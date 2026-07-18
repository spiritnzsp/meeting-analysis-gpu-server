"""
Video Encoding Worker

Orchestrates video encoding for queued requests. Manages pending uploads
(binary frame data streamed to temp files) and processes encoding jobs
via VideoEncoderProcessor.
"""
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict

from .config import Config
from .queue_manager import QueueManager, QueuedRequest
from .protocol import (
    VideoEncodeRequest, VideoEncodeResult, ProgressMessage, ProcessingStage,
    ErrorMessage
)
from .processors.video_encoder import VideoEncoderProcessor
from .binary_protocol import BinaryFrameType, encode_binary_frame
from .utils.temp_video_file import TempVideoFile
from .logging_config import (
    get_logger, set_request_context, clear_request_context,
    LogEvents
)

logger = get_logger(__name__)


# Chunk-send tuning for the WebSocket result download path. These are
# protocol-internal performance constants - not user-facing config -
# because the wrong values silently re-create a real production bug:
#
# Originally 64KB chunks were sent in a tight `await websocket.send(...)`
# loop with no event-loop yield. On a fast LAN a 550MB result produced
# ~8800 sends; pong frames queue behind data frames in the websockets
# outgoing buffer and the client's ping_timeout fired mid-transfer.
# On a slow link (e.g. transcontinental VPN) bandwidth backpressure
# kept the queue shallow and the bug never showed.
#
# Combined with ping_timeout=None on both ends (server.py / client),
# these values eliminate the failure mode: fewer awaits and explicit
# cooperative yields keep the event loop fair regardless of network.
_CHUNK_SIZE = 1024 * 1024   # 1MB
_YIELD_EVERY_CHUNKS = 8     # cooperative `await asyncio.sleep(0)` cadence


class VideoWorker:
    """
    Video encoding worker.

    Manages:
    - Pending uploads (request_id -> TempVideoFile)
    - Processing loop: dequeue ready requests and encode
    - Upload timeout enforcement
    """

    def __init__(self, config: Config, queue: QueueManager, arbiter=None):
        self.config = config
        self.queue = queue
        # The arbiter reserves the transient NVENC VRAM per encode (wired in D2.3).
        self._arbiter = arbiter

        self._encoder: Optional[VideoEncoderProcessor] = None
        self._pending_uploads: Dict[str, TempVideoFile] = {}
        self._running = False
        self._current_request: Optional[str] = None
        self._timeout_task: Optional[asyncio.Task] = None

        # Exponential backoff for error recovery
        self._error_backoff_seconds: float = 1.0
        self._max_error_backoff_seconds: float = 60.0

    async def start(self) -> None:
        """Start the video worker processing loop."""
        logger.info("Video Worker starting")

        # Initialize encoder
        self._encoder = VideoEncoderProcessor(self.config.video_encoding)
        await self._encoder.initialize()

        codecs = self._encoder.codec_capabilities.available_codecs
        if codecs:
            logger.info(f"Video Worker ready with NVENC codecs: {codecs}")
        else:
            logger.warning("Video Worker started but no NVENC codecs detected")

        self._running = True

        # Start upload timeout checker
        self._timeout_task = asyncio.create_task(self._check_upload_timeouts())

        while self._running:
            try:
                queued = await self.queue.wait_for_request()

                if queued.cancelled:
                    set_request_context(request_id=queued.request.request_id)
                    if queued.timeout_expired:
                        logger.info("Video request timed out in queue")
                        try:
                            await queued.websocket.send(ErrorMessage(
                                request_id=queued.request.request_id,
                                error="Video request timed out while waiting in queue",
                                recoverable=True,
                                error_code="QUEUE_TIMEOUT",
                            ).to_json())
                        except Exception:
                            pass
                    # Clean up any pending upload
                    self._cleanup_upload(queued.request.request_id)
                    clear_request_context()
                    continue

                # Process the video request
                processing_timeout = self.config.video_queue.processing_timeout
                try:
                    await asyncio.wait_for(
                        self._process_video_request(queued),
                        timeout=processing_timeout,
                    )
                except asyncio.TimeoutError:
                    request_id = queued.request.request_id
                    set_request_context(request_id=request_id)
                    logger.error(f"Video encoding timed out after {processing_timeout}s")

                    if self._encoder:
                        self._encoder.cancel(request_id)

                    try:
                        await queued.websocket.send(ErrorMessage(
                            request_id=request_id,
                            error=f"Video encoding timed out after {processing_timeout} seconds",
                            recoverable=False,
                            error_code="PROCESSING_TIMEOUT",
                        ).to_json())
                    except Exception:
                        pass

                    self._cleanup_upload(request_id)
                    self._current_request = None
                    clear_request_context()

                self._error_backoff_seconds = 1.0

            except asyncio.CancelledError:
                logger.info("Video Worker cancelled")
                break
            except Exception as e:
                logger.error(f"Video Worker error: {e}", exc_info=True)
                await asyncio.sleep(self._error_backoff_seconds)
                self._error_backoff_seconds = min(
                    self._error_backoff_seconds * 2,
                    self._max_error_backoff_seconds,
                )

        logger.info("Video Worker stopped")

    async def stop(self) -> None:
        """Stop the worker."""
        self._running = False

        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass

        # Clean up all pending uploads
        for request_id in list(self._pending_uploads.keys()):
            self._cleanup_upload(request_id)

        if self._encoder:
            self._encoder.shutdown()

    def register_upload(self, request_id: str) -> TempVideoFile:
        """
        Register a new pending upload for a video encode request.

        Args:
            request_id: The request ID

        Returns:
            TempVideoFile for streaming data
        """
        temp_file = TempVideoFile(
            suffix=".mp4",
            max_size=self.config.video_encoding.max_input_size,
            temp_dir=self.config.video_encoding.temp_directory,
        )
        self._pending_uploads[request_id] = temp_file
        logger.info(f"Registered upload for {request_id}")
        return temp_file

    def receive_video_data(self, request_id: str, data: bytes) -> bool:
        """
        Receive a chunk of video data for a pending upload.

        Args:
            request_id: The request ID
            data: Video data chunk

        Returns:
            True if data was accepted

        Raises:
            ValueError: If data exceeds size limit
            KeyError: If request_id not found in pending uploads
        """
        if request_id not in self._pending_uploads:
            raise KeyError(f"No pending upload for request {request_id}")

        temp_file = self._pending_uploads[request_id]
        temp_file.append(data)
        return True

    async def finalize_upload(self, request_id: str) -> bool:
        """
        Finalize a pending upload and mark the queue entry as ready.

        Args:
            request_id: The request ID

        Returns:
            True if finalized successfully
        """
        if request_id not in self._pending_uploads:
            return False

        temp_file = self._pending_uploads[request_id]
        temp_file.finalize()

        # Mark queue entry as ready for processing
        await self.queue.mark_data_ready(request_id)
        logger.info(f"Upload finalized for {request_id}: {temp_file.current_size} bytes")
        return True

    def _cleanup_upload(self, request_id: str) -> None:
        """Clean up a pending upload's temp file."""
        temp_file = self._pending_uploads.pop(request_id, None)
        if temp_file:
            temp_file.cleanup()

    async def _check_upload_timeouts(self) -> None:
        """Periodically check for stale uploads and cancel them."""
        timeout = self.config.video_encoding.data_upload_timeout
        while self._running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                stale = []
                for request_id, temp_file in self._pending_uploads.items():
                    if not temp_file.is_complete and temp_file.seconds_since_last_write > timeout:
                        stale.append(request_id)

                for request_id in stale:
                    logger.warning(f"Upload timeout for {request_id} (no data for {timeout}s)")
                    self._cleanup_upload(request_id)
                    await self.queue.cancel(request_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Upload timeout check error: {e}")

    async def _process_video_request(self, queued: QueuedRequest) -> None:
        """Process a single video encoding request."""
        request: VideoEncodeRequest = queued.request
        websocket = queued.websocket

        self._current_request = request.request_id
        start_time = time.time()

        set_request_context(request_id=request.request_id)
        logger.info(
            f"Processing video encode request: {request.filename} "
            f"(transfer_method={request.transfer_method})"
        )

        if request.transfer_method == "shared_fs":
            await self._process_shared_fs_request(queued, start_time)
        else:
            await self._process_websocket_request(queued, start_time)

    async def _process_shared_fs_request(
        self, queued: QueuedRequest, start_time: float,
    ) -> None:
        """Process a shared filesystem video encode request.

        Input is read directly from the shared path. Output is written
        directly to the shared path. No binary data is transferred
        over the WebSocket — only JSON control messages.
        """
        request: VideoEncodeRequest = queued.request
        websocket = queued.websocket

        input_path = Path(request.input_path)
        output_path = Path(request.output_path)

        try:
            if not input_path.exists():
                await websocket.send(VideoEncodeResult(
                    request_id=request.request_id,
                    success=False,
                    error_message=f"Input file not found on shared filesystem: {input_path}",
                ).to_json())
                return

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Progress callback
            async def on_progress(percent: int, message: str):
                try:
                    await websocket.send(ProgressMessage(
                        request_id=request.request_id,
                        stage=ProcessingStage.ENCODING,
                        percent=percent,
                        message=message,
                    ).to_json())
                except Exception:
                    pass

            await on_progress(5, "Starting shared filesystem encoding...")

            # Encode directly from/to shared filesystem paths
            success, error, codec_used = await self._encoder.encode(
                input_path=input_path,
                output_path=output_path,
                options=request.options,
                progress_callback=on_progress,
                request_id=request.request_id,
            )

            encoding_time = time.time() - start_time

            if success:
                output_size = output_path.stat().st_size if output_path.exists() else 0

                # Send result — no binary frames needed, client reads from shared fs
                await websocket.send(VideoEncodeResult(
                    request_id=request.request_id,
                    success=True,
                    output_size=output_size,
                    codec_used=codec_used,
                    encoding_time=encoding_time,
                ).to_json())

                logger.info(
                    f"Shared filesystem encoding complete: {codec_used}, "
                    f"{output_size} bytes, {encoding_time:.1f}s, "
                    f"output: {output_path}"
                )
            else:
                await websocket.send(VideoEncodeResult(
                    request_id=request.request_id,
                    success=False,
                    codec_used=codec_used,
                    encoding_time=encoding_time,
                    error_message=error,
                ).to_json())
                logger.error(f"Shared filesystem encoding failed: {error}")

        except Exception as e:
            logger.error(f"Shared filesystem video processing error: {e}", exc_info=True)
            try:
                await websocket.send(VideoEncodeResult(
                    request_id=request.request_id,
                    success=False,
                    error_message="Video encoding failed",
                ).to_json())
            except Exception:
                pass
        finally:
            self._current_request = None
            clear_request_context()

    async def _process_websocket_request(
        self, queued: QueuedRequest, start_time: float,
    ) -> None:
        """Process a WebSocket chunked video encode request (v1.0 flow)."""
        request: VideoEncodeRequest = queued.request
        websocket = queued.websocket

        # Get the finalized input temp file
        temp_input = self._pending_uploads.get(request.request_id)
        if not temp_input or not temp_input.is_complete:
            logger.error(f"No finalized input for request {request.request_id}")
            await websocket.send(VideoEncodeResult(
                request_id=request.request_id,
                success=False,
                error_message="Upload data not available",
            ).to_json())
            self._cleanup_upload(request.request_id)
            self._current_request = None
            clear_request_context()
            return

        input_path = temp_input.path

        # Create output temp file
        output_temp = TempVideoFile(
            suffix=".mp4",
            max_size=self.config.video_encoding.max_input_size,
            temp_dir=self.config.video_encoding.temp_directory,
        )
        output_path = output_temp.path

        try:
            # Progress callback
            async def on_progress(percent: int, message: str):
                try:
                    await websocket.send(ProgressMessage(
                        request_id=request.request_id,
                        stage=ProcessingStage.ENCODING,
                        percent=percent,
                        message=message,
                    ).to_json())
                except Exception:
                    pass

            await on_progress(5, "Starting video encoding...")

            # Encode
            success, error, codec_used = await self._encoder.encode(
                input_path=input_path,
                output_path=output_path,
                options=request.options,
                progress_callback=on_progress,
                request_id=request.request_id,
            )

            encoding_time = time.time() - start_time

            if success:
                # Read output and send result + binary data
                output_size = output_path.stat().st_size if output_path.exists() else 0

                # Send result metadata
                await websocket.send(VideoEncodeResult(
                    request_id=request.request_id,
                    success=True,
                    output_size=output_size,
                    codec_used=codec_used,
                    encoding_time=encoding_time,
                ).to_json())

                # Send encoded video data as binary frames. See the
                # _CHUNK_SIZE / _YIELD_EVERY_CHUNKS comment at module
                # top for the constants' rationale.
                if output_path.exists():
                    loop = asyncio.get_running_loop()
                    with open(output_path, 'rb') as f:
                        chunk_count = 0
                        while True:
                            # Off-thread read so disk IO does not block
                            # the event loop while the kernel readahead
                            # populates the next chunk.
                            chunk = await loop.run_in_executor(
                                None, f.read, _CHUNK_SIZE,
                            )
                            if not chunk:
                                break
                            frame = encode_binary_frame(
                                BinaryFrameType.VIDEO_OUTPUT,
                                request.request_id,
                                chunk,
                            )
                            await websocket.send(frame)
                            chunk_count += 1
                            if chunk_count % _YIELD_EVERY_CHUNKS == 0:
                                # Cooperative yield: lets the ping/pong
                                # handler and any other concurrent tasks
                                # run before the next chunk is queued.
                                await asyncio.sleep(0)

                    # Send empty final frame to signal end
                    await websocket.send(encode_binary_frame(
                        BinaryFrameType.VIDEO_OUTPUT,
                        request.request_id,
                        b"",
                    ))

                logger.info(
                    f"Video encoding complete: {codec_used}, "
                    f"{output_size} bytes, {encoding_time:.1f}s"
                )
            else:
                await websocket.send(VideoEncodeResult(
                    request_id=request.request_id,
                    success=False,
                    codec_used=codec_used,
                    encoding_time=encoding_time,
                    error_message=error,
                ).to_json())
                logger.error(f"Video encoding failed: {error}")

        except Exception as e:
            logger.error(f"Video processing error: {e}", exc_info=True)
            try:
                await websocket.send(VideoEncodeResult(
                    request_id=request.request_id,
                    success=False,
                    error_message="Video encoding failed",
                ).to_json())
            except Exception:
                pass
        finally:
            # Clean up both temp files
            self._cleanup_upload(request.request_id)
            output_temp.cleanup()
            self._current_request = None
            clear_request_context()

    @property
    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self._current_request is not None

    @property
    def current_request_id(self) -> Optional[str]:
        """Get ID of currently processing request."""
        return self._current_request

    @property
    def pending_upload_count(self) -> int:
        """Number of pending uploads."""
        return len(self._pending_uploads)
