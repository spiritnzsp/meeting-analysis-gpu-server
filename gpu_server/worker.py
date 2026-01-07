"""
GPU Processing Worker

Orchestrates Whisper and PyAnnote processing for queued requests.
"""
import asyncio
import logging
import time
from typing import Optional, Callable, Awaitable

from .config import Config
from .queue_manager import QueueManager, QueuedRequest
from .protocol import (
    ProcessRequest, ProcessingResult, ProgressMessage, ProcessingStage,
    TranscriptSegment, DiarizationSegment, SpeakerEmbedding
)
from .processors import WhisperProcessor, PyAnnoteProcessor

logger = logging.getLogger(__name__)


class GPUWorker:
    """
    GPU processing worker.

    Continuously processes requests from the queue using
    Whisper for transcription and PyAnnote for diarization.
    """

    def __init__(self, config: Config, queue: QueueManager):
        """
        Initialize the GPU worker.

        Args:
            config: Server configuration
            queue: Request queue manager
        """
        self.config = config
        self.queue = queue

        self._whisper: Optional[WhisperProcessor] = None
        self._pyannote: Optional[PyAnnoteProcessor] = None

        self._running = False
        self._current_request: Optional[str] = None

    async def start(self):
        """Start the worker processing loop."""
        logger.info("GPU Worker starting...")

        # Initialize processors (lazy load models on first use)
        self._whisper = WhisperProcessor(self.config.whisper)
        self._pyannote = PyAnnoteProcessor(self.config.pyannote)

        self._running = True
        logger.info("GPU Worker started")

        while self._running:
            try:
                # Wait for next request
                queued = await self.queue.wait_for_request()

                if queued.cancelled:
                    # Check if it was a timeout vs user cancellation
                    if getattr(queued, 'timeout_expired', False):
                        logger.info(f"Request timed out in queue: {queued.request.request_id}")
                        # Notify client of timeout
                        try:
                            from .protocol import ErrorMessage
                            await queued.websocket.send(ErrorMessage(
                                request_id=queued.request.request_id,
                                error="Request timed out while waiting in queue",
                                recoverable=True,
                            ).to_json())
                        except Exception as e:
                            logger.warning(f"Failed to send timeout error: {e}")
                    else:
                        logger.info(f"Skipping cancelled request: {queued.request.request_id}")
                    continue

                # Process the request with timeout
                processing_timeout = self.config.queue.processing_timeout
                try:
                    await asyncio.wait_for(
                        self._process_request(queued),
                        timeout=processing_timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"Request {queued.request.request_id} timed out after "
                        f"{processing_timeout}s of processing"
                    )
                    # Send timeout error to client
                    try:
                        from .protocol import ErrorMessage
                        await queued.websocket.send(ErrorMessage(
                            request_id=queued.request.request_id,
                            error=f"Processing timed out after {processing_timeout} seconds",
                            recoverable=False,
                        ).to_json())
                    except Exception as e:
                        logger.warning(f"Failed to send processing timeout error: {e}")
                    self._current_request = None

            except asyncio.CancelledError:
                logger.info("Worker cancelled")
                break
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                await asyncio.sleep(1)  # Brief pause before continuing

        logger.info("GPU Worker stopped")

    async def stop(self):
        """Stop the worker."""
        self._running = False

        # Unload models
        if self._whisper:
            self._whisper.unload()
        if self._pyannote:
            self._pyannote.unload()

    async def _process_request(self, queued: QueuedRequest):
        """
        Process a single request.

        Args:
            queued: The queued request to process
        """
        request = queued.request
        websocket = queued.websocket

        self._current_request = request.request_id
        start_time = time.time()

        logger.info(f"Processing request: {request.request_id} ({request.meeting_name})")

        # Create progress callback
        async def send_progress(msg: ProgressMessage):
            try:
                await websocket.send(msg.to_json())
            except Exception as e:
                logger.warning(f"Failed to send progress: {e}")

        result = ProcessingResult(
            request_id=request.request_id,
            success=False,
        )

        try:
            await send_progress(ProgressMessage(
                request_id=request.request_id,
                stage=ProcessingStage.LOADING_MODELS,
                percent=5,
                message="Loading models..."
            ))

            transcript_segments = []
            full_text = ""
            detected_language = ""
            diarization_segments = []
            speaker_embeddings = []

            # Transcription
            if request.options.transcribe:
                transcript_segments, full_text, detected_language = await self._whisper.transcribe(
                    audio_data=request.audio_data,
                    language=request.options.language,
                    model_override=request.options.whisper_model,
                    progress_callback=send_progress,
                    request_id=request.request_id,
                )

            # Diarization
            if request.options.diarize:
                diarization_segments = await self._pyannote.diarize(
                    audio_data=request.audio_data,
                    num_speakers=request.options.num_speakers,
                    progress_callback=send_progress,
                    request_id=request.request_id,
                )

                # Align transcript with diarization
                if transcript_segments and diarization_segments:
                    transcript_segments = self._pyannote.align_transcript_with_diarization(
                        transcript_segments, diarization_segments
                    )

            # Extract embeddings
            if request.options.extract_embeddings and diarization_segments:
                speaker_embeddings = await self._pyannote.extract_embeddings(
                    audio_data=request.audio_data,
                    diarization_segments=diarization_segments,
                    meeting_id=request.request_id,
                    progress_callback=send_progress,
                    request_id=request.request_id,
                )

            # Build result
            processing_time = time.time() - start_time

            result = ProcessingResult(
                request_id=request.request_id,
                success=True,
                transcript_segments=transcript_segments,
                diarization_segments=diarization_segments,
                speaker_embeddings=speaker_embeddings,
                full_text=full_text,
                detected_language=detected_language,
                processing_time_seconds=processing_time,
            )

            logger.info(
                f"Request {request.request_id} completed in {processing_time:.1f}s: "
                f"{len(transcript_segments)} transcript segments, "
                f"{len(diarization_segments)} diarization segments, "
                f"{len(speaker_embeddings)} embeddings"
            )

        except Exception as e:
            logger.error(f"Processing error for {request.request_id}: {e}", exc_info=True)
            result = ProcessingResult(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
                processing_time_seconds=time.time() - start_time,
            )

        finally:
            self._current_request = None

        # Send result
        try:
            await send_progress(ProgressMessage(
                request_id=request.request_id,
                stage=ProcessingStage.COMPLETE if result.success else ProcessingStage.FAILED,
                percent=100,
                message="Complete" if result.success else result.error_message,
            ))
            await websocket.send(result.to_json())
        except Exception as e:
            logger.error(f"Failed to send result: {e}")

    @property
    def is_processing(self) -> bool:
        """Check if currently processing a request."""
        return self._current_request is not None

    @property
    def current_request_id(self) -> Optional[str]:
        """Get ID of currently processing request."""
        return self._current_request
