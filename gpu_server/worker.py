"""
GPU Processing Worker

Orchestrates Whisper and PyAnnote processing for queued requests.
"""
import asyncio
import time
from typing import Optional, Callable, Awaitable

from .config import Config
from .queue_manager import QueueManager, QueuedRequest
from .protocol import (
    ProcessRequest, ProcessingResult, ProgressMessage, ProcessingStage,
    TranscriptSegment, DiarizationSegment, SpeakerEmbedding, ErrorMessage
)
from .processors import WhisperProcessor, PyAnnoteProcessor, ProcessorCancelled
from .utils.temp_file import cleanup_orphaned_temp_files
from .logging_config import (
    get_logger, set_request_context, clear_request_context,
    LogEvents, PerformanceTimer
)

logger = get_logger(__name__)


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

        # Exponential backoff for error recovery (prevents log flooding)
        self._error_backoff_seconds: float = 1.0
        self._max_error_backoff_seconds: float = 60.0

    async def start(self):
        """Start the worker processing loop."""
        logger.info(
            "GPU Worker starting",
            data={
                'whisper_model': self.config.whisper.model,
                'whisper_device': self.config.whisper.device,
                'pyannote_model': self.config.pyannote.model,
                'pyannote_device': self.config.pyannote.device,
            }
        )

        # Clean up orphaned temp files from previous runs (crash recovery)
        orphans_cleaned = cleanup_orphaned_temp_files()
        if orphans_cleaned > 0:
            logger.info(f"Cleaned {orphans_cleaned} orphaned temp files from previous runs")

        # Initialize processors (lazy load models on first use)
        self._whisper = WhisperProcessor(self.config.whisper)
        self._pyannote = PyAnnoteProcessor(self.config.pyannote)

        self._running = True
        logger.info("GPU Worker started and ready for requests")

        while self._running:
            try:
                # Wait for next request
                queued = await self.queue.wait_for_request()

                if queued.cancelled:
                    set_request_context(request_id=queued.request.request_id)
                    # Check if it was a timeout vs user cancellation
                    if queued.timeout_expired:
                        logger.info(
                            LogEvents.REQUEST_TIMEOUT,
                            data={'stage': 'queue', 'reason': 'queue_timeout'}
                        )
                        # Notify client of timeout
                        try:
                            await queued.websocket.send(ErrorMessage(
                                request_id=queued.request.request_id,
                                error="Request timed out while waiting in queue",
                                recoverable=True,
                                error_code="QUEUE_TIMEOUT",
                            ).to_json())
                        except Exception as e:
                            logger.warning(f"Failed to send timeout error: {e}")
                    else:
                        logger.info(LogEvents.REQUEST_CANCELLED, data={'stage': 'queue'})
                    clear_request_context()
                    continue

                # Process the request with timeout
                processing_timeout = self.config.queue.processing_timeout
                try:
                    await asyncio.wait_for(
                        self._process_request(queued),
                        timeout=processing_timeout
                    )
                except asyncio.TimeoutError:
                    request_id = queued.request.request_id
                    set_request_context(request_id=request_id)
                    logger.error(
                        LogEvents.REQUEST_TIMEOUT,
                        data={
                            'stage': 'processing',
                            'timeout_seconds': processing_timeout,
                        }
                    )

                    # Cancel processors to stop ongoing GPU operations
                    if self._whisper:
                        self._whisper.cancel(request_id)
                    if self._pyannote:
                        self._pyannote.cancel(request_id)

                    # Wait for GPU operations to complete before continuing
                    # This prevents race conditions where next request starts
                    # while GPU is still busy with the timed-out request
                    logger.info("Waiting for GPU operations to complete...")
                    wait_timeout = 30.0  # Max time to wait for GPU to finish

                    # Wait for Whisper processor, recover if stuck
                    if self._whisper:
                        whisper_idle = await self._whisper.wait_for_idle(wait_timeout)
                        if not whisper_idle:
                            logger.error(
                                "Whisper processor stuck after timeout, recreating processor"
                            )
                            try:
                                self._whisper.shutdown()
                            except Exception as e:
                                logger.warning(f"Error shutting down stuck Whisper processor: {e}")
                            self._whisper = WhisperProcessor(self.config.whisper)
                            logger.info("Whisper processor recreated successfully")

                    # Wait for PyAnnote processor, recover if stuck
                    if self._pyannote:
                        pyannote_idle = await self._pyannote.wait_for_idle(wait_timeout)
                        if not pyannote_idle:
                            logger.error(
                                "PyAnnote processor stuck after timeout, recreating processor"
                            )
                            try:
                                self._pyannote.shutdown()
                            except Exception as e:
                                logger.warning(f"Error shutting down stuck PyAnnote processor: {e}")
                            self._pyannote = PyAnnoteProcessor(self.config.pyannote)
                            logger.info("PyAnnote processor recreated successfully")

                    logger.info("GPU operations completed, ready for next request")

                    # Send timeout error to client
                    try:
                        await queued.websocket.send(ErrorMessage(
                            request_id=request_id,
                            error=f"Processing timed out after {processing_timeout} seconds",
                            recoverable=False,
                            error_code="PROCESSING_TIMEOUT",
                        ).to_json())
                    except Exception as e:
                        logger.warning(f"Failed to send processing timeout error: {e}")

                    # Clean up GPU memory after timeout
                    self._cleanup_gpu_memory()

                    self._current_request = None
                    clear_request_context()

                # Reset error backoff on successful request processing
                self._error_backoff_seconds = 1.0

            except asyncio.CancelledError:
                logger.info("Worker cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Worker error: {e}",
                    exc_info=True,
                    data={'backoff_seconds': self._error_backoff_seconds}
                )
                await asyncio.sleep(self._error_backoff_seconds)

                # Exponential backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s max
                self._error_backoff_seconds = min(
                    self._error_backoff_seconds * 2,
                    self._max_error_backoff_seconds
                )

        logger.info("GPU Worker stopped")

    async def stop(self):
        """Stop the worker."""
        self._running = False

        # Shutdown processors (unloads models and shuts down executors)
        if self._whisper:
            self._whisper.shutdown()
        if self._pyannote:
            self._pyannote.shutdown()

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

        # Set request context for all logging in this method
        set_request_context(
            request_id=request.request_id,
            meeting_name=request.meeting_name,
        )

        logger.info(
            LogEvents.REQUEST_STARTED,
            data={
                'audio_size_bytes': len(request.audio_data),
                'transcribe': request.options.transcribe,
                'diarize': request.options.diarize,
                'extract_embeddings': request.options.extract_embeddings,
            }
        )

        # Track consecutive progress send failures to detect disconnected clients
        consecutive_failures = 0
        max_consecutive_failures = 3

        # Create progress callback with early termination support
        async def send_progress(msg: ProgressMessage):
            nonlocal consecutive_failures
            try:
                await websocket.send(msg.to_json())
                consecutive_failures = 0  # Reset on success
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(
                        f"Client appears disconnected ({consecutive_failures} consecutive failures), "
                        f"cancelling processing for {request.request_id}"
                    )
                    # Cancel the processors to stop wasting GPU resources
                    if self._whisper:
                        self._whisper.cancel(request.request_id)
                    if self._pyannote:
                        self._pyannote.cancel(request.request_id)
                else:
                    logger.warning(f"Failed to send progress ({consecutive_failures}/{max_consecutive_failures}): {e}")

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
                logger.info(LogEvents.TRANSCRIPTION_STARTED)
                transcribe_start = time.time()

                transcript_segments, full_text, detected_language = await self._whisper.transcribe(
                    audio_data=request.audio_data,
                    language=request.options.language,
                    model_override=request.options.whisper_model,
                    progress_callback=send_progress,
                    request_id=request.request_id,
                )

                logger.info(
                    LogEvents.TRANSCRIPTION_COMPLETED,
                    data={
                        'duration_ms': (time.time() - transcribe_start) * 1000,
                        'segments': len(transcript_segments),
                        'detected_language': detected_language,
                        'text_length': len(full_text),
                    }
                )

            # Diarization
            if request.options.diarize:
                logger.info(LogEvents.DIARIZATION_STARTED)
                diarize_start = time.time()

                diarization_segments = await self._pyannote.diarize(
                    audio_data=request.audio_data,
                    num_speakers=request.options.num_speakers,
                    progress_callback=send_progress,
                    request_id=request.request_id,
                )

                unique_speakers = len(set(seg.speaker for seg in diarization_segments))
                logger.info(
                    LogEvents.DIARIZATION_COMPLETED,
                    data={
                        'duration_ms': (time.time() - diarize_start) * 1000,
                        'segments': len(diarization_segments),
                        'speakers': unique_speakers,
                    }
                )

                # Align transcript with diarization
                if transcript_segments and diarization_segments:
                    transcript_segments = self._pyannote.align_transcript_with_diarization(
                        transcript_segments, diarization_segments
                    )

            # Extract embeddings (non-fatal - attendee registry is optional)
            if request.options.extract_embeddings and diarization_segments:
                logger.info(LogEvents.EMBEDDING_EXTRACTION_STARTED)
                embed_start = time.time()

                try:
                    speaker_embeddings = await self._pyannote.extract_embeddings(
                        audio_data=request.audio_data,
                        diarization_segments=diarization_segments,
                        meeting_id=request.request_id,
                        progress_callback=send_progress,
                        request_id=request.request_id,
                        hf_token=request.options.hf_token,
                    )

                    logger.info(
                        LogEvents.EMBEDDING_EXTRACTION_COMPLETED,
                        data={
                            'duration_ms': (time.time() - embed_start) * 1000,
                            'embeddings': len(speaker_embeddings),
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Embedding extraction failed (non-fatal): {e}",
                        exc_info=False,
                    )
                    speaker_embeddings = []

            # Build result
            processing_time = time.time() - start_time

            # Check for partial failures - requested operations that produced no output
            warnings = []
            if request.options.transcribe and not transcript_segments:
                warnings.append("Transcription requested but produced no segments")
            if request.options.diarize and not diarization_segments:
                warnings.append("Diarization requested but produced no speaker segments")
            if request.options.extract_embeddings and request.options.diarize and diarization_segments and not speaker_embeddings:
                # Only warn if diarization succeeded but embedding extraction failed
                warnings.append("Embedding extraction requested but produced no embeddings")

            result = ProcessingResult(
                request_id=request.request_id,
                success=True,
                transcript_segments=transcript_segments,
                diarization_segments=diarization_segments,
                speaker_embeddings=speaker_embeddings,
                full_text=full_text,
                detected_language=detected_language,
                processing_time_seconds=processing_time,
                warnings=warnings,
            )

            if warnings:
                logger.warning(
                    "Processing completed with warnings",
                    data={'warnings': warnings}
                )

            logger.info(
                LogEvents.REQUEST_COMPLETED,
                data={
                    'processing_time_seconds': round(processing_time, 2),
                    'transcript_segments': len(transcript_segments),
                    'diarization_segments': len(diarization_segments),
                    'speaker_embeddings': len(speaker_embeddings),
                    'detected_language': detected_language,
                    'warnings_count': len(warnings),
                }
            )

        except ProcessorCancelled:
            # Processing was cancelled (timeout or user request)
            processing_time = time.time() - start_time
            logger.info(
                LogEvents.REQUEST_CANCELLED,
                data={
                    'processing_time_seconds': round(processing_time, 2),
                    'reason': 'processor_cancelled',
                }
            )
            result = ProcessingResult(
                request_id=request.request_id,
                success=False,
                error_message="Processing was cancelled",
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                LogEvents.REQUEST_FAILED,
                data={
                    'error_type': type(e).__name__,
                    'error': str(e),
                    'processing_time_seconds': round(processing_time, 2),
                },
                exc_info=True
            )
            result = ProcessingResult(
                request_id=request.request_id,
                success=False,
                error_message="Processing failed",  # Don't leak internal details to clients
                processing_time_seconds=processing_time,
            )

        finally:
            self._current_request = None
            clear_request_context()

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

    def _cleanup_gpu_memory(self):
        """
        Force GPU memory cleanup.

        Called after processing timeouts to free any stuck GPU memory.
        Note: This may not cancel in-flight operations, but will release
        memory after they complete.
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for any pending GPU ops
                logger.info("GPU memory cache cleared after timeout")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to cleanup GPU memory: {e}")

    @property
    def is_processing(self) -> bool:
        """Check if currently processing a request."""
        return self._current_request is not None

    @property
    def current_request_id(self) -> Optional[str]:
        """Get ID of currently processing request."""
        return self._current_request
