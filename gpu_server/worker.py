"""
GPU Processing Worker

Orchestrates Whisper and PyAnnote processing for queued requests, gated through
the VRAM arbiter so audio serializes/evicts around the LLM and video workloads on
a single card instead of independently OOMing.

Each request runs under an arbiter lease acquired OUTSIDE the per-request
processing timeout (F4): the lease keeps the request's required models resident
for the whole job (a concurrent LLM admission cannot evict them mid-transcription)
and, conversely, admitting this job may evict the idle LLM to make room. On a
processing timeout the processors are cancelled and drained WHILE STILL HOLDING
THE LEASE; a wedged processor is recreated through its ResidentProcessorHandle
(rebind, under the arbiter's admission barrier) so the arbiter registry never
desyncs.
"""
import asyncio
import time
from typing import Optional, List

from .config import Config
from .queue_manager import QueueManager, QueuedRequest
from .protocol import (
    ProcessRequest, ProcessingResult, ProgressMessage, ProcessingStage,
    TranscriptSegment, DiarizationSegment, SpeakerEmbedding, ErrorMessage
)
from .backoff import ErrorBackoff
from .processors import WhisperProcessor, PyAnnoteProcessor, ProcessorCancelled
from .processors.whisper_processor import WHISPER_MODEL_KEY
from .processors.pyannote_processor import PYANNOTE_MODEL_KEY, PYANNOTE_EMBEDDING_KEY
from .orchestrator import (
    VramArbiter, WorkloadNeed, ResidentModel, ResidentProcessorHandle,
)
from .utils.temp_file import cleanup_orphaned_temp_files
from .logging_config import (
    get_logger, set_request_context, clear_request_context,
    LogEvents, PerformanceTimer
)

logger = get_logger(__name__)


class GPUWorker:
    """
    GPU processing worker.

    Continuously processes requests from the queue using Whisper for
    transcription and PyAnnote for diarization, gated through the VRAM arbiter.
    """

    def __init__(self, config: Config, queue: QueueManager, arbiter: VramArbiter):
        """
        Initialize the GPU worker.

        Args:
            config: Server configuration
            queue: Request queue manager
            arbiter: VRAM admission arbiter (shared across all workloads)
        """
        self.config = config
        self.queue = queue
        self._arbiter = arbiter

        # The processors are owned by ResidentProcessorHandles so a stuck-timeout
        # recovery can recreate the underlying processor and rebind its arbiter
        # residents as one transaction, keeping the registry identity stable (F5).
        self._whisper_handle = ResidentProcessorHandle(
            lambda: WhisperProcessor(config.whisper)
        )
        self._pyannote_handle = ResidentProcessorHandle(
            lambda: PyAnnoteProcessor(config.pyannote)
        )

        self._running = False
        self._current_request: Optional[str] = None

        # Exponential backoff for error recovery (prevents log flooding)
        self._backoff = ErrorBackoff()

    def residents(self) -> List[ResidentModel]:
        """The arbiter residents this worker owns (whisper + the two pyannote
        models). The server registers these synchronously at startup (F9)."""
        return self._whisper_handle.residents() + self._pyannote_handle.residents()

    # The current processor instances, resolved through the handles so a
    # stuck-recovery swap is transparent to _process_request. Read-only.
    @property
    def _whisper(self) -> WhisperProcessor:
        return self._whisper_handle.processor

    @property
    def _pyannote(self) -> PyAnnoteProcessor:
        return self._pyannote_handle.processor

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

        self._running = True
        logger.info("GPU Worker started and ready for requests")

        while self._running:
            try:
                queued = await self.queue.wait_for_request()

                if queued.cancelled:
                    await self._handle_queue_cancelled(queued)
                    continue

                await self._serve(queued)

                # Reset error backoff on successful request processing
                self._backoff.reset()

            except asyncio.CancelledError:
                logger.info("Worker cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Worker error: {e}",
                    exc_info=True,
                    data={'backoff_seconds': self._backoff.current_seconds}
                )
                await self._backoff.sleep()

        logger.info("GPU Worker stopped")

    async def _handle_queue_cancelled(self, queued: QueuedRequest) -> None:
        set_request_context(request_id=queued.request.request_id)
        if queued.timeout_expired:
            logger.info(
                LogEvents.REQUEST_TIMEOUT,
                data={'stage': 'queue', 'reason': 'queue_timeout'}
            )
            await self._send_error(
                queued.websocket, queued.request.request_id,
                "Request timed out while waiting in queue",
                recoverable=True, error_code="QUEUE_TIMEOUT",
            )
        else:
            logger.info(LogEvents.REQUEST_CANCELLED, data={'stage': 'queue'})
        clear_request_context()

    def _workload_need(self, request: ProcessRequest) -> WorkloadNeed:
        """Derive the required models from what the request actually asks for
        (request-derived, not hardcoded): transcribe→whisper, diarize→pyannote,
        extract_embeddings (only meaningful alongside diarize)→pyannote_embedding.
        This pins only the VRAM the job will use, which is what makes the two
        pyannote residents pay off."""
        keys: list[str] = []
        if request.options.transcribe:
            keys.append(WHISPER_MODEL_KEY)
        if request.options.diarize:
            keys.append(PYANNOTE_MODEL_KEY)
            if request.options.extract_embeddings:
                keys.append(PYANNOTE_EMBEDDING_KEY)
        return WorkloadNeed(required_models=tuple(keys))

    async def _serve(self, queued: QueuedRequest) -> None:
        """Acquire the VRAM lease (outside the timeout), process under it, and be
        the SOLE sender of the result. On timeout, recover while still holding the
        lease, then send the outcome."""
        request = queued.request
        websocket = queued.websocket
        need = self._workload_need(request)
        processing_timeout = self.config.queue.processing_timeout

        try:
            timed_out = False
            result: Optional[ProcessingResult] = None
            async with await self._arbiter.acquire(need):
                task = asyncio.ensure_future(self._process_request(queued))
                try:
                    # shield so the timeout does not cancel the coroutine (a
                    # run_in_executor job can't be cancelled anyway); on timeout
                    # we cancel+drain+recover explicitly, still holding the lease.
                    result = await asyncio.wait_for(
                        asyncio.shield(task), timeout=processing_timeout
                    )
                except asyncio.TimeoutError:
                    await self._recover_from_timeout(queued, task)
                    timed_out = True
            # Lease released here — AFTER any recovery, so no concurrent admission
            # can evict a model mid-teardown.
            if timed_out:
                # Preserve the prior wire contract: a processing timeout is
                # reported as PROCESSING_TIMEOUT (not a ProcessingResult), uniform
                # with the video worker.
                await self._send_error(
                    websocket, request.request_id,
                    f"Processing timed out after {processing_timeout} seconds",
                    recoverable=False, error_code="PROCESSING_TIMEOUT",
                )
            else:
                await self._send_result(websocket, request.request_id, result)
        except Exception as e:
            # acquire() can raise (unknown model, forced-load OOM, hold failure);
            # emit a failure result rather than letting the client hang (F-D).
            logger.error(f"Audio admission/processing failed: {e}", exc_info=True)
            await self._send_error(
                websocket, request.request_id, "Processing failed",
                recoverable=False, error_code="PROCESSING_FAILED",
            )

    async def _recover_from_timeout(
        self, queued: QueuedRequest, task: asyncio.Future
    ) -> None:
        """A processing timeout fired. Cancel the in-flight work, drain it, and
        recreate any wedged processor — all WHILE STILL HOLDING THE LEASE (the
        required models stay held, op_count>0, so a concurrent admission cannot
        reserve/evict the model being torn down). The client is told
        PROCESSING_TIMEOUT by the caller regardless of the drained outcome, which
        is discarded here."""
        request_id = queued.request.request_id
        set_request_context(request_id=request_id)
        logger.error(
            LogEvents.REQUEST_TIMEOUT,
            data={'stage': 'processing',
                  'timeout_seconds': self.config.queue.processing_timeout}
        )

        # Flag cancellation; the executor jobs unwind at the next _check_cancelled.
        self._whisper.cancel(request_id)
        self._pyannote.cancel(request_id)

        wait_timeout = 30.0
        whisper_idle = await self._whisper.wait_for_idle(wait_timeout)
        pyannote_idle = await self._pyannote.wait_for_idle(wait_timeout)

        # Recreate any WEDGED processor through its handle. The recreate runs
        # under the admission barrier so no concurrent eviction pass touches its
        # shared executor mid-swap (CB2); the barrier is reachable because a
        # parked P0-3 waiter releases the admission lock (BLOCKER-1).
        if not whisper_idle:
            logger.error("Whisper processor stuck after timeout, recreating")
            async with self._arbiter.admission_barrier():
                await self._whisper_handle.recreate()
        if not pyannote_idle:
            logger.error("PyAnnote processor stuck after timeout, recreating")
            async with self._arbiter.admission_barrier():
                await self._pyannote_handle.recreate()

        # Discard the timed-out request's task outcome — the client gets
        # PROCESSING_TIMEOUT regardless. If the processors went idle the task has
        # already finished (result dropped); if we recreated, its executor future
        # is on the dead executor — either way cancel/await detaches it cleanly.
        task.cancel()
        try:
            await task
        except BaseException:
            pass

        self._cleanup_gpu_memory()
        self._current_request = None
        clear_request_context()

    async def stop(self):
        """Stop the worker."""
        self._running = False

        # Shutdown processors (unloads models and shuts down executors)
        self._whisper.shutdown()
        self._pyannote.shutdown()

    async def _process_request(self, queued: QueuedRequest) -> ProcessingResult:
        """
        Process a single request and RETURN its result (the caller, _serve, owns
        sending it — so there is exactly one sender per request, uniform across
        workers). Progress messages are still sent here as work proceeds.

        The required models are guaranteed resident by the arbiter lease the
        caller holds; the processors assert-resident rather than self-load.
        """
        request = queued.request
        websocket = queued.websocket

        self._current_request = request.request_id
        start_time = time.time()

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

        async def send_progress(msg: ProgressMessage):
            nonlocal consecutive_failures
            try:
                await websocket.send(msg.to_json())
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(
                        f"Client appears disconnected ({consecutive_failures} consecutive failures), "
                        f"cancelling processing for {request.request_id}"
                    )
                    self._whisper.cancel(request.request_id)
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
                error_message="Processing failed",  # Don't leak internal details
                processing_time_seconds=processing_time,
            )

        finally:
            self._current_request = None
            clear_request_context()

        return result

    async def _send_result(self, websocket, request_id: str, result: ProcessingResult) -> None:
        """Send the final progress + result to the client (the sole send site)."""
        try:
            await websocket.send(ProgressMessage(
                request_id=request_id,
                stage=ProcessingStage.COMPLETE if result.success else ProcessingStage.FAILED,
                percent=100,
                message="Complete" if result.success else result.error_message,
            ).to_json())
            await websocket.send(result.to_json())
        except Exception as e:
            logger.error(f"Failed to send result: {e}")

    @staticmethod
    async def _send_error(websocket, request_id: str, error: str,
                          recoverable: bool = False,
                          error_code: str = "PROCESSING_FAILED") -> None:
        try:
            await websocket.send(ErrorMessage(
                request_id=request_id,
                error=error,
                recoverable=recoverable,
                error_code=error_code,
            ).to_json())
        except Exception as e:
            logger.warning(f"Failed to send error message: {e}")

    def _cleanup_gpu_memory(self):
        """
        Force GPU memory cleanup after a timeout. May not cancel in-flight
        operations but releases cache after they complete.
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
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
