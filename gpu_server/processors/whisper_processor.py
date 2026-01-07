"""
Whisper Transcription Processor

GPU-accelerated speech-to-text using faster-whisper.
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Callable, Awaitable, Tuple

from ..config import WhisperConfig
from ..protocol import TranscriptSegment, ProgressMessage, ProcessingStage
from ..logging_config import get_logger, LogEvents
from ..utils.temp_file import TempAudioFile

logger = get_logger(__name__)


# Import from package __init__.py - defined there to avoid duplication
# This import happens after __init__.py defines ProcessorCancelled but before
# it tries to import WhisperProcessor, so no circular import issue
from . import ProcessorCancelled


class WhisperProcessor:
    """
    Whisper transcription processor using faster-whisper.

    Features:
    - GPU acceleration with CUDA
    - Word-level timestamps
    - Language detection
    - Multiple model sizes
    - Non-blocking async interface (GPU ops run in executor)
    - Safe cancellation with operation tracking
    """

    def __init__(self, config: WhisperConfig):
        """
        Initialize the Whisper processor.

        Args:
            config: Whisper configuration
        """
        self.config = config
        self._model = None
        self._model_name = None
        self._shutdown = False

        # Cancellation tracking - use request_id to avoid TOCTOU race
        # When cancel is called, we store the request_id that should be cancelled
        # This avoids the race where clear() loses a cancellation
        self._cancelled_request_id: Optional[str] = None
        self._cancel_lock = threading.Lock()

        # Operation tracking for safe timeout handling
        self._operation_complete = asyncio.Event()
        self._operation_complete.set()  # Initially not processing
        self._is_processing = False

        # Dedicated thread pool for GPU operations to avoid blocking event loop
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper_gpu")

    def _ensure_model_sync(self, model_name: Optional[str] = None):
        """Load the model if not already loaded (synchronous, runs in executor)."""
        target_model = model_name or self.config.model

        if self._model is not None and self._model_name == target_model:
            return  # Already loaded

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model: {target_model}")
            self._model = WhisperModel(
                target_model,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            self._model_name = target_model
            logger.info(f"Whisper model loaded: {target_model}")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _check_cancelled(self, request_id: str) -> None:
        """Check if processing was cancelled and raise if so."""
        with self._cancel_lock:
            # Check for specific request cancellation or wildcard cancellation
            if self._cancelled_request_id == request_id or self._cancelled_request_id == "__ANY__":
                raise ProcessorCancelled("Transcription cancelled")

    def _transcribe_sync(
        self,
        audio_path: Path,
        language: Optional[str],
        request_id: str,
    ) -> Tuple[List[dict], str, float]:
        """
        Synchronous transcription (runs in executor).

        Returns:
            Tuple of (raw_segments, detected_language, language_probability)

        Raises:
            ProcessorCancelled: If cancel event is set
        """
        self._check_cancelled(request_id)

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language or self.config.language,
            beam_size=self.config.beam_size,
            word_timestamps=True,
            vad_filter=True,
        )

        # Convert iterator to list (this consumes the generator)
        # Check for cancellation periodically between segments
        raw_segments = []
        for segment in segments_iter:
            self._check_cancelled(request_id)

            words = []
            if segment.words:
                for word in segment.words:
                    words.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    })

            raw_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "avg_logprob": getattr(segment, 'avg_logprob', 0.0),
                "words": words,
            })

        return raw_segments, info.language, info.language_probability

    async def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        model_override: Optional[str] = None,
        progress_callback: Optional[Callable[[ProgressMessage], Awaitable[None]]] = None,
        request_id: str = "",
    ) -> Tuple[List[TranscriptSegment], str, str]:
        """
        Transcribe audio data (non-blocking).

        Args:
            audio_data: Raw audio bytes (opus, wav, etc.)
            language: Force language (None for auto-detect)
            model_override: Override default model
            progress_callback: Callback for progress updates
            request_id: Request ID for progress messages

        Returns:
            Tuple of (segments, full_text, detected_language)

        Raises:
            RuntimeError: If processor has been shut down
            ProcessorCancelled: If processing was cancelled
        """
        if self._shutdown:
            raise RuntimeError("WhisperProcessor has been shut down")

        # Mark operation as in-progress (for timeout handling)
        self._operation_complete.clear()
        self._is_processing = True

        # Clear any stale cancellation from previous request
        with self._cancel_lock:
            self._cancelled_request_id = None

        loop = asyncio.get_running_loop()

        try:
            # Load model in executor (blocking operation)
            await loop.run_in_executor(
                self._executor,
                self._ensure_model_sync,
                model_override
            )

            # Write audio to temp file with robust cleanup (faster-whisper needs file path)
            with TempAudioFile(audio_data, suffix=".opus") as audio_path:
                try:
                    if progress_callback:
                        await progress_callback(ProgressMessage(
                            request_id=request_id,
                            stage=ProcessingStage.TRANSCRIBING,
                            percent=10,
                            message="Starting transcription..."
                        ))

                    # Run transcription in executor (blocking GPU operation)
                    raw_segments, detected_language, lang_prob = await loop.run_in_executor(
                        self._executor,
                        self._transcribe_sync,
                        audio_path,
                        language,
                        request_id,
                    )

                    logger.info(f"Detected language: {detected_language} (prob: {lang_prob:.2f})")

                    # Convert to TranscriptSegment objects (non-blocking)
                    segments: List[TranscriptSegment] = []
                    full_text_parts = []

                    for i, raw_seg in enumerate(raw_segments):
                        segments.append(TranscriptSegment(
                            start=raw_seg["start"],
                            end=raw_seg["end"],
                            text=raw_seg["text"],
                            confidence=raw_seg["avg_logprob"],
                            words=raw_seg["words"],
                        ))
                        full_text_parts.append(raw_seg["text"])

                        # Progress update every 10 segments
                        if progress_callback and i % 10 == 0:
                            percent = 10 + int((i / max(len(raw_segments), 1)) * 40)  # 10-50%
                            await progress_callback(ProgressMessage(
                                request_id=request_id,
                                stage=ProcessingStage.TRANSCRIBING,
                                percent=percent,
                                message=f"Processing {i+1}/{len(raw_segments)} segments"
                            ))

                    full_text = " ".join(full_text_parts)

                    if progress_callback:
                        await progress_callback(ProgressMessage(
                            request_id=request_id,
                            stage=ProcessingStage.TRANSCRIBING,
                            percent=50,
                            message="Transcription complete"
                        ))

                    logger.info(f"Transcription complete: {len(segments)} segments, {len(full_text)} chars")
                    return segments, full_text, detected_language

                except ProcessorCancelled:
                    logger.warning(f"Transcription cancelled for request {request_id}")
                    raise
                except asyncio.CancelledError:
                    logger.warning(f"Transcription async cancelled for request {request_id}")
                    raise
        finally:
            # Mark operation as complete
            self._is_processing = False
            self._operation_complete.set()

    def cancel(self, request_id: str = "") -> None:
        """
        Cancel processing for a specific request.

        Args:
            request_id: The request ID to cancel. If empty, cancels any in-progress request.
        """
        with self._cancel_lock:
            if request_id:
                self._cancelled_request_id = request_id
            elif self._is_processing:
                # Cancel whatever is currently processing
                self._cancelled_request_id = "__ANY__"
        logger.info(f"Whisper processor cancellation requested for {request_id or 'current request'}")

    async def wait_for_idle(self, timeout: float = 30.0) -> bool:
        """
        Wait for the processor to become idle.

        Use this after cancelling to ensure GPU operations have completed
        before starting a new request.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if processor is idle, False if timeout occurred
        """
        if not self._is_processing:
            return True

        try:
            await asyncio.wait_for(self._operation_complete.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for Whisper processor to become idle")
            return False

    @property
    def is_processing(self) -> bool:
        """Check if processor is currently running a GPU operation."""
        return self._is_processing

    def unload(self):
        """Unload the model to free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_name = None
            logger.info("Whisper model unloaded")

            # Force GPU memory cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def shutdown(self, timeout: float = 30.0):
        """
        Shutdown the processor and release all resources. Safe to call multiple times.

        Args:
            timeout: Maximum time to wait for executor shutdown (default 30s)
        """
        if self._shutdown:
            return  # Already shut down

        self._shutdown = True

        # Signal cancellation to any running operations
        self.cancel()

        self.unload()

        # Shutdown the executor with timeout
        if self._executor is not None:
            logger.info("Shutting down Whisper executor...")

            # Use a thread to implement timeout since ThreadPoolExecutor.shutdown() doesn't have one
            shutdown_complete = threading.Event()

            def do_shutdown():
                try:
                    self._executor.shutdown(wait=True, cancel_futures=False)
                finally:
                    shutdown_complete.set()

            shutdown_thread = threading.Thread(target=do_shutdown, daemon=True)
            shutdown_thread.start()

            if shutdown_complete.wait(timeout=timeout):
                logger.info("Whisper executor shutdown complete")
            else:
                logger.warning(
                    f"Whisper executor shutdown timed out after {timeout}s, "
                    "forcing shutdown (operations may still be running)"
                )
                # Force shutdown without waiting
                self._executor.shutdown(wait=False, cancel_futures=True)

            self._executor = None
