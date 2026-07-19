"""
Whisper Transcription Processor

GPU-accelerated speech-to-text using faster-whisper.
"""
import asyncio
from pathlib import Path
from typing import Optional, List, Callable, Awaitable, Tuple

from ..config import WhisperConfig
from ..protocol import TranscriptSegment, ProgressMessage, ProcessingStage
from ..logging_config import get_logger
from ..utils.temp_file import TempAudioFile

logger = get_logger(__name__)


# Import from package __init__.py - defined there to avoid duplication
# This import happens after __init__.py defines ProcessorCancelled but before
# it tries to import WhisperProcessor, so no circular import issue
from . import ProcessorCancelled
from .base_processor import BaseProcessor
from ..orchestrator.resident_model import ResidentBinding
from ..whisper_models import resolve_effective, WhisperModelSwapError

WHISPER_MODEL_KEY = "whisper"


class WhisperProcessor(BaseProcessor):
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
        super().__init__(thread_name_prefix="whisper_gpu")
        self.config = config
        self._model = None
        # Which variant is currently loaded, and the variant the NEXT cold load
        # should build (threaded by the worker before acquire, so a cold load
        # builds the requested variant directly instead of loading the ceiling
        # then swapping down). config.model is the CEILING; any variant is ≤ it.
        self._model_name: Optional[str] = None
        self._pending_model: Optional[str] = None

    @property
    def processor_name(self) -> str:
        return "Whisper"

    # --- residency (driven by the arbiter via ResidentProcessorHandle) --------

    def resident_bindings(self) -> list[ResidentBinding]:
        """Declare this processor's single arbiter resident. The handle builds
        the ResidentModel from this; the arbiter loads/evicts it. Bound to
        ``config.model`` — the server runs exactly the configured model (see
        ``transcribe`` for why per-request overrides are dropped)."""
        return [
            ResidentBinding(
                key=WHISPER_MODEL_KEY,
                estimated_vram_bytes=self.config.estimated_vram_bytes,
                load_fn=self._load_model_sync,
                unload_fn=self._unload_resources,
            )
        ]

    def is_loaded(self) -> bool:
        return self._model is not None

    def request_model(self, requested: Optional[str]) -> None:
        """Thread the client-requested model to the NEXT cold load. The RAW
        request is passed (the processor owns the ceiling/resolve). Set by the
        (serial) audio worker before acquire; a cold load builds the resolved
        variant directly, avoiding a load-ceiling-then-swap double load."""
        self._pending_model = requested

    def _unload_resources(self) -> None:
        """Unload the model to free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_name = None
            logger.info("Whisper model unloaded")
            self._empty_cuda_cache()

    def _build_model(self, name: str):
        from faster_whisper import WhisperModel
        logger.info(f"Loading Whisper model: {name}")
        model = WhisperModel(name, device=self.config.device,
                             compute_type=self.config.compute_type)
        logger.info(f"Whisper model loaded: {name}")
        return model

    def _load_model_sync(self) -> None:
        """Cold load (blocking; runs on the executor). Builds the resolved variant
        of the pending request, capped at the ceiling (config.model). Idempotent."""
        if self._model is not None:
            return  # Already loaded
        target = resolve_effective(self._pending_model, self.config.model)
        try:
            self._model = self._build_model(target)
            self._model_name = target
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _swap_sync(self, target: str) -> None:
        """Swap the loaded variant to ``target`` as ONE executor task (F-D):
        unload the current model FIRST (peak VRAM ≤ the ceiling reservation), then
        build the target. On a build failure the processor is left with NO model —
        raise WhisperModelSwapError so the worker recreates it (the arbiter still
        counts whisper resident, so it won't reload on its own: F-A)."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_name = None
            self._empty_cuda_cache()
        try:
            self._model = self._build_model(target)
            self._model_name = target
        except Exception as e:
            logger.error(f"Whisper model swap to '{target}' failed: {e}")
            raise WhisperModelSwapError(str(e)) from e

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
        self._check_shutdown()

        # config.model is the CEILING: honour a smaller requested model, pin a
        # bigger one to the ceiling.
        effective = resolve_effective(model_override, self.config.model)

        # The model is loaded by the arbiter (the caller holds a lease that
        # requires "whisper"); this processor does not self-load. Fail fast if
        # that contract is broken — BEFORE _begin_operation so a raise can't leak
        # the idle-tracking flag.
        if not self.is_loaded():
            raise RuntimeError(
                "Whisper model is not resident — the arbiter lease must require "
                "'whisper' before transcribe()"
            )

        self._begin_operation()
        loop = asyncio.get_running_loop()

        try:
            # If the resident variant differs from the requested one (a warm
            # resident from a prior request wanting a different model), swap it —
            # ONE executor task, under the held lease (op_count>0 → the arbiter
            # can't evict whisper), VRAM peak ≤ the fixed ceiling reservation. The
            # arbiter's residency accounting is unchanged (a whisper IS loaded); a
            # swap FAILURE raises WhisperModelSwapError → the worker recreates the
            # processor to resync (F-A).
            if self._model_name != effective:
                await loop.run_in_executor(self._executor, self._swap_sync, effective)
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
            self._end_operation()
