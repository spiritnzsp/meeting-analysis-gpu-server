"""
Whisper Transcription Processor

GPU-accelerated speech-to-text using faster-whisper.
"""
import asyncio
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Callable, Awaitable, Tuple

from ..config import WhisperConfig
from ..protocol import TranscriptSegment, ProgressMessage, ProcessingStage

logger = logging.getLogger(__name__)

# Dedicated thread pool for GPU operations to avoid blocking event loop
_gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper_gpu")


class WhisperProcessor:
    """
    Whisper transcription processor using faster-whisper.

    Features:
    - GPU acceleration with CUDA
    - Word-level timestamps
    - Language detection
    - Multiple model sizes
    - Non-blocking async interface (GPU ops run in executor)
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

    def _transcribe_sync(
        self,
        audio_path: Path,
        language: Optional[str],
    ) -> Tuple[List[dict], str, float]:
        """
        Synchronous transcription (runs in executor).

        Returns:
            Tuple of (raw_segments, detected_language, language_probability)
        """
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language or self.config.language,
            beam_size=self.config.beam_size,
            word_timestamps=True,
            vad_filter=True,
        )

        # Convert iterator to list (this consumes the generator)
        raw_segments = []
        for segment in segments_iter:
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
        """
        loop = asyncio.get_running_loop()

        # Load model in executor (blocking operation)
        await loop.run_in_executor(
            _gpu_executor,
            self._ensure_model_sync,
            model_override
        )

        # Write audio to temp file (faster-whisper needs file path)
        with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as f:
            f.write(audio_data)
            audio_path = Path(f.name)

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
                _gpu_executor,
                self._transcribe_sync,
                audio_path,
                language,
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

        finally:
            # Clean up temp file
            audio_path.unlink(missing_ok=True)

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
