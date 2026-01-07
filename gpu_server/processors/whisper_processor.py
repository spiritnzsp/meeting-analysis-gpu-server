"""
Whisper Transcription Processor

GPU-accelerated speech-to-text using faster-whisper.
"""
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Callable, Awaitable

from ..config import WhisperConfig
from ..protocol import TranscriptSegment, ProgressMessage, ProcessingStage

logger = logging.getLogger(__name__)


class WhisperProcessor:
    """
    Whisper transcription processor using faster-whisper.

    Features:
    - GPU acceleration with CUDA
    - Word-level timestamps
    - Language detection
    - Multiple model sizes
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

    def _ensure_model(self, model_name: Optional[str] = None):
        """Load the model if not already loaded."""
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

    async def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        model_override: Optional[str] = None,
        progress_callback: Optional[Callable[[ProgressMessage], Awaitable[None]]] = None,
        request_id: str = "",
    ) -> tuple[List[TranscriptSegment], str, str]:
        """
        Transcribe audio data.

        Args:
            audio_data: Raw audio bytes (opus, wav, etc.)
            language: Force language (None for auto-detect)
            model_override: Override default model
            progress_callback: Callback for progress updates
            request_id: Request ID for progress messages

        Returns:
            Tuple of (segments, full_text, detected_language)
        """
        # Load model
        self._ensure_model(model_override)

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

            # Run transcription
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                language=language or self.config.language,
                beam_size=self.config.beam_size,
                word_timestamps=True,
                vad_filter=True,
            )

            detected_language = info.language
            logger.info(f"Detected language: {detected_language} (prob: {info.language_probability:.2f})")

            # Convert segments
            segments: List[TranscriptSegment] = []
            full_text_parts = []

            segment_list = list(segments_iter)
            total_segments = len(segment_list)

            for i, segment in enumerate(segment_list):
                # Extract word-level timestamps
                words = []
                if segment.words:
                    for word in segment.words:
                        words.append({
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability,
                        })

                segments.append(TranscriptSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    confidence=segment.avg_logprob if hasattr(segment, 'avg_logprob') else 1.0,
                    words=words,
                ))
                full_text_parts.append(segment.text.strip())

                # Progress update every 10 segments
                if progress_callback and i % 10 == 0:
                    percent = 10 + int((i / max(total_segments, 1)) * 40)  # 10-50%
                    await progress_callback(ProgressMessage(
                        request_id=request_id,
                        stage=ProcessingStage.TRANSCRIBING,
                        percent=percent,
                        message=f"Transcribed {i+1}/{total_segments} segments"
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
