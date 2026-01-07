"""
PyAnnote Speaker Diarization Processor

GPU-accelerated speaker diarization and embedding extraction.
"""
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Callable, Awaitable

import numpy as np

from ..config import PyAnnoteConfig
from ..protocol import (
    DiarizationSegment, SpeakerEmbedding, TranscriptSegment,
    ProgressMessage, ProcessingStage
)

logger = logging.getLogger(__name__)


class PyAnnoteProcessor:
    """
    PyAnnote speaker diarization processor.

    Features:
    - GPU-accelerated diarization
    - Speaker embedding extraction
    - Transcript-diarization alignment
    """

    def __init__(self, config: PyAnnoteConfig):
        """
        Initialize the PyAnnote processor.

        Args:
            config: PyAnnote configuration
        """
        self.config = config
        self._pipeline = None
        self._embedding_model = None

    def _ensure_pipeline(self):
        """Load the diarization pipeline if not already loaded."""
        if self._pipeline is not None:
            return

        try:
            # PyTorch 2.6+ compatibility for pyannote model loading
            import torch
            from torch.version import TorchVersion
            torch.serialization.add_safe_globals([TorchVersion])

            # Additional pyannote classes that need whitelisting
            try:
                from pyannote.audio.core import task as pyannote_task
                for cls_name in ['Specifications', 'Problem', 'Resolution', 'Task']:
                    if hasattr(pyannote_task, cls_name):
                        torch.serialization.add_safe_globals([getattr(pyannote_task, cls_name)])
            except ImportError:
                pass

            from pyannote.audio import Pipeline

            logger.info(f"Loading PyAnnote pipeline: {self.config.model}")

            self._pipeline = Pipeline.from_pretrained(
                self.config.model,
                use_auth_token=self.config.huggingface_token,
            )

            if self.config.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self._pipeline = self._pipeline.to(torch.device("cuda"))
                    logger.info("PyAnnote pipeline moved to CUDA")
                else:
                    logger.warning("CUDA requested but not available, using CPU")

            logger.info("PyAnnote pipeline loaded")

        except Exception as e:
            logger.error(f"Failed to load PyAnnote pipeline: {e}")
            raise

    def _ensure_embedding_model(self):
        """Load the embedding model if not already loaded."""
        if self._embedding_model is not None:
            return

        try:
            from pyannote.audio import Model

            logger.info("Loading PyAnnote embedding model")

            self._embedding_model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=self.config.huggingface_token,
            )

            if self.config.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self._embedding_model = self._embedding_model.to(torch.device("cuda"))

            logger.info("PyAnnote embedding model loaded")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    async def diarize(
        self,
        audio_data: bytes,
        num_speakers: Optional[int] = None,
        progress_callback: Optional[Callable[[ProgressMessage], Awaitable[None]]] = None,
        request_id: str = "",
    ) -> List[DiarizationSegment]:
        """
        Perform speaker diarization on audio.

        Args:
            audio_data: Raw audio bytes
            num_speakers: Number of speakers hint (None for auto)
            progress_callback: Callback for progress updates
            request_id: Request ID for progress messages

        Returns:
            List of DiarizationSegment
        """
        self._ensure_pipeline()

        # Write audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as f:
            f.write(audio_data)
            audio_path = Path(f.name)

        try:
            if progress_callback:
                await progress_callback(ProgressMessage(
                    request_id=request_id,
                    stage=ProcessingStage.DIARIZING,
                    percent=55,
                    message="Starting diarization..."
                ))

            # Run diarization
            diarization_params = {}
            if num_speakers:
                diarization_params['num_speakers'] = num_speakers

            diarization = self._pipeline(str(audio_path), **diarization_params)

            # Convert to segments
            segments: List[DiarizationSegment] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(DiarizationSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                ))

            # Normalize speaker labels (SPEAKER_00 -> Person-1)
            speaker_map = {}
            for seg in segments:
                if seg.speaker not in speaker_map:
                    speaker_map[seg.speaker] = f"Person-{len(speaker_map) + 1}"
                seg.speaker = speaker_map[seg.speaker]

            if progress_callback:
                await progress_callback(ProgressMessage(
                    request_id=request_id,
                    stage=ProcessingStage.DIARIZING,
                    percent=75,
                    message=f"Diarization complete: {len(speaker_map)} speakers"
                ))

            logger.info(f"Diarization complete: {len(segments)} segments, {len(speaker_map)} speakers")
            return segments

        finally:
            audio_path.unlink(missing_ok=True)

    async def extract_embeddings(
        self,
        audio_data: bytes,
        diarization_segments: List[DiarizationSegment],
        meeting_id: str = "",
        progress_callback: Optional[Callable[[ProgressMessage], Awaitable[None]]] = None,
        request_id: str = "",
    ) -> List[SpeakerEmbedding]:
        """
        Extract speaker embeddings from audio.

        Args:
            audio_data: Raw audio bytes
            diarization_segments: Diarization results
            meeting_id: Meeting ID for embedding metadata
            progress_callback: Callback for progress updates
            request_id: Request ID for progress messages

        Returns:
            List of SpeakerEmbedding (one per unique speaker)
        """
        self._ensure_embedding_model()

        # Write audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as f:
            f.write(audio_data)
            audio_path = Path(f.name)

        try:
            if progress_callback:
                await progress_callback(ProgressMessage(
                    request_id=request_id,
                    stage=ProcessingStage.EXTRACTING_EMBEDDINGS,
                    percent=80,
                    message="Extracting speaker embeddings..."
                ))

            import torch
            import torchaudio
            from pyannote.audio import Inference

            # Load audio
            waveform, sample_rate = torchaudio.load(str(audio_path))

            # Create inference object
            inference = Inference(self._embedding_model, window="whole")

            # Group segments by speaker
            speaker_segments: Dict[str, List[DiarizationSegment]] = {}
            for seg in diarization_segments:
                if seg.speaker not in speaker_segments:
                    speaker_segments[seg.speaker] = []
                speaker_segments[seg.speaker].append(seg)

            embeddings: List[SpeakerEmbedding] = []

            for speaker, segs in speaker_segments.items():
                # Find best segment (longest, for better embedding quality)
                best_seg = max(segs, key=lambda s: s.end - s.start)
                duration = best_seg.end - best_seg.start

                # Skip very short segments
                if duration < 1.0:
                    logger.warning(f"Skipping short segment for {speaker}: {duration:.1f}s")
                    continue

                try:
                    # Extract segment audio
                    start_sample = int(best_seg.start * sample_rate)
                    end_sample = int(best_seg.end * sample_rate)
                    segment_audio = waveform[:, start_sample:end_sample]

                    # Get embedding
                    with torch.no_grad():
                        embedding = inference({"waveform": segment_audio, "sample_rate": sample_rate})

                    # Convert to list for JSON serialization
                    embedding_list = embedding.flatten().tolist()

                    embeddings.append(SpeakerEmbedding(
                        speaker_label=speaker,
                        meeting_id=meeting_id,
                        segment_start=best_seg.start,
                        segment_duration=duration,
                        embedding=embedding_list,
                        quality_score=min(1.0, duration / 10.0),  # Longer = better quality
                    ))

                except Exception as e:
                    logger.warning(f"Failed to extract embedding for {speaker}: {e}")

            if progress_callback:
                await progress_callback(ProgressMessage(
                    request_id=request_id,
                    stage=ProcessingStage.EXTRACTING_EMBEDDINGS,
                    percent=90,
                    message=f"Extracted {len(embeddings)} embeddings"
                ))

            logger.info(f"Extracted {len(embeddings)} speaker embeddings")
            return embeddings

        finally:
            audio_path.unlink(missing_ok=True)

    def align_transcript_with_diarization(
        self,
        transcript_segments: List[TranscriptSegment],
        diarization_segments: List[DiarizationSegment],
    ) -> List[TranscriptSegment]:
        """
        Align transcript segments with diarization to assign speakers.

        Args:
            transcript_segments: Whisper transcript segments
            diarization_segments: PyAnnote diarization segments

        Returns:
            Transcript segments with speaker assignments
        """
        for ts in transcript_segments:
            ts_mid = (ts.start + ts.end) / 2

            # Find overlapping diarization segment
            best_speaker = None
            best_overlap = 0

            for ds in diarization_segments:
                # Calculate overlap
                overlap_start = max(ts.start, ds.start)
                overlap_end = min(ts.end, ds.end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = ds.speaker

            ts.speaker = best_speaker

        return transcript_segments

    def unload(self):
        """Unload models to free GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            logger.info("PyAnnote pipeline unloaded")

        if self._embedding_model is not None:
            del self._embedding_model
            self._embedding_model = None
            logger.info("PyAnnote embedding model unloaded")

        # Force GPU memory cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
