"""
PyAnnote Speaker Diarization Processor

GPU-accelerated speaker diarization and embedding extraction.
"""
import asyncio
import math
from pathlib import Path
from typing import Optional, List, Dict, Callable, Awaitable, Tuple

from ..config import PyAnnoteConfig
from ..protocol import (
    DiarizationSegment, SpeakerEmbedding, TranscriptSegment,
    ProgressMessage, ProcessingStage
)
from ..logging_config import get_logger
from ..utils.temp_file import TempAudioFile

logger = get_logger(__name__)


# Import from package __init__.py - defined there to avoid duplication
from . import ProcessorCancelled
from .base_processor import BaseProcessor
from ..orchestrator.resident_model import ResidentBinding

PYANNOTE_MODEL_KEY = "pyannote"
PYANNOTE_EMBEDDING_KEY = "pyannote_embedding"


class PyAnnoteProcessor(BaseProcessor):
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
        super().__init__(thread_name_prefix="pyannote_gpu")
        self.config = config
        self._pipeline = None
        self._embedding_model = None

    @property
    def processor_name(self) -> str:
        return "PyAnnote"

    # --- residency (driven by the arbiter via ResidentProcessorHandle) --------

    def resident_bindings(self) -> list[ResidentBinding]:
        """Declare this processor's TWO independently-evictable arbiter
        residents: the diarization pipeline and the speaker-embedding model.
        They MUST have separate load/unload so evicting one does not desync the
        other's loaded-ness (the embedding model is often unused and should be
        evictable on its own). The embedding load uses the server's configured
        HF token (no per-request client token on the arbiter path)."""
        return [
            ResidentBinding(
                key=PYANNOTE_MODEL_KEY,
                estimated_vram_bytes=self.config.estimated_vram_bytes,
                load_fn=self._ensure_pipeline_sync,
                unload_fn=self._unload_pipeline,
            ),
            ResidentBinding(
                key=PYANNOTE_EMBEDDING_KEY,
                estimated_vram_bytes=self.config.embedding_estimated_vram_bytes,
                load_fn=self._ensure_embedding_model_sync,
                unload_fn=self._unload_embedding,
            ),
        ]

    def is_pipeline_loaded(self) -> bool:
        return self._pipeline is not None

    def is_embedding_loaded(self) -> bool:
        return self._embedding_model is not None

    def _unload_pipeline(self) -> None:
        """Free ONLY the diarization pipeline (one of the two residents)."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            logger.info("PyAnnote pipeline unloaded")
            self._empty_cuda_cache()

    def _unload_embedding(self) -> None:
        """Free ONLY the speaker-embedding model (one of the two residents)."""
        if self._embedding_model is not None:
            del self._embedding_model
            self._embedding_model = None
            logger.info("PyAnnote embedding model unloaded")
            self._empty_cuda_cache()

    def _unload_resources(self) -> None:
        """Unload BOTH models — the BaseProcessor.shutdown path frees everything.
        Per-resident eviction goes through _unload_pipeline / _unload_embedding."""
        self._unload_pipeline()
        self._unload_embedding()

    def _ensure_pipeline_sync(self):
        """Load the diarization pipeline if not already loaded (synchronous, runs in executor)."""
        if self._pipeline is not None:
            return

        try:
            # PyTorch 2.6+ compatibility for pyannote model loading
            import torch
            try:
                from torch.version import TorchVersion
                torch.serialization.add_safe_globals([TorchVersion])
            except ImportError:
                # TorchVersion was removed in PyTorch 2.6+
                pass

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

            # Try new API (token) first, fall back to old API (use_auth_token)
            try:
                self._pipeline = Pipeline.from_pretrained(
                    self.config.model,
                    token=self.config.huggingface_token,
                )
            except TypeError:
                # Older pyannote versions use use_auth_token
                self._pipeline = Pipeline.from_pretrained(
                    self.config.model,
                    use_auth_token=self.config.huggingface_token,
                )

            # Support both "cuda" and "cuda:N" device specifications
            if self.config.device.startswith("cuda"):
                import torch
                if torch.cuda.is_available():
                    device = torch.device(self.config.device)
                    self._pipeline = self._pipeline.to(device)
                    logger.info(f"PyAnnote pipeline moved to {self.config.device}")
                else:
                    logger.warning("CUDA requested but not available, using CPU")

            logger.info("PyAnnote pipeline loaded")

        except Exception as e:
            logger.error(f"Failed to load PyAnnote pipeline: {e}")
            raise

    def _ensure_embedding_model_sync(self):
        """Load the embedding model if not already loaded (synchronous, runs on
        the executor via the arbiter's ResidentModel). Uses the server's
        configured HuggingFace token — the arbiter-driven load has no per-request
        client token."""
        if self._embedding_model is not None:
            return

        # Mirror the client's model choice so server-produced embeddings
        # live in the same 256-dim vector space as client-produced ones
        # (e.g. samples extracted by the voice-sample editor). No
        # fallback: a 512-dim emergency model would produce vectors
        # incompatible with the existing registry and find_matches would
        # silently skip every comparison on dimension mismatch. Fail
        # loud instead.
        candidate_models = [
            "pyannote/wespeaker-voxceleb-resnet34-LM",
        ]

        token = self.config.huggingface_token

        last_error: Optional[Exception] = None
        for model_name in candidate_models:
            try:
                from pyannote.audio import Model

                logger.info(f"Trying embedding model: {model_name}")

                try:
                    self._embedding_model = Model.from_pretrained(model_name, token=token)
                except TypeError:
                    self._embedding_model = Model.from_pretrained(model_name, use_auth_token=token)

                # Support both "cuda" and "cuda:N" device specifications
                if self.config.device.startswith("cuda"):
                    import torch
                    if torch.cuda.is_available():
                        device = torch.device(self.config.device)
                        self._embedding_model = self._embedding_model.to(device)
                        logger.info(f"PyAnnote embedding model moved to {self.config.device}")

                logger.info(f"Loaded embedding model {model_name}")
                return
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load {model_name}: {e}")
                self._embedding_model = None

        logger.error(f"Failed to load any embedding model. Last error: {last_error}")
        raise RuntimeError(
            f"Could not load any embedding model. Last error: {last_error}"
        )

    def _diarize_sync(
        self,
        audio_path: Path,
        num_speakers: Optional[int],
        request_id: str,
    ) -> List[DiarizationSegment]:
        """
        Synchronous diarization (runs in executor).

        Returns:
            List of DiarizationSegment with normalized speaker labels

        Raises:
            ProcessorCancelled: If cancel event is set
        """
        import torchaudio

        self._check_cancelled(request_id)

        # Pre-load audio with torchaudio for optimal performance
        # This is much faster than letting PyAnnote decode the file internally
        logger.info(f"Loading audio file: {audio_path}")
        waveform, sample_rate = torchaudio.load(str(audio_path))
        logger.info(f"Loaded audio: shape={waveform.shape}, sample_rate={sample_rate}Hz")

        # Resample to 16kHz if needed (optimal for PyAnnote)
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            logger.info(f"Resampling from {sample_rate}Hz to {target_sample_rate}Hz")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate
            )
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
            logger.info(f"Resampled audio: shape={waveform.shape}")

        # Prepare audio input as waveform tensor (much faster than file path)
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        self._check_cancelled(request_id)

        # Run diarization. Embeddings are extracted in a separate pass via
        # _extract_embeddings_sync (best-segment per speaker) so that each
        # vector corresponds to a single, known audio window rather than a
        # centroid averaged across every segment tagged with the speaker
        # label - the latter contaminates the embedding whenever the
        # diariser clumps multiple real speakers under one label.
        diarization_params = {}
        if num_speakers:
            diarization_params['num_speakers'] = num_speakers

        logger.info("Starting PyAnnote diarization...")
        diarization_result = self._pipeline(audio_input, **diarization_params)

        self._check_cancelled(request_id)

        diarization_output = diarization_result

        # DiarizeOutput (PyAnnote 3.x) wraps Annotation in speaker_diarization attribute
        if hasattr(diarization_output, 'speaker_diarization'):
            diarization = diarization_output.speaker_diarization
            logger.info("Extracted Annotation from DiarizeOutput")
        else:
            diarization = diarization_output

        # Convert to segments
        segments: List[DiarizationSegment] = []
        if hasattr(diarization, 'itertracks'):
            # Standard Annotation object
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(DiarizationSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                ))
        elif hasattr(diarization, 'items'):
            # Dict-like object fallback
            for (segment, track), speaker in diarization.items():
                segments.append(DiarizationSegment(
                    start=segment.start,
                    end=segment.end,
                    speaker=speaker,
                ))
        else:
            raise AttributeError(f"Cannot iterate diarization result of type {type(diarization)}")

        # Normalize speaker labels (SPEAKER_00 -> Person-1)
        speaker_map = {}
        for seg in segments:
            if seg.speaker not in speaker_map:
                speaker_map[seg.speaker] = f"Person-{len(speaker_map) + 1}"
            seg.speaker = speaker_map[seg.speaker]

        # Embeddings are extracted in a separate pass (best-segment per speaker)
        # via _extract_embeddings_sync, not here — centroid aggregation across all
        # segments tagged with one label inherits diariser clumping errors and was
        # the cause of false-positive registry matches.

        logger.info(f"Diarization complete: {len(segments)} segments, {len(speaker_map)} speakers")
        return segments

    async def diarize(
        self,
        audio_data: bytes,
        num_speakers: Optional[int] = None,
        progress_callback: Optional[Callable[[ProgressMessage], Awaitable[None]]] = None,
        request_id: str = "",
    ) -> List[DiarizationSegment]:
        """
        Perform speaker diarization on audio (non-blocking).

        Args:
            audio_data: Raw audio bytes
            num_speakers: Number of speakers hint (None for auto)
            progress_callback: Callback for progress updates
            request_id: Request ID for progress messages

        Returns:
            List of DiarizationSegment

        Raises:
            RuntimeError: If processor has been shut down
            ProcessorCancelled: If processing was cancelled
        """
        self._check_shutdown()
        self._begin_operation()

        # The pipeline is loaded by the arbiter (the caller holds a lease that
        # requires "pyannote"); no self-load — the arbiter owns loaded-ness.
        if not self.is_pipeline_loaded():
            raise RuntimeError(
                "PyAnnote pipeline is not resident — the arbiter lease must "
                "require 'pyannote' before diarize()"
            )

        loop = asyncio.get_running_loop()

        try:
            # Write audio to temp file with robust cleanup
            with TempAudioFile(audio_data, suffix=".opus") as audio_path:
                try:
                    if progress_callback:
                        await progress_callback(ProgressMessage(
                            request_id=request_id,
                            stage=ProcessingStage.DIARIZING,
                            percent=55,
                            message="Starting diarization..."
                        ))

                    # Run diarization in executor (blocking GPU operation)
                    segments = await loop.run_in_executor(
                        self._executor,
                        self._diarize_sync,
                        audio_path,
                        num_speakers,
                        request_id,
                    )

                    if progress_callback:
                        # Count unique speakers
                        unique_speakers = len(set(seg.speaker for seg in segments))
                        await progress_callback(ProgressMessage(
                            request_id=request_id,
                            stage=ProcessingStage.DIARIZING,
                            percent=75,
                            message=f"Diarization complete: {unique_speakers} speakers"
                        ))

                    return segments

                except ProcessorCancelled:
                    logger.warning(f"Diarization cancelled for request {request_id}")
                    raise
                except asyncio.CancelledError:
                    logger.warning(f"Diarization async cancelled for request {request_id}")
                    raise
        finally:
            self._end_operation()

    def _extract_embeddings_sync(
        self,
        audio_path: Path,
        diarization_segments: List[DiarizationSegment],
        meeting_id: str,
        request_id: str,
    ) -> List[SpeakerEmbedding]:
        """
        Synchronous embedding extraction (runs in executor).

        Returns:
            List of SpeakerEmbedding (one per unique speaker)

        Raises:
            ProcessorCancelled: If cancel event is set
        """
        import torch
        import torchaudio
        from pyannote.audio import Inference

        self._check_cancelled(request_id)

        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))

        try:
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
                self._check_cancelled(request_id)

                # Find best segment (longest, for better embedding quality)
                best_seg = max(segs, key=lambda s: s.end - s.start)
                duration = best_seg.end - best_seg.start

                # Skip very short segments
                if duration < 1.0:
                    logger.warning(f"Skipping short segment for {speaker}: {duration:.1f}s")
                    continue

                try:
                    # Extract segment audio - use round() for accurate sample boundaries
                    start_sample = round(best_seg.start * sample_rate)
                    end_sample = round(best_seg.end * sample_rate)
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

                    # Explicitly delete segment audio to free memory
                    del segment_audio
                    del embedding

                except Exception as e:
                    logger.warning(f"Failed to extract embedding for {speaker}: {e}")

            logger.info(f"Extracted {len(embeddings)} speaker embeddings")
            return embeddings

        finally:
            # Explicitly release waveform tensor to free memory
            del waveform
            # Clear GPU cache after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def extract_embeddings(
        self,
        audio_data: bytes,
        diarization_segments: List[DiarizationSegment],
        meeting_id: str = "",
        progress_callback: Optional[Callable[[ProgressMessage], Awaitable[None]]] = None,
        request_id: str = "",
    ) -> List[SpeakerEmbedding]:
        """
        Extract speaker embeddings from audio (non-blocking).

        Args:
            audio_data: Raw audio bytes
            diarization_segments: Diarization results
            meeting_id: Meeting ID for embedding metadata
            progress_callback: Callback for progress updates
            request_id: Request ID for progress messages

        Returns:
            List of SpeakerEmbedding (one per unique speaker)

        Raises:
            RuntimeError: If processor has been shut down or the embedding model
                is not resident
            ProcessorCancelled: If processing was cancelled
        """
        self._check_shutdown()

        # The embedding model is loaded by the arbiter (the caller holds a lease
        # that requires "pyannote_embedding"); no self-load.
        if not self.is_embedding_loaded():
            raise RuntimeError(
                "PyAnnote embedding model is not resident — the arbiter lease "
                "must require 'pyannote_embedding' before extract_embeddings()"
            )

        self._begin_operation()

        loop = asyncio.get_running_loop()

        try:
            # Write audio to temp file with robust cleanup
            with TempAudioFile(audio_data, suffix=".opus") as audio_path:
                try:
                    if progress_callback:
                        await progress_callback(ProgressMessage(
                            request_id=request_id,
                            stage=ProcessingStage.EXTRACTING_EMBEDDINGS,
                            percent=80,
                            message="Extracting speaker embeddings..."
                        ))

                    # Run embedding extraction in executor (blocking GPU operation)
                    embeddings = await loop.run_in_executor(
                        self._executor,
                        self._extract_embeddings_sync,
                        audio_path,
                        diarization_segments,
                        meeting_id,
                        request_id,
                    )

                    if progress_callback:
                        await progress_callback(ProgressMessage(
                            request_id=request_id,
                            stage=ProcessingStage.EXTRACTING_EMBEDDINGS,
                            percent=90,
                            message=f"Extracted {len(embeddings)} embeddings"
                        ))

                    return embeddings

                except ProcessorCancelled:
                    logger.warning(f"Embedding extraction cancelled for request {request_id}")
                    raise
                except asyncio.CancelledError:
                    logger.warning(f"Embedding extraction async cancelled for request {request_id}")
                    raise
        finally:
            self._end_operation()

    def _is_valid_timestamp(self, start: float, end: float) -> bool:
        """Check if timestamps are valid (not NaN, not negative, end >= start)."""
        if math.isnan(start) or math.isnan(end):
            return False
        if start < 0 or end < 0:
            return False
        if end < start:
            return False
        return True

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
        # Filter diarization segments with valid timestamps
        valid_diarization = [
            ds for ds in diarization_segments
            if self._is_valid_timestamp(ds.start, ds.end)
        ]
        if len(valid_diarization) != len(diarization_segments):
            skipped = len(diarization_segments) - len(valid_diarization)
            logger.warning(f"Skipped {skipped} diarization segments with invalid timestamps")

        for ts in transcript_segments:
            # Skip transcript segments with invalid timestamps
            if not self._is_valid_timestamp(ts.start, ts.end):
                logger.warning(f"Skipping transcript segment with invalid timestamps: start={ts.start}, end={ts.end}")
                continue

            # Find overlapping diarization segment
            best_speaker = None
            best_overlap = 0

            for ds in valid_diarization:
                # Calculate overlap
                overlap_start = max(ts.start, ds.start)
                overlap_end = min(ts.end, ds.end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = ds.speaker

            ts.speaker = best_speaker

        return transcript_segments
