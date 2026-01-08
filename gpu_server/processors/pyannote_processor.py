"""
PyAnnote Speaker Diarization Processor

GPU-accelerated speaker diarization and embedding extraction.
"""
import asyncio
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict, Callable, Awaitable, Tuple

import numpy as np

from ..config import PyAnnoteConfig
from ..protocol import (
    DiarizationSegment, SpeakerEmbedding, TranscriptSegment,
    ProgressMessage, ProcessingStage
)
from ..logging_config import get_logger, LogEvents
from ..utils.temp_file import TempAudioFile

logger = get_logger(__name__)


# Import from package __init__.py - defined there to avoid duplication
from . import ProcessorCancelled


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
        self._shutdown = False

        # Cancellation tracking - use request_id to avoid TOCTOU race
        self._cancelled_request_id: Optional[str] = None
        self._cancel_lock = threading.Lock()

        # Operation tracking for safe timeout handling
        self._operation_complete = asyncio.Event()
        self._operation_complete.set()  # Initially not processing
        self._is_processing = False

        # Store pre-computed embeddings from diarization (avoids loading separate model)
        self._last_embeddings: Dict[str, List[tuple]] = {}
        self._last_speaker_map: Dict[str, str] = {}  # SPEAKER_00 -> Person-1 mapping

        # Dedicated thread pool for GPU operations to avoid blocking event loop
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pyannote_gpu")

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

    def _ensure_embedding_model_sync(self, client_hf_token: Optional[str] = None):
        """Load the embedding model if not already loaded (synchronous, runs in executor).

        Args:
            client_hf_token: Optional token from client (used if model not cached).
                             Once model is downloaded, token is no longer needed.
        """
        if self._embedding_model is not None:
            return

        try:
            from pyannote.audio import Model

            logger.info("Loading PyAnnote embedding model")

            # Use client's token if provided, otherwise fall back to server config
            # Client token is especially useful for first-time model download
            token = client_hf_token or self.config.huggingface_token
            if client_hf_token:
                logger.info("Using client-provided HuggingFace token for embedding model")

            # Try new API (token) first, fall back to old API (use_auth_token)
            try:
                self._embedding_model = Model.from_pretrained(
                    "pyannote/embedding",
                    token=token,
                )
            except TypeError:
                # Older pyannote versions use use_auth_token
                self._embedding_model = Model.from_pretrained(
                    "pyannote/embedding",
                    use_auth_token=token,
                )

            # Support both "cuda" and "cuda:N" device specifications
            if self.config.device.startswith("cuda"):
                import torch
                if torch.cuda.is_available():
                    device = torch.device(self.config.device)
                    self._embedding_model = self._embedding_model.to(device)
                    logger.info(f"PyAnnote embedding model moved to {self.config.device}")

            logger.info("PyAnnote embedding model loaded")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _check_cancelled(self, request_id: str) -> None:
        """Check if processing was cancelled and raise if so."""
        with self._cancel_lock:
            if self._cancelled_request_id == request_id or self._cancelled_request_id == "__ANY__":
                raise ProcessorCancelled("Processing cancelled")

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

        # Run diarization
        diarization_params = {}
        if num_speakers:
            diarization_params['num_speakers'] = num_speakers

        # Request embeddings to be returned with diarization result
        # This is essential for the attendee registry speaker identification feature
        diarization_params['return_embeddings'] = True

        logger.info("Starting PyAnnote diarization...")
        diarization_result = self._pipeline(audio_input, **diarization_params)

        self._check_cancelled(request_id)

        # Handle different PyAnnote return types
        # When return_embeddings=True, PyAnnote returns (Annotation/DiarizeOutput, embeddings_tensor)
        raw_embeddings_from_result = None
        if isinstance(diarization_result, tuple) and len(diarization_result) == 2:
            logger.info("Received tuple result from PyAnnote (diarization, embeddings)")
            diarization_output, raw_embeddings_from_result = diarization_result
            logger.info(f"Embeddings tensor type: {type(raw_embeddings_from_result)}, "
                       f"shape: {raw_embeddings_from_result.shape if hasattr(raw_embeddings_from_result, 'shape') else 'N/A'}")
        else:
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

        # Store speaker map for embedding key conversion
        self._last_speaker_map = speaker_map

        # Extract pre-computed embeddings from diarization result
        # This avoids needing to load a separate embedding model
        self._last_embeddings = {}
        try:
            # Use embeddings from tuple result (return_embeddings=True)
            # or fall back to speaker_embeddings attribute (DiarizeOutput)
            raw_embeddings = raw_embeddings_from_result
            if raw_embeddings is None:
                raw_embeddings = getattr(diarization_output, 'speaker_embeddings', None)

            if raw_embeddings is not None:
                logger.info(f"Found pre-computed embeddings: type={type(raw_embeddings)}, "
                           f"shape={raw_embeddings.shape if hasattr(raw_embeddings, 'shape') else 'N/A'}")

                # Get unique speakers for index mapping
                unique_speakers = list(speaker_map.keys())

                # Build a map of best segment per speaker (longest segment for quality)
                # Segments already have normalized labels at this point
                speaker_best_segment: Dict[str, DiarizationSegment] = {}
                for seg in segments:
                    duration = seg.end - seg.start
                    if seg.speaker not in speaker_best_segment:
                        speaker_best_segment[seg.speaker] = seg
                    elif duration > (speaker_best_segment[seg.speaker].end - speaker_best_segment[seg.speaker].start):
                        speaker_best_segment[seg.speaker] = seg

                # Handle SlidingWindowFeature format (frame-level embeddings from return_embeddings=True)
                if hasattr(raw_embeddings, 'data') and hasattr(raw_embeddings, 'sliding_window'):
                    # SlidingWindowFeature: compute per-speaker centroids
                    logger.info("Processing SlidingWindowFeature embeddings...")
                    frame_embeddings = raw_embeddings.data  # shape (num_frames, embedding_dim)
                    sliding_window = raw_embeddings.sliding_window
                    logger.info(f"Frame embeddings shape: {frame_embeddings.shape}, "
                               f"num_frames: {len(frame_embeddings)}")

                    # At this point, segments have normalized labels (Person-1, Person-2, etc.)
                    # Compute speaker centroids by averaging frame embeddings within speaker segments
                    normalized_speakers = set(seg.speaker for seg in segments)
                    for normalized_label in normalized_speakers:
                        speaker_frames = []
                        for seg in segments:
                            if seg.speaker == normalized_label:
                                # Find frames within this segment's time range
                                for frame_idx in range(len(frame_embeddings)):
                                    frame_start = sliding_window[frame_idx].start
                                    frame_end = sliding_window[frame_idx].end
                                    frame_mid = (frame_start + frame_end) / 2
                                    if seg.start <= frame_mid <= seg.end:
                                        speaker_frames.append(frame_embeddings[frame_idx])

                        if speaker_frames:
                            # Compute centroid (mean of frame embeddings)
                            centroid = np.mean(speaker_frames, axis=0)
                            # Get timing from best segment
                            best_seg = speaker_best_segment.get(normalized_label)
                            if best_seg:
                                seg_start = best_seg.start
                                seg_duration = best_seg.end - best_seg.start
                                quality = min(1.0, seg_duration / 10.0)
                            else:
                                seg_start, seg_duration, quality = 0.0, 0.0, 0.8
                            self._last_embeddings[normalized_label] = [(centroid, seg_start, seg_duration, quality)]
                            logger.info(f"Computed centroid for {normalized_label}: dim={len(centroid)}, "
                                       f"frames={len(speaker_frames)}, segment={seg_start:.1f}-{seg_start+seg_duration:.1f}s")

                elif isinstance(raw_embeddings, np.ndarray) and len(raw_embeddings) > 0:
                    # Embeddings are array: shape (num_speakers, embedding_dim)
                    for idx, orig_label in enumerate(unique_speakers):
                        if idx < len(raw_embeddings):
                            emb_np = raw_embeddings[idx]
                            if hasattr(emb_np, 'cpu'):
                                emb_np = emb_np.cpu().numpy()
                            normalized_label = speaker_map[orig_label]
                            # Get timing from best segment
                            best_seg = speaker_best_segment.get(normalized_label)
                            if best_seg:
                                seg_start = best_seg.start
                                seg_duration = best_seg.end - best_seg.start
                                quality = min(1.0, seg_duration / 10.0)
                            else:
                                seg_start, seg_duration, quality = 0.0, 0.0, 0.8
                            self._last_embeddings[normalized_label] = [(emb_np, seg_start, seg_duration, quality)]
                            logger.info(f"Stored embedding for {normalized_label}: dim={len(emb_np)}, "
                                       f"segment={seg_start:.1f}-{seg_start+seg_duration:.1f}s")

                elif hasattr(raw_embeddings, 'items'):
                    # Dict format
                    for orig_label, embedding in raw_embeddings.items():
                        if hasattr(embedding, 'cpu'):
                            emb_np = embedding.cpu().numpy()
                        elif hasattr(embedding, 'numpy'):
                            emb_np = embedding.numpy()
                        else:
                            emb_np = np.array(embedding)
                        normalized_label = speaker_map.get(orig_label, orig_label)
                        # Get timing from best segment
                        best_seg = speaker_best_segment.get(normalized_label)
                        if best_seg:
                            seg_start = best_seg.start
                            seg_duration = best_seg.end - best_seg.start
                            quality = min(1.0, seg_duration / 10.0)
                        else:
                            seg_start, seg_duration, quality = 0.0, 0.0, 0.8
                        self._last_embeddings[normalized_label] = [(emb_np, seg_start, seg_duration, quality)]
                        logger.info(f"Stored embedding for {normalized_label}: dim={len(emb_np)}, "
                                   f"segment={seg_start:.1f}-{seg_start+seg_duration:.1f}s")

                if self._last_embeddings:
                    logger.info(f"Extracted {len(self._last_embeddings)} pre-computed embeddings:")
                    for label, emb_list in self._last_embeddings.items():
                        if emb_list:
                            emb_np, start, dur, qual = emb_list[0]
                            logger.info(f"  {label}: dim={len(emb_np)}, segment={start:.1f}s-{start+dur:.1f}s, quality={qual:.2f}")
                else:
                    logger.warning("Could not extract speaker embeddings from provided format")
            else:
                logger.info("No pre-computed speaker_embeddings in diarization result")
        except Exception as e:
            logger.warning(f"Failed to extract pre-computed embeddings: {e}")

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
        if self._shutdown:
            raise RuntimeError("PyAnnoteProcessor has been shut down")

        # Mark operation as in-progress
        self._operation_complete.clear()
        self._is_processing = True

        # Clear any stale cancellation from previous request
        with self._cancel_lock:
            self._cancelled_request_id = None

        loop = asyncio.get_running_loop()

        try:
            # Load pipeline in executor (blocking operation)
            await loop.run_in_executor(
                self._executor,
                self._ensure_pipeline_sync,
            )

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
            # Mark operation as complete
            self._is_processing = False
            self._operation_complete.set()

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
        hf_token: Optional[str] = None,
    ) -> List[SpeakerEmbedding]:
        """
        Extract speaker embeddings from audio (non-blocking).

        Args:
            audio_data: Raw audio bytes
            diarization_segments: Diarization results
            meeting_id: Meeting ID for embedding metadata
            progress_callback: Callback for progress updates
            request_id: Request ID for progress messages
            hf_token: Optional client HuggingFace token (for first-time model download)

        Returns:
            List of SpeakerEmbedding (one per unique speaker)

        Raises:
            RuntimeError: If processor has been shut down
            ProcessorCancelled: If processing was cancelled
        """
        if self._shutdown:
            raise RuntimeError("PyAnnoteProcessor has been shut down")

        # Check for pre-computed embeddings from diarization (preferred method)
        if self._last_embeddings:
            logger.info(f"Using {len(self._last_embeddings)} pre-computed embeddings from diarization")
            embeddings = []
            for speaker_label, emb_list in self._last_embeddings.items():
                for emb_np, start, duration, quality in emb_list:
                    embeddings.append(SpeakerEmbedding(
                        speaker_label=speaker_label,
                        meeting_id=meeting_id,
                        segment_start=start,
                        segment_duration=duration,
                        embedding=emb_np.tolist() if hasattr(emb_np, 'tolist') else list(emb_np),
                        quality_score=quality,
                    ))
            # Clear after use to avoid stale data
            self._last_embeddings = {}
            return embeddings

        # No pre-computed embeddings - fall back to loading separate model
        logger.info("No pre-computed embeddings available, loading embedding model...")

        # Mark operation as in-progress
        self._operation_complete.clear()
        self._is_processing = True

        # Clear any stale cancellation from previous requests
        # This is safe because if cancel was called during diarize(), it would have
        # already raised ProcessorCancelled there and we wouldn't reach this point.
        # Any remaining cancellation is from a PREVIOUS request and should be cleared.
        with self._cancel_lock:
            self._cancelled_request_id = None

        loop = asyncio.get_running_loop()

        try:
            # Load embedding model in executor (blocking operation)
            # Pass client token for first-time model download
            from functools import partial
            await loop.run_in_executor(
                self._executor,
                partial(self._ensure_embedding_model_sync, client_hf_token=hf_token),
            )

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
            # Mark operation as complete
            self._is_processing = False
            self._operation_complete.set()

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
        logger.info(f"PyAnnote processor cancellation requested for {request_id or 'current request'}")

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
            logger.warning(f"Timeout waiting for PyAnnote processor to become idle")
            return False

    @property
    def is_processing(self) -> bool:
        """Check if processor is currently running a GPU operation."""
        return self._is_processing

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
            logger.info("Shutting down PyAnnote executor...")

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
                logger.info("PyAnnote executor shutdown complete")
            else:
                logger.warning(
                    f"PyAnnote executor shutdown timed out after {timeout}s, "
                    "forcing shutdown (operations may still be running)"
                )
                # Force shutdown without waiting
                self._executor.shutdown(wait=False, cancel_futures=True)

            self._executor = None
