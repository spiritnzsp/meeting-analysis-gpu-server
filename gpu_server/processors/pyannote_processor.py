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

        # Dedicated thread pool for GPU operations to avoid blocking event loop
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pyannote_gpu")

    def _ensure_pipeline_sync(self):
        """Load the diarization pipeline if not already loaded (synchronous, runs in executor)."""
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
        """Load the embedding model if not already loaded (synchronous, runs in executor)."""
        if self._embedding_model is not None:
            return

        try:
            from pyannote.audio import Model

            logger.info("Loading PyAnnote embedding model")

            self._embedding_model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=self.config.huggingface_token,
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
        self._check_cancelled(request_id)

        # Run diarization
        diarization_params = {}
        if num_speakers:
            diarization_params['num_speakers'] = num_speakers

        diarization = self._pipeline(str(audio_path), **diarization_params)

        self._check_cancelled(request_id)

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
            RuntimeError: If processor has been shut down
            ProcessorCancelled: If processing was cancelled
        """
        if self._shutdown:
            raise RuntimeError("PyAnnoteProcessor has been shut down")

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
            await loop.run_in_executor(
                self._executor,
                self._ensure_embedding_model_sync,
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
