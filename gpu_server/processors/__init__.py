"""
Audio Processors

GPU-accelerated processing modules for transcription and diarization.
"""


class ProcessorCancelled(Exception):
    """Raised when processing is cancelled via cancel event."""
    pass


from .base_processor import BaseProcessor
from .whisper_processor import WhisperProcessor
from .pyannote_processor import PyAnnoteProcessor
from .video_encoder import VideoEncoderProcessor, CodecCapabilities

__all__ = [
    'BaseProcessor', 'WhisperProcessor', 'PyAnnoteProcessor',
    'VideoEncoderProcessor', 'CodecCapabilities', 'ProcessorCancelled',
]
