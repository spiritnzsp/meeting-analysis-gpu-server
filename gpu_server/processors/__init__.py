"""
Audio Processors

GPU-accelerated processing modules for transcription and diarization.
"""


class ProcessorCancelled(Exception):
    """Raised when processing is cancelled via cancel event."""
    pass


from .whisper_processor import WhisperProcessor
from .pyannote_processor import PyAnnoteProcessor

__all__ = ['WhisperProcessor', 'PyAnnoteProcessor', 'ProcessorCancelled']
