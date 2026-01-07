"""
Audio Processors

GPU-accelerated processing modules for transcription and diarization.
"""
from .whisper_processor import WhisperProcessor
from .pyannote_processor import PyAnnoteProcessor

__all__ = ['WhisperProcessor', 'PyAnnoteProcessor']
