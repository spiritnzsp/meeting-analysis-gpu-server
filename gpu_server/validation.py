"""
Input Validation Module

Provides validators for protocol message fields to prevent injection attacks
and ensure data integrity.
"""
import re
from typing import Optional, List

# Validation constants
MAX_REQUEST_ID_LENGTH = 128
MAX_MEETING_NAME_LENGTH = 256
MAX_NUM_SPEAKERS = 100
MIN_NUM_SPEAKERS = 1

# Allowed Whisper models
ALLOWED_WHISPER_MODELS = {
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large", "large-v1", "large-v2", "large-v3",
}

# Pattern for valid request IDs (alphanumeric, dash, underscore, dot)
REQUEST_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_.\-]+$')

# Control characters to strip from text fields
CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x1f\x7f-\x9f]')


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


def validate_request_id(request_id: str) -> str:
    """
    Validate and sanitize a request ID.

    Args:
        request_id: The request ID to validate

    Returns:
        The validated request ID

    Raises:
        ValidationError: If validation fails
    """
    if not request_id:
        raise ValidationError("request_id", "Request ID is required")

    if not isinstance(request_id, str):
        raise ValidationError("request_id", "Request ID must be a string")

    if len(request_id) > MAX_REQUEST_ID_LENGTH:
        raise ValidationError(
            "request_id",
            f"Request ID exceeds maximum length ({MAX_REQUEST_ID_LENGTH} chars)"
        )

    if not REQUEST_ID_PATTERN.match(request_id):
        raise ValidationError(
            "request_id",
            "Request ID contains invalid characters (allowed: alphanumeric, dash, underscore, dot)"
        )

    return request_id


def validate_meeting_name(meeting_name: str) -> str:
    """
    Validate and sanitize a meeting name.

    Args:
        meeting_name: The meeting name to validate

    Returns:
        The sanitized meeting name

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(meeting_name, str):
        raise ValidationError("meeting_name", "Meeting name must be a string")

    # Strip control characters (prevent log injection)
    sanitized = CONTROL_CHAR_PATTERN.sub('', meeting_name)

    if len(sanitized) > MAX_MEETING_NAME_LENGTH:
        raise ValidationError(
            "meeting_name",
            f"Meeting name exceeds maximum length ({MAX_MEETING_NAME_LENGTH} chars)"
        )

    return sanitized.strip()


def validate_num_speakers(num_speakers: Optional[int]) -> Optional[int]:
    """
    Validate the number of speakers hint.

    Args:
        num_speakers: The number of speakers (None for auto-detect)

    Returns:
        The validated number or None

    Raises:
        ValidationError: If validation fails
    """
    if num_speakers is None:
        return None

    # Check for bool before int because bool is a subclass of int in Python
    if isinstance(num_speakers, bool):
        raise ValidationError("num_speakers", "Number of speakers must be an integer, not a boolean")

    if not isinstance(num_speakers, int):
        raise ValidationError("num_speakers", "Number of speakers must be an integer")

    if num_speakers < MIN_NUM_SPEAKERS:
        raise ValidationError(
            "num_speakers",
            f"Number of speakers must be at least {MIN_NUM_SPEAKERS}"
        )

    if num_speakers > MAX_NUM_SPEAKERS:
        raise ValidationError(
            "num_speakers",
            f"Number of speakers cannot exceed {MAX_NUM_SPEAKERS}"
        )

    return num_speakers


def validate_whisper_model(model: Optional[str]) -> Optional[str]:
    """
    Validate the Whisper model name.

    Args:
        model: The model name (None to use server default)

    Returns:
        The validated model name or None

    Raises:
        ValidationError: If validation fails
    """
    if model is None:
        return None

    if not isinstance(model, str):
        raise ValidationError("whisper_model", "Whisper model must be a string")

    if model not in ALLOWED_WHISPER_MODELS:
        raise ValidationError(
            "whisper_model",
            f"Invalid Whisper model. Allowed: {', '.join(sorted(ALLOWED_WHISPER_MODELS))}"
        )

    return model


def validate_language(language: Optional[str]) -> Optional[str]:
    """
    Validate the language code.

    Args:
        language: ISO language code (None for auto-detect)

    Returns:
        The validated language code or None

    Raises:
        ValidationError: If validation fails
    """
    if language is None:
        return None

    if not isinstance(language, str):
        raise ValidationError("language", "Language must be a string")

    # Basic format check (2-3 letter codes)
    if not re.match(r'^[a-z]{2,3}$', language.lower()):
        raise ValidationError(
            "language",
            "Language must be a 2-3 letter ISO code (e.g., 'en', 'de', 'fra')"
        )

    return language.lower()


def validate_priority(priority: int) -> int:
    """
    Validate the request priority.

    Args:
        priority: The priority value

    Returns:
        The validated priority

    Raises:
        ValidationError: If validation fails
    """
    # Check for bool before int because bool is a subclass of int in Python
    if isinstance(priority, bool):
        raise ValidationError("priority", "Priority must be an integer, not a boolean")

    if not isinstance(priority, int):
        raise ValidationError("priority", "Priority must be an integer")

    if priority < 0 or priority > 100:
        raise ValidationError("priority", "Priority must be between 0 and 100")

    return priority


# Audio format magic bytes for validation
# Format: (magic_bytes, offset, format_name)
AUDIO_MAGIC_BYTES = [
    (b'OggS', 0, 'OGG/Opus'),           # OGG container (often contains Opus)
    (b'RIFF', 0, 'WAV'),                 # WAV files start with RIFF
    (b'fLaC', 0, 'FLAC'),               # FLAC audio
    (b'ID3', 0, 'MP3'),                  # MP3 with ID3 tag
    (b'\xff\xfb', 0, 'MP3'),             # MP3 frame sync (MPEG 1 Layer 3)
    (b'\xff\xfa', 0, 'MP3'),             # MP3 frame sync
    (b'\xff\xf3', 0, 'MP3'),             # MP3 frame sync (MPEG 2 Layer 3)
    (b'\xff\xf2', 0, 'MP3'),             # MP3 frame sync
    (b'ftyp', 4, 'M4A/MP4'),             # M4A/MP4 container (ftyp at offset 4)
    (b'FORM', 0, 'AIFF'),               # AIFF audio
]


def detect_audio_format(audio_data: bytes) -> Optional[str]:
    """
    Detect audio format from magic bytes.

    Args:
        audio_data: The audio bytes

    Returns:
        Format name if recognized, None otherwise
    """
    if len(audio_data) < 12:  # Need at least 12 bytes for reliable detection
        return None

    for magic, offset, format_name in AUDIO_MAGIC_BYTES:
        if len(audio_data) > offset + len(magic):
            if audio_data[offset:offset + len(magic)] == magic:
                return format_name

    return None


def validate_audio_data(
    audio_data: bytes,
    max_size: int = 100 * 1024 * 1024,
    require_valid_format: bool = True
) -> bytes:
    """
    Validate audio data including format check via magic bytes.

    Args:
        audio_data: The audio bytes
        max_size: Maximum allowed size in bytes
        require_valid_format: If True, reject unrecognized formats

    Returns:
        The validated audio data

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(audio_data, bytes):
        raise ValidationError("audio_data", "Audio data must be bytes")

    if len(audio_data) == 0:
        raise ValidationError("audio_data", "Audio data is empty")

    if len(audio_data) > max_size:
        raise ValidationError(
            "audio_data",
            f"Audio data exceeds maximum size ({max_size / 1024 / 1024:.0f} MB)"
        )

    # Check audio format via magic bytes
    if require_valid_format:
        detected_format = detect_audio_format(audio_data)
        if detected_format is None:
            raise ValidationError(
                "audio_data",
                "Unrecognized audio format. Supported: OGG/Opus, WAV, MP3, FLAC, M4A, AIFF"
            )

    return audio_data
