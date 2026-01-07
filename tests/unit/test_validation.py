"""
Tests for the validation module.
"""
import pytest

from gpu_server.validation import (
    ValidationError,
    validate_request_id,
    validate_meeting_name,
    validate_num_speakers,
    validate_whisper_model,
    validate_language,
    validate_priority,
    validate_audio_data,
    MAX_REQUEST_ID_LENGTH,
    MAX_MEETING_NAME_LENGTH,
    MAX_NUM_SPEAKERS,
    MIN_NUM_SPEAKERS,
    ALLOWED_WHISPER_MODELS,
)


class TestValidateRequestId:
    """Tests for request_id validation."""

    def test_valid_request_id(self):
        """Valid request IDs should pass."""
        assert validate_request_id("abc123") == "abc123"
        assert validate_request_id("request-001") == "request-001"
        assert validate_request_id("req_123.456") == "req_123.456"

    def test_empty_request_id(self):
        """Empty request ID should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_request_id("")
        assert exc_info.value.field == "request_id"
        assert "required" in exc_info.value.message.lower()

    def test_request_id_too_long(self):
        """Request ID exceeding max length should raise ValidationError."""
        long_id = "a" * (MAX_REQUEST_ID_LENGTH + 1)
        with pytest.raises(ValidationError) as exc_info:
            validate_request_id(long_id)
        assert exc_info.value.field == "request_id"
        assert "128" in exc_info.value.message

    def test_request_id_invalid_characters(self):
        """Request ID with invalid characters should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_request_id("req@123")  # @ not allowed
        assert exc_info.value.field == "request_id"

        with pytest.raises(ValidationError) as exc_info:
            validate_request_id("req 123")  # space not allowed
        assert exc_info.value.field == "request_id"

    def test_request_id_control_characters(self):
        """Request ID with control characters should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_request_id("req\n123")
        assert exc_info.value.field == "request_id"


class TestValidateMeetingName:
    """Tests for meeting_name validation."""

    def test_valid_meeting_name(self):
        """Valid meeting names should pass."""
        assert validate_meeting_name("Team Standup") == "Team Standup"
        assert validate_meeting_name("") == ""  # Empty is allowed
        assert validate_meeting_name("Meeting 2024-01-15") == "Meeting 2024-01-15"

    def test_meeting_name_too_long(self):
        """Meeting name exceeding max length should raise ValidationError."""
        long_name = "a" * (MAX_MEETING_NAME_LENGTH + 1)
        with pytest.raises(ValidationError) as exc_info:
            validate_meeting_name(long_name)
        assert exc_info.value.field == "meeting_name"

    def test_meeting_name_strips_control_chars(self):
        """Meeting names with control characters should have them stripped."""
        # Control characters should be stripped
        result = validate_meeting_name("Meeting\x00Name")
        assert "\x00" not in result
        assert "Meeting" in result

    def test_meeting_name_whitespace(self):
        """Meeting names with excessive whitespace should be handled."""
        result = validate_meeting_name("  Meeting  Name  ")
        assert result == "Meeting  Name"  # Leading/trailing stripped


class TestValidateNumSpeakers:
    """Tests for num_speakers validation."""

    def test_valid_num_speakers(self):
        """Valid num_speakers should pass."""
        assert validate_num_speakers(2) == 2
        assert validate_num_speakers(10) == 10
        assert validate_num_speakers(1) == 1
        assert validate_num_speakers(100) == 100

    def test_none_num_speakers(self):
        """None should be allowed (auto-detect)."""
        assert validate_num_speakers(None) is None

    def test_num_speakers_too_low(self):
        """num_speakers below minimum should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_num_speakers(0)
        assert exc_info.value.field == "num_speakers"

    def test_num_speakers_too_high(self):
        """num_speakers above maximum should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_num_speakers(MAX_NUM_SPEAKERS + 1)
        assert exc_info.value.field == "num_speakers"

    def test_num_speakers_negative(self):
        """Negative num_speakers should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_num_speakers(-1)
        assert exc_info.value.field == "num_speakers"

    def test_num_speakers_boolean_rejected(self):
        """Boolean values should be rejected (bool is subclass of int)."""
        with pytest.raises(ValidationError) as exc_info:
            validate_num_speakers(True)
        assert exc_info.value.field == "num_speakers"
        assert "boolean" in exc_info.value.message.lower()

        with pytest.raises(ValidationError) as exc_info:
            validate_num_speakers(False)
        assert exc_info.value.field == "num_speakers"


class TestValidateWhisperModel:
    """Tests for whisper_model validation."""

    def test_valid_whisper_models(self):
        """Valid Whisper models should pass."""
        assert validate_whisper_model("tiny") == "tiny"
        assert validate_whisper_model("base") == "base"
        assert validate_whisper_model("small") == "small"
        assert validate_whisper_model("medium") == "medium"
        assert validate_whisper_model("large-v3") == "large-v3"

    def test_none_whisper_model(self):
        """None should be allowed (use default)."""
        assert validate_whisper_model(None) is None

    def test_invalid_whisper_model(self):
        """Invalid Whisper model should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_whisper_model("invalid-model")
        assert exc_info.value.field == "whisper_model"

    def test_whisper_model_case_sensitive(self):
        """Whisper model names should be case-sensitive."""
        with pytest.raises(ValidationError) as exc_info:
            validate_whisper_model("TINY")  # Should be lowercase
        assert exc_info.value.field == "whisper_model"


class TestValidateLanguage:
    """Tests for language validation."""

    def test_valid_languages(self):
        """Valid language codes should pass."""
        assert validate_language("en") == "en"
        assert validate_language("es") == "es"
        assert validate_language("zh") == "zh"

    def test_none_language(self):
        """None should be allowed (auto-detect)."""
        assert validate_language(None) is None

    def test_language_too_long(self):
        """Language code too long should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_language("english")  # Too long
        assert exc_info.value.field == "language"

    def test_language_invalid_format(self):
        """Language code with invalid format should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_language("12")  # Not letters
        assert exc_info.value.field == "language"


class TestValidatePriority:
    """Tests for priority validation."""

    def test_valid_priority(self):
        """Valid priorities should pass."""
        assert validate_priority(0) == 0
        assert validate_priority(50) == 50
        assert validate_priority(100) == 100

    def test_priority_negative(self):
        """Negative priority should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_priority(-1)
        assert exc_info.value.field == "priority"

    def test_priority_too_high(self):
        """Priority above 100 should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_priority(101)
        assert exc_info.value.field == "priority"

    def test_priority_boolean_rejected(self):
        """Boolean values should be rejected (bool is subclass of int)."""
        with pytest.raises(ValidationError) as exc_info:
            validate_priority(True)
        assert exc_info.value.field == "priority"
        assert "boolean" in exc_info.value.message.lower()

        with pytest.raises(ValidationError) as exc_info:
            validate_priority(False)
        assert exc_info.value.field == "priority"


class TestValidateAudioData:
    """Tests for audio_data validation."""

    def test_valid_audio_data(self):
        """Valid audio data with OGG header should pass."""
        # OGG container magic bytes (OggS) + padding to reach min size
        data = b"OggS" + b"\x00" * 100  # OGG header + padding
        result = validate_audio_data(data)
        assert result == data

    def test_valid_wav_audio_data(self):
        """Valid audio data with WAV header should pass."""
        # RIFF....WAVE header
        data = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100
        result = validate_audio_data(data)
        assert result == data

    def test_invalid_format_rejected(self):
        """Invalid audio format should be rejected."""
        data = b"not valid audio data at all"
        with pytest.raises(ValidationError) as exc_info:
            validate_audio_data(data)
        assert exc_info.value.field == "audio_data"
        assert "Unrecognized audio format" in exc_info.value.message

    def test_format_check_can_be_disabled(self):
        """Format check can be disabled for backward compatibility."""
        data = b"fake audio data"
        result = validate_audio_data(data, require_valid_format=False)
        assert result == data

    def test_empty_audio_data(self):
        """Empty audio data should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_audio_data(b"")
        assert exc_info.value.field == "audio_data"

    def test_audio_data_too_large(self):
        """Audio data exceeding max size should raise ValidationError."""
        max_size = 1024  # 1 KB for testing
        large_data = b"x" * (max_size + 1)
        with pytest.raises(ValidationError) as exc_info:
            validate_audio_data(large_data, max_size=max_size)
        assert exc_info.value.field == "audio_data"


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error_message(self):
        """ValidationError should have proper message format."""
        error = ValidationError("test_field", "test message")
        assert error.field == "test_field"
        assert error.message == "test message"
        assert str(error) == "test_field: test message"
