"""Tests for video validation functions."""
import pytest
from gpu_server.validation import (
    ValidationError,
    detect_video_format,
    validate_video_data,
    validate_video_codec,
    validate_nvenc_preset,
    validate_resolution,
    validate_video_quality,
)


class TestDetectVideoFormat:
    """Test video format detection via magic bytes."""

    def test_mp4_ftyp(self):
        # MP4 files have 'ftyp' at offset 4
        data = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 100
        assert detect_video_format(data) == "MP4"

    def test_webm_mkv(self):
        # WebM/MKV EBML header
        data = b"\x1a\x45\xdf\xa3" + b"\x00" * 100
        assert detect_video_format(data) == "WebM/MKV"

    def test_avi(self):
        data = b"RIFF" + b"\x00" * 100
        assert detect_video_format(data) == "AVI"

    def test_mpeg_ts(self):
        data = b"\x47" + b"\x00" * 100
        assert detect_video_format(data) == "MPEG-TS"

    def test_mpeg_ps(self):
        data = b"\x00\x00\x01\xba" + b"\x00" * 100
        assert detect_video_format(data) == "MPEG-PS"

    def test_unknown_format(self):
        data = b"\xde\xad\xbe\xef" + b"\x00" * 100
        assert detect_video_format(data) is None

    def test_too_short(self):
        assert detect_video_format(b"\x00\x00") is None


class TestValidateVideoData:
    """Test video data validation."""

    def test_valid_mp4(self):
        data = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 100
        result = validate_video_data(data, max_size=1024)
        assert result == data

    def test_empty_data(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_video_data(b"", max_size=1024)

    def test_exceeds_max_size(self):
        data = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 2000
        with pytest.raises(ValidationError, match="exceeds maximum"):
            validate_video_data(data, max_size=1024)

    def test_unrecognized_format(self):
        data = b"\xde\xad\xbe\xef" + b"\x00" * 100
        with pytest.raises(ValidationError, match="Unrecognized video format"):
            validate_video_data(data, max_size=1024)

    def test_not_bytes(self):
        with pytest.raises(ValidationError, match="must be bytes"):
            validate_video_data("not bytes", max_size=1024)


class TestValidateVideoCodec:
    """Test video codec validation."""

    def test_valid_codecs(self):
        assert validate_video_codec("h264_nvenc") == "h264_nvenc"
        assert validate_video_codec("hevc_nvenc") == "hevc_nvenc"
        assert validate_video_codec("av1_nvenc") == "av1_nvenc"

    def test_none(self):
        assert validate_video_codec(None) is None

    def test_invalid_codec(self):
        with pytest.raises(ValidationError, match="Invalid video codec"):
            validate_video_codec("libx264")

    def test_not_string(self):
        with pytest.raises(ValidationError, match="must be a string"):
            validate_video_codec(123)


class TestValidateNvencPreset:
    """Test NVENC preset validation."""

    def test_valid_presets(self):
        for i in range(1, 8):
            assert validate_nvenc_preset(f"p{i}") == f"p{i}"

    def test_none(self):
        assert validate_nvenc_preset(None) is None

    def test_invalid_preset(self):
        with pytest.raises(ValidationError, match="Invalid NVENC preset"):
            validate_nvenc_preset("fast")

    def test_p0_invalid(self):
        with pytest.raises(ValidationError):
            validate_nvenc_preset("p0")

    def test_p8_invalid(self):
        with pytest.raises(ValidationError):
            validate_nvenc_preset("p8")


class TestValidateResolution:
    """Test resolution validation."""

    def test_valid_resolutions(self):
        assert validate_resolution("1920x1080") == "1920x1080"
        assert validate_resolution("3840x2160") == "3840x2160"
        assert validate_resolution("1280x720") == "1280x720"

    def test_none(self):
        assert validate_resolution(None) is None

    def test_invalid_format(self):
        with pytest.raises(ValidationError, match="WIDTHxHEIGHT"):
            validate_resolution("1920-1080")

    def test_too_small(self):
        with pytest.raises(ValidationError, match="at least 16"):
            validate_resolution("8x8")

    def test_too_large(self):
        with pytest.raises(ValidationError, match="exceed maximum"):
            validate_resolution("10000x10000")


class TestValidateVideoQuality:
    """Test video quality validation."""

    def test_valid_range(self):
        assert validate_video_quality(0) == 0
        assert validate_video_quality(23) == 23
        assert validate_video_quality(51) == 51

    def test_none(self):
        assert validate_video_quality(None) is None

    def test_negative(self):
        with pytest.raises(ValidationError, match="between 0 and 51"):
            validate_video_quality(-1)

    def test_too_high(self):
        with pytest.raises(ValidationError, match="between 0 and 51"):
            validate_video_quality(52)

    def test_boolean_rejected(self):
        with pytest.raises(ValidationError, match="not a boolean"):
            validate_video_quality(True)
