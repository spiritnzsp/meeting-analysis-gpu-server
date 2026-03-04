"""Tests for video protocol extensions."""
import json
import pytest
from gpu_server.protocol import (
    MessageType, ProcessingStage,
    VideoCodecOptions, VideoEncodeRequest, VideoEncodeResult,
)
from gpu_server.validation import ValidationError


class TestMessageTypeExtensions:
    """Test new message type values."""

    def test_video_encode_value(self):
        assert MessageType.VIDEO_ENCODE == "video_encode"

    def test_video_result_value(self):
        assert MessageType.VIDEO_RESULT == "video_result"


class TestProcessingStageExtensions:
    """Test new processing stage values."""

    def test_receiving_video(self):
        assert ProcessingStage.RECEIVING_VIDEO == "receiving_video"

    def test_probing_input(self):
        assert ProcessingStage.PROBING_INPUT == "probing_input"

    def test_encoding(self):
        assert ProcessingStage.ENCODING == "encoding"


class TestVideoCodecOptions:
    """Test VideoCodecOptions dataclass."""

    def test_defaults(self):
        opts = VideoCodecOptions()
        assert opts.codec is None
        assert opts.preset is None
        assert opts.quality is None
        assert opts.resolution is None
        assert opts.framerate is None
        assert opts.bitrate is None
        assert opts.pixel_format is None

    def test_custom_values(self):
        opts = VideoCodecOptions(
            codec="hevc_nvenc",
            preset="p4",
            quality=23,
            resolution="1920x1080",
            framerate=30,
            bitrate="5M",
            pixel_format="yuv420p",
        )
        assert opts.codec == "hevc_nvenc"
        assert opts.preset == "p4"
        assert opts.quality == 23


class TestVideoEncodeRequest:
    """Test VideoEncodeRequest deserialization."""

    def test_from_dict_minimal(self):
        data = {
            "request_id": "req-123",
            "filename": "video.mp4",
            "input_size": 1000000,
        }
        req = VideoEncodeRequest.from_dict(data)
        assert req.request_id == "req-123"
        assert req.filename == "video.mp4"
        assert req.input_size == 1000000
        assert req.priority == 0
        assert req.options.codec is None

    def test_from_dict_with_options(self):
        data = {
            "request_id": "req-456",
            "filename": "movie.mkv",
            "input_size": 5000000,
            "priority": 5,
            "options": {
                "codec": "hevc_nvenc",
                "preset": "p6",
                "quality": 20,
            },
        }
        req = VideoEncodeRequest.from_dict(data)
        assert req.priority == 5
        assert req.options.codec == "hevc_nvenc"
        assert req.options.preset == "p6"
        assert req.options.quality == 20

    def test_from_dict_missing_request_id(self):
        data = {"filename": "video.mp4", "input_size": 1000}
        with pytest.raises(ValidationError):
            VideoEncodeRequest.from_dict(data)

    def test_from_dict_missing_filename(self):
        data = {"request_id": "req-1", "input_size": 1000}
        with pytest.raises(ValidationError, match="Filename"):
            VideoEncodeRequest.from_dict(data)

    def test_from_dict_invalid_input_size(self):
        data = {"request_id": "req-1", "filename": "v.mp4", "input_size": -1}
        with pytest.raises(ValidationError, match="positive integer"):
            VideoEncodeRequest.from_dict(data)

    def test_from_dict_zero_input_size(self):
        data = {"request_id": "req-1", "filename": "v.mp4", "input_size": 0}
        with pytest.raises(ValidationError, match="positive integer"):
            VideoEncodeRequest.from_dict(data)

    def test_filename_sanitization(self):
        data = {
            "request_id": "req-1",
            "filename": "../../etc/passwd",
            "input_size": 1000,
        }
        req = VideoEncodeRequest.from_dict(data)
        assert "/" not in req.filename
        assert "\\" not in req.filename


class TestVideoEncodeResult:
    """Test VideoEncodeResult serialization."""

    def test_success_to_json(self):
        result = VideoEncodeResult(
            request_id="req-123",
            success=True,
            output_size=500000,
            codec_used="hevc_nvenc",
            encoding_time=12.5,
        )
        data = json.loads(result.to_json())
        assert data["type"] == "video_result"
        assert data["success"] is True
        assert data["output_size"] == 500000
        assert data["codec_used"] == "hevc_nvenc"
        assert data["encoding_time"] == 12.5

    def test_failure_to_json(self):
        result = VideoEncodeResult(
            request_id="req-456",
            success=False,
            error_message="No NVENC codec available",
        )
        data = json.loads(result.to_json())
        assert data["success"] is False
        assert data["error_message"] == "No NVENC codec available"
