"""Tests for protocol module."""
import json
import pytest

from gpu_server.protocol import (
    MessageType, ProcessingStage, ProcessRequest, ProcessingOptions,
    ProgressMessage, ProcessingResult, parse_message
)


class TestProcessRequest:
    """Tests for ProcessRequest serialization."""

    def test_to_json_and_back(self):
        """Test round-trip serialization."""
        request = ProcessRequest(
            request_id="test-123",
            audio_data=b"fake audio data",
            options=ProcessingOptions(
                transcribe=True,
                diarize=True,
                extract_embeddings=False,
            ),
            priority=5,
            meeting_name="Test Meeting",
        )

        json_str = request.to_json()
        data = json.loads(json_str)

        # Deserialize
        restored = ProcessRequest.from_dict(data)

        assert restored.request_id == request.request_id
        assert restored.audio_data == request.audio_data
        assert restored.options.transcribe == request.options.transcribe
        assert restored.options.diarize == request.options.diarize
        assert restored.options.extract_embeddings == request.options.extract_embeddings
        assert restored.priority == request.priority
        assert restored.meeting_name == request.meeting_name

    def test_audio_data_base64_encoding(self):
        """Test that audio data is properly base64 encoded."""
        audio = b"\x00\x01\x02\x03\xff\xfe\xfd"
        request = ProcessRequest(
            request_id="test",
            audio_data=audio,
        )

        json_str = request.to_json()
        data = json.loads(json_str)

        # Should be valid base64
        assert isinstance(data["audio_data"], str)

        # Round-trip should preserve data
        restored = ProcessRequest.from_dict(data)
        assert restored.audio_data == audio


class TestProgressMessage:
    """Tests for ProgressMessage."""

    def test_to_json(self):
        """Test JSON serialization."""
        msg = ProgressMessage(
            request_id="test-123",
            stage=ProcessingStage.TRANSCRIBING,
            percent=45,
            message="Processing..."
        )

        json_str = msg.to_json()
        data = json.loads(json_str)

        assert data["type"] == MessageType.PROGRESS
        assert data["request_id"] == "test-123"
        assert data["stage"] == "transcribing"
        assert data["percent"] == 45
        assert data["message"] == "Processing..."


class TestProcessingResult:
    """Tests for ProcessingResult."""

    def test_success_result(self):
        """Test successful result serialization."""
        result = ProcessingResult(
            request_id="test-123",
            success=True,
            full_text="Hello world",
            detected_language="en",
            processing_time_seconds=5.5,
        )

        json_str = result.to_json()
        data = json.loads(json_str)

        assert data["type"] == MessageType.RESULT
        assert data["success"] is True
        assert data["full_text"] == "Hello world"

    def test_failed_result(self):
        """Test failed result serialization."""
        result = ProcessingResult(
            request_id="test-123",
            success=False,
            error_message="GPU out of memory",
        )

        json_str = result.to_json()
        data = json.loads(json_str)

        assert data["success"] is False
        assert data["error_message"] == "GPU out of memory"


class TestParseMessage:
    """Tests for message parsing."""

    def test_valid_message(self):
        """Test parsing valid JSON message."""
        data = parse_message('{"type": "process", "request_id": "123"}')
        assert data["type"] == "process"
        assert data["request_id"] == "123"

    def test_invalid_json(self):
        """Test parsing invalid JSON."""
        data = parse_message("not valid json")
        assert data["type"] == "error"
        assert "Invalid JSON" in data["error"]

    def test_missing_type(self):
        """Test message without type field."""
        data = parse_message('{"request_id": "123"}')
        assert data["type"] == "error"
        assert "Missing message type" in data["error"]
