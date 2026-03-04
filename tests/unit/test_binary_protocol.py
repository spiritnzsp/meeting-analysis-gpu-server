"""Tests for binary frame protocol."""
import pytest
from gpu_server.binary_protocol import (
    BinaryFrameType, BinaryFrameHeader,
    encode_binary_frame, decode_binary_frame,
)


class TestEncodeDecode:
    """Test binary frame encoding and decoding."""

    def test_roundtrip_video_input(self):
        payload = b"\x00\x01\x02\x03" * 100
        request_id = "req-abc-123"

        encoded = encode_binary_frame(BinaryFrameType.VIDEO_INPUT, request_id, payload)
        header, decoded_payload = decode_binary_frame(encoded)

        assert header.frame_type == BinaryFrameType.VIDEO_INPUT
        assert header.request_id == request_id
        assert decoded_payload == payload

    def test_roundtrip_video_output(self):
        payload = b"encoded video data"
        request_id = "test-456"

        encoded = encode_binary_frame(BinaryFrameType.VIDEO_OUTPUT, request_id, payload)
        header, decoded_payload = decode_binary_frame(encoded)

        assert header.frame_type == BinaryFrameType.VIDEO_OUTPUT
        assert header.request_id == request_id
        assert decoded_payload == payload

    def test_empty_payload(self):
        """Empty payload is valid (signals end of stream)."""
        encoded = encode_binary_frame(BinaryFrameType.VIDEO_INPUT, "req-1", b"")
        header, payload = decode_binary_frame(encoded)
        assert payload == b""
        assert header.request_id == "req-1"

    def test_uuid_request_id(self):
        """Typical UUID request ID."""
        request_id = "550e8400-e29b-41d4-a716-446655440000"
        payload = b"data"

        encoded = encode_binary_frame(BinaryFrameType.VIDEO_INPUT, request_id, payload)
        header, decoded_payload = decode_binary_frame(encoded)

        assert header.request_id == request_id
        assert decoded_payload == payload

    def test_header_structure(self):
        """Verify the binary structure matches the spec."""
        request_id = "abc"
        payload = b"\xff\xfe"

        encoded = encode_binary_frame(BinaryFrameType.VIDEO_INPUT, request_id, payload)

        assert encoded[0] == 0x01  # VIDEO_INPUT
        assert encoded[1] == 3     # id_length = len("abc")
        assert encoded[2:5] == b"abc"
        assert encoded[5:] == b"\xff\xfe"

    def test_overhead_is_minimal(self):
        """Verify overhead is 2 + len(request_id)."""
        request_id = "550e8400-e29b-41d4-a716-446655440000"
        payload = b"\x00" * 1000

        encoded = encode_binary_frame(BinaryFrameType.VIDEO_INPUT, request_id, payload)
        expected_overhead = 2 + len(request_id.encode('utf-8'))

        assert len(encoded) == expected_overhead + len(payload)


class TestDecodeErrors:
    """Test error handling in decode."""

    def test_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            decode_binary_frame(b"\x01")

    def test_truncated_request_id(self):
        # Says id_length=10 but only has 3 bytes
        data = bytes([0x01, 10]) + b"abc"
        with pytest.raises(ValueError, match="truncated"):
            decode_binary_frame(data)

    def test_unknown_frame_type(self):
        data = bytes([0xFF, 3]) + b"abc" + b"payload"
        with pytest.raises(ValueError, match="Unknown binary frame type"):
            decode_binary_frame(data)


class TestEncodeErrors:
    """Test error handling in encode."""

    def test_request_id_too_long(self):
        long_id = "a" * 256
        with pytest.raises(ValueError, match="too long"):
            encode_binary_frame(BinaryFrameType.VIDEO_INPUT, long_id, b"data")
