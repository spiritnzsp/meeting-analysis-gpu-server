"""
Binary Frame Protocol

Minimal binary framing for efficient video data transfer over WebSocket.

Frame format:
    [1 byte: frame_type][1 byte: id_length][N bytes: request_id_utf8][payload]

Total overhead: 2 + len(request_id) bytes (~38 bytes typically). Negligible for video.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple


class BinaryFrameType(IntEnum):
    """Binary frame types for video data transfer."""
    VIDEO_INPUT = 0x01   # Client -> Server: video data chunk
    VIDEO_OUTPUT = 0x02  # Server -> Client: encoded video data chunk


@dataclass
class BinaryFrameHeader:
    """Decoded binary frame header."""
    frame_type: BinaryFrameType
    request_id: str


def encode_binary_frame(frame_type: BinaryFrameType, request_id: str, payload: bytes) -> bytes:
    """
    Encode a binary frame with header and payload.

    Args:
        frame_type: Type of binary frame
        request_id: Request ID (UTF-8, max 255 bytes)
        payload: Raw payload bytes

    Returns:
        Encoded binary frame

    Raises:
        ValueError: If request_id is too long
    """
    request_id_bytes = request_id.encode('utf-8')
    if len(request_id_bytes) > 255:
        raise ValueError(f"Request ID too long for binary frame: {len(request_id_bytes)} bytes (max 255)")

    header = bytes([int(frame_type), len(request_id_bytes)]) + request_id_bytes
    return header + payload


def decode_binary_frame(data: bytes) -> Tuple[BinaryFrameHeader, bytes]:
    """
    Decode a binary frame into header and payload.

    Args:
        data: Raw binary frame data

    Returns:
        Tuple of (BinaryFrameHeader, payload_bytes)

    Raises:
        ValueError: If frame is malformed
    """
    if len(data) < 2:
        raise ValueError("Binary frame too short (minimum 2 bytes)")

    frame_type_byte = data[0]
    id_length = data[1]

    if len(data) < 2 + id_length:
        raise ValueError(f"Binary frame truncated: expected {2 + id_length} header bytes, got {len(data)}")

    try:
        frame_type = BinaryFrameType(frame_type_byte)
    except ValueError:
        raise ValueError(f"Unknown binary frame type: 0x{frame_type_byte:02x}")

    request_id = data[2:2 + id_length].decode('utf-8')
    payload = data[2 + id_length:]

    return BinaryFrameHeader(frame_type=frame_type, request_id=request_id), payload
