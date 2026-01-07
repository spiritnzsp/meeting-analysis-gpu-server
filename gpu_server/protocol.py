"""
WebSocket Protocol Definitions

Defines message types and serialisation for client-server communication.
"""
import json
import base64
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


# Protocol versioning
# Format: (major, minor) - major version changes break compatibility
PROTOCOL_VERSION = (1, 0)
PROTOCOL_VERSION_STRING = f"{PROTOCOL_VERSION[0]}.{PROTOCOL_VERSION[1]}"
MIN_COMPATIBLE_VERSION = (1, 0)  # Minimum client version server will accept


def parse_version(version_str: str) -> Tuple[int, int]:
    """Parse a version string like '1.0' into a tuple (1, 0)."""
    try:
        parts = version_str.split(".")
        if len(parts) >= 2:
            return (int(parts[0]), int(parts[1]))
        elif len(parts) == 1:
            return (int(parts[0]), 0)
    except (ValueError, AttributeError):
        pass
    return (0, 0)


def is_version_compatible(client_version: str) -> Tuple[bool, str]:
    """
    Check if a client protocol version is compatible with this server.

    Returns:
        Tuple of (is_compatible, error_message)
    """
    client_ver = parse_version(client_version)

    # Major version must match
    if client_ver[0] != PROTOCOL_VERSION[0]:
        return False, (
            f"Protocol version mismatch: client v{client_version}, "
            f"server v{PROTOCOL_VERSION_STRING}. Major versions must match."
        )

    # Client minor version must be >= minimum
    if client_ver < MIN_COMPATIBLE_VERSION:
        return False, (
            f"Client protocol version {client_version} is too old. "
            f"Minimum required: {MIN_COMPATIBLE_VERSION[0]}.{MIN_COMPATIBLE_VERSION[1]}"
        )

    return True, ""


class MessageType(str, Enum):
    """Message types for WebSocket protocol."""
    # Client -> Server
    AUTH = "auth"
    PROCESS = "process"
    CANCEL = "cancel"
    PING = "ping"

    # Server -> Client
    AUTH_OK = "auth_ok"
    AUTH_FAILED = "auth_failed"
    QUEUED = "queued"
    PROGRESS = "progress"
    RESULT = "result"
    ERROR = "error"
    PONG = "pong"


class ProcessingStage(str, Enum):
    """Processing stages for progress updates."""
    QUEUED = "queued"
    LOADING_MODELS = "loading_models"
    DECODING_AUDIO = "decoding_audio"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    EXTRACTING_EMBEDDINGS = "extracting_embeddings"
    ALIGNING = "aligning"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingOptions:
    """Options for processing request."""
    transcribe: bool = True
    diarize: bool = True
    extract_embeddings: bool = True
    whisper_model: Optional[str] = None  # Override server default
    language: Optional[str] = None  # Force language (null = auto)
    num_speakers: Optional[int] = None  # Hint for diarization


@dataclass
class AuthMessage:
    """Authentication message from client."""
    api_key: str
    client_version: str = "unknown"
    protocol_version: str = "1.0"  # Protocol version for compatibility checking

    def to_json(self) -> str:
        return json.dumps({"type": MessageType.AUTH, **asdict(self)})

    @classmethod
    def from_dict(cls, data: Dict) -> 'AuthMessage':
        return cls(
            api_key=data.get("api_key", ""),
            client_version=data.get("client_version", "unknown"),
            protocol_version=data.get("protocol_version", "1.0"),
        )


@dataclass
class ProcessRequest:
    """Processing request from client."""
    request_id: str
    audio_data: bytes  # Raw opus/wav bytes
    options: ProcessingOptions = field(default_factory=ProcessingOptions)
    priority: int = 0  # Higher = more priority
    meeting_name: str = ""  # For logging/display

    def to_json(self) -> str:
        data = {
            "type": MessageType.PROCESS,
            "request_id": self.request_id,
            "audio_data": base64.b64encode(self.audio_data).decode('ascii'),
            "options": asdict(self.options),
            "priority": self.priority,
            "meeting_name": self.meeting_name,
        }
        return json.dumps(data)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ProcessRequest':
        audio_b64 = data.get("audio_data", "")
        audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""

        options_dict = data.get("options", {})
        options = ProcessingOptions(
            transcribe=options_dict.get("transcribe", True),
            diarize=options_dict.get("diarize", True),
            extract_embeddings=options_dict.get("extract_embeddings", True),
            whisper_model=options_dict.get("whisper_model"),
            language=options_dict.get("language"),
            num_speakers=options_dict.get("num_speakers"),
        )

        return cls(
            request_id=data.get("request_id", ""),
            audio_data=audio_bytes,
            options=options,
            priority=data.get("priority", 0),
            meeting_name=data.get("meeting_name", ""),
        )


@dataclass
class ProgressMessage:
    """Progress update from server."""
    request_id: str
    stage: ProcessingStage
    percent: int  # 0-100
    message: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.PROGRESS,
            "request_id": self.request_id,
            "stage": self.stage.value,
            "percent": self.percent,
            "message": self.message,
        })


@dataclass
class TranscriptSegment:
    """A segment of transcribed text."""
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: float = 1.0
    words: List[Dict] = field(default_factory=list)  # Word-level timestamps


@dataclass
class DiarizationSegment:
    """A speaker diarization segment."""
    start: float
    end: float
    speaker: str


@dataclass
class SpeakerEmbedding:
    """Voice embedding for a speaker."""
    speaker_label: str
    meeting_id: str
    segment_start: float
    segment_duration: float
    embedding: List[float]  # 512-dim vector as list
    quality_score: float = 1.0


@dataclass
class ProcessingResult:
    """Processing result from server."""
    request_id: str
    success: bool
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)
    diarization_segments: List[DiarizationSegment] = field(default_factory=list)
    speaker_embeddings: List[SpeakerEmbedding] = field(default_factory=list)
    full_text: str = ""
    detected_language: str = ""
    error_message: str = ""
    processing_time_seconds: float = 0.0

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.RESULT,
            "request_id": self.request_id,
            "success": self.success,
            "transcript_segments": [asdict(s) for s in self.transcript_segments],
            "diarization_segments": [asdict(s) for s in self.diarization_segments],
            "speaker_embeddings": [asdict(e) for e in self.speaker_embeddings],
            "full_text": self.full_text,
            "detected_language": self.detected_language,
            "error_message": self.error_message,
            "processing_time_seconds": self.processing_time_seconds,
        })


@dataclass
class ErrorMessage:
    """Error message from server."""
    request_id: str
    error: str
    recoverable: bool = True

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.ERROR,
            "request_id": self.request_id,
            "error": self.error,
            "recoverable": self.recoverable,
        })


def parse_message(data: str) -> Dict[str, Any]:
    """Parse a JSON message and return as dict with 'type' field."""
    try:
        msg = json.loads(data)
        if "type" not in msg:
            return {"type": "error", "error": "Missing message type"}
        return msg
    except json.JSONDecodeError as e:
        return {"type": "error", "error": f"Invalid JSON: {e}"}
