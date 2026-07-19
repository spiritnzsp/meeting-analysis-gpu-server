"""
WebSocket Protocol Definitions

Defines message types and serialisation for client-server communication.
"""
import binascii
import json
import base64
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from .validation import (
    ValidationError,
    validate_request_id,
    validate_meeting_name,
    validate_num_speakers,
    validate_whisper_model,
    validate_language,
    validate_priority,
    validate_audio_data,
)


# Protocol versioning
# Format: (major, minor) - major version changes break compatibility
PROTOCOL_VERSION = (1, 2)  # v1.2: additive LLM_GENERATE workload + gpu/workloads capability
PROTOCOL_VERSION_STRING = f"{PROTOCOL_VERSION[0]}.{PROTOCOL_VERSION[1]}"
MIN_COMPATIBLE_VERSION = (1, 0)  # Minimum client version server will accept (v1.0 clients still supported)

# Upper bound on an LLM request's combined prompt size, enforced at validation
# so oversized input is rejected before it reaches the GPU. ~1M chars is far
# above any real meeting transcript (~110K usable at 28K tokens).
MAX_LLM_PROMPT_CHARS = 1_000_000


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
    VIDEO_ENCODE = "video_encode"
    LLM_GENERATE = "llm_generate"

    # Server -> Client
    AUTH_OK = "auth_ok"
    AUTH_FAILED = "auth_failed"
    QUEUED = "queued"
    PROGRESS = "progress"
    RESULT = "result"
    ERROR = "error"
    PONG = "pong"
    CANCELLED = "cancelled"
    VIDEO_RESULT = "video_result"
    LLM_RESULT = "llm_result"


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
    # Video encoding stages
    RECEIVING_VIDEO = "receiving_video"
    PROBING_INPUT = "probing_input"
    ENCODING = "encoding"


@dataclass
class ProcessingOptions:
    """Options for processing request."""
    transcribe: bool = True
    diarize: bool = True
    extract_embeddings: bool = True
    whisper_model: Optional[str] = None  # Override server default (ignored when arbiter-gated; server runs config.whisper.model)
    language: Optional[str] = None  # Force language (null = auto)
    num_speakers: Optional[int] = None  # Hint for diarization
    # DEPRECATED/IGNORED since the audio path is arbiter-gated: the embedding model
    # is loaded by the arbiter using the server's configured HuggingFace token, so
    # a client-supplied token is no longer consulted. Kept for wire compatibility.
    hf_token: Optional[str] = None


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
        """
        Deserialize a ProcessRequest from a dictionary.

        Args:
            data: The dictionary to deserialize

        Returns:
            ProcessRequest instance

        Raises:
            ValidationError: If any field fails validation
        """
        # Decode and validate audio data
        audio_b64 = data.get("audio_data", "")
        try:
            audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
        except (binascii.Error, ValueError):
            # binascii.Error for invalid base64, ValueError for incorrect padding
            raise ValidationError("audio_data", "Invalid base64 encoding")

        audio_bytes = validate_audio_data(audio_bytes)

        # Validate request ID
        request_id = validate_request_id(data.get("request_id", ""))

        # Validate meeting name
        meeting_name = validate_meeting_name(data.get("meeting_name", ""))

        # Validate priority
        priority = validate_priority(data.get("priority", 0))

        # Validate and build options
        options_dict = data.get("options", {})

        # Helper to validate boolean options - reject non-boolean types
        def validate_bool_option(name: str, value, default: bool) -> bool:
            if value is None:
                return default
            if not isinstance(value, bool):
                raise ValidationError(
                    f"options.{name}",
                    f"Must be a boolean (true/false), not {type(value).__name__}"
                )
            return value

        options = ProcessingOptions(
            transcribe=validate_bool_option("transcribe", options_dict.get("transcribe"), True),
            diarize=validate_bool_option("diarize", options_dict.get("diarize"), True),
            extract_embeddings=validate_bool_option("extract_embeddings", options_dict.get("extract_embeddings"), True),
            whisper_model=validate_whisper_model(options_dict.get("whisper_model")),
            language=validate_language(options_dict.get("language")),
            num_speakers=validate_num_speakers(options_dict.get("num_speakers")),
            hf_token=options_dict.get("hf_token"),  # Client's HuggingFace token
        )

        return cls(
            request_id=request_id,
            audio_data=audio_bytes,
            options=options,
            priority=priority,
            meeting_name=meeting_name,
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
    warnings: List[str] = field(default_factory=list)  # Warnings for partial failures

    def to_json(self) -> str:
        data = {
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
        }
        # Only include warnings if there are any (backward compatibility)
        if self.warnings:
            data["warnings"] = self.warnings
        return json.dumps(data)


@dataclass
class LlmGenerateRequest:
    """LLM generation request from client (v1.2).

    The server is a GENERIC LLM executor: the client sends the complete prompts
    (any summarisation coverage/consolidation steering is applied client-side,
    keeping prompt policy in one place). ``response_format`` of "json_object"
    asks llama.cpp to constrain output to valid JSON.
    """
    request_id: str
    system_prompt: str
    user_prompt: str
    temperature: Optional[float] = None   # None -> server config default
    max_tokens: Optional[int] = None      # None -> server config default
    response_format: Optional[str] = None  # e.g. "json_object"
    priority: int = 0
    meeting_name: str = ""                 # for logging/display only

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.LLM_GENERATE,
            "request_id": self.request_id,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": self.response_format,
            "priority": self.priority,
            "meeting_name": self.meeting_name,
        })

    @classmethod
    def from_dict(cls, data: Dict) -> 'LlmGenerateRequest':
        # Reuse the shared validators every other request type uses (charset +
        # length caps; control-char stripping) rather than ad-hoc checks.
        request_id = validate_request_id(data.get("request_id"))
        user_prompt = data.get("user_prompt")
        if not isinstance(user_prompt, str) or not user_prompt:
            raise ValidationError("user_prompt", "must be a non-empty string")
        system_prompt = data.get("system_prompt", "")
        if not isinstance(system_prompt, str):
            raise ValidationError("system_prompt", "must be a string")
        # Bound prompt size at validation so an oversized prompt is rejected
        # BEFORE it is enqueued, tokenized on the GPU thread, and possibly
        # triggers an eviction. Far above any real transcript (~110K usable).
        if len(user_prompt) + len(system_prompt) > MAX_LLM_PROMPT_CHARS:
            raise ValidationError(
                "user_prompt", f"combined prompt exceeds {MAX_LLM_PROMPT_CHARS} characters"
            )
        temperature = data.get("temperature")
        if temperature is not None:
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
                raise ValidationError("temperature", "must be a number or null")
            if not (0.0 <= float(temperature) <= 2.0):
                raise ValidationError("temperature", "must be between 0.0 and 2.0")
        max_tokens = data.get("max_tokens")
        if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0
        ):
            raise ValidationError("max_tokens", "must be a positive integer or null")
        response_format = data.get("response_format")
        if response_format is not None and response_format != "json_object":
            raise ValidationError("response_format", "must be null or 'json_object'")
        return cls(
            request_id=request_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            priority=validate_priority(data.get("priority", 0)),
            meeting_name=validate_meeting_name(data.get("meeting_name", "")),
        )


@dataclass
class LlmGenerateResult:
    """LLM generation result from server (v1.2)."""
    request_id: str
    success: bool
    text: str = ""
    finish_reason: str = ""        # "stop" | "length" | ""
    error_message: str = ""
    error_code: str = ""           # e.g. "CONTEXT_TOO_LONG" (additive; old clients ignore it)
    processing_time_seconds: float = 0.0

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.LLM_RESULT,
            "request_id": self.request_id,
            "success": self.success,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "processing_time_seconds": self.processing_time_seconds,
        })

    @classmethod
    def from_dict(cls, data: Dict) -> 'LlmGenerateResult':
        return cls(
            request_id=str(data.get("request_id", "")),
            success=bool(data.get("success", False)),
            text=str(data.get("text", "")),
            finish_reason=str(data.get("finish_reason", "")),
            error_message=str(data.get("error_message", "")),
            error_code=str(data.get("error_code", "")),
            processing_time_seconds=float(data.get("processing_time_seconds", 0.0)),
        )


@dataclass
class ErrorMessage:
    """Error message from server."""
    request_id: str
    error: str
    recoverable: bool = True
    error_code: Optional[str] = None  # Optional error code for programmatic handling

    def to_json(self) -> str:
        data = {
            "type": MessageType.ERROR,
            "request_id": self.request_id,
            "error": self.error,
            "recoverable": self.recoverable,
        }
        if self.error_code:
            data["error_code"] = self.error_code
        return json.dumps(data)


@dataclass
class AuthErrorMessage:
    """Authentication error message from server (no request_id)."""
    error: str
    error_code: str = "AUTH_ERROR"
    server_protocol_version: Optional[str] = None

    def to_json(self) -> str:
        data = {
            "type": MessageType.AUTH_FAILED,
            "error": self.error,
            "error_code": self.error_code,
        }
        if self.server_protocol_version:
            data["server_protocol_version"] = self.server_protocol_version
        return json.dumps(data)


@dataclass
class CancelledMessage:
    """Cancellation confirmation message."""
    request_id: str

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.CANCELLED,
            "request_id": self.request_id,
        })


@dataclass
class VideoCodecOptions:
    """Video encoding options."""
    codec: Optional[str] = None          # e.g. "h264_nvenc", None = auto-select
    preset: Optional[str] = None         # p1-p7, None = use config default
    quality: Optional[int] = None        # CQ value 0-51, None = use config default
    resolution: Optional[str] = None     # e.g. "1920x1080", None = keep original
    framerate: Optional[int] = None      # e.g. 30, None = keep original
    bitrate: Optional[str] = None        # e.g. "5M", None = use CQ mode
    pixel_format: Optional[str] = None   # e.g. "yuv420p", None = auto


@dataclass
class VideoEncodeRequest:
    """Video encoding request from client."""
    request_id: str
    filename: str
    input_size: int  # Expected size in bytes
    options: VideoCodecOptions = field(default_factory=VideoCodecOptions)
    priority: int = 0
    # Protocol v1.1 transfer negotiation fields
    transfer_method: str = "websocket"  # "websocket" or "shared_fs"
    input_path: Optional[str] = None    # For shared_fs: server-side input path
    output_path: Optional[str] = None   # For shared_fs: server-side output path
    resume_offset: int = 0              # For websocket resume: byte offset to continue from

    @classmethod
    def from_dict(cls, data: Dict) -> 'VideoEncodeRequest':
        """
        Deserialize a VideoEncodeRequest from a dictionary.

        Raises:
            ValidationError: If any field fails validation
        """
        request_id = validate_request_id(data.get("request_id", ""))

        filename = data.get("filename", "")
        if not filename or not isinstance(filename, str):
            raise ValidationError("filename", "Filename is required")
        # Basic filename sanitization - strip path separators
        filename = filename.replace("/", "_").replace("\\", "_")

        input_size = data.get("input_size", 0)
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValidationError("input_size", "Input size must be a positive integer")

        priority = validate_priority(data.get("priority", 0))

        # Parse codec options
        opts = data.get("options", {})
        options = VideoCodecOptions(
            codec=opts.get("codec"),
            preset=opts.get("preset"),
            quality=opts.get("quality"),
            resolution=opts.get("resolution"),
            framerate=opts.get("framerate"),
            bitrate=opts.get("bitrate"),
            pixel_format=opts.get("pixel_format"),
        )

        # Protocol v1.1 transfer method
        transfer_method = data.get("transfer_method", "websocket")
        if transfer_method not in ("websocket", "shared_fs"):
            transfer_method = "websocket"

        input_path = data.get("input_path")
        output_path = data.get("output_path")

        # Validate shared_fs paths
        if transfer_method == "shared_fs":
            if not input_path or not isinstance(input_path, str):
                raise ValidationError("input_path", "input_path is required for shared_fs transfer")
            if not output_path or not isinstance(output_path, str):
                raise ValidationError("output_path", "output_path is required for shared_fs transfer")

        # Resume offset for WebSocket uploads
        resume_offset = data.get("resume_offset", 0)
        if not isinstance(resume_offset, int) or resume_offset < 0:
            resume_offset = 0

        return cls(
            request_id=request_id,
            filename=filename,
            input_size=input_size,
            options=options,
            priority=priority,
            transfer_method=transfer_method,
            input_path=input_path,
            output_path=output_path,
            resume_offset=resume_offset,
        )


@dataclass
class VideoEncodeResult:
    """Video encoding result from server."""
    request_id: str
    success: bool
    output_size: int = 0
    codec_used: str = ""
    encoding_time: float = 0.0
    error_message: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "type": MessageType.VIDEO_RESULT,
            "request_id": self.request_id,
            "success": self.success,
            "output_size": self.output_size,
            "codec_used": self.codec_used,
            "encoding_time": self.encoding_time,
            "error_message": self.error_message,
        })


# Maximum JSON message size (defense-in-depth, WebSocket also has limits)
# Needs to be large enough for base64-encoded audio files (1 hour opus ~ 30MB raw, ~40MB base64)
# 500MB allows for recordings up to ~4 hours
MAX_JSON_SIZE = 500 * 1024 * 1024  # 500MB to handle very long recordings


def parse_message(data: str, max_size: int = MAX_JSON_SIZE) -> Dict[str, Any]:
    """
    Parse a JSON message and return as dict with 'type' field.

    Args:
        data: JSON string to parse
        max_size: Maximum allowed size in bytes (default 10MB)

    Returns:
        Parsed message dict with 'type' field, or error dict
    """
    # Size limit check (defense-in-depth)
    if len(data) > max_size:
        return {"type": "error", "error": f"Message too large ({len(data)} bytes, max {max_size})"}

    try:
        msg = json.loads(data)
        if "type" not in msg:
            return {"type": "error", "error": "Missing message type"}
        return msg
    except json.JSONDecodeError:
        return {"type": "error", "error": "Invalid JSON format"}
    except RecursionError:
        return {"type": "error", "error": "JSON nesting depth exceeded"}
