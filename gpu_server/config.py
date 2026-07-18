"""
Configuration Management

Loads and validates server configuration from YAML files.
"""
import hashlib
import hmac
import os
import secrets
import yaml
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .validation import ALLOWED_WHISPER_MODELS

logger = logging.getLogger(__name__)


# Argon2 parameters for API key hashing (OWASP recommended for high-security)
# These are intentionally slow to prevent brute-force attacks
ARGON2_TIME_COST = 3        # Number of iterations
ARGON2_MEMORY_COST = 65536  # 64 MiB memory
ARGON2_PARALLELISM = 4      # Parallel threads
ARGON2_HASH_LEN = 32        # Output hash length
ARGON2_SALT_LEN = 16        # Salt length

# Note: Secure memory zeroing is not implemented because Python's string
# immutability makes it ineffective. Strings cannot be modified in place,
# and any "zeroing" would just zero a copy. For sensitive data, minimize
# exposure time and let garbage collection handle cleanup.


def _is_argon2_hash(hash_str: str) -> bool:
    """Check if a hash string is in argon2 format."""
    return hash_str.startswith('$argon2')


def _is_legacy_sha256_hash(hash_str: str) -> bool:
    """Check if a hash string is a legacy SHA-256 hash (64 hex chars)."""
    if len(hash_str) != 64:
        return False
    try:
        int(hash_str, 16)
        return True
    except ValueError:
        return False


def _hash_api_key_legacy(api_key: str) -> str:
    """
    Legacy SHA-256 hashing for backward compatibility.

    DEPRECATED: Use hash_api_key() which uses argon2.
    This is only used for verifying existing SHA-256 hashes.
    """
    salted = f"gpu-server-v1:{api_key}"
    return hashlib.sha256(salted.encode('utf-8')).hexdigest()


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for secure storage using argon2id.

    Argon2id is the recommended algorithm for password/key hashing as it's
    resistant to both GPU and side-channel attacks.

    Args:
        api_key: The plaintext API key

    Returns:
        Argon2id hash of the API key (includes salt, params, and hash)
    """
    try:
        from argon2 import PasswordHasher
        from argon2.profiles import RFC_9106_LOW_MEMORY

        # Use argon2id with secure parameters
        ph = PasswordHasher(
            time_cost=ARGON2_TIME_COST,
            memory_cost=ARGON2_MEMORY_COST,
            parallelism=ARGON2_PARALLELISM,
            hash_len=ARGON2_HASH_LEN,
            salt_len=ARGON2_SALT_LEN,
        )

        return ph.hash(api_key)
    except ImportError:
        # Fallback to SHA-256 if argon2 not installed (with warning)
        logger.warning(
            "argon2-cffi not installed, falling back to SHA-256. "
            "Install argon2-cffi for secure API key hashing: pip install argon2-cffi"
        )
        return _hash_api_key_legacy(api_key)


def verify_api_key(provided_key: str, stored_hashes: List[str]) -> bool:
    """
    Verify an API key against stored hashes.

    Supports both argon2 (preferred) and legacy SHA-256 hashes for
    backward compatibility. Uses constant-time comparison.

    Args:
        provided_key: The API key provided by the client
        stored_hashes: List of valid API key hashes (argon2 or SHA-256)

    Returns:
        True if the key is valid, False otherwise
    """
    if not provided_key or not stored_hashes:
        return False

    valid = False
    legacy_hash_found = False

    try:
        from argon2 import PasswordHasher
        from argon2.exceptions import VerifyMismatchError, InvalidHashError

        ph = PasswordHasher(
            time_cost=ARGON2_TIME_COST,
            memory_cost=ARGON2_MEMORY_COST,
            parallelism=ARGON2_PARALLELISM,
            hash_len=ARGON2_HASH_LEN,
            salt_len=ARGON2_SALT_LEN,
        )

        # Check against all stored hashes
        # We iterate through ALL hashes to prevent timing attacks
        for stored_hash in stored_hashes:
            if _is_argon2_hash(stored_hash):
                # Argon2 hash - use argon2 verification
                try:
                    ph.verify(stored_hash, provided_key)
                    valid = True
                    # Don't break - continue checking to maintain constant time
                except (VerifyMismatchError, InvalidHashError):
                    pass
            elif _is_legacy_sha256_hash(stored_hash):
                # Legacy SHA-256 hash - use legacy verification
                legacy_hash_found = True
                provided_legacy_hash = _hash_api_key_legacy(provided_key)
                if hmac.compare_digest(provided_legacy_hash, stored_hash):
                    valid = True
                    # Don't break - continue checking
    except ImportError:
        # argon2 not installed, fall back to legacy verification only
        provided_legacy_hash = _hash_api_key_legacy(provided_key)
        for stored_hash in stored_hashes:
            if _is_legacy_sha256_hash(stored_hash):
                legacy_hash_found = True
                if hmac.compare_digest(provided_legacy_hash, stored_hash):
                    valid = True

    # Log deprecation warning if legacy hashes are still in use
    if legacy_hash_found and valid:
        logger.warning(
            "API key verified using legacy SHA-256 hash. "
            "Please regenerate API keys and use argon2 hashes for improved security."
        )

    return valid


def generate_api_key() -> str:
    """
    Generate a cryptographically secure API key.

    Returns:
        A URL-safe base64-encoded 32-byte random key
    """
    return secrets.token_urlsafe(32)


@dataclass
class ServerConfig:
    """WebSocket server configuration."""
    host: str = "0.0.0.0"
    port: int = 8765
    max_connections: int = 10
    max_message_size: int = 500 * 1024 * 1024  # 500MB default for long recordings
    # Rate limiting for PROCESS requests (stricter)
    rate_limit_requests: int = 10  # Max PROCESS requests per window
    rate_limit_window: int = 60  # Window in seconds
    # Rate limiting for ALL messages (DoS protection)
    rate_limit_messages: int = 100  # Max messages per second per connection
    rate_limit_messages_window: int = 1  # Window in seconds (1 = per second)


@dataclass
class AuthConfig:
    """Authentication configuration."""
    enabled: bool = True
    api_keys: List[str] = field(default_factory=list)  # Plaintext keys (for backward compat, will be hashed)
    api_key_hashes: List[str] = field(default_factory=list)  # Pre-hashed keys (preferred)


@dataclass
class QueueConfig:
    """Request queue configuration."""
    max_size: int = 100
    request_timeout: int = 3600  # seconds - timeout for requests waiting in queue
    processing_timeout: int = 1800  # seconds - timeout for active processing (30 min default)


@dataclass
class WhisperConfig:
    """Whisper transcription configuration."""
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    language: Optional[str] = None


@dataclass
class PyAnnoteConfig:
    """PyAnnote diarization configuration."""
    device: str = "cuda"
    huggingface_token: str = ""
    model: str = "pyannote/speaker-diarization-3.1"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: Optional[str] = None


@dataclass
class TLSConfig:
    """TLS/SSL configuration."""
    enabled: bool = False
    cert_file: Optional[str] = None  # Path to certificate file (.pem or .crt)
    key_file: Optional[str] = None   # Path to private key file (.pem or .key)
    ca_file: Optional[str] = None    # Path to CA certificate for client verification (optional)
    require_client_cert: bool = False  # Whether to require client certificates


@dataclass
class VideoEncodingConfig:
    """Video encoding configuration."""
    enabled: bool = False
    ffmpeg_path: str = "ffmpeg"
    preferred_codecs: List[str] = field(default_factory=lambda: ["av1_nvenc", "hevc_nvenc", "h264_nvenc"])
    default_preset: str = "p4"          # p1(fastest)..p7(best quality)
    default_quality: int = 23           # CQ mode, 0-51
    max_sessions: int = 2              # NVENC hardware session limit
    max_input_size: int = 2 * 1024 * 1024 * 1024  # 2GB
    data_upload_timeout: int = 300     # 5 min to complete upload after control message
    temp_directory: Optional[str] = None  # Must NOT be tmpfs for large files
    # Protocol v1.1: shared filesystem transfer
    shared_paths: List[str] = field(default_factory=list)  # Paths accessible to both client and server


@dataclass
class VideoQueueConfig:
    """Video encoding queue configuration."""
    max_size: int = 50
    request_timeout: int = 7200        # 2 hours
    processing_timeout: int = 3600     # 1 hour


@dataclass
class LlmConfig:
    """On-device LLM (summarisation) configuration.

    Disabled by default: the LLM workload is opt-in and only runs where a GGUF
    model and enough VRAM exist. When enabled the server advertises the ``llm``
    capability and accepts LLM_GENERATE requests. The tuned defaults mirror the
    app's LlamaCppProvider (Qwen2.5-14B-Q5_K_M on a 16GB card): q8 KV cache +
    flash attention so a 14B fits ~28K context.

    ``estimated_vram_gb`` is the FULL resident footprint the arbiter reserves:
    weights + the ENTIRE q8 KV cache (llama.cpp allocates all of n_ctx up front
    at construction — it does NOT grow during generation) + compute buffers
    (~13 GB for 14B-Q5 at 28K, measured in trials). ``kv_headroom_gb`` is only a
    small TRANSIENT scratch margin per request — NOT KV growth, which is already
    resident above.
    """
    enabled: bool = False
    model_path: str = ""
    n_ctx: int = 28672
    n_gpu_layers: int = -1
    flash_attn: bool = True
    kv_cache_type: int = 8  # llama_cpp type 8 = Q8_0 KV
    temperature: float = 0.3
    max_tokens: int = 4000
    estimated_vram_gb: float = 13.0        # weights + FULL q8 28K KV + buffers (all resident)
    kv_headroom_gb: float = 0.5            # small transient per-request scratch margin

    @property
    def estimated_vram_bytes(self) -> int:
        return int(self.estimated_vram_gb * 1024 ** 3)

    @property
    def kv_headroom_bytes(self) -> int:
        return int(self.kv_headroom_gb * 1024 ** 3)


@dataclass
class LlmQueueConfig:
    """LLM request queue configuration."""
    max_size: int = 50
    request_timeout: int = 1800        # 30 min waiting in queue
    processing_timeout: int = 600      # 10 min per generation


@dataclass
class GpuConfig:
    """GPU resource-arbiter configuration."""
    vram_headroom_gb: float = 1.0  # safety margin the arbiter never hands out

    @property
    def vram_headroom_bytes(self) -> int:
        return int(self.vram_headroom_gb * 1024 ** 3)


@dataclass
class Config:
    """Main configuration container."""
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    pyannote: PyAnnoteConfig = field(default_factory=PyAnnoteConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tls: TLSConfig = field(default_factory=TLSConfig)
    video_encoding: VideoEncodingConfig = field(default_factory=VideoEncodingConfig)
    video_queue: VideoQueueConfig = field(default_factory=VideoQueueConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    llm_queue: LlmQueueConfig = field(default_factory=LlmQueueConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)


def load_config(config_path: Optional[Path] = None, fail_on_error: bool = True) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, searches default locations.
        fail_on_error: If True (default), raise ConfigurationError on parse errors.
            If False, log error and continue with defaults.

    Returns:
        Config object with loaded values.

    Raises:
        ConfigurationError: If fail_on_error is True and config file cannot be parsed.
    """
    # Default search paths
    search_paths = [
        Path("config.yaml"),
        Path("config.yml"),
        Path.home() / ".config" / "gpu-server" / "config.yaml",
        Path("/etc/gpu-server/config.yaml"),
    ]

    if config_path:
        search_paths.insert(0, Path(config_path))

    # Find first existing config file
    config_file = None
    for path in search_paths:
        if path.exists():
            config_file = path
            break

    config = Config()

    if config_file:
        logger.info(f"Loading configuration from: {config_file}")
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f) or {}

            # Server config
            if 'server' in data:
                config.server = ServerConfig(
                    host=data['server'].get('host', config.server.host),
                    port=data['server'].get('port', config.server.port),
                    max_connections=data['server'].get('max_connections', config.server.max_connections),
                    max_message_size=data['server'].get('max_message_size', config.server.max_message_size),
                    rate_limit_requests=data['server'].get('rate_limit_requests', config.server.rate_limit_requests),
                    rate_limit_window=data['server'].get('rate_limit_window', config.server.rate_limit_window),
                    rate_limit_messages=data['server'].get('rate_limit_messages', config.server.rate_limit_messages),
                    rate_limit_messages_window=data['server'].get('rate_limit_messages_window', config.server.rate_limit_messages_window),
                )

            # Auth config
            if 'auth' in data:
                config.auth = AuthConfig(
                    enabled=data['auth'].get('enabled', config.auth.enabled),
                    api_keys=data['auth'].get('api_keys', config.auth.api_keys),
                    api_key_hashes=data['auth'].get('api_key_hashes', config.auth.api_key_hashes),
                )

            # Queue config
            if 'queue' in data:
                config.queue = QueueConfig(
                    max_size=data['queue'].get('max_size', config.queue.max_size),
                    request_timeout=data['queue'].get('request_timeout', config.queue.request_timeout),
                    processing_timeout=data['queue'].get('processing_timeout', config.queue.processing_timeout),
                )

            # Whisper config
            if 'whisper' in data:
                config.whisper = WhisperConfig(
                    model=data['whisper'].get('model', config.whisper.model),
                    device=data['whisper'].get('device', config.whisper.device),
                    compute_type=data['whisper'].get('compute_type', config.whisper.compute_type),
                    beam_size=data['whisper'].get('beam_size', config.whisper.beam_size),
                    language=data['whisper'].get('language'),
                )

            # PyAnnote config
            if 'pyannote' in data:
                config.pyannote = PyAnnoteConfig(
                    device=data['pyannote'].get('device', config.pyannote.device),
                    huggingface_token=data['pyannote'].get('huggingface_token', config.pyannote.huggingface_token),
                    model=data['pyannote'].get('model', config.pyannote.model),
                )

            # Logging config
            if 'logging' in data:
                config.logging = LoggingConfig(
                    level=data['logging'].get('level', config.logging.level),
                    file=data['logging'].get('file'),
                )

            # TLS config
            if 'tls' in data:
                config.tls = TLSConfig(
                    enabled=data['tls'].get('enabled', config.tls.enabled),
                    cert_file=data['tls'].get('cert_file'),
                    key_file=data['tls'].get('key_file'),
                    ca_file=data['tls'].get('ca_file'),
                    require_client_cert=data['tls'].get('require_client_cert', config.tls.require_client_cert),
                )

            # Video encoding config
            if 'video_encoding' in data:
                ve = data['video_encoding']
                config.video_encoding = VideoEncodingConfig(
                    enabled=ve.get('enabled', config.video_encoding.enabled),
                    ffmpeg_path=ve.get('ffmpeg_path', config.video_encoding.ffmpeg_path),
                    preferred_codecs=ve.get('preferred_codecs', config.video_encoding.preferred_codecs),
                    default_preset=ve.get('default_preset', config.video_encoding.default_preset),
                    default_quality=ve.get('default_quality', config.video_encoding.default_quality),
                    max_sessions=ve.get('max_sessions', config.video_encoding.max_sessions),
                    max_input_size=ve.get('max_input_size', config.video_encoding.max_input_size),
                    data_upload_timeout=ve.get('data_upload_timeout', config.video_encoding.data_upload_timeout),
                    temp_directory=ve.get('temp_directory', config.video_encoding.temp_directory),
                    shared_paths=ve.get('shared_paths', config.video_encoding.shared_paths),
                )

            # Video queue config
            if 'video_queue' in data:
                vq = data['video_queue']
                config.video_queue = VideoQueueConfig(
                    max_size=vq.get('max_size', config.video_queue.max_size),
                    request_timeout=vq.get('request_timeout', config.video_queue.request_timeout),
                    processing_timeout=vq.get('processing_timeout', config.video_queue.processing_timeout),
                )

            if 'llm' in data:
                llm = data['llm']
                config.llm = LlmConfig(
                    enabled=llm.get('enabled', config.llm.enabled),
                    model_path=llm.get('model_path', config.llm.model_path),
                    n_ctx=llm.get('n_ctx', config.llm.n_ctx),
                    n_gpu_layers=llm.get('n_gpu_layers', config.llm.n_gpu_layers),
                    flash_attn=llm.get('flash_attn', config.llm.flash_attn),
                    kv_cache_type=llm.get('kv_cache_type', config.llm.kv_cache_type),
                    temperature=llm.get('temperature', config.llm.temperature),
                    max_tokens=llm.get('max_tokens', config.llm.max_tokens),
                    estimated_vram_gb=llm.get('estimated_vram_gb', config.llm.estimated_vram_gb),
                    kv_headroom_gb=llm.get('kv_headroom_gb', config.llm.kv_headroom_gb),
                )

            if 'llm_queue' in data:
                lq = data['llm_queue']
                config.llm_queue = LlmQueueConfig(
                    max_size=lq.get('max_size', config.llm_queue.max_size),
                    request_timeout=lq.get('request_timeout', config.llm_queue.request_timeout),
                    processing_timeout=lq.get('processing_timeout', config.llm_queue.processing_timeout),
                )

            if 'gpu' in data:
                config.gpu = GpuConfig(
                    vram_headroom_gb=data['gpu'].get('vram_headroom_gb', config.gpu.vram_headroom_gb),
                )

        except Exception as e:
            if fail_on_error:
                raise ConfigurationError(f"Failed to parse config file '{config_file}': {e}") from e
            else:
                logger.error(f"Error loading config file: {e}")
                logger.info("Using default configuration")
    else:
        logger.warning("No configuration file found, using defaults")

    # Environment variable overrides
    if os.environ.get('GPU_SERVER_HOST'):
        config.server.host = os.environ['GPU_SERVER_HOST']
    if os.environ.get('GPU_SERVER_PORT'):
        try:
            config.server.port = int(os.environ['GPU_SERVER_PORT'])
        except ValueError:
            raise ConfigurationError(
                f"Invalid GPU_SERVER_PORT environment variable: '{os.environ['GPU_SERVER_PORT']}'. "
                "Must be a valid integer."
            )
    if os.environ.get('GPU_SERVER_API_KEY'):
        # Read and immediately clear sensitive env var
        api_key = os.environ.pop('GPU_SERVER_API_KEY')
        config.auth.api_keys.append(api_key)
        del api_key  # Clear reference to minimize exposure time
    if os.environ.get('HUGGINGFACE_TOKEN'):
        # Read and immediately clear sensitive env var (consistent with API key handling)
        config.pyannote.huggingface_token = os.environ.pop('HUGGINGFACE_TOKEN')

    # LLM environment variable overrides
    if os.environ.get('GPU_SERVER_LLM_ENABLED'):
        config.llm.enabled = os.environ['GPU_SERVER_LLM_ENABLED'].lower() in ('true', '1', 'yes')
    if os.environ.get('GPU_SERVER_LLM_MODEL_PATH'):
        config.llm.model_path = os.environ['GPU_SERVER_LLM_MODEL_PATH']

    # Video encoding environment variable overrides
    if os.environ.get('GPU_SERVER_VIDEO_ENABLED'):
        config.video_encoding.enabled = os.environ['GPU_SERVER_VIDEO_ENABLED'].lower() in ('true', '1', 'yes')
    if os.environ.get('GPU_SERVER_FFMPEG_PATH'):
        config.video_encoding.ffmpeg_path = os.environ['GPU_SERVER_FFMPEG_PATH']
    if os.environ.get('GPU_SERVER_VIDEO_TEMP_DIR'):
        config.video_encoding.temp_directory = os.environ['GPU_SERVER_VIDEO_TEMP_DIR']
    if os.environ.get('GPU_SERVER_SHARED_PATHS'):
        # Comma-separated list of shared paths
        config.video_encoding.shared_paths = [
            p.strip() for p in os.environ['GPU_SERVER_SHARED_PATHS'].split(',') if p.strip()
        ]

    # TLS environment variable overrides
    if os.environ.get('GPU_SERVER_TLS_ENABLED'):
        config.tls.enabled = os.environ['GPU_SERVER_TLS_ENABLED'].lower() in ('true', '1', 'yes')
    if os.environ.get('GPU_SERVER_TLS_CERT'):
        config.tls.cert_file = os.environ['GPU_SERVER_TLS_CERT']
    if os.environ.get('GPU_SERVER_TLS_KEY'):
        config.tls.key_file = os.environ['GPU_SERVER_TLS_KEY']
    if os.environ.get('GPU_SERVER_TLS_CA'):
        config.tls.ca_file = os.environ['GPU_SERVER_TLS_CA']

    # Hash any plaintext API keys and add to hashed list
    # This allows backward compatibility while storing keys securely
    if config.auth.api_keys:
        for plaintext_key in config.auth.api_keys:
            hashed = hash_api_key(plaintext_key)
            if hashed not in config.auth.api_key_hashes:
                config.auth.api_key_hashes.append(hashed)
        # Clear plaintext keys list
        config.auth.api_keys = []
        logger.info("Plaintext API keys have been hashed (argon2) and stored securely")

    return config


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


def validate_config(config: Config, strict: bool = True):
    """
    Validate configuration for required settings.

    Args:
        config: The configuration to validate
        strict: If True (default), raises ConfigurationError for critical issues.
                If False, logs errors but continues (not recommended for production).

    Raises:
        ConfigurationError: If strict=True and validation fails
    """
    errors = []
    warnings = []

    # Auth validation - this is CRITICAL, always an error
    if config.auth.enabled and not config.auth.api_key_hashes:
        errors.append(
            "Authentication is enabled but no API keys are configured. "
            "Either add API keys or set auth.enabled=False"
        )

    # TLS validation
    if config.tls.enabled:
        if not config.tls.cert_file:
            errors.append("TLS is enabled but cert_file is not set")
        if not config.tls.key_file:
            errors.append("TLS is enabled but key_file is not set")

    # PyAnnote validation (HuggingFace token required for diarization)
    if not config.pyannote.huggingface_token:
        warnings.append("PyAnnote HuggingFace token not set - diarization will fail")

    # Server config validation
    if config.server.port < 1 or config.server.port > 65535:
        errors.append(f"Invalid server port: {config.server.port}")
    if config.server.max_connections < 1:
        errors.append(f"Invalid max_connections: {config.server.max_connections}")
    if config.server.max_message_size < 1024 * 1024:  # Minimum 1MB
        errors.append(f"max_message_size too small (minimum 1MB): {config.server.max_message_size}")
    if config.server.max_message_size > 500 * 1024 * 1024:  # Maximum 500MB
        warnings.append(f"max_message_size very large ({config.server.max_message_size / 1024 / 1024:.0f}MB), may cause memory issues")

    # Whisper config validation
    if config.whisper.model not in ALLOWED_WHISPER_MODELS:
        errors.append(
            f"Invalid whisper.model: '{config.whisper.model}'. "
            f"Must be one of: {', '.join(sorted(ALLOWED_WHISPER_MODELS))}"
        )
    valid_compute_types = {'float16', 'float32', 'int8', 'int8_float16', 'int8_float32'}
    if config.whisper.compute_type not in valid_compute_types:
        errors.append(
            f"Invalid whisper.compute_type: '{config.whisper.compute_type}'. "
            f"Must be one of: {', '.join(sorted(valid_compute_types))}"
        )
    if config.whisper.beam_size < 1 or config.whisper.beam_size > 10:
        errors.append(
            f"Invalid whisper.beam_size: {config.whisper.beam_size}. "
            "Must be between 1 and 10"
        )

    # Device validation for both Whisper and PyAnnote
    valid_device_patterns = {'cpu', 'cuda'}
    for name, device in [('whisper', config.whisper.device), ('pyannote', config.pyannote.device)]:
        # Allow "cpu", "cuda", or "cuda:N" where N is a digit
        if device not in valid_device_patterns:
            if device.startswith('cuda:'):
                suffix = device[5:]
                if not suffix.isdigit():
                    errors.append(
                        f"Invalid {name}.device: '{device}'. "
                        "Must be 'cpu', 'cuda', or 'cuda:N' where N is a GPU index"
                    )
            else:
                errors.append(
                    f"Invalid {name}.device: '{device}'. "
                    "Must be 'cpu', 'cuda', or 'cuda:N' where N is a GPU index"
                )

    # Video encoding validation
    if config.video_encoding.enabled:
        if config.video_encoding.default_quality < 0 or config.video_encoding.default_quality > 51:
            errors.append(f"Invalid video_encoding.default_quality: {config.video_encoding.default_quality}. Must be 0-51")
        valid_presets = {f"p{i}" for i in range(1, 8)}
        if config.video_encoding.default_preset not in valid_presets:
            errors.append(f"Invalid video_encoding.default_preset: '{config.video_encoding.default_preset}'. Must be p1-p7")
        if config.video_encoding.max_sessions < 1:
            errors.append(f"Invalid video_encoding.max_sessions: {config.video_encoding.max_sessions}. Must be >= 1")
        if config.video_encoding.temp_directory:
            temp_path = Path(config.video_encoding.temp_directory)
            if not temp_path.exists():
                warnings.append(f"Video encoding temp_directory does not exist: {temp_path}")

    # LLM validation — catch a misconfigured LLM at startup rather than on the
    # first request (after a lease acquire and possible eviction).
    if config.llm.enabled:
        if not config.llm.model_path:
            errors.append("llm.enabled is true but llm.model_path is empty")
        elif not Path(config.llm.model_path).is_file():
            errors.append(f"llm.model_path does not exist: {config.llm.model_path}")
        if config.llm.n_ctx < 512:
            errors.append(f"Invalid llm.n_ctx: {config.llm.n_ctx}. Must be >= 512")
        if config.llm.estimated_vram_gb <= 0:
            errors.append(f"Invalid llm.estimated_vram_gb: {config.llm.estimated_vram_gb}. Must be > 0")

    # Log warnings
    for warning in warnings:
        logger.warning(f"Configuration warning: {warning}")

    # Handle errors
    if errors:
        error_msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        if strict:
            raise ConfigurationError(error_msg)
        else:
            logger.error(error_msg)

    return len(errors) == 0


def setup_logging(config: LoggingConfig):
    """Configure logging based on config."""
    level = getattr(logging, config.level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if config.file:
        handlers.append(logging.FileHandler(config.file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def create_ssl_context(tls_config: TLSConfig):
    """
    Create an SSL context for TLS connections.

    Args:
        tls_config: TLS configuration

    Returns:
        ssl.SSLContext if TLS is enabled, None otherwise

    Raises:
        ValueError: If TLS is enabled but required files are missing
    """
    if not tls_config.enabled:
        return None

    import ssl

    # Validate required files
    if not tls_config.cert_file:
        raise ValueError("TLS is enabled but cert_file is not set")
    if not tls_config.key_file:
        raise ValueError("TLS is enabled but key_file is not set")

    cert_path = Path(tls_config.cert_file)
    key_path = Path(tls_config.key_file)

    if not cert_path.exists():
        raise ValueError(f"TLS certificate file not found: {cert_path}")
    if not key_path.exists():
        raise ValueError(f"TLS key file not found: {key_path}")

    # Create SSL context for server
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(
        certfile=str(cert_path),
        keyfile=str(key_path),
    )

    # Optional: Load CA certificate for client verification
    if tls_config.ca_file:
        ca_path = Path(tls_config.ca_file)
        if not ca_path.exists():
            raise ValueError(f"TLS CA file not found: {ca_path}")
        ssl_context.load_verify_locations(cafile=str(ca_path))

    # Configure client certificate verification
    if tls_config.require_client_cert:
        ssl_context.verify_mode = ssl.CERT_REQUIRED
    elif tls_config.ca_file:
        ssl_context.verify_mode = ssl.CERT_OPTIONAL
    else:
        ssl_context.verify_mode = ssl.CERT_NONE

    # Use secure defaults
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
    ssl_context.set_ciphers('ECDHE+AESGCM:DHE+AESGCM:ECDHE+CHACHA20:DHE+CHACHA20')

    logger.info(f"TLS enabled with certificate: {cert_path}")
    return ssl_context
