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
from typing import List, Optional

logger = logging.getLogger(__name__)


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for secure storage.

    Uses SHA-256 with a constant salt prefix to prevent rainbow table attacks
    while keeping hashes deterministic for comparison.

    Args:
        api_key: The plaintext API key

    Returns:
        Hex-encoded hash of the API key
    """
    # Use a fixed prefix as salt (deterministic for comparisons)
    # This prevents rainbow table attacks while allowing hash comparison
    salted = f"gpu-server-v1:{api_key}"
    return hashlib.sha256(salted.encode('utf-8')).hexdigest()


def verify_api_key(provided_key: str, stored_hashes: List[str]) -> bool:
    """
    Verify an API key using constant-time comparison.

    Args:
        provided_key: The API key provided by the client
        stored_hashes: List of valid API key hashes

    Returns:
        True if the key is valid, False otherwise
    """
    if not provided_key or not stored_hashes:
        return False

    provided_hash = hash_api_key(provided_key)

    # Check against all stored hashes using constant-time comparison
    # We iterate through ALL hashes to prevent timing attacks
    valid = False
    for stored_hash in stored_hashes:
        if hmac.compare_digest(provided_hash, stored_hash):
            valid = True
        # Don't break early - check all to maintain constant time

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
    max_message_size: int = 100 * 1024 * 1024  # 100MB default
    # Rate limiting
    rate_limit_requests: int = 10  # Max requests per window
    rate_limit_window: int = 60  # Window in seconds


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
class Config:
    """Main configuration container."""
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    pyannote: PyAnnoteConfig = field(default_factory=PyAnnoteConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tls: TLSConfig = field(default_factory=TLSConfig)


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, searches default locations.

    Returns:
        Config object with loaded values.
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

        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            logger.info("Using default configuration")
    else:
        logger.warning("No configuration file found, using defaults")

    # Environment variable overrides
    if os.environ.get('GPU_SERVER_HOST'):
        config.server.host = os.environ['GPU_SERVER_HOST']
    if os.environ.get('GPU_SERVER_PORT'):
        config.server.port = int(os.environ['GPU_SERVER_PORT'])
    if os.environ.get('GPU_SERVER_API_KEY'):
        config.auth.api_keys.append(os.environ['GPU_SERVER_API_KEY'])
    if os.environ.get('HUGGINGFACE_TOKEN'):
        config.pyannote.huggingface_token = os.environ['HUGGINGFACE_TOKEN']

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
        # Clear plaintext keys from memory (security best practice)
        config.auth.api_keys = []
        logger.info("Plaintext API keys have been hashed and stored securely")

    return config


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


def validate_config(config: Config, strict: bool = False):
    """
    Validate configuration for required settings.

    Args:
        config: The configuration to validate
        strict: If True, raises ConfigurationError for issues. If False, logs warnings.

    Raises:
        ConfigurationError: If strict=True and validation fails
    """
    errors = []
    warnings = []

    # Auth validation
    if config.auth.enabled and not config.auth.api_key_hashes:
        errors.append("Authentication is enabled but no API keys are configured")

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
