"""
Configuration Management

Loads and validates server configuration from YAML files.
"""
import os
import yaml
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """WebSocket server configuration."""
    host: str = "0.0.0.0"
    port: int = 8765
    max_connections: int = 10


@dataclass
class AuthConfig:
    """Authentication configuration."""
    enabled: bool = True
    api_keys: List[str] = field(default_factory=list)


@dataclass
class QueueConfig:
    """Request queue configuration."""
    max_size: int = 100
    request_timeout: int = 3600  # seconds


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
class Config:
    """Main configuration container."""
    server: ServerConfig = field(default_factory=ServerConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    pyannote: PyAnnoteConfig = field(default_factory=PyAnnoteConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


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
                )

            # Auth config
            if 'auth' in data:
                config.auth = AuthConfig(
                    enabled=data['auth'].get('enabled', config.auth.enabled),
                    api_keys=data['auth'].get('api_keys', config.auth.api_keys),
                )

            # Queue config
            if 'queue' in data:
                config.queue = QueueConfig(
                    max_size=data['queue'].get('max_size', config.queue.max_size),
                    request_timeout=data['queue'].get('request_timeout', config.queue.request_timeout),
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

    return config


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
