"""
Structured Logging Configuration

Provides consistent, structured logging across the GPU server with:
- Request correlation via context variables
- JSON formatting for production
- Performance timing helpers
- Standardized log fields
"""
import asyncio
import contextvars
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from functools import wraps
from typing import Optional, Dict, Any, Callable

# Context variables for request tracking
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar('request_id', default='')
client_addr_var: contextvars.ContextVar[str] = contextvars.ContextVar('client_addr', default='')
meeting_name_var: contextvars.ContextVar[str] = contextvars.ContextVar('meeting_name', default='')


@dataclass
class LogContext:
    """Structured log context data."""
    request_id: str = ''
    client_addr: str = ''
    meeting_name: str = ''

    @classmethod
    def current(cls) -> 'LogContext':
        """Get current context from context variables."""
        return cls(
            request_id=request_id_var.get(),
            client_addr=client_addr_var.get(),
            meeting_name=meeting_name_var.get(),
        )


def set_request_context(
    request_id: str = '',
    client_addr: str = '',
    meeting_name: str = '',
) -> None:
    """Set request context for logging."""
    if request_id:
        request_id_var.set(request_id)
    if client_addr:
        client_addr_var.set(client_addr)
    if meeting_name:
        meeting_name_var.set(meeting_name)


def clear_request_context() -> None:
    """Clear request context after request completion."""
    request_id_var.set('')
    client_addr_var.set('')
    meeting_name_var.set('')


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Produces logs in JSON format suitable for log aggregation systems.
    """

    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add context if available
        if self.include_context:
            ctx = LogContext.current()
            if ctx.request_id:
                log_data['request_id'] = ctx.request_id
            if ctx.client_addr:
                log_data['client_addr'] = ctx.client_addr
            if ctx.meeting_name:
                log_data['meeting_name'] = ctx.meeting_name

        # Add extra fields from record
        if hasattr(record, 'extra_data') and record.extra_data:
            log_data['data'] = record.extra_data

        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add source location for errors
        if record.levelno >= logging.WARNING:
            log_data['location'] = {
                'file': record.pathname,
                'line': record.lineno,
                'function': record.funcName,
            }

        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable formatter for console output.

    Includes request context when available for easier debugging.
    """

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        # Get context
        ctx = LogContext.current()

        # Build context prefix
        context_parts = []
        if ctx.request_id:
            context_parts.append(f"req={ctx.request_id[:12]}")
        if ctx.client_addr:
            context_parts.append(f"client={ctx.client_addr}")

        context_str = f"[{' '.join(context_parts)}] " if context_parts else ""

        # Format timestamp
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

        # Build message
        level = record.levelname
        if self.use_colors and level in self.COLORS:
            level_str = f"{self.COLORS[level]}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        message = f"{timestamp} {level_str} {record.name:25} {context_str}{record.getMessage()}"

        # Add exception if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes request context.

    Use this instead of logging.getLogger() for request-aware logging.
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        # Add extra data if provided
        extra = kwargs.get('extra', {})
        if 'data' in kwargs:
            extra['extra_data'] = kwargs.pop('data')
            kwargs['extra'] = extra
        return msg, kwargs

    def with_data(self, **data) -> 'ContextLogger':
        """Create a child logger with additional data context."""
        new_extra = {**self.extra, 'extra_data': data}
        return ContextLogger(self.logger, new_extra)


def get_logger(name: str) -> ContextLogger:
    """
    Get a context-aware logger.

    Args:
        name: Logger name (usually __name__)

    Returns:
        ContextLogger instance
    """
    return ContextLogger(logging.getLogger(name), {})


def configure_logging(
    level: str = 'INFO',
    json_format: bool = False,
    use_colors: bool = True,
) -> None:
    """
    Configure logging for the GPU server.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Use JSON format (for production/log aggregation)
        use_colors: Use ANSI colors in console output
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(ConsoleFormatter(use_colors=use_colors))

    root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


@dataclass
class PerformanceMetrics:
    """Container for performance timing metrics."""
    operation: str
    start_time: float
    end_time: float = 0.0
    success: bool = True
    error: str = ''
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def finish(self, success: bool = True, error: str = '') -> 'PerformanceMetrics':
        """Mark operation as complete."""
        self.end_time = time.time()
        self.success = success
        self.error = error
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'operation': self.operation,
            'duration_ms': round(self.duration_ms, 2),
            'success': self.success,
            'error': self.error,
            **self.metadata,
        }


class PerformanceTimer:
    """
    Context manager for timing operations.

    Usage:
        with PerformanceTimer('transcribe', logger) as timer:
            timer.add_metadata('model', 'large-v3')
            result = do_transcription()
            timer.add_metadata('segments', len(result))
    """

    def __init__(
        self,
        operation: str,
        logger: Optional[ContextLogger] = None,
        log_level: int = logging.INFO,
    ):
        self.metrics = PerformanceMetrics(operation=operation, start_time=time.time())
        self.logger = logger
        self.log_level = log_level

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the timing record."""
        self.metrics.metadata[key] = value

    def __enter__(self) -> 'PerformanceTimer':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            self.metrics.finish(success=False, error=str(exc_val))
        else:
            self.metrics.finish(success=True)

        if self.logger:
            data = self.metrics.to_dict()
            self.logger.log(
                self.log_level,
                f"{self.metrics.operation} completed in {self.metrics.duration_ms:.1f}ms",
                data=data,
            )


def timed(operation: str, logger_name: str = None):
    """
    Decorator for timing function execution.

    Args:
        operation: Name of the operation for logging
        logger_name: Logger name (defaults to function's module)

    Usage:
        @timed('transcribe_audio')
        async def transcribe(audio_data: bytes) -> str:
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = logger_name or func.__module__
            log = get_logger(name)
            with PerformanceTimer(operation, log):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = logger_name or func.__module__
            log = get_logger(name)
            with PerformanceTimer(operation, log):
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Standard log event types for consistency
class LogEvents:
    """Standard log event names for consistency."""
    # Connection events
    CLIENT_CONNECTED = 'client_connected'
    CLIENT_DISCONNECTED = 'client_disconnected'
    CLIENT_AUTHENTICATED = 'client_authenticated'
    AUTH_FAILED = 'auth_failed'

    # Request events
    REQUEST_RECEIVED = 'request_received'
    REQUEST_QUEUED = 'request_queued'
    REQUEST_STARTED = 'request_started'
    REQUEST_COMPLETED = 'request_completed'
    REQUEST_FAILED = 'request_failed'
    REQUEST_CANCELLED = 'request_cancelled'
    REQUEST_TIMEOUT = 'request_timeout'

    # Processing events
    MODEL_LOADING = 'model_loading'
    MODEL_LOADED = 'model_loaded'
    TRANSCRIPTION_STARTED = 'transcription_started'
    TRANSCRIPTION_COMPLETED = 'transcription_completed'
    DIARIZATION_STARTED = 'diarization_started'
    DIARIZATION_COMPLETED = 'diarization_completed'
    EMBEDDING_EXTRACTION_STARTED = 'embedding_extraction_started'
    EMBEDDING_EXTRACTION_COMPLETED = 'embedding_extraction_completed'

    # Server events
    SERVER_STARTED = 'server_started'
    SERVER_STOPPED = 'server_stopped'
    QUEUE_FULL = 'queue_full'
    RATE_LIMITED = 'rate_limited'
