"""
WebSocket Server

Main server that handles client connections and routes requests.
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List

import websockets
from websockets.server import WebSocketServerProtocol

from . import __version__
from .config import Config, create_ssl_context, verify_api_key
from .logging_config import (
    get_logger, set_request_context, clear_request_context,
    LogEvents, PerformanceTimer
)
from .protocol import (
    MessageType, parse_message, ProcessRequest, AuthMessage,
    ProgressMessage, ProcessingStage, ErrorMessage, AuthErrorMessage,
    CancelledMessage, PROTOCOL_VERSION_STRING, is_version_compatible
)
from .queue_manager import QueueManager
from .validation import ValidationError
from .worker import GPUWorker

logger = get_logger(__name__)


@dataclass
class RateLimitTracker:
    """Tracks rate limit state for a client (for PROCESS requests)."""
    request_times: List[float] = field(default_factory=list)
    # Maximum stored entries to prevent unbounded growth
    # Even if pruning isn't called, list stays bounded
    _max_entries: int = 1000

    def record_request(self):
        """Record a request timestamp."""
        # Use monotonic time to avoid issues with clock adjustments
        self.request_times.append(time.monotonic())
        # Cap list size to prevent unbounded growth
        if len(self.request_times) > self._max_entries:
            # Keep only the most recent entries
            self.request_times = self.request_times[-self._max_entries:]

    def get_request_count(self, window_seconds: float) -> int:
        """Get number of requests in the time window."""
        cutoff = time.monotonic() - window_seconds
        # Prune old entries
        self.request_times = [t for t in self.request_times if t > cutoff]
        return len(self.request_times)


@dataclass
class MessageRateLimiter:
    """
    Tracks per-message rate limits for DoS protection.

    Uses a sliding window counter for efficient rate limiting of all
    message types, protecting against flood attacks.
    """
    message_times: List[float] = field(default_factory=list)
    _max_entries: int = 200  # Small since window is typically 1 second

    def check_and_record(self, limit: int, window_seconds: float) -> bool:
        """
        Check if message is allowed and record it.

        Args:
            limit: Maximum messages allowed in window
            window_seconds: Time window in seconds

        Returns:
            True if message is allowed, False if rate limited
        """
        # Use monotonic time to avoid issues with clock adjustments
        now = time.monotonic()
        cutoff = now - window_seconds

        # Prune old entries
        self.message_times = [t for t in self.message_times if t > cutoff]

        # Check limit
        if len(self.message_times) >= limit:
            return False

        # Record this message
        self.message_times.append(now)

        # Cap list size (shouldn't hit this often with small windows)
        if len(self.message_times) > self._max_entries:
            self.message_times = self.message_times[-self._max_entries:]

        return True

    def get_message_count(self, window_seconds: float) -> int:
        """Get current message count in window (for logging)."""
        cutoff = time.monotonic() - window_seconds
        return sum(1 for t in self.message_times if t > cutoff)


class GPUServer:
    """
    WebSocket server for GPU processing requests.

    Handles:
    - Client authentication
    - Request queuing
    - Progress streaming
    - Result delivery
    """

    def __init__(self, config: Config):
        """
        Initialize the server.

        Args:
            config: Server configuration
        """
        self.config = config
        self.queue = QueueManager(
            max_size=config.queue.max_size,
            request_timeout=config.queue.request_timeout,
        )
        self.worker = GPUWorker(config, self.queue)

        self._authenticated_clients: Set[WebSocketServerProtocol] = set()
        self._pending_connections: Set[WebSocketServerProtocol] = set()  # Pre-auth connections
        self._client_info: Dict[WebSocketServerProtocol, dict] = {}
        self._rate_limiters: Dict[WebSocketServerProtocol, RateLimitTracker] = {}
        self._message_limiters: Dict[WebSocketServerProtocol, MessageRateLimiter] = {}
        self._server = None
        self._worker_task = None

    async def start(self):
        """Start the server."""
        logger.info(
            "Starting GPU Server",
            data={
                'host': self.config.server.host,
                'port': self.config.server.port,
                'max_connections': self.config.server.max_connections,
                'queue_max_size': self.config.queue.max_size,
            }
        )

        # Create SSL context if TLS is enabled
        ssl_context = create_ssl_context(self.config.tls)

        # Start the worker
        self._worker_task = asyncio.create_task(self.worker.start())

        # Start WebSocket server
        self._server = await websockets.serve(
            self._handle_connection,
            self.config.server.host,
            self.config.server.port,
            max_size=self.config.server.max_message_size,
            ping_interval=30,
            ping_timeout=10,
            ssl=ssl_context,
        )

        max_mb = self.config.server.max_message_size / (1024 * 1024)
        protocol = "wss" if ssl_context else "ws"

        logger.info(
            LogEvents.SERVER_STARTED,
            data={
                'protocol': protocol,
                'host': self.config.server.host,
                'port': self.config.server.port,
                'max_message_size_mb': max_mb,
                'tls_enabled': ssl_context is not None,
                'auth_enabled': self.config.auth.enabled,
            }
        )

    async def stop(self):
        """Stop the server."""
        logger.info("Stopping GPU Server...", data={'connected_clients': len(self._authenticated_clients)})

        # Stop accepting new connections
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Stop the worker
        await self.worker.stop()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        # Close all client connections with timeout to prevent hanging on unresponsive clients
        close_timeout = 5.0  # 5 seconds per client
        for ws in list(self._authenticated_clients):
            try:
                await asyncio.wait_for(ws.close(), timeout=close_timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Client close timed out after {close_timeout}s, forcing disconnect")
            except Exception as e:
                logger.warning(f"Error closing client connection: {e}")

        # Also close any pending (pre-auth) connections
        for ws in list(self._pending_connections):
            try:
                await asyncio.wait_for(ws.close(), timeout=close_timeout)
            except (asyncio.TimeoutError, Exception):
                pass  # Best effort for pending connections

        logger.info(LogEvents.SERVER_STOPPED)

    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str = ""):
        """
        Handle a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            path: Request path (unused)
        """
        client_addr = str(websocket.remote_address)
        set_request_context(client_addr=client_addr)

        logger.info(
            LogEvents.CLIENT_CONNECTED,
            data={'remote_address': client_addr, 'path': path}
        )

        # Enforce connection limit - check TOTAL connections (pending + authenticated)
        # This prevents FD exhaustion by rejecting connections early
        max_conn = self.config.server.max_connections
        total_connections = len(self._authenticated_clients) + len(self._pending_connections)
        if total_connections >= max_conn:
            logger.warning(
                f"Connection limit reached ({max_conn}), rejecting client",
                data={
                    'max_connections': max_conn,
                    'authenticated': len(self._authenticated_clients),
                    'pending': len(self._pending_connections),
                }
            )
            await websocket.send(AuthErrorMessage(
                error="Server connection limit reached, try again later",
                error_code="CONNECTION_LIMIT",
            ).to_json())
            await websocket.close(1013, "Server overloaded")  # 1013 = Try again later
            clear_request_context()
            return

        # Track as pending connection until authenticated
        self._pending_connections.add(websocket)

        try:
            # Wait for authentication
            authenticated = await self._authenticate(websocket)
            if not authenticated:
                return

            # Move from pending to authenticated
            self._pending_connections.discard(websocket)
            self._authenticated_clients.add(websocket)
            self._rate_limiters[websocket] = RateLimitTracker()
            self._message_limiters[websocket] = MessageRateLimiter()

            # Handle messages
            async for message in websocket:
                await self._handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(
                LogEvents.CLIENT_DISCONNECTED,
                data={'close_code': e.code, 'reason': str(e.reason) if e.reason else ''}
            )
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            logger.info(LogEvents.CLIENT_DISCONNECTED)  # Log disconnect for non-ConnectionClosed errors
        finally:
            self._pending_connections.discard(websocket)
            self._authenticated_clients.discard(websocket)
            self._client_info.pop(websocket, None)
            self._rate_limiters.pop(websocket, None)
            self._message_limiters.pop(websocket, None)
            clear_request_context()

    async def _authenticate(self, websocket: WebSocketServerProtocol) -> bool:
        """
        Authenticate a client connection.

        Args:
            websocket: The WebSocket connection

        Returns:
            True if authenticated, False otherwise
        """
        if not self.config.auth.enabled:
            # Auth disabled, accept all connections
            await websocket.send(json.dumps({
                "type": MessageType.AUTH_OK,
                "server_version": __version__,
                "protocol_version": PROTOCOL_VERSION_STRING,
            }))
            logger.info(LogEvents.CLIENT_AUTHENTICATED, data={'auth_disabled': True})
            return True

        try:
            # Wait for auth message (with timeout)
            message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            data = parse_message(message)

            if data.get("type") != MessageType.AUTH:
                logger.warning(
                    LogEvents.AUTH_FAILED,
                    data={'reason': 'invalid_message_type', 'received_type': data.get('type')}
                )
                await websocket.send(AuthErrorMessage(
                    error="Expected auth message",
                    error_code="INVALID_MESSAGE_TYPE",
                ).to_json())
                return False

            auth_msg = AuthMessage.from_dict(data)

            # Check protocol version compatibility
            compatible, version_error = is_version_compatible(auth_msg.protocol_version)
            if not compatible:
                logger.warning(
                    LogEvents.AUTH_FAILED,
                    data={
                        'reason': 'protocol_version_mismatch',
                        'client_version': auth_msg.protocol_version,
                        'server_version': PROTOCOL_VERSION_STRING,
                    }
                )
                await websocket.send(AuthErrorMessage(
                    error=version_error,
                    error_code="PROTOCOL_VERSION_MISMATCH",
                    server_protocol_version=PROTOCOL_VERSION_STRING,
                ).to_json())
                return False

            # Check API key using constant-time comparison
            if not verify_api_key(auth_msg.api_key, self.config.auth.api_key_hashes):
                logger.warning(LogEvents.AUTH_FAILED, data={'reason': 'invalid_api_key'})
                await websocket.send(AuthErrorMessage(
                    error="Invalid API key",
                    error_code="INVALID_API_KEY",
                ).to_json())
                return False

            # Store client info
            # Use monotonic time for consistent duration calculations
            self._client_info[websocket] = {
                "client_version": auth_msg.client_version,
                "protocol_version": auth_msg.protocol_version,
                "authenticated_at": time.monotonic(),
            }

            await websocket.send(json.dumps({
                "type": MessageType.AUTH_OK,
                "server_version": __version__,
                "protocol_version": PROTOCOL_VERSION_STRING,
                "queue_size": self.queue.size,
            }))

            logger.info(
                LogEvents.CLIENT_AUTHENTICATED,
                data={'client_version': auth_msg.client_version, 'protocol_version': auth_msg.protocol_version}
            )
            return True

        except asyncio.TimeoutError:
            logger.warning(LogEvents.AUTH_FAILED, data={'reason': 'timeout'})
            await websocket.send(AuthErrorMessage(
                error="Authentication timeout",
                error_code="AUTH_TIMEOUT",
            ).to_json())
            return False
        except Exception as e:
            logger.error(f"Authentication error: {e}", data={'reason': 'exception', 'error': str(e)})
            await websocket.send(AuthErrorMessage(
                error="Authentication error",
                error_code="AUTH_ERROR",
            ).to_json())
            return False

    async def _handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """
        Handle an incoming message from an authenticated client.

        Args:
            websocket: The WebSocket connection
            message: The raw message string
        """
        # Global per-message rate limiting (DoS protection)
        message_limiter = self._message_limiters.get(websocket)
        if message_limiter:
            allowed = message_limiter.check_and_record(
                limit=self.config.server.rate_limit_messages,
                window_seconds=self.config.server.rate_limit_messages_window,
            )
            if not allowed:
                logger.warning(
                    "Message rate limit exceeded",
                    data={
                        'limit': self.config.server.rate_limit_messages,
                        'window_seconds': self.config.server.rate_limit_messages_window,
                    }
                )
                await websocket.send(ErrorMessage(
                    request_id="",
                    error=f"Message rate limit exceeded. Max {self.config.server.rate_limit_messages} "
                          f"messages per {self.config.server.rate_limit_messages_window} second(s).",
                    recoverable=True,
                    error_code="MESSAGE_RATE_LIMIT_EXCEEDED",
                ).to_json())
                return

        data = parse_message(message)
        msg_type = data.get("type")

        if msg_type == MessageType.PROCESS:
            await self._handle_process_request(websocket, data)

        elif msg_type == MessageType.CANCEL:
            await self._handle_cancel(websocket, data)

        elif msg_type == MessageType.PING:
            await websocket.send(json.dumps({
                "type": MessageType.PONG,
                "queue_size": self.queue.size,
                "is_processing": self.worker.is_processing,
            }))

        else:
            logger.warning(f"Unknown message type: {msg_type}")
            await websocket.send(ErrorMessage(
                request_id=data.get("request_id", ""),
                error=f"Unknown message type: {msg_type}",
                error_code="UNKNOWN_MESSAGE_TYPE",
            ).to_json())

    async def _handle_process_request(self, websocket: WebSocketServerProtocol, data: dict):
        """Handle a processing request."""
        request_id = data.get("request_id", "unknown")

        try:
            request = ProcessRequest.from_dict(data)
            request_id = request.request_id

            # Set request context for logging
            set_request_context(
                request_id=request_id,
                meeting_name=request.meeting_name,
            )

            # Check rate limit
            rate_limiter = self._rate_limiters.get(websocket)
            if rate_limiter:
                request_count = rate_limiter.get_request_count(self.config.server.rate_limit_window)
                if request_count >= self.config.server.rate_limit_requests:
                    logger.warning(
                        LogEvents.RATE_LIMITED,
                        data={
                            'request_count': request_count,
                            'window_seconds': self.config.server.rate_limit_window,
                            'limit': self.config.server.rate_limit_requests,
                        }
                    )
                    await websocket.send(ErrorMessage(
                        request_id=request.request_id,
                        error=f"Rate limit exceeded. Max {self.config.server.rate_limit_requests} "
                              f"requests per {self.config.server.rate_limit_window} seconds.",
                        recoverable=True,
                        error_code="RATE_LIMIT_EXCEEDED",
                    ).to_json())
                    return
                rate_limiter.record_request()

            logger.info(
                LogEvents.REQUEST_RECEIVED,
                data={
                    'audio_size_bytes': len(request.audio_data),
                    'meeting_name': request.meeting_name,
                    'options': {
                        'transcribe': request.options.transcribe,
                        'diarize': request.options.diarize,
                        'extract_embeddings': request.options.extract_embeddings,
                        'whisper_model': request.options.whisper_model,
                        'language': request.options.language,
                        'num_speakers': request.options.num_speakers,
                    },
                    'priority': request.priority,
                }
            )

            # Create progress callback for queue position updates
            async def on_progress(msg: ProgressMessage):
                try:
                    await websocket.send(msg.to_json())
                except Exception as e:
                    logger.debug(f"Failed to send progress (connection may have closed): {e}")

            # Enqueue the request (atomic check-and-add inside the lock)
            success = await self.queue.enqueue(request, websocket, on_progress)

            if success:
                logger.info(LogEvents.REQUEST_QUEUED, data={'position': self.queue.size})
                await websocket.send(json.dumps({
                    "type": MessageType.QUEUED,
                    "request_id": request.request_id,
                    "position": self.queue.size,
                }))
            else:
                # Enqueue failed - queue is full (check happens atomically inside enqueue)
                logger.warning(LogEvents.QUEUE_FULL, data={'queue_size': self.queue.size})
                await websocket.send(ErrorMessage(
                    request_id=request.request_id,
                    error="Queue is full, please try again later",
                    recoverable=True,
                    error_code="QUEUE_FULL",
                ).to_json())

        except ValidationError as e:
            logger.warning(
                LogEvents.REQUEST_FAILED,
                data={'reason': 'validation_error', 'field': e.field, 'message': e.message}
            )
            await websocket.send(ErrorMessage(
                request_id=request_id,
                error=str(e),
                recoverable=False,
                error_code="VALIDATION_ERROR",
            ).to_json())

        except Exception as e:
            logger.error(
                LogEvents.REQUEST_FAILED,
                data={'reason': 'internal_error', 'error_type': type(e).__name__, 'error': str(e)},
                exc_info=True
            )
            await websocket.send(ErrorMessage(
                request_id=request_id,
                error="Internal server error",
                error_code="INTERNAL_ERROR",
            ).to_json())

    async def _handle_cancel(self, websocket: WebSocketServerProtocol, data: dict):
        """Handle a cancel request."""
        request_id = data.get("request_id", "")
        if not request_id:
            return

        set_request_context(request_id=request_id)
        success = await self.queue.cancel(request_id)

        if success:
            await websocket.send(CancelledMessage(request_id=request_id).to_json())
            logger.info(LogEvents.REQUEST_CANCELLED)
        else:
            logger.warning(
                f"Cancel failed: request not found or already processing",
                data={'reason': 'not_found_or_processing'}
            )
            await websocket.send(ErrorMessage(
                request_id=request_id,
                error="Request not found or already processing",
                error_code="CANCEL_NOT_FOUND",
            ).to_json())

    @property
    def connected_clients(self) -> int:
        """Number of connected clients."""
        return len(self._authenticated_clients)

    def get_stats(self) -> dict:
        """Get server statistics."""
        return {
            "connected_clients": self.connected_clients,
            "queue": self.queue.get_stats(),
            "worker": {
                "is_processing": self.worker.is_processing,
                "current_request": self.worker.current_request_id,
            },
        }
