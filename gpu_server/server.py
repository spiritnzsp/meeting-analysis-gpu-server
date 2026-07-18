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
    CancelledMessage, PROTOCOL_VERSION_STRING, is_version_compatible,
    VideoEncodeRequest, LlmGenerateRequest,
)
from .binary_protocol import BinaryFrameType, decode_binary_frame
from .queue_manager import QueueManager
from .llm_worker import LlmWorker
from .validation import ValidationError
from .worker import GPUWorker
from .video_worker import VideoWorker

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
    - Request queuing (audio + video)
    - Progress streaming
    - Result delivery
    - Binary frame dispatch for video data
    """

    def __init__(self, config: Config):
        """
        Initialize the server.

        Args:
            config: Server configuration
        """
        self.config = config

        # Audio queue and worker
        self.queue = QueueManager(
            max_size=config.queue.max_size,
            request_timeout=config.queue.request_timeout,
            queue_name="audio",
        )
        self.worker = GPUWorker(config, self.queue)

        # Video queue and worker (created if enabled)
        self.video_queue: Optional[QueueManager] = None
        self.video_worker: Optional[VideoWorker] = None
        if config.video_encoding.enabled:
            self.video_queue = QueueManager(
                max_size=config.video_queue.max_size,
                request_timeout=config.video_queue.request_timeout,
                queue_name="video",
            )
            self.video_worker = VideoWorker(config, self.video_queue)

        # LLM queue + worker + GPU arbiter (created if enabled). The arbiter is
        # the single VRAM admission point; today only the LLM worker is routed
        # through it (audio/video fold in later). Constructing TorchMemoryProbe
        # is lazy — the first (CUDA-context-creating) probe happens on the first
        # acquire, off the constructor.
        self.arbiter = None
        self.llm_queue: Optional[QueueManager] = None
        self.llm_worker: Optional[LlmWorker] = None
        if config.llm.enabled:
            from .gpu_info import TorchMemoryProbe
            from .orchestrator import VramArbiter
            self.arbiter = VramArbiter(
                TorchMemoryProbe(), config.gpu.vram_headroom_bytes
            )
            self.llm_queue = QueueManager(
                max_size=config.llm_queue.max_size,
                request_timeout=config.llm_queue.request_timeout,
                queue_name="llm",
            )
            self.llm_worker = LlmWorker(config, self.llm_queue, self.arbiter)

        self._authenticated_clients: Set[WebSocketServerProtocol] = set()
        self._pending_connections: Set[WebSocketServerProtocol] = set()  # Pre-auth connections
        self._client_info: Dict[WebSocketServerProtocol, dict] = {}
        self._rate_limiters: Dict[WebSocketServerProtocol, RateLimitTracker] = {}
        self._message_limiters: Dict[WebSocketServerProtocol, MessageRateLimiter] = {}
        self._server = None
        self._worker_task = None
        self._video_worker_task = None
        self._llm_worker_task = None

    def _build_workloads(self) -> dict:
        """Advertise which workloads this server can perform, so a client can
        discover (e.g.) that remote LLM summarisation is available. Additive
        capability field in auth_ok; old clients ignore it."""
        return {
            "transcribe": True,
            "diarize": True,
            "encode": self.config.video_encoding.enabled,
            "llm": self.config.llm.enabled,
        }

    def _build_transfer_capabilities(self) -> dict:
        """Build transfer_capabilities dict for auth_ok response."""
        methods = ["websocket"]
        shared_paths = []
        if self.config.video_encoding.enabled:
            if self.config.video_encoding.shared_paths:
                methods.insert(0, "shared_fs")
                shared_paths = self.config.video_encoding.shared_paths
        return {
            "methods": methods,
            "shared_paths": shared_paths,
            "max_websocket_chunk": 512 * 1024,
            "max_input_size": self.config.video_encoding.max_input_size,
        }

    async def start(self):
        """Start the server."""
        logger.info(
            "Starting GPU Server",
            data={
                'host': self.config.server.host,
                'port': self.config.server.port,
                'max_connections': self.config.server.max_connections,
                'queue_max_size': self.config.queue.max_size,
                'video_encoding_enabled': self.config.video_encoding.enabled,
            }
        )

        # Create SSL context if TLS is enabled
        ssl_context = create_ssl_context(self.config.tls)

        # Start the audio worker
        self._worker_task = asyncio.create_task(self.worker.start())

        # Start the video worker if enabled
        if self.video_worker:
            self._video_worker_task = asyncio.create_task(self.video_worker.start())

        # Start the LLM worker if enabled
        if self.llm_worker:
            self._llm_worker_task = asyncio.create_task(self.llm_worker.start())

        # Start WebSocket server
        self._server = await websockets.serve(
            self._handle_connection,
            self.config.server.host,
            self.config.server.port,
            max_size=self.config.server.max_message_size,
            ping_interval=30,
            # ping_timeout=None: do not terminate a connection because a
            # pong was slow. Successful sends are themselves liveness
            # proof; pings are kept as a periodic diagnostic and to
            # service NAT/middlebox keepalive on long-haul VPN paths.
            # See video_worker chunk-send loop comments for context.
            ping_timeout=None,
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
                'video_encoding_enabled': self.config.video_encoding.enabled,
            }
        )

    async def stop(self):
        """Stop the server."""
        logger.info("Stopping GPU Server...", data={'connected_clients': len(self._authenticated_clients)})

        # Stop accepting new connections
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Stop the audio worker
        await self.worker.stop()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        # Stop the video worker
        if self.video_worker:
            await self.video_worker.stop()
        if self._video_worker_task:
            self._video_worker_task.cancel()
            try:
                await self._video_worker_task
            except asyncio.CancelledError:
                pass

        # Stop the LLM worker
        if self.llm_worker:
            await self.llm_worker.stop()
        if self._llm_worker_task:
            self._llm_worker_task.cancel()
            try:
                await self._llm_worker_task
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

            # Handle messages - support both text and binary frames
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._handle_binary_frame(websocket, message)
                else:
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
                "video_encoding_enabled": self.config.video_encoding.enabled,
                "transfer_capabilities": self._build_transfer_capabilities(),
                "workloads": self._build_workloads(),
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
                "video_encoding_enabled": self.config.video_encoding.enabled,
                "transfer_capabilities": self._build_transfer_capabilities(),
                "workloads": self._build_workloads(),
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
        Handle an incoming text message from an authenticated client.

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

        # Handle parsing errors (parse_message returns {"type": "error", "error": "..."} on failure)
        if msg_type == "error" and "error" in data:
            logger.warning(f"Message parsing error: {data.get('error')}")
            await websocket.send(ErrorMessage(
                request_id="",
                error=data.get("error", "Message parsing failed"),
                error_code="MESSAGE_PARSE_ERROR",
            ).to_json())
            return

        if msg_type == MessageType.PROCESS:
            await self._handle_process_request(websocket, data)

        elif msg_type == MessageType.VIDEO_ENCODE:
            await self._handle_video_encode_request(websocket, data)

        elif msg_type == MessageType.LLM_GENERATE:
            await self._handle_llm_request(websocket, data)

        elif msg_type == MessageType.CANCEL:
            await self._handle_cancel(websocket, data)

        elif msg_type == MessageType.PING:
            pong_data = {
                "type": MessageType.PONG,
                "queue_size": self.queue.size,
                "is_processing": self.worker.is_processing,
            }
            if self.video_queue:
                pong_data["video_queue_size"] = self.video_queue.size
            await websocket.send(json.dumps(pong_data))

        elif msg_type == "auth":
            # Client sent auth message but already authenticated - ignore silently
            # This happens when auth is disabled (auto-auth on connect) but client
            # still sends auth message
            logger.debug("Ignoring auth message from already-authenticated client")

        else:
            logger.warning(f"Unknown message type: {msg_type}")
            await websocket.send(ErrorMessage(
                request_id=data.get("request_id", ""),
                error=f"Unknown message type: {msg_type}",
                error_code="UNKNOWN_MESSAGE_TYPE",
            ).to_json())

    async def _handle_llm_request(self, websocket: WebSocketServerProtocol, data: dict):
        """Validate and queue an LLM_GENERATE request for the LLM worker."""
        request_id = data.get("request_id", "unknown")
        if not self.llm_worker or not self.llm_queue:
            await websocket.send(ErrorMessage(
                request_id=request_id,
                error="LLM workload is not enabled on this server",
                recoverable=False,
                error_code="LLM_NOT_ENABLED",
            ).to_json())
            return
        try:
            request = LlmGenerateRequest.from_dict(data)
        except ValidationError as e:
            await websocket.send(ErrorMessage(
                request_id=request_id,
                error=f"Invalid LLM request: {e}",
                recoverable=False,
                error_code="VALIDATION_ERROR",
            ).to_json())
            return

        # Per-request rate limit — LLM generations are long (minutes); without
        # this a client could fill the queue behind the global flood limiter.
        rate_limiter = self._rate_limiters.get(websocket)
        if rate_limiter:
            if (rate_limiter.get_request_count(self.config.server.rate_limit_window)
                    >= self.config.server.rate_limit_requests):
                await websocket.send(ErrorMessage(
                    request_id=request.request_id,
                    error=f"Rate limit exceeded. Max {self.config.server.rate_limit_requests} "
                          f"requests per {self.config.server.rate_limit_window} seconds.",
                    recoverable=True,
                    error_code="RATE_LIMIT_EXCEEDED",
                ).to_json())
                return
            rate_limiter.record_request()

        success = await self.llm_queue.enqueue(request, websocket)
        if success:
            await websocket.send(json.dumps({
                "type": MessageType.QUEUED,
                "request_id": request.request_id,
                "queue_size": self.llm_queue.size,
            }))
        else:
            await websocket.send(ErrorMessage(
                request_id=request.request_id,
                error="LLM queue is full",
                recoverable=True,
                error_code="QUEUE_FULL",
            ).to_json())

    async def _handle_binary_frame(self, websocket: WebSocketServerProtocol, data: bytes):
        """
        Handle an incoming binary WebSocket frame.

        Binary frames bypass the message rate limiter since video data
        can arrive as many frames.
        """
        try:
            header, payload = decode_binary_frame(data)
        except ValueError as e:
            logger.warning(f"Invalid binary frame: {e}")
            return

        if header.frame_type == BinaryFrameType.VIDEO_INPUT:
            await self._handle_video_data(websocket, header.request_id, payload)
        else:
            logger.warning(f"Unexpected binary frame type: {header.frame_type}")

    async def _handle_video_data(self, websocket: WebSocketServerProtocol, request_id: str, payload: bytes):
        """Handle incoming video data chunk."""
        if not self.config.video_encoding.enabled or not self.video_worker:
            await websocket.send(ErrorMessage(
                request_id=request_id,
                error="Video encoding is not enabled on this server",
                error_code="VIDEO_ENCODING_DISABLED",
            ).to_json())
            return

        try:
            if len(payload) == 0:
                # Empty payload = final frame, finalize upload
                await self.video_worker.finalize_upload(request_id)
            else:
                self.video_worker.receive_video_data(request_id, payload)
        except KeyError:
            logger.warning(f"Video data for unknown request: {request_id}")
            await websocket.send(ErrorMessage(
                request_id=request_id,
                error="No pending video upload for this request",
                error_code="NO_PENDING_UPLOAD",
            ).to_json())
        except ValueError as e:
            logger.warning(f"Video data error: {e}")
            await websocket.send(ErrorMessage(
                request_id=request_id,
                error=str(e),
                recoverable=False,
                error_code="VIDEO_DATA_ERROR",
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

    async def _handle_video_encode_request(self, websocket: WebSocketServerProtocol, data: dict):
        """Handle a video encoding request."""
        request_id = data.get("request_id", "unknown")

        # Check if video encoding is enabled
        if not self.config.video_encoding.enabled or not self.video_worker or not self.video_queue:
            await websocket.send(ErrorMessage(
                request_id=request_id,
                error="Video encoding is not enabled on this server",
                error_code="VIDEO_ENCODING_DISABLED",
            ).to_json())
            return

        try:
            request = VideoEncodeRequest.from_dict(data)
            request_id = request.request_id

            set_request_context(request_id=request_id)

            # Validate input size against config limit
            if request.input_size > self.config.video_encoding.max_input_size:
                await websocket.send(ErrorMessage(
                    request_id=request_id,
                    error=f"Input size exceeds maximum "
                          f"({request.input_size / 1024 / 1024:.0f} MB > "
                          f"{self.config.video_encoding.max_input_size / 1024 / 1024:.0f} MB)",
                    recoverable=False,
                    error_code="INPUT_TOO_LARGE",
                ).to_json())
                return

            # Check rate limit (shared with audio)
            rate_limiter = self._rate_limiters.get(websocket)
            if rate_limiter:
                request_count = rate_limiter.get_request_count(self.config.server.rate_limit_window)
                if request_count >= self.config.server.rate_limit_requests:
                    await websocket.send(ErrorMessage(
                        request_id=request_id,
                        error="Rate limit exceeded",
                        recoverable=True,
                        error_code="RATE_LIMIT_EXCEEDED",
                    ).to_json())
                    return
                rate_limiter.record_request()

            logger.info(
                "Video encode request received",
                data={
                    'filename': request.filename,
                    'input_size': request.input_size,
                    'priority': request.priority,
                    'transfer_method': request.transfer_method,
                }
            )

            if request.transfer_method == "shared_fs":
                # Shared filesystem: validate paths are under allowed shared_paths
                for path_label, path_value in [("input_path", request.input_path), ("output_path", request.output_path)]:
                    if not self._validate_shared_path(path_value, request_id, websocket):
                        await websocket.send(ErrorMessage(
                            request_id=request_id,
                            error=f"Path not under any configured shared_paths: {path_value}",
                            recoverable=False,
                            error_code="SHARED_FS_PATH_NOT_ALLOWED",
                        ).to_json())
                        return

                # Verify input file exists and is readable
                input_path = Path(request.input_path)
                if not input_path.exists():
                    await websocket.send(ErrorMessage(
                        request_id=request_id,
                        error=f"Shared filesystem input file not found: {request.input_path}",
                        recoverable=False,
                        error_code="SHARED_FS_INPUT_NOT_FOUND",
                    ).to_json())
                    return

                # Enqueue immediately as ready (no binary upload needed)
                success = await self.video_queue.enqueue(
                    request, websocket, pending_data=False,
                )

                if success:
                    await websocket.send(json.dumps({
                        "type": MessageType.QUEUED,
                        "request_id": request_id,
                        "position": self.video_queue.size,
                    }))
                else:
                    logger.warning("Video queue full")
                    await websocket.send(ErrorMessage(
                        request_id=request_id,
                        error="Video queue is full, please try again later",
                        recoverable=True,
                        error_code="QUEUE_FULL",
                    ).to_json())
            elif request.resume_offset > 0:
                # WebSocket resume: client reconnected after a drop
                temp_file = self.video_worker._pending_uploads.get(request_id)
                if temp_file is None:
                    logger.warning(
                        f"Resume failed: no pending upload for {request_id} "
                        f"(may have timed out)"
                    )
                    await websocket.send(ErrorMessage(
                        request_id=request_id,
                        error="No pending upload found for this request. "
                              "The partial upload may have expired.",
                        recoverable=False,
                        error_code="RESUME_NOT_FOUND",
                    ).to_json())
                    return

                server_offset = temp_file.current_size
                if server_offset != request.resume_offset:
                    logger.warning(
                        f"Resume offset mismatch for {request_id}: "
                        f"client={request.resume_offset}, server={server_offset}"
                    )
                    await websocket.send(ErrorMessage(
                        request_id=request_id,
                        error=f"Resume offset mismatch: server has {server_offset} bytes, "
                              f"client claims {request.resume_offset}",
                        recoverable=False,
                        error_code="RESUME_OFFSET_MISMATCH",
                    ).to_json())
                    return

                # Update the queue entry's websocket to the new connection
                await self.video_queue.update_websocket(request_id, websocket)

                logger.info(
                    f"Resuming upload for {request_id} at offset "
                    f"{server_offset / 1024 / 1024:.1f}MB"
                )

                await websocket.send(json.dumps({
                    "type": MessageType.QUEUED,
                    "request_id": request_id,
                    "position": self.video_queue.size,
                    "resume_offset": server_offset,
                }))

                await websocket.send(ProgressMessage(
                    request_id=request_id,
                    stage=ProcessingStage.RECEIVING_VIDEO,
                    percent=int(server_offset / request.input_size * 100) if request.input_size > 0 else 0,
                    message=f"Resuming upload from {server_offset / 1024 / 1024:.1f}MB",
                ).to_json())

            else:
                # WebSocket chunked transfer (fresh upload)
                # Register upload (creates temp file)
                self.video_worker.register_upload(request_id)

                # Enqueue placeholder with pending_data=True
                success = await self.video_queue.enqueue(
                    request, websocket, pending_data=True,
                )

                if success:
                    await websocket.send(json.dumps({
                        "type": MessageType.QUEUED,
                        "request_id": request_id,
                        "position": self.video_queue.size,
                    }))

                    # Send progress: waiting for data
                    await websocket.send(ProgressMessage(
                        request_id=request_id,
                        stage=ProcessingStage.RECEIVING_VIDEO,
                        percent=0,
                        message="Ready to receive video data",
                    ).to_json())
                else:
                    logger.warning("Video queue full")
                    self.video_worker._cleanup_upload(request_id)
                    await websocket.send(ErrorMessage(
                        request_id=request_id,
                        error="Video queue is full, please try again later",
                        recoverable=True,
                        error_code="QUEUE_FULL",
                    ).to_json())

        except ValidationError as e:
            await websocket.send(ErrorMessage(
                request_id=request_id,
                error=str(e),
                recoverable=False,
                error_code="VALIDATION_ERROR",
            ).to_json())
        except Exception as e:
            logger.error(f"Video encode request error: {e}", exc_info=True)
            await websocket.send(ErrorMessage(
                request_id=request_id,
                error="Internal server error",
                error_code="INTERNAL_ERROR",
            ).to_json())

    def _validate_shared_path(self, path_str: str, request_id: str, websocket) -> bool:
        """Validate that a path is under one of the configured shared_paths.

        Security: prevents path traversal attacks by resolving the path
        and checking it's genuinely under an allowed shared mount.

        Returns True if valid. Sends error and returns False if invalid.
        This is a synchronous check — caller must await the websocket.send
        if this returns False (error is sent inline via _validate_shared_path_async).
        """
        resolved = Path(path_str).resolve()
        for shared_path_str in self.config.video_encoding.shared_paths:
            shared_path = Path(shared_path_str).resolve()
            if resolved.is_relative_to(shared_path):
                return True

        logger.warning(
            f"Shared path validation failed: {path_str} is not under any "
            f"configured shared_paths: {self.config.video_encoding.shared_paths}"
        )
        # We can't await here since this is sync — caller handles async send
        return False

    async def _handle_cancel(self, websocket: WebSocketServerProtocol, data: dict):
        """Handle a cancel request."""
        request_id = data.get("request_id", "")
        if not request_id:
            return

        set_request_context(request_id=request_id)

        # Try audio queue first, then video queue
        success = await self.queue.cancel(request_id)
        if not success and self.video_queue:
            success = await self.video_queue.cancel(request_id)
            if success and self.video_worker:
                self.video_worker._cleanup_upload(request_id)

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
        stats = {
            "connected_clients": self.connected_clients,
            "queue": self.queue.get_stats(),
            "worker": {
                "is_processing": self.worker.is_processing,
                "current_request": self.worker.current_request_id,
            },
        }
        if self.video_queue and self.video_worker:
            stats["video_queue"] = self.video_queue.get_stats()
            stats["video_worker"] = {
                "is_processing": self.video_worker.is_processing,
                "current_request": self.video_worker.current_request_id,
                "pending_uploads": self.video_worker.pending_upload_count,
            }
        return stats
