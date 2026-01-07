"""
WebSocket Server

Main server that handles client connections and routes requests.
"""
import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List

import websockets
from websockets.server import WebSocketServerProtocol

from .config import Config, create_ssl_context, verify_api_key


@dataclass
class RateLimitTracker:
    """Tracks rate limit state for a client."""
    request_times: List[float] = field(default_factory=list)

    def record_request(self):
        """Record a request timestamp."""
        self.request_times.append(time.time())

    def get_request_count(self, window_seconds: float) -> int:
        """Get number of requests in the time window."""
        cutoff = time.time() - window_seconds
        # Prune old entries
        self.request_times = [t for t in self.request_times if t > cutoff]
        return len(self.request_times)


from .queue_manager import QueueManager
from .worker import GPUWorker
from .protocol import (
    MessageType, parse_message, ProcessRequest, AuthMessage,
    ProgressMessage, ProcessingStage, ErrorMessage,
    PROTOCOL_VERSION_STRING, is_version_compatible
)

logger = logging.getLogger(__name__)


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
        self._client_info: Dict[WebSocketServerProtocol, dict] = {}
        self._rate_limiters: Dict[WebSocketServerProtocol, RateLimitTracker] = {}
        self._server = None
        self._worker_task = None

    async def start(self):
        """Start the server."""
        logger.info(f"Starting GPU Server on {self.config.server.host}:{self.config.server.port}")

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
        logger.info(f"Max message size: {max_mb:.0f} MB")

        protocol = "wss" if ssl_context else "ws"
        logger.info(f"GPU Server listening on {protocol}://{self.config.server.host}:{self.config.server.port}")

    async def stop(self):
        """Stop the server."""
        logger.info("Stopping GPU Server...")

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

        # Close all client connections
        for ws in list(self._authenticated_clients):
            await ws.close()

        logger.info("GPU Server stopped")

    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str = ""):
        """
        Handle a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            path: Request path (unused)
        """
        client_addr = websocket.remote_address
        logger.info(f"New connection from {client_addr}")

        # Enforce connection limit
        max_conn = self.config.server.max_connections
        if len(self._authenticated_clients) >= max_conn:
            logger.warning(f"Connection limit reached ({max_conn}), rejecting {client_addr}")
            await websocket.send(json.dumps({
                "type": MessageType.AUTH_FAILED,
                "error": "Server connection limit reached, try again later"
            }))
            await websocket.close(1013, "Server overloaded")  # 1013 = Try again later
            return

        try:
            # Wait for authentication
            authenticated = await self._authenticate(websocket)
            if not authenticated:
                return

            self._authenticated_clients.add(websocket)
            self._rate_limiters[websocket] = RateLimitTracker()

            # Handle messages
            async for message in websocket:
                await self._handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed: {client_addr} ({e.code})")
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
        finally:
            self._authenticated_clients.discard(websocket)
            self._client_info.pop(websocket, None)
            self._rate_limiters.pop(websocket, None)
            logger.info(f"Client disconnected: {client_addr}")

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
            await websocket.send(json.dumps({"type": MessageType.AUTH_OK}))
            return True

        try:
            # Wait for auth message (with timeout)
            message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
            data = parse_message(message)

            if data.get("type") != MessageType.AUTH:
                await websocket.send(json.dumps({
                    "type": MessageType.AUTH_FAILED,
                    "error": "Expected auth message"
                }))
                return False

            auth_msg = AuthMessage.from_dict(data)

            # Check protocol version compatibility
            compatible, version_error = is_version_compatible(auth_msg.protocol_version)
            if not compatible:
                logger.warning(f"Protocol version incompatible from {websocket.remote_address}: {version_error}")
                await websocket.send(json.dumps({
                    "type": MessageType.AUTH_FAILED,
                    "error": version_error,
                    "server_protocol_version": PROTOCOL_VERSION_STRING,
                }))
                return False

            # Check API key using constant-time comparison
            if not verify_api_key(auth_msg.api_key, self.config.auth.api_key_hashes):
                logger.warning(f"Authentication failed: invalid API key from {websocket.remote_address}")
                await websocket.send(json.dumps({
                    "type": MessageType.AUTH_FAILED,
                    "error": "Invalid API key"
                }))
                return False

            # Store client info
            self._client_info[websocket] = {
                "client_version": auth_msg.client_version,
                "protocol_version": auth_msg.protocol_version,
                "authenticated_at": asyncio.get_event_loop().time(),
            }

            await websocket.send(json.dumps({
                "type": MessageType.AUTH_OK,
                "server_version": "0.1.0",
                "protocol_version": PROTOCOL_VERSION_STRING,
                "queue_size": self.queue.size,
            }))

            logger.info(f"Client authenticated: {websocket.remote_address} (version: {auth_msg.client_version})")
            return True

        except asyncio.TimeoutError:
            logger.warning(f"Authentication timeout: {websocket.remote_address}")
            await websocket.send(json.dumps({
                "type": MessageType.AUTH_FAILED,
                "error": "Authentication timeout"
            }))
            return False
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    async def _handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """
        Handle an incoming message from an authenticated client.

        Args:
            websocket: The WebSocket connection
            message: The raw message string
        """
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
            ).to_json())

    async def _handle_process_request(self, websocket: WebSocketServerProtocol, data: dict):
        """Handle a processing request."""
        try:
            request = ProcessRequest.from_dict(data)

            # Check rate limit
            rate_limiter = self._rate_limiters.get(websocket)
            if rate_limiter:
                request_count = rate_limiter.get_request_count(self.config.server.rate_limit_window)
                if request_count >= self.config.server.rate_limit_requests:
                    logger.warning(
                        f"Rate limit exceeded for {websocket.remote_address}: "
                        f"{request_count} requests in {self.config.server.rate_limit_window}s"
                    )
                    await websocket.send(ErrorMessage(
                        request_id=request.request_id,
                        error=f"Rate limit exceeded. Max {self.config.server.rate_limit_requests} "
                              f"requests per {self.config.server.rate_limit_window} seconds.",
                        recoverable=True,
                    ).to_json())
                    return
                rate_limiter.record_request()

            logger.info(
                f"Received process request: {request.request_id} "
                f"({len(request.audio_data)} bytes, {request.meeting_name})"
            )

            # Check queue capacity
            if self.queue.is_full:
                await websocket.send(ErrorMessage(
                    request_id=request.request_id,
                    error="Queue is full, please try again later",
                    recoverable=True,
                ).to_json())
                return

            # Create progress callback for queue position updates
            async def on_progress(msg: ProgressMessage):
                await websocket.send(msg.to_json())

            # Enqueue the request
            success = await self.queue.enqueue(request, websocket, on_progress)

            if success:
                await websocket.send(json.dumps({
                    "type": MessageType.QUEUED,
                    "request_id": request.request_id,
                    "position": self.queue.size,
                }))
            else:
                await websocket.send(ErrorMessage(
                    request_id=request.request_id,
                    error="Failed to enqueue request",
                ).to_json())

        except Exception as e:
            logger.error(f"Error handling process request: {e}", exc_info=True)
            await websocket.send(ErrorMessage(
                request_id=data.get("request_id", ""),
                error=str(e),
            ).to_json())

    async def _handle_cancel(self, websocket: WebSocketServerProtocol, data: dict):
        """Handle a cancel request."""
        request_id = data.get("request_id", "")
        if not request_id:
            return

        success = await self.queue.cancel(request_id)

        if success:
            await websocket.send(json.dumps({
                "type": "cancelled",
                "request_id": request_id,
            }))
            logger.info(f"Request cancelled: {request_id}")
        else:
            await websocket.send(ErrorMessage(
                request_id=request_id,
                error="Request not found or already processing",
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
