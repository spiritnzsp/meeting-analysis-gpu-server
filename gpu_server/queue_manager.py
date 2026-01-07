"""
Request Queue Manager

Manages the queue of processing requests with priority support.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Callable, Awaitable
from heapq import heappush, heappop

from .protocol import ProcessRequest, ProgressMessage, ProcessingStage

logger = logging.getLogger(__name__)


@dataclass(order=True)
class QueuedRequest:
    """A request in the queue with priority ordering."""
    # Priority tuple: (negative priority, timestamp) - lower = processed first
    sort_key: tuple = field(compare=True)
    request: ProcessRequest = field(compare=False)
    websocket: object = field(compare=False)  # WebSocket connection
    queued_at: datetime = field(compare=False, default_factory=datetime.now)
    cancelled: bool = field(compare=False, default=False)


class QueueManager:
    """
    Manages the processing request queue.

    Features:
    - Priority-based ordering (higher priority processed first)
    - FIFO within same priority
    - Request cancellation
    - Queue size limits
    - Progress callback support
    """

    def __init__(self, max_size: int = 100, request_timeout: int = 3600):
        """
        Initialize the queue manager.

        Args:
            max_size: Maximum number of queued requests
            request_timeout: Request timeout in seconds
        """
        self.max_size = max_size
        self.request_timeout = request_timeout

        self._queue: list = []  # Heap queue
        self._requests: Dict[str, QueuedRequest] = {}  # request_id -> QueuedRequest
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        self._processing = False

    @property
    def size(self) -> int:
        """Current queue size."""
        return len(self._queue)

    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self.size >= self.max_size

    async def enqueue(
        self,
        request: ProcessRequest,
        websocket,
        progress_callback: Optional[Callable[[ProgressMessage], Awaitable[None]]] = None
    ) -> bool:
        """
        Add a request to the queue.

        Args:
            request: The processing request
            websocket: WebSocket connection for sending results
            progress_callback: Optional callback for progress updates

        Returns:
            True if enqueued, False if queue is full
        """
        async with self._lock:
            if self.is_full:
                logger.warning(f"Queue full, rejecting request {request.request_id}")
                return False

            # Create priority tuple: (-priority, timestamp) so higher priority comes first
            sort_key = (-request.priority, datetime.now().timestamp())

            queued = QueuedRequest(
                sort_key=sort_key,
                request=request,
                websocket=websocket,
            )

            heappush(self._queue, queued)
            self._requests[request.request_id] = queued

            logger.info(
                f"Enqueued request {request.request_id} "
                f"(priority={request.priority}, queue_size={self.size})"
            )

            # Signal that queue is not empty
            self._not_empty.set()

            # Send progress update
            if progress_callback:
                await progress_callback(ProgressMessage(
                    request_id=request.request_id,
                    stage=ProcessingStage.QUEUED,
                    percent=0,
                    message=f"Queued (position {self.size})"
                ))

            return True

    async def dequeue(self) -> Optional[QueuedRequest]:
        """
        Get the next request from the queue.

        Returns:
            Next QueuedRequest or None if queue is empty
        """
        async with self._lock:
            while self._queue:
                queued = heappop(self._queue)

                # Skip cancelled requests
                if queued.cancelled:
                    del self._requests[queued.request.request_id]
                    continue

                # Remove from lookup
                if queued.request.request_id in self._requests:
                    del self._requests[queued.request.request_id]

                # Clear not_empty if queue is now empty
                if not self._queue:
                    self._not_empty.clear()

                return queued

            self._not_empty.clear()
            return None

    async def wait_for_request(self) -> QueuedRequest:
        """
        Wait for and return the next request.

        Blocks until a request is available.
        """
        while True:
            await self._not_empty.wait()
            request = await self.dequeue()
            if request:
                return request

    async def cancel(self, request_id: str) -> bool:
        """
        Cancel a queued request.

        Args:
            request_id: ID of request to cancel

        Returns:
            True if cancelled, False if not found
        """
        async with self._lock:
            if request_id in self._requests:
                self._requests[request_id].cancelled = True
                logger.info(f"Cancelled request {request_id}")
                return True
            return False

    async def get_position(self, request_id: str) -> Optional[int]:
        """
        Get queue position for a request.

        Args:
            request_id: Request ID

        Returns:
            Position (1-based) or None if not found
        """
        async with self._lock:
            if request_id not in self._requests:
                return None

            target = self._requests[request_id]
            position = 1
            for queued in sorted(self._queue):
                if queued.cancelled:
                    continue
                if queued.request.request_id == request_id:
                    return position
                position += 1
            return None

    async def clear(self):
        """Clear all queued requests."""
        async with self._lock:
            self._queue.clear()
            self._requests.clear()
            self._not_empty.clear()
            logger.info("Queue cleared")

    def get_stats(self) -> Dict:
        """Get queue statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "is_full": self.is_full,
        }
