"""
Request Queue Manager

Manages the queue of processing requests with priority support.
Generic enough to handle both audio processing and video encoding requests.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, Awaitable, Protocol, runtime_checkable
from heapq import heappush, heappop

from .protocol import ProgressMessage, ProcessingStage

logger = logging.getLogger(__name__)


@runtime_checkable
class Queueable(Protocol):
    """Protocol for objects that can be enqueued."""
    @property
    def request_id(self) -> str: ...
    @property
    def priority(self) -> int: ...


@dataclass(order=True)
class QueuedRequest:
    """A request in the queue with priority ordering."""
    # Priority tuple: (negative priority, timestamp) - lower = processed first
    sort_key: tuple = field(compare=True)
    request: Any = field(compare=False)  # Queueable (ProcessRequest or VideoEncodeRequest)
    websocket: object = field(compare=False)  # WebSocket connection
    queued_at: float = field(compare=False, default_factory=time.monotonic)  # Monotonic time for reliable timeouts
    cancelled: bool = field(compare=False, default=False)
    timeout_expired: bool = field(compare=False, default=False)  # True if request timed out in queue
    pending_data: bool = field(compare=False, default=False)  # True if waiting for data upload (video)


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

    def __init__(self, max_size: int = 100, request_timeout: int = 3600, queue_name: str = "audio"):
        """
        Initialize the queue manager.

        Args:
            max_size: Maximum number of queued requests
            request_timeout: Request timeout in seconds
            queue_name: Name for logging (e.g. "audio", "video")
        """
        self.max_size = max_size
        self.request_timeout = request_timeout
        self.queue_name = queue_name

        self._queue: list = []  # Heap queue
        self._requests: Dict[str, QueuedRequest] = {}  # request_id -> QueuedRequest
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        self._processing = False

    @property
    def size(self) -> int:
        """Current queue size (approximate, for monitoring only)."""
        # Note: This is not thread-safe but is acceptable for monitoring/stats
        return len(self._queue)

    def _size_locked(self) -> int:
        """Get size when lock is already held."""
        return len(self._queue)

    def _is_full_locked(self) -> bool:
        """Check if queue is full when lock is already held."""
        return self._size_locked() >= self.max_size

    @property
    def is_full(self) -> bool:
        """Check if queue is full (approximate, for monitoring only)."""
        # Note: This is not thread-safe but is acceptable for monitoring/stats
        # For actual enqueue decisions, use _is_full_locked() inside the lock
        return self.size >= self.max_size

    async def enqueue(
        self,
        request: Any,
        websocket,
        progress_callback: Optional[Callable[[ProgressMessage], Awaitable[None]]] = None,
        pending_data: bool = False,
    ) -> bool:
        """
        Add a request to the queue.

        Args:
            request: The processing request (must have request_id and priority attributes)
            websocket: WebSocket connection for sending results
            progress_callback: Optional callback for progress updates
            pending_data: If True, marks as waiting for data upload (video)

        Returns:
            True if enqueued, False if queue is full
        """
        async with self._lock:
            if self._is_full_locked():
                logger.warning(f"[{self.queue_name}] Queue full, rejecting request {request.request_id}")
                return False

            # Create priority tuple: (-priority, timestamp) so higher priority comes first
            # Use monotonic time for consistent ordering regardless of clock adjustments
            sort_key = (-request.priority, time.monotonic())

            queued = QueuedRequest(
                sort_key=sort_key,
                request=request,
                websocket=websocket,
                pending_data=pending_data,
            )

            heappush(self._queue, queued)
            self._requests[request.request_id] = queued

            queue_size = self._size_locked()
            logger.info(
                f"[{self.queue_name}] Enqueued request {request.request_id} "
                f"(priority={request.priority}, queue_size={queue_size}, pending_data={pending_data})"
            )

            # Signal that queue is not empty
            self._not_empty.set()

            # Send progress update
            if progress_callback and not pending_data:
                await progress_callback(ProgressMessage(
                    request_id=request.request_id,
                    stage=ProcessingStage.QUEUED,
                    percent=0,
                    message=f"Queued (position {queue_size})"
                ))

            return True

    async def mark_data_ready(self, request_id: str) -> bool:
        """
        Mark a pending_data request as ready for processing.

        Args:
            request_id: The request to mark as ready

        Returns:
            True if marked, False if not found
        """
        async with self._lock:
            if request_id in self._requests:
                self._requests[request_id].pending_data = False
                # Re-signal in case worker is waiting
                self._not_empty.set()
                logger.info(f"[{self.queue_name}] Data ready for request {request_id}")
                return True
            return False

    async def dequeue(self) -> Optional[QueuedRequest]:
        """
        Get the next request from the queue.

        Returns:
            Next QueuedRequest or None if queue is empty
        """
        async with self._lock:
            while self._queue:
                queued = heappop(self._queue)

                # Skip cancelled requests (already removed from _requests in cancel())
                if queued.cancelled:
                    # Clean up from _requests if somehow still there (defensive)
                    self._requests.pop(queued.request.request_id, None)
                    continue

                # Skip pending_data requests - put them back
                if queued.pending_data:
                    heappush(self._queue, queued)
                    break

                # Check if request has expired while waiting in queue
                # Use monotonic time difference for reliable timeout regardless of clock adjustments
                wait_time = time.monotonic() - queued.queued_at
                if wait_time > self.request_timeout:
                    logger.warning(
                        f"[{self.queue_name}] Request {queued.request.request_id} expired after "
                        f"{wait_time:.1f}s in queue (timeout: {self.request_timeout}s)"
                    )
                    # Remove from _requests dict BEFORE returning. This is intentional:
                    # - The request is no longer "queued" - it's being handed off for timeout handling
                    # - Lookups (get_position, cancel) should not find it since it's already expired
                    # - The returned object carries all state needed for the worker to notify the client
                    self._requests.pop(queued.request.request_id, None)
                    # Mark as timed out so worker can send appropriate error
                    queued.cancelled = True
                    queued.timeout_expired = True
                    # Return it so worker can notify client of the timeout
                    if not self._queue:
                        self._not_empty.clear()
                    return queued

                # Remove from lookup
                self._requests.pop(queued.request.request_id, None)

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
                queued = self._requests[request_id]
                queued.cancelled = True
                # Remove from _requests dict immediately to free memory
                # The heap entry remains but will be skipped/cleaned in dequeue()
                del self._requests[request_id]
                logger.info(f"[{self.queue_name}] Cancelled request {request_id}")
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
            logger.info(f"[{self.queue_name}] Queue cleared")

    def get_stats(self) -> Dict:
        """Get queue statistics."""
        return {
            "name": self.queue_name,
            "size": self.size,
            "max_size": self.max_size,
            "is_full": self.is_full,
        }
