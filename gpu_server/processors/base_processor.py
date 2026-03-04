"""
Base Processor

Abstract base class for ThreadPoolExecutor-based GPU processors.
Extracts common lifecycle code shared by WhisperProcessor and PyAnnoteProcessor.
"""
import asyncio
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

# Import from package __init__.py
from . import ProcessorCancelled


class BaseProcessor(ABC):
    """
    Base for ThreadPoolExecutor-based GPU processors.

    Provides:
    - Cancellation tracking with request_id to avoid TOCTOU races
    - Operation tracking for safe timeout handling
    - Dedicated thread pool for GPU operations
    - Graceful shutdown with timeout
    """

    def __init__(self, thread_name_prefix: str):
        self._shutdown = False

        # Cancellation tracking - use request_id to avoid TOCTOU race
        self._cancelled_request_id: Optional[str] = None
        self._cancel_lock = threading.Lock()

        # Operation tracking for safe timeout handling
        self._operation_complete = asyncio.Event()
        self._operation_complete.set()  # Initially not processing
        self._is_processing = False

        # Dedicated thread pool for GPU operations to avoid blocking event loop
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name_prefix)

    @property
    @abstractmethod
    def processor_name(self) -> str:
        """Human-readable processor name for logging."""
        ...

    @abstractmethod
    def _unload_resources(self) -> None:
        """Unload models/resources to free GPU memory."""
        ...

    def _check_cancelled(self, request_id: str) -> None:
        """Check if processing was cancelled and raise if so."""
        with self._cancel_lock:
            if self._cancelled_request_id == request_id or self._cancelled_request_id == "__ANY__":
                raise ProcessorCancelled(f"{self.processor_name} processing cancelled")

    def cancel(self, request_id: str = "") -> None:
        """
        Cancel processing for a specific request.

        Args:
            request_id: The request ID to cancel. If empty, cancels any in-progress request.
        """
        with self._cancel_lock:
            if request_id:
                self._cancelled_request_id = request_id
            elif self._is_processing:
                self._cancelled_request_id = "__ANY__"
        logger.info(f"{self.processor_name} cancellation requested for {request_id or 'current request'}")

    async def wait_for_idle(self, timeout: float = 30.0) -> bool:
        """
        Wait for the processor to become idle.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if processor is idle, False if timeout occurred
        """
        if not self._is_processing:
            return True

        try:
            await asyncio.wait_for(self._operation_complete.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for {self.processor_name} to become idle")
            return False

    @property
    def is_processing(self) -> bool:
        """Check if processor is currently running a GPU operation."""
        return self._is_processing

    def _begin_operation(self) -> None:
        """Mark the start of a processing operation. Clear stale cancellation."""
        self._operation_complete.clear()
        self._is_processing = True
        with self._cancel_lock:
            self._cancelled_request_id = None

    def _end_operation(self) -> None:
        """Mark the end of a processing operation."""
        self._is_processing = False
        self._operation_complete.set()

    def _check_shutdown(self) -> None:
        """Raise RuntimeError if processor has been shut down."""
        if self._shutdown:
            raise RuntimeError(f"{self.processor_name} has been shut down")

    def shutdown(self, timeout: float = 30.0) -> None:
        """
        Shutdown the processor and release all resources. Safe to call multiple times.

        Args:
            timeout: Maximum time to wait for executor shutdown (default 30s)
        """
        if self._shutdown:
            return

        self._shutdown = True

        # Signal cancellation to any running operations
        self.cancel()

        self._unload_resources()

        # Shutdown the executor with timeout
        if self._executor is not None:
            logger.info(f"Shutting down {self.processor_name} executor...")

            shutdown_complete = threading.Event()

            def do_shutdown():
                try:
                    self._executor.shutdown(wait=True, cancel_futures=False)
                finally:
                    shutdown_complete.set()

            shutdown_thread = threading.Thread(target=do_shutdown, daemon=True)
            shutdown_thread.start()

            if shutdown_complete.wait(timeout=timeout):
                logger.info(f"{self.processor_name} executor shutdown complete")
            else:
                logger.warning(
                    f"{self.processor_name} executor shutdown timed out after {timeout}s, "
                    "forcing shutdown (operations may still be running)"
                )
                self._executor.shutdown(wait=False, cancel_futures=True)

            self._executor = None
