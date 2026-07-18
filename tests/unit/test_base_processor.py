"""Tests for BaseProcessor abstract base class."""
import asyncio
import threading
import pytest
from unittest.mock import MagicMock

from gpu_server.processors.base_processor import BaseProcessor
from gpu_server.processors import ProcessorCancelled


class ConcreteProcessor(BaseProcessor):
    """Concrete implementation for testing."""

    def __init__(self):
        super().__init__(thread_name_prefix="test_gpu")
        self.unloaded = False

    @property
    def processor_name(self) -> str:
        return "TestProcessor"

    def _unload_resources(self) -> None:
        self.unloaded = True


class TestBaseProcessorLifecycle:
    """Test lifecycle management."""

    def test_initial_state(self):
        proc = ConcreteProcessor()
        assert not proc.is_processing
        assert not proc._shutdown

    def test_begin_end_operation(self):
        proc = ConcreteProcessor()
        proc._begin_operation()
        assert proc.is_processing
        assert proc._cancelled_request_id is None

        proc._end_operation()
        assert not proc.is_processing

    def test_check_shutdown_raises_when_shutdown(self):
        proc = ConcreteProcessor()
        proc._shutdown = True
        with pytest.raises(RuntimeError, match="TestProcessor has been shut down"):
            proc._check_shutdown()

    def test_check_shutdown_ok_when_running(self):
        proc = ConcreteProcessor()
        proc._check_shutdown()  # Should not raise

    def test_shutdown_calls_unload(self):
        proc = ConcreteProcessor()
        proc.shutdown()
        assert proc.unloaded
        assert proc._shutdown

    def test_shutdown_idempotent(self):
        proc = ConcreteProcessor()
        proc.shutdown()
        proc.shutdown()  # Should not raise
        assert proc._shutdown

    def test_shutdown_skips_unload_when_drain_times_out(self):
        # P1-1: if the executor drain times out, a GPU kernel may still be live,
        # so the model must be LEAKED (not freed) to avoid CUDA-context
        # corruption. _unload_resources must NOT run in that case.
        proc = ConcreteProcessor()
        started = threading.Event()
        release = threading.Event()

        def block():
            started.set()
            release.wait(5)

        proc._executor.submit(block)
        assert started.wait(1)
        try:
            proc.shutdown(timeout=0.2)  # cannot drain while `block` holds the thread
            assert proc._shutdown
            assert not proc.unloaded  # leaked, not freed
        finally:
            release.set()


class TestBaseProcessorCancellation:
    """Test cancellation tracking."""

    def test_check_cancelled_specific_request(self):
        proc = ConcreteProcessor()
        proc._cancelled_request_id = "req-123"
        with pytest.raises(ProcessorCancelled):
            proc._check_cancelled("req-123")

    def test_check_cancelled_wildcard(self):
        proc = ConcreteProcessor()
        proc._cancelled_request_id = "__ANY__"
        with pytest.raises(ProcessorCancelled):
            proc._check_cancelled("any-request")

    def test_check_cancelled_different_request_ok(self):
        proc = ConcreteProcessor()
        proc._cancelled_request_id = "req-123"
        proc._check_cancelled("req-456")  # Should not raise

    def test_cancel_with_request_id(self):
        proc = ConcreteProcessor()
        proc.cancel("req-123")
        assert proc._cancelled_request_id == "req-123"

    def test_cancel_without_request_id_when_processing(self):
        proc = ConcreteProcessor()
        proc._is_processing = True
        proc.cancel()
        assert proc._cancelled_request_id == "__ANY__"

    def test_cancel_without_request_id_when_idle(self):
        proc = ConcreteProcessor()
        proc.cancel()
        assert proc._cancelled_request_id is None

    def test_begin_operation_clears_stale_cancellation(self):
        proc = ConcreteProcessor()
        proc._cancelled_request_id = "old-req"
        proc._begin_operation()
        assert proc._cancelled_request_id is None


class TestBaseProcessorWaitForIdle:
    """Test wait_for_idle."""

    @pytest.mark.asyncio
    async def test_wait_when_idle(self):
        proc = ConcreteProcessor()
        result = await proc.wait_for_idle(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_when_processing_then_completes(self):
        proc = ConcreteProcessor()
        proc._begin_operation()

        async def complete_later():
            await asyncio.sleep(0.1)
            proc._end_operation()

        asyncio.create_task(complete_later())
        result = await proc.wait_for_idle(timeout=2.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_timeout(self):
        proc = ConcreteProcessor()
        proc._begin_operation()
        result = await proc.wait_for_idle(timeout=0.1)
        assert result is False
        proc._end_operation()
