"""Tests for ResidentModel — VRAM residency lifecycle and eviction coordination.

Uses a REAL single-thread ThreadPoolExecutor so the load/unload marshalling
(the F1 safety property: torch teardown must run on the model's own executor,
never the event loop) is actually exercised, not mocked away.
"""
import threading

import pytest
from concurrent.futures import ThreadPoolExecutor

from gpu_server.orchestrator.resident_model import ResidentModel

GB = 1024 ** 3


@pytest.fixture
def executor():
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_resident")
    yield ex
    ex.shutdown(wait=True)


def _model(executor, load_fn=None, unload_fn=None, vram=5 * GB):
    return ResidentModel(
        key="m",
        estimated_vram_bytes=vram,
        executor=executor,
        load_fn=load_fn or (lambda: None),
        unload_fn=unload_fn or (lambda: None),
    )


async def test_load_runs_on_executor_thread_not_the_loop(executor):
    loop_thread = threading.get_ident()
    ran_on = {}

    def load_fn():
        ran_on["thread"] = threading.get_ident()

    m = _model(executor, load_fn=load_fn)
    assert not m.is_loaded()
    await m.load()
    assert m.is_loaded()
    # F1: the blocking teardown/build must NOT run on the event-loop thread.
    assert ran_on["thread"] != loop_thread


async def test_load_is_idempotent(executor):
    calls = []
    m = _model(executor, load_fn=lambda: calls.append(1))
    await m.load()
    await m.load()
    assert calls == [1]


async def test_unload_runs_unload_fn_and_clears_loaded(executor):
    unloaded = []
    m = _model(executor, unload_fn=lambda: unloaded.append(1))
    await m.load()
    assert m.try_reserve_for_eviction()
    await m.unload()
    assert unloaded == [1]
    assert not m.is_loaded()


async def test_reserve_for_eviction_requires_loaded_idle_and_unreserved(executor):
    m = _model(executor)
    # not loaded -> cannot reserve
    assert not m.try_reserve_for_eviction()
    await m.load()
    # loaded + idle -> reserve succeeds
    assert m.try_reserve_for_eviction()
    # already reserved -> second reserve fails
    assert not m.try_reserve_for_eviction()
    m.cancel_eviction_reservation()
    assert m.try_reserve_for_eviction()


async def test_busy_model_cannot_be_reserved_for_eviction(executor):
    # F3: an in-flight operation blocks eviction so teardown can't race inference.
    m = _model(executor)
    await m.load()
    assert m.begin_operation() is True
    assert m.is_busy
    assert not m.try_reserve_for_eviction()
    m.end_operation()
    assert m.try_reserve_for_eviction()


async def test_begin_operation_refuses_while_evicting(executor):
    # F3: once reserved for eviction, no new operation may start on the model.
    m = _model(executor)
    await m.load()
    assert m.try_reserve_for_eviction()
    assert m.begin_operation() is False


async def test_begin_operation_refuses_when_not_loaded(executor):
    m = _model(executor)
    assert m.begin_operation() is False


async def test_load_failure_leaves_model_unloaded(executor):
    def boom():
        raise RuntimeError("CUDA out of memory")

    m = _model(executor, load_fn=boom)
    with pytest.raises(RuntimeError, match="out of memory"):
        await m.load()
    assert not m.is_loaded()


async def test_rebind_resets_residency_to_unloaded(executor):
    # F5: after a stuck-timeout recreation, the same stable-keyed resident is
    # rebound to a fresh executor/model and starts unloaded.
    m = _model(executor)
    await m.load()
    assert m.is_loaded()

    new_ex = ThreadPoolExecutor(max_workers=1)
    try:
        loaded_again = []
        m.rebind(new_ex, load_fn=lambda: loaded_again.append(1), unload_fn=lambda: None)
        assert not m.is_loaded()
        assert m.key == "m"  # identity preserved
        await m.load()
        assert loaded_again == [1]
    finally:
        new_ex.shutdown(wait=True)
