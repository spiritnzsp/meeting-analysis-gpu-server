"""Tests for VramArbiter — admission, eviction, protection, OOM-tolerance, leases.

Uses fake models (satisfying the ManagedModel protocol, no GPU) and a scripted
memory probe so admission decisions are deterministic. A model becomes evictable
only after its lease is released (end_operation), so tests use ``async with`` to
model a finished job vs. an in-use one.
"""
import asyncio
import gc

import pytest

from gpu_server.gpu_info import GpuSnapshot
from gpu_server.orchestrator.vram_arbiter import VramArbiter, WorkloadNeed

GB = 1024 ** 3


class FakeModel:
    """In-memory ManagedModel: tracks load/unload calls, simulates OOM (once or
    always), a non-OOM load error, and the refcounted busy/eviction state."""

    def __init__(self, key, vram_gb, oom_once=False, oom_always=False, load_error=None):
        self.key = key
        self.estimated_vram_bytes = int(vram_gb * GB)
        self._loaded = False
        self._op_count = 0
        self._evicting = False
        self._oom_once = oom_once
        self._oom_always = oom_always
        self._load_error = load_error
        self.load_calls = 0
        self.unload_calls = 0

    def is_loaded(self):
        return self._loaded

    @property
    def is_busy(self):
        return self._op_count > 0

    async def load(self):
        self.load_calls += 1
        if self._load_error is not None:
            raise self._load_error
        if self._oom_always:
            raise RuntimeError("CUDA out of memory")
        if self._oom_once:
            self._oom_once = False
            raise RuntimeError("CUDA out of memory")
        self._loaded = True

    async def unload(self):
        self.unload_calls += 1
        self._loaded = False
        self._evicting = False

    def try_reserve_for_eviction(self):
        if not self._loaded or self._op_count > 0 or self._evicting:
            return False
        self._evicting = True
        return True

    def cancel_eviction_reservation(self):
        self._evicting = False

    def begin_operation(self):
        if not self._loaded or self._evicting:
            return False
        self._op_count += 1
        return True

    def end_operation(self):
        if self._op_count > 0:
            self._op_count -= 1


class FakeProbe:
    def __init__(self, total_gb=16, free_gb=None):
        self._total = int(total_gb * GB)
        self._free = int((free_gb if free_gb is not None else total_gb) * GB)

    def snapshot(self):
        return GpuSnapshot(total_bytes=self._total, free_bytes=self._free)


class FakeClock:
    """Monotonic, increments on each read so acquisition order is deterministic."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def _arbiter(probe=None, headroom_gb=0):
    return VramArbiter(
        probe=probe or FakeProbe(total_gb=16),
        headroom_bytes=int(headroom_gb * GB),
        clock=FakeClock(),
    )


async def test_acquire_loads_required_model():
    arb = _arbiter()
    m = FakeModel("whisper", 6)
    arb.register(m)
    async with await arb.acquire(WorkloadNeed(required_models=("whisper",))):
        assert m.is_loaded()
        assert m.load_calls == 1


async def test_already_loaded_model_is_not_reloaded():
    arb = _arbiter()
    m = FakeModel("whisper", 6)
    arb.register(m)
    async with await arb.acquire(WorkloadNeed(required_models=("whisper",))):
        pass
    async with await arb.acquire(WorkloadNeed(required_models=("whisper",))):
        pass
    assert m.load_calls == 1


async def test_evicts_lru_to_make_room():
    arb = _arbiter()  # 16GB, no headroom
    whisper, pyannote, llm = FakeModel("whisper", 6), FakeModel("pyannote", 6), FakeModel("llm", 6)
    for m in (whisper, pyannote, llm):
        arb.register(m)
    # Two finished jobs leave whisper (older) and pyannote (newer) idle+loaded.
    async with await arb.acquire(WorkloadNeed(required_models=("whisper",))):
        pass
    async with await arb.acquire(WorkloadNeed(required_models=("pyannote",))):
        pass
    async with await arb.acquire(WorkloadNeed(required_models=("llm",))):
        assert whisper.unload_calls == 1     # LRU idle victim
        assert not whisper.is_loaded()
        assert set(arb.loaded_keys()) == {"pyannote", "llm"}


async def test_in_use_required_model_is_not_evicted_by_concurrent_admission():
    # S1: the model an in-flight job holds must not be evicted to admit another.
    arb = _arbiter()  # 16GB
    whisper, pyannote, llm = FakeModel("whisper", 6), FakeModel("pyannote", 6), FakeModel("llm", 6)
    for m in (whisper, pyannote, llm):
        arb.register(m)
    async with await arb.acquire(WorkloadNeed(required_models=("whisper",))):  # whisper IN USE
        async with await arb.acquire(WorkloadNeed(required_models=("pyannote",))):
            pass  # pyannote now idle+loaded; card holds whisper(6)+pyannote(6)
        async with await arb.acquire(WorkloadNeed(required_models=("llm",))):
            assert whisper.is_loaded()
            assert whisper.unload_calls == 0   # in-use model protected
            assert not pyannote.is_loaded()    # the idle one was evicted instead


async def test_required_models_are_protected_from_eviction():
    arb = _arbiter()  # 16GB
    a, b, c = FakeModel("a", 6), FakeModel("b", 6), FakeModel("c", 6)
    for m in (a, b, c):
        arb.register(m)
    async with await arb.acquire(WorkloadNeed(required_models=("a",))):
        pass
    async with await arb.acquire(WorkloadNeed(required_models=("c",))):
        pass  # a and c idle+loaded -> 12GB
    # Need both a and b; b must load, room must come from idle c, NOT from a.
    async with await arb.acquire(WorkloadNeed(required_models=("a", "b"))):
        assert a.is_loaded()           # protected: survived
        assert b.is_loaded()
        assert not c.is_loaded()       # evicted instead
        assert c.unload_calls == 1


async def test_held_model_is_never_evicted():
    arb = _arbiter(probe=FakeProbe(total_gb=10))  # tight: forces an eviction attempt
    held, target = FakeModel("held", 6), FakeModel("target", 6)
    arb.register(held)
    arb.register(target)
    async with await arb.acquire(WorkloadNeed(required_models=("held",))):  # in use
        async with await arb.acquire(WorkloadNeed(required_models=("target",))):
            assert held.unload_calls == 0   # a held model is never torn down
            assert held.is_loaded()


async def test_oom_on_load_triggers_aggressive_evict_and_retry():
    arb = _arbiter()
    victim = FakeModel("victim", 6)
    llm = FakeModel("llm", 6, oom_once=True)
    arb.register(victim)
    arb.register(llm)
    async with await arb.acquire(WorkloadNeed(required_models=("victim",))):
        pass  # victim idle+loaded
    async with await arb.acquire(WorkloadNeed(required_models=("llm",))):
        assert llm.load_calls == 2       # failed once, retried once
        assert llm.is_loaded()
        assert victim.unload_calls == 1  # freed on the aggressive pass


async def test_persistent_oom_propagates_after_retry():
    arb = _arbiter()
    llm = FakeModel("llm", 6, oom_always=True)
    arb.register(llm)
    with pytest.raises(RuntimeError, match="out of memory"):
        await arb.acquire(WorkloadNeed(required_models=("llm",)))
    assert llm.load_calls == 2  # tried, aggressive-evicted, retried, still OOM -> raised


async def test_non_oom_load_error_propagates_immediately():
    arb = _arbiter()
    m = FakeModel("m", 6, load_error=RuntimeError("boom"))
    arb.register(m)
    with pytest.raises(RuntimeError, match="boom"):
        await arb.acquire(WorkloadNeed(required_models=("m",)))
    assert m.load_calls == 1  # no retry for a non-OOM error


async def test_admission_uses_driver_free_when_lower_than_accounting():
    # F6 pessimism: our accounting says room, but the driver reports little free
    # (another process holds VRAM) -> the driver figure must force an eviction.
    arb = _arbiter(probe=FakeProbe(total_gb=16, free_gb=4))
    a, b = FakeModel("a", 6), FakeModel("b", 6)
    arb.register(a)
    arb.register(b)
    async with await arb.acquire(WorkloadNeed(required_models=("a",))):
        pass  # a idle+loaded; accounting=10 free, driver=4 free
    async with await arb.acquire(WorkloadNeed(required_models=("b",))):
        assert a.unload_calls == 1  # driver figure (4<6) forced the eviction


async def test_unknown_required_model_raises():
    arb = _arbiter()
    with pytest.raises(KeyError):
        await arb.acquire(WorkloadNeed(required_models=("nope",)))


async def test_transient_reservation_is_held_then_released():
    arb = _arbiter()
    async with await arb.acquire(WorkloadNeed(transient_bytes=4 * GB)) as lease:
        assert arb.transient_reserved_bytes == 4 * GB
        assert not lease._released
    assert arb.transient_reserved_bytes == 0


async def test_lease_release_is_idempotent():
    arb = _arbiter()
    lease = await arb.acquire(WorkloadNeed(transient_bytes=2 * GB))
    lease.release()
    lease.release()  # must not double-decrement below zero
    assert arb.transient_reserved_bytes == 0


async def test_dropped_lease_is_released_by_del_backstop():
    # S3: a lease that is never entered/released must not permanently leak.
    arb = _arbiter()
    m = FakeModel("m", 6)
    arb.register(m)
    lease = await arb.acquire(WorkloadNeed(required_models=("m",), transient_bytes=2 * GB))
    assert arb.transient_reserved_bytes == 2 * GB
    assert m.is_busy
    del lease
    gc.collect()
    assert arb.transient_reserved_bytes == 0  # backstop released the reservation
    assert not m.is_busy                       # and the residency hold


async def test_transient_reservation_forces_eviction():
    arb = _arbiter(probe=FakeProbe(total_gb=10))
    resident = FakeModel("resident", 6)
    arb.register(resident)
    async with await arb.acquire(WorkloadNeed(required_models=("resident",))):
        pass  # resident idle+loaded, 4GB free
    async with await arb.acquire(WorkloadNeed(transient_bytes=6 * GB)):
        assert resident.unload_calls == 1  # evicted to fit the transient workload
        assert arb.transient_reserved_bytes == 6 * GB


async def test_rebind_via_registry_keeps_accounting_consistent():
    # F5 at the arbiter level: a rebound resident is unloaded and not double-counted.
    from concurrent.futures import ThreadPoolExecutor
    from gpu_server.orchestrator.resident_model import ResidentModel

    ex = ThreadPoolExecutor(max_workers=1)
    try:
        m = ResidentModel("whisper", 6 * GB, ex, load_fn=lambda: None, unload_fn=lambda: None)
        arb = _arbiter()
        arb.register(m)
        async with await arb.acquire(WorkloadNeed(required_models=("whisper",))):
            pass
        assert "whisper" in arb.loaded_keys()
        # Simulate stuck-timeout recovery: rebind resets residency to unloaded.
        m.rebind(ex, load_fn=lambda: None, unload_fn=lambda: None)
        assert arb.loaded_keys() == []  # no phantom "loaded" entry
    finally:
        ex.shutdown(wait=True)
