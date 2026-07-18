"""Tests for VramArbiter — admission, eviction, protection, OOM-tolerance, leases.

Uses fake models (satisfying the ManagedModel protocol, no GPU) and a scripted
memory probe so admission decisions are deterministic. The probe reports all
memory free, so admission is driven by the arbiter's own resident accounting.
"""
import pytest

from gpu_server.gpu_info import GpuSnapshot
from gpu_server.orchestrator.vram_arbiter import VramArbiter, WorkloadNeed

GB = 1024 ** 3


class FakeModel:
    """In-memory ManagedModel: tracks load/unload calls, can simulate a
    one-shot OOM and a busy (in-operation) state."""

    def __init__(self, key, vram_gb, oom_once=False):
        self.key = key
        self.estimated_vram_bytes = int(vram_gb * GB)
        self._loaded = False
        self._busy = False
        self._evicting = False
        self._oom_once = oom_once
        self.load_calls = 0
        self.unload_calls = 0

    def is_loaded(self):
        return self._loaded

    async def load(self):
        self.load_calls += 1
        if self._oom_once:
            self._oom_once = False
            raise RuntimeError("CUDA out of memory")
        self._loaded = True

    async def unload(self):
        self.unload_calls += 1
        self._loaded = False
        self._evicting = False

    def try_reserve_for_eviction(self):
        if not self._loaded or self._busy or self._evicting:
            return False
        self._evicting = True
        return True

    def cancel_eviction_reservation(self):
        self._evicting = False

    def begin_operation(self):
        if not self._loaded or self._evicting:
            return False
        self._busy = True
        return True

    def end_operation(self):
        self._busy = False


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
    await arb.acquire(WorkloadNeed(required_models=("whisper",)))
    await arb.acquire(WorkloadNeed(required_models=("whisper",)))
    assert m.load_calls == 1


async def test_evicts_lru_to_make_room():
    arb = _arbiter()  # 16GB, no headroom
    whisper, pyannote, llm = FakeModel("whisper", 6), FakeModel("pyannote", 6), FakeModel("llm", 6)
    for m in (whisper, pyannote, llm):
        arb.register(m)
    await arb.acquire(WorkloadNeed(required_models=("whisper",)))    # loaded first (oldest)
    await arb.acquire(WorkloadNeed(required_models=("pyannote",)))   # 12GB resident now
    await arb.acquire(WorkloadNeed(required_models=("llm",)))        # needs 6, only 4 free
    # Oldest (whisper) evicted to fit the LLM; pyannote (more recent) survives.
    assert whisper.unload_calls == 1
    assert not whisper.is_loaded()
    assert set(arb.loaded_keys()) == {"pyannote", "llm"}


async def test_required_models_are_protected_from_eviction():
    arb = _arbiter()  # 16GB
    a, b, c = FakeModel("a", 6), FakeModel("b", 6), FakeModel("c", 6)
    for m in (a, b, c):
        arb.register(m)
    await arb.acquire(WorkloadNeed(required_models=("a",)))   # a loaded
    await arb.acquire(WorkloadNeed(required_models=("c",)))   # c loaded -> 12GB
    # Now need both a and b; b must load, room must come from c, NOT from a.
    await arb.acquire(WorkloadNeed(required_models=("a", "b")))
    assert a.is_loaded()           # protected: survived
    assert b.is_loaded()
    assert not c.is_loaded()       # evicted instead
    assert c.unload_calls == 1


async def test_busy_model_is_never_evicted():
    arb = _arbiter(probe=FakeProbe(total_gb=10))  # tight: forces an eviction attempt
    busy, target = FakeModel("busy", 6), FakeModel("target", 6)
    arb.register(busy)
    arb.register(target)
    await arb.acquire(WorkloadNeed(required_models=("busy",)))
    assert busy.begin_operation()  # busy is mid-operation
    # target needs 6GB but only 4 is free; the only candidate (busy) is in-op.
    await arb.acquire(WorkloadNeed(required_models=("target",)))
    assert busy.unload_calls == 0  # F3: a busy model is never torn down
    assert busy.is_loaded()


async def test_oom_on_load_triggers_aggressive_evict_and_retry():
    arb = _arbiter()
    victim = FakeModel("victim", 6)
    llm = FakeModel("llm", 6, oom_once=True)
    arb.register(victim)
    arb.register(llm)
    await arb.acquire(WorkloadNeed(required_models=("victim",)))  # victim resident
    # llm's first load raises OOM despite accounting; arbiter evicts and retries.
    await arb.acquire(WorkloadNeed(required_models=("llm",)))
    assert llm.load_calls == 2      # failed once, retried once
    assert llm.is_loaded()
    assert victim.unload_calls == 1  # freed on the aggressive pass


async def test_unknown_required_model_raises():
    arb = _arbiter()
    with pytest.raises(KeyError):
        await arb.acquire(WorkloadNeed(required_models=("nope",)))


async def test_transient_reservation_is_held_then_released():
    arb = _arbiter()
    lease = await arb.acquire(WorkloadNeed(transient_bytes=4 * GB))
    assert arb.transient_reserved_bytes == 4 * GB
    async with lease:
        assert arb.transient_reserved_bytes == 4 * GB
    assert arb.transient_reserved_bytes == 0


async def test_lease_release_is_idempotent():
    arb = _arbiter()
    lease = await arb.acquire(WorkloadNeed(transient_bytes=2 * GB))
    lease.release()
    lease.release()  # must not double-decrement below zero
    assert arb.transient_reserved_bytes == 0


async def test_transient_reservation_forces_eviction():
    # A transient need (e.g. NVENC) must be able to evict a resident model.
    arb = _arbiter(probe=FakeProbe(total_gb=10))
    resident = FakeModel("resident", 6)
    arb.register(resident)
    await arb.acquire(WorkloadNeed(required_models=("resident",)))  # 6GB resident, 4 free
    async with await arb.acquire(WorkloadNeed(transient_bytes=6 * GB)):
        assert resident.unload_calls == 1  # evicted to fit the transient workload
        assert arb.transient_reserved_bytes == 6 * GB
