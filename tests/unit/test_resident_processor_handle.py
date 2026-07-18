"""Tests for ResidentProcessorHandle — recreate + rebind of a processor's
arbiter residents as one transaction (the stuck-timeout recovery seam, F5).

Uses a fake processor (real single-thread executor so the ResidentModel
marshalling is genuinely exercised) rather than the heavy real processors."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpu_server.orchestrator.resident_processor_handle import (
    ResidentBinding, ResidentProcessorHandle,
)

GB = 1024 ** 3


class FakeProcessor:
    """Minimal ResidentCapable: a real executor, some residents, a shutdown."""

    def __init__(self, tag: int, keys=("a", "b")):
        self.tag = tag
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._keys = keys
        self.loaded: dict[str, int] = {}
        self.shutdown_called = False
        self.shutdown_thread_id = None

    @property
    def executor(self):
        return self._executor

    def resident_bindings(self):
        return [
            ResidentBinding(
                key=k,
                estimated_vram_bytes=GB,
                load_fn=self._make_load(k),
                unload_fn=self._make_unload(k),
            )
            for k in self._keys
        ]

    def _make_load(self, key):
        def load():
            # Record WHICH processor instance actually did the load, so a test
            # can prove rebind repointed the resident at the new instance.
            self.loaded[key] = self.tag
        return load

    def _make_unload(self, key):
        def unload():
            self.loaded.pop(key, None)
        return unload

    def shutdown(self, timeout: float = 30.0):
        self.shutdown_called = True
        self.shutdown_thread_id = threading.get_ident()
        self._executor.shutdown(wait=True)


@pytest.fixture
def factory():
    created: list[FakeProcessor] = []
    counter = {"n": 0}

    def make():
        proc = FakeProcessor(tag=counter["n"])
        counter["n"] += 1
        created.append(proc)
        return proc

    yield make, created
    for proc in created:
        proc._executor.shutdown(wait=True)


def test_residents_built_once_with_right_keys(factory):
    make, _ = factory
    handle = ResidentProcessorHandle(make)
    keys = {r.key for r in handle.residents()}
    assert keys == {"a", "b"}
    # Sized from the bindings.
    assert all(r.estimated_vram_bytes == GB for r in handle.residents())


def test_duplicate_resident_key_rejected():
    def make():
        proc = FakeProcessor(tag=0, keys=("a", "a"))
        return proc

    with pytest.raises(ValueError, match="duplicate resident key"):
        ResidentProcessorHandle(make)


async def test_recreate_shuts_down_old_off_the_loop_and_rebinds(factory):
    make, created = factory
    handle = ResidentProcessorHandle(make)
    residents_before = handle.residents()
    resident_a = next(r for r in residents_before if r.key == "a")

    # Load on the first processor instance (tag 0).
    await resident_a.load()
    assert handle.processor.tag == 0
    assert handle.processor.loaded["a"] == 0
    assert resident_a.is_loaded()

    loop_thread_id = threading.get_ident()
    await handle.recreate()

    old = created[0]
    new = created[1]
    # Old processor was shut down, and NOT on the event-loop thread (P1-1).
    assert old.shutdown_called
    assert old.shutdown_thread_id is not None
    assert old.shutdown_thread_id != loop_thread_id

    # Same resident OBJECTS survive (identity), now reset to unloaded.
    assert handle.residents() and set(handle.residents()) == set(residents_before)
    assert not resident_a.is_loaded()

    # Loading again runs on the NEW processor instance (tag 1) — rebind worked.
    await resident_a.load()
    assert handle.processor is new
    assert new.loaded["a"] == 1


async def test_recreate_rejects_processor_missing_a_resident():
    counter = {"n": 0}
    execs: list = []

    def make():
        # First instance has (a, b); the second is missing "b".
        keys = ("a", "b") if counter["n"] == 0 else ("a",)
        proc = FakeProcessor(tag=counter["n"], keys=keys)
        counter["n"] += 1
        execs.append(proc)
        return proc

    handle = ResidentProcessorHandle(make)
    try:
        with pytest.raises(KeyError, match="missing residents"):
            await handle.recreate()
    finally:
        for proc in execs:
            proc._executor.shutdown(wait=True)
