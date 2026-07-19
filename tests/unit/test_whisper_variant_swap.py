"""Tests for the WhisperProcessor per-request variant load/swap (ceiling feature)."""
import sys
import types

import pytest

from gpu_server.config import WhisperConfig
from gpu_server.processors.whisper_processor import WhisperProcessor
from gpu_server.whisper_models import WhisperModelSwapError


def _fake_faster_whisper(monkeypatch, fail_on=None):
    calls = []

    class FakeModel:
        def __init__(self, name, device, compute_type):
            calls.append(name)
            if fail_on and name == fail_on:
                raise RuntimeError(f"simulated load failure for {name}")

    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)
    return calls


@pytest.fixture
def proc():
    procs = []

    def make(ceiling):
        p = WhisperProcessor(WhisperConfig(model=ceiling))
        procs.append(p)
        return p

    yield make
    for p in procs:
        p.shutdown()


def test_cold_load_builds_requested_variant(proc, monkeypatch):
    calls = _fake_faster_whisper(monkeypatch)
    p = proc("large-v3")
    p.request_model("medium")       # smaller than the ceiling
    p._load_model_sync()
    assert p._model_name == "medium"
    assert calls == ["medium"]       # built medium directly — no load-ceiling-then-swap


def test_cold_load_pins_bigger_request_to_ceiling(proc, monkeypatch):
    calls = _fake_faster_whisper(monkeypatch)
    p = proc("medium")               # ceiling = medium
    p.request_model("large-v3")      # bigger than ceiling
    p._load_model_sync()
    assert p._model_name == "medium"
    assert calls == ["medium"]


def test_swap_changes_the_loaded_variant(proc, monkeypatch):
    calls = _fake_faster_whisper(monkeypatch)
    p = proc("large-v3")
    p.request_model("small")
    p._load_model_sync()
    assert p._model_name == "small"
    p._swap_sync("large-v3")         # warm-swap to a different variant
    assert p._model_name == "large-v3"
    assert calls == ["small", "large-v3"]  # unloaded small, built large-v3


def test_swap_failure_raises_and_leaves_no_model(proc, monkeypatch):
    # F-A: a failed swap-load leaves the processor with NO model; it must raise
    # so the worker recreates it (the arbiter still counts whisper resident).
    calls = _fake_faster_whisper(monkeypatch, fail_on="large-v3")
    p = proc("large-v3")
    p.request_model("small")
    p._load_model_sync()             # small loads fine
    with pytest.raises(WhisperModelSwapError):
        p._swap_sync("large-v3")     # build fails
    assert p._model is None
    assert p._model_name is None     # not left pointing at an unloaded variant
    assert p.is_loaded() is False
