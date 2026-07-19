"""Tests for the Phase-D resident wiring on the real audio processors:
resident_bindings, is_loaded, the pyannote per-resident unload split, and the
whisper model-constraint collapse. Model loading is faked (no GPU/model files)."""
import sys
import types

import pytest

from gpu_server.config import WhisperConfig, PyAnnoteConfig
from gpu_server.processors.whisper_processor import (
    WhisperProcessor, WHISPER_MODEL_KEY,
)
from gpu_server.processors.pyannote_processor import (
    PyAnnoteProcessor, PYANNOTE_MODEL_KEY, PYANNOTE_EMBEDDING_KEY,
)

GB = 1024 ** 3


@pytest.fixture
def whisper():
    proc = WhisperProcessor(WhisperConfig(model="large-v3", estimated_vram_gb=3.0))
    yield proc
    proc.shutdown()


@pytest.fixture
def pyannote():
    proc = PyAnnoteProcessor(
        PyAnnoteConfig(estimated_vram_gb=2.0, embedding_estimated_vram_gb=0.5)
    )
    yield proc
    proc.shutdown()


# --- whisper ---------------------------------------------------------------

def test_whisper_single_resident_binding(whisper):
    bindings = whisper.resident_bindings()
    assert len(bindings) == 1
    b = bindings[0]
    assert b.key == WHISPER_MODEL_KEY == "whisper"
    assert b.estimated_vram_bytes == int(3.0 * GB)
    assert b.load_fn == whisper._load_model_sync
    assert b.unload_fn == whisper._unload_resources


def test_whisper_is_loaded_tracks_model(whisper):
    assert not whisper.is_loaded()
    whisper._model = object()
    assert whisper.is_loaded()


def test_whisper_load_is_constrained_to_config_model(whisper, monkeypatch):
    # Fake faster_whisper so no real model is fetched.
    recorded = {}

    class FakeWhisperModel:
        def __init__(self, model, device, compute_type):
            recorded["model"] = model
            recorded["device"] = device

    fake_mod = types.ModuleType("faster_whisper")
    fake_mod.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_mod)

    whisper._load_model_sync()
    assert whisper.is_loaded()
    assert recorded["model"] == "large-v3"  # config.model, no override path
    # Idempotent: a second call does not rebuild.
    recorded.clear()
    whisper._load_model_sync()
    assert recorded == {}


# --- pyannote --------------------------------------------------------------

def test_pyannote_two_resident_bindings(pyannote):
    bindings = {b.key: b for b in pyannote.resident_bindings()}
    assert set(bindings) == {PYANNOTE_MODEL_KEY, PYANNOTE_EMBEDDING_KEY}
    assert bindings[PYANNOTE_MODEL_KEY].estimated_vram_bytes == int(2.0 * GB)
    assert bindings[PYANNOTE_EMBEDDING_KEY].estimated_vram_bytes == int(0.5 * GB)
    # Each resident has its OWN unload (P0-2) so evicting one can't desync the other.
    assert bindings[PYANNOTE_MODEL_KEY].unload_fn == pyannote._unload_pipeline
    assert bindings[PYANNOTE_EMBEDDING_KEY].unload_fn == pyannote._unload_embedding
    assert bindings[PYANNOTE_MODEL_KEY].unload_fn != bindings[PYANNOTE_EMBEDDING_KEY].unload_fn


def test_pyannote_is_loaded_per_resident(pyannote):
    assert not pyannote.is_pipeline_loaded()
    assert not pyannote.is_embedding_loaded()
    pyannote._pipeline = object()
    assert pyannote.is_pipeline_loaded()
    assert not pyannote.is_embedding_loaded()


def test_pyannote_unload_pipeline_leaves_embedding(pyannote):
    pyannote._pipeline = object()
    pyannote._embedding_model = object()
    pyannote._unload_pipeline()
    assert pyannote._pipeline is None
    assert pyannote._embedding_model is not None  # embedding untouched
    assert pyannote.is_embedding_loaded()


def test_pyannote_unload_embedding_leaves_pipeline(pyannote):
    pyannote._pipeline = object()
    pyannote._embedding_model = object()
    pyannote._unload_embedding()
    assert pyannote._embedding_model is None
    assert pyannote._pipeline is not None  # pipeline untouched
    assert pyannote.is_pipeline_loaded()


def test_pyannote_unload_resources_frees_both(pyannote):
    pyannote._pipeline = object()
    pyannote._embedding_model = object()
    pyannote._unload_resources()
    assert pyannote._pipeline is None
    assert pyannote._embedding_model is None
