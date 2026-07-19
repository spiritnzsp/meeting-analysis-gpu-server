"""Tests for whisper model ceiling/variant resolution + auto-detection."""
import pytest

from gpu_server.validation import ALLOWED_WHISPER_MODELS
from gpu_server.whisper_models import (
    WHISPER_MODEL_RANK, WHISPER_MODEL_VRAM_GB,
    resolve_effective, largest_fitting_model, hardware_ceiling, DEFAULT_CEILING,
)


def test_rank_and_vram_cover_every_allowed_model():
    # F-B: a valid-but-unranked model would be silently pinned UP to the ceiling.
    for m in ALLOWED_WHISPER_MODELS:
        assert m in WHISPER_MODEL_RANK, f"{m} missing from WHISPER_MODEL_RANK"
        assert m in WHISPER_MODEL_VRAM_GB, f"{m} missing from WHISPER_MODEL_VRAM_GB"


def test_resolve_honours_smaller_pins_bigger():
    assert resolve_effective("medium", "large-v3") == "medium"   # smaller honoured
    assert resolve_effective("large-v3", "medium") == "medium"   # bigger pinned
    assert resolve_effective("small", "small") == "small"        # equal
    assert resolve_effective("tiny", "large-v3") == "tiny"


def test_resolve_defaults_to_ceiling_on_empty_or_unknown():
    assert resolve_effective(None, "medium") == "medium"
    assert resolve_effective("", "medium") == "medium"
    assert resolve_effective("nonexistent-model", "medium") == "medium"  # unknown → ceiling (safe)


def test_en_variants_rank_equal_to_base():
    assert resolve_effective("medium.en", "large-v3") == "medium.en"
    assert WHISPER_MODEL_RANK["medium.en"] == WHISPER_MODEL_RANK["medium"]


def test_largest_fitting_scales_with_vram():
    # coresident (pyannote+embedding) 2.5 GB, headroom 1 GB.
    assert largest_fitting_model(16.0, 2.5, 1.0) == "large-v3"  # 12.5 GB budget
    assert largest_fitting_model(6.0, 2.5, 1.0) == "medium"     # 2.5 GB budget
    assert largest_fitting_model(4.0, 2.5, 1.0) == "tiny"       # 0.5 GB budget
    assert largest_fitting_model(3.0, 2.5, 1.0) == "tiny"       # nothing fits → smallest


def test_hardware_ceiling_clamps_and_autodetects():
    # Default (large-v3) auto-detects the largest that fits.
    assert hardware_ceiling("large-v3", 16.0, 2.5, 1.0) == "large-v3"
    assert hardware_ceiling("large-v3", 6.0, 2.5, 1.0) == "medium"   # clamped down
    # An explicit smaller model is honoured (and still clamped).
    assert hardware_ceiling("small", 16.0, 2.5, 1.0) == "small"
    assert hardware_ceiling("large-v3", 4.0, 2.5, 1.0) == "tiny"
    # No detected VRAM → configured model unclamped.
    assert hardware_ceiling("medium", None, 2.5, 1.0) == "medium"
    assert hardware_ceiling("", None, 2.5, 1.0) == DEFAULT_CEILING
