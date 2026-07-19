"""
Whisper model size metadata + capability-aware ceiling/variant resolution.

`config.whisper.model` is a CEILING: the server honours a client-requested model
if it is ≤ the ceiling by size, and pins anything larger to the ceiling. The
ceiling itself is clamped to what the detected GPU can actually run alongside the
other audio residents (pyannote + embedding) — so the same config adapts from a
4 GB card to a 24 GB one without change.

Why this is safe with the arbiter: every allowed variant is ≤ the ceiling, and the
arbiter reserves a FIXED ceiling-sized footprint for the "whisper" resident, so it
never needs to know which variant is actually loaded (see whisper_processor's
in-lease swap).
"""
from __future__ import annotations

from typing import Optional

# Size rank of every ALLOWED whisper model (validation.ALLOWED_WHISPER_MODELS).
# `.en` variants are the same size as their multilingual base; `large` is the
# legacy alias for large-v2. A unit test asserts this covers every allowed model —
# an unranked-but-valid model would be silently pinned UP to the ceiling.
WHISPER_MODEL_RANK = {
    "tiny": 1, "tiny.en": 1,
    "base": 2, "base.en": 2,
    "small": 3, "small.en": 3,
    "medium": 4, "medium.en": 4,
    "large": 5, "large-v1": 5, "large-v2": 5, "large-v3": 5,
}

# Conservative faster-whisper fp16 VRAM footprint (GB) per model, used to pick the
# largest model the card can run. Deliberately generous so the auto-ceiling never
# selects a model that then OOMs.
WHISPER_MODEL_VRAM_GB = {
    "tiny": 0.5, "tiny.en": 0.5,
    "base": 0.7, "base.en": 0.7,
    "small": 1.2, "small.en": 1.2,
    "medium": 2.5, "medium.en": 2.5,
    "large": 3.0, "large-v1": 3.0, "large-v2": 3.0, "large-v3": 3.0,
}

# The canonical model to pick for each rank tier when auto-selecting a ceiling
# (prefer the newest/multilingual variant of a size).
_PREFERRED_BY_RANK = {5: "large-v3", 4: "medium", 3: "small", 2: "base", 1: "tiny"}

# Fallback ceiling when GPU VRAM cannot be detected (CPU-only / no CUDA).
DEFAULT_CEILING = "small"


class WhisperModelSwapError(RuntimeError):
    """A per-request variant swap (unload old + load new) failed, leaving the
    processor with no model loaded while the arbiter still counts it resident.
    The worker recreates the processor to resync (F5 rebind)."""


def resolve_effective(requested: Optional[str], ceiling: str) -> str:
    """The model to actually run for a request: ``requested`` if it is ≤ ``ceiling``
    by size, else the ceiling. Falsy or unknown ``requested`` → ceiling (never
    exceed it). An unknown ``ceiling`` (shouldn't happen — it's config-validated)
    also degrades to returning the ceiling as-is."""
    if not requested:
        return ceiling
    rr = WHISPER_MODEL_RANK.get(requested)
    rc = WHISPER_MODEL_RANK.get(ceiling)
    if rr is None or rc is None:
        return ceiling
    return requested if rr <= rc else ceiling


def largest_fitting_model(total_vram_gb: float, coresident_gb: float,
                          headroom_gb: float) -> str:
    """The biggest whisper model that fits alongside the other audio residents:
    ``vram(model) + coresident + headroom ≤ total``. If nothing fits, the smallest
    model (best effort)."""
    budget = total_vram_gb - coresident_gb - headroom_gb
    fitting_ranks = [
        WHISPER_MODEL_RANK[m] for m, v in WHISPER_MODEL_VRAM_GB.items() if v <= budget
    ]
    best_rank = max(fitting_ranks) if fitting_ranks else 1
    return _PREFERRED_BY_RANK[best_rank]


def hardware_ceiling(configured_model: str, total_vram_gb: Optional[float],
                     coresident_gb: float, headroom_gb: float) -> str:
    """Resolve the effective whisper ceiling for this hardware: the configured
    model, clamped so it never exceeds what the detected card can run. With
    ``configured_model`` at the default (large-v3, the biggest model) this yields
    "the largest model that fits" — i.e. auto-detection. An explicit smaller model
    is honoured (and still clamped). If VRAM can't be detected, the configured
    model is used unclamped (falling back to DEFAULT_CEILING only if it's unusable)."""
    if total_vram_gb is None:
        return configured_model or DEFAULT_CEILING
    fits = largest_fitting_model(total_vram_gb, coresident_gb, headroom_gb)
    return resolve_effective(configured_model, fits)
