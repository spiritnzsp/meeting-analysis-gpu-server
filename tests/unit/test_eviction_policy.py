"""Tests for LruEvictionPolicy — victim choice under VRAM pressure."""
from gpu_server.orchestrator.eviction_policy import (
    EvictionCandidate,
    LruEvictionPolicy,
)

GB = 1024 ** 3


def _c(key, vram_gb, last_used):
    return EvictionCandidate(key=key, estimated_vram_bytes=int(vram_gb * GB), last_used=last_used)


def test_evicts_least_recently_used_first():
    policy = LruEvictionPolicy()
    candidates = [
        _c("recent", 5, last_used=100.0),
        _c("oldest", 5, last_used=10.0),
        _c("middle", 5, last_used=50.0),
    ]
    # Need to free 5GB -> the single oldest suffices.
    assert policy.select_victims(candidates, 5 * GB) == ["oldest"]


def test_accumulates_until_shortfall_covered():
    policy = LruEvictionPolicy()
    candidates = [
        _c("a", 3, last_used=1.0),
        _c("b", 3, last_used=2.0),
        _c("c", 3, last_used=3.0),
    ]
    # Need 5GB: oldest 'a' (3) not enough, add 'b' (3) -> 6 >= 5, stop before 'c'.
    assert policy.select_victims(candidates, 5 * GB) == ["a", "b"]


def test_selects_all_when_shortfall_exceeds_total():
    policy = LruEvictionPolicy()
    candidates = [_c("a", 3, 1.0), _c("b", 3, 2.0)]
    assert policy.select_victims(candidates, 100 * GB) == ["a", "b"]


def test_zero_or_negative_shortfall_evicts_nothing():
    policy = LruEvictionPolicy()
    candidates = [_c("a", 3, 1.0)]
    assert policy.select_victims(candidates, 0) == []
    assert policy.select_victims(candidates, -5) == []


def test_no_candidates_returns_empty():
    assert LruEvictionPolicy().select_victims([], 5 * GB) == []
