"""Tests for Phase-D VRAM sizing config: the VramEstimate VO plus the
whisper/pyannote/video sizing fields, their parsing, and validation."""
import textwrap

import pytest

from gpu_server.config import (
    Config, VramEstimate, WhisperConfig, PyAnnoteConfig, VideoEncodingConfig,
    load_config, validate_config, ConfigurationError,
)

GB = 1024 ** 3


def test_vram_estimate_converts_gb_to_bytes():
    assert VramEstimate(1.0).bytes == GB
    assert VramEstimate(0.5).bytes == GB // 2
    assert VramEstimate(0).bytes == 0


def test_vram_estimate_is_frozen():
    est = VramEstimate(2.0)
    with pytest.raises(Exception):
        est.gb = 3.0  # frozen dataclass


def test_default_sizing_bytes_match_gb():
    w = WhisperConfig()
    assert w.estimated_vram_bytes == int(w.estimated_vram_gb * GB)
    assert w.estimated_vram_gb == 3.0

    p = PyAnnoteConfig()
    assert p.estimated_vram_bytes == int(p.estimated_vram_gb * GB)
    assert p.embedding_estimated_vram_bytes == int(p.embedding_estimated_vram_gb * GB)
    assert p.estimated_vram_gb == 2.0
    assert p.embedding_estimated_vram_gb == 0.5

    v = VideoEncodingConfig()
    assert v.per_session_vram_bytes == int(v.per_session_vram_gb * GB)
    assert v.per_session_vram_gb == 1.5


def test_load_config_parses_sizing(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent("""
        auth:
          enabled: false
        whisper:
          model: large-v3
          estimated_vram_gb: 4.5
        pyannote:
          huggingface_token: tok
          estimated_vram_gb: 2.5
          embedding_estimated_vram_gb: 0.75
        video_encoding:
          enabled: true
          per_session_vram_gb: 2.0
    """))
    config = load_config(cfg_file)
    assert config.whisper.estimated_vram_gb == 4.5
    assert config.pyannote.estimated_vram_gb == 2.5
    assert config.pyannote.embedding_estimated_vram_gb == 0.75
    assert config.video_encoding.per_session_vram_gb == 2.0


def test_validate_rejects_nonpositive_whisper_sizing():
    config = Config()
    config.auth.enabled = False
    config.whisper.estimated_vram_gb = 0
    with pytest.raises(ConfigurationError, match="whisper.estimated_vram_gb"):
        validate_config(config, strict=True)


def test_validate_rejects_nonpositive_pyannote_embedding_sizing():
    config = Config()
    config.auth.enabled = False
    config.pyannote.embedding_estimated_vram_gb = -1.0
    with pytest.raises(ConfigurationError, match="embedding_estimated_vram_gb"):
        validate_config(config, strict=True)


def test_validate_rejects_nonpositive_video_session_sizing():
    config = Config()
    config.auth.enabled = False
    config.video_encoding.enabled = True
    config.video_encoding.per_session_vram_gb = 0
    with pytest.raises(ConfigurationError, match="per_session_vram_gb"):
        validate_config(config, strict=True)


def test_validate_accepts_default_sizing():
    config = Config()
    config.auth.enabled = False
    config.pyannote.huggingface_token = "tok"
    # Should not raise on the sizing fields (defaults are all positive).
    assert validate_config(config, strict=True) is True
