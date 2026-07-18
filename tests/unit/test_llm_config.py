"""Tests for LLM config parsing + validation (the C1 gap that shipped)."""
import pytest

from gpu_server.config import (
    Config,
    ConfigurationError,
    LlmConfig,
    load_config,
    validate_config,
)


def test_llm_config_defaults():
    c = LlmConfig()
    assert c.enabled is False
    assert c.estimated_vram_gb == 13.0
    assert c.estimated_vram_bytes == int(13.0 * 1024 ** 3)


def test_load_config_parses_llm_gpu_sections(tmp_path):
    # The omission of these parse blocks made the feature unreachable via YAML.
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "llm:\n"
        "  enabled: true\n"
        "  model_path: /models/x.gguf\n"
        "  n_ctx: 16384\n"
        "  estimated_vram_gb: 12.0\n"
        "llm_queue:\n"
        "  processing_timeout: 300\n"
        "gpu:\n"
        "  vram_headroom_gb: 2.0\n"
    )
    cfg = load_config(yaml_path, fail_on_error=True)
    assert cfg.llm.enabled is True
    assert cfg.llm.model_path == "/models/x.gguf"
    assert cfg.llm.n_ctx == 16384
    assert cfg.llm.estimated_vram_gb == 12.0
    assert cfg.llm_queue.processing_timeout == 300
    assert cfg.gpu.vram_headroom_gb == 2.0


def test_env_override_enables_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_SERVER_LLM_ENABLED", "true")
    monkeypatch.setenv("GPU_SERVER_LLM_MODEL_PATH", "/models/env.gguf")
    cfg = load_config(tmp_path / "missing.yaml", fail_on_error=False)
    assert cfg.llm.enabled is True
    assert cfg.llm.model_path == "/models/env.gguf"


def test_validate_rejects_enabled_without_model_path():
    cfg = Config()
    cfg.auth.enabled = False
    cfg.llm.enabled = True
    cfg.llm.model_path = ""
    with pytest.raises(ConfigurationError):
        validate_config(cfg)


def test_validate_rejects_missing_model_file(tmp_path):
    cfg = Config()
    cfg.auth.enabled = False
    cfg.llm.enabled = True
    cfg.llm.model_path = str(tmp_path / "nope.gguf")
    with pytest.raises(ConfigurationError):
        validate_config(cfg)


def test_validate_passes_with_real_model_file(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"stub")
    cfg = Config()
    cfg.auth.enabled = False
    cfg.llm.enabled = True
    cfg.llm.model_path = str(gguf)
    assert validate_config(cfg) is True


def test_validate_ignores_llm_when_disabled():
    cfg = Config()  # llm disabled, empty model_path
    cfg.auth.enabled = False
    assert validate_config(cfg) is True
