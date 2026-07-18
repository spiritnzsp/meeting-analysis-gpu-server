"""Tests for LlmProcessor — request shaping, streaming, clamp, cancellation.

Injects a fake llama_cpp.Llama directly (proc._llm) so nothing loads a real
model or touches a GPU.
"""
import pytest

from gpu_server.config import LlmConfig
from gpu_server.processors import ProcessorCancelled
from gpu_server.processors.llm_processor import LlmProcessor


class FakeLlama:
    def __init__(self, chunks=None, finish="stop"):
        self._chunks = chunks if chunks is not None else ["Hello", " world"]
        self._finish = finish
        self.last_kwargs = None
        self.closed = False

    def create_chat_completion(self, **kwargs):
        self.last_kwargs = kwargs
        n = len(self._chunks)
        for i, piece in enumerate(self._chunks):
            reason = self._finish if i == n - 1 else None
            yield {"choices": [{"delta": {"content": piece}, "finish_reason": reason}]}

    def tokenize(self, data, add_bos=True, special=True):
        return [0] * (len(data) // 4)

    def close(self):
        self.closed = True


class CancellingLlama(FakeLlama):
    """Cancels the owning processor part-way through the stream."""

    def __init__(self, proc, request_id):
        super().__init__(chunks=["partial", "more"])
        self._proc = proc
        self._rid = request_id

    def create_chat_completion(self, **kwargs):
        self.last_kwargs = kwargs
        yield {"choices": [{"delta": {"content": "partial"}, "finish_reason": None}]}
        self._proc.cancel(self._rid)  # cancel mid-stream
        yield {"choices": [{"delta": {"content": "more"}, "finish_reason": None}]}


def _proc(**cfg):
    return LlmProcessor(LlmConfig(**cfg))


def test_llm_config_byte_properties_and_opt_in():
    c = LlmConfig(estimated_vram_gb=11.0, kv_headroom_gb=2.0)
    assert c.estimated_vram_bytes == int(11.0 * 1024 ** 3)
    assert c.kv_headroom_bytes == int(2.0 * 1024 ** 3)
    assert c.enabled is False  # opt-in by default


async def test_generate_shapes_request_and_concatenates_stream():
    proc = _proc(n_ctx=4096)
    proc._llm = FakeLlama(chunks=['{', '"ok": true}'], finish="stop")
    text, finish = await proc.generate(
        "SYS", "USER", temperature=0.3, max_tokens=100,
        response_format="json_object", request_id="r1",
    )
    assert text == '{"ok": true}'
    assert finish == "stop"
    kw = proc._llm.last_kwargs
    assert kw["messages"][0] == {"role": "system", "content": "SYS"}
    assert kw["messages"][1] == {"role": "user", "content": "USER"}
    assert kw["stream"] is True
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["temperature"] == 0.3
    proc.shutdown()
    assert proc._llm is None


async def test_generate_uses_config_defaults_when_args_omitted():
    proc = _proc(n_ctx=4096, temperature=0.5, max_tokens=222)
    proc._llm = FakeLlama()
    await proc.generate("s", "u", request_id="r")
    kw = proc._llm.last_kwargs
    assert kw["temperature"] == 0.5
    assert kw["max_tokens"] == 222  # room is ample at n_ctx=4096
    proc.shutdown()


async def test_generate_clamps_max_tokens_to_remaining_context():
    proc = _proc(n_ctx=1000)
    proc._llm = FakeLlama()
    await proc.generate("SYS", "USER", max_tokens=100000, request_id="r")
    # room = n_ctx - prompt_tokens - 64; must be far below the requested max.
    assert proc._llm.last_kwargs["max_tokens"] < 100000
    assert proc._llm.last_kwargs["max_tokens"] > 0
    proc.shutdown()


async def test_generate_raises_when_prompt_exceeds_context():
    proc = _proc(n_ctx=64)  # tiny: no room to generate
    proc._llm = FakeLlama()
    with pytest.raises(RuntimeError, match="too long"):
        await proc.generate("x" * 4000, "y" * 4000, request_id="r")
    proc.shutdown()


async def test_generate_honors_cancellation_mid_stream():
    proc = _proc(n_ctx=4096)
    proc._llm = CancellingLlama(proc, "r1")
    with pytest.raises(ProcessorCancelled):
        await proc.generate("s", "u", request_id="r1")
    proc.shutdown()


async def test_generate_raises_if_model_not_loaded():
    proc = _proc()
    with pytest.raises(RuntimeError, match="before the model was loaded"):
        await proc.generate("s", "u", request_id="r")
    proc.shutdown()


def test_make_resident_model_carries_config_vram():
    proc = _proc(estimated_vram_gb=11.0)
    rm = proc.make_resident_model()
    assert rm.key == "llm"
    assert rm.estimated_vram_bytes == int(11.0 * 1024 ** 3)
    assert not rm.is_loaded()
    proc.shutdown()
