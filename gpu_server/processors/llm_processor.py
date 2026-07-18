"""
LLM Processor — on-device text generation via llama-cpp-python.

A GENERIC LLM executor: it runs whatever prompts it is given and returns the
text. All summarisation-specific prompt policy stays client-side, so this
processor stays reusable and has one responsibility — inference.

Residency (loading the GGUF into VRAM, unloading it) is owned by the arbiter via
the ``ResidentModel`` this processor produces (``make_resident_model``); the
blocking load/unload run on this processor's own single-thread executor, so they
serialise behind any in-flight generation (the F1 safety property). ``generate``
assumes the model is already resident — the caller (LlmWorker) holds an arbiter
lease that guarantees it.

Cancellation: a single blocking ``create_chat_completion`` cannot observe the
BaseProcessor cancel flag, so generation is streamed and the flag is checked
between chunks (the token-level cancel the review called for).
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Tuple

from ..config import LlmConfig
from ..logging_config import get_logger
from .base_processor import BaseProcessor

logger = get_logger(__name__)


class LlmProcessor(BaseProcessor):
    """llama.cpp-backed text generation processor."""

    def __init__(self, config: LlmConfig):
        super().__init__(thread_name_prefix="llm_gpu")
        self._config = config
        self._llm = None  # lazily-loaded llama_cpp.Llama

    @property
    def processor_name(self) -> str:
        return "LLM"

    # --- residency (driven by the arbiter's ResidentModel) -----------------

    def make_resident_model(self, key: str = "llm"):
        """Build the arbiter ``ResidentModel`` for this processor. Load/unload
        are bound to THIS processor's executor so residency changes serialise
        behind generation on the same single-thread queue."""
        from ..orchestrator.resident_model import ResidentModel
        return ResidentModel(
            key=key,
            estimated_vram_bytes=self._config.estimated_vram_bytes,
            executor=self._executor,
            load_fn=self._load_model_sync,
            unload_fn=self._unload_resources,
        )

    def _load_model_sync(self) -> None:
        """Blocking model build — runs on the executor thread via ResidentModel."""
        if self._llm is not None:
            return
        if not self._config.model_path:
            raise RuntimeError("LLM model_path is not configured")
        # Import torch BEFORE llama_cpp: the llama-cpp-python wheel bundles an
        # older CUDA runtime (cu124) than torch (cu128); if llama_cpp initialises
        # CUDA first, torch's later import dies with an undefined-symbol error
        # against the already-loaded libcudart. torch-first pins the newer
        # runtime, which the cu124 llama build runs against fine. The server
        # usually imports torch at startup (gpu_info) and the arbiter probe forces
        # it before this load, but pin it here so the LLM path is safe regardless.
        try:
            import torch  # noqa: F401
        except ImportError:
            pass
        from llama_cpp import Llama

        t0 = time.time()
        # q8 KV + flash attention: required for a 14B to fit ~28K context in 16GB
        # (mirrors the app's LlamaCppProvider). n_gpu_layers=-1 offloads all.
        self._llm = Llama(
            model_path=self._config.model_path,
            n_ctx=self._config.n_ctx,
            n_gpu_layers=self._config.n_gpu_layers,
            flash_attn=self._config.flash_attn,
            type_k=self._config.kv_cache_type,
            type_v=self._config.kv_cache_type,
            verbose=False,
        )
        logger.info(
            f"LLM model loaded in {time.time() - t0:.1f}s "
            f"(n_ctx={self._config.n_ctx})"
        )

    def _unload_resources(self) -> None:
        """Free the model's VRAM. Runs on the executor thread (via ResidentModel
        or BaseProcessor.shutdown). Idempotent."""
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Ignoring error closing llama model: {e}")
            self._llm = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    # --- generation --------------------------------------------------------

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        request_id: str = "",
    ) -> Tuple[str, str]:
        """Generate a completion. Returns (text, finish_reason). Raises
        ProcessorCancelled if cancelled mid-stream. Assumes the model is resident
        (the caller holds an arbiter lease)."""
        self._check_shutdown()
        loop = asyncio.get_running_loop()
        self._begin_operation()
        try:
            return await loop.run_in_executor(
                self._executor,
                self._generate_sync,
                system_prompt,
                user_prompt,
                temperature if temperature is not None else self._config.temperature,
                max_tokens if max_tokens is not None else self._config.max_tokens,
                response_format,
                request_id,
            )
        finally:
            self._end_operation()

    def _generate_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[str],
        request_id: str,
    ) -> Tuple[str, str]:
        # Executor thread. The model must be loaded (arbiter guarantees it).
        if self._llm is None:
            raise RuntimeError("LLM generate called before the model was loaded")

        # Clamp to the context window: the KV cache is n_ctx total (prompt +
        # generation), so requesting more than remains would be silently clipped.
        prompt_tokens = self._count_prompt_tokens(system_prompt, user_prompt)
        room = self._config.n_ctx - prompt_tokens - 64
        if room < 256:
            raise RuntimeError(
                f"Prompt is too long for the model's context "
                f"({prompt_tokens} tokens vs n_ctx={self._config.n_ctx})"
            )
        effective_max = min(max_tokens, room)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = dict(
            messages=messages,
            temperature=temperature,
            max_tokens=effective_max,
            stream=True,  # stream so cancellation can be observed between chunks
        )
        if response_format:
            kwargs["response_format"] = {"type": response_format}

        parts: list[str] = []
        finish_reason = ""
        for chunk in self._llm.create_chat_completion(**kwargs):
            # Honour cancellation token-by-token (the flag can't interrupt a
            # single blocking call, so we stream).
            self._check_cancelled(request_id)
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                parts.append(piece)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
        return "".join(parts), finish_reason

    def _count_prompt_tokens(self, system_prompt: str, user_prompt: str) -> int:
        try:
            text = (system_prompt + user_prompt).encode("utf-8")
            return len(self._llm.tokenize(text, add_bos=True, special=True))
        except Exception:
            return 0
