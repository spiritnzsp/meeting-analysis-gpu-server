"""
LLM Worker — drains the LLM request queue, gated by the GPU arbiter.

Mirrors the audio/video worker loop but is the FIRST workload routed through the
VramArbiter: each generation runs under a lease that keeps the LLM model
resident (un-evictable) for its duration. The lease is acquired OUTSIDE the
per-request processing timeout (F4) — time spent waiting for VRAM (e.g. behind
another workload) is not charged to the generation.
"""
from __future__ import annotations

import asyncio
import time

from .backoff import ErrorBackoff
from .config import Config
from .logging_config import get_logger
from .orchestrator import VramArbiter, WorkloadNeed
from .processors import ProcessorCancelled
from .processors.llm_processor import LlmProcessor, PromptTooLongError
from .protocol import LlmGenerateResult
from .queue_manager import QueueManager

logger = get_logger(__name__)

LLM_MODEL_KEY = "llm"


class LlmWorker:
    """Processes LLM_GENERATE requests one at a time, arbiter-gated."""

    def __init__(self, config: Config, queue: QueueManager, arbiter: VramArbiter):
        self.config = config
        self.queue = queue
        self._arbiter = arbiter
        # Build the processor and its resident EAGERLY so the server can register
        # the resident synchronously at construction, uniformly with the audio
        # residents (F9 register-before-acquire in one place). LlmProcessor.__init__
        # only spawns an executor thread — no model load / CUDA at construction.
        self._processor = LlmProcessor(self.config.llm)
        self._resident = self._processor.make_resident_model(LLM_MODEL_KEY)
        self._running = False
        self._backoff = ErrorBackoff()

    def residents(self):
        """The arbiter resident this worker owns (the LLM model). Registered by
        the server alongside the audio residents (F9)."""
        return [self._resident]

    async def start(self):
        self._running = True
        logger.info("LLM worker started")
        await self._run_loop()

    async def stop(self):
        self._running = False
        if self._processor:
            self._processor.shutdown()
        logger.info("LLM worker stopped")

    @property
    def is_processing(self) -> bool:
        return self._processor.is_processing if self._processor else False

    async def _run_loop(self):
        while self._running:
            try:
                queued = await self.queue.wait_for_request()
                if queued.cancelled:
                    continue
                await self._process(queued)
                self._backoff.reset()
            except asyncio.CancelledError:
                logger.info("LLM worker cancelled")
                break
            except Exception as e:  # noqa: BLE001 - keep the worker alive
                logger.error(f"LLM worker loop error: {e}", exc_info=True)
                await self._backoff.sleep()

    async def _process(self, queued):
        request = queued.request
        websocket = queued.websocket
        t0 = time.time()
        # Acquire the lease OUTSIDE the processing timeout (F4). The lease keeps
        # the LLM resident for the whole generation; the arbiter loads it
        # (evicting others if needed) and reserves a small compute-scratch margin.
        need = WorkloadNeed(
            required_models=(LLM_MODEL_KEY,),
            transient_bytes=self.config.llm.kv_headroom_bytes,
        )
        try:
            async with await self._arbiter.acquire(need):
                gen_task = asyncio.ensure_future(self._processor.generate(
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    response_format=request.response_format,
                    request_id=request.request_id,
                ))
                try:
                    # shield so the timeout does NOT cancel the coroutine — a
                    # run_in_executor job can't be cancelled anyway. On timeout we
                    # signal cancel and DRAIN (still holding the lease) so the
                    # executor thread actually stops before the next request's
                    # _begin_operation clears this request's cancel flag (C2), and
                    # so it stops head-of-line-blocking the single-thread executor.
                    text, finish_reason = await asyncio.wait_for(
                        asyncio.shield(gen_task),
                        timeout=self.config.llm_queue.processing_timeout,
                    )
                except asyncio.TimeoutError:
                    self._processor.cancel(request.request_id)
                    try:
                        await gen_task  # drain: raises ProcessorCancelled when the stream sees the flag
                    except (ProcessorCancelled, Exception):
                        pass
                    await self._send(websocket, LlmGenerateResult(
                        request_id=request.request_id, success=False,
                        error_message=(
                            f"LLM generation timed out after "
                            f"{self.config.llm_queue.processing_timeout}s"
                        ),
                        processing_time_seconds=time.time() - t0,
                    ))
                    return
            await self._send(websocket, LlmGenerateResult(
                request_id=request.request_id,
                success=True,
                text=text,
                finish_reason=finish_reason,
                processing_time_seconds=time.time() - t0,
            ))
        except ProcessorCancelled:
            await self._send(websocket, LlmGenerateResult(
                request_id=request.request_id, success=False,
                error_message="Generation cancelled",
                processing_time_seconds=time.time() - t0,
            ))
        except PromptTooLongError as e:
            # INPUT too long — report the real reason + a distinct error_code so
            # the client can chunk/re-split (NOT the generic mask below). This is
            # actionable to the caller, unlike an internal failure.
            logger.warning(f"LLM prompt too long: {e}")
            await self._send(websocket, LlmGenerateResult(
                request_id=request.request_id, success=False,
                error_message=str(e), error_code="CONTEXT_TOO_LONG",
                processing_time_seconds=time.time() - t0,
            ))
        except Exception as e:  # noqa: BLE001
            # Mask internal detail from the client (uniform with the audio/video
            # workers); the specifics are logged server-side.
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            await self._send(websocket, LlmGenerateResult(
                request_id=request.request_id, success=False,
                error_message="LLM generation failed",
                processing_time_seconds=time.time() - t0,
            ))

    @staticmethod
    async def _send(websocket, result: LlmGenerateResult):
        try:
            await websocket.send(result.to_json())
        except Exception as e:  # pragma: no cover - client vanished
            logger.warning(f"Failed to send LLM result: {e}")
