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
from typing import Optional

from .config import Config
from .logging_config import get_logger
from .orchestrator import VramArbiter, WorkloadNeed
from .processors import ProcessorCancelled
from .processors.llm_processor import LlmProcessor
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
        self._processor: Optional[LlmProcessor] = None
        self._running = False
        self._error_backoff_seconds = 1.0
        self._max_error_backoff_seconds = 60.0

    async def start(self):
        self._processor = LlmProcessor(self.config.llm)
        # Register the LLM as an arbiter resident so it participates in VRAM
        # admission/eviction alongside the other workloads (once those are also
        # arbiter-gated). Registration is synchronous, before the drain loop.
        self._arbiter.register(self._processor.make_resident_model(LLM_MODEL_KEY))
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
                self._error_backoff_seconds = 1.0
            except asyncio.CancelledError:
                logger.info("LLM worker cancelled")
                break
            except Exception as e:  # noqa: BLE001 - keep the worker alive
                logger.error(f"LLM worker loop error: {e}", exc_info=True)
                await asyncio.sleep(self._error_backoff_seconds)
                self._error_backoff_seconds = min(
                    self._error_backoff_seconds * 2, self._max_error_backoff_seconds
                )

    async def _process(self, queued):
        request = queued.request
        websocket = queued.websocket
        t0 = time.time()
        try:
            # Acquire the lease OUTSIDE the processing timeout (F4). The lease
            # keeps the LLM resident for the whole generation; the arbiter loads
            # it (evicting others if needed) and reserves KV-growth headroom.
            need = WorkloadNeed(
                required_models=(LLM_MODEL_KEY,),
                transient_bytes=self.config.llm.kv_headroom_bytes,
            )
            async with await self._arbiter.acquire(need):
                text, finish_reason = await asyncio.wait_for(
                    self._processor.generate(
                        system_prompt=request.system_prompt,
                        user_prompt=request.user_prompt,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        response_format=request.response_format,
                        request_id=request.request_id,
                    ),
                    timeout=self.config.llm_queue.processing_timeout,
                )
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
        except asyncio.TimeoutError:
            # Stop the in-flight generation; the executor-serialized teardown
            # keeps VRAM safe even though the lease has been released.
            if self._processor:
                self._processor.cancel(request.request_id)
            await self._send(websocket, LlmGenerateResult(
                request_id=request.request_id, success=False,
                error_message=(
                    f"LLM generation timed out after "
                    f"{self.config.llm_queue.processing_timeout}s"
                ),
                processing_time_seconds=time.time() - t0,
            ))
        except Exception as e:  # noqa: BLE001
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            await self._send(websocket, LlmGenerateResult(
                request_id=request.request_id, success=False,
                error_message=str(e),
                processing_time_seconds=time.time() - t0,
            ))

    @staticmethod
    async def _send(websocket, result: LlmGenerateResult):
        try:
            await websocket.send(result.to_json())
        except Exception as e:  # pragma: no cover - client vanished
            logger.warning(f"Failed to send LLM result: {e}")
