"""
INKFORGE — Celery Worker Configuration (GPU-Optimized)

Configures Celery with Redis as broker for async handwriting generation tasks.

CRITICAL GPU CONSTRAINTS:
    - concurrency=1: Only ONE task runs per worker to prevent CUDA OOM errors.
      A 7B+ param model typically consumes 14-16 GB of VRAM. Running two
      inference tasks simultaneously would exceed the 24 GB budget on an A10G
      or RTX 4090.

    - worker_max_memory_per_child: Prevents memory leaks from accumulating
      across task executions. After N tasks, the worker process is recycled.

    - prefetch_multiplier=1: Worker fetches one task at a time, preventing
      a backlog from building up on a slow GPU worker.

For horizontal scaling, spin up additional workers on separate GPU instances
rather than increasing concurrency on a single worker.
"""

from __future__ import annotations

import logging
import time

from celery import Celery
from celery.signals import worker_init, worker_shutdown

from app.config import settings

logger = logging.getLogger("inkforge.worker")

# ============================================================
# Celery App
# ============================================================

worker = Celery(
    "inkforge",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

worker.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task tracking
    task_track_started=True,
    # --- GPU-CRITICAL SETTINGS ---
    # Hard limit: 120s for long documents (multi-page generation)
    task_time_limit=120,
    # Soft limit: 90s — task gets SoftTimeLimitExceeded, can gracefully save state
    task_soft_time_limit=90,
    # ONE task at a time per worker — prevents CUDA OOM
    # To scale: run multiple workers across multiple GPUs
    worker_concurrency=1,
    # Fetch one task at a time — no prefetching on GPU workers
    worker_prefetch_multiplier=1,
    # Recycle worker after 50 tasks to prevent VRAM fragmentation / memory leaks
    worker_max_tasks_per_child=50,
    # Memory guard: kill worker if it exceeds 20 GB resident memory
    # (model ~14 GB + KV cache ~2 GB + overhead)
    worker_max_memory_per_child=20_000_000,  # 20 GB in KB
    # Late ACK: only acknowledge task AFTER completion
    # Prevents task loss if worker crashes mid-inference
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Result expiration: clean up after 1 hour
    result_expires=3600,
)


# ============================================================
# Worker Lifecycle Signals
# ============================================================

# Holds the engine instance at the worker process level
_worker_engine = None


@worker_init.connect
def on_worker_init(**kwargs):
    """
    Called when a Celery worker process starts.

    Loads the LLM model into GPU memory ONCE.
    This is the worker-level equivalent of FastAPI's lifespan.

    NOTE: In production, the Celery worker and FastAPI server may run
    in separate processes (or even separate machines). Each gets its
    own LLMEngine instance — this is by design for GPU isolation.
    """
    global _worker_engine

    import asyncio

    from app.ml.llm_engine import EngineConfig, LLMEngine

    logger.info("=" * 60)
    logger.info("CELERY WORKER — Initializing LLM Engine")
    logger.info("=" * 60)

    _worker_engine = LLMEngine.get_instance()

    config = EngineConfig(
        model_name=settings.model_name,
        checkpoint_path=settings.model_checkpoint_path,
        device=settings.device,
        engine_backend=settings.engine_backend,
        gpu_memory_fraction=settings.gpu_memory_fraction,
        quantization_bits=settings.quantization_bits,
        kv_cache_size_gb=settings.kv_cache_size_gb,
        max_seq_len=settings.max_seq_len,
        max_concurrent_requests=1,  # Celery worker: always 1
        stream_chunk_delay_ms=0,  # No delay needed for batch processing
    )

    # Run async init in sync context
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_worker_engine.initialize_model(config))
    finally:
        loop.close()

    logger.info("CELERY WORKER — Engine ready, accepting tasks")


@worker_shutdown.connect
def on_worker_shutdown(**kwargs):
    """Release GPU resources when the worker shuts down."""
    global _worker_engine

    if _worker_engine is not None:
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_worker_engine.shutdown())
        finally:
            loop.close()

        _worker_engine = None
        logger.info("CELERY WORKER — Engine shut down")


# ============================================================
# Tasks
# ============================================================


@worker.task(
    bind=True,
    name="inkforge.generate",
    max_retries=2,
    default_retry_delay=5,
)
def generate_handwriting(
    self,
    text: str,
    style_id: str,
    params: dict,
) -> dict:
    """
    Async task: Generate handwriting strokes for the given text.

    This is the Celery batch path — used for background jobs where
    the client doesn't need real-time streaming (e.g., bulk export,
    PDF generation, API calls).

    For real-time streaming, use the SSE endpoint instead.

    Args:
        text: Input text to synthesize.
        style_id: ID of the style preset to use.
        params: Humanization parameters dict.

    Returns:
        Dict containing stroke sequence and metadata.
    """
    import asyncio

    global _worker_engine

    if _worker_engine is None or not _worker_engine.is_ready:
        raise RuntimeError(
            "LLM Engine not initialized. Ensure worker_init signal loaded the model."
        )

    task_id = self.request.id
    logger.info(f"Task {task_id}: starting generation ({len(text)} chars)")

    start_time = time.monotonic()

    # Collect all strokes from the async generator
    strokes = []

    async def _collect_strokes():
        async for event in _worker_engine.stream_generate(
            text=text,
            style_id=style_id,
            params=params,
        ):
            if event.get("type") == "stroke":
                strokes.append(event["data"])
            elif event.get("type") == "complete":
                pass  # Final event
            elif event.get("type") == "error":
                error_msg = event.get("data") or event.get("message", "Unknown stream error")
                logger.error(f"Task {task_id}: stream error — {error_msg}")
                raise RuntimeError(f"Stream generation failed: {error_msg}")

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_collect_strokes())
    finally:
        loop.close()

    elapsed_ms = round((time.monotonic() - start_time) * 1000, 1)

    logger.info(f"Task {task_id}: complete — {len(strokes)} strokes in {elapsed_ms}ms")

    return {
        "job_id": task_id,
        "status": "complete",
        "strokes": strokes,
        "metadata": {
            "total_strokes": len(strokes),
            "generation_time_ms": elapsed_ms,
            "text_length": len(text),
            "style_id": style_id,
        },
    }
