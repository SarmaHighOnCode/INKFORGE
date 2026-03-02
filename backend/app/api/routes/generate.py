"""
INKFORGE — Generation Routes

POST /api/generate — Submit a handwriting generation job
WS   /api/ws/{job_id}  — WebSocket stream for real-time stroke output
GET  /api/stream/{job_id} — Server-Sent Events (fallback)
GET  /api/job/{job_id} — Poll job status

The WebSocket streaming endpoint is the critical path for real-time inference:
strokes are yielded one-by-one as the model generates them,
giving the frontend a real-time animated preview.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.models.schemas import GenerateRequest, GenerateResponse, JobStatusResponse

logger = logging.getLogger("inkforge.routes.generate")

router = APIRouter()

# In-memory job store (replace with Redis in production)
_jobs: dict[str, dict[str, Any]] = {}

# TTL for job cleanup (1 hour)
_JOB_TTL_SECONDS = 3600
_MAX_JOBS = 500


def _cleanup_old_jobs() -> None:
    """Remove jobs older than TTL and enforce a size cap."""
    if len(_jobs) <= _MAX_JOBS // 2:
        return  # skip cleanup when well under capacity

    now = time.monotonic()
    expired = [
        jid for jid, job in _jobs.items() if now - job.get("created_at", now) > _JOB_TTL_SECONDS
    ]
    for jid in expired:
        del _jobs[jid]

    # If still over capacity, evict oldest
    if len(_jobs) > _MAX_JOBS:
        sorted_ids = sorted(
            _jobs.keys(),
            key=lambda jid: _jobs[jid].get("created_at", 0),
        )
        for jid in sorted_ids[: len(_jobs) - _MAX_JOBS]:
            del _jobs[jid]


@router.post("/generate", response_model=GenerateResponse, status_code=202)
async def generate_handwriting(
    request_body: GenerateRequest,
    request: Request,
) -> GenerateResponse:
    """
    Submit a handwriting generation job.

    Accepts text, style preset, and humanization parameters.
    Creates a job_id and returns the WebSocket URL for real-time
    stroke streaming.

    The actual inference runs lazily when the client connects to the
    stream endpoint — no Celery task is dispatched for streaming jobs.
    For batch/async jobs, use the Celery worker path instead.
    """
    # Periodic cleanup of expired/stale jobs
    _cleanup_old_jobs()

    job_id = str(uuid.uuid4())

    # Store job metadata
    _jobs[job_id] = {
        "status": "queued",
        "created_at": time.monotonic(),
        "text": request_body.text,
        "style_id": request_body.style_id,
        "params": request_body.params.model_dump(),
        "paper_texture": request_body.paper_texture.value,
        "ink_color": request_body.ink_color.value,
        "font_size": request_body.font_size.value,
        "progress": 0.0,
        "error": None,
        "result": None,
    }

    # Build the WebSocket URL (matches the /api/ws/{job_id} route)
    base_url = str(request.base_url).rstrip("/")
    # Convert http(s) to ws(s)
    ws_base = base_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_base}/api/ws/{job_id}"

    # Build the SSE stream URL too
    stream_url = f"{base_url}/api/stream/{job_id}"

    logger.info(
        f"Job {job_id} created: {len(request_body.text)} chars, style={request_body.style_id}"
    )

    return GenerateResponse(
        job_id=job_id,
        ws_url=ws_url,
        stream_url=stream_url,
        status="queued",
    )


@router.websocket("/ws/{job_id}")
async def websocket_stream(websocket: WebSocket, job_id: str) -> None:
    """
    WebSocket endpoint for real-time stroke streaming.

    Connects to the LLMEngine and streams stroke data as the model
    generates it. Each message is a JSON-encoded event:

        {"type": "stroke", "index": 0, "data": {"dx": 8.2, "dy": -0.3, "p1": 1, "p2": 0, "p3": 0}}

    The stream ends with a completion event:

        {"type": "complete", "total_strokes": 1247}
    """
    await websocket.accept()

    if job_id not in _jobs:
        await websocket.send_json({"type": "error", "message": f"Job {job_id} not found"})
        await websocket.close(code=4004)
        return

    job = _jobs[job_id]

    # Get the engine from app state (loaded during lifespan)
    engine = getattr(websocket.app.state, "engine", None)

    if engine is None:
        await websocket.send_json({"type": "error", "message": "Engine unavailable"})
        await websocket.close(code=4003)
        return

    if not engine.is_ready:
        await websocket.send_json(
            {"type": "error", "message": "Engine not ready — model still loading"}
        )
        await websocket.close(code=4003)
        return

    # Mark job as processing
    job["status"] = "processing"

    # Wrap the generator in a task so we can cancel it on disconnect
    gen_task: asyncio.Task | None = None

    try:
        stroke_count = 0

        async def _run_stream():
            nonlocal stroke_count
            async for event in engine.stream_generate(
                text=job["text"],
                style_id=job["style_id"],
                params=job["params"],
            ):
                await websocket.send_json(event)

                if event.get("type") == "stroke":
                    stroke_count += 1
                elif event.get("type") == "complete":
                    job["status"] = "complete"
                    job["result"] = {
                        "total_strokes": event.get("total_strokes", stroke_count),
                    }

        gen_task = asyncio.create_task(_run_stream())
        await gen_task

        # If we exited without a complete event, mark accordingly
        if job["status"] != "complete":
            job["status"] = "complete"

        await websocket.close(code=1000)

    except WebSocketDisconnect:
        logger.info(f"Job {job_id}: client disconnected")
        job["status"] = "cancelled"
        # Cancel the running generation task so the engine stops work
        if gen_task is not None and not gen_task.done():
            gen_task.cancel()

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)

        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=4000)
        except Exception:
            pass


@router.get("/stream/{job_id}")
async def stream_strokes(job_id: str, request: Request) -> StreamingResponse:
    """
    Server-Sent Events (SSE) endpoint for real-time stroke streaming (fallback).

    Connects to the LLMEngine and streams stroke data as the model
    generates it. Each event is a JSON-encoded stroke tuple:

        data: {"type": "stroke", "index": 0, "data": {"dx": 8.2, "dy": -0.3, ...}}

    The stream ends with a completion event:

        data: {"type": "complete", "total_strokes": 1247, ...}
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs[job_id]

    # Get the engine from app state (loaded during lifespan)
    engine = getattr(request.app.state, "engine", None)

    if engine is None:
        raise HTTPException(status_code=503, detail="Engine unavailable")

    if not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Engine not ready — model still loading",
        )

    # Mark job as processing
    job["status"] = "processing"

    async def event_generator():
        """
        Async generator that yields SSE-formatted events.

        SSE format:
            data: {json}\n\n

        Each event is followed by two newlines per the SSE spec.
        """
        try:
            stroke_count = 0

            async for event in engine.stream_generate(
                text=job["text"],
                style_id=job["style_id"],
                params=job["params"],
            ):
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info(f"Job {job_id}: client disconnected")
                    break

                # Format as SSE
                event_data = json.dumps(event, ensure_ascii=False)
                yield f"data: {event_data}\n\n"

                if event.get("type") == "stroke":
                    stroke_count += 1
                elif event.get("type") == "complete":
                    job["status"] = "complete"
                    job["result"] = {
                        "total_strokes": event.get("total_strokes", stroke_count),
                    }

            # If we exited without a complete event, mark accordingly
            if job["status"] != "complete":
                job["status"] = "complete"

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            job["status"] = "failed"
            job["error"] = str(e)

            error_event = json.dumps(
                {
                    "type": "error",
                    "message": str(e),
                }
            )
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Poll job status.

    Returns the current state of a generation job:
    queued → processing → complete | failed
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs[job_id]

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        error=job.get("error"),
    )
