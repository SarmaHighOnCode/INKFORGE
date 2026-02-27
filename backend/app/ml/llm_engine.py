"""
INKFORGE — LLM Engine (Model Serving Wrapper)

Singleton engine class that manages the LSTM+MDN handwriting model.
Supports both mock mode (development) and real inference (production).

The engine abstracts model lifecycle management:
    - One-time loading on startup
    - VRAM management (GPU) or CPU fallback
    - Async streaming generation
    - Graceful shutdown

Usage:
    engine = LLMEngine.get_instance()
    await engine.initialize_model(config)
    async for stroke in engine.stream_generate(text, style_id, params):
        yield stroke
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator

logger = logging.getLogger("inkforge.engine")


# ============================================================
# Engine Configuration
# ============================================================

class QuantizationMode(str, Enum):
    """Quantization strategy for model weights."""
    NONE = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class EngineConfig:
    """
    Configuration blob passed to the engine at initialization.
    Maps directly to Settings fields — separated so the engine
    has no hard dependency on FastAPI/Pydantic.
    """
    model_name: str = "inkforge-lstm-mdn-v1"
    checkpoint_path: str = "checkpoints/lstm_mdn_v1_best.pt"
    device: str = "cpu"
    engine_backend: str = "lstm"  # "mock" | "lstm" (real model)
    gpu_memory_fraction: float = 0.85
    quantization_bits: int = 0
    kv_cache_size_gb: float = 2.0
    max_seq_len: int = 2048
    max_concurrent_requests: int = 4
    stream_chunk_delay_ms: int = 20  # Faster for real model


@dataclass
class EngineStatus:
    """Runtime status snapshot of the engine."""
    model_loaded: bool = False
    model_name: str = ""
    engine_backend: str = "mock"
    device: str = "cpu"
    gpu_available: bool = False
    gpu_name: str | None = None
    vram_total_gb: float = 0.0
    vram_allocated_gb: float = 0.0
    vram_reserved_gb: float = 0.0
    quantization: str = "fp16"
    kv_cache_gb: float = 0.0
    max_seq_len: int = 2048
    active_requests: int = 0
    total_requests_served: int = 0
    uptime_seconds: float = 0.0


# ============================================================
# Singleton LLM Engine
# ============================================================

class LLMEngine:
    """
    Singleton engine that manages the lifecycle of the LSTM+MDN model.

    Responsibilities:
        - One-time model loading (checkpoint, CUDA setup)
        - VRAM budget enforcement
        - Async streaming generation (yields strokes one by one)
        - Thread-safe singleton — only one instance per process
        - Clean shutdown with memory deallocation

    The engine supports two backends:
        - "mock"  → simulated inference with random strokes (development)
        - "lstm"  → real LSTM+MDN model inference (production)
    """

    _instance: LLMEngine | None = None
    _creation_lock = __import__("threading").Lock()  # thread-safe singleton
    # asyncio lock created in __init__ per instance

    def __new__(cls) -> LLMEngine:
        with cls._creation_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    @classmethod
    def get_instance(cls) -> LLMEngine:
        """Get or create the singleton engine instance."""
        return cls()

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # State
        self._model_loaded: bool = False
        self._config: EngineConfig | None = None
        self._start_time: float = 0.0
        self._active_requests: int = 0
        self._total_requests: int = 0
        self._actual_device: str = "cpu"  # resolved device after init

        # VRAM tracking
        self._vram_total_gb: float = 0.0
        self._vram_allocated_gb: float = 0.0
        self._vram_reserved_gb: float = 0.0
        self._gpu_name: str | None = None

        # Semaphore for concurrent request limiting
        self._semaphore: asyncio.Semaphore | None = None

        # Model objects
        self._inference_service: Any = None  # InferenceService instance
        self._use_real_model: bool = False

        # Instance-bound asyncio lock (safe as it's created during fastapi lifespan)
        self._lock = asyncio.Lock()

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    async def initialize_model(self, config: EngineConfig) -> None:
        """
        Load the model into memory. Called ONCE during FastAPI lifespan startup.

        Args:
            config: Engine configuration from application settings.
        """
        async with self._lock:
            if self._model_loaded:
                logger.warning("Model already loaded — skipping re-initialization")
                return

            self._config = config
            self._start_time = time.monotonic()
            self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

            logger.info("=" * 60)
            logger.info("INKFORGE ENGINE — Initialization Sequence")
            logger.info("=" * 60)

            # --- Step 1: Device Detection ---
            logger.info("[1/4] Probing compute devices...")
            actual_device = config.device

            if config.device == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        self._gpu_name = torch.cuda.get_device_name(0)
                        self._vram_total_gb = round(
                            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1
                        )
                        logger.info(f"  → CUDA device found: {self._gpu_name}")
                        logger.info(f"  → Total VRAM: {self._vram_total_gb} GB")
                    else:
                        logger.warning("  → CUDA requested but not available — falling back to CPU")
                        actual_device = "cpu"
                except ImportError:
                    logger.warning("  → PyTorch not installed — falling back to CPU")
                    actual_device = "cpu"
            else:
                logger.info("  → Using CPU (no GPU acceleration)")

            # --- Step 2: Load Model ---
            logger.info(f"[2/4] Loading model: {config.model_name}")

            if config.engine_backend == "lstm":
                # Try to load real LSTM+MDN model
                checkpoint_path = Path(config.checkpoint_path)

                if checkpoint_path.exists():
                    try:
                        from app.services.inference import InferenceService

                        self._inference_service = InferenceService(
                            checkpoint_path=str(checkpoint_path),
                            device=actual_device,
                        )
                        self._inference_service.load_model()
                        self._inference_service.warmup()

                        self._use_real_model = True
                        self._vram_allocated_gb = 0.5  # LSTM is small
                        logger.info(f"  → LSTM+MDN model loaded from: {checkpoint_path}")
                        logger.info(f"  → Device: {actual_device}")

                    except Exception as e:
                        logger.warning(f"  → Failed to load model: {e}")
                        logger.info("  → Falling back to mock mode")
                        self._use_real_model = False
                else:
                    logger.info(f"  → Checkpoint not found: {checkpoint_path}")
                    logger.info("  → Using mock mode (train a model first)")
                    self._use_real_model = False

            elif config.engine_backend == "mock":
                logger.info("  → Mock mode enabled (random stroke generation)")
                self._use_real_model = False
            else:
                logger.warning(f"  → Unknown backend: {config.engine_backend}, using mock")
                self._use_real_model = False

            # --- Step 3: Configuration ---
            logger.info("[3/4] Configuring engine...")
            logger.info(f"  → Max concurrent requests: {config.max_concurrent_requests}")
            logger.info(f"  → Stream delay: {config.stream_chunk_delay_ms}ms")

            # --- Step 4: Warmup ---
            logger.info("[4/4] Engine ready")

            # --- Ready ---
            self._model_loaded = True
            self._actual_device = actual_device
            logger.info("=" * 60)
            logger.info(f"ENGINE READY — {config.model_name}")
            logger.info(f"  Backend:  {'LSTM+MDN' if self._use_real_model else 'Mock'}")
            logger.info(f"  Device:   {actual_device}")
            if self._gpu_name:
                logger.info(f"  GPU:      {self._gpu_name}")
            logger.info("=" * 60)

    async def shutdown(self) -> None:
        """
        Release model from memory. Called during FastAPI lifespan shutdown.
        """
        async with self._lock:
            logger.info("ENGINE SHUTDOWN — Releasing resources...")

            if self._inference_service is not None:
                del self._inference_service
                self._inference_service = None

            self._vram_allocated_gb = 0.0
            self._vram_reserved_gb = 0.0
            self._model_loaded = False
            self._use_real_model = False

            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("  → Resources released")
            logger.info("ENGINE SHUTDOWN — Complete")

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    async def stream_generate(
        self,
        text: str,
        style_id: str = "neat_cursive",
        params: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Async generator that yields stroke data one at a time.

        Uses the real LSTM+MDN model if loaded, otherwise falls back
        to mock stroke generation for development.

        Args:
            text: Input text to synthesize as handwriting.
            style_id: Style preset identifier.
            params: Humanization parameters dict.

        Yields:
            Dicts with type="stroke" containing (dx, dy, p1, p2, p3) data,
            followed by a type="complete" event.
        """
        if not self._model_loaded:
            raise RuntimeError("Engine not initialized — call initialize_model() first")

        params = params or {}
        config = self._config

        # Enforce concurrency limit
        async with self._semaphore:
            self._active_requests += 1
            self._total_requests += 1
            request_id = self._total_requests
            start_time = time.monotonic()

            logger.info(
                f"[req-{request_id}] Starting generation: "
                f"{len(text)} chars, style={style_id}, "
                f"active={self._active_requests}/{config.max_concurrent_requests}"
            )

            try:
                if self._use_real_model and self._inference_service is not None:
                    # Real LSTM+MDN inference
                    async for event in self._stream_real_model(
                        text, style_id, params, config, request_id, start_time
                    ):
                        yield event
                else:
                    # Mock inference (fallback)
                    async for event in self._stream_mock(
                        text, style_id, params, config, request_id, start_time
                    ):
                        yield event

            finally:
                self._active_requests -= 1

    async def _stream_real_model(
        self,
        text: str,
        style_id: str,
        params: dict[str, Any],
        config: EngineConfig,
        request_id: int,
        start_time: float,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream strokes from the real LSTM+MDN model.
        """
        temperature = params.get("character_inconsistency", 0.4)

        # Generate strokes using InferenceService
        # Run in thread pool to avoid blocking the async event loop
        try:
            strokes = await asyncio.to_thread(
                self._inference_service.generate,
                text=text,
                style_id=style_id,
                params=params,
                temperature=temperature,
                max_strokes=50,  # Per character
            )
        except Exception as e:
            logger.error(f"[req-{request_id}] Model inference failed: {e}")
            # Fall back to mock on error
            async for event in self._stream_mock(
                text, style_id, params, config, request_id, start_time
            ):
                yield event
            return

        # Stream the strokes with layout positioning
        stroke_index = 0
        cursor_x = 40.0
        cursor_y = 0.0
        line_num = 0
        char_idx = 0

        page_width = 700.0
        line_height = 28.0
        margin_left = 40.0
        margin_right = page_width - 40.0

        for dx, dy, p1, p2, p3 in strokes:
            # Track position for layout
            cursor_x += dx
            cursor_y += dy

            # Line wrap check
            if cursor_x > margin_right:
                cursor_x = margin_left
                cursor_y += line_height
                line_num += 1

            # Get current character if available
            current_char = text[char_idx] if char_idx < len(text) else ""
            if p2 == 1:  # Pen up = move to next character
                char_idx += 1

            stroke_event = {
                "type": "stroke",
                "index": stroke_index,
                "data": {
                    "dx": round(dx, 3),
                    "dy": round(dy, 3),
                    "p1": int(p1),
                    "p2": int(p2),
                    "p3": int(p3),
                    "char": current_char,
                    "x": round(cursor_x, 2),
                    "y": round(cursor_y, 2),
                },
            }

            yield stroke_event
            stroke_index += 1

            # Small delay for streaming effect
            delay = config.stream_chunk_delay_ms / 1000.0
            await asyncio.sleep(delay)

            # Check for end of sequence
            if p3 == 1:
                break

        # Completion event
        elapsed_ms = (time.monotonic() - start_time) * 1000
        yield {
            "type": "complete",
            "total_strokes": stroke_index,
            "total_chars": len(text),
            "lines": line_num + 1,
            "generation_time_ms": round(elapsed_ms, 1),
        }

        logger.info(
            f"[req-{request_id}] Complete: "
            f"{stroke_index} strokes, {line_num + 1} lines, "
            f"{elapsed_ms:.1f}ms"
        )

    async def _stream_mock(
        self,
        text: str,
        style_id: str,
        params: dict[str, Any],
        config: EngineConfig,
        request_id: int,
        start_time: float,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream mock strokes for development/demo when no model is available.
        """
        # Use a non-whitespace sentinel so \n survives .split()
        words = text.replace("\n", " __NL__ ").split()
        total_words = len([w for w in words if w != "__NL__"])

        temperature = params.get("character_inconsistency", 0.4)
        fatigue = params.get("fatigue_simulation", 0.3)
        slant = params.get("slant_angle", 5.0)
        baseline_drift = params.get("baseline_drift", 0.3)

        # Layout state
        cursor_x = 0.0
        cursor_y = 0.0
        line_num = 0
        page_width = 700.0
        line_height = 28.0
        margin_left = 40.0
        margin_right = page_width - 40.0
        stroke_index = 0

        for word_idx, word in enumerate(words):
            # Handle paragraph breaks
            if word == "__NL__":
                cursor_x = margin_left + random.uniform(-2, 5)
                cursor_y += line_height * 1.2
                line_num += 1
                continue

            word_width = len(word) * random.uniform(8.0, 12.0)

            # Line wrap
            if cursor_x + word_width > margin_right:
                cursor_x = margin_left + random.uniform(-2, 3)
                cursor_y += line_height + random.uniform(-1.5, 1.5)
                line_num += 1

            # Baseline drift (Exaggerated for visual awareness in mock mode)
            global_drift = baseline_drift * 8.0 * math.sin(
                2 * math.pi * line_num / max(total_words / 5.0, 3.0) + random.uniform(0, 0.5)
            )

            char_x = cursor_x
            for char_idx, char in enumerate(word):
                fatigue_factor = 1.0 + fatigue * (word_idx / max(total_words, 1))

                # Generate a small scribble for the character (~15 strokes)
                num_strokes = max(1, int(random.gauss(15, 3)))
                char_width_target = max(0.5, random.gauss(8.0, 2.0 * temperature * fatigue_factor))
                
                # We need to end up char_width_target to the right
                dx_base = char_width_target / num_strokes
                
                for step in range(num_strokes):
                    # Progress through the character 0.0 -> 1.0
                    t = step / num_strokes
                    
                    # Create some loops and zig-zags
                    # dx is mostly forward, but occasionally backwards
                    dx = random.gauss(dx_base, 1.0 * temperature)
                    
                    # dy oscillates to draw height
                    # frequency is roughly 2-3 up/down sweeps per character
                    freq = random.uniform(2.0, 3.0)
                    dy = math.sin(t * math.pi * 2 * freq) * random.uniform(3.0, 6.0)
                    dy += random.gauss(0.0 + global_drift, 0.5 * temperature)
                    dy += slant * 0.1 * random.gauss(0, 0.3)

                    stroke_event = {
                        "type": "stroke",
                        "index": stroke_index,
                        "data": {
                            "dx": round(dx, 3),
                            "dy": round(dy, 3),
                            "p1": 1,  # pen always down during char
                            "p2": 0,
                            "p3": 0,
                            "char": char if step == 0 else "",  # only send char on first stroke
                            "x": round(char_x, 2),
                            "y": round(cursor_y + dy, 2),
                        },
                    }

                    yield stroke_event
                    stroke_index += 1
                    char_x += dx

                    delay = config.stream_chunk_delay_ms / 1000.0
                    delay *= random.uniform(0.2, 0.8) # faster for mock strokes
                    await asyncio.sleep(delay)

            # Pen-up between words
            word_space = random.uniform(8, 14)
            yield {
                "type": "stroke",
                "index": stroke_index,
                "data": {
                    "dx": round(word_space, 3),
                    "dy": 0.0,
                    "p1": 0,
                    "p2": 1,
                    "p3": 0,
                    "char": " ",
                    "word_idx": word_idx,
                    "x": round(char_x + word_space, 2),
                    "y": round(cursor_y, 2),
                    "pressure": 0.0,
                },
            }
            stroke_index += 1
            cursor_x = char_x + word_space

            await asyncio.sleep(0.02)

        # Completion event
        elapsed_ms = (time.monotonic() - start_time) * 1000
        yield {
            "type": "complete",
            "total_strokes": stroke_index,
            "total_words": total_words,
            "lines": line_num + 1,
            "generation_time_ms": round(elapsed_ms, 1),
        }

        logger.info(
            f"[req-{request_id}] Complete (mock): "
            f"{stroke_index} strokes, {line_num + 1} lines"
        )

    # --------------------------------------------------------
    # Status & Introspection
    # --------------------------------------------------------

    def get_status(self) -> EngineStatus:
        """Return a snapshot of the engine's current state."""
        config = self._config or EngineConfig()
        quant = self._resolve_quantization(config.quantization_bits)

        uptime = 0.0
        if self._start_time > 0:
            uptime = time.monotonic() - self._start_time

        backend = "lstm" if self._use_real_model else "mock"

        return EngineStatus(
            model_loaded=self._model_loaded,
            model_name=config.model_name,
            engine_backend=backend,
            device=config.device,
            gpu_available=self._gpu_name is not None,
            gpu_name=self._gpu_name,
            vram_total_gb=self._vram_total_gb,
            vram_allocated_gb=self._vram_allocated_gb,
            vram_reserved_gb=self._vram_reserved_gb,
            quantization=quant.value,
            kv_cache_gb=config.kv_cache_size_gb,
            max_seq_len=config.max_seq_len,
            active_requests=self._active_requests,
            total_requests_served=self._total_requests,
            uptime_seconds=round(uptime, 1),
        )

    @property
    def is_ready(self) -> bool:
        """Whether the engine is loaded and ready for inference."""
        return self._model_loaded

    @staticmethod
    def _resolve_quantization(bits: int) -> QuantizationMode:
        """Map quantization bits config to enum."""
        if bits == 4:
            return QuantizationMode.INT4
        elif bits == 8:
            return QuantizationMode.INT8
        return QuantizationMode.NONE
