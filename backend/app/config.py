"""
INKFORGE — Application Configuration

Loads environment variables and defines application settings using Pydantic.
Includes LLM infrastructure settings for VRAM management, quantization,
and inference engine backend selection.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "inkforge"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "info"

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # --- LLM / Model Infrastructure ---
    # Engine backend: "lstm" for real model, "mock" for development without model
    # The engine will automatically fall back to mock if no checkpoint is found
    engine_backend: str = "lstm"
    model_name: str = "inkforge-lstm-mdn-v1"
    model_checkpoint_path: str = "checkpoints/lstm_mdn_v1_best.pt"
    device: str = "cpu"  # "cpu" or "cuda"

    # GPU / VRAM Management
    gpu_memory_fraction: float = 0.85  # Fraction of total VRAM to allocate
    quantization_bits: int = 0  # 0=fp16 (no quant), 4=int4, 8=int8
    kv_cache_size_gb: float = 2.0  # KV cache allocation in GB
    max_seq_len: int = 2048  # Maximum sequence length for generation

    # Inference
    max_concurrent_requests: int = 4  # Max parallel inference requests
    stream_chunk_delay_ms: int = 20  # Per-stroke streaming delay (ms)

    # Export
    export_dir: str = "./exports"
    export_dpi: int = 300

    # CORS
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


settings = Settings()
