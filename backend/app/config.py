"""
INKFORGE — Application Configuration

Loads environment variables and defines application settings using Pydantic.
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
    host: str = "0.0.0.0"
    port: int = 8000

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Model
    model_checkpoint_path: str = "../checkpoints/lstm_mdn_v1.pt"
    device: str = "cpu"  # "cpu" or "cuda"

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
