"""Environment-driven settings for the inference API.

Every value is overridable with a ``TITANIC_`` environment variable, so the
service can be reconfigured for a load test without editing code::

    $env:TITANIC_MAX_CONCURRENCY=1; uvicorn api.main:app
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from titanic.config import Paths


class Settings(BaseSettings):
    """Runtime configuration for the API process.

    Attributes:
        artifacts_dir: Where trained bundles live.
        max_concurrency: Parallel inference slots.
        max_queue: Requests allowed to wait for a slot.
        queue_timeout_s: Maximum wait before a request is rejected.
        default_model: Model used when a request does not name one.
        admin_token: Guards /admin/reload. Reload is disabled when unset.
        cors_origins: Origins allowed to call the API.
    """

    model_config = SettingsConfigDict(env_prefix="TITANIC_", case_sensitive=False)

    artifacts_dir: Path = Paths().artifacts
    max_concurrency: int = 2
    max_queue: int = 64
    queue_timeout_s: float = 5.0
    default_model: str | None = None
    admin_token: str | None = None

    # Localhost only: the Streamlit app is the single intended browser client,
    # and a wildcard would be an unnecessary default on a service with no auth.
    cors_origins: list[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
    ]


def get_settings() -> Settings:
    """Build settings from the environment.

    Returns:
        A populated :class:`Settings`.
    """
    return Settings()
