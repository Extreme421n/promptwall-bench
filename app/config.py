"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


# Sensible defaults so a developer running a frontend locally on the usual
# Vite / Next / CRA dev-server ports works without any extra env setup.
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]


class Settings(BaseSettings):
    """App settings. Values come from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ----- DB -----
    database_url: str = (
        "postgresql+psycopg://promptwall:promptwall@localhost:5544/promptwall_bench"
    )
    test_database_url: str = "sqlite:///./test.db"

    # ----- LLM -----
    llm_provider: Literal["mock", "openai_compatible"] = "mock"
    default_model: str = "mock-1"
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None

    # ----- HTTP / CORS -----
    # Comma-separated list of allowed origins, e.g.
    #   CORS_ORIGINS=http://localhost:3000,https://demo.example.com
    # Use the literal string ``*`` to allow any origin (development only).
    # ``NoDecode`` tells pydantic-settings *not* to JSON-decode the env value
    # — we want plain comma-separated strings, not a JSON array.
    cors_origins: Annotated[list[str], NoDecode] = _DEFAULT_CORS_ORIGINS

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, v: Any) -> Any:
        """Accept either a list or a comma-separated string from the env."""
        if v is None or v == "":
            return _DEFAULT_CORS_ORIGINS
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


settings = Settings()
