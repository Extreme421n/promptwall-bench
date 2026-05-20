"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


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


settings = Settings()
