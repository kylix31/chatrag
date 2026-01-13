"""
Infrastructure Layer - Configuration
Manages application settings using Pydantic Settings
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    Principle: Separation of Concerns - configuration isolated from business logic
    """

    openai_api_key: str = Field(...)
    openai_embedding_model: str = "text-embedding-3-large"
    openai_chat_model: str = "gpt-4o-mini"

    azure_search_endpoint: str = Field(...)
    azure_search_key: str = Field(...)
    azure_search_index_name: str = Field(...)

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    max_clarifications: int = 2

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )


# Singleton pattern for settings
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Factory method to get singleton instance of settings
    Pattern: Singleton
    """
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings
