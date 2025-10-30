"""Application configuration using environment variables."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key: str = Field(..., alias="CHATTERBOX_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    elevenlabs_api_key: Optional[str] = Field(default=None, alias="ELEVENLABS_API_KEY")
    data_dir: Path = Field(default=Path("data"), alias="CHATTERBOX_DATA_DIR")
    output_dir: Path = Field(default=Path("output"), alias="CHATTERBOX_OUTPUT_DIR")
    temp_dir: Path = Field(default=Path("api_temp"), alias="CHATTERBOX_TEMP_DIR")
    prepend_phrase: str = Field(default="This is", alias="CHATTERBOX_PREPEND_PHRASE")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir.resolve()

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir.resolve()

    @property
    def resolved_temp_dir(self) -> Path:
        return self.temp_dir.resolve()


settings = Settings()
