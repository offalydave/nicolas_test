from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    # Platform database
    neon_platform_connection_string: str = ""
    # Data pipeline database
    neon_data_pipeline_connection_string: str = ""

    @field_validator("neon_platform_connection_string", "neon_data_pipeline_connection_string")
    @classmethod
    def validate_connection_string(cls, v: str) -> str:
        if not v:
            raise ValueError("Connection string is required")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",           # Use .env file for local development.
        env_file_encoding="utf-8"  # Ensure correct encoding.
    )