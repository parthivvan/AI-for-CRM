from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    ai_service_api_key: str = "dev-ai-service-key"
    pilot_branch_id: str = "pilot-branch-001"
    log_level: str = "INFO"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-1.5-flash"

    # Local model (ONNX) — primary vision provider
    vision_provider: str = "local_model"  # local_model | gemini | deterministic
    enable_gemini_fallback: bool = True
    model_path: str = "models/current/mobilenetv3_large_skin_ai.onnx"
    model_labels_path: str = "models/current/labels.json"
    model_version: str = "genx-vision-v0.1"
    model_threshold: float = 0.65

    crm_base_url: str | None = None
    crm_api_key: str | None = None
    crm_timeout_seconds: float = 15.0
    push_analysis_to_crm: bool = False

    database_url: str = Field(default="sqlite:///./genx_ai.db")

    enable_scheduler: bool = False
    recommendation_cron_hour: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()

