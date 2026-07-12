from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    fireworks_api_key: SecretStr | None = None
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    allowed_models: str = ""
    local_model_path: str = "models/model.gguf"
    global_timeout_seconds: int = 600
    request_timeout_seconds: int = 30
    use_local_cascade: bool = False

