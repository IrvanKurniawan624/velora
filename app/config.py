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
    allowed_models: str = ""  # Comma-separated list of allowed models from env

    local_model_path: str = "app/models/gemma-2-2b-it-Q8_0.gguf"
    max_runtime_seconds: int = 570  # Global pipeline runtime limit (9.5 minutes)

    @property
    def allowed_models_list(self) -> list[str]:
        return [m.strip() for m in self.allowed_models.split(",") if m.strip()]
