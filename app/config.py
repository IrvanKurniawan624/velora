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
    ep_completion_params: str = '[{"model": "<Your model here>"}]'

    fireworks_model: str = "accounts/fireworks/models/qwen3-30b-a3b"

    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model_3b: str = "qwen3-3b"
    lmstudio_model_7b: str = "qwen3-7b"

    # Router tuning parameters
    confidence_threshold: float = 0.7
    jit_cost_lambda: float = 0.3


settings = Settings()
