from app.config import Settings


def test_settings_loading() -> None:
    settings = Settings()
    assert settings.fireworks_base_url == "https://api.fireworks.ai/inference/v1"
