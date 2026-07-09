import asyncio

from app.config import Settings
from app.clients.local_client import LocalClient
from app.clients.fireworks_client import FireworksClient
from app.services.classifier import ClassifierService
from app.services.agent import AgentService


MODEL_CLASSIFIER = "___ISI_MODEL_CLASSIFIER_DISINI___"
MODEL_AGENT = "___ISI_MODEL_AGENT_DISINI___"


async def main() -> None:
    settings = Settings()

    local = LocalClient()
    fireworks = FireworksClient(
        api_key=settings.fireworks_api_key,
        base_url=settings.fireworks_base_url,
    )

    classifier = ClassifierService(client=fireworks, model=MODEL_CLASSIFIER)
    agent = AgentService(local_client=local, fireworks_client=fireworks)


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
