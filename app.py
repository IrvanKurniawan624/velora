import asyncio

from app.clients.fireworks_client import FireworksClient
from app.clients.local_client import LocalClient
from app.config import Settings
from app.services.agent import AgentService
from app.services.classifier import ClassifierService

MODEL_CLASSIFIER = ""
MODEL_AGENT = ""


async def main() -> None:
    settings = Settings()

    local = LocalClient()
    fireworks = FireworksClient(
        api_key=settings.fireworks_api_key,
        base_url=settings.fireworks_base_url,
    )

    classifier = ClassifierService(client=fireworks, model=MODEL_CLASSIFIER)  # noqa: F841
    agent = AgentService(local_client=local, fireworks_client=fireworks)  # noqa: F841


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
