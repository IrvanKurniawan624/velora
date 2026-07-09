from abc import ABC, abstractmethod

from app.models import ChatRequest, ChatResponse


class BaseClient(ABC):
    @abstractmethod
    async def chat_complete(self, request: ChatRequest) -> ChatResponse:
        pass
