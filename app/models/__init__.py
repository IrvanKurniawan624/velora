from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False


class ChatResponse(BaseModel):
    content: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
