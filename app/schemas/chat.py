from pydantic import BaseModel

class ChatResponse(BaseModel):
    content: str
    model: str
    confidence: float = 1.0
    remote_tokens_used: int = 0
