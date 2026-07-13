"""Lightweight data models used by the agent clients and orchestrator."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatResponse:
    """A single response from a local or remote model."""
    content: str
    model: str = "local"
    confidence: float = 1.0
    remote_tokens_used: int = 0


@dataclass
class TaskItem:
    """Internal representation of one task after classification."""
    task_id: str
    prompt: str
    category: str


@dataclass
class TaskResult:
    """Final answer to be written to results.json."""
    task_id: str
    answer: str
    confidence: float = 0.0
    category: str = ""
