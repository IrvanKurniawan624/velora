from pathlib import Path

import pytest

from app.clients.base_client import BaseClient
from app.core.constants import Category
from app.models import ChatRequest, ChatResponse
from app.schemas.result import Result
from app.services.classifier import ClassifierService
from app.services.router import RouterService
from app.services.self_check import SelfCheckService
from app.utils.io import load_tasks, save_results


class MockClient(BaseClient):
    def __init__(self, response_text: str = "") -> None:
        self.response_text = response_text
        self.last_request = None

    async def chat_complete(self, request: ChatRequest) -> ChatResponse:
        self.last_request = request
        return ChatResponse(
            content=self.response_text,
            model=request.model,
            finish_reason="stop",
        )


def test_router_logic() -> None:
    router = RouterService()
    # Low-risk -> local
    assert router.route(Category.SENTIMENT) == "local"
    assert router.route(Category.NER) == "local"
    assert router.route(Category.FACTUAL) == "local"
    assert router.route(Category.SUMMARIZE) == "local"

    # High-risk -> fireworks
    assert router.route(Category.MATH) == "fireworks"
    assert router.route(Category.LOGIC) == "fireworks"
    assert router.route(Category.CODE_DEBUG) == "fireworks"
    assert router.route(Category.CODE_GEN) == "fireworks"


@pytest.mark.anyio
async def test_classifier_rules() -> None:
    mock_client = MockClient("factual")
    classifier = ClassifierService(client=mock_client)

    # Rule-based matches
    assert (
        await classifier.classify("What is the sentiment of this text?")
        == Category.SENTIMENT
    )
    assert (
        await classifier.classify("Extract named entities from here.") == Category.NER
    )
    assert (
        await classifier.classify("Summarize the following document.")
        == Category.SUMMARIZE
    )
    assert (
        await classifier.classify("Fix the bug in this python code.")
        == Category.CODE_DEBUG
    )
    assert (
        await classifier.classify("Write a python script to parse logs.")
        == Category.CODE_GEN
    )
    assert await classifier.classify("Solve the math equation.") == Category.MATH
    assert await classifier.classify("Solve this riddle.") == Category.LOGIC

    # Fallback to LLM
    mock_client.response_text = "math"
    assert (
        await classifier.classify("An ambiguous query with no clear words.")
        == Category.MATH
    )
    assert mock_client.last_request is not None
    assert "You are a query classifier" in mock_client.last_request.messages[0].content


def test_self_check_validation() -> None:
    checker = SelfCheckService()

    # Refusals
    assert not checker.check(
        "test prompt", "I am unable to answer this question.", Category.FACTUAL
    )
    assert not checker.check("test prompt", "I don't know.", Category.FACTUAL)
    assert not checker.check("test prompt", "", Category.FACTUAL)

    # Valid factual
    assert checker.check(
        "What is python?", "Python is a programming language.", Category.FACTUAL
    )

    # Sentiment checks
    assert checker.check(
        "sentiment check", "The sentiment is positive", Category.SENTIMENT
    )
    # Invalid sentiment format
    assert not checker.check(
        "sentiment check",
        "This is just text explaining nothing about labels",
        Category.SENTIMENT,
    )


def test_io_utilities(tmp_path: Path) -> None:
    tasks_file = tmp_path / "tasks.json"
    results_file = tmp_path / "results.json"

    # Write dummy tasks
    tasks_content = """
    [
        {"id": "t1", "prompt": "Sentiment of: I love it!"},
        {"id": "t2", "prompt": "What is 2+2?"}
    ]
    """
    tasks_file.write_text(tasks_content, encoding="utf-8")

    # Load tasks
    loaded_tasks = load_tasks(tasks_file)
    assert len(loaded_tasks) == 2
    assert loaded_tasks[0].task_id == "t1"
    assert loaded_tasks[1].prompt == "What is 2+2?"

    # Save results
    results_to_save = [
        Result(task_id="t1", answer="positive"),
        Result(task_id="t2", answer="4"),
    ]
    save_results(results_file, results_to_save)

    # Read back results to verify
    assert results_file.exists()
    import json

    with open(results_file) as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["task_id"] == "t1"
    assert data[0]["answer"] == "positive"


def test_advanced_routing() -> None:
    router = RouterService()

    # 1. Low complexity heavy category (e.g., MATH, very short query)
    features_easy_math = {
        "word_count": 5,
        "math_score": 0.0,
        "code_tokens": 0,
        "instruction_count": 0,
        "constraint_count": 0,
        "entity_density": 0.0,
        "conditionals": 0,
        "ambiguities": 0,
    }
    # Should route to local because C < 0.18 and length < 15
    assert router.route(Category.MATH, features_easy_math) == "local"

    # 2. High complexity heavy category (long or many operators)
    features_hard_math = {
        "word_count": 20,
        "math_score": 3.0,
        "code_tokens": 0,
        "instruction_count": 2,
        "constraint_count": 1,
        "entity_density": 0.0,
        "conditionals": 1,
        "ambiguities": 0,
    }
    # Should route to fireworks
    assert router.route(Category.MATH, features_hard_math) == "fireworks"

    # 3. Light category (e.g., FACTUAL) but exceeding length threshold (> 220 words)
    features_long_factual = {
        "word_count": 250,
        "math_score": 0.0,
        "code_tokens": 0,
        "instruction_count": 1,
        "constraint_count": 0,
        "entity_density": 0.0,
        "conditionals": 0,
        "ambiguities": 0,
    }
    assert router.route(Category.FACTUAL, features_long_factual) == "fireworks"

    # 4. Light category with constraint count >= 2
    features_constrained = {
        "word_count": 30,
        "math_score": 0.0,
        "code_tokens": 0,
        "instruction_count": 1,
        "constraint_count": 2,
        "entity_density": 0.0,
        "conditionals": 0,
        "ambiguities": 0,
    }
    assert router.route(Category.SUMMARIZE, features_constrained) == "fireworks"

    # 5. Light category with ambiguity
    features_ambiguous = {
        "word_count": 10,
        "math_score": 0.0,
        "code_tokens": 0,
        "instruction_count": 0,
        "constraint_count": 0,
        "entity_density": 0.0,
        "conditionals": 0,
        "ambiguities": 1,
    }
    assert router.route(Category.SENTIMENT, features_ambiguous) == "fireworks"
