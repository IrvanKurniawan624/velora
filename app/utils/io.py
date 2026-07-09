import json
from pathlib import Path

from app.core.logger import get_logger
from app.schemas.result import Result
from app.schemas.task import Task

logger = get_logger("io_utils")


def load_tasks(file_path: str | Path) -> list[Task]:
    """Load and validate tasks from tasks.json file."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Task file not found at {path}")
        return []

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            tasks = [Task.model_validate(item) for item in data]
        elif isinstance(data, dict) and "tasks" in data:
            tasks = [Task.model_validate(item) for item in data["tasks"]]
        else:
            raise ValueError(
                "Root element must be a list or a dict containing a 'tasks' list."
            )

        logger.info(f"Successfully loaded {len(tasks)} tasks from {path}")
        return tasks
    except Exception as e:
        logger.error(f"Failed to load tasks from {path}: {e}")
        raise


def save_results(file_path: str | Path, results: list[Result]) -> None:
    """Save results to results.json file, validating schema first."""
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Convert results to dict list conforming to the Result schema
        data_to_save = [res.model_dump() for res in results]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved {len(results)} results to {path}")
    except Exception as e:
        logger.error(f"Failed to save results to {path}: {e}")
        raise
