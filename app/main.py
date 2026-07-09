import asyncio
import sys
from pathlib import Path
from app.config import Settings
from app.clients.local_client import LocalClient
from app.clients.fireworks_client import FireworksClient
from app.services.agent import AgentService
from app.schemas.result import Result
from app.utils.io import load_tasks, save_results
from app.utils.timer import TimerGuard
from app.core.logger import get_logger

logger = get_logger("main")


async def run_pipeline() -> None:
    settings = Settings()

    # 1. Initialize Timer
    timer = TimerGuard(max_total_seconds=settings.max_runtime_seconds)

    # 2. Paths
    input_path = Path("input/tasks.json")
    output_path = Path("output/results.json")

    logger.info(f"Starting pipeline. Input: {input_path}, Output: {output_path}")

    # 3. Load tasks
    try:
        tasks = load_tasks(input_path)
    except Exception as e:
        logger.error(f"Failed to load tasks, exiting. Error: {e}")
        sys.exit(1)

    if not tasks:
        logger.warning("No tasks found to process. Exiting.")
        save_results(output_path, [])
        sys.exit(0)

    # 4. Initialize Clients (once at startup to prevent load overhead)
    logger.info("Initializing inference clients...")
    try:
        local_client = LocalClient()
        fireworks_client = FireworksClient()
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        sys.exit(1)

    agent = AgentService(local_client=local_client, fireworks_client=fireworks_client)

    results = []

    # 5. Process tasks
    for idx, task in enumerate(tasks):
        logger.info(f"Processing task {idx + 1}/{len(tasks)}: id={task.task_id}")

        # Check global timer
        if timer.is_exceeded():
            logger.error("Global pipeline time limit exceeded! Stopping execution.")
            break

        try:
            # Run with task-level timeout
            answer = await timer.run_with_timeout(
                agent.run(task.prompt),
                task_timeout=settings.max_task_runtime_seconds,
            )
            results.append(Result(task_id=task.task_id, answer=answer))
        except TimeoutError as te:
            logger.error(f"Task {task.task_id} timed out: {te}. Saving fallback error.")
            results.append(Result(task_id=task.task_id, answer="Error: Task execution timed out."))
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}. Saving fallback error.")
            results.append(Result(task_id=task.task_id, answer="Error: Internal system failure."))

    # 6. Save results
    logger.info("Task processing completed. Saving results...")
    try:
        save_results(output_path, results)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        sys.exit(1)

    logger.info(f"Pipeline finished successfully. Total elapsed time: {timer.elapsed():.2f}s")


def main() -> None:
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Pipeline crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
