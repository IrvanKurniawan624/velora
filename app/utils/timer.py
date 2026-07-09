import asyncio
import time

from app.core.logger import get_logger

logger = get_logger("timer")


class TimerGuard:
    def __init__(self, max_total_seconds: float = 570.0) -> None:
        self.max_total_seconds = max_total_seconds
        self.start_time = time.time()
        logger.info(
            f"TimerGuard initialized with {max_total_seconds} seconds total budget."
        )

    def elapsed(self) -> float:
        """Returns elapsed time in seconds since initialization."""
        return time.time() - self.start_time

    def remaining(self) -> float:
        """Returns remaining budget in seconds."""
        rem = self.max_total_seconds - self.elapsed()
        return max(0.0, rem)

    def is_exceeded(self) -> bool:
        """Checks if the global budget has been exceeded."""
        return self.elapsed() >= self.max_total_seconds

    async def run_with_timeout(self, coro, task_timeout: float | None = None):
        """Run an async coroutine with a timeout.

        The timeout is the minimum of the task_timeout (if provided) and the remaining global time budget.
        """
        remaining_time = self.remaining()
        actual_timeout = (
            min(task_timeout, remaining_time)
            if task_timeout is not None
            else remaining_time
        )

        if actual_timeout <= 0:
            raise TimeoutError("No time remaining in the global pipeline budget.")

        try:
            return await asyncio.wait_for(coro, timeout=actual_timeout)
        except TimeoutError as te:
            logger.error(
                f"Task timed out after {actual_timeout:.2f}s (global limit check)."
            )
            raise TimeoutError(f"Task timed out after {actual_timeout:.2f}s.") from te
