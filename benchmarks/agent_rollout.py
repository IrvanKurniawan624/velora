import asyncio
import os
import time
from eval_protocol.pytest.rollout_processor import RolloutProcessor
from eval_protocol.models import EvaluationRow, Message
from app.services.agent import AgentService
from app.clients.local_client import LocalClient
from app.clients.fireworks_client import FireworksClient

class AgentPipelineRolloutProcessor(RolloutProcessor):
    def setup(self) -> None:
        self.local_client = LocalClient()
        self.fireworks_client = FireworksClient()
        self.agent = AgentService(
            local_client=self.local_client,
            fireworks_client=self.fireworks_client
        )

    def __call__(self, rows: list[EvaluationRow], config) -> list[asyncio.Task[EvaluationRow]]:
        tasks = []
        for row in rows:
            tasks.append(asyncio.create_task(self.process_row(row)))
        return tasks

    async def process_row(self, row: EvaluationRow) -> EvaluationRow:
        start_time = time.perf_counter()
        
        # Get the query (the last user message)
        user_messages = row.get_user_messages()
        if not user_messages:
            raise ValueError("No user messages found in evaluation row.")
        
        user_query = user_messages[-1].content
        
        # Run our Agent Pipeline
        response_content = await self.agent.run(user_query)
        
        # Append the assistant message
        row.messages = list(row.messages) + [
            Message(
                role="assistant",
                content=response_content
            )
        ]
        
        row.execution_metadata.rollout_duration_seconds = time.perf_counter() - start_time
        return row
