from pydantic import BaseModel, Field


class Task(BaseModel):
    task_id: str = Field(alias="id")
    prompt: str

    model_config = {
        "populate_by_name": True,
    }
