from pydantic import BaseModel


class Result(BaseModel):
    task_id: str
    answer: str
