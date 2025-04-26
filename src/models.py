from pydantic import BaseModel, ConfigDict, Field


class Question(BaseModel):
    task_id: str
    question: str
    file_name: str
    level: str = Field(..., alias="Level")

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class Result(BaseModel):
    question: Question
    answer: str
