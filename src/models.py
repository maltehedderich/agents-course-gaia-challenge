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

    def get_answer(self) -> dict[str, str]:
        return {"task_id": self.question.task_id, "submitted_answer": self.answer}


class EvaluationResponse(BaseModel):
    username: str
    score: int
    correct_count: int
    total_attempted: int
    message: str
    timestamp: str

    model_config = ConfigDict(from_attributes=True)
