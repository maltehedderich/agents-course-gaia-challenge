from pathlib import Path

import httpx
from pydantic import HttpUrl

from src.models import Question


class EvaluationService:
    def __init__(self, base_url: HttpUrl):
        self.base_url = base_url

    def get_questions(self) -> list[Question]:
        """
        Get the questions from the evaluation service.
        """
        url = str(self.base_url) + "questions"
        response = httpx.get(url)
        response.raise_for_status()

        return [Question.model_validate(question) for question in response.json()]

    def get_file(self, question: Question, file_path: Path) -> None:
        """
        Get the file from the evaluation service.
        """
        assert question.file_name, "Question does not have a file attached"

        url = str(self.base_url) + "files/" + question.task_id
        response = httpx.get(url)
        response.raise_for_status()

        # Save the file to the specified path
        file_path.write_bytes(response.content)
