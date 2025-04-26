import logging
from pathlib import Path

import httpx
from pydantic import HttpUrl

from src.models import EvaluationResponse, Question, Result

log = logging.getLogger(__name__)


class EvaluationService:
    def __init__(self, base_url: HttpUrl):
        self.base_url = base_url

    def get_questions(self) -> list[Question]:
        """
        Get the questions from the evaluation service.
        """
        log.info("Fetching questions from the evaluation service")
        url = str(self.base_url) + "questions"
        response = httpx.get(url)
        response.raise_for_status()

        return [Question.model_validate(question) for question in response.json()]

    def get_file(self, question: Question, file_path: Path) -> None:
        """
        Get the file from the evaluation service.
        """
        assert question.file_name, "Question does not have a file attached"
        log.info(f"Downloading file {question.file_name} for task {question.task_id}")

        url = str(self.base_url) + "files/" + question.task_id
        response = httpx.get(url)
        response.raise_for_status()

        # Save the file to the specified path
        file_path.write_bytes(response.content)

    def submit(
        self, username: str, agent_code: HttpUrl, results: list[Result]
    ) -> EvaluationResponse:
        """
        Submit the results to the evaluation service.
        """
        log.info("Submitting results to the evaluation service")
        url = str(self.base_url) + "submit"
        payload = {
            "username": username,
            "agent_code": str(agent_code),
            "answers": [result.get_answer() for result in results],
        }
        response = httpx.post(url, json=payload)
        response.raise_for_status()

        log.info("Results submitted successfully")
        return EvaluationResponse.model_validate(response.json())
