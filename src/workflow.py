from pathlib import Path
from typing import Any

from google import genai
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from pydantic import SecretStr

from src.services.evaluation_service import EvaluationService, Question


class QuestionStartEvent(StartEvent):
    question: Question


class DownloadFileEvent(Event):
    pass


class LanguageModelEvent(Event):
    file_path: Path | None = None


class QuestionWorkflow(Workflow):
    def __init__(
        self,
        *args: Any,
        model: str,
        gemini_api_key: SecretStr,
        evaluation_service: EvaluationService,
        data_path: Path = Path("data"),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.evaluation_service = evaluation_service
        self.gemini_client = genai.Client(api_key=gemini_api_key.get_secret_value())

        # Create the data path if it doesn't exist
        data_path.mkdir(parents=True, exist_ok=True)
        self.data_path = data_path

    @step
    async def start(
        self,
        context: Context,
        event: QuestionStartEvent,
    ) -> LanguageModelEvent | DownloadFileEvent:
        """
        Start the workflow with a question.
        """
        # Store the question in the context
        await context.set("question", event.question)

        # Check if the question has a file name
        return DownloadFileEvent() if event.question.file_name else LanguageModelEvent()

    @step
    async def download_file(
        self,
        context: Context,
        event: DownloadFileEvent,
    ) -> LanguageModelEvent:
        """
        Download the file from the evaluation service.
        """
        question = await context.get("question")
        assert isinstance(question, Question), "`question` not found in context"

        # Download the file
        file_path = self.data_path / question.file_name
        self.evaluation_service.get_file(question, file_path)

        return LanguageModelEvent(file_path=file_path)

    @step
    async def call_language_model(
        self,
        context: Context,
        _: LanguageModelEvent,
    ) -> StopEvent:
        """
        Call the language model with the question and file.
        """
        question = await context.get("question")
        assert isinstance(question, Question), "`question` not found in context"

        # Call the language model with the question and file
        print(f"Calling language model {self.model} with question: {question.question}")

        return StopEvent()
