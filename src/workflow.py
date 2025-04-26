import logging
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore
from google import genai
from google.genai.types import (
    Content,
    FunctionCall,
    GenerateContentConfig,
    Part,
)
from google.genai.types import Tool as GenaiTool
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from pydantic import SecretStr

from src.models import Result
from src.services.evaluation_service import EvaluationService, Question
from src.tools import Tool

log = logging.getLogger(__name__)


class QuestionStartEvent(StartEvent):
    question: Question


class DownloadFileEvent(Event):
    pass


class UploadFileEvent(Event):
    file_path: Path


class LanguageModelEvent(Event):
    pass


class FunctionCallEvent(Event):
    function_calls: list[FunctionCall]


class ExtractAnswerEvent(Event):
    pass


class QuestionWorkflow(Workflow):
    def __init__(
        self,
        *args: Any,
        model: str,
        tools: list[Tool],
        gemini_api_key: SecretStr,
        evaluation_service: EvaluationService,
        data_path: Path = Path("data"),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.tools = tools
        self.evaluation_service = evaluation_service
        self.gemini_client = genai.Client(api_key=gemini_api_key.get_secret_value())
        self.gemini_config = GenerateContentConfig(
            temperature=0.0,
            tools=[
                GenaiTool(
                    function_declarations=[tool.function_declaration],
                )  # type: ignore
                for tool in tools
            ],
        )

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
        log.info(f"Starting workflow for question: {event.question.question}")

        # Store the question in the context
        await context.set("question", event.question)

        # Add initial content for the language model
        await context.set(
            "contents",
            [Content(role="user", parts=[Part(text=event.question.question)])],
        )

        # Check if the question has a file name
        return DownloadFileEvent() if event.question.file_name else LanguageModelEvent()

    @step
    async def download_file(
        self,
        context: Context,
        _: DownloadFileEvent,
    ) -> UploadFileEvent:
        """
        Download the file from the evaluation service.
        """
        question = await context.get("question")
        assert isinstance(question, Question), "`question` not found in context"

        # Download the file
        file_path = self.data_path / question.file_name
        self.evaluation_service.get_file(question, file_path)

        return UploadFileEvent(file_path=file_path)

    @step
    async def upload_file(
        self,
        context: Context,
        event: UploadFileEvent,
    ) -> LanguageModelEvent:
        """
        Upload the file to the language model.
        """
        contents = await context.get("contents")
        assert isinstance(contents, list), "`question` not found in context"

        # Handle Excel files differently due to compatibility issues with gemini
        if event.file_path.suffix in [".xls", ".xlsx"]:
            log.info(f"Reading Excel file {event.file_path}")
            df = pd.read_excel(event.file_path)
            contents.append(
                Content(
                    role="user",
                    parts=[
                        Part(text=f"<excel-content>{df.to_markdown()}</excel-content>"),
                    ],
                )
            )
        else:
            # Upload the file to gemini
            log.info(f"Uploading file {event.file_path} to Gemini")
            file_ = await self.gemini_client.aio.files.upload(file=event.file_path)

            # Add the file to the contents
            contents.append(file_)
        await context.set("contents", contents)

        return LanguageModelEvent()

    @step
    async def call_language_model(
        self,
        context: Context,
        _: LanguageModelEvent,
    ) -> FunctionCallEvent | ExtractAnswerEvent:
        """
        Call the language model with the question and file.
        """
        question = await context.get("question")
        contents = await context.get("contents")
        assert isinstance(question, Question), "`question` not found in context"
        assert isinstance(contents, list), "`contents` not found in context"

        # Call the language model
        response = await self.gemini_client.aio.models.generate_content(
            model=self.model,
            contents=contents,  # type: ignore
            config=self.gemini_config,
        )

        # Check if the response contains function calls
        if response.function_calls:
            return FunctionCallEvent(function_calls=response.function_calls)

        # If the response contains no function calls, append the final message and extract the answer
        assert response.text, "Response text is empty"
        contents.append(
            Content(
                role="model",
                parts=[Part(text=response.text)],
            )
        )
        await context.set("contents", contents)

        # If no function calls, extract the answer directly
        return ExtractAnswerEvent()

    @step
    async def call_function(
        self,
        context: Context,
        event: FunctionCallEvent,
    ) -> LanguageModelEvent:
        """
        Call the function with the function calls from the language model.
        """
        contents = await context.get("contents")
        assert isinstance(contents, list), "`contents` not found in context"

        # Execute all function calls
        for function_call in event.function_calls:
            log.info(f"Calling function: {function_call.name}")
            # Add the function call to the contents
            contents.append(
                Content(role="model", parts=[Part(function_call=function_call)])
            )

            # Execute the function
            tool: Tool | None = next(
                filter(lambda tool: tool.name == function_call.name, self.tools)
            )
            assert tool, f"Tool {function_call.name} not found"
            response = await tool.function(
                **function_call.args if function_call.args else {}
            )

            # Add the result to the contents
            contents.append(
                Content(
                    role="user",
                    parts=[
                        Part.from_function_response(
                            name=tool.name, response={"result": response}
                        )
                    ],
                )
            )

        # Return the result
        return LanguageModelEvent()

    @step
    async def extract_answer(
        self,
        context: Context,
        _: ExtractAnswerEvent,
    ) -> StopEvent:
        """
        Extract the answer from the language model response.
        """
        question = await context.get("question")
        contents = await context.get("contents")
        assert isinstance(question, Question), "`question` not found in context"
        assert isinstance(contents, list), "`contents` not found in context"

        contents.append(
            Content(
                role="user",
                parts=[
                    Part(
                        text=(
                            "Extract the answer to the initial question from the content above. "
                            "Answer purely based on the content above, without any additional information.\n"
                            "IMPORTANT: ONLY RETURN THE ANSWER, WITHOUT ANY ADDITIONAL TEXT.\n"
                            f"\n\nQuestion: {question.question}\n\n"
                            "Answer: "
                        )
                    )
                ],
            )
        )

        # Get the last content
        response = await self.gemini_client.aio.models.generate_content(
            model=self.model,
            contents=contents,  # type: ignore
            config=GenerateContentConfig(temperature=0.0),
        )

        # Return the result
        assert response.text, "Response text is empty"
        return StopEvent(result=Result(question=question, answer=response.text.strip()))
