import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, get_args

from src.services import EvaluationService
from src.settings import GEMINI_MODELS, Settings
from src.tools import get_tools
from src.workflow import QuestionStartEvent, QuestionWorkflow

settings = Settings()  # type: ignore
evaluation_service = EvaluationService(settings.evaluation_api_base_url)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ALLOWED_COMMANDS = ["generate_answers"]


async def generate_answers() -> None:
    questions = evaluation_service.get_questions()
    workflow = QuestionWorkflow(
        model=settings.gemini_model,
        tools=get_tools(),
        gemini_api_key=settings.gemini_api_key,
        evaluation_service=evaluation_service,
        timeout=120,
    )
    result_path = Path("results") / settings.gemini_model
    result_path.mkdir(parents=True, exist_ok=True)
    for question in questions:
        result_file_path = result_path / f"{question.task_id}.json"
        if result_file_path.exists():
            log.info(
                f"Result file already exists for task {question.task_id}. Skipping."
            )
            continue

        log.info(f"Processing Task: {question.task_id}")
        result = await workflow.run(
            start_event=QuestionStartEvent(
                question=question,
            )
        )

        result_file_path.write_text(result.model_dump_json(indent=4))


async def main(command: str, **kwargs: Any) -> None:
    if command not in ALLOWED_COMMANDS:
        print(
            f"Please provide a valid command from the following commands: [{', '.join(ALLOWED_COMMANDS)}], e.g. `python -m src.main {ALLOWED_COMMANDS[0]}`"
        )
        sys.exit(1)

    print(kwargs)
    if command == "generate_answers":
        # Use the provided model if specified
        if "--model" in kwargs:
            model = kwargs["--model"]
            if model not in get_args(GEMINI_MODELS):
                print(
                    "Please provide a valid model from the following models: "
                    f"[{', '.join(get_args(GEMINI_MODELS))}], e.g. `python -m src.main {command}"
                    f" --model={get_args(GEMINI_MODELS)[0]}`"
                )
                sys.exit(1)
            settings.gemini_model = model
        await generate_answers()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], **dict(arg.split("=") for arg in sys.argv[2:])))
