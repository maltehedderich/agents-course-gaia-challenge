import asyncio
import logging
import sys
from typing import Any, get_args

from llama_index.utils.workflow import draw_all_possible_flows  # type: ignore

from src.models import EvaluationResponse, Result
from src.services import EvaluationService
from src.settings import GEMINI_MODELS, Settings
from src.tools import get_tools
from src.workflow import QuestionStartEvent, QuestionWorkflow

settings = Settings()  # type: ignore
evaluation_service = EvaluationService(settings.evaluation_api_base_url)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ALLOWED_COMMANDS = ["generate_answers", "submit_answers", "draw_workflow"]

questions = evaluation_service.get_questions()


async def generate_answers() -> None:
    workflow = QuestionWorkflow(
        model=settings.gemini_model,
        tools=get_tools(),
        gemini_api_key=settings.gemini_api_key,
        evaluation_service=evaluation_service,
        timeout=120,
    )

    for question in questions:
        result_file_path = settings.result_path / f"{question.task_id}.json"
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


async def submit_answers() -> None:
    answer_files = list(settings.result_path.glob("*.json"))
    # Check if all answer files are generated
    if len(answer_files) < len(questions):
        log.warning(
            f"Not all answers are generated. Found {len(list(answer_files))} answer files, "
            f"but expected {len(questions)}. Please run the generate_answers command first."
        )
        return

    results = [Result.model_validate_json(file_.read_text()) for file_ in answer_files]
    response: EvaluationResponse = evaluation_service.submit(
        username=settings.huggingface_username,
        agent_code=settings.huggingface_space,
        results=results,
    )
    log.info(
        f"Submission successful.\n"
        f"Score: {response.score}.\n"
        f"Correct Count: {response.correct_count}.\n"
        f"Total Attempted: {response.total_attempted}.\n"
        f"Message: {response.message}.\n"
        f"Timestamp: {response.timestamp}.\n"
    )


async def draw_workflow() -> None:
    draw_all_possible_flows(QuestionWorkflow, "workflow.html")


async def main(command: str, **kwargs: Any) -> None:
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

    if command == "generate_answers":
        await generate_answers()
    elif command == "submit_answers":
        await submit_answers()
    elif command == "draw_workflow":
        await draw_workflow()
    else:
        raise ValueError(
            f"Command {command} is not implemented. Please use one of the following commands: [{', '.join(ALLOWED_COMMANDS)}]"
        )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], **dict(arg.split("=") for arg in sys.argv[2:])))
