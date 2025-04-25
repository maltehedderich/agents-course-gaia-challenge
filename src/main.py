import asyncio

from src.services import EvaluationService
from src.settings import Settings
from src.workflow import QuestionStartEvent, QuestionWorkflow

settings = Settings()  # type: ignore
evaluation_service = EvaluationService(settings.evaluation_api_base_url)


async def main():
    questions = evaluation_service.get_questions()
    workflow = QuestionWorkflow(
        model=settings.gemini_model,
        gemini_api_key=settings.gemini_api_key,
        evaluation_service=evaluation_service,
    )
    for question in questions:
        await workflow.run(
            start_event=QuestionStartEvent(
                question=question,
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
