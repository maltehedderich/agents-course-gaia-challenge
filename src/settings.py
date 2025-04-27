from pathlib import Path
from typing import Literal

from pydantic import HttpUrl, SecretStr
from pydantic_settings import BaseSettings

GEMINI_MODELS = Literal[
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-03-25",
]


class Settings(BaseSettings, env_file=".env"):
    evaluation_api_base_url: HttpUrl = HttpUrl(
        "https://agents-course-unit4-scoring.hf.space"
    )

    gemini_api_key: SecretStr
    gemini_model: GEMINI_MODELS = "gemini-2.0-flash"

    huggingface_username: str = "hedderich"
    huggingface_space: HttpUrl = HttpUrl(
        "https://huggingface.co/spaces/hedderich/agents-course-gaia-challenge/tree/main"
    )

    @property
    def result_path(self) -> Path:
        path = Path("results") / self.gemini_model
        path.mkdir(parents=True, exist_ok=True)
        return path
