from typing import Literal

from pydantic import HttpUrl, SecretStr
from pydantic_settings import BaseSettings

GEMINI_MODELS = Literal["gemini-2.0-flash", "gemini-2.5-pro-preview-03-25"]


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
