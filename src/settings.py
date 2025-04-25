from pydantic import HttpUrl, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings, env_file=".env"):
    evaluation_api_base_url: HttpUrl = HttpUrl(
        "https://agents-course-unit4-scoring.hf.space"
    )

    gemini_api_key: SecretStr
    gemini_model: str = "gemini-2.5-pro-preview-03-25"
