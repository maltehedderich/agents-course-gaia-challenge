import logging
from typing import Callable

import pandas as pd  # type: ignore
from google import genai
from google.genai.types import (
    Content,
    FileData,
    FunctionDeclaration,
    GenerateContentConfig,
    GoogleSearch,
    Part,
)
from google.genai.types import Tool as GenaiTool
from mediawikiapi import MediaWikiAPI  # type: ignore
from pydantic import BaseModel

from src.settings import Settings

settings = Settings()  # type: ignore
log = logging.getLogger(__name__)


class Tool(BaseModel):
    function: Callable
    name: str
    description: str
    function_declaration: FunctionDeclaration

    @classmethod
    def from_function(cls, function: Callable) -> "Tool":
        """
        Create a Tool instance from a function.
        """
        assert callable(function), "Function must be callable"
        assert function.__name__, "Function must have a name"
        assert function.__doc__, "Function must have a docstring"

        return cls.model_validate(
            {
                "function": function,
                "name": function.__name__,
                "description": function.__doc__,
                "function_declaration": FunctionDeclaration.from_callable_with_api_option(
                    callable=function
                ),
            }
        )


# Add Wikipedia Tool
async def wikipedia_search(wikipedia_title: str) -> str:
    """
    Search Wikipedia for a given title and return the content of the top result.
    Use this tool whenever you need information from Wikipedia.

    Parameters
    ----------
    wikipedia_title : str
        The title to search for on Wikipedia.

    Returns
    -------
    str
        Wikipedia content of the top result.
    """
    log.info(f"Searching Wikipedia for: {wikipedia_title}")
    wikipedia = MediaWikiAPI()
    titles = wikipedia.search(wikipedia_title, results=1)
    if not titles:
        return "No results found."

    output_template = "{title}\n```{content}\n```\n"

    # Get the first result
    page = wikipedia.page(titles[0])
    tables = pd.read_html(page.url)
    content = (
        page.content
        + "\n\nTables:\n"
        + "\n".join(table.to_markdown() for table in tables)
    )

    return output_template.format(title=page.title, content=content)


# Add YouTube Tool
async def youtube_search(question: str, video_url: str) -> str | None:
    """
    Answer a question based on the content of a YouTube video.

    Parameters
    ----------
    question : str
        The question to answer.
    video_url : str
        The URL of the YouTube video. This needs to be a valid YouTube URL.
        The URL should be in the format of "https://www.youtube.com/watch?v=VIDEO_ID".

    Returns
    -------
    str
        The answer to the question based on the content of the YouTube video.
    """
    assert video_url.startswith("https://www.youtube.com/watch?v="), (
        "Invalid YouTube URL"
    )

    log.info(f"Answering question: {question} based on video: {video_url}")
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    config = GenerateContentConfig(
        temperature=0.0,
    )
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=Content(
            role="user",
            parts=[
                Part(
                    file_data=FileData(file_uri=video_url),
                ),
                Part(
                    text=f"Based on the content of this YouTube video, answer the following question:\n\n{question}"
                ),
            ],
        ),
        config=config,
    )
    return response.text


async def google_search(question: str) -> str | None:
    """
    Search Google for a given question and return a concise answer based on the top search results.
    Use this tool when you need up-to-date or general information from the web, such as news, facts, or broad topics.

    Parameters
    ----------
    question : str
        The question you want answered.

    Returns
    -------
    str
        A concise answer based on the most relevant web results.
    """
    log.info(f"Searching Google for: {question}")
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    config = GenerateContentConfig(
        temperature=0.0,
        tools=[
            GenaiTool(google_search=GoogleSearch()),
        ],
    )
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=question,
        config=config,
    )
    return response.text


async def decode_text(text: str) -> str:
    """
    Decode the given text and return the decoded text. Use this tool when you need to decode a text that has been
    encoded, obfuscated, or altered in any way that makes it not immediately human-readable. This includes texts that
    have been reversed, encoded using a specific algorithm or format, or otherwise transformed.

    Use this tool whenever you encounter a text that is not in a standard readable format, such as reversed text or
    text encoded in another way.

    Parameters
    ----------
    text : str
        The text to decode.

    Returns
    -------
    str
        The decoded text.
    """
    log.info(f"Decoding text: {text}")
    client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
    config = GenerateContentConfig(
        temperature=0.0,
    )
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=f"Decode the following text, ONLY respond with the decoded text:\n\n{text}",
        config=config,
    )
    assert response.text, "No text returned from decoding"
    return response.text


def get_tools() -> list[Tool]:
    """
    Get the list of tools.
    """
    return [
        Tool.from_function(wikipedia_search),
        Tool.from_function(youtube_search),
        Tool.from_function(google_search),
        Tool.from_function(decode_text),
    ]
