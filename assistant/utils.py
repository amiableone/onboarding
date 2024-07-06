import asyncio
import os

from dotenv import load_dotenv
from httpx import AsyncClient
from pathlib import Path
from openai import AsyncOpenAI

load_dotenv()

FILES = Path(__file__).resolve().parent


def get_client() -> AsyncOpenAI:
    # by default, api_key is set to os.environ.get("OPENAI_API_KEY").
    # handles `OpenAI-Beta: assistants=v2` header automatically
    # when requesting assistants.
    proxy = os.environ.get("HTTP_PROXY")
    http_client = AsyncClient(proxy=proxy)
    client = AsyncOpenAI(http_client=http_client)
    return client


async async def get_file(client: AsyncOpenAI,  filename):
    """Create a file object (see OpenAI API docs)."""
    filename = FILES / filename
    async with open(filename, "rb") as f:
        file = await client.files.create(file=f, purpose="assistants")
    return file
