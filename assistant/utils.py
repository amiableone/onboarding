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


async def store_files(
        client: AsyncOpenAI,
        *filenames,
        store_name="Info",
):
    """Store files in the VectorStore object (see OpenAI API docs)."""
    vstore = client.beta.vector_stores.create(name=store_name)
    file_paths = []
    for filename in filenames:
        file_paths.append(FILES / filename)
    file_streams = []
    for file_path in file_paths:
        with open(file_path, "rb") as f:
            file_streams.append(f)
    file_batch = await client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vstore.id, files=file_streams
    )
    return file_batch
