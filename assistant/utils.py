import asyncio
import os

from dotenv import load_dotenv
from httpx import AsyncClient
from pathlib import Path
from openai import AsyncOpenAI
from openai.types import FileObject
from openai.types.beta import Assistant
from openai.types.beta.vector_stores import VectorStoreFileBatch

load_dotenv()

FILES = Path(__file__).resolve().parent
FILE_SEARCH_TOOL = {"type": "file_search"}


def get_client() -> AsyncOpenAI:
    # by default, api_key is set to os.environ.get("OPENAI_API_KEY").
    # handles `OpenAI-Beta: assistants=v2` header automatically
    # when requesting assistants.
    proxy = os.environ.get("HTTP_PROXY")
    http_client = AsyncClient(proxy=proxy)
    client = AsyncOpenAI(http_client=http_client)
    return client


async def get_file(client: AsyncOpenAI,  filename) -> FileObject:
    """Create a file object (see OpenAI API docs)."""
    filename = FILES / filename
    async with open(filename, "rb") as f:
        file = await client.files.create(file=f, purpose="assistants")
    return file


async def store_files(
        client: AsyncOpenAI,
        *filenames,
        store_name="Info",
) -> VectorStoreFileBatch:
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


async def create_assistant(
        client: AsyncOpenAI,
        vector_store_id: str,
        instructions: str | None = None,
        model="gpt-4o",
) -> Assistant:
    # only the file_search tool is supported for now.
    tools = [FILE_SEARCH_TOOL],
    tool_resources = {
        "file_search": {
            "vector_store_ids": [vector_store_id],
        },
    }
    instructions = (
        instructions
        or "You're an HR assistant that has access to files with information"
           "about the company, its culture, and the hackathon test the job"
           "candidates must pass to get onboard. You share views strongly "
           "correlated with the company culture - striving for productivity"
           "and intolerating incompetence. You're manner of speech resembles"
           "that of Ben Horowitz, a founder of a16z (you can find some info"
           "on the topic in the files)."
    )
    assistant = await client.beta.assistants.create(
        instructions = instructions,
        name="Startup HR Assistant",
        tools=tools,
        tool_resources=tool_resources,
        temperature=0.3,
        model=model,
    )
    return assistant


async def create_thread():
    pass
