import asyncio
import os

from dotenv import load_dotenv
from httpx import AsyncClient
from openai import AsyncOpenAI
from openai.types import FileObject
from openai.types.beta import Assistant, Thread
from openai.types.beta.vector_stores import VectorStoreFileBatch
from pathlib import Path

load_dotenv()

FILES = Path(__file__).resolve().parent.parent / "files"
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
    with open(filename, "rb") as f:
        file = await client.files.create(file=f, purpose="assistants")
    return file


async def store_files(
        client: AsyncOpenAI,
        store_name="Info",
) -> VectorStoreFileBatch:
    """Store files in the VectorStore object (see OpenAI API docs)."""
    vstore = await client.beta.vector_stores.create(name=store_name)
    tasks = []
    for path in os.listdir(FILES):
        task = asyncio.create_task(get_file(client, path))
        tasks.append(task)
    files = await asyncio.gather(*tasks)
    file_ids = [file.id for file in files]
    batch = await client.beta.vector_stores.file_batches.create(
        vector_store_id=vstore.id,
        file_ids=file_ids,
    )
    return batch


async def create_assistant(
        client: AsyncOpenAI,
        instructions: str | None = None,
        vector_store_id: str | None = None,
        model="gpt-4o",
) -> Assistant:
    # only the file_search tool is supported for now.
    if vector_store_id:
        tools = FILE_SEARCH_TOOL,
        tool_resources = {
            "file_search": {
                "vector_store_ids": [vector_store_id],
            },
        }
        instructions = (
            instructions
            or "1. You're an HR assistant that has access to files with information "
                "about (1) Latokent company, (2) its culture, and (3) the hackathon "
                "test the job candidates must pass to get on board. "
                "2. You share views strongly correlated with the company culture - "
                "striving for productivity and intolerating incompetence. "
                "3. Give answers in the manner resembling that of Ben Horowitz, "
                "a founder of a16z (you can find some info on the topic in the "
                "files). "
                "4. When information is not in the files provided, don't mention that. "
                "Just answer the question. "
                "5. When asked about a hackathon, answer using info about the AI Bot "
                "Hackathon Task, including timelines."
                "6. You can ask questions from the Test for candidates. The file with "
                "test questions to ask has 'Проверьте, готовы ли вы к результативному "
                "хакатону и интервью с помощью этого короткого теста' in it."
        )
        return await client.beta.assistants.create(
            instructions = instructions,
            name="Startup HR Assistant",
            tools=tools,
            tool_resources=tool_resources,
            temperature=0.3,
            model=model,
        )
    return await client.beta.assistants.create(
        name="Startup HR Assistant",
        temperature=0.3,
        model=model,
    )


def to_messages(text, role):
    # images are not supported at this point.
    return [{"role": role, "content": text}]


async def get_filename(
        client: AsyncOpenAI,
        file_id: str,
) -> FileObject:
    f = await client.files.retrieve(file_id)
    return f.filename


async def handle_annotations(
        response: str,
        annotations: list,
):
    """
    Replace file references with empty strings.
    """
    for annotation in annotations:
        placeholder = annotation.text
        response = response.replace(placeholder, "")
    return response