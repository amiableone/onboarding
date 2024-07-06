import asyncio
from typing import Dict

from utils import (
    get_client,
    store_files,
    create_assistant,
    AsyncOpenAI,
    Assistant,
    Thread,
    VectorStoreFileBatch,
)


class QueryDispatcher:
    """Dispatch user queries to OpenAI API."""

    # attributes for interacting with the OpenAI API
    client: AsyncOpenAI
    assistant: Assistant
    batch: VectorStoreFileBatch
    # map chat ids to Thread objects.
    threads: Dict[str, Thread]

    # provide interface for the telegram bot to interact with
    # this class. `messages` and `responses` are queues of
    # Tuple[str, str] where the first value is a chat_id.
    queries: asyncio.Queue
    responses: asyncio.Queue

    def __init__(self):
        self.client = get_client()

    @classmethod
    async def setup(cls):
        qd = cls()

        # store information for the assistant in the VectorStore object
        # of the OpenAI API.
        batch = await store_files(qd.client)
        qd.batch = batch
        vectore_store_id = batch.vector_store_id

        # create an assistant.
        self.assistant = await create_assistant(qd.client, vectore_store_id)
