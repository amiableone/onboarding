import asyncio
from openai.types.beta.threads import Run
from typing import Dict

from utils import (
    get_client,
    store_files,
    create_assistant,
    to_messages,
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
    # map chat ids to thread ids.
    threads: Dict

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

    async def thread_message(self, chat_id, text, role="user"):
        try:
            thread_id = self.threads[chat_id]
        except KeyError:
            thread_id = await self.create_thread(chat_id, text, role)
        await self.client.beta.threads.messages.create(
            thread_id, content=text, role=role
        )
        return thread_id

    async def create_thread(self, chat_id, text, role="user"):
        thread = await self.client.beta.threads.create(
            messages=to_messages(text, role)
        )
        thread_id = thread.id
        self.threads[chat_id] = thread_id
        return thread_id

    async def run_thread(self, thread_id) -> Run:
        # create a run and poll until it reaches a terminal state.
        run = await self.client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=self.assistant.id,
        )
        return run
