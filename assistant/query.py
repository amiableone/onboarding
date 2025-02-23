import asyncio

from openai.types.beta.threads import Run
from openai._types import NotGiven, NOT_GIVEN
from typing import Dict, Set

from .utils import (
    get_client,
    store_files,
    create_assistant,
    to_messages,
    handle_annotations,
    AsyncOpenAI,
    Assistant,
)


class QueryDispatcher:
    """Dispatch user queries to OpenAI API."""

    # attributes for interacting with the OpenAI API
    client: AsyncOpenAI
    assistant: Assistant
    # map chat ids to thread ids.
    threads: Dict

    # provide interface for the telegram bot to interact with
    # this class. `messages` and `responses` are queues of
    # Tuple[str, str] where the first value is a chat_id.
    queries: asyncio.Queue
    responses: asyncio.Queue
    chats: Set[str]
    handlers: Set[asyncio.Task]

    def __init__(self):
        self.client = get_client()
        self.threads = {}
        self.queries = asyncio.Queue()
        self.responses = asyncio.Queue()
        self.chats = set()
        self.handlers = set()
        self.running = False

    @classmethod
    async def setup(cls, files=True):
        qd = cls()

        if files:
            # store information for the assistant in the VectorStore
            # object of the OpenAI API.
            batch = await store_files(qd.client)
            vstore_id = batch.vector_store_id

            # create an assistant.
            qd.assistant = await create_assistant(qd.client, vstore_id)
        else:
            qd.assistant = await create_assistant(qd.client)
        return qd

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

    async def get_response(self, run: Run, last_id: str | NotGiven):
        # if run is completed, retrieve the message it generated.
        if run.status == "completed":
            paginator =  self.client.beta.threads.messages.list(
                thread_id=run.thread_id,
                run_id=run.id,
                after=last_id,
            )
            async for message in paginator:
                last_id = message.id
                try:
                    response = message.content[0].text.value
                    annotations = message.content[0].text.annotations
                    response = await handle_annotations(response, annotations)
                except AttributeError as err:
                    # message content type is probably not 'text'.
                    pass
            return response, last_id

    async def handle_user(self, chat_id, query):
        """Handle a single user interaction with the OpenAI API."""
        last_id = NOT_GIVEN
        while True:
            thread_id = await self.thread_message(chat_id, query)
            run = await self.run_thread(thread_id)
            result = await self.get_response(run, last_id)
            if result is not None:
                response, last_id = result
                self.responses.put_nowait((chat_id, response))
            await asyncio.sleep(1)

    async def run(self):
        """Run QueryDispatcher."""
        self.running = True
        while self.running:
            try:
                async with asyncio.timeout(5):
                    chat_id, query = await self.queries.get()
            except TimeoutError:
                continue
            if chat_id not in self.chats:
                handler = asyncio.create_task(self.handle_user(chat_id, query))
                self.handlers.add(handler)
                handler.add_done_callback(self.handlers.discard)
        await asyncio.gather(*self.handlers)

    def stop(self):
        for task in self.handlers:
            task.cancel()
        self.running = False
