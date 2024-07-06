import os

from dotenv import load_dotenv
from httpx import AsyncClient
from openai import AsyncOpenAI

load_dotenv()


def get_client():
    # by default, api_key is set to os.environ.get("OPENAI_API_KEY").
    # handles `OpenAI-Beta: assistants=v2` header automatically
    # when requesting assistants.
    proxy = os.environ.get("HTTP_PROXY")
    http_client = AsyncClient(proxy=proxy)
    client = AsyncOpenAI(http_client=http_client)
    return client
