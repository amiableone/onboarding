import aiohttp
import asyncio
import json
import logging
import os

from datetime import datetime, timedelta
from typing import Optional, Dict, Callable
from urllib.parse import urlencode

logger = logging.getLogger("Bot")


class BotCommandBase:
    """
    This class represents a command object in Telegram Bot API.
    """
    _description = ""
    callback: Optional[Callable] = None

    def __init__(self, bot: "BotBase", callback=None, description=None):
        self.bot = bot
        if callback:
            self.register_callback(callback)
        self.description = description
        logger.info("Created command %s", self.command)

    @property
    def command(self):
        return self.__class__.__name__.lower()

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, desc):
        try:
            self._description = desc.capitalize()[:25]
        except AttributeError:
            self._description = self.__class__.__name__

    def register_callback(self, callback):
        """
        Register callback to make instance callable.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        self.callback = callback

    def __call__(self, *args, **kwargs):
        res = self.callback(*args, **kwargs)
        logger.info("Called command /%s", self.command)
        try:
            # this makes __call__ effectively awaitable
            # when self.callback is a coroutine function.
            return self.bot._loop.create_task(res)
        except TypeError:
            return res


class BotBase:
    """
    This class manages telegram bot runtime and allows making
    get and post requests.
    """
    base_url = "https://api.telegram.org"
    token = "/bot%s"
    method = "/%s"

    def __init__(self, token):
        if not token or not isinstance(token, str):
            raise ValueError("token must be a non-empty string")
        self.token %= token
        self._initiated: Optional[asyncio.Future] = None
        self.session = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.is_running = False
        self._tasks = set()
        self._work_complete: Optional[asyncio.Future] = None
        self._stop_session: Optional[asyncio.Future] = None

    async def run(self):
        self._loop = asyncio.get_running_loop()
        self._initiated = self._loop.create_future()
        async with aiohttp.ClientSession(self.base_url) as session:
            self.session = session
            self.is_running = True
            self._stop_session = self._loop.create_future()
            self._work_complete = self._loop.create_future()
            self._initiated.set_result(None)
            logger.info("Bot is set and running.")
            await self._stop_session
            self.is_running = False
            logger.info("Bot is stopping...")
            await self._work_complete
        self._stop_session = None
        self._work_complete = None
        logger.info("Bot is stopped.")

    def add_tasks(self, *tasks):
        for task in tasks:
            task: asyncio.Task
            self._tasks.add(task)
            task.add_done_callback(self.complete_work)
            logger.debug(
                "Task wrapping coro '%s' is added to the bot.",
                task._coro.__name__,
            )

    def complete_work(self, task):
        self._tasks.discard(task)
        logger.debug(
            "Task wrapping coro %s is discarded.",
            task._coro.__name__,
        )
        if not self._tasks:
            self._work_complete.set_result(None)

    def stop_session(self):
        self._stop_session.set_result(None)

    async def get(self, method, data=None):
        data = data if isinstance(data, dict) else {}
        data = "?" + urlencode(data) if data else ""
        url = self.token + self.method % method + data
        async with self.session.get(url) as response:
            contents = await response.read()
            contents = json.loads(contents)
        return contents

    async def post(self, method, data, headers=None, as_json=True):
        url = self.token + self.method % method
        if as_json:
            data = json.dumps(data)
            hdrs = {"Content-Type": "application/json; charset=utf-8"}
        else:
            data = urlencode(data).encode()
            hdrs = {
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            }
        headers = headers if isinstance(headers, dict) else {}
        hdrs.update(headers)
        async with self.session.post(url, data=data, headers=hdrs) as response:
            contents = await response.read()
            contents = json.loads(contents)
        return contents


class BotCommandManagerMixin:
    """
    This mixin adds command managing features to BotBase subclass.
    """
    commands: Dict[str, BotCommandBase] = {}
    _get_commands = "getMyCommands"
    _set_commands = "setMyCommands"
    has_unset_commands = False

    def __init__(self):
        raise NotImplementedError(
            "Can't instantiate this class on its own. "
            "Use as mixin for a BotBase subclass."
        )

    def add_start(self, callback):
        self.add_commands(start=callback)

    def add_help(self, callback):
        self.add_commands(help=callback)

    def add_settings(self, callback):
        self.add_commands(settings=callback)

    def add_commands(self, **commands_callbacks):
        for command, callback in commands_callbacks.items():
            if self.commands.get(command, False):
                continue
            cmd_class = type(
                command.capitalize(),
                (BotCommandBase,),
                {},
            )
            cmd_instance = cmd_class(self, callback)
            self.__class__.commands[command] = cmd_instance
            self.__class__.has_unset_commands = True

    async def get_commands(self):
        res = await self.get(self._get_commands)
        if res["ok"]:
            return res["result"]

    async def set_commands(self):
        """
        Sets telegram bot commands taken from self.commands.
        Commands already set but missing from self.commands will be unset.
        """
        cmds = {
            "commands": [],
        }
        for name, command in self.commands.items():
            cmd_dict = {
                "command": name,
                "description": command.description,
            }
            cmds["commands"].append(cmd_dict)
        logger.info("Setting commands %s", cmds)
        res = await self.post(self._set_commands, cmds)
        if res["ok"]:
            self.__class__.has_unset_commands = False
        logger.info("Commands are set.")
        return res


class BotUpdateHandlerMixin:
    """
    This mixin adds update polling feature to BotBase subclass.
    """
    _get_updates = "getUpdates"
    allowed_updates = ("message", "edited_message")
    limit = 100
    offset = 1
    timeout = 100   # max allowed
    _reset_period: timedelta
    last_date: Optional[datetime]
    last_id: int
    updates: asyncio.Queue
    queries: asyncio.Queue
    cmds_pending: asyncio.Queue

    def conf_updates(self):
        # if no update retrieved for 7 days, id of the next update is set randomly
        self._reset_period = timedelta(days=7)
        self.last_date = None
        self.last_id = 0
        self.updates = asyncio.Queue()
        self.queries = asyncio.Queue()
        self.cmds_pending = asyncio.Queue()

    def new_last_date(self, date):
        # `date` is param of Update object from Telegram Bot API.
        self.last_date = datetime.fromtimestamp(date)

    def new_last_id(self, luid):
        if self.last_date and self.last_date > datetime.today() - timedelta(days=7):
            self.last_id = max(self.last_id, luid)
        else:
            self.last_id = luid

    def new_offset(self):
        self.offset = self.last_id + 1

    async def get_updates(self):
        data = self._get_updates_params()
        getter = asyncio.create_task(self.get(self._get_updates, data))
        stopper = self._stop_session
        logger.debug("Waiting for updates...")
        done, pending = await asyncio.wait(
            [getter, stopper], return_when=asyncio.FIRST_COMPLETED,
        )
        first = done.pop()
        if getter is not first:
            getter.cancel()
            await asyncio.sleep(0)
            return
        res = getter.result()
        if res["ok"]:
            for update in res["result"]:
                self.updates.put_nowait(update)

    def _get_updates_params(self):
        return {
            "offset": self.offset,
            "limit": self.limit,
            "timeout": self.timeout,
            "allowed_updates": self.allowed_updates,
        }

    def process_updates(self):
        """
        Process updates and recalculate offset param.
        """
        date = self.last_date
        # while stmt references attr of BotBase.
        while not self.updates.empty():
            update = self.updates.get_nowait()
            msg_obj = update.get("message") or update.get("edited_message")
            # Update class attributes
            date = msg_obj.get("date") or msg_obj.get("edit_date") or date
            self.new_last_date(date)
            self.new_last_id(update["update_id"])
            self.new_offset()
            try:
                self.process_message(msg_obj)
            except TypeError:
                continue
        logger.info("New update offset value is %s", self.offset)

    def process_message(self, msg_obj):
        """
        Parse message for command or query and put processed data into
        the corresponding queue.
        """
        try:
            chat_id = msg_obj["chat"]["id"]
            message = msg_obj["text"]
            logger.info("Received message '%s' from chat %s", message, chat_id)
            query = chat_id, message
            self.queries.put_nowait(query)
        except (TypeError, KeyError):
            raise TypeError("Neither a message nor command")

    async def run_polling(self):
        logger.info("Polling procedure initiated.")
        try:
            while self.is_running:
                await self.get_updates()
                self.process_updates()
        except Exception as exc:
            logger.exception("Exception in run_polling: %s", exc)


class Bot(BotBase, BotUpdateHandlerMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf_updates()