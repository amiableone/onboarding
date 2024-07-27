import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv
from functools import partial

from tg.bot import Bot
from assistant.query import QueryDispatcher

load_dotenv()
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Configure bot settings.")
parser.add_argument(
    "--debug", "-d",
    action="store_true",
    help="run in debug mode to see more detailed logs",
)
parser.add_argument(
    "--files", "-f",
    action="store_true",
    help="tell QueryDispatcher to create a VectorStore object for the assistant",
)


async def _start_cb(bot: Bot, chat_id, *args):
    text = "Hi"
    data = {"chat_id": chat_id, "text": text}
    await bot.post("sendMessage", data)


async def _help_cb(bot: Bot, chat_id, *args):
    text = ("Ask me anything you want to know about Latoken "
            "and the application process.")
    data = {"chat_id": chat_id, "text": text}
    await bot.post("sendMessage", data)


async def handle_cmds(bot: Bot):
    handlers = set()
    while bot.is_running:
        chat, cmd, params = await bot.cmds_pending.get()
        # Only coro funcs are supported as command callbacks for now.
        try:
            handler = bot.commands[cmd](chat, params)
        except KeyError:
            continue
        handlers.add(handler)
        handler.add_done_callback(handlers.discard)
        logger.debug("Handling command /%s from chat %s.", cmd, chat)
    await asyncio.gather(*handlers)


async def dispatch_queries(bot: Bot, qd: QueryDispatcher):
    while bot.is_running:
        try:
            chat, query = await bot.queries.get()
            qd.queries.put_nowait((chat, query))
            logger.debug("Dispatched message from chat %s: %s", chat, query)
        except Exception as exc:
            logger.exception("Exception in dispatcher: %s", exc)


async def dispatch_responses(
        bot: Bot,
        qd: QueryDispatcher,
        poller: asyncio.Task,
):
    senders = set()
    while bot.is_running:
        getter = asyncio.create_task(qd.responses.get())
        done, pending = await asyncio.wait(
            [getter, poller],
            return_when=asyncio.FIRST_COMPLETED,
        )
        first = done.pop()
        if getter is not first:
            await asyncio.gather(*senders)
            getter.cancel()
            return
        chat, msg = getter.result()
        data = {"chat_id": chat, "text": msg}
        sender = asyncio.create_task(bot.post("sendMessage", data))
        senders.add(sender)
        sender.add_done_callback(senders.discard)
        logger.debug("Sending query results to chat %s", chat)
    await asyncio.gather(*senders)


async def log_state():
    loop = asyncio.get_event_loop()
    while not bot._work_complete.done():
        i = 0
        while True:
            await asyncio.sleep(5)
            i += 5
            if i == 60:
                break
        logger.debug(
            "Running tasks: %s",
            [task._coro.__name__ for task in asyncio.all_tasks(loop)],
        )


async def main(
        bot: Bot,
        debug: bool,
        files: bool,
        handle_commands: bool,
):
    logging.basicConfig(
        format="%(name)s:%(asctime)s:%(funcName)s::%(message)s",
        level=logging.DEBUG if debug else logging.INFO,
    )
    logger.debug("Running in debug mode.")

    # instantiate QueryDispatcher.
    qd = await QueryDispatcher.setup(files=files)
    try:
        bot_runner = asyncio.create_task(bot.run())
        qd_runner = asyncio.create_task(qd.run())
        # Give control to bot.run() to create the _initiated future.
        await asyncio.sleep(0.01)
        poller = asyncio.create_task(bot.run_polling())
        query_dispatcher = asyncio.create_task(dispatch_queries(bot, qd))
        response_dispatcher = asyncio.create_task(dispatch_responses(bot, qd, poller))
        bot.add_tasks(
            poller,
            query_dispatcher,
            response_dispatcher,
        )
        tasks = [bot_runner, qd_runner]
        if debug:
            tasks.append(asyncio.create_task(log_state()))
        await bot._initiated
        # Now we can make requests.
        if handle_commands:
            # Set commands if commands in bot instance differ
            # from the ones returned to the 'getMyCommands' request.
            cmd_handler = asyncio.create_task(handle_cmds(bot))
            bot.add_tasks(cmd_handler)
            cmd_array = await bot.get_commands()
            for c1, c2 in zip(cmd_array, bot.commands.values()):
                if c1["command"] != c2.command:
                    tasks.append(bot.set_commands())
                    break
                if c1["description"] != c2.description:
                    tasks.append(bot.set_commands())
                    break
        gather = asyncio.gather(
            *tasks,
            *bot._tasks,
            return_exceptions=True,
        )
        await asyncio.shield(gather)
    except asyncio.CancelledError:
        bot.stop_session()
        qd.stop()
        # cancel query_dispatcher and cmd_handler to stop getting new updates and
        # allow other tasks finish processing remaining updates.
        query_dispatcher.cancel()
        if handle_commands:
            cmd_handler.cancel()
        await gather


if __name__ == "__main__":
    args = parser.parse_args()
    DEBUG = args.debug
    FILES = args.files

    TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
    bot = Bot(TOKEN)
    HANDLE_COMMANDS = hasattr(bot, "_get_commands")

    if HANDLE_COMMANDS:
        # configure the bot.
        start_cb = partial(_start_cb, bot)
        help_cb = partial(_help_cb, bot)
        bot.add_commands(start=start_cb, help=help_cb)
        bot.commands["start"].description = "Let me greet you."
        bot.commands["help"].description = "See what I can help you with."

    asyncio.run(main(bot, DEBUG, FILES, HANDLE_COMMANDS))

