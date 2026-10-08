from queue import Queue
from shared_logging import get_logger

from poker.application.commands import Command, SessionContext
from poker.application.interfaces import CommandHandler
from poker.application.views import CommandResponse
from poker.messaging.interfaces import CommandEnvelope, CommandQueue


class LocalCommandQueue(CommandQueue):
    def __init__(self) -> None:
        self._queue: Queue[CommandEnvelope | None] = Queue()

    def put(self, message: CommandEnvelope | None) -> None:
        self._queue.put(message)

    def get(self) -> CommandEnvelope | None:
        return self._queue.get()


class QueuedCommandHandler(CommandHandler):
    """Network threads enqueue and wait; they never access the engine or SQL."""

    def __init__(self, queue: CommandQueue) -> None:
        self._queue = queue

    def handle(self, command: Command, session: SessionContext) -> CommandResponse:
        message = CommandEnvelope(command, session)
        self._queue.put(message)
        return message.reply.get()


class QueueWorker:
    def __init__(self, queue: CommandQueue, handler: CommandHandler) -> None:
        self._queue = queue
        self._handler = handler

    def process_next(self) -> bool:
        message = self._queue.get()
        if message is None:
            return False
        try:
            message.reply.put(self._handler.handle(message.command, message.session))
        except Exception:
            get_logger("server.queue").bind(correlation_id=message.session.correlation_id).exception(
                "queue.command_failed", "Queue consumer failed", {"command": message.command.kind.value},
            )
            raise
        return True

    def run(self) -> None:
        while self.process_next():
            pass

    def stop(self) -> None:
        self._queue.put(None)
