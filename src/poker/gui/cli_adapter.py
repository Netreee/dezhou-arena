from queue import Empty, Queue
from threading import Lock, Thread
from shared_logging import get_logger

from poker.application.views import CommandResponse
from poker.domain.types import TableId
from poker.client.cli import PokerCli
from poker.gui.interfaces import CliGuiPort
from poker.gui.models import CliLifecycleUpdate, CliResponseUpdate, CliUpdate, GuiError


class PokerCliGuiAdapter(CliGuiPort):
    """One thread owns PokerCli; all GUI traffic crosses typed queues."""

    def __init__(self, cli: PokerCli, *, poll_interval: float = 1.0) -> None:
        self._cli = cli
        self._poll_interval = poll_interval
        self._input: Queue[str | None] = Queue()
        self._updates: Queue[CliUpdate] = Queue()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._closed = False

    def start(self, name: str, table_id: TableId | None = None) -> None:
        with self._lock:
            if self._thread is not None or self._closed:
                raise RuntimeError("CLI adapter can only start once")
            self._input.put(f"join_table {table_id} {name}" if table_id is not None else f"join {name}")
            self._thread = Thread(target=self._run, name="gui-cli", daemon=True)
            self._thread.start()

    def submit_line(self, line: str) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive() or self._closed:
                raise RuntimeError("CLI adapter is not running")
            self._input.put(line)

    def next_update(self) -> CliUpdate | None:
        try:
            return self._updates.get_nowait()
        except Empty:
            return None

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._input.put("quit")
            thread = self._thread
        if thread is not None:
            thread.join(timeout=6)
            if thread.is_alive():
                raise RuntimeError("CLI owner did not finish within its network timeout")

    def _response(self, source: str, response: CommandResponse) -> None:
        self._updates.put(CliResponseUpdate(source, response))

    def _lifecycle(self, code: str, message: str | None, source: str | None) -> None:
        self._updates.put(CliLifecycleUpdate(
            GuiError(code, message) if message is not None else None, code != "input_error", source,
        ))

    def _run(self) -> None:
        try:
            self._cli.run(input_queue=self._input, poll_interval=self._poll_interval,
                          write=lambda _: None, on_response=self._response, on_lifecycle=self._lifecycle)
        except Exception as error:
            get_logger("gui.cli").exception("gui.runtime_failed", "GUI CLI runtime failed")
            self._updates.put(CliLifecycleUpdate(GuiError("cli_runtime_failed", str(error))))
