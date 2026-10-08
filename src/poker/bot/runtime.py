import argparse
from collections.abc import Callable
from queue import Queue

from shared_logging import LoggingConfig, get_logger, process_logging
from shared_logging.observations import bot_decision
from poker.application.views import CommandResponse, PlayerView
from poker.client.cli import PokerCli
from poker.client.parser import CommandParser
from poker.domain.types import TableId
from poker.transport.local_tcp import LocalTcpClient


def passive_decision(view: PlayerView) -> str | None:
    """Select a server-offered action; no local poker rules or strategy claims."""
    offered = {action.kind.value for action in view.me.legal_actions}
    return next((kind for kind in ("check", "call", "fold") if kind in offered), None)


def run_bot(cli: PokerCli, name: str, decision: Callable[[PlayerView], str | None] = passive_decision,
            *, max_actions: int | None = None) -> None:
    if max_actions is not None and max_actions < 1:
        raise ValueError("max_actions must be positive")
    log = get_logger("bot.decision")
    inputs: Queue[str | None] = Queue()
    inputs.put(f"join {name}")
    pending = False
    submitted = 0

    def response(source: str, result: CommandResponse) -> None:
        nonlocal pending, submitted
        if source != "poll":
            pending = False
        if result.error is not None:
            log.emit("ERROR", "bot.failed", "Bot command rejected", {"code": result.error.code.value})
            inputs.put("quit")
            return
        if max_actions is not None and submitted >= max_actions and not pending:
            inputs.put("quit")
            return
        view = result.view
        if pending or view is None or view.actor_id != view.me.player_id:
            return
        try:
            with bot_decision(log, player_id=view.me.player_id, hand_id=view.hand_id):
                line = decision(view)
        except Exception:
            inputs.put("quit")
            raise
        if line is not None:
            inputs.put(line)
            pending = True
            submitted += 1
            log.bind(player_id=view.me.player_id, hand_id=view.hand_id).emit(
                "INFO", "bot.command_submitted", "Bot submitted a CLI command", {"command": line.split()[0]},
            )

    log.emit("INFO", "bot.started", "Bot runtime started")
    def lifecycle(code: str, message: str | None, source: str | None) -> None:
        if code in ("connection_failed", "input_error"):
            log.emit("ERROR", "bot.failed", "Bot CLI failed", {"reason": message, "code": code})
            inputs.put("quit")

    try:
        cli.run(input_queue=inputs, write=lambda _: None, on_response=response, on_lifecycle=lifecycle)
    finally:
        log.emit("INFO", "bot.stopped", "Bot runtime stopped", {"submitted_actions": submitted})


def main() -> None:
    parser = argparse.ArgumentParser(description="Passive CLI bot with structured decision logs")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--table", help="Table to join; omitted selects the server's default table")
    parser.add_argument("--name", default="Bot")
    parser.add_argument("--max-actions", type=int)
    args = parser.parse_args()
    with process_logging(LoggingConfig.from_environment("bot"), component="bot.process"):
        run_bot(PokerCli(LocalTcpClient(args.port), CommandParser(TableId(args.table) if args.table else None)), args.name, max_actions=args.max_actions)


if __name__ == "__main__":
    main()
