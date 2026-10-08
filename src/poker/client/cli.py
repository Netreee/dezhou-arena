from collections.abc import Callable
from dataclasses import replace
from math import isfinite
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic
from shared_logging import get_logger

from poker.application.commands import ActCommand, Command, StateCommand
from poker.application.views import CommandResponse, PlayerView
from poker.client.guarded import GuardedInput, GuardedResult
from poker.client.parser import CommandParser
from poker.transport.interfaces import CommandClient


class PokerCli:
    def __init__(self, client: CommandClient, parser: CommandParser) -> None:
        self._client = client
        self._parser = parser
        self._latest: PlayerView | None = None
        self._displayed: PlayerView | None = None

    @property
    def latest(self) -> PlayerView | None:
        return self._latest

    def submit(self, line: str) -> CommandResponse:
        return self._remember(self._client.send(self._parse(line)))

    def _parse(self, line: str) -> Command:
        hand_id = self._latest.hand_id if self._latest is not None else None
        return self._parser.parse(line, hand_id)

    def refresh(self) -> CommandResponse:
        return self._remember(self._client.send(StateCommand()))

    def _remember(self, response: CommandResponse) -> CommandResponse:
        if response.view is not None:
            self._latest = response.view
        return response

    def run(
        self, *, poll_interval: float = 1.0,
        input_queue: Queue[str | None] | None = None,
        clock: Callable[[], float] = monotonic,
        write: Callable[[str], None] = print,
        on_response: Callable[[str, CommandResponse], None] | None = None,
        on_lifecycle: Callable[[str, str | None, str | None], None] | None = None,
        guarded_queue: Queue[GuardedInput] | None = None,
        on_guarded_result: Callable[[GuardedResult], None] | None = None,
    ) -> None:
        """Own connect/send/close here; optional queue/clock/output drive fast tests."""
        if not isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("Polling interval must be positive and finite")
        if guarded_queue is not None and on_guarded_result is None:
            raise ValueError("Guarded input requires a result callback")
        inputs = input_queue if input_queue is not None else Queue[str | None]()
        stop = Event()
        reader: Thread | None = None
        self._displayed = None
        try:
            self._client.connect()
            write("命令: join 姓名 | join_table 桌号 姓名 | state | start | fold | check | call | bet_to N | raise_to N | all_in | quit")
            write("输入命令后按回车；入座后每秒自动刷新。金额表示本街总投入，以服务器给出的合法动作为准。")
            if input_queue is None:
                reader = Thread(target=self._read_input, args=(inputs, stop), name="poker-input", daemon=True)
                reader.start()
            next_poll = clock() + poll_interval if self._latest is not None else None
            while True:
                timeout = max(0.0, next_poll - clock()) if next_poll is not None else None
                try:
                    item = self._next_input(inputs, guarded_queue, timeout)
                except Empty:
                    # A guarded queue needs a short wakeup interval, not extra polls.
                    if next_poll is None or clock() < next_poll:
                        continue
                    response = self.refresh()
                    self._show(response, write, force=False)
                    if on_response is not None:
                        on_response("poll", response)
                    next_poll = clock() + poll_interval
                    continue
                if isinstance(item, GuardedInput):
                    self._guarded(item, write, on_response, on_guarded_result)
                    next_poll = clock() + poll_interval if self._latest is not None else None
                    continue
                line = item
                if line is None or line.strip() == "quit":
                    break
                if next_poll is not None and clock() >= next_poll:
                    response = self.refresh()
                    self._show(response, write, force=False)
                    if on_response is not None:
                        on_response("poll", response)
                    next_poll = clock() + poll_interval
                if not line.strip():
                    continue
                try:
                    command = self._parse(line)
                except ValueError as error:
                    get_logger("client.cli").emit("WARNING", "client.input_rejected", "CLI input rejected", {"reason": str(error)})
                    write(f"输入错误: {error}")
                    if on_lifecycle is not None:
                        on_lifecycle("input_error", str(error), line)
                    continue
                get_logger("client.cli").emit("INFO", "client.command_submitted", "CLI command submitted", {"command": command.kind.value})
                response = self._remember(self._client.send(command))
                if command.kind.value == "join" and response.view is not None:
                    get_logger("client.cli").bind(player_id=response.view.me.player_id).emit("INFO", "client.joined", "CLI joined")
                get_logger("client.cli").bind(player_id=response.view.me.player_id if response.view else None,
                                             hand_id=response.view.hand_id if response.view else None).emit(
                    "INFO", "client.response_received", "CLI command response", {"command": command.kind.value, "ok": response.ok},
                )
                self._show(response, write, force=True)
                if on_response is not None:
                    on_response(line, response)
                if next_poll is None and self._latest is not None:
                    next_poll = clock() + poll_interval
        except KeyboardInterrupt:
            pass
        except OSError as error:
            get_logger("client.cli").emit("ERROR", "client.connection_failed", "CLI connection failed", {"reason": str(error)})
            write(f"连接结束: {error}")
            if on_lifecycle is not None:
                on_lifecycle("connection_failed", str(error), None)
        finally:
            stop.set()
            self._client.close()
            if reader is not None:
                # stdin cannot be cancelled portably; the daemon never owns the socket.
                reader.join(timeout=0.1)
            if on_lifecycle is not None:
                on_lifecycle("closed", None, None)

    @staticmethod
    def _next_input(
        inputs: Queue[str | None], guarded_queue: Queue[GuardedInput] | None,
        timeout: float | None,
    ) -> str | None | GuardedInput:
        if guarded_queue is None:
            return inputs.get(timeout=timeout)
        # Normal control input (especially quit) takes priority. Both queues are
        # consumed by this CLI owner; producers never access the socket.
        try:
            return inputs.get_nowait()
        except Empty:
            pass
        try:
            return guarded_queue.get_nowait()
        except Empty:
            return inputs.get(timeout=0.05 if timeout is None else min(timeout, 0.05))

    def _guarded(
        self, item: GuardedInput, write: Callable[[str], None],
        on_response: Callable[[str, CommandResponse], None] | None,
        on_result: Callable[[GuardedResult], None] | None,
    ) -> None:
        def report(result: GuardedResult) -> None:
            if result.error is not None:
                get_logger("client.cli").emit("WARNING", "client.guarded_rejected", "CLI guarded input rejected",
                                              {"request_id": result.request_id, "code": result.error})
            if on_result is not None:
                on_result(result)

        try:
            command = self._parser.parse(item.line, item.token.hand_id)
            if not isinstance(command, ActCommand):
                raise ValueError("Guarded input only permits a player action")
        except ValueError:
            report(GuardedResult(item.request_id, error="invalid_action"))
            return
        try:
            fresh = self.refresh()
            self._show(fresh, write, force=False)
            if on_response is not None:
                on_response("guarded_refresh", fresh)
            if fresh.view is None:
                report(GuardedResult(item.request_id, error="refresh_failed"))
                return
            if not item.token.matches(fresh.view) or fresh.view.actor_id != item.token.player_id:
                report(GuardedResult(item.request_id, error="stale_observation"))
                return
            # Preserve the decision's hand and bind its revision atomically at
            # the server as well; another connection may race the refresh.
            command = replace(command, expected_revision=item.token.revision)
            get_logger("client.cli").emit("INFO", "client.command_submitted", "CLI guarded command submitted",
                                          {"command": command.kind.value, "request_id": item.request_id})
            response = self._remember(self._client.send(command))
            get_logger("client.cli").bind(player_id=item.token.player_id, hand_id=item.token.hand_id).emit(
                "INFO", "client.response_received", "CLI guarded command response",
                {"command": command.kind.value, "request_id": item.request_id, "ok": response.ok},
            )
            self._show(response, write, force=True)
            if on_response is not None:
                on_response(item.line, response)
            report(GuardedResult(item.request_id, response=response))
        except OSError:
            report(GuardedResult(item.request_id, error="connection_failed"))
            raise

    @staticmethod
    def _read_input(inputs: Queue[str | None], stop: Event) -> None:
        while not stop.is_set():
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                inputs.put(None)
                return
            if stop.is_set():
                return
            inputs.put(line)
            if line.strip() == "quit":
                return

    def _show(self, response: CommandResponse, write: Callable[[str], None], *, force: bool) -> None:
        if response.error is not None:
            write(f"服务器拒绝 [{response.error.code.value}]: {response.error.message}")
        elif response.view is not None and (force or response.view != self._displayed):
            write(self._render(response.view))
            self._displayed = response.view

    @staticmethod
    def _render(view: PlayerView) -> str:
        names = {p.player_id: p.name for p in view.players}
        actor = names.get(view.actor_id, str(view.actor_id)) if view.actor_id is not None else "无"
        phase = view.phase.value if view.phase is not None else "未开局"
        lines = [
            f"\n牌桌: {view.table_id} | 版本: {view.revision}",
            f"本手: {view.hand_id or '未开局'} | 阶段: {phase}",
            f"庄位: {view.button_seat if view.button_seat is not None else '无'} | 行动者: {actor}",
            f"公共牌: {' '.join(view.board) or '无'}",
            f"底池: {view.pot_total} | 当前下注: {view.current_bet}",
            "座位 | 玩家 | 筹码 | 状态 | 本街投入 | 本手投入 | 已公开底牌",
        ]
        for player in view.players:
            name = player.name + (" (你)" if player.player_id == view.me.player_id else "")
            status = player.status.value if player.status is not None else "未参局"
            lines.append(
                f"{player.seat} | {name} | {player.stack} | {status} | "
                f"{player.street_commit} | {player.hand_commit} | {' '.join(player.revealed_cards) or '无'}"
            )
        lines.append(f"本人底牌: {' '.join(view.me.hole_cards) or '无'}")
        options = []
        for option in view.me.legal_actions:
            label = option.kind.value
            if option.min_to is not None and option.max_to is not None:
                label += f" {option.min_to}..{option.max_to} (本街总额)"
            elif option.pay is not None:
                label += f" (支付 {option.pay})"
            options.append(label)
        lines.append(f"合法动作: {' | '.join(options) or '无'}")
        if view.result is not None:
            label = "本手结果" if view.result.hand_id == view.hand_id else "上一手结果"
            lines.append(f"{label}: {view.result.hand_id}")
            for index, award in enumerate(view.result.awards, 1):
                shares = ", ".join(f"{names.get(s.player_id, str(s.player_id))}: {s.amount}" for s in award.shares)
                lines.append(f"池 {index}: {award.amount} | 分配: {shares}")
            if view.result.refunds:
                refunds = ", ".join(f"{names.get(s.player_id, str(s.player_id))}: {s.amount}" for s in view.result.refunds)
                lines.append(f"退款: {refunds}")
        return "\n".join(lines)
