import socket
import socketserver
from typing import TextIO, cast
from uuid import uuid4

from shared_logging import get_logger

from poker.application.commands import Command, SessionContext
from poker.application.interfaces import CommandHandler
from poker.application.views import CommandResponse, ErrorInfo
from poker.domain.types import ErrorCode
from poker.transport.codec import JsonLineCodec
from poker.transport.interfaces import CommandClient, CommandServer

_log = get_logger("transport")


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = cast(_SocketServer, self.server)
        session = SessionContext()
        log = _log.bind(connection_id=session.connection_id)
        log.emit("INFO", "connection.opened", "Game connection opened")
        try:
            while raw := self.rfile.readline():
                session.correlation_id = None
                try:
                    command, session.correlation_id = server.codec.decode_request(raw.decode("utf-8"))
                except (ValueError, UnicodeError) as error:
                    log.emit("WARNING", "command.rejected", "Invalid protocol request", {"reason": str(error)})
                    response = CommandResponse(error=ErrorInfo(ErrorCode.BAD_REQUEST, str(error)))
                else:
                    response = server.gateway.handle(command, session)
                self.wfile.write((server.codec.encode_response(response) + "\n").encode("utf-8"))
                self.wfile.flush()
        except Exception:
            log.bind(correlation_id=session.correlation_id, player_id=session.player_id).exception("connection.failed", "Game connection failed")
            raise
        finally:
            log.emit("INFO", "connection.closed", "Game connection closed")


class _SocketServer(socketserver.ThreadingTCPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], gateway: CommandHandler) -> None:
        self.gateway = gateway
        self.codec = JsonLineCodec()
        super().__init__(address, _RequestHandler)


class LocalTcpServer(CommandServer):
    def __init__(self, gateway: CommandHandler, port: int = 8765) -> None:
        # Local-only by construction; do not expose a configurable public host.
        self._server = _SocketServer(("127.0.0.1", port), gateway)
        self._started = False

    @property
    def address(self) -> tuple[str, int]:
        address = self._server.server_address
        return str(address[0]), int(address[1])

    def serve_forever(self) -> None:
        self._started = True
        self._server.serve_forever(poll_interval=0.1)

    def close(self) -> None:
        # shutdown waits for the serving loop; an unstarted listener only closes.
        if self._started:
            self._server.shutdown()
        self._server.server_close()


class LocalTcpClient(CommandClient):
    def __init__(self, port: int = 8765, *, timeout: float = 5.0) -> None:
        self._port = port
        self._timeout = timeout
        self._socket: socket.socket | None = None
        self._stream: TextIO | None = None
        self._codec = JsonLineCodec()

    def connect(self) -> None:
        if self._socket is not None:
            raise RuntimeError("Client is already connected")
        self._socket = socket.create_connection(("127.0.0.1", self._port), timeout=self._timeout)
        self._stream = self._socket.makefile("rw", encoding="utf-8", newline="\n")

    def send(self, command: Command) -> CommandResponse:
        if self._stream is None:
            raise RuntimeError("Connect before sending commands")
        correlation_id = uuid4().hex
        _log.bind(correlation_id=correlation_id).emit("DEBUG", "client.transport_sent", "Game request sent", {"command": command.kind.value})
        self._stream.write(self._codec.encode_command(command, correlation_id=correlation_id) + "\n")
        self._stream.flush()
        raw = self._stream.readline()
        if not raw:
            raise ConnectionError("Server closed the connection")
        return self._codec.decode_response(raw)

    def close(self) -> None:
        stream, self._stream = self._stream, None
        connection, self._socket = self._socket, None
        try:
            if stream is not None:
                stream.close()
        finally:
            if connection is not None:
                connection.close()
