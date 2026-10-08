import argparse
from pathlib import Path

import uvicorn
from shared_logging import LoggingConfig, process_logging

from poker.client.cli import PokerCli
from poker.client.parser import CommandParser
from poker.gui.cli_adapter import PokerCliGuiAdapter
from poker.gui.controller import LocalGuiSessionManager
from poker.gui.http_app import create_app
from poker.gui.interfaces import CliGuiPort
from poker.transport.local_tcp import LocalTcpClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Hold'em browser client")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP listen address; use 0.0.0.0 for LAN access")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--game-port", type=int, default=8765)
    parser.add_argument("--assets-dir", type=Path, default=Path(__file__).with_name("web") / "dist")
    args = parser.parse_args()

    def new_port() -> CliGuiPort:
        return PokerCliGuiAdapter(PokerCli(LocalTcpClient(args.game_port), CommandParser()))

    with process_logging(LoggingConfig.from_environment("gui"), component="gui.process"):
        app = create_app(LocalGuiSessionManager(new_port), assets_dir=args.assets_dir)
        uvicorn.run(app, host=args.host, port=args.port, log_config=None)
