"""Composition root: the only place that selects concrete adapters."""

import argparse
from pathlib import Path
from uuid import uuid4
from shared_logging import LoggingConfig, process_logging

from poker.application.commands import JoinCommand
from poker.client.cli import PokerCli
from poker.client.parser import CommandParser
from poker.domain.models import Table
from poker.domain.types import TableId
from poker.engine.holdem import HoldemEngine
from poker.engine.policies import FiveCardHighEvaluator, NoLimitBettingRules, SidePotAllocator
from poker.persistence.sqlite import SqliteTableRepository
from poker.server import TableRegistry, TableRuntime
from poker.transport.local_tcp import LocalTcpClient, LocalTcpServer


def server_main() -> None:
    parser = argparse.ArgumentParser(description="Local Hold'em server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", type=Path, default=Path(".data/poker.sqlite3"))
    parser.add_argument("--table", action="append", help="Active table ID; repeat for multiple tables")
    args = parser.parse_args()
    table_ids = tuple(TableId(value) for value in args.table) if args.table else (TableId(uuid4().hex),)
    try:
        for table_id in table_ids:
            JoinCommand("validate", table_id)
        if len(set(table_ids)) != len(table_ids):
            raise ValueError("Table IDs must be unique")
    except ValueError as error:
        parser.error(str(error))
    with process_logging(LoggingConfig.from_environment("server"), component="server.process"):
        repository = SqliteTableRepository(args.db)
        registry = TableRegistry(tuple(TableRuntime(
            Table(table_id), repository,
            HoldemEngine(NoLimitBettingRules(), FiveCardHighEvaluator(), SidePotAllocator()),
        ) for table_id in table_ids))
        server = LocalTcpServer(registry, args.port)
        try:
            registry.start()
            print(f"Local Hold'em on {server.address}; table={registry.default_table_id}")
            print(f"Active tables: {', '.join(registry.table_ids)}")
            print("Use a client to join; seated clients poll automatically for table updates.")
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.close()
            registry.close()


def client_main() -> None:
    parser = argparse.ArgumentParser(description="Local Hold'em client with automatic state polling")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--table", help="Table to join; omitted selects the server's default table")
    args = parser.parse_args()
    with process_logging(LoggingConfig.from_environment("cli"), component="client.process"):
        client = LocalTcpClient(args.port)
        cli = PokerCli(client, CommandParser(TableId(args.table) if args.table else None))
        cli.run()
