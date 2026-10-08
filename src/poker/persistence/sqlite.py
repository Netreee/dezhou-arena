import sqlite3
from contextlib import closing
from pathlib import Path

from poker.application.interfaces import TableRepository
from poker.domain.models import Table
from poker.domain.types import TableId
from poker.persistence.snapshot import TableSnapshotCodec
from shared_logging import get_logger

_log = get_logger("server.persistence")


class SqliteTableRepository(TableRepository):
    """One file, one snapshot per table. Each operation closes its connection."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._codec = TableSnapshotCodec()
        path.parent.mkdir(parents=True, exist_ok=True)
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        with closing(sqlite3.connect(path)) as connection:
            with connection:
                connection.executescript(schema)

    def load(self, table_id: TableId) -> Table | None:
        with closing(sqlite3.connect(self._path)) as connection:
            row = connection.execute(
                "SELECT payload_json FROM table_snapshots WHERE table_id = ?", (table_id,),
            ).fetchone()
        return None if row is None else self._codec.decode(str(row[0]))

    def save(self, table: Table) -> None:
        payload = self._codec.encode(table)
        try:
            with closing(sqlite3.connect(self._path)) as connection:
                with connection:
                    connection.execute(
                        "INSERT INTO table_snapshots(table_id, payload_json) VALUES (?, ?) "
                        "ON CONFLICT(table_id) DO UPDATE SET payload_json = excluded.payload_json",
                        (table.id, payload),
                    )
        except Exception:
            _log.exception("snapshot.save_failed", "Snapshot write failed", {"table_id": table.id})
            raise
        _log.emit("INFO", "snapshot.saved", "Snapshot committed", {"table_id": table.id, "revision": table.revision})
