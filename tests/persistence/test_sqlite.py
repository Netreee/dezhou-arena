import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from poker.domain.models import ChipShare, HandResult, Pot, PotAward
from poker.domain.types import PlayerId, TableId
from poker.persistence.sqlite import SqliteTableRepository
from tests.fixtures import make_table


class SqliteTests(unittest.TestCase):
    def test_private_snapshot_round_trip_and_reopen(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "poker.sqlite3"
            repository = SqliteTableRepository(path)
            table = make_table(with_hand=True)
            repository.save(table)
            self.assertEqual(SqliteTableRepository(path).load(table.id), table)

    def test_loaded_state_is_detached_and_save_replaces_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            repository = SqliteTableRepository(Path(directory) / "poker.sqlite3")
            table = make_table()
            repository.save(table)
            loaded = repository.load(table.id)
            assert loaded is not None
            loaded.players[0].stack = 123
            self.assertEqual(repository.load(table.id), table)
            repository.save(loaded)
            self.assertEqual(repository.load(table.id), loaded)

    def test_unknown_table_is_absent(self) -> None:
        with TemporaryDirectory() as directory:
            repository = SqliteTableRepository(Path(directory) / "poker.sqlite3")
            self.assertIsNone(repository.load(TableId("absent")))

    def test_result_and_public_reveal_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            repository = SqliteTableRepository(Path(directory) / "poker.sqlite3")
            table = make_table(with_hand=True)
            assert table.hand is not None
            refund = ChipShare(PlayerId("p2"), 5)
            table.hand.refunds.append(refund)
            table.last_result = HandResult(
                table.hand.id,
                (PotAward(Pot(20, (PlayerId("p1"), PlayerId("p2"))),
                          (ChipShare(PlayerId("p1"), 20),)),),
                (refund,), {pid: member.hole_cards for pid, member in table.hand.players.items()},
            )
            repository.save(table)
            self.assertEqual(repository.load(table.id), table)
