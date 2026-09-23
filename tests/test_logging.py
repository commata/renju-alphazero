import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation import GameResult, MatchResult, save_match_logs
from renju import BLACK, WHITE


class MatchLoggingTest(unittest.TestCase):
    def test_save_match_logs_writes_csv_and_json(self):
        results = (
            GameResult(
                winner=BLACK,
                number_of_moves=3,
                elapsed_seconds=1.25,
                black_agent="A",
                white_agent="B",
                history=((7, 7), (7, 8), (8, 7)),
            ),
            GameResult(
                winner=WHITE,
                number_of_moves=4,
                elapsed_seconds=2.5,
                black_agent="A",
                white_agent="B",
                history=((7, 7), (7, 8), (8, 7), (8, 8)),
            ),
        )
        match = MatchResult(
            games=2,
            black_wins=1,
            white_wins=1,
            draws=0,
            total_moves=7,
            average_moves=3.5,
            elapsed_seconds=3.75,
            games_per_second=2 / 3.75,
            results=results,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = save_match_logs(
                directory,
                [("a_vs_b", match)],
                {"seed": 42, "simulations": 25},
            )

            for path in paths.values():
                self.assertTrue(Path(path).exists())

            with Path(paths["games_csv"]).open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["winner"], "BLACK")
            self.assertEqual(rows[1]["winner_agent"], "B")

            with Path(paths["moves_csv"]).open(encoding="utf-8-sig", newline="") as file:
                moves = list(csv.DictReader(file))
            self.assertEqual(len(moves), 7)
            self.assertEqual(moves[0]["row0"], "7")
            self.assertEqual(moves[0]["row"], "8")
            self.assertEqual(moves[1]["player"], "WHITE")

            with Path(paths["games_json"]).open(encoding="utf-8") as file:
                payload = json.load(file)
            self.assertEqual(payload["config"]["seed"], 42)
            self.assertEqual(len(payload["games"]), 2)
            self.assertEqual(payload["games"][0]["moves"][0]["col"], 8)


if __name__ == "__main__":
    unittest.main()
