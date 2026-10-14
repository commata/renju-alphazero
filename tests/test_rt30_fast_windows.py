"""rt30.fast_windows: exact replacement of search.mcts_v5._threat_windows (docs/web-ser5.md §3.1)."""
import json
from pathlib import Path
from random import Random
import unittest

from analysis import mcts_v8
from analysis.mcts_v8 import SearchDiagnostics, _own_vct_attack
from renju import Game
from rt30 import fast_windows
from search import mcts_v5, mcts_v7, threat_patterns
from search.mcts_v5 import _RootContext

ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / 'docs' / 'mcts-v8-results' / 'e3'


def recorded_position(seed: int, line: int, ply: int) -> Game:
    """The position before the V8 move at ``ply`` of game ``line`` in the E3-A JSONL of ``seed``."""
    with (E3 / f'puct_policy_vct2_50k_{seed}.jsonl').open(encoding='utf-8') as handle:
        record = json.loads(handle.readlines()[line])
    game = Game()
    for move in record['moves'][:ply]:
        game.play(*move)
    return game


def random_positions(count: int, seed: int = 20261014):
    """Positions of random legal games (every length up to the end of the game)."""
    rng = Random(seed)
    positions = []
    while len(positions) < count:
        game = Game()
        while not game.done and len(positions) < count:
            legal = game.legal_moves()
            if not legal:
                break
            # Mostly near the stones, so threat windows are common.
            near = [m for m in legal if any(abs(m[0] - r) <= 2 and abs(m[1] - c) <= 2 for r, c in game.history)]
            game.play(*rng.choice(near if near and rng.random() < 0.9 else legal))
            positions.append([row[:] for row in game.board])
    return positions


class _Board:
    def __init__(self, board):
        self.board = board


IMPLEMENTATIONS = [fast_windows.threat_windows_pure] + (
    [fast_windows.threat_windows_numpy] if fast_windows.threat_windows_numpy else [])


class SameWindowsTest(unittest.TestCase):
    def test_same_list_in_the_same_order_on_random_positions(self):
        # list(...) == list(...): the order (WINDOWS order) must match, not only the set.
        for board in random_positions(3000):
            game = _Board(board)
            for player in (1, -1):
                for minimum in (0, 1, 2, 3, 4, 5):
                    expected = list(fast_windows.FROZEN(game, player, minimum))
                    for implementation in IMPLEMENTATIONS:
                        self.assertEqual(list(implementation(game, player, minimum)), expected)

    def test_empty_and_full_boards(self):
        for board in ([[0] * 15 for _ in range(15)], [[1] * 15 for _ in range(15)],
                      [[(r + c) % 3 - 1 for c in range(15)] for r in range(15)]):
            for player in (1, -1):
                for minimum in (2, 3, 4):
                    expected = list(fast_windows.FROZEN(_Board(board), player, minimum))
                    for implementation in IMPLEMENTATIONS:
                        self.assertEqual(list(implementation(_Board(board), player, minimum)), expected)

    def test_default_minimum_is_three(self):
        board = random_positions(40)[-1]
        for implementation in IMPLEMENTATIONS:
            self.assertEqual(list(implementation(_Board(board), 1)), list(fast_windows.FROZEN(_Board(board), 1)))

    def test_early_exit_callers_still_work(self):
        # ``next(_threat_windows(...), None)`` (mcts_v7) needs an iterator.
        board = random_positions(40)[-1]
        for implementation in IMPLEMENTATIONS:
            self.assertEqual(next(implementation(_Board(board), 1, 2), None),
                             next(fast_windows.FROZEN(_Board(board), 1, 2), None))


class InstallTest(unittest.TestCase):
    def tearDown(self):
        fast_windows.uninstall()

    def test_install_rebinds_every_importer_and_uninstall_restores(self):
        modules = (mcts_v5, mcts_v7, mcts_v8, threat_patterns)
        self.assertGreaterEqual(fast_windows.install(), len(modules))
        self.assertTrue(fast_windows.installed())
        for module in modules:
            self.assertIs(module._threat_windows, fast_windows.threat_windows)
        self.assertEqual(fast_windows.FROZEN.__module__, 'search.mcts_v5')
        fast_windows.uninstall()
        self.assertFalse(fast_windows.installed())
        for module in modules:
            self.assertIs(module._threat_windows, fast_windows.FROZEN)

    def test_context_manager(self):
        with fast_windows.fast_windows(fast_windows.threat_windows_pure):
            self.assertIs(mcts_v7._threat_windows, fast_windows.threat_windows_pure)
        self.assertIs(mcts_v7._threat_windows, fast_windows.FROZEN)


class SameSearchTest(unittest.TestCase):
    """End to end: V8-B on a recorded own-VCT position, identical with and without the speed-up."""

    def run_attack(self, game):
        diag = SearchDiagnostics()
        move = _own_vct_attack(game, _RootContext(game.legal_moves(), diag), diag,
                               node_limit=20_000, call_limit=10_000, node_budget=200_000)
        return move, diag.v8_attack_checked, diag.v8_attack_nodes, diag.v8_attack_calls

    def test_v8b_identical(self):
        game = recorded_position(8415, 11, 15)
        expected = self.run_attack(game)
        self.assertIsNotNone(expected[0])
        for implementation in IMPLEMENTATIONS:
            with fast_windows.fast_windows(implementation):
                self.assertEqual(self.run_attack(game), expected)


if __name__ == '__main__':
    unittest.main()
