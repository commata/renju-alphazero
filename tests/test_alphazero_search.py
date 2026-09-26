import random
import unittest
from copy import deepcopy

from renju import BLACK, WHITE, Game
from search.alphazero import (Node, SearchConfig, backup, run_search, search_with_tree,
                              select_child, terminal_value)
from search.evaluator import (EvaluationResult, EvaluatorOutputError, ScriptedEvaluator,
                              UniformEvaluator)


def cfg(n, **kwargs):
    return SearchConfig(num_simulations=n, **{'noise_enabled': False, **kwargs})


def action(row, col):
    return row * 15 + col


def play(game, moves):
    for move in moves:
        game.play(*move)
    return game


def opened_game():
    return play(Game(), ((7, 7), (7, 8), (8, 8)))


def black_open_four():
    """Black to play with an open four on row 7: (7,3) or (7,8) wins."""
    return play(Game(), ((7, 7), (0, 0), (7, 6), (0, 2), (7, 5), (0, 4), (7, 4), (0, 6)))


def white_must_block():
    """White to play; black has a closed four (7,4)-(7,7) and threatens (7,8)."""
    return play(Game(), ((7, 7), (7, 3), (7, 6), (0, 2), (7, 5), (0, 4), (7, 4)))


def _pattern(r, c):
    return BLACK if ((c + 2 * (r % 2) + (r // 2) % 2) // 2) % 2 == 0 else WHITE


def near_full_draw():
    """Full no-five board with two empty cells; every line ends in a no-legal draw."""
    game = Game()
    game.board = [[_pattern(r, c) for c in range(15)] for r in range(15)]
    game.board[0][2] = game.board[0][3] = 0
    game.to_play = WHITE
    return game


def state(game):
    return (deepcopy(game.board), list(game.history), game.to_play, game.winner, game.done)


class CountingGame(Game):
    legal_calls = 0

    def legal_moves(self):
        type(self).legal_calls += 1
        return super().legal_moves()


class ConfigTest(unittest.TestCase):
    def test_validation(self):
        for kwargs in ({'num_simulations': 0}, {'c_puct': 0}, {'fpu': 0.1}, {'tau': 0},
                       {'temperature_moves': -1}, {'dirichlet_alpha': 0},
                       {'dirichlet_epsilon': 1.5}, {'noise_enabled': 1},
                       {'evaluator_batch_size': 2}, {'num_simulations': True}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                SearchConfig(**kwargs)

    def test_round_trip(self):
        config = SearchConfig(num_simulations=8, c_puct=2)
        data = config.to_dict()
        self.assertEqual(set(data), {'format_version', 'num_simulations', 'c_puct', 'fpu', 'tau',
                                     'temperature_moves', 'dirichlet_alpha', 'dirichlet_epsilon',
                                     'noise_enabled', 'evaluator_batch_size'})
        self.assertEqual(SearchConfig.from_dict(data), config)


class RootContractTest(unittest.TestCase):
    def test_done_root_rejected(self):
        game = play(black_open_four(), ((7, 3),))
        self.assertTrue(game.done)
        with self.assertRaises(ValueError):
            run_search(game, UniformEvaluator(), cfg(4))

    def test_root_pre_expansion_not_counted(self):
        game = opened_game()
        for n in (1, 2, 7):
            evaluator = UniformEvaluator()
            result, root = search_with_tree(game, evaluator, cfg(n))
            self.assertEqual(root.visit_count, n)
            self.assertEqual(sum(c.visit_count for c in root.children.values()), n)
            self.assertEqual(sum(result.visit_counts), n)
            self.assertEqual(evaluator.calls, n + 1)  # root + one leaf per simulation
            self.assertEqual(result.evaluator_calls, n + 1)
            self.assertEqual(root.value_sum, 0.0)
            self.assertFalse(result.fast_path)

    def test_root_value_not_backed_up(self):
        game = opened_game()  # white to play at the root, leaves are black to play
        evaluator = ScriptedEvaluator(value=lambda s: 0.75 if s.to_play == WHITE else 0.0)
        _, root = search_with_tree(game, evaluator, cfg(1))
        child = next(c for c in root.children.values() if c.visit_count)
        self.assertEqual(child.value_sum, 0.0)  # only the leaf value (0) was backed up

    def test_children_are_exactly_legal_moves(self):
        game = opened_game()
        _, root = search_with_tree(game, UniformEvaluator(), cfg(3))
        self.assertEqual(sorted(root.children), sorted(action(*m) for m in game.legal_moves()))
        self.assertTrue(all(c.player_who_moved == WHITE for c in root.children.values()))
        self.assertEqual(root.to_play, WHITE)

    def test_single_legal_fast_path(self):
        game = Game()
        evaluator = ScriptedEvaluator()
        rng = random.Random(5)
        before = rng.getstate()
        result, root = search_with_tree(game, evaluator, SearchConfig(num_simulations=16), rng)
        self.assertIsNone(root)
        self.assertTrue(result.fast_path)
        self.assertEqual(evaluator.calls, 0)
        self.assertEqual(result.evaluator_calls, 0)
        self.assertEqual(rng.getstate(), before)
        self.assertEqual(sum(result.visit_counts), 1)
        self.assertEqual(result.visit_counts[action(7, 7)], 1)
        self.assertEqual(result.to_play, BLACK)

    def test_non_terminal_root_without_legal_moves_raises(self):
        game = Game()
        game.board = [[_pattern(r, c) for c in range(15)] for r in range(15)]
        with self.assertRaises(RuntimeError):
            run_search(game, UniformEvaluator(), cfg(2))


class PuctTest(unittest.TestCase):
    def test_prior_preference(self):
        target = action(3, 3)
        result = run_search(opened_game(), ScriptedEvaluator(weights={target: 50.0}), cfg(1))
        self.assertEqual(result.visit_counts[target], 1)

    def test_index_tie_break_and_determinism(self):
        game = opened_game()
        legal = sorted(action(*m) for m in game.legal_moves())
        result = run_search(game, UniformEvaluator(), cfg(3))
        self.assertEqual([a for a, n in enumerate(result.visit_counts) if n], legal[:3])
        again = run_search(game, UniformEvaluator(), cfg(3))
        self.assertEqual(result.visit_counts, again.visit_counts)

    def test_prior_tie_break_on_equal_score(self):
        parent = Node(None, 1.0, None)
        parent.visit_count = 1
        high = Node(200, 0.5, BLACK)
        high.visit_count = 1              # U = 0.5 * 1 / 2 = 0.25, Q = 0
        low = Node(3, 0.25, BLACK)        # U = 0.25 * 1 / 1 = 0.25
        parent.children = {3: low, 200: high}
        self.assertEqual(select_child(parent, 1.0).action, 200)
        same = Node(4, 0.25, BLACK)
        parent.children = {4: same, 3: low}
        self.assertEqual(select_child(parent, 1.0).action, 3)

    def test_parent_zero_visits_uses_sqrt_one(self):
        parent = Node(None, 1.0, None)
        low = Node(0, 0.25, BLACK)
        high = Node(1, 0.75, BLACK)
        parent.children = {0: low, 1: high}
        self.assertIs(select_child(parent, 1.0), high)


class BackupTest(unittest.TestCase):
    def test_backup_alternates_by_player_identity(self):
        root = Node(None, 1.0, None)
        a = Node(1, 0.5, BLACK)
        b = Node(2, 0.5, WHITE)
        c = Node(3, 0.5, BLACK)
        backup(root, [a, b, c], 0.5, WHITE)  # leaf value from WHITE's view
        self.assertEqual((a.value_sum, b.value_sum, c.value_sum), (-0.5, 0.5, -0.5))
        self.assertEqual((root.visit_count, a.visit_count, c.visit_count), (1, 1, 1))
        self.assertEqual(root.value_sum, 0.0)

    def test_winning_terminal_leaf_keeps_to_play(self):
        root = Node(None, 1.0, None)
        win = Node(4, 1.0, BLACK)
        backup(root, [win], 1.0, BLACK)  # to_play == winner == player_who_moved
        self.assertEqual(win.q, 1.0)

    def test_leaf_value_sign_in_search(self):
        _, root = search_with_tree(opened_game(), ScriptedEvaluator(value=0.5), cfg(1))
        child = next(c for c in root.children.values() if c.visit_count)
        self.assertEqual(child.to_play, BLACK)
        self.assertEqual(child.q, -0.5)

    def test_terminal_value_function(self):
        game = black_open_four()
        with self.assertRaises(ValueError):
            terminal_value(game, BLACK)
        game.play(7, 3)
        self.assertEqual((game.winner, game.to_play), (BLACK, BLACK))
        self.assertEqual(terminal_value(game, BLACK), 1.0)
        self.assertEqual(terminal_value(game, WHITE), -1.0)


class TerminalTest(unittest.TestCase):
    def test_terminal_win_sign(self):
        win = action(7, 3)
        _, root = search_with_tree(black_open_four(), ScriptedEvaluator(weights={win: 100.0}),
                                   cfg(6))
        child = root.children[win]
        self.assertTrue(child.is_terminal and child.is_expanded)
        self.assertEqual(child.children, {})
        self.assertEqual((child.to_play, child.player_who_moved), (BLACK, BLACK))
        self.assertEqual(child.terminal_value, 1.0)
        self.assertEqual(child.q, 1.0)

    def test_terminal_not_evaluated_and_revisit_cached(self):
        game = play(CountingGame(), ((7, 7), (0, 0), (7, 6), (0, 2), (7, 5), (0, 4), (7, 4),
                                     (0, 6)))
        CountingGame.legal_calls = 0
        win = action(7, 3)
        evaluator = ScriptedEvaluator(weights={win: 100.0})
        n = 12
        _, root = search_with_tree(game, evaluator, cfg(n))
        visits = root.children[win].visit_count
        self.assertGreater(visits, 5)
        self.assertEqual(evaluator.calls, 1 + n - visits)
        self.assertEqual(CountingGame.legal_calls, 1 + n - visits)

    def test_no_legal_draw_value_zero(self):
        evaluator = UniformEvaluator()
        _, root = search_with_tree(near_full_draw(), evaluator, cfg(6))
        for child in root.children.values():
            self.assertEqual(child.to_play, BLACK)
            self.assertFalse(child.is_terminal)
            (grandchild,) = child.children.values()
            self.assertTrue(grandchild.is_terminal)
            self.assertEqual(grandchild.to_play, WHITE)  # turn already passed on draw
            self.assertEqual(grandchild.terminal_value, 0.0)
            self.assertEqual(grandchild.value_sum, 0.0)
        self.assertEqual(evaluator.calls, 3)  # root + two single-legal inner nodes

    def test_avoids_opponent_win(self):
        block = action(7, 8)
        others = (action(1, 1), action(13, 13))
        weights = {block: 5.0, others[0]: 5.0, others[1]: 5.0}
        evaluator = ScriptedEvaluator(weights=weights, default_weight=1e-4)
        result, root = search_with_tree(white_must_block(), evaluator, cfg(40))
        self.assertEqual(max(range(225), key=lambda a: result.visit_counts[a]), block)
        for other in others:
            self.assertLess(root.children[other].q, -0.5)

    def test_non_terminal_leaf_without_legal_moves_raises(self):
        class Broken(Game):
            def legal_moves(self):
                return [] if len(self.history) > 3 else super().legal_moves()
        game = play(Broken(), ((7, 7), (7, 8), (8, 8)))
        before = state(game)
        with self.assertRaises(RuntimeError):
            run_search(game, UniformEvaluator(), cfg(2))
        self.assertEqual(state(game), before)


class GameImmutabilityTest(unittest.TestCase):
    def test_game_restored_after_search(self):
        for game in (opened_game(), black_open_four(), near_full_draw(), white_must_block()):
            before = state(game)
            run_search(game, UniformEvaluator(), cfg(10))
            run_search(game, UniformEvaluator(), SearchConfig(num_simulations=5),
                       random.Random(1))
            self.assertEqual(state(game), before)

    def test_game_unchanged_after_validation_failure(self):
        game = opened_game()
        before = state(game)
        calls = []

        def script(snapshot):
            calls.append(snapshot)
            return EvaluationResult((0.0,) * 225, 0.0) if len(calls) == 3 else None
        with self.assertRaises(EvaluatorOutputError):
            run_search(game, ScriptedEvaluator(script=script), cfg(5))
        self.assertEqual(state(game), before)
        with self.assertRaises(EvaluatorOutputError):
            run_search(game, ScriptedEvaluator(value=2.0), cfg(5))
        self.assertEqual(state(game), before)


if __name__ == '__main__':
    unittest.main()
