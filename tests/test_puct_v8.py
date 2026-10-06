"""H5 (§12.16): PUCT tree on V8's action sets; only the prior differs between arms."""
from random import Random
import unittest

from analysis.mcts_v8 import V8_DEFAULTS, SearchDiagnostics, mcts_search_v8
from analysis.mcts_v8_agent import MCTSV8Agent
from analysis.puct_v8 import PUCTStats, child_priors, prior_entropy, search_tree_puct
from renju import Game
from search.mcts_v321 import _search_candidates_v321

TREE = dict(candidate_limit=20, neighborhood_radius=2, priority_top_k=8)
OFF = {**V8_DEFAULTS, 'stage_vct_safety': False, 'own_vct_attack': False, 'root_vct_safety': False}
# Black to move with a four on row 7 blocked at (7, 6): only (7, 11) makes five.
BLACK_FOUR = [[7, 7], [7, 6], [7, 8], [2, 5], [7, 9], [2, 12], [7, 10], [12, 2]]
# White has a four on row 3 blocked at (3, 2); black to move must block at (3, 7).
WHITE_FOUR = [[7, 7], [3, 3], [3, 2], [3, 4], [11, 11], [3, 5], [11, 1], [3, 6]]
QUIET = [[7, 7], [6, 8], [8, 8], [6, 6], [7, 9]]


def _game(moves):
    game = Game()
    for move in moves:
        game.play(*move)
    return game


class _FakePolicy:
    def __init__(self, scores=None):
        self.scores, self.calls = dict(scores or {}), 0

    def __call__(self, game):
        self.calls += 1
        return {m: self.scores.get(m, 1e-6) for m in game.legal_moves()}


class PriorTest(unittest.TestCase):
    MOVES = [(7, 6), (6, 7), (8, 9), (5, 5)]

    def test_priors_are_distributions_over_the_children(self):
        game = _game(QUIET)
        stats = PUCTStats()
        uniform = child_priors(game, self.MOVES, 'uniform', None, stats)
        heuristic = child_priors(game, self.MOVES, 'heuristic', None, stats)
        policy = child_priors(game, self.MOVES, 'policy', _FakePolicy({(5, 5): 3.0, (6, 7): 1.0}), stats)
        for priors in (uniform, heuristic, policy):
            self.assertAlmostEqual(sum(priors), 1.0)
        self.assertEqual(uniform, [0.25] * 4)
        self.assertEqual(heuristic, sorted(heuristic, reverse=True))  # 1/rank
        self.assertAlmostEqual(heuristic[0] / heuristic[1], 2.0)
        self.assertGreater(policy[3], 0.74)  # restricted to the children and renormalized
        self.assertEqual(stats.nn_calls, 1)

    def test_policy_without_mass_falls_back_to_uniform(self):
        stats = PUCTStats()
        zero = lambda game: {}  # noqa: E731
        self.assertEqual(child_priors(_game(QUIET), self.MOVES, 'policy', zero, stats), [0.25] * 4)
        self.assertEqual(stats.prior_fallbacks, 1)

    def test_entropy(self):
        self.assertAlmostEqual(prior_entropy([0.25] * 4), 1.0)
        self.assertEqual(prior_entropy([1.0]), 0.0)


class TreeTest(unittest.TestCase):
    def test_value_sign(self):
        # Values are stored for the player who moved into a node: the five is +1 for black,
        # the move that leaves white's five is -1 for black, whatever the prior says.
        _, children = search_tree_puct(_game(BLACK_FOUR), [(12, 12), (7, 11)], 20, c_puct=1.5,
                                       prior='heuristic', random=Random(1), **TREE)
        win = next(c for c in children if c.move == (7, 11))
        self.assertEqual((win.player_just_moved, win.mean_value), (1, 1.0))
        self.assertGreater(win.visits, 0)
        policy = _FakePolicy({(11, 13): 0.99, (3, 7): 0.01})
        _, children = search_tree_puct(_game(WHITE_FOUR), [(11, 13), (3, 7)], 40, c_puct=1.5,
                                       prior='policy', policy=policy, random=Random(2), **TREE)
        losing = next(c for c in children if c.move == (11, 13))
        self.assertEqual((losing.mean_value, round(losing.prior, 2)), (-1.0, 0.99))

    def test_uniform_prior_picks_by_value(self):
        for seed in range(3):
            move, _ = search_tree_puct(_game(BLACK_FOUR), [(12, 12), (7, 11)], 60, c_puct=1.5,
                                       prior='uniform', random=Random(seed), **TREE)
            self.assertEqual(move, (7, 11))
            move, _ = search_tree_puct(_game(WHITE_FOUR), [(11, 13), (3, 7)], 40, c_puct=1.5,
                                       prior='uniform', random=Random(seed), **TREE)
            self.assertEqual(move, (3, 7))

    def test_same_action_sets_for_every_prior(self):
        game = _game(QUIET)
        root = _search_candidates_v321(game, 20, 2)
        for prior, policy in (('uniform', None), ('heuristic', None), ('policy', _FakePolicy())):
            _, children = search_tree_puct(game, root, 30, c_puct=1.5, prior=prior, policy=policy,
                                           random=Random(4), **TREE)
            self.assertEqual([c.move for c in children], root)
            for child in children:
                if child.children:
                    after = _game(QUIET + [list(child.move)])
                    self.assertEqual([c.move for c in child.children], _search_candidates_v321(after, 20, 2))

    def test_does_not_touch_the_game(self):
        game = _game(QUIET)
        history = list(game.history)
        search_tree_puct(game, _search_candidates_v321(game, 20, 2), 10, c_puct=1.5, prior='uniform',
                         random=Random(5), **TREE)
        self.assertEqual(game.history, history)


class V8IntegrationTest(unittest.TestCase):
    def test_policy_prior_fails_fast_without_a_policy(self):
        with self.assertRaises(ValueError):
            MCTSV8Agent(tree_mode='puct', puct_prior='policy')
        with self.assertRaises(ValueError):
            mcts_search_v8(_game(QUIET), **{**OFF, 'tree_mode': 'puct', 'puct_prior': 'policy'}, random=Random(1))

    def test_puct_rejects_h4_options_and_bad_values(self):
        for bad in ({'root_policy_order': True}, {'root_policy_extra': 4}):
            with self.assertRaises(ValueError):
                MCTSV8Agent(tree_mode='puct', root_policy=_FakePolicy(), **bad)
        for bad in ({'tree_mode': 'alpha'}, {'puct_prior': 'nn'}, {'puct_c': 0}, {'puct_c': True}):
            with self.assertRaises(ValueError):
                MCTSV8Agent(**bad)

    def test_puct_tree_route_records_diagnostics(self):
        policy = _FakePolicy()
        diag = SearchDiagnostics()
        config = {**OFF, 'tree_mode': 'puct', 'puct_prior': 'policy', 'simulations': 12,
                  'tactical_simulations': 12}
        move = mcts_search_v8(_game(QUIET), **config, root_policy=policy, random=Random(3), diagnostics=diag)
        self.assertEqual(diag.v8_route, 'tree')
        self.assertEqual((diag.v8_tree_mode, diag.v8_puct_prior, diag.v8_tree_simulations), ('puct', 'policy', 12))
        self.assertIn(move, diag.root_candidates)
        self.assertEqual(policy.calls, diag.v8_puct_nn_calls)
        self.assertTrue(1 < diag.v8_puct_nn_calls <= 13)  # root + at most one per simulation
        self.assertIn(diag.v8_puct_prior_top, diag.root_candidates)
        self.assertGreater(diag.v8_tree_seconds, 0)

    def test_v5_tree_records_its_mode_only(self):
        diag = SearchDiagnostics()
        mcts_search_v8(_game(QUIET), **{**OFF, 'simulations': 8, 'tactical_simulations': 8},
                       random=Random(3), diagnostics=diag)
        self.assertEqual((diag.v8_tree_mode, diag.v8_puct_prior, diag.v8_puct_nn_calls), ('v5', '', 0))


class BenchmarkArgsTest(unittest.TestCase):
    def test_puct_arms_need_c_and_keys_carry_it(self):
        from types import SimpleNamespace
        from scripts.run_mcts_v8_benchmark import build_tasks, main, needs_policy, v8_config
        with self.assertRaises(SystemExit):
            main(['--arm', 'puct_heur', '--pairs', '1'])
        with self.assertRaises(SystemExit):
            main(['--arm', 'full', '--puct-c', '1.5', '--pairs', '1'])
        with self.assertRaises(SystemExit):
            main(['--arm', 'full', '--opponent', 'v8:puct_heur', '--pairs', '1'])
        self.assertTrue(needs_policy(v8_config('puct_policy')))
        self.assertFalse(needs_policy(v8_config('puct_heur')))
        args = SimpleNamespace(arm='puct_heur', opponent='v8:full', pairs=1, seed=11, opening_random_plies=2,
                               opening_radius=2, counterfactual=False, search_overrides={},
                               arm_overrides={'puct_c': 1.5}, policy_checkpoint=None)
        tasks, _ = build_tasks(args)
        self.assertEqual(tasks[0]['key'], 'puct_heur[puct_c=1.5]@v8:full/11/0/black')
        self.assertEqual(tasks[0]['v8_overrides'], {'puct_c': 1.5})
        self.assertEqual(tasks[0]['v7_overrides'], {})


class ScriptsTest(unittest.TestCase):
    def test_policy_arm_game_and_h5_summary(self):
        try:
            import torch
        except ImportError:
            self.skipTest('requires torch')
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from model.checkpoint import save_checkpoint
        from model.config import ModelConfig
        from model.network import PolicyValueNet
        from scripts.run_mcts_v8_benchmark import build_tasks, play_one
        from scripts.summarize_h5 import summarize, verdict
        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp, 'best.pt')
            torch.manual_seed(0)
            save_checkpoint(ckpt, PolicyValueNet(ModelConfig(channels=8, blocks=1)))
            ckpt.with_suffix('.json').write_text(json.dumps({
                'policy_trained': True, 'value_trained': False, 'step': 0,
                'config': {'channels': 8, 'blocks': 1}}), encoding='utf-8')
            args = SimpleNamespace(arm='puct_policy', opponent='v8:off', pairs=1, seed=11, opening_random_plies=2,
                                   opening_radius=2, counterfactual=False, search_overrides={},
                                   arm_overrides={'puct_c': 1.5}, policy_checkpoint=str(ckpt))
            tasks, _ = build_tasks(args)
            off = {'stage_vct_safety': False, 'own_vct_attack': False, 'root_vct_safety': False}
            task = {**tasks[0], 'v8_overrides': {**tasks[0]['v8_overrides'], 'simulations': 4,
                                                 'tactical_simulations': 6, **off},
                    'v7_overrides': {'simulations': 4, 'tactical_simulations': 6}}
            record = play_one(task)
        tree = [m for m in record['v8_moves'] if m['route'] == 'tree']
        self.assertTrue(tree)
        for m in tree:
            self.assertEqual((m['tree']['mode'], m['tree']['prior']), ('puct', 'policy'))
            self.assertGreaterEqual(m['tree']['nn_calls'], 1)
        row = summarize({'arm': 'puct_policy', 'seeds': [11], 'games': [{**record, 'run_seed': 11}]})
        self.assertEqual(row['tree_moves'], len(tree))
        self.assertEqual(row['tree_status']['UNCHECKED'], len(tree))  # V8-C off here
        self.assertIsNotNone(row['tree_chose_prior_top_rate'])
        self.assertEqual(verdict({'puct_policy': row}, {})['decision'], 'incomplete')

    def test_calibration_rule(self):
        from scripts.h5_calibrate_puct import decide
        ok = {'visited': 5, 'top_share': 0.5, 'policy_top': 0.3}
        table = {}
        for c in (0.5, 1.0, 2.0):
            table[('uniform', c)] = {**ok, 'policy_top': 0.2}
            table[('heuristic', c)] = dict(ok)
            table[('policy', c)] = dict(ok)
        table[('heuristic', 1.0)]['top_share'] = 0.9  # collapsed
        table[('policy', 2.0)]['policy_top'] = 0.22   # prior ignored
        chosen, reasons = decide(table, (0.5, 1.0, 2.0))
        self.assertEqual(chosen, 0.5)
        self.assertTrue(reasons['1.0'] and reasons['2.0'] and not reasons['0.5'])


if __name__ == '__main__':
    unittest.main()
