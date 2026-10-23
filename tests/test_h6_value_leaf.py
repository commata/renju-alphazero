"""H6-4: value-leaf PUCT (analysis.puct_value_v1), its routing in analysis.mcts_v8, TB-V1-Value and the loader."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis import puct_value_v1 as pv
from analysis.mcts_v8_agent import MCTSV8Agent
from analysis.puct_v8 import PUCTNode
from analysis.tb_engine_v1 import TB_ENGINE_V1, config_hash as tb_hash
from analysis.tb_v1_value import TB_V1_VALUE, VALUE_CHECKPOINT_SHA256, config_hash, make_agent
from renju import BLACK, WHITE, Game
from tests.tb_engine_v1_regression import FIXTURE, PLIES, decide, hashed_policy

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None

ROOT = Path(__file__).resolve().parents[1]
WEB = json.loads((ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json').read_text(encoding='utf-8'))


def play(moves) -> Game:
    game = Game()
    for move in moves:
        game.play(*move)
    return game


def by_last_move(table, default=0.0):
    """A stand-in leaf: v (side to move's view) looked up by the last move played."""
    return lambda game: table.get(tuple(game.history[-1]), default)


def search(game, root_moves, leaf, simulations=4):
    return pv.search_tree_puct_value(game, root_moves, simulations, c_puct=1.5, prior='uniform', leaf_value=leaf,
                                     candidate_limit=8, neighborhood_radius=2, priority_top_k=8)


class ExactLeafTest(unittest.TestCase):
    def test_own_five_wins(self):
        # black to move with a four on row 7, closed at (7, 6): one five at (7, 11)
        game = play([(7, 7), (7, 6), (7, 8), (0, 0), (7, 9), (0, 2), (7, 10), (14, 14)])
        self.assertEqual(game.to_play, BLACK)
        self.assertEqual(pv.exact_leaf(game), 1.0)

    def test_one_opponent_five_is_left_to_the_value(self):
        # black to move, white's four is closed at (3, 2): one five, blockable, so no shortcut
        game = play([(7, 7), (3, 3), (3, 2), (3, 4), (0, 0), (3, 5), (0, 2), (3, 6)])
        self.assertEqual(game.to_play, BLACK)
        self.assertIsNone(pv.exact_leaf(game))

    def test_opponent_double_five_loses(self):
        # black to move, white has an open four (two fives): a two-move forced win for white
        game = play([(7, 7), (3, 3), (0, 0), (3, 4), (0, 2), (3, 5), (0, 4), (3, 6)])
        self.assertEqual(game.to_play, BLACK)
        self.assertEqual(pv.exact_leaf(game), -1.0)


class BackupTest(unittest.TestCase):
    def test_parent_child_signs(self):
        root = PUCTNode(parent=None, move=None, player_just_moved=None)
        child = PUCTNode(parent=root, move=(0, 0), player_just_moved=BLACK)
        grandchild = PUCTNode(parent=child, move=(0, 1), player_just_moved=WHITE)
        pv.backup(grandchild, 0.6, BLACK)  # +0.6 for black (black to move at the leaf)
        self.assertAlmostEqual(grandchild.value_sum, -0.6)  # white moved into it
        self.assertAlmostEqual(child.value_sum, 0.6)        # black moved into it
        self.assertEqual((root.visits, child.visits, grandchild.visits), (1, 1, 1))
        pv.backup(grandchild, 0.0, WHITE)  # a draw adds nothing
        self.assertAlmostEqual(child.value_sum, 0.6)
        self.assertEqual(child.visits, 2)


class SearchSignTest(unittest.TestCase):
    def test_the_child_whose_leaf_is_bad_for_the_opponent_wins(self):
        game = play(WEB['moves'][:4])
        a, b = (5, 5), (5, 6)
        self.assertTrue({a, b} <= set(game.legal_moves()))
        # after a the opponent (to move) is losing (-0.8), after b winning (+0.8)
        move, children = search(game, [a, b], by_last_move({a: -0.8, b: 0.8}))
        self.assertEqual(move, a)
        q = {c.move: c.mean_value for c in children}
        self.assertGreater(q[a], 0)
        self.assertLess(q[b], 0)

    def test_a_sign_bug_would_be_caught(self):
        """The same search with the value's sign flipped picks the other move, so the test above bites."""
        game = play(WEB['moves'][:4])
        a, b = (5, 5), (5, 6)
        move, _ = search(game, [a, b], by_last_move({a: 0.8, b: -0.8}))
        self.assertEqual(move, b)

    def test_immediate_win_is_taken_with_any_value(self):
        game = play([(7, 7), (7, 6), (7, 8), (0, 0), (7, 9), (0, 2), (7, 10), (14, 14)])
        # (a leaf of -1 would make the other moves wins too, so it is not used here)
        for leaf in (lambda g: 0.0, lambda g: 1.0, lambda g: 0.5):
            move, children = search(game, [(3, 3), (7, 11), (10, 10)], leaf, simulations=6)
            self.assertEqual(move, (7, 11))
            self.assertEqual({c.move: c.mean_value for c in children}[(7, 11)], 1.0)

    def test_constant_value_follows_the_prior(self):
        game = play(WEB['moves'][:4])
        moves = [(5, 5), (5, 6), (6, 5)]
        self.assertTrue(set(moves) <= set(game.legal_moves()))
        prior = lambda g: {m: (0.7 if m == (5, 6) else 0.15) for m in g.legal_moves()}  # noqa: E731
        move, children = pv.search_tree_puct_value(
            game, moves, 20, c_puct=1.5, prior='policy', policy=prior, leaf_value=lambda g: 0.0,
            candidate_limit=8, neighborhood_radius=2, priority_top_k=8)
        self.assertEqual(move, (5, 6))

    def test_value_outside_range_is_refused(self):
        with self.assertRaises(ValueError):
            search(play(WEB['moves'][:4]), [(5, 5)], lambda g: 1.5)

    def test_leaf_value_is_required(self):
        with self.assertRaises(ValueError):
            pv.search_tree_puct_value(play(WEB['moves'][:20]), [(5, 5)], 2, c_puct=1.5, prior='uniform',
                                      candidate_limit=8, neighborhood_radius=2, priority_top_k=8)


class RoutingTest(unittest.TestCase):
    def test_tb_engine_v1_puct_path_is_unchanged(self):
        """Decision fields on fixed positions equal those recorded before the engine-version step."""
        want = json.loads(FIXTURE.read_text(encoding='utf-8'))
        self.assertEqual([decide(p) for p in PLIES], want)

    def test_candidate_config(self):
        self.assertEqual({k for k in TB_V1_VALUE if TB_V1_VALUE[k] != TB_ENGINE_V1[k]}, {'tree_mode'})
        self.assertEqual(TB_V1_VALUE['tree_mode'], 'puct_value')
        self.assertEqual(len(config_hash()), 64)
        self.assertNotEqual(config_hash(), tb_hash())
        from scripts.run_mcts_v8_benchmark import v8_config
        self.assertEqual(v8_config('puct_value_vct2_50k', {'puct_c': 1.5}), TB_V1_VALUE)
        self.assertEqual(v8_config('puct_policy_vct2_50k', {'puct_c': 1.5}), TB_ENGINE_V1)

    def test_leaf_value_is_required_exactly_with_the_value_tree(self):
        with self.assertRaises(ValueError):
            MCTSV8Agent(root_policy=hashed_policy, **TB_V1_VALUE)
        with self.assertRaises(ValueError):
            MCTSV8Agent(root_policy=hashed_policy, leaf_value=lambda g: 0.0, **TB_ENGINE_V1)

    def test_candidate_agent_runs_the_value_leaf(self):
        calls = []
        agent = make_agent(hashed_policy, lambda g: calls.append(1) or 0.1, seed=7)
        game = play(WEB['moves'][:20])
        move = agent.select_move(game)
        d = agent.diagnostics
        self.assertIn(move, game.legal_moves())
        self.assertEqual((d.v8_route, d.v8_tree_mode), ('tree', 'puct_value'))
        self.assertEqual(d.v8_value_calls, len(calls))
        self.assertGreater(d.v8_value_calls, 0)
        self.assertEqual(d.v8_value_calls + d.v8_value_exact_leaves + d.v8_value_terminal_leaves,
                         d.v8_tree_simulations)
        self.assertEqual(d.v8_value_black_leaves + d.v8_value_white_leaves, d.v8_value_calls)
        self.assertGreater(d.v8_puct_nn_calls, 0)  # the H3 prior is still used


class RunnerTest(unittest.TestCase):
    def test_value_arm_is_not_an_opponent(self):
        from scripts.run_mcts_v8_benchmark import needs_policy, needs_value, parse_opponent, v8_config
        with self.assertRaises(ValueError):
            parse_opponent('v8:puct_value_vct2_50k')
        self.assertEqual(parse_opponent('v8:puct_policy_vct2_50k'), 'v8:puct_policy_vct2_50k')
        config = v8_config('puct_value_vct2_50k')
        self.assertTrue(needs_value(config) and needs_policy(config))

    def test_value_arm_needs_the_checkpoint(self):
        from scripts.run_mcts_v8_benchmark import main
        with TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                main(['--arm', 'puct_value_vct2_50k', '--puct-c', '1.5', '--policy-checkpoint', 'missing.pt',
                      '--output', str(Path(tmp, 'o.json'))])
            with self.assertRaises(SystemExit):
                main(['--arm', 'full', '--value-checkpoint', 'x.pt', '--output', str(Path(tmp, 'o.json'))])

    def test_root_q(self):
        from scripts.run_mcts_v8_benchmark import root_q
        self.assertIsNone(root_q([((0, 0), 0, 0.0)]))
        q = root_q([((0, 0), 3, 1.0), ((0, 1), 2, 0.5), ((0, 2), 0, 0.0)])
        self.assertEqual((q['children'], q['range'], q['saturated']), (2, 0.5, 0.5))

    def test_p92_guard(self):
        from scripts.run_s1_probes import summarize

        def runs(arm, far):
            return [{'probe': 'P92', 'arm': arm, 'route': 'tree', 'played': [3, 4] if i < far else [12, 12],
                     'root_value_spread': {'saturated_visited': None},
                     'p92': {'chose_far': i < far, 'chose_region': False, 'root_has_region': True,
                             'region_beats_far_rank': True, 'region_beats_far_visits': True, 'both_present': True,
                             'far_absent': False, 'region_strictly_beats_far': True, 'region_visit_share': 0.1,
                             'far_visit_share': 0.1}} for i in range(10)]

        def guard(base, cand):
            return summarize(runs('puct_policy_vct2_50k', base) + runs('puct_value_vct2_50k', cand))['h6_5_p92_guard']

        self.assertEqual(guard(0, 0)['verdict'], 'PASS')
        self.assertEqual(guard(2, 1)['verdict'], 'PASS')
        self.assertEqual(guard(0, 1)['verdict'], 'FAIL')
        self.assertEqual(guard(10, 10)['verdict'], 'NOT_DISCRIMINATING')
        self.assertEqual(guard(0, 0)['p93_p94'], 'diagnostic_only')


class LeafCheckTest(unittest.TestCase):
    def test_check_runs_both_engines(self):
        from scripts import h6_value_leaf_check as check
        source = ROOT / 'docs' / 'mcts-v8-results' / 'e1h' / 'puct_policy_vct2_50k_8417.json'
        with TemporaryDirectory() as tmp:
            out = Path(tmp, 'check.json')
            factories = {'policy': hashed_policy, 'leaf': lambda g: 0.0}
            self.assertEqual(check.main(['--games', str(source), '--per-game', '1', '--limit', '2', '--allow-dirty',
                                         '--output', str(out)], factories=factories), 0)
            result = json.loads(out.read_text(encoding='utf-8'))
        s = result['summary']
        self.assertEqual(s['positions'], 2)
        self.assertGreater(s['tree_route_share'], 0)
        self.assertGreater(s['value']['calls'], 0)
        self.assertIsNone(result['verdict'])


@unittest.skipIf(torch is None, 'requires torch')
class ValueLeafTest(unittest.TestCase):
    def _checkpoint(self, root: Path) -> Path:
        from model.checkpoint import save_checkpoint
        from tests.test_h6_cross_eval import MODEL_CONFIG
        from tests.test_h6_value import h3_model
        save_checkpoint(root / 'best.pt', h3_model(seed=12))
        (root / 'best.json').write_text(json.dumps({'stage': 'H6-2a', 'value_trained': True, 'labels_sha256': 'x',
                                                    'model_config': MODEL_CONFIG}), encoding='utf-8')
        return root / 'best.pt'

    def test_refuses_another_checkpoint(self):
        from hybrid.h6_value_leaf import V3_CHECKPOINT_SHA256, ValueLeaf
        self.assertEqual(V3_CHECKPOINT_SHA256, VALUE_CHECKPOINT_SHA256)
        with TemporaryDirectory() as tmp:
            path = self._checkpoint(Path(tmp))
            with self.assertRaises(ValueError):
                ValueLeaf(path)
            meta = json.loads(path.with_name('best.json').read_text(encoding='utf-8'))
            path.with_name('best.json').write_text(json.dumps({**meta, 'stage': 'H3'}), encoding='utf-8')
            with self.assertRaises(ValueError):
                ValueLeaf(path, expected_sha256=None)

    def test_d4_invariant_cached_and_in_range(self):
        from hybrid.h6_value_leaf import ValueLeaf
        with TemporaryDirectory() as tmp:
            leaf = ValueLeaf(self._checkpoint(Path(tmp)), expected_sha256=None)
            moves = WEB['moves'][:21]
            v = leaf(play(moves))
            mirrored = leaf(play([(r, 14 - c) for r, c in moves]))
            rotated = leaf(play([(c, 14 - r) for r, c in moves]))
            self.assertTrue(-1.0 <= v <= 1.0)
            self.assertAlmostEqual(v, mirrored, places=5)
            self.assertAlmostEqual(v, rotated, places=5)
            self.assertEqual(leaf(play(moves)), v)
            self.assertEqual((leaf.calls, leaf.cache_hits), (4, 1))


if __name__ == '__main__':
    unittest.main()
