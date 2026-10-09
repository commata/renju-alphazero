"""E2 opponent: selective depth-2 own attack (analysis.vct2_attack) and its benchmark wiring."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from analysis.mcts_v8 import SearchDiagnostics, _own_vct_attack
from analysis.threats import UNSAFE, ThreatSolver, decision_status
from analysis.vct2_attack import ROUTE, VCT2_ATTACK_BUDGET, VCT2AttackAgent, own_vct2_attack
from renju import Game
from scripts.run_mcts_v8_benchmark import build_tasks, opponent_arm, opponent_config, parse_opponent, summarize
from search.mcts_v5 import _RootContext

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json'


def p93() -> Game:
    """White to move after black 93 (3,4): a proven depth-2 win for white, witness white 94 = (11, 12)."""
    game = Game()
    for move in json.loads(PROBE.read_text(encoding='utf-8'))['moves'][:93]:
        game.play(*move)
    return game


class FakeBase:
    def __init__(self, move, route):
        self.name, self.move, self.route, self.diagnostics, self.calls = 'fake', move, route, SearchDiagnostics(), 0

    def select_move(self, game):
        self.calls += 1
        self.diagnostics.v8_route = self.route
        return self.move


class AttackTest(unittest.TestCase):
    def test_p93_witness_found_where_v8b_finds_none(self):
        game = p93()
        before = len(game.history)
        diag = SearchDiagnostics()
        self.assertIsNone(_own_vct_attack(game, _RootContext(game.legal_moves(), diag), diag,
                                          node_limit=20_000, call_limit=10_000, node_budget=200_000))
        agent = VCT2AttackAgent(FakeBase((0, 14), 'tree'))
        move = agent.select_move(game)
        self.assertEqual(move, (11, 12))
        self.assertEqual(len(game.history), before)  # game restored
        self.assertEqual(agent.diagnostics.v8_route, ROUTE)
        self.assertEqual((agent.attack_info['status'], agent.attack_info['base_route']), ('WIN', 'tree'))
        self.assertLessEqual(agent.attack_info['nodes'], VCT2_ATTACK_BUDGET['node_budget'])
        # The selective proof is a proof: the full solver (every quiet move) agrees.
        game.play(*move)
        try:
            counts, _ = ThreatSolver(node_limit=20_000).decision(game, 1, stop_at_safe=True)
        finally:
            game.undo()
        self.assertEqual(decision_status(counts), UNSAFE)

    def test_no_attack_keeps_the_base_move(self):
        game = Game()
        for move in ((7, 7), (7, 8), (8, 7)):
            game.play(*move)
        move, info = own_vct2_attack(game)
        self.assertIsNone(move)
        self.assertEqual(info['status'], '')
        agent = VCT2AttackAgent(FakeBase((6, 6), 'tree'))
        self.assertEqual(agent.select_move(game), (6, 6))
        self.assertEqual(agent.diagnostics.v8_route, 'tree')

    def test_routes_with_an_own_win_or_a_forced_move_are_left_alone(self):
        game = p93()
        for route in ('stage1', 'stage2', 'stage3', 'own_vcf', 'own_vct'):
            agent = VCT2AttackAgent(FakeBase((0, 14), route))
            self.assertEqual(agent.select_move(game), (0, 14))
            self.assertEqual(agent.attack_info, {})


class BenchmarkWiringTest(unittest.TestCase):
    def test_opponent_name(self):
        self.assertEqual(parse_opponent('v8:full+vct2atk'), 'v8:full+vct2atk')
        self.assertEqual(opponent_arm('v8:full+vct2atk'), 'full')
        for bad in ('v7+vct2atk', 'v8:nope+vct2atk', 'v8:+vct2atk'):
            with self.assertRaises(ValueError):
                parse_opponent(bad)
        config = opponent_config('v8:full+vct2atk')
        self.assertEqual(config['vct2_attack']['budget'], VCT2_ATTACK_BUDGET)
        self.assertEqual(VCT2_ATTACK_BUDGET['node_budget'], 200_000)  # E2, fixed in §12.28
        e2b = opponent_config('v8:full+vct2atk400k')['vct2_attack']['budget']
        self.assertEqual(e2b, {**VCT2_ATTACK_BUDGET, 'node_budget': 400_000})
        self.assertEqual(opponent_arm('v8:full+vct2atk400k'), 'full')
        for bad in ('v8:full+vct2atk0k', 'v8:full+vct2atk400', 'v8:full+vct2atkk'):
            with self.assertRaises(ValueError):
                parse_opponent(bad)
        self.assertNotIn('vct2_attack', opponent_config('v8:full'))
        self.assertIsNone(opponent_config('v7'))

    def test_keys_are_separate_and_pairing_is_kept(self):
        def args(opponent):
            return SimpleNamespace(arm='full', opponent=opponent, pairs=1, seed=8413, opening_random_plies=2,
                                   opening_radius=2, counterfactual=False, search_overrides={},
                                   policy_checkpoint=None)
        base, base_openings = build_tasks(args('v8:full'))
        tasks, openings = build_tasks(args('v8:full+vct2atk'))
        e2b, _ = build_tasks(args('v8:full+vct2atk400k'))
        self.assertEqual(e2b[0]['key'], 'full@v8:full+vct2atk400k/8413/0/black')  # E2b never resumes E2 lines
        self.assertEqual(openings, base_openings)
        self.assertEqual([t['seed'] for t in tasks], [t['seed'] for t in base])
        self.assertEqual(tasks[0]['key'], 'full@v8:full+vct2atk/8413/0/black')

    def test_summary_counts_punished_vct2_losses(self):
        def v8_move(ply, checked):
            return {'ply': ply, 'seconds': 0.1, 'route': 'tree', 'changed': False, 'v7_move': None,
                    'attack': {'candidates': 0, 'calls': 0, 'exhausted': False, 'seconds': 0.0},
                    'vct': {'checked': [], 'widened': False, 'exhausted': False, 'seconds': 0.0},
                    'root': {'checked': [], 'exhausted': False, 'seconds': 0.0},
                    'vct2': {'checked': checked, 'switched': False, 'nodes': 1, 'seconds': 0.1}}
        moves = [[7, 7], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
        game = {'key': 'k', 'v8_color': 'white', 'result': 'loss', 'winner': 1, 'length': 6, 'moves': moves,
                'v8_moves': [v8_move(1, [[[1, 1], 'UNSAFE']]),            # kept a proven loss, punished
                             v8_move(3, [[[3, 3], 'UNSAFE']]),            # kept, not punished
                             v8_move(5, [[[5, 5], 'UNSAFE']])],           # kept, game over next
                'opponent_move_seconds': [0.1, 0.1],
                'opponent_vct2_attack': [
                    {'ply': 2, 'route': 'own_vct2', 'base_route': 'tree', 'status': 'WIN', 'exhausted': False,
                     'seconds': 1.0, 'move_seconds': 2.0},
                    {'ply': 4, 'route': 'tree', 'base_route': 'tree', 'status': '', 'exhausted': True,
                     'seconds': 3.0, 'move_seconds': 4.0}]}
        summary = summarize([game])
        self.assertEqual(summary['v8_played_vct2_lost'],
                         {'moves': 3, 'opponent_next': {'own_vct2': 1, 'other': 1, 'game_over': 1}})
        attack = summary['opponent_vct2_attack']
        self.assertEqual((attack['ran'], attack['wins'], attack['budget_exhausted']), (2, 1, 1))
        self.assertEqual(attack['wins_by_base_route'], {'tree': 1})



class E2CompareTest(unittest.TestCase):
    def _runs(self, arm, result=None, opponent='v8:full+vct2atk'):
        runs = []
        for seed in (8401, 8402):
            run = json.loads((ROOT / 'docs' / 'mcts-v8-results' / f'h5_policy_{seed}.json').read_text(encoding='utf-8'))
            run['arm'], run['opponent'] = arm, opponent
            for game in run['games']:
                if result is not None:
                    game['result'] = result
            runs.append(run)
        return runs

    def test_reading(self):
        from scripts.s3_compare import compare
        base = self._runs('puct_policy')
        same = compare(base, self._runs('puct_policy_vct2'), opponent='v8:full+vct2atk')
        self.assertNotIn('decision', same)
        self.assertEqual(same['e2']['reading'], 'NOT_ESTABLISHED')
        self.assertEqual(same['e2']['exposure'], {'opponent_vct2_wins': 0, 'minimum': 10, 'low_power': True})
        self.assertIn('move_seconds_total', same['e2']['opponent_time']['vct2'])
        self.assertEqual(same['e2']['punishment']['vct2']['opponent_vct2_wins'], 0)
        better = compare(base, self._runs('puct_policy_vct2', result='win'), opponent='v8:full+vct2atk')
        self.assertEqual(better['e2']['reading'], 'EFFICACY')
        worse = compare(base, self._runs('puct_policy_vct2', result='loss'), opponent='v8:full+vct2atk')
        self.assertEqual(worse['e2']['reading'], 'HARM')

    def test_opponent_must_match(self):
        from scripts.s3_compare import compare
        with self.assertRaises(ValueError):
            compare(self._runs('puct_policy', opponent='v8:full'), self._runs('puct_policy_vct2'),
                    opponent='v8:full+vct2atk')


if __name__ == '__main__':
    unittest.main()
