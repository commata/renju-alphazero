"""E3 arms (analysis.e3_arms), their benchmark wiring, the dev replay cases and the E3 reading (§12.32)."""
import json
from pathlib import Path
import unittest

from analysis.e3_arms import (
    E3_A, STAGE_VCT2_BUDGET, StageVCT2Agent, make_agent, selective_status, stage_vct2_veto,
)
from analysis.mcts_v8 import V8_DEFAULTS, SearchDiagnostics
from analysis.s3_vct2_v1 import S3_VCT2_V1
from analysis.threats import UNSAFE, UNKNOWN
from renju import Game
from scripts.run_mcts_v8_benchmark import ARMS, _stage_vct2_summary, _vct2_cost, v8_config

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'docs' / 'mcts-v8-results'


def uniform(game):
    legal = game.legal_moves()
    return {m: 1.0 / len(legal) for m in legal}


def position(manifest, key):
    entry = next(p for p in json.loads((RESULTS / manifest).read_text(encoding='utf-8'))['positions'] if p['key'] == key)
    game = Game()
    for move in entry['history']:
        game.play(*move)
    return entry, game


class ConfigTest(unittest.TestCase):
    def test_e3a_changes_only_the_budget(self):
        self.assertEqual({k for k in E3_A if E3_A[k] != S3_VCT2_V1[k]}, {'vct2_node_budget'})
        self.assertEqual(E3_A['vct2_node_budget'], 50_000)
        self.assertEqual(STAGE_VCT2_BUDGET, {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': 10_000})

    def test_benchmark_arms(self):
        base = v8_config('puct_policy_vct2', {'puct_c': 1.5})
        self.assertEqual(base, S3_VCT2_V1)  # S3-VCT2-v1 is the E3 baseline arm
        e3a = v8_config('puct_policy_vct2_50k', {'puct_c': 1.5})
        self.assertEqual(e3a, E3_A)
        e3s = v8_config('puct_policy_vct2_stage', {'puct_c': 1.5})
        self.assertEqual({k: v for k, v in e3s.items() if k in V8_DEFAULTS}, S3_VCT2_V1)
        self.assertEqual(e3s['stage_vct2'], STAGE_VCT2_BUDGET)
        self.assertIn('puct_policy_vct2_stage', ARMS)


class FakeBase:
    candidate_limit, neighborhood_radius, name = 20, 2, 'fake'

    def __init__(self, move, route):
        self.move, self.route, self.diagnostics = move, route, SearchDiagnostics()

    def select_move(self, game):
        self.diagnostics.v8_route = self.route
        return self.move


class StageTest(unittest.TestCase):
    def test_other_routes_are_untouched(self):
        game = Game()
        game.play(7, 7)
        for route in ('tree', 'stage1', 'stage2', 'stage3', 'own_vcf', 'own_vct'):
            agent = StageVCT2Agent(FakeBase((6, 6), route))
            self.assertEqual(agent.select_move(game), (6, 6))
            self.assertEqual(agent.stage_info, {})

    def test_e1s_miss_is_rescued(self):
        # §12.31: V8-A played (8,6) (VCT1 SAFE, a depth-2 loss); (11,9) in the forced tier is VCT2_CLEAR.
        entry, game = position('e1s/e1s_manifest.json', '8412-p2-black/12')
        final, info = stage_vct2_veto(game, 'stage4', tuple(entry['played']), tuple(entry['v7_move']),
                                      entry['v8a_checked'])
        self.assertEqual(final, (11, 9))
        self.assertEqual((info['switched'], info['final_tier']), (True, 'forced'))
        self.assertEqual(info['checked'][0][:2], [[8, 6], UNSAFE])
        # The whole agent: the frozen S3 engine reaches the same V8-A move, the wrapper replaces it.
        agent = make_agent('e3s', uniform, seed=1)
        self.assertEqual(agent.select_move(game), (11, 9))
        self.assertEqual(agent.diagnostics.v8_route, 'stage4')


class BudgetTest(unittest.TestCase):
    def test_e3a_budget_catches_an_e1_dev_miss(self):
        # E1-dev 8411-p8 (7,8): 10k UNKNOWN, proven with 11,320 selective nodes (§12.30).
        _, game = position('e1/e1_manifest.json', '8411-p8-white/11')
        budget = {'node_limit': E3_A['vct2_node_limit'], 'call_limit': E3_A['vct2_call_limit'],
                  'node_budget': E3_A['vct2_node_budget']}
        self.assertEqual(selective_status(game, (7, 8), STAGE_VCT2_BUDGET)[0], UNKNOWN)
        status, nodes = selective_status(game, (7, 8), budget)
        self.assertEqual((status, nodes), (UNSAFE, 11_320))


class SummaryTest(unittest.TestCase):
    def test_cost_split(self):
        moves = [{'vct2': {'checked': [[[1, 1], 'UNSAFE'], [[2, 2], 'UNKNOWN']], 'check_nodes': [50_000, 30_000],
                           'nodes': 80_000, 'switched': True}},
                 {'vct2': {'checked': [[[3, 3], 'SAFE']], 'check_nodes': [100], 'nodes': 100, 'switched': False}},
                 {'vct2': {'checked': []}},
                 {'vct2': {'checked': [[[4, 4], 'SAFE']], 'nodes': 7, 'switched': False}}]  # an old record
        cost = _vct2_cost(moves)
        self.assertEqual((cost['moves_checked'], cost['replacement_checks'], cost['replacements']), (3, 1, 1))
        self.assertEqual((cost['nodes_selected']['n'], cost['nodes_replacements']['max']), (2, 30_000))
        stage = _stage_vct2_summary([{'stage_vct2': {'checked': [[[1, 1], 'UNSAFE', 900, 'played'],
                                                                 [[2, 2], 'SAFE', 50, 'widened']],
                                                     'switched': True, 'final_tier': 'widened', 'nodes': 950,
                                                     'seconds': 1.0}}, {}])
        self.assertEqual((stage['played_move_proven_lost'], stage['switched_by_tier']['widened']), (1, 1))
        self.assertIsNone(_stage_vct2_summary([{}]))


class DevReplayTest(unittest.TestCase):
    def test_cases(self):
        from scripts.e3_dev_replay import build_cases
        cases = build_cases()
        kinds = {k: sum(c['kind'] == k for c in cases) for k in ('e2-tree', 'e2-stage', 'e1-tree', 'e1s-miss')}
        self.assertEqual(kinds, {'e2-tree': 6, 'e2-stage': 5, 'e1-tree': 3, 'e1s-miss': 1})
        for c in cases:
            self.assertEqual(c['arm'], 'e3a' if c['kind'].endswith('tree') else 'e3s')
            self.assertNotIn('8415', c['key'])
            self.assertNotIn('8416', c['key'])

    def test_classify(self):
        from scripts.e3_dev_replay import classify
        final = lambda sel, full: {'final': {'selective_200k': sel, 'full': {'status': full}}}  # noqa: E731
        self.assertEqual(classify({'detected': False, 'switched': False}, final('SAFE', 'VCT2_CLEAR')), 'NOT_DETECTED')
        self.assertEqual(classify({'detected': True, 'switched': False}, final('UNSAFE', 'PROVEN_LOSS')), 'KEPT')
        self.assertEqual(classify({'detected': True, 'switched': True}, final('SAFE', 'VCT2_CLEAR')), 'RESCUED')
        self.assertEqual(classify({'detected': True, 'switched': True}, final('UNSAFE', 'UNKNOWN')), 'LOSS_TO_LOSS')
        self.assertEqual(classify({'detected': True, 'switched': True}, final('UNKNOWN', 'UNKNOWN')), 'SWITCH_UNRESOLVED')


class ReadingTest(unittest.TestCase):
    def test_reading(self):
        from scripts.e3_compare import reading
        ok = {'baseline': {'safety_violations': []}, 'arm': {'safety_violations': []}}
        d = lambda x, lo, hi: {'diff': x, 'ci95': [lo, hi]}  # noqa: E731
        self.assertEqual(reading(ok, d(0.05, 0.01, 0.1), 0, 3, {'RESCUED': 2}, 0), 'EFFICACY')
        self.assertEqual(reading(ok, d(0.02, -0.02, 0.06), 0, 3, {'RESCUED': 1}, 0),
                         'MECHANISM_ESTABLISHED_BUT_MATCH_UNPROVEN')
        self.assertEqual(reading(ok, d(0.02, -0.02, 0.06), 0, 3, {'RESCUED': 1, 'LOSS_TO_LOSS': 1}, 0),
                         'NOT_ESTABLISHED')
        self.assertEqual(reading(ok, d(0.0, -0.02, 0.02), 0, 3, {'RESCUED': 1}, 0), 'NOT_ESTABLISHED')
        self.assertEqual(reading(ok, d(0.02, -0.02, 0.06), 3, 3, {'RESCUED': 1}, 0), 'NOT_ESTABLISHED')
        self.assertEqual(reading(ok, d(-0.06, -0.1, 0.0), 0, 3, {}, 0), 'HARM')
        self.assertEqual(reading(ok, d(0.05, 0.01, 0.1), 0, 3, {}, 1), 'SAFETY_VIOLATION')


if __name__ == '__main__':
    unittest.main()
