"""H7-E3 tooling: TB-Engine-v2 as an opponent, the value arm on it, d2 --table / --legacy-frozen, the P92 guard
pair and the guard v2 summary (the H6-6 rules, unchanged)."""
import json
import unittest

from analysis.tb_engine_v2 import TB_ENGINE_V2
from scripts import h6_5d
from scripts import run_mcts_v8_benchmark as bench
from scripts import run_s1_probes as s1
from scripts import summarize_h6_6 as h66
from scripts import summarize_h7_e3 as e3
from tests.tb_engine_v1_regression import WEB, hashed_policy

MOVES = json.loads(WEB.read_text(encoding='utf-8'))['moves']


class ArmsTest(unittest.TestCase):
    def test_arms_and_opponent(self):
        self.assertEqual(bench.v8_config('puct_policy_vct2_50k_tt', {'puct_c': 1.5}), {**TB_ENGINE_V2, 'root_vcf_table': True})
        self.assertEqual(bench.v8_config('puct_value_vct2_50k_tt', {'puct_c': 1.5}),
                         {**TB_ENGINE_V2, 'tree_mode': 'puct_value', 'root_vcf_table': True})
        self.assertEqual(bench.parse_opponent('v8:puct_policy_vct2_50k_tt'), 'v8:puct_policy_vct2_50k_tt')
        with self.assertRaises(ValueError):
            bench.parse_opponent('v8:puct_value_vct2_50k_tt')
        saved = bench.load_policy
        bench.load_policy = lambda checkpoint: hashed_policy
        try:
            opp = bench.make_opponent('v8:puct_policy_vct2_50k_tt', 1, {'puct_c': 1.5}, 'x')
            plain = bench.make_opponent('v8:puct_policy_vct2_50k', 1, {'puct_c': 1.5}, 'x')
        finally:
            bench.load_policy = saved
        self.assertTrue(opp.root_vcf_table)
        self.assertFalse(plain.root_vcf_table)


class D2Test(unittest.TestCase):
    def test_table_check_and_legacy(self):
        on = h6_5d.check_status(MOVES[:94], MOVES[94], h6_5d.STANDARD, table=True)
        off = h6_5d.check_status(MOVES[:94], MOVES[94], h6_5d.STANDARD)
        self.assertEqual((on['status'], off['status']), ('UNSAFE', 'UNSAFE'))
        self.assertLessEqual(on['nodes'], off['nodes'])
        task = {'id': 'x', 'arm': 'a', 'set': 'control', 'game_key': 'k', 'game_kind': 'loss', 'ply': 94,
                'moves': MOVES[:94], 'tree_move': MOVES[94], 'recorded_status': 'UNSAFE',
                'standard': h6_5d.STANDARD, 'large': h6_5d.STANDARD, 'table': True, 'legacy': True}
        row = h6_5d.d2_task(task)
        self.assertTrue(row['table'])
        self.assertEqual((row['replay'], row['legacy_large']['status']), ('exact', 'UNSAFE'))
        self.assertNotIn('legacy_large', h6_5d.d2_task({**task, 'legacy': False}))


def p92(far):
    return {'chose_far': far, 'chose_region': not far, 'root_has_region': True, 'region_beats_far_rank': not far,
            'region_beats_far_visits': not far, 'both_present': True, 'far_absent': False,
            'region_strictly_beats_far': not far, 'region_visit_share': 0.5, 'far_visit_share': 0.2}


class P92Test(unittest.TestCase):
    def test_guard_for_the_new_pair(self):
        records = [{'arm': arm, 'probe': 'P92', 'route': 'tree', 'played': [3, 4] if far else [5, 5],
                    'root_value_spread': {'saturated_visited': None}, 'p92': p92(far)}
                   for arm, fars in (('puct_policy_vct2_50k_tt', (True, False)), ('puct_value_vct2_50k_tt', (False, False)))
                   for far in fars]
        out = s1.summarize(records)
        self.assertEqual(out['h7_e3_p92_guard']['verdict'], 'PASS')
        self.assertNotIn('h6_5_p92_guard', out)


def row(seeds, score=0.6, low=0.52, black=0.6, white=0.6, time_ratio=1.0, violations=()):
    return {'seeds': seeds, 'score': score, 'ci95_pairs': [low, 0.7], 'by_colour': {'black': black, 'white': white},
            'time_ratio': time_ratio, 'safety_violations': list(violations)}


def proofs(unsafe='PASS', unknown='PASS'):
    return {'UNSAFE': {'target': 'UNSAFE', 'verdict': unsafe, 'ci95': [0.0, 0.02]},
            'UNKNOWN': {'target': 'UNKNOWN', 'verdict': unknown, 'ci95': [0.0, 0.02]}}


class GuardTest(unittest.TestCase):
    def test_rules_are_guard_v2(self):
        self.assertEqual(e3.RULES, h66.RULES)
        self.assertEqual(e3.RULES['proof_margin'], 0.03)

    def test_same_decisions_as_h6_6(self):
        """Every case of tests/test_h6_6_summary.py, on the E3 seeds, decides the same."""
        cases = [({}, 'PASS'), ({'score': 0.5}, 'PASS'), ({'score': 0.49}, 'PASS'), ({'time_ratio': 1.51}, 'PASS'),
                 ({'white': 0.44}, 'PASS'), ({}, 'FAIL'), ({}, 'NOT_DISCRIMINATING'), ({'score': 0.54}, 'PASS'),
                 ({'low': 0.5}, 'PASS'), ({}, None)]
        for stage66, stage in zip(h66.STAGES, e3.STAGES):
            for main, p92 in cases:
                for p in (proofs(), proofs('REJECT'), proofs(unknown='INCONCLUSIVE'), proofs('REJECT', 'REJECT')):
                    want = h66.decide(row(stage66, **main), {h66.CANDIDATE: row(stage66), h66.BASELINE: row(stage66)},
                                      p, p92)['decision']
                    got = e3.decide(row(stage, **main), {e3.CANDIDATE: row(stage), e3.BASELINE: row(stage)},
                                    p, p92)['decision']
                    self.assertEqual(got, {'RUN_8405': 'RUN_8408'}.get(want, want), (stage, main, p92, p))
        self.assertEqual(e3.decide(row([8404]), {e3.CANDIDATE: row([8404]), e3.BASELINE: row([8404])},
                                   proofs(), 'PASS')['decision'], 'INCOMPLETE')

    def test_proof_status_needs_the_table(self):
        run = {'arm': e3.CANDIDATE, 'games': []}
        good = {'table': True, 'legacy_frozen': True, 'fast': {'requested': 'fast3'}, 'rows': []}
        self.assertEqual(e3.proof_status([run], [good]), {})
        with self.assertRaises(ValueError):
            e3.proof_status([run], [{**good, 'table': False}])
        with self.assertRaises(ValueError):
            e3.check_run({'arm': e3.BASELINE, 'opponent': 'v8:full', 'seed': 8407, 'v8_config': {'puct_c': 1.5},
                          'fast': {'requested': 'fast3'}}, (e3.CANDIDATE, e3.BASELINE), 'v8:full')


if __name__ == '__main__':
    unittest.main()
