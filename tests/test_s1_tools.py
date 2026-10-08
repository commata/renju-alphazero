import json
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import unittest

from scripts.run_mcts_v8_benchmark import file_sha256, main as benchmark_main, provenance
from scripts.run_s1_probes import FAR_MOVE, main as probes_main, p92_metrics, root_value_spread
from scripts.s1_loss_analysis import causal_fields, primary_cause, signature_ply, signature_table
from scripts.s2_position_truth import DEFAULT_BUDGET, load_done, main as truth_main, summarize as truth_summary
from scripts.s2_policy_diag import describe
from scripts.s2_vct2_detector_eval import summarize as detector_summary
from scripts.s2_join_p92 import join as join_p92

SMOKE = ['--pairs', '1', '--seed', '3', '--simulations', '4', '--tactical-simulations', '8']


def _move(ply, route, played, root=(), vct=(), switch=''):
    return {'ply': ply, 'route': route, 'played': list(played),
            'root': {'checked': [[list(m), s] for m, s in root], 'exhausted': False, 'switch': switch},
            'vct': {'checked': [[list(m), s] for m, s in vct], 'exhausted': False}}


class ProvenanceTest(unittest.TestCase):
    def test_checkpoint_hashes(self):
        with TemporaryDirectory() as tmp:
            ckpt = Path(tmp, 'best.pt')
            ckpt.write_bytes(b'weights')
            ckpt.with_suffix('.json').write_text('{}', encoding='utf-8')
            info = provenance(ckpt)
            self.assertEqual(info['policy_checkpoint_sha256'], hashlib.sha256(b'weights').hexdigest())
            self.assertEqual(info['policy_metadata_sha256'], hashlib.sha256(b'{}').hexdigest())
            self.assertIsNone(file_sha256(Path(tmp, 'missing.pt')))
        self.assertIsNone(provenance(None)['policy_checkpoint_sha256'])

    def test_benchmark_output_records_provenance(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp, 'a.json')
            benchmark_main(['--arm', 'off', *SMOKE, '--output', str(out)])
            payload = json.loads(out.read_text(encoding='utf-8'))
            self.assertIn(payload['provenance']['git_dirty'], (True, False, None))
            self.assertIsNone(payload['provenance']['policy_checkpoint_sha256'])  # not a policy arm


class ProbeMetricsTest(unittest.TestCase):
    def test_region_against_far_move(self):
        children = [((10, 12), 6, -0.5), (FAR_MOVE, 4, -1.0), ((3, 3), 2, -1.0)]
        m = p92_metrics(children, (10, 12), priors={(10, 12): 0.3, FAR_MOVE: 0.01})
        self.assertTrue(m['root_has_region'] and m['region_beats_far_rank'] and m['region_beats_far_visits'])
        self.assertTrue(m['both_present'] and m['region_strictly_beats_far'])
        self.assertEqual(m['region_best'], [11, 13])
        self.assertEqual((m['region_best_rank'], m['far_rank']), (1, 2))
        self.assertFalse(m['chose_far'])
        self.assertEqual(m['region_best_prior'], 0.3)

    def test_far_move_preferred_and_region_missing(self):
        m = p92_metrics([(FAR_MOVE, 4, -1.0), ((3, 3), 4, -1.0)], FAR_MOVE)
        self.assertTrue(m['chose_far'])
        self.assertFalse(m['root_has_region'] or m['region_beats_far_rank'] or m['region_beats_far_visits'])
        self.assertIsNone(m['far_prior'])

    def test_value_spread(self):
        self.assertTrue(root_value_spread([((0, 0), 4, -1.0), ((0, 1), 3, -1.0)])['saturated_visited'])
        self.assertFalse(root_value_spread([((0, 0), 4, -1.0), ((0, 1), 3, 0.2)])['saturated_visited'])

    def test_unvisited_children_do_not_hide_saturation(self):
        # PUCT keeps unvisited children at Q = 0; P93 looked unsaturated because of them.
        spread = root_value_spread([((0, 0), 22, 1.0), ((0, 1), 13, 1.0), ((0, 2), 0, 0.0)])
        self.assertEqual((spread['range_all'], spread['range_visited']), (1.0, 0.0))
        self.assertTrue(spread['saturated_visited'])
        self.assertEqual((spread['visited'], spread['children']), (2, 3))

    def test_far_absent_is_not_a_strict_win(self):
        m = p92_metrics([((3, 3), 4, -1.0), ((9, 8), 3, -1.0)], (3, 3))
        self.assertTrue(m['region_beats_far_rank'] and m['far_absent'])  # the §12.19 gate's reading
        self.assertFalse(m['both_present'] or m['region_strictly_beats_far'])

    def test_probe_runner_smoke(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp, 'p.json')
            probes_main(['--arms', 'off', '--names', 'P92', '--seeds', '1', '--simulations', '4',
                         '--tactical-simulations', '8', '--output', str(out)])
            payload = json.loads(out.read_text(encoding='utf-8'))
            run = payload['runs'][0]
            self.assertEqual((run['probe'], run['arm']), ('P92', 'off'))
            self.assertEqual(len(run['played']), 2)
            self.assertIn('gate', payload['summary']['off']['P92'])


class LossAnalysisTest(unittest.TestCase):
    def test_signature(self):
        hit = [_move(10, 'tree', (1, 1), root=[((1, 1), 'SAFE')]),
               _move(12, 'stage4', (2, 2), vct=[((2, 2), 'UNSAFE'), ((3, 3), 'UNSAFE')])]
        self.assertEqual(signature_ply({'v8_moves': hit}), 10)
        safe_defence = [hit[0], _move(12, 'stage4', (2, 2), vct=[((2, 2), 'SAFE')])]
        self.assertIsNone(signature_ply({'v8_moves': safe_defence}))

    def test_signature_counts_openings_once(self):
        hit = [_move(10, 'tree', (1, 1), root=[((1, 1), 'SAFE')]),
               _move(12, 'stage4', (2, 2), vct=[((2, 2), 'UNSAFE')])]
        games = [{'pair': 0, 'v8_color': c, 'result': 'loss', 'moves': [], 'v8_moves': hit} for c in ('black', 'white')]
        games.append({'pair': 1, 'v8_color': 'black', 'result': 'win', 'moves': [], 'v8_moves': []})
        row = signature_table([{'arm': 'x', 'seed': 1, 'games': games}])['x']
        self.assertEqual((row['affected_games'], row['games']), (2, 3))
        self.assertEqual((row['affected_openings'], row['unique_openings']), (1, 2))

    def test_primary_cause(self):
        tree = _move(30, 'tree', (5, 5), root=[((5, 5), 'SAFE')])
        block = _move(32, 'stage2', (6, 6))
        last = _move(34, 'stage4', (7, 7), vct=[((7, 7), 'UNSAFE')])
        entry = lambda record, depth: {'ply': record['ply'], 'depth': depth, 'status': 'UNSAFE' if depth is not None else 'SAFE', 'record': record}
        cause, decisive = primary_cause([entry(last, 0), entry(tree, 2), entry(block, None)])
        self.assertEqual((cause, decisive['ply']), ('VCT2_HORIZON', 30))
        self.assertEqual(primary_cause([entry(last, 0), entry(tree, 1), entry(block, None)])[0], 'VCT1_LOSS')
        self.assertEqual(primary_cause([entry(last, 0), entry(block, 0), entry(tree, None)])[0], 'DEEPER_OR_POSITIONAL')
        self.assertEqual(primary_cause([entry(last, None)])[0], 'UNRESOLVED')

    def test_causal_fields(self):
        lost = {'depth': 2, 'status': 'UNSAFE'}
        self.assertEqual(causal_fields('VCT2_HORIZON', [lost, {'depth': None, 'status': 'SAFE'}]),
                         {'boundary_status': 'SAFE', 'causal_status': 'CONFIRMED'})
        self.assertEqual(causal_fields('VCT2_HORIZON', [lost, {'depth': None, 'status': 'UNKNOWN'}])['causal_status'],
                         'TENTATIVE')
        self.assertEqual(causal_fields('VCT1_LOSS', [lost]),  # the lost run reaches V8's first move
                         {'boundary_status': 'START', 'causal_status': 'CONFIRMED'})
        self.assertEqual(causal_fields('DEEPER_OR_POSITIONAL', [lost])['causal_status'], 'NOT_APPLICABLE')


class PositionTruthTest(unittest.TestCase):
    def test_verdicts_and_recall(self):
        rows = [{'move': [1, 1], 'lost_depth': 0, 'status': 'PROVEN_LOSS'},
                {'move': [2, 2], 'lost_depth': None, 'status': 'SAFE'}]
        summary = truth_summary(rows, 2, {'arm': {(2, 2), (3, 3)}})
        self.assertEqual(summary['verdict'], 'SAVING_MOVES_FOUND')
        self.assertEqual(summary['root_candidate_recall']['arm']['recall'], 1.0)
        lost = truth_summary(rows[:1], 1, {})
        self.assertEqual(lost['verdict'], 'PROVEN_LOSS')
        self.assertEqual(truth_summary(rows[:1], 1, {}, restricted=True)['verdict'], 'RESTRICTED_PROVEN_LOSS')
        self.assertEqual(truth_summary(rows[:1], 2, {})['verdict'], 'UNRESOLVED')  # one move not classified

    def test_p94_moves_are_lost_and_resume(self):
        with TemporaryDirectory() as tmp:
            jsonl, out = Path(tmp, 't.jsonl'), Path(tmp, 't.json')
            truth_main(['--name', 'P94', '--moves', '11,13', '11,12', '--jsonl', str(jsonl), '--output', str(out)])
            truth_main(['--name', 'P94', '--moves', '11,13', '11,12', '--jsonl', str(jsonl), '--output', str(out)])
            self.assertEqual(len(jsonl.read_text(encoding='utf-8').splitlines()), 2)  # resumed, not redone
            payload = json.loads(out.read_text(encoding='utf-8'))
            self.assertTrue(payload['restricted'])
            self.assertEqual(payload['summary']['verdict'], 'RESTRICTED_PROVEN_LOSS')
            self.assertTrue(all(r['status'] == 'PROVEN_LOSS' for r in payload['moves']))


class TruthResumeTest(unittest.TestCase):
    def test_legacy_torn_and_budget_mismatch(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, 't.jsonl')
            path.write_text(json.dumps({'move': [1, 1], 'lost_depth': 0, 'status': 'PROVEN_LOSS'}) + '\n'
                            + json.dumps({'move': [2, 2], 'lost_depth': None, 'status': 'SAFE',
                                          'budget': DEFAULT_BUDGET}) + '\n'
                            + '{"move": [3, 3], "lost_d', encoding='utf-8')  # torn by an interrupted write
            done, info = load_done(path, dict(DEFAULT_BUDGET))
            self.assertEqual(set(done), {(1, 1), (2, 2)})
            self.assertEqual(info, {'legacy_rows': 1, 'torn_lines': 1})
            with self.assertRaises(ValueError):  # one file never mixes budgets
                load_done(path, {**DEFAULT_BUDGET, 'node_budget': 1})

    def test_rerun_with_more_workers_only_computes_missing_moves(self):
        with TemporaryDirectory() as tmp:
            jsonl = Path(tmp, 't.jsonl')
            truth_main(['--name', 'P94', '--moves', '11,13', '--jsonl', str(jsonl)])
            truth_main(['--name', 'P94', '--moves', '11,13', '11,12', '11,14', '--workers', '2', '--jsonl', str(jsonl)])
            rows = [json.loads(line) for line in jsonl.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(sorted(tuple(r['move']) for r in rows), [(11, 12), (11, 13), (11, 14)])
            self.assertTrue(all(r['budget'] == DEFAULT_BUDGET for r in rows))


class DetectorEvalSummaryTest(unittest.TestCase):
    def _row(self, name, expect, status, verified=None, seconds=1.0, added=0.5, played=None):
        return {'key': f'{name}/{expect}/{status}/{seconds}', 'set': name, 'expect': expect, 'verified': verified,
                'meta': {'played': played},
                'result': {'status': status, 'seconds': seconds, 'stage_seconds': {'selective': added}}}

    def test_gate(self):
        rows = [self._row('A', 'PROVEN_LOSS', 'PROVEN_LOSS', True, played=True),
                self._row('A', 'PROVEN_LOSS', 'UNKNOWN', seconds=2.0),  # a missed alternative does not fail the gate
                self._row('A', 'NOT_PROVEN_LOSS', 'NO_TARGETED_VCT2_FOUND'),
                self._row('D', None, 'NO_TARGETED_VCT2_FOUND', seconds=30.0, added=1.0)]
        gate = detector_summary(rows)['gate']
        self.assertTrue(gate['A_all_detected'])
        self.assertEqual(gate['soundness_violations'], [])
        self.assertFalse(gate['D_cost_ok_total'])   # one 30 s check: p95 > 10 s
        self.assertTrue(gate['D_cost_ok_added'])     # its selective stage took 1 s
        self.assertTrue(gate['pass_added'] and not gate['pass_total'])

    def test_flagging_a_move_without_a_loss_is_a_violation(self):
        rows = [self._row('A', 'NOT_PROVEN_LOSS', 'PROVEN_LOSS', True)]
        self.assertEqual(len(detector_summary(rows)['gate']['soundness_violations']), 1)


class JoinP92Test(unittest.TestCase):
    def test_committed_inputs(self):
        load = lambda name: json.loads(Path('docs/mcts-v8-results', name).read_text(encoding='utf-8'))
        joined = join_p92(load('s2_p92_truth.json'), load('s2_policy_diag.json'),
                          [load(f's1_probes_{a}.json') for a in ('full', 'puct_heur', 'puct_policy')])
        totals = joined['totals']
        self.assertEqual((totals['proven_loss'], totals['unknown']), (129, 4))
        self.assertEqual(totals['best_policy_rank_not_proven_loss'], 45)
        self.assertEqual(joined['provenance']['truth_legacy_rows'], 41)


class PolicyDiagTest(unittest.TestCase):
    def test_ranks_are_one_indexed_over_all_moves(self):
        info = describe({(0, 0): 0.1, (2, 3): 0.6, (5, 5): 0.3}, [(3, 4), (9, 9)], top=2)
        self.assertEqual(info['top'], [[[3, 4], 0.6], [[6, 6], 0.3]])
        self.assertEqual(info['moves']['3,4'], {'rank': 1, 'prob': 0.6})
        self.assertEqual(info['moves']['9,9'], {'rank': None, 'prob': 0.0})  # not legal / no mass


if __name__ == '__main__':
    unittest.main()
