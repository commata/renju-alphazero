import json
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import unittest

from scripts.run_mcts_v8_benchmark import file_sha256, main as benchmark_main, provenance
from scripts.run_s1_probes import FAR_MOVE, main as probes_main, p92_metrics, root_value_spread
from scripts.s1_loss_analysis import causal_fields, primary_cause, signature_ply, signature_table
from scripts.s2_position_truth import main as truth_main, summarize as truth_summary
from scripts.s2_policy_diag import describe

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


class PolicyDiagTest(unittest.TestCase):
    def test_ranks_are_one_indexed_over_all_moves(self):
        info = describe({(0, 0): 0.1, (2, 3): 0.6, (5, 5): 0.3}, [(3, 4), (9, 9)], top=2)
        self.assertEqual(info['top'], [[[3, 4], 0.6], [[6, 6], 0.3]])
        self.assertEqual(info['moves']['3,4'], {'rank': 1, 'prob': 0.6})
        self.assertEqual(info['moves']['9,9'], {'rank': None, 'prob': 0.0})  # not legal / no mass


if __name__ == '__main__':
    unittest.main()
