"""H6-3 offline gate (hybrid.h6_gate, scripts/h6_offline_gate.py)."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None

if torch is not None:
    from hybrid import h6_gate
    from hybrid.h6_gate import THRESHOLDS, d4_metrics, ece, judge, part_of, partition, probes, split_metrics
    from hybrid.h6_value_cache import KIND_OUTCOME, KIND_PROOF, build_value_cache, load_value_cache
    from tests.test_h6_value import h3_model, label_rows, random_games, write_report, write_rows

ROOT = Path(__file__).resolve().parents[1]
PROBE_FILE = ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json'


def side_value(planes):
    """A value that only reads plane 3 (current_is_black): +0.5 for white to move, -0.5 for black."""
    return 0.5 - planes[:, 3, 0, 0]


def fake(v, t, kind, ply):
    return torch.tensor(v), {'value': torch.tensor(t, dtype=torch.float32),
                             'kind': torch.tensor(kind, dtype=torch.int8), 'ply': torch.tensor(ply, dtype=torch.int16)}


@unittest.skipIf(torch is None, 'requires torch')
class PartitionTest(unittest.TestCase):
    def test_partition_is_deterministic_disjoint_and_whole_games(self):
        ids = [f'renjunet:{i}' for i in range(200)]
        self.assertEqual([part_of(g) for g in ids], [part_of(g) for g in ids])
        self.assertTrue(40 < sum(part_of(g) == 'A' for g in ids) < 160)
        data = {'game': torch.tensor([i % 200 for i in range(1000)], dtype=torch.int32),
                'value': torch.zeros(1000)}
        a, ia = partition(data, ids, 'A')
        b, ib = partition(data, ids, 'B')
        self.assertEqual(ia['states'] + ib['states'], 1000)
        self.assertEqual(ia['games'] + ib['games'], 200)
        self.assertFalse(set(a['game'].tolist()) & set(b['game'].tolist()))
        self.assertTrue(all(part_of(ids[g]) == 'A' for g in a['game'].tolist()))


@unittest.skipIf(torch is None, 'requires torch')
class MetricTest(unittest.TestCase):
    def test_gates_from_metrics(self):
        # proof: +1 all right, -1 9/10 right; outcome: 0.7 / 0.6 by side, small errors
        v, t, kind, ply = [], [], [], []
        for i in range(20):
            v.append(0.9); t.append(1); kind.append(KIND_PROOF); ply.append(10 + i)
        for i in range(10):
            v.append(-0.8 if i else 0.3); t.append(-1); kind.append(KIND_PROOF); ply.append(11 + i)
        for i in range(100):
            target = 1 if i % 2 else -1
            right = (i % 10) < 7
            v.append(0.4 * target if right else -0.2 * target); t.append(target); kind.append(KIND_OUTCOME)
            ply.append(5 + i)
        vv, data = fake(v, t, kind, ply)
        m = split_metrics(vv, data)
        self.assertAlmostEqual(m['proof']['minus_accuracy'], 0.9)
        self.assertAlmostEqual(m['proof']['balanced_accuracy'], 0.95)
        self.assertLess(m['outcome']['mse'], 1.0)
        d4 = {'mean_std': 0.01, 'p95_std': 0.02, 'confident_sign_agreement': 1.0}
        probe = {'P93': {'pass': True}, 'P94': {'pass': True}}
        verdict = judge(m, d4, probe)
        self.assertTrue(verdict['gates']['G1'])
        self.assertIn(verdict['lambda'], (0.5, 1.0))
        # G1 fails when proof -1 accuracy drops below 0.85
        v2 = list(v)
        for i in range(20, 23):
            v2[i] = 0.1
        m2 = split_metrics(torch.tensor(v2), data)
        self.assertFalse(judge(m2, d4, probe)['gates']['G1'])
        # G4: P94 alone failing points at --prove-losses, P93 failing does not
        bad = judge(m, d4, {'P93': {'pass': True}, 'P94': {'pass': False}})
        self.assertIn('prove-losses', bad['next_steps_if_failed']['G4'])
        bad = judge(m, d4, {'P93': {'pass': False}, 'P94': {'pass': True}})
        self.assertNotIn('--prove-losses v3', bad['next_steps_if_failed']['G4'])

    def test_g2_g5_thresholds(self):
        # outcome MSE >= 1 fails G2 even with good accuracy; a colour gap > 0.05 fails G5
        v, t, kind, ply = [], [], [], []
        for i in range(400):
            target = 1 if i % 2 else -1
            black = (i // 2) % 2 == 0
            right = (i % 20) < (16 if black else 12)  # black 0.8, white 0.6
            v.append(0.99 * target if right else -0.99 * target); t.append(target); kind.append(KIND_OUTCOME)
            ply.append(10 if black else 11)
        vv, data = fake(v, t, kind, ply)
        m = split_metrics(vv, data)
        verdict = judge(m, {'mean_std': 0, 'p95_std': 0, 'confident_sign_agreement': 1.0},
                        {'P93': {'pass': True}, 'P94': {'pass': True}})
        self.assertGreater(m['outcome']['mse'], 1.0)
        self.assertFalse(verdict['gates']['G2'])
        self.assertFalse(verdict['gates']['G5'])
        self.assertFalse(verdict['gates']['G1'])  # no proof rows -> no proof accuracy -> fail, not pass

    def test_ece(self):
        self.assertAlmostEqual(ece(torch.tensor([0.0, 0.0]), torch.tensor([1.0, -1.0])), 0.0)
        self.assertAlmostEqual(ece(torch.tensor([1.0, 1.0]), torch.tensor([-1.0, -1.0])), 1.0)


@unittest.skipIf(torch is None, 'requires torch')
class ProbeAndD4Test(unittest.TestCase):
    def test_probes_use_side_to_move(self):
        out = probes(side_value, PROBE_FILE, 'cpu')
        self.assertEqual(out['P93']['side_to_move'], 'white')
        self.assertEqual(out['P94']['side_to_move'], 'black')
        self.assertTrue(out['P93']['pass'] and out['P94']['pass'])
        self.assertEqual(out['P93']['d4_std'], 0.0)
        self.assertGreaterEqual(out['P92']['rank'], 1)
        self.assertEqual(out['P92']['played_0idx'], [2, 3])  # (3,4) 1-indexed
        flipped = probes(lambda p: -side_value(p), PROBE_FILE, 'cpu')
        self.assertFalse(flipped['P93']['pass'] or flipped['P94']['pass'])

    def test_d4_metrics_and_end_to_end(self):
        from hybrid.h6_value_train import value_forward
        from model.checkpoint import save_checkpoint
        from scripts.h6_offline_gate import main
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = label_rows(random_games())
            write_rows(root / 'rows.jsonl.gz', rows)
            write_report(root / 'report.json', rows)
            manifest = build_value_cache(root / 'rows.jsonl.gz', root / 'cache', report=root / 'report.json')
            data, _ = load_value_cache(root / 'cache', 'test')
            d4 = d4_metrics(side_value, data, 'cpu')  # side_value is D4-invariant
            self.assertEqual(d4['mean_std'], 0.0)
            self.assertEqual(d4['confident_sign_agreement'], 1.0)
            model = h3_model()
            save_checkpoint(root / 'best.pt', model)
            (root / 'best.json').write_text(json.dumps({
                'stage': 'H6-2a', 'value_trained': True, 'labels_sha256': manifest['labels_sha256'],
                'model_config': {'board_size': 15, 'input_planes': 6, 'channels': 64, 'blocks': 4,
                                 'policy_channels': 2, 'value_channels': 1, 'value_hidden': 64},
                'step': 1}), encoding='utf-8')
            out = root / 'gate.json'
            args = ['--checkpoint', str(root / 'best.pt'), '--cache-dir', str(root / 'cache'), '--part', 'A',
                    '--output', str(out), '--allow-dirty']
            self.assertEqual(main(args), 0)
            result = json.loads(out.read_text(encoding='utf-8'))
            self.assertIn(result['verdict'], ('PASS', 'FAIL'))
            self.assertEqual(result['thresholds'], THRESHOLDS)
            self.assertEqual(result['partition']['part'], 'A')
            with self.assertRaises(SystemExit):  # one shot
                main(args)
            value_forward  # noqa: B018  (imported for the checkpoint path)


if __name__ == '__main__':
    unittest.main()
