"""H6-3D diagnostics (hybrid.h6_diag, scripts/h6_diagnose.py)."""
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
    from hybrid.h3_train import d4_tables
    from hybrid.h6_diag import d1, d2, d4_forward, deep_loss_children, inverse_spatial
    from hybrid.h6_value_cache import build_value_cache, load_value_cache
    from hybrid.h6_value_train import augment_planes
    from model.config import ModelConfig
    from model.network import PolicyValueNet
    from tests.test_h6_value import h3_model, label_rows, random_games, write_report, write_rows

ROOT = Path(__file__).resolve().parents[1]


def equivariant_model():
    """Identity trunk: features are the input planes, so the trunk is exactly D4-equivariant."""
    torch.manual_seed(3)
    model = PolicyValueNet(ModelConfig(channels=6))
    model.trunk = torch.nn.Sequential()
    return model.eval()


@unittest.skipIf(torch is None, 'torch is not installed')
class DiagTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        root = Path(cls.tmp.name)
        rows = label_rows(random_games())
        write_rows(root / 'rows.jsonl.gz', rows)
        write_report(root / 'report.json', rows)
        cls.manifest = build_value_cache(root / 'rows.jsonl.gz', root / 'cache', report=root / 'report.json')
        cls.root = root
        cls.data, _ = load_value_cache(root / 'cache', 'test')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_inverse_spatial_undoes_augment(self):
        dest, src = d4_tables('cpu')
        x = torch.randn(8, 3, 15, 15)
        s = torch.arange(8)
        self.assertTrue(torch.equal(inverse_spatial(augment_planes(x, s, src), s, dest), x))

    def test_equivariant_trunk_reads_zero(self):
        dest, src = d4_tables('cpu')
        from hybrid.h6_value_cache import planes_from_value_cache
        planes, _, _ = planes_from_value_cache(self.data, torch.arange(5), 'cpu')
        out = d4_forward(equivariant_model(), planes, dest, src)
        self.assertLess(float(out['rel_l2'].max()), 1e-6)
        self.assertGreater(float(out['cosine'].min()), 1 - 1e-6)
        # with exactly equivariant features, head-only anisotropy equals the end-to-end spread
        self.assertTrue(torch.allclose(out['head_values'], out['values'], atol=1e-6))
        result, v8 = d1(equivariant_model(), self.data, 'cpu')
        self.assertEqual(v8.shape, (self.data['value'].shape[0], 8))
        self.assertLess(result['trunk_equivariance']['rel_l2_nonidentity']['max'], 1e-6)
        self.assertEqual(result['per_symmetry_abs_diff_from_identity'][0], 0.0)

    def test_real_trunk_is_measured(self):
        result, _ = d1(h3_model(), {k: v[:20] for k, v in self.data.items()}, 'cpu')
        self.assertGreater(result['trunk_equivariance']['rel_l2_nonidentity']['mean'], 0.0)
        self.assertEqual(len(result['policy_top1_agreement_per_symmetry']), 8)
        self.assertEqual(result['policy_top1_agreement_per_symmetry'][0], 1.0)

    def test_deep_loss_children_and_scores(self):
        manifests = sorted((ROOT / 'docs' / 'mcts-v8-results').glob('e1*/*manifest.json'))
        rows = deep_loss_children(manifests)
        self.assertGreater(len(rows), 100)
        self.assertTrue(all(r['side_to_move'] == ('black' if r['ply'] % 2 == 0 else 'white') for r in rows))
        plane3 = torch.stack([r['planes'][3, 0, 0] for r in rows])
        self.assertTrue(all((p == 1.0) == (r['side_to_move'] == 'black') for p, r in zip(plane3, rows)))
        always_plus = d2(lambda p: torch.full((p.shape[0],), 0.5), rows, 'cpu')
        self.assertEqual(always_plus['raw']['move_accuracy'], 1.0)
        self.assertLessEqual(always_plus['raw']['unique_children'], always_plus['raw']['moves'])
        self.assertIn('0', always_plus['raw']['by_lost_depth'])

    def test_end_to_end(self):
        from model.checkpoint import save_checkpoint
        from scripts.h6_diagnose import main
        save_checkpoint(self.root / 'best.pt', h3_model())
        (self.root / 'best.json').write_text(json.dumps({
            'stage': 'H6-2a', 'labels_sha256': self.manifest['labels_sha256'],
            'model_config': {'board_size': 15, 'input_planes': 6, 'channels': 64, 'blocks': 4,
                             'policy_channels': 2, 'value_channels': 1, 'value_hidden': 64}}), encoding='utf-8')
        out = self.root / 'diag.json'
        args = ['--checkpoint', str(self.root / 'best.pt'), '--cache-dir', str(self.root / 'cache'),
                '--output', str(out), '--allow-dirty', '--limit', '30', '--latency-repeats', '5']
        self.assertEqual(main(args), 0)
        result = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual(result['partition']['part'], 'A')
        self.assertIn('outcome_ece', result['D3']['table'])
        self.assertIn('slowdown_throughput', result['D3']['latency'])
        self.assertIn('P94', result['D3']['dev_probes_P93_P94'])


if __name__ == '__main__':
    unittest.main()
