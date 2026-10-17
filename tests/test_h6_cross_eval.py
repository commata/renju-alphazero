"""scripts/h6_cross_eval.py: H6-2a vs H6-2a' on the v3 cache's states, split by what v3 changed."""
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
    from hybrid.h6_labels import check_leakage, game_rows, mask_cross_split
    from hybrid.h6_value_cache import build_value_cache
    from model.checkpoint import save_checkpoint
    from renju import BLACK
    from tests.test_h6_value import h3_model, random_games, write_report, write_rows

MODEL_CONFIG = {'board_size': 15, 'input_planes': 6, 'channels': 64, 'blocks': 4,
                'policy_channels': 2, 'value_channels': 1, 'value_hidden': 64}


def rows_for(games, *, with_vcf_loss: bool):
    """v2-like rows (a vcf proof per decided game); v3 adds a vcf_loss two plies before the end."""
    rows = []
    for g in games:
        proofs = {}
        if g['winner'] is not None:
            proofs[9] = {'kind': 'vcf', 'value': 1 if (9 % 2 == 0) == (g['winner'] == BLACK) else -1, 'depth': 0}
            last = len(g['moves']) - 2
            loser_to_move = last if (last % 2 == 0) != (g['winner'] == BLACK) else last - 1
            if with_vcf_loss and loser_to_move > 9:
                proofs[loser_to_move] = {'kind': 'vcf_loss', 'value': -1, 'depth': 0}
        rows.extend(game_rows(g, source='test', split=g['split'], outcome_weight=0.1, proof_weight=1.0,
                              proofs=proofs, solver_class='vcf', from_ply=5))
    mask_cross_split(rows)
    check_leakage(rows)
    return rows


@unittest.skipIf(torch is None, 'requires torch')
class CrossEvalTest(unittest.TestCase):
    def test_new_losses_are_the_added_vcf_loss_rows(self):
        from scripts.h6_cross_eval import main
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            games = random_games(count=60)
            for version, extra in (('v2', False), ('v3', True)):
                rows = rows_for(games, with_vcf_loss=extra)
                write_rows(root / f'{version}.jsonl.gz', rows)
                write_report(root / f'{version}.json', rows)
                manifest = build_value_cache(root / f'{version}.jsonl.gz', root / f'cache_{version}',
                                             report=root / f'{version}.json')
                (root / version).mkdir()
                save_checkpoint(root / version / 'best.pt', h3_model(seed=11 if version == 'v2' else 12))
                (root / version / 'best.json').write_text(json.dumps({
                    'stage': 'H6-2a', 'labels_sha256': manifest['labels_sha256'], 'model_config': MODEL_CONFIG,
                    'step': 1}), encoding='utf-8')
            out = root / 'cross.json'
            args = ['--model', f"v2={root / 'v2' / 'best.pt'}", '--model', f"v3={root / 'v3' / 'best.pt'}",
                    '--v2-cache', str(root / 'cache_v2'), '--v3-cache', str(root / 'cache_v3'),
                    '--output', str(out), '--allow-dirty']
            self.assertEqual(main(args), 0)
            result = json.loads(out.read_text(encoding='utf-8'))
            v3_rows = [r for r in rows_for(games, with_vcf_loss=True)
                       if r.label_kind == 'proof_vcf_loss' and not r.value_mask and r.split == 'val']
            val = result['sets']['val']['subsets']
            self.assertEqual(val['new_loss'], len(v3_rows))
            self.assertGreater(val['new_loss'], 0)
            old = [r for r in rows_for(games, with_vcf_loss=False)
                   if r.label_kind.startswith('proof_') and r.value_target < 0 and not r.value_mask and r.split == 'val']
            self.assertEqual(val['old_loss'], len(old))  # proof -1 in both versions
            self.assertEqual(sum(val[k] for k in ('new_loss', 'old_loss', 'proof_win', 'outcome_win', 'outcome_loss')),
                             val['all'])
            for name in ('v2', 'v3'):
                for set_name in ('val', 'testA'):
                    d4 = result['models'][name]['sets'][set_name]['d4']
                    self.assertEqual(d4['all']['states'], result['sets'][set_name]['subsets']['all'])
            with self.assertRaises(SystemExit):  # never overwrites
                main(args)


if __name__ == '__main__':
    unittest.main()
