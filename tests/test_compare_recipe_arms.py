"""scripts/compare_recipe_arms.py: the pre-registered C2 recipe verdict on synthetic runs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

from compare_recipe_arms import (direct_result, heavy_non_inferiority,  # noqa: E402
                                 sign_test, verdict)


def write_direct(path: Path, seed: int, wins: int, losses: int, splits: int = 10) -> Path:
    path.write_text(json.dumps({'seed': seed, 'matches': [{'summary': {
        'a': 'CTL1520', 'b': 'LR1520', 'a_pair_wins': losses, 'a_pair_losses': wins,
        'pair_splits': splits}}]}))
    return path


def write_run(root: Path, one_sided: bool, heavy_wins: list[int], probe_top1: list[bool]):
    (root / 'self_play').mkdir(parents=True)
    for g in range(4):
        results = [1 if g % 2 else -1] * 4 if one_sided else [1, -1, 1, -1]
        (root / 'self_play' / f'gen{g:03d}.json').write_text(json.dumps({'games': [
            {'record': {'moves': [112] + [i + 3 * k for i in range(19)], 'winner': w}}
            for k, w in enumerate(results)]}))
    games = [{'result': 'win' if w else 'loss', 'moves': [[7, 7], [6, 6], [i % 15, 3]],
              'opening_plies': 3, 'model_color': 'black' if i % 2 == 0 else 'white'}
             for i, w in enumerate(heavy_wins)]
    (root / 'external_eval').mkdir()
    (root / 'external_eval' / 'gen004_heavy.json').write_text(json.dumps(
        {'opponents': {'mcts_v7': {'games': games}}}))
    (root / 'probes').mkdir()
    (root / 'probes' / 'gen004.json').write_text(json.dumps({'rows': [
        {'id': f'p{i}', 'kind': 'vcf', 'top1': t} for i, t in enumerate(probe_top1)]}))
    return root


class CompareRecipeArmsTest(unittest.TestCase):
    def test_sign_test(self):
        self.assertEqual(sign_test(0, 0), 1.0)
        self.assertAlmostEqual(sign_test(10, 0), 2 / 1024)

    def test_direct_pools_distinct_seeds_and_refuses_shared_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = write_direct(Path(tmp) / 'a.json', 9109, 12, 4)
            b = write_direct(Path(tmp) / 'b.json', 9110, 11, 4)
            result = direct_result([a, b], 'LR', 'CTL')
            self.assertEqual((result['treatment_pair_wins'], result['treatment_pair_losses']),
                             (23, 8))
            self.assertEqual(result['winner'], 'treatment')
            c = write_direct(Path(tmp) / 'c.json', 9109, 5, 5)
            with self.assertRaises(SystemExit):
                direct_result([a, c], 'LR', 'CTL')

    def test_heavy_non_inferiority_is_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = write_run(Path(tmp) / 'c', False, [1, 0] * 100, [True] * 10)
            same = write_run(Path(tmp) / 't', False, [1, 0] * 100, [True] * 10)
            result = heavy_non_inferiority(control, same, [4], 0.05)
            self.assertEqual(result['treatment_minus_control'], 0.0)
            self.assertTrue(result['non_inferior'])        # identical games: no variance
        with tempfile.TemporaryDirectory() as tmp:
            control = write_run(Path(tmp) / 'c', False, [1] * 200, [True] * 10)
            worse = write_run(Path(tmp) / 't', False, [1, 1, 1, 0] * 50, [True] * 10)
            self.assertFalse(heavy_non_inferiority(control, worse, [4], 0.05)['non_inferior'])

    def test_verdict_rules(self):
        undecided = {'winner': None}
        unstable = {'one_sided_share': 0.6, 'mean_abs_color_margin': 0.8}
        stable = {'one_sided_share': 0.4, 'mean_abs_color_margin': 0.6}
        ok_heavy, ok_probes = {'non_inferior': True}, {'regression': False}
        self.assertEqual(verdict(undecided, unstable, stable, ok_heavy, ok_probes, 0.15)[0],
                         'adopt_stability')
        self.assertEqual(verdict(undecided, unstable, stable, {'non_inferior': False},
                                 ok_probes, 0.15)[0], 'keep_control')
        self.assertEqual(verdict(undecided, unstable, {'one_sided_share': 0.5,
                                                       'mean_abs_color_margin': 0.6},
                                 ok_heavy, ok_probes, 0.15)[0], 'keep_control')
        self.assertEqual(verdict({'winner': 'control'}, unstable, stable, ok_heavy, ok_probes,
                                 0.15)[0], 'reject')
        self.assertEqual(verdict({'winner': 'treatment'}, stable, unstable, ok_heavy,
                                 ok_probes, 0.15)[0], 'adopt_stronger')

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = write_run(Path(tmp) / 'c', True, [1, 0] * 100, [True] * 10)
            treatment = write_run(Path(tmp) / 't', False, [1, 0] * 100, [True] * 10)
            write_direct(Path(tmp) / 'lr_gen1520.json', 9109, 6, 5)
            write_direct(Path(tmp) / 'lr_gen1560.json', 9110, 5, 6)
            out = Path(tmp) / 'verdict.json'
            done = subprocess.run(
                [sys.executable, str(ROOT / 'scripts' / 'compare_recipe_arms.py'),
                 '--control', str(control), '--treatment', str(treatment), '--from', '0',
                 '--to', '4', '--points', '4', '--direct', str(Path(tmp) / 'lr_gen*.json'),
                 '--output', str(out)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            data = json.loads(out.read_text())
            self.assertEqual(data['stability']['control']['one_sided_share'], 1.0)
            self.assertEqual(data['stability']['treatment']['mean_abs_color_margin'], 0.0)
            self.assertEqual(data['verdict'], 'adopt_stability')


if __name__ == '__main__':
    unittest.main()
