"""R2-D3 preparation: the additive macro decomposition with each game's original denominator, partitions, per-game
weight shares, replicate-level additivity, composition margins, the prediction checks and the record cross-check."""
from random import Random
import unittest

import torch

from hybrid import h7r_d3 as d3
from hybrid import h7r_r2 as r2
from scripts import h7r_d3_prep as prep


def fixture(n_games=12, seed=3):
    rnd = Random(seed)
    game, kind, ply, label_kind, weight, target, route = [], [], [], [], [], [], []
    for g in range(n_games):
        for p in range(5, 5 + rnd.randint(3, 9)):
            proof = rnd.random() < 0.25
            game.append(g); ply.append(p if rnd.random() < 0.7 else p + 40)
            kind.append(1 if proof else 0)
            label_kind.append(rnd.choice(['proof_vcf', 'proof_forced_loss']) if proof else 'outcome')
            weight.append(1.0 if proof else 0.1)
            target.append(rnd.choice([-1.0, 1.0]) if proof else rnd.choice([-1.0, 0.0, 1.0]))
            route.append('exact_leaf' if proof and rnd.random() < 0.3 else 'network')
    a = [rnd.uniform(-1, 1) for _ in game]
    b = [rnd.uniform(-1, 1) for _ in game]
    return dict(game=game, kind=kind, ply=ply, label_kind=label_kind, weight=weight, target=target, route=route,
                a=a, b=b, n_games=n_games)


def network_parts(f):
    rows = [i for i, r in enumerate(f['route']) if r == 'network']
    return rows, {'outcome': [i for i in rows if f['kind'][i] == 0], 'proof': [i for i in rows if f['kind'][i] == 1]}


class DecompositionTest(unittest.TestCase):
    def test_parts_add_up_to_the_r2_macro_difference(self):
        for seed in range(5):
            f = fixture(seed=seed)
            rows, parts = network_parts(f)
            out = d3.decompose(f['a'], f['b'], f['target'], f['weight'], f['game'], rows, parts, f['n_games'])
            direct = r2.difference(r2.game_sums(f['a'], f['target'], f['weight'], f['game'], rows, f['n_games']),
                                   r2.game_sums(f['b'], f['target'], f['weight'], f['game'], rows, f['n_games']),
                                   'macro')
            self.assertAlmostEqual(out['total'], direct, places=14)
            self.assertAlmostEqual(sum(p['contribution'] for p in out['parts'].values()), direct, places=14)

    def test_each_game_keeps_its_original_denominator(self):
        # game 0: one outcome row (w 0.1) and one proof row (w 1); game 1: one outcome row only
        game, weight, target = [0, 0, 1], [0.1, 1.0, 0.1], [1.0, 1.0, -1.0]
        a, b = [1.0, 0.0, -1.0], [0.0, 1.0, 0.0]
        parts = {'outcome': [0, 2], 'proof': [1]}
        out = d3.decompose(a, b, target, weight, game, [0, 1, 2], parts, 2)
        # outcome: game 0 (0 - 0.1*1)/1.1, game 1 (0 - 0.1*1)/0.1 = -1; mean over both games
        self.assertAlmostEqual(out['parts']['outcome']['contribution'], ((-0.1 / 1.1) + (-1.0)) / 2)
        # proof: game 0 (1*1 - 0)/1.1, game 1 has no proof row but stays in the mean with 0
        self.assertAlmostEqual(out['parts']['proof']['contribution'], (1.0 / 1.1) / 2)
        self.assertEqual(out['parts']['proof']['games_with_rows'], 1)
        # not the part's own macro (proof-only denominators, proof games only): that would be 1.0
        own = r2.difference(r2.game_sums(a, target, weight, game, [1], 2), r2.game_sums(b, target, weight, game, [1], 2),
                            'macro')
        self.assertAlmostEqual(own, 1.0)
        self.assertNotAlmostEqual(out['parts']['proof']['contribution'], own)

    def test_parts_must_be_a_partition(self):
        f = fixture()
        rows, parts = network_parts(f)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            d3.decompose(f['a'], f['b'], f['target'], f['weight'], f['game'], rows,
                         {**parts, 'again': parts['proof'][:1]}, f['n_games'])
        with self.assertRaisesRegex(ValueError, 'partition'):
            d3.decompose(f['a'], f['b'], f['target'], f['weight'], f['game'], rows,
                         {'outcome': parts['outcome']}, f['n_games'])

    def test_bootstrap_parts_add_up_replicate_by_replicate(self):
        f = fixture()
        rows, parts = network_parts(f)
        counts = r2.multiplicities(f['n_games'], 300, 7)
        den = d3.denominators(f['weight'], f['game'], rows, f['n_games'])
        draws = [r2.replicate(counts, d3.part_sums(f['a'], f['b'], f['target'], f['weight'], f['game'], p, den,
                                                   f['n_games']), None, 'macro')[0] for p in parts.values()]
        total, _ = r2.replicate(counts, r2.game_sums(f['a'], f['target'], f['weight'], f['game'], rows, f['n_games']),
                                r2.game_sums(f['b'], f['target'], f['weight'], f['game'], rows, f['n_games']), 'macro')
        self.assertTrue(torch.allclose(sum(draws), total, atol=1e-12))
        out = d3.decompose(f['a'], f['b'], f['target'], f['weight'], f['game'], rows, parts, f['n_games'], counts)
        self.assertEqual(len(out['parts']['outcome']['ci95']['ci95']), 2)

    def test_macro_weight_shares(self):
        game, weight = [0, 0, 1, 1, 1], [0.1, 1.0, 0.1, 0.1, 0.1]
        shares = d3.macro_shares(weight, game, [0, 1, 2, 3, 4], {'outcome': [0, 2, 3, 4], 'proof': [1]}, 2)
        self.assertAlmostEqual(shares['outcome'], ((0.1 / 1.1) + 1.0) / 2)
        self.assertAlmostEqual(shares['proof'], (1.0 / 1.1) / 2)
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        pooled = 0.4 / 1.4  # the pooled weight share would be another number
        self.assertNotAlmostEqual(shares['outcome'], pooled)


class CompositionTest(unittest.TestCase):
    def test_margins_add_up_and_kinds_are_labelled(self):
        f = fixture()
        out = d3.composition(f['route'], f['kind'], f['ply'], f['label_kind'], f['weight'], f['game'], [5, 20, 40, 80])
        self.assertEqual(out['rows'], len(f['game']))
        kinds = {c['kind'] for c in out['kind_route']}
        self.assertTrue(kinds <= {'outcome', 'proof:vcf', 'proof:forced_loss'})
        self.assertAlmostEqual(sum(c['weight'] for c in out['cells']), sum(f['weight']))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            d3.composition(['network'], [1], [5], ['outcome'], [1.0], [0], [5, 20])



class AnalyseWiringTest(unittest.TestCase):
    def test_analyse_on_a_synthetic_cache(self):
        f = fixture(seed=11)
        info = {'game': f['game'], 'ply': f['ply'], 'game_ids': [f'g{i}' for i in range(f['n_games'])],
                'result': ['black' if g % 2 else 'white' for g in f['game']], 'label_kind': f['label_kind']}
        values = {m: [((i * 7 + k) % 19) / 10 - 0.9 for i in range(len(f['game']))] for k, m in enumerate(prep.MODELS)}
        contract = {'metrics': {'ply_bins': [5, 20, 40, 80]}, 'bootstrap': {'level': 0.95}}
        counts = r2.multiplicities(f['n_games'], 200, 1)
        out = prep.analyse(contract, f['kind'], info, f['route'], f['target'], f['weight'], values, counts)
        self.assertEqual(set(out['decomposition']), set(prep.COMPARISONS))
        for d in out['decomposition'].values():
            self.assertAlmostEqual(sum(p['contribution'] for p in d['kind'].values()), d['total'], places=12)
            self.assertAlmostEqual(sum(p['contribution'] for p in d['kind_ply'].values()), d['total'], places=12)
            self.assertEqual(sum(p['rows'] for p in d['kind_ply'].values()), f['route'].count('network'))
        self.assertAlmostEqual(sum(out['macro_weight_shares']['kind'].values()), 1.0)
        self.assertAlmostEqual(sum(v for v in out['macro_weight_shares']['kind_ply'].values()), 1.0)

class PredictionCheckTest(unittest.TestCase):
    def setUp(self):
        self.info = {'game': [0, 0, 1, 2], 'ply': [5, 6, 5, 5], 'game_ids': ['a', 'b', 'c']}
        self.routes = ['network', 'exact_leaf', 'network', 'network']

    def pred(self, **change):
        p = {'contract_sha256': 's', 'game': torch.tensor(self.info['game'], dtype=torch.int32),
             'ply': torch.tensor(self.info['ply'], dtype=torch.int16), 'route': [True, False, True, True],
             'models': {m: {'raw': torch.zeros(4, dtype=torch.float64), 'd4': torch.full((4,), .5, dtype=torch.float64)}
                        for m in prep.MODELS}}
        p.update(change)
        return p

    def test_aligned_predictions_pass_and_others_fail(self):
        self.assertEqual(prep.checked_predictions(self.pred(), self.info, self.routes, 's')['c5'], [.5] * 4)
        bad_value = self.pred()
        bad_value['models']['c8']['d4'][1] = 1.5
        bad_model = self.pred()
        bad_model['models'].pop('c8')
        cases = {'contract': self.pred(contract_sha256='x'),
                 'order': self.pred(game=torch.tensor([0, 1, 0, 2], dtype=torch.int32)),
                 'missing row': self.pred(ply=torch.tensor([5, 6, 5], dtype=torch.int16)),
                 'route': self.pred(route=[True] * 4), 'value range': bad_value, 'model set': bad_model,
                 'float game': self.pred(game=torch.tensor([0., 0., 1., 2.]))}
        for name, p in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                prep.checked_predictions(p, self.info, self.routes, 's')


class RecordCheckTest(unittest.TestCase):
    def result(self, delta=0.0):
        out = {'decomposition': {}}
        for spec in prep.COMPARISONS:
            out['decomposition'][spec] = {'total': -0.004 + delta,
                                          'component_macro': {'outcome': -0.02, 'proof': 0.001}}
        return out

    def records(self):
        metrics = [('network', 'macro', -0.004), ('network/outcome', 'macro', -0.02), ('network/proof', 'macro', 0.001),
                   ('network', 'pooled', -0.001)]
        eps = {'paired': {spec: [{'metric': {'group': g, 'aggregate': a}, 'difference': v} for g, a, v in metrics]
                          for spec in prep.COMPARISONS[:-1]}}
        return eps, {'summary': {'c8:c5': {'primary': {'difference': -0.004000003}}}}

    def test_equal_totals_pass_and_differences_stop(self):
        eps, a = self.records()
        checks = prep.check_against_records(self.result(), eps, a)
        self.assertEqual(len(checks), len(prep.COMPARISONS))
        with self.assertRaisesRegex(ValueError, 'differs from R2-0B'):
            prep.check_against_records(self.result(1e-6), eps, a)

    def test_absent_predictions_stop(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, 'the epsilon record names'):
            prep.bound_inputs(root, 'runs/none/seed_predictions.pt')
        if not (root / 'runs/h7r/R2-DIAG-1/r2_0b/seed_predictions.pt').exists():  # this checkout has no raw data
            with self.assertRaisesRegex(ValueError, 'absent'):
                prep.bound_inputs(root, 'runs/h7r/R2-DIAG-1/r2_0b/seed_predictions.pt')



class RunVerifyTest(unittest.TestCase):
    def test_new_record_only_and_verification_catches_edits(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        body = {'format': prep.FORMAT, 'git': {'commit': 'c', 'dirty': False},
                'decomposition': {'c8:c5': {'total': -0.0045, 'kind': {'outcome': {'contribution': -0.005}}}}}
        with TemporaryDirectory() as tmp, patch.object(prep, 'compute', side_effect=lambda *a, **k: json.loads(
                json.dumps(body))):
            out = Path(tmp) / 'prep.json'
            args = SimpleNamespace(output=str(out), cache='unused', predictions='unused')
            written = prep.run(args, root=tmp, log=lambda *a, **k: None)
            self.assertIn('seconds', written)
            self.assertEqual(prep.verify(args, root=tmp, log=lambda *a, **k: None), [])
            with self.assertRaisesRegex(ValueError, 'never overwritten'):
                prep.run(args, root=tmp)
            saved = json.loads(out.read_text())
            saved['decomposition']['c8:c5']['kind']['outcome']['contribution'] = -0.004
            out.write_text(json.dumps(saved))
            self.assertTrue(prep.verify(args, root=tmp, log=lambda *a, **k: None))

    def test_verify_rejects_ambiguous_json_and_replaced_file(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        from h7r_d4_full_check import strict_json

        body = {'format': prep.FORMAT}
        with TemporaryDirectory() as tmp, patch.object(prep, 'compute', return_value=body):
            out = Path(tmp) / 'prep.json'
            args = SimpleNamespace(output=str(out))
            out.write_text('{"format":"wrong","format":"h7r-d3-prep-v1","seconds":0}')
            with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
                prep.verify(args)
            out.write_text(json.dumps({**body, 'seconds': 0}))

            def replace_after_read(raw):
                out.write_text(json.dumps({**body, 'seconds': 1}))
                return strict_json(raw)

            with patch('h7r_d4_full_check.strict_json', side_effect=replace_after_read):
                with self.assertRaisesRegex(ValueError, 'record changed'):
                    prep.verify(args)



class DesktopBlockTest(unittest.TestCase):
    """The plan's \u00a79.26 blocks stop on a failed step, the pipeline included, and never reach a commit."""

    def test_blocks_stop_on_failures(self):
        import os
        import re
        import shutil
        import subprocess
        from pathlib import Path
        from tempfile import TemporaryDirectory
        if not shutil.which('bash'):
            self.skipTest('Bash required')
        doc = (Path(__file__).resolve().parents[1] / 'docs/h7-rl-plan.md').read_text(encoding='utf-8')
        blocks = re.findall(r'```bash\n(.*?)```', doc.split('### 9.26', 1)[1].split('\n### ', 1)[0], re.S)
        self.assertEqual(len(blocks), 4)
        for index, failure in ((0, 'fetch'), (0, 'hash'), (1, 'run'), (2, 'verify'),
                               (3, 'extra'), (3, 'verify'), (3, 'staged')):
            with self.subTest(index=index, failure=failure), TemporaryDirectory() as tmp:
                root = Path(tmp)
                bins = root / '.venv/bin'
                bins.mkdir(parents=True)
                (bins / 'activate').write_text('export PATH="$PWD/.venv/bin:$PATH"\n')
                (bins / 'python').write_text('''#!/bin/bash
case "$D3_FAIL:$*" in
verify:*h7r_d3_prep.py*--verify*) exit 23;;
run:*h7r_d3_prep.py*) case "$*" in *--verify*) exit 0;; *) exit 22;; esac;;
staged:*verify-staged*) exit 24;;
esac
exit 0
''')
                (bins / 'git').write_text('''#!/bin/bash
if [ "$D3_FAIL:$1" = fetch:fetch ]; then exit 25; fi
if [ "$1" = commit ] || [ "$1" = push ]; then echo "$1" >> "$D3_SCRATCH/commits"; fi
exit 0
''')
                for name in ('python', 'git'):
                    (bins / name).chmod(0o755)
                if failure == 'hash':
                    for path in ('docs/h7r/r2/R2-DIAG-1/r2_0b/epsilon.json',
                                 'runs/h7r/R2-DIAG-1/r2_0b/seed_predictions.pt'):
                        file = root / path
                        file.parent.mkdir(parents=True, exist_ok=True)
                        file.write_text('wrong hash')
                if index == 3:
                    records = root / 'docs/h7r/r2/R2-D3-NO-OUTCOME-1'
                    records.mkdir(parents=True)
                    (records / 'prep.json').write_text('{}')
                    if failure == 'extra':
                        (records / 'extra.json').write_text('{}')
                script = blocks[index].replace('cd ~/renju-h6', 'cd "$D3_SCRATCH"', 1)
                result = subprocess.run(['bash'], input=script, text=True, capture_output=True,
                                        env={**os.environ, 'D3_SCRATCH': str(root), 'D3_FAIL': failure})
                expected = {'fetch': 25, 'hash': 1, 'run': 22, 'verify': 23, 'extra': 1, 'staged': 24}[failure]
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                if failure == 'hash':
                    self.assertNotIn('PRECHECK PASS', result.stdout)
                if failure == 'extra':
                    self.assertIn('unexpected record path', result.stderr)
                self.assertFalse((root / 'commits').exists())


if __name__ == '__main__':
    unittest.main()
