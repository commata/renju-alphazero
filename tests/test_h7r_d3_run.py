"""R2-D3-NO-OUTCOME-1: the outcome mask on a real (synthetic) cycle cache and its label rows, the one-call trainer
wrapper on the real ``train_value``, the direct comparison with the control record, the control reproduction and the
R2 reading rule, the contract loader on the real committed records, and the train wiring (heavy steps replaced)."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from hybrid import h7c_train as tr
from hybrid import h7r_d3_mask as dm
from hybrid import h7r_r2 as r2
from scripts import h7r_d3 as d3

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'docs/h7r/r2/R2-D3-NO-OUTCOME-1/d3-contract.json'
QUIET = lambda *a, **k: None  # noqa: E731


class MaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_h7c_train import cycle_value_rows
        from tests.test_tb_v3_net import write_models
        cls.tmp = TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        policy, value = write_models(tmp / 'm', seed=3)
        cls.rows, cls.cache = cycle_value_rows(tmp, 'c8', policy, value, cycle=8)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outcome_rows_removed_proof_rows_kept_every_row_linked(self):
        from hybrid.h6_value_cache import load_value_cache
        keep, body = dm.build_outcome_mask(self.rows, self.cache)
        for split, info in body['splits'].items():
            data, _ = load_value_cache(self.cache, split)
            self.assertTrue(torch.equal(keep[split], data['kind'] == 1))
            self.assertEqual(info['removed_rows'] + info['kept_rows'], info['rows'])
            self.assertAlmostEqual(info['removed_weight'], float(data['weight'].double()[data['kind'] == 0].sum()))
            self.assertEqual(sum(c['rows'] for lk, c in info['by_label_kind'].items() if lk == 'outcome'),
                             info['removed_rows'])
            self.assertTrue(all(c['kept'] == lk.startswith('proof_') for lk, c in info['by_label_kind'].items()))
        self.assertGreater(body['splits']['train']['kept_rows'], 0)  # the fixture has proof rows
        self.assertGreater(body['splits']['train']['removed_rows'], 0)

    def test_save_load_and_refusals(self):
        keep, body = dm.build_outcome_mask(self.rows, self.cache)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'mask.pt'
            dm.save_mask(path, keep, body)
            loaded = dm.load_mask(path, body)
            self.assertTrue(all(torch.equal(loaded[s], keep[s]) for s in keep))
            with self.assertRaisesRegex(ValueError, 'never overwritten'):
                dm.save_mask(path, keep, body)
            other = {**body, 'keep_sha256': '0' * 64}
            with self.assertRaisesRegex(ValueError, 'not the mask'):
                dm.load_mask(path, other)
            cache = Path(tmp) / 'cache'
            shutil.copytree(self.cache, cache)
            games = json.loads((cache / 'train_games.json').read_text())
            (cache / 'train_games.json').write_text(json.dumps(list(reversed(games))))
            with self.assertRaisesRegex(ValueError, 'game order'):
                dm.build_outcome_mask(self.rows, cache)

    def test_label_kind_must_match_the_cache_kind(self):
        self.assertEqual(dm.label_kind_of(1, 'proof_vcf'), 'proof_vcf')
        self.assertEqual(dm.label_kind_of(0, 'outcome'), 'outcome')
        for kind, label in ((0, 'proof_vcf'), (1, 'outcome'), (2, 'outcome'), (0, 'other')):
            with self.subTest(kind=kind, label=label), self.assertRaises(ValueError):
                dm.label_kind_of(kind, label)


def tiny_caches():
    from tests.test_h7c_train import value_split
    cycle_train, _ = value_split(1, 4)
    replay_train, _ = value_split(2, 10)
    cycle_val, _ = value_split(3, 2)
    replay_val, _ = value_split(4, 2)
    for data in (cycle_train, cycle_val):  # give the cycle some proof rows (weight 1) next to the outcome rows
        data['kind'][::3] = 1
        data['weight'][::3] = 1.0
    cycle = {'train': cycle_train, 'val': cycle_val}
    keep = {s: cycle[s]['kind'] == 1 for s in cycle}
    body = {'splits': {s: {'kept_weight': float(cycle[s]['weight'].double()[keep[s]].sum()),
                           'rows': int(keep[s].numel()), 'removed_rows': int((~keep[s]).sum())} for s in cycle},
            'keep_sha256': 'k' * 64}
    return cycle, {'train': replay_train, 'val': replay_val}, keep, body


class WrapperTest(unittest.TestCase):
    def test_one_call_with_the_mask_and_the_original_selection(self):
        from tests.test_h7c_train import small_model
        cycle, replay, keep, body = tiny_caches()
        cfg = tr.ValueConfig(batch_size=16, epochs=1.0, evaluations=2, learning_rate=1e-3, h6_guard_ratio=100.0,
                             seed=31001)
        original = tr.train_value
        with d3.d3_trainer(keep, body) as calls:
            out = tr.train_value(cfg, small_model(), cycle, replay, log=QUIET)  # as h7c_train_value.run calls it
            with self.assertRaisesRegex(ValueError, 'twice'):
                tr.train_value(cfg, small_model(), cycle, replay, log=QUIET)
        self.assertIs(tr.train_value, original)  # restored
        self.assertEqual(calls, [31001])
        record = out['record']
        self.assertEqual(record['selection']['mode'], 'original')
        self.assertTrue(all(abs(e['cycle_kept_weight'] - body['splits']['train']['kept_weight']) < 1e-4
                            for e in record['mask']['epochs']))
        direct = tr.train_value(cfg, small_model(), cycle, replay, log=QUIET, cycle_keep=keep,
                                cycle_selection='original')
        self.assertEqual(record['evaluations'], direct['record']['evaluations'])  # the same call, nothing else

    def test_refusals(self):
        from tests.test_h7c_train import small_model
        cycle, replay, keep, body = tiny_caches()
        cfg = tr.ValueConfig(batch_size=16, epochs=1.0, evaluations=2)
        with d3.d3_trainer(keep, body), self.assertRaisesRegex(ValueError, 'unexpected trainer options'):
            tr.train_value(cfg, small_model(), cycle, replay, log=QUIET, cycle_keep=keep)
        wrong = {**keep, 'train': ~keep['train']}
        with d3.d3_trainer(wrong, body), self.assertRaisesRegex(ValueError, 'kind == proof'):
            tr.train_value(cfg, small_model(), cycle, replay, log=QUIET)
        heavy = {'splits': {**body['splits'], 'train': {'kept_weight': 1e9}}}
        with d3.d3_trainer(keep, heavy), self.assertRaisesRegex(ValueError, 'kept weight'):
            tr.train_value(cfg, small_model(), cycle, replay, log=QUIET)


ASSIGNMENT = {'version': 'v', 'n_cycle': 100, 'n_replay': 900, 'seed': 31001, 'batch_size': 16, 'total_steps': 13,
              'epochs': 1.0, 'augmentation_sha256': 'a' * 64, 'sha256': 'f' * 64}


def comparison(**change):
    run = {'assignment': dict(ASSIGNMENT), 'config': {'sha256': 'c' * 64}}
    trainer = {'total_steps': 13, 'train_rows_per_epoch': 200, 'selection': {'mode': 'original'},
               'mask': {'epochs': [{'cycle_kept_weight': 12.5}]}}
    control = {'assignment': dict(ASSIGNMENT), 'config': {'sha256': 'c' * 64}}
    spec = {'assignment_sha256': 'f' * 64, 'total_steps': 13}
    body = {'splits': {'train': {'kept_weight': 12.5}}}
    for target, key, value in change.get('set', ()):
        {'run': run, 'trainer': trainer, 'control': control, 'spec': spec}[target][key] = value
    return d3.compare_with_control(run, trainer, control, spec, body)


class CompareTest(unittest.TestCase):
    def test_the_same_seed_compares_directly_with_the_recorded_control(self):
        self.assertTrue(comparison()['passed'])
        cases = {'assignment_sha256': ('run', 'assignment', {**ASSIGNMENT, 'sha256': 'e' * 64}),
                 'assignment': ('run', 'assignment', {**ASSIGNMENT, 'n_replay': 901}),
                 'total_steps': ('trainer', 'total_steps', 14),
                 'train_rows_per_epoch': ('trainer', 'train_rows_per_epoch', 198),
                 'config': ('run', 'config', {'sha256': 'd' * 64}),
                 'selection_original': ('trainer', 'selection', None),
                 'mask_every_epoch': ('trainer', 'mask', {'epochs': [{'cycle_kept_weight': 12.5},
                                                                     {'cycle_kept_weight': 12.0}]})}
        for check, change in cases.items():
            with self.subTest(check):
                out = comparison(set=[change])
                self.assertFalse(out['passed'])
                self.assertFalse(out['checks'][check])


class ReadingTest(unittest.TestCase):
    eps = 0.0019461168062766864

    def paired(self, improvements, uppers):
        return {s: {'improvement': i, 'ci95': [u - 0.002, u]} for s, i, u in zip(d3.SEEDS, improvements, uppers)}

    def reproduced(self, ok=True):
        return {'reproduced': ok}

    def test_development_signal_needs_all_three_seeds(self):
        control = {s: 0.399 for s in d3.SEEDS}
        r = d3.reading(self.paired([0.003, 0.0025, 0.004], [-0.0005, -0.0001, -0.002]), self.eps, control,
                       self.reproduced())
        self.assertEqual(r['label'], 'DEVELOPMENT_SIGNAL')
        r = d3.reading(self.paired([0.003, 0.0015, 0.004], [-0.0005, -0.0001, -0.002]), self.eps, control,
                       self.reproduced())
        self.assertEqual(r['label'], 'NOT_ESTABLISHED')  # one seed not beyond epsilon
        r = d3.reading(self.paired([0.003, 0.0025, 0.004], [-0.0005, 0.0, -0.002]), self.eps, control,
                       self.reproduced())
        self.assertEqual(r['label'], 'NOT_ESTABLISHED')  # one upper bound not below 0

    def test_hold_on_unreproduced_controls_or_a_wide_control_range(self):
        good = self.paired([0.003] * 3, [-0.001] * 3)
        r = d3.reading(good, self.eps, {s: 0.399 for s in d3.SEEDS}, self.reproduced(False))
        self.assertEqual(r['label'], 'HOLD')
        wide = {'31001': 0.396, '31002': 0.399, '31003': 0.3985}
        r = d3.reading(good, self.eps, wide, self.reproduced())
        self.assertEqual(r['label'], 'HOLD')
        self.assertIn('seed range', r['hold'][0])

    def test_control_reproduction_is_an_aggregate_metric_tolerance(self):
        recorded = {s: [0.399, 0.74, 0.10, 0.42] for s in d3.SEEDS}
        near = {s: [v + 9e-6 for v in recorded[s]] for s in d3.SEEDS}
        self.assertTrue(d3.control_reproduction(near, recorded, 1e-5)['reproduced'])
        far = deepcopy(near)
        far['31002'][2] += 2e-5
        out = d3.control_reproduction(far, recorded, 1e-5)
        self.assertFalse(out['reproduced'])
        self.assertEqual(len(out['failures']), 1)
        self.assertIn('control 31002 metric 2', out['failures'][0])


class AnalyseTest(unittest.TestCase):
    """The pure part of ``evaluate`` on synthetic H-dev rows: control reproduction, paired reading, references."""

    def world(self, d3_gain=0.5):
        from random import Random
        from tests.test_h7r_d3 import fixture
        f = fixture(n_games=30, seed=5)
        info = {'game': f['game'], 'ply': f['ply'], 'game_ids': [f'g{i}' for i in range(f['n_games'])],
                'result': ['black' if g % 2 else 'white' for g in f['game']], 'label_kind': f['label_kind']}
        rnd = Random(1)
        noise = lambda scale: [max(-1.0, min(1.0, t + rnd.uniform(-scale, scale))) for t in f['target']]  # noqa: E731
        pred = {'c5': {'d4': noise(0.9), 'raw': noise(0.9)}}
        control = {'d4': noise(0.8), 'raw': noise(0.8)}  # one control world: a seed range of 0 (no hold)
        for s in d3.SEEDS:
            pred[f'control{s}'] = {k: list(v) for k, v in control.items()}
            pred[f'd3{s}'] = {'d4': noise(0.8 * (1 - d3_gain)), 'raw': noise(0.8 * (1 - d3_gain))}
        r2c = json.loads((ROOT / 'docs/h7r/r2/R2-DIAG-1.contract.json').read_text())
        metrics = [r2c['metrics']['primary'], *r2c['metrics']['components']]
        groups = r2.row_groups(f['route'], f['kind'], f['ply'], f['target'], info['result'], f['label_kind'],
                               [5, 20, 40, 80])
        value = lambda name, m: r2.point_metrics(r2.game_sums(pred[name][m['prediction']], f['target'], f['weight'],
                                                              f['game'], groups[m['group']], f['n_games']))[m['aggregate']]  # noqa: E731
        recorded = {s: {'primary': value(f'control{s}', metrics[0]),
                        'components': [value(f'control{s}', m) for m in metrics[1:]]} for s in d3.SEEDS}
        return f, info, pred, metrics, recorded

    def run_analyse(self, f, info, pred, metrics, recorded, epsilon=1e-4):
        counts = r2.multiplicities(f['n_games'], 300, 3)
        return d3.analyse(pred, f['route'], f['kind'], info, f['target'], f['weight'], [5, 20, 40, 80],
                          {'tolerance': 1e-5, 'metrics': metrics}, recorded, epsilon, counts, 0.95)

    def test_reproduced_controls_and_better_d3_runs_give_the_signal(self):
        f, info, pred, metrics, recorded = self.world()
        reproduced, paired, references, result = self.run_analyse(f, info, pred, metrics, recorded)
        self.assertTrue(reproduced['reproduced'])
        self.assertTrue(all(p['improvement'] > 0 and p['ci95'][1] < 0 for p in paired.values()))
        self.assertEqual(result['label'], 'DEVELOPMENT_SIGNAL')
        for s in d3.SEEDS:
            parts = references[s]['decomposition_kind']
            self.assertAlmostEqual(sum(v['contribution'] for v in parts.values()), paired[s]['difference'], places=12)
            self.assertEqual(len(references[s]['components']), len(metrics) - 1)

    def test_unreproduced_controls_hold_the_reading(self):
        f, info, pred, metrics, recorded = self.world()
        recorded['31002']['components'][1] += 2e-5
        reproduced, _, _, result = self.run_analyse(f, info, pred, metrics, recorded)
        self.assertFalse(reproduced['reproduced'])
        self.assertEqual(result['label'], 'HOLD')

    def test_a_wide_control_seed_range_holds_the_reading(self):
        f, info, pred, metrics, recorded = self.world()
        pred['control31003'] = {k: [max(-1.0, min(1.0, v + 0.3)) for v in vs] for k, vs in pred['control31003'].items()}
        groups = r2.row_groups(f['route'], f['kind'], f['ply'], f['target'], info['result'], f['label_kind'],
                               [5, 20, 40, 80])
        recorded['31003'] = {'primary': r2.point_metrics(r2.game_sums(pred['control31003']['d4'], f['target'],
                                                                      f['weight'], f['game'], groups['network'],
                                                                      f['n_games']))['macro'],
                             'components': [r2.point_metrics(r2.game_sums(pred['control31003'][m['prediction']],
                                                                          f['target'], f['weight'], f['game'],
                                                                          groups[m['group']], f['n_games']))[
                                                m['aggregate']] for m in metrics[1:]]}
        _, _, _, result = self.run_analyse(f, info, pred, metrics, recorded)
        self.assertEqual(result['label'], 'HOLD')
        self.assertIn('seed range', result['hold'][0])

    def test_no_gain_is_not_established(self):
        f, info, pred, metrics, recorded = self.world(d3_gain=0.0)
        _, _, _, result = self.run_analyse(f, info, pred, metrics, recorded, epsilon=0.5)
        self.assertEqual(result['label'], 'NOT_ESTABLISHED')


class ContractTest(unittest.TestCase):
    def load(self, contract, stage='mask'):
        with TemporaryDirectory(dir=ROOT / 'docs') as tmp:
            path = Path(tmp) / 'd3.json'
            path.write_text(json.dumps(contract))
            with mock.patch('hybrid.h7c_nmask.committed', return_value=True):
                return d3.load_d3(path, ROOT, stage)

    def approved(self):
        c = json.loads((ROOT / CONTRACT).read_text())
        c['approved'] = {s: True for s in d3.STAGES}
        return c

    def test_the_committed_contract_approves_mask_and_train_only(self):
        c = json.loads((ROOT / CONTRACT).read_text())
        self.assertEqual((c['status'], c['approved']), ('fixed', {'mask': True, 'train': True, 'evaluate': False}))
        for stage in ('evaluate',):
            with self.subTest(stage), self.assertRaisesRegex(ValueError, f'{stage} stage is not approved'):
                d3.load_d3(ROOT / CONTRACT, ROOT, stage)
        unapproved = {**c, 'approved': {s: False for s in d3.STAGES}}
        self.assertEqual(d3.experiment_sha256(unapproved), d3.experiment_sha256(c))  # the approval is not the experiment

    def test_an_approved_copy_loads_against_the_real_records(self):
        c, _, eid = self.load(self.approved(), 'train')
        self.assertEqual(eid, d3.experiment_sha256(json.loads((ROOT / CONTRACT).read_text())))  # approvals aside
        self.assertEqual(c['controls']['31001']['total_steps'], 210)
        self.assertEqual(c['reference']['proof_contribution_same_recipe_range'], [-0.000771, 0.000656])
        self.assertEqual(c['control_reproduction']['tolerance'], 1e-5)

    def test_a_stage_needs_the_earlier_stages_approved(self):
        c = self.approved()
        c['approved']['mask'] = False
        with self.assertRaisesRegex(ValueError, 'train stage needs'):
            self.load(c, 'train')
        c['approved'].update(mask=True, train=False)
        with self.assertRaisesRegex(ValueError, 'evaluate stage needs'):
            self.load(c, 'evaluate')

    def test_wrong_pins_are_refused(self):
        cases = {
            'pinned values are not': lambda c: c['controls']['31002'].update(best_step=85),
            'not the R2-0B epsilon': lambda c: c['ruling'].update(epsilon=0.002),
            'condition differs': lambda c: c['condition'].update(cycle_selection='masked'),
            'checkout differs': lambda c: c['sources'].update({'scripts/h7r_d3.py': '0' * 64}),
            'R0 reproduced': lambda c: c['sources'].update({'src/hybrid/h7c_train.py': '1' * 64}),
            'primary and component': lambda c: c['control_reproduction'].update(metrics=[]),
            'under runs/': lambda c: c['output'].update(mask='docs/mask.pt'),
            'keys differ': lambda c: c.update(extra=1),
            'one bool per stage': lambda c: c.update(approved={'mask': True}),
        }
        for message, change in cases.items():
            with self.subTest(message):
                c = self.approved()
                change(c)
                with self.assertRaisesRegex(ValueError, message):
                    self.load(c)


class TrainWiringTest(unittest.TestCase):
    """``train`` with seed.train replaced by a stand-in that calls the trainer as h7c_train_value.run does."""

    def test_records_the_comparison_and_refuses_a_second_run(self):
        from tests.test_h7c_train import small_model
        c = json.loads((ROOT / CONTRACT).read_text())
        control = json.loads((ROOT / c['controls']['31001']['run_record']['path']).read_text())
        cycle, replay, keep, body = tiny_caches()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_train(ns, log=None):
                out = tr.train_value(tr.ValueConfig(batch_size=16, epochs=1.0, evaluations=2, seed=31001,
                                                    h6_guard_ratio=100.0),
                                     small_model(), cycle, replay, log=QUIET)
                ns.out_dir.mkdir(parents=True)
                (ns.out_dir / 'record.json').write_text(json.dumps({**out['record'], 'config': {}}))
                run = {'assignment': control['assignment'], 'config': control['config'], 'verdict': 'CANDIDATE',
                       'best_step': out['record']['best']['step'], 'candidate': None}
                (ns.out_dir / 'r2_train.json').write_text(json.dumps(run))
                self.seen = ns
                return run
            contract = c
            args = SimpleNamespace(d3_contract='x', root=root, seed='31001', cycle_cache='cc', cycle_rows='cr',
                                   replay_cache='rc', parent='p', device='cpu')
            with mock.patch.object(d3, 'load_d3', return_value=(contract, 's' * 64, 'e' * 64)), \
                    mock.patch.object(d3, 'checked_mask', return_value=(keep, body)), \
                    mock.patch.object(d3, 'git_state', return_value={'commit': 'c' * 40, 'dirty': False}), \
                    mock.patch.object(d3, 'pinned', return_value=control), \
                    mock.patch('h7r_r2_seed.train', side_effect=fake_train):
                (root / c['output']['mask_manifest']).parent.mkdir(parents=True)
                (root / c['output']['mask_manifest']).write_text('{}')
                with self.assertRaisesRegex(ValueError, 'not comparable with its control'):
                    d3.train(args, log=QUIET)  # the stand-in trainer runs 13 steps, the control 210
                saved = json.loads((root / c['output']['run_dirs']['31001'] / 'd3_train.json').read_text())
                self.assertFalse(saved['comparison']['passed'])  # the failed comparison is kept
                self.assertTrue(saved['comparison']['checks']['assignment'])
                self.assertTrue(saved['comparison']['checks']['selection_original'])
                self.assertEqual(self.seen.seed, 31001)
                self.assertEqual(self.seen.reproduction, root / c['r0']['reproduction']['path'])
                with self.assertRaisesRegex(ValueError, 'never overwritten'):
                    d3.train(args, log=QUIET)


class ProcedureTest(unittest.TestCase):
    def test_the_draft_blocks_parse_and_fail_closed(self):
        import re
        import subprocess
        doc = (ROOT / 'docs/h7r/r2/R2-D3-NO-OUTCOME-1-draft.md').read_text()
        section = re.split(r'\n##+ ', doc.split('### 2.6', 1)[1], maxsplit=1)[0]  # up to the next heading
        blocks = re.findall(r'```bash\n(.*?)```', section, re.S)
        self.assertEqual(len(blocks), 4)
        for block in blocks:
            self.assertEqual(subprocess.run(['bash', '-n'], input=block, text=True).returncode, 0)
            self.assertIn('set -euo pipefail', block)
            self.assertIn('__EXIT__', block)


if __name__ == '__main__':
    unittest.main()
