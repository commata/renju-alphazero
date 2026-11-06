"""R2-D1-HEAD-WIDTH-1 (1,000-row stage): the prediction-preserving widening (eval and train mode, its byte checks, its
own random generator), the wide checkpoint saved and loaded back through the evaluators' loader, the fit probe on the
real trainer (no change to the run, the parent fit equal in both arms, the last-step state kept apart from the
selected checkpoint), the subset, the floors, the two readings, the contract on the real records, and the train and
fit stages wired end to end on stand-in data."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from hybrid import h7c_train as tr
from hybrid import h7r_d1 as d1
from hybrid import h7r_r2 as r2
from scripts import h7r_d1 as cli
from tests.test_h7c_train import small_model, value_split

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1/d1-1k-contract.json'
QUIET = lambda *a, **k: None  # noqa: E731


def caches():
    cycle_train, _ = value_split(1, 4)
    replay_train, _ = value_split(2, 10)
    cycle_val, _ = value_split(3, 2)
    replay_val, _ = value_split(4, 2)
    for data in (cycle_train, cycle_val):  # some proof rows next to the outcome rows
        data['kind'][::3] = 1
        data['weight'][::3] = 1.0
    return {'train': cycle_train, 'val': cycle_val}, {'train': replay_train, 'val': replay_val}


class WidenTest(unittest.TestCase):
    def test_prediction_preserved_in_eval_and_train_mode(self):
        parent = small_model()
        cycle, _ = caches()
        wide = d1.widen_value_head(parent, 4, 2026110710)
        self.assertEqual(wide.config.value_channels, 4)
        self.assertEqual(asdict(wide.config), {**asdict(parent.config), 'value_channels': 4})
        checks = d1.preservation(parent, wide)
        self.assertTrue(checks['passed'], checks)
        for data in (cycle['train'], cycle['val']):
            diff = (d1.predict_rows(wide, data, 'cpu') - d1.predict_rows(parent, data, 'cpu')).abs().max()
            self.assertLessEqual(float(diff), 1e-6)
        self.assertLessEqual(d1.train_mode_difference(parent, wide, cycle['train'], 'cpu'), 1e-6)
        # the new channels are not dead: nonzero conv weights, zero outgoing columns
        w = wide.state_dict()
        self.assertGreater(float(w['value_head.0.weight'][1:].abs().sum()), 0)
        self.assertEqual(float(w['value_head.4.weight'][:, 225:].abs().sum()), 0.0)

    def test_own_generator_deterministic_and_trainer_streams_untouched(self):
        parent = small_model()
        torch.manual_seed(5)
        before = torch.get_rng_state()
        a = d1.widen_value_head(parent, 4, 2026110710)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))  # the global stream is not consumed
        b = d1.widen_value_head(parent, 4, 2026110710)
        c = d1.widen_value_head(parent, 4, 2026110711)
        key = 'value_head.0.weight'
        self.assertTrue(torch.equal(a.state_dict()[key], b.state_dict()[key]))
        self.assertFalse(torch.equal(a.state_dict()[key][1:], c.state_dict()[key][1:]))
        self.assertTrue(torch.equal(a.state_dict()[key][:1], c.state_dict()[key][:1]))
        with self.assertRaisesRegex(ValueError, 'not wider'):
            d1.widen_value_head(parent, 1, 1)

    def test_a_broken_widening_fails_its_checks(self):
        parent = small_model()
        wide = d1.widen_value_head(parent, 4, 1)
        state = wide.state_dict()
        state['value_head.4.weight'][0, 300] = 0.5
        wide.load_state_dict(state)
        checks = d1.preservation(parent, wide)
        self.assertFalse(checks['passed'])
        self.assertFalse(checks['checks']['new_columns_zero'])


class CheckpointTest(unittest.TestCase):
    def test_the_wide_checkpoint_loads_through_the_evaluators_loader(self):
        from model.checkpoint import load_checkpoint, save_checkpoint
        from model.config import ModelConfig
        from hybrid.h6_value_train import frozen_state, state_sha256
        parent = small_model()
        wide = d1.widen_value_head(parent, 4, 2026110710)
        cycle, _ = caches()
        with TemporaryDirectory() as tmp:
            ckpt, meta = Path(tmp) / 'wide.pt', Path(tmp) / 'wide.json'
            save_checkpoint(ckpt, wide)
            frozen = state_sha256(frozen_state(parent))
            meta.write_text(json.dumps({'model_config': asdict(wide.config), 'frozen_state_sha256': frozen}))
            loaded, _ = r2.load_value_model(ckpt, meta, 'cpu')
            self.assertEqual(loaded.config.value_channels, 4)
            self.assertTrue(torch.equal(d1.predict_rows(loaded, cycle['train'], 'cpu'),
                                        d1.predict_rows(wide, cycle['train'], 'cpu')))
            spec = {'value_checkpoint_sha256': r2.file_sha256(ckpt), 'value_metadata_sha256': r2.file_sha256(meta),
                    'policy_checkpoint_sha256': 'p' * 64, 'policy_metadata_sha256': 'q' * 64}
            model, identity = r2.check_model('wide', spec, ckpt, meta, frozen, 'cpu')
            self.assertEqual(identity['frozen_state_sha256']['recomputed'], frozen)
            with self.assertRaises((ValueError, RuntimeError)):  # the narrow architecture cannot load it
                load_checkpoint(ckpt, ModelConfig())


class FitProbeTest(unittest.TestCase):
    def run_arm(self, parent, cycle, replay, probe=True, seed=31001):
        cfg = tr.ValueConfig(batch_size=16, epochs=1.0, evaluations=3, learning_rate=1e-3, h6_guard_ratio=100.0,
                             seed=seed)
        n = 2 * cycle['train']['value'].shape[0]
        total = max(1, round(cfg.epochs * -(-n // cfg.batch_size)))
        steps = tr.evaluation_steps(total, cfg.evaluations)
        if not probe:
            return tr.train_value(cfg, parent, cycle, replay, log=QUIET), None, steps
        with d1.fit_probe(cycle['train'], steps, 'cpu') as seen:
            out = tr.train_value(cfg, parent, cycle, replay, log=QUIET)
        return out, seen, steps

    def test_the_probe_does_not_change_the_run_and_keeps_the_last_step(self):
        cycle, replay = caches()
        plain, _, _ = self.run_arm(small_model(), cycle, replay, probe=False)
        probed, seen, steps = self.run_arm(small_model(), cycle, replay)
        self.assertEqual(plain['record']['evaluations'], probed['record']['evaluations'])
        self.assertTrue(all(torch.equal(plain['state'][k], probed['state'][k]) for k in plain['state']))
        self.assertEqual([f['at'] for f in seen['fits']], ['parent', *steps])
        model = small_model()
        model.load_state_dict(seen['final_state'])
        from hybrid.h6_value_train import evaluate
        self.assertAlmostEqual(seen['fits'][-1]['weighted_mse'], evaluate(model, cycle['train'], 'cpu')['weighted_mse'],
                               places=6)
        best = probed['record']['best']['step']
        selected = seen['fits'][1 + steps.index(best)]
        model.load_state_dict(probed['state'])
        self.assertAlmostEqual(selected['weighted_mse'], evaluate(model, cycle['train'], 'cpu')['weighted_mse'],
                               places=6)

    def test_both_arms_start_from_the_same_fit_with_the_same_assignment(self):
        cycle, replay = caches()
        parent = small_model()
        _, narrow, _ = self.run_arm(parent, cycle, replay)
        _, wide, _ = self.run_arm(d1.widen_value_head(parent, 4, 2026110710), cycle, replay)
        self.assertAlmostEqual(narrow['fits'][0]['weighted_mse'], wide['fits'][0]['weighted_mse'], places=6)
        n, m = cycle['train']['value'].shape[0], replay['train']['value'].shape[0]
        self.assertEqual(r2.data_assignment(n, m, 31001, 16, 1.0, True), r2.data_assignment(n, m, 31001, 16, 1.0, True))


class SubsetFloorTest(unittest.TestCase):
    def test_whole_games_nested_and_seeded(self):
        ids = [f'g{i:02d}' for i in range(40)]
        rows = {g: 10 + i % 7 for i, g in enumerate(ids)}
        out = d1.select_subset(ids, rows, 7, [100, 300])
        small, big = out[100], out[300]
        self.assertEqual(big['games'][:len(small['games'])], small['games'])  # nested prefix
        self.assertGreaterEqual(small['rows'], 100)
        self.assertLess(small['rows'] - rows[small['games'][-1]], 100)  # the first prefix that reaches it
        self.assertEqual(d1.select_subset(list(reversed(ids)), rows, 7, [100, 300]), out)  # order-independent
        self.assertNotEqual(d1.select_subset(ids, rows, 8, [100])[100]['games'], small['games'])
        with self.assertRaisesRegex(ValueError, 'fewer rows'):
            d1.select_subset(ids, rows, 7, [10_000])

    def test_floor_of_conflicting_targets(self):
        keys = ['a', 'a', 'b', 'c', 'c']
        target = torch.tensor([1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
        weight = torch.tensor([0.1, 0.1, 1.0, 0.1, 1.0], dtype=torch.float64)
        out = d1.floor(keys, target, weight)
        self.assertAlmostEqual(out['floor'], (0.1 * 1 + 0.1 * 1) / 2.3)  # group a predicted 0
        self.assertEqual((out['groups'], out['conflicting_groups']), (3, 1))
        cycle, _ = caches()
        both = d1.floors(cycle['train'])
        self.assertGreaterEqual(both['d4']['floor'], both['exact']['floor'] - 1e-12)  # D4 merges more rows
        self.assertLessEqual(both['d4']['groups'], both['exact']['groups'])


class ReadingTest(unittest.TestCase):
    seeds = ('31001', '31002', '31003')

    def final(self, narrow, wide):
        return {'narrow': dict(zip(self.seeds, narrow)), 'wide': dict(zip(self.seeds, wide))}

    def test_fit_categories(self):
        cases = {
            'WIDTH_EFFECT': ([0.20, 0.21, 0.205], [0.15, 0.16, 0.15]),
            'BOTH_FIT': ([0.012, 0.011, 0.0115], [0.011, 0.0112, 0.0118]),  # mixed signs: not read when all fit
            'SEED_DISAGREEMENT': ([0.20, 0.21, 0.205], [0.15, 0.22, 0.15]),
            'NEITHER_FIT': ([0.30, 0.31, 0.32], [0.299, 0.309, 0.319]),  # wide lower, but within the narrow spread
            'MIXED_FIT': ([0.016, 0.017, 0.0165], [0.0145, 0.0165, 0.016]),  # one fits, the rest do not
        }
        self.assertEqual(set(cases), set(d1.FIT_CATEGORIES))
        for category, (narrow, wide) in cases.items():
            with self.subTest(category):
                out = d1.fit_reading(self.final(narrow, wide), 0.005, 0.01, self.seeds)
                self.assertEqual(out['category'], category, out)

    def test_hdev_rule(self):
        def paired(i, upper):
            return {s: {'improvement': a, 'ci95': [b - 0.003, b]} for s, a, b in zip(self.seeds, i, upper)}
        control = dict(zip(self.seeds, [0.40, 0.4005, 0.4001]))
        eps = 0.0019461168062766864
        self.assertEqual(d1.hdev_reading(paired([0.003] * 3, [-0.001] * 3), eps, control, self.seeds)['label'],
                         'DEVELOPMENT_SIGNAL')
        self.assertEqual(d1.hdev_reading(paired([0.003, 0.001, 0.003], [-0.001] * 3), eps, control, self.seeds)
                         ['label'], 'NOT_ESTABLISHED')
        wide_range = dict(zip(self.seeds, [0.40, 0.403, 0.401]))
        self.assertEqual(d1.hdev_reading(paired([0.003] * 3, [-0.001] * 3), eps, wide_range, self.seeds)['label'],
                         'HOLD')


class ContractTest(unittest.TestCase):
    def load(self, contract, stage='prepare'):
        with TemporaryDirectory(dir=ROOT / 'docs') as tmp:
            path = Path(tmp) / 'd1.json'
            path.write_text(json.dumps(contract))
            with mock.patch('hybrid.h7c_nmask.committed', return_value=True):
                return cli.load_d1(path, ROOT, stage)

    def approved(self):
        c = json.loads((ROOT / CONTRACT).read_text())
        c['approved'] = {s: True for s in cli.STAGES}
        return c

    def test_the_committed_approvals_are_a_prefix_of_the_stages(self):
        c = json.loads((ROOT / CONTRACT).read_text())
        flags = [c['approved'][s] for s in cli.STAGES]
        self.assertEqual(flags, sorted(flags, reverse=True))  # approved stages first, never a later one alone
        for stage, flag in zip(cli.STAGES, flags):
            if not flag:
                with self.subTest(stage), self.assertRaisesRegex(ValueError, f'{stage} stage is not approved'):
                    cli.load_d1(ROOT / CONTRACT, ROOT, stage)

    def test_an_approved_copy_loads_and_the_experiment_ignores_approvals(self):
        c, _, eid = self.load(self.approved(), 'evaluate')
        self.assertEqual((c['budget']['total_steps'], c['fit']['margin'], c['subset']['rows']), (210, 0.01, 1000))
        self.assertEqual(eid, cli.experiment_sha256(json.loads((ROOT / CONTRACT).read_text())))

    def test_wrong_pins_are_refused(self):
        cases = {
            'arms differ': lambda c: c['arms']['wide'].update(value_channels=8),
            'not the R2-0B epsilon': lambda c: c['ruling'].update(epsilon=0.002),
            'not the A1': lambda c: c['budget'].update(config={'path': 'x.json', 'sha256': '0' * 64}),
            'checkout differs': lambda c: c['sources'].update({'scripts/h7r_d1.py': '0' * 64}),
            '5,000 rows': lambda c: c['subset'].update(rows=2000),
            'last step': lambda c: c['fit'].update(measured_at='selected'),
            'under runs/': lambda c: c['output'].update(predictions='docs/p.pt'),
            'keys differ': lambda c: c.update(extra=1),
        }
        for message, change in cases.items():
            with self.subTest(message):
                c = self.approved()
                change(c)
                with self.assertRaisesRegex(ValueError, message):
                    self.load(c)

    def test_a_stage_needs_the_earlier_stages(self):
        c = self.approved()
        c['approved']['prepare'] = False
        with self.assertRaisesRegex(ValueError, 'train stage needs'):
            self.load(c, 'train')


class TrainFitWiringTest(unittest.TestCase):
    """``train`` for both arms and three seeds and then ``fit``, on stand-in caches and a stand-in c5."""

    def test_six_runs_then_the_fit_reading(self):
        from model.checkpoint import save_checkpoint
        from hybrid.h6_value_train import frozen_state, state_sha256
        c = json.loads((ROOT / CONTRACT).read_text())
        c['budget'] = {**c['budget'], 'total_steps': 4}
        cycle, replay = caches()
        parent = small_model()
        frozen = state_sha256(frozen_state(parent))
        r2c = {'frozen_state_sha256': frozen, 'models': {'c5': {'value_checkpoint_sha256': 'v' * 64}}}
        config = {'seed': 1, 'batch_size': 16, 'epochs': 1.0, 'learning_rate': 1e-3, 'weight_decay': 1e-4,
                  'min_lr_ratio': 0.05, 'augment_d4': True, 'evaluations': 2, 'h6_guard_ratio': 100.0}
        desc = {'games': 4, 'rows': int(cycle['train']['value'].shape[0]), 'games_sha256': 'g' * 64, 'nested': {}}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parents = root / c['output']['parents']
            parents.mkdir(parents=True)
            widened = {}
            for s in cli.SEEDS:
                wide = d1.widen_value_head(parent, 4, c['widen']['init_seeds'][s])
                ckpt, meta = cli.parent_paths(c, root, s)
                save_checkpoint(ckpt, wide)
                meta.write_text(json.dumps({'model_config': asdict(wide.config), 'frozen_state_sha256': frozen}))
                widened[s] = {'checkpoint_sha256': r2.file_sha256(ckpt), 'metadata_sha256': r2.file_sha256(meta)}
            prep = {'subset': desc, 'widened': widened, 'floors': {'d4': {'floor': 0.0}}}
            args = lambda arm, s: SimpleNamespace(d1_contract='x', root=root, arm=arm, seed=s, cycle_cache='cc',  # noqa: E731
                                                  cycle_rows='cr', replay_cache='rc', c5='c5.pt', device='cpu')
            with mock.patch.object(cli, 'load_d1', return_value=(c, 's' * 64, 'e' * 64)), \
                    mock.patch.object(cli, 'git_state', return_value={'commit': 'c' * 40, 'dirty': False}), \
                    mock.patch.object(cli, 'checked_prepare', return_value=prep), \
                    mock.patch.object(cli, 'reference_inputs', return_value=(r2c, {}, config)), \
                    mock.patch.object(cli, 'checked_caches', return_value=(cycle, replay)), \
                    mock.patch.object(cli, 'subset_of', return_value=(cycle['train'], desc)), \
                    mock.patch.object(cli, 'load_c5', return_value=(copy.deepcopy(parent), {})):
                runs = {(arm, s): cli.train(args(arm, s), log=QUIET) for arm in cli.ARMS for s in cli.SEEDS}
                with self.assertRaisesRegex(ValueError, 'never overwritten'):
                    cli.train(args('wide', '31001'), log=QUIET)
                out = cli.fit(SimpleNamespace(d1_contract='x', root=root), log=QUIET)
            for (arm, s), rec in runs.items():
                self.assertEqual(rec['total_steps'], 4)
                self.assertEqual(rec['last_step']['step'], 4)
                self.assertEqual(rec['value_channels'], 1 if arm == 'narrow' else 4)
                self.assertEqual(rec['fits'][0]['at'], 'parent')
                run_dir = root / c['output']['run_dirs'][arm][s]
                loaded, _ = r2.load_value_model(run_dir / 'final.pt', run_dir / 'final.json', 'cpu')
                self.assertEqual(loaded.config.value_channels, rec['value_channels'])
                if rec['selected']['state']:
                    self.assertTrue((run_dir / 'best.pt').is_file())
            for s in cli.SEEDS:  # paired: the same assignment, and the same starting fit (prediction preserved)
                a, b = runs[('narrow', s)], runs[('wide', s)]
                self.assertEqual(a['assignment'], b['assignment'])
                self.assertAlmostEqual(a['fits'][0]['weighted_mse'], b['fits'][0]['weighted_mse'], places=6)
            self.assertIn(out['reading']['category'], d1.FIT_CATEGORIES)
            self.assertEqual(set(out['selected_step']), set(cli.ARMS))
            # evaluate needs this train-fit record committed and equal to the run records
            recs = cli.run_records(c, root, 'e' * 64)
            with mock.patch('hybrid.h7c_nmask.committed', return_value=False):
                with self.assertRaisesRegex(ValueError, 'not committed'):
                    cli.checked_fit(c, root, 'e' * 64, prep, recs)
            with mock.patch('hybrid.h7c_nmask.committed', return_value=True):
                self.assertEqual(cli.checked_fit(c, root, 'e' * 64, prep, recs), json.loads(json.dumps(out['reading'])))
                path = root / c['output']['fit']
                edited = json.loads(path.read_text())
                was = out['reading']['category']
                edited['reading']['category'] = 'BOTH_FIT' if was == 'WIDTH_EFFECT' else 'WIDTH_EFFECT'
                path.write_text(json.dumps(edited))
                with self.assertRaisesRegex(ValueError, 'differs from the run records'):
                    cli.checked_fit(c, root, 'e' * 64, prep, recs)


class ProcedureTest(unittest.TestCase):
    def test_the_blocks_parse_and_fail_closed(self):
        import re
        import subprocess
        doc = (ROOT / 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1-draft.md').read_text()
        section = re.split(r'\n## ', doc.split('\n## 10.', 1)[1], maxsplit=1)[0]
        blocks = re.findall(r'```bash\n(.*?)```', section, re.S)
        self.assertEqual(len(blocks), 4)
        for block in blocks:
            self.assertEqual(subprocess.run(['bash', '-n'], input=block, text=True).returncode, 0)
            self.assertIn('set -euo pipefail', block)
            self.assertIn('__EXIT__', block)


if __name__ == '__main__':
    unittest.main()
