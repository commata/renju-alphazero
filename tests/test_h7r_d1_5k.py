"""The D1 5,000-row stage (``scripts/h7r_d1_5k.py``): its contract pins (the 1,000-row records it links to, the
fixed values it keeps, the subset it expects), the 5,000-row prefix equal to the nested set of the 1,000-row
selection, prepare and train equal to the pinned 1,000-row tool on the same stand-in inputs (with the parent
state check added), the train-fit reading, the H-dev evaluation equal to the A1 tool with its interruption and
resume, and the 1,000-row contract, amendment and records still loading unchanged."""
import contextlib
import copy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from hybrid import h7r_d1 as d1
from hybrid import h7r_r2 as r2
from scripts import h7r_d1 as cli1
from tests.test_h7c_train import small_model, value_split
from tests.test_h7r_d1 import caches
import h7r_d1 as cli  # the module the 5k tool and the A1 tool import
import h7r_d1_5k as k5
import h7r_d1_evaluate as ev
from tests import test_h7r_d1_evaluate as te

ROOT = Path(__file__).resolve().parents[1]
D = 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1'
CONTRACT = f'{D}/d1-5k-contract.json'
V2 = f'{D}/d1-1k-v2-contract.json'
QUIET = lambda *a, **k: None  # noqa: E731


def contract():
    return json.loads((ROOT / CONTRACT).read_text())


class LoaderTest(unittest.TestCase):
    def load(self, c, stage='prepare'):
        with TemporaryDirectory(dir=ROOT / 'docs') as tmp:
            path = Path(tmp) / 'd1-5k.json'
            path.write_text(json.dumps(c))
            with mock.patch('hybrid.h7c_nmask.committed', return_value=True):
                return k5.load_d1_5k(path, ROOT, stage)

    def approved(self):
        c = contract()
        c['approved'] = {s: True for s in k5.STAGES}
        return c

    def test_the_committed_contract(self):
        c = contract()
        self.assertEqual((c['status'], c['stage'], c['format']), ('fixed', '5k', k5.FORMAT))
        flags = [c['approved'][s] for s in k5.STAGES]
        self.assertEqual(flags, sorted(flags, reverse=True))
        for stage, flag in zip(k5.STAGES, flags):
            if not flag:
                with self.subTest(stage), self.assertRaisesRegex(ValueError, f'{stage} stage is not approved'):
                    k5.load_d1_5k(ROOT / CONTRACT, ROOT, stage)
        draft = json.loads((ROOT / f'{D}/d1-5k-contract.draft.json').read_text())
        same = {k for k in draft if k not in ('status', 'approved', 'sources', 'note')}
        self.assertEqual({k: c[k] for k in same}, {k: draft[k] for k in same})  # fixed = the reviewed draft

    def test_an_approved_copy_loads_every_stage(self):
        for stage in k5.STAGES:
            c, _, eid = self.load(self.approved(), stage)
        self.assertEqual(eid, cli1.experiment_sha256(contract()))
        self.assertNotEqual(eid, cli1.experiment_sha256(json.loads((ROOT / V2).read_text())))

    def test_wrong_pins_are_refused(self):
        cases = {
            'a draft is never run': lambda c: c.update(status='draft'),
            'every file pinned': lambda c: c['sources'].update({k5.TOOL_PATH: None}),
            'checkout differs': lambda c: c['sources'].update({'scripts/h7r_d1.py': '0' * 64}),
            'nested 5,000-row prefix': lambda c: c['subset']['expected'].update(rows=5175),
            '210 steps': lambda c: c['budget'].update(total_steps=1155),
            'not the 1,000-row arms': lambda c: c['arms']['wide'].update(value_channels=8),
            'not the 1,000-row fit': lambda c: c['fit'].update(margin=0.02),
            'not the 1,000-row rules': lambda c: c['ruling'].update(hdev='x'),
            'construction, init seeds': lambda c: c['widen']['exact'].update(tolerance=1e-9),
            'not the 1,000-row v2 experiment': lambda c: c['predecessor'].update(experiment_sha256='0' * 64),
            'not the passed 1,000-row prepare': lambda c: c['predecessor']['prepare'].update(
                path=c['predecessor']['fit']['path'], sha256=c['predecessor']['fit']['sha256']),
            'readings differ': lambda c: c['predecessor']['readings'].update(hdev='HOLD'),
            'records under docs': lambda c: c['output'].update(fit='docs/other/fit.json'),
            'predictions record under runs': lambda c: c['evaluate'].update(predicted_record='docs/x.json'),
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
        c['approved']['train'] = False
        with self.assertRaisesRegex(ValueError, 'evaluate stage needs'):
            self.load(c, 'evaluate')


class SubsetTest(unittest.TestCase):
    """The 5,000-row prefix of the 5k tool is the ``nested`` set of the 1,000-row selection (scaled down)."""

    def setup(self, tmp):
        cycle_train, _ = value_split(11, 30)
        ids = [f'game-{g:03d}' for g in range(30)]
        (Path(tmp) / 'train_games.json').write_text(json.dumps(ids))
        return {'train': cycle_train}

    def test_the_prefix_is_the_recorded_nested_set(self):
        with TemporaryDirectory() as tmp:
            cycle = self.setup(tmp)
            v2like = {'subset': {'seed': 2026110701, 'rows': 40, 'nested_rows': 120}}
            _, desc1 = cli1.subset_of(v2like, cycle, tmp)
            c = {'subset': {'seed': 2026110701, 'rows': 120, 'expected': desc1['nested']}}
            data, desc = k5.subset_5k(c, cycle, tmp)
            self.assertEqual(desc, desc1['nested'])
            self.assertEqual(int(data['value'].shape[0]), desc['rows'])
            small, _ = cli1.subset_of(v2like, cycle, tmp)
            self.assertLess(int(small['value'].shape[0]), desc['rows'])
            c['subset']['seed'] = 2026110799
            with self.assertRaisesRegex(ValueError, 'not the nested set'):
                k5.subset_5k(c, cycle, tmp)


@contextlib.contextmanager
def stack(patches):
    with contextlib.ExitStack() as s:
        for p in patches:
            s.enter_context(p)
        yield


class PrepareTrainTest(unittest.TestCase):
    """prepare and train of the 5k tool against the pinned 1,000-row tool on the same stand-in inputs."""

    def inputs(self, root):
        cycle, replay = caches()
        parent = small_model()
        from hybrid.h6_value_train import frozen_state, state_sha256
        frozen = state_sha256(frozen_state(parent))
        (root / 'c5').mkdir()
        from model.checkpoint import save_checkpoint
        save_checkpoint(root / 'c5' / 'best.pt', parent)
        (root / 'c5' / 'best.json').write_text(json.dumps({'frozen_state_sha256': frozen}))
        desc = {'games': 4, 'rows': int(cycle['train']['value'].shape[0]), 'games_sha256': 'g' * 64}
        r2c = {'frozen_state_sha256': frozen, 'models': {'c5': {'value_checkpoint_sha256': 'v' * 64}}}
        return cycle, replay, parent, desc, r2c

    def contracts(self, desc):
        c1 = json.loads((ROOT / V2).read_text())
        c5k = contract()
        for c in (c1, c5k):
            c['widen']['exact'].update(replay_val_rows=8, train_mode_rows=12)
            c['widen']['device_check'].update(batch_size=16)
        c5k['subset']['expected'] = desc
        return c1, c5k

    def common(self, module_cli, cycle, replay, parent, desc, r2c, config=None):
        return [mock.patch.object(module_cli, 'git_state', return_value={'commit': 'c' * 40, 'dirty': False}),
                mock.patch.object(module_cli, 'reference_inputs', return_value=(r2c, {}, config or {})),
                mock.patch.object(module_cli, 'checked_caches', return_value=(cycle, replay)),
                mock.patch.object(module_cli, 'load_c5', side_effect=lambda r, p, dv: (
                    copy.deepcopy(parent), {'value_checkpoint_sha256': 'c5'}))]

    def test_prepare_equals_the_1k_prepare_and_checks_the_parent(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            ra, rb = Path(a), Path(b)
            cycle, replay, parent, desc, r2c = self.inputs(ra)
            self.inputs(rb)
            c1, c5k = self.contracts(desc)
            args = lambda root: SimpleNamespace(d1_contract='x', root=root, cycle_cache='cc', cycle_rows='cr',  # noqa: E731
                                                replay_cache='rc', c5=str(root / 'c5' / 'best.pt'), device='cpu')
            with stack([*self.common(cli1, cycle, replay, parent, desc, r2c),
                        mock.patch.object(cli1, 'load_d1', return_value=(c1, 's' * 64, 'e' * 64)),
                        mock.patch.object(cli1, 'subset_of', return_value=(cycle['train'], {**desc, 'nested': desc}))]):
                one = cli1.prepare(args(ra), log=QUIET)
            prep1 = {'c5': {'value_checkpoint_sha256': 'c5'}, 'subset': {'nested': desc},
                     'widened': {s: {'state_sha256': w['state_sha256']} for s, w in one['widened'].items()}}

            def run(root, prep):
                with stack([*self.common(cli, cycle, replay, parent, desc, r2c),
                            mock.patch.object(k5, 'load_d1_5k', return_value=(c5k, 't' * 64, 'f' * 64)),
                            mock.patch.object(k5, 'predecessor', return_value=(c1, prep)),
                            mock.patch.object(k5, 'subset_5k', return_value=(cycle['train'], desc))]):
                    try:
                        return k5.prepare(args(root), log=QUIET), None
                    except ValueError as exc:
                        return json.loads((root / c5k['output']['prepare']).read_text()), exc
            five, err = run(rb, prep1)
            self.assertIsNone(err)
            self.assertTrue(five['passed'] and five['subset_matches_1k_nested'])
            self.assertEqual((five['floors'], five['composition']), (one['floors'], one['composition']))
            for s in cli.SEEDS:
                w1, w5 = one['widened'][s], five['widened'][s]
                for key in ('checks', 'state_sha256', 'frozen_state_sha256', 'exact', 'device', 'layers', 'reload',
                            'checkpoint_sha256'):
                    self.assertEqual(w5[key], w1[key], (s, key))
                self.assertEqual(w5['gates'], {**w1['gates'], 'same_parent_as_1k': True})
            with TemporaryDirectory() as c_:
                rc = Path(c_)
                self.inputs(rc)
                other = copy.deepcopy(prep1)
                other['widened']['31002']['state_sha256'] = '0' * 64
                rec, err = run(rc, other)
            self.assertRegex(str(err), 'no training')
            self.assertFalse(rec['passed'])
            self.assertFalse(rec['widened']['31002']['gates']['same_parent_as_1k'])
            self.assertTrue(rec['widened']['31001']['passed'])

    def test_train_equals_the_1k_train_then_fit(self):
        config = {'seed': 1, 'batch_size': 16, 'epochs': 1.0, 'learning_rate': 1e-3, 'weight_decay': 1e-4,
                  'min_lr_ratio': 0.05, 'augment_d4': True, 'evaluations': 2, 'h6_guard_ratio': 100.0}
        from model.checkpoint import save_checkpoint
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            ra, rb = Path(a), Path(b)
            cycle, replay, parent, desc, r2c = self.inputs(ra)
            self.inputs(rb)
            c1, c5k = self.contracts(desc)
            for c in (c1, c5k):
                c['budget'] = {**c['budget'], 'total_steps': 4}
            preps = {}
            for root, c, mod in ((ra, c1, cli1), (rb, c5k, k5)):
                widened = {}
                for s in cli.SEEDS:
                    wide = d1.widen_value_head(parent, 4, c['widen']['init_seeds'][s])
                    ckpt, meta = mod.parent_paths(c, root, s)
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    save_checkpoint(ckpt, wide)
                    meta.write_text(json.dumps({'model_config': asdict(wide.config),
                                                'frozen_state_sha256': r2c['frozen_state_sha256']}))
                    widened[s] = {'checkpoint_sha256': r2.file_sha256(ckpt), 'metadata_sha256': r2.file_sha256(meta)}
                preps[mod] = {'subset': desc if mod is k5 else {**desc, 'nested': {}}, 'widened': widened,
                              'floors': {'d4': {'floor': 0.0}}}
            args = lambda root, arm, s: SimpleNamespace(d1_contract='x', root=root, arm=arm, seed=s,  # noqa: E731
                                                        cycle_cache='cc', cycle_rows='cr', replay_cache='rc',
                                                        c5='c5.pt', device='cpu')
            with stack([*self.common(cli1, cycle, replay, parent, desc, r2c, config),
                        mock.patch.object(cli1, 'load_d1', return_value=(c1, 's' * 64, 'e' * 64)),
                        mock.patch.object(cli1, 'checked_prepare', return_value=preps[cli1]),
                        mock.patch.object(cli1, 'subset_of', return_value=(cycle['train'], {**desc, 'nested': {}}))]):
                ones = {(arm, s): cli1.train(args(ra, arm, s), log=QUIET) for arm in cli.ARMS for s in cli.SEEDS}
            with stack([*self.common(cli, cycle, replay, parent, desc, r2c, config),
                        mock.patch.object(k5, 'load_d1_5k', return_value=(c5k, 't' * 64, 'f' * 64)),
                        mock.patch.object(k5, 'checked_prepare', return_value=preps[k5]),
                        mock.patch.object(k5, 'subset_5k', return_value=(cycle['train'], desc))]):
                fives = {(arm, s): k5.train(args(rb, arm, s), log=QUIET) for arm in cli.ARMS for s in cli.SEEDS}
                with self.assertRaisesRegex(ValueError, 'never overwritten'):
                    k5.train(args(rb, 'wide', '31001'), log=QUIET)
                out = k5.fit(SimpleNamespace(d1_contract='x', root=rb), log=QUIET)
            for key, one in ones.items():
                five = fives[key]
                for k in ('arm', 'seed', 'value_channels', 'assignment', 'total_steps', 'evaluation_steps', 'fits',
                          'verdict'):
                    self.assertEqual(five[k], one[k], (key, k))
                self.assertEqual(five['last_step']['fit'], one['last_step']['fit'])
                self.assertEqual(five['last_step']['state']['checkpoint_sha256'],
                                 one['last_step']['state']['checkpoint_sha256'])
                self.assertEqual({k: five['selected'][k] for k in ('step', 'objective', 'fit')},
                                 {k: one['selected'][k] for k in ('step', 'objective', 'fit')})
                self.assertEqual(five['experiment_sha256'], 'f' * 64)
            self.assertIn(out['reading']['category'], d1.FIT_CATEGORIES)
            self.assertEqual(out['experiment_sha256'], 'f' * 64)


class EvaluateTest(unittest.TestCase):
    """The 5k evaluate against the A1 tool on the same stand-in models, predictions and H-dev."""

    def stage(self, tmp):
        st = te.Stage(Path(tmp))
        c = contract()
        c['output'] = {**c['output'], 'run_dirs': st.c['output']['run_dirs']}  # the stand-in run folders
        st.c5k = c
        return st

    def run5k(self, st, resume=False):
        patches = [mock.patch.object(k5, 'load_d1_5k', return_value=(st.c5k, 't' * 64, 'f' * 64)),
                   mock.patch.object(k5, 'checked_prepare', return_value={})]
        with st.patched(cli), stack(patches):
            return k5.evaluate(SimpleNamespace(d1_contract='x', root=st.root, cache='hdev', c5='c5.pt',
                                               device='cpu', resume=resume), log=QUIET)

    def out(self, st, name):
        return st.root / {'result': st.c5k['output']['result'], 'predictions': st.c5k['output']['predictions'],
                          'record': st.c5k['evaluate']['predicted_record']}[name]

    def test_the_reading_equals_the_a1_tool_and_an_interruption_resumes(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            ref = te.Stage(Path(a)).run()
            st = self.stage(b)
            with mock.patch.object(k5, 'publish_result', side_effect=OSError('interrupted')):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    self.run5k(st)
            self.assertTrue(self.out(st, 'predictions').is_file() and self.out(st, 'record').is_file())
            with self.assertRaisesRegex(ValueError, 'complete them with --resume'):
                self.run5k(st)
            calls = st.predict_calls
            out = self.run5k(st, resume=True)
            self.assertEqual(st.predict_calls, calls)
            self.assertIsNotNone(out['resumed'])
            self.assertEqual({k: out[k] for k in te.READING_KEYS}, {k: ref[k] for k in te.READING_KEYS})
            self.assertEqual(out['experiment_sha256'], 'f' * 64)
            self.assertIn('5,000-row', out['scope'])
            written = json.loads(self.out(st, 'result').read_text())
            self.assertEqual(written['predictions']['sha256'], r2.file_sha256(self.out(st, 'predictions')))
            with self.assertRaisesRegex(ValueError, 'nothing to resume'):
                self.run5k(st, resume=True)

    def test_straight_run_and_refusals(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            ref = te.Stage(Path(a)).run()
            st = self.stage(b)
            out = self.run5k(st)
            self.assertIsNone(out['resumed'])
            self.assertEqual({k: out[k] for k in te.READING_KEYS}, {k: ref[k] for k in te.READING_KEYS})
            with self.assertRaisesRegex(ValueError, 'never overwritten'):
                self.run5k(st)
        with TemporaryDirectory() as tmp:
            st = self.stage(tmp)
            partial = self.out(st, 'predictions').with_name('predictions.pt.partial')
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b'unfinished')
            with self.assertRaisesRegex(ValueError, 'without its record'):
                self.run5k(st)
            with self.assertRaisesRegex(ValueError, 'no predictions record'):
                self.run5k(st, resume=True)
            self.assertEqual(partial.read_bytes(), b'unfinished')
        with TemporaryDirectory() as tmp:
            st = self.stage(tmp)
            with mock.patch.object(k5, 'publish_result', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    self.run5k(st)
            preds = self.out(st, 'predictions')
            stored = torch.load(preds, weights_only=True)
            stored['models']['wide31003']['d4'][1] += 0.25
            preds.unlink()
            torch.save(stored, preds)
            with self.assertRaisesRegex(ValueError, 'not the recorded one'):
                self.run5k(st, resume=True)


class OneThousandRegressionTest(unittest.TestCase):
    def test_the_1k_contract_amendment_and_records_still_load(self):
        for stage in cli1.STAGES:
            c, _, eid = cli1.load_d1(ROOT / V2, ROOT, stage)
        self.assertEqual(eid[:12], 'fc309887e4d7')
        a, _ = ev.load_amendment(ROOT / ev.AMENDMENT, ROOT, ROOT / V2, eid)
        self.assertTrue(a['approved'])
        self.assertTrue(cli1.checked_prepare(c, ROOT, eid)['passed'])
        five = contract()
        self.assertEqual(five['predecessor']['experiment_sha256'], eid)
        self.assertEqual({p: five['sources'][p] for p in cli1.SOURCES}, c['sources'])  # same pinned 1k files


if __name__ == '__main__':
    unittest.main()
