"""The D1 full-scale stage (``scripts/h7r_d1_full.py``): its contract pins (the 5,000-row records it links to, the
fixed values it keeps, the whole-split data, the R2-0B reference, the environment and pairing rules), the whole
split used without selection, prepare and train equal to the 5,000-row tool on the same stand-in inputs (with the
parent check against the earlier stages and the runtime environment added), the comparability stop with its
reasons recorded, the R2-0B reference beside the reading, the H-dev evaluation equal to the 5,000-row tool with
interruption, resume, tampering and a HEAD change, and the earlier contracts and records still loading."""
import contextlib
import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from hybrid import h7r_d1 as d1
from hybrid import h7r_r2 as r2
from tests.test_h7c_train import small_model, value_split
from tests.test_h7r_d1 import caches
import h7r_d1 as cli  # the module the 5k and full tools import
import h7r_d1_5k as k5
import h7r_d1_full as kf
from tests import test_h7r_d1_evaluate as te

ROOT = Path(__file__).resolve().parents[1]
D = 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1'
CONTRACT = f'{D}/d1-full-contract.json'
QUIET = lambda *a, **k: None  # noqa: E731


def contract():
    return json.loads((ROOT / CONTRACT).read_text())


@contextlib.contextmanager
def stack(patches):
    with contextlib.ExitStack() as s:
        for p in patches:
            s.enter_context(p)
        yield


class LoaderTest(unittest.TestCase):
    def load(self, c, stage='prepare'):
        with TemporaryDirectory(dir=ROOT / 'docs') as tmp:
            path = Path(tmp) / 'd1-full.json'
            path.write_text(json.dumps(c))
            with mock.patch('hybrid.h7c_nmask.committed', return_value=True):
                return kf.load_d1_full(path, ROOT, stage)

    def approved(self):
        c = contract()
        c['approved'] = {s: True for s in kf.STAGES}
        return c

    def test_the_committed_contract(self):
        c = contract()
        self.assertEqual((c['status'], c['stage'], c['format']), ('fixed', 'full', kf.FORMAT))
        flags = [c['approved'][s] for s in kf.STAGES]
        self.assertEqual(flags, sorted(flags, reverse=True))
        for stage, flag in zip(kf.STAGES, flags):
            if not flag:
                with self.subTest(stage), self.assertRaisesRegex(ValueError, f'{stage} stage is not approved'):
                    kf.load_d1_full(ROOT / CONTRACT, ROOT, stage)
        draft = json.loads((ROOT / f'{D}/d1-full-contract.draft.json').read_text())
        reviewed = {'status', 'approved', 'sources', 'note', 'pairing', 'environment', 'next'}
        kept = [k for k in draft if k not in reviewed]
        self.assertEqual({k: c[k] for k in kept}, {k: draft[k] for k in kept})
        self.assertEqual(c['pairing']['per_seed'], draft['pairing']['per_seed'])
        self.assertEqual(tuple(c['environment']['fields']), kf.ENV_FIELDS)
        self.assertEqual(set(kf.ENV_FIELDS), set(d1.numeric_settings('cpu')))  # every recorded field is checked
        self.assertIn('NOT_ESTABLISHED', c['next']['d4'])

    def test_an_approved_copy_loads_every_stage(self):
        for stage in kf.STAGES:
            _, _, eid = self.load(self.approved(), stage)
        self.assertEqual(eid, cli.experiment_sha256(contract()))
        five = json.loads((ROOT / f'{D}/d1-5k-contract.json').read_text())
        self.assertNotEqual(eid, cli.experiment_sha256(five))

    def test_wrong_pins_are_refused(self):
        cases = {
            'a draft is never run': lambda c: c.update(status='draft'),
            'every file pinned': lambda c: c['sources'].update({kf.TOOL_PATH: None}),
            'checkout differs': lambda c: c['sources'].update({kf.TOOL_PATH: '0' * 64}),
            'earlier tools are not': lambda c: c['sources'].update({'scripts/h7r_d1_5k.py': '1' * 64}),
            '35,816 rows': lambda c: c['data'].update(rows=35000),
            'c8 config, 210 steps': lambda c: c['budget'].update(total_steps=420),
            'not the 5,000-row arms': lambda c: c['arms']['wide'].update(value_channels=8),
            'not the 5,000-row fit': lambda c: c['fit'].update(margin=0.02),
            'not the earlier rules': lambda c: c['ruling'].update(hdev='x'),
            'construction, init seeds': lambda c: c['widen']['exact'].update(tolerance=1e-9),
            'not the 5,000-row experiment': lambda c: c['predecessor'].update(experiment_sha256='0' * 64),
            'readings differ': lambda c: c['predecessor']['readings'].update(hdev='HOLD'),
            'both arms trained anew': lambda c: c['control'].update(kind='reused'),
            'R2-0B reference runs': lambda c: c['control']['r2_0b_reference'].pop('31003'),
            'environment: the fields': lambda c: c['environment']['fields'].pop(),
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
        c['approved']['prepare'] = False
        with self.assertRaisesRegex(ValueError, 'train stage needs'):
            self.load(c, 'train')


class DataTest(unittest.TestCase):
    def test_the_whole_split_is_used_and_checked(self):
        with TemporaryDirectory() as tmp:
            train, _ = value_split(13, 9)
            ids = [f'game-{g:03d}' for g in range(9)]
            (Path(tmp) / 'train_games.json').write_text(json.dumps(ids))
            n = int(train['value'].shape[0])
            train['kind'][::5] = 1
            known = {'outcome_rows': int((train['kind'] == 0).sum()), 'proof_rows': int((train['kind'] == 1).sum())}
            c = {'data': {'rows': n, 'composition_known': known}}
            data, desc = kf.full_data(c, {'train': train}, tmp)
            self.assertIs(data, train)  # no selection
            self.assertEqual((desc['games'], desc['rows']), (9, n))
            from hybrid.h7r_d4 import digest
            self.assertEqual(desc['games_sha256'], digest(sorted(ids)))
            with self.assertRaisesRegex(ValueError, 'rows, the contract'):
                kf.full_data({'data': {'rows': n + 1, 'composition_known': known}}, {'train': train}, tmp)
            with self.assertRaisesRegex(ValueError, 'composition'):
                kf.full_data({'data': {'rows': n, 'composition_known': {**known, 'proof_rows': 0}}},
                             {'train': train}, tmp)
        c = contract()
        ok = {'inputs': {'cycle_cache_splits': {'train': c['data']['split_sha256']},
                         'cycle_labels_sha256': c['data']['labels_sha256']}}
        kf.check_reference_inputs(c, ok)
        bad = copy.deepcopy(ok)
        bad['inputs']['cycle_cache_splits']['train'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'not the split and labels'):
            kf.check_reference_inputs(c, bad)


class StandIns:
    def __init__(self, root):
        from hybrid.h6_value_train import frozen_state, state_sha256
        from model.checkpoint import save_checkpoint
        self.cycle, self.replay = caches()
        self.parent = small_model()
        frozen = state_sha256(frozen_state(self.parent))
        (root / 'c5').mkdir()
        save_checkpoint(root / 'c5' / 'best.pt', self.parent)
        (root / 'c5' / 'best.json').write_text(json.dumps({'frozen_state_sha256': frozen}))
        self.desc = {'games': 4, 'rows': int(self.cycle['train']['value'].shape[0]), 'games_sha256': 'g' * 64}
        self.r2c = {'frozen_state_sha256': frozen, 'models': {'c5': {'value_checkpoint_sha256': 'v' * 64}}}
        for s in cli.SEEDS:  # the pinned R2-0B reference records, at their paths under this root
            for name in ('record', 'r2_train'):
                rel = contract()['control']['r2_0b_reference'][s][name]['path']
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / rel, root / rel)

    def common(self, config=None):
        return [mock.patch.object(cli, 'git_state', return_value={'commit': 'c' * 40, 'dirty': False}),
                mock.patch.object(cli, 'reference_inputs', return_value=(self.r2c, {}, config or {})),
                mock.patch.object(cli, 'checked_caches', return_value=(self.cycle, self.replay)),
                mock.patch.object(cli, 'load_c5', side_effect=lambda r, p, dv: (
                    copy.deepcopy(self.parent), {'value_checkpoint_sha256': 'c5'}))]


def contracts(desc):
    c5k, cf = json.loads((ROOT / f'{D}/d1-5k-contract.json').read_text()), contract()
    for c in (c5k, cf):
        c['widen']['exact'].update(replay_val_rows=8, train_mode_rows=12)
        c['widen']['device_check'].update(batch_size=16)
    c5k['subset']['expected'] = desc
    cf['data'] = {**cf['data'], 'rows': desc['rows']}
    return c5k, cf


class PrepareTest(unittest.TestCase):
    def test_prepare_equals_the_5k_prepare_and_checks_the_parent(self):
        args = lambda root: SimpleNamespace(d1_contract='x', root=root, cycle_cache='cc', cycle_rows='cr',  # noqa: E731
                                            replay_cache='rc', c5=str(root / 'c5' / 'best.pt'), device='cpu')
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            ra, rb = Path(a), Path(b)
            si = StandIns(ra)
            StandIns(rb)
            c5k, cf = contracts(si.desc)
            prep1 = {'c5': {'value_checkpoint_sha256': 'c5'}, 'subset': {'nested': si.desc},
                     'widened': {s: {'state_sha256': d1.preservation(si.parent, d1.widen_value_head(
                         si.parent, 4, c5k['widen']['init_seeds'][s]))['state_sha256']} for s in cli.SEEDS}}
            with stack([*si.common(), mock.patch.object(k5, 'load_d1_5k', return_value=(c5k, 't' * 64, 'f' * 64)),
                        mock.patch.object(k5, 'predecessor', return_value=({}, prep1)),
                        mock.patch.object(k5, 'subset_5k', return_value=(si.cycle['train'], si.desc))]):
                five = k5.prepare(args(ra), log=QUIET)
            prep5 = {'c5': five['c5'], 'widened': {s: {'state_sha256': w['state_sha256'],
                                                       'parent_state_sha256_1k': w['parent_state_sha256_1k']}
                                                   for s, w in five['widened'].items()}}

            def run(root, prep):
                with stack([*si.common(),
                            mock.patch.object(kf, 'load_d1_full', return_value=(cf, 'u' * 64, 'h' * 64)),
                            mock.patch.object(kf, 'predecessor_prepare', return_value=prep),
                            mock.patch.object(kf, 'check_reference_inputs', return_value=None),
                            mock.patch.object(kf, 'full_data', return_value=(si.cycle['train'], si.desc))]):
                    try:
                        return kf.prepare(args(root), log=QUIET), None
                    except ValueError as exc:
                        return json.loads((root / cf['output']['prepare']).read_text()), exc
            full, err = run(rb, prep5)
            self.assertIsNone(err)
            self.assertTrue(full['passed'])
            self.assertEqual((full['floors'], full['composition'], full['data']),
                             (five['floors'], five['composition'], five['subset']))
            for s in cli.SEEDS:
                w5, wf = five['widened'][s], full['widened'][s]
                for key in ('checks', 'state_sha256', 'exact', 'device', 'layers', 'reload', 'checkpoint_sha256'):
                    self.assertEqual(wf[key], w5[key], (s, key))
                self.assertEqual({k: v for k, v in wf['gates'].items() if k != 'same_parent_as_earlier'},
                                 {k: v for k, v in w5['gates'].items() if k != 'same_parent_as_1k'})
                self.assertTrue(wf['gates']['same_parent_as_earlier'])
            self.assertTrue(all(isinstance(v, float) for v in full['timings_seconds'].values()))
            with TemporaryDirectory() as c_:
                rc = Path(c_)
                StandIns(rc)
                other = copy.deepcopy(prep5)
                other['widened']['31003']['state_sha256'] = '0' * 64
                rec, err = run(rc, other)
            self.assertRegex(str(err), 'no training')
            self.assertFalse(rec['widened']['31003']['gates']['same_parent_as_earlier'])
            self.assertTrue(rec['widened']['31001']['passed'])


class TrainFitTest(unittest.TestCase):
    CONFIG = {'seed': 1, 'batch_size': 16, 'epochs': 1.0, 'learning_rate': 1e-3, 'weight_decay': 1e-4,
              'min_lr_ratio': 0.05, 'augment_d4': True, 'evaluations': 2, 'h6_guard_ratio': 100.0}

    def setup_runs(self, root, si, c, mod):
        from model.checkpoint import save_checkpoint
        widened = {}
        for s in cli.SEEDS:
            wide = d1.widen_value_head(si.parent, 4, c['widen']['init_seeds'][s])
            ckpt, meta = k5.parent_paths(c, root, s)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(ckpt, wide)
            meta.write_text(json.dumps({'model_config': asdict(wide.config),
                                        'frozen_state_sha256': si.r2c['frozen_state_sha256']}))
            widened[s] = {'checkpoint_sha256': r2.file_sha256(ckpt), 'metadata_sha256': r2.file_sha256(meta)}
        prep = {'widened': widened, 'floors': {'d4': {'floor': 0.0}},
                'device_check': {'settings': d1.numeric_settings('cpu')}}
        prep['subset' if mod is k5 else 'data'] = si.desc
        return prep

    def args(self, root, arm, s):
        return SimpleNamespace(d1_contract='x', root=root, arm=arm, seed=s, cycle_cache='cc', cycle_rows='cr',
                               replay_cache='rc', c5='c5.pt', device='cpu')

    def run_full(self, root, si, cf, prep):
        with stack([*si.common(self.CONFIG),
                    mock.patch.object(kf, 'load_d1_full', return_value=(cf, 'u' * 64, 'h' * 64)),
                    mock.patch.object(kf, 'checked_prepare', return_value=prep),
                    mock.patch.object(kf, 'check_reference_inputs', return_value=None),
                    mock.patch.object(kf, 'full_data', return_value=(si.cycle['train'], si.desc))]):
            return {(arm, s): kf.train(self.args(root, arm, s), log=QUIET) for arm in cli.ARMS for s in cli.SEEDS}

    def fit(self, root, cf, prep):
        with stack([mock.patch.object(kf, 'load_d1_full', return_value=(cf, 'u' * 64, 'h' * 64)),
                    mock.patch.object(kf, 'checked_prepare', return_value=prep)]):
            return kf.fit(SimpleNamespace(d1_contract='x', root=root), log=QUIET)

    def test_train_equals_the_5k_train_records_the_environment_then_fit(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            ra, rb = Path(a), Path(b)
            si = StandIns(ra)
            StandIns(rb)
            c5k, cf = contracts(si.desc)
            for c in (c5k, cf):
                c['budget'] = {**c['budget'], 'total_steps': 4}
            p5, pf = self.setup_runs(ra, si, c5k, k5), self.setup_runs(rb, si, cf, kf)
            with stack([*si.common(self.CONFIG),
                        mock.patch.object(k5, 'load_d1_5k', return_value=(c5k, 't' * 64, 'f' * 64)),
                        mock.patch.object(k5, 'checked_prepare', return_value=p5),
                        mock.patch.object(k5, 'subset_5k', return_value=(si.cycle['train'], si.desc))]):
                fives = {(arm, s): k5.train(self.args(ra, arm, s), log=QUIET) for arm in cli.ARMS for s in cli.SEEDS}
            fulls = self.run_full(rb, si, cf, pf)
            for key, five in fives.items():
                full = fulls[key]
                for k in ('arm', 'seed', 'value_channels', 'assignment', 'total_steps', 'evaluation_steps', 'fits',
                          'verdict', 'subset'):
                    self.assertEqual(full[k], five[k], (key, k))
                self.assertEqual(full['last_step']['state']['checkpoint_sha256'],
                                 five['last_step']['state']['checkpoint_sha256'])
                self.assertEqual({k: full['selected'][k] for k in ('step', 'objective', 'fit')},
                                 {k: five['selected'][k] for k in ('step', 'objective', 'fit')})
                self.assertEqual(full['environment'], d1.numeric_settings('cpu'))
            out = self.fit(rb, cf, pf)
            self.assertTrue(out['comparable'])
            self.assertEqual(out['comparability_reasons'], [])
            self.assertIn(out['reading']['category'], d1.FIT_CATEGORIES)
            ref = out['r2_0b_reference']
            self.assertEqual(set(ref), set(cli.SEEDS))
            old = json.loads((ROOT / cf['control']['r2_0b_reference']['31002']['record']['path']).read_text())
            self.assertEqual(ref['31002']['r2_0b']['selected_step'], old['best']['step'])
            self.assertEqual(ref['31002']['narrow']['selected_step'], fulls[('narrow', '31002')]['selected']['step'])

    def test_an_environment_or_pairing_difference_stops_the_reading(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            si = StandIns(root)
            _, cf = contracts(si.desc)
            cf['budget'] = {**cf['budget'], 'total_steps': 4}
            prep = self.setup_runs(root, si, cf, kf)
            self.run_full(root, si, cf, prep)
            path = root / cf['output']['run_dirs']['wide']['31002'] / 'd1_train.json'
            rec = json.loads(path.read_text())
            rec['environment']['threads'] += 1
            path.write_text(json.dumps(rec))
            with self.assertRaisesRegex(ValueError, 'not comparable'):
                self.fit(root, cf, prep)
            written = json.loads((root / cf['output']['fit']).read_text())
            self.assertFalse(written['comparable'])
            self.assertIsNone(written['reading'])
            self.assertTrue(any('wide 31002' in r and 'threads' in r for r in written['comparability_reasons']))
            with mock.patch('hybrid.h7c_nmask.committed', return_value=True), \
                    self.assertRaisesRegex(ValueError, 'not comparable: no H-dev reading'):
                kf.checked_fit(cf, root, 'h' * 64, prep, kf.load_runs(cf, root, 'h' * 64))
        recs = {(arm, s): {'environment': prep['device_check']['settings'], 'subset': si.desc,
                           'total_steps': 4, 'assignment': 'a', 'evaluation_steps': [2, 4],
                           'git': {'commit': 'c' * 40, 'dirty': False}} for arm in cli.ARMS for s in cli.SEEDS}
        self.assertEqual(kf.comparability(cf, prep, recs), [])
        cases = {
            'the arms differ in evaluation_steps': lambda r: r[('wide', '31001')].update(evaluation_steps=[3, 4]),
            'the arms differ in assignment': lambda r: r[('narrow', '31003')].update(assignment='b'),
            'one clean commit': lambda r: r[('wide', '31003')].update(git={'commit': 'd' * 40, 'dirty': False}),
            'no runtime environment': lambda r: r[('narrow', '31001')].pop('environment'),
            'data is not the prepared one': lambda r: r[('narrow', '31002')].update(subset={'rows': 1}),
        }
        for message, change in cases.items():
            with self.subTest(message):
                r = copy.deepcopy(recs)
                change(r)
                self.assertTrue(any(message in x for x in kf.comparability(cf, prep, r)), kf.comparability(cf, prep, r))


class EvaluateTest(unittest.TestCase):
    """The full evaluate against the 5k evaluate on the same stand-in models, predictions and H-dev."""

    class Stage(te.Stage):
        def pinned(self, root, ref, what):
            if what == 'R2 contract':
                return self.r2c
            return {'hdev': self.hdev, 'seeds': {s: {'primary': 0.4} for s in cli.SEEDS}}

    def stage(self, tmp, contract_of):
        st = self.Stage(Path(tmp))
        c = contract_of()
        c['output'] = {**c['output'], 'run_dirs': st.c['output']['run_dirs']}
        st.cx = c
        return st

    def run_full(self, st, resume=False, git_states=None):
        patches = [mock.patch.object(kf, 'load_d1_full', return_value=(st.cx, 'u' * 64, 'h' * 64)),
                   mock.patch.object(kf, 'checked_prepare', return_value={}),
                   mock.patch.object(kf, 'load_runs', return_value=st.recs),
                   mock.patch.object(kf, 'checked_fit', return_value=st.fit)]
        if git_states is not None:
            patches.append(mock.patch.object(cli, 'git_state', side_effect=list(git_states)))
        with st.patched(cli), stack(patches):
            return kf.evaluate(SimpleNamespace(d1_contract='x', root=st.root, cache='hdev', c5='c5.pt',
                                               device='cpu', resume=resume), log=QUIET)

    def run_5k(self, st):
        patches = [mock.patch.object(k5, 'load_d1_5k', return_value=(st.cx, 't' * 64, 'f' * 64)),
                   mock.patch.object(k5, 'checked_prepare', return_value={})]
        with st.patched(cli), stack(patches):
            return k5.evaluate(SimpleNamespace(d1_contract='x', root=st.root, cache='hdev', c5='c5.pt',
                                               device='cpu', resume=False), log=QUIET)

    def out(self, st, name):
        return st.root / {'result': st.cx['output']['result'], 'predictions': st.cx['output']['predictions'],
                          'record': st.cx['evaluate']['predicted_record']}[name]

    def five_contract(self):
        return json.loads((ROOT / f'{D}/d1-5k-contract.json').read_text())

    def test_the_reading_equals_the_5k_tool_with_the_r2_0b_reference(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            ref = self.run_5k(self.stage(a, self.five_contract))
            st = self.stage(b, contract)
            out = self.run_full(st)
            self.assertEqual({k: out[k] for k in te.READING_KEYS}, {k: ref[k] for k in te.READING_KEYS})
            self.assertIsNone(out['resumed'])
            self.assertIn('full-scale', out['scope'])
            for s in cli.SEEDS:
                self.assertAlmostEqual(out['r2_0b_reference'][s]['difference'], out['paired'][s]['control'] - 0.4)
            with self.assertRaisesRegex(ValueError, 'never overwritten'):
                self.run_full(st)

    def test_interruption_resume_tampering_and_a_head_change(self):
        clean = {'commit': 'c' * 40, 'dirty': False}
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            straight = self.run_full(self.stage(a, contract))
            st = self.stage(b, contract)
            with mock.patch.object(kf, 'publish_result', side_effect=OSError('interrupted')):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    self.run_full(st)
            with self.assertRaisesRegex(ValueError, 'complete them with --resume'):
                self.run_full(st)
            saved = (self.out(st, 'predictions').read_bytes(), self.out(st, 'record').read_bytes())
            with self.assertRaisesRegex(ValueError, 'changed during the evaluation: no result'):
                self.run_full(st, resume=True, git_states=[clean, {'commit': 'd' * 40, 'dirty': False}])
            self.assertFalse(self.out(st, 'result').exists())
            self.assertEqual((self.out(st, 'predictions').read_bytes(), self.out(st, 'record').read_bytes()), saved)
            calls = st.predict_calls
            out = self.run_full(st, resume=True)
            self.assertEqual(st.predict_calls, calls)
            self.assertIsNotNone(out['resumed'])
            self.assertEqual({k: out[k] for k in te.READING_KEYS}, {k: straight[k] for k in te.READING_KEYS})
        with TemporaryDirectory() as tmp:
            st = self.stage(tmp, contract)
            with self.assertRaisesRegex(ValueError, 'changed during the evaluation: no result'):
                self.run_full(st, git_states=[clean, clean, {'commit': 'c' * 40, 'dirty': True}])
            self.assertTrue(self.out(st, 'record').is_file() and not self.out(st, 'result').exists())
            preds = self.out(st, 'predictions')
            stored = torch.load(preds, weights_only=True)
            stored['models']['narrow31001']['raw'][0] += 0.1
            preds.unlink()
            torch.save(stored, preds)
            with self.assertRaisesRegex(ValueError, 'not the recorded one'):
                self.run_full(st, resume=True)
        with TemporaryDirectory() as tmp:
            st = self.stage(tmp, contract)
            partial = self.out(st, 'predictions').with_name('predictions.pt.partial')
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b'unfinished')
            with self.assertRaisesRegex(ValueError, 'without its record'):
                self.run_full(st)
            with self.assertRaisesRegex(ValueError, 'no predictions record'):
                self.run_full(st, resume=True)
            self.assertEqual(partial.read_bytes(), b'unfinished')


class RegressionTest(unittest.TestCase):
    def test_the_earlier_contracts_and_records_still_load(self):
        V2 = ROOT / f'{D}/d1-1k-v2-contract.json'
        for stage in cli.STAGES:
            c1, _, eid1 = cli.load_d1(V2, ROOT, stage)
            c5, _, eid5 = k5.load_d1_5k(ROOT / f'{D}/d1-5k-contract.json', ROOT, stage)
        self.assertEqual((eid1[:12], eid5[:12]), ('fc309887e4d7', '84ed0100d792'))
        self.assertTrue(k5.checked_prepare(c5, ROOT, eid5)['passed'])
        full = contract()
        self.assertEqual(full['predecessor']['experiment_sha256'], eid5)
        self.assertEqual({p: full['sources'][p] for p in k5.SOURCES}, c5['sources'])


if __name__ == '__main__':
    unittest.main()
