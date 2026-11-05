"""R2-D3 checkpoint selection option of ``h7c_train.train_value``: the omitted option is the pre-change trainer bit for
bit (unmasked recipe and H7C-NMASK-1), ``cycle_selection='original'`` trains the masked trajectory but selects on the
original cycle val WMSE (the masked one a diagnostic), with the same guard, ties and NO_CANDIDATE handling."""
import contextlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

from hybrid import h7c_train as tr
from tests.test_h7c_train import small_model, value_split

QUIET = lambda *a, **k: None  # noqa: E731
ROOT = Path(__file__).resolve().parents[1]
BEFORE = '3066dbae9f13b928027afa69d5a6a8edeae3e25a'  # the last commit before the option (trainer 529cffb8...)
BEFORE_SHA256 = '529cffb875fb7bf4268badccbaf69506be27770b51360523fa9a851de7b5485b'


def caches():
    cycle_train, _ = value_split(1, 4)
    replay_train, _ = value_split(2, 10)
    cycle_val, _ = value_split(3, 2)
    replay_val, _ = value_split(4, 2)
    return {'train': cycle_train, 'val': cycle_val}, {'train': replay_train, 'val': replay_val}


def half_keep(cycle):
    return {s: torch.arange(cycle[s]['value'].shape[0]) % 2 == 0 for s in ('train', 'val')}


def pre_change_trainer():
    """``hybrid.h7c_train`` as it was at BEFORE, loaded beside the current one (same package, its own module)."""
    import hashlib
    try:
        source = subprocess.run(['git', 'show', f'{BEFORE}:src/hybrid/h7c_train.py'], cwd=ROOT, capture_output=True,
                                check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    assert hashlib.sha256(source).hexdigest() == BEFORE_SHA256
    tmp = TemporaryDirectory()
    path = Path(tmp.name) / 'h7c_train_before.py'
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location('hybrid.h7c_train_before', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._tmp = tmp  # keep the file for the module's lifetime
    return module


def same(a, b, path=''):
    """Exact equality of records and state dicts (tensors bit for bit)."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return [] if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype \
            and a.shape == b.shape and torch.equal(a, b) else [path]
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            return [f'{path}: keys {list(a)} != {list(b)}']
        return [d for k in a for d in same(a[k], b[k], f'{path}/{k}')]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f'{path}: length']
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f'{path}[{i}]')]
    return [] if type(a) is type(b) and a == b else [f'{path}: {a!r} != {b!r}']


CONFIGS = [dict(batch_size=16, epochs=1.0, evaluations=3, learning_rate=1e-3, h6_guard_ratio=100.0, seed=7),
           dict(batch_size=8, epochs=2.0, evaluations=4, learning_rate=3e-3, h6_guard_ratio=1.0, seed=11),
           dict(batch_size=4, epochs=1.0, evaluations=2, learning_rate=1e-3, h6_guard_ratio=0.0, seed=5)]


class OmittedOptionTest(unittest.TestCase):
    """Without the option the plain recipe and NMASK are the pre-change trainer: records and states bit for bit."""

    @classmethod
    def setUpClass(cls):
        cls.before = pre_change_trainer()

    def setUp(self):
        if self.before is None:
            self.skipTest('the pre-change trainer is not in this clone')

    def test_plain_and_nmask_equal_the_pre_change_trainer(self):
        verdicts = set()
        for i, kw in enumerate(CONFIGS):
            for name in ('plain', 'nmask'):
                with self.subTest(config=i, path=name):
                    cycle, replay = caches()
                    extra = {} if name == 'plain' else {'cycle_keep': half_keep(cycle)}
                    new = tr.train_value(tr.ValueConfig(**kw), small_model(), cycle, replay, log=QUIET, **extra)
                    old = self.before.train_value(self.before.ValueConfig(**kw), small_model(), cycle, replay,
                                                  log=QUIET, **extra)
                    self.assertEqual(same(new['record'], old['record']), [])
                    self.assertEqual(same(new['state'], old['state']), [])
                    self.assertEqual(list(new), list(old))
                    if name == 'nmask':
                        self.assertEqual(same(new['val_sets'], old['val_sets']), [])
                    self.assertNotIn('selection', new['record'])
                    verdicts.add(new['record']['verdict'])
        self.assertEqual(verdicts, {'CANDIDATE', 'NO_CANDIDATE'})  # both outcomes of the guard are covered


class OriginalSelectionTest(unittest.TestCase):
    def test_trajectory_is_the_masked_one_and_selection_reads_the_original_metric(self):
        cycle, replay = caches()
        keep = half_keep(cycle)
        cfg = tr.ValueConfig(**CONFIGS[0])
        nmask = tr.train_value(cfg, small_model(), cycle, replay, log=QUIET, cycle_keep=keep)
        d3 = tr.train_value(cfg, small_model(), cycle, replay, log=QUIET, cycle_keep=keep, cycle_selection='original')
        a, b = nmask['record'], d3['record']
        self.assertEqual(same(a['mask'], b['mask']), [])  # same batches, same kept weights
        for x, y in zip(a['evaluations'], b['evaluations']):
            self.assertEqual(x['step'], y['step'])
            self.assertEqual(x['cycle_wmse_original'], y['cycle_wmse'])  # the same model at every evaluation
            self.assertEqual(x['cycle_wmse'], y['cycle_wmse_masked'])
            self.assertEqual(x['replay_wmse'], y['replay_wmse'])
            self.assertEqual(x['eligible'], y['eligible'])
            self.assertEqual(y['objective'], (y['cycle_wmse'] + y['replay_wmse']) / 2)
        self.assertEqual(a['parent_val']['cycle_wmse_original'], b['parent_val']['cycle_wmse'])
        self.assertEqual(a['parent_val']['cycle_wmse'], b['parent_val']['cycle_wmse_masked'])
        self.assertEqual(b['selection']['mode'], 'original')
        self.assertNotIn('cycle_wmse_original', b['evaluations'][0])
        best = min((e for e in b['evaluations'] if e['eligible']), key=lambda e: e['objective'])
        self.assertEqual(b['best']['step'], best['step'])

    def test_an_all_true_mask_with_original_selection_is_the_plain_recipe(self):
        cycle, replay = caches()
        cfg = tr.ValueConfig(**CONFIGS[1])
        plain = tr.train_value(cfg, small_model(), cycle, replay, log=QUIET)
        keep = {s: torch.ones(cycle[s]['value'].shape[0], dtype=torch.bool) for s in ('train', 'val')}
        d3 = tr.train_value(cfg, small_model(), cycle, replay, log=QUIET, cycle_keep=keep, cycle_selection='original')
        for x, y in zip(plain['record']['evaluations'], d3['record']['evaluations']):
            self.assertEqual((x['step'], x['objective'], x['eligible']), (y['step'], y['objective'], y['eligible']))
        self.assertEqual(plain['record']['verdict'], d3['record']['verdict'])
        self.assertEqual(same(plain['state'], d3['state']), [])

    def test_zero_cycle_weight_batches_keep_the_replay_term(self):
        cycle, replay = caches()
        n = cycle['train']['value'].shape[0]
        keep = {'train': torch.arange(n) % 2 == 0, 'val': torch.ones(cycle['val']['value'].shape[0], dtype=bool)}
        cfg = tr.ValueConfig(batch_size=4, epochs=1.0, evaluations=2, learning_rate=1e-3, h6_guard_ratio=100.0)
        out = tr.train_value(cfg, small_model(), cycle, replay, log=QUIET, cycle_keep=keep, cycle_selection='original')
        mask = out['record']['mask']
        self.assertGreater(mask['zero_cycle_weight_batches'] + mask['batches_without_cycle_rows'], 0)
        nmask = tr.train_value(cfg, small_model(), cycle, replay, log=QUIET, cycle_keep=keep)
        self.assertEqual(same(mask, nmask['record']['mask']), [])

    def test_refusals(self):
        cycle, replay = caches()
        cfg = tr.ValueConfig(**CONFIGS[0])
        with self.assertRaisesRegex(ValueError, 'cycle_keep'):
            tr.train_value(cfg, small_model(), cycle, replay, log=QUIET, cycle_selection='original')
        with self.assertRaisesRegex(ValueError, 'only'):
            tr.train_value(cfg, small_model(), cycle, replay, log=QUIET, cycle_keep=half_keep(cycle),
                           cycle_selection='masked')


class ScriptedMetricTest(unittest.TestCase):
    """Validation metrics replaced by scripted values (training itself is real): the two metrics prefer different
    checkpoints; the guard, ties and NO_CANDIDATE follow the trainer's rules."""

    @contextlib.contextmanager
    def scripted(self, original, masked, replay):
        """Per evaluation (parent first): original cycle val WMSE, masked cycle val WMSE, H6 val WMSE."""
        calls = {'evaluate': 0, 'masked': 0}

        def evaluate(model, data, device):
            i = calls['evaluate']
            calls['evaluate'] += 1
            value = (original if i % 2 == 0 else replay)[i // 2]
            return {'weighted_mse': value, 'all': {'scripted': value}}

        def masked_metric(model, data, device, batch_size=4096):
            i = calls['masked']
            calls['masked'] += 1
            return {'rows': 0, 'kept_rows': 0, 'weight': 1.0, 'kept_weight': 1.0, 'masked_wmse': masked[i]}
        with mock.patch('hybrid.h6_value_train.evaluate', side_effect=evaluate), \
                mock.patch.object(tr, 'masked_metric', side_effect=masked_metric):
            yield calls

    def run_both(self, original, masked, replay, guard=1.05):
        cycle, replay_sets = caches()
        keep = half_keep(cycle)
        cfg = tr.ValueConfig(batch_size=16, epochs=1.0, evaluations=3, learning_rate=1e-3, h6_guard_ratio=guard)
        out = {}
        for name, extra in (('nmask', {}), ('d3', {'cycle_selection': 'original'})):
            with self.scripted(original, masked, replay) as calls:
                out[name] = tr.train_value(cfg, small_model(), cycle, replay_sets, log=QUIET, cycle_keep=keep,
                                           **extra)['record']
                self.assertEqual(calls['evaluate'], 2 * 4)  # parent + three evaluations
        return out

    def test_the_original_metric_decides_and_the_masked_one_stays_a_diagnostic(self):
        r = self.run_both(original=[.50, .40, .30, .35], masked=[.50, .20, .35, .38], replay=[.10] * 4)
        steps = [e['step'] for e in r['d3']['evaluations']]
        self.assertEqual(r['nmask']['best']['step'], steps[0])  # masked .20 is the lowest
        self.assertEqual(r['d3']['best']['step'], steps[1])  # original .30 is the lowest
        self.assertEqual([e['cycle_wmse'] for e in r['d3']['evaluations']], [.40, .30, .35])
        self.assertEqual([e['cycle_wmse_masked'] for e in r['d3']['evaluations']], [.20, .35, .38])
        self.assertEqual(r['d3']['parent_val']['cycle_wmse'], .50)
        self.assertEqual(r['d3']['parent_val']['cycle_wmse_masked'], .50)
        self.assertAlmostEqual(r['d3']['best']['objective'], (.30 + .10) / 2)

    def test_guard_tie_and_no_candidate(self):
        # the best original objective (eval 2) fails the H6 guard: eval 3 is selected, never eval 2
        r = self.run_both(original=[.5, .40, .20, .30], masked=[.5] * 4, replay=[.10, .10, .20, .10])
        steps = [e['step'] for e in r['d3']['evaluations']]
        self.assertEqual([e['eligible'] for e in r['d3']['evaluations']], [True, False, True])
        self.assertEqual(r['d3']['best']['step'], steps[2])
        # a tie: the first evaluation with the lowest objective is kept (strict improvement only)
        r = self.run_both(original=[.5, .30, .30, .40], masked=[.5] * 4, replay=[.10] * 4)
        self.assertEqual(r['d3']['best']['step'], r['d3']['evaluations'][0]['step'])
        # every evaluation fails the guard: NO_CANDIDATE, as the trainer does
        r = self.run_both(original=[.5, .1, .1, .1], masked=[.5] * 4, replay=[.10, .20, .20, .20])
        self.assertEqual((r['d3']['verdict'], r['d3']['best']), ('NO_CANDIDATE', None))


if __name__ == '__main__':
    unittest.main()
