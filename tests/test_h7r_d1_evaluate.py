"""The D1 v2 evaluate stage end to end on stand-in models and a stand-in H-dev: the amended tool
(``scripts/h7r_d1_evaluate.py``) reads the same as the contract-pinned ``scripts/h7r_d1.py evaluate``; an
interruption after the predictions leaves a recorded state that ``--resume`` completes without a new inference and
with the same reading; altered predictions, a changed checkpoint, an unrecorded partial file and an unapproved or
mismatched amendment are refused."""
import contextlib
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import h7r_holdout
import h7r_r2 as r2cli_top
from hybrid import h7r_d1 as d1
from hybrid import h7r_r2 as r2
from scripts import h7r_d1 as cli
from scripts import h7r_d1_evaluate as ev
from tests.test_h7c_train import small_model, value_split

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1/d1-1k-v2-contract.json'
QUIET = lambda *a, **k: None  # noqa: E731
REAL_PREDICT = r2.predict
PREDICTIONS = {}
NAMES = ['c5', *(f'{arm}{s}' for arm in cli.ARMS for s in cli.SEEDS)]


def hdev_standin():
    data, _ = value_split(7, 24)
    n = int(data['value'].shape[0])
    data['kind'] = torch.tensor([1 if i % 4 == 0 else 0 for i in range(n)], dtype=data['kind'].dtype)
    data['weight'] = torch.where(data['kind'] == 1, torch.tensor(1.0), torch.tensor(0.1)).to(data['weight'].dtype)
    games = data['game'].tolist()
    present = sorted(set(games))
    dense = {g: i for i, g in enumerate(present)}
    info = {'game': [dense[g] for g in games], 'game_ids': [f'g{g}' for g in present], 'ply': data['ply'].tolist(),
            'result': [r2.RESULTS[g % 3] for g in games],
            'label_kind': ['proof_vcf' if k == 1 else 'outcome' for k in data['kind'].tolist()]}
    route = ['exact_leaf' if i % 9 == 0 or p < 5 else 'network' for i, p in enumerate(info['ply'])]
    return data, info, route


class Stage:
    """A tmp checkout: the real v2 contract (outputs under tmp), stand-in models, records and H-dev."""

    def __init__(self, root: Path):
        self.root = root
        self.c = json.loads((ROOT / CONTRACT).read_text())
        self.contract_path = root / CONTRACT
        self.contract_path.parent.mkdir(parents=True)
        self.contract_path.write_text(json.dumps(self.c))
        self.eid = cli.experiment_sha256(self.c)
        self.r2c = json.loads((ROOT / 'docs/h7r/r2/R2-DIAG-1.contract.json').read_text())
        self.r2c['bootstrap'] = {**self.r2c['bootstrap'], 'samples': 300}
        self.data, self.info, self.route = hdev_standin()
        self.hdev = {'rows': int(self.data['value'].shape[0]), 'stand_in': True}
        parent = small_model()
        self.models = {'c5': parent}
        self.recs = {}
        for s in cli.SEEDS:
            for arm in cli.ARMS:
                m = copy.deepcopy(parent) if arm == 'narrow' else d1.widen_value_head(parent, 4, int(s))
                with torch.no_grad():  # a trained head stand-in: perturb the value head only
                    g = torch.Generator().manual_seed(int(s) + (0 if arm == 'narrow' else 7))
                    for n, t in m.named_parameters():
                        if n.startswith('value_head.'):
                            t.add_(0.05 * torch.randn(t.shape, generator=g))
                self.models[f'{arm}{s}'] = m.eval()
                run_dir = root / self.c['output']['run_dirs'][arm][s]
                run_dir.mkdir(parents=True)
                (run_dir / 'best.pt').write_bytes(f'{arm}{s}'.encode())
                (run_dir / 'best.json').write_text('{}')
                self.recs[(arm, s)] = {'verdict': 'CANDIDATE', 'selected': {'state': {
                    'checkpoint_sha256': r2.file_sha256(run_dir / 'best.pt'),
                    'metadata_sha256': r2.file_sha256(run_dir / 'best.json')}}}
        self.fit = {'category': 'WIDTH_EFFECT'}
        self.amendment = {
            'format': ev.AMENDMENT_FORMAT, 'amendment_id': 'test-A1', 'approved': True,
            'contract': {'path': CONTRACT, 'experiment_sha256': self.eid},
            'tool': {'path': ev.TOOL_PATH, 'sha256': ev.tool_sha256()},
            'predicted_record': 'runs/h7r/R2-D1-HEAD-WIDTH-1/1k-v2/evaluate-predicted.json',
            'unchanged': 'x', 'reason': 'x'}
        self.amendment_path = root / ev.AMENDMENT
        self.amendment_path.write_text(json.dumps(self.amendment))
        self.predict_calls = 0

    def check_model(self, name, spec, ckpt, meta, frozen, device):
        r2.require(r2.file_sha256(ckpt) == spec['value_checkpoint_sha256'], f'model {name}: checkpoint differs')
        return copy.deepcopy(self.models[name]), {'value_checkpoint_sha256': spec['value_checkpoint_sha256'],
                                                  'name': name}

    def predict(self, model, data, device):
        from hybrid.h6_value_train import state_sha256
        self.predict_calls += 1
        key = state_sha256(model.state_dict())
        if key not in PREDICTIONS:  # the stand-ins are deterministic: one real inference per model across tests
            PREDICTIONS[key] = REAL_PREDICT(model, data, 'cpu')
        return copy.deepcopy(PREDICTIONS[key])

    def pinned(self, root, ref, what):
        return self.r2c if what == 'R2 contract' else {'hdev': self.hdev}

    @contextlib.contextmanager
    def patched(self, module):
        """The same stand-ins under ``module`` (the pinned CLI, or the CLI module the amended tool imports)."""
        patches = [
            mock.patch.object(module, 'load_d1', return_value=(self.c, 's' * 64, self.eid)),
            mock.patch.object(module, 'git_state', return_value={'commit': 'c' * 40, 'dirty': False}),
            mock.patch.object(module, 'checked_prepare', return_value={}),
            mock.patch.object(module, 'run_records', return_value=self.recs),
            mock.patch.object(module, 'checked_fit', return_value=self.fit),
            mock.patch.object(module, 'pinned', side_effect=self.pinned),
            mock.patch.object(module, 'load_c5', side_effect=lambda r2c, path, dev: (
                copy.deepcopy(self.models['c5']), {'value_checkpoint_sha256': 'c5'})),
            mock.patch.object(r2, 'check_model', side_effect=self.check_model),
            mock.patch.object(r2, 'predict', side_effect=self.predict),
            mock.patch.object(r2cli_top, 'check_hdev', return_value=(None, self.data, self.info, self.hdev)),
            mock.patch.object(h7r_holdout, 'routes', return_value=self.route),
            mock.patch('hybrid.h7c_nmask.committed', return_value=True),
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            yield

    def args(self, resume=False):
        return SimpleNamespace(d1_contract=self.contract_path, amendment=self.amendment_path, root=self.root,
                               cache='hdev', c5='c5.pt', device='cpu', resume=resume)

    def run(self, resume=False):
        with self.patched(ev.cli):
            return ev.evaluate(self.args(resume), log=QUIET)

    def out(self, name):
        return self.root / ({'result': self.c['output']['result'], 'predictions': self.c['output']['predictions'],
                             'record': self.amendment['predicted_record']}[name])


READING_KEYS = ('hdev', 'hdev_reading', 'paired', 'references', 'train_fit_reading')


class EvaluateTest(unittest.TestCase):
    def test_the_amended_tool_reads_as_the_pinned_evaluate(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            pinned_stage, amended = Stage(Path(a)), Stage(Path(b))
            with pinned_stage.patched(cli):
                ref = cli.evaluate(pinned_stage.args(), log=QUIET)
            out = amended.run()
            pinned_reading = json.loads(json.dumps({k: ref[k] for k in READING_KEYS}))
            self.assertEqual({k: out[k] for k in READING_KEYS}, pinned_reading)
            self.assertIn(out['hdev_reading']['label'], ('DEVELOPMENT_SIGNAL', 'HOLD', 'NOT_ESTABLISHED'))
            self.assertEqual(set(out['paired']), set(cli.SEEDS))
            self.assertIsNone(out['resumed'])
            self.assertEqual(amended.predict_calls, 7)
            written = json.loads(amended.out('result').read_text())
            self.assertEqual(written['evaluate_tool']['tool']['sha256'], ev.tool_sha256())
            self.assertEqual(written['predictions']['sha256'], r2.file_sha256(amended.out('predictions')))
            record = json.loads(amended.out('record').read_text())
            self.assertEqual(record['predictions']['sha256'], written['predictions']['sha256'])
            self.assertEqual(record['models'], sorted(NAMES))
            self.assertFalse(amended.out('predictions').with_name('predictions.pt.partial').exists())
            with self.assertRaisesRegex(ValueError, 'never overwritten'):
                amended.run()

    def test_an_interrupted_reading_is_resumed_without_a_new_inference(self):
        with TemporaryDirectory() as tmp:
            stage = Stage(Path(tmp))
            with mock.patch.object(ev, 'publish_result', side_effect=OSError('interrupted')):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    stage.run()
            self.assertTrue(stage.out('predictions').is_file() and stage.out('record').is_file())
            self.assertFalse(stage.out('result').exists())
            with self.assertRaisesRegex(ValueError, 'complete them with --resume'):
                stage.run()
            calls = stage.predict_calls
            out = stage.run(resume=True)
            self.assertEqual(stage.predict_calls, calls)  # no new inference
            self.assertIsNotNone(out['resumed'])
            self.assertEqual(out['resumed']['predicted_record_sha256'], r2.file_sha256(stage.out('record')))
            with TemporaryDirectory() as other:
                straight = Stage(Path(other)).run()
            self.assertEqual({k: out[k] for k in READING_KEYS}, {k: straight[k] for k in READING_KEYS})
            with self.assertRaisesRegex(ValueError, 'nothing to resume'):
                stage.run(resume=True)

    def interrupted(self, tmp):
        stage = Stage(Path(tmp))
        with mock.patch.object(ev, 'publish_result', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                stage.run()
        return stage

    def test_resume_refuses_altered_predictions_and_changed_checkpoints(self):
        with TemporaryDirectory() as tmp:
            stage = self.interrupted(tmp)
            preds = stage.out('predictions')
            saved = preds.read_bytes()
            stored = torch.load(preds, weights_only=True)
            stored['models']['wide31001']['d4'][0] += 0.5
            preds.unlink()
            torch.save(stored, preds)
            with self.assertRaisesRegex(ValueError, 'not the recorded one'):
                stage.run(resume=True)
            preds.unlink()
            preds.write_bytes(saved)
            stage.recs[('wide', '31002')]['selected']['state']['checkpoint_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
                stage.run(resume=True)
        with TemporaryDirectory() as tmp:
            stage = self.interrupted(tmp)
            stage.models['narrow31003'] = small_model()  # same file hashes, other weights: the identity record
            original = stage.check_model

            def other_identity(name, spec, ckpt, meta, frozen, device):
                model, ident = original(name, spec, ckpt, meta, frozen, device)
                return model, {**ident, 'name': name + '-changed'} if name == 'narrow31003' else ident
            stage.check_model = other_identity
            with self.assertRaisesRegex(ValueError, 'differs from the one the predictions were made with'):
                stage.run(resume=True)

    def test_a_link_interruption_is_completed_and_an_unrecorded_partial_is_never_used(self):
        with TemporaryDirectory() as tmp:
            stage = self.interrupted(tmp)
            preds = stage.out('predictions')
            partial = preds.with_name(preds.name + '.partial')
            preds.rename(partial)  # interrupted between the record and the link
            out = stage.run(resume=True)
            self.assertTrue(preds.is_file() and not partial.exists())
            self.assertIsNotNone(out['resumed'])
        with TemporaryDirectory() as tmp:
            stage = Stage(Path(tmp))
            partial = stage.out('predictions').with_name('predictions.pt.partial')
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b'unfinished')
            with self.assertRaisesRegex(ValueError, 'without its record'):
                stage.run()
            with self.assertRaisesRegex(ValueError, 'no predictions record'):
                stage.run(resume=True)
            self.assertEqual(partial.read_bytes(), b'unfinished')  # left for inspection

    def test_the_amendment_is_checked(self):
        cases = {
            'not approved': lambda a: a.update(approved=False),
            'not the tool the amendment pins': lambda a: a['tool'].update(sha256='0' * 64),
            'not for this D1 contract': lambda a: a['contract'].update(experiment_sha256='0' * 64),
            'under runs/': lambda a: a.update(predicted_record='docs/x.json'),
            'not a D1 evaluate amendment': lambda a: a.update(extra=1),
        }
        for message, change in cases.items():
            with self.subTest(message), TemporaryDirectory() as tmp:
                stage = Stage(Path(tmp))
                change(stage.amendment)
                stage.amendment_path.write_text(json.dumps(stage.amendment))
                with self.assertRaisesRegex(ValueError, message):
                    stage.run()
        with TemporaryDirectory() as tmp:
            stage = Stage(Path(tmp))
            with mock.patch('hybrid.h7c_nmask.committed', return_value=False), \
                    self.assertRaisesRegex(ValueError, 'not committed'):
                ev.load_amendment(stage.amendment_path, stage.root, stage.contract_path, stage.eid)


class CommittedAmendmentTest(unittest.TestCase):
    def test_the_committed_amendment_pins_this_tool_and_the_v2_experiment(self):
        a = json.loads((ROOT / ev.AMENDMENT).read_text())
        c = json.loads((ROOT / CONTRACT).read_text())
        self.assertEqual(a['contract'], {'path': CONTRACT, 'experiment_sha256': cli.experiment_sha256(c)})
        self.assertEqual(a['tool'], {'path': ev.TOOL_PATH, 'sha256': ev.tool_sha256()})
        self.assertIn(a['approved'], (True, False))
        # the v2 contract's pinned sources are untouched by the amendment
        self.assertEqual([p for p in cli.SOURCES if r2.file_sha256(ROOT / p) != c['sources'][p]], [])


if __name__ == '__main__':
    unittest.main()
