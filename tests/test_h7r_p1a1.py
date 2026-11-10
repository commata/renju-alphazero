"""R2-P1-POLICY-REPLAY-A1 (scripts/h7r_p1a1.py): the contract and its experiment id (not the amended one); the
deterministic settings (refused without the environment, set for the run, restored after); the stages on a real
temporary git repository: prepare, the five gate runs through the main-run entry point, the exact gate (strict checks,
G1, G2) with the c2 record as a reference only, the train gate re-judged from the committed gate records, the numeric
settings of a main run held to the gate's, link, the Arena contract and the judge; and the desktop blocks."""
import copy
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

from hybrid import h7c_train as tr
import h7r_p1 as p1
import h7r_p1a1 as a1
from tests.test_h7c_train import policy_rows, small_model
from tests.test_h7r_p1 import QUIET, STUB_EVAL, fake_cache, git, guard_eval

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / 'docs/h7r/r2/R2-P1-POLICY-REPLAY-A1/p1a1-contract.draft.json'
CUBLAS = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}


def draft_contract():
    c = json.loads(DRAFT.read_text())
    return {**c, 'status': 'fixed', 'sources': []}


class ContractTest(unittest.TestCase):
    def write(self, tmp, c):
        path = Path(tmp) / 'c.json'
        path.write_text(json.dumps(c))
        return path

    def test_contract_rules_and_the_experiment_id(self):
        c = draft_contract()
        old = json.loads((ROOT / p1.CONTRACT).read_text())
        with TemporaryDirectory() as tmp, mock.patch.object(a1, 'committed', return_value=True):
            _, _, eid = a1.load_contract(self.write(tmp, c), ROOT)
            self.assertNotEqual(eid, p1.experiment_sha256(old))  # a new experiment
            self.assertEqual(c['amendment']['of']['experiment_sha256'], p1.experiment_sha256(old))
            with self.assertRaisesRegex(ValueError, 'gate stage is not approved'):
                a1.load_contract(self.write(tmp, {**c, 'approved': {**c['approved'], 'prepare': True}}), ROOT, 'gate')
            cases = {
                'approved before': {**c, 'approved': {'prepare': True, 'gate': False, 'train': True, 'arena': False}},
                'approved flags': {**c, 'approved': {'prepare': True, 'train': False, 'arena': False}},
                'gate runs / seed': {**c, 'stages': {**c['stages'], 'gate': {**c['stages']['gate'],
                                                                             'runs': {'E1': 'old', 'E2': 'C'}}}},
                'deterministic settings': {**c, 'determinism': {**c['determinism'], 'settings': {
                    **c['determinism']['settings'], 'deterministic_algorithms': False}}},
                'not the R2-P1-POLICY-REPLAY-A1 contract': old,
            }
            for message, bad in cases.items():
                with self.subTest(message):
                    with self.assertRaisesRegex(ValueError, message):
                        a1.load_contract(self.write(tmp, bad), ROOT)
            approved = {'prepare': True, 'gate': True, 'train': True, 'arena': False}
            _, _, eid2 = a1.load_contract(self.write(tmp, {**c, 'approved': approved}), ROOT, 'train')
            self.assertEqual(eid2, eid)  # an approval never changes the experiment id

    def test_the_unchanged_parts_are_the_p1_contract(self):
        c, old = draft_contract(), json.loads((ROOT / p1.CONTRACT).read_text())
        for k in ('goal', 'question', 'inputs', 'arms', 'seeds', 'arena', 'reading', 'data_rule', 'separate_stages'):
            self.assertEqual(c[k], old[k], k)
        self.assertEqual({k: v for k, v in c['training'].items() if k != 'numeric'}, old['training'])
        exp = old['stages']['reproduce']['expected']
        self.assertEqual(c['stages']['gate']['expected'],
                         {k: exp[k] for k in ('total_steps', 'steps_per_epoch', 'warmup_steps', 'evaluation_steps')})
        self.assertEqual(c['stages']['gate']['tolerance'], 0)
        for rel in ('docs/h7r/r2/R2-P1-POLICY-REPLAY-1/reproduce.json', 'docs/h7r/r2/R2-P1-DIAG-1/diagnosis.json'):
            self.assertIn(rel, c['evidence'])

    def test_the_fixed_contract_matches_its_draft(self):
        fixed = json.loads((ROOT / a1.CONTRACT).read_text())
        draft = json.loads(DRAFT.read_text())
        self.assertEqual(set(fixed['approved']), set(a1.STAGES))  # approvals change over time, never the id
        self.assertEqual(a1.experiment_sha256(fixed),
                         'faeb72b0f625baba213cc88504388dcd1d5eed03a16307de8da2217583c02648')
        self.assertEqual(fixed['draft'], {'path': DRAFT.relative_to(ROOT).as_posix(), 'sha256': p1.file_sha256(DRAFT)})
        same = {k: v for k, v in fixed.items() if k not in ('status', 'sources', 'draft', 'approved')}
        self.assertEqual(same, {k: v for k, v in draft.items() if k not in ('status', 'sources', 'approved')})
        paths = {s['path'] for s in fixed['sources']}
        for rel in ('scripts/h7r_p1a1.py', 'scripts/h7r_p1.py', 'scripts/h7r_p1_diag.py', 'src/hybrid/h7r_p1_train.py',
                    'src/hybrid/h7c_train.py', 'src/hybrid/h7r_d1.py'):
            self.assertIn(rel, paths)
        a1.load_contract(None, ROOT, check_sources=True)  # every pinned source and evidence file is this checkout's


class DeterministicTest(unittest.TestCase):
    def test_refused_without_the_environment_set_then_restored(self):
        c = draft_contract()
        with mock.patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ''}):
            with self.assertRaisesRegex(ValueError, 'CUBLAS_WORKSPACE_CONFIG must be'):
                with a1.deterministic(c, 'cpu'):
                    pass
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        with mock.patch.dict(os.environ, CUBLAS):
            with a1.deterministic(c, 'cpu') as s:
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertTrue(torch.backends.cudnn.deterministic)
                self.assertEqual(s['environment'], CUBLAS)
                for k in ('torch', 'cuda', 'cudnn', 'device_name', 'threads', 'cudnn_allow_tf32'):
                    self.assertIn(k, s)
            torch.backends.cuda.matmul.allow_tf32 = True  # (mock.patch.object does not restore this switch)
            try:
                with self.assertRaisesRegex(ValueError, 'matmul_allow_tf32'):
                    with a1.deterministic(c, 'cpu'):
                        pass
            finally:
                torch.backends.cuda.matmul.allow_tf32 = False
            self.assertEqual(torch.get_float32_matmul_precision(), 'highest')
        self.assertFalse(torch.are_deterministic_algorithms_enabled())  # restored
        self.assertFalse(torch.backends.cudnn.deterministic)

    def test_differences(self):
        a = {'record': {'evaluations': [1], 'best': 2}, 'fingerprints': {'loss': [0.5]}, 'x': 1}
        self.assertEqual(a1.differences(a, copy.deepcopy(a)), [])
        b = copy.deepcopy(a)
        b['record']['best'] = 3
        b['fingerprints']['loss'][0] = 0.5000001
        self.assertEqual(a1.differences(a, b), ['fingerprints.loss', 'record.best'])


class RealGitTest(unittest.TestCase):
    """The stages on a real temporary git repository with the repository's real .gitignore; stand-in inputs (tiny
    rows, small networks, a fake cache) under the ignored runs/ and data/; CPU under the deterministic settings."""

    SOURCE = 'src/hybrid/h7r_p1_train.py'

    def setUp(self):
        import gzip
        from analysis import tb_v3
        from model.checkpoint import save_checkpoint
        from model.config import ModelConfig
        from tests.test_h7c_train import source_of

        env = mock.patch.dict(os.environ, CUBLAS)
        env.start()
        self.addCleanup(env.stop)
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.root = Path(self.tmp.name)
        git(root, 'init', '-q', '-b', 'feat/h7-c1')
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(root, 'config', k, v)
        shutil.copy(ROOT / '.gitignore', root / '.gitignore')
        (root / self.SOURCE).parent.mkdir(parents=True)
        shutil.copy(ROOT / self.SOURCE, root / self.SOURCE)
        cfg = ModelConfig()
        h3 = root / p1.PATHS['h3']
        h3.parent.mkdir(parents=True)
        save_checkpoint(h3, small_model())
        (root / p1.h3_meta_path()).write_text(json.dumps({'policy_trained': True, 'value_trained': False,
                                                           'config': {'channels': cfg.channels, 'blocks': cfg.blocks}}))
        value = root / p1.PATHS['c5_value']
        value.parent.mkdir(parents=True)
        save_checkpoint(value, small_model(5))
        (root / p1.c5_meta_path()).write_text(json.dumps({'stage': 'H6-2a', 'trunk_frozen': True,
                                                           'parent_h3_checkpoint_sha256': p1.file_sha256(h3)}))
        source = source_of(h3, value)
        rows = policy_rows(6, source=source, cycle=2) + policy_rows(2, 'val', source=source, cycle=2)
        path = root / p1.PATHS['cycle_rows']
        path.parent.mkdir(parents=True)
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        cache_dir, expected = fake_cache(root)
        target = root / p1.PATHS['cache']
        target.parent.mkdir(parents=True)
        shutil.move(str(cache_dir), str(target))
        for s in ('O0', 'O1', 'O2'):
            dest = root / f'docs/h7r/suites/{s}.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / f'docs/h7r/suites/{s}.json', dest)
        c = draft_contract()
        c['evidence'] = {}
        c['sources'] = [{'path': self.SOURCE, 'sha256': p1.file_sha256(root / self.SOURCE)}]
        c['inputs']['cycle_rows'] = {'path': p1.PATHS['cycle_rows'], 'sha256': p1.file_sha256(path),
                                     'train_rows': sum(r['split'] == 'train' for r in rows),
                                     'val_rows': sum(r['split'] == 'val' for r in rows)}
        c['inputs']['parent_and_teacher'] = {'policy_checkpoint_sha256': p1.file_sha256(h3),
                                             'policy_metadata_sha256': p1.file_sha256(root / p1.h3_meta_path())}
        c['inputs']['value_c5'] = {'value_checkpoint_sha256': p1.file_sha256(value),
                                   'value_metadata_sha256': p1.file_sha256(root / p1.c5_meta_path())}
        c['inputs']['renjunet_cache']['expected'] = expected
        c['training'] = {**c['training'], 'batch_size': 16, 'epochs': 1.0, 'evaluations': 3, 'guard_drop': 1.0,
                         'device': 'cpu'}
        c['arms']['T']['replay'] = {**c['arms']['T']['replay'], 'batch': 8}
        frozen = mock.patch.dict(tb_v3.MODELS, {**tb_v3.MODELS, **c['inputs']['parent_and_teacher']})
        frozen.start()
        self.addCleanup(frozen.stop)
        # the "c2" record: the unchanged h7c trainer on the same inputs, default settings
        checked = p1.check_inputs(c, root)
        cfg_ = p1.policy_config(c, 20261024, 'cpu')
        old = tr.train_policy(cfg_, small_model(), small_model(), tr.policy_tensors(checked['rows']['train']),
                              tr.policy_tensors(checked['rows']['val']), checked['cache']['val'], guard_eval, log=QUIET)
        rec = old['record']
        inputs = {'policy_rows_sha256': c['inputs']['cycle_rows']['sha256'],
                  'parent_checkpoint_sha256': p1.file_sha256(h3),
                  'parent_metadata_sha256': p1.file_sha256(root / p1.h3_meta_path()),
                  'teacher_checkpoint_sha256': p1.file_sha256(h3)}
        self.c2 = {'config': tr.config_dict(cfg_), 'inputs': inputs, **rec}
        (root / p1.C2_RECORD).parent.mkdir(parents=True)
        (root / p1.C2_RECORD).write_text(json.dumps(self.c2))
        c['stages']['gate']['expected'] = {
            'total_steps': rec['total_steps'], 'steps_per_epoch': rec['steps_per_epoch'],
            'warmup_steps': rec['warmup_steps'], 'evaluation_steps': [e['step'] for e in rec['evaluations']]}
        self.c = c
        self.write_contract(c)
        git(root, 'add', '-A')
        git(root, 'commit', '-qm', 'contract')
        self.assertEqual(git(root, 'ls-files', 'runs', 'data'), '')  # inputs stay git-ignored

    def write_contract(self, c, indent=None):
        (self.root / a1.CONTRACT).parent.mkdir(parents=True, exist_ok=True)
        (self.root / a1.CONTRACT).write_text(json.dumps(c, indent=indent))

    def approve(self, stage):
        c = json.loads((self.root / a1.CONTRACT).read_text())
        c['approved'][stage] = True
        self.write_contract(c, indent=1)
        git(self.root, 'commit', '-qam', f'approve {stage}')

    def commit(self, *paths):
        git(self.root, 'add', *paths)
        git(self.root, 'commit', '-qm', 'records')

    def ns(self, **k):
        import argparse
        return argparse.Namespace(contract=None, device='cpu', **k)

    def copy_gate_records(self):
        for run in a1.GATE_RUNS:
            dst = self.root / a1.gate_record_path(run)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.root / a1.RUNS / 'gate' / run / 'record.json', dst)
        self.commit(*a1.gate_record_paths())

    def through_gate_runs(self):
        self.approve('prepare')
        self.assertTrue(a1.prepare(self.ns(), self.root)['passed'])
        self.commit(f'{a1.D}/prepare.json')
        self.approve('gate')
        recs = {run: a1.gate_run(self.ns(run=run), self.root, log=QUIET) for run in a1.GATE_RUNS}
        self.copy_gate_records()
        return recs

    def through_gate(self):
        recs = self.through_gate_runs()
        g = a1.gate(self.ns(), self.root)
        self.assertTrue(g['passed'], g)
        self.commit(f'{a1.D}/gate.json')
        return recs, g

    def test_the_stages_on_real_git(self):
        recs, g = self.through_gate()
        eid = g['experiment_sha256']
        self.assertEqual(recs['E1']['kind'], 'old')
        self.assertIsNone(recs['E1']['p1'])
        self.assertIsNone(recs['E2']['p1']['replay'])
        self.assertEqual(recs['E4']['p1']['replay']['batch'], 8)
        self.assertNotEqual(recs['E4']['selected_state_sha256'], recs['E2']['selected_state_sha256'])
        self.assertEqual(recs['E1']['selected_state_sha256'], recs['E2']['selected_state_sha256'])
        self.assertEqual(recs['E2']['settings']['deterministic_algorithms'], True)
        self.assertTrue((self.root / a1.RUNS / 'gate/E2/state.pt').is_file())
        self.assertEqual(git(self.root, 'ls-files', 'runs'), '')
        self.assertEqual(g['G1_determinism'], {'E2=E3': [], 'E4=E5': []})
        self.assertEqual(g['G2_equivalence'], {'E1=E2': []})
        self.assertIn('evaluations', g['c2_reference']['E2'])
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            a1.gate(self.ns(), self.root)
        with self.assertRaisesRegex(ValueError, 'already holds a run'):
            a1.gate_run(self.ns(run='E1'), self.root, log=QUIET)
        with self.assertRaisesRegex(ValueError, 'train stage is not approved'):
            a1.train(self.ns(arm='C', seed=20261024), self.root, log=QUIET)
        self.approve('train')
        for seed in self.c['seeds']['training']:
            for arm in a1.ARMS:
                out = a1.train(self.ns(arm=arm, seed=seed), self.root, log=QUIET)
                self.assertEqual((out['experiment_sha256'], out['verdict']), (eid, 'CANDIDATE'))
                self.assertEqual(out['settings'], g['settings'])
                dst = self.root / a1.run_record_path(arm, seed)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(self.root / a1.RUNS / f'{arm}-{seed}' / 'record.json', dst)
                self.commit(a1.run_record_path(arm, seed))
        c24 = json.loads((self.root / a1.run_record_path('C', 20261024)).read_text())
        self.assertEqual(c24['selected_state_sha256'], recs['E2']['selected_state_sha256'])  # the gate's entry point
        self.assertEqual(c24['fingerprints'], recs['E2']['fingerprints'])
        t24 = json.loads((self.root / a1.run_record_path('T', 20261024)).read_text())
        self.assertEqual(t24['fixed_logits'], recs['E4']['fixed_logits'])
        lk = a1.link(self.ns(), self.root)
        self.assertEqual((lk['candidate_seed'], lk['experiment_sha256']), (20261024, eid))
        self.commit(f'{a1.D}/link.json')
        self.approve('arena')
        body = a1.arena_contract(self.ns(), self.root)
        self.commit(a1.ARENA_CONTRACT)
        self.assertEqual(body['contract_id'], 'R2-P1-A1-ARENA-1')
        self.assertEqual(sorted({a['seed'] for a in body['arms']}), [8788, 8789, 8790])
        self.assertEqual(body['models']['T']['identity']['policy_checkpoint_sha256'], t24['candidate']['checkpoint_sha256'])
        arena = json.loads((self.root / a1.ARENA_CONTRACT).read_text())
        bad = copy.deepcopy(arena)
        bad['arms'][0]['pairs'] = 1
        (self.root / a1.ARENA_CONTRACT).write_text(json.dumps(bad))
        self.commit(a1.ARENA_CONTRACT)
        with self.assertRaisesRegex(ValueError, 'not the one the A1 contract and the verified link give'):
            a1.judge(self.ns(arms_dir=str(self.root / 'arms')), self.root)
        (self.root / a1.ARENA_CONTRACT).write_text(json.dumps(arena))
        self.commit(a1.ARENA_CONTRACT)
        arms_dir = self.root / 'arms'
        arms_dir.mkdir()
        sizes = {a['name']: a['pairs'] for a in arena['arms']}
        for name in sizes:
            (arms_dir / f'{name}.json').write_text('{}')

        def load(path, contract, sha, root):
            name = Path(path).stem
            score = 1.0 if '_T_' in name else 0.5
            pairs = [{'opening_id': i, 'pair_score': score, 'class_id': i // 4} for i in range(sizes[name])]
            return {'table': {'valid': True, 'pairs': pairs}, 'summary': {'pairs': sizes[name]}}
        import h7r_eval as cli
        with mock.patch.object(cli, 'load_arm', side_effect=load):
            j = a1.judge(self.ns(arms_dir=str(arms_dir)), self.root)
        self.assertEqual(j['readings'], {'P': 'STRONGER', 'D': 'BETTER_THAN_CONTROL'})
        self.assertEqual(j['experiment_sha256'], eid)

    def test_the_gate_is_exact_and_rejudged(self):
        self.through_gate_runs()
        paths = {run: self.root / a1.gate_record_path(run) for run in a1.GATE_RUNS}
        good = {run: json.loads(p.read_text()) for run, p in paths.items()}
        cases = {
            'a loss 1 ulp off (G1)': ('E3', lambda r: r['fingerprints']['loss'].__setitem__(
                1, float(torch.nextafter(torch.tensor(r['fingerprints']['loss'][1], dtype=torch.float64),
                                         torch.tensor(1e9, dtype=torch.float64))))),
            'another replay KL record (G1)': ('E5', lambda r: r['p1']['evaluations'][0].update(renjunet_val_kl={})),
            'another selected state (G2)': ('E1', lambda r: r.update(selected_state_sha256='0' * 64)),
            'another evaluation value (G2)': ('E1', lambda r: r['record']['evaluations'][0]['val'].update(ce=0.0)),
            'non-deterministic settings': ('E4', lambda r: r['settings'].update(deterministic_algorithms=False)),
            'other threads': ('E4', lambda r: r['settings'].update(threads=999)),
            'another learning rate': ('E2', lambda r: r['config'].update(learning_rate=1.0)),
            'another step structure': ('E2', lambda r: r['record'].update(warmup_steps=99)),
            'replay in an arm C run': ('E3', lambda r: r['p1'].update(replay={'mu': 0.5})),
            'the diagnostic off': ('E2', lambda r: r['p1'].update(parent_diag=None)),
            'another input': ('E5', lambda r: r['inputs'].update(policy_rows_sha256='0' * 64)),
        }
        for what, (run, change) in cases.items():
            with self.subTest(what):
                bad = copy.deepcopy(good[run])
                change(bad)
                paths[run].write_text(json.dumps(bad))
                self.commit(a1.gate_record_path(run))
                body = a1.judge_gate(self.c, a1.experiment_sha256(self.c), self.root)
                self.assertFalse(body['passed'], what)
                paths[run].write_text(json.dumps(good[run]))
                self.commit(a1.gate_record_path(run))
        # a passed gate record whose gate records change afterwards no longer opens training
        g = a1.gate(self.ns(), self.root)
        self.assertTrue(g['passed'])
        self.commit(f'{a1.D}/gate.json')
        self.approve('train')
        bad = copy.deepcopy(good['E5'])
        bad['fingerprints']['loss'][0] += 1.0
        paths['E5'].write_text(json.dumps(bad))
        self.commit(a1.gate_record_path('E5'))
        with self.assertRaisesRegex(ValueError, 'not what the committed gate records give'):
            a1.train(self.ns(arm='C', seed=20261024), self.root, log=QUIET)
        self.assertFalse((self.root / a1.RUNS / 'C-20261024').exists())

    def test_a_failed_gate_is_recorded_and_stops_training(self):
        self.through_gate_runs()
        path = self.root / a1.gate_record_path('E3')
        bad = json.loads(path.read_text())
        bad['record']['evaluations'][-1]['guard_top1'] += 1e-12
        path.write_text(json.dumps(bad))
        self.commit(a1.gate_record_path('E3'))
        g = a1.gate(self.ns(), self.root)
        self.assertFalse(g['passed'])
        self.assertEqual(g['G1_determinism']['E2=E3'], ['record.evaluations'])
        self.commit(f'{a1.D}/gate.json')
        self.approve('train')
        with self.assertRaisesRegex(ValueError, 'the gate did not pass'):
            a1.train(self.ns(arm='T', seed=20261024), self.root, log=QUIET)

    def test_the_c2_values_are_a_reference_only(self):
        c2 = copy.deepcopy(self.c2)
        for e in c2['evaluations']:
            e['eligible'] = not e['eligible']
            e['val']['objective'] += 0.5
        c2['best'] = {**c2['evaluations'][0], 'step': -1}
        (self.root / p1.C2_RECORD).write_text(json.dumps(c2))
        self.commit(p1.C2_RECORD)
        self.through_gate_runs()
        g = a1.gate(self.ns(), self.root)
        self.assertTrue(g['passed'], g)
        ref = g['c2_reference']['E2']
        self.assertFalse(ref['eligibility_equal'])
        self.assertEqual(ref['c2_selected_step'], -1)
        self.assertAlmostEqual(ref['evaluations']['objective'], 0.5, places=6)
        # the c2 step structure and inputs stay strict
        c2['evaluations'] = c2['evaluations'][:-1]
        (self.root / p1.C2_RECORD).write_text(json.dumps(c2))
        self.commit(p1.C2_RECORD)
        body = a1.judge_gate(self.c, a1.experiment_sha256(self.c), self.root)
        self.assertFalse(body['passed'])
        self.assertIn('c2', body['strict'])

    def test_main_runs_keep_the_gate_settings_and_publish_nothing_on_a_change(self):
        self.through_gate()
        self.approve('train')
        import hybrid.h7r_d1 as d1
        real = d1.numeric_settings
        with mock.patch.object(d1, 'numeric_settings', side_effect=lambda device: {**real(device), 'threads': 999}):
            with self.assertRaisesRegex(ValueError, 'differ from the gate'):
                a1.train(self.ns(arm='C', seed=20261111), self.root, log=QUIET)
        self.assertFalse((self.root / a1.RUNS / 'C-20261111').exists())
        with mock.patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':16:8'}):
            with self.assertRaisesRegex(ValueError, 'CUBLAS_WORKSPACE_CONFIG must be'):
                a1.train(self.ns(arm='C', seed=20261111), self.root, log=QUIET)
        real_execute = a1.execute

        def during(*a, **k):
            out = real_execute(*a, **k)
            git(self.root, 'commit', '-q', '--allow-empty', '-m', 'moved')
            return out
        with mock.patch.object(a1, 'execute', side_effect=during):
            with self.assertRaisesRegex(ValueError, 'changed during the run'):
                a1.train(self.ns(arm='T', seed=20261112), self.root, log=QUIET)
        out = self.root / a1.RUNS / 'T-20261112'
        self.assertEqual(sorted(x.name for x in out.iterdir()), [])  # no checkpoint, metadata, logits or record
        self.assertFalse(torch.are_deterministic_algorithms_enabled())  # restored after every run


STUB_A1 = """import json, os, sys
from pathlib import Path
D = 'docs/h7r/r2/R2-P1-POLICY-REPLAY-A1'
R = 'runs/h7r/R2-P1-POLICY-REPLAY-A1'
fail = set(filter(None, os.environ.get('A1_FAIL', '').split(',')))
forbid = set(filter(None, os.environ.get('A1_FORBID', '').split(',')))
cmd = sys.argv[1]
arg = lambda k: sys.argv[sys.argv.index(k) + 1]
def out(path, body):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(body))
if cmd in ('gate-run', 'train'):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        print('STOP: CUBLAS_WORKSPACE_CONFIG'); sys.exit(7)
    run = arg('--run') if cmd == 'gate-run' else f"{arg('--arm')}-{arg('--seed')}"
    if run in forbid:
        print('STOP: this run must not be run again'); sys.exit(9)
    if run in fail:
        print('STOP: RuntimeError: stand-in failure'); sys.exit(1)
    out(f"{R}/gate/{run}/record.json" if cmd == 'gate-run' else f'{R}/{run}/record.json', {'run': run})
    print('ran', run); sys.exit(0)
if cmd in fail:
    print(f'STOP: stand-in {cmd} failure')
    if cmd == 'gate':
        out(f'{D}/gate.json', {'passed': False}); sys.exit(2)
    sys.exit(1)
out(f'{D}/{cmd}.json', {'stage': cmd}); print('ok', cmd)
"""


class BlockTest(unittest.TestCase):
    """The desktop blocks of the draft (section 7) run by bash on a real temporary repository with a bare origin,
    the repository's real .gitignore and the real record tool; the A1 and Arena commands are stand-ins."""

    NAMES = ('a1-1', 'a1-F', 'a1-G', 'a1-2', 'a1-4')

    def setUp(self):
        import re
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.origin, self.work = base / 'origin.git', base / 'work'
        git(base, 'init', '-q', '--bare', str(self.origin))
        git(base, 'init', '-q', '-b', 'feat/h7-c1', str(self.work))
        w = self.work
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(w, 'config', k, v)
        shutil.copy(ROOT / '.gitignore', w / '.gitignore')
        for rel in ('scripts/h7r_d4_records.py', 'src/hybrid/h7r_ledger.py'):
            (w / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, w / rel)
        (w / 'src/hybrid/__init__.py').write_text('')
        (w / 'scripts/check_track_b_frozen.py').write_text('print("frozen PASS")\n')
        (w / 'scripts/h7r_p1a1.py').write_text(STUB_A1)
        (w / 'scripts/h7r_eval.py').write_text(STUB_EVAL)
        (w / '.venv/bin').mkdir(parents=True)
        (w / '.venv/bin/activate').write_text('')
        arena = {'arms': [{'name': n} for n in ('O0_T_vs_champion', 'O0_C_vs_champion')]}
        (w / 'docs/h7r/contracts').mkdir(parents=True)
        (w / 'docs/h7r/contracts/R2-P1-A1-ARENA-1.json').write_text(json.dumps(arena))
        git(w, 'add', '-A')
        git(w, 'commit', '-qm', 'init')
        git(w, 'remote', 'add', 'origin', str(self.origin))
        git(w, 'push', '-q', 'origin', 'feat/h7-c1')
        doc = (ROOT / 'docs/h7r/r2/R2-P1-POLICY-REPLAY-A1-draft.md').read_text()
        blocks = re.findall(r'```bash\n(.*?)```', doc.split('\n## 7. 실행 절차', 1)[1], re.S)
        self.assertEqual(len(blocks), 5)
        self.blocks = dict(zip(self.NAMES, (b.replace('~/renju-h6', str(w)) for b in blocks)))
        self.D = 'docs/h7r/r2/R2-P1-POLICY-REPLAY-A1'

    def bash(self, name, **env):
        import subprocess
        e = {k: v for k, v in os.environ.items() if k != 'CUBLAS_WORKSPACE_CONFIG'}
        e = {**e, 'PYTHONDONTWRITEBYTECODE': '1', **env}
        r = subprocess.run(['bash', '-c', self.blocks[name]], cwd=self.work, capture_output=True, text=True, env=e)
        return r.returncode, r.stdout + r.stderr

    def files(self, rel):
        return git(self.work, 'ls-files', rel).splitlines()

    def synced(self):
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), git(self.origin, 'rev-parse', 'feat/h7-c1'))

    def test_prepare_pass_fail_and_rerun(self):
        import time
        rc, out = self.bash('a1-1', A1_FAIL='prepare')
        self.assertEqual(rc, 1, out)
        self.assertEqual(len(self.files(f'{self.D}/stage-logs')), 2)
        self.assertIn('prepare FAIL (exit 1)', git(self.work, 'log', '-1', '--format=%s'))
        time.sleep(1.1)
        rc, out = self.bash('a1-1')
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.files(f'{self.D}/prepare.json'))
        self.synced()
        rc, out = self.bash('a1-1')
        self.assertEqual(rc, 1)
        self.assertIn('already recorded', out)

    def test_the_gate_block(self):
        rc, out = self.bash('a1-G')
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(self.files(f'{self.D}/gate-records')), 5)
        self.assertTrue(self.files(f'{self.D}/gate.json'))
        logs = [f for f in self.files(f'{self.D}/stage-logs') if f.endswith('.log')]
        self.assertEqual(len(logs), 6)  # five gate runs and the gate, each committed
        self.assertEqual(self.files('runs'), [])
        subjects = git(self.work, 'log', '--format=%s').splitlines()
        self.assertEqual(sum('gate run E' in s for s in subjects), 5)  # each run committed alone
        self.synced()
        rc, out = self.bash('a1-G')
        self.assertEqual(rc, 1)
        self.assertIn('already recorded', out)

    def test_a_failed_gate_run_stops_and_is_never_repeated(self):
        rc, out = self.bash('a1-G', A1_FAIL='E3')
        self.assertEqual(rc, 1, out)
        self.assertIn('gate run E3 failed', out)
        self.assertEqual(len(self.files(f'{self.D}/gate-records')), 2)
        failed = [f for f in self.files(f'{self.D}/stage-logs') if '/gate-run-E3-' in f and f.endswith('.log')]
        self.assertEqual(len(failed), 1)
        self.assertIn('__EXIT__ 1', (self.work / failed[0]).read_text())
        self.assertFalse(self.files(f'{self.D}/gate.json'))
        rc, out = self.bash('a1-G', A1_FORBID='E1,E2,E3,E4,E5')
        self.assertEqual(rc, 1, out)
        self.assertIn('a gate run failed before', out)
        self.synced()

    def test_a_failed_gate_is_recorded_and_stops(self):
        rc, out = self.bash('a1-G', A1_FAIL='gate')
        self.assertEqual(rc, 1, out)
        self.assertIn('the gate did not pass (exit 2)', out)
        self.assertTrue(self.files(f'{self.D}/gate.json'))
        self.assertIn('equivalence gate exit 2', git(self.work, 'log', '-1', '--format=%s'))

    def test_a_finished_but_uncommitted_gate_run_is_copied_not_rerun(self):
        R = self.work / 'runs/h7r/R2-P1-POLICY-REPLAY-A1/gate/E1'
        R.mkdir(parents=True)
        (R / 'record.json').write_text('{"run": "E1"}')
        rc, out = self.bash('a1-G', A1_FORBID='E1')
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(self.files(f'{self.D}/gate-records')), 5)

    def test_training_needs_the_gate_stops_records_and_resumes(self):
        import time
        rc, out = self.bash('a1-2')
        self.assertEqual(rc, 1)
        self.assertIn('the gate is not recorded', out)
        rc, out = self.bash('a1-G')
        self.assertEqual(rc, 0, out)
        time.sleep(1.1)
        rc, out = self.bash('a1-2', A1_FAIL='T-20261111', A1_FORBID='E1,E2,E3,E4,E5')
        self.assertEqual(rc, 1, out)
        self.assertIn('run T 20261111 failed', out)
        self.assertEqual(len(self.files(f'{self.D}/seed-records')), 3)
        R = self.work / 'runs/h7r/R2-P1-POLICY-REPLAY-A1'
        (R / 'C-20261112').mkdir(parents=True)
        (R / 'C-20261112/record.json').write_text('{"run": "C-20261112"}')
        time.sleep(1.1)
        rc, out = self.bash('a1-2', A1_FORBID='C-20261024,T-20261024,C-20261111,C-20261112')
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(self.files(f'{self.D}/seed-records')), 6)
        self.synced()

    def test_recovery_commits_unrecorded_records_only(self):
        R = self.work / 'runs/h7r/R2-P1-POLICY-REPLAY-A1'
        (R / 'gate/E2').mkdir(parents=True)
        (R / 'gate/E2/record.json').write_text('{"run": "E2"}')
        (R / 'logs').mkdir()
        (R / 'logs/gate-run-E2-20261110-000000.log').write_text('ran E2\n__EXIT__ 0\n')
        rc, out = self.bash('a1-F')
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.files(f'{self.D}/gate-records/E2/record.json'))
        self.assertTrue(self.files(f'{self.D}/stage-logs/gate-run-E2-20261110-000000.log'))
        rc, out = self.bash('a1-F')
        self.assertEqual(rc, 0, out)
        self.assertIn('nothing left to record', out)

    def test_arena_failure_keeps_head_and_resumes(self):
        import time
        head = git(self.work, 'rev-parse', 'HEAD')
        rc, out = self.bash('a1-4', EVAL_FAIL='O0_C_vs_champion')
        self.assertEqual(rc, 1, out)
        self.assertIn('nothing committed, HEAD kept', out)
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), head)
        rc, out = self.bash('a1-F')
        self.assertEqual(rc, 1, out)
        self.assertIn('the Arena is unfinished', out)
        time.sleep(1.1)
        rc, out = self.bash('a1-4')
        self.assertEqual(rc, 0, out)
        self.assertIn('resuming at', out)
        self.assertTrue(self.files(f'{self.D}/judge.json'))
        rc, out = self.bash('a1-4')
        self.assertEqual(rc, 1)
        self.assertIn('already recorded', out)


if __name__ == '__main__':
    unittest.main()
