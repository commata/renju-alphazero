"""R2-D3 end-to-end flow in a temporary git repository: the approvals turned on one stage at a time (mask, train,
evaluate), each as its own commit, the stage records linked through the experiment SHA-256 although the contract
file changes with every approval, and the draft's records block run for real (copy, manifests, staged-set check,
commit, push to a local bare remote) with the repository's own .gitignore. Heavy steps (the mask build, the seed
training, the GPU H-dev inference) are replaced; the trainer itself runs on tiny caches."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from hybrid import h7c_train as tr
from hybrid import h7r_d3_mask as dm
from hybrid import h7r_r2 as r2
from scripts import h7r_d3 as d3
from tests.test_h7r_d3_run import tiny_caches

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'docs/h7r/r2/R2-D3-NO-OUTCOME-1/d3-contract.json'
DRAFT = 'docs/h7r/r2/R2-D3-NO-OUTCOME-1-draft.md'
QUIET = lambda *a, **k: None  # noqa: E731


class Reached(Exception):
    """The evaluation got past every record check to the H-dev inference."""


class Repo:
    def __init__(self, tmp: Path):
        self.root = tmp / 'renju-h6'
        self.root.mkdir()
        self.git('init', '-q', '-b', 'feat/h7-c1')
        self.git('config', 'user.email', 't@t')
        self.git('config', 'user.name', 't')
        c = json.loads((ROOT / CONTRACT).read_text())
        c['approved'] = {s: False for s in d3.STAGES}
        a1 = json.loads((ROOT / c['amendment']['path']).read_text())['r2_0b']
        files = {CONTRACT, '.gitignore', c['r2_contract']['path'], c['amendment']['path'], c['epsilon']['path'],
                 c['r0']['result']['path'], c['r0']['reproduction']['path'], a1['reference_record']['path'],
                 a1['config']['path'], 'scripts/h7r_d4_records.py', 'src/hybrid/__init__.py',
                 'src/hybrid/h7r_ledger.py', 'docs/h7r/r2/R2-DIAG-1/r2_0b/reproduction.json', *d3.SOURCES, *(v['run_record']['path'] for v in c['controls'].values())}
        for f in files:
            (self.root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, self.root / f)
        unapproved = json.loads((self.root / CONTRACT).read_text())  # the flow starts before any approval
        unapproved['approved'] = {s: False for s in d3.STAGES}
        (self.root / CONTRACT).write_text(json.dumps(unapproved, indent=2) + '\n')
        for v in c['controls'].values():  # the controls' local run records (runs/, as on the desktop)
            (self.root / v['run_dir']).mkdir(parents=True)
            shutil.copy2(ROOT / v['run_record']['path'], self.root / v['run_dir'] / 'r2_train.json')
        self.git('add', '-A')
        self.git('commit', '-qm', 'base')
        self.remote = tmp / 'origin.git'
        subprocess.run(['git', 'init', '-q', '--bare', str(self.remote)], check=True)
        self.git('remote', 'add', 'origin', str(self.remote))
        self.contract = c

    def git(self, *a) -> str:
        return subprocess.run(['git', *a], cwd=self.root, check=True, capture_output=True, text=True).stdout.strip()

    def approve(self, stage):
        path = self.root / CONTRACT
        c = json.loads(path.read_text())
        c['approved'][stage] = True
        path.write_text(json.dumps(c, indent=2) + '\n')
        self.git('commit', '-qam', f'approve {stage}')
        return r2.file_sha256(path)

    def args(self, **kw):
        return SimpleNamespace(d3_contract=self.root / CONTRACT, root=self.root, **kw)


class FlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.repo = Repo(Path(self.tmp.name))
        self.cycle, self.replay, self.keep, _ = tiny_caches()

    def tearDown(self):
        self.tmp.cleanup()

    def mask_stage(self):
        import torch
        from hybrid.h3_cache import tensor_sha256
        repo, c = self.repo, self.repo.contract
        ref = json.loads((repo.root / json.loads((repo.root / c['amendment']['path']).read_text())['r2_0b'][
            'reference_record']['path']).read_text())
        cache = repo.root / 'runs/h7c/c8/value_cache'
        cache.mkdir(parents=True)
        (cache / 'manifest.json').write_text(json.dumps({
            'labels_sha256': ref['inputs']['cycle_labels_sha256'],
            'splits': {k: {'sha256': v} for k, v in ref['inputs']['cycle_cache_splits'].items()}}))
        rows = repo.root / 'runs/h7c/c8/value_rows.jsonl.gz'
        rows.write_bytes(b'rows')
        keep = self.keep
        body = {'format': dm.MASK_FORMAT, 'definition': dm.DEFINITION, 'labels_sha256': 'l' * 64,
                'splits': {s: {'rows': int(keep[s].numel()), 'removed_rows': int((~keep[s]).sum()),
                               'kept_rows': int(keep[s].sum()),
                               'kept_weight': float(self.cycle[s]['weight'].double()[keep[s]].sum())} for s in keep},
                'keep_sha256': tensor_sha256({s: keep[s].to(torch.uint8) for s in keep})}
        with mock.patch.object(dm, 'build_outcome_mask', return_value=(keep, body)):
            record = d3.mask(repo.args(cycle_cache=str(cache), cycle_rows=str(rows)), log=QUIET)
        repo.git('add', c['output']['mask_manifest'])
        repo.git('commit', '-qm', 'mask manifest (records only)')
        return record

    def fake_seed_train(self, control_records):
        cycle, replay = self.cycle, self.replay

        def train(ns, log=None):
            from tests.test_h7c_train import small_model
            out = tr.train_value(tr.ValueConfig(batch_size=16, epochs=1.0, evaluations=2, seed=ns.seed,
                                                h6_guard_ratio=100.0), small_model(), cycle, replay, log=QUIET)
            control = control_records[str(ns.seed)]
            ns.out_dir.mkdir(parents=True)
            record = {**out['record'], 'config': {}, 'total_steps': control['assignment']['total_steps'],
                      'train_rows_per_epoch': 2 * control['assignment']['n_cycle']}
            (ns.out_dir / 'record.json').write_text(json.dumps(record))
            (ns.out_dir / 'config.json').write_text('{}')
            run = {'assignment': control['assignment'], 'config': control['config'], 'verdict': 'CANDIDATE',
                   'best_step': out['record']['best']['step'], 'seed': ns.seed, 'role': 'seed_spread',
                   'candidate': {'checkpoint_sha256': 'c' * 64, 'metadata_sha256': 'm' * 64}}
            (ns.out_dir / 'r2_train.json').write_text(json.dumps(run))
            return run
        return train

    def test_staged_approvals_link_through_the_experiment_and_the_records_commit(self):
        repo, c = self.repo, self.repo.contract
        for stage in d3.STAGES:  # nothing runs before its approval
            with self.assertRaisesRegex(ValueError, f'{stage} stage is not approved'):
                d3.load_d3(repo.root / CONTRACT, repo.root, stage)
        shas = {'mask': repo.approve('mask')}
        with self.assertRaisesRegex(ValueError, 'train stage is not approved'):
            d3.load_d3(repo.root / CONTRACT, repo.root, 'train')
        mask_record = self.mask_stage()

        shas['train'] = repo.approve('train')
        controls = {s: json.loads((repo.root / v['run_record']['path']).read_text()) for s, v in c['controls'].items()}
        with mock.patch('h7r_r2_seed.train', side_effect=self.fake_seed_train(controls)):
            trains = {s: d3.train(repo.args(seed=s, cycle_cache='cc', cycle_rows='cr', replay_cache='rc', parent='p',
                                            device='cpu'), log=QUIET) for s in d3.SEEDS}
        for s, rec in trains.items():
            self.assertTrue(rec['comparison']['passed'], rec['comparison'])
            self.assertEqual(rec['contract_sha256'], shas['train'])

        shas['evaluate'] = repo.approve('evaluate')
        _, csha, eid = d3.load_d3(repo.root / CONTRACT, repo.root, 'evaluate')
        self.assertEqual(csha, shas['evaluate'])
        self.assertEqual(len({shas['mask'], shas['train'], shas['evaluate']}), 3)  # the file changed every time
        self.assertEqual({mask_record['experiment_sha256'], *(r['experiment_sha256'] for r in trains.values())},
                         {eid})  # ... the experiment did not
        self.assertEqual(mask_record['contract_sha256'], shas['mask'])
        with mock.patch('h7r_r2_seed.approved', return_value=({'models': {'c5': {}}, 'frozen_state_sha256': 'f'},
                                                              'r', {}, 'a')), \
                mock.patch('h7r_r2_seed.checked_training_run',
                           return_value=({'candidate': {'checkpoint_sha256': 'c' * 64, 'metadata_sha256': 'm' * 64}},
                                         {})), \
                mock.patch.object(r2, 'check_model', return_value=(object(), {})), \
                mock.patch('h7r_r2.check_hdev', side_effect=Reached):
            with self.assertRaises(Reached):  # every record check passed: the H-dev inference would start
                d3.evaluate(repo.args(cache='h', c5='p.pt,p.json', device='cpu'), log=QUIET)

            # a change beyond the approvals is another experiment: the earlier stage records are refused
            path = repo.root / CONTRACT
            changed = json.loads(path.read_text())
            changed['reference']['note'] += ' (edited)'
            path.write_text(json.dumps(changed, indent=2) + '\n')
            repo.git('commit', '-qam', 'edit the contract')
            with self.assertRaisesRegex(ValueError, 'not a comparable candidate of this experiment'):
                d3.evaluate(repo.args(cache='h', c5='p.pt,p.json', device='cpu'), log=QUIET)
            repo.git('revert', '--no-edit', 'HEAD')

        self.records_block(repo, c)

    def records_block(self, repo, c):
        """The draft's records block, run for real on the repository's .gitignore and a local bare remote."""
        X = repo.root / 'docs/h7r/r2/R2-D3-NO-OUTCOME-1/d3'
        R = repo.root / 'runs/h7r/R2-D3-NO-OUTCOME-1/d3'
        (X / 'd3_eval.json').write_text(json.dumps({'format': d3.EVAL_FORMAT}))
        (R / 'logs').mkdir(parents=True, exist_ok=True)
        for name in ('mask', 'train-31001', 'evaluate'):
            (R / 'logs' / f'{name}-0.log').write_text('__START__\n__EXIT__ 0\n')
        block = self.block(4)
        (repo.root / '.venv/bin').mkdir(parents=True)
        (repo.root / '.venv/bin/activate').write_text('')
        run = lambda script: subprocess.run(['bash'], input=script.replace('cd ~/renju-h6', f'cd "{repo.root}"', 1),  # noqa: E731
                                            text=True, capture_output=True, env={**os.environ})
        old = run(block.replace('seed-records', 'runs'))  # the earlier path is ignored: git add refuses it
        self.assertNotEqual(old.returncode, 0)
        self.assertIn('ignored', old.stdout + old.stderr)
        for leftover in (X / 'runs', X / 'run-logs', X / 'log_manifest.json', X / 'records_manifest.json'):
            shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink(missing_ok=True)
        repo.git('reset', '-q')
        out = run(block)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        files = repo.git('show', '--name-only', '--format=', 'HEAD').splitlines()
        self.assertEqual(sorted(f for f in files if '/seed-records/' in f),
                         sorted(f'docs/h7r/r2/R2-D3-NO-OUTCOME-1/d3/seed-records/seed-{s}/{n}' for s in d3.SEEDS
                                for n in ('d3_train.json', 'r2_train.json', 'record.json', 'config.json')))
        self.assertIn('docs/h7r/r2/R2-D3-NO-OUTCOME-1/d3/d3_eval.json', files)
        self.assertFalse(any(f.endswith('.pt') for f in files))
        self.assertEqual(repo.git('rev-parse', 'HEAD'),
                         subprocess.run(['git', '--git-dir', str(repo.remote), 'rev-parse', 'feat/h7-c1'],
                                        capture_output=True, text=True, check=True).stdout.strip())

    def block(self, number):
        doc = (ROOT / DRAFT).read_text()
        section = re.split(r'\n##+ ', doc.split('### 2.6', 1)[1], maxsplit=1)[0]
        return re.findall(r'```bash\n(.*?)```', section, re.S)[number - 1]


if __name__ == '__main__':
    unittest.main()
