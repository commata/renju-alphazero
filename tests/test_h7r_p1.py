"""R2-P1-POLICY-REPLAY-1: the trainer (replay off is h7c_train.train_policy; the replay term, its generator and its
KL formula) and the tool (contract and experiment id, cache checks, the reproduction rules, the Arena contract the
R1 tools accept, the stratified paired judge and its readings)."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

from hybrid import h7c_train as tr, h7r_p1_train as pt
import h7r_p1 as p1
from tests.test_h7c_train import guard_data, policy_rows, small_model

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / 'docs/h7r/r2/R2-P1-POLICY-REPLAY-1/p1-contract.draft.json'
QUIET = lambda *a, **k: None  # noqa: E731


def guard_eval(model, data):
    from hybrid.h3_train import H3Config, evaluate
    return evaluate(model, data, H3Config(), torch.device('cpu'))


class TrainerTest(unittest.TestCase):
    def setUp(self):
        self.train = tr.policy_tensors(policy_rows(6))
        self.val = tr.policy_tensors(policy_rows(2, 'val'))
        self.guard = guard_data(policy_rows(2, 'val'))
        self.replay_data = guard_data(policy_rows(4, 'train'))
        self.cfg = tr.PolicyConfig(batch_size=16, epochs=1.0, evaluations=3, learning_rate=1e-3, guard_drop=1.0)

    def run_new(self, **kw):
        return pt.train_policy(self.cfg, small_model(), small_model(), self.train, self.val, self.guard, guard_eval,
                               log=QUIET, **kw)

    @staticmethod
    def same(a, b):
        return a['record'] == b['record'] and all(torch.equal(a['state'][k], b['state'][k]) for k in a['state'])

    def test_replay_off_is_the_c2_trainer(self):
        old = tr.train_policy(self.cfg, small_model(), small_model(), self.train, self.val, self.guard, guard_eval,
                              log=QUIET)
        self.assertTrue(self.same(old, self.run_new()))
        with_diag = self.run_new(diag=self.guard)  # the diagnostic KL never touches the training path
        self.assertTrue(self.same(old, with_diag))
        self.assertEqual(len(with_diag['p1']['evaluations']), 3)
        self.assertIsNone(with_diag['p1']['replay'])
        with self.assertRaisesRegex(ValueError, 'both its settings and its positions'):
            self.run_new(replay=pt.Replay())

    def test_the_replay_term_and_its_own_generator(self):
        seen = {'C': [], 'T': []}
        real = tr.planes_from_states

        def spy(arm):
            def planes(data, index, device='cpu'):
                if data is self.train:
                    seen[arm].append(index.tolist())
                return real(data, index, device)
            return planes
        with mock.patch.object(tr, 'planes_from_states', side_effect=spy('C')):
            c = self.run_new(diag=self.guard)
        with mock.patch.object(tr, 'planes_from_states', side_effect=spy('T')):
            t = self.run_new(replay=pt.Replay(batch=8), replay_data=self.replay_data, diag=self.guard)
        self.assertEqual(seen['C'], seen['T'])  # the same cycle batches in the same order
        self.assertFalse(self.same(c, t))
        self.assertEqual(t['p1']['replay'], {'mu': 0.5, 'batch': 8, 'seed_offset': 1000003,
                                             'generator_seed': self.cfg.seed + 1000003})
        self.assertEqual(t['p1']['evaluations'][0]['step'], 1)
        self.assertIsNotNone(t['p1']['evaluations'][-1]['replay_kl_train_mean'])

    def test_replay_kl_formula(self):
        student, teacher = small_model(1), small_model(2)
        from hybrid.h3_cache import planes_from_cache
        from hybrid.h3_train import policy_logits
        planes, legal, _ = planes_from_cache(self.replay_data, torch.arange(5))
        got = pt.replay_kl(student, teacher, planes, legal)
        for i in range(5):
            idx = legal[i].nonzero().flatten()
            ps = torch.softmax(policy_logits(student, planes[i:i + 1])[0, idx].float(), 0)
            pt_ = torch.softmax(policy_logits(teacher, planes[i:i + 1])[0, idx].float(), 0)
            self.assertAlmostEqual(float(got[i].detach()), float((pt_ * (pt_.log() - ps.log())).sum().detach()),
                                   places=5)
        self.assertTrue(torch.allclose(pt.replay_kl(teacher, teacher, planes, legal), torch.zeros(5), atol=1e-6))
        got.mean().backward()  # the student gets a gradient, H3 none
        self.assertTrue(any(p.grad is not None for p in student.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))


def draft_contract():
    c = json.loads(DRAFT.read_text())
    c.pop('fixing', None)
    return {**c, 'status': 'fixed', 'sources': [], 'draft': {}}


class ContractTest(unittest.TestCase):
    def write(self, tmp, c):
        path = Path(tmp) / 'c.json'
        path.write_text(json.dumps(c))
        return path

    def test_contract_rules_and_the_experiment_id(self):
        c = draft_contract()
        with TemporaryDirectory() as tmp, mock.patch.object(p1, 'committed', return_value=True):
            path = self.write(tmp, c)
            _, sha, eid = p1.load_contract(path, ROOT)
            with self.assertRaisesRegex(ValueError, 'prepare stage is not approved'):
                p1.load_contract(path, ROOT, 'prepare')
            for approved, message in (({'prepare': False, 'train': True, 'arena': False}, 'approved before'),
                                      ({'prepare': True}, 'approved flags')):
                with self.subTest(message):
                    with self.assertRaisesRegex(ValueError, message):
                        p1.load_contract(self.write(tmp, {**c, 'approved': approved}), ROOT)
            path = self.write(tmp, {**c, 'approved': {'prepare': True, 'train': True, 'arena': False}})
            c2, sha2, eid2 = p1.load_contract(path, ROOT, 'train')
            self.assertEqual(eid2, eid)  # an approval never changes the experiment id
            self.assertNotEqual(sha2, sha)
            with self.assertRaisesRegex(ValueError, 'not fixed'):
                p1.load_contract(self.write(tmp, {**c, 'status': 'draft'}), ROOT)
            bad = {**c, 'sources': [{'path': 'scripts/h7r_p1.py', 'sha256': '0' * 64}]}
            with self.assertRaisesRegex(ValueError, 'differs from the contract'):
                p1.load_contract(self.write(tmp, bad), ROOT)
        with TemporaryDirectory() as tmp, mock.patch.object(p1, 'committed', return_value=False):
            with self.assertRaisesRegex(ValueError, 'not committed'):
                p1.load_contract(self.write(tmp, c), ROOT)

    def test_the_fixed_contract_is_the_draft_with_its_sources(self):
        fixed = json.loads((ROOT / p1.CONTRACT).read_text())
        draft = json.loads(DRAFT.read_text())
        draft.pop('fixing', None)
        strip = lambda c: {k: v for k, v in c.items() if k not in ('status', 'approved', 'sources', 'draft')}  # noqa: E731
        self.assertEqual(strip(fixed), strip(draft))
        self.assertEqual(fixed['draft']['sha256'], p1.file_sha256(DRAFT))
        with mock.patch.object(p1, 'committed', return_value=True):
            c, _, eid = p1.load_contract(ROOT / p1.CONTRACT, ROOT)
        self.assertEqual(eid, p1.experiment_sha256(fixed))
        self.assertIn('scripts/h7r_p1.py', [s['path'] for s in c['sources']])

    def test_the_fixed_values(self):
        c = draft_contract()
        self.assertEqual(c['seeds']['training'], [20261024, 20261111, 20261112])
        self.assertEqual(c['seeds']['arena_candidate_training_seed'], 20261024)
        self.assertEqual(c['seeds']['arena_games'], {'O0': 8788, 'O1': 8789, 'O2_aux': 8790})
        r = p1.replay_of(c, 'T')
        self.assertEqual((r.mu, r.batch, r.seed_offset), (0.5, 512, 1000003))
        self.assertIsNone(p1.replay_of(c, 'C'))
        cfg = p1.policy_config(c, 20261024, 'cuda')
        c2 = json.loads((ROOT / p1.C2_RECORD).read_text())
        self.assertEqual(tr.config_dict(cfg), c2['config'])  # the c2 settings exactly
        exp = c['inputs']['renjunet_cache']['expected']
        report = json.loads((ROOT / 'docs/mcts-v8-results/h3_cache_report.json').read_text())
        for s in ('train', 'val'):
            self.assertEqual((exp[s]['tensor_sha256'], exp[s]['states'], exp[s]['games']),
                             (report['splits'][s]['sha256'], report['splits'][s]['states'], report['splits'][s]['games']))


def fake_cache(tmp, overlap=False):
    from hybrid.h3_cache import code_identity, tensor_sha256
    base = guard_data(policy_rows(4))
    out = Path(tmp) / 'cache'
    out.mkdir()
    splits, expected = {}, {'format': 'h3-cache-v1', 'h2_output_sha256': 'h' * 64, **code_identity()}
    for i, s in enumerate(('train', 'val')):
        n = base['target'].shape[0]
        data = {**base, 'game_id': torch.arange(n) // 5 + (0 if (s == 'train' or overlap) else 1000)}
        torch.save(data, out / f'{s}.pt')
        splits[s] = {'states': n, 'games': len(set(data['game_id'].tolist())), 'sha256': tensor_sha256(data)}
        expected[s] = {'tensor_sha256': splits[s]['sha256'], 'states': n, 'games': splits[s]['games']}
    (out / 'manifest.json').write_text(json.dumps({**{k: expected[k] for k in expected if k not in ('train', 'val')},
                                                   'splits': splits}))
    return out, expected


class CacheTest(unittest.TestCase):
    def test_expectations_and_disjoint_games(self):
        with TemporaryDirectory() as tmp:
            cache, exp = fake_cache(tmp)
            out = p1.check_cache(cache, exp)
            self.assertEqual(set(out), {'train', 'val'})
            for change, message in ((lambda e: e['train'].update(tensor_sha256='0' * 64), 'manifest hash differs'),
                                    (lambda e: e['val'].update(states=1), 'states'),
                                    (lambda e: e.update(h2_output_sha256='x'), 'h2_output_sha256')):
                with self.subTest(message):
                    e = copy.deepcopy(exp)
                    change(e)
                    with self.assertRaisesRegex(ValueError, message):
                        p1.check_cache(cache, e)
        with TemporaryDirectory() as tmp:
            cache, exp = fake_cache(tmp, overlap=True)
            with self.assertRaisesRegex(ValueError, 'a game in both train and val'):
                p1.check_cache(cache, exp)


class ReproductionTest(unittest.TestCase):
    def test_rules(self):
        c2 = json.loads((ROOT / p1.C2_RECORD).read_text())
        exp = draft_contract()['stages']['reproduce']['expected']
        self.assertEqual(p1.reproduction_check(copy.deepcopy(c2), c2, exp), [])
        cases = {
            'eligibility': lambda r: r['evaluations'][2].update(eligible=True),
            'selected step': lambda r: r.update(best={**r['best'], 'step': 48}),
            'evaluation steps': lambda r: r['evaluations'][0].update(step=11),
            'total_steps': lambda r: r.update(total_steps=97),
            'beyond': lambda r: r['evaluations'][4]['val'].update(objective=r['evaluations'][4]['val']['objective'] + 1e-5),
            'teacher_guard': lambda r: r['teacher_guard'].update(top1=r['teacher_guard']['top1'] + 1e-5),
        }
        for message, change in cases.items():
            with self.subTest(message):
                new = copy.deepcopy(c2)
                change(new)
                self.assertTrue(any(message in m for m in p1.reproduction_check(new, c2, exp)), message)
        near = copy.deepcopy(c2)
        near['evaluations'][4]['val']['objective'] += 1e-9
        self.assertEqual(p1.reproduction_check(near, c2, exp), [])
        self.assertTrue(any('does not show' in m for m in p1.reproduction_check(c2, c2, {**exp, 'selected_step': 48})))
        self.assertEqual((c2['best']['step'], [e['eligible'] for e in c2['evaluations']]),
                         (77, [True, True, False, False, True, True, False, True, False, False]))


def link_record(c, missing=()):
    models = {}
    for seed in c['seeds']['training']:
        for arm in p1.ARMS:
            key = f'{arm}{seed}'
            models[key] = None if key in missing else {
                'policy': f'{p1.RUNS}/{arm}-{seed}/best.pt', 'value': p1.PATHS['c5_value'],
                'identity': {'policy_checkpoint_sha256': hashlib.sha256(key.encode()).hexdigest(),
                             'policy_metadata_sha256': hashlib.sha256(key.encode() + b'm').hexdigest(),
                             **c['inputs']['value_c5']}}
    return {'models': models}


class ArenaContractTest(unittest.TestCase):
    def test_the_r1_tools_accept_it(self):
        from hybrid import h7r_eval as ev
        c = draft_contract()
        body = p1.arena_contract_body(c, 'e' * 64, link_record(c), 'l' * 64, ROOT)
        # Amendment A1 replaced P1 before its Arena and took over P1's Arena seeds (its draft keeps the seeds
        # section of P1): its committed Arena contract holds exactly these seeds, so it is set aside, and every other
        # committed contract is still checked.
        successor = ROOT / 'docs/h7r/contracts/R2-P1-A1-ARENA-1.json'
        self.assertEqual(ev.contract_seeds(json.loads(successor.read_text())), ev.contract_seeds(body))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'R2-P1-ARENA-1.json'
            ev.check_contract(body, path, ROOT, check_files=False)
            path.write_text(json.dumps(body))
            root = Path(tmp) / 'root'
            (root / 'docs/h7r/contracts').mkdir(parents=True)
            for other in (ROOT / 'docs/h7r/contracts').glob('*.json'):
                if other != successor:
                    shutil.copy(other, root / 'docs/h7r/contracts' / other.name)
            for rel in {a['suite']['path'] for a in body['arms']}:  # the suite SHA-256 check reads them
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / rel, root / rel)
            ev.load_contract(path, root, check_files=False, contracts_dir=tmp)  # no other contract uses its seeds
        names = [a['name'] for a in body['arms']]
        self.assertEqual(names[:4], ['O0_T_vs_champion', 'O0_C_vs_champion', 'O1_T_vs_champion', 'O1_C_vs_champion'])
        self.assertEqual(len(names), 8)
        main = body['arms'][:4]
        self.assertEqual([(a['pairs'], a['seed'], a['pairing']) for a in main],
                         [(300, 8788, 'O0'), (300, 8788, 'O0'), (300, 8789, 'O1'), (300, 8789, 'O1')])
        self.assertTrue(all(a['opponent'] == {'model': 'champion', 'budget': 1} for a in body['arms']))
        self.assertEqual(body['models']['champion']['identity']['policy_checkpoint_sha256'],
                         c['inputs']['parent_and_teacher']['policy_checkpoint_sha256'])
        less = p1.arena_contract_body(c, 'e' * 64, link_record(c, missing=('T20261111',)), 'l' * 64, ROOT)
        self.assertEqual(len(less['arms']), 7)  # an auxiliary run without a candidate has no arm
        self.assertEqual(body['p1']['experiment_sha256'], 'e' * 64)


class JudgeTest(unittest.TestCase):
    def strata(self, t, c, n=40):
        return {s: [(t, c, i // 4) for i in range(n)] for s in p1.SUITES}

    def test_readings(self):
        mixed = {s: [(1.0 if i % 2 else 0.5, 0.5, i // 4) for i in range(40)] for s in p1.SUITES}
        b = p1.stratified_bounds(mixed, 2000, 7, 0.025)
        self.assertEqual(p1.labels(b), {'P': 'STRONGER', 'D': 'BETTER_THAN_CONTROL'})
        self.assertAlmostEqual(b['P']['estimate'], 0.25)
        even = {s: [((0.0, 1.0)[i % 2], (0.0, 1.0)[(i // 2) % 2], i) for i in range(40)] for s in p1.SUITES}
        lab = p1.labels(p1.stratified_bounds(even, 2000, 7, 0.025))
        self.assertEqual(lab, {'P': 'NOT_ESTABLISHED', 'D': 'NOT_ESTABLISHED'})
        self.assertIn('not established', p1.interpret(lab))
        self.assertIn('regression mitigated only', p1.interpret({'P': 'NOT_ESTABLISHED', 'D': 'BETTER_THAN_CONTROL'}))
        self.assertIn('confirmation stage', p1.interpret({'P': 'STRONGER', 'D': 'NOT_ESTABLISHED'}))
        worse = {s: [(0.0 if i % 2 else 0.5, 0.5, i) for i in range(40)] for s in p1.SUITES}
        self.assertEqual(p1.labels(p1.stratified_bounds(worse, 2000, 7, 0.025)), {'P': 'WEAKER', 'D': 'WORSE_THAN_CONTROL'})

    def test_equal_suite_weights_and_class_units(self):
        lopsided = {'O0': [(1.0, 0.5, i) for i in range(10)], 'O1': [(0.5, 0.5, i) for i in range(90)]}
        b = p1.stratified_bounds(lopsided, 500, 3, 0.025)
        self.assertAlmostEqual(b['P']['estimate'], 0.25)  # 0.5 x 0.5 + 0.5 x 0.0, not the pooled 0.05
        cls = p1.stratified_bounds(self.strata(1.0, 0.5), 200, 3, 0.025, unit='class')
        self.assertEqual(cls['units'], {'O0': 10, 'O1': 10})

    def test_paired_openings(self):
        pair = lambda k, s: {'opening_id': k, 'pair_score': s, 'class_id': k}  # noqa: E731
        tables = {f'{s}_{a}_vs_champion': {'pairs': [pair(k, 0.5) for k in range(3)]} for s in p1.SUITES for a in 'TC'}
        self.assertEqual(len(p1.paired_strata(tables)['O0']), 3)
        tables['O1_C_vs_champion']['pairs'].pop()
        with self.assertRaisesRegex(ValueError, 'not played on the same openings'):
            p1.paired_strata(tables)



def git(root, *args):
    import subprocess
    return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


class RealGitTest(unittest.TestCase):
    """The stages on a real temporary git repository with the repository's real .gitignore: approval commits,
    records written by the tool, copies into seed-records, staging and commits; stand-in inputs (tiny rows, small
    networks, a fake cache) under the ignored runs/ and data/; nothing about git is mocked."""

    SOURCE = 'src/hybrid/h7r_p1_train.py'

    def setUp(self):
        import gzip
        import shutil
        from analysis import tb_v3
        from model.checkpoint import save_checkpoint
        from model.config import ModelConfig
        from tests.test_h7c_train import source_of

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
        c['training'] = {**c['training'], 'batch_size': 16, 'epochs': 1.0, 'evaluations': 3, 'guard_drop': 1.0}
        c['arms']['T']['replay'] = {**c['arms']['T']['replay'], 'batch': 8}
        c['approved'] = {'prepare': False, 'train': False, 'arena': False}
        frozen = mock.patch.dict(tb_v3.MODELS, {**tb_v3.MODELS, **c['inputs']['parent_and_teacher']})
        frozen.start()
        self.addCleanup(frozen.stop)
        # the "c2" record: the unchanged h7c trainer on the same inputs
        checked = p1.check_inputs(c, root)
        result, inputs, cfg_, _ = p1.run_training(c, 'C', 20261024, 'cpu', checked, root, log=QUIET)
        old = tr.train_policy(cfg_, small_model(), small_model(), tr.policy_tensors(checked['rows']['train']),
                              tr.policy_tensors(checked['rows']['val']), checked['cache']['val'], guard_eval, log=QUIET)
        self.assertEqual(result['record'], old['record'])
        rec = old['record']
        (root / p1.C2_RECORD).parent.mkdir(parents=True)
        (root / p1.C2_RECORD).write_text(json.dumps({'config': tr.config_dict(cfg_), 'inputs': inputs, **rec}))
        c['stages']['reproduce']['expected'] = {
            'total_steps': rec['total_steps'], 'steps_per_epoch': rec['steps_per_epoch'],
            'warmup_steps': rec['warmup_steps'], 'evaluation_steps': [e['step'] for e in rec['evaluations']],
            'eligibility': [e['eligible'] for e in rec['evaluations']], 'selected_step': rec['best']['step']}
        self.c = c
        self.write_contract(c)
        git(root, 'add', '-A')
        git(root, 'commit', '-qm', 'contract')
        self.assertEqual(git(root, 'ls-files', 'runs', 'data'), '')  # inputs stay git-ignored

    def write_contract(self, c, indent=None):
        (self.root / p1.CONTRACT).parent.mkdir(parents=True, exist_ok=True)
        (self.root / p1.CONTRACT).write_text(json.dumps(c, indent=indent))

    def approve(self, stage):
        c = json.loads((self.root / p1.CONTRACT).read_text())
        c['approved'][stage] = True
        self.write_contract(c, indent=1)
        git(self.root, 'commit', '-qam', f'approve {stage}')

    def commit(self, *paths):
        git(self.root, 'add', *paths)
        git(self.root, 'commit', '-qm', 'records')
        for p in paths:
            self.assertTrue(git(self.root, 'ls-files', '--error-unmatch', p))

    def ns(self, **k):
        import argparse
        return argparse.Namespace(contract=None, device='cpu', **k)

    def through_training(self, seeds=None):
        import shutil
        self.approve('prepare')
        self.assertTrue(p1.prepare(self.ns(), self.root)['passed'])
        self.commit(f'{p1.D}/prepare.json')
        self.approve('train')
        rep = p1.reproduce(self.ns(), self.root, log=QUIET)
        self.assertEqual(rep['mismatches'], [])
        self.commit(f'{p1.D}/reproduce.json')
        for seed in seeds or self.c['seeds']['training']:
            for arm in p1.ARMS:
                out = p1.train(self.ns(arm=arm, seed=seed), self.root, log=QUIET)
                self.assertEqual((out['experiment_sha256'], out['verdict']), (rep['experiment_sha256'], 'CANDIDATE'))
                dst = self.root / p1.run_record_path(arm, seed)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(self.root / p1.RUNS / f'{arm}-{seed}' / 'record.json', dst)
                self.commit(p1.run_record_path(arm, seed))  # seed-records is not git-ignored
        return rep

    def test_the_stages_on_real_git(self):
        rep = self.through_training()
        eid = rep['experiment_sha256']
        old = self.root / f'{p1.D}/runs/C-20261024/record.json'  # the old path: git-ignored, never committed
        old.parent.mkdir(parents=True)
        old.write_text('{}')
        self.assertTrue(git(self.root, 'check-ignore', str(old.relative_to(self.root))))
        old.unlink()
        lk = p1.link(self.ns(), self.root)
        self.assertEqual((lk['candidate_seed'], lk['experiment_sha256']), (20261024, eid))
        self.commit(f'{p1.D}/link.json')
        self.approve('arena')  # the contract file changes, the experiment does not
        body = p1.arena_contract(self.ns(), self.root)
        self.commit(p1.ARENA_CONTRACT)
        t = json.loads((self.root / p1.run_record_path('T', 20261024)).read_text())
        self.assertEqual(body['models']['T']['identity']['policy_checkpoint_sha256'], t['candidate']['checkpoint_sha256'])
        self.assertEqual(len(body['arms']), 8)
        with self.assertRaisesRegex(ValueError, 'already holds a run'):
            p1.train(self.ns(arm='T', seed=20261024), self.root, log=QUIET)
        # judge: the Arena contract must be exactly the expected one
        arena = json.loads((self.root / p1.ARENA_CONTRACT).read_text())
        tampered = {
            'fewer pairs': lambda a: a['arms'][0].update(pairs=1),
            'another model': lambda a: a['models']['T']['identity'].update(policy_checkpoint_sha256='0' * 64),
            'a missing arm': lambda a: a['arms'].pop(1),
            'an added arm': lambda a: a['arms'].append({**a['arms'][0], 'name': 'extra'}),
            'another budget': lambda a: a['arms'][2]['tested'].update(budget=8),
            'other statistics': lambda a: a['stats']['bootstrap'].update(samples=1000),
            'another link': lambda a: a['p1'].update(link_sha256='0' * 64),
        }
        for what, change in tampered.items():
            with self.subTest(what):
                bad = copy.deepcopy(arena)
                change(bad)
                (self.root / p1.ARENA_CONTRACT).write_text(json.dumps(bad))
                self.commit(p1.ARENA_CONTRACT)
                with self.assertRaisesRegex(ValueError, 'not the one the P1 contract and the verified link give'):
                    p1.judge(self.ns(arms_dir=str(self.root / 'arms')), self.root)
        (self.root / p1.ARENA_CONTRACT).write_text(json.dumps(arena))
        self.commit(p1.ARENA_CONTRACT)
        with self.assertRaises((ValueError, OSError)):  # the right contract, but no arm records yet
            p1.judge(self.ns(arms_dir=str(self.root / 'arms')), self.root)
        self.assertFalse((self.root / p1.D / 'judge.json').exists())
        # verified arm records (stand-ins for h7r_eval.load_arm): a short arm is refused, complete ones are judged
        arms_dir = self.root / 'arms'
        arms_dir.mkdir()
        sizes = {a['name']: a['pairs'] for a in arena['arms']}
        for name in sizes:
            (arms_dir / f'{name}.json').write_text('{}')

        def loaded(short=None):
            def load(path, contract, sha, root):
                name = Path(path).stem
                n = sizes[name] - (1 if name == short else 0)
                score = 1.0 if '_T_' in name else 0.5
                pairs = [{'opening_id': i, 'pair_score': score, 'class_id': i // 4} for i in range(n)]
                return {'table': {'valid': True, 'pairs': pairs}, 'summary': {'pairs': n}}
            return load
        import h7r_eval as cli
        with mock.patch.object(cli, 'load_arm', side_effect=loaded(short='O1_C_vs_champion')):
            with self.assertRaisesRegex(ValueError, 'the sample is never reduced'):
                p1.judge(self.ns(arms_dir=str(arms_dir)), self.root)
        with mock.patch.object(cli, 'load_arm', side_effect=loaded()):
            j = p1.judge(self.ns(arms_dir=str(arms_dir)), self.root)
        self.assertEqual(j['readings'], {'P': 'STRONGER', 'D': 'BETTER_THAN_CONTROL'})
        self.assertEqual(j['experiment_sha256'], eid)
        self.commit(f'{p1.D}/judge.json')

    def test_inputs_changed_after_prepare_are_refused(self):
        from hybrid.h3_cache import tensor_sha256
        self.approve('prepare')
        p1.prepare(self.ns(), self.root)
        self.commit(f'{p1.D}/prepare.json')
        self.approve('train')
        cache = self.root / p1.PATHS['cache']
        data = torch.load(cache / 'train.pt', weights_only=True)
        data['black'] = data['black'].roll(1, 0)  # other stones (every fake target is the same move)
        self.assertNotEqual(tensor_sha256(data), self.c['inputs']['renjunet_cache']['expected']['train']['tensor_sha256'])
        torch.save(data, cache / 'train.pt')
        manifest = json.loads((cache / 'manifest.json').read_text())
        manifest['splits']['train']['sha256'] = tensor_sha256(data)  # the cache and its own manifest together
        (cache / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'cache train: manifest hash differs'):
            p1.reproduce(self.ns(), self.root, log=QUIET)
        self.assertFalse((self.root / p1.D / 'reproduce.json').exists())

    def test_a_change_during_the_run_publishes_nothing(self):
        import shutil
        self.approve('prepare')
        p1.prepare(self.ns(), self.root)
        self.commit(f'{p1.D}/prepare.json')
        self.approve('train')
        p1.reproduce(self.ns(), self.root, log=QUIET)
        self.commit(f'{p1.D}/reproduce.json')
        real = p1.run_training
        changes = {
            'HEAD': lambda: git(self.root, 'commit', '-q', '--allow-empty', '-m', 'moved'),
            'dirty': lambda: (self.root / self.SOURCE).write_text('# edited\n'),
            'input': lambda: (self.root / p1.c5_meta_path()).write_text((self.root / p1.c5_meta_path()).read_text() + ' '),
        }
        import model.checkpoint as mc
        real_save = mc.save_checkpoint

        def save_then_move(*a, **k):
            real_save(*a, **k)
            git(self.root, 'commit', '-q', '--allow-empty', '-m', 'moved while saving')
        with mock.patch.object(mc, 'save_checkpoint', side_effect=save_then_move):
            with self.assertRaisesRegex(ValueError, 'changed during the run'):
                p1.train(self.ns(arm='C', seed=20261112), self.root, log=QUIET)
        out = self.root / p1.RUNS / 'C-20261112'
        self.assertEqual(sorted(x.name for x in out.iterdir()), [])  # no checkpoint, metadata, record or temporary
        shutil.rmtree(out)
        for what, change in changes.items():
            with self.subTest(what):
                def during(*a, **k):
                    out = real(*a, **k)
                    change()
                    return out
                with mock.patch.object(p1, 'run_training', side_effect=during):
                    with self.assertRaisesRegex(ValueError, 'changed during the run'):
                        p1.train(self.ns(arm='C', seed=20261111), self.root, log=QUIET)
                out = self.root / p1.RUNS / 'C-20261111'
                self.assertFalse((out / 'record.json').exists())
                self.assertFalse((out / 'best.pt').exists())
                git(self.root, 'checkout', '--', self.SOURCE)
                meta = json.loads((self.root / p1.c5_meta_path()).read_text())
                (self.root / p1.c5_meta_path()).write_text(json.dumps(meta))
                if out.exists():
                    shutil.rmtree(out)

    def test_the_train_gate_rejudges_the_reproduction(self):
        self.approve('prepare')
        p1.prepare(self.ns(), self.root)
        self.commit(f'{p1.D}/prepare.json')
        self.approve('train')
        p1.reproduce(self.ns(), self.root, log=QUIET)
        rep_path = self.root / p1.D / 'reproduce.json'
        good = json.loads(rep_path.read_text())
        cases = {
            'a changed run with passed kept true': lambda r: r['run']['best'].update(step=1),
            'another seed': lambda r: r.update(seed=20261111),
            'another c2 record': lambda r: r.update(c2_record_sha256='0' * 64),
            'other settings': lambda r: r['settings'].update(learning_rate=1.0),
        }
        for what, change in cases.items():
            with self.subTest(what):
                bad = copy.deepcopy(good)
                change(bad)
                rep_path.write_text(json.dumps(bad))
                self.commit(f'{p1.D}/reproduce.json')
                with self.assertRaisesRegex(ValueError, 'reproduction'):
                    p1.train(self.ns(arm='C', seed=20261024), self.root, log=QUIET)
        rep_path.write_text(json.dumps(good))
        self.commit(f'{p1.D}/reproduce.json')
        self.assertEqual(p1.train(self.ns(arm='C', seed=20261024), self.root, log=QUIET)['verdict'], 'CANDIDATE')


STUB_P1 = """import json, os, sys
from pathlib import Path
D = 'docs/h7r/r2/R2-P1-POLICY-REPLAY-1'
R = 'runs/h7r/R2-P1-POLICY-REPLAY-1'
fail = set(filter(None, os.environ.get('P1_FAIL', '').split(',')))
forbid = set(filter(None, os.environ.get('P1_FORBID', '').split(',')))
cmd = sys.argv[1]
def out(path, body):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(body))
if cmd == 'train':
    run = f"{sys.argv[sys.argv.index('--arm') + 1]}-{sys.argv[sys.argv.index('--seed') + 1]}"
    if run in forbid:
        print('STOP: this run must not be trained again'); sys.exit(9)
    if run in fail:
        print('STOP: ValueError: stand-in failure'); sys.exit(1)
    out(f'{R}/{run}/record.json', {'run': run}); print('trained', run); sys.exit(0)
if cmd in fail:
    print(f'STOP: ValueError: stand-in {cmd} failure')
    if cmd == 'reproduce':
        out(f'{D}/reproduce.json', {'passed': False}); sys.exit(2)
    sys.exit(1)
out(f'{D}/{cmd}.json', {'stage': cmd}); print('ok', cmd)
"""
STUB_EVAL = """import json, os, subprocess, sys
from pathlib import Path
arg = lambda k: sys.argv[sys.argv.index(k) + 1]
arm, jl, out = arg('--arm'), Path(arg('--jsonl')), Path(arg('--output'))
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if jl.exists():  # like the real arm signature: a journal resumes only at the HEAD it was started at
    if json.loads(jl.read_text().splitlines()[0])['head'] != head:
        print('STOP: the journal is of another signature (HEAD)'); sys.exit(3)
jl.parent.mkdir(parents=True, exist_ok=True)
with jl.open('a') as f:
    f.write(json.dumps({'head': head}) + chr(10))
if arm in os.environ.get('EVAL_FAIL', '').split(','):
    print('interrupted'); sys.exit(1)
out.write_text(json.dumps({'arm': arm, 'head': head}))
print('played', arm)
"""


class BlockTest(unittest.TestCase):
    """The desktop blocks of the draft (section 11) run by bash on a real temporary repository with a bare origin,
    the repository's real .gitignore and the real record tool; the P1 and Arena commands are stand-ins."""

    def setUp(self):
        import re
        import shutil
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
        (w / 'scripts/h7r_p1.py').write_text(STUB_P1)
        (w / 'scripts/h7r_eval.py').write_text(STUB_EVAL)
        (w / '.venv/bin').mkdir(parents=True)
        (w / '.venv/bin/activate').write_text('')
        arena = {'arms': [{'name': n} for n in ('O0_T_vs_champion', 'O0_C_vs_champion')]}
        (w / 'docs/h7r/contracts').mkdir(parents=True)
        (w / 'docs/h7r/contracts/R2-P1-ARENA-1.json').write_text(json.dumps(arena))
        git(w, 'add', '-A')
        git(w, 'commit', '-qm', 'init')
        git(w, 'remote', 'add', 'origin', str(self.origin))
        git(w, 'push', '-q', 'origin', 'feat/h7-c1')
        doc = (ROOT / 'docs/h7r/r2/R2-P1-POLICY-REPLAY-1-draft.md').read_text()
        sec = doc.split('\n## 11. 실행 절차', 1)[1]
        blocks = re.findall(r'```bash\n(.*?)```', sec, re.S)
        self.assertEqual(len(blocks), 4)
        self.blocks = dict(zip(('p1-1', 'p1-F', 'p1-2', 'p1-4'), (b.replace('~/renju-h6', str(w)) for b in blocks)))
        self.D = 'docs/h7r/r2/R2-P1-POLICY-REPLAY-1'

    def bash(self, name, **env):
        import os
        import subprocess
        e = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', **env}
        r = subprocess.run(['bash', '-c', self.blocks[name]], cwd=self.work, capture_output=True, text=True, env=e)
        return r.returncode, r.stdout + r.stderr

    def tracked(self, rel):
        return bool(git(self.work, 'ls-files', rel))

    def test_prepare_pass_fail_and_rerun(self):
        rc, out = self.bash('p1-1', P1_FAIL='prepare')
        self.assertEqual(rc, 1, out)
        logs = git(self.work, 'ls-files', f'{self.D}/stage-logs').splitlines()
        self.assertEqual(len(logs), 2)  # the failed attempt's log and manifest are committed
        log = next(l for l in logs if l.endswith('.log'))
        self.assertIn('__EXIT__ 1', (self.work / log).read_text())
        self.assertIn('prepare FAIL (exit 1)', git(self.work, 'log', '-1', '--format=%s'))
        self.assertFalse(self.tracked(f'{self.D}/prepare.json'))
        import time
        time.sleep(1.1)  # a new log name
        rc, out = self.bash('p1-1')
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.tracked(f'{self.D}/prepare.json'))
        self.assertEqual(len(git(self.work, 'ls-files', f'{self.D}/stage-logs').splitlines()), 4)  # nothing overwritten
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), git(self.origin, 'rev-parse', 'feat/h7-c1'))
        rc, out = self.bash('p1-1')
        self.assertEqual(rc, 1)
        self.assertIn('already recorded', out)

    def test_recovery_after_an_interruption(self):
        import subprocess
        L = self.work / 'runs/h7r/R2-P1-POLICY-REPLAY-1/logs'
        L.mkdir(parents=True)
        (L / 'prepare-20261110-000000.log').write_text('ok prepare\n')  # stopped before its records commit
        subprocess.run(['python', 'scripts/h7r_p1.py', 'prepare'], cwd=self.work, check=True, capture_output=True)
        rc, out = self.bash('p1-1')
        self.assertEqual(rc, 1)
        self.assertIn('run p1-F', out)
        rc, out = self.bash('p1-F')
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.tracked(f'{self.D}/prepare.json'))
        self.assertTrue(self.tracked(f'{self.D}/stage-logs/prepare-20261110-000000.log'))
        rc, out = self.bash('p1-F')
        self.assertEqual(rc, 0, out)
        self.assertIn('nothing left to record', out)

    def test_recovery_refuses_while_a_p1_command_runs(self):
        import subprocess
        import time
        alias = self.work / 'scripts/h7r_p1.py'
        alias.write_text('import time; time.sleep(30)\n')
        proc = subprocess.Popen(['python', 'scripts/h7r_p1.py', 'train'], cwd=self.work)
        self.addCleanup(lambda: (proc.kill(), proc.wait()))
        for _ in range(200):  # until the stand-in is visible as a process (not a fixed sleep)
            if subprocess.run(['pgrep', '-f', 'scripts/h7r_p[1][.]py train'], capture_output=True).returncode == 0:
                break
            time.sleep(0.05)
        rc, out = self.bash('p1-F')
        self.assertEqual(rc, 1, out)
        self.assertIn('still running', out)

    def test_training_stops_records_and_resumes_without_retraining(self):
        import time
        rc, out = self.bash('p1-2', P1_FAIL='T-20261111')
        self.assertEqual(rc, 1, out)
        self.assertIn('run T 20261111 failed', out)
        self.assertTrue(self.tracked(f'{self.D}/reproduce.json'))
        done = git(self.work, 'ls-files', f'{self.D}/seed-records').splitlines()
        self.assertEqual(len(done), 3)  # C/T 20261024 and C 20261111, each committed alone
        failed = [l for l in git(self.work, 'ls-files', f'{self.D}/stage-logs').splitlines() if 'train-T-20261111' in l]
        self.assertEqual(len(failed), 2)  # the failed run's log and manifest
        # a run that finished but was not committed is copied, not trained again
        R = self.work / 'runs/h7r/R2-P1-POLICY-REPLAY-1'
        (R / 'C-20261112').mkdir(parents=True)
        (R / 'C-20261112/record.json').write_text('{"run": "C-20261112"}')
        time.sleep(1.1)
        rc, out = self.bash('p1-2', P1_FORBID='C-20261024,T-20261024,C-20261111,C-20261112')
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(git(self.work, 'ls-files', f'{self.D}/seed-records').splitlines()), 6)
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), git(self.origin, 'rev-parse', 'feat/h7-c1'))

    def test_a_failed_reproduction_stops_before_training(self):
        rc, out = self.bash('p1-2', P1_FAIL='reproduce')
        self.assertEqual(rc, 1, out)
        self.assertIn('reproduction gate did not pass', out)
        self.assertTrue(self.tracked(f'{self.D}/reproduce.json'))
        self.assertEqual(git(self.work, 'ls-files', f'{self.D}/seed-records'), '')

    def test_arena_failure_keeps_head_and_resumes(self):
        import time
        head = git(self.work, 'rev-parse', 'HEAD')
        rc, out = self.bash('p1-4', EVAL_FAIL='O0_C_vs_champion')
        self.assertEqual(rc, 1, out)
        self.assertIn('nothing committed, HEAD kept', out)
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), head)  # no commit while an arm is unfinished
        rc, out = self.bash('p1-F')
        self.assertEqual(rc, 1, out)
        self.assertIn('the Arena is unfinished', out)
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), head)
        time.sleep(1.1)
        rc, out = self.bash('p1-4')
        self.assertEqual(rc, 0, out)
        self.assertIn('resuming at', out)
        self.assertTrue(self.tracked(f'{self.D}/judge.json'))
        logs = [l for l in git(self.work, 'ls-files', f'{self.D}/stage-logs').splitlines() if l.endswith('.log')]
        self.assertEqual(len([l for l in logs if 'play-O0_C' in l]), 2)  # the failed attempt's log too
        self.assertEqual(len(logs), 4)

    def test_arena_refuses_a_moved_head(self):
        rc, out = self.bash('p1-4', EVAL_FAIL='O0_T_vs_champion')
        self.assertEqual(rc, 1, out)
        git(self.work, 'commit', '-q', '--allow-empty', '-m', 'moved')
        rc, out = self.bash('p1-4')
        self.assertEqual(rc, 1, out)
        self.assertIn('HEAD moved since the Arena started', out)
        self.assertFalse(self.tracked(f'{self.D}/judge.json'))

    def test_arena_block(self):
        rc, out = self.bash('p1-4')
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.tracked(f'{self.D}/judge.json'))
        logs = git(self.work, 'ls-files', f'{self.D}/stage-logs').splitlines()
        self.assertEqual(len(logs), 6)  # two arms and the judge, each a log and a manifest
        rc, out = self.bash('p1-4')
        self.assertEqual(rc, 1)
        self.assertIn('already recorded', out)

if __name__ == '__main__':
    unittest.main()
