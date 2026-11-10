"""R2-P1-POLICY-REPLAY-1: the trainer (replay off is h7c_train.train_policy; the replay term, its generator and its
KL formula) and the tool (contract and experiment id, cache checks, the reproduction rules, the Arena contract the
R1 tools accept, the stratified paired judge and its readings)."""
import copy
import hashlib
import json
from pathlib import Path
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
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'R2-P1-ARENA-1.json'
            ev.check_contract(body, path, ROOT, check_files=False)
            path.write_text(json.dumps(body))
            ev.load_contract(path, ROOT, check_files=False, contracts_dir=tmp)  # no other contract uses its seeds
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



class EndToEndTest(unittest.TestCase):
    """prepare -> reproduce -> train (6 runs) -> link -> arena-contract on a repository-shaped temporary tree with
    stand-in inputs (tiny rows, small networks, a fake cache), through the real code paths."""

    def build(self, root):
        import gzip
        import shutil
        from model.checkpoint import save_checkpoint
        from model.config import ModelConfig
        from tests.test_h7c_train import source_of

        cfg = ModelConfig()
        h3 = root / p1.PATHS['h3']
        h3.parent.mkdir(parents=True)
        save_checkpoint(h3, small_model())
        h3.with_suffix('.json').write_text(json.dumps({'policy_trained': True, 'value_trained': False,
                                                        'config': {'channels': cfg.channels, 'blocks': cfg.blocks}}))
        value = root / p1.PATHS['c5_value']
        value.parent.mkdir(parents=True)
        save_checkpoint(value, small_model(5))
        value.with_name('best.json').write_text(json.dumps({'stage': 'H6-2a', 'trunk_frozen': True,
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
        c['inputs']['cycle_rows'] = {'path': p1.PATHS['cycle_rows'], 'sha256': p1.file_sha256(path),
                                     'train_rows': sum(r['split'] == 'train' for r in rows),
                                     'val_rows': sum(r['split'] == 'val' for r in rows)}
        c['inputs']['parent_and_teacher'] = {'policy_checkpoint_sha256': p1.file_sha256(h3),
                                             'policy_metadata_sha256': p1.file_sha256(h3.with_suffix('.json'))}
        c['inputs']['value_c5'] = {'value_checkpoint_sha256': p1.file_sha256(value),
                                   'value_metadata_sha256': p1.file_sha256(value.with_name('best.json'))}
        c['inputs']['renjunet_cache']['expected'] = expected
        c['training'] = {**c['training'], 'batch_size': 16, 'epochs': 1.0, 'evaluations': 3, 'guard_drop': 1.0}
        c['arms']['T']['replay'] = {**c['arms']['T']['replay'], 'batch': 8}
        c['approved'] = {'prepare': True, 'train': True, 'arena': True}
        # the "c2" record: the unchanged h7c trainer on the same inputs
        result, inputs, cfg_, _ = p1.run_training(c, 'C', 20261024, 'cpu', root, log=QUIET)
        old = tr.train_policy(cfg_, small_model(), small_model(), tr.policy_tensors([r for r in rows if r['split'] == 'train']),
                              tr.policy_tensors([r for r in rows if r['split'] == 'val']),
                              p1.check_cache(target, expected)['val'], guard_eval, log=QUIET)
        rec = old['record']
        c2 = {'config': tr.config_dict(cfg_), 'inputs': inputs, **rec}
        (root / p1.C2_RECORD).parent.mkdir(parents=True)
        (root / p1.C2_RECORD).write_text(json.dumps(c2))
        c['stages']['reproduce']['expected'] = {
            'total_steps': rec['total_steps'], 'steps_per_epoch': rec['steps_per_epoch'],
            'warmup_steps': rec['warmup_steps'], 'evaluation_steps': [e['step'] for e in rec['evaluations']],
            'eligibility': [e['eligible'] for e in rec['evaluations']], 'selected_step': rec['best']['step']}
        (root / p1.CONTRACT).parent.mkdir(parents=True, exist_ok=True)
        (root / p1.CONTRACT).write_text(json.dumps(c))
        return c

    def test_the_stages_link_through_the_experiment_id(self):
        import argparse
        import shutil
        from analysis import tb_v3

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self.build(root)
            frozen = {**tb_v3.MODELS, **c['inputs']['parent_and_teacher']}
            ns = lambda **k: argparse.Namespace(contract=None, device='cpu', **k)  # noqa: E731
            with mock.patch.object(p1, 'committed', return_value=True), \
                    mock.patch.object(p1, 'git_state', return_value={'commit': 'x', 'dirty': False}), \
                    mock.patch.dict(tb_v3.MODELS, frozen):
                self.assertTrue(p1.prepare(ns(), root)['passed'])
                rep = p1.reproduce(ns(), root, log=QUIET)
                self.assertEqual(rep['mismatches'], [])
                eid = rep['experiment_sha256']
                for seed in c['seeds']['training']:
                    for arm in p1.ARMS:
                        out = p1.train(ns(arm=arm, seed=seed), root, log=QUIET)
                        self.assertEqual((out['experiment_sha256'], out['verdict']), (eid, 'CANDIDATE'))
                        dst = root / p1.run_record_path(arm, seed)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(root / p1.RUNS / f'{arm}-{seed}' / 'record.json', dst)
                t = json.loads((root / p1.run_record_path('T', 20261024)).read_text())
                cc = json.loads((root / p1.run_record_path('C', 20261024)).read_text())
                self.assertIsNotNone(t['p1']['replay'])
                self.assertIsNone(cc['p1']['replay'])
                self.assertNotEqual(t['candidate'], cc['candidate'])
                with self.assertRaisesRegex(ValueError, 'already holds a run'):
                    p1.train(ns(arm='T', seed=20261024), root, log=QUIET)
                # an approval commit changes the contract file, not the experiment: the stages still link
                (root / p1.CONTRACT).write_text(json.dumps({**c, 'approved': {'prepare': True, 'train': True,
                                                                              'arena': True}}, indent=1))
                lk = p1.link(ns(), root)
                self.assertEqual(lk['candidate_seed'], 20261024)
                body = p1.arena_contract(ns(), root)
                self.assertEqual(body['models']['T']['identity']['policy_checkpoint_sha256'],
                                 t['candidate']['checkpoint_sha256'])
                self.assertEqual(len(body['arms']), 8)
                # a reproduction that misses stops training
                (root / p1.D / 'reproduce.json').unlink()
                bad = json.loads((root / p1.C2_RECORD).read_text())
                bad['evaluations'][0]['val']['objective'] += 1e-3
                (root / p1.C2_RECORD).write_text(json.dumps(bad))
                rep2 = p1.reproduce(ns(), root, log=QUIET)
                self.assertFalse(rep2['passed'])
                with self.assertRaisesRegex(ValueError, 'reproduction gate did not pass'):
                    p1.train(ns(arm='C', seed=20261111), root, log=QUIET)

if __name__ == '__main__':
    unittest.main()
