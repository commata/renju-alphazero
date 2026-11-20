"""R2-S400-PRESERVE-1 (``scripts/h7r_preserve.py``, ``scripts/h7r_dev_suites_2.py``, ``hybrid.h7r_suite_seeds_dev2``): the new
suites (registry, exclusion, rebuild, the O1-dev-a / b check kept); the contract (pins, mu, approvals); on a temporary
repository with the R2-S400-TRAIN fixture: prepare, the exact reproduction gate (mu 0.5), the mu 2.0 runs (only the
coefficient changes: the same draws and fingerprints), verify-run, the v2 Arena contract (new suites, seeds 9103 / 9104),
the four arms with an in-process fake game and the D'' / P'' judge; the pairing and reading rules; the desktop blocks
under real bash. Training and the engine games are stand-ins; the rules and records are the tools'."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import h7r_dev_suites as ds
import h7r_dev_suites_2 as ds2
import h7r_eval as cli
import h7r_p1 as p1
import h7r_p1a1 as a1
import h7r_preserve as pv
import h7r_s400_train as st
from hybrid import h7r_suite_seeds_dev2 as seeds2
from tests.test_h7r_s400_train import QUIET, RepoBase, git

ROOT = Path(__file__).resolve().parents[1]


# --- the new suites --------------------------------------------------------------------------------------------------

def before_the_suites(root):
    """The copied checkout as it was before O1-dev-c / d: the two suites, their record and every later v2 contract that
    plays them (the PRESERVE and R3 Arenas; they name suite files that do not exist yet) are removed."""
    root = Path(root)
    suites = {ds2.suite_path('c'), ds2.suite_path('d')}
    for rel in (ds2.RECORD, *suites):
        (root / rel).unlink(missing_ok=True)
    for path in sorted((root / 'docs/h7r/contracts-v2').glob('*.json')):
        arms = json.loads(path.read_text(encoding='utf-8')).get('arms') or []
        if any((a.get('suite') or {}).get('path') in suites for a in arms):
            path.unlink()


class SuiteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.root = Path(self.tmp.name)
        git(root, 'init', '-q')
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(root, 'config', k, v)
        for rel in ('docs/h7r/suites', 'docs/h7r/contracts', 'docs/h7r/contracts-v2'):
            shutil.copytree(ROOT / rel, root / rel)
        before_the_suites(root)
        git(root, 'add', '-A')
        git(root, 'commit', '-qm', 'sources')

    def test_an_unregistered_seed_is_refused(self):
        with mock.patch.object(ds2, 'SUITES', (('c', 202611131), ('d', 202611192))):
            with self.assertRaisesRegex(ValueError, 'not a registered h7r-suite-seeds-dev2 suite seed'):
                ds2.build_suites(ds2.sources(self.root))


# --- the run on the R2-S400-TRAIN fixture ----------------------------------------------------------------------------

def stand_in(root, rel, text, meta_text=None, kind='policy'):
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = path.with_suffix('.json') if kind == 'policy' else path.with_name('best.json')
    path.write_text(text)
    meta.write_text(meta_text if meta_text is not None else text + ' metadata')
    return p1.file_sha256(path), p1.file_sha256(meta)


class PreserveBase(RepoBase):
    def setUp(self):
        super().setUp()
        root = self.root
        c4 = json.loads((root / st.CONTRACT).read_text())
        h3 = stand_in(root, p1.PATHS['h3'], 'H3 policy', json.dumps({'config': {'channels': 64, 'blocks': 4}}))
        c4['inputs']['parent_and_teacher'] = {'policy_checkpoint_sha256': h3[0], 'policy_metadata_sha256': h3[1]}
        v = stand_in(root, p1.PATHS['c5_value'], 'c5 value', kind='value')
        c4['inputs']['value_c5'] = {'value_checkpoint_sha256': v[0], 'value_metadata_sha256': v[1]}
        self.write_contract(c4)
        git(root, 'commit', '-qam', 'stand-in model SHA-256')
        self.eid4 = self.eid()
        rows = root / st.RUNS / 'policy_rows_s400.jsonl.gz'
        rows.parent.mkdir(parents=True, exist_ok=True)
        rows.write_bytes(b'stand-in candidate rows')
        self.rows_rec = {'experiment_sha256': self.eid4, 'candidate_rows_sha256': p1.file_sha256(rows),
                         'c2_rows_sha256': 'c' * 64, 's400_records_sha256': 's' * 64}
        self.commit_record(f'{st.D}/rows.json', self.rows_rec)
        self.commit_record(f'{st.D}/gate.json', {'experiment_sha256': self.eid4, 'passed': True, 'differences': [],
                                                 'run': st.gate_view(self.a1_rec), 'settings': self.a1_rec['settings']})
        for s in pv.SEEDS:
            run = self.preserve_run(s, pv.BASE_MU)
            files = stand_in(root, f'{st.RUNS}/S-{s}/best.pt', f'S {s} checkpoint')
            rec = {'format': 'h7r-s400-train-run-v1', 'experiment_sha256': self.eid4, 'arm': 'S', 'seed': s,
                   **st.gate_view(self.view_source(run)), 'inputs': pv.run_inputs(run, self.rows_rec),
                   'candidate': {'checkpoint_sha256': files[0], 'metadata_sha256': files[1]}}
            self.commit_record(st.run_record_path(s), rec)
        shutil.copytree(ROOT / 'docs/h7r/contracts-v2', root / 'docs/h7r/contracts-v2')  # suite sources
        before_the_suites(root)
        self.c = pv.contract_body(root, sources=())
        self.write_preserve(self.c)
        git(root, 'add', '-A')
        git(root, 'commit', '-qm', 'preserve contract')
        self.calls = []
        for module, name, value in ((st, 'verified_rows', lambda c, e, r: (self.rows_rec, [])),
                                    (p1, 'check_inputs', lambda c, r: {'rows': {'train': [], 'val': []}, 'h3': {},
                                                                        'cache': {}, 'source': {'models': {}}}),
                                    (st, 'input_files', lambda c: []), (a1, 'execute', self.fake_execute)):
            patch = mock.patch.object(module, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def write_preserve(self, c):
        path = self.root / pv.CONTRACT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(c, indent=1) + '\n')

    def approve_preserve(self, *stages):
        for s in stages:
            self.c['approved'][s] = True
        self.write_preserve(self.c)
        git(self.root, 'commit', '-qam', 'approve ' + ' '.join(stages))

    def view_source(self, run):
        from hybrid import h7c_train as tr
        return {**run['summary'], **run['result']['record'], 'p1': run['result']['p1'],
                'config': tr.config_dict(run['cfg']), 'settings': run['settings']}

    def preserve_run(self, seed, mu, change=None, candidate=False):
        """What ``h7r_p1a1.execute`` returns: the fixture's A1 T run, its replay coefficient ``mu``; another mu moves
        the per-step losses and the selected state (the draws and the other fingerprints stay)."""
        run = self.fake_run(change)
        run['result']['p1'] = copy.deepcopy(run['result']['p1'])
        run['result']['p1']['replay']['mu'] = mu
        if mu != pv.BASE_MU:
            fp = run['summary']['fingerprints'] = copy.deepcopy(run['summary']['fingerprints'])
            fp['loss'] = [x + 0.01 for x in fp['loss']]
            run['summary']['selected_state_sha256'] = 'f' * 64
        if candidate:
            from tests.test_h7c_train import small_model
            import torch
            run['result']['state'] = {}
            run['parent'] = small_model()
            run['logits'] = torch.zeros(3)
        return run

    def fake_execute(self, c, kind, seed, device, checked, root, log, expect_settings=None):
        mu = c['arms']['T']['replay']['weight_mu']
        self.calls.append({'kind': kind, 'seed': seed, 'mu': mu, 'settings': expect_settings})
        return self.preserve_run(seed, mu, getattr(self, 'change', None), candidate=getattr(self, 'candidate', True))

    def ns(self, **k):
        return argparse.Namespace(contract=None, device='cpu', log_name=None, **k)

    def through_suites(self):
        self.approve_preserve('prepare', 'suites', 'gate')
        pv.prepare(self.ns(), self.root)
        self.commit(self.c['output']['prepare'])
        ds2.build(argparse.Namespace(allow_dirty=False), self.root, log=QUIET)
        git(self.root, 'add', 'docs/h7r/suites')
        git(self.root, 'commit', '-qm', 'suites')

    def through_gate(self):
        self.through_suites()
        rec = pv.gate(self.ns(), self.root, log=QUIET)
        self.commit(self.c['output']['gate'])
        return rec

    def commit(self, rel):
        git(self.root, 'add', rel)
        git(self.root, 'commit', '-qm', rel)


class FlowTest(PreserveBase):
    def test_the_contract_pins(self):
        c, _, eid = pv.load_contract(None, self.root)
        self.assertEqual((c['replay']['mu'], c['replay']['base_mu']), (2.0, 0.5))
        self.assertEqual(c['s400_train']['experiment_sha256'], self.eid4)
        self.assertEqual(c['rows']['sha256'], self.rows_rec['candidate_rows_sha256'])
        self.assertEqual(sorted(c['base_runs']), ['20261024', '20261111', '20261112'])
        s4, _, _ = pv.s400(c, self.root)
        mixed = pv.with_mu(s4, 2.0)
        self.assertEqual(mixed['arms']['T']['replay']['weight_mu'], 2.0)
        self.assertEqual(s4['arms']['T']['replay']['weight_mu'], 0.5)  # a copy: the contract itself unchanged
        self.assertEqual({k: v for k, v in mixed.items() if k != 'arms'}, {k: v for k, v in s4.items() if k != 'arms'})
        bad = {**self.c, 'replay': {**self.c['replay'], 'mu': 1.0}}
        self.write_preserve(bad)
        git(self.root, 'commit', '-qam', 'mu changed')
        with self.assertRaisesRegex(ValueError, 'mu, seeds or statistics differ'):
            pv.load_contract(None, self.root)
        self.write_preserve({**self.c, 'approved': {**self.c['approved'], 'gate': True}})
        git(self.root, 'commit', '-qam', 'out of order')
        with self.assertRaisesRegex(ValueError, 'gate approved before'):
            pv.load_contract(None, self.root)

    def test_a_changed_s400_contract_is_refused(self):
        c4 = json.loads((self.root / st.CONTRACT).read_text())
        c4['note'] = 'edited'
        self.write_contract(c4)
        git(self.root, 'commit', '-qam', 'S400 contract edited')
        with self.assertRaisesRegex(ValueError, 'not the pinned one'):
            pv.s400(self.c, self.root)

    def test_prepare_gate_train_and_verify(self):
        with self.assertRaisesRegex(ValueError, 'the prepare stage is not approved'):
            pv.prepare(self.ns(), self.root)
        gate = self.through_gate()
        self.assertEqual((gate['passed'], gate['differences'], gate['mu']), (True, [], 0.5))
        self.assertEqual(self.calls[-1], {'kind': 'T', 'seed': 20261024, 'mu': 0.5,
                                          'settings': self.a1_rec['settings']})
        self.approve_preserve('train')
        rec = pv.train(self.ns(seed=20261024), self.root, log=QUIET)
        self.assertEqual(self.calls[-1]['mu'], 2.0)
        self.assertEqual((rec['mu'], rec['p1']['replay']['mu'], rec['verdict'], rec['arm']), (2.0, 2.0, 'CANDIDATE', 'S2'))
        self.assertEqual(rec['base_fingerprints_same'], {'steps': True, 'batch_sha256': True, 'sym_sha256': True,
                                                         'lr_sha256': True})  # the same draws; the losses moved
        self.assertNotEqual(rec['fingerprints']['loss'], json.loads(
            (self.root / st.run_record_path(20261024)).read_text())['fingerprints']['loss'])
        out = self.root / pv.RUNS / 'S2-20261024'
        self.assertEqual(rec['candidate'], {'checkpoint_sha256': p1.file_sha256(out / 'best.pt'),
                                            'metadata_sha256': p1.file_sha256(out / 'best.json')})
        self.assertEqual(json.loads((out / 'best.json').read_text())['preserve']['mu'], 2.0)
        self.assertEqual(rec['inputs']['policy_rows_sha256'], self.rows_rec['candidate_rows_sha256'])
        with self.assertRaisesRegex(ValueError, 'already holds a run'):
            pv.train(self.ns(seed=20261024), self.root, log=QUIET)
        with self.assertRaisesRegex(ValueError, 'seed not in the contract'):
            pv.train(self.ns(seed=1), self.root, log=QUIET)
        dst = self.root / pv.run_record_path(20261024)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out / 'record.json', dst)
        self.assertEqual(pv.verify_run(self.ns(seed=20261024), self.root)['verdict'], 'CANDIDATE')
        (out / 'best.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'best.json: its SHA-256 is not the one'):
            pv.verify_run(self.ns(seed=20261024), self.root)
        (out / 'best.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'best.pt is missing'):
            pv.verify_run(self.ns(seed=20261024), self.root)

    def test_the_gate_is_exact(self):
        self.through_suites()
        self.change = lambda r: r['evaluations'][0]['val'].update(ce=r['evaluations'][0]['val']['ce'] + 1e-12)
        rec = pv.gate(self.ns(), self.root, log=QUIET)
        self.assertFalse(rec['passed'])
        self.assertTrue(any(d.startswith('evaluations') for d in rec['differences']))
        self.commit(self.c['output']['gate'])
        self.approve_preserve('train')
        with self.assertRaisesRegex(ValueError, 'the reproduction gate did not pass'):
            pv.train(self.ns(seed=20261024), self.root, log=QUIET)

    def test_a_tampered_gate_record_is_refused(self):
        self.through_gate()
        path = self.root / self.c['output']['gate']
        rec = json.loads(path.read_text())
        rec['run']['selected_state_sha256'] = '0' * 64  # still "passed"
        self.commit_record(self.c['output']['gate'], rec)
        self.approve_preserve('train')
        with self.assertRaisesRegex(ValueError, 'the reproduction gate did not pass'):
            pv.train(self.ns(seed=20261024), self.root, log=QUIET)

    def test_a_run_that_did_not_use_the_contract_mu_publishes_nothing(self):
        self.through_gate()
        self.approve_preserve('train')
        with mock.patch.object(a1, 'execute', lambda c, *a, **k: self.preserve_run(20261111, pv.BASE_MU)):
            with self.assertRaisesRegex(ValueError, 'did not use the contract mu'):
                pv.train(self.ns(seed=20261111), self.root, log=QUIET)
        self.assertFalse((self.root / pv.RUNS / 'S2-20261111' / 'record.json').exists())

    def test_the_gate_needs_the_committed_suites(self):
        self.approve_preserve('prepare', 'suites', 'gate')
        pv.prepare(self.ns(), self.root)
        self.commit(self.c['output']['prepare'])
        with self.assertRaisesRegex(ValueError, 'O1-dev-cd-record.json: not committed'):
            pv.gate(self.ns(), self.root, log=QUIET)
        self.assertEqual(self.calls, [])  # nothing trained
        self.assertFalse((self.root / self.c['output']['gate']).exists())

    def test_a_run_with_other_draws_publishes_nothing(self):
        self.through_gate()
        self.approve_preserve('train')
        for key in pv.DRAWS:
            with self.subTest(key):
                def other(c, *a, key=key, **k):
                    run = self.preserve_run(20261111, pv.MU)
                    fp = run['summary']['fingerprints']
                    fp[key] = fp[key] + 1 if key == 'steps' else '0' * 64
                    return run
                with mock.patch.object(a1, 'execute', other):
                    with self.assertRaisesRegex(ValueError, rf"\['{key}'\] differ from the mu 0.5 run"):
                        pv.train(self.ns(seed=20261111), self.root, log=QUIET)
                out = self.root / pv.RUNS / 'S2-20261111'
                self.assertFalse(out.exists() and any(out.iterdir()))

    def test_prepare_checks_the_arena_base_checkpoint(self):
        self.approve_preserve('prepare')
        (self.root / st.RUNS / 'S-20261024' / 'best.pt').write_text('another checkpoint')
        with self.assertRaisesRegex(ValueError, 'the checkpoint files differ from its record'):
            pv.prepare(self.ns(), self.root)
        self.assertFalse((self.root / self.c['output']['prepare']).exists())


class SuiteFlowTest(PreserveBase):
    def test_registry_build_and_check(self):
        self.assertEqual(seeds2.conflicts(), {k: [] for k in seeds2.conflicts()})
        self.assertEqual(sorted(seeds2.SUITE_SEEDS), [202611191, 202611192])
        for stages, message in (((), 'the suites stage is not approved'),
                                (('prepare', 'suites'), 'the prepare record is not committed')):
            with self.subTest(stages):
                if stages:
                    self.approve_preserve(*stages)
                with self.assertRaisesRegex(ValueError, message):
                    ds2.build(argparse.Namespace(allow_dirty=False), self.root, log=QUIET)
                self.assertFalse((self.root / ds2.RECORD).exists() or (self.root / ds2.suite_path('c')).exists())
        pv.prepare(self.ns(), self.root)
        self.commit(self.c['output']['prepare'])
        rec = ds2.build(argparse.Namespace(allow_dirty=False), self.root, log=QUIET)
        self.assertEqual({k: v['seed'] for k, v in rec['suites'].items()}, {'c': 202611191, 'd': 202611192})
        self.assertEqual(rec['overlaps']['between_c_and_d'], {'class_id': 0, 'history_id': 0})
        paths = {s['path'] for s in rec['sources']}
        self.assertTrue({'docs/h7r/suites/O1-dev-a.json', 'docs/h7r/suites/O1-dev-b.json',
                         'docs/h7r/contracts-v2/R2-S400-TRAIN-ARENA-1.json'} <= paths)
        for label in ('c', 'd'):
            body = json.loads((self.root / ds2.suite_path(label)).read_text())
            self.assertEqual((body['count'], len({o['class_id'] for o in body['openings']})), (300, 300))
            self.assertEqual(body['seed_policy'], seeds2.POLICY)
            for src, counts in rec['overlaps'][label]['by_source'].items():
                self.assertEqual(counts['class_id'], 0, src)
        git(self.root, 'add', 'docs/h7r/suites')
        git(self.root, 'commit', '-qm', 'suites')
        self.assertTrue(ds2.check(None, self.root, log=QUIET)['passed'])
        self.assertTrue(ds.check(None, self.root, log=QUIET)['passed'])  # O1-dev-a / b still check beside them
        with self.assertRaisesRegex(ValueError, 'exist: never overwritten'):
            ds2.build(argparse.Namespace(allow_dirty=False), self.root, log=QUIET)
        body = json.loads((self.root / ds2.suite_path('d')).read_text())
        body['git'] = {**body['git'], 'commit': '0' * 40}  # a valid suite, not the recorded file
        (self.root / ds2.suite_path('d')).write_text(json.dumps(body, indent=1) + '\n')
        with self.assertRaisesRegex(ValueError, 'suite d: the file is not the recorded one'):
            ds2.check(None, self.root, log=QUIET)



def fake_play_game(task, make_tested=None, make_opponent=None, max_plies=None):
    """A complete game decided by the opening: the tested side wins as black; as white, S2 at about half of the
    openings and S05 at about a third."""
    from renju import BLACK
    from tests.h7r_helpers import winning_moves

    ctx = task['context']
    black = task['tested_color'] == BLACK
    h = int(hashlib.sha256(task['opening_id'].encode()).hexdigest(), 16)
    white_win = h % 2 == 0 if '_S2_' in ctx['arm'] else h % 3 == 0
    result = 'win' if black or white_win else 'loss'
    tested = BLACK if black else -BLACK
    winner = tested if result == 'win' else -tested
    moves = winning_moves(task['opening'], winner)
    return {'key': task['key'], 'signature': ctx['signature'], 'arm': task['arm'], 'pair': task['pair'],
            'seed': task['seed'], 'tested_color': 'black' if black else 'white', 'result': result,
            'winner': winner, 'length': len(moves), 'moves': moves, 'illegal': None,
            'truncated': False, 'tested_moves': [], 'opponent_routes': {}, 'game_seconds': 1.0}


class ArenaTest(PreserveBase):
    def setUp(self):
        super().setUp()
        self.through_gate()
        self.approve_preserve('train')
        pv.train(self.ns(seed=20261024), self.root, log=QUIET)
        dst = pv.run_record_path(20261024)
        (self.root / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.root / pv.RUNS / 'S2-20261024' / 'record.json', self.root / dst)
        self.commit(dst)
        self.approve_preserve('arena')

    def head(self):
        return {'commit': git(self.root, 'rev-parse', 'HEAD'), 'dirty': False}

    def args(self, **k):
        return argparse.Namespace(contract=None, **k)

    def test_contract_play_and_judge(self):
        body = pv.arena_contract(self.ns(), self.root)
        self.assertEqual({(a['pairing'], a['seed'], a['pairs']) for a in body['arms']},
                         {('O1-dev-c', 9103, 300), ('O1-dev-d', 9104, 300)})
        self.assertEqual(sorted(a['name'] for a in body['arms']),
                         sorted(pv.arm_name(s, m) for s in ('O1-dev-c', 'O1-dev-d') for m in pv.ARMS))
        self.assertEqual(body['models']['S05']['identity']['policy_checkpoint_sha256'],
                         self.c['base_runs']['20261024']['checkpoint_sha256'])
        self.assertEqual(body['stats'], pv.STATS)
        self.assertFalse(pv.check_contract_file(self.args(), self.root)['committed'])
        with self.assertRaisesRegex(ValueError, 'exists: never overwritten'):
            pv.arena_contract(self.ns(), self.root)
        self.commit(pv.ARENA_CONTRACT)
        a = pv.load(self.args(), self.root)
        self.assertEqual(sorted(pv.models_command(self.args(), self.root, log=QUIET)['models_checked']),
                         ['S05', 'S2', 'champion'])
        folder = self.root / pv.ARMS_DIR
        from hybrid.h7r_ledger import publish_json
        state = self.head()
        for arm in a['arena']['arms']:
            args = self.args(arm=arm['name'], device='cpu', workers=1, jsonl=folder / f"{arm['name']}.jsonl",
                             allow_dirty=False, threads_for_tests=True)
            with mock.patch('h7c_arena.play_game', fake_play_game), mock.patch.object(cli, '_git', lambda _: dict(state)):
                publish_json(folder / f"{arm['name']}.json", pv.play(args, log=QUIET, root=self.root))
        out = pv.judge(self.args(arms_dir=folder), self.root)
        self.assertEqual(out['bounds']['units'], {'O1-dev-c': 300, 'O1-dev-d': 300})
        for key, (hi, lo) in (("P''", ('STRONGER', 'WEAKER')), ("D''", ('BETTER', 'WORSE'))):
            b = out['bounds'][key]
            self.assertEqual(out['readings'][key],
                             hi if b['lower_one_sided'] > 0 else lo if b['upper_one_sided'] < 0 else 'NOT_ESTABLISHED')
        self.assertGreater(out['bounds']["D''"]['estimate'], 0)  # S2 wins more as white in the fake
        self.assertEqual(out['use'], pv.use(out['readings']))
        self.assertEqual(out['arms_commit'], out['git']['commit'])
        self.assertEqual(pv.verify_judge(self.args(arms_dir=folder), self.root), out)


class RuleTest(unittest.TestCase):
    def setUp(self):
        names = ('O1-dev-c', 'O1-dev-d')
        self.arena = {'arms': [{'name': pv.arm_name(s, m), 'pairing': s, 'pairs': 3} for s in names for m in pv.ARMS]}
        self.tables = {pv.arm_name(s, m): {'pairs': [{'opening_id': f'{s}-{i}', 'class_id': f'c{i}',
                                                      'pair_score': score} for i in range(3)]}
                       for s in names for m, score in (('S2', 1.0), ('S05', 0.5))}

    def test_pairs_must_match(self):
        strata = pv.paired_strata(self.arena, self.tables)
        self.assertEqual(strata['O1-dev-c'][0], (1.0, 0.5, 'c0'))
        b = pv.bounds_of(strata, 'opening')
        self.assertEqual((b["P''"]['estimate'], b["D''"]['estimate']), (0.5, 0.5))
        t = pv.arm_name('O1-dev-d', 'S05')
        for message, change in (('same openings in the same order', lambda tb: tb[t]['pairs'].reverse()),
                                ('differ in class', lambda tb: tb[t]['pairs'][1].update(class_id='cX')),
                                ('the sample is never reduced', lambda tb: tb[t]['pairs'].pop())):
            with self.subTest(message):
                tables = json.loads(json.dumps(self.tables))
                change(tables)
                with self.assertRaisesRegex(ValueError, message):
                    pv.paired_strata(self.arena, tables)
        with self.assertRaisesRegex(ValueError, 'the arms are not S2 and S05'):
            pv.paired_strata({'arms': self.arena['arms'][1:]}, self.tables)

    def test_the_use_table(self):
        lab = lambda p, d: {"P''": p, "D''": d}  # noqa: E731
        out = pv.use(lab('STRONGER', 'BETTER'))
        self.assertEqual(out['decision'], 'MAY_OPEN_CONFIRMATION')
        self.assertIn('improvement over the old mu 0.5 candidate', out['note'])
        out = pv.use(lab('STRONGER', 'NOT_ESTABLISHED'))
        self.assertEqual(out['decision'], 'MAY_OPEN_CONFIRMATION')
        self.assertIn('undetermined', out['note'])
        out = pv.use(lab('NOT_ESTABLISHED', 'BETTER'))
        self.assertEqual((out['decision'], out['note']),
                         ('END', 'the stronger preservation is recorded as an improvement over the old mu 0.5 '
                                 'candidate; the contract ends'))
        out = pv.use(lab('STRONGER', 'WORSE'))
        self.assertEqual((out['decision'], out['worse']), ('MAY_OPEN_CONFIRMATION', ["D''"]))
        self.assertIn('worse than the old mu 0.5 candidate', out['note'])
        self.assertNotIn('undetermined', out['note'])
        out = pv.use(lab('WEAKER', 'WORSE'))
        self.assertEqual((out['decision'], out['worse']), ('END', ["P''", "D''"]))
        self.assertEqual(out['note'], 'the stronger preservation is recorded as worse than the old mu 0.5 candidate; '
                                      'the contract ends')
        self.assertEqual(pv.use(lab('NOT_ESTABLISHED', 'NOT_ESTABLISHED'))['note'], 'the contract ends')


class FixedContractTest(unittest.TestCase):
    def test_the_committed_contract(self):
        c, _, eid = pv.load_contract(None, ROOT)
        self.assertEqual(list(c['approved']), list(pv.STAGES))  # approvals change over time, never the id
        self.assertEqual({k: v for k, v in c.items() if k not in ('sources', 'approved')},
                         {k: v for k, v in pv.contract_body(ROOT).items() if k not in ('sources', 'approved')})
        self.assertEqual([s['path'] for s in c['sources']], list(pv.SOURCES))
        self.assertEqual(c['s400_train']['experiment_sha256'],
                         '67c6ba3c8081da661eb1cd0330a325499f62f52d03ff61f9597dd496eca4d731')
        self.assertEqual(eid, 'd832623f66ec616005b5b4a78dd8516328373deef21ba32124993953f39b3fc1')


# --- the desktop blocks ----------------------------------------------------------------------------------------------

PV_STUB = """import hashlib, json, os, sys
from pathlib import Path
D = 'docs/h7r/r2/R2-S400-PRESERVE-1'
R = 'runs/h7r/R2-S400-PRESERVE-1'
A = Path('docs/h7r/contracts-v2/R2-S400-PRESERVE-ARENA-1.json')
cmd = sys.argv[1]
log = sys.argv[sys.argv.index('--log-name') + 1] if '--log-name' in sys.argv else None
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if os.environ.get('PV_FORBID') and cmd not in ('verify-run', 'check-contract', 'verify-judge'):
    print('STOP: must not run again'); sys.exit(9)
if cmd in ('gate', 'train') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
    print('STOP: CUBLAS_WORKSPACE_CONFIG'); sys.exit(7)
if os.environ.get('PV_FAIL') in (cmd, cmd + ':' + (sys.argv[sys.argv.index('--seed') + 1] if '--seed' in sys.argv else '')):
    print('STOP: ValueError: stand-in failure'); sys.exit(1)
if cmd in ('prepare', 'gate'):
    passed = not (cmd == 'gate' and os.environ.get('PV_GATE_FAIL'))
    Path(D, cmd + '.json').write_text(json.dumps({'format': 'h7r-preserve-' + cmd + '-v1', 'passed': passed,
                                                  'log_name': log}))
    print(cmd, passed); sys.exit(0 if passed else 2)
if cmd == 'train':
    seed = int(sys.argv[sys.argv.index('--seed') + 1])
    out = Path(R, 'S2-' + str(seed))
    out.mkdir(parents=True, exist_ok=True)
    for name in ('best.pt', 'best.json', 'logits.pt'):
        (out / name).write_bytes((name + ' ' + str(seed)).encode())
    (out / 'record.json').write_text(json.dumps({'format': 'h7r-preserve-run-v1', 'seed': seed, 'mu': 2.0,
        'log_name': log, 'verdict': 'CANDIDATE', 'candidate': {'checkpoint_sha256': sha(out / 'best.pt'),
                                                               'metadata_sha256': sha(out / 'best.json')}}))
    if os.environ.get('PV_TAMPER') == str(seed):  # the checkpoint changes after its record is written
        (out / 'best.pt').write_bytes(b'replaced')
    print('train', seed); sys.exit(0)
if cmd == 'verify-run':
    seed = sys.argv[sys.argv.index('--seed') + 1]
    rec = json.loads(Path(D, 'seed-records', 'S2-' + seed, 'record.json').read_text())
    f = Path(R, 'S2-' + seed, 'best.pt')
    if not f.is_file():
        print('STOP: ValueError: best.pt is missing'); sys.exit(1)
    if sha(f) != rec['candidate']['checkpoint_sha256']:
        print('STOP: ValueError: best.pt: its SHA-256 is not the one the record records'); sys.exit(1)
    print('verified', seed); sys.exit(0)
if cmd == 'arena-contract':
    A.parent.mkdir(parents=True, exist_ok=True)
    arms = [{'name': s + '_' + m + '_vs_champion'} for s in ('O1-dev-c', 'O1-dev-d') for m in ('S2', 'S05')]
    A.write_text(json.dumps({'format': 'stand-in arena', 'arms': arms}))
    print('arena contract written'); sys.exit(0)
if cmd == 'check-contract':
    ok = json.loads(A.read_text()).get('format') == 'stand-in arena'
    print('checked' if ok else 'STOP: ValueError: not the contract'); sys.exit(0 if ok else 1)
if cmd == 'check-models':
    print('models as contracted'); sys.exit(0)
if cmd == 'play':
    arm = sys.argv[sys.argv.index('--arm') + 1]
    if os.environ.get('PV_ARM_FAIL') == arm:
        print('STOP: ValueError: stand-in game failure'); sys.exit(1)
    Path(sys.argv[sys.argv.index('--output') + 1]).write_text(json.dumps({'arm': arm}))
    print('played', arm); sys.exit(0)
if cmd in ('judge', 'verify-judge'):
    if cmd == 'judge':
        Path(D, 'judge.json').write_text(json.dumps({'format': 'h7r-preserve-judge-v1'}))
    print(cmd); sys.exit(0)
print('STOP: unknown command ' + cmd); sys.exit(1)
"""


class BlockTest(unittest.TestCase):
    """The desktop blocks of the draft (section 11) run by bash on a real temporary repository with a bare origin, the
    real .gitignore and record tool; the preserve tool is a stand-in."""

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
        for rel in ('scripts/h7r_d4_records.py', 'src/hybrid/h7r_ledger.py', 'scripts/h7r_attempts.py'):
            (w / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, w / rel)
        (w / 'src/hybrid/__init__.py').write_text('')
        (w / 'scripts/check_track_b_frozen.py').write_text('print("frozen PASS")' + chr(10))
        (w / 'scripts/h7r_preserve.py').write_text(PV_STUB)
        (w / '.venv/bin').mkdir(parents=True)
        (w / '.venv/bin/activate').write_text('')
        (w / pv.D).mkdir(parents=True)
        (w / pv.CONTRACT).write_text(json.dumps({'format': 'stand-in', 'contract_id': pv.CONTRACT_ID}))
        git(w, 'add', '-A')
        git(w, 'commit', '-qm', 'init')
        git(w, 'remote', 'add', 'origin', str(self.origin))
        git(w, 'push', '-q', 'origin', 'feat/h7-c1')
        doc = (ROOT / 'docs/h7r/r2/R2-S400-PRESERVE-1-draft.md').read_text()
        blocks = re.findall(r'```bash' + chr(10) + r'(.*?)```', doc, re.S)
        self.assertEqual(len(blocks), 5)
        self.blocks = dict(zip(('pv-P', 'pv-G', 'pv-T', 'pv-AC', 'pv-A'),
                               (b.replace('~/renju-h6', w.as_posix()) for b in blocks)))

    def bash(self, name, **env):
        import os
        import subprocess
        e = {k: v for k, v in os.environ.items() if k != 'CUBLAS_WORKSPACE_CONFIG'}
        e = {**e, 'PYTHONDONTWRITEBYTECODE': '1', **env}
        r = subprocess.run(['bash', '-c', self.blocks[name]], cwd=self.work, capture_output=True, text=True, env=e,
                           start_new_session=True)
        return r.returncode, r.stdout + r.stderr

    def files(self, rel):
        return git(self.work, 'ls-files', rel).splitlines()

    def through(self, *blocks):
        for b in blocks:
            code, out = self.bash(b)
            self.assertEqual(code, 0, f'{b}: {out}')

    def assert_pushed(self):
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), git(self.origin, 'rev-parse', 'feat/h7-c1'))
        self.assertEqual(git(self.work, 'status', '--porcelain', '--untracked-files=no'), '')

    def subjects(self):
        return git(self.work, 'log', '--format=%s').splitlines()

    def test_prepare_then_gate(self):
        code, out = self.bash('pv-G')
        self.assertEqual(code, 1)
        self.assertIn('prepare is not recorded', out)
        self.through('pv-P', 'pv-G')
        self.assertTrue(self.files(f'{pv.D}/gate.json'))
        self.assertEqual(len([f for f in self.files(f'{pv.D}/stage-logs') if f.endswith('.log')]), 2)
        self.assert_pushed()
        code, out = self.bash('pv-G')
        self.assertEqual(code, 1)
        self.assertIn('already recorded', out)

    def test_a_failed_gate_is_recorded_and_stops(self):
        self.through('pv-P')
        code, out = self.bash('pv-G', PV_GATE_FAIL='1')
        self.assertEqual(code, 2, out)
        self.assertTrue(self.files(f'{pv.D}/gate.json'))
        self.assertEqual(self.subjects()[0], 'H7-R PRESERVE: gate exit 2 (records only)')

    def test_an_interrupted_gate_is_recorded_from_its_result(self):
        self.through('pv-P')
        L = self.work / pv.RUNS / 'logs'
        name = 'gate-AbC123xy.log'
        (L / name).write_text('gate True' + chr(10))  # killed before its exit line
        (self.work / pv.D / 'gate.json').write_text(json.dumps(
            {'format': 'h7r-preserve-gate-v1', 'passed': True, 'log_name': name}))
        code, out = self.bash('pv-G', PV_FORBID='1')
        self.assertEqual(code, 0, out)
        self.assertIn('Recovered exit code', (self.work / pv.D / 'stage-logs' / name).read_text())

    def test_train_three_seeds_one_commit_each_then_verified(self):
        code, out = self.bash('pv-T')
        self.assertEqual(code, 1)
        self.assertIn('the gate is not recorded', out)
        self.through('pv-P', 'pv-G')
        code, out = self.bash('pv-T', PV_FAIL='train:20261111')
        self.assertEqual(code, 1, out)
        self.assertIn('STOP: train S2 20261111 failed', out)
        self.assertEqual(self.subjects()[:2], ['H7-R PRESERVE: train S2 20261111 exit 1 (records only)',
                                               'H7-R PRESERVE: train S2 20261024 exit 0 (records only)'])
        self.assertFalse(self.files(f'{pv.SEED_RECORDS}/S2-20261112'))
        self.through('pv-T')  # 20261024 verified and skipped, the rest trained
        self.assertEqual(self.subjects()[:2], ['H7-R PRESERVE: train S2 20261112 exit 0 (records only)',
                                               'H7-R PRESERVE: train S2 20261111 exit 0 (records only)'])
        for s in pv.SEEDS:
            self.assertEqual(self.files(pv.run_record_path(s)), [pv.run_record_path(s)])
        self.assert_pushed()
        (self.work / pv.RUNS / 'S2-20261024' / 'best.pt').write_text('replaced')
        code, out = self.bash('pv-T', PV_FORBID='1')
        self.assertEqual(code, 1, out)
        self.assertIn('its SHA-256 is not the one', out)

    def test_checkpoint_files_are_checked_before_the_record_is_staged(self):
        self.through('pv-P', 'pv-G')
        head = git(self.work, 'rev-parse', 'HEAD')
        code, out = self.bash('pv-T', PV_TAMPER='20261024')
        self.assertEqual(code, 1, out)
        self.assertIn('the files of S2-20261024 are not the ones its record names; nothing committed', out)
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), head)  # nothing committed or pushed
        self.assertEqual(git(self.origin, 'rev-parse', 'feat/h7-c1'), head)
        self.assertEqual(git(self.work, 'diff', '--cached', '--name-only'), '')
        (self.work / pv.RUNS / 'S2-20261024' / 'best.pt').unlink()  # the recovery path checks them too
        code, out = self.bash('pv-T', PV_FORBID='1')
        self.assertEqual(code, 1, out)
        self.assertIn('an uncommitted', out)
        self.assertIn('nothing committed', out)
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), head)

    def test_files_without_a_record_stop_the_train_block(self):
        self.through('pv-P', 'pv-G')
        out_dir = self.work / pv.RUNS / 'S2-20261024'
        out_dir.mkdir(parents=True)
        (out_dir / 'best.pt.tmp').write_text('partial')
        code, out = self.bash('pv-T', PV_FORBID='1')
        self.assertEqual(code, 1, out)
        self.assertIn('without its record', out)
        self.assertTrue((out_dir / 'best.pt.tmp').exists())

    def test_arena_contract_then_arena(self):
        code, out = self.bash('pv-AC')
        self.assertEqual(code, 1)
        self.assertIn('train S2 20261024 is not recorded', out)
        self.through('pv-P', 'pv-G', 'pv-T')
        code, out = self.bash('pv-A')
        self.assertEqual(code, 1)
        self.assertIn('run pv-AC first', out)
        self.through('pv-AC')
        self.assertTrue(self.files(pv.ARENA_CONTRACT))
        code, out = self.bash('pv-A', PV_ARM_FAIL='O1-dev-c_S05_vs_champion')
        self.assertEqual(code, 1, out)
        self.assertIn('rerun this block to resume', out)
        self.assertFalse(self.files(pv.JUDGE))
        code, out = self.bash('pv-A')
        self.assertEqual(code, 0, out)
        self.assertIn('done: O1-dev-c_S2_vs_champion', out)
        self.assertTrue(self.files(pv.JUDGE))
        self.assertEqual(self.subjects()[0], 'H7-R PRESERVE: Arena and judge (records only)')
        self.assert_pushed()
