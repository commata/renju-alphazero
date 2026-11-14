"""scripts/h7r_s400_train.py: the contract (A1's unchanged parts, the control checkpoints, v2 Arena seeds); the
candidate rows (only train visit targets replaced, the S200 target kept without S400 mass, no row added, every other
row and field unchanged, a deterministic file) and their check; the T reproduction gate (exact, before any search);
the search order (S200 check, then S400; a difference stops); the provenance of a candidate run; the v2 Arena
contract. The engine search and the training are stand-ins; the rules and the records are the tool's."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from hybrid import h7c_retarget as hr
from hybrid.h6_labels import canonical_hash, split_for
import h7r_p1 as p1
import h7r_s400_train as st
from tests.test_h7r_p1 import git

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / 'docs/h7r/r2/R2-S400-TRAIN/s400-train-contract.draft.json'
QUIET = lambda *a, **k: None  # noqa: E731


def draft():
    return {**json.loads(DRAFT.read_text()), 'status': 'fixed'}


class ContractTest(unittest.TestCase):
    def test_rules_and_the_unchanged_parts(self):
        c = draft()
        a1c = json.loads((ROOT / 'docs/h7r/r2/R2-P1-POLICY-REPLAY-A1/p1a1-contract.json').read_text())
        for k in ('inputs', 'training', 'determinism', 'data_rule'):
            self.assertEqual(c[k], a1c[k], k)
        self.assertEqual(c['arms']['T'], a1c['arms']['T'])
        self.assertEqual(c['arms']['S']['replay'], a1c['arms']['T']['replay'])
        self.assertEqual(c['training_structure'], a1c['stages']['gate']['expected'])
        self.assertEqual(c['seeds']['training'], a1c['seeds']['training'])
        self.assertEqual(c['approved'], {s: False for s in st.STAGES})
        for seed, ref in c['control']['models'].items():
            rec = json.loads((ROOT / ref['seed_record']).read_text())
            self.assertEqual((rec['arm'], rec['seed'], rec['candidate']),
                             ('T', int(seed), {'checkpoint_sha256': ref['checkpoint_sha256'],
                                               'metadata_sha256': ref['metadata_sha256']}))
        from hybrid import h7r_seeds as seeds
        for s in c['arena']['suites']:
            self.assertEqual(p1.file_sha256(ROOT / s['path']), s['sha256'])
            self.assertIn(s['seed'], seeds.ARENA_SEEDS)
        self.assertEqual(c['search']['scope_eligible'],
                         json.loads((ROOT / 'docs/h7r/r2/R2-S400-PRE/prepare.json').read_text())['sample']['scopes']
                         ['training_eligible'])
        with TemporaryDirectory() as tmp, mock.patch.object(st, 'committed', return_value=True):
            path = Path(tmp) / 'c.json'
            for message, bad in (
                    ('approved before', {**c, 'approved': {**c['approved'], 'search': True}}),
                    ('gate / candidate seed', {**c, 'gate': {**c['gate'], 'seed': 20261111}}),
                    ('search budgets', {**c, 'search': {**c['search'], 'budgets': {
                        **c['search']['budgets'], 's400': {**c['search']['budgets']['s400'], 'factor': 4}}}})):
                with self.subTest(message):
                    path.write_text(json.dumps({**bad, 'sources': [], 'evidence': {}}))
                    with self.assertRaisesRegex(ValueError, message):
                        st.load_contract(path, ROOT)


class FixedContractTest(unittest.TestCase):
    def test_the_fixed_contract_is_its_draft_with_sources(self):
        fixed = json.loads((ROOT / st.CONTRACT).read_text())
        d = json.loads(DRAFT.read_text())
        self.assertEqual(set(fixed['approved']), set(st.STAGES))  # approvals change over time, never the id
        self.assertEqual(p1.experiment_sha256(fixed),
                         '67c6ba3c8081da661eb1cd0330a325499f62f52d03ff61f9597dd496eca4d731')
        self.assertEqual(fixed['draft'], {'path': DRAFT.relative_to(ROOT).as_posix(), 'sha256': p1.file_sha256(DRAFT)})
        self.assertEqual({k: v for k, v in fixed.items() if k not in ('status', 'sources', 'draft', 'approved')},
                         {k: v for k, v in d.items() if k not in ('status', 'sources', 'approved')})
        st.load_contract(None, ROOT)  # every pinned source and evidence file is this checkout's (and committed)


def stand_in_games(n=12):
    """Cycle-1 stand-in games; game 0 plays a ply whose S200 search puts all mass on proven losses (no c2 row)."""
    from tests.test_h7c_labels import game
    return [game(i, game_seed=100 + i) for i in range(n)]


def c2_records(games):
    out = {}
    for g in games:
        for p in hr.eligible_plies(g):
            row = {'ply': p['ply'], 'color': p['color'], 'seed': 7, 'canonical_hash': canonical_hash(g['moves'][:p['ply']]),
                   'route': 'tree', 'engine_move': p['engine_move'],
                   'visits': [[*p['engine_move'], 150, 0.2], [3, 3, 50, 0.0]], 'proven_loss': [], 'e4': {},
                   'tree_budget': 200, 'seconds': 0.5, 'cooldown_restored': False}
            if g['game_index'] == 0 and p['ply'] == hr.eligible_plies(g)[0]['ply']:
                row['visits'] = [[3, 3, 50, 0.0]]
                row['proven_loss'] = [[3, 3]]  # S200 leaves no mass: no c2 row here
            out[(g['game_index'], p['ply'])] = row
    return out


def s400_records(games, c2):
    """S400: another split everywhere; at game 1's first ply all mass on a proven loss (the S200 target is kept)."""
    out = {}
    for (gi, ply), row in c2.items():
        g = games[gi]
        p = next(x for x in g['plies'] if x['ply'] == ply)
        new = {**copy.deepcopy(row), 'visits': [[*p['engine_move'], 300, 0.25], [4, 4, 100, 0.0]], 'proven_loss': [],
               'tree_budget': 400}
        if gi == 1 and ply == hr.eligible_plies(g)[0]['ply']:
            new['visits'] = [[*p['engine_move'], 10, -0.9]]
            new['proven_loss'] = [p['engine_move']]
        out[(gi, ply)] = new
    return out


def c2_rows(games, c2):
    import h7c_labels as lab
    provenance = lambda g: {'engine_config_hash': g['engine_hash'], 'model_hash': g['model_hash'],  # noqa: E731
                            'git_commit': 'x', 'h7c_provenance': {}}
    rows = [r for g in games for r in lab.policy_rows(
        g, split_for(lab.game_id(g), fractions=lab.SPLIT_FRACTIONS, salt=lab.SPLIT_SALT), provenance(g),
        retarget={'rows': c2, 'info': {'sha256': 'f' * 64}})]
    return rows


class RowsTest(unittest.TestCase):
    def setUp(self):
        import h7c_labels as lab
        self.games = stand_in_games()
        self.c2 = c2_records(self.games)
        self.s400 = s400_records(self.games, self.c2)
        self.rows = c2_rows(self.games, self.c2)
        self.by_id = {lab.game_id(g): g for g in self.games}
        self.rows[-1]['mask'], self.rows[-1]['flags'] = True, ['probe_position']

    def test_only_train_visit_targets_change(self):
        cand, counts = st.candidate_rows(self.rows, self.by_id, self.s400)
        self.assertEqual(len(cand), len(self.rows))
        self.assertEqual(st.check_candidate(self.rows, cand, self.by_id, self.s400), [])
        visits = [r for r in self.rows if r['split'] == 'train' and r['kind'] == 'visits']
        self.assertEqual(counts['train_visit_rows'], len(visits))
        self.assertEqual(counts['s200_kept_no_s400_mass'], 1)
        self.assertEqual(counts['s400'], len(visits) - 1)
        self.assertAlmostEqual(counts['s400_share'], round((len(visits) - 1) / len(visits), 6))
        for a, b in zip(self.rows, cand):
            if a['split'] == 'train' and a['kind'] == 'visits':
                self.assertIn(b['s400']['target'], ('s400', 's200_kept'))
                if b['s400']['target'] == 's400':
                    self.assertNotEqual(a['target'], b['target'])
                else:
                    self.assertEqual(a['target'], b['target'])
            else:
                self.assertEqual(a, b)  # val, proof and masked rows unchanged
        trained = [g for g in self.games]
        self.assertEqual(st.s400_only_targets(self.rows, trained, self.s400), 1)  # game 0's first ply: not added

    def test_the_check_refuses_any_other_change(self):
        cand, _ = st.candidate_rows(self.rows, self.by_id, self.s400)
        i_val = next((i for i, r in enumerate(self.rows) if r['split'] != 'train'), None)
        i_train = next(i for i, r in enumerate(cand) if r.get('s400', {}).get('target') == 's400')
        i_proof = next(i for i, r in enumerate(self.rows) if r['kind'] == 'proof_one_hot')
        cases = {
            'a row less': lambda c: c.pop(),
            'another target': lambda c: c[i_train]['target'][0].__setitem__(2, 0.5),
            'a mask': lambda c: c[i_train].update(mask=True),
            'the order': lambda c: c.__setitem__(slice(0, 2), [c[1], c[0]]),
            'a proof row': lambda c: c[i_proof].update(target=[[0, 0, 1.0]]),
            'a fallback claimed as S400': lambda c: next(r for r in c if r.get('s400', {}).get('target') == 's200_kept')
            .update(s400={'target': 's400'}),
        }
        if i_val is not None:
            cases['a val target'] = lambda c: c[i_val].update(target=[[3, 3, 1.0]])
        for what, change in cases.items():
            with self.subTest(what):
                bad = copy.deepcopy(cand)
                change(bad)
                self.assertTrue(st.check_candidate(self.rows, bad, self.by_id, self.s400), what)

    def test_the_file_is_deterministic(self):
        cand, _ = st.candidate_rows(self.rows, self.by_id, self.s400)
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a' / 'r.jsonl.gz', Path(tmp) / 'b' / 'r.jsonl.gz'
            st.write_rows(a, cand)
            st.write_rows(b, copy.deepcopy(cand))
            self.assertEqual(p1.file_sha256(a), p1.file_sha256(b))
            self.assertEqual(st.read_rows(a), cand)


class GateTest(unittest.TestCase):
    def test_the_comparison_is_exact(self):
        rec = json.loads((ROOT / 'docs/h7r/r2/R2-P1-POLICY-REPLAY-A1/seed-records/T-20261024/record.json').read_text())
        import h7r_p1a1 as a1
        self.assertEqual(a1.differences(st.gate_view(rec), st.gate_view(copy.deepcopy(rec))), [])
        bad = copy.deepcopy(rec)
        bad['fingerprints']['loss'][3] += 1e-12
        self.assertEqual(a1.differences(st.gate_view(bad), st.gate_view(rec)), ['fingerprints.loss'])
        bad = copy.deepcopy(rec)
        bad['settings']['threads'] = 999
        self.assertEqual(a1.differences(st.gate_view(bad), st.gate_view(rec)), ['settings.threads'])
        self.assertNotIn('seconds', st.GATE_COMPARED)


class RepoBase(unittest.TestCase):
    """gate, search, train and arena-contract on a temporary repository: the contract with stand-in input SHAs, the
    committed A1 records it names, stand-in checkpoints; ``h7r_p1a1.execute`` and the engine search are faked."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.root = Path(self.tmp.name)
        git(root, 'init', '-q', '-b', 'feat/h7-c1')
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(root, 'config', k, v)
        shutil.copy(ROOT / '.gitignore', root / '.gitignore')
        for rel in ('docs/h7r/suites', 'docs/h7r/contracts', 'docs/h7r/r2/R2-P1-POLICY-REPLAY-A1/seed-records'):
            shutil.copytree(ROOT / rel, root / rel)
        for rel in ('docs/h7r/r2/R2-P1-POLICY-REPLAY-A1/gate.json',):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, root / rel)
        c = draft()
        c['evidence'], c['sources'] = {}, []
        for seed, ref in c['control']['models'].items():  # stand-in A1 checkpoints named by the copied seed records
            for kind in ('checkpoint', 'metadata'):
                path = root / ref[kind]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f'{seed} {kind}')
                ref[f'{kind}_sha256'] = p1.file_sha256(path)
            rec_path = root / ref['seed_record']
            rec = json.loads(rec_path.read_text())
            rec['candidate'] = {'checkpoint_sha256': ref['checkpoint_sha256'], 'metadata_sha256': ref['metadata_sha256']}
            rec_path.write_text(json.dumps(rec))
            ref['seed_record_sha256'] = p1.file_sha256(rec_path)
        self.c = c
        self.write_contract(c)
        git(root, 'add', '-A')
        git(root, 'commit', '-qm', 'contract')
        self.a1_rec = json.loads((root / c['control']['models']['20261024']['seed_record']).read_text())

    def write_contract(self, c):
        path = self.root / st.CONTRACT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(c, indent=1))

    def approve(self, *stages):
        c = json.loads((self.root / st.CONTRACT).read_text())
        for s in stages:
            c['approved'][s] = True
        self.write_contract(c)
        git(self.root, 'commit', '-qam', 'approve')

    def commit_record(self, rel, body):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body))
        git(self.root, 'add', rel)
        git(self.root, 'commit', '-qm', rel)

    def ns(self, **k):
        import argparse
        return argparse.Namespace(contract=None, device='cpu', **k)

    def eid(self):
        return st.load_contract(None, self.root, check_sources=False)[2]

    def fake_run(self, change=None):
        """What ``h7r_p1a1.execute`` returns for T(20261024) reproducing its A1 record (changed by ``change``)."""
        rec = copy.deepcopy(self.a1_rec)
        if change:
            change(rec)
        from hybrid import h7c_train as tr
        cfg = tr.PolicyConfig(**rec['config'])
        summary = {k: rec[k] for k in ('initial_state_sha256', 'fingerprints', 'selected_state_sha256', 'fixed_logits')}
        record = {k: rec[k] for k in ('total_steps', 'steps_per_epoch', 'warmup_steps', 'train_rows', 'val_rows',
                                      'teacher_guard', 'parent_guard_top1', 'parent_val', 'evaluations', 'best',
                                      'verdict')}
        return {'result': {'record': record, 'p1': rec['p1'], 'state': None}, 'cfg': cfg, 'parent': None,
                'settings': rec['settings'], 'logits': None, 'summary': summary, 'seconds': 1.0,
                'inputs': {'policy_rows': p1.PATHS['cycle_rows'], 'policy_rows_sha256': 'c' * 64,
                           'parent_checkpoint_sha256': 'p' * 64, 'parent_metadata_sha256': 'm' * 64,
                           'teacher_checkpoint_sha256': 'p' * 64, 'source': {'models': {}}}}

    def through_prepare(self):
        self.approve('prepare', 'gate')
        self.commit_record(f'{st.D}/prepare.json', {'experiment_sha256': self.eid(), 'passed': True})

    def gate_with(self, change=None):
        with mock.patch.object(st, 'input_files', return_value=[]), \
                mock.patch.object(p1, 'check_inputs', return_value={}), \
                mock.patch('h7r_p1a1.execute', return_value=self.fake_run(change)) as ex:
            out = st.gate(self.ns(), self.root, log=QUIET)
        self.assertEqual(ex.call_args.kwargs['expect_settings'], json.loads(
            (self.root / self.c['gate']['a1_gate']['path']).read_text())['settings'])
        return out

class FlowTest(RepoBase):
    def test_the_gate_passes_only_on_an_exact_reproduction(self):
        self.through_prepare()
        bad = self.gate_with(lambda r: r['evaluations'][0]['val'].update(ce=0.0))
        self.assertFalse(bad['passed'])
        self.assertEqual(bad['differences'], ['evaluations'])
        (self.root / st.D / 'gate.json').unlink()
        good = self.gate_with()
        self.assertTrue(good['passed'], good['differences'])
        git(self.root, 'add', f'{st.D}/gate.json')
        git(self.root, 'commit', '-qm', 'gate')
        st.verified_gate(st.load_contract(None, self.root, check_sources=False)[0], self.eid(), self.root)
        rec = json.loads((self.root / st.D / 'gate.json').read_text())
        rec['run']['best']['step'] = 1  # a passed record whose run no longer matches A1
        (self.root / st.D / 'gate.json').write_text(json.dumps(rec))
        git(self.root, 'commit', '-qam', 'tampered')
        with self.assertRaisesRegex(ValueError, 'the T reproduction gate did not pass'):
            st.verified_gate(st.load_contract(None, self.root, check_sources=False)[0], self.eid(), self.root)

    def test_search_checks_s200_first_and_stops_on_a_difference(self):
        self.through_prepare()
        self.approve('search')
        self.commit_record(f'{st.D}/gate.json', {'experiment_sha256': self.eid(), 'passed': True})
        games = stand_in_games()
        c2 = c2_records(games)
        inp = {'games': games, 'selfplay': hr.selfplay_identity(games), 's200_c2': c2, 'sample': games[:6],
               'sample_ids': [], 'by_id': {}, 'row_mask': {}, 'scopes': {}}
        calls = []

        def search(view, root, gs, identity, jsonl, device, workers, log, change=None):
            calls.append((identity['factor'], [g['game_index'] for g in gs]))
            rows = {}
            for g in gs:
                rows[g['game_index']] = [copy.deepcopy(c2[(g['game_index'], p['ply'])]) for p in hr.eligible_plies(g)]
                if identity['factor'] == 4 and g['game_index'] == 2:
                    rows[g['game_index']][0]['visits'][0][2] += 1
            return {'rows': rows, 'wall_seconds': 1.0, 'resumed_games': 0, 'games': len(gs)}
        c = st.load_contract(None, self.root, check_sources=False)[0]
        c['search']['scope_eligible'] = sum(len(hr.eligible_plies(g)) for g in games[5:])
        with mock.patch.object(st, 'load_contract', return_value=(c, 'f' * 64, self.eid())), \
                mock.patch.object(st, 'verified_gate', return_value={}), \
                mock.patch.object(st, 'input_files', return_value=[]), \
                mock.patch.object(st.pre, 'load_inputs', return_value=inp), \
                mock.patch.object(st, 'train_scope', return_value=games[5:]), \
                mock.patch.object(st.pre, 'search_games', side_effect=search):
            out = st.search(self.ns(), self.root, log=QUIET)
        self.assertEqual(out['status'], 'S200_NOT_REPRODUCED')
        self.assertEqual(calls, [(4, [0, 1, 2, 3, 4])])  # the S400 search never started
        self.assertTrue((self.root / st.D / 'search.json').is_file())

    CAND_ROWS = [{'format': 'h7c-policy-row-v2', 'split': 'train', 'mask': False, 'id': 1},
                 {'format': 'h7c-policy-row-v2', 'split': 'train', 'mask': True, 'id': 2},
                 {'format': 'h7c-policy-row-v2', 'split': 'val', 'mask': False, 'id': 3}]

    def candidate_setup(self):
        self.through_prepare()
        self.approve('search', 'train')
        cand = self.root / st.RUNS / 'policy_rows_s400.jsonl.gz'
        st.write_rows(cand, self.CAND_ROWS)
        (self.root / st.RUNS / 's400.jsonl').write_text('{}' + chr(10))
        rows_rec = {'c2_rows_sha256': 'c' * 64, 'candidate_rows_sha256': p1.file_sha256(cand),
                    's400_records_sha256': p1.file_sha256(self.root / st.RUNS / 's400.jsonl')}
        for name in ('gate', 'search', 'rows'):
            self.commit_record(f'{st.D}/{name}.json', {'experiment_sha256': self.eid(), **rows_rec})
        return cand, rows_rec

    def train_with(self, rows_rec, execute):
        with mock.patch.object(st, 'input_files', return_value=[]), \
                mock.patch.object(st, 'verified_gate', return_value={'settings': self.a1_rec['settings']}), \
                mock.patch.object(st, 'verified_rows', return_value=(rows_rec, copy.deepcopy(self.CAND_ROWS))), \
                mock.patch.object(p1, 'check_inputs', return_value={'rows': 'c2 rows'}), \
                mock.patch('hybrid.h7c_train.load_policy_rows', side_effect=AssertionError('read again')), \
                mock.patch('h7r_p1a1.execute', side_effect=execute) as ex:
            return st.train(self.ns(seed=20261111), self.root, log=QUIET), ex

    def test_a_candidate_run_trains_on_the_verified_rows_and_records_them(self):
        cand, rows_rec = self.candidate_setup()
        out, ex = self.train_with(rows_rec, lambda *a, **k: self.fake_run())
        self.assertEqual(ex.call_args.args[4]['rows'], {'train': [self.CAND_ROWS[0]], 'val': [self.CAND_ROWS[2]]})
        self.assertEqual(ex.call_args.args[1:3], ('T', 20261111))
        self.assertEqual(ex.call_args.kwargs['expect_settings'], self.a1_rec['settings'])
        self.assertEqual(out['inputs']['policy_rows_sha256'], rows_rec['candidate_rows_sha256'])
        self.assertEqual(out['inputs']['c2_policy_rows_sha256'], 'c' * 64)
        self.assertEqual(out['inputs']['s400_records_sha256'], rows_rec['s400_records_sha256'])
        self.assertEqual(out['inputs']['policy_rows'], f'{st.RUNS}/policy_rows_s400.jsonl.gz')
        self.assertEqual((out['arm'], out['seed']), ('S', 20261111))

    def test_a_candidate_file_replaced_during_the_run_publishes_nothing(self):
        cand, rows_rec = self.candidate_setup()

        def replace_then_run(*a, **k):
            st.write_rows(cand, [{**self.CAND_ROWS[0], 'id': 99}])
            return self.fake_run()
        with self.assertRaisesRegex(ValueError, 'changed during the run'):
            self.train_with(rows_rec, replace_then_run)
        self.assertFalse((self.root / st.RUNS / 'S-20261111' / 'record.json').exists())

    def test_a_candidate_file_other_than_the_verified_one_is_refused(self):
        cand, rows_rec = self.candidate_setup()
        st.write_rows(cand, [{**self.CAND_ROWS[0], 'id': 98}])  # replaced before the run (the check reads the record)
        with self.assertRaisesRegex(ValueError, 'changed between the start of the run and their check'):
            self.train_with(rows_rec, lambda *a, **k: self.fake_run())

    def test_the_arena_contract_is_a_v2_contract(self):
        from hybrid import h7r_eval_v2 as ev2
        self.approve('prepare', 'gate', 'search', 'train', 'arena')
        rec = {'experiment_sha256': self.eid(), 'verdict': 'CANDIDATE', 'arm': 'S', 'seed': 20261024,
               'candidate': {'checkpoint_sha256': 'a' * 64, 'metadata_sha256': 'b' * 64}}
        self.commit_record(st.run_record_path(20261024), rec)
        body = st.arena_contract(self.ns(), self.root)
        git(self.root, 'add', st.ARENA_CONTRACT)
        git(self.root, 'commit', '-qm', 'arena')
        ev2.load_contract(self.root / st.ARENA_CONTRACT, self.root, check_files=False)
        self.assertEqual(len(body['arms']), 4)
        self.assertEqual({(a['pairing'], a['seed'], a['pairs']) for a in body['arms']},
                         {('O1-dev-a', 9101, 300), ('O1-dev-b', 9102, 300)})
        self.assertEqual(body['models']['T']['identity']['policy_checkpoint_sha256'],
                         self.c['control']['models']['20261024']['checkpoint_sha256'])
        import h7r_dev_suites as ds
        self.assertTrue(ds.check(None, self.root, log=QUIET)['passed'])  # the suites still check beside it

    def test_an_arena_seed_conflict_writes_nothing(self):
        self.approve('prepare', 'gate', 'search', 'train', 'arena')
        rec = {'experiment_sha256': self.eid(), 'verdict': 'CANDIDATE', 'arm': 'S', 'seed': 20261024,
               'candidate': {'checkpoint_sha256': 'a' * 64, 'metadata_sha256': 'b' * 64}}
        self.commit_record(st.run_record_path(20261024), rec)
        other = {'format': 'h7r-eval-contract-v2', 'contract_id': 'OTHER', 'arms': [{'seed': 9101}]}
        self.commit_record('docs/h7r/contracts-v2/OTHER.json', other)
        with self.assertRaisesRegex(ValueError, r'seeds \[9101\] are already used by OTHER.json'):
            st.arena_contract(self.ns(), self.root)
        self.assertFalse((self.root / st.ARENA_CONTRACT).exists())
        git(self.root, 'rm', '-q', 'docs/h7r/contracts-v2/OTHER.json')
        git(self.root, 'commit', '-qm', 'conflict removed')
        st.arena_contract(self.ns(), self.root)  # the next attempt is not blocked by a leftover file
        self.assertTrue((self.root / st.ARENA_CONTRACT).exists())


class RowsFlowTest(RepoBase):
    """rows and verified_rows with real files: stand-in self-play, c2 S200 records, c2 rows and S400 records."""

    def test_rows_then_the_pre_training_check(self):
        import h7c_labels as lab
        from h7r_p1 import digest
        games = stand_in_games()
        c2, rows = c2_records(games), None
        rows = c2_rows(games, c2)
        c = json.loads((self.root / st.CONTRACT).read_text())
        s = c['search']
        files = {k: self.root / s['inputs'][k]['path'] for k in st.SEARCH_KEYS}
        for f in files.values():
            f.parent.mkdir(parents=True, exist_ok=True)
        files['selfplay'].write_text(''.join(json.dumps(g) + chr(10) for g in games))
        for k in ('policy_checkpoint', 'policy_metadata', 'value_checkpoint', 'value_metadata'):
            files[k].write_text(k)
        ident = {'format': hr.FORMAT, 'purpose': 'retarget', 'factor': 4, 'budget': hr.budget(4), 'mode': 'restored',
                 'selfplay': hr.selfplay_identity(games), 'coverage': hr.coverage(games)}
        files['retarget_s200'].write_text(''.join(json.dumps({'identity': ident, 'game_index': g['game_index'], 'rows': [
            c2[(g['game_index'], p['ply'])] for p in hr.eligible_plies(g)]}) + chr(10) for g in games))
        rows_path = self.root / p1.PATHS['cycle_rows']
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(rows_path, 'wt', encoding='utf-8') as f:
            f.write(''.join(json.dumps(r) + chr(10) for r in rows))
        for k, f in files.items():
            s['inputs'][k]['sha256'] = p1.file_sha256(f)
        c['inputs']['cycle_rows']['sha256'] = p1.file_sha256(rows_path)
        s['selfplay_identity'] = hr.selfplay_identity(games)
        s['reproduction_sample']['games'] = 6
        c['approved'] = {k: k in ('prepare', 'gate', 'search') for k in st.STAGES}
        self.write_contract(c)
        git(self.root, 'commit', '-qam', 'stand-in inputs')
        eid = self.eid()
        train_ids = {r['game_id'] for r in rows if r['split'] == 'train'}
        scope = [g for g in games if lab.game_id(g) in train_ids]
        s400 = s400_records(games, c2)
        ident400 = {'purpose': st.PURPOSE, 'factor': 8}
        (self.root / st.RUNS).mkdir(parents=True, exist_ok=True)
        (self.root / st.RUNS / 's400.jsonl').write_text(''.join(json.dumps({'identity': ident400, 'game_index': g['game_index'],
            'rows': [s400[(g['game_index'], p['ply'])] for p in hr.eligible_plies(g)]}) + chr(10) for g in scope))
        self.commit_record(f'{st.D}/search.json', {'experiment_sha256': eid, 'status': 'SEARCHED', 's400': {
            'records_sha256': p1.file_sha256(self.root / st.RUNS / 's400.jsonl'), 'identity_sha256': digest(ident400)}})
        with mock.patch.object(st, 'input_files', return_value=[]):
            out = st.rows(self.ns(), self.root, log=QUIET)
        self.assertEqual(out['check'], 'passed')
        self.assertEqual(out['rows'], len(rows))
        self.assertEqual(out['replacement']['s200_kept_no_s400_mass'], int(lab.game_id(games[1]) in train_ids))
        git(self.root, 'add', f'{st.D}/rows.json')
        git(self.root, 'commit', '-qm', 'rows')
        rec, cand = st.verified_rows(c, eid, self.root)
        self.assertEqual(len(cand), len(rows))
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            st.rows(self.ns(), self.root, log=QUIET)
        cand_path = self.root / st.RUNS / 'policy_rows_s400.jsonl.gz'
        bad = st.read_rows(cand_path)
        i = next(i for i, r in enumerate(bad) if r.get('s400', {}).get('target') == 's400')
        bad[i]['target'][0][2] += 0.01
        st.write_rows(cand_path, bad)
        with self.assertRaisesRegex(ValueError, 'not the recorded one'):
            st.verified_rows(c, eid, self.root)
        empty = self.root / st.RUNS / 's400.jsonl'
        lines = empty.read_text().splitlines()
        first = json.loads(lines[0])
        first['rows'] = []
        empty.write_text(chr(10).join([json.dumps(first), *lines[1:]]) + chr(10))
        with self.assertRaises(ValueError):  # an empty S400 record: refused before any candidate row is used
            st.s400_records(empty, scope, digest(ident400))
        good = [json.dumps({'identity': ident400, 'game_index': g['game_index'], 'rows': [
            s400[(g['game_index'], p['ply'])] for p in hr.eligible_plies(g)]}) for g in scope]
        outside = next(g for g in games if g not in scope) if len(scope) < len(games) else None
        cases = {'twice in the file': good + [good[0]], 'missing for 1 games': good[1:]}
        if outside is not None:
            cases['not a game of the train scope'] = good + [json.dumps({'identity': ident400, 'game_index':
                outside['game_index'], 'rows': []})]
        for message, lines in cases.items():
            with self.subTest(message):
                empty.write_text(chr(10).join(lines) + chr(10))
                with self.assertRaisesRegex(ValueError, message):
                    st.s400_records(empty, scope, digest(ident400))


STUB = """import gzip, hashlib, json, os, signal, subprocess, sys
from pathlib import Path
D = 'docs/h7r/r2/R2-S400-TRAIN'
R = 'runs/h7r/R2-S400-TRAIN'
cmd = sys.argv[1]
log = sys.argv[sys.argv.index('--log-name') + 1] if '--log-name' in sys.argv else None
def kill():  # the whole block dies (power loss, a closed terminal)
    sys.stdout.flush()
    os.killpg(os.getpgid(0), signal.SIGKILL)
if os.environ.get('S5_FORBID'):
    print('STOP: must not run again'); sys.exit(9)
if cmd == 'gate' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
    print('STOP: CUBLAS_WORKSPACE_CONFIG'); sys.exit(7)
if os.environ.get('S5_FAIL') == cmd:
    print('STOP: ValueError: stand-in failure'); sys.exit(1)
if os.environ.get('S5_KILL') == cmd + ':before':
    kill()
c = json.loads(Path(D, 's400-train-contract.json').read_text())
eid = hashlib.sha256(json.dumps({k: v for k, v in c.items() if k != 'approved'}, sort_keys=True,
                                separators=(',', ':')).encode()).hexdigest()
commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
stamp = {'experiment_sha256': eid, 'git': {'commit': commit, 'dirty': False}}
if cmd in ('prepare', 'gate'):
    passed = not (cmd == 'gate' and os.environ.get('S5_GATE_FAIL'))
    body, code = {'passed': passed, 'log_name': log}, 0 if passed else 2
elif cmd == 'search':
    Path(R).mkdir(parents=True, exist_ok=True)
    Path(R, 's200-check.jsonl').write_text('s200' + chr(10))
    if os.environ.get('S5_S200_FAIL'):
        body, code = {'status': 'S200_NOT_REPRODUCED'}, 2
    else:
        Path(R, 's400.jsonl').write_text('s400' + chr(10))
        body, code = {'status': 'SEARCHED',
                      's400': {'records_sha256': hashlib.sha256(Path(R, 's400.jsonl').read_bytes()).hexdigest()}}, 0
else:  # rows: the candidate file first, then the record (the order of the real tool)
    cand = Path(R, 'policy_rows_s400.jsonl.gz')
    if cand.exists():
        print('STOP: ValueError: ' + str(cand) + ' exists: never overwritten'); sys.exit(1)
    with gzip.GzipFile(cand, 'wb', mtime=0) as f:
        f.write(b'{"row": 1}' + bytes([10]))
    if os.environ.get('S5_ROWS_ABORT') == 'kill':
        kill()
    if os.environ.get('S5_ROWS_ABORT') == 'exit':
        print('STOP: ValueError: a read file changed: nothing is published'); sys.exit(1)
    body, code = {'candidate_rows_sha256': hashlib.sha256(cand.read_bytes()).hexdigest()}, 0
Path(D, cmd + '.json').write_text(json.dumps({'format': 'h7r-s400-train-' + cmd + '-v1', **stamp, **body}))
if os.environ.get('S5_TAMPER') == cmd:  # the output the result names changes after the result is written
    out = Path(R, 's400.jsonl' if cmd == 'search' else 'policy_rows_s400.jsonl.gz')
    out.write_bytes(out.read_bytes() + b'x')
if os.environ.get('S5_KILL') == cmd + ':after':
    kill()
print(cmd, code)
sys.exit(code)
"""


class BlockTest(unittest.TestCase):
    """The desktop blocks of the draft (section 9) run by bash on a real temporary repository with a bare origin, the
    real .gitignore and record tool; the training tool is a stand-in."""

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
        (w / 'scripts/h7r_s400_train.py').write_text(STUB)
        (w / '.venv/bin').mkdir(parents=True)
        (w / '.venv/bin/activate').write_text('')
        self.contract = {'format': 'stand-in', 'contract_id': 'R2-S400-TRAIN', 'note': 'block test',
                         'approved': {s: True for s in st.STAGES}}
        (w / 'docs/h7r/r2/R2-S400-TRAIN').mkdir(parents=True)
        (w / 'docs/h7r/r2/R2-S400-TRAIN/s400-train-contract.json').write_text(json.dumps(self.contract))
        hook = w / '.git/hooks/pre-commit'  # S5_KILL_AT_COMMIT: the block dies with its records staged
        hook.write_text('#!/bin/sh' + chr(10) + '[ -z "$S5_KILL_AT_COMMIT" ] || kill -9 0' + chr(10))
        hook.chmod(0o755)
        git(w, 'add', '-A')
        git(w, 'commit', '-qm', 'init')
        git(w, 'remote', 'add', 'origin', str(self.origin))
        git(w, 'push', '-q', 'origin', 'feat/h7-c1')
        doc = (ROOT / 'docs/h7r/r2/R2-S400-TRAIN-draft.md').read_text()
        blocks = re.findall(r'```bash' + chr(10) + r'(.*?)```', doc, re.S)
        self.assertEqual(len(blocks), 4)
        self.blocks = dict(zip(('s5-1', 's5-G', 's5-S', 's5-R'),
                               (b.replace('~/renju-h6', w.as_posix()) for b in blocks)))
        self.D = 'docs/h7r/r2/R2-S400-TRAIN'
        self.R = 'runs/h7r/R2-S400-TRAIN'
        self.cand = f'{self.R}/policy_rows_s400.jsonl.gz'

    def bash(self, name, **env):
        import os
        import subprocess
        e = {k: v for k, v in os.environ.items() if k != 'CUBLAS_WORKSPACE_CONFIG'}
        e = {**e, 'PYTHONDONTWRITEBYTECODE': '1', **env}
        r = subprocess.run(['bash', '-c', self.blocks[name]], cwd=self.work, capture_output=True, text=True, env=e,
                           start_new_session=True)  # a stand-in kill takes the block's process group only
        return r.returncode, r.stdout + r.stderr

    def files(self, rel):
        return git(self.work, 'ls-files', rel).splitlines()

    def test_prepare_then_gate(self):
        code, out = self.bash('s5-G')
        self.assertEqual(code, 1)
        self.assertIn('prepare is not recorded', out)
        code, out = self.bash('s5-1')
        self.assertEqual(code, 0, out)
        code, out = self.bash('s5-G')
        self.assertEqual(code, 0, out)
        self.assertTrue(self.files(f'{self.D}/gate.json'))
        self.assertEqual(len([f for f in self.files(f'{self.D}/stage-logs') if f.endswith('.log')]), 2)
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), git(self.origin, 'rev-parse', 'feat/h7-c1'))
        code, out = self.bash('s5-G')
        self.assertEqual(code, 1)
        self.assertIn('already recorded', out)

    def test_a_failed_gate_is_recorded_and_stops(self):
        self.assertEqual(self.bash('s5-1')[0], 0)
        code, out = self.bash('s5-G', S5_GATE_FAIL='1')
        self.assertEqual(code, 2, out)
        self.assertTrue(self.files(f'{self.D}/gate.json'))
        self.assertIn('gate exit 2', git(self.work, 'log', '-1', '--format=%s'))

    def test_an_interrupted_gate_is_recorded_from_its_result_and_log(self):
        self.assertEqual(self.bash('s5-1')[0], 0)
        L = self.work / 'runs/h7r/R2-S400-TRAIN/logs'
        name = 'gate-AbC123xy.log'
        (L / name).write_text('gate True' + chr(10) + '__EXIT__ 0' + chr(10))
        (self.work / self.D / 'gate.json').write_text(json.dumps(
            {'format': 'h7r-s400-train-gate-v1', 'passed': True, 'log_name': name}))
        git(self.work, 'add', f'{self.D}/gate.json')  # interrupted right after staging
        code, out = self.bash('s5-G', S5_FORBID='1')
        self.assertEqual(code, 0, out)
        self.assertTrue(self.files(f'{self.D}/stage-logs/{name}'))

    def test_a_result_whose_log_disagrees_stops(self):
        self.assertEqual(self.bash('s5-1')[0], 0)
        L = self.work / 'runs/h7r/R2-S400-TRAIN/logs'
        name = 'gate-AbC123xz.log'
        (L / name).write_text('__EXIT__ 0' + chr(10))
        (self.work / self.D / 'gate.json').write_text(json.dumps(
            {'format': 'h7r-s400-train-gate-v1', 'passed': False, 'log_name': name}))
        code, out = self.bash('s5-G', S5_FORBID='1')
        self.assertEqual(code, 1, out)
        self.assertIn('disagree on the exit code', out)


    # --- search and rows: the attempt records (section 9.1) ---------------------------------------------------------

    def attempts(self, stage):
        """The committed attempt records of a stage, oldest first."""
        out = [json.loads((self.work / f).read_text()) for f in self.files(f'{self.D}/stage-logs')
               if f.endswith('.attempt.json') and Path(f).name.startswith(stage + '-')]
        # a closed attempt comes before the one that closed it (the start times have one-second resolution)
        return sorted(out, key=lambda a: (a['started'], a['finish'].get('closed_by') is None))

    def sha(self, rel):
        return p1.file_sha256(self.work / rel)

    def head(self):
        return git(self.work, 'rev-parse', 'HEAD')

    def through(self, *blocks):
        for b in blocks:
            code, out = self.bash(b)
            self.assertEqual(code, 0, out)

    def assert_pushed(self):
        self.assertEqual(self.head(), git(self.origin, 'rev-parse', 'feat/h7-c1'))
        self.assertEqual(git(self.work, 'status', '--porcelain', '--untracked-files=no'), '')

    def test_search_then_rows(self):
        self.through('s5-1', 's5-G')
        code, out = self.bash('s5-R')
        self.assertEqual(code, 1)
        self.assertIn('search is not recorded', out)
        before = self.head()
        self.through('s5-S')
        [a] = self.attempts('search')
        self.assertEqual((a['stage'], a['head'], a['experiment_sha256']),
                         ('search', before, p1.experiment_sha256(self.contract)))
        self.assertEqual(a['contract_file_sha256'], self.sha(f'{self.D}/s400-train-contract.json'))
        f = a['finish']
        self.assertEqual((f['state'], f['exit_code'], f['recovered']), ('exited', 0, False))
        self.assertEqual(f['result_sha256'], self.sha(f'{self.D}/search.json'))
        self.assertEqual(f['log_sha256'], self.sha(f"{self.D}/stage-logs/{a['name']}.log"))
        self.assertEqual(f['outputs'], {f'{self.R}/s200-check.jsonl': self.sha(f'{self.R}/s200-check.jsonl'),
                                        f'{self.R}/s400.jsonl': self.sha(f'{self.R}/s400.jsonl')})
        self.assertIn('search exit 0', git(self.work, 'log', '-1', '--format=%s'))
        self.assert_pushed()
        code, out = self.bash('s5-S')
        self.assertEqual(code, 1)
        self.assertIn('already recorded', out)
        self.through('s5-R')
        [r] = self.attempts('rows')
        self.assertEqual(r['set_aside'], [])
        self.assertEqual(r['finish']['outputs'], {self.cand: self.sha(self.cand)})
        self.assertEqual(r['finish']['result_sha256'], self.sha(f'{self.D}/rows.json'))
        self.assert_pushed()

    def test_s200_not_reproduced_is_recorded_and_rows_refuse(self):
        self.through('s5-1', 's5-G')
        code, out = self.bash('s5-S', S5_S200_FAIL='1')
        self.assertEqual(code, 2, out)
        self.assertIn('search exit 2', git(self.work, 'log', '-1', '--format=%s'))
        [a] = self.attempts('search')
        self.assertEqual(a['finish']['exit_code'], 2)
        self.assert_pushed()
        before = self.head()
        code, out = self.bash('s5-R')
        self.assertEqual(code, 1)
        self.assertIn('search ended S200_NOT_REPRODUCED', out)
        self.assertEqual(self.head(), before)
        self.assertEqual(list((self.work / self.R / 'logs').glob('rows-*')), [])

    def test_interrupted_after_staging_is_recorded_without_a_new_run(self):
        self.through('s5-1', 's5-G')
        before = self.head()
        code, out = self.bash('s5-S', S5_KILL_AT_COMMIT='1')
        self.assertNotEqual(code, 0, out)
        self.assertEqual(self.head(), before)
        self.assertIn(f'{self.D}/search.json', git(self.work, 'diff', '--cached', '--name-only'))
        code, out = self.bash('s5-S', S5_FORBID='1')
        self.assertEqual(code, 0, out)
        [a] = self.attempts('search')
        self.assertEqual((a['finish']['exit_code'], a['finish']['recovered']), (0, False))
        self.assertTrue(self.files(f'{self.D}/search.json'))
        self.assert_pushed()

    def test_killed_after_the_result_is_recovered_from_the_open_attempt(self):
        self.through('s5-1', 's5-G')
        code, out = self.bash('s5-S', S5_KILL='search:after')
        self.assertNotEqual(code, 0, out)
        [open_] = [json.loads(p.read_text()) for p in (self.work / self.R / 'logs').glob('search-*.attempt.json')]
        self.assertIsNone(open_['finish'])
        code, out = self.bash('s5-S', S5_FORBID='1')
        self.assertEqual(code, 0, out)
        [a] = self.attempts('search')
        self.assertEqual((a['name'], a['finish']['exit_code'], a['finish']['recovered']), (open_['name'], 0, True))
        log = (self.work / self.D / 'stage-logs' / f"{a['name']}.log").read_text().splitlines()
        self.assertEqual(log[-2:], [f'Recovered exit code from {self.D}/search.json', '__EXIT__ 0'])
        self.assert_pushed()

    def test_killed_before_the_result_is_closed_and_recorded_with_the_next_run(self):
        self.through('s5-1', 's5-G')
        code, out = self.bash('s5-S', S5_KILL='search:before')
        self.assertNotEqual(code, 0, out)
        self.assertFalse((self.work / self.D / 'search.json').exists())
        self.through('s5-S')
        old, new = self.attempts('search')
        self.assertEqual((old['finish']['state'], old['finish']['exit_code'], old['finish']['closed_by']),
                         ('interrupted', None, new['name']))
        log = (self.work / self.D / 'stage-logs' / f"{old['name']}.log").read_text().splitlines()
        self.assertTrue(log[-1].startswith('__INTERRUPTED__'))
        self.assertEqual((new['finish']['state'], new['finish']['exit_code']), ('exited', 0))
        self.assert_pushed()

    def candidate_only(self, abort):
        """An interrupted rows run: the candidate file written, rows.json not."""
        self.through('s5-1', 's5-G', 's5-S')
        code, out = self.bash('s5-R', S5_ROWS_ABORT=abort)
        self.assertNotEqual(code, 0, out)
        self.assertTrue((self.work / self.cand).exists())
        self.assertFalse((self.work / self.D / 'rows.json').exists())
        return self.sha(self.cand)

    def check_set_aside(self, left_sha):
        before, staged = self.head(), git(self.work, 'diff', '--cached', '--name-only')
        code, out = self.bash('s5-R')
        self.assertEqual(code, 1)
        self.assertIn('an interrupted rows run left', out)
        self.assertIn(f'export S5_SET_ASIDE={left_sha[:12]}', out)
        self.assertEqual((self.head(), git(self.work, 'diff', '--cached', '--name-only'), self.sha(self.cand)),
                         (before, staged, left_sha))  # nothing changed
        code, out = self.bash('s5-R', S5_SET_ASIDE='0' * 12)
        self.assertEqual(code, 1)
        self.assertEqual(self.sha(self.cand), left_sha)
        code, out = self.bash('s5-R', S5_SET_ASIDE=left_sha[:12])
        self.assertEqual(code, 0, out)
        [new] = [a for a in self.attempts('rows') if a['set_aside']]
        older = [a for a in self.attempts('rows') if not a['set_aside']]
        [aside] = new['set_aside']
        self.assertEqual(aside['to'], f"{self.R}/interrupted/{new['name']}.policy_rows_s400.jsonl.gz")
        self.assertEqual((aside['from'], aside['sha256'], self.sha(aside['to'])), (self.cand, left_sha, left_sha))
        self.assertEqual(aside['left_by'], older[-1]['name'])
        self.assertEqual(older[-1]['finish']['outputs'], {self.cand: left_sha})
        self.assertEqual(new['finish']['same_as_set_aside'], {self.cand: True})
        self.assertEqual(new['finish']['outputs'], {self.cand: left_sha})
        self.assertTrue(self.files(f'{self.D}/rows.json'))
        self.assert_pushed()
        return older, new

    def test_candidate_only_after_a_kill_is_kept_and_set_aside_on_request(self):
        left = self.candidate_only('kill')
        [old], new = self.check_set_aside(left)
        self.assertEqual((old['finish']['state'], old['finish']['closed_by']), ('interrupted', new['name']))

    def test_candidate_only_after_a_failed_publish_is_kept_and_set_aside_on_request(self):
        left = self.candidate_only('exit')
        self.assertIn('rows exit 1', git(self.work, 'log', '-1', '--format=%s'))  # the failure is recorded
        [old], new = self.check_set_aside(left)
        self.assertEqual((old['finish']['state'], old['finish']['exit_code']), ('exited', 1))

    def test_a_partial_temporary_file_is_set_aside(self):
        self.through('s5-1', 's5-G', 's5-S')
        tmp = self.work / f'{self.cand}.tmp'
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(b'partial')
        self.through('s5-R')
        [r] = self.attempts('rows')
        [aside] = r['set_aside']
        self.assertEqual((aside['from'], aside['left_by']), (f'{self.cand}.tmp', None))
        self.assertEqual((self.work / aside['to']).read_bytes(), b'partial')
        self.assertFalse(tmp.exists())

    OUTPUT = {'search': 'runs/h7r/R2-S400-TRAIN/s400.jsonl',
              'rows': 'runs/h7r/R2-S400-TRAIN/policy_rows_s400.jsonl.gz'}
    BLOCK = {'search': 's5-S', 'rows': 's5-R'}

    def before(self, stage):
        self.through('s5-1', 's5-G', *(('s5-S',) if stage == 'rows' else ()))

    def check_output_changed_after_the_run(self, stage):
        self.before(stage)
        before = self.head()
        code, out = self.bash(self.BLOCK[stage], S5_TAMPER=stage)
        self.assertEqual(code, 1, out)
        self.assertIn(f"{self.OUTPUT[stage]}: its SHA-256 is not the one {self.D}/{stage}.json records", out)
        self.assertEqual((self.head(), git(self.work, 'diff', '--cached', '--name-only')), (before, ''))
        [a] = [json.loads(p.read_text()) for p in (self.work / self.R / 'logs').glob(f'{stage}-*.attempt.json')]
        self.assertIsNone(a['finish'])  # no success recorded
        code, out = self.bash(self.BLOCK[stage], S5_FORBID='1')  # the next run recovers it: refused again
        self.assertEqual(code, 1, out)
        self.assertIn('its SHA-256 is not the one', out)
        self.assertEqual((self.head(), git(self.work, 'diff', '--cached', '--name-only')), (before, ''))

    def test_search_output_changed_after_the_run_is_not_recorded(self):
        self.check_output_changed_after_the_run('search')

    def test_rows_output_changed_after_the_run_is_not_recorded(self):
        self.check_output_changed_after_the_run('rows')

    def check_output_gone_before_the_commit(self, stage):
        self.before(stage)
        before = self.head()
        code, out = self.bash(self.BLOCK[stage], S5_KILL_AT_COMMIT='1')
        self.assertNotEqual(code, 0, out)
        staged = git(self.work, 'diff', '--cached', '--name-only')
        self.assertIn(f'{self.D}/{stage}.json', staged)
        (self.work / self.OUTPUT[stage]).unlink()
        code, out = self.bash(self.BLOCK[stage], S5_FORBID='1')
        self.assertEqual(code, 1, out)
        self.assertIn(f'{self.OUTPUT[stage]} is missing: {self.D}/{stage}.json records SHA-256', out)
        self.assertEqual((self.head(), git(self.work, 'diff', '--cached', '--name-only')), (before, staged))
        self.assertEqual(git(self.origin, 'rev-parse', 'feat/h7-c1'), before)

    def test_search_output_gone_before_the_commit_is_not_recovered(self):
        self.check_output_gone_before_the_commit('search')

    def test_rows_output_gone_before_the_commit_is_not_recovered(self):
        self.check_output_gone_before_the_commit('rows')

    def test_a_running_block_is_refused(self):
        import fcntl

        self.through('s5-1', 's5-G')
        lock = self.work / self.R / 'logs/search.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('w') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            code, out = self.bash('s5-S')
        self.assertEqual(code, 1)
        self.assertIn('another search block is running', out)
        self.assertFalse((self.work / self.D / 'search.json').exists())


class AttemptToolTest(unittest.TestCase):
    """``scripts/h7r_attempts.py`` refusals the blocks rely on."""

    def setUp(self):
        import h7r_attempts as att

        self.att = att
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        w = self.root = Path(self.tmp.name)
        git(w, 'init', '-q')
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(w, 'config', k, v)
        self.c = {'contract_id': 'R2-S400-TRAIN', 'x': 1, 'approved': {'search': True}}
        (w / 'c.json').write_text(json.dumps(self.c))
        git(w, 'add', '-A')
        git(w, 'commit', '-qm', 'init')
        (w / 'logs').mkdir()
        self.log = w / 'logs/search-Ab12Cd34.log'
        self.log.write_text('run' + chr(10))
        self.assertEqual(att.main(['--root', str(w), 'start', '--stage', 'search', '--log', 'logs/search-Ab12Cd34.log',
                                   '--result', 'search.json', '--contract', 'c.json',
                                   '--output', f'{st.RUNS}/s200-check.jsonl', '--output', self.S400]), 0)
        self.path = w / 'logs/search-Ab12Cd34.attempt.json'

    S400 = f'{st.RUNS}/s400.jsonl'

    def result(self, **over):
        body = {'format': 'h7r-s400-train-search-v1', 'status': 'SEARCHED',
                'experiment_sha256': p1.experiment_sha256(self.c), 'git': {'commit': git(self.root, 'rev-parse', 'HEAD')}}
        (self.root / 'search.json').write_text(json.dumps({**body, **over}))

    def test_the_experiment_id_is_the_tool_s(self):
        real = json.loads((ROOT / st.D / 's400-train-contract.json').read_text(encoding='utf-8'))
        self.assertEqual(self.att.experiment_sha256(real), p1.experiment_sha256(real))
        self.assertTrue(self.att.experiment_sha256(real).startswith('67c6ba3c8081'))
        self.assertEqual(self.att.load(self.path)['experiment_sha256'], p1.experiment_sha256(self.c))

    def test_finish_refuses_inconsistent_ends(self):
        with self.log.open('a') as f:
            f.write('__EXIT__ 0' + chr(10))
        with self.assertRaisesRegex(ValueError, 'did not produce its result'):
            self.att.finish(self.root, self.path, 0)
        with self.assertRaisesRegex(ValueError, 'last line is not __EXIT__ 1'):
            self.att.finish(self.root, self.path, 1)
        self.result(status='S200_NOT_REPRODUCED')
        with self.assertRaisesRegex(ValueError, 'means exit 2'):
            self.att.finish(self.root, self.path, 0)
        self.result(git={'commit': '0' * 40})
        with self.assertRaisesRegex(ValueError, "not of this attempt's experiment and HEAD"):
            self.att.finish(self.root, self.path, 0)
        self.assertIsNone(self.att.load(self.path)['finish'])

    def s400(self, data=b's400'):
        path = self.root / self.S400
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def test_finish_checks_the_outputs_the_result_names(self):
        with self.log.open('a') as f:
            f.write('__EXIT__ 0' + chr(10))
        recorded = self.s400()
        self.s400(b'changed')
        self.result(s400={'records_sha256': recorded})
        with self.assertRaisesRegex(ValueError, f'{self.S400}: its SHA-256 is not the one search.json records'):
            self.att.finish(self.root, self.path, 0)
        (self.root / self.S400).unlink()
        with self.assertRaisesRegex(ValueError, f'{self.S400} is missing'):
            self.att.finish(self.root, self.path, 0)
        self.result()  # SEARCHED without the S400 records' SHA-256
        with self.assertRaisesRegex(ValueError, f'no SHA-256 for {self.S400}'):
            self.att.finish(self.root, self.path, 0)
        self.assertIsNone(self.att.load(self.path)['finish'])
        self.result(s400={'records_sha256': self.s400()})
        f = self.att.finish(self.root, self.path, 0)['finish']
        self.assertEqual((f['exit_code'], f['outputs'][self.S400]), (0, recorded))

    def test_a_not_reproduced_search_names_no_output(self):
        with self.log.open('a') as f:
            f.write('__EXIT__ 2' + chr(10))
        self.result(status='S200_NOT_REPRODUCED')
        self.assertEqual(self.att.finish(self.root, self.path, 2)['finish']['exit_code'], 2)

    def test_a_named_output_must_be_an_output_of_the_attempt(self):
        log = self.root / 'logs/rows-Ab12Cd34.log'
        log.write_text('run' + chr(10) + '__EXIT__ 0' + chr(10))
        self.assertEqual(self.att.main(['--root', str(self.root), 'start', '--stage', 'rows', '--log',
                                        'logs/rows-Ab12Cd34.log', '--result', 'rows.json', '--contract', 'c.json']), 0)
        cand = self.root / st.RUNS / 'policy_rows_s400.jsonl.gz'
        cand.parent.mkdir(parents=True, exist_ok=True)
        cand.write_bytes(b'rows')
        (self.root / 'rows.json').write_text(json.dumps({
            'format': 'h7r-s400-train-rows-v1', 'experiment_sha256': p1.experiment_sha256(self.c),
            'git': {'commit': git(self.root, 'rev-parse', 'HEAD')}, 'candidate_rows_sha256': p1.file_sha256(cand)}))
        with self.assertRaisesRegex(ValueError, f'rows.json names {st.RUNS}/policy_rows_s400.jsonl.gz, not an output '
                                                'of attempt rows-Ab12Cd34'):
            self.att.finish(self.root, self.root / 'logs/rows-Ab12Cd34.attempt.json', 0)

    def test_recover_checks_the_outputs_of_a_finished_attempt(self):
        from types import SimpleNamespace

        with self.log.open('a') as f:
            f.write('__EXIT__ 0' + chr(10))
        self.result(s400={'records_sha256': self.s400()})
        self.att.finish(self.root, self.path, 0)
        args = SimpleNamespace(stage='search', log_dir='logs', result='search.json')
        self.assertEqual(self.att.recover(args, self.root), ('search-Ab12Cd34', 0))
        (self.root / self.S400).unlink()
        with self.assertRaisesRegex(ValueError, f'{self.S400} is missing'):
            self.att.recover(args, self.root)
        self.s400(b'changed')
        with self.assertRaisesRegex(ValueError, 'its SHA-256 is not the one'):
            self.att.recover(args, self.root)

    def test_recover_needs_the_attempt_that_wrote_the_result(self):
        from types import SimpleNamespace

        args = SimpleNamespace(stage='search', log_dir='logs', result='search.json')
        self.result(experiment_sha256='f' * 64)
        with self.assertRaisesRegex(ValueError, '0 open attempts'):
            self.att.recover(args, self.root)
        self.result()
        with self.log.open('a') as f:
            f.write('__EXIT__ 1' + chr(10))
        with self.assertRaisesRegex(ValueError, 'disagree on the exit code'):
            self.att.recover(args, self.root)
        self.assertIsNone(self.att.load(self.path)['finish'])

    def test_start_never_runs_over_a_result(self):
        self.result()
        log = self.root / 'logs/search-Zz99Yy88.log'
        log.write_text('')
        self.assertEqual(self.att.main(['--root', str(self.root), 'start', '--stage', 'search', '--log',
                                        'logs/search-Zz99Yy88.log', '--result', 'search.json', '--contract',
                                        'c.json']), 1)
        self.assertIsNone(self.att.load(self.path)['finish'])  # the open attempt is left for recover


if __name__ == '__main__':
    unittest.main()
