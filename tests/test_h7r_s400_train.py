"""scripts/h7r_s400_train.py: the contract (A1's unchanged parts, the control checkpoints, v2 Arena seeds); the
candidate rows (only train visit targets replaced, the S200 target kept without S400 mass, no row added, every other
row and field unchanged, a deterministic file) and their check; the T reproduction gate (exact, before any search);
the search order (S200 check, then S400; a difference stops); the provenance of a candidate run; the v2 Arena
contract. The engine search and the training are stand-ins; the rules and the records are the tool's."""
import copy
import gzip
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
                         '6c579b5ba682c1fe780aec5c1e0d47f921e420c72096da37ef95a81fbcda60d5')
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

    def test_a_candidate_run_records_its_rows(self):
        self.through_prepare()
        self.approve('search', 'train')
        cand = self.root / st.RUNS / 'policy_rows_s400.jsonl.gz'
        cand.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cand, 'wt') as f:
            f.write('{}\n')
        (self.root / st.RUNS / 's400.jsonl').write_text('{}\n')
        rows_rec = {'c2_rows_sha256': 'c' * 64, 'candidate_rows_sha256': p1.file_sha256(cand),
                    's400_records_sha256': p1.file_sha256(self.root / st.RUNS / 's400.jsonl')}
        self.commit_record(f'{st.D}/gate.json', {'experiment_sha256': self.eid(), 'settings': self.a1_rec['settings']})
        self.commit_record(f'{st.D}/rows.json', {'experiment_sha256': self.eid(), **rows_rec})
        loaded = {'train': ['candidate rows'], 'val': []}
        with mock.patch.object(st, 'input_files', return_value=[]), \
                mock.patch.object(st, 'verified_gate', return_value={'settings': self.a1_rec['settings']}), \
                mock.patch.object(st, 'verified_rows', return_value=(rows_rec, [])), \
                mock.patch.object(p1, 'check_inputs', return_value={'rows': 'c2 rows'}), \
                mock.patch('hybrid.h7c_train.load_policy_rows', return_value=loaded) as load, \
                mock.patch('h7r_p1a1.execute', return_value=self.fake_run()) as ex:
            out = st.train(self.ns(seed=20261111), self.root, log=QUIET)
        self.assertEqual(load.call_args.args[0], self.root / st.RUNS / 'policy_rows_s400.jsonl.gz')
        self.assertIs(ex.call_args.args[4]['rows'], loaded)  # trained on the candidate rows
        self.assertEqual(ex.call_args.args[1:3], ('T', 20261111))
        self.assertEqual(ex.call_args.kwargs['expect_settings'], self.a1_rec['settings'])
        self.assertEqual(out['inputs']['policy_rows_sha256'], rows_rec['candidate_rows_sha256'])
        self.assertEqual(out['inputs']['c2_policy_rows_sha256'], 'c' * 64)
        self.assertEqual(out['inputs']['s400_records_sha256'], rows_rec['s400_records_sha256'])
        self.assertEqual(out['inputs']['policy_rows'], f'{st.RUNS}/policy_rows_s400.jsonl.gz')
        self.assertEqual((out['arm'], out['seed']), ('S', 20261111))

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


if __name__ == '__main__':
    unittest.main()
