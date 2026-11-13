"""scripts/h7r_s400_pre.py: the metrics by hand; the contract rules; prepare and measure on a real temporary repository
with stand-in self-play games, c2 policy rows and c2 S200 records (the engine search faked): the fixed sample, the early
and the whole-sample S200 reproduction checks stopping before S400, the denominators, the bootstrap and the time
estimates, nothing overwritten."""
import copy
import gzip
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from hybrid import h7c_retarget as hr
from hybrid.h6_labels import canonical_hash, split_for
import h7r_p1 as p1
import h7r_s400_pre as pre
from tests.test_h7r_p1 import git

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / 'docs/h7r/r2/R2-S400-PRE/s400-pre-contract.draft.json'


class MetricTest(unittest.TestCase):
    def test_by_hand(self):
        a = pre.distribution([[0, 0, 30, 0.1], [0, 1, 10, 0.0], [1, 1, 0, 0.0]])
        b = pre.distribution([[0, 0, 20, 0.1], [0, 1, 20, 0.0]])
        self.assertEqual(a, {(0, 0): 0.75, (0, 1): 0.25})
        self.assertEqual(pre.top1(b), (0, 0))  # tie: row-major first
        self.assertAlmostEqual(pre.tv(a, b), 0.25)
        self.assertAlmostEqual(pre.entropy(b), 1.0)
        want = 0.5 * (0.75 * math.log2(0.75 / 0.625) + 0.25 * math.log2(0.25 / 0.375)) + \
            0.5 * (0.5 * math.log2(0.5 / 0.625) + 0.5 * math.log2(0.5 / 0.375))
        self.assertAlmostEqual(pre.js(a, b), want)
        self.assertEqual(pre.js(a, a), 0.0)
        self.assertEqual(pre.pair_metrics(a, b)['top1_changed'], 0.0)
        self.assertEqual([pre.loss_relation(x, y) for x, y in (([[1, 1]], [[1, 1]]), ([[1, 1], [2, 2]], [[1, 1]]),
                                                               ([], [[3, 3]]), ([[1, 1]], [[2, 2]]))],
                         ['same', 'first_only', 'second_only', 'both_unique'])
        edges = [[1, 10], [11, 20], [21, 40], [41, None]]
        self.assertEqual([pre.bucket(p, edges) for p in (0, 9, 10, 40, 99)], ['1-10', '1-10', '11-20', '41-', '41-'])

    def test_sample_order_is_fixed(self):
        ids = {f'h7c:c1:8801:{i}' for i in range(30)}
        s = pre.select_sample(ids, 'r2-s400-pre-v1', 10)
        self.assertEqual(s, pre.select_sample(set(sorted(ids, reverse=True)), 'r2-s400-pre-v1', 10))
        self.assertEqual(s, sorted(s, key=lambda g: pre.game_key('r2-s400-pre-v1', g)))
        self.assertNotEqual(s, pre.select_sample(ids, 'another-salt', 10))
        with self.assertRaisesRegex(ValueError, 'train games'):
            pre.select_sample(ids, 'x', 31)

    def test_bootstrap_by_game(self):
        series = {'m': [(0, 1.0), (0, 1.0), (1, 0.0)]}
        out = pre.bootstrap_means(series, [0, 1], 2000, 1)
        lo, hi = out['m']['ci95']
        self.assertEqual((lo, hi), (0.0, 1.0))  # a draw of only game 1 gives 0, only game 0 gives 1
        self.assertEqual(out, pre.bootstrap_means(series, [0, 1], 2000, 1))  # seeded


def contract_draft():
    c = json.loads(DRAFT.read_text())
    return {**c, 'status': 'fixed', 'sources': [], 'evidence': {}}


class ContractTest(unittest.TestCase):
    def test_rules(self):
        c = contract_draft()
        self.assertEqual(c['approved'], {'prepare': False, 'measure': False})
        self.assertEqual(c['budgets']['s400']['budget'], {'simulations': 400, 'tactical_simulations': 800})
        self.assertEqual(c['reproduction']['mean_value_tolerance'], hr.Q_TOLERANCE)
        ret = json.loads((ROOT / 'docs/h7c/c2/retarget_s200.json').read_text())
        self.assertEqual(c['selfplay_identity'], ret['identity']['selfplay'])
        self.assertEqual(c['inputs']['retarget_s200']['sha256'], ret['jsonl_sha256'])
        self.assertEqual((c['time_reference']['c2_s200_wall_seconds'], c['time_reference']['eligible']),
                         (8857.0, 20910))
        with TemporaryDirectory() as tmp, mock.patch.object(pre, 'committed', return_value=True):
            path = Path(tmp) / 'c.json'
            for message, bad in (
                    ('measure approved before prepare', {**c, 'approved': {'prepare': False, 'measure': True}}),
                    ('budgets', {**c, 'budgets': {**c['budgets'], 's400': {**c['budgets']['s400'], 'factor': 4}}}),
                    ('sample sizes', {**c, 'sample': {**c['sample'], 'early_check_games': 100}})):
                with self.subTest(message):
                    path.write_text(json.dumps(bad))
                    with self.assertRaisesRegex(ValueError, message):
                        pre.load_contract(path, ROOT)
            path.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, 'prepare stage is not approved'):
                pre.load_contract(path, ROOT, 'prepare')


class FixedContractTest(unittest.TestCase):
    def test_the_fixed_contract_is_its_draft_with_sources(self):
        fixed = json.loads((ROOT / pre.CONTRACT).read_text())
        draft = json.loads(DRAFT.read_text())
        self.assertEqual(fixed['approved'], {'prepare': False, 'measure': False})
        self.assertEqual(fixed['draft'], {'path': DRAFT.relative_to(ROOT).as_posix(), 'sha256': p1.file_sha256(DRAFT)})
        self.assertEqual({k: v for k, v in fixed.items() if k not in ('status', 'sources', 'draft', 'approved')},
                         {k: v for k, v in draft.items() if k not in ('status', 'sources', 'approved')})
        self.assertIn('scripts/h7r_s400_pre.py', {s['path'] for s in fixed['sources']})
        pre.load_contract(None, ROOT)  # sources and evidence are this checkout's (and the file is committed)


class FlowTest(unittest.TestCase):
    """prepare and measure on a temporary repository: 30 stand-in games (cycle 1), their c2-style policy rows (one
    masked), a c2-style S200 file; the search is faked by ``fake_search``."""

    GAMES, SAMPLE, EARLY = 30, 10, 3

    def setUp(self):
        import h7c_labels as lab
        from model.checkpoint import save_checkpoint
        from model.config import ModelConfig
        from tests.test_h7c_labels import game
        from tests.test_h7c_train import small_model

        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.root = Path(self.tmp.name)
        git(root, 'init', '-q', '-b', 'feat/h7-c1')
        for k, v in (('user.email', 't@t'), ('user.name', 't')):
            git(root, 'config', k, v)
        shutil.copy(ROOT / '.gitignore', root / '.gitignore')
        self.games = [game(i, game_seed=100 + i) for i in range(self.GAMES)]
        c = contract_draft()
        files = {k: root / v['path'] for k, v in c['inputs'].items()}
        for f in files.values():
            f.parent.mkdir(parents=True, exist_ok=True)
        files['selfplay'].write_text(''.join(json.dumps(g) + '\n' for g in self.games))
        save_checkpoint(files['policy_checkpoint'], small_model())
        cfg = ModelConfig()
        files['policy_metadata'].write_text(json.dumps({'policy_trained': True, 'value_trained': False,
                                                        'config': {'channels': cfg.channels, 'blocks': cfg.blocks}}))
        files['value_checkpoint'].write_bytes(b'value')
        files['value_metadata'].write_text('{}')
        ident = {'format': hr.FORMAT, 'purpose': 'retarget', 'factor': 4, 'budget': hr.budget(4), 'mode': 'restored',
                 'selfplay_jsonl_sha256': p1.file_sha256(files['selfplay']), 'selfplay': hr.selfplay_identity(self.games),
                 'coverage': hr.coverage(self.games)}
        self.c2 = {}
        lines = []
        for g in self.games:
            rows = []
            for p in hr.eligible_plies(g):
                row = {'ply': p['ply'], 'color': p['color'], 'seed': 7, 'canonical_hash': canonical_hash(g['moves'][:p['ply']]),
                       'route': 'tree', 'engine_move': p['engine_move'],
                       'visits': [[*p['engine_move'], 150, 0.2], [3, 3, 50, 0.0]], 'proven_loss': [], 'e4': {},
                       'tree_budget': 200, 'seconds': 0.5, 'cooldown_restored': False}
                rows.append(row)
                self.c2[(g['game_index'], p['ply'])] = row
            lines.append(json.dumps({'identity': ident, 'game_index': g['game_index'], 'rows': rows}))
        files['retarget_s200'].write_text('\n'.join(lines) + '\n')
        provenance = lambda g: {'engine_config_hash': g['engine_hash'], 'model_hash': g['model_hash'],  # noqa: E731
                                'git_commit': 'x', 'h7c_provenance': {}}
        retarget = {'rows': self.c2, 'info': {'sha256': 'f' * 64}}
        policy = [r for g in self.games for r in lab.policy_rows(
            g, split_for(lab.game_id(g), fractions=lab.SPLIT_FRACTIONS, salt=lab.SPLIT_SALT), provenance(g),
            retarget=retarget)]
        self.train_ids = {r['game_id'] for r in policy if r['split'] == 'train'}
        self.assertGreaterEqual(len(self.train_ids), self.SAMPLE)
        sample = pre.select_sample(self.train_ids, c['sample']['salt'], self.SAMPLE)
        self.masked = (sample[1], next(r['ply'] for r in policy if r['game_id'] == sample[1] and r['kind'] == 'visits'))
        for r in policy:
            if (r['game_id'], r['ply']) == self.masked:
                r['mask'], r['flags'] = True, ['probe_position']
        with gzip.open(files['policy_rows'], 'wt', encoding='utf-8') as handle:
            handle.write(''.join(json.dumps(r) + '\n' for r in policy))
        for k, f in files.items():
            c['inputs'][k]['sha256'] = p1.file_sha256(f)
        c['selfplay_identity'] = hr.selfplay_identity(self.games)
        c['sample'] = {**c['sample'], 'games': self.SAMPLE, 'early_check_games': self.EARLY}
        c['time_reference'] = {**c['time_reference'], 'eligible': sum(len(hr.eligible_plies(g)) for g in self.games)}
        c['device'] = 'cpu'
        self.c = c
        self.write_contract(c)
        git(root, 'add', '-A')
        git(root, 'commit', '-qm', 'contract')
        self.assertEqual(git(root, 'ls-files', 'runs'), '')
        engine = mock.patch('h7c_retarget.check_engine')
        engine.start()
        self.addCleanup(engine.stop)
        self.calls = []

    def write_contract(self, c):
        path = self.root / pre.CONTRACT
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(c, indent=1))

    def approve(self, stage):
        c = json.loads((self.root / pre.CONTRACT).read_text())
        c['approved'][stage] = True
        self.write_contract(c)
        git(self.root, 'commit', '-qam', f'approve {stage}')

    def ns(self):
        import argparse
        return argparse.Namespace(contract=None, device='cpu')

    def fake_search(self, change200=None):
        """S200 gives the c2 record (changed by ``change200``); S400 another visit split."""
        def search(c, root, games, identity, jsonl, device, workers, log):
            self.calls.append((identity['factor'], [g['game_index'] for g in games]))
            rows = {}
            for g in games:
                out = []
                for p in hr.eligible_plies(g):
                    row = copy.deepcopy(self.c2[(g['game_index'], p['ply'])])
                    if identity['factor'] == 4 and change200:
                        change200(g, p, row)
                    if identity['factor'] == 8:
                        row['visits'] = [[*p['engine_move'], 300, 0.25], [3, 3, 300, 0.1], [4, 4, 200, 0.0]]
                        row['tree_budget'] = 400
                        row['seconds'] = 0.9
                    out.append(row)
                rows[g['game_index']] = out
            return {'rows': rows, 'wall_seconds': 10.0 * identity['factor'] / 4 * len(games) / self.SAMPLE,
                    'resumed_games': 0, 'games': len(games)}
        return search

    def through_prepare(self):
        self.approve('prepare')
        rec = pre.prepare(self.ns(), self.root)
        git(self.root, 'add', f'{pre.D}/prepare.json')
        git(self.root, 'commit', '-qm', 'prepare')
        self.approve('measure')
        return rec

    def test_prepare_then_measure(self):
        rec = self.through_prepare()
        s = rec['sample']
        self.assertEqual(s['game_ids'], pre.select_sample(self.train_ids, self.c['sample']['salt'], self.SAMPLE))
        self.assertEqual(s['scopes']['full_eligible'], self.c['time_reference']['eligible'])
        self.assertEqual(s['scopes']['train_games'], len(self.train_ids))
        with mock.patch.object(pre, 'search_games', side_effect=self.fake_search()):
            out = pre.measure(self.ns(), self.root, log=lambda *a, **k: None)
        self.assertEqual(out['status'], 'MEASURED')
        self.assertEqual([f for f, _ in self.calls], [4, 4, 8])  # early S200, whole S200, then S400
        self.assertEqual(len(self.calls[0][1]), self.EARLY)
        self.assertEqual(self.calls[2][1], s['game_indices'])
        self.assertTrue(out['s200_reproduction']['sample']['reproduced'])
        d = out['denominators']
        self.assertEqual(d['positions'], s['coverage']['eligible'])
        self.assertEqual(d['masked'], 1)
        self.assertEqual(d['common_positions'], d['unmasked'])
        self.assertEqual((d['s400_only_target_unmasked'], d['s400_only_target_mask_unknown']), (0, 0))
        self.assertNotIn('s400_only_target', d)
        t = out['tables']
        # S50 -> S200 and S200 -> S400 by hand: actor 40/10, S200 150/50, S400 300/300/200 over the actor's move
        self.assertAlmostEqual(t['raw.s50_s200.tv']['mean'], 0.05)
        self.assertAlmostEqual(t['raw.s200_s400.tv']['mean'], 0.375)
        moves = [tuple(p['engine_move']) for g in self.games if g['game_index'] in s['game_indices']
                 for p in hr.eligible_plies(g)]
        changed = sum((3, 3) < m for m in moves) / len(moves)  # S400 ties the move with (3, 3): row-major first
        self.assertAlmostEqual(t['raw.s200_s400.top1_changed']['mean'], round(changed, 6))
        self.assertIsNotNone(t['target.common.kl_h3.s400']['ci95_game_bootstrap'])
        self.assertEqual(set(t['raw.s50_s200.tv']['by_bucket']), {'1-10'})
        self.assertEqual(out['proven_loss_sets']['s200_s400'], {'same': d['positions']})
        tm = out['timing']
        self.assertAlmostEqual(tm['s400_over_s200_search_seconds'], 0.9 / 0.5)
        ref = self.c['time_reference']
        self.assertAlmostEqual(tm['estimates']['full_re_search']['ratio_method_seconds'],
                               round(ref['c2_s200_wall_seconds'] * 1.8, 1))
        by_id = {f"h7c:c1:8801:{g['game_index']}": g for g in self.games}
        self.assertEqual(tm['estimates']['training_data']['positions'],
                         sum(len(hr.eligible_plies(by_id[i])) for i in self.train_ids))
        self.assertIn('extrapolated_seconds', tm['estimates']['full_re_search'])
        self.assertTrue((self.root / pre.RUNS / 'positions.jsonl.gz').is_file())
        self.assertEqual(git(self.root, 'ls-files', 'runs'), '')
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            pre.measure(self.ns(), self.root, log=lambda *a, **k: None)

    def test_an_early_s200_difference_stops_before_the_rest_and_s400(self):
        self.through_prepare()

        def change(g, p, row):
            row['visits'][0][2] += 1
        with mock.patch.object(pre, 'search_games', side_effect=self.fake_search(change)):
            out = pre.measure(self.ns(), self.root, log=lambda *a, **k: None)
        self.assertEqual(out['status'], 'S200_EARLY_NOT_REPRODUCED')
        self.assertEqual([f for f, _ in self.calls], [4])
        self.assertIn('visits', out['s200_reproduction']['early']['differing_fields'])
        self.assertNotIn('tables', out)

    def test_a_later_s200_difference_stops_before_s400(self):
        rec = self.through_prepare()
        late = rec['sample']['game_indices'][-1]

        def change(g, p, row):
            if g['game_index'] == late:
                row['visits'][0][3] += 1e-4  # beyond the G0 mean-value tolerance
        with mock.patch.object(pre, 'search_games', side_effect=self.fake_search(change)):
            out = pre.measure(self.ns(), self.root, log=lambda *a, **k: None)
        self.assertEqual(out['status'], 'S200_NOT_REPRODUCED')
        self.assertEqual([f for f, _ in self.calls], [4, 4])
        self.assertTrue(out['s200_reproduction']['early']['reproduced'])
        self.assertEqual(out['s200_reproduction']['sample']['differing_fields'], {'mean_value': len(
            hr.eligible_plies(self.games[late]))})

    def test_mean_values_within_the_g0_tolerance_reproduce(self):
        self.through_prepare()

        def change(g, p, row):
            row['visits'][1][3] += 5e-6
        with mock.patch.object(pre, 'search_games', side_effect=self.fake_search(change)):
            out = pre.measure(self.ns(), self.root, log=lambda *a, **k: None)
        self.assertEqual(out['status'], 'MEASURED')

    def resumed_records(self, change):
        """A S200 record file holding the early sample games, each changed by ``change(rows)``."""
        rec = self.through_prepare()
        c, _, eid = pre.load_contract(None, self.root, check_sources=False)
        ident = pre.search_identity(c, eid, 's200', rec['sample']['sample_sha256'], hr.selfplay_identity(self.games))
        path = self.root / pre.RUNS / 's200.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for gi in rec['sample']['game_indices'][:self.EARLY]:
            g = self.games[gi]
            rows = [copy.deepcopy(self.c2[(gi, p['ply'])]) for p in hr.eligible_plies(g)]
            lines.append(json.dumps({'identity': ident, 'game_index': gi, 'rows': change(rows)}))
        path.write_text('\n'.join(lines) + '\n')
        return rec

    def test_a_resumed_record_with_empty_rows_stops_before_any_comparison(self):
        self.resumed_records(lambda rows: [])
        with mock.patch('h7c_retarget.ProcessPoolExecutor', side_effect=AssertionError('searched')):
            with self.assertRaisesRegex(ValueError, 'not its eligible plies'):
                pre.measure(self.ns(), self.root, log=lambda *a, **k: None)
        self.assertFalse((self.root / pre.D / 'measure.json').exists())
        self.assertFalse((self.root / pre.RUNS / 's400.jsonl').exists())

    def test_a_resumed_record_with_a_nan_mean_value_stops(self):
        def nan(rows):
            rows[0]['visits'][0][3] = float('nan')
            return rows
        self.resumed_records(nan)
        with mock.patch('h7c_retarget.ProcessPoolExecutor', side_effect=AssertionError('searched')):
            with self.assertRaisesRegex(ValueError, 'not a finite number'):
                pre.measure(self.ns(), self.root, log=lambda *a, **k: None)

    def test_the_reproduction_needs_every_position_and_finite_values(self):
        g = self.games[0]
        rows = [copy.deepcopy(self.c2[(0, p['ply'])]) for p in hr.eligible_plies(g)]
        full = pre.check_reproduction([g], {0: rows}, self.c2)
        self.assertTrue(full['reproduced'])
        self.assertEqual(full['compared'], full['expected'])
        empty = pre.check_reproduction([g], {0: []}, self.c2)
        self.assertFalse(empty['reproduced'])
        self.assertEqual(empty['differing_fields'], {'not_compared': len(rows)})
        self.assertFalse(pre.check_reproduction([g], {}, self.c2)['reproduced'])
        bad = copy.deepcopy(rows)
        bad[0]['visits'][0][3] = float('nan')
        self.assertEqual(pre.check_reproduction([g], {0: bad}, self.c2)['differing_fields'], {'non_finite': 1})

    def test_the_sample_is_rejudged_and_inputs_are_checked(self):
        self.through_prepare()
        path = self.root / pre.D / 'prepare.json'
        bad = json.loads(path.read_text())
        bad['sample']['game_ids'] = bad['sample']['game_ids'][::-1]
        path.write_text(json.dumps(bad))
        git(self.root, 'commit', '-qam', 'tampered')
        with mock.patch.object(pre, 'search_games', side_effect=AssertionError('searched')):
            with self.assertRaisesRegex(ValueError, "not the committed prepare record"):
                pre.measure(self.ns(), self.root, log=lambda *a, **k: None)
        rows = self.root / self.c['inputs']['policy_rows']['path']
        with gzip.open(rows, 'at', encoding='utf-8') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'input policy_rows'):
            pre.load_inputs(self.c, self.root, 'cpu')



def _fast5():
    from rt30 import fast5
    return fast5.available()


@unittest.skipUnless(_fast5(), 'python -m rt30.vcfcore.build first')
class EngineTest(unittest.TestCase):
    """Real TB-V3 / fast5 agents on random stand-in networks (CPU): a real S200 re-search through h7c_retarget stands in
    for the c2 records; the tool's own S200 search must reproduce it, and its S400 search passes the per-row checks."""

    def test_s200_reproduces_a_real_re_search_and_s400_runs(self):
        import argparse
        import h7c_retarget as rt
        import h7c_selfplay as sp
        from analysis import tb_v3_net
        from hybrid.h6_fast import install
        from rt30 import fast5
        from tests.test_tb_v3_net import write_models

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            policy, value = write_models(tmp / 'm', seed=5)
            sp._init_worker(str(policy), str(value), 'cpu')
            self.addCleanup(lambda: (fast5.uninstall(), install('off')))
            models = tb_v3_net.model_identity(policy, value)
            games = []
            for i in range(2):
                task = {'identity': {**sp.run_identity(2, 8802, models, 'cpu'),
                                     'generator_git': {'commit': 'x', 'dirty': False}},
                        'game_index': i, 'game_seed': sp.game_seed(8802, 2, i),
                        'opening_seed': sp.opening_seed(8802, 2, i)}
                games.append(sp.play_game(task, max_plies=10))
            selfplay = hr.selfplay_identity(games)
            quiet = lambda *a, **k: None  # noqa: E731
            c2_ident = {'format': hr.FORMAT, 'purpose': 'retarget', 'factor': 4, 'budget': hr.budget(4),
                        'mode': 'restored', 'selfplay': selfplay}
            c2_args = argparse.Namespace(jsonl=tmp / 'c2.jsonl', workers=1, device='cpu', policy_checkpoint=policy,
                                         value_checkpoint=value)
            c2 = rt._execute(c2_args, games, c2_ident, ('restored',), 4, quiet)
            c2_rows = hr.index_rows(games, [{'identity': c2_ident, 'game_index': i, 'rows': r['rows']}
                                            for i, r in c2.items()], c2_ident)
            self.assertGreater(len(c2_rows), 0)
            c = {'mode': 'restored', 'inputs': {'policy_checkpoint': {'path': str(policy)},
                                                'value_checkpoint': {'path': str(value)}},
                 'budgets': {'s200': {'factor': 4}, 's400': {'factor': 8}}}
            got = {}
            for b in ('s200', 's400'):
                ident = {**c2_ident, 'purpose': pre.PURPOSE, 'factor': c['budgets'][b]['factor'],
                         'budget': hr.budget(c['budgets'][b]['factor']), 'sample_sha256': 's' * 64}
                got[b] = pre.search_games(c, '/', games, ident, tmp / f'{b}.jsonl', 'cpu', 1, quiet)
            repro = pre.check_reproduction(games, got['s200']['rows'], c2_rows)
            self.assertTrue(repro['reproduced'], repro)
            self.assertEqual(repro['compared'], len(c2_rows))
            rows400 = [r for g in got['s400']['rows'].values() for r in g]
            self.assertTrue(all(r['tree_budget'] in (400, 800) for r in rows400))
            again = pre.search_games(c, '/', games, {**c2_ident, 'purpose': pre.PURPOSE, 'factor': 8,
                                                      'budget': hr.budget(8), 'sample_sha256': 's' * 64},
                                     tmp / 's400.jsonl', 'cpu', 1, quiet)
            self.assertEqual(again['resumed_games'], 2)  # resumed from the file, nothing searched again



STUB = """import json, os, sys
from pathlib import Path
D = 'docs/h7r/r2/R2-S400-PRE'
cmd = sys.argv[1]
log = sys.argv[sys.argv.index('--log-name') + 1]
if os.environ.get('S4_FORBID'):
    print('STOP: must not run again'); sys.exit(9)
if os.environ.get('S4_FAIL') == cmd:
    print('STOP: ValueError: stand-in failure'); sys.exit(1)
Path(D).mkdir(parents=True, exist_ok=True)
if cmd == 'prepare':
    Path(D, 'prepare.json').write_text(json.dumps({'format': 'h7r-s400-pre-prepare-v1', 'passed': True,
                                                   'log_name': log}))
    print('ok prepare'); sys.exit(0)
status = os.environ.get('S4_STATUS', 'MEASURED')
Path(D, 'measure.json').write_text(json.dumps({'format': 'h7r-s400-pre-measure-v1', 'status': status, 'log_name': log}))
print('status', status)
sys.exit(0 if status == 'MEASURED' else 2)
"""


class BlockTest(unittest.TestCase):
    """The desktop blocks of the plan (section 9) run by bash on a real temporary repository with a bare origin, the
    real .gitignore and record tool; the measurement tool is a stand-in."""

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
        (w / 'scripts/h7r_s400_pre.py').write_text(STUB)
        (w / '.venv/bin').mkdir(parents=True)
        (w / '.venv/bin/activate').write_text('')
        git(w, 'add', '-A')
        git(w, 'commit', '-qm', 'init')
        git(w, 'remote', 'add', 'origin', str(self.origin))
        git(w, 'push', '-q', 'origin', 'feat/h7-c1')
        doc = (ROOT / 'docs/h7r/r2/R2-S400-PRE-draft.md').read_text()
        blocks = re.findall(r'```bash\n(.*?)```', doc, re.S)
        self.assertEqual(len(blocks), 2)
        self.blocks = dict(zip(('s4-1', 's4-2'), (b.replace('~/renju-h6', str(w)) for b in blocks)))
        self.D = 'docs/h7r/r2/R2-S400-PRE'

    def bash(self, name, **env):
        import os
        import subprocess
        e = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', **env}
        r = subprocess.run(['bash', '-c', self.blocks[name]], cwd=self.work, capture_output=True, text=True, env=e)
        return r.returncode, r.stdout + r.stderr

    def files(self, rel):
        return git(self.work, 'ls-files', rel).splitlines()

    def test_prepare_then_measure(self):
        code, out = self.bash('s4-2')
        self.assertEqual(code, 1)
        self.assertIn('prepare is not recorded', out)
        code, out = self.bash('s4-1')
        self.assertEqual(code, 0, out)
        self.assertTrue(self.files(f'{self.D}/prepare.json'))
        code, out = self.bash('s4-2')
        self.assertEqual(code, 0, out)
        self.assertTrue(self.files(f'{self.D}/measure.json'))
        self.assertEqual(len([f for f in self.files(f'{self.D}/stage-logs') if f.endswith('.log')]), 2)
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), git(self.origin, 'rev-parse', 'feat/h7-c1'))
        code, out = self.bash('s4-2')
        self.assertEqual(code, 1)
        self.assertIn('already recorded', out)

    def test_a_stop_is_recorded_and_a_failure_is_logged(self):
        import time
        self.assertEqual(self.bash('s4-1')[0], 0)
        code, out = self.bash('s4-2', S4_FAIL='measure')
        self.assertEqual(code, 1, out)
        self.assertFalse(self.files(f'{self.D}/measure.json'))
        self.assertIn('measure exit 1', git(self.work, 'log', '-1', '--format=%s'))
        time.sleep(1.1)
        code, out = self.bash('s4-2', S4_STATUS='S200_NOT_REPRODUCED')
        self.assertEqual(code, 2, out)
        self.assertTrue(self.files(f'{self.D}/measure.json'))
        self.assertIn('measure exit 2', git(self.work, 'log', '-1', '--format=%s'))

    def result_without_commit(self, status='S200_EARLY_NOT_REPRODUCED', exit_line='__EXIT__ 2', log=True):
        """A measure run whose result (and log) exist, interrupted before its records commit."""
        L = self.work / 'runs/h7r/R2-S400-PRE/logs'
        L.mkdir(parents=True, exist_ok=True)
        name = 'measure-AbC123xy.log'
        if log:
            (L / name).write_text('status ' + status + chr(10) + (exit_line + chr(10) if exit_line else ''))
        (self.work / self.D / 'measure.json').write_text(json.dumps(
            {'format': 'h7r-s400-pre-measure-v1', 'status': status, 'log_name': name}))
        return name

    def test_an_uncommitted_result_is_recorded_without_a_rerun(self):
        self.assertEqual(self.bash('s4-1')[0], 0)
        name = self.result_without_commit()
        code, out = self.bash('s4-2', S4_FORBID='1')
        self.assertEqual(code, 2, out)
        self.assertIn('no new run', out)
        self.assertTrue(self.files(f'{self.D}/measure.json'))
        self.assertTrue(self.files(f'{self.D}/stage-logs/{name}'))

    def test_a_result_staged_before_an_interruption_is_recorded(self):
        self.assertEqual(self.bash('s4-1')[0], 0)
        name = self.result_without_commit(status='MEASURED', exit_line='__EXIT__ 0')
        git(self.work, 'add', f'{self.D}/measure.json')  # interrupted right after staging
        code, out = self.bash('s4-2', S4_FORBID='1')
        self.assertEqual(code, 0, out)
        self.assertTrue(self.files(f'{self.D}/stage-logs/{name}'))
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), git(self.origin, 'rev-parse', 'feat/h7-c1'))

    def test_a_result_without_its_log_or_with_another_exit_code_stops(self):
        self.assertEqual(self.bash('s4-1')[0], 0)
        head = git(self.work, 'rev-parse', 'HEAD')
        self.result_without_commit(log=False)
        code, out = self.bash('s4-2', S4_FORBID='1')
        self.assertEqual(code, 1, out)
        self.assertIn("the result's log is missing", out)
        self.result_without_commit(exit_line='__EXIT__ 0')  # the result says 2
        code, out = self.bash('s4-2', S4_FORBID='1')
        self.assertEqual(code, 1, out)
        self.assertIn('disagree on the exit code', out)
        self.assertEqual(git(self.work, 'rev-parse', 'HEAD'), head)  # nothing committed

    def test_a_log_cut_before_its_exit_line_is_recovered_from_the_result(self):
        self.assertEqual(self.bash('s4-1')[0], 0)
        name = self.result_without_commit(exit_line=None)
        code, out = self.bash('s4-2', S4_FORBID='1')
        self.assertEqual(code, 2, out)
        text = (self.work / self.D / 'stage-logs' / name).read_text()
        self.assertIn('Recovered exit code from', text)
        self.assertTrue(text.rstrip().endswith('__EXIT__ 2'))

    def test_an_unrelated_staged_file_stops(self):
        self.assertEqual(self.bash('s4-1')[0], 0)
        self.result_without_commit()
        (self.work / 'other.txt').write_text('x')
        git(self.work, 'add', 'other.txt')
        code, out = self.bash('s4-2', S4_FORBID='1')
        self.assertEqual(code, 1, out)
        self.assertIn('unrelated staged path: other.txt', out)


if __name__ == '__main__':
    unittest.main()
