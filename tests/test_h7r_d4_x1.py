"""R2-D4 X1 post-hoc description: root Q validity and perspective, the prediction join through the dense cache index,
quartile ties and empty cells, concentration, repeated histories, seed overlap through shared branches, split totals,
provenance without the current HEAD as run commit, and an end-to-end pass on a real wrapper run with untouched inputs."""
import contextlib
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hybrid import h7r_d4 as d4
from scripts import h7r_d4_x1 as x1
from tests.test_h7r_d4_full_check import Full
from tests.test_h7r_d4_smoke_check import tree_hashes

R2_SHA = 'r' * 64


def row(compare='seed31001:c5', game=0, ply=10, color='black', outcome='equal', delta=0.0, move_a=(7, 7),
        move_b=(7, 8), history=((7, 7),), game_id=None):
    return {'compare': compare, 'model': compare.split(':')[0], 'game_index': game,
            'game_id': game_id or f'h7c:c1:1:{game}', 'ply': ply, 'color': color, 'history': tuple(history),
            'move_a': list(move_a), 'move_b': list(move_b),
            'key_a': f'a:{history}:{move_a}', 'key_b': f'b:{history}:{move_b}', 'outcome': outcome, 'delta': delta}


def tree(visits):
    return {'route': 'tree', 'engine_move': [0, 0], 'visits': visits}


def predictions_for(info, n_values=None, **change):
    n = len(info['ply'])
    pred = {'contract_sha256': R2_SHA, 'game': torch.tensor(info['game'], dtype=torch.int32),
            'ply': torch.tensor(info['ply'], dtype=torch.int16), 'route': [True] * n,
            'models': {m: {'raw': torch.zeros(n, dtype=torch.float64),
                           'd4': torch.arange(n, dtype=torch.float64) * (i + 1) / 100}
                       for i, m in enumerate(x1.PREDICTION_MODELS)}}
    pred.update(change)
    return pred


class RootQTest(unittest.TestCase):
    def test_visits_zero_default_q_is_never_a_value(self):
        # PUCTNode.mean_value is 0.0 for an unvisited child: that 0 must not become a dQ.
        rec = tree([[7, 7, 0, 0.0], [7, 8, 30, -0.2]])
        self.assertEqual(x1.within_model_q(rec, [7, 7], [7, 8]), {'status': 'visits_zero', 'which': 'candidate_move'})
        rec = tree([[7, 7, 0, 0.0], [7, 8, 0, 0.0]])
        self.assertEqual(x1.within_model_q(rec, [7, 7], [7, 8])['which'], 'both')
        self.assertIsNone(x1.own_q(tree([[7, 7, 0, 0.0]]), [7, 7]))

    def test_one_move_missing_and_non_tree(self):
        rec = tree([[7, 8, 30, -0.2]])
        self.assertEqual(x1.within_model_q(rec, [7, 7], [7, 8]), {'status': 'missing', 'which': 'candidate_move'})
        self.assertEqual(x1.within_model_q(rec, [7, 8], [1, 1]), {'status': 'missing', 'which': 'c5_move'})
        self.assertEqual(x1.within_model_q({'route': 'vcf', 'visits': [[7, 8, 3, .1]]}, [7, 8], [7, 8]),
                         {'status': 'non_tree'})

    def test_valid_q_difference_visit_share_and_precision(self):
        rec = tree([[7, 7, 30, 0.123457], [7, 8, 10, 0.1], [3, 3, 0, 0.0]])
        res = x1.within_model_q(rec, [7, 7], [7, 8])
        self.assertEqual(res['status'], 'valid')
        self.assertEqual(res['dq'], 0.023457)  # rounded to the recorded 6 decimals
        self.assertAlmostEqual(res['share_diff'], (30 - 10) / 40)  # positive visits of that root only

    def test_black_and_white_keep_the_root_mover_perspective(self):
        rec = tree([[7, 7, 30, 0.3], [7, 8, 10, -0.1]])
        sets = {'seed31001': rec, 'c5': rec, 'seed31002': rec, 'seed31003': rec, 'c8': rec}
        rows = [row(ply=10, color='black', game=0), row(ply=11, color='white', game=1)]
        plies = {(0, 10): {'sets': sets}, (1, 11): {'sets': sets}}
        out = x1.q_analysis(rows, plies)['seed31001:c5']['candidate']
        self.assertEqual([r['q']['candidate']['dq'] for r in rows], [0.4, 0.4])  # no sign flip by colour
        self.assertEqual((out['valid'], out['dq']['positive']), (2, 2))

    def test_coverage_counts_add_up_and_cross_model_is_distribution_only(self):
        good = tree([[7, 7, 30, 0.3], [7, 8, 10, -0.1]])
        missing = tree([[7, 8, 10, -0.1]])
        zero = tree([[7, 7, 0, 0.0], [7, 8, 10, -0.1]])
        cases = [good, missing, zero, {'route': 'vcf', 'engine_move': [7, 8], 'visits': []}]
        rows, plies = [], {}
        for i, rec in enumerate(cases):
            rows.append(row(game=i))
            plies[(i, 10)] = {'sets': {'seed31001': rec, 'c5': good}}
        out = x1.q_analysis(rows, plies)['seed31001:c5']
        self.assertEqual(out['candidate']['valid'], 1)
        self.assertEqual(out['candidate']['excluded'], {'missing:candidate_move': 1, 'non_tree': 1,
                                                        'visits_zero:candidate_move': 1})
        self.assertEqual(out['c5']['valid'], 4)
        self.assertEqual(out['cross_model_q_diff']['count'] + out['cross_model_q_diff']['excluded'], 4)
        self.assertNotIn('cells', out['cross_model_q_diff'])


class PredictionJoinTest(unittest.TestCase):
    def setUp(self):
        # Original game indexes 3, 7, 12: the dense cache index 0, 1, 2 is not the game index.
        self.manifest = {'rows': 6, 'games': [{'game_id': f'h7c:c1:1:{i}', 'game_index': i, 'plies': [5, 6]}
                                              for i in (3, 7, 12)]}
        self.info = {'game_ids': [f'h7c:c1:1:{i}' for i in (3, 7, 12)], 'game': [0, 0, 1, 1, 2, 2],
                     'ply': [5, 6, 5, 6, 5, 6]}
        self.routes = ['network'] * 6

    def join(self, pred, info=None):
        return x1.checked_predictions(pred, info or self.info, self.routes, self.manifest, R2_SHA)

    def test_dense_index_maps_to_the_original_game(self):
        values, facts = self.join(predictions_for(self.info))
        self.assertEqual(facts['rows'], 6)
        self.assertEqual(values[('h7c:c1:1:7', 6)]['c5'], 3 / 100)  # row 3 is game 7 (dense 1), ply 6
        eligible = [(gi, p, 'black') for gi in (3, 7, 12) for p in (5, 6)]
        rows = [row(compare=c, game=7, ply=6, game_id='h7c:c1:1:7') for c in x1.ANALYSIS['comparisons']]
        game_ids = {e['game_index']: e['game_id'] for e in self.manifest['games']}
        out = x1.value_change(eligible, rows, values, game_ids)
        self.assertAlmostEqual(rows[0]['dv_abs'], abs(values[('h7c:c1:1:7', 6)]['seed31001'] - 3 / 100))
        self.assertEqual(sum(c['disagreements'] for c in out['seed31001:c5']['cells'].values()), 1)

    def test_missing_duplicate_misjoined_and_foreign_rows_fail(self):
        info = self.info
        cases = {
            'missing': predictions_for({**info, 'game': info['game'][:5], 'ply': info['ply'][:5]}),
            'duplicate': predictions_for({**info, 'ply': [5, 5, 5, 6, 5, 6]}),
            'original index used': predictions_for({**info, 'game': [3, 3, 7, 7, 12, 12]}),
            'out of order': predictions_for({**info, 'game': [1, 1, 0, 0, 2, 2]}),
            'another contract': predictions_for(info, contract_sha256='x' * 64),
        }
        bad_model = predictions_for(info)
        bad_model['models']['c6'] = bad_model['models'].pop('c8')
        cases['another model set'] = bad_model
        nan = predictions_for(info)
        nan['models']['c5']['d4'][2] = float('nan')
        cases['non-finite'] = nan
        for name, pred in cases.items():
            with self.subTest(name), self.assertRaises(x1.Failure):
                self.join(pred)

    def test_duplicate_or_extra_rows_in_the_cache_view_fail(self):
        info = {**self.info, 'ply': [5, 5, 5, 6, 5, 6]}
        with self.assertRaisesRegex(x1.Failure, 'twice'):
            self.join(predictions_for(info), info)
        info = {**self.info, 'ply': [5, 7, 5, 6, 5, 6]}
        with self.assertRaisesRegex(x1.Failure, 'eligible rows differ'):
            self.join(predictions_for(info), info)


class QuartileTest(unittest.TestCase):
    def test_linear_interpolation(self):
        self.assertEqual(x1.quantiles([1, 2, 3, 4, 5]), [2, 3, 4])
        self.assertEqual(x1.quantiles([0, 10]), [2.5, 5, 7.5])
        self.assertEqual(x1.quantiles([4]), [4, 4, 4])
        self.assertIsNone(x1.quantiles([]))

    def test_equal_values_give_duplicate_boundaries_and_empty_cells(self):
        items = [(0.5, row(outcome='better', delta=1.0)), (0.5, row(outcome='worse', delta=-1.0)),
                 (0.5, row())]
        out = x1.quartile_cells(items)
        self.assertEqual(out['boundaries']['duplicates'], ['q25=q50', 'q50=q75'])
        self.assertEqual(out['empty_cells'], ['q25<x<=q50', 'q50<x<=q75', 'x>q75'])
        self.assertEqual(out['cells']['x<=q25']['n'], 3)  # no tie splitting, no jitter
        self.assertEqual(out['cells']['x>q75']['mean_delta'], None)
        empty = x1.quartile_cells([])
        self.assertEqual((empty['boundaries']['q50'], len(empty['empty_cells'])), (None, 4))

    def test_cells_add_up_to_the_population(self):
        items = [(float(i % 7), row(outcome=('better', 'worse', 'equal')[i % 3], delta=(1.0, -1.0, 0.0)[i % 3]))
                 for i in range(50)]
        out = x1.quartile_cells(items)
        self.assertEqual(sum(c['n'] for c in out['cells'].values()), 50)
        self.assertEqual(sum(c['better'] for c in out['cells'].values()), x1.tally([r for _, r in items])['better'])


class ConcentrationTest(unittest.TestCase):
    def test_net_zero_top_games_order_and_null_share(self):
        rows = [row(game=5, outcome='better', delta=1.0), row(game=5, ply=11, outcome='worse', delta=-1.0),
                row(game=2, outcome='better', delta=0.5), row(game=9, outcome='worse', delta=-0.5),
                row(game=4, outcome='equal')]
        c = x1.concentration(rows)['seed31001:c5']
        self.assertEqual(c['overall']['net'], 0)
        self.assertEqual([g['game_index'] for g in c['top_games']['list']], [5, 2, 9])  # count, then game index
        self.assertEqual((c['top_games']['net'], c['top_games']['decisive_share']), (0, 1.0))
        self.assertEqual(c['top_games']['delta_sum'], 0.0)
        self.assertEqual(c['decisive_games'], 3)
        self.assertEqual(c['per_game_max']['decisive'], 2)
        self.assertNotIn('net_share', c['top_games'])  # no ratio over the overall net
        none = x1.concentration([row(outcome='equal')])['seed31001:c5']
        self.assertIsNone(none['top_games']['decisive_share'])
        self.assertEqual(none['top_games']['list'], [])

    def test_repeated_histories(self):
        h = ((7, 7), (7, 8))
        rows = [row(game=0, history=h), row(game=1, history=h, outcome='better', delta=1.0),
                row(game=2, history=h, move_a=(1, 1)), row(game=3, history=((0, 0),))]
        rep = x1.concentration(rows)['seed31001:c5']['repeated_history']
        self.assertEqual((rep['distinct_histories'], rep['repeated_histories'], rep['observations']), (2, 1, 3))
        self.assertEqual((rep['max_repeats'], rep['decisive']), (3, 1))
        self.assertEqual(rep['observations_with_identical_branch_pair'], 2)


class SeedOverlapTest(unittest.TestCase):
    def test_shared_branches_give_automatic_agreement_split_by_candidate_move(self):
        rows = [row('seed31001:c5', game=0, outcome='better', delta=1.0),
                row('seed31002:c5', game=0, outcome='better', delta=1.0),  # same candidate: both branches shared
                row('seed31001:c5', game=1, outcome='worse', delta=-1.0),
                row('seed31002:c5', game=1, move_a=(1, 1), outcome='equal'),  # another candidate move
                row('seed31003:c5', game=2)]
        rows[3]['key_a'] = 'other'
        out = x1.seed_overlap(rows)
        pair = out['pairs']['seed31001:c5|seed31002:c5']
        self.assertEqual((pair['common_positions'], pair['same_candidate_move'], pair['different_candidate_move']),
                         (2, 1, 1))
        self.assertEqual(pair['shared_branches'], {'c5_branch': 2, 'candidate_branch': 1})
        agree = pair['outcome_agreement']
        self.assertEqual((agree['same_candidate_move']['agree'], agree['different_candidate_move']['agree']), (1, 0))
        self.assertEqual(agree['different_candidate_move']['table'], {'worse|equal': 1})
        self.assertEqual(out['union']['positions'], 3)
        self.assertEqual(out['union']['by_seed_count'], {'1': 1, '2': 2, '3': 0})

    def test_a_shared_branch_pair_with_two_outcomes_is_an_error(self):
        rows = [row('seed31001:c5', outcome='better', delta=1.0), row('seed31002:c5', outcome='worse', delta=-1.0)]
        with self.assertRaisesRegex(x1.Failure, 'two outcomes'):
            x1.seed_overlap(rows)


class PlySideTest(unittest.TestCase):
    def test_cells_add_up_to_the_whole_and_zero_denominators_are_null(self):
        eligible = [(0, p, 'black' if p % 2 == 0 else 'white') for p in range(5, 100)]
        rows = [row(ply=6), row(ply=21, color='white', outcome='better', delta=1.0),
                row(ply=90, outcome='worse', delta=-0.5)]
        out = x1.ply_side(eligible, rows)['seed31001:c5']
        self.assertEqual(out['total']['disagreements'], 3)
        self.assertEqual(sum(c['eligible'] for c in out['cells'].values()), len(eligible))
        self.assertEqual(out['cells']['[80,inf)|black']['mean_delta'], -0.5)
        self.assertIsNone(out['cells']['[20,40)|black']['mean_delta'])
        empty = x1.ply_side([(0, 6, 'black')], [])['c8:c5']
        self.assertIsNone(empty['cells']['[5,20)|white']['rate'])
        with self.assertRaises(x1.Failure):
            x1.ply_bin(4)


class ProvenanceTest(unittest.TestCase):
    def repo(self, tmp):
        root = Path(tmp)
        def git(*a):
            return subprocess.run(['git', *a], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        git('init', '-q'); git('config', 'user.email', 't@t'); git('config', 'user.name', 't')
        for name in ('contract.json', 'approval.json', 'eligible.json', 'src.py'):
            (root / name).write_text(name)
        git('add', '.'); git('commit', '-qm', 'run')
        run = git('rev-parse', 'HEAD')
        (root / 'later.txt').write_text('x'); git('add', '.'); git('commit', '-qm', 'x1')  # HEAD != run commit
        plan = {'run_commit': run, 'run_sources': {'src.py': d4.file_sha256(root / 'src.py')},
                'inputs': {k: {'path': f'{k}.json', 'sha256': d4.file_sha256(root / f'{k}.json')}
                           for k in ('contract', 'approval', 'eligible')}}
        return root, plan, git

    def test_run_links_hold_at_a_later_head_and_catch_edits(self):
        with TemporaryDirectory() as tmp:
            root, plan, git = self.repo(tmp)
            self.assertEqual(x1.check_run_links(plan, root)['run_commit'], plan['run_commit'])
            (root / 'src.py').write_text('changed'); git('commit', '-qam', 'edit')
            changed = {**plan, 'run_sources': {'src.py': d4.file_sha256(root / 'src.py')}}
            with self.assertRaisesRegex(x1.Failure, 'run commit has another version'):
                x1.check_run_links(changed, root)
            with self.assertRaisesRegex(x1.Failure, 'checkout differs'):
                x1.check_run_links(plan, root)
            with self.assertRaisesRegex(x1.Failure, 'not an ancestor'):
                x1.check_run_links({**plan, 'run_commit': '0' * 40}, root)

    def test_absent_input_stops_instead_of_narrowing(self):
        with TemporaryDirectory() as tmp:
            plan = {'inputs': {k: {'path': f'{k}.bin', 'sha256': '0' * 64} for k in x1.INPUTS}}
            for k in x1.INPUTS[:-1]:
                (Path(tmp) / f'{k}.bin').write_text(k)
            with self.assertRaisesRegex(x1.Failure, r"absent: \['predictions.bin'\]"):
                x1.input_shas(plan, Path(tmp))
            (Path(tmp) / 'predictions.bin').write_text('p')
            with self.assertRaisesRegex(x1.Failure, 'differs from the plan'):
                x1.input_shas(plan, Path(tmp))

    def test_committed_plan_matches_the_tool(self):
        root = Path(__file__).resolve().parents[1]
        plan = json.loads((root / x1.PLAN).read_text())
        self.assertEqual((plan['status'], plan['nature'], plan['analysis']), ('fixed', 'post_hoc_exploratory',
                                                                              x1.ANALYSIS))
        self.assertEqual(tuple(plan['inputs']), x1.INPUTS)
        self.assertTrue(all(len(v['sha256']) == 64 for v in plan['inputs'].values()))
        self.assertEqual(len(plan['run_commit']), 40)
        self.assertTrue(plan['output']['rows'].startswith('runs/'))


class EndToEndTest(unittest.TestCase):
    """The real wrapper's full run on the synthetic fixture, rebuilt and described by X1."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.run_ = Full(Path(self.tmp.name))
        self.summary = self.run_.run()
        r, folder = self.run_, self.run_.folder
        rel = lambda p: str(Path(p).relative_to(r.root))  # noqa: E731
        self.plan = {'run_commit': 'fixture', 'inputs': {
            'identity': {'path': rel(folder / 'identity.json')}, 'summary': {'path': rel(folder / 'summary.json')},
            'replay': {'path': rel(folder / 'replay.jsonl')}, 'branches': {'path': rel(folder / 'branches.jsonl')},
            'eligible': {'path': rel(r.sample)}}}
        self.shas = {'contract': 's', 'approval': d4.file_sha256(r.approval), 'eligible': d4.file_sha256(r.sample),
                     'replay': d4.file_sha256(folder / 'replay.jsonl'),
                     'branches': d4.file_sha256(folder / 'branches.jsonl')}
        self.info = {'game_ids': [g['game_id'] for g in r.manifest['games']],
                     'game': [i for i in range(4) for _ in range(4)], 'ply': [5, 6, 7, 8] * 4}

    def tearDown(self):
        self.tmp.cleanup()

    def rebuild(self, **change):
        plan = deepcopy(self.plan)
        plan.update(change)
        with self.run_.patches():
            return x1.reconstruct(plan, self.run_.contract, self.shas, 'unused', self.run_.root, log=lambda *a, **k: None)

    def test_rebuild_equals_the_summary_and_inputs_stay_untouched(self):
        before = tree_hashes(self.run_.root)
        rec = self.rebuild()
        self.assertEqual(rec['actual']['observations'], self.summary['observations'])
        values, _ = x1.checked_predictions(predictions_for(self.info), self.info, ['network'] * 16,
                                           rec['manifest'], R2_SHA)
        result, rows = x1.analyse(rec, values)
        self.assertEqual(len(rows), self.summary['observations'])
        for compare in x1.ANALYSIS['comparisons']:
            a = result['ply_side'][compare]['total']
            s = self.summary['comparisons'][compare]
            self.assertEqual((a['disagreements'], a['better'], a['worse'], a['equal']),
                             (s['observations'], s['better'], s['worse'], s['equal']))
            q = result['within_model_q'][compare]
            # ply 8 of the fixture is a forced (non-tree) position: counted, never given a Q of 0
            self.assertEqual((q['candidate']['valid'], q['candidate']['excluded']), (0, {'non_tree': 4}))
            self.assertEqual(q['c5']['excluded'], {'non_tree': 4})
            self.assertEqual(q['candidate']['dq']['count'], 0)
            self.assertEqual(sum(c['disagreements'] for c in result['value_change'][compare]['cells'].values()), 4)
            self.assertEqual(result['concentration'][compare]['repeated_history']['repeated_histories'], 1)
        pair = result['seed_overlap']['pairs']['seed31001:c5|seed31002:c5']
        self.assertEqual((pair['same_candidate_move'], pair['outcome_agreement']['same_candidate_move']['agree']),
                         (4, 4))
        self.assertEqual(tree_hashes(self.run_.root), before)  # nothing written, nothing changed

    def test_the_run_commit_of_the_plan_is_required(self):
        with self.assertRaisesRegex(x1.Failure, 'not the clean run commit'):
            self.rebuild(run_commit='another')

    def test_tampered_journal_fails(self):
        path = self.run_.folder / 'replay.jsonl'
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b''.join(lines[:-1]))
        with self.assertRaises(x1.Failure):
            self.rebuild()


class CommandTest(unittest.TestCase):
    """run(): plan binding, new outputs only, rows under runs/, inputs re-hashed after the analysis."""

    @contextlib.contextmanager
    def setup(self, change_input=False):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = {}
            for k in x1.INPUTS:
                p = root / 'in' / f'{k}.bin'
                p.parent.mkdir(exist_ok=True)
                p.write_text(k)
                inputs[k] = {'path': f'in/{k}.bin', 'sha256': d4.file_sha256(p)}
            plan = {'format': x1.PLAN_FORMAT, 'status': 'fixed', 'nature': 'post_hoc_exploratory',
                    'analysis': x1.ANALYSIS, 'inputs': inputs, 'run_commit': 'f' * 40, 'hdev_cache': 'cache',
                    'output': {'summary': 'records/x1/x1_summary.json', 'rows': 'runs/x1/x1_rows.jsonl'}}
            (root / x1.PLAN).parent.mkdir(parents=True)
            (root / x1.PLAN).write_text(json.dumps(plan))
            rec = {'identity_sha256': 'i', 'games': {}, 'manifest': {'games': []}, 'filtered': [],
                   'actual': {'eligible_plies': 0, 'warmup_plies': 0, 'observations': 0, 'branches': 0,
                              'branch_outcomes': {}, 'actor_check': {}, 'reading': 'NOT_ESTABLISHED'}}

            def analyse(*a):
                if change_input:
                    (root / 'in' / 'replay.bin').write_text('changed')
                return {'ply_side': {}}, [row()]
            with patch('hybrid.h7c_nmask.committed', return_value=True), \
                    patch('h7r_d4.git_state', return_value={'commit': 'x1', 'dirty': False}), \
                    patch.object(x1.d4, 'load_contract', return_value=({}, inputs['contract']['sha256'])), \
                    patch.object(x1, 'check_run_links', return_value={}), \
                    patch.object(x1, 'check_records', return_value={}), \
                    patch.object(x1, 'check_prediction_source', return_value={}), \
                    patch.object(x1, 'reconstruct', return_value=rec), \
                    patch.object(x1, 'hdev_rows', return_value=({}, [])), \
                    patch.object(x1, 'load_predictions', return_value={}), \
                    patch.object(x1, 'checked_predictions', return_value=({}, {})), \
                    patch.object(x1, 'analyse', side_effect=analyse):
                yield root, SimpleNamespace(plan=str(root / x1.PLAN), cache=None)

    def test_outputs_are_new_files_and_rows_stay_under_runs(self):
        with self.setup() as (root, args):
            body = x1.run(args, root=root, log=lambda *a, **k: None)
            self.assertTrue(body['inputs_unchanged'])
            self.assertEqual(body['x1_git'], {'commit': 'x1', 'dirty': False})
            self.assertEqual(body['provenance']['run'], {})
            self.assertTrue((root / 'runs/x1/x1_rows.jsonl').is_file())
            self.assertEqual(body['rows']['count'], 1)
            with self.assertRaisesRegex(x1.Failure, 'never overwritten'):
                x1.run(args, root=root, log=lambda *a, **k: None)

    def test_an_input_changed_during_the_analysis_stops_without_output(self):
        with self.setup(change_input=True) as (root, args):
            with self.assertRaisesRegex(x1.Failure, 'replay.bin: SHA-256 differs'):
                x1.run(args, root=root, log=lambda *a, **k: None)
            self.assertFalse((root / 'records/x1/x1_summary.json').exists())
            self.assertFalse((root / 'runs/x1/x1_rows.jsonl').exists())

    def test_another_plan_or_analysis_is_refused(self):
        with self.setup() as (root, args):
            plan = json.loads((root / x1.PLAN).read_text())
            plan['analysis'] = {**plan['analysis'], 'top_games': 20}
            (root / x1.PLAN).write_text(json.dumps(plan))
            with self.assertRaisesRegex(x1.Failure, 'drift'):
                x1.run(args, root=root, log=lambda *a, **k: None)
            with self.assertRaisesRegex(x1.Failure, 'X1 plan is'):
                x1.run(SimpleNamespace(plan=str(root / 'other.json'), cache=None), root=root)



class DesktopCheckTest(unittest.TestCase):
    def test_precheck_reports_absent_and_bad_inputs(self):
        from scripts import h7r_d4_x1_precheck as pre
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a.bin').write_text('a'); (root / 'b.bin').write_text('b')
            plan = {'inputs': {'a': {'path': 'a.bin', 'sha256': d4.file_sha256(root / 'a.bin')},
                               'b': {'path': 'b.bin', 'sha256': '0' * 64},
                               'c': {'path': 'c.bin', 'sha256': '0' * 64}},
                    'output': {'summary': 'out.json', 'rows': 'runs/rows.jsonl'}}
            (root / pre.PLAN).parent.mkdir(parents=True)
            (root / pre.PLAN).write_text(json.dumps(plan))
            lines = []
            self.assertFalse(pre.precheck(root, log=lambda m, **k: lines.append(m)))
            self.assertEqual([l.split()[0] for l in lines], ['OK', 'BAD', 'ABSENT', 'PRECHECK'])
            self.assertIn('FAIL', lines[-1])
            self.assertEqual(pre.verify_output(root, log=lambda *a, **k: None), [f"{root / 'out.json'} is absent"])
            plan['inputs'] = {'a': plan['inputs']['a']}
            (root / pre.PLAN).write_text(json.dumps(plan))
            self.assertTrue(pre.precheck(root, log=lambda *a, **k: None))


if __name__ == '__main__':
    unittest.main()
