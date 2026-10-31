"""D4 contract, clustering, bounds, source alignment, actor replay and append/resume checks.
Synthetic/fake engines and stand-in networks; no production H-dev, CUDA or real Rapfi.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from hybrid import h7r_d4 as d4
from scripts import h7r_d4 as cli
from tests.test_h7c_labels import game, MOVES

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/h7r/r2/R2-D4-BASELINE-1.contract.json'


def observations(n=40, a=1., b=.5, unresolved=0):
    obs, outcomes = [], {}
    for i in range(n):
        ka, kb = f'a{i}', f'b{i}'
        obs.append({'game_index': i % 10, 'key_a': ka, 'key_b': kb})
        outcomes[ka] = {'kind': 'unresolved' if i < unresolved else 'resolved', 'score': None if i < unresolved else a}
        outcomes[kb] = {'kind': 'resolved', 'score': b}
    return obs, outcomes


class StatisticsTest(unittest.TestCase):
    def setUp(self): self.counts = d4.multiplicities(10, 1000, 47)

    def stats(self, n=40, a=1., b=.5, unresolved=0):
        obs, outcomes = observations(n, a, b, unresolved)
        return d4.selection_stats(obs, outcomes, list(range(10)), self.counts)

    def test_directions_ties_and_no_decisive_share(self):
        positive, negative, ties = self.stats(), self.stats(a=0.), self.stats(a=.5)
        self.assertEqual(positive['label_individual'], 'BETTER')
        self.assertEqual(negative['label_individual'], 'WORSE')
        self.assertEqual(ties['label95'], 'INCONCLUSIVE')
        self.assertIsNone(ties['better_share'])
        self.assertEqual(ties['both']['draw'], 40)
        self.assertEqual(ties['better_share_ci95']['undefined'], 1000)

    def test_two_sided_bounds_never_treat_missing_as_a_loss(self):
        out = self.stats(a=.5, b=.5, unresolved=2)
        self.assertLess(out['mean_delta_bounds'][0], 0)
        self.assertGreater(out['mean_delta_bounds'][1], 0)
        self.assertEqual(out['label95'], 'INCONCLUSIVE')
        self.assertNotIn('UNRESOLVED_OVER_5_PERCENT', out['failures'])
        self.assertIn('UNRESOLVED_OVER_5_PERCENT', self.stats(unresolved=3)['failures'])
        # Resolved evidence can survive the conservative bounds when failures are small.
        self.assertEqual(self.stats(unresolved=1)['label95'], 'BETTER')

    def test_no_observations_and_empty_bootstrap_are_not_evidence(self):
        self.assertIsNone(self.stats(n=0)['mean_delta_bounds'])
        self.assertEqual(self.stats(n=0)['label95'], 'INCONCLUSIVE')
        obs, outcomes = observations(20)
        for o in obs: o['game_index'] = 0
        counts = torch.tensor([[0., 10., *([0.]*8)], [10., *([0.]*9)]], dtype=torch.float64)
        out = d4.selection_stats(obs, outcomes, list(range(10)), counts)
        self.assertEqual(out['intervals']['0.95']['lower']['undefined'], 1)
        self.assertEqual(out['label95'], 'INCONCLUSIVE')
        self.assertIn('UNDEFINED_BOOTSTRAP', out['failures'])  # the reason is recorded, not only the label
        self.assertNotIn('UNDEFINED_BOOTSTRAP', self.stats()['failures'])

    def test_decisive_observations_and_their_games(self):
        obs, outcomes = observations(40)
        for i in range(40):  # games 0-1 decisive (better / worse), the rest ties
            outcomes[f'a{i}']['score'] = {0: 1., 1: 0.}.get(i % 10, .5)
        out = d4.selection_stats(obs, outcomes, list(range(10)), self.counts)
        self.assertEqual((out['better'], out['worse'], out['equal']), (4, 4, 32))
        self.assertEqual((out['decisive'], out['decisive_games'], out['resolved_games']), (8, 2, 10))

    def test_macro_does_not_replace_observation_weighting(self):
        obs, outcomes = observations(40)
        for i in range(30):
            obs[i]['game_index'] = 0
            outcomes[f'a{i}']['score'] = 0
        out = d4.selection_stats(obs, outcomes, list(range(10)), self.counts)
        self.assertNotAlmostEqual(out['resolved_mean_delta'], out['resolved_game_macro'])
        # Same draws over the same games are shared across comparisons, never independent per seed.
        twin = d4.selection_stats(obs, outcomes, list(range(10)), self.counts)
        self.assertEqual(out, twin)

    def test_joint_iut_and_reference_is_not_a_fourth_primary(self):
        out = {p: {'label95': 'BETTER', 'label_individual': 'INCONCLUSIVE'} for p in d4.PRIMARY}
        out['c8:c5'] = {'label95': 'WORSE', 'label_individual': 'WORSE'}
        self.assertEqual(d4.joint_reading(out), 'ALL_THREE_BETTER')
        out[d4.PRIMARY[0]]['label95'] = 'INCONCLUSIVE'
        self.assertEqual(d4.joint_reading(out), 'NOT_ESTABLISHED')
        out[d4.PRIMARY[0]]['label_individual'] = 'BETTER'
        self.assertEqual(d4.joint_reading(out), 'PARTIAL_SIGNAL')
        out[d4.PRIMARY[1]]['label_individual'] = 'WORSE'
        self.assertEqual(d4.joint_reading(out), 'MIXED_DIRECTION')


class SensitivityTest(unittest.TestCase):
    def test_first_difference_ply_per_game(self):
        def ply(n, diff):
            sets = {m: {'route': 'tree', 'engine_move': [7, 7], 'visits': [[7, 7, 10, 0.1]]}
                    for m in ('c5', 'c8', 'seed31001', 'seed31002', 'seed31003')}
            if diff:
                sets['seed31001'] = {'route': 'tree', 'engine_move': [8, 8], 'visits': [[8, 8, 10, 0.2]]}
            return {'ply': n, 'color': 'black' if n % 2 == 0 else 'white', 'sets': sets}
        records = [{'game_index': 0, 'plies': [ply(5, False), ply(6, True), ply(7, True)]},
                   {'game_index': 1, 'plies': [ply(5, False), ply(6, False), ply(9, True)]},
                   {'game_index': 2, 'plies': [ply(5, False)]}]
        out = cli.sensitivity(records, d4.multiplicities(3, 50, 1))
        self.assertEqual(out['seed31001:c5']['first_difference_ply'], {'games': 2, 'median': 7.5, 'min': 6, 'max': 9})
        self.assertEqual(out['seed31001:c5']['games_with_difference'], 2)
        self.assertEqual(out['seed31002:c5']['first_difference_ply'],
                         {'games': 0, 'median': None, 'min': None, 'max': None})


class ContractTest(unittest.TestCase):
    def test_fixed_contract_and_pinned_prerequisites(self):
        c, sha = d4.load_contract(CONTRACT, ROOT)
        self.assertEqual(c['status'], 'fixed')
        self.assertEqual(c['execution'], {'prepare': True, 'smoke': False, 'full': False})
        self.assertEqual(c['hdev']['games'], 396)
        self.assertEqual(c['hdev']['rows'], 15985)
        eps = json.loads((ROOT/c['epsilon_result']).read_text())
        for n in ('seed31001', 'seed31002', 'seed31003'):
            self.assertEqual(c['models'][n]['value_checkpoint_sha256'], eps['seeds'][n[4:]]['identity']['value_checkpoint_sha256'])
        self.assertEqual(c['runtime_identity']['runtime_hash'], '7c727150922dc87207e61447e26f9348cec4a19d15a61d16d00e6f209d98ad0e')
        self.assertEqual(len(sha), 64)

    def test_contract_mutations_and_runtime_source_changes_fail(self):
        c = json.loads(CONTRACT.read_text())
        for change in ('draft', 'stats', 'model', 'source'):
            broken = deepcopy(c)
            if change == 'draft': broken['status'] = 'draft'
            if change == 'stats': broken['stats']['samples'] = 100
            if change == 'model': broken['models']['c5']['value_checkpoint_sha256'] = None
            if change == 'source': broken['runtime_sources'][0]['sha256'] = '0'*64
            with TemporaryDirectory() as tmp:
                p = Path(tmp)/'contract.json'; p.write_text(json.dumps(broken))
                with self.assertRaises(ValueError, msg=change): d4.load_contract(p, ROOT)

    def test_separate_committed_execution_permission(self):
        with self.assertRaises(ValueError): cli.approval(None, 's', 'smoke')
        with TemporaryDirectory() as tmp:
            p = Path(tmp)/'approval.json'
            a = {'format': 'h7r-d4-execution-v1', 'contract_sha256': 's', 'mode': 'smoke', 'approved': True}
            p.write_text(json.dumps(a))
            with patch.object(cli, 'committed', return_value=True):
                self.assertEqual(cli.approval(p, 's', 'smoke'), d4.file_sha256(p))
                for sha, mode in [('other', 'smoke'), ('s', 'full')]:
                    with self.assertRaises(ValueError): cli.approval(p, sha, mode)
            with patch.object(cli, 'committed', return_value=False), self.assertRaises(ValueError):
                cli.approval(p, 's', 'smoke')


class SourceTest(unittest.TestCase):
    def test_real_holdout_cache_path_and_source_alignment(self):
        from tests.test_h7r_r2 import EvaluateTest
        from tests.test_h7r_holdout import ACTOR
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cp, _ = EvaluateTest().fixture(tmp)
            c = json.loads(cp.read_text())
            c['models']['c5'] = ACTOR
            manifest, games, _ = cli.inputs(c, 's', tmp/'runs/cache', tmp)
            self.assertEqual(len(manifest['games']), 4)
            self.assertEqual(manifest['rows'], 16)
            self.assertEqual(manifest['warmup_plies'], 8)
            self.assertEqual([e['plies'] for e in manifest['games']], [[5, 6, 7, 8]]*4)
            # R2's file bindings refuse an altered cache before replay.
            with (tmp/'runs/cache/test.pt').open('ab') as handle: handle.write(b'changed')
            with self.assertRaises(ValueError): cli.inputs(c, 's', tmp/'runs/cache', tmp)

    def test_missing_duplicate_or_nonengine_eligible_rows(self):
        import h7c_labels as lab
        g = game()
        info = {'game_ids': [lab.game_id(g)], 'game': [0, 0], 'ply': [5, 6]}
        self.assertEqual(cli.eligible_manifest({'hdev': {}}, 's', [g], info)['rows'], 2)
        for plies in ([5, 5], [4, 5], [5, 100]):
            with self.assertRaises(ValueError):
                cli.eligible_manifest({'hdev': {}}, 's', [g], {**info, 'ply': plies})
        for change in ({'game_ids':info['game_ids']*2}, {'game':[0]}, {'game':[0,1]}):
            with self.assertRaises(ValueError):
                cli.eligible_manifest({'hdev':{}}, 's', [g], {**info,**change})
        broken = deepcopy(g); broken['plies'].pop(2)
        with self.assertRaises(ValueError): cli.eligible_manifest({'hdev': {}}, 's', [broken], info)

    def test_replay_keeps_warmup_but_observations_use_only_eligible(self):
        import h7c_sensitivity as sn
        g = game()
        models = {name: {} for name in ('c5', 'c8', 'seed31001', 'seed31002', 'seed31003')}
        call = []
        def search(agent, board):
            p = g['plies'][len(board.history)-3]
            call.append(p['ply'])
            return {k: p[k] for k in ('route', 'engine_move', 'visits')}
        r = sn.replay_game(g, models, 'cpu', make=lambda side, seed: object(), search=search)
        sn.check_replay(r, {0:g}, models, 'c5')
        self.assertEqual(len(call), 6*5)
        self.assertIn(3, call)
        m = {'games': [{'game_index':0, 'plies':[5, 6, 7, 8]}]}
        filtered = cli.eligible_replays([r], m)
        self.assertEqual([p['ply'] for p in filtered[0]['plies']], [5, 6, 7, 8])
        self.assertEqual(cli.observations_for(filtered, {0:g}), [])
        r['plies'][0]['sets']['c5']['visits'] = []
        with self.assertRaisesRegex(ValueError, 'reproduction'): sn.check_replay(r, {0:g}, models, 'c5')


class ResumeTest(unittest.TestCase):
    def test_full_wrapper_interruption_resume_summary_and_refusals(self):
        """Execute the whole development wrapper with source fixtures and terminal fixed branches.
        Only heavy preflight/model/engine dependencies are replaced; aggregation and resume are real.
        """
        import h7c_adjudicate as ad
        import h7c_labels as lab
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = json.loads(CONTRACT.read_text())
            eps = ROOT/c['epsilon_result']
            c['epsilon_result'] = str(eps)
            games = {i:game(i) for i in range(4)}
            info = {'game_ids':[lab.game_id(g) for g in games.values()],
                    'game':[i for i in range(4) for _ in range(4)], 'ply':[5,6,7,8]*4}
            manifest = cli.eligible_manifest(c, 's', list(games.values()), info)
            sample = root/'sample.json'; sample.write_text(json.dumps(manifest))
            models = {n:{} for n in c['models']}
            args = SimpleNamespace(contract=CONTRACT, cache='unused', sample=sample, approval='unused',
                                   mode='full', model=[], device='cuda', workers=2, rapfi_dir='unused',
                                   out_dir=root/'runs/h7r/R2-D4-BASELINE-1/full')
            def replay(tasks):
                rows = []
                for t in tasks:
                    plies = []
                    for p in t['game']['plies']:
                        sets = {n:{k:deepcopy(p[k]) for k in ('route','engine_move','visits')} for n in models}
                        if p['ply'] == 8:
                            for n in models:
                                if n != 'c5': sets[n]['engine_move'] = [7,6]
                        plies.append({'ply':p['ply'], 'color':p['color'], 'actor_move':p['engine_move'],
                                      'actor_route':p['route'], 'sets':sets})
                    rows.append({'game_index':t['game_index'], 'identity_sha256':t['identity_sha256'], 'plies':plies})
                return rows
            with patch.object(cli,'ROOT',root), patch.object(cli,'contract_at',return_value=(c,'s')), \
                    patch.object(cli,'git_state',return_value={'commit':'fixture','dirty':False}), \
                    patch.object(cli,'approval',return_value='approved'), patch.object(cli,'committed',return_value=True), \
                    patch.object(cli,'inputs',return_value=(manifest,games,{})), \
                    patch.object(cli,'checked_models',return_value=(models,{})), \
                    patch('analysis.tb_v3_fast5.runtime_hash',return_value=c['runtime_identity']['runtime_hash']), \
                    patch('torch.cuda.get_device_name',return_value='fixture'), \
                    patch.object(cli,'ProcessPoolExecutor',side_effect=lambda max_workers,mp_context:ThreadPoolExecutor(max_workers)), \
                    patch('h7c_adjudicate.engine_spec',return_value={'identity':c['rapfi_identity']}), \
                    patch.object(cli,'replay_chunk',side_effect=replay) as rp:
                with patch('h7c_adjudicate._chunk',side_effect=RuntimeError('interrupted')):
                    with self.assertRaisesRegex(RuntimeError,'interrupted'): cli.run(args)
                self.assertEqual(rp.call_count, 4)
                rp.reset_mock()
                result = cli.run(args)
                self.assertEqual(rp.call_count, 0)
                self.assertEqual(result['reading'],'NOT_ESTABLISHED')
                self.assertEqual(result['eligible_plies'],16)
                self.assertEqual(result['warmup_plies'],8)
                self.assertEqual(result['branches'],2)  # full-history+move sharing across games and seeds
                self.assertEqual(result['observations'],16)
                self.assertTrue(result['actor_check']['reproduced'])
                self.assertEqual(result['comparisons']['seed31001:c5']['both']['win'],4)
                self.assertNotIn('label_individual',result['comparisons']['c8:c5'])
                with self.assertRaisesRegex(ValueError,'completed summary'): cli.run(args)
                (args.out_dir/'summary.json').unlink()
                journal = args.out_dir/'branches.jsonl'
                records = [json.loads(line) for line in journal.read_text().splitlines()]
                records[0]['score'] = 0.
                journal.write_text(''.join(json.dumps(r)+'\n' for r in records))
                with self.assertRaisesRegex(ValueError,'outcome differs'): cli.run(args)

    def test_resume_id_duplicates_torn_tail_and_middle_corruption(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp)/'replay.jsonl'
            record = {'game_index': 0, 'identity_sha256': 'i'}
            cli.append_record(p, record)
            with p.open('ab') as f: f.write(b'{"broken"')
            self.assertEqual(cli.read_journal(p, 'game_index', 'i', {0}), {0:record})
            self.assertEqual(len(list(Path(tmp).glob('*.partial-*'))), 1)
            before = p.read_bytes()
            cli.append_record(p, record)
            with self.assertRaises(ValueError): cli.read_journal(p, 'game_index', 'i', {0})
            p.write_bytes(before)
            for identity, keys in [('j', {0}), ('i', {1})]:
                with self.assertRaises(ValueError): cli.read_journal(p, 'game_index', identity, keys)
            p.write_bytes(b'bad\n'+before)
            with self.assertRaises(ValueError): cli.read_journal(p, 'game_index', 'i', {0})

    def test_unordered_execution_persists_and_resumes_without_work(self):
        with TemporaryDirectory() as tmp, ThreadPoolExecutor(2) as pool:
            p = Path(tmp)/'out.jsonl'
            tasks = [{'game_index':i, 'identity_sha256':'i'} for i in range(3)]
            def worker(chunk): return chunk
            def validate(r): self.assertEqual(r['identity_sha256'], 'i')
            out = cli.execute(pool, worker, tasks, p, validate, {}, 'game_index', 'fixture', lambda _:1)
            saved = p.read_bytes()
            self.assertEqual(len(out), 3)
            with patch.object(pool, 'submit', side_effect=AssertionError('must resume')):
                cli.execute(pool, worker, tasks, p, validate, out, 'game_index', 'fixture', lambda _:1)
            self.assertEqual(saved, p.read_bytes())
            out[0]['identity_sha256'] = 'bad'
            with self.assertRaises(AssertionError):
                cli.execute(pool, worker, tasks, p, validate, out, 'game_index', 'fixture', lambda _:1)

    def test_branch_replay_and_wrong_status_identity_or_illegal_prefix(self):
        import h7c_adjudicate as ad
        from tests.test_h7c_adjudicate import task
        t = task(MOVES[:8], [7,11])
        r = ad.play_branch(t)
        self.assertEqual(cli.validate_branch(r, t, d4.POLICY)['score'], 1.)
        for key, value in [('score', 0.), ('identity_sha256', 'bad'), ('order', 9), ('mover','white'),
                           ('continuation', [[0,0]])]:
            with self.assertRaises(ValueError): cli.validate_branch({**r, key:value}, t, d4.POLICY)
        truncated = {**r, 'status':'INVALID', 'score':None, 'reason':'truncated'}
        with self.assertRaises(ValueError): cli.validate_branch(truncated, t, d4.POLICY)

    def test_structural_normalization_and_execution_failure_remain_distinct(self):
        from tests.test_h7r_confirm import STRUCTURAL, PREFIX, FIXED
        from tests.test_h7c_adjudicate import task
        t = task(PREFIX, FIXED)
        r = {**STRUCTURAL, 'key':t['key'], 'order':0, 'identity_sha256':t['identity_sha256'],
             'move':FIXED, 'ply':len(PREFIX), 'mover':'black' if len(PREFIX)%2 == 0 else 'white'}
        self.assertEqual(cli.validate_branch(r,t,d4.POLICY)['kind'],'structural')
        t = task(MOVES[:3], [0,2])
        r = {'status':'INVALID', 'score':None, 'reason':'timeout', 'key':t['key'], 'order':0,
             'identity_sha256':t['identity_sha256'], 'move':t['move'], 'ply':3, 'mover':'white',
             'length':4, 'continuation':[]}
        self.assertEqual(cli.validate_branch(r,t,d4.POLICY)['kind'],'unresolved')


class NativeReplayTest(unittest.TestCase):
    def test_worker_replays_real_fast5_actor_with_stand_in_models_on_cpu(self):
        from rt30 import fast5
        if not fast5.available(): self.skipTest('build native fast5 first')
        from tests.test_tb_v3_net import write_models
        import h7c_selfplay as sp
        import h7c_sensitivity as sn
        from analysis.tb_v3_net import model_identity
        from hybrid.h6_fast import install
        import h7c_arena as ar
        try:
            with TemporaryDirectory() as tmp:
                policy, value = write_models(Path(tmp)/'models',seed=5)
                sp._init_worker(str(policy),str(value),'cpu')
                ident = model_identity(policy,value)
                t = {'identity':{**sp.run_identity(8,8808,ident,'cpu'),
                                  'generator_git':{'commit':'fixture','dirty':False}},
                     'game_index':0, 'game_seed':sp.game_seed(8808,8,0), 'opening_seed':sp.opening_seed(8808,8,0)}
                g = sp.play_game(t,max_plies=10)
                models = {n:{'policy':str(policy),'value':str(value)} for n in ('c5','seed31001')}
                out = cli.replay_chunk([{'game':g,'models':models,'device':'cpu','identity_sha256':'i'}])[0]
                sn.check_replay(out,{0:g},models,'c5')
                self.assertEqual(out['identity_sha256'],'i')
        finally:
            fast5.uninstall(); install('off'); ar._MODELS.clear()

    def test_fake_protocol_processes_and_shared_branches(self):
        import h7c_adjudicate as ad
        from tests.test_h7c_adjudicate import task, spec
        with TemporaryDirectory() as tmp:
            log = Path(tmp)/'fake.log'
            engine = spec('--seed', '3', '--log', str(log))
            for move in ([0,2], [7,9]):
                t = task(MOVES[:3], move, engine)
                r = ad.play_branch(t)
                self.assertIn(cli.validate_branch(r, t, d4.POLICY)['score'], (0., .5, 1.))
            self.assertEqual(sum(line.startswith('START') for line in log.read_text().splitlines()), 2)
            self.assertEqual(log.read_text().splitlines().count('END'), 2)


if __name__ == '__main__': unittest.main()
