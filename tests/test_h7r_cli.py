"""H7-R evaluation CLI (scripts/h7r_eval.py): suites, contract arms, records, sizing, selection and the judgment
end to end with an in-process fake game; the game wiring (budgets) and the time gate on the real engine with
stand-in networks (skipped without the native core)."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from hybrid import h7r_eval as ev
from rt30 import fast5
from scripts import h7r_eval as cli
from tests.test_h7r_eval import STATS

QUIET = lambda *a, **k: None  # noqa: E731


def fake_play_game(task, make_tested=None, make_opponent=None, max_plies=None):
    """A complete game decided by the opening: the tested side wins as black, except at openings whose last
    coordinate is even, where the candidate (``context['arm']`` starts with 'candidate') also wins as white."""
    from renju import BLACK
    ctx = task['context']
    black = task['tested_color'] == BLACK
    bonus = ctx['arm'].startswith('candidate') and task['opening'][-1][1] % 2 == 0
    result = 'win' if black or bonus else 'loss'
    from tests.h7r_helpers import winning_moves
    tested = BLACK if black else -BLACK
    winner = tested if result == 'win' else -tested
    moves = winning_moves(task['opening'], winner)
    return {'key': task['key'], 'signature': ctx['signature'], 'arm': task['arm'], 'pair': task['pair'],
            'seed': task['seed'], 'tested_color': 'black' if black else 'white', 'result': result,
            'winner': winner, 'length': len(moves), 'moves': moves, 'illegal': None,
            'truncated': False, 'tested_moves': [], 'opponent_routes': {}, 'game_seconds': 2.0}


def world(tmp: Path) -> dict:
    from analysis import tb_v3_net
    from tests.test_tb_v3_net import write_models

    policy, champ = write_models(tmp / 'm' / 'champ', seed=1)
    _, cand = write_models(tmp / 'm' / 'cand', seed=2)
    _, orig = write_models(tmp / 'm' / 'orig', seed=3)
    rel = lambda p: str(Path(p).relative_to(tmp))  # noqa: E731
    models = {n: {'policy': rel(policy), 'value': rel(v), 'identity': tb_v3_net.model_identity(policy, v)}
              for n, v in (('cand', cand), ('champ', champ), ('orig', orig))}
    suite = cli.suite(argparse.Namespace(name='O0', count=40, seed=8781, compare_with=[], allow_dirty=True),
                      log=QUIET)
    (tmp / 'suites').mkdir()
    (tmp / 'suites/o0.json').write_text(json.dumps(suite))
    ref = {'path': 'suites/o0.json', 'sha256': ev.file_sha256(tmp / 'suites/o0.json')}
    arm = lambda name, t, o: {'name': name, 'tested': {'model': t, 'budget': 1},  # noqa: E731
                              'opponent': {'model': o, 'budget': 1}, 'pairs': 30, 'seed': 8790, 'pairing': 'P',
                              'suite': ref}
    body = {'format': ev.CONTRACT_FORMAT, 'contract_id': 'T-CONFIRM', 'purpose': 'confirm', 'status': 'fixed',
            'models': models, 'roles': {'candidate': 'cand', 'champion': 'champ', 'anchor': 'orig'},
            'arms': [arm('candidate_vs_champion', 'cand', 'champ'), arm('candidate_vs_anchor', 'cand', 'orig'),
                     arm('champion_vs_anchor', 'champ', 'orig')],
            'stats': STATS, 'informative': {'kind': 'score_split', 'm': 5},
            'anchor': {'model': 'orig', 'margin': -0.02}, 'selection': {'margin': -0.02, 'alpha': 0.025,
                                                                         'min_observations': 20, 'holdout_contract_sha256': 'f'*64,
                                                                         'engine': {'registered_engine': True}},
            'gates': {'time': {'mean_ratio_max': 1.1, 'p95_ratio_max': 1.2}, 'tactical_rule': 'h7c-tactical-v2',
                      'forbidden_errors': 0}}
    (tmp / 'contracts').mkdir()
    path = tmp / 'contracts/confirm.json'
    path.write_text(json.dumps(body))
    return {'root': tmp, 'contract': path, 'body': body, 'policy': policy, 'cand': cand, 'champ': champ}


class FlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.w = world(Path(self.tmp.name))

    def play(self, arm, **kw):
        args = argparse.Namespace(contract=self.w['contract'], arm=arm, device='cpu', workers=1,
                                  jsonl=self.w['root'] / 'runs' / f'{arm}.jsonl', allow_dirty=True,
                                  threads_for_tests=True, **kw)
        with mock.patch('h7c_arena.play_game', fake_play_game):
            return cli.play(args, log=QUIET, root=self.w['root'])

    def save(self, record, name):
        path = self.w['root'] / 'out' / f'{name}.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(record))
        return path

    def test_confirm_flow(self):
        records = {a: self.save(self.play(a), a) for a in ev.ROLE_ARMS}
        rec = json.loads(records['candidate_vs_champion'].read_text())
        s = rec['summary']
        self.assertEqual(s['pairs'], 30)
        self.assertEqual(s['move_split'], s['score_split'])  # the fake splits both together
        self.assertGreater(s['score_split'], 0)
        # resume: nothing new, the same record
        again = self.play('candidate_vs_champion')
        self.assertEqual(again['games_played_now'], 0)
        self.assertEqual(again['summary'], s)
        # the three arms play the same openings (one pairing)
        ids = {a: [o['history_id'] for o in json.loads(p.read_text())['openings']] for a, p in records.items()}
        self.assertEqual(len({tuple(v) for v in ids.values()}), 1)
        self.assertEqual(len(set(ids['candidate_vs_champion'])), 30)
        # selection rows, timing record (as the tools write them), judgment
        from tests.h7r_helpers import selection_source, timing_record, tactical_record
        # Freeze the holdout and registered engine in the confirmation contract before evidence generation.
        selection_source(self.w['root'], self.w['body'], 'pending')
        self.w['contract'].write_text(json.dumps(self.w['body']))
        _, sha = ev.load_contract(self.w['contract'], self.w['root'], check_files=False)
        # Replay arm fixtures under the final contract identity.
        import shutil
        shutil.rmtree(self.w['root'] / 'runs')
        records = {a: self.save(self.play(a), a) for a in ev.ROLE_ARMS}
        source = selection_source(self.w['root'], self.w['body'], sha)
        selpath = self.w['root'] / 'out' / 'sel.json'
        sel = cli.selection(argparse.Namespace(contract=self.w['contract'], evidence=source, output=selpath,
                                               allow_dirty=True, root=self.w['root']))
        self.assertEqual(sel['status'], 'NONINFERIOR')
        timing = timing_record(self.w['body'], sha, self.w['root'])
        args = argparse.Namespace(contract=self.w['contract'], arm_record=list(records.values()),
                                  selection=self.save(sel, 'sel'), tactical=self.save(tactical_record(self.w['body']), 'tact'),
                                  timing=self.save(timing, 'time'), allow_dirty=True)
        verdict = cli.judge(args, root=self.w['root'])
        self.assertEqual(verdict['decision'], 'PROMOTE', verdict['reasons'])
        self.assertIn('changes no champion', verdict['meaning'])
        # missing records keep the champion
        with self.assertRaisesRegex(ValueError, 'different confirmation inputs'):
            cli.judge(argparse.Namespace(**{**vars(args), 'selection': None, 'timing': None}), root=self.w['root'])
        self.assertEqual(cli.judge(args, root=self.w['root'])['decision'], verdict['decision'])
        # a tampered record is refused
        rec = json.loads(records['candidate_vs_champion'].read_text())
        rec['games'][0]['result'] = 'draw'
        bad = self.save(rec, 'bad')
        with self.assertRaisesRegex(ValueError, 'result|summary'):
            cli.load_arm(bad, *ev.load_contract(self.w['contract'], self.w['root'], check_files=False,
                                                contracts_dir=self.w['contract'].parent), root=self.w['root'])
        # sizing from a (pilot-like) arm record
        size = cli.size(argparse.Namespace(contract=self.w['contract'], arm_record=[records['candidate_vs_champion']],
                                           delta=0.03, power=0.8, m=[10, 20], allow_dirty=True,
                                           root=self.w['root']))
        arm = size['arms']['candidate_vs_champion']
        self.assertGreater(arm['pairs_needed'], 0)
        self.assertEqual(arm['worker_seconds_per_pair'], 4.0)

    def test_play_refusals(self):
        body = dict(self.w['body'])
        for change, message in (({'status': 'draft'}, 'draft'),
                                ({'informative': {'kind': 'score_split', 'm': None}}, 'unset')):
            self.w['contract'].write_text(json.dumps({**body, **change}))
            with self.assertRaisesRegex(ValueError, message):
                self.play('candidate_vs_champion')
        self.w['contract'].write_text(json.dumps(body))
        with self.assertRaisesRegex(ValueError, 'commit the contract'):
            args = argparse.Namespace(contract=self.w['contract'], arm='candidate_vs_champion', device='cpu',
                                      workers=1, jsonl=self.w['root'] / 'x.jsonl', allow_dirty=False)
            with mock.patch.object(cli, '_git', lambda allow: {'commit': 'x', 'dirty': False}):
                cli.play(args, log=QUIET, root=self.w['root'])
        self.play('candidate_vs_champion')
        changed = {**body, 'arms': [dict(a, pairs=29) if a['name'] == 'candidate_vs_champion' else a
                                    for a in body['arms']]}
        self.w['contract'].write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'different confirmation inputs|another setup'):  # immutable binding
            self.play('candidate_vs_champion')

    def test_main_never_overwrites(self):
        out = self.w['root'] / 'suite.json'
        self.assertEqual(cli.main(['suite', '--name', 'O2', '--output', str(out), '--allow-dirty']), 0)
        with self.assertRaises(SystemExit):
            cli.main(['suite', '--name', 'O2', '--output', str(out), '--allow-dirty'])


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class EngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.w = world(Path(self.tmp.name))

    def tearDown(self):
        from hybrid.h6_fast import install
        from scripts import h7c_arena as ar
        fast5.uninstall()
        install('off')
        ar._MODELS.clear()

    def test_budget_wiring(self):
        import h7c_budget as sb
        from renju import BLACK
        calls = []
        real = sb.make_budget_agent

        def spy(factor, *a):
            calls.append(factor)
            return real(factor, *a)

        sides = {'tested': {'policy': str(self.w['policy']), 'value': str(self.w['cand']), 'budget': 8},
                 'opponent': {'policy': str(self.w['policy']), 'value': str(self.w['champ']), 'budget': 1}}
        task = {'key': 'k', 'arm': 'official', 'pair': 0, 'seed': 1, 'tested_color': BLACK,
                'opening': [[7, 7], [6, 7], [8, 8]], 'opening_id': 'x', 'max_plies': 6,
                'context': {'models': {'candidate': sides['tested'], 'parent': sides['opponent']}, 'sides': sides,
                            'device': 'cpu', 'signature': 's', 'arm': 'a'}}
        with mock.patch.object(sb, 'make_budget_agent', spy):
            row = cli._play_chunk([task])[0]
        self.assertEqual(sorted(calls), [1, 8])
        self.assertEqual((row['arm'], row['opening_id'], row['length']), ('a', 'x', 6))

    def test_timing_record(self):
        args = argparse.Namespace(contract=self.w['contract'], side=['cand', 'champ'],
                                  suite=self.w['root'] / 'suites/o0.json', positions=2, warmup=1, device='cpu',
                                  seed=1, bootstrap_samples=100, mean_ratio_max=1.1, p95_ratio_max=1.2,
                                  allow_dirty=True)
        out = cli.timing(args, log=QUIET, root=self.w['root'])
        self.assertEqual(out['positions'], 2)
        self.assertEqual(out['gate']['limits'], self.w['body']['gates']['time'])
        self.assertIn('platform', out['environment'])
        self.assertEqual(set(out['sides']), {'cand', 'champ'})


if __name__ == '__main__':
    unittest.main()
