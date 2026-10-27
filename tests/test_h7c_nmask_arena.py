"""H7C-NMASK-1 Arena (scripts/h7c_nmask_arena.py): the identity, the verdict on recorded games, and real TB-V3 /
fast5 games with stand-in networks (skipped without the native core)."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from hybrid import h7c_nmask as nm
from rt30 import fast5
from scripts import h7c_arena as ar
from scripts import h7c_nmask_arena as na
from tests.test_h7c_arena import game, judge_record

PARENT, CONTROL, CANDIDATE = ({'v': c} for c in ('c5', 'c8', 'mask'))
DECISION = {'parent': PARENT, 'control': CONTROL, 'evaluation': {'official': {'seed': 8761}}}


def write_records(tmp, results, *, tactical_items=None, **identity):
    ident = {'format': na.FORMAT, 'arm': 'official', 'seed': 8761, 'decision_sha256': 'd' * 64,
             'candidate': CANDIDATE, 'parent': PARENT, 'candidate_record_sha256': 'r' * 64,
             'git': {'commit': 'abc', 'dirty': False}, **identity}
    sig = ar.signature(ident)
    games = []
    for pair, (black, white) in enumerate(results):
        for color, result in (('black', black), ('white', white)):
            illegal = {'side': 'tested'} if result == 'loss_illegal' else None
            games.append({**game(pair, color, 'loss' if result == 'loss_illegal' else result, illegal),
                          'signature': sig})
    arm = Path(tmp) / 'official.json'
    arm.write_text(json.dumps({'format': f'{na.FORMAT}-arm', 'identity': ident, 'signature': sig, 'games': games,
                               'summary': ar.summarize_arm(games), 'fingerprint': ar.fingerprint(games)}))
    tactical = Path(tmp) / 'tactical.json'
    tactical.write_text(json.dumps({'format': f'{na.FORMAT}-tactical', 'git': {'commit': 'abc', 'dirty': False},
                                    'candidate': CANDIDATE, 'parent': PARENT, 'decision_sha256': 'd' * 64,
                                    'candidate_record_sha256': 'r' * 64, 'rule_version': ar.TACTICAL_RULE,
                                    'judge': judge_record(**(tactical_items or {}))}))
    return argparse.Namespace(arm=arm, tactical=tactical, decision=None, allow_uncommitted=True, allow_dirty=False)


class VerdictTest(unittest.TestCase):
    def verdict(self, args):
        with mock.patch.object(nm, 'load_decision', lambda *a, **k: (DECISION, 'd' * 64)):
            return na.verdict(args)

    def test_promote_keep_and_gate(self):
        with TemporaryDirectory() as tmp:
            out = self.verdict(write_records(tmp, [('win', 'win')] * 30 + [('win', 'loss')] * 20))
            self.assertEqual(out['promotion']['decision'], 'PROMOTE')
            self.assertIn('changes no champion', out['meaning'])
            self.assertEqual(out['seed'], 8761)
        with TemporaryDirectory() as tmp:
            out = self.verdict(write_records(tmp, [('win', 'loss')] * 50))
            self.assertEqual(out['promotion']['decision'], 'KEEP_CHAMPION')
        with TemporaryDirectory() as tmp:
            out = self.verdict(write_records(tmp, [('win', 'win')] * 49 + [('win', 'loss_illegal')]))
            self.assertEqual((out['tactical']['verdict'], out['tactical']['forbidden_errors']), ('FAIL', 1))
            self.assertEqual(out['promotion']['decision'], 'KEEP_CHAMPION')

    def test_refusals(self):
        with TemporaryDirectory() as tmp:
            args = write_records(tmp, [('win', 'win')] * 50, decision_sha256='e' * 64)
            with self.assertRaisesRegex(ValueError, 'decision'):
                self.verdict(args)
        with TemporaryDirectory() as tmp:
            args = write_records(tmp, [('win', 'win')] * 50, seed=8731)
            with self.assertRaisesRegex(ValueError, 'seed'):
                self.verdict(args)
        with TemporaryDirectory() as tmp:
            args = write_records(tmp, [('win', 'win')] * 50, git={'commit': 'abc', 'dirty': True})
            with self.assertRaisesRegex(ValueError, 'clean'):
                self.verdict(args)
        with TemporaryDirectory() as tmp:
            args = write_records(tmp, [('win', 'win')] * 49)
            with self.assertRaisesRegex(ValueError, 'complete pairs'):
                self.verdict(args)
        with TemporaryDirectory() as tmp:
            args = write_records(tmp, [('win', 'win')] * 50)
            record = json.loads(args.arm.read_text())
            record['games'][0]['result'] = 'loss'
            args.arm.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'summary'):
                self.verdict(args)


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class PlayTest(unittest.TestCase):
    def tearDown(self):
        from hybrid.h6_fast import install
        fast5.uninstall()
        install('off')
        ar._MODELS.clear()

    def test_official_games_seed_and_resume(self):
        from analysis import tb_v3_net
        from tests.test_tb_v3_net import write_models

        quiet = lambda *a, **k: None  # noqa: E731
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            policy, parent_value = write_models(tmp / 'p', seed=2)
            _, value = write_models(tmp / 'c', seed=3)
            value.with_name('record.json').write_text('{}')
            parent = tb_v3_net.model_identity(policy, parent_value)
            candidate = tb_v3_net.model_identity(policy, value)
            decision = {**DECISION, 'parent': parent}
            args = argparse.Namespace(decision=None, allow_uncommitted=True, allow_dirty=True,
                                      candidate_value=value, policy=policy, parent_value=parent_value, device='cpu',
                                      workers=1, jsonl=tmp / 'official.jsonl', pairs=1, max_plies=7)
            with mock.patch.object(nm, 'load_decision', lambda *a, **k: (decision, 'd' * 64)), \
                    mock.patch('h7c_nmask.check_candidate', lambda *a, **k: {
                        'checkpoint_sha256': candidate['value_checkpoint_sha256']}):
                first = na.play(args, log=quiet)
                self.assertEqual(len(first['games']), 2)
                self.assertEqual(first['identity']['seed'], 8761)
                self.assertTrue(all(g['key'].startswith('official/8761/') for g in first['games']))
                self.assertEqual(first['identity']['candidate'], candidate)
                again = na.play(args, log=quiet)
                self.assertEqual([g['moves'] for g in again['games']], [g['moves'] for g in first['games']])
                with self.assertRaisesRegex(ValueError, 'parent'):  # another parent than c5
                    na.play(argparse.Namespace(**{**vars(args), 'parent_value': value}), log=quiet)
                with mock.patch.object(nm, 'load_decision', lambda *a, **k: (
                        {**decision, 'evaluation': {'official': {'seed': 8731}}}, 'd' * 64)):
                    with self.assertRaisesRegex(ValueError, 'reserved'):
                        na.play(argparse.Namespace(**{**vars(args), 'jsonl': tmp / 'o.jsonl'}), log=quiet)


if __name__ == '__main__':
    unittest.main()
