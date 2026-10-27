"""Read-only check of the v1 adjudication (scripts/h7c_adjudicate_check.py) on a fake-engine v1 run."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from renju import Game
from scripts import h7c_adjudicate as ad
from scripts import h7c_adjudicate_check as ck
from tests.test_h7c_adjudicate import DOUBLE_THREE, FAKE, MOVES, write_inputs

ROOT = Path(__file__).resolve().parents[1]


def v1_run(tmp: Path) -> dict:
    """A complete v1 run with the fake engine (unregistered, dirty: the two facts a test run cannot change)."""
    paths = write_inputs(tmp)
    out = tmp / 'out'
    argv = ['run'] + sum([[f'--{k.replace("_", "-")}', str(paths[k])] for k in (
        'manifest', 'summary', 'sensitivity_jsonl', 'selfplay', 'verdict')], []) + [
        '--rapfi-dir', str(tmp / 'rapfi'), '--workers', '2', '--allow-dirty', '--allow-unregistered-engine',
        '--engine-command', f'{sys.executable} {FAKE} --seed 5', '--jsonl', str(out / 'branches.jsonl'),
        '--output', str(out / 'adjudication.json'), '--manifest-out', str(out / 'manifest.json')]
    with mock.patch('builtins.print'):
        ad.main(argv)
    return {**paths, 'adjudication': out / 'adjudication.json', 'adjudication_manifest': out / 'manifest.json',
            'jsonl': out / 'branches.jsonl'}


def args_for(paths: dict, **change) -> argparse.Namespace:
    keys = ('adjudication', 'adjudication_manifest', 'jsonl', 'manifest', 'summary', 'sensitivity_jsonl',
            'selfplay', 'verdict')
    return argparse.Namespace(**{**{k: paths[k] for k in keys}, 'run_log': None, **change})


def rehash(paths: dict) -> None:
    """After a deliberate edit of branches.jsonl: the manifest and the summary name the new file."""
    for name, key in (('adjudication_manifest', 'files_sha256'),):
        data = json.loads(Path(paths[name]).read_text())
        data[key] = {ad._rel(paths['jsonl']): ad.file_sha(paths['jsonl'])}
        Path(paths[name]).write_text(json.dumps(data))
    result = json.loads(Path(paths['adjudication']).read_text())
    result['jsonl_sha256'] = ad.file_sha(paths['jsonl'])
    Path(paths['adjudication']).write_text(json.dumps(result))
    manifest = json.loads(Path(paths['adjudication_manifest']).read_text())
    manifest['document_files_sha256'] = {ad._rel(paths['adjudication']): ad.file_sha(paths['adjudication'])}
    Path(paths['adjudication_manifest']).write_text(json.dumps(manifest))


TEST_RUN_FAILS = ['a full, clean run']  # the fake engine is unregistered and the test tree dirty


class CheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = v1_run(Path(self.tmp.name))
        self.quiet = lambda *a, **k: None  # noqa: E731

    def run_check(self, **change):
        with mock.patch.object(ck.ad, 'EXPECTED_DISAGREEMENTS', {'c8:c5': 12, 'c5:orig': 8}):
            return ck.check(args_for(self.paths, **change), log=self.quiet)

    def test_a_faithful_run_passes_every_record_check(self):
        out = self.run_check()
        self.assertEqual(out['failed'], TEST_RUN_FAILS, out['checks'])
        for name in ('comparisons reproduced', 'overlap reproduced', 'every branch replays with Game',
                     'v1 code at the recorded commit = this checkout', 'exactly the branch list'):
            self.assertTrue(out['checks'][name]['pass'], name)
        shared = out['shared']
        # write_inputs: ply 6 is the one position both comparisons share, and there c8 = orig = [5, 5]
        self.assertEqual((shared['shared_positions'], shared['mirror_positions']), (1, 1))
        self.assertEqual(shared['c8:c5']['mirror']['observations'], 4)       # four games
        self.assertEqual(shared['c8:c5']['mirror']['source_games'], 4)       # nothing merged
        self.assertEqual(shared['c8:c5']['mirror']['positions'], 1)
        self.assertEqual(shared['c8:c5']['only_here']['observations'], 8)
        self.assertEqual(shared['c5:orig']['only_here']['observations'], 4)
        m8, m5 = shared['c8:c5']['mirror'], shared['c5:orig']['mirror']
        self.assertEqual((m8['better'], m8['worse']), (m5['worse'], m5['better']))  # the sign reversed
        self.assertEqual(shared['mirror_deltas_opposite'], 1)
        self.assertFalse(out['shared']['documented_matches'])               # 85 / 67 are the real run's
        self.assertTrue(out['sensitivity_abnormal_end_as_draw']['not_a_reading'])
        self.assertEqual(out['abnormal_end']['branches'], 0)
        self.assertEqual(out['timing']['worker_seconds']['branches'], 8)

    def test_raw_record_problems_fail(self):
        lines = Path(self.paths['jsonl']).read_text().splitlines()
        Path(self.paths['jsonl']).write_text('\n'.join(lines[:-1]) + '\n')  # one branch missing, manifest not updated
        self.assertIn('branches.jsonl = the v1 manifest', self.run_check()['failed'])
        rehash(self.paths)
        out = self.run_check()
        self.assertIn('exactly the branch list', out['failed'])
        self.assertIn('comparisons reproduced', out['failed'])

        Path(self.paths['jsonl']).write_text('\n'.join(lines + lines[:1]) + '\n')  # a branch twice
        rehash(self.paths)
        self.assertIn('each branch once', self.run_check()['failed'])

        rows = [json.loads(x) for x in lines]
        valid = next(r for r in rows if r['status'] == 'VALID' and r['continuation'])
        valid['score'] = 1.0 - valid['score'] if valid['score'] != 0.5 else 0.0
        Path(self.paths['jsonl']).write_text(''.join(json.dumps(r) + '\n' for r in rows))
        rehash(self.paths)
        out = self.run_check()
        self.assertIn('every branch replays with Game', out['failed'])

        Path(self.paths['jsonl']).unlink()
        out = self.run_check()
        self.assertIn('branches.jsonl present', out['failed'])
        self.assertEqual(out['verdict'], 'FAIL')

    def test_a_broken_v1_input_stops_the_check(self):
        Path(self.paths['selfplay']).write_text(Path(self.paths['selfplay']).read_text() + '\n')
        with self.assertRaises(ValueError):
            self.run_check()

    def test_run_log_timing(self):
        log = Path(self.tmp.name) / 'run.log'
        log.write_text('[h7c-adjudicate] 445 observations ({}), 708 unique branches; 0 done, 708 to play, workers 4\n'
                       '[h7c-adjudicate] 300/708 branches  42.4%  10.0 min  ETA 14 min  300/708 branches\n'
                       '[h7c-adjudicate] 445 observations ({}), 708 unique branches; 300 done, 408 to play, workers 4\n'
                       '[h7c-adjudicate] 408/408 branches  100.0%  12.5 min  ETA 0 min  408/408 branches\n')
        t = ck.timing([{'seconds': 4.0}, {'seconds': 6.0}], log)
        self.assertEqual(t['run_log']['pool_wall_minutes'], 22.5)
        self.assertEqual([s['done_before'] for s in t['run_log']['sessions']], [0, 300])
        self.assertEqual(t['worker_seconds']['sum'], 10.0)
        self.assertIsNone(ck.timing([], None)['run_log'])


class ReplayTest(unittest.TestCase):
    def test_abnormal_end_explained_only_when_black_is_stuck(self):
        record = {'key': 'k', 'move': [14, 14], 'continuation': [[13, 13]], 'status': 'INVALID',
                  'reason': 'abnormal_end', 'length': 10, 'score': None}
        original = Game.play

        def end_after(game_self, *a, **k):  # the rules end the game here without a winner (as when black is stuck)
            original(game_self, *a, **k)
            if len(game_self.history) == 9:
                game_self.done = True

        with mock.patch.object(Game, 'play', end_after):
            out = ck.replay(record, DOUBLE_THREE)
        self.assertTrue(any('continuation move 1 after the game was over' in p or 'abnormal_end is not' in p
                            for p in out['problems']), out)
        self.assertIn('abnormal', out)
        self.assertFalse(out['abnormal']['every_empty_point_black_forbidden'])

    def test_valid_records_replay(self):
        game_over = {'key': 'k', 'move': [7, 11], 'continuation': [], 'status': 'VALID', 'score': 1.0,
                     'winner': 'black', 'length': 9, 'forbid_checked': 0}
        self.assertEqual(ck.replay(game_over, MOVES[:8])['problems'], [])
        wrong = {**game_over, 'winner': 'white', 'score': 0.0}
        self.assertEqual(len(ck.replay(wrong, MOVES[:8])['problems']), 2)
        illegal = {**game_over, 'move': [7, 7]}
        self.assertIn('illegal', ck.replay(illegal, MOVES[:8])['problems'][0])


if __name__ == '__main__':
    unittest.main()
