"""scripts/h7r_s400_arena.py: the R2-S400-TRAIN Arena on a temporary repository: the committed v2 Arena contract must
be the one the S400 contract gives; the model files are checked before any game; four arms played through the v2
play loop with an in-process fake game; the judge reads D' and P' only when S and T played the same openings in each
suite, from one clean commit, with the full sample; verify-judge records a judgment only when the arm records give it
again. The engine games are stand-ins; the records, checks and statistics are the tools'."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

import h7r_eval as cli
import h7r_p1 as p1
import h7r_s400_arena as sa
import h7r_s400_train as st
from tests.test_h7r_s400_train import QUIET, RepoBase, git

SUITES = ('O1-dev-a', 'O1-dev-b')


def fake_play_game(task, make_tested=None, make_opponent=None, max_plies=None):
    """A complete game decided by the opening: the tested side wins as black; as white, S wins at about half of the
    openings and T at about a third (so D' and P' are not zero)."""
    from renju import BLACK
    from tests.h7r_helpers import winning_moves

    ctx = task['context']
    black = task['tested_color'] == BLACK
    h = int(hashlib.sha256(task['opening_id'].encode()).hexdigest(), 16)
    white_win = h % 2 == 0 if '_S_' in ctx['arm'] else h % 3 == 0
    result = 'win' if black or white_win else 'loss'
    tested = BLACK if black else -BLACK
    winner = tested if result == 'win' else -tested
    moves = winning_moves(task['opening'], winner)
    return {'key': task['key'], 'signature': ctx['signature'], 'arm': task['arm'], 'pair': task['pair'],
            'seed': task['seed'], 'tested_color': 'black' if black else 'white', 'result': result,
            'winner': winner, 'length': len(moves), 'moves': moves, 'illegal': None,
            'truncated': False, 'tested_moves': [], 'opponent_routes': {}, 'game_seconds': 1.0}


class ArenaBase(RepoBase):
    def setUp(self):
        super().setUp()
        root, c = self.root, json.loads((self.root / st.CONTRACT).read_text())
        self.stand_in(p1.PATHS['h3'], 'champion policy', c['inputs']['parent_and_teacher'], 'policy')
        self.stand_in(p1.PATHS['c5_value'], 'c5 value', c['inputs']['value_c5'], 'value')
        self.write_contract(c)
        git(root, 'commit', '-qam', 'stand-in model SHA-256')
        self.approve('prepare', 'gate', 'search', 'train', 'arena')
        s_dir = root / st.RUNS / 'S-20261024'
        s_dir.mkdir(parents=True)
        (s_dir / 'best.pt').write_text('S checkpoint')
        (s_dir / 'best.json').write_text('S metadata')
        self.commit_record(st.run_record_path(20261024), {
            'format': 'h7r-s400-train-run-v1', 'experiment_sha256': self.eid(), 'verdict': 'CANDIDATE', 'arm': 'S',
            'seed': 20261024, 'candidate': {'checkpoint_sha256': p1.file_sha256(s_dir / 'best.pt'),
                                            'metadata_sha256': p1.file_sha256(s_dir / 'best.json')}})
        st.arena_contract(self.ns(), root)
        git(root, 'add', st.ARENA_CONTRACT)
        git(root, 'commit', '-qm', 'arena contract')
        self.arena = json.loads((root / st.ARENA_CONTRACT).read_text())
        self.arms_dir = root / sa.ARMS_DIR

    def stand_in(self, rel, text, identity, kind):
        """Model files at ``rel`` (its metadata beside it) and their SHA-256 in the contract's identity."""
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = path.with_suffix('.json') if kind == 'policy' else path.with_name('best.json')
        path.write_text(text)
        meta.write_text(text + ' metadata')
        identity[f'{kind}_checkpoint_sha256'] = p1.file_sha256(path)
        identity[f'{kind}_metadata_sha256'] = p1.file_sha256(meta)

    def head(self):
        return {'commit': git(self.root, 'rev-parse', 'HEAD'), 'dirty': False}

    def args(self, **k):
        return argparse.Namespace(contract=None, **k)

    def play(self, arm, git_state=None, save=True, folder=None):
        folder = folder or self.arms_dir
        args = self.args(arm=arm, device='cpu', workers=1, jsonl=folder / f'{arm}.jsonl', allow_dirty=False,
                         threads_for_tests=True)
        state = git_state or self.head()
        with mock.patch('h7c_arena.play_game', fake_play_game), mock.patch.object(cli, '_git', lambda _: dict(state)):
            record = sa.play(args, log=QUIET, root=self.root)
        if save:
            from hybrid.h7r_ledger import publish_json
            publish_json(folder / f'{arm}.json', record)
        return record

    def play_all(self):
        return {a['name']: self.play(a['name']) for a in self.arena['arms']}

    def judge(self, folder=None):
        return sa.judge(self.args(arms_dir=folder or self.arms_dir), self.root)

    def verify(self):
        return sa.verify_judge(self.args(arms_dir=self.arms_dir), self.root)


class ContractTest(ArenaBase):
    def test_the_arena_contract_must_be_the_one_the_contract_gives(self):
        a = sa.load(self.args(), self.root)
        self.assertEqual(sa.suites(a['arena']), list(SUITES))
        bad = json.loads((self.root / st.ARENA_CONTRACT).read_text())
        bad['meaning'] = 'another reading'
        self.commit_record(st.ARENA_CONTRACT, bad)  # committed, yet not the contract's Arena
        with self.assertRaisesRegex(ValueError, 'not the one the S400 contract'):
            sa.load(self.args(), self.root)
        self.commit_record(st.ARENA_CONTRACT, self.arena)
        sa.load(self.args(), self.root)
        git(self.root, 'rm', '-q', '--cached', st.ARENA_CONTRACT)
        git(self.root, 'commit', '-qm', 'untracked')
        with self.assertRaisesRegex(ValueError, 'the Arena contract is not committed'):
            sa.load(self.args(), self.root)

    def test_check_contract_before_and_after_its_commit(self):
        self.assertTrue(sa.check_contract_file(self.args(), self.root)['committed'])
        git(self.root, 'rm', '-q', '--cached', st.ARENA_CONTRACT)  # written, not yet committed
        out = sa.check_contract_file(self.args(), self.root)
        self.assertEqual((out['committed'], out['arena_contract_sha256']),
                         (False, p1.file_sha256(self.root / st.ARENA_CONTRACT)))
        (self.root / st.ARENA_CONTRACT).write_text(json.dumps({**self.arena, 'meaning': 'another'}))
        with self.assertRaisesRegex(ValueError, 'not the one the S400 contract'):
            sa.check_contract_file(self.args(), self.root)
        (self.root / st.ARENA_CONTRACT).unlink()
        with self.assertRaisesRegex(ValueError, 'no Arena contract'):
            sa.check_contract_file(self.args(), self.root)

    def test_arena_approval_is_required(self):
        c = json.loads((self.root / st.CONTRACT).read_text())
        c['approved']['arena'] = False
        self.write_contract(c)
        git(self.root, 'commit', '-qam', 'unapprove')
        with self.assertRaisesRegex(ValueError, 'arena'):
            sa.load(self.args(), self.root)


class ModelCheckTest(ArenaBase):
    def test_every_model_file_is_checked_before_any_game(self):
        out = sa.models_command(self.args(), self.root, log=QUIET)
        self.assertEqual(out['models_checked'], ['S', 'T', 'champion'])
        s_meta = self.root / st.RUNS / 'S-20261024/best.json'
        s_meta.write_text('another metadata')
        with self.assertRaisesRegex(ValueError, r'S runs/h7r/R2-S400-TRAIN/S-20261024/best.json: [0-9a-f]{12}\.\.\.'):
            sa.models_command(self.args(), self.root, log=QUIET)
        name = self.arena['arms'][0]['name']
        with self.assertRaisesRegex(ValueError, 'model files differ'):
            self.play(name, save=False)
        self.assertFalse((self.arms_dir / f'{name}.jsonl').exists())  # refused before the journal is opened
        s_meta.write_text('S metadata')
        (self.root / p1.PATHS['c5_value']).with_name('best.json').unlink()
        with self.assertRaisesRegex(ValueError, r'runs/h7c/c5/value/best.json: missing'):
            sa.models_command(self.args(), self.root, log=QUIET)


class PlayJudgeTest(ArenaBase):
    def test_play_resume_judge_and_verify(self):
        import shutil

        records = self.play_all()
        names = list(records)
        # refusals first (nothing is published): an arm from another commit, a missing arm, a changed record
        other = self.root / 'runs/other-commit'
        for n in names[1:]:
            other.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.arms_dir / f'{n}.json', other / f'{n}.json')
        self.play(names[0], git_state={'commit': 'f' * 40, 'dirty': False}, folder=other)
        with self.assertRaisesRegex(ValueError, 'not played from one clean code commit'):
            self.judge(other)
        (other / f'{names[0]}.json').unlink()
        with self.assertRaises(OSError):
            self.judge(other)
        bad = json.loads(json.dumps(records[names[0]]))
        bad['games'][0]['result'] = 'draw'  # a game its replay contradicts
        (other / f'{names[0]}.json').write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            self.judge(other)
        self.assertFalse((self.root / sa.JUDGE).exists())
        for name, rec in records.items():
            self.assertEqual((rec['summary']['pairs'], rec['summary']['valid'], len(rec['games'])), (300, True, 600))
            self.assertEqual(rec['identity']['contract_id'], 'R2-S400-TRAIN-ARENA-1')
        for s in SUITES:  # one pairing per suite: the same openings
            self.assertEqual([o['history_id'] for o in records[f'{s}_S_vs_champion']['openings']],
                             [o['history_id'] for o in records[f'{s}_T_vs_champion']['openings']])
        # resume: a cut journal plays only the cut games, to the same record
        name = f'{SUITES[0]}_S_vs_champion'
        jsonl = self.arms_dir / f'{name}.jsonl'
        lines = jsonl.read_text().splitlines()
        jsonl.write_text('\n'.join(lines[:400]) + '\n')
        again = self.play(name, save=False)
        self.assertEqual((again['games_played_now'], again['summary']), (200, records[name]['summary']))
        out = self.judge()
        # the estimates are the 50:50 means of the pair scores
        score = {n: {p['opening_id']: p['pair_score'] for p in ev_table(r)} for n, r in records.items()}
        p_est = sum(sum(score[f'{s}_S_vs_champion'].values()) / 300 for s in SUITES) / 2 - 0.5
        d_est = sum(sum(v - score[f'{s}_T_vs_champion'][k] for k, v in score[f'{s}_S_vs_champion'].items()) / 300
                    for s in SUITES) / 2
        self.assertAlmostEqual(out['bounds']["P'"]['estimate'], round(p_est, 6))
        self.assertAlmostEqual(out['bounds']["D'"]['estimate'], round(d_est, 6))
        self.assertGreater(d_est, 0)
        for key, (hi, lo) in (("P'", ('STRONGER', 'WEAKER')), ("D'", ('BETTER', 'WORSE'))):
            b = out['bounds'][key]
            want = hi if b['lower_one_sided'] > 0 else lo if b['upper_one_sided'] < 0 else 'NOT_ESTABLISHED'
            self.assertEqual(out['readings'][key], want)
        self.assertEqual(out['bounds']['units'], {s: 300 for s in SUITES})
        self.assertEqual(out['sensitivity_class_level']['bounds']['unit'], 'class')
        self.assertEqual(out['arms_commit'], out['git']['commit'])
        self.assertEqual(out['arm_records_sha256'],
                         {n: p1.file_sha256(self.arms_dir / f'{n}.json') for n in records})
        self.assertEqual(json.loads((self.root / sa.JUDGE).read_text()), out)
        with self.assertRaisesRegex(ValueError, 'exists: never overwritten'):
            self.judge()
        self.assertEqual(self.verify(), out)  # recomputed from the arm records
        again = {k: v for k, v in out.items() if k != 'git'}
        changed = {**out, 'readings': {**out['readings'], "P'": 'STRONGER' if out['readings']["P'"] != 'STRONGER'
                                       else 'WEAKER'}}
        with mock.patch.object(sa, 'compute', return_value=again):  # the recomputation just checked, reused
            (self.root / sa.JUDGE).write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'not the judgment its arm records give'):
                self.verify()
            (self.root / sa.JUDGE).write_text(json.dumps({**out, 'git': {'commit': '0' * 40, 'dirty': False}}))
            with self.assertRaisesRegex(ValueError, "not judged from a clean tree at the arms' commit"):
                self.verify()


def ev_table(record):
    from hybrid import h7r_eval as ev
    return ev.pair_table(record['games'], record['openings'])['pairs']


class PairingTest(unittest.TestCase):
    """D' and P' are read only when S and T of each suite are the same openings in the same order (and classes),
    with the arm's full number of pairs."""

    def setUp(self):
        arms = [{'name': f'{s}_{m}_vs_champion', 'pairing': s, 'pairs': 3} for s in SUITES for m in 'ST']
        self.arena = {'arms': arms}
        self.tables = {}
        for s in SUITES:
            for m, score in (('S', 1.0), ('T', 0.5)):
                self.tables[f'{s}_{m}_vs_champion'] = {'pairs': [
                    {'opening_id': f'{s}-{i}', 'class_id': f'c{i}', 'pair_score': score} for i in range(3)]}

    def test_matching_pairs_are_read(self):
        strata = sa.paired_strata(self.arena, self.tables)
        self.assertEqual(strata[SUITES[0]], [(1.0, 0.5, 'c0'), (1.0, 0.5, 'c1'), (1.0, 0.5, 'c2')])
        b = sa.bounds(strata, {'alpha_one_sided': 0.025, 'bootstrap': {'samples': 200, 'seed': 1}}, 'opening')
        self.assertEqual((b["P'"]['estimate'], b["D'"]['estimate']), (0.5, 0.5))
        self.assertEqual(sa.readings(b), {"P'": 'STRONGER', "D'": 'BETTER'})

    def test_any_mismatch_is_refused(self):
        t = f'{SUITES[1]}_T_vs_champion'
        cases = {
            'same openings in the same order': lambda tb: tb[t]['pairs'].reverse(),
            'not played on the same openings': lambda tb: tb[t]['pairs'][0].update(opening_id='other'),
            'differ in class': lambda tb: tb[t]['pairs'][1].update(class_id='cX'),
            'the sample is never reduced': lambda tb: tb[t]['pairs'].pop(),
        }
        for message, change in cases.items():
            with self.subTest(message):
                tables = json.loads(json.dumps(self.tables))
                change(tables)
                with self.assertRaisesRegex(ValueError, message):
                    sa.paired_strata(self.arena, tables)

    def test_each_suite_has_one_s_and_one_t_arm(self):
        arena = {'arms': [a for a in self.arena['arms'] if a['name'] != f'{SUITES[0]}_T_vs_champion']}
        with self.assertRaisesRegex(ValueError, 'the arms are not S and T'):
            sa.paired_strata(arena, self.tables)

    def test_readings_follow_the_contract(self):
        b = lambda lo, hi: {'lower_one_sided': lo, 'upper_one_sided': hi}  # noqa: E731
        self.assertEqual(sa.readings({"P'": b(-0.02, -0.001), "D'": b(-0.01, 0.02)}),
                         {"P'": 'WEAKER', "D'": 'NOT_ESTABLISHED'})
        self.assertEqual(sa.readings({"P'": b(0.0, 0.03), "D'": b(-0.03, -0.0001)}),
                         {"P'": 'NOT_ESTABLISHED', "D'": 'WORSE'})


ARENA_STUB = """import json, os, signal, subprocess, sys
from pathlib import Path
D = 'docs/h7r/r2/R2-S400-TRAIN'
A = 'docs/h7r/contracts-v2/R2-S400-TRAIN-ARENA-1.json'
cmd = sys.argv[1]
def arg(name):
    return sys.argv[sys.argv.index(name) + 1]
def stop(message):
    print('STOP: ValueError: ' + message); sys.exit(1)
def kill():
    sys.stdout.flush()
    os.killpg(os.getpgid(0), signal.SIGKILL)
head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
if os.environ.get('S5_FORBID_ARENA') == cmd:
    print('STOP: must not run again'); sys.exit(9)
if cmd == 'check-contract':
    if json.loads(Path(A).read_text()).get('format') != 'stand-in arena' or os.environ.get('S5_BAD_CONTRACT'):
        stop('not the Arena contract the S400 contract gives')
    print('contract checked'); sys.exit(0)
if cmd == 'check-models':
    if os.environ.get('S5_BAD_MODELS'):
        stop('model files differ from the Arena contract')
    print('models checked'); sys.exit(0)
arms = [a['name'] for a in json.loads(Path(A).read_text())['arms']]
def one_commit(folder):
    if {json.loads(Path(folder, a + '.json').read_text())['head'] for a in arms} != {head}:
        stop('the arms were not played from one clean code commit')
if cmd == 'play':
    arm, out, journal = arg('--arm'), Path(arg('--output')), Path(arg('--jsonl'))
    if out.exists():
        stop(str(out) + ' exists')
    with journal.open('a') as f:
        f.write(json.dumps({'arm': arm, 'head': head}) + chr(10))
    if os.environ.get('S5_PLAY_FAIL') == arm:
        print('STOP: RuntimeError: CUDA error'); sys.exit(1)
    games = len(journal.read_text().splitlines())
    out.write_text(json.dumps({'arm': arm, 'head': head, 'journal_lines': games}))
    print('played', arm); sys.exit(0)
judge = Path(D, 'judge.json')
if cmd == 'judge':
    one_commit(arg('--arms-dir'))
    if judge.exists():
        stop(str(judge) + ' exists')
    judge.write_text(json.dumps({'format': 'stand-in judge', 'readings': {'P': 'NOT_ESTABLISHED', 'D': 'BETTER'},
                                 'git': {'commit': head, 'dirty': False}}))
    if os.environ.get('S5_KILL') == 'judge:after':
        kill()
    print('judged'); sys.exit(0)
if cmd == 'verify-judge':
    one_commit(arg('--arms-dir'))
    if json.loads(judge.read_text())['git']['commit'] != head or os.environ.get('S5_VERIFY_FAIL'):
        stop('not the judgment its arm records give')
    print('verified'); sys.exit(0)
stop('unknown command')
"""


class ArenaBlockTest(unittest.TestCase):
    """s5-AC and s5-A of the draft (section 9.3) run by bash on a real temporary repository with a bare origin; the
    training and Arena tools are stand-ins (``tests.test_h7r_s400_train.BlockTest``'s repository)."""

    from tests.test_h7r_s400_train import BlockTest as _B
    bash, files, through, head, assert_pushed = _B.bash, _B.files, _B.through, _B.head, _B.assert_pushed

    def setUp(self):
        self._B.setUp(self)
        (self.work / 'scripts/h7r_s400_arena.py').write_text(ARENA_STUB)
        git(self.work, 'add', 'scripts/h7r_s400_arena.py')
        git(self.work, 'commit', '-qm', 'arena stand-in')
        git(self.work, 'push', '-q', 'origin', 'feat/h7-c1')
        self.A = 'docs/h7r/contracts-v2/R2-S400-TRAIN-ARENA-1.json'
        self.AR = self.work / self.R / 'arena'
        self.arms = [f'{s}_{m}_vs_champion' for s in SUITES for m in 'ST']

    def logs(self, prefix):
        return sorted(f for f in self.files(f'{self.D}/stage-logs') if Path(f).name.startswith(prefix)
                      and f.endswith('.log'))

    def trained(self):
        self.through('s5-1', 's5-G', 's5-S', 's5-R', 's5-T')

    def staged(self):
        return git(self.work, 'diff', '--cached', '--name-only')

    def test_contract_then_arena_and_judge(self):
        self.trained()
        code, out = self.bash('s5-A')
        self.assertEqual(code, 1)
        self.assertIn('the Arena contract is not recorded (run s5-AC first)', out)
        self.assertFalse((self.AR / 'HEAD').exists())
        self.through('s5-AC')
        self.assertTrue(self.files(self.A))
        self.assertEqual((len(self.logs('arena-contract-')), len(self.logs('arena-check-'))), (1, 1))
        self.assertIn('Arena contract exit 0', git(self.work, 'log', '-1', '--format=%s'))
        self.assert_pushed()
        code, out = self.bash('s5-AC')
        self.assertEqual(code, 1)
        self.assertIn('the Arena contract is already recorded', out)
        head = self.head()
        self.through('s5-A')
        self.assertEqual((self.AR / 'HEAD').read_text().strip(), head)
        for arm in self.arms:
            self.assertEqual(json.loads((self.AR / f'{arm}.json').read_text())['head'], head)
            self.assertEqual(len([f for f in self.logs('play-') if f'play-{arm}-' in f]), 1)
        self.assertTrue(self.files(f'{self.D}/judge.json'))
        self.assertEqual((len(self.logs('check-models-')), len(self.logs('judge-'))), (1, 1))
        self.assertEqual(git(self.work, 'log', '-1', '--format=%s'), 'H7-R S400-TRAIN: Arena and judge (records only)')
        self.assert_pushed()
        code, out = self.bash('s5-A')
        self.assertEqual(code, 1)
        self.assertIn('the judge is already recorded', out)

    def test_a_contract_written_before_a_kill_is_checked_and_recorded(self):
        self.trained()
        code, out = self.bash('s5-AC', S5_KILL='arena-contract:after')
        self.assertNotEqual(code, 0, out)
        self.assertTrue((self.work / self.A).exists())
        code, out = self.bash('s5-AC', S5_FORBID='1', S5_BAD_CONTRACT='1')
        self.assertEqual(code, 1, out)
        self.assertIn('the Arena contract does not check; nothing committed', out)
        self.assertFalse(self.files(self.A))
        code, out = self.bash('s5-AC', S5_FORBID='1')
        self.assertEqual(code, 0, out)
        self.assertTrue(self.files(self.A))
        self.assertEqual(len(self.logs('arena-contract-')), 1)  # the killed run's log is kept and recorded
        self.assertEqual(len(self.logs('arena-check-')), 2)
        self.assert_pushed()

    def test_wrong_model_files_stop_before_any_game(self):
        self.trained()
        self.through('s5-AC')
        before = self.head()
        code, out = self.bash('s5-A', S5_BAD_MODELS='1')
        self.assertEqual(code, 1, out)
        self.assertIn("the model files are not the Arena contract's; nothing played or committed", out)
        self.assertEqual(sorted(p.name for p in self.AR.iterdir()), ['HEAD'])  # no journal, no arm record
        self.assertEqual((self.head(), self.staged()), (before, ''))
        self.through('s5-A')  # fixed: the same HEAD, played and judged
        self.assertEqual(len(self.logs('check-models-')), 2)
        self.assert_pushed()

    def test_a_failed_arm_resumes_at_the_pinned_head(self):
        self.trained()
        self.through('s5-AC')
        pinned = self.head()
        code, out = self.bash('s5-A', S5_PLAY_FAIL=self.arms[1])
        self.assertEqual(code, 1, out)
        self.assertIn(f'arm {self.arms[1]} failed (exit 1); nothing committed, HEAD kept', out)
        self.assertEqual((self.head(), self.staged()), (pinned, ''))
        self.assertTrue((self.AR / f'{self.arms[0]}.json').exists())
        git(self.work, 'commit', '-q', '--allow-empty', '-m', 'moved')
        code, out = self.bash('s5-A')
        self.assertEqual(code, 1)
        self.assertIn('HEAD moved since the Arena started', out)
        git(self.work, 'reset', '-q', '--hard', pinned)
        code, out = self.bash('s5-A')
        self.assertEqual(code, 0, out)
        self.assertIn(f'done: {self.arms[0]}', out)
        rec = json.loads((self.AR / f'{self.arms[1]}.json').read_text())
        self.assertEqual((rec['head'], rec['journal_lines']), (pinned, 2))  # resumed from its journal
        self.assertEqual(len([f for f in self.logs('play-') if f'play-{self.arms[1]}-' in f]), 2)  # failure kept
        self.assertEqual(len([f for f in self.logs('play-') if f'play-{self.arms[0]}-' in f]), 1)
        self.assert_pushed()

    def check_recovered(self, **env):
        self.trained()
        self.through('s5-AC')
        pinned = self.head()
        code, out = self.bash('s5-A', **env)
        self.assertNotEqual(code, 0, out)
        self.assertEqual(self.head(), pinned)
        self.assertTrue((self.work / self.D / 'judge.json').exists())
        code, out = self.bash('s5-A', S5_VERIFY_FAIL='1', S5_FORBID_ARENA='judge')
        self.assertEqual(code, 1, out)
        self.assertIn('is not what the arm records give: report it; nothing committed', out)
        self.assertEqual(self.head(), pinned)
        code, out = self.bash('s5-A', S5_FORBID_ARENA='judge')  # verified, never judged again
        self.assertEqual(code, 0, out)
        self.assertIn('verifying it against the arm records (no new judge)', out)
        self.assertTrue(self.files(f'{self.D}/judge.json'))
        self.assertEqual(len(self.logs('verify-judge-')), 2)
        self.assertEqual(len(self.logs('judge-')), 1)
        self.assert_pushed()

    def test_a_judge_killed_after_its_record_is_verified_not_judged_again(self):
        self.check_recovered(S5_KILL='judge:after')

    def test_an_arena_interrupted_after_staging_is_verified_and_recorded(self):
        self.check_recovered(S5_KILL_AT_COMMIT='1')


if __name__ == '__main__':
    unittest.main()
