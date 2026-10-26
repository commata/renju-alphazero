"""B2 program/decision/guard contracts and resumable stage execution, without match engine dependencies."""
import copy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from hybrid import h7c_b2 as b2, h7c_recipe
from hybrid.h6_labels import RESERVED_SEEDS, check_reserved_seed
from scripts import h7c_loop as loop

ROOT = Path(__file__).resolve().parents[1]


class ProgramTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for relative in ('docs/h7c/b2_program.json', 'docs/h7c/c3/arena_verdict.json', 'configs/h7c/c1_value.json'):
            dest = self.root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, dest)
        self.path = self.root / 'docs/h7c/b2_program.json'
        self.p, self.sha = b2.load_program(self.path, self.root)
        self.v = b2.read(self.root / 'docs/h7c/c3/arena_verdict.json')

    def decision(self, cycle=4, previous=None, previous_sha=None):
        return b2.decision(self.p, self.sha, previous or self.v,
                           previous_sha or self.p['start']['verdict_sha256'], cycle)

    def test_seed_registration_and_no_training_overlap(self):
        all_eval = set()
        for c in range(4, 11):
            s = b2.seeds(self.p, c)
            self.assertEqual(s['selfplay_run_seed'], 8800+c)
            check_reserved_seed(s['selfplay_run_seed'])
            for field in ('arena', 'x2', 'drift'):
                self.assertIn(s[field], RESERVED_SEEDS)
                self.assertNotIn(s[field], all_eval)
                all_eval.add(s[field])
                with self.assertRaises(ValueError):
                    check_reserved_seed(s[field])
        for c in (3, 11):
            with self.assertRaises(ValueError):
                b2.seeds(self.p, c)

    def test_deterministic_first_decision(self):
        d = self.decision()
        self.assertEqual(d, self.decision())
        self.assertEqual(d['parent'], self.v['next_champion'])
        self.assertEqual(d['program']['sha256'], self.sha)
        b2.validate_decision(d, self.root)

    def test_recipe_reader_validates_program_reference(self):
        d = self.decision()
        dp = self.root / 'decision.json'
        loop.atomic(dp, d)
        with mock.patch.object(b2, 'ROOT', self.root):
            self.assertEqual(h7c_recipe.load(dp, 4)[0], d)
            d['seeds']['selfplay_run_seed'] = 8809
            loop.atomic(dp, d)
            with self.assertRaisesRegex(ValueError, 'deterministic'):
                h7c_recipe.load(dp, 4)

    def test_changed_program_or_start_verdict_is_rejected(self):
        d = self.decision()
        p = b2.read(self.path)
        p['approved_on'] = 'changed'
        loop.atomic(self.path, p)
        with self.assertRaisesRegex(ValueError, 'program SHA'):
            b2.validate_decision(d, self.root)
        with (self.root / self.p['start']['verdict']).open('ab') as f:
            f.write(b' ')
        with self.assertRaisesRegex(ValueError, 'start verdict SHA'):
            b2.load_program(self.path, self.root)

    def test_program_cannot_change_rules(self):
        for field, value in (('last_cycle', 11), ('recipe', 'JOINT'), ('stops', {}), ('drift', {}),
                             ('promotion', {}), ('seeds', {}), ('execution', {})):
            with self.subTest(field=field):
                p = copy.deepcopy(self.p)
                p[field] = value
                loop.atomic(self.path, p)
                with self.assertRaises(ValueError):
                    b2.load_program(self.path, self.root)

    def test_following_cycle_requires_program_bound_champion(self):
        previous = {**self.v, 'cycle': 4, 'b2_program_sha256': self.sha}
        d = self.decision(5, previous, 'a'*64)
        self.assertEqual(d['champion_record_sha256'], 'a'*64)
        previous['b2_program_sha256'] = 'b'*64
        with self.assertRaises(ValueError):
            self.decision(5, previous, 'a'*64)

    def test_policy_changes_and_path_escape_rejected(self):
        previous = copy.deepcopy(self.v)
        previous['next_champion']['policy_checkpoint_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'policy changed'):
            self.decision(previous=previous)
        for path in ('../outside', str(self.root.resolve())):
            with self.assertRaises(ValueError):
                b2.contained(self.root, path)


def verdict(cycle=4, promote=False, tactical='PASS'):
    return {'cycle': cycle, 'promotion': {'decision': 'PROMOTE' if promote else 'KEEP_CHAMPION'},
            'tactical': {'verdict': tactical}}


class StopTest(unittest.TestCase):
    def setUp(self):
        self.p = b2.read(ROOT / 'docs/h7c/b2_program.json')

    def test_b1_convergence_does_not_stop_b2(self):
        v = {**verdict(), 'convergence': {'converged': True}}
        self.assertIsNone(b2.stop_reason(self.p, [v]))

    def test_no_promotion_streak_and_reset(self):
        self.assertEqual(b2.stop_reason(self.p, [verdict(c) for c in (4, 5, 6)]), 'NO_PROMOTION_STREAK')
        self.assertIsNone(b2.stop_reason(self.p, [verdict(4), verdict(5, True), verdict(6)]))

    def test_no_candidate_counts_as_no_promotion_not_tactical_failure(self):
        rows = [verdict(c, tactical=None) for c in (4, 5, 6)]
        self.assertEqual(b2.stop_reason(self.p, rows), 'NO_PROMOTION_STREAK')
        self.assertIsNone(b2.stop_reason(self.p, rows[:2]))

    def test_tactical_failure_streak_and_reset(self):
        self.assertEqual(b2.stop_reason(self.p, [verdict(4, tactical='FAIL'), verdict(5, tactical='FAIL')]),
                         'TACTICAL_FAIL_STREAK')
        self.assertIsNone(b2.stop_reason(self.p, [verdict(4, tactical='FAIL'), verdict(5, True), verdict(6, tactical='FAIL')]))

    def test_max_cycle_and_drift_priority(self):
        self.assertEqual(b2.stop_reason(self.p, [verdict(10, True)]), 'MAX_CYCLE')
        self.assertEqual(b2.stop_reason(self.p, [verdict(10, True)], {'stop': True, 'reason': 'DRIFT_DETECTED'}),
                         'DRIFT_DETECTED')

    def test_invalid_official_evaluation_stops_without_changing_verdict(self):
        v = {**verdict(), 'official': {'valid': False}}
        self.assertEqual(b2.stop_reason(self.p, [v]), 'OFFICIAL_INVALID')
        self.assertEqual(v['promotion']['decision'], 'KEEP_CHAMPION')

    def test_drift_boundary_and_invalid_results(self):
        s = {'games': 100, 'complete_pairs': 50, 'incomplete_pairs': [], 'truncated': 0,
             'opponent_illegal_pairs': [], 'illegal': {'tested': 0, 'opponent': 0}, 'improvement_ci95': [-.1, 0.]}
        self.assertFalse(b2.drift_judge(s)['stop'])  # upper == 0 is NOT below zero
        self.assertTrue(b2.drift_judge({**s, 'improvement_ci95': [-.2, -.005]})['alarm'])
        for change in ({'complete_pairs': 49}, {'truncated': 1}, {'illegal': {'tested': 1}},
                       {'improvement_ci95': None}, {'opponent_illegal_pairs': [1]}):
            with self.subTest(change=change):
                j = b2.drift_judge({**s, **change})
                self.assertEqual(j['reason'], 'DRIFT_INVALID')
                self.assertTrue(j['stop'])


class StageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / 'input.json'
        self.input.write_text('input')
        self.output, self.jsonl = self.root / 'out.json', self.root / 'games.jsonl'
        self.identity = {'git': {'commit': 'fixed', 'dirty': False}, 'program_sha256': 'a'*64}
        self.calls = []
        def executor(argv, log, root, running):
            self.calls.append(argv)
            self.output.write_text('valid')
            self.jsonl.write_text('game')
            return 0
        self.executor = executor

    def stages(self, **kw):
        return loop.Stages(self.root, self.root / 'receipts', self.identity, kw.get('executor', self.executor))

    def step(self, store, **kw):
        store.step('c4-official', ['python', 'match.py'], [self.input], [self.output, self.jsonl],
                   lambda: self.assertEqual(self.output.read_text(), 'valid'), **kw)

    def test_completed_stage_skips_only_when_hashes_match(self):
        self.step(self.stages())
        self.step(self.stages())
        self.assertEqual(len(self.calls), 1)

    def test_output_tamper_or_deletion_blocks_resume(self):
        self.step(self.stages())
        self.output.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'output changed'):
            self.stages()
        self.output.unlink()
        with self.assertRaises(ValueError):
            self.stages()

    def test_input_tamper_blocks_completed_stage_skip(self):
        self.step(self.stages())
        self.input.write_text('other')
        with self.assertRaisesRegex(ValueError, 'input bytes changed'):
            self.step(self.stages())

    def test_changed_command_or_code_blocks_resume(self):
        self.step(self.stages())
        with self.assertRaisesRegex(ValueError, 'argv'):
            self.stages().step('c4-official', ['other'], [self.input], [self.output], lambda: None)
        self.identity['git']['commit'] = 'new-code'
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.stages()

    def test_existing_output_without_receipt_is_never_adopted(self):
        self.output.write_text('valuable')
        with self.assertRaisesRegex(ValueError, 'unreceipted output'):
            self.step(self.stages())
        self.assertEqual(self.output.read_text(), 'valuable')
        self.assertEqual(self.calls, [])

    def test_interrupted_game_jsonl_resumes(self):
        def failed(argv, log, root, running):
            self.jsonl.write_text('partial')
            return 2
        with self.assertRaisesRegex(ValueError, 'command failed'):
            self.step(self.stages(executor=failed), resume_jsonl=self.jsonl)
        self.step(self.stages(), resume_jsonl=self.jsonl)
        self.assertEqual(len(self.calls), 1)

    def test_partial_final_file_is_not_overwritten(self):
        def failed(argv, log, root, running):
            self.output.write_text('partial model/cache')
            return 2
        with self.assertRaises(ValueError):
            self.step(self.stages(executor=failed))
        with self.assertRaisesRegex(ValueError, 'unreceipted output'):
            self.step(self.stages())
        self.assertEqual(self.output.read_text(), 'partial model/cache')

    def test_live_child_blocks_second_executor(self):
        s = self.stages()
        s.state['stages']['c4-official'] = {'status': 'RUNNING', 'pid': 123}
        s.save()
        with mock.patch.object(loop, 'alive', return_value=True), self.assertRaisesRegex(ValueError, 'still alive'):
            self.stages()

    def test_write_once_never_replaces_evidence(self):
        loop.write_once(self.output, b'first')
        loop.write_once(self.output, b'first')
        with self.assertRaisesRegex(ValueError, 'protected output'):
            loop.write_once(self.output, b'second')

    def test_lock_excludes_duplicate_loop(self):
        with loop.exclusive(self.root / 'lock'):
            with self.assertRaisesRegex(ValueError, 'another B2'):
                with loop.exclusive(self.root / 'lock'):
                    self.fail('second lock acquired')


if __name__ == '__main__':
    unittest.main()
