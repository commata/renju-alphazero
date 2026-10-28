"""H7-R confirmations end to end (docs/h7-rl-plan.md 9.7): the versioned INVALID policy, the h7r tactical record of
a candidate without cycle lineage, the judgment order and its ledger, the forbidden-move attribution. Fixtures,
stand-in networks and mocked runners only: no real evaluation, Rapfi or GPU."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from hybrid import h7r_eval as ev, h7r_evidence as evidence
from hybrid.h7r_ledger import bind, bind_all, conflicts
from scripts import h7r_eval as cli, h7r_holdout as ho
from tests.h7r_helpers import selection_source, tactical_record, tactical_results, timing_record
from tests.test_h7r_cli import CLEAN, QUIET, fake_play_game, world
from tests import test_h7r_eval as eval_tests
from tests.test_h7r_eval import STATS, games, openings

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / 'docs/h7r/invalid_policy_v1.draft.json').read_text())['invalid_policy']
ABNORMAL = json.loads((ROOT / 'tests/fixtures/h7r_abnormal_end_branch.json').read_text())
PREFIX, FIXED, CONT = ABNORMAL['prefix'], ABNORMAL['fixed_move'], ABNORMAL['continuation']
STRUCTURAL = {'status': 'INVALID', 'reason': 'abnormal_end', 'score': None, 'continuation': CONT,
              'length': len(PREFIX) + 1 + len(CONT)}
SPEC = {'margin': -0.02, 'alpha': 0.025, 'min_observations': 20}


def key(move, moves=PREFIX):
    import h7c_adjudicate as ad
    return ad.branch_key(moves, move)


def record(move=FIXED, **fields):
    return {'key': key(move), 'ply': len(PREFIX), 'move': list(move), **fields}


def task(move=FIXED):
    return {'key': key(move), 'moves': PREFIX, 'move': list(move)}


def ledger_files(root):
    folder = Path(root) / 'runs/h7r/confirmations'
    return {p.name: p.read_bytes() for p in folder.glob('*.json')} if folder.exists() else {}


class PolicyContractTest(unittest.TestCase):
    def test_draft_policy_is_valid_and_every_field_is_checked(self):
        ev.check_invalid_policy(POLICY)
        self.assertEqual((POLICY['max_invalid_share'], POLICY['denominator']), (0.05, 'disagreement_observations'))
        for change in ({'version': 'h7r-invalid-v0'}, {'structural_abnormal_end': 'draw'}, {'unresolved': 'drop'},
                       {'unresolved_reasons': []}, {'unresolved_reasons': ['timeout', 'truncated']},
                       {'unresolved_reasons': ['timeout', 'timeout']}, {'max_invalid_share': 1},
                       {'max_invalid_share': -0.01}, {'max_invalid_share': float('nan')},
                       {'denominator': 'branches'}):
            with self.subTest(change), self.assertRaises(ValueError):
                ev.check_invalid_policy({**POLICY, **change})
        with self.assertRaises(ValueError):
            ev.check_invalid_policy({k: v for k, v in POLICY.items() if k != 'denominator'})

    def test_contract_checks_a_policy_and_keeps_strict_without_one(self):
        from tests.test_h7r_eval import contract
        with TemporaryDirectory() as tmp:
            body = contract(Path(tmp))
            ev.check_contract(body, 'strict', tmp, check_files=False)  # no policy: the strict rule
            body['selection']['invalid_policy'] = POLICY
            ev.check_contract(body, 'policy', tmp, check_files=False)
            body['selection']['invalid_policy'] = {**POLICY, 'max_invalid_share': 0.5, 'unresolved': 'ignore'}
            with self.assertRaisesRegex(ValueError, 'worst_case'):
                ev.check_contract(body, 'bad', tmp, check_files=False)


class BranchOutcomeTest(unittest.TestCase):
    def test_structural_abnormal_end_only_by_policy(self):
        r = record(**STRUCTURAL)
        self.assertEqual(evidence.branch_outcome(r, task(), POLICY), {'kind': 'structural', 'score': 0.5,
                                                                       'reason': 'abnormal_end'})
        unresolved = {**POLICY, 'structural_abnormal_end': 'unresolved'}
        self.assertEqual(evidence.branch_outcome(r, task(), unresolved)['kind'], 'unresolved')
        self.assertEqual(evidence.branch_outcome(r, task(), None)['kind'], 'unresolved')  # strict
        self.assertEqual(r['status'], 'INVALID')  # the source record is not rewritten

    def test_condition_mismatch_and_tampering_are_data_errors(self):
        cases = {
            'not over': record(**{**STRUCTURAL, 'continuation': CONT[:-2], 'length': STRUCTURAL['length'] - 2}),
            'illegal move': record(**{**STRUCTURAL, 'continuation': [*CONT[:5], CONT[4], *CONT[6:]]}),
            'length': record(**{**STRUCTURAL, 'length': STRUCTURAL['length'] - 1}),
            'scored': record(**{**STRUCTURAL, 'score': 0.5}),
            'valid claim': record(**{**STRUCTURAL, 'status': 'VALID', 'score': 0.5, 'winner': None}),
            'reason': record(**{**STRUCTURAL, 'reason': 'truncated'}),
        }
        for name, r in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                evidence.branch_outcome(r, task(), POLICY)
        with self.assertRaisesRegex(ValueError, 'wrong fixed branch'):
            evidence.branch_outcome({**record(**STRUCTURAL), 'move': [14, 13]}, task(), POLICY)
        with self.assertRaisesRegex(ValueError, 'wrong fixed branch'):
            evidence.branch_outcome({**record(**STRUCTURAL), 'ply': 9}, task(), POLICY)
        # the five [7, 6] claimed as an abnormal end: a winner, many empty points
        five = record([7, 6], status='INVALID', reason='abnormal_end', score=None, continuation=[], length=9)
        with self.assertRaisesRegex(ValueError, 'not structural'):
            evidence.branch_outcome(five, task([7, 6]), POLICY)
        valid = record([7, 6], status='VALID', score=1.0, winner='black', continuation=[], length=9)
        self.assertEqual(evidence.branch_outcome(valid, task([7, 6]), POLICY)['score'], 1.0)
        with self.assertRaisesRegex(ValueError, 'differs from replay'):
            evidence.branch_outcome({**valid, 'score': 0.0}, task([7, 6]), POLICY)

    def test_execution_failures(self):
        failed = lambda reason, **kw: record(status='INVALID', reason=reason, score=None,  # noqa: E731
                                             **{'continuation': [], 'length': len(PREFIX) + 1, **kw})
        self.assertEqual(evidence.branch_outcome(failed('timeout'), task(), POLICY)['kind'], 'unresolved')
        self.assertEqual(evidence.branch_outcome(failed('engine_error'), task(), None)['kind'], 'unresolved')
        only_timeout = {**POLICY, 'unresolved_reasons': ['timeout']}
        self.assertEqual(evidence.branch_outcome(failed('engine_error'), task(), only_timeout)['kind'], 'unhandled')
        self.assertEqual(evidence.branch_outcome(failed('engine_illegal', move_at=[7, 7]), task(), POLICY)['kind'],
                         'unresolved')
        black_to_move = {'continuation': [[14, 0]], 'length': len(PREFIX) + 2, 'ply_at': len(PREFIX) + 2}
        self.assertEqual(evidence.branch_outcome(failed('forbid_mismatch', ours_only=[], engine_only=[[3, 3]],
                                                        **black_to_move), task(), POLICY)['kind'], 'unresolved')
        refused = {
            'after the end': failed('timeout', continuation=CONT, length=STRUCTURAL['length']),
            'legal engine move': failed('engine_illegal', move_at=[3, 3]),
            'invented forbidden point': failed('forbid_mismatch', ours_only=[[3, 3]], engine_only=[],
                                               **black_to_move),
            'wrong ply': failed('forbid_mismatch', ours_only=[], engine_only=[[3, 3]],
                                **{**black_to_move, 'ply_at': 3}),
            'not a failure': failed('fixed_move_illegal'),
            'no reason': failed(None),
        }
        for name, r in refused.items():
            with self.subTest(name), self.assertRaises(ValueError):
                evidence.branch_outcome(r, task(), POLICY)


def outcomes_and_obs(spec):
    """``spec``: per observation (game, a kind/score, b kind/score); every branch distinct unless named."""
    outcomes, obs = {}, []
    for i, (game, a, b) in enumerate(spec):
        for side, value in (('a', a), ('b', b)):
            name = value[2] if len(value) > 2 else f'{i}{side}'
            outcomes[name] = {'kind': value[0], 'score': value[1], 'reason': None if value[0] == 'resolved'
                              else 'abnormal_end' if value[0] == 'structural' else 'timeout'}
        obs.append({'game': game, 'key_a': a[2] if len(a) > 2 else f'{i}a', 'key_b': b[2] if len(b) > 2 else f'{i}b'})
    return obs, outcomes


R1, R0, U, S = ('resolved', 1.0), ('resolved', 0.0), ('unresolved', None), ('structural', 0.5)


class PolicySelectionTest(unittest.TestCase):
    def select(self, spec, policy=POLICY, **change):
        obs, outcomes = outcomes_and_obs(spec)
        return ev.policy_selection(obs, outcomes, policy, {**SPEC, **change}, STATS['bootstrap'])

    def test_worst_case_values(self):
        obs, outcomes = outcomes_and_obs([('g0', U, R1), ('g1', R1, U), ('g2', U, U), ('g3', S, R1), ('g4', R0, S),
                                          ('g5', R1, R0)])
        rows = ev.worst_case_rows(obs, outcomes)
        self.assertEqual([r['delta'] for r in rows], [-1.0, 0.0, -1.0, -0.5, -0.5, 1.0])
        self.assertEqual([r['resolved'] for r in rows], [False, False, False, True, True, True])
        self.assertEqual([r['structural'] for r in rows], [False, False, False, True, True, False])

    def test_share_counts_observations_and_shared_branches(self):
        good = [(f'g{i}', R1, R1) for i in range(37)]
        shared = [(f's{i}', ('unresolved', None, 'one'), R1) for i in range(3)]  # one branch in three observations
        out = self.select(good + shared)
        self.assertEqual((out['unresolved_observations'], out['branches']['unresolved_by_reason']), (3, {'timeout': 1}))
        self.assertEqual((out['status'], out['reason'], out['invalid_share']), ('INCONCLUSIVE', 'INVALID_SHARE_EXCEEDED', 0.075))

    def test_share_boundary_zero_and_resolved_minimum(self):
        at = self.select([(f'g{i}', R1, R1) for i in range(38)] + [('u0', U, R1), ('u1', R1, U)])
        self.assertEqual(at['invalid_share'], 0.05)  # 2 / 40: at the maximum, judged by the bootstrap
        self.assertNotEqual(at.get('reason'), 'INVALID_SHARE_EXCEEDED')
        self.assertIn('lower_one_sided', at)
        exact = self.select([(f'g{i}', R1, R1) for i in range(19)] + [('u', R1, U)], min_observations=19)
        self.assertNotEqual(exact.get('reason'), 'INVALID_SHARE_EXCEEDED')  # 1 / 20 is exactly 0.05
        over = self.select([(f'g{i}', R1, R1) for i in range(37)] + [('u0', U, R1), ('u1', U, R1), ('u2', U, R1)])
        self.assertEqual(over['reason'], 'INVALID_SHARE_EXCEEDED')
        self.assertEqual(self.select([])['reason'], 'NO_OBSERVATIONS')
        few = self.select([(f'g{i}', R1, R1) for i in range(19)] + [('u', R1, U)])
        self.assertEqual((few['reason'], few['resolved_observations'], few['observations_total']),
                         ('TOO_FEW_RESOLVED', 19, 20))  # the imputed observation never fills the minimum
        self.assertEqual(self.select([(f'g{i}', R1, R1) for i in range(30)] + [('x', ('unhandled', None), R1)])['reason'],
                         'UNHANDLED_INVALID')

    def test_structural_and_unresolved_observation_counts_can_overlap(self):
        good = [(f'g{i}', R1, R1) for i in range(38)]
        out = self.select(good + [('candidate-draw', S, U), ('champion-draw', U, S)])
        self.assertEqual((out['structural_observations'], out['unresolved_observations'],
                          out['resolved_observations'], out['observations_total']), (2, 2, 38, 40))
        self.assertEqual(out['normalized']['abnormal_end_evaluation_draw'], 2)
        self.assertEqual(out['invalid_share'], 0.05)
        self.assertEqual(out['mean_delta'], -0.025)

    def test_structural_branches_are_counted_apart(self):
        out = self.select([(f'g{i}', R1, R1) for i in range(30)] + [('s', S, R1)])
        self.assertEqual((out['structural_observations'], out['unresolved_observations'], out['invalid_share']),
                         (1, 0, 0.0))
        self.assertEqual(out['normalized']['abnormal_end_evaluation_draw'], 1)

    def test_dropping_invalid_would_pass_but_the_worst_case_does_not(self):
        spec = [(f'g{i}', R1, R1) for i in range(38)] + [('u0', U, R1), ('u1', U, R1)]
        obs, outcomes = outcomes_and_obs(spec)
        dropped = [{'game': r['game'], 'delta': r['delta']} for r in ev.worst_case_rows(obs, outcomes) if r['resolved']]
        self.assertEqual(ev.cluster_noninferiority(dropped, -0.02, 0.025, STATS['bootstrap'], 20)['status'],
                         'NONINFERIOR')
        out = self.select(spec)
        self.assertNotEqual(out['status'], 'NONINFERIOR')
        self.assertEqual(out['mean_delta'], -0.05)


def confirm_world(tmp, policy=POLICY, candidate_moves=None, branch_records=None):
    w = world(Path(tmp))
    if policy is not None:
        w['body']['selection']['invalid_policy'] = policy
    selection_source(w['root'], w['body'], 'pending', candidate_moves, branch_records)
    w['contract'].write_text(json.dumps(w['body']))
    _, w['sha'] = ev.load_contract(w['contract'], w['root'], check_files=False)
    w['source'] = selection_source(w['root'], w['body'], w['sha'], candidate_moves, branch_records)
    return w


class StructuralSelectionTest(unittest.TestCase):
    """Game 0's candidate plays [14, 14]; its branch is the structural abnormal_end fixture."""

    def test_policy_normalizes_and_strict_stays_inconclusive(self):
        moves, records = {0: FIXED}, {tuple(FIXED): STRUCTURAL}
        with TemporaryDirectory() as tmp:
            w = confirm_world(tmp, POLICY, moves, records)
            out = evidence.selection(w['source'], w['body'], w['sha'], w['root'])
            self.assertEqual((out['structural_observations'], out['unresolved_observations'], out['invalid']), (1, 0, 0))
            self.assertEqual(out['normalized']['abnormal_end_evaluation_draw'], 1)
            self.assertEqual((out['mean_delta'], out['invalid_share']), (-0.02, 0.0))  # -0.5 / 25 observations
            self.assertNotEqual(out.get('reason'), 'INVALID_BRANCHES')
            manifest = json.loads(w['source'].read_text())
            branches = ho.read_ref(manifest['branches'], w['source'].parent)
            kept = [json.loads(x) for x in branches.read_text().splitlines() if '"abnormal_end"' in x]
            self.assertEqual(len(kept), 1)
            self.assertEqual((kept[0]['status'], kept[0]['score']), ('INVALID', None))  # the source is unchanged
            # a cut continuation behind the same file reference is refused, never unresolved
            lines = [json.loads(x) for x in branches.read_text().splitlines()]
            for row in lines:
                if row.get('reason') == 'abnormal_end':
                    row.update(continuation=row['continuation'][:-2], length=row['length'] - 2)
            old = branches.read_bytes()
            try:
                branches.write_text(''.join(json.dumps(r) + '\n' for r in lines))
                manifest['branches'] = ho.file_ref(branches, w['source'].parent)
                w['source'].write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'not structural'):
                    evidence.selection(w['source'], w['body'], w['sha'], w['root'])
            finally:
                branches.write_bytes(old)
        with TemporaryDirectory() as tmp:
            w = confirm_world(tmp, None, moves, records)
            out = evidence.selection(w['source'], w['body'], w['sha'], w['root'])
            self.assertEqual((out['status'], out['reason'], out['invalid'], out['invalid_policy']),
                             ('INCONCLUSIVE', 'INVALID_BRANCHES', 1, None))

    def test_unresolved_timeout_takes_the_worst_delta(self):
        timeout = {'status': 'INVALID', 'reason': 'timeout', 'score': None, 'continuation': [],
                   'length': len(PREFIX) + 1}
        with TemporaryDirectory() as tmp:
            w = confirm_world(tmp, POLICY, {0: FIXED}, {tuple(FIXED): timeout})
            out = evidence.selection(w['source'], w['body'], w['sha'], w['root'])
            self.assertEqual((out['unresolved_observations'], out['resolved_observations'], out['invalid_share']),
                             (1, 24, 0.04))
            self.assertEqual(out['mean_delta'], -0.04)  # 0 - 1 for game 0, the rest 1 - 1


class ConfirmFlowTest(unittest.TestCase):
    """A policy contract with stand-in models whose candidate has no cycle metadata: tactical from the command,
    every evidence record, the judgment to PROMOTE and the ledger's behaviour around it."""

    def setUp(self):
        git_patch = mock.patch.object(cli, '_git', return_value=dict(CLEAN))
        git_patch.start()
        self.addCleanup(git_patch.stop)
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.w = confirm_world(self.tmp.name)
        self.root, self.body, self.sha = self.w['root'], self.w['body'], self.w['sha']
        self.arms = {a: self.play(a) for a in ev.ROLE_ARMS}
        sel = cli.selection(argparse.Namespace(contract=self.w['contract'], evidence=self.w['source'],
                                               output=self.root / 'out/sel.json', allow_dirty=True, root=self.root))
        self.selection = self.save(sel, 'sel')
        self.timing = self.save(timing_record(self.body, self.sha, self.root), 'time')
        self.tactical = self.save(self.make_tactical(), 'tact')

    def save(self, value, name):
        path = self.root / 'out' / f'{name}.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def play(self, arm, git=CLEAN, name=None):
        args = argparse.Namespace(contract=self.w['contract'], arm=arm, device='cpu', workers=1,
                                  jsonl=self.root / 'runs' / f'{name or arm}.jsonl', allow_dirty=True,
                                  threads_for_tests=True)
        with mock.patch('h7c_arena.play_game', fake_play_game), mock.patch.object(cli, '_git', lambda _: git):
            return self.save(cli.play(args, log=QUIET, root=self.root), name or arm)

    def make_tactical(self, git=CLEAN, calls=None):
        result = tactical_results()
        calls = [] if calls is None else calls
        args = argparse.Namespace(contract=self.w['contract'], device='cpu', workers=1, allow_dirty=True,
                                  root=self.root)
        with mock.patch.object(cli, '_git', lambda _: git), \
                mock.patch('h7c_arena.value_probes', lambda v, d: calls.append(('value', v)) or deepcopy(result['value_probes'])), \
                mock.patch('h7c_arena.must_block', lambda p: calls.append(('policy', p)) or deepcopy(result['must_block'])), \
                mock.patch('h7c_arena.e1_suite', lambda s, d, n, label, log=print: calls.append(('e1', s)) or deepcopy(result['e1'])):
            return cli.tactical(args, log=QUIET, root=self.root)

    def judge_args(self, **change):
        return argparse.Namespace(**{'contract': self.w['contract'], 'arm_record': list(self.arms.values()),
                                     'selection': self.selection, 'tactical': self.tactical, 'timing': self.timing,
                                     'allow_dirty': True, **change})

    def test_tactical_command_for_a_candidate_without_cycle_lineage(self):
        meta = json.loads((self.w['cand']).with_name('best.json').read_text())
        self.assertFalse({'h7c_stage', 'cycle'} & set(meta))
        calls = []
        record = self.make_tactical(calls=calls)
        self.assertEqual((record['format'], record['contract_sha256'], record['roles']),
                         (ev.TACTICAL_FORMAT, self.sha, {'candidate': 'cand', 'champion': 'champ'}))
        self.assertEqual((record['candidate'], record['champion']),
                         (self.body['models']['cand']['identity'], self.body['models']['champ']['identity']))
        paths = {self.body['models'][n][k] for n in ('cand', 'champ') for k in ('policy', 'value')}
        self.assertEqual({str(Path(c[1]['value'] if c[0] == 'e1' else c[1]).relative_to(self.root)) for c in calls},
                         paths)
        self.assertEqual(evidence.tactical(record, self.body, self.sha, 'test', confirmation=True)['verdict'], 'PASS')
        args = argparse.Namespace(contract=self.w['contract'], device='cpu', workers=1, allow_dirty=True,
                                  root=self.root)
        draft = {**self.body, 'status': 'draft'}
        self.w['contract'].write_text(json.dumps(draft))
        try:
            with mock.patch.object(cli, '_git', lambda _: CLEAN), self.assertRaisesRegex(ValueError, 'draft'):
                cli.tactical(args, log=QUIET, root=self.root)
        finally:
            self.w['contract'].write_text(json.dumps(self.body))

    def test_judge_promotes_and_records_ledger_entries(self):
        verdict = cli.judge(self.judge_args(), root=self.root)
        self.assertEqual((verdict['decision'], verdict['keep_kind'], verdict['code_commit']), ('PROMOTE', None, 'test'))
        self.assertEqual(verdict['judge_git'], CLEAN)
        code_entry = next(e for e in verdict['ledger']['entries'] if e['category'] == 'judge-code')
        self.assertEqual(json.loads((self.root / code_entry['entry']).read_text()), CLEAN)
        self.assertEqual(verdict['selection']['invalid_policy'], POLICY)
        for entry in verdict['ledger']['entries']:
            self.assertEqual(ev.file_sha256(self.root / entry['entry']), entry['sha256'])
        self.assertEqual(cli.judge(self.judge_args(), root=self.root), verdict)  # identical audit recomputation
        # changed inputs of a judged contract are refused; the ledger stays as it was
        before = ledger_files(self.root)
        regenerated = self.root / 'out/time2.json'
        regenerated.write_text(json.dumps(json.loads(self.timing.read_text()), indent=1))
        with self.assertRaisesRegex(ValueError, 'different confirmation inputs'):
            cli.judge(self.judge_args(timing=regenerated), root=self.root)
        self.assertEqual(ledger_files(self.root), before)

    def test_current_judge_code_is_checked_before_recording(self):
        before = ledger_files(self.root)
        args = self.judge_args(allow_dirty=False)
        cases = (({'commit': 'test', 'dirty': True}, 'clean committed code'),
                 ({'commit': 'test', 'dirty': None}, 'clean committed code'),
                 ({'commit': None, 'dirty': False}, 'clean committed code'),
                 ({'commit': 'different-current-code', 'dirty': False}, 'judge code commit differs'))
        with mock.patch('hybrid.h7c_nmask.committed', return_value=True):
            for state, message in cases:
                with self.subTest(state=state), mock.patch.object(cli, '_git', return_value=state) as reader:
                    with self.assertRaisesRegex(ValueError, message):
                        cli.judge(args, root=self.root)
                    reader.assert_called_once_with(False)
                    self.assertEqual(ledger_files(self.root), before)
            # Clean matching code is accepted without the test-only dirty-tree option.
            verdict = cli.judge(args, root=self.root)
            self.assertEqual((verdict['decision'], verdict['judge_git']), ('PROMOTE', CLEAN))

    def test_refusals_leave_the_ledger_unchanged_and_a_complete_run_then_succeeds(self):
        before = ledger_files(self.root)
        other_commit = self.save(self.make_tactical(git={'commit': 'other', 'dirty': False}), 'tact-other')
        bad_timing = json.loads(self.timing.read_text())
        bad_timing['sides']['cand']['samples'] = [9.0, 9.0]
        legacy = self.save({**tactical_record(self.body), 'contract_sha256': self.sha}, 'tact-legacy')
        changed_probe = json.loads(self.tactical.read_text())
        changed_probe['files']['fresh_probes'] = '0' * 64
        other_model = json.loads(self.tactical.read_text())
        other_model['candidate'] = self.body['models']['orig']['identity']
        other_contract = json.loads(self.tactical.read_text())
        other_contract['contract_sha256'] = '0' * 64
        cases = {
            'missing selection': (dict(selection=None), 'needs every evidence record'),
            'missing tactical': (dict(tactical=None), 'needs every evidence record'),
            'absent file': (dict(timing=self.root / 'out/none.json'), 'no such evidence file'),
            'tactical commit': (dict(tactical=other_commit), 'commit differs'),
            'timing samples': (dict(timing=self.save(bad_timing, 'time-bad')), 'timing'),
            'legacy tactical': (dict(tactical=legacy), 'never re-bound'),
            'probe file': (dict(tactical=self.save(changed_probe, 'tact-probe')), 'probes'),
            'model': (dict(tactical=self.save(other_model, 'tact-model')), 'candidate and champion'),
            'contract': (dict(tactical=self.save(other_contract, 'tact-contract')), 'another contract'),
        }
        for name, (change, message) in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, message):
                    cli.judge(self.judge_args(**change), root=self.root)
                self.assertEqual(ledger_files(self.root), before)
        self.assertEqual(cli.judge(self.judge_args(), root=self.root)['decision'], 'PROMOTE')
        self.assertGreater(len(ledger_files(self.root)), len(before))

    def test_arms_of_different_commits_are_refused(self):
        other = self.play('champion_vs_anchor', git={'commit': 'other', 'dirty': False}, name='cva-other')
        arms = [self.arms['candidate_vs_champion'], self.arms['candidate_vs_anchor'], other]
        before = ledger_files(self.root)
        with self.assertRaisesRegex(ValueError, 'different code commits'):
            cli.judge(self.judge_args(arm_record=arms), root=self.root)
        self.assertEqual(ledger_files(self.root), before)

    def test_allow_dirty_never_writes_the_repository_ledger(self):
        before = ledger_files(cli.ROOT)
        with self.assertRaisesRegex(ValueError, 'never writes'):
            cli.judge(self.judge_args(), root=cli.ROOT)
        with self.assertRaisesRegex(ValueError, 'never writes'):
            bind_all(cli.ROOT, [('contract', 'T-CONFIRM', {'sha256': self.sha})], allow_dirty=True)
        self.assertEqual(ledger_files(cli.ROOT), before)

    def test_selection_source_reserves_the_hconf_before_evaluating_and_resumes(self):
        manifest = json.loads(self.w['source'].read_text())
        rows = {json.loads(x)['game_index']: json.loads(x)
                for x in (self.w['source'].parent / 'sensitivity.jsonl').read_text().splitlines()}
        folder = self.root / 'runs/source'

        def args(contract):
            return argparse.Namespace(contract=contract, root=self.root, cache=self.w['source'].parent / 'cache',
                                      source_dir=folder, output=folder / 'source.json', program=folder / 'p.json',
                                      rapfi_dir=folder, workers=1, device='cuda', allow_dirty=True,
                                      threads_for_tests=True)

        def sensitivity(tasks):
            return [{**deepcopy(rows[t['game']['game_index']]), 'identity': t['identity']} for t in tasks]

        def branch(tasks):
            return [{'key': t['key'], 'identity_sha256': t['identity_sha256'], 'order': t['order'], 'status': 'VALID',
                     'score': 1., 'length': 9, 'winner': 'black', 'ply': 8, 'move': t['move'], 'continuation': []}
                    for t in tasks]

        def crash(tasks):
            raise RuntimeError('interrupted')

        patches = lambda chunk: (mock.patch.object(cli, '_git', lambda _: CLEAN),  # noqa: E731
                                 mock.patch('h7c_labels._under_data_or_runs', return_value=True),
                                 mock.patch('h7c_adjudicate.engine_spec', return_value={'identity': manifest['engine']}),
                                 mock.patch.object(cli, '_sensitivity_chunk', side_effect=chunk),
                                 mock.patch.object(cli, '_branch_chunk', side_effect=branch))
        hold = ('holdout-use', self.body['selection']['holdout_contract_sha256'], {'confirmation': self.sha})
        self.assertEqual(conflicts(self.root, [(*hold[:2], {'confirmation': 'another'})]), [])
        first = patches(crash)
        with first[0], first[1], first[2], first[3], first[4]:
            with self.assertRaises(RuntimeError):
                cli.selection_source(args(self.w['contract']), root=self.root)
        # the reservation came before the evaluation and survives the interruption
        self.assertEqual(conflicts(self.root, [(*hold[:2], {'confirmation': 'another'})]), [f'{hold[0]}:{hold[1]}'])
        second = patches(sensitivity)
        with second[0], second[1], second[2], second[3], second[4]:
            out = cli.selection_source(args(self.w['contract']), root=self.root)  # the same inputs resume
            self.assertEqual(out['contract_sha256'], self.sha)
            # another confirmation of this H-conf is refused before anything of it is recorded
            body2 = {**deepcopy(self.body), 'contract_id': 'T-CONFIRM-2'}
            for arm in body2['arms']:
                arm['seed'] = 8791
            path2 = self.w['contract'].with_name('confirm2.json')
            path2.write_text(json.dumps(body2))
            before = ledger_files(self.root)
            with self.assertRaisesRegex(ValueError, 'holdout-use'):
                cli.selection_source(args(path2), root=self.root)
            self.assertEqual(ledger_files(self.root), before)


class LedgerTest(unittest.TestCase):
    def test_bind_all_checks_every_conflict_before_writing(self):
        with TemporaryDirectory() as tmp:
            bind(tmp, 'a', '1', {'x': 1})
            before = ledger_files(tmp)
            with self.assertRaisesRegex(ValueError, 'a:1'):
                bind_all(tmp, [('b', '2', {'y': 1}), ('a', '1', {'x': 2}), ('c', '3', {'z': 1})])
            self.assertEqual(ledger_files(tmp), before)
            with self.assertRaisesRegex(ValueError, 'twice'):
                bind_all(tmp, [('b', '2', {'y': 1}), ('b', '2', {'y': 2})])
            entries = bind_all(tmp, [('a', '1', {'x': 1}), ('b', '2', {'y': 1})])
            self.assertEqual([e['category'] for e in entries], ['a', 'b'])
            self.assertEqual(conflicts(tmp, [('b', '2', {'y': 1})]), [])
            self.assertEqual(conflicts(tmp, [('b', '2', {'y': 9})]), ['b:2'])


class ForbiddenAttributionTest(unittest.TestCase):
    def illegal(self, arms, name, side):
        """The first game of ``name`` ends on an illegal attempt by ``side`` (the tested side is black there)."""
        ops = openings(40)
        g = games(ops, eval_tests.JudgeTest.GOOD['cand_results' if name == 'candidate_vs_champion' else 'cand_anchor'],
                  replayable=True)
        cut = 4 if side == 'tested' else 5
        g[0] = {**g[0], 'result': 'loss' if side == 'tested' else 'opponent_illegal', 'winner': None,
                'moves': g[0]['moves'][:cut], 'length': cut, 'illegal': {'side': side, 'ply': cut, 'move': [7, 7]}}
        arms[name]['table'] = ev.pair_table(g, ops)
        arms[name]['summary'] = ev.arm_summary(arms[name]['table'], STATS)
        return arms

    def test_anchor_arms_without_valid_pairs_keep_the_champion(self):
        t = eval_tests.JudgeTest()
        for empty_arms in (('candidate_vs_anchor',), ('champion_vs_anchor',),
                           ('candidate_vs_anchor', 'champion_vs_anchor')):
            with self.subTest(empty_arms=empty_arms):
                arms, ops = t.arms(**t.GOOD), openings(40)
                for name in empty_arms:
                    records = games(ops, t.GOOD['cand_anchor'], replayable=True)
                    for g in records:
                        cut = 5 if g['tested_color'] == 'black' else 4
                        g.update(result='opponent_illegal', winner=None, moves=g['moves'][:cut], length=cut,
                                 illegal={'side': 'opponent', 'ply': cut, 'move': [7, 7]})
                    table = ev.pair_table(records, ops)  # genuine replay, every pair invalidated
                    self.assertEqual((table['pairs'], len(table['opponent_illegal_pairs'])), ([], 40))
                    arms[name] = {'table': table, 'summary': ev.arm_summary(table, STATS)}
                out = t.judge(arms=arms)
                self.assertEqual((out['decision'], out['keep_kind']), ('KEEP_CHAMPION', 'NOT_JUDGEABLE'))
                self.assertIn('ARM_INVALID', out['reasons'])
                self.assertEqual(out['anchor'], {'status': 'NOT_COMPUTED', 'reason': 'ARM_INVALID'})

    def test_candidate_and_reference_faults_are_told_apart(self):
        t = eval_tests.JudgeTest()
        cand = t.judge(arms=self.illegal(t.arms(**t.GOOD), 'candidate_vs_anchor', 'tested'))
        self.assertIn('GATE_FORBIDDEN', cand['reasons'])
        self.assertEqual((cand['keep_kind'], cand['candidate_illegal_games']['candidate_vs_anchor']),
                         ('CANDIDATE_FAILED', 1))
        self.assertIn('GATE_FORBIDDEN', cand['reason_classes']['candidate_failed'])
        ref = t.judge(arms=self.illegal(t.arms(**t.GOOD), 'champion_vs_anchor', 'tested'))
        self.assertEqual(ref['decision'], 'KEEP_CHAMPION')
        self.assertNotIn('GATE_FORBIDDEN', ref['reasons'])
        self.assertTrue(ref['gates']['forbidden'])
        self.assertIn('ARM_INVALID', ref['reasons'])
        self.assertEqual(ref['arm_invalid'], [{'arm': 'champion_vs_anchor', 'cause': 'CHAMPION_ILLEGAL_IN_REFERENCE_ARM',
                                               'games': 1}])
        self.assertEqual(ref['keep_kind'], 'NOT_JUDGEABLE')
        opponent = t.judge(arms=self.illegal(t.arms(**t.GOOD), 'champion_vs_anchor', 'opponent'))
        self.assertEqual([c['cause'] for c in opponent['arm_invalid']], ['OPPONENT_ILLEGAL'])
        self.assertEqual((opponent['anchor']['status'], opponent['keep_kind']), ('NOT_COMPUTED', 'NOT_JUDGEABLE'))


if __name__ == '__main__':
    unittest.main()
