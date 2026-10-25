"""H7-c automation Auto-v1 (hybrid/h7c_auto.py, scripts/h7c_auto.py): no evidence combination moves on wrongly."""
import json
from pathlib import Path
import unittest

from hybrid import h7c_auto as auto
from scripts import h7c_auto as cli

ALL_TOOLS = {name: True for name in auto.TOOLING}


def e(cycle, decision='KEEP_CHAMPION', lower=-0.1, reason=None, mode='JOINT'):
    if reason is None:
        reason = 'PROMOTED' if decision == 'PROMOTE' else 'NOT_PROMOTED'
    if reason in ('NO_CANDIDATE', 'OFFICIAL_INVALID'):
        lower = None
    return {'cycle': cycle, 'decision': decision, 'reason': reason, 'lower': lower, 'candidate_mode': mode}


C1 = e(1, lower=-0.2)   # the recorded c1 verdict


def state(cycles, tooling=ALL_TOOLS):
    return auto.next_cycle_decision(cycles, tooling)['state']


class NextCycleTest(unittest.TestCase):
    def test_c2_after_c1(self):
        self.assertEqual(state([C1], {}), 'NEXT_CYCLE')

    def test_e2_right_after_c2(self):
        # the pre-registered E2 has no minimum: c1 and c2 stalled, no promotion in the evaluated window
        self.assertEqual(state([C1, e(2)]), 'E2_REQUIRED')
        self.assertEqual(state([C1, e(2)], {}), 'E2_SB_TOOLING_REQUIRED')
        self.assertEqual(state([C1, e(2, lower=0.0)]), 'E2_REQUIRED')     # lower <= 0 includes 0

    def test_hold(self):
        cycles = [C1, e(2, 'PROMOTE', 0.02), e(3), e(4)]
        d = auto.next_cycle_decision(cycles, ALL_TOOLS)
        self.assertEqual((d['state'], d['why']['stagnant'], d['why']['hold']), ('V9_HOLD', True, True))
        self.assertEqual(state(cycles + [e(5)]), 'E2_REQUIRED')          # c3..c5: no promotion

    def test_tactical_keep_with_a_positive_bound_is_no_stall(self):
        self.assertEqual(state([C1, e(2, lower=0.01)]), 'NEXT_CYCLE')

    def test_no_candidate(self):
        self.assertEqual(state([C1, e(2, reason='NO_CANDIDATE')]), 'NEXT_CYCLE')   # breaks the run, not a stall
        self.assertEqual(state([C1, e(2, reason='NO_CANDIDATE'), e(3)]), 'NEXT_CYCLE')
        d = auto.next_cycle_decision([C1, e(2, reason='NO_CANDIDATE'), e(3, reason='NO_CANDIDATE')], ALL_TOOLS)
        self.assertEqual((d['state'], d['why']['rule']), ('MANUAL_REVIEW_REQUIRED', 'R2'))
        # a no-candidate cycle is not evaluated: the hold window skips it
        cycles = [C1, e(2, 'PROMOTE', 0.02), e(3), e(4, reason='NO_CANDIDATE'), e(5), e(6)]
        self.assertEqual(state(cycles), 'E2_REQUIRED')                     # window c3, c5, c6

    def test_official_invalid_first(self):
        d = auto.next_cycle_decision([C1, e(2), e(3, reason='OFFICIAL_INVALID')], ALL_TOOLS)
        self.assertEqual((d['state'], d['why']['rule']), ('MANUAL_REVIEW_REQUIRED', 'R6'))

    def test_fallback_stall_needs_its_diagnosis(self):
        cycles = [C1, e(2, mode='VALUE_ONLY_FALLBACK')]
        self.assertEqual(state(cycles, {'e2_sb_champion': True}), 'E2_FALLBACK_DIAG_TOOLING_REQUIRED')
        self.assertEqual(state(cycles), 'E2_REQUIRED')

    def test_tooling_and_seed_range(self):
        self.assertEqual(state([C1, e(2, 'PROMOTE', 0.02)], {}), 'C3_RECIPE_TOOLING_REQUIRED')
        cycles = [C1] + [e(c, 'PROMOTE', 0.02) for c in range(2, 11)]
        d = auto.next_cycle_decision(cycles, ALL_TOOLS)
        self.assertEqual((d['state'], d['why']['cycle']), ('BLOCKED_SEED_RANGE', 11))

    def test_history_must_be_complete(self):
        with self.assertRaises(ValueError):
            auto.next_cycle_decision([C1, e(3)], ALL_TOOLS)


class CycleStageTest(unittest.TestCase):
    FULL = {'cycle': 2, 'ready': True, 'recipe': True, 'selfplay': True, 'preflight': 'WARN', 'retarget': True,
            'labels': True, 'policy_verdict': 'CANDIDATE', 'value_verdict': 'CANDIDATE', 'verdict': True, 'x2': True,
            'cycle_index': True}

    def stage(self, **changes):
        return auto.cycle_stage({**self.FULL, **changes})['state']

    def test_order(self):
        self.assertEqual(self.stage(), 'CYCLE_COMPLETE')
        self.assertEqual(self.stage(ready='missing: g0'), 'NOT_C2_READY')
        self.assertEqual(self.stage(recipe=False), 'RECIPE_REQUIRED')
        self.assertEqual(self.stage(selfplay=False), 'SELFPLAY_READY')
        self.assertEqual(auto.cycle_stage({**self.FULL, 'selfplay': False})['why']['seed'], 8802)
        self.assertEqual(self.stage(retarget=False), 'RETARGET_READY')
        self.assertEqual(self.stage(labels=False), 'LABELS_READY')
        self.assertEqual(self.stage(policy_verdict=None), 'POLICY_TRAIN_READY')
        self.assertEqual(self.stage(value_verdict=None), 'VALUE_TRAIN_READY')
        self.assertEqual(self.stage(verdict=False), 'ARENA_BUNDLE_READY')
        self.assertEqual(self.stage(x2=False), 'X2_MONITOR_READY')
        self.assertEqual(self.stage(cycle_index=False), 'CYCLE_INDEX_READY')

    def test_retarget_stops(self):
        self.assertEqual(self.stage(retarget=False, retarget_partial=True), 'MANUAL_COST_RESUME_REVIEW_REQUIRED')
        self.assertEqual(self.stage(retarget=False, preflight='MANUAL_COST_PREFLIGHT_REVIEW_REQUIRED'),
                         'MANUAL_COST_PREFLIGHT_REVIEW_REQUIRED')

    def test_training_results(self):
        self.assertEqual(self.stage(train_error='policy: CANDIDATE without its metadata file'), 'BLOCKED_TRAIN')
        self.assertEqual(self.stage(policy_verdict='NO_CANDIDATE', verdict=False), 'ARENA_BUNDLE_READY')  # fallback
        self.assertEqual(self.stage(value_verdict='NO_CANDIDATE', verdict=False), 'NO_CANDIDATE_VERDICT_READY')
        self.assertEqual(self.stage(value_verdict='NO_CANDIDATE', x2=False), 'CYCLE_COMPLETE')   # X2 not applicable

    def test_later_cycles_need_the_recipe_tool(self):
        self.assertEqual(auto.cycle_stage({'cycle': 3})['state'], 'C3_RECIPE_TOOLING_REQUIRED')
        self.assertEqual(auto.cycle_stage({'cycle': 11, 'tooling': ALL_TOOLS})['state'], 'BLOCKED_SEED_RANGE')


class TrainArtifactTest(unittest.TestCase):
    def test_records(self):
        rec = {'cycle': 2, 'inputs': {'parent_checkpoint_sha256': 'p'}, 'verdict': 'CANDIDATE',
               'candidate': {'metadata_sha256': 'm'}}
        self.assertIsNone(auto.train_artifacts_ok(rec, 'm', 2, 'p'))
        self.assertIn('without', auto.train_artifacts_ok(rec, None, 2, 'p'))
        self.assertIn('differs', auto.train_artifacts_ok(rec, 'x', 2, 'p'))
        self.assertIn('parent', auto.train_artifacts_ok(rec, 'm', 2, 'q'))
        self.assertIn('cycle', auto.train_artifacts_ok(rec, 'm', 3, 'p'))
        none = {**rec, 'verdict': 'NO_CANDIDATE', 'candidate': None}
        self.assertIsNone(auto.train_artifacts_ok(none, None, 2, 'p'))
        self.assertIsNotNone(auto.train_artifacts_ok(none, 'm', 2, 'p'))

    def test_resolve(self):
        self.assertEqual(auto.resolve_candidate('CANDIDATE', 'CANDIDATE', 2), 'JOINT')
        self.assertEqual(auto.resolve_candidate('NO_CANDIDATE', 'CANDIDATE', 2), 'VALUE_ONLY_FALLBACK')
        self.assertEqual(auto.resolve_candidate('NO_CANDIDATE', 'CANDIDATE', 1), 'NO_CANDIDATE')
        self.assertEqual(auto.resolve_candidate('CANDIDATE', 'NO_CANDIDATE', 2), 'NO_CANDIDATE')


class PreflightTest(unittest.TestCase):
    def test_boundaries(self):
        def status(rows):
            return auto.cost_preflight(rows, 2.8334, 6, 10728.5)['status']
        self.assertEqual(status(21582), 'PASS')
        self.assertEqual(status(21583), 'WARN')
        self.assertEqual(status(22718), 'WARN')
        self.assertEqual(status(22719), 'MANUAL_COST_PREFLIGHT_REVIEW_REQUIRED')
        self.assertEqual(status(21956), 'WARN')                                 # c1's visits rows
        self.assertAlmostEqual(auto.cost_preflight(21956, 2.8334, 6, 10728.5)['projected_seconds'], 10368.4)


class E2Test(unittest.TestCase):
    def test_sb_branch(self):
        cases = {(2, 8): 'E2_BUDGET_NONMONOTONIC_REVIEW', (2,): 'E2_BUDGET_NONMONOTONIC_REVIEW',
                 (4,): 'E2_CURRENT_TEACHER_STILL_STRONG_REVIEW', (4, 8): 'NEW_TEACHER_REVIEW',
                 (8,): 'NEW_TEACHER_REVIEW', (2, 4, 8): 'NEW_TEACHER_REVIEW', (): 'E2_SATURATION_REQUIRED'}
        for stronger, expected in cases.items():
            self.assertEqual(auto.sb_branch(set(stronger))['state'], expected, stronger)
        self.assertEqual(auto.sb_branch({4, 8})['why']['factor'], 8)

    def test_episode(self):
        none = {'B_POLICY': False, 'B_VALUE': False}
        self.assertEqual(auto.e2_state(None, None, None)['state'], 'E2_DIAGNOSIS_REQUIRED')
        self.assertEqual(auto.e2_state(None, None, none)['state'], 'E2_SB_REQUIRED')
        self.assertEqual(auto.e2_state(set(), None, none)['state'], 'E2_SATURATION_REQUIRED')
        self.assertEqual(auto.e2_state(set(), 'STRONGER', none)['why']['factor'], 8)
        self.assertEqual(auto.e2_state(set(), 'INCONCLUSIVE', none)['state'], 'REPLICATE_REQUIRED')
        self.assertEqual(auto.e2_state(set(), 'SATURATED', none)['state'], 'V9_RESEARCH_READY')
        with self.assertRaises(ValueError):
            auto.e2_state(set(), 'MAYBE', none)

    def test_prescriptions(self):
        both = {'B_POLICY': True, 'B_VALUE': True}

        def run(**p):
            return auto.e2_state(set(), 'SATURATED', both, p)['state']
        self.assertEqual(run(), 'MANUAL_RECIPE_CHANGE_REQUIRED')
        self.assertEqual(auto.e2_state(set(), 'SATURATED', both)['why']['trigger'], 'B_POLICY')
        self.assertEqual(run(B_POLICY=e(5, 'PROMOTE', 0.02)), 'E2_EPISODE_CLOSED')
        self.assertEqual(run(B_POLICY=e(5, reason='NO_CANDIDATE')), 'MANUAL_RECIPE_CHANGE_REQUIRED')
        self.assertEqual(run(B_POLICY=e(5, reason='OFFICIAL_INVALID')), 'MANUAL_RECIPE_CHANGE_REQUIRED')
        self.assertEqual(auto.e2_state(set(), 'SATURATED', both, {'B_POLICY': e(5)})['why']['trigger'], 'B_VALUE')
        self.assertEqual(run(B_POLICY=e(5), B_VALUE=e(6)), 'V9_RESEARCH_READY')
        self.assertEqual(run(B_POLICY=e(5), B_VALUE=e(6, lower=0.01)), 'MANUAL_RECIPE_CHANGE_REQUIRED')

    def test_seeds(self):
        self.assertEqual([auto.e2_seed(n) for n in (1, 8, 9)], [8742, 8749, None])
        self.assertEqual(auto.selfplay_seed(2), 8802)


class EvidenceOnlyTest(unittest.TestCase):
    def test_paths(self):
        ok = ['docs/h7c/c2/selfplay_c2_seed8802.json', 'docs/external/x2_c2.json']
        self.assertEqual(auto.evidence_only(ok, 2), [])
        bad = ['docs/h7c-auto.md', 'src/hybrid/h7c_auto.py', 'docs/h7c/c3/recipe.json', 'docs/external/x2_c1.json',
               'docs/h7c/e2/e1/plan.json']
        self.assertEqual(auto.evidence_only(ok + bad, 2), sorted(bad))
        self.assertEqual(auto.evidence_only(['docs/h7c/e2/e1/plan.json'], 2, episode=1), [])

    def test_fingerprint(self):
        a = {'cycle': 2, 'teacher_factor': 4, 'code_base_commit': 'x', 'source_decision_sha256': 'd'}
        b = {'cycle': 3, 'teacher_factor': 4, 'code_base_commit': 'y', 'parent_recipe_sha256': 'r'}
        self.assertEqual(auto.recipe_fingerprint(a), auto.recipe_fingerprint(b))
        self.assertNotEqual(auto.recipe_fingerprint(a), auto.recipe_fingerprint({**b, 'teacher_factor': 8}))


class RepositoryTest(unittest.TestCase):
    """The committed c1 evidence as the CLI reads it."""

    def test_c1_chain_and_entry(self):
        found = cli.verdicts()
        self.assertGreaterEqual(len(found), 1)
        self.assertEqual(cli.chain_errors(found), [])
        entry = auto.cycle_entry(found[0]['record'])
        self.assertEqual((entry['decision'], entry['lower'], entry['candidate_mode']), ('KEEP_CHAMPION', -0.2, 'JOINT'))

    def test_recorded_numbers(self):
        review = json.loads((cli.C1['cost_review']).read_text(encoding='utf-8'))
        decision = json.loads((cli.C1['decision']).read_text(encoding='utf-8'))
        self.assertEqual(review['hard_cap_seconds'], 10728.5)
        self.assertEqual(decision['budget']['tree_seconds_mean'], 2.8334)

    def test_commands_exist_only_for_existing_tools(self):
        history = [auto.cycle_entry(cli.verdicts()[0]['record'])]
        lines = cli.command({'cycle': 2, 'history': history, 'step': auto.cycle_stage(
            {**CycleStageTest.FULL, 'selfplay': False})})
        self.assertIn('--run-seed 8802', lines[0])
        self.assertTrue(Path(cli.ROOT / 'scripts' / 'h7c_selfplay.py').exists())
        self.assertEqual(cli.command({'cycle': 3, 'history': history, 'step': auto.cycle_stage({'cycle': 3})}), [])


if __name__ == '__main__':
    unittest.main()
