"""H7-c automation Auto-v1 (hybrid/h7c_auto.py, scripts/h7c_auto.py): no evidence combination moves on wrongly."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

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
            'labels': True, 'manifest': True, 'policy_verdict': 'CANDIDATE', 'value_verdict': 'CANDIDATE',
            'verdict': True, 'x2': True,
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
        self.assertEqual(self.stage(manifest=False), 'EVIDENCE_MANIFEST_READY')
        self.assertEqual(self.stage(policy_verdict=None), 'POLICY_TRAIN_READY')
        self.assertEqual(self.stage(value_verdict=None), 'VALUE_TRAIN_READY')
        self.assertEqual(self.stage(verdict=False), 'ARENA_BUNDLE_READY')
        self.assertEqual(self.stage(x2=False, x2_config=True), 'X2_MONITOR_READY')
        self.assertEqual(self.stage(cycle_index=False), 'CYCLE_INDEX_READY')

    def test_retarget_stops(self):
        self.assertEqual(self.stage(retarget=False, retarget_partial=True), 'MANUAL_COST_RESUME_REVIEW_REQUIRED')
        self.assertEqual(self.stage(retarget=False, preflight='MANUAL_COST_PREFLIGHT_REVIEW_REQUIRED'),
                         'MANUAL_COST_PREFLIGHT_REVIEW_REQUIRED')

    def test_failing_evidence_blocks(self):
        for stage, state in (('selfplay', 'BLOCKED_SELFPLAY'), ('retarget', 'BLOCKED_RETARGET'),
                             ('labels', 'BLOCKED_LABELS'), ('arena', 'BLOCKED_ARENA'), ('x2', 'BLOCKED_X2'),
                             ('cycle_index', 'BLOCKED_CYCLE_INDEX'), ('recipe', 'BLOCKED_RECIPE'),
                             ('manifest', 'BLOCKED_MANIFEST')):
            self.assertEqual(self.stage(errors={stage: ['x']}), state)
        self.assertEqual(self.stage(errors={'selfplay': []}), 'CYCLE_COMPLETE')
        self.assertEqual(self.stage(champion_error='not on this machine', verdict=False), 'BLOCKED_CHAMPION_FILES')
        # after the Arena, before X2: a candidate checkpoint gone or changed blocks the X2 step
        self.assertEqual(self.stage(x2=False, x2_config=True, errors={'train': ['value: checkpoint']}), 'BLOCKED_TRAIN')
        self.assertEqual(self.stage(x2=False, x2_config=True, champion_error='parent policy gone'),
                         'BLOCKED_CHAMPION_FILES')

    def test_checkpoints_needed(self):
        self.assertTrue(auto.checkpoints_needed(False, False, None))          # training / Arena
        self.assertTrue(auto.checkpoints_needed(True, False, 'JOINT'))        # X2 still to run
        self.assertTrue(auto.checkpoints_needed(True, False, 'VALUE_ONLY_FALLBACK'))
        self.assertFalse(auto.checkpoints_needed(True, True, 'JOINT'))
        self.assertFalse(auto.checkpoints_needed(True, False, 'NO_CANDIDATE'))

    def test_an_index_alone_is_no_cycle(self):
        self.assertEqual(auto.cycle_stage({'cycle': 2, 'ready': True, 'cycle_index': True})['state'],
                         'RECIPE_REQUIRED')
        self.assertEqual(self.stage(labels=False), 'LABELS_READY')

    def test_x2_config(self):
        self.assertEqual(self.stage(x2=False, x2_config=False), 'X2_CONFIG_REQUIRED')
        self.assertEqual(self.stage(x2=False, x2_config=True, errors={'x2_config': ['engine']}), 'BLOCKED_X2_CONFIG')
        good = {'engine': '/e/rapfi', 'engine_name': 'rapfi', 'engine_info': ['max_node=300'],
                'engine_files': ['/e/config.toml'], 'workers': 4}
        self.assertEqual(auto.x2_config_errors(good), [])
        for bad in ({}, [], {**good, 'engine': ''}, {**good, 'engine_info': 'max_node=300'},
                    {**good, 'engine_files': [1]}, {**good, 'workers': 0}, {**good, 'workers': '4'},
                    {**good, 'workers': True}, {**good, 'engine_name': ''}):
            self.assertTrue(auto.x2_config_errors(bad), bad)

    def test_training_results(self):
        self.assertEqual(self.stage(policy_verdict='NO_CANDIDATE', verdict=False), 'ARENA_BUNDLE_READY')  # fallback
        self.assertEqual(self.stage(value_verdict='NO_CANDIDATE', verdict=False), 'NO_CANDIDATE_VERDICT_READY')
        self.assertEqual(self.stage(value_verdict='NO_CANDIDATE', x2=False), 'CYCLE_COMPLETE')   # X2 not applicable

    def test_later_cycles_need_the_recipe_tool(self):
        self.assertEqual(auto.cycle_stage({'cycle': 3})['state'], 'C3_RECIPE_TOOLING_REQUIRED')
        self.assertEqual(auto.cycle_stage({'cycle': 11, 'tooling': ALL_TOOLS})['state'], 'BLOCKED_SEED_RANGE')


class TrainArtifactTest(unittest.TestCase):
    PARENT = {'policy_checkpoint_sha256': 'pc', 'policy_metadata_sha256': 'pm', 'value_checkpoint_sha256': 'vc',
              'value_metadata_sha256': 'vm'}

    def rec(self, **changes):
        return {'format': 'h7c-policy-train-v1', 'cycle': 2, 'git': {'dirty': False},
                'inputs': {'parent_checkpoint_sha256': 'pc', 'parent_metadata_sha256': 'pm',
                           'source': {'retarget': {'sha256': 'r'}}},
                'verdict': 'CANDIDATE', 'candidate': {'checkpoint_sha256': 'c', 'metadata_sha256': 'm'}, **changes}

    def errors(self, rec, meta='m', ckpt='c', local=True, source=None):
        return auto.train_errors('policy', rec, 2, self.PARENT, meta, ckpt, local,
                                 source or {'source.retarget': {'sha256': 'r'}})

    def test_records(self):
        self.assertEqual(self.errors(self.rec()), [])
        self.assertTrue(self.errors(self.rec(), meta=None))                    # no metadata
        self.assertTrue(self.errors(self.rec(), meta='x'))                     # metadata SHA
        self.assertTrue(self.errors(self.rec(), ckpt=None))                    # best.pt not on this machine
        self.assertTrue(self.errors(self.rec(), ckpt='x'))                     # wrong best.pt
        self.assertEqual(self.errors(self.rec(), ckpt=None, local=False), [])  # after the Arena: docs only
        self.assertTrue(self.errors(self.rec(inputs={'parent_checkpoint_sha256': 'pc', 'parent_metadata_sha256': 'x',
                                                     'source': {'retarget': {'sha256': 'r'}}})))
        self.assertTrue(self.errors(self.rec(), source={'source.retarget': {'sha256': 'other'}}))
        self.assertTrue(self.errors(self.rec(cycle=3)))
        self.assertTrue(self.errors(self.rec(git={'dirty': True})))
        none = self.rec(verdict='NO_CANDIDATE', candidate=None)
        self.assertEqual(self.errors(none, meta=None, ckpt=None), [])
        self.assertTrue(self.errors(none, meta='m'))

    def test_blocked_train_stage(self):
        full = {**CycleStageTest.FULL, 'errors': {'train': ['policy: wrong checkpoint']}}
        self.assertEqual(auto.cycle_stage(full)['state'], 'BLOCKED_TRAIN')

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

    def test_commit_gate(self):
        recipe = ['docs/h7c/c2/recipe.json']
        self.assertEqual(auto.commit_gate(recipe, [], 2)['state'], 'EVIDENCE_COMMIT_REQUIRED')     # uncommitted recipe
        self.assertIsNone(auto.commit_gate([], recipe, 2))
        self.assertEqual(auto.commit_gate(['src/hybrid/h7c_auto.py'], [], 2)['state'],
                         'MANUAL_CODE_CHANGE_REVIEW_REQUIRED')
        self.assertEqual(auto.commit_gate([], ['docs/h7c-auto.md'], 2)['state'], 'MANUAL_CODE_CHANGE_REVIEW_REQUIRED')
        g0 = ('docs/h7c/c1/retarget_g0.json', 'docs/h7c/c1/retarget_g0b.json')
        self.assertEqual(auto.commit_gate(list(g0), [], 2, g0)['state'], 'EVIDENCE_COMMIT_REQUIRED')
        self.assertEqual(auto.commit_gate(list(g0), [], 2)['state'], 'MANUAL_CODE_CHANGE_REVIEW_REQUIRED')
        # the previous cycle's index waiting for its commit; committed changes to it after the base are not allowed
        self.assertEqual(auto.commit_gate(['docs/h7c/c2/cycle.json'], [], 3)['state'], 'EVIDENCE_COMMIT_REQUIRED')
        self.assertEqual(auto.commit_gate([], ['docs/h7c/c2/cycle.json'], 3)['state'],
                         'MANUAL_CODE_CHANGE_REVIEW_REQUIRED')

    def test_fingerprint(self):
        a = {'format': 'h7c-recipe-v1', 'cycle': 2, 'recipe': 'S200_RETARGET', 'teacher_factor': 4,
             'budget': {'simulations': 200, 'tactical_simulations': 400}, 'mode': 'restored',
             'hard_cap_seconds': 10728.5, 'tree_seconds_mean': 2.8334, 'workers': 6, 'actor': {'engine': 'TB-V3'},
             'arena_seed': 8732, 'selfplay_seed': 8802, 'parent_verdict_sha256': 'v1', 'code_base_commit': 'x',
             'source_decision_sha256': 'd', 'policy_config_sha256': 'p', 'value_config_sha256': 'v',
             'engine_hash': 'e', 'runtime_hash': 'r', 'value_device': 'cuda'}
        b = {**a, 'cycle': 3, 'arena_seed': 8733, 'selfplay_seed': 8803, 'parent_verdict_sha256': 'v2',
             'code_base_commit': 'y', 'parent_recipe_sha256': 'r2', 'source_verdict_sha256': 's'}
        self.assertEqual(auto.recipe_fingerprint(a), auto.recipe_fingerprint(b))
        self.assertNotEqual(auto.recipe_fingerprint(a), auto.recipe_fingerprint({**b, 'teacher_factor': 8}))
        self.assertNotEqual(auto.recipe_fingerprint(a), auto.recipe_fingerprint({**b, 'intervention': 'proof_off'}))


class ManifestTest(unittest.TestCase):
    RETARGET = {'jsonl_sha256': 'r' * 64, 'identity': {'selfplay_jsonl_sha256': 's' * 64}}

    def manifest(self, **changes):
        return {'format': 'h7c-evidence-v2', 'cycle': 2, 'selfplay_jsonl_sha256': 's' * 64,
                'retarget_jsonl_sha256': 'r' * 64, 'labels_report_sha256': 'l' * 64, 'policy_rows_sha256': 'p' * 64,
                'value_rows_sha256': 'v' * 64, **changes}

    def test_mutations(self):
        self.assertEqual(auto.manifest_errors(self.manifest(), 2, self.RETARGET, 'l' * 64), [])
        for key in ('selfplay_jsonl_sha256', 'retarget_jsonl_sha256', 'labels_report_sha256'):
            self.assertTrue(auto.manifest_errors(self.manifest(**{key: 'x' * 64}), 2, self.RETARGET, 'l' * 64), key)
        self.assertTrue(auto.manifest_errors(self.manifest(policy_rows_sha256=None), 2, self.RETARGET, 'l' * 64))
        self.assertTrue(auto.manifest_errors(self.manifest(), 3, self.RETARGET, 'l' * 64))
        # the policy record must name the manifest's row file
        rec = TrainArtifactTest().rec(inputs={'parent_checkpoint_sha256': 'pc', 'parent_metadata_sha256': 'pm',
                                              'policy_rows_sha256': 'x' * 64})
        self.assertTrue(auto.train_errors('policy', rec, 2, TrainArtifactTest.PARENT, 'm', 'c', True,
                                          {'policy_rows_sha256': 'p' * 64}))


class RawFileTest(unittest.TestCase):
    MANIFEST = ManifestTest().manifest()

    def test_pure(self):
        now = {k: self.MANIFEST[k] for k in auto.MANIFEST_KEYS}
        self.assertEqual(auto.raw_file_errors(self.MANIFEST, now), [])
        for key in ('value_rows_sha256', 'policy_rows_sha256'):
            self.assertTrue(auto.raw_file_errors(self.MANIFEST, {**now, key: 'x' * 64}), key)
            self.assertTrue(auto.raw_file_errors(self.MANIFEST, {**now, key: None}), key)
        self.assertEqual(auto.raw_file_errors(self.MANIFEST, {**now, 'policy_rows_sha256': None},
                                              ['value_rows_sha256']), [])

    def test_changed_rows_after_the_manifest(self):
        """manifest written -> a row file changes under runs/ -> the stage blocks before the trainer reads it."""
        with TemporaryDirectory() as tmp:
            files = {k: Path(tmp) / k for k in auto.MANIFEST_KEYS}
            for path in files.values():
                path.write_bytes(path.name.encode())
            with mock.patch.object(cli, 'raw_files', return_value=files):
                manifest = {**cli.raw_manifest(2), 'cycle': 2}
                now = lambda: {k: cli._sha(p) for k, p in cli.raw_files(2).items()}  # noqa: E731
                self.assertEqual(auto.raw_file_errors(manifest, now()), [])
                for key in ('value_rows_sha256', 'policy_rows_sha256'):
                    files[key].write_bytes(b'changed')
                    errors = auto.raw_file_errors(manifest, now())
                    self.assertTrue(any(e.startswith(key) for e in errors), key)
                    stage = auto.cycle_stage({**CycleStageTest.FULL, 'policy_verdict': None, 'value_verdict': None,
                                              'errors': {'manifest': errors}})
                    self.assertEqual(stage['state'], 'BLOCKED_MANIFEST')


class G0Test(unittest.TestCase):
    SELFPLAY = {'cycle': 1, 'run_seed': 8801, 'engine_hash': 'e', 'runtime_hash': 'r', 'value_device': 'cuda',
                'models': {'m': 1}, 'model_hash': 'h', 'generator_git': {'commit': 'g', 'dirty': False}}

    def rec(self, factor, games, modes, **changes):
        ident = {'format': 'h7c-retarget-v1', 'purpose': 'g0', 'factor': factor, 'modes': modes,
                 'selfplay_jsonl_sha256': 'j', 'selfplay': self.SELFPLAY,
                 'coverage': {'eligible': 10 * games, 'games': games}, 'git': {'dirty': False}}
        out = {'format': 'h7c-retarget-v1-check', 'identity': ident, 'rows': 10 * games, 'games': games,
               'actor': {m: {} for m in modes} if factor == 1 else None,
               'modes_differ': {} if len(modes) > 1 else None,
               'projection': {'compared': 9000, 'differ': 0, 'missing': 0} if factor == 1 else None,
               'verdict': 'PASS', 'failures': []}
        return {**out, **changes}

    def errors(self, g0=None, g0b=None):
        return auto.g0_errors(g0 or self.rec(1, 1000, ['restored']), g0b or self.rec(4, 20, ['restored', 'sequential']),
                              self.SELFPLAY, 'j')

    def test_contract(self):
        self.assertEqual(self.errors(), [])
        self.assertTrue(auto.g0_errors(None, None, self.SELFPLAY, 'j'))
        subset = self.rec(1, 50, ['restored'])
        self.assertTrue(self.errors(g0=subset))                                       # --games-limit
        self.assertTrue(self.errors(g0=self.rec(1, 1000, ['restored'], projection=None)))   # no --policy-rows
        self.assertTrue(self.errors(g0=self.rec(1, 1000, ['restored'],
                                               projection={'compared': 0, 'differ': 0, 'missing': 0})))
        self.assertTrue(self.errors(g0b=self.rec(4, 20, ['restored'])))                # one mode only
        self.assertTrue(self.errors(g0=self.rec(1, 1000, ['restored'], actor={})))     # no per-mode report
        self.assertTrue(self.errors(g0=self.rec(1, 1000, ['restored'], actor={'restored': {'move': 3}})))
        self.assertTrue(self.errors(g0=self.rec(1, 1000, ['sequential'])))             # G0 is the restored mode
        self.assertTrue(self.errors(g0b=self.rec(2, 20, ['restored', 'sequential'])))  # wrong factor
        self.assertTrue(self.errors(g0=self.rec(1, 1000, ['restored'], verdict='FAIL')))
        other = {**self.rec(4, 20, ['restored', 'sequential'])}
        other['identity'] = {**other['identity'], 'selfplay_jsonl_sha256': 'other'}
        self.assertTrue(self.errors(g0b=other))                                         # another self-play file


class RepositoryTest(unittest.TestCase):
    """The committed c1 evidence as the CLI reads it."""

    def test_c1_chain_and_entry(self):
        found = cli.verdicts()
        self.assertGreaterEqual(len(found), 1)
        self.assertEqual(cli.chain_errors(found), [])
        entry = auto.cycle_entry(found[0]['record'])
        self.assertEqual((entry['decision'], entry['lower'], entry['candidate_mode']), ('KEEP_CHAMPION', -0.2, 'JOINT'))

    def test_c1_evidence_checks(self):
        found = cli.verdicts()
        verdict = found[0]['record']
        parent = verdict['parent']
        docs = cli.DOCS / 'c1'
        records = {n: json.loads((docs / f'{n}_record.json').read_text(encoding='utf-8')) for n in ('policy', 'value')}
        for name, record in records.items():
            meta = cli._sha(docs / f'{name}_best.json')
            self.assertEqual(auto.train_errors(name, record, 1, parent, meta, None, False,
                                               {'source.models': parent}), [])
        candidate = auto.expected_candidate('JOINT', records['policy'], records['value'], parent)
        self.assertEqual(auto.verdict_errors(verdict, 1, parent, None, 'JOINT', candidate, {}), [])
        tampered = {**verdict, 'next_champion': candidate}       # a KEEP that names the candidate as champion
        self.assertTrue(auto.verdict_errors(tampered, 1, parent, None, 'JOINT', candidate, {}))
        x2 = json.loads((cli.ROOT / 'docs' / 'external' / 'x2_c1.json').read_text(encoding='utf-8'))
        self.assertEqual(auto.x2_errors(x2, 1, candidate), [])
        self.assertTrue(auto.x2_errors(x2, 1, parent))
        selfplay = json.loads(cli.C1['selfplay'].read_text(encoding='utf-8'))
        recipe = {k: selfplay['identity'][k] for k in ('engine_hash', 'runtime_hash', 'value_device')}
        self.assertEqual(auto.selfplay_errors(selfplay, 1, parent, recipe), [])
        bad = auto.selfplay_errors({**selfplay, 'identity': {**selfplay['identity'], 'cycle': 2}}, 2, parent, recipe)
        self.assertTrue(any('run seed' in e for e in bad))

    def test_broken_chain(self):
        found = cli.verdicts()
        fake = {'path': found[0]['path'], 'record': {**found[0]['record'], 'cycle': 2, 'parent': {'x': 1}}}
        self.assertTrue(cli.chain_errors(found + [fake]))

    def test_x2_command(self):
        history = [auto.cycle_entry(cli.verdicts()[0]['record'])]
        cfg = {'engine': '/e/rapfi', 'engine_info': ['max_node=300'], 'engine_files': ['/e/a.bin'], 'workers': 4}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x2_config.json'
            path.write_text(json.dumps(cfg), encoding='utf-8')
            with mock.patch.object(cli, 'X2_CONFIG', path):
                lines = {mode: cli.command({'cycle': 2, 'history': history, 'step': auto._step(
                    'X2_MONITOR_READY', 'RUN_X2', candidate_mode=mode)})[0]
                    for mode in ('JOINT', 'VALUE_ONLY_FALLBACK')}
        joint, fallback = lines['JOINT'], lines['VALUE_ONLY_FALLBACK']
        self.assertIn('--policy-checkpoint runs/h7c/c2/policy/best.pt', joint)
        self.assertNotIn('--candidate-mode', joint)
        self.assertNotIn('--policy-record', joint)
        self.assertIn(f"--policy-checkpoint {cli.FROZEN['policy']}", fallback)   # the parent policy
        self.assertIn('--candidate-mode VALUE_ONLY_FALLBACK', fallback)
        self.assertIn('--policy-record runs/h7c/c2/policy/record.json', fallback)
        self.assertIn('--engine /e/rapfi', joint)
        self.assertIn('--engine-info max_node=300', joint)
        self.assertIn('--engine-file /e/a.bin', joint)
        self.assertIn('--workers 4', joint)
        self.assertNotIn('...', joint + fallback)

    def test_x2_paths_are_quoted(self):
        history = [auto.cycle_entry(cli.verdicts()[0]['record'])]
        cfg = {'engine': '/home/u/My Engines/rapfi', 'engine_info': ['max_node=300'],
               'engine_files': ['/home/u/My Engines/a b.bin'], 'workers': 4}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x2_config.json'
            path.write_text(json.dumps(cfg), encoding='utf-8')
            with mock.patch.object(cli, 'X2_CONFIG', path):
                line = cli.command({'cycle': 2, 'history': history,
                                    'step': auto._step('X2_MONITOR_READY', 'RUN_X2', candidate_mode='JOINT')})[0]
        self.assertIn("--engine '/home/u/My Engines/rapfi'", line)
        self.assertIn("--engine-file '/home/u/My Engines/a b.bin'", line)

    def test_error_states(self):
        for state in ('BLOCKED_TRAIN', 'BLOCKED_EVIDENCE', 'BLOCKED_SEED_RANGE', 'MANUAL_CODE_CHANGE_REVIEW_REQUIRED'):
            self.assertTrue(auto.is_error(state), state)
        for state in ('NOT_C2_READY', 'SELFPLAY_READY', 'CYCLE_COMPLETE', 'X2_CONFIG_REQUIRED',
                      'C3_RECIPE_TOOLING_REQUIRED', 'EVIDENCE_COMMIT_REQUIRED', 'MANUAL_REVIEW_REQUIRED'):
            self.assertFalse(auto.is_error(state), state)
        with mock.patch.object(cli, 'dirty_paths', return_value=['src/x.py']):
            self.assertEqual(cli.main(['verify', '--skip-frozen']), 1)
        with mock.patch.object(cli, 'dirty_paths', return_value=[]):
            self.assertEqual(cli.main(['verify', '--skip-frozen']), 0)   # NOT_C2_READY is not an error

    def test_uncommitted_recipe_is_a_commit_step(self):
        with mock.patch.object(cli, 'dirty_paths', return_value=['docs/h7c/c2/recipe.json']):
            self.assertEqual(cli.current()['step']['state'], 'EVIDENCE_COMMIT_REQUIRED')
        with mock.patch.object(cli, 'dirty_paths', return_value=['scripts/h7c_selfplay.py']):
            self.assertEqual(cli.current()['step']['state'], 'MANUAL_CODE_CHANGE_REVIEW_REQUIRED')

    def test_a_lone_cycle_index_does_not_advance(self):
        with TemporaryDirectory() as tmp:
            docs = Path(tmp)
            shutil.copytree(cli.DOCS / 'c1', docs / 'c1')
            (docs / 'c2').mkdir()
            (docs / 'c2' / 'cycle.json').write_text('{}', encoding='utf-8')
            with mock.patch.object(cli, 'DOCS', docs), mock.patch.object(cli, 'dirty_paths', return_value=[]):
                state = cli.current()
        self.assertEqual(state['cycle'], 2)
        self.assertNotEqual(state['step']['state'], 'CYCLE_COMPLETE')

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
