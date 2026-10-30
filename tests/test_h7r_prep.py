"""H7-R preparation after the pilot (docs/h7-rl-plan.md 9.11): suites drawn without openings already used (exact
history ids recomputed from moves, sources recorded, shortfalls refused, reproducible), the anchor non-inferiority
sizing on the paired difference, and the H-dev draft contract. No game is played."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hybrid import h7r_eval as ev
from hybrid import h7r_openings as op
from scripts import h7r_eval as cli
from tests.test_h7r_cli import QUIET

ROOT = Path(__file__).resolve().parents[1]


def record(tmp: Path, name: str, moves: list, fmt='h7c-arena-v1-arm', history_id='0' * 64) -> Path:
    path = tmp / name
    path.write_text(json.dumps({'format': fmt, 'openings': [{'pair': i, 'moves': m, 'history_id': history_id}
                                                            for i, m in enumerate(moves)]}))
    return path


class ExclusionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = op.build_suite('O1', 40, 11)

    def test_union_of_sources_with_ids_recomputed_from_moves(self):
        first = [o['moves'] for o in self.base['openings'][:5]]
        second = [o['moves'] for o in self.base['openings'][3:8]]  # three shared with the first
        a = record(self.root, 'a.json', first)  # a wrong stored history_id is ignored
        b = record(self.root, 'b.json', second, fmt='external-match-v1')
        ex = op.exclusion([a, b], self.root)
        self.assertEqual(ex['history_ids'], {o['history_id'] for o in self.base['openings'][:8]})
        self.assertEqual([(s['path'], s['openings'], s['distinct_histories']) for s in ex['sources']],
                         [('a.json', 5, 5), ('b.json', 5, 5)])
        self.assertEqual(ex['sources'][0]['sha256'], ev.file_sha256(a))
        suite = op.build_suite('O1', 40, 11, exclude=ex['history_ids'])
        self.assertFalse({o['history_id'] for o in suite['openings']} & ex['history_ids'])
        self.assertEqual(suite['exclusion']['excluded_draws'], 8)  # the same draws, the used ones skipped
        self.assertEqual(op.build_suite('O1', 40, 11, exclude=ex['history_ids']), suite)  # reproducible
        op.check_suite(suite)
        tampered = deepcopy(suite)
        tampered['exclusion']['history_ids'].append(suite['openings'][0]['history_id'])
        tampered['exclusion']['history_ids_sha256'] = op._sha(sorted(tampered['exclusion']['history_ids']))
        with self.assertRaisesRegex(ValueError, 'excluded'):
            op.check_suite(tampered)

    def test_contract_source_takes_each_arm_sample_not_the_suite(self):
        used = op.used_openings(ROOT / 'docs/h7r/contracts/R1-PILOT-1.json', ROOT)
        self.assertEqual(len(used['moves']), 1020)  # 9 x 60 + 3 x 160 drawn openings, not 3 suites of 300 / 184
        self.assertEqual(len({op.ids(m)['history_id'] for m in used['moves']}), 486)
        with self.assertRaisesRegex(ValueError, 'without arms'):  # a holdout contract has no arm samples
            op.used_openings(ROOT / 'docs/h7r/contracts/R1-SMOKE-HOLDOUT.json', ROOT)
        self.assertIn('not proof the arms were played', used['kind'])

    def test_refused_sources(self):
        suite_file = self.root / 'suite.json'
        suite_file.write_text(json.dumps(self.base))
        with self.assertRaisesRegex(ValueError, 'lists candidates'):
            op.exclusion([suite_file], self.root)
        with self.assertRaisesRegex(ValueError, 'not a known evaluation record'):
            op.exclusion([record(self.root, 'x.json', [[[7, 7]]], fmt='something-else')], self.root)
        with self.assertRaisesRegex(ValueError, 'not a move list'):
            op.exclusion([record(self.root, 'y.json', [[[7, 7], [99, 1]]])], self.root)
        with self.assertRaises(FileNotFoundError):
            op.exclusion([self.root / 'absent.json'], self.root)
        for name, body in (('missing', {'format': 'h7r-arena-v1-arm'}),
                           ('null', {'format': 'h7c-arena-v1-arm', 'openings': None}),
                           ('empty', {'format': 'external-match-v1', 'openings': []})):
            (self.root / f'{name}.json').write_text(json.dumps(body))
            with self.subTest(name), self.assertRaisesRegex(ValueError, 'without its openings list'):
                op.exclusion([self.root / f'{name}.json'], self.root)
        for name, moves in (('twice', [[7, 7], [7, 7]]), ('off-centre', [[6, 6], [7, 7], [8, 8]]),
                            ('no moves', None)):
            with self.subTest(name), self.assertRaisesRegex(ValueError, 'illegal|not a move list'):
                op.exclusion([record(self.root, f'{name.replace(" ", "_")}.json', [moves])], self.root)
        broken = json.loads((ROOT / 'docs/h7r/contracts/R1-SMOKE-1.json').read_text())
        broken['arms'][0]['suite']['sha256'] = '0' * 64
        (self.root / 'c.json').write_text(json.dumps(broken))
        with self.assertRaisesRegex(ValueError, 'not its SHA-256'):
            op.used_openings(self.root / 'c.json', ROOT)

    def test_a_shortfall_fails_and_is_explained(self):
        o2 = op.build_suite('O2', None, None)
        used = record(self.root, 'u.json', [o['moves'] for o in o2['openings'][:150]])
        ex = op.exclusion([used], self.root)
        with self.assertRaisesRegex(ValueError, '34 distinct openings left after excluding 150 used ones; 40 asked'):
            op.build_suite('O2', 40, None, exclude=ex['history_ids'])
        with self.assertRaisesRegex(ValueError, 'draws excluded.*no repeat fills'):
            op.build_suite('O1', 40, 11, max_attempts=45, exclude={o['history_id'] for o in self.base['openings']})

    def test_disjoint_suites_and_shared_beginnings(self):
        main = op.build_suite('O1', 30, 5)
        path = self.root / 'main.json'
        path.write_text(json.dumps(main))
        ex = op.exclusion([], self.root, [path])
        anchor = op.build_suite('O1', 30, 6, exclude=ex['history_ids'])
        self.assertFalse({o['history_id'] for o in main['openings']} & {o['history_id'] for o in anchor['openings']})
        self.assertEqual(ex['sources'][0]['kind'], 'suite reserved for another sample (disjoint)')
        prefixes = [o['moves'][:3] for o in main['openings'][:4]]
        self.assertEqual(op.prefix_overlap(main['openings'], prefixes)['3'],
                         sum(o['moves'][:3] in prefixes for o in main['openings']))
        self.assertEqual(op.prefix_overlap(main['openings'], [o['moves'] for o in main['openings']]), {})

    def test_suite_command_records_sources_and_overlaps(self):
        used = record(self.root, 'arena.json', [o['moves'] for o in self.base['openings'][:6]])
        three = record(self.root, 'old.json', [o['moves'][:3] for o in self.base['openings'][:2]])
        args = argparse.Namespace(name='O1', count=40, seed=11, compare_with=[], exclude_from=[used, three],
                                  disjoint_from=[], allow_dirty=True, root=self.root)
        body = cli.suite(args, log=QUIET)
        ex = body['exclusion']
        self.assertEqual([s['path'] for s in ex['sources']], ['arena.json', 'old.json'])
        self.assertEqual(ex['overlap']['by_source']['arena.json']['history_id'], 0)
        self.assertIn('3', ex['overlap']['shared_beginnings'])
        op.check_suite(body)


class AnchorRecordsTest(unittest.TestCase):
    """Measured anchor sizing reads exactly the two anchor arms of a confirm contract (fake games, no engine)."""

    def setUp(self):
        from unittest import mock
        from tests.test_h7r_cli import CLEAN, fake_play_game, world
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.w = world(Path(self.tmp.name))
        self.mock, self.clean, self.fake = mock, CLEAN, fake_play_game

    def play(self, arm, name=None, git=None, game=None):
        args = argparse.Namespace(contract=self.w['contract'], arm=arm, device='cpu', workers=1,
                                  jsonl=self.w['root'] / 'runs' / f'{name or arm}.jsonl', allow_dirty=True,
                                  threads_for_tests=True)
        with self.mock.patch('h7c_arena.play_game', game or self.fake), \
                self.mock.patch.object(cli, '_git', lambda _: git or self.clean):
            record = cli.play(args, log=QUIET, root=self.w['root'])
        path = self.w['root'] / 'out' / f'{name or arm}.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(record))
        return path

    def measure(self, *records):
        return cli.anchor_size(argparse.Namespace(margin=-0.02, true_difference=0.0, alpha=0.025, power=0.8,
                                                  sd=[0.205], pairs=[1040], arm_record=list(records),
                                                  contract=self.w['contract'], allow_dirty=True,
                                                  root=self.w['root']))

    def test_roles_order_and_refusals(self):
        cand, champ = self.play('candidate_vs_anchor'), self.play('champion_vs_anchor')
        out = self.measure(cand, champ)['measured']
        self.assertEqual((out['status'], out['direction'], out['pairs']),
                         ('MEASURED', 'candidate_vs_anchor - champion_vs_anchor', 30))
        self.assertGreater(out['pairs_needed_at_measured_sd'], 1)
        self.assertEqual(self.measure(champ, cand)['measured'], out)  # either order: candidate minus champion
        copy = self.w['root'] / 'out/candidate_copy.json'
        copy.write_bytes(cand.read_bytes())
        main = self.play('candidate_vs_champion')
        other = self.play('champion_vs_anchor', name='cva_other', git={'commit': 'other', 'dirty': False})
        for records, message in (((cand, cand), 'two different files'), ((cand, copy), 'once each'),
                                 ((main, champ), 'once each'), ((cand,), 'two different files'),
                                 ((cand, other), 'one clean code commit')):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.measure(*records)

    def test_zero_empirical_variance_gives_no_size(self):
        def flat(task, **kw):  # the candidate never gets the colour bonus: every pair 0.5, like the champion's
            ctx = task['context']
            return self.fake({**task, 'context': {**ctx, 'arm': 'flat'}}, **kw) | {'arm': task['arm']}
        out = self.measure(self.play('candidate_vs_anchor', name='flat', game=flat),
                           self.play('champion_vs_anchor'))['measured']
        self.assertEqual((out['status'], out['sd'], out['pairs_needed_at_measured_sd']),
                         ('ZERO_EMPIRICAL_VARIANCE', 0.0, None))
        self.assertIn('no empirical size', out['note'])


class AnchorSizingTest(unittest.TestCase):
    def test_known_values(self):
        out = ev.anchor_sizing(-0.02, [0.18, 0.205, 0.23], [400, 800, 1040])
        self.assertEqual([r['pairs_needed'] for r in out['rows']], [636, 825, 1039])
        self.assertEqual([r['power_at']['800'] for r in out['rows']], [0.8815, 0.788, 0.6913])
        self.assertEqual([r['power_at']['400'] for r in out['rows']], [0.6034, 0.4965, 0.4126])
        self.assertEqual(out['rows'][2]['power_at']['1040'], 0.8007)
        with self.assertRaisesRegex(ValueError, 'above the margin'):
            ev.noninferiority_pairs(0.2, -0.02, true_difference=-0.03)

    def test_inputs_are_checked_in_the_library(self):
        bad = (dict(sds=[-0.2]), dict(sds=[0]), dict(sds=[float('nan')]), dict(sds=[float('inf')]),
               dict(pairs=[0]), dict(pairs=[-5]), dict(pairs=[1.5]), dict(pairs=[True]), dict(alpha=0.0),
               dict(alpha=0.6), dict(power=0.3), dict(power=1.0), dict(margin=0.0), dict(margin=float('nan')),
               dict(true_difference=float('nan')), dict(sds=[]), dict(pairs=[]))
        for change in bad:
            kw = {'margin': -0.02, 'sds': [0.2], 'pairs': [400], 'true_difference': 0.0, 'alpha': 0.025,
                  'power': 0.8, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                ev.anchor_sizing(kw['margin'], kw['sds'], kw['pairs'], kw['true_difference'], kw['alpha'],
                                 kw['power'])
        with self.assertRaises(ValueError):
            ev.noninferiority_power(400, 0, -0.02)  # the lower-level functions check too
        with self.assertRaises(ValueError):
            ev.noninferiority_pairs(-0.2, -0.02)

    def test_paired_sd_uses_the_covariance(self):
        ids = [f'o{i}' for i in range(8)]
        a_scores = [1.0, 0.5, 0.5, 0.0, 1.0, 0.5, 0.5, 0.0]
        table = lambda scores: {'pairs': [{'opening_id': i, 'pair_score': s} for i, s in zip(ids, scores)]}  # noqa
        same = ev.paired_difference_sd(table(a_scores), table(a_scores))
        self.assertEqual((same['sd'], same['mean']), (0.0, 0.0))  # identical arms: D_i = 0 whatever each arm's SD
        flipped = ev.paired_difference_sd(table(a_scores), table([1 - s for s in a_scores]))
        self.assertGreater(flipped['sd'], 0.5)
        with self.assertRaisesRegex(ValueError, 'twice'):
            ev.paired_difference_sd(table(a_scores), {'pairs': [{'opening_id': 'x', 'pair_score': 0.5}] * 8})
        with self.assertRaisesRegex(ValueError, 'same openings'):
            ev.paired_difference_sd(table(a_scores), table(a_scores[:7] + [0.5]) | {'pairs': table(a_scores)['pairs'][:7]})
        with self.assertRaisesRegex(ValueError, 'no pair score'):
            ev.paired_difference_sd(table(a_scores), table([0.3] + a_scores[1:]))
        with self.assertRaisesRegex(ValueError, 'invalid pairs'):
            ev.paired_difference_sd(table(a_scores), {**table(a_scores), 'valid': False})
        self.assertEqual(same['status'], 'ZERO_EMPIRICAL_VARIANCE')

    def test_command(self):
        args = argparse.Namespace(margin=-0.02, true_difference=0.0, alpha=0.025, power=0.8, sd=[0.205],
                                  pairs=[1040], arm_record=[], contract=None)
        out = cli.anchor_size(args)
        self.assertEqual((out['format'], out['rows'][0]['pairs_needed']), ('h7r-anchor-sizing-v1', 825))
        with self.assertRaisesRegex(ValueError, '--contract'):
            cli.anchor_size(argparse.Namespace(**{**vars(args), 'arm_record': [Path('a'), Path('b')]}))


class HdevContractTest(unittest.TestCase):
    """R1-HDEV-1 (docs/h7-rl-plan.md 9.12): fixed, one reserved seed, the generator accepts exactly its run, the
    training path refuses its seed, and the access rules are in the contract."""

    def setUp(self):
        self.path = ROOT / 'docs/h7r/contracts/R1-HDEV-1.json'
        self.contract, self.sha = ev.load_contract(self.path, ROOT, check_files=False)

    def test_fixed_run_and_one_reserved_seed(self):
        import h7c_selfplay as sp
        from hybrid.h6_labels import RESERVED_SEEDS
        h = self.contract['holdout']
        self.assertEqual((self.contract['status'], h['split'], h['games'], h['actor'], h['seed'], h['cycle']),
                         ('fixed', 'dev', 400, 'c5', 8787, 1))
        self.assertEqual(ev.contract_seeds(self.contract), {8787})
        self.assertIn(8787, RESERVED_SEEDS)
        sp.check_seeds(8787, 1, range(400), holdout=True)
        with self.assertRaisesRegex(ValueError, 'run-seed 8787'):
            sp.check_seeds(8787, 1, range(3))  # the training generator never takes the H-dev seed
        pilot = json.loads((ROOT / 'docs/h7r/contracts/R1-PILOT-1.json').read_text())
        self.assertEqual(self.contract['models']['c5'], pilot['models']['c5'])  # the research champion c5

    def test_the_generator_accepts_exactly_this_run(self):
        import h7c_selfplay as sp
        actor = self.contract['models']['c5']['identity']
        ok = argparse.Namespace(run_seed=8787, cycle=1, start=0, games=400, allow_dirty=True)
        self.assertEqual(sp.holdout_identity(self.path, ok, actor)['holdout'],
                         {'contract_id': 'R1-HDEV-1', 'contract_sha256': self.sha, 'split': 'dev'})
        orig = json.loads((ROOT / 'docs/h7r/contracts/R1-PILOT-1.json').read_text())['models']['orig']['identity']
        for change, models in (({'run_seed': 8788}, actor), ({'cycle': 2}, actor), ({'games': 401}, actor),
                               ({}, orig)):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'not the holdout contract'):
                sp.holdout_identity(self.path, argparse.Namespace(**{**vars(ok), **change}), models)

    def test_access_rules_are_fixed(self):
        access = self.contract['access']
        self.assertIn('checkpoint / recipe selection', access['allowed'])
        self.assertTrue(any('gradient training' in f for f in access['forbidden']))
        self.assertTrue(any('confirmation' in f for f in access['forbidden']))
        self.assertIn('H-conf', access['separate'])
        self.assertIn('not the H-dev reading rule', self.contract['holdout_notes']['stats'])
        self.assertEqual(self.contract['research_baseline']['champion'], 'c5')


if __name__ == '__main__':
    unittest.main()
