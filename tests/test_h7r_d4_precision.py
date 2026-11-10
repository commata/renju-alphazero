"""scripts/h7r_d4_precision.py: the sizing arithmetic, the log phases, the pre-fixed rule, the summary mode on the
committed records, and the journal mode's rebuild-and-compare check and per-game statistics (synthetic journals)."""
import json
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import h7r_d4_precision as pr

ROOT = Path(__file__).resolve().parents[1]


class ArithmeticTest(unittest.TestCase):
    def test_sizing(self):
        se = 0.02
        self.assertAlmostEqual(pr.Z_ALPHA, 1.959964, places=5)
        g = pr.games_needed(se, 0.03, pr.z_power(0.8))
        self.assertEqual(g, -(-396 * ((pr.Z_ALPHA + pr.z_power(0.8)) * se / 0.03) ** 2 // 1))
        self.assertGreater(pr.games_needed(se, 0.03, pr.z_power(0.8, 3, 'independent')), g)
        self.assertAlmostEqual(pr.mde(se, pr.z_power(0.8), pr.G0), (pr.Z_ALPHA + pr.z_power(0.8)) * se)
        self.assertAlmostEqual(pr.mde(se, pr.z_power(0.8), 4 * pr.G0), pr.mde(se, pr.z_power(0.8), pr.G0) / 2)
        self.assertAlmostEqual(pr.z_power(0.8, 3, 'independent'), pr.Z(0.8 ** (1 / 3)))

    def test_phases_and_rule(self):
        log = '\n'.join(['100.0 __START__', '160.0 [x-replay] 1/2', '760.0 [x-replay] 2/2',
                         '770.0 [x-rapfi] 1/3', '890.0 [x-rapfi] 3/3', '891.0 __EXIT__ 0'])
        self.assertEqual(pr.phase_minutes(log, 'x'), {'replay': 10.0, 'rapfi': 2.0})
        with self.assertRaisesRegex(ValueError, 'no replay'):
            pr.phase_minutes(log, 'y')
        proj = pr.projection({'a': 0.02, 'b': 0.01, 'c': 0.015}, hours_per_g0=1.0)
        self.assertEqual(set(proj), {'0.02', '0.03', '0.05', 'mde'})
        self.assertEqual(proj['0.03']['all_three_games']['dependent_bound'],
                         pr.games_needed(0.02, 0.03, pr.z_power(0.8)))  # the largest SE decides
        cheap = pr.proposal(proj)
        self.assertEqual(cheap['evaluation_expansion'], 'PROPOSE')
        dear = pr.proposal(pr.projection({'a': 0.026, 'b': 0.02, 'c': 0.021}, hours_per_g0=5.5))
        self.assertEqual(dear['evaluation_expansion'], 'DO_NOT_PROPOSE')
        self.assertIn('next_d4_mde_at_396_games', dear)

    def test_fixed_values(self):
        self.assertEqual((pr.EFFECTS, pr.PRIMARY_EFFECT, pr.POWER, pr.ALPHA_ONE_SIDED, pr.BUDGET_HOURS, pr.PRIMARY_K),
                         ((0.02, 0.03, 0.05), 0.03, 0.80, 0.025, 24.0, 3))


class SummaryModeTest(unittest.TestCase):
    def test_on_the_committed_records(self):
        r = pr.summary_report(ROOT)
        self.assertEqual(set(r['studies']), {'D1', 'BASELINE'})
        d1 = r['studies']['D1']
        self.assertEqual(d1['reading'], 'NOT_ESTABLISHED')
        self.assertEqual(d1['comparisons']['wide31002:narrow31002']['observations'], 520)
        self.assertEqual(d1['cost']['models'], 7)
        self.assertGreater(d1['cost']['replay_minutes'], 250)
        for name, st in r['studies'].items():
            joint = st['projection']['0.03']['all_three_games']
            self.assertLessEqual(joint['dependent_bound'], joint['independent_bound'], name)


def outcome(score):
    return {'kind': 'resolved', 'score': score, 'reason': None}


class PerGameTest(unittest.TestCase):
    def test_concentration_and_standard_errors(self):
        games = list(range(10))
        outcomes = {'w': outcome(1), 'd': outcome(.5), 'l': outcome(0)}
        obs = ([{'game_index': 0, 'key_a': 'w', 'key_b': 'l'}] * 5 + [{'game_index': 1, 'key_a': 'l', 'key_b': 'w'}]
               + [{'game_index': g, 'key_a': 'd', 'key_b': 'd'} for g in (2, 3, 4)])
        r = pr.per_game(obs, outcomes, games)
        self.assertEqual((r['observations'], r['decisive'], r['games_with_observations'], r['games_with_decisive']),
                         (9, 6, 5, 2))
        self.assertAlmostEqual(r['mean_delta'], 4 / 9)
        self.assertEqual((r['max_observations_in_a_game'], r['top10pct_games_share_of_observations']), (5, 5 / 9))
        self.assertAlmostEqual(r['top10pct_games_share_of_decisive'], 5 / 6)
        m = 4 / 9
        s, n = [5, -1, 0, 0, 0] + [0] * 5, [5, 1, 1, 1, 1] + [0] * 5
        self.assertAlmostEqual(r['cluster_se'], sqrt(10 / 9 * sum((a - m * b) ** 2 for a, b in zip(s, n))) / 9)
        self.assertGreater(r['design_effect'], 1)  # five identical results in one game
        self.assertAlmostEqual(r['largest_one_game_influence'], abs(m - (-1 / 4)))
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            pr.per_game(obs, {**outcomes, 'w': {'kind': 'unresolved', 'score': None}}, games)


class JournalModeTest(unittest.TestCase):
    """The rebuild-and-compare path on synthetic journals: the committed statistics come back, a changed summary or
    a foreign journal line is refused."""

    def setUp(self):
        from hybrid import h7r_d4 as d4
        import h7r_d1_d4 as dd
        self.d4, self.dd = d4, dd
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.root = Path(self.tmp.name)
        D = 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1/d4'
        self.c = json.loads((ROOT / f'{D}/d1-d4-contract.json').read_text())
        self.c['stats'] = {**self.c['stats'], 'samples': 200}
        self.games = {g: {'game_index': g} for g in range(12)}
        self.manifest = {'games': [{'game_index': g, 'plies': [5]} for g in self.games]}
        identity = {'run': 'synthetic'}
        self.sha = d4.digest(identity)
        self.obs = []
        for g in self.games:
            for p in dd.COMPARISONS:
                if (g + len(p)) % 3:
                    self.obs.append({'game_index': g, 'compare': p, 'key_a': f'a{g}{p}', 'key_b': f'b{g}{p}',
                                     'moves': [], 'move_a': [7, 7], 'move_b': [7, 8]})
        self.outcomes = {o[k]: outcome([0, .5, 1][(o['game_index'] + i) % 3])
                         for o in self.obs for i, k in enumerate(('key_a', 'key_b'))}
        counts = d4.multiplicities(len(self.games), self.c['stats']['samples'], self.c['stats']['seed'])
        comparisons = {p: d4.selection_stats([o for o in self.obs if o['compare'] == p], self.outcomes,
                                             sorted(self.games), counts, self.c['stats']) for p in dd.COMPARISONS}
        for p in dd.REFERENCE:
            for k in ('label95', 'label_individual'):
                comparisons[p].pop(k)
        self.summary = {'identity_sha256': self.sha, 'comparisons': json.loads(json.dumps(comparisons))}
        for rel, body in ((f'{D}/d1-d4-contract.json', self.c), (f'{D}/full/identity.json', identity),
                          (f'{D}/full/summary.json', self.summary), (self.c['output']['sample'], self.manifest),
                          ('run/identity.json', identity)):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(json.dumps(body))
        (root / 'run/replay.jsonl').write_text(''.join(
            json.dumps({'game_index': g, 'identity_sha256': self.sha, 'plies': []}) + '\n' for g in self.games))
        keys = sorted({o[k] for o in self.obs for k in ('key_a', 'key_b')})
        (root / 'run/branches.jsonl').write_text(''.join(
            json.dumps({'key': k, 'identity_sha256': self.sha}) + '\n' for k in keys))
        log = root / pr.STUDIES['D1']['logs'] / 'full-1.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('0.0 [d1-d4-replay] a\n600.0 [d1-d4-replay] b\n660.0 [d1-d4-rapfi] c\n720.0 [d1-d4-rapfi] d\n')

    def run_it(self):
        import argparse
        import h7c_adjudicate as ad
        import h7r_d4 as base
        table = {k: {'key': k} for k in self.outcomes}
        with mock.patch.object(self.dd, 'inputs', return_value=(self.manifest, self.games, {})), \
                mock.patch.object(self.dd, 'observations_for', return_value=self.obs), \
                mock.patch.object(ad, 'branch_table', return_value=table), \
                mock.patch.object(base, 'validate_branch', side_effect=lambda r, t, p: self.outcomes[r['key']]):
            return pr.journals_report(argparse.Namespace(run_dir=self.root / 'run', cache='c'), self.root)

    def test_rebuild_compare_and_per_game(self):
        out = self.run_it()
        self.assertEqual(out['mode'], 'journals')
        self.assertEqual(set(out['per_game']), set(self.dd.COMPARISONS))
        row = out['per_game']['wide31001:narrow31001']
        self.assertEqual(row['se_used'], max(row['bootstrap_se'], row['cluster_se'], row['jackknife_se']))
        self.assertEqual(out['cost']['replay_minutes'], 10.0)

    def test_refusals(self):
        path = self.root / 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1/d4/full/summary.json'
        bad = json.loads(path.read_text())
        bad['comparisons']['wide31001:narrow31001']['better'] += 1
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'do not give the committed statistics back'):
            self.run_it()
        path.write_text(json.dumps(self.summary))
        with open(self.root / 'run/branches.jsonl', 'a') as f:
            f.write(json.dumps({'key': 'other', 'identity_sha256': 'x'}) + '\n')
        with self.assertRaisesRegex(ValueError, 'not a record of this run'):
            self.run_it()


if __name__ == '__main__':
    unittest.main()
