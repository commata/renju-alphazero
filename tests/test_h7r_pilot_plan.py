"""The R1-PILOT-1 draft (docs/h7-rl-plan.md 9.2): its recorded planning numbers are the ones its own sources and
formulas give, and the contract loads with the current tools. Planning only: nothing is played."""
import json
from pathlib import Path
from statistics import NormalDist
import unittest

from hybrid import h7r_eval as ev
from hybrid import h7r_openings as op

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'docs/h7r/contracts/R1-PILOT-1.json'


def power(n, sd, delta=0.077, alpha=0.025):
    return NormalDist().cdf(delta * n ** 0.5 / sd - NormalDist().inv_cdf(1 - alpha))


class PilotPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = json.loads(PILOT.read_text())
        cls.arms = {a['name']: a for a in cls.body['arms']}

    def test_the_draft_loads_and_stays_draft(self):
        contract, _ = ev.load_contract(PILOT, ROOT, check_files=False)
        self.assertEqual((contract['purpose'], contract['status']), ('pilot', 'draft'))

    def test_sizes_seeds_and_samples(self):
        planning = self.body['planning']
        self.assertEqual(sum(a['pairs'] for a in self.body['arms']), planning['total_pairs_proposed'])
        self.assertEqual(2 * planning['total_pairs_proposed'], planning['total_games_proposed'])
        for suite in ('O0', 'O1', 'O2'):
            small = [self.arms[f'{suite}_{c}'] for c in ('c8_vs_c5', 'mask_vs_c5', 'c5_vs_orig')]
            control = self.arms[f'{suite}_c5s400_vs_c5']
            self.assertEqual({(a['seed'], a['pairs'], a['pairing']) for a in small}, {(small[0]['seed'], 60,
                                                                                       f'{suite}_small')})
            self.assertEqual((control['pairs'], control['pairing'], control['tested']['budget']),
                             (160, f'{suite}_positive_control', 8))
            body = json.loads((ROOT / control['suite']['path']).read_text())
            ids = lambda a: {o['history_id'] for o in op.pick(body, a['pairs'], a['seed'])}  # noqa: E731
            self.assertEqual(len(ids(small[0]) & ids(control)), planning['opening_overlap_small_vs_control'][suite])
        self.assertEqual(sorted(self.body['execution_plan']['arm_order_proposed']), sorted(self.arms))

    def test_cost_scenarios_follow_the_stated_formula(self):
        cost = self.body['planning']['cost']
        c8 = json.loads((ROOT / 'docs/h7c/c8/arena/official.json').read_text())
        self.assertAlmostEqual(cost['s50_seconds_per_pair_6_workers_proxy'], c8['wall_seconds'] / 50, places=3)
        sb = json.loads((ROOT / 'docs/h7c/sb_summary.json').read_text())['arms']['S400']['cost']
        self.assertAlmostEqual(cost['original_sb_arm_proxy'], (1 + sb['tested_seconds'] / sb['opponent_seconds']) / 2,
                               places=4)
        m = cost['s400_arm_cost_multiplier_assumed']
        for row in cost['scenario_estimates']:
            weight = 9 * row['small_pairs_per_arm'] + 3 * row['control_pairs_per_arm'] * m
            self.assertAlmostEqual(row['hours_6_workers_proxy'],
                                   weight * cost['s50_seconds_per_pair_6_workers_proxy'] / 3600, places=3)
            self.assertAlmostEqual(row['hours_2_workers_proxy'],
                                   weight * cost['s50_seconds_per_pair_2_workers_smoke'] / 3600, places=2)
            self.assertAlmostEqual(row['control_power_per_suite_approx'], power(row['control_pairs_per_arm'], 0.35),
                                   places=4)

    def test_control_power_and_its_sources(self):
        pp = self.body['positive_control']['planning_power']
        saturation = json.loads((ROOT / pp['effect_source']['path']).read_text())['saturation']
        self.assertEqual(ev.file_sha256(ROOT / pp['effect_source']['path']), pp['effect_source']['sha256'])
        self.assertAlmostEqual(saturation['score'] - 0.5, pp['effect_source_observed_improvement'], places=4)
        self.assertAlmostEqual(pp['power_approx'], power(160, pp['pair_sd_assumed']), places=4)
        z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)
        self.assertEqual(pp['pairs_for_power_at_least_0_8_approx'], int(-(-(z * 0.35 / 0.077) ** 2 // 1)))
        lo, hi = saturation['improvement_ci95']
        sd = (hi - lo) / 2 * saturation['pairs'] ** 0.5 / 1.96
        self.assertAlmostEqual(pp['sd_sensitivity']['pair_sd_approx'], sd, places=3)
        self.assertAlmostEqual(pp['sd_sensitivity']['power_at_160_approx'], power(160, sd), places=3)


if __name__ == '__main__':
    unittest.main()
