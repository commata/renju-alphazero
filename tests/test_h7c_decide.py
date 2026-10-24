"""H7-c next-cycle decision (scripts/h7c_decide.py): the pre-registered triggers and the case order."""
import unittest

from scripts import h7c_decide as hd


def d2(proof_dce=-0.02, visits_dce=-0.01, visits_dtop1=0.01, n_proof=100, n_visits=400):
    return {f'proof_one_hot/{s}': {'n': n_proof // 2, 'delta_ce': proof_dce, 'delta_top1': 0.0} for s in ('black', 'white')} | \
           {f'visits/{s}': {'n': n_visits // 2, 'delta_ce': visits_dce, 'delta_top1': visits_dtop1} for s in ('black', 'white')}


def d4(overall=0.06, network=0.05):
    return {'training_metric': {'by_route': {'all': {'wmse_relative_improvement': overall},
                                             'network': {'wmse_relative_improvement': network},
                                             'exact_leaf': {'n': 0}}}}


def sb(reading='NOT_DETECTED', stronger=(), saturation=None, seconds=1.0):
    arms = {f'S{50 * k}': {'factor': k, 'stronger': k in stronger, 'cost': {'tree_seconds_mean': seconds * k}}
            for k in (2, 4, 8)}
    return {'reading': reading, 'arms': arms,
            'saturation': None if saturation is None else {'reading': saturation, 'rule': {'factor': 8}}}


SELFPLAY = {'wall_seconds': 4291.4, 'summary': {'routes': {'tree': 22068}}}


def case(diag_d2=None, diag_d4=None, summary=None):
    return hd.decide({'parts': {'d2': diag_d2 or d2(), 'd4': diag_d4 or d4()}}, summary or sb(), SELFPLAY)


class TriggerTest(unittest.TestCase):
    def test_policy_trigger(self):
        # proof rows are 100 of 500; reduction 100*0.02 = 2 of 2 + 400*0.01 = 6: share 1/3
        self.assertFalse(hd.policy_trigger(d2())['fired'])
        self.assertAlmostEqual(hd.policy_trigger(d2())['proof_share_of_reduction'], 2 / 6, places=4)
        self.assertTrue(hd.policy_trigger(d2(proof_dce=-0.04))['fired'])          # 4 of 8: exactly 0.5
        self.assertTrue(hd.policy_trigger(d2(visits_dtop1=0.0))['fired'])         # visits top-1 did not rise
        self.assertTrue(hd.policy_trigger(d2(visits_dce=0.01))['fired'])          # the visits rows got worse: share > 1

    def test_value_trigger(self):
        self.assertFalse(hd.value_trigger(d4(0.06, 0.03))['fired'])               # exactly half: not below
        self.assertTrue(hd.value_trigger(d4(0.06, 0.029))['fired'])
        self.assertTrue(hd.value_trigger(d4(0.06, 0.0))['fired'])
        self.assertTrue(hd.value_trigger(d4(-0.01, -0.02))['fired'])

    def test_cost_ceiling(self):
        # 22,068 tree positions / 6 workers against 2 x 4,291.4 s: at most 2.334 s per tree move
        ok = hd.budget_case(sb('BUDGET_HELPS', (4,), seconds=0.58), SELFPLAY)
        self.assertEqual((ok['factor'], ok['affordable']), (4, True))
        self.assertEqual(ok['budget'], {'simulations': 200, 'tactical_simulations': 400})
        dear = hd.budget_case(sb('BUDGET_HELPS', (4,), seconds=0.59), SELFPLAY)    # 2.36 s per move at S200
        self.assertFalse(dear['affordable'])


class CaseOrderTest(unittest.TestCase):
    def test_order(self):
        self.assertEqual(case(summary=sb('BUDGET_HELPS', (2, 4)), diag_d2=d2(visits_dtop1=0))['case'], 'A')  # A first
        self.assertEqual(case(summary=sb('BUDGET_HELPS', (8,), seconds=1.0))['case'], 'A_COST_REVIEW')
        self.assertEqual(case(diag_d2=d2(visits_dtop1=0), diag_d4=d4(0.06, 0.0))['case'], 'B_POLICY')
        self.assertEqual(case(diag_d4=d4(0.06, 0.0))['case'], 'B_VALUE')
        self.assertEqual(case()['case'], 'RUN_SATURATION')
        v9 = case(summary=sb(saturation='SATURATED'))
        self.assertEqual((v9['case'], v9['v9_e1']['met']), ('V9_ENTRY_E1', True))
        self.assertEqual(case(summary=sb(saturation='INCONCLUSIVE'))['case'], 'REPLICATE')
        late = case(summary=sb(saturation='STRONGER', seconds=0.2))
        self.assertEqual((late['case'], late['budget']['factor']), ('A', 8))
        blocked = case(summary=sb(saturation='SATURATED'), diag_d2=d2(visits_dtop1=0))
        self.assertEqual((blocked['case'], blocked['v9_e1']['met']), ('B_POLICY', False))  # B before V9


if __name__ == '__main__':
    unittest.main()
