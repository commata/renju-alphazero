"""rt30.engine_v2: TB-RT30-v1 plus the V8-B slack retry (docs/web-ser5.md §9.5)."""
import unittest

from analysis.mcts_v8 import WIN, SearchDiagnostics, _BudgetedSolver
from analysis.mcts_v8_agent import MCTSV8Agent
from rt30 import engine_v2, fast, tb_rt30_v1, tb_rt30_v2
from rt30.deadline import Deadline
from rt30.engine import RT30Agent
from rt30.engine_v2 import RT30AgentV2
from search.mcts_v5 import _RootContext

from test_rt30_engine import CONFIG, POSITIONS, TickClock
from test_rt30_fast_windows import recorded_position


def without_seconds(diag):
    return {k: v for k, v in vars(diag).items() if not k.endswith('_seconds')}


class _Engine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fast.install_all()

    @classmethod
    def tearDownClass(cls):
        fast.uninstall_all()


class NoDeadlineIsTB1Test(_Engine):
    def test_same_move_and_diagnostics_on_every_route(self):
        for name, (key, route) in POSITIONS.items():
            game = recorded_position(*key)
            frozen = MCTSV8Agent(seed=7, **CONFIG)
            expected = frozen.select_move(game)
            v2 = RT30AgentV2(seed=7, **CONFIG)
            self.assertEqual(v2.select_move(game), expected, name)
            self.assertEqual(without_seconds(v2.diagnostics), without_seconds(frozen.diagnostics), name)
            info = v2.rt_info
            self.assertEqual((info.timeouts, info.attack_cap_hit, info.attack_retried), ([], False, False), name)


class RetryOffIsV1Test(_Engine):
    def test_same_as_v1_under_the_same_clock(self):
        for name, at in (('own_vct', 0.2), ('tree', 0.5), ('stage4', 1.0)):
            game = recorded_position(*POSITIONS[name][0])
            v1, v2 = RT30Agent(seed=7, **CONFIG), RT30AgentV2(seed=7, attack_retry=False, **CONFIG)
            m1 = v1.select_move(game, deadline=Deadline(at, TickClock()))
            m2 = v2.select_move(game, deadline=Deadline(at, TickClock()))
            self.assertEqual(m1, m2, name)
            self.assertEqual(without_seconds(v1.diagnostics), without_seconds(v2.diagnostics), name)
            self.assertEqual((v1.rt_info.timeouts, v1.rt_info.skipped), (v2.rt_info.timeouts, v2.rt_info.skipped))


class RetryTest(_Engine):
    KEY = POSITIONS['own_vct'][0]

    def attack_ticks(self):
        """Clock readings of the full V8-B search on the position (it proves a win)."""
        game = recorded_position(*self.KEY)
        clock = TickClock(1.0)
        diag = SearchDiagnostics()
        move, state = engine_v2._own_vct_attack(
            game, _RootContext(game.legal_moves(), diag), diag, node_limit=20_000, call_limit=10_000,
            node_budget=200_000, deadline=Deadline(1e12, clock))
        self.assertIsNotNone(move)
        self.assertIsNone(state)
        return clock.now

    def test_soft_cap_cut_win_is_recovered_in_the_time_left(self):
        ticks = self.attack_ticks()
        step = 0.001
        cap = 0.5 * ticks * step  # the soft cap stops V8-B half way
        game = recorded_position(*self.KEY)
        v1 = RT30Agent(seed=7, timing={'attack_seconds': cap}, **CONFIG)
        m1 = v1.select_move(game, deadline=Deadline(1e9, TickClock(step)))
        self.assertIn('attack', v1.rt_info.timeouts)
        self.assertNotEqual(v1.diagnostics.v8_route, 'own_vct')

        v2 = RT30AgentV2(seed=7, timing={'attack_seconds': cap}, **CONFIG)
        m2 = v2.select_move(game, deadline=Deadline(1e9, TickClock(step)))
        info = v2.rt_info
        self.assertTrue(info.attack_cap_hit and info.attack_retried and info.attack_retry_win)
        self.assertNotIn('attack', info.timeouts)
        self.assertEqual(v2.diagnostics.v8_route, 'own_vct')
        self.assertEqual(info.best[-1], ('own_vct_retry', tuple(m2)))
        self.assertNotEqual(m1, m2)
        # A proof, re-checked with a fresh frozen solver.
        solver = _BudgetedSolver(node_limit=20_000, call_limit=10_000, node_budget=200_000)
        self.assertEqual(solver.attack_status(game, m2), WIN)
        self.assertEqual(game.history, recorded_position(*self.KEY).history)  # the game is left as it was

    def test_no_retry_when_the_move_deadline_cut_v8b(self):
        game = recorded_position(*self.KEY)
        v2 = RT30AgentV2(seed=7, **CONFIG)
        v2.select_move(game, deadline=Deadline(0.2, TickClock()))
        info = v2.rt_info
        self.assertIn('attack', info.timeouts)
        self.assertEqual((info.attack_cap_hit, info.attack_retried), (False, False))

    def test_no_retry_without_time_left(self):
        ticks = self.attack_ticks()
        step = 0.001
        cap = 0.5 * ticks * step
        game = recorded_position(*self.KEY)
        # The move's deadline leaves the cap but not the final margin plus the retry minimum after it.
        v2 = RT30AgentV2(seed=7, timing={'attack_seconds': cap, 'puct_reserve': 0.0}, **CONFIG)
        at = cap + 1.5 + 0.2
        move = v2.select_move(game, deadline=Deadline(at, TickClock(step)))
        info = v2.rt_info
        self.assertTrue(info.attack_cap_hit)
        self.assertFalse(info.attack_retried)
        self.assertIn('attack', info.timeouts)
        self.assertIn(move, game.legal_moves())

    def test_budget_end_is_not_resumable(self):
        game = recorded_position(*self.KEY)
        diag = SearchDiagnostics()
        move, state = engine_v2._own_vct_attack(
            game, _RootContext(game.legal_moves(), diag), diag, node_limit=20_000, call_limit=10_000,
            node_budget=50, deadline=None)
        self.assertIsNone(move)
        self.assertIsNone(state)


class ConfigTest(unittest.TestCase):
    def test_v2_is_v1_plus_the_retry(self):
        v1, v2 = tb_rt30_v1.TB_RT30_V1, tb_rt30_v2.TB_RT30_V2
        self.assertEqual({k for k in v2 if v2[k] != v1.get(k)}, {'timing', 'attack_retry'})
        self.assertEqual({k: v for k, v in v2['timing'].items() if k != 'attack_retry_min'}, v1['timing'])
        self.assertTrue(v2['attack_retry'])
        self.assertEqual(tb_rt30_v2.STATUS, 'candidate')
        self.assertNotEqual(tb_rt30_v2.config_hash(), tb_rt30_v1.config_hash())

    def test_make_agent(self):
        agent = tb_rt30_v2.make_agent(None, seed=3, uniform_prior=True)
        try:
            self.assertIsInstance(agent, RT30AgentV2)
            self.assertEqual(agent.name, 'TB-RT30-v2 (uniform prior)')
            self.assertEqual(agent.timing['attack_seconds'], 8.0)
            self.assertTrue(agent.attack_retry)
        finally:
            fast.uninstall_all()


if __name__ == '__main__':
    unittest.main()
