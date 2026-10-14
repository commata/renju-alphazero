"""rt30.engine: TB-Engine-v1 without a deadline; bounded, always-playable with one (docs/web-ser5.md §3.3)."""
import unittest

from analysis.mcts_v8_agent import MCTSV8Agent
from analysis.tb_engine_v1 import TB_ENGINE_V1
from rt30 import fast_windows
from rt30.deadline import Deadline
from rt30.engine import RT30Agent, RT30_TIMING, check_config, limited

from test_rt30_fast_windows import recorded_position

# The tree route needs the H3 checkpoint (not in the repository): the TB1 configuration with a
# uniform PUCT prior exercises the same flow and modules.
CONFIG = {**TB_ENGINE_V1, 'puct_prior': 'uniform'}
POSITIONS = {  # recorded E3-A positions with quick searches: (seed, line, ply), route
    'tree': ((8415, 0, 13), 'tree'),
    'own_vct': ((8415, 11, 15), 'own_vct'),
    'stage4': ((8415, 6, 17), 'stage4'),       # every forced defense UNSAFE: widening and fallback
    'stage4_safe': ((8415, 3, 11), 'stage4'),
}
SLOW_ATTACK = (8415, 9, 26)  # V8-B dominated: 51 s frozen, 24 s with the fast windows (this container)


class TickClock:
    """Deterministic clock: every reading advances ``step`` seconds."""

    def __init__(self, step=0.001):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


class _Engine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fast_windows.install()  # exact (test_rt30_fast_windows); only makes these tests faster

    @classmethod
    def tearDownClass(cls):
        fast_windows.uninstall()


class NoDeadlineIsTB1Test(_Engine):
    def test_same_move_and_diagnostics_on_every_route(self):
        for name, (key, route) in POSITIONS.items():
            game = recorded_position(*key)
            frozen = MCTSV8Agent(seed=7, **CONFIG)
            expected = frozen.select_move(game)
            rt = RT30Agent(seed=7, **CONFIG)
            progress = []
            move = rt.select_move(game, progress=lambda stage, m: progress.append(stage))
            self.assertEqual(move, expected, name)
            self.assertEqual(rt.diagnostics.v8_route, route, name)
            same = {k: v for k, v in vars(rt.diagnostics).items() if not k.endswith('_seconds')}
            want = {k: v for k, v in vars(frozen.diagnostics).items() if not k.endswith('_seconds')}
            self.assertEqual(same, want, name)
            self.assertEqual((rt.rt_info.timeouts, rt.rt_info.skipped), ([], []))
            self.assertEqual(progress[0], 'emergency')
            self.assertEqual(rt.rt_info.best[-1][1], tuple(move))

    def test_same_random_stream_over_several_moves(self):
        game = recorded_position(*POSITIONS['tree'][0])
        frozen, rt = MCTSV8Agent(seed=3, **CONFIG), RT30Agent(seed=3, **CONFIG)
        self.assertEqual([frozen.select_move(game) for _ in range(2)], [rt.select_move(game) for _ in range(2)])


class DeadlineTest(_Engine):
    def run_rt(self, name, at, step=0.001):
        game = recorded_position(*POSITIONS[name][0])
        clock = TickClock(step)
        agent = RT30Agent(seed=7, **CONFIG)
        move = agent.select_move(game, deadline=Deadline(at, clock))
        self.assertIn(move, game.legal_moves())
        return agent, move, clock

    def test_real_clock_bounds_a_slow_attack_search(self):
        game = recorded_position(*SLOW_ATTACK)
        agent = RT30Agent(seed=7, **CONFIG)
        deadline = Deadline.after(5.0)
        move = agent.select_move(game, deadline=deadline)
        self.assertIn(move, game.legal_moves())
        self.assertIn('attack', agent.rt_info.timeouts)
        # V8-B stops at min(4 s, deadline - margin); the frozen V7 modules after it have no deadline
        # (docs/web-ser5.md §3.2), so allow them a few seconds here.
        self.assertLess(-deadline.remaining(), 8.0)
        self.assertLess(agent.rt_info.seconds, 13.0)

    def test_tiny_deadline_skips_the_tree_and_plays_the_first_root_candidate(self):
        agent, move, _ = self.run_rt('tree', at=0.5)
        self.assertIn('puct', agent.rt_info.skipped)
        self.assertEqual(move, agent.diagnostics.root_candidates[0])
        self.assertEqual(agent.rt_info.best[-1], ('root_order', move))

    def test_attack_cut_is_never_played_as_a_win(self):
        # The V8-B win needs a finished proof: with the search cut it is not found, and the move
        # comes from a later route.
        agent, move, _ = self.run_rt('own_vct', at=0.2)
        self.assertIn('attack', agent.rt_info.timeouts)
        self.assertNotEqual(agent.diagnostics.v8_route, 'own_vct')
        self.assertEqual(agent.diagnostics.v8_attack_status, '')

    def test_stage_route_falls_back_by_the_frozen_rule(self):
        agent, move, _ = self.run_rt('stage4', at=1.0)
        diag = agent.diagnostics
        self.assertEqual(diag.v8_route, 'stage4')
        statuses = dict(diag.v8_vct_checked)
        if statuses.get(move) == 'UNSAFE':
            # Only when nothing unrefuted is left (the frozen rule keeps V7's move then).
            self.assertEqual(move, diag.v8_v7_move)

    def test_emergency_move_comes_first_and_is_legal(self):
        agent, move, _ = self.run_rt('stage4_safe', at=0.01)
        self.assertEqual(agent.rt_info.best[0][0], 'emergency')


class HelpersTest(unittest.TestCase):
    def test_limited(self):
        clock = TickClock(0.0)
        d = Deadline(10.0, clock)
        self.assertEqual(limited(d, reserve=2.0).at, 8.0)
        self.assertEqual(limited(d, reserve=2.0, cap=3.0).at, 3.0)
        self.assertIsNone(limited(None, reserve=2.0))

    def test_rejects_configurations_outside_the_tb1_flow(self):
        with self.assertRaises(ValueError):
            check_config({**TB_ENGINE_V1, 'tree_mode': 'v5'})
        with self.assertRaises(ValueError):
            RT30Agent(**{**TB_ENGINE_V1, 'puct_prior': 'uniform', 'root_policy_extra': 2})

    def test_timing_defaults(self):
        self.assertLess(RT30_TIMING['final_margin'] + RT30_TIMING['puct_reserve'], 27.5)


if __name__ == '__main__':
    unittest.main()
