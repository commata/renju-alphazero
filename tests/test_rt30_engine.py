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


class AttackVariantsTest(_Engine):
    """Opt-in V8-B order / scheduler (docs/web-ser5.md §3.8): proofs stay proofs; defaults are V8's."""

    def attack(self, key, **kwargs):
        from analysis.mcts_v8 import SearchDiagnostics
        from rt30.engine import _own_vct_attack
        from search.mcts_v5 import _RootContext

        game = recorded_position(*key)
        diag = SearchDiagnostics()
        move, _ = _own_vct_attack(game, _RootContext(game.legal_moves(), diag), diag, node_limit=20_000,
                                  call_limit=10_000, node_budget=200_000, deadline=None, **kwargs)
        return game, move, diag

    def test_threes_first_keeps_the_candidates_and_puts_open_threes_first(self):
        from analysis.mcts_v8 import SearchDiagnostics, _attack_candidates
        from rt30.engine import attack_order
        from search.mcts_v321 import _fast_pattern_features_for_move
        from search.mcts_v5 import _RootContext

        game = recorded_position(8415, 9, 26)
        frozen = _attack_candidates(game, _RootContext(game.legal_moves(), SearchDiagnostics()))
        ordered = attack_order(game, frozen, 'threes_first')
        self.assertEqual(sorted(ordered), sorted(frozen))
        threes = [_fast_pattern_features_for_move(game, game.to_play, m, assume_legal=True).open_three_directions
                  for m in ordered]
        self.assertEqual(threes, sorted(threes, reverse=True))
        self.assertEqual(attack_order(game, frozen, 'frozen'), frozen)

    def test_every_variant_finds_a_proven_win_where_v8_does(self):
        from analysis.mcts_v8 import WIN, _BudgetedSolver

        key = POSITIONS['own_vct'][0]
        _, expected, _ = self.attack(key)
        self.assertIsNotNone(expected)
        for order in ('frozen', 'threes_first'):
            for scheduler in ('fair', 'widen'):
                game, move, diag = self.attack(key, order=order, scheduler=scheduler)
                self.assertIsNotNone(move, (order, scheduler))
                self.assertEqual(diag.v8_attack_status, WIN)
                # Re-prove the move with a fresh frozen solver: a variant can only find proofs.
                solver = _BudgetedSolver(node_limit=20_000, call_limit=10_000, node_budget=200_000)
                self.assertEqual(solver.attack_status(game, move), WIN)

    def test_default_is_v8(self):
        _, move, diag = self.attack(POSITIONS['own_vct'][0])
        _, move2, diag2 = self.attack(POSITIONS['own_vct'][0], order='frozen', scheduler='fair')
        self.assertEqual((move, diag.v8_attack_checked, diag.v8_attack_nodes),
                         (move2, diag2.v8_attack_checked, diag2.v8_attack_nodes))

    def test_unknown_names_rejected(self):
        with self.assertRaises(ValueError):
            RT30Agent(**CONFIG, attack_order='nope')
        with self.assertRaises(ValueError):
            self.attack(POSITIONS['own_vct'][0], scheduler='nope')


class _FakeSolver:
    """Scripted V8-C statuses with a deadline: one check per status_after; at the deadline the
    solver stops like DeadlineSolver (UNKNOWN, cut, exhausted, timed out)."""

    script: dict = {}

    def __init__(self, *, node_limit, call_limit, node_budget, deadline):
        self.node_limit, self.call_limit, self.node_budget, self.deadline = node_limit, call_limit, node_budget, deadline
        self.nodes_used = self.vcf_calls = 0
        self.exhausted = self.timed_out = self.last_cut = False

    def status_after(self, game, move, share=None, call_share=None):
        self.last_cut = True
        if self.exhausted:
            return 'UNKNOWN'
        if self.deadline is not None and self.deadline.expired():
            self.timed_out = self.exhausted = True
            return 'UNKNOWN'
        self.nodes_used += 1
        self.vcf_calls += 1
        self.last_cut = False
        return self.script[tuple(move)]


class RootRescueTest(unittest.TestCase):
    """V8-C root rescue (docs/web-ser5.md §3.6.2) on scripted statuses."""

    def setUp(self):
        from collections import namedtuple

        import rt30.engine as engine

        self.engine = engine
        original = engine.DeadlineSolver
        engine.DeadlineSolver = _FakeSolver
        self.addCleanup(setattr, engine, 'DeadlineSolver', original)
        original_lost = engine._not_immediately_lost
        engine._not_immediately_lost = lambda game, move: True
        self.addCleanup(setattr, engine, '_not_immediately_lost', original_lost)
        Child = namedtuple('Child', 'move visits mean_value')
        # Tree order: (0,0) chosen, then (1,1) (2,2) (3,3) (4,4) by visits.
        self.children = [Child((i, i), 10 - i, 0.0) for i in range(5)]

    def verify(self, script, deadline, rescue=None):
        from analysis.mcts_v8 import SearchDiagnostics
        from rt30.deadline import CheckLimit

        _FakeSolver.script = script
        diag = SearchDiagnostics()
        move, cut, rescued = self.engine._verify_root_choice(
            None, (0, 0), self.children, diag, node_limit=10, call_limit=1000, node_budget=1000, max_children=4,
            mode='aggressive', deadline=CheckLimit(deadline), rescue_deadline=None if rescue is None else CheckLimit(rescue))
        return move, cut, rescued, dict(diag.v8_root_checked)

    def test_bug_path_without_rescue_and_the_fix(self):
        # The tree move is proven lost, the next child is lost too, the third is the safe one.
        script = {(0, 0): 'UNSAFE', (1, 1): 'UNSAFE', (2, 2): 'SAFE', (3, 3): 'UNSAFE', (4, 4): 'UNSAFE'}
        # Deadline right after the tree move's proof: the old rule plays the first unchecked child.
        move, cut, rescued, statuses = self.verify(script, deadline=1)
        self.assertEqual((move, cut, rescued), ((1, 1), True, False))
        self.assertEqual(statuses.get((1, 1)), 'UNKNOWN')  # unchecked, though the script says it loses
        # With the rescue the child search goes on and finds the proven SAFE child.
        move, cut, rescued, statuses = self.verify(script, deadline=1, rescue=10 ** 6)
        self.assertEqual((move, rescued), ((2, 2), True))
        self.assertEqual(statuses[(1, 1)], 'UNSAFE')
        self.assertEqual(statuses[(2, 2)], 'SAFE')

    def test_no_rescue_when_the_tree_move_is_not_proven_lost(self):
        for status in ('SAFE', 'UNKNOWN'):
            script = {(i, i): 'SAFE' for i in range(5)}
            script[(0, 0)] = status
            _, _, rescued, _ = self.verify(script, deadline=1, rescue=10 ** 6)
            self.assertFalse(rescued, status)

    def test_never_plays_its_own_proven_loss_while_an_unrefuted_child_exists(self):
        from itertools import product

        for statuses in product(('UNSAFE', 'SAFE', 'UNKNOWN'), repeat=4):
            script = {(0, 0): 'UNSAFE', **{(i + 1, i + 1): s for i, s in enumerate(statuses)}}
            for deadline, rescue in ((1, None), (2, None), (3, None), (1, 2), (1, 10 ** 6), (10 ** 6, None)):
                move, _, _, seen = self.verify(script, deadline, rescue)
                if seen.get(move) == 'UNSAFE':
                    # Only when every child the search could choose is proven lost.
                    others = [m for m in [(i, i) for i in range(1, 5)] if seen.get(m, 'UNKNOWN') != 'UNSAFE']
                    self.assertEqual(others, [], (script, deadline, rescue, move, seen))

    def test_engine_gives_vct2_time_to_the_rescue(self):
        # rt30_search: when V8-C reports a rescue, VCT2 is skipped and the move info says so.
        engine = self.engine
        original = engine._verify_root_choice

        def rescued(game, chosen, children, diag, **kwargs):
            self.assertIsNotNone(kwargs['rescue_deadline'])
            move, cut, _ = original(game, chosen, children, diag, **{**kwargs, 'rescue_deadline': None})
            return move, cut, True

        engine._verify_root_choice = rescued
        self.addCleanup(setattr, engine, '_verify_root_choice', original)
        engine.DeadlineSolver = TestRealSolver.real()
        _FakeSolver.script = {}
        game = recorded_position(*POSITIONS['tree'][0])
        agent = RT30Agent(seed=7, **CONFIG)
        agent.select_move(game, deadline=Deadline.after(600.0))
        self.assertTrue(agent.rt_info.rescued)
        self.assertIn('vct2_root_rescue', agent.rt_info.skipped)
        self.assertEqual(agent.diagnostics.v8_vct2_checked, ())

    def test_rescue_off_and_no_deadline_keep_the_old_rule(self):
        engine = self.engine
        calls = []
        original = engine._verify_root_choice

        def spy(*args, **kwargs):
            calls.append(kwargs['rescue_deadline'])
            return original(*args, **kwargs)

        engine._verify_root_choice = spy
        self.addCleanup(setattr, engine, '_verify_root_choice', original)
        engine.DeadlineSolver = TestRealSolver.real()
        game = recorded_position(*POSITIONS['tree'][0])
        RT30Agent(seed=7, **CONFIG).select_move(game)                                   # no deadline
        RT30Agent(seed=7, root_rescue=False, **CONFIG).select_move(game, deadline=Deadline.after(600.0))
        self.assertEqual(calls, [None, None])


class TestRealSolver:
    @staticmethod
    def real():
        from rt30.solver import DeadlineSolver

        return DeadlineSolver
