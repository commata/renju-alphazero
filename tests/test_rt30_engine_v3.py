"""rt30.engine_v3 (X2: V8-B in a helper process) and rt30.fast3 (X1-A native scan) (docs/web-ser5.md §9.8)."""
import unittest

from analysis.mcts_v8 import WIN, _BudgetedSolver
from analysis.mcts_v8_agent import MCTSV8Agent
from rt30 import attack_helper, fast, fast3, fast3_windows, fast_windows, tb_rt30_v1, tb_rt30_v3
from rt30.deadline import Deadline
from rt30.engine import RT30Agent
from rt30.engine_v3 import RT30AgentV3

from test_rt30_engine import CONFIG, POSITIONS, SLOW_ATTACK, TickClock
from test_rt30_fast_windows import _Board, random_positions, recorded_position


def without_seconds(diag):
    return {k: v for k, v in vars(diag).items() if not k.endswith('_seconds')}


class NativeWindowsTest(unittest.TestCase):
    """X1-A: the native scan is the frozen scan (list and order) on every input."""

    @classmethod
    def setUpClass(cls):
        if not fast3_windows.available():
            raise unittest.SkipTest('rt30.native not built (python -m rt30.native.build)')

    def tearDown(self):
        fast3.uninstall_all()

    def test_same_windows_in_the_same_order(self):
        for board in random_positions(3000, seed=20261016):
            game = _Board(board)
            for player in (1, -1):
                for minimum in (0, 1, 2, 3, 4, 5):
                    self.assertEqual(list(fast3_windows.threat_windows_native(game, player, minimum)),
                                     list(fast_windows.FROZEN(game, player, minimum)))

    def test_default_minimum_and_board_untouched(self):
        board = random_positions(50)[-1]
        copy = [row[:] for row in board]
        self.assertEqual(list(fast3_windows.threat_windows_native(_Board(board), -1)),
                         list(fast_windows.FROZEN(_Board(board), -1)))
        self.assertEqual(board, copy)

    def test_install_switches_from_fast2_and_back(self):
        from search import mcts_v5, mcts_v7

        fast.install_all()
        self.assertEqual(fast3.install_all(), ['threat_windows_native', 'winning_extensions_in_direction'])
        self.assertIs(mcts_v7._threat_windows, fast3_windows.threat_windows_native)
        self.assertEqual(fast3.active()[0], 'threat_windows_native')
        fast3.uninstall_all()
        self.assertIs(mcts_v5._threat_windows, fast_windows.FROZEN)
        self.assertIs(mcts_v7._threat_windows, fast_windows.FROZEN)

    def test_v8b_search_identical(self):
        from analysis.mcts_v8 import SearchDiagnostics, _own_vct_attack
        from search.mcts_v5 import _RootContext

        def run():
            game = recorded_position(*POSITIONS['own_vct'][0])
            diag = SearchDiagnostics()
            move = _own_vct_attack(game, _RootContext(game.legal_moves(), diag), diag,
                                   node_limit=20_000, call_limit=10_000, node_budget=200_000)
            return move, diag.v8_attack_checked, diag.v8_attack_nodes, diag.v8_attack_calls

        expected = run()
        fast3.install_all()
        self.assertEqual(run(), expected)


class _Engine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fast3.install_all()
        if attack_helper.get_helper() is None:
            raise unittest.SkipTest(f'no helper process here: {attack_helper.unavailable_reason()}')

    @classmethod
    def tearDownClass(cls):
        fast3.uninstall_all()


class NoDeadlineIsTB1Test(_Engine):
    def test_same_move_and_diagnostics_on_every_route(self):
        for name, (key, route) in POSITIONS.items():
            game = recorded_position(*key)
            frozen = MCTSV8Agent(seed=7, **CONFIG)
            expected = frozen.select_move(game)
            v3 = RT30AgentV3(seed=7, **CONFIG)
            self.assertEqual(v3.select_move(game), expected, name)
            self.assertEqual(without_seconds(v3.diagnostics), without_seconds(frozen.diagnostics), name)
            self.assertEqual(v3.rt_info.timeouts, [], name)
            self.assertTrue(v3.rt_info.attack_parallel, name)

    def test_random_stream_after_a_v8b_win_is_tb1_s(self):
        # The main search ran PUCT rollouts while V8-B proved the win; the next move must still be TB1's.
        sequence = [POSITIONS['own_vct'][0], POSITIONS['tree'][0], POSITIONS['own_vct'][0], POSITIONS['tree'][0]]
        frozen, v3 = MCTSV8Agent(seed=3, **CONFIG), RT30AgentV3(seed=3, **CONFIG)
        expected = [frozen.select_move(recorded_position(*k)) for k in sequence]
        self.assertEqual([v3.select_move(recorded_position(*k)) for k in sequence], expected)
        self.assertEqual(frozen._random.getstate(), v3._random.getstate())


class DeadlineTest(_Engine):
    def test_slow_v8b_gets_the_whole_move_beside_the_rest(self):
        # V8-B dominated (51 s frozen); v1 stops it at its cap, v3 lets it run to the deadline in the helper.
        game = recorded_position(*SLOW_ATTACK)
        v3 = RT30AgentV3(seed=7, **CONFIG)
        deadline = Deadline.after(12.0)
        move = v3.select_move(game, deadline=deadline)
        info = v3.rt_info
        self.assertIn(move, game.legal_moves())
        self.assertTrue(info.attack_parallel)
        self.assertLess(-deadline.remaining(), 1.0)  # done before the deadline (final margin 1.5 s)
        if v3.diagnostics.v8_route == 'own_vct':
            self.assertEqual(_BudgetedSolver(node_limit=20_000, call_limit=10_000,
                                             node_budget=200_000).attack_status(game, move), WIN)
        else:
            self.assertIn('attack', info.timeouts)

    def test_quick_win_ends_the_move_early(self):
        game = recorded_position(*POSITIONS['own_vct'][0])
        v3 = RT30AgentV3(seed=7, **CONFIG)
        move = v3.select_move(game, deadline=Deadline.after(27.5))
        frozen = MCTSV8Agent(seed=7, **CONFIG)
        self.assertEqual(move, frozen.select_move(game))
        self.assertEqual(v3.diagnostics.v8_route, 'own_vct')

    def test_test_clock_runs_v8b_inline_as_v1(self):
        game = recorded_position(*POSITIONS['own_vct'][0])
        v1, v3 = RT30Agent(seed=7, **CONFIG), RT30AgentV3(seed=7, **CONFIG)
        m1 = v1.select_move(game, deadline=Deadline(0.2, TickClock()))
        # A test clock is not the machine's monotonic clock: the helper cannot share it, V8-B runs inline.
        m3 = v3.select_move(game, deadline=Deadline(0.2, TickClock()))
        self.assertEqual(m1, m3)
        self.assertEqual(without_seconds(v1.diagnostics), without_seconds(v3.diagnostics))
        self.assertEqual(v3.rt_info.attack_helper_failed, 'test clock')

    def test_off_is_v1(self):
        game = recorded_position(*POSITIONS['tree'][0])
        v1, v3 = RT30Agent(seed=7, **CONFIG), RT30AgentV3(seed=7, parallel_attack=False, **CONFIG)
        self.assertEqual(v1.select_move(game, deadline=Deadline(0.5, TickClock())),
                         v3.select_move(game, deadline=Deadline(0.5, TickClock())))
        self.assertEqual(without_seconds(v1.diagnostics), without_seconds(v3.diagnostics))
        self.assertFalse(v3.rt_info.attack_parallel)


class ConfigTest(unittest.TestCase):
    def test_v3_is_v1_plus_x1_x2(self):
        v1, v3 = tb_rt30_v1.TB_RT30_V1, tb_rt30_v3.TB_RT30_V3
        self.assertEqual({k for k in v3 if v3[k] != v1.get(k)}, {'parallel_attack', 'exact_speedups'})
        self.assertEqual(v3['exact_speedups'][0], 'threat_windows_native')
        self.assertEqual(tb_rt30_v3.STATUS, 'candidate')


class MakeAgentTest(unittest.TestCase):
    """make_agent never returns an engine that would run as v1 under the v3 name."""

    def setUp(self):
        self._get_helper, self._reason = attack_helper.get_helper, attack_helper.unavailable_reason
        attack_helper.get_helper = lambda: None
        attack_helper.unavailable_reason = lambda: 'AssertionError: daemonic processes are not allowed to have children'

    def tearDown(self):
        attack_helper.get_helper, attack_helper.unavailable_reason = self._get_helper, self._reason
        fast3.uninstall_all()

    def test_refuses_without_the_helper(self):
        with self.assertRaisesRegex(RuntimeError, 'V8-B helper'):
            tb_rt30_v3.make_agent(None, seed=3, uniform_prior=True, allow_python=True)

    def test_inline_attack_only_when_asked(self):
        agent = tb_rt30_v3.make_agent(None, seed=3, uniform_prior=True, allow_python=True, allow_inline_attack=True)
        self.assertTrue(agent.name.startswith(tb_rt30_v3.NAME))


if __name__ == '__main__':
    unittest.main()
