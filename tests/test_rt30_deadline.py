"""rt30 deadline: a cut is UNKNOWN, never a proof; state and caches stay clean (docs/web-ser5.md §3.2)."""
from copy import deepcopy
import unittest

from analysis.mcts_v8 import WIN, _BudgetedSolver, _first_proven
from analysis.selective_vct import SelectiveSolver
from analysis.threats import SAFE, UNKNOWN, UNSAFE, ordered_moves
from rt30 import fast_windows
from rt30.deadline import CheckLimit, Deadline, SearchTimeout
from rt30.solver import DeadlineSelectiveSolver, DeadlineSolver
from rt30.vcf import find_vcf_with_stats
from search.mcts_v7 import _find_vcf_with_stats

from test_rt30_fast_windows import recorded_position

V8C = {'node_limit': 20_000, 'call_limit': 10_000, 'node_budget': 400_000}
V8B = {'node_limit': 20_000, 'call_limit': 10_000, 'node_budget': 200_000}
CASES = [  # (seed, line, ply, move, budget, check): recorded E3-A positions with small searches
    (8415, 0, 13, (10, 8), V8C, 'status'),    # V8-C SAFE
    (8416, 18, 38, (5, 6), V8C, 'status'),    # V8-C UNSAFE
    (8415, 11, 15, None, V8B, 'attack'),      # V8-B WIN (the recorded rank-1 candidate)
]


def snapshot(game):
    return deepcopy(vars(game))


def evaluate(solver, game, move, check):
    if check == 'status':
        return solver.status_after(game, move)
    return solver.attack_status(game, move)


def attack_move(game):
    from analysis.mcts_v8 import SearchDiagnostics, _attack_candidates
    from search.mcts_v5 import _RootContext
    return _attack_candidates(game, _RootContext(game.legal_moves(), SearchDiagnostics()))[0]


def run(cls, game, move, budget, check, deadline=None):
    kwargs = {} if deadline is None and cls in (_BudgetedSolver, SelectiveSolver) else {'deadline': deadline}
    solver = cls(**budget, **kwargs)
    return evaluate(solver, game, move, check), solver


def counting_limit(cls, game, move, budget, check) -> int:
    """How many deadline checks the full search makes."""
    limit = CheckLimit(10 ** 9)
    run(cls, game, move, budget, check, limit)
    return limit.checks


def cut_points(total: int, count: int = 12) -> list[int]:
    points = {0, 1, total - 1, total}
    step = 1.0
    while step < total:
        points.add(int(step))
        step *= total ** (1.0 / count)
    return sorted(p for p in points if 0 <= p <= total)


class _Cases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fast_windows.install()  # exact (test_rt30_fast_windows); only makes these tests faster
        cls.cases = []
        for seed, line, ply, move, budget, check in CASES:
            game = recorded_position(seed, line, ply)
            cls.cases.append((game, move or attack_move(game), budget, check))

    @classmethod
    def tearDownClass(cls):
        fast_windows.uninstall()


class NoDeadlineIsFrozenTest(_Cases):
    def test_same_status_nodes_calls_and_caches(self):
        expected_status = {0: SAFE, 1: UNSAFE, 2: WIN}
        for index, (game, move, budget, check) in enumerate(self.cases):
            frozen_status, frozen = run(_BudgetedSolver, game, move, budget, check)
            self.assertEqual(frozen_status, expected_status[index])
            for deadline in (None, Deadline.after(3600.0)):
                status, solver = run(DeadlineSolver, game, move, budget, check, deadline)
                self.assertEqual(status, frozen_status)
                self.assertEqual((solver.nodes_used, solver.vcf_calls, solver.exhausted, solver.vcf_exhausted),
                                 (frozen.nodes_used, frozen.vcf_calls, frozen.exhausted, frozen.vcf_exhausted))
                self.assertEqual(solver._vcf_cache, frozen._vcf_cache)
                self.assertEqual(solver._after_cache, frozen._after_cache)
                self.assertFalse(solver.timed_out)

    def test_selective_solver_matches(self):
        game, move, budget, _ = self.cases[1]
        budget = {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': 50_000}
        check = lambda s: s._bounded(game, move, None, None, lambda g: s.after_move(g, 2)[0])  # noqa: E731
        frozen, rt = SelectiveSolver(**budget), DeadlineSelectiveSolver(**budget)
        self.assertEqual(check(rt), check(frozen))
        self.assertEqual((rt.nodes_used, rt.vcf_calls), (frozen.nodes_used, frozen.vcf_calls))
        self.assertEqual(rt.quiet_moves(game), frozen.quiet_moves(game))


class CutIsNeverAProofTest(_Cases):
    """Cut every search at many points: UNKNOWN or the full result; caches are a subset of the
    full run's; the game is unchanged; the solver stays stopped."""

    def test_cuts(self):
        for game, move, budget, check in self.cases:
            before = snapshot(game)
            full_status, full = run(DeadlineSolver, game, move, budget, check)
            total = counting_limit(DeadlineSolver, game, move, budget, check)
            self.assertGreater(total, 10)
            for k in cut_points(total):
                status, solver = run(DeadlineSolver, game, move, budget, check, CheckLimit(k))
                self.assertEqual(snapshot(game), before, f'state changed at cut {k}')
                if k < total:
                    self.assertEqual(status, UNKNOWN, f'cut {k}/{total}')
                    self.assertTrue(solver.timed_out and solver.exhausted)
                    self.assertTrue(solver.last_cut)
                    # A stopped solver never resumes: a later call is UNKNOWN at once.
                    self.assertEqual(evaluate(solver, game, move, check), UNKNOWN)
                else:
                    self.assertEqual(status, full_status)
                for key, value in solver._vcf_cache.items():
                    self.assertEqual(full._vcf_cache.get(key), value, f'cut {k}: VCF cache entry not in the full run')
                for key, value in solver._after_cache.items():
                    self.assertEqual(full._after_cache.get(key), value, f'cut {k}: status cache entry not in the full run')

    def test_expired_deadline_is_unknown_without_search(self):
        game, move, budget, check = self.cases[0]
        status, solver = run(DeadlineSolver, game, move, budget, check, Deadline(at=0.0))
        self.assertEqual(status, UNKNOWN)
        self.assertEqual((solver.nodes_used, solver.vcf_calls), (0, 0))

    def test_fair_scheduler_stops_at_the_deadline(self):
        # ``_proven`` loops while the solver is not exhausted; a timeout must end it.
        game, move, budget, _ = self.cases[1]
        moves = game.legal_moves()[:6]
        for k in (0, 50, 500):
            solver = DeadlineSolver(**budget, deadline=CheckLimit(k))
            statuses = {}
            before = snapshot(game)
            found = _first_proven(game, moves, statuses, solver, solver.status_after, SAFE)
            self.assertEqual(snapshot(game), before)
            self.assertTrue(solver.timed_out)
            full = DeadlineSolver(**budget)
            for m, status in statuses.items():
                if status != UNKNOWN:
                    self.assertEqual(status, full.status_after(game, m))
            if found is not None:
                self.assertEqual(statuses[found], SAFE)


class VCFDeadlineTest(_Cases):
    def vcf_case(self):
        """After the V8-B attack and a defender reply: the attacker's VCF search (WIN)."""
        game, move, _, _ = self.cases[2]
        attacker = game.to_play
        game.play(*move)
        for reply in ordered_moves(game):
            game.play(*reply)
            expected = _find_vcf_with_stats(game, attacker, max_fours=10, node_limit=5000)
            if expected[0] is not None and expected[1] > 3:
                return game, attacker, expected
            game.undo()
        self.fail('no VCF case')

    def test_vcf_cut_raises_and_restores_the_board(self):
        game, attacker, expected = self.vcf_case()
        try:
            self.assertEqual(find_vcf_with_stats(game, attacker, max_fours=10, node_limit=5000), expected)
            nodes = expected[1]
            before = snapshot(game)
            for k in range(0, nodes + 2):
                try:
                    result = find_vcf_with_stats(game, attacker, max_fours=10, node_limit=5000,
                                                 deadline=CheckLimit(k))
                except SearchTimeout:
                    self.assertLess(k, nodes)
                else:
                    self.assertGreaterEqual(k, nodes)
                    self.assertEqual(result, expected)
                self.assertEqual(snapshot(game), before, f'board changed at cut {k}')
        finally:
            game.undo()
            game.undo()

    def test_bad_arguments_raise_like_the_frozen_function(self):
        game, _, _, _ = self.cases[0]
        with self.assertRaises(ValueError):
            find_vcf_with_stats(game, 1, max_fours=0, node_limit=10, deadline=CheckLimit(5))


if __name__ == '__main__':
    unittest.main()
