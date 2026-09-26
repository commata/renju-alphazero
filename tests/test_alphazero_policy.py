import math
import random
import unittest

from renju import Game
from search.alphazero import (SearchConfig, SearchResult, SearchTiming, argmax_action,
                              run_search, sample_action, sample_dirichlet, select_action,
                              visit_policy)
from search.evaluator import ScriptedEvaluator, UniformEvaluator


def opened_game():
    game = Game()
    for move in ((7, 7), (7, 8), (8, 8)):
        game.play(*move)
    return game


def legal_actions(game):
    return sorted(r * 15 + c for r, c in game.legal_moves())


def result_with(counts: dict, priors: dict) -> SearchResult:
    visit_counts = [0] * 225
    used = [0.0] * 225
    for action, count in counts.items():
        visit_counts[action] = count
    for action, prior in priors.items():
        used[action] = prior
    return SearchResult(tuple(visit_counts), tuple(used), 1, False, 1, SearchTiming())


class FixedRandom:
    def __init__(self, value=0.0, gamma=1.0):
        self.value = value
        self.gamma = gamma
        self.random_calls = 0

    def random(self):
        self.random_calls += 1
        return self.value

    def gammavariate(self, alpha, beta):
        return self.gamma


class VisitPolicyTest(unittest.TestCase):
    def test_visits_to_pi(self):
        game = opened_game()
        result = run_search(game, UniformEvaluator(), SearchConfig(num_simulations=9,
                                                                   noise_enabled=False))
        pi = visit_policy(result.visit_counts)
        legal = set(legal_actions(game))
        self.assertTrue(all(isinstance(n, int) for n in result.visit_counts))
        self.assertEqual(sum(result.visit_counts), 9)
        self.assertAlmostEqual(sum(pi), 1.0, delta=1e-12)
        self.assertEqual(sum(p for a, p in enumerate(pi) if a not in legal), 0)
        self.assertTrue(all(result.visit_counts[a] == 0 for a in range(225) if a not in legal))
        for a in range(225):
            self.assertEqual(pi[a], result.visit_counts[a] / 9)

    def test_empty_counts_rejected(self):
        with self.assertRaises(ValueError):
            visit_policy((0,) * 225)


class TemperatureTest(unittest.TestCase):
    def setUp(self):
        self.config = SearchConfig(num_simulations=8, temperature_moves=4)
        self.result = result_with({10: 5, 20: 3}, {10: 0.4, 20: 0.6})

    def test_temperature_does_not_alter_stored_target(self):
        counts = self.result.visit_counts
        pi = visit_policy(counts)
        rng = random.Random(0)
        for ply in (0, 1, 3, 4, 50):
            select_action(self.result, ply, self.config, rng)
        self.assertEqual(self.result.visit_counts, counts)
        self.assertEqual(visit_policy(self.result.visit_counts), pi)

    def test_ply_window(self):
        rng = FixedRandom(0.99)
        self.assertEqual(select_action(self.result, 3, self.config, rng), 20)  # sampled
        self.assertEqual(rng.random_calls, 1)
        self.assertEqual(select_action(self.result, 4, self.config, rng), 10)  # argmax
        self.assertEqual(rng.random_calls, 1)
        zero = SearchConfig(num_simulations=8, temperature_moves=0)
        self.assertEqual(select_action(self.result, 0, zero, None), 10)

    def test_sampling_cumulative_ascending(self):
        counts = [0] * 225
        counts[30], counts[40] = 3, 1
        # tau=1: weights 1 and 1/3 (normalized by the peak), total 4/3.
        self.assertEqual(sample_action(counts, 1.0, FixedRandom(0.74)), 30)
        self.assertEqual(sample_action(counts, 1.0, FixedRandom(0.76)), 40)
        self.assertEqual(sample_action(counts, 1.0, FixedRandom(0.0)), 30)
        # tau -> 0 approaches argmax; large tau approaches uniform over visited actions.
        self.assertEqual(sample_action(counts, 0.01, FixedRandom(0.999999)), 30)
        self.assertEqual(sample_action(counts, 1000.0, FixedRandom(0.51)), 40)

    def test_fast_path_uses_no_rng(self):
        result = run_search(Game(), UniformEvaluator(), self.config, random.Random(0))
        rng = FixedRandom()
        self.assertEqual(select_action(result, 0, self.config, rng), 112)
        self.assertEqual(rng.random_calls, 0)


class ArgmaxTieBreakTest(unittest.TestCase):
    def test_visit_count_first(self):
        self.assertEqual(argmax_action(result_with({5: 4, 6: 3}, {5: 0.1, 6: 0.9})), 5)

    def test_used_prior_second(self):
        self.assertEqual(argmax_action(result_with({5: 4, 6: 4}, {5: 0.1, 6: 0.9})), 6)

    def test_action_index_third(self):
        self.assertEqual(argmax_action(result_with({9: 4, 6: 4}, {9: 0.5, 6: 0.5})), 6)

    def test_uses_noised_prior(self):
        game = opened_game()
        config = SearchConfig(num_simulations=1, temperature_moves=0)
        result = run_search(game, UniformEvaluator(), config, random.Random(3))
        uniform = 1 / len(game.legal_moves())
        self.assertTrue(any(abs(p - uniform) > 1e-9 for p in result.priors if p > 0))
        visited = [a for a, n in enumerate(result.visit_counts) if n]
        self.assertEqual(argmax_action(result), visited[0])


class DirichletTest(unittest.TestCase):
    def test_noise_only_on_legal_children_exact_formula(self):
        game = opened_game()
        legal = legal_actions(game)
        config = SearchConfig(num_simulations=4, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
        rng = random.Random(21)
        result = run_search(game, UniformEvaluator(), config, rng)
        expected_rng = random.Random(21)
        eta = sample_dirichlet(expected_rng, 0.3, len(legal))
        self.assertEqual(rng.getstate(), expected_rng.getstate())  # exactly len(legal) draws
        for action, noise in zip(legal, eta):
            self.assertAlmostEqual(result.priors[action],
                                   0.75 * (1 / len(legal)) + 0.25 * noise, delta=1e-12)
        legal_set = set(legal)
        self.assertTrue(all(p == 0 for a, p in enumerate(result.priors) if a not in legal_set))
        self.assertAlmostEqual(sum(result.priors), 1.0, delta=1e-9)

    def test_noise_off_consumes_no_rng(self):
        rng = random.Random(4)
        before = rng.getstate()
        result = run_search(opened_game(), ScriptedEvaluator(),
                            SearchConfig(num_simulations=3, noise_enabled=False), rng)
        self.assertEqual(rng.getstate(), before)
        self.assertEqual(sum(result.visit_counts), 3)

    def test_gamma_total_guarded(self):
        for gamma in (0.0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                sample_dirichlet(FixedRandom(gamma=gamma), 0.05, 5)
            with self.assertRaises(ValueError):
                run_search(opened_game(), UniformEvaluator(), SearchConfig(num_simulations=2),
                           FixedRandom(gamma=gamma))

    def test_noise_requires_rng(self):
        with self.assertRaises(ValueError):
            run_search(opened_game(), UniformEvaluator(), SearchConfig(num_simulations=2))

    def test_same_seed_reproducible(self):
        config = SearchConfig(num_simulations=6, temperature_moves=10)
        runs = []
        for _ in range(2):
            rng = random.Random(99)
            result = run_search(opened_game(), UniformEvaluator(), config, rng)
            runs.append((result.visit_counts, result.priors, select_action(result, 3, config, rng)))
        self.assertEqual(runs[0], runs[1])

    def test_module_random_state_unchanged(self):
        random.seed(1234)
        before = random.getstate()
        config = SearchConfig(num_simulations=6)
        rng = random.Random(8)
        for game in (Game(), opened_game()):
            result = run_search(game, UniformEvaluator(), config, rng)
            select_action(result, len(game.history), config, rng)
        self.assertEqual(random.getstate(), before)


if __name__ == '__main__':
    unittest.main()
