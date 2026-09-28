"""Stage 7 profiling / batch benchmark: measurement must not change search results."""
import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    import search.alphazero as alphazero
    from model.config import ModelConfig
    from model.evaluator import PolicyValueEvaluator
    from model.network import PolicyValueNet
    from renju import Game
    from search.alphazero import SearchConfig, run_search
    from search.evaluator import EvaluationSnapshot
    from scripts.benchmark_stage7_batch import benchmark
    from scripts.profile_stage7_search import load_positions
    from training.profiling import ProfiledEvaluator, instrument_search, profile_searches


@unittest.skipIf(torch is None, 'requires torch')
class ProfilingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(5)
        cls.model = PolicyValueNet(ModelConfig(channels=8, blocks=1, value_hidden=8)).eval()
        cls.games = load_positions(6)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_profiled_evaluator_outputs_are_identical(self):
        plain = PolicyValueEvaluator(self.model)
        profiled = ProfiledEvaluator(self.model)
        snapshots = [EvaluationSnapshot.from_game(g, g.legal_moves()) for g in self.games]
        self.assertEqual(plain.evaluate_batch(snapshots), profiled.evaluate_batch(snapshots))
        self.assertEqual(profiled.calls, len(snapshots))
        self.assertGreater(profiled.seconds['forward'], 0)

    def test_instrumented_search_gives_identical_visits_and_restores_patches(self):
        originals = (alphazero._legal_moves, alphazero.tactical_filter, Game.play, Game.undo)
        for rules in (False, True):
            config = SearchConfig(num_simulations=12, temperature_moves=0, noise_enabled=False,
                                  tactical_rules=rules)
            expected = [run_search(g, PolicyValueEvaluator(self.model), config, None).visit_counts
                        for g in self.games]
            with instrument_search() as (seconds, counts):
                actual = [run_search(g, ProfiledEvaluator(self.model), config, None).visit_counts
                          for g in self.games]
            self.assertEqual(actual, expected, rules)
            self.assertGreater(counts['play'], 0)
            self.assertEqual(counts['rule_filter'] > 0, rules)
        self.assertEqual((alphazero._legal_moves, alphazero.tactical_filter, Game.play,
                          Game.undo), originals)

    def test_breakdown_adds_up_to_total(self):
        config = SearchConfig(num_simulations=12, temperature_moves=0, noise_enabled=False,
                              tactical_rules=True)
        result = profile_searches(self.model, self.games, config)
        parts = sum(result['per_move_ms'].values())
        # wrapper overhead is attributed to legal_moves / tree_other; allow a small margin
        self.assertAlmostEqual(parts / result['total_ms_per_move'], 1.0, delta=0.1)
        self.assertEqual(result['searched_positions'], len(self.games))
        self.assertGreater(result['evaluator_calls_per_move'], 1)

    def test_batch_benchmark_rows_and_numerics(self):
        snapshots = [EvaluationSnapshot.from_game(g, g.legal_moves()) for g in load_positions(8)]
        rows = benchmark(self.model, snapshots, [1, 2, 4], repeats=1, warmup=0)
        self.assertEqual([r['batch'] for r in rows], [1, 2, 4])
        self.assertEqual(rows[0]['max_abs_prior_diff_vs_b1'], 0.0)
        for row in rows:
            self.assertEqual(row['positions'], 8)
            self.assertLess(row['max_abs_prior_diff_vs_b1'], 1e-4)
            self.assertLess(row['max_abs_value_diff_vs_b1'], 1e-4)
            self.assertGreater(row['full_positions_per_second'], 0)


if __name__ == '__main__':
    unittest.main()
