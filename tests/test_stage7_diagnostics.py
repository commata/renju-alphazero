"""Stage 7 diagnostics: value-target sign at tensor level, FPU option, search probes."""
import unittest

from renju import BLACK, WHITE
from search.alphazero import SearchConfig

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from pathlib import Path

    from model.config import ModelConfig
    from model.evaluator import PolicyValueEvaluator
    from model.network import PolicyValueNet
    from search.alphazero import run_search
    from training.dataset import samples_from_record
    from training.probes import evaluate_search_probes, load_probe_set
    from training.self_play import play_self_play_game

    PROBES = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'stage7_probes_v1.json'


class FpuConfigTest(unittest.TestCase):
    def test_default_keeps_stage5_hash_fields(self):
        self.assertNotIn('fpu_reduction', SearchConfig().to_dict())
        self.assertEqual(SearchConfig(fpu_reduction=0.25).to_dict()['fpu_reduction'], 0.25)
        with self.assertRaises(ValueError):
            SearchConfig(fpu_reduction=-0.1)
        round_trip = SearchConfig.from_dict(SearchConfig(fpu_reduction=0.5).to_dict())
        self.assertEqual(round_trip.fpu_reduction, 0.5)


@unittest.skipIf(torch is None, 'requires torch')
class ValueTargetSignTest(unittest.TestCase):
    def test_z_alternates_and_matches_side_to_move_plane(self):
        torch.manual_seed(0)
        evaluator = PolicyValueEvaluator(
            PolicyValueNet(ModelConfig(channels=8, blocks=1, value_hidden=8)).eval())
        config = SearchConfig(num_simulations=4)
        winners = set()
        for seed in range(40):
            record = play_self_play_game(evaluator, config, seed).record
            if record.winner is None or record.winner in winners:
                continue
            winners.add(record.winner)
            samples = samples_from_record(record, generation=0, game_id=seed)
            for index, sample in enumerate(samples):
                # plane 3 = current_is_black: the tensor's side to move
                mover = BLACK if float(sample.state[3].max()) == 1.0 else WHITE
                self.assertEqual(mover, BLACK if index % 2 == 0 else WHITE)
                self.assertEqual(sample.value, 1.0 if mover == record.winner else -1.0)
            self.assertEqual(samples[-1].value, 1.0)  # the winning move's state
            if winners == {BLACK, WHITE}:
                return
        self.fail(f'expected games won by both colours, got {winners}')


@unittest.skipIf(torch is None, 'requires torch')
class FpuSearchAndSearchProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1)
        cls.model = PolicyValueNet(ModelConfig(channels=8, blocks=1, value_hidden=8)).eval()
        cls.probes, _ = load_probe_set(PROBES)

    def _game(self, kind):
        from renju import Game
        probe = next(p for p in self.probes if p.kind == kind)
        game = Game()
        for move in probe.moves:
            game.play(*move)
        return game

    def test_fpu_changes_search_but_keeps_visit_invariants(self):
        game = self._game('must_block')
        evaluator = PolicyValueEvaluator(self.model)
        base = SearchConfig(num_simulations=32, temperature_moves=0, noise_enabled=False)
        strict = SearchConfig(num_simulations=32, temperature_moves=0, noise_enabled=False,
                              fpu_reduction=1.0)
        a = run_search(game, evaluator, base, None)
        b = run_search(game, evaluator, strict, None)
        self.assertEqual(sum(a.visit_counts), 32)
        self.assertEqual(sum(b.visit_counts), 32)
        # A large reduction makes unvisited children unattractive: fewer children visited.
        self.assertLess(sum(1 for c in b.visit_counts if c),
                        sum(1 for c in a.visit_counts if c))

    def test_search_probe_summary(self):
        config = SearchConfig(num_simulations=4, temperature_moves=0, noise_enabled=False)
        subset = [p for p in self.probes if p.kind in ('immediate_win', 'must_block')][:6]
        result = evaluate_search_probes(self.model, subset, config,
                                        ('immediate_win', 'must_block'))
        self.assertEqual(sum(s['probes'] for s in result['summary'].values()), len(subset))
        for summary in result['summary'].values():
            self.assertTrue(0.0 <= summary['solved'] <= 1.0)
            self.assertTrue(0.0 <= summary['visit_share'] <= 1.0)
        with self.assertRaises(ValueError):
            evaluate_search_probes(self.model, subset, SearchConfig(num_simulations=4))


if __name__ == '__main__':
    unittest.main()
