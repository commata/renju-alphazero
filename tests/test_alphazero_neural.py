import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from renju import Game
from renju.rules import forbidden_reason
from search.alphazero import SearchConfig, run_search
from search.evaluator import EvaluationSnapshot, validate_evaluation

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from model.checkpoint import load_checkpoint
    from model.encoding import encode_game
    from model.evaluator import PolicyValueEvaluator, create_random_checkpoint, file_sha256
    from model.masking import legal_moves_to_mask
    from model.network import PolicyValueNet


def play(moves):
    game = Game()
    for move in moves:
        game.play(*move)
    return game


FORBIDDEN_MOVES = ((7, 7), (14, 14), (3, 4), (14, 12), (3, 5), (14, 10), (1, 6), (14, 8),
                   (2, 6), (12, 14))


def positions():
    return {
        'initial': Game(),
        'white_to_play': play(((7, 7),)),
        'black_to_play': play(((7, 7), (7, 8))),
        'black_with_forbidden': play(FORBIDDEN_MOVES),
        'white_after_forbidden_setup': play(FORBIDDEN_MOVES + ((0, 0),)),
    }


def seeded_model(seed=7):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PolicyValueNet().eval()


@unittest.skipIf(torch is None, 'install the neural extra')
class PolicyValueEvaluatorTest(unittest.TestCase):
    def setUp(self):
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        self.evaluator = PolicyValueEvaluator(seeded_model())

    def test_forbidden_fixture(self):
        game = positions()['black_with_forbidden']
        self.assertIsNotNone(forbidden_reason(game.board, 3, 6))
        self.assertNotIn((3, 6), game.legal_moves())

    def test_encoding_matches_encode_game_on_live_game(self):
        for name, game in positions().items():
            legal = game.legal_moves()
            mask = legal_moves_to_mask(legal)
            expected = encode_game(game, mask)
            planes, used_mask = self.evaluator.encode(EvaluationSnapshot.from_game(game, legal))
            self.assertTrue(torch.equal(planes, expected), name)
            self.assertTrue(torch.equal(used_mask, mask), name)

    def test_outputs_pass_search_validation(self):
        for name, game in positions().items():
            snapshot = EvaluationSnapshot.from_game(game, game.legal_moves())
            result = validate_evaluation(self.evaluator.evaluate(snapshot), snapshot)
            self.assertTrue(-1 <= result.value <= 1, name)

    def test_uses_supplied_legal_moves_without_recomputing(self):
        game = positions()['black_to_play']
        subset = ((0, 0), (5, 5), (9, 3))
        snapshot = EvaluationSnapshot.from_game(game, subset)
        with patch.object(Game, 'legal_moves', side_effect=AssertionError('recomputed')), \
                patch.object(Game, 'has_legal_move', side_effect=AssertionError('recomputed')):
            result = self.evaluator.evaluate(snapshot)
            planes, _ = self.evaluator.encode(snapshot)
        nonzero = {a for a, p in enumerate(result.priors) if p > 0}
        self.assertEqual(nonzero, {0, 5 * 15 + 5, 9 * 15 + 3})
        self.assertEqual(planes[5].sum().item(), 3)
        validate_evaluation(result, snapshot)

    def test_eval_and_inference_mode(self):
        model = seeded_model().train()
        evaluator = PolicyValueEvaluator(model)
        self.assertFalse(evaluator.model.training)
        seen = []
        original = evaluator.model.forward

        def spy(x):
            seen.append((torch.is_inference_mode_enabled(), evaluator.model.training))
            return original(x)
        evaluator.model.forward = spy
        game = positions()['black_to_play']
        evaluator.evaluate(EvaluationSnapshot.from_game(game, game.legal_moves()))
        self.assertEqual(seen, [(True, False)])
        evaluator.model.train()
        with self.assertRaises(RuntimeError):
            evaluator.evaluate(EvaluationSnapshot.from_game(game, game.legal_moves()))

    def test_batch_matches_single(self):
        # atol 1e-6 failed on CPU (max |diff| ~2.5e-6 prior, ~1.9e-6 value): batched and
        # single convolutions take different float32 kernel paths. 1e-5 matches the
        # policy-sum tolerance; bit-exactness is only required for a fixed batch size.
        snapshots = [EvaluationSnapshot.from_game(g, g.legal_moves()) for g in positions().values()]
        batch = self.evaluator.evaluate_batch(snapshots)
        self.assertEqual(self.evaluator.evaluate_batch([]), [])
        for snapshot, batched in zip(snapshots, batch):
            single = self.evaluator.evaluate(snapshot)
            self.assertTrue(torch.allclose(torch.tensor(single.priors), torch.tensor(batched.priors),
                                           atol=1e-5, rtol=0))
            self.assertAlmostEqual(single.value, batched.value, delta=1e-5)
            illegal = [a for a, p in enumerate(single.priors) if p == 0]
            self.assertTrue(all(batched.priors[a] == 0 for a in illegal))

    def test_search_with_neural_evaluator(self):
        game = positions()['black_to_play']
        before = ([row[:] for row in game.board], list(game.history))
        torch_rng = torch.random.get_rng_state()
        result = run_search(game, self.evaluator, SearchConfig(num_simulations=6,
                                                               noise_enabled=False))
        self.assertEqual(sum(result.visit_counts), 6)
        self.assertEqual(result.evaluator_calls, 7)
        self.assertTrue(torch.equal(torch_rng, torch.random.get_rng_state()))
        self.assertEqual(before, ([row[:] for row in game.board], list(game.history)))


@unittest.skipIf(torch is None, 'install the neural extra')
class RandomCheckpointTest(unittest.TestCase):
    def test_save_hash_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nested' / 'random.pt'
            rng_before = torch.random.get_rng_state()
            digest = create_random_checkpoint(path, 11)
            self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
            self.assertEqual(digest, file_sha256(path))
            self.assertEqual(len(digest), 64)
            evaluator = PolicyValueEvaluator.from_checkpoint(path)
            self.assertFalse(evaluator.model.training)
            other = Path(tmp) / 'again.pt'
            create_random_checkpoint(other, 11)
            first, second = load_checkpoint(path).state_dict(), load_checkpoint(other).state_dict()
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assertTrue(torch.equal(first[key], second[key]), key)
            reference = seeded_model(11).state_dict()
            self.assertTrue(all(torch.equal(first[k], reference[k]) for k in first))


if __name__ == '__main__':
    unittest.main()
