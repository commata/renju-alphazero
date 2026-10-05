"""Stage 8 G0: lock-step batched self-play gives the same games as one at a time."""
import hashlib
import json
from pathlib import Path
from random import Random
import tempfile
import unittest

from model.config import ACTION_COUNT
from search.alphazero import SearchConfig, drive, search_steps, search_with_tree
from search.batched import LockstepStats, run_lockstep
from search.evaluator import EvaluationResult, ScriptedEvaluator
from renju import Game
from training.self_play import play_self_play_game, record_hash, self_play_steps

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None

ROOT = Path(__file__).resolve().parents[1]
CONFIG = SearchConfig(num_simulations=12, temperature_moves=2, tactical_rules=True)


def position_dependent(snapshot):
    """Deterministic, position-dependent priors and value (no batch dependence)."""
    digest = hashlib.sha256(repr((snapshot.board, snapshot.to_play)).encode()).digest()
    actions = snapshot.legal_actions()
    weights = [1 + digest[a % 32] + (a * 7919 % 13) for a in actions]
    total = sum(weights)
    priors = [0.0] * ACTION_COUNT
    for action, weight in zip(actions, weights):
        priors[action] = weight / total
    return EvaluationResult(tuple(priors), digest[0] / 255 * 1.6 - 0.8)


class BatchedSelfPlayTest(unittest.TestCase):
    def test_search_generator_matches_the_direct_search(self):
        game = Game()
        for move in [(7, 7), (7, 8), (8, 8)]:
            game.play(*move)
        a, _ = search_with_tree(game, ScriptedEvaluator(script=position_dependent), CONFIG,
                                Random(5))
        b, _ = drive(search_steps(game, CONFIG, Random(5)),
                     ScriptedEvaluator(script=position_dependent))
        self.assertEqual(a.visit_counts, b.visit_counts)
        self.assertEqual(a.priors, b.priors)

    def test_lockstep_games_equal_sequential_games(self):
        seeds = [11, 12, 13, 14, 15]
        sequential = [play_self_play_game(ScriptedEvaluator(script=position_dependent),
                                          CONFIG, seed).record for seed in seeds]
        evaluator = ScriptedEvaluator(script=position_dependent)
        stats = LockstepStats()
        games = run_lockstep([self_play_steps(CONFIG, seed) for seed in seeds], evaluator, stats)
        batched = [g.record for g in games]
        self.assertEqual([record_hash(r) for r in batched], [record_hash(r) for r in sequential])
        self.assertEqual(evaluator.calls, stats.evaluations)
        self.assertEqual(evaluator.batch_calls, stats.rounds)
        self.assertGreater(stats.mean_batch, 1.5)          # games really ran together
        self.assertEqual(max(int(k) for k in stats.to_dict()['batch_sizes']), len(seeds))


@unittest.skipIf(torch is None, 'requires torch')
class BatchedTrainingLoopTest(unittest.TestCase):
    def _run(self, tmp: Path, parallel: int):
        from training.config import load_config
        from training.loop import run_training

        config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        config['training'].update(generations=1, games_per_generation=4)
        config['self_play_parallel_games'] = parallel
        run = tmp / f'run{parallel}'
        run_training(config, run_dir=run, log=lambda m: None)
        records = [json.loads((run / 'self_play' / f'gen{g:03d}.json').read_text())
                   for g in range(1)]
        events = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()
                  if line.strip()]
        return records, [e for e in events if e.get('type') == 'generation']

    def test_checkpoint_without_the_key_still_resumes(self):
        """Checkpoints written before the key existed (every Stage 8 run) must resume."""
        from training.config import load_config
        from training.loop import run_training
        from training.training_checkpoint import (load_checkpoint_payload,
                                                  load_training_state, save_atomic)

        threads = torch.get_num_threads()
        config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        config['training'].update(generations=1, games_per_generation=2)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp) / 'run'
                run_training(config, run_dir=run, log=lambda m: None)
                latest = run / 'checkpoints' / 'latest.pt'
                payload = load_checkpoint_payload(latest)
                payload['config'].pop('self_play_parallel_games')
                save_atomic(latest, payload)
                old_style = {k: v for k, v in payload['config'].items()}
                load_training_state(latest, {**old_style, 'device': 'cpu'})
                resumed = dict(config, self_play_parallel_games=2)
                resumed['training'] = dict(config['training'], generations=2)
                state = run_training(resumed, resume=latest, log=lambda m: None)
                self.assertEqual(state.generation, 2)
        finally:
            torch.set_num_threads(threads)

    def test_parallel_games_change_nothing_but_the_batching(self):
        threads = torch.get_num_threads()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                one, events_one = self._run(Path(tmp), 1)
                four, events_four = self._run(Path(tmp), 4)
        finally:
            torch.set_num_threads(threads)
        hashes = lambda runs: [[g['record_sha256'] for g in r['games']] for r in runs]  # noqa: E731
        self.assertEqual(hashes(one), hashes(four))
        self.assertNotIn('self_play_batching', events_one[0])
        self.assertGreater(events_four[0]['self_play_batching']['mean_batch'], 1.0)


if __name__ == '__main__':
    unittest.main()


@unittest.skipIf(torch is None, 'requires torch')
class BatchEncodingTest(unittest.TestCase):
    def test_batch_encoding_equals_per_state_encoding(self):
        from model.encoding import encode_batch
        from model.evaluator import PolicyValueEvaluator
        from model.network import PolicyValueNet
        from search.evaluator import EvaluationSnapshot

        rng = Random(3)
        snapshots = []
        for length in (0, 1, 2, 5, 9, 14):
            game = Game()
            while len(game.history) < length and not game.done:
                game.play(*rng.choice(game.legal_moves()))
            snapshots.append(EvaluationSnapshot.from_game(game, game.legal_moves()))
        evaluator = PolicyValueEvaluator(PolicyValueNet().eval())
        old = [evaluator.encode(s) for s in snapshots]
        masks = torch.stack([m for _, m in old])
        planes = encode_batch([s.board for s in snapshots], [s.to_play for s in snapshots],
                              [s.last_move for s in snapshots], masks)
        self.assertTrue(torch.equal(planes, torch.stack([p for p, _ in old])))
        batched = evaluator.evaluate_batch(snapshots)
        with torch.inference_mode():
            from model.masking import masked_softmax
            logits, values = evaluator.model(torch.stack([p for p, _ in old]))
            priors = masked_softmax(logits, masks)
        self.assertEqual([r.priors for r in batched], [tuple(r) for r in priors.tolist()])
        self.assertEqual([r.value for r in batched], values.reshape(-1).tolist())
