"""Stage 8 MP: self-play worker processes (src/training/workers.py)."""
import json
from pathlib import Path
from random import Random
import tempfile
import unittest

from training.config import critical_config_hash, load_config, resolve_config, ConfigError

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from model.config import ModelConfig
    from model.evaluator import PolicyValueEvaluator
    from model.network import PolicyValueNet
    from search.alphazero import SearchConfig, drive
    from training.self_play import record_hash, self_play_steps
    from training.workers import SelfPlayPool, deal, undeal


class DealTest(unittest.TestCase):
    def test_round_robin_and_inverse(self):
        seeds = list(range(10))
        shares = deal(seeds, 3)
        self.assertEqual(shares, [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]])
        self.assertEqual(undeal(shares, len(seeds)), seeds)
        self.assertEqual(deal(seeds[:2], 4), [[0], [1]])        # never an empty share
        self.assertEqual(undeal(deal(seeds[:2], 4), 2), [0, 1])

    def test_workers_key_is_execution_only(self):
        base = load_config(ROOT / 'configs' / 'stage8_ada_sims400.yaml')
        self.assertEqual(critical_config_hash(base),
                         critical_config_hash(dict(base, self_play_workers=4)))
        with self.assertRaises(ConfigError):
            resolve_config({'self_play_workers': 0})


@unittest.skipIf(torch is None, 'requires torch')
class SelfPlayPoolTest(unittest.TestCase):
    def test_pool_games_equal_single_process(self):
        torch.manual_seed(3)
        model = PolicyValueNet(ModelConfig(channels=8, blocks=1, value_hidden=8)).eval()
        search = SearchConfig(num_simulations=4, temperature_moves=4, tactical_rules=True)
        rng = Random(5)
        seeds = [rng.getrandbits(63) for _ in range(5)]
        evaluator = PolicyValueEvaluator(model)
        expected = [record_hash(drive(self_play_steps(search, s), evaluator).record)
                    for s in seeds]
        with SelfPlayPool(2, 'cpu') as pool:
            games, summary = pool.play(model, search, seeds, parallel=4)
            again, _ = pool.play(model, search, seeds[:3], parallel=1)   # pool is reused
        self.assertEqual([record_hash(g.record) for g in games], expected)
        self.assertEqual([record_hash(g.record) for g in again], expected[:3])
        self.assertEqual(summary['workers'], 2)
        self.assertEqual(len(summary['worker_seconds']), 2)
        self.assertGreater(summary['mean_batch'], 1.0)
        self.assertTrue(model.training is False)


@unittest.skipIf(torch is None, 'requires torch')
class WorkersTrainingLoopTest(unittest.TestCase):
    def _run(self, tmp: Path, workers: int):
        from training.loop import run_training

        config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        config['training'].update(generations=2, games_per_generation=4)
        config['self_play_parallel_games'] = 4
        config['self_play_workers'] = workers
        run = tmp / f'run{workers}'
        state = run_training(config, run_dir=run, log=lambda m: None)
        records = [[g['record_sha256'] for g in
                    json.loads((run / 'self_play' / f'gen{g:03d}.json').read_text())['games']]
                   for g in range(2)]
        events = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()
                  if line.strip()]
        return records, [e for e in events if e.get('type') == 'generation'], state

    def test_workers_change_nothing_but_the_execution(self):
        threads = torch.get_num_threads()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                one, _, state_one = self._run(Path(tmp), 1)
                two, events_two, state_two = self._run(Path(tmp), 2)
                metadata = json.loads((Path(tmp) / 'run2' / 'metadata.json').read_text())
        finally:
            torch.set_num_threads(threads)
        self.assertEqual(one, two)
        for key, value in state_one.model.state_dict().items():
            self.assertTrue(torch.equal(value, state_two.model.state_dict()[key]), key)
        self.assertEqual(events_two[0]['self_play_batching']['workers'], 2)
        self.assertEqual(metadata['segments'][0]['self_play_workers'], 2)


if __name__ == '__main__':
    unittest.main()
