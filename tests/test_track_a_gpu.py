"""Track A GPU plan: BatchedEvaluator, device roles, batched self-play/evaluation/h2h.

On the CPU the batched paths reproduce the serial ones exactly. The CUDA path (pinned
buffers, events) is exercised by ``test_device_matches_cpu_within_tolerance`` when a GPU exists.
"""
from copy import deepcopy
import json
from pathlib import Path
from random import Random
import shutil
import sys
import tempfile
import unittest

from renju import Game
from search.alphazero import SearchConfig
from search.evaluator import EvaluationSnapshot
from training.config import (ConfigError, critical_config_hash, load_config, parallel_setting,
                             resolve_config, role_device)

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from model.batched_evaluator import BatchedEvaluator, encode_snapshots
    from model.config import ModelConfig
    from model.encoding import encode_game
    from model.evaluator import PolicyValueEvaluator, _SnapshotView
    from model.masking import legal_moves_to_mask
    from model.network import PolicyValueNet
    from training.dataset import augment_batch, augment_batch_grouped, build_batch
    from training.evaluation import evaluate_generation
    from training.loop import run_training
    from training.metrics import read_metrics
    from training.self_play import play_self_play_games, record_hash
    from training.trainer import model_for_device
    from training.training_state import init_training_state

TINY = {'channels': 8, 'blocks': 1, 'policy_channels': 2, 'value_channels': 1, 'value_hidden': 8}


def snapshots(count=10, seed=3):
    rng = Random(seed)
    result = []
    for n in range(count):
        game = Game()
        for _ in range(2 * n):
            if game.done:
                break
            game.play(*rng.choice(game.legal_moves()))
        if not game.done:
            result.append(EvaluationSnapshot.from_game(game, game.legal_moves()))
    return result


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet(ModelConfig(**TINY)).eval()


class ConfigTest(unittest.TestCase):
    def test_devices_and_parallel_are_not_critical(self):
        base = load_config(ROOT / 'configs' / 'stage8_b400_temp4.yaml')
        changed = deepcopy(base)
        changed['devices'] = {'training': 'cuda', 'self_play': 'cuda', 'evaluation': 'cpu'}
        changed['parallel'] = {'self_play': 'batched', 'evaluation': 'batched', 'max_batch': 16,
                               'eager': True}
        self.assertEqual(critical_config_hash(base), critical_config_hash(changed))

    def test_role_device_falls_back_to_device(self):
        config = resolve_config({'device': 'cpu', 'devices': {'training': 'cuda'}})
        self.assertEqual(role_device(config, 'training'), 'cuda')
        self.assertEqual(role_device(config, 'self_play'), 'cpu')
        old = resolve_config({})
        del old['devices'], old['parallel']   # a config written before the keys existed
        self.assertEqual(role_device(old, 'evaluation'), 'cpu')
        self.assertEqual(parallel_setting(old, 'self_play'), 'serial')
        with self.assertRaises(ValueError):
            role_device(config, 'probe')

    def test_validation(self):
        for bad in ({'parallel': {'self_play': 'gpu'}}, {'parallel': {'max_batch': 0}},
                    {'parallel': {'eager': 1}}, {'devices': {'training': 3}}):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                resolve_config(bad)


@unittest.skipIf(torch is None, 'requires torch')
class BatchedEvaluatorTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_encode_matches_encode_game(self):
        items = snapshots()
        planes, masks = encode_snapshots(items)
        expected = torch.stack([encode_game(_SnapshotView(s), legal_moves_to_mask(s.legal_moves))
                                for s in items])
        self.assertTrue(torch.equal(planes, expected))
        self.assertTrue(torch.equal(masks, expected[:, 5].reshape(len(items), -1).bool()))

    def test_outputs_equal_policy_value_evaluator(self):
        model = tiny_model()
        items = snapshots()
        self.assertEqual(BatchedEvaluator(model).evaluate_batch(items),
                         PolicyValueEvaluator(model).evaluate_batch(items))
        evaluator = BatchedEvaluator(model)
        self.assertEqual(evaluator.evaluate(items[0]), PolicyValueEvaluator(model).evaluate(items[0]))
        self.assertEqual((evaluator.calls, evaluator.batches), (1, 1))
        self.assertEqual(evaluator.evaluate_batch([]), [])

    def test_contract(self):
        model = tiny_model()
        evaluator = BatchedEvaluator(model)
        pending = evaluator.submit(snapshots(2))
        with self.assertRaises(RuntimeError):
            evaluator.submit(snapshots(2))
        evaluator.collect(pending)
        model.train()
        with self.assertRaises(RuntimeError):
            evaluator.submit(snapshots(2))

    def test_device_matches_cpu_within_tolerance(self):
        # CUDA when present (pinned buffers, async event); otherwise the CPU path, which is
        # exact. The CI policy allows no skips, so this test never skips.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = tiny_model()
        items = snapshots()
        cpu = PolicyValueEvaluator(model).evaluate_batch(items)
        evaluator = BatchedEvaluator(deepcopy(model), device=device)
        pending = evaluator.submit(items)
        device_results = evaluator.collect(pending)
        self.assertTrue(pending.done())
        for a, b in zip(cpu, device_results):
            self.assertLess(max(abs(x - y) for x, y in zip(a.priors, b.priors)), 1e-4)
            self.assertLess(abs(a.value - b.value), 1e-4)

    def test_batched_self_play_equals_serial_on_cpu(self):
        model = tiny_model(1)
        config = SearchConfig(num_simulations=6, temperature_moves=4, tactical_rules=True)
        seeds = [1, 2, 3, 4]
        serial = play_self_play_games(PolicyValueEvaluator(model), config, seeds)
        batched = play_self_play_games(BatchedEvaluator(model), config, seeds, batched=True,
                                       max_batch=3)
        self.assertEqual([record_hash(g.record) for g in serial],
                         [record_hash(g.record) for g in batched])


@unittest.skipIf(torch is None, 'requires torch')
class AugmentationTest(unittest.TestCase):
    def test_grouped_equals_per_row(self):
        model_config = resolve_config({'model': TINY})
        state = init_training_state(model_config)
        from training.self_play import play_self_play_game
        from training.dataset import samples_from_record
        from search.evaluator import UniformEvaluator
        game = play_self_play_game(UniformEvaluator(), SearchConfig(num_simulations=2), 5)
        state.buffer.extend(samples_from_record(game.record, generation=0, game_id=0))
        batch = state.buffer.get(list(range(min(16, len(state.buffer)))))
        a, sym_a = augment_batch(batch, Random(9))
        b, sym_b = augment_batch_grouped(batch, Random(9))
        self.assertEqual(sym_a, sym_b)
        for x, y in ((a.states, b.states), (a.policies, b.policies),
                     (a.legal_masks, b.legal_masks), (a.values, b.values)):
            self.assertTrue(torch.equal(x, y))
        cpu = build_batch(state.buffer, 8, sample_rng=Random(1), augment_rng=Random(2),
                          augment=True, device='cpu')
        plain = build_batch(state.buffer, 8, sample_rng=Random(1), augment_rng=Random(2),
                            augment=True)
        self.assertTrue(torch.equal(cpu.states, plain.states))

    def test_model_for_device_keeps_same_object(self):
        model = tiny_model()
        self.assertIs(model_for_device(model, 'cpu'), model)


@unittest.skipIf(torch is None, 'requires torch')
class BatchedLoopTest(unittest.TestCase):
    """Two tiny runs, serial vs batched self-play and evaluation: identical results."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = Path(tempfile.mkdtemp())
        base = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        base['training']['games_per_generation'] = 3
        cls.runs = {}
        for name, mode in (('serial', 'serial'), ('batched', 'batched')):
            config = deepcopy(base)
            config['parallel'] = {'self_play': mode, 'evaluation': mode, 'max_batch': None,
                                  'eager': False}
            run_dir = cls.tmp / name
            state = run_training(config, run_dir=run_dir, log=lambda message: None)
            cls.runs[name] = (run_dir, state)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_same_games_and_weights(self):
        (serial_dir, serial), (batched_dir, batched) = self.runs['serial'], self.runs['batched']
        paths = sorted((serial_dir / 'self_play').glob('*.json'))
        self.assertEqual(len(paths), 2)
        for path in paths:
            expected = [g['record_sha256'] for g in json.loads(path.read_text())['games']]
            other = json.loads((batched_dir / 'self_play' / path.name).read_text())
            self.assertEqual(len(expected), 3)
            self.assertEqual([g['record_sha256'] for g in other['games']], expected)
        for key, value in serial.model.state_dict().items():
            self.assertTrue(torch.equal(value, batched.model.state_dict()[key]), key)

    def test_generation_event_records_batching(self):
        events = [e for e in read_metrics(self.runs['batched'][0] / 'metrics.jsonl')
                  if e['type'] == 'generation']
        self.assertTrue(events)
        for event in events:
            self.assertEqual(event['devices'], {'training': 'cpu', 'self_play': 'cpu',
                                                'evaluation': 'cpu'})
            self.assertEqual(event['self_play_batching']['tasks'], 3)
            self.assertLessEqual(event['self_play_batching']['max_batch'], 3)
        serial = [e for e in read_metrics(self.runs['serial'][0] / 'metrics.jsonl')
                  if e['type'] == 'generation']
        self.assertNotIn('self_play_batching', serial[0])

    def test_batched_evaluation_equals_serial(self):
        config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        model = init_training_state(config).model
        previous = init_training_state(config).model
        serial = evaluate_generation(model, previous, 0, config, 1)
        batched_config = deepcopy(config)
        batched_config['parallel']['evaluation'] = 'batched'
        batched = evaluate_generation(model, previous, 0, batched_config, 1)
        for name in serial['opponents']:
            self.assertEqual([g['moves'] for g in serial['opponents'][name]['games']],
                             [g['moves'] for g in batched['opponents'][name]['games']], name)
            self.assertIn('batching', batched['opponents'][name]['summary'])
        self.assertTrue(model.training)


@unittest.skipIf(torch is None, 'requires torch')
class HeadToHeadBatchedTest(unittest.TestCase):
    def test_batched_matches_equal_serial(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        from run_stage8_head_to_head import play_match, play_matches_batched
        from training.evaluation import PUCTAgent

        search = SearchConfig(num_simulations=3, temperature_moves=0, noise_enabled=False)
        models = [tiny_model(seed) for seed in (1, 2, 3)]
        serial_players = [{'label': f'm{i}', 'agent': PUCTAgent(f'm{i}', m, search)}
                          for i, m in enumerate(models)]
        batched_players = [{'label': f'm{i}', 'agent': PUCTAgent(
            f'm{i}', m, search, evaluator=BatchedEvaluator(m))} for i, m in enumerate(models)]
        pairs = [(0, 1), (0, 2), (1, 2)]
        quiet = lambda message: None  # noqa: E731
        serial = [play_match(serial_players[a], serial_players[b], pairs=2, seed=5,
                             opening_plies=2, radius=2, log=quiet) for a, b in pairs]
        batched, stats = play_matches_batched(
            [(batched_players[a], batched_players[b]) for a, b in pairs], pairs=2, seed=5,
            opening_plies=2, radius=2, max_active=5, max_batch=None, eager=False, log=quiet)
        for s, b in zip(serial, batched):
            self.assertEqual([g['moves'] for g in s['games']], [g['moves'] for g in b['games']])
            self.assertEqual([g['pair'] for g in s['games']], [g['pair'] for g in b['games']])
            self.assertEqual(s['summary']['a_wins'], b['summary']['a_wins'])
        self.assertEqual(stats['tasks'], 12)
        self.assertGreater(stats['max_batch'], 1)


if __name__ == '__main__':
    unittest.main()
