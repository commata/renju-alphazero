"""Resume equivalence: continuous 3 generations vs stop/reload and crash/reload.

CPU, single process, no DataLoader, fixed seed, torch threads 1 and deterministic
algorithms. Everything is compared with exact equality (torch.equal / ==).
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.config import load_config

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from training import loop as loop_module
    from training.loop import run_training
    from training.metrics import read_metrics
    from training.training_checkpoint import load_checkpoint_payload
    from training.training_state import COMPONENT_RNGS

QUIET = dict(log=lambda message: None)


def resume_config():
    config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
    config['training']['generations'] = 3
    config['evaluation']['every'] = 2  # gen 0 and the final gen 2 are evaluated
    config['evaluation']['tactical'] = {'black_games': 0, 'white_games': 0}  # runtime only
    return config


def assert_tensor_tree_equal(test, a, b, path='root'):
    if isinstance(a, torch.Tensor):
        test.assertTrue(isinstance(b, torch.Tensor) and a.dtype == b.dtype
                        and torch.equal(a, b), path)
    elif isinstance(a, dict):
        test.assertEqual(sorted(a, key=str), sorted(b, key=str), path)
        for key in a:
            assert_tensor_tree_equal(test, a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b), path)
        for index, (x, y) in enumerate(zip(a, b)):
            assert_tensor_tree_equal(test, x, y, f'{path}[{index}]')
    else:
        test.assertEqual(a, b, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def games(run_dir, kind, generation):
    data = read_json(Path(run_dir) / kind / f'gen{generation:03d}.json')
    if kind == 'self_play':
        return [(g['record_sha256'], g['record']['moves']) for g in data['games']]
    return {name: [(g['moves'], g['result']) for g in opponent['games']]
            for name, opponent in data['opponents'].items()}


@unittest.skipIf(torch is None, 'requires torch')
class ResumeEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        cls.deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.a, cls.b, cls.c = root / 'continuous', root / 'stopped', root / 'crashed'

        run_training(resume_config(), run_dir=cls.a, **QUIET)

        stopped = run_training(resume_config(), run_dir=cls.b, stop_after=2, **QUIET)
        cls.b_stop_generation = stopped.generation
        run_training(None, resume=cls.b / 'checkpoints' / 'latest.pt', **QUIET)

        real_train_step = loop_module.train_step
        calls = {'n': 0}

        def crash_in_generation_1(*args, **kwargs):
            calls['n'] += 1
            steps = resume_config()['training']['steps_per_generation']
            if calls['n'] == steps + 2:  # second step of generation 1
                raise RuntimeError('injected crash')
            return real_train_step(*args, **kwargs)

        with patch.object(loop_module, 'train_step', crash_in_generation_1):
            with cls.assertRaises(cls(), RuntimeError):
                run_training(resume_config(), run_dir=cls.c, **QUIET)
        cls.c_crash_latest = load_checkpoint_payload(cls.c / 'checkpoints' / 'latest.pt')
        cls.c_crash_events = read_metrics(cls.c / 'metrics.jsonl')
        run_training(resume_config(), resume=cls.c / 'checkpoints' / 'latest.pt', **QUIET)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        torch.use_deterministic_algorithms(cls.deterministic)
        cls.tmp.cleanup()

    def final(self, run_dir):
        return load_checkpoint_payload(Path(run_dir) / 'checkpoints' / 'latest.pt')

    def assert_same_final_state(self, other):
        a, b = self.final(self.a), self.final(other)
        self.assertEqual((a['generation'], a['global_step']), (3, 6))
        self.assertEqual((b['generation'], b['global_step']), (3, 6))
        assert_tensor_tree_equal(self, a['model_state_dict'], b['model_state_dict'], 'model')
        assert_tensor_tree_equal(self, a['optimizer_state_dict'], b['optimizer_state_dict'],
                                 'optimizer')
        assert_tensor_tree_equal(self, a['replay_buffer'], b['replay_buffer'], 'buffer')
        for name in COMPONENT_RNGS:
            self.assertEqual(a['component_rng'][name], b['component_rng'][name], name)
        assert_tensor_tree_equal(self, a['global_rng'], b['global_rng'], 'global_rng')
        self.assertEqual(a['critical_config_hash'], b['critical_config_hash'])
        for generation in range(3):
            self.assertEqual(games(self.a, 'self_play', generation),
                             games(other, 'self_play', generation), generation)
        for generation in (0, 2):
            self.assertEqual(games(self.a, 'evaluation', generation),
                             games(other, 'evaluation', generation), generation)

    def test_stop_and_resume_matches_continuous(self):
        self.assertEqual(self.b_stop_generation, 2)
        self.assert_same_final_state(self.b)

    def test_crash_resume_matches_continuous(self):
        self.assert_same_final_state(self.c)

    def test_crash_left_generation_1_checkpoint(self):
        # Crash during generation 1 -> latest holds generation 1 (re-run gen 1, no off-by-one).
        self.assertEqual(self.c_crash_latest['generation'], 1)
        self.assertTrue(any(e['generation'] == 1 and e['type'] == 'train'
                            for e in self.c_crash_events))
        metadata = read_json(self.c / 'metadata.json')
        first, second = metadata['segments']
        self.assertEqual((first['status'], first['end_generation']), ('failed', 1))
        self.assertIn('injected crash', first['error'])
        self.assertEqual((second['resumed'], second['start_generation'],
                          second['end_generation'], second['status']), (True, 1, 3, 'completed'))
        self.assertTrue(list(self.c.glob('metrics.jsonl.bak-*')))

    def test_metrics_have_no_duplicate_generations(self):
        steps = resume_config()['training']['steps_per_generation']
        continuous = read_metrics(self.a / 'metrics.jsonl')
        for run_dir in (self.b, self.c):
            events = read_metrics(run_dir / 'metrics.jsonl')
            train = [e['global_step'] for e in events if e['type'] == 'train']
            self.assertEqual(train, list(range(1, 3 * steps + 1)), run_dir.name)
            for kind in ('generation', 'checkpoint'):
                self.assertEqual([e['generation'] for e in events if e['type'] == kind],
                                 [0, 1, 2], (run_dir.name, kind))
            evaluations = sorted((e['generation'], e['opponent']) for e in events
                                 if e['type'] == 'evaluation')
            self.assertEqual(evaluations, sorted((e['generation'], e['opponent'])
                                                 for e in continuous
                                                 if e['type'] == 'evaluation'))
            self.assertEqual([e['total_loss'] for e in events if e['type'] == 'train'],
                             [e['total_loss'] for e in continuous if e['type'] == 'train'])

    def test_resume_rejects_changed_training_config(self):
        from training.config import ConfigError

        config = resume_config()
        config['optimizer']['lr'] = 0.5
        with self.assertRaises(ConfigError):
            run_training(config, resume=self.a / 'checkpoints' / 'latest.pt', **QUIET)


if __name__ == '__main__':
    unittest.main()
