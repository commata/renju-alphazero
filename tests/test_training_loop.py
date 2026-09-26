import json
import math
import tempfile
import unittest
from pathlib import Path

from training.config import load_config

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from training.loop import run_training, verify_checkpoint
    from training.metrics import read_metrics
    from training.self_play import GameRecord, replay_record
    from training.training_checkpoint import load_checkpoint_payload
    from scripts.verify_stage6_evaluation import verify_run


@unittest.skipIf(torch is None, 'requires torch')
class TrainingLoopEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        cls.run_dir = Path(cls.tmp.name) / 'run'
        cls.state = run_training(cls.config, run_dir=cls.run_dir, log=lambda message: None)
        cls.events = read_metrics(cls.run_dir / 'metrics.jsonl')

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        cls.tmp.cleanup()

    def test_run_directory_layout(self):
        checkpoints = self.run_dir / 'checkpoints'
        self.assertEqual(sorted(p.name for p in checkpoints.iterdir()),
                         ['checkpoint_gen001.pt', 'checkpoint_gen002.pt',
                          'checkpoint_init.pt', 'latest.pt'])
        for name in ('config.yaml', 'metadata.json', 'metrics.jsonl', 'self_play/gen000.json',
                     'self_play/gen001.json', 'evaluation/gen000.json',
                     'evaluation/gen001.json'):
            self.assertTrue((self.run_dir / name).exists(), name)
        self.assertEqual(load_config(self.run_dir / 'config.yaml'), self.config)

    def test_checkpoint_generation_is_next_generation(self):
        checkpoints = self.run_dir / 'checkpoints'
        for name, generation in (('checkpoint_init.pt', 0), ('checkpoint_gen001.pt', 1),
                                 ('checkpoint_gen002.pt', 2), ('latest.pt', 2)):
            self.assertEqual(load_checkpoint_payload(checkpoints / name)['generation'],
                             generation, name)
        self.assertEqual((checkpoints / 'latest.pt').read_bytes(),
                         (checkpoints / 'checkpoint_gen002.pt').read_bytes())
        self.assertEqual(self.state.generation, 2)
        self.assertEqual(self.state.global_step, 2 * self.config['training']['steps_per_generation'])

    def test_metrics_events(self):
        t = self.config['training']
        train = [e for e in self.events if e['type'] == 'train']
        self.assertEqual([e['global_step'] for e in train],
                         list(range(1, 2 * t['steps_per_generation'] + 1)))
        for event in train:
            for key in ('policy_loss', 'value_loss', 'total_loss', 'grad_norm'):
                self.assertTrue(math.isfinite(event[key]), key)
        generations = [e for e in self.events if e['type'] == 'generation']
        self.assertEqual([e['generation'] for e in generations], [0, 1])
        buffer = 0
        for event in generations:
            buffer += event['new_samples']
            self.assertEqual(event['buffer_size'], buffer)
            self.assertEqual(event['self_play_games'], t['games_per_generation'])
            self.assertEqual(event['black_wins'] + event['white_wins'] + event['draws'],
                             t['games_per_generation'])
            self.assertEqual(event['samples_drawn'], t['steps_per_generation'] * t['batch_size'])
            self.assertAlmostEqual(event['sample_reuse_ratio'],
                                   event['samples_drawn'] / event['new_samples'])
            self.assertEqual(event['illegal_moves'], 0)
        evaluations = [e for e in self.events if e['type'] == 'evaluation']
        self.assertEqual(sorted((e['generation'], e['opponent']) for e in evaluations),
                         [(g, o) for g in (0, 1) for o in ('previous', 'random', 'tactical')])
        for event in evaluations:
            self.assertEqual(event['illegal_moves'], 0)
            self.assertEqual(event['games'], 2)
            self.assertIn('unique_games', event)

    def test_metadata(self):
        metadata = json.loads((self.run_dir / 'metadata.json').read_text(encoding='utf-8'))
        for key in ('git_branch', 'git_commit', 'git_dirty', 'seed', 'device', 'python',
                    'torch', 'model_version'):
            self.assertIn(key, metadata)
        segment, = metadata['segments']
        self.assertEqual((segment['start_generation'], segment['end_generation'],
                          segment['resumed'], segment['status']), (0, 2, False, 'completed'))

    def test_self_play_records_replay(self):
        for gen in (0, 1):
            data = json.loads((self.run_dir / 'self_play' / f'gen{gen:03d}.json')
                              .read_text(encoding='utf-8'))
            for game in data['games']:
                replay_record(GameRecord.from_dict(game['record']))

    def test_latest_reload_plays_one_self_play_game(self):
        result = verify_checkpoint(self.run_dir / 'checkpoints' / 'latest.pt')
        self.assertEqual(result['generation'], 2)
        self.assertGreater(result['moves'], 0)
        self.assertEqual(result['illegal_moves'], 0)

    def test_persisted_evaluation_wld_audit(self):
        result = verify_run(self.run_dir)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['generations'], [0, 1])
        self.assertEqual(result['opponents'], 6)
        self.assertEqual(result['games'], 12)

    def test_new_run_refuses_non_empty_directory(self):
        with self.assertRaises(FileExistsError):
            run_training(self.config, run_dir=self.run_dir, log=lambda message: None)


if __name__ == '__main__':
    unittest.main()
