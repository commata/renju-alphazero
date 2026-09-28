"""Stage 8-G: external evaluation schedule, colour-regression summary, orchestration."""
import argparse
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from training.schedule import color_regression_summary, next_stop, schedule_points

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None


class ScheduleTest(unittest.TestCase):
    def test_points_follow_the_anchor(self):
        self.assertEqual(schedule_points(160, 20, 240), [160, 180, 200, 220, 240])
        self.assertEqual(schedule_points(160, 80, 480), [160, 240, 320, 400, 480])
        self.assertEqual(schedule_points(160, 20, 150), [])

    def test_next_stop(self):
        self.assertEqual(next_stop(163, 160, 20, 192), 180)   # desktop baseline left gen 163
        self.assertEqual(next_stop(180, 160, 20, 192), 192)   # capped at the gate target
        self.assertEqual(next_stop(160, 160, 20, 288), 180)
        self.assertEqual(next_stop(150, 160, 20, 288), 160)


def _light(generation, black, white, games=25):
    side = lambda wins: {'wins': wins, 'losses': games - wins, 'draws': 0, 'games': games}
    return {'generation': generation,
            'opponents': {'mcts_v2': {'summary': {'black': side(black), 'white': side(white)}}}}


class ColorSummaryTest(unittest.TestCase):
    def test_d32_like_white_collapse(self):
        # unordered input on purpose: the summary replays in generation order
        summary = color_regression_summary([_light(200, 15, 1), _light(160, 16, 17),
                                            _light(180, 15, 3)])
        self.assertEqual([r['generation'] for r in summary['rows']], [160, 180, 200])
        self.assertEqual([r['white']['status'] for r in summary['rows']],
                         ['healthy', 'candidate', 'confirmed'])
        self.assertEqual([r['black']['status'] for r in summary['rows']], ['healthy'] * 3)
        self.assertEqual([(a['generation'], a['color'], a['status']) for a in summary['alerts']],
                         [(180, 'white', 'candidate'), (200, 'white', 'confirmed')])

    def test_d16_like_run_has_no_alerts(self):
        summary = color_regression_summary([_light(160, 18, 21), _light(180, 22, 17),
                                            _light(200, 20, 23)])
        self.assertEqual(summary['alerts'], [])


@unittest.skipIf(torch is None, 'requires torch')
class OrchestrationTest(unittest.TestCase):
    def test_segmented_run_evaluates_on_schedule_and_matches_uninterrupted(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        from run_stage8_training import build_parser, orchestrate
        from training.config import load_config
        from training.loop import run_training

        threads = torch.get_num_threads()
        config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        config['training'].update(generations=1, keep_every=1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / 'run'
                run_training(config, run_dir=run_dir, log=lambda m: None)   # checkpoint_gen001
                target = deepcopy(config)
                target['training']['generations'] = 3
                args = build_parser().parse_args([
                    '--run-dir', str(run_dir), '--anchor', '1', '--light-every', '1',
                    '--heavy-every', '2', '--light-opponents', 'random', '--light-pairs', '1',
                    '--heavy-opponents', 'tactical', '--heavy-pairs', '1'])
                logs = []
                self.assertEqual(orchestrate(run_dir, target, args, log=logs.append), 3)
                out = run_dir / 'external_eval'
                self.assertEqual(sorted(p.name for p in out.glob('gen*.json')),
                                 ['gen001_heavy.json', 'gen001_light.json', 'gen001_replay.json',
                                  'gen002_light.json', 'gen002_replay.json', 'gen003_heavy.json',
                                  'gen003_light.json', 'gen003_replay.json'])
                replay = json.loads((out / 'gen003_replay.json').read_text(encoding='utf-8'))
                self.assertEqual(replay['black_to_move']['samples']
                                 + replay['white_to_move']['samples'], replay['samples'])
                self.assertEqual(sorted(p.name for p in (run_dir / 'probes').glob('*.json')),
                                 ['gen001.json', 'gen001_defense.json', 'gen002.json',
                                  'gen002_defense.json', 'gen003.json', 'gen003_defense.json'])
                light = json.loads((out / 'gen002_light.json').read_text(encoding='utf-8'))
                self.assertEqual(light['generation'], 2)
                self.assertEqual(light['opponents']['random']['summary']['games'], 2)
                summary = json.loads((out / 'color_regression.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['opponent'], 'mcts_v2')   # not in this light set
                self.assertEqual(summary['rows'], [])

                # idempotent: an eval-only re-run rewrites nothing
                stamps = {p: p.stat().st_mtime_ns for p in out.glob('gen*.json')}
                args.eval_only = True
                self.assertEqual(orchestrate(run_dir, None, args, log=logs.append), 3)
                self.assertEqual({p: p.stat().st_mtime_ns for p in out.glob('gen*.json')}, stamps)

                # segmented (stop/eval/resume) training == one uninterrupted run
                straight = run_training(target, run_dir=Path(tmp) / 'straight',
                                        log=lambda m: None)
                resumed = torch.load(run_dir / 'checkpoints' / 'latest.pt', weights_only=True)
                for key, value in straight.model.state_dict().items():
                    self.assertTrue(torch.equal(value, resumed['model_state_dict'][key]), key)
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
