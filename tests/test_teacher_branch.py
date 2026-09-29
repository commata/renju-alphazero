"""Teacher arm: proof-labelled dataset and the fine-tuned branch (docs/v7-nn-integration-review.md)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from analysis.tactical_labels import label_position, vcf_first_moves
from analysis.threats import ThreatSolver
from renju import Game

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None


def play(moves):
    game = Game()
    for move in moves:
        game.play(*move)
    return game


# Black (7,4..7,7) with both ends open, white to move -> forced loss for white.
OPEN_FOUR = [(7, 7), (0, 0), (7, 6), (0, 14), (7, 5), (14, 0), (7, 4)]
# Black three (7,5..7,7) open, white elsewhere, black to move -> open four wins.
OPEN_THREE_BLACK_TO_MOVE = [(7, 7), (0, 0), (7, 6), (0, 14), (7, 5), (14, 0)]
# White closed four (3,3..3,6) blocked by black (3,2); black to move must block (3,7).
CLOSED_FOUR = [(7, 7), (3, 3), (3, 2), (3, 4), (7, 6), (3, 5), (12, 12), (3, 6)]


class TacticalLabelTest(unittest.TestCase):
    def setUp(self):
        self.solver = ThreatSolver(node_limit=20_000)

    def test_labels(self):
        self.assertEqual(label_position(play(OPEN_FOUR), self.solver),
                         {'kind': 'forced_loss', 'policy': [], 'value': -1})
        win = play(OPEN_FOUR[:-1] + [(7, 4), (14, 14)])
        self.assertEqual(label_position(win, self.solver)['kind'], 'immediate_win')
        self.assertEqual(label_position(play(CLOSED_FOUR), self.solver),
                         {'kind': 'must_block', 'policy': [(3, 7)], 'value': None})
        label = label_position(play(OPEN_THREE_BLACK_TO_MOVE), self.solver)
        self.assertEqual(label['kind'], 'unstoppable_four')
        self.assertEqual(label['policy'], [(7, 4), (7, 8)])
        self.assertEqual(label['value'], 1)
        self.assertIsNone(label_position(play([(7, 7), (0, 0)]), self.solver))

    def test_vcf_first_moves_include_every_open_four(self):
        game = play(OPEN_THREE_BLACK_TO_MOVE)
        starts = vcf_first_moves(game, self.solver)
        self.assertIn((7, 4), starts)
        self.assertIn((7, 8), starts)
        self.assertEqual(game.history, OPEN_THREE_BLACK_TO_MOVE)

    def test_dataset_excludes_probe_positions_under_d4(self):
        from build_tactical_dataset import build, canonical_key

        games = [{'source': 't', 'moves': OPEN_FOUR + [(7, 3)]}]
        full = build(iter(games), exclude=set(), node_limit=20_000, prove_losses=False,
                     log=lambda m: None)
        kinds = [p['kind'] for p in full['positions']]
        self.assertIn('forced_loss', kinds)
        mirrored = [(r, 14 - c) for r, c in OPEN_FOUR]
        excluded = build(iter(games), exclude={canonical_key(mirrored)}, node_limit=20_000,
                         prove_losses=False, log=lambda m: None)
        self.assertNotIn('forced_loss', [p['kind'] for p in excluded['positions']])
        self.assertEqual(excluded['stats']['excluded_probe'], 1)


@unittest.skipIf(torch is None, 'requires torch')
class TeacherBranchTest(unittest.TestCase):
    def test_branch_changes_only_weights_and_resumes(self):
        from make_teacher_branch import make_teacher_branch
        from training.config import load_config
        from training.loop import run_training
        from training.replay_buffer import ReplayBuffer
        from training.training_checkpoint import load_checkpoint_payload

        threads = torch.get_num_threads()
        config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        config['training'].update(generations=3, keep_every=1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / 'source'
                run_training(config, run_dir=source, stop_after=2, log=lambda m: None)
                (source / 'external_eval').mkdir()
                (source / 'external_eval' / 'gen002_light.json').write_text('{}')
                (source / 'external_eval' / 'gen001_light.json').write_text('{}')
                dataset = Path(tmp) / 'tactical.json'
                dataset.write_text(json.dumps({'format': 'tactical-dataset-v1', 'positions': [
                    {'moves': OPEN_FOUR, 'kind': 'forced_loss', 'policy': [], 'value': -1},
                    {'moves': CLOSED_FOUR, 'kind': 'must_block', 'policy': [[3, 7]],
                     'value': None},
                    {'moves': OPEN_THREE_BLACK_TO_MOVE, 'kind': 'unstoppable_four',
                     'policy': [[7, 4], [7, 8]], 'value': 1},
                ]}))
                dest = Path(tmp) / 'teacher'
                result = make_teacher_branch(source, 2, dataset, dest, steps=5, batch_size=8,
                                             log=lambda m: None)
                self.assertEqual(result['removed_results'], ['external_eval/gen002_light.json'])
                self.assertTrue((dest / 'external_eval' / 'gen001_light.json').is_file())
                self.assertEqual(result['dataset_kinds'],
                                 {'forced_loss': 1, 'must_block': 1, 'unstoppable_four': 1})
                original = load_checkpoint_payload(source / 'checkpoints' / 'checkpoint_gen002.pt')
                tuned = load_checkpoint_payload(dest / 'checkpoints' / 'latest.pt')
                self.assertTrue(any(not torch.equal(original['model_state_dict'][k],
                                                    tuned['model_state_dict'][k])
                                    for k in original['model_state_dict']))
                for key in ('generation', 'global_step', 'critical_config_hash', 'component_rng'):
                    self.assertEqual(original[key], tuned[key], key)
                capacity = config['training']['replay_capacity']
                buffers = []
                for payload in (original, tuned):
                    buffer = ReplayBuffer(capacity)
                    buffer.load_state_dict(payload['replay_buffer'])
                    buffers.append(buffer)
                self.assertGreater(len(buffers[0]), 0)
                self.assertEqual(len(buffers[0]), len(buffers[1]))
                everything = torch.arange(len(buffers[0]))
                self.assertTrue(torch.equal(buffers[0].get(everything).states,
                                            buffers[1].get(everything).states))
                self.assertEqual(original['optimizer_state_dict']['state'].keys(),
                                 tuned['optimizer_state_dict']['state'].keys())
                state = run_training(None, resume=dest / 'checkpoints' / 'latest.pt',
                                     log=lambda m: None)
                self.assertEqual(state.generation, 3)
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
