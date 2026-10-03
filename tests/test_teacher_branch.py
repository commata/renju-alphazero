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
# White (3,3)(3,4) + (4,5)(5,5), white to move: (3,5) is a 3-3 (legal for white).
DOUBLE_THREE_WHITE_TO_MOVE = [(7, 7), (3, 3), (12, 1), (3, 4), (12, 5), (4, 5), (1, 12),
                              (5, 5), (12, 9)]


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
        # The whole game is dropped, not only the probe position (neighbouring plies leak).
        self.assertEqual(excluded['positions'], [])
        self.assertEqual(excluded['stats']['excluded_probe_games'], 1)
        self.assertIn('forced_loss|WHITE|opening', full['balance'])

    def test_probe_at_the_final_position_drops_the_game(self):
        from build_tactical_dataset import build, canonical_key

        moves = OPEN_FOUR + [(7, 3)]              # black completes five: final position
        games = [{'source': 't', 'moves': moves}]
        result = build(iter(games), exclude={canonical_key(moves)}, node_limit=20_000,
                       prove_losses=False, log=lambda m: None)
        self.assertEqual(result['positions'], [])
        self.assertEqual(result['stats']['excluded_probe_games'], 1)

    def test_vct_threat_proof(self):
        from analysis.tactical_labels import defense_label, proves_threat

        game = play(DOUBLE_THREE_WHITE_TO_MOVE)
        self.assertTrue(proves_threat(game, (3, 5), self.solver))   # white 3-3
        self.assertFalse(proves_threat(game, (14, 14), self.solver))
        self.assertEqual(game.history, DOUBLE_THREE_WHITE_TO_MOVE)
        game.undo()  # black to move before the threat: far too many candidates
        self.assertIsNone(defense_label(game, self.solver, max_candidates=5))

    def test_vct_candidates_come_from_the_continuation(self):
        from build_tactical_dataset import canonical_key, vct_tasks

        moves = DOUBLE_THREE_WHITE_TO_MOVE + [(3, 5), (3, 6), (3, 2)]
        keys = [canonical_key(moves[:ply]) for ply in range(len(moves))]
        t = len(DOUBLE_THREE_WHITE_TO_MOVE)
        labels = {keys[t + 2]: {'kind': 'unstoppable_four'}}
        tasks = vct_tasks([{'moves': moves, 'keys': keys, 'source': 's'}], labels)
        self.assertEqual([len(task['moves']) for task in tasks], [t + 1])
        self.assertEqual(tasks[0]['moves'][-1], [3, 5])
        self.assertTrue(tasks[0]['defend'])

    def test_vct_task_emits_attack_and_loss_labels(self):
        from build_tactical_dataset import _init_worker, _vct_task

        _init_worker(20_000)
        task = {'moves': [list(m) for m in DOUBLE_THREE_WHITE_TO_MOVE + [(3, 5)]],
                'defend': False, 'source': 's'}
        found = _vct_task((task, 1, 12))
        self.assertEqual([label['kind'] for _, label in found], ['vct_attack', 'vcf_loss'])
        self.assertEqual(found[0][1]['policy'], [(3, 5)])
        self.assertEqual(len(found[1][0]), len(DOUBLE_THREE_WHITE_TO_MOVE) + 1)


@unittest.skipIf(torch is None, 'requires torch')
class TeacherBranchTest(unittest.TestCase):
    def test_kind_shares_and_weights(self):
        from make_teacher_branch import kind_shares, parse_kind_weights

        kinds = ['must_block'] * 8 + ['vcf'] * 2
        self.assertEqual(kind_shares(kinds, False, None), {'must_block': 0.8, 'vcf': 0.2})
        self.assertEqual(kind_shares(kinds, True, None), {'must_block': 0.5, 'vcf': 0.5})
        weights = parse_kind_weights('vcf=3')
        self.assertEqual(kind_shares(kinds, True, weights), {'must_block': 0.25, 'vcf': 0.75})
        with self.assertRaises(ValueError):
            parse_kind_weights('vcf=-1')

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
                                             balance_kinds=True, log=lambda m: None)
                self.assertEqual(result['settings']['teacher_rows_drawn'], 20)
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
                reset_dest = Path(tmp) / 'teacher_reset'
                result = make_teacher_branch(source, 2, dataset, reset_dest, steps=2,
                                             batch_size=8, optimizer_state='reset',
                                             log=lambda m: None)
                self.assertEqual(result['settings']['run_optimizer_state'], 'reset')
                reset = load_checkpoint_payload(reset_dest / 'checkpoints' / 'latest.pt')
                self.assertEqual(reset['optimizer_state_dict']['state'], {})
                self.assertTrue(original['optimizer_state_dict']['state'])
                state = run_training(None, resume=reset_dest / 'checkpoints' / 'latest.pt',
                                     log=lambda m: None)
                self.assertEqual(state.generation, 3)
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()


@unittest.skipIf(torch is None, 'requires torch')
class RecipeBranchTest(unittest.TestCase):
    def test_recipe_branch_changes_only_the_config_and_resumes(self):
        import yaml
        from make_recipe_branch import make_recipe_branch
        from training.config import ConfigError, load_config
        from training.loop import run_training
        from training.training_checkpoint import load_checkpoint_payload

        threads = torch.get_num_threads()
        config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        config['training'].update(generations=3, keep_every=1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / 'source'
                run_training(config, run_dir=source, stop_after=2, log=lambda m: None)
                recipe = json.loads(json.dumps(config))
                recipe['self_play']['temperature_moves'] = 1
                recipe_path = Path(tmp) / 'recipe.yaml'
                recipe_path.write_text(yaml.safe_dump(recipe), encoding='utf-8')
                dest = Path(tmp) / 'recipe'
                result = make_recipe_branch(source, 2, recipe_path, dest)
                self.assertEqual(list(result['critical_changes']), ['self_play.temperature_moves'])
                original = load_checkpoint_payload(source / 'checkpoints' / 'checkpoint_gen002.pt')
                changed = load_checkpoint_payload(dest / 'checkpoints' / 'latest.pt')
                self.assertNotEqual(original['critical_config_hash'],
                                    changed['critical_config_hash'])
                for key in ('generation', 'global_step', 'component_rng'):
                    self.assertEqual(original[key], changed[key], key)
                self.assertTrue(all(torch.equal(original['model_state_dict'][k],
                                                changed['model_state_dict'][k])
                                    for k in original['model_state_dict']))
                with self.assertRaises(ConfigError):   # the old recipe no longer resumes it
                    run_training(config, resume=dest / 'checkpoints' / 'latest.pt',
                                 log=lambda m: None)
                state = run_training(load_config(recipe_path),
                                     resume=dest / 'checkpoints' / 'latest.pt', log=lambda m: None)
                self.assertEqual(state.generation, 3)
                self.assertEqual(state.config['self_play']['temperature_moves'], 1)
                with self.assertRaises(ValueError):    # no critical change left
                    make_recipe_branch(None, 3, recipe_path, dest, in_place=True)
        finally:
            torch.set_num_threads(threads)


    def test_recipe_branch_learning_rate_reaches_the_optimizer(self):
        import yaml
        from make_recipe_branch import make_recipe_branch
        from training.config import load_config
        from training.loop import run_training
        from training.training_checkpoint import load_training_state

        threads = torch.get_num_threads()
        config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        config['training'].update(generations=3, keep_every=1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / 'source'
                run_training(config, run_dir=source, stop_after=2, log=lambda m: None)
                recipe = json.loads(json.dumps(config))
                recipe['optimizer']['lr'] = config['optimizer']['lr'] * 0.3
                recipe_path = Path(tmp) / 'recipe.yaml'
                recipe_path.write_text(yaml.safe_dump(recipe), encoding='utf-8')
                dest = Path(tmp) / 'recipe'
                make_recipe_branch(source, 2, recipe_path, dest)
                # the saved Adam state still carries the old lr; the config must win
                state = load_training_state(dest / 'checkpoints' / 'latest.pt')
                self.assertEqual({g['lr'] for g in state.optimizer.param_groups},
                                 {recipe['optimizer']['lr']})
                self.assertTrue(state.optimizer.state)          # moments kept
                plain = load_training_state(source / 'checkpoints' / 'latest.pt')
                self.assertEqual({g['lr'] for g in plain.optimizer.param_groups},
                                 {config['optimizer']['lr']})
        finally:
            torch.set_num_threads(threads)


class RecipeConfigTest(unittest.TestCase):
    def test_temp4_differs_from_g3_b_only_in_temperature(self):
        from training.config import config_differences, critical_config, load_config

        base = load_config(ROOT / 'configs' / 'stage8_g3_b.yaml')
        temp4 = load_config(ROOT / 'configs' / 'stage8_b400_temp4.yaml')
        self.assertEqual(config_differences(critical_config(base), critical_config(temp4)),
                         ['self_play.temperature_moves'])
        self.assertEqual(temp4['self_play']['temperature_moves'], 4)

    def test_lr_arm_differs_from_ada_only_in_learning_rate(self):
        from training.config import config_differences, critical_config, load_config

        ada = load_config(ROOT / 'configs' / 'stage8_s2_adaptive.yaml')
        lr3 = load_config(ROOT / 'configs' / 'stage8_ada_lr3e4.yaml')
        self.assertEqual(config_differences(critical_config(ada), critical_config(lr3)),
                         ['optimizer.lr'])
        self.assertEqual(lr3['optimizer']['lr'], 0.0003)

    def test_temp2_differs_from_temp4_only_in_temperature(self):
        from training.config import config_differences, critical_config, load_config

        temp4 = load_config(ROOT / 'configs' / 'stage8_b400_temp4.yaml')
        temp2 = load_config(ROOT / 'configs' / 'stage8_s640_temp2.yaml')
        self.assertEqual(config_differences(critical_config(temp4), critical_config(temp2)),
                         ['self_play.temperature_moves'])
        self.assertEqual(temp2['self_play']['temperature_moves'], 2)

    def test_sims100_differs_from_temp2_only_in_simulations(self):
        from training.config import config_differences, critical_config, load_config

        temp2 = load_config(ROOT / 'configs' / 'stage8_s640_temp2.yaml')
        sims = load_config(ROOT / 'configs' / 'stage8_s880_sims100.yaml')
        self.assertEqual(config_differences(critical_config(temp2), critical_config(sims)),
                         ['self_play.simulations'])
        self.assertEqual(sims['self_play']['simulations'], 100)
