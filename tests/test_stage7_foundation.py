"""Stage 7 foundation: continuation config, probe fixture labels, probe/eval tools."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from renju import BLACK, WHITE, Game
from search.mcts import _is_legal_for_player, _wins_for_player
from training.config import critical_config_hash, load_config

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / 'tests' / 'fixtures' / 'stage7_probes_v1.json'

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from training.loop import run_training
    from training.probes import load_probe_set, run_probe_file
    from scripts.run_stage7_checkpoint_eval import evaluate_checkpoint


def _replay(moves):
    game = Game()
    for move in moves:
        game.play(*move)
    return game


def _winning_points(game, player):
    return sorted((r, c) for r in range(15) for c in range(15)
                  if _is_legal_for_player(game, player, (r, c))
                  and _wins_for_player(game, player, (r, c)))


class Stage7ContinuationConfigTest(unittest.TestCase):
    def test_only_execution_control_keys_differ_from_stage6_mvp(self):
        stage6 = load_config(ROOT / 'configs' / 'stage6_mvp.yaml')
        stage7 = load_config(ROOT / 'configs' / 'stage7a_continuation.yaml')
        self.assertEqual(critical_config_hash(stage6), critical_config_hash(stage7))
        self.assertEqual(stage7['training']['generations'], 30)
        self.assertGreaterEqual(stage7['training']['keep_checkpoints'], 28)
        for key in ('games_per_generation', 'batch_size', 'steps_per_generation',
                    'replay_capacity'):
            self.assertEqual(stage6['training'][key], stage7['training'][key], key)
        self.assertEqual(stage6['self_play'], stage7['self_play'])
        self.assertEqual(stage6['model'], stage7['model'])
        self.assertEqual(stage6['evaluation'], stage7['evaluation'])


class Stage7ProbeFixtureTest(unittest.TestCase):
    """Labels are re-derived from the rules engine, independent of the builder."""

    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(PROBES.read_text(encoding='utf-8'))

    def test_format_ids_and_balance(self):
        self.assertEqual(self.data['format'], 'stage7-probes-v1')
        probes = self.data['probes']
        self.assertEqual(len({p['id'] for p in probes}), len(probes))
        self.assertEqual(len({json.dumps(p['moves']) + p['kind'] for p in probes}),
                         len(probes))
        for kind in ('immediate_win', 'must_block', 'forced_loss', 'vcf'):
            colours = {p['to_play'] for p in probes if p['kind'] == kind}
            self.assertEqual(colours, {'BLACK', 'WHITE'}, kind)

    def test_labels_match_rules_engine(self):
        for probe in self.data['probes']:
            with self.subTest(probe=probe['id']):
                game = _replay([tuple(m) for m in probe['moves']])
                self.assertFalse(game.done)
                player = game.to_play
                self.assertEqual(probe['to_play'], 'BLACK' if player == BLACK else 'WHITE')
                legal = set(game.legal_moves())
                correct = [tuple(m) for m in probe['correct_moves']]
                self.assertTrue(set(correct) <= legal)
                own = _winning_points(game, player)
                threats = _winning_points(game, -player)
                kind = probe['kind']
                if kind == 'immediate_win':
                    self.assertEqual(correct, own)
                    self.assertEqual(probe['value_sign'], 1)
                elif kind == 'must_block':
                    self.assertEqual(own, [])
                    self.assertEqual(correct, threats)
                    self.assertEqual(len(correct), 1)
                    self.assertIsNone(probe['value_sign'])
                elif kind == 'forced_loss':
                    self.assertEqual(own, [])
                    self.assertGreaterEqual(len(threats), 2)
                    self.assertEqual(correct, [])
                    self.assertEqual(probe['value_sign'], -1)
                elif kind == 'vcf':
                    self.assertTrue(correct)
                    self.assertEqual(probe['value_sign'], 1)
                elif kind == 'avoid':
                    self.assertTrue({tuple(m) for m in probe['avoid_moves']} <= legal)
                else:
                    self.fail(f'unknown kind {kind}')


@unittest.skipIf(torch is None, 'requires torch')
class Stage7ToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.tmp.name) / 'stage6'
        cls.config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        cls.config['training']['generations'] = 1
        run_training(cls.config, run_dir=cls.run_dir, log=lambda _: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        torch.set_num_threads(cls.threads)

    def test_resume_of_copied_run_leaves_original_untouched(self):
        copy = Path(self.tmp.name) / 'stage7a'
        shutil.copytree(self.run_dir, copy)
        before = {p.relative_to(self.run_dir): p.read_bytes()
                  for p in self.run_dir.rglob('*') if p.is_file()}
        config = json.loads(json.dumps(self.config))
        config['training']['generations'] = 2
        config['training']['keep_checkpoints'] = 32
        config['output']['run_name'] = 'stage7a_test'
        state = run_training(config, resume=copy / 'checkpoints' / 'latest.pt',
                             log=lambda _: None)
        self.assertEqual(state.generation, 2)
        self.assertEqual(Path(state.run_dir).resolve(), copy.resolve())
        self.assertTrue((copy / 'checkpoints' / 'checkpoint_gen002.pt').is_file())
        after = {p.relative_to(self.run_dir): p.read_bytes()
                 for p in self.run_dir.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_probe_results_are_complete_and_deterministic(self):
        checkpoint = self.run_dir / 'checkpoints' / 'checkpoint_gen001.pt'
        first = run_probe_file(checkpoint, PROBES)
        second = run_probe_file(checkpoint, PROBES)
        self.assertEqual(first['summary'], second['summary'])
        probes, sha = load_probe_set(PROBES)
        self.assertEqual(first['probe_set_sha256'], sha)
        self.assertEqual(len(first['rows']), len(probes))
        self.assertEqual(first['generation'], 1)
        for kind in ('immediate_win', 'must_block', 'vcf'):
            summary = first['summary'][kind]
            self.assertAlmostEqual(summary['mass_lift'],
                                   summary['correct_mass'] / summary['uniform_mass'])
            self.assertTrue(0.0 <= summary['top1'] <= summary['top3'] <= 1.0)
        value = first['value_overall']
        self.assertEqual(value['win_probes'] + value['loss_probes'],
                         sum(p.value_sign is not None for p in probes))
        self.assertGreater(first['forbidden_diagnostic']['positions'], 0)

    def test_checkpoint_eval_uses_generation_independent_openings(self):
        checkpoints = [self.run_dir / 'checkpoints' / name
                       for name in ('checkpoint_init.pt', 'checkpoint_gen001.pt')]
        results = [evaluate_checkpoint(path, ['random'], pairs=1, seed=3,
                                       simulations=2, log=lambda _: None)
                   for path in checkpoints]
        openings = [[g['moves'][:3] for g in r['opponents']['random']['games']]
                    for r in results]
        self.assertEqual(openings[0], openings[1])
        summary = results[1]['opponents']['random']['summary']
        self.assertEqual(summary['games'], 2)
        self.assertEqual(summary['illegal_moves'], 0)
        self.assertEqual(results[1]['model_search']['num_simulations'], 2)


if __name__ == '__main__':
    unittest.main()
