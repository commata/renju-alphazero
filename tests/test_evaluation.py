import random
import unittest
from pathlib import Path
from unittest.mock import patch

from renju import BLACK, Game
from training.config import evaluation_search_config, load_config

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from training import evaluation as evaluation_module
    from training.evaluation import (PUCTAgent, evaluate_generation, make_opening,
                                     opponents_for_generation, summarize_games)
    from training.training_state import COMPONENT_RNGS, init_training_state


def fake_game(color, result, moves):
    return {'model_color': color, 'result': result, 'moves': moves, 'length': len(moves),
            'opening_plies': 3, 'model_move_seconds': [0.2, 0.4],
            'opponent_move_seconds': [0.1]}


@unittest.skipIf(torch is None, 'requires torch')
class EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        cls.state = init_training_state(cls.config)
        cls.rng_before = {n: getattr(cls.state, n).getstate() for n in COMPONENT_RNGS}
        cls.global_before = random.getstate()
        cls.state.model.train()
        cls.previous = init_training_state(cls.config).model
        cls.results = evaluate_generation(cls.state.model, cls.previous, 0, cls.config, 1)

    def test_all_configured_opponents_colour_balanced(self):
        self.assertEqual(sorted(self.results['opponents']), ['previous', 'random', 'tactical'])
        for name, data in self.results['opponents'].items():
            summary = data['summary']
            self.assertEqual(summary['black']['games'], 1, name)
            self.assertEqual(summary['white']['games'], 1, name)
            self.assertEqual(summary['wins'] + summary['losses'] + summary['draws'], 2)
            for key in ('wins', 'losses', 'draws'):
                self.assertEqual(summary[key], summary['black'][key] + summary['white'][key])
            self.assertEqual(summary['illegal_moves'], 0)

    def test_opening_pairs_share_opening_and_swap_colours(self):
        for data in self.results['opponents'].values():
            black, white = data['games']
            self.assertEqual((black['model_color'], white['model_color']), ('black', 'white'))
            self.assertEqual(black['opening_plies'], 3)
            self.assertEqual(black['moves'][:3], white['moves'][:3])
            self.assertEqual(black['moves'][0], [7, 7])

    def test_games_replay_legally_with_consistent_result(self):
        for data in self.results['opponents'].values():
            for record in data['games']:
                game = Game()
                for move in record['moves']:
                    game.play(*move)
                self.assertTrue(game.done)
                self.assertEqual(game.winner, record['winner'])

    def test_evaluation_is_stateless_and_reproducible(self):
        again = evaluate_generation(self.state.model, self.previous, 0, self.config, 1)
        for name in self.results['opponents']:
            self.assertEqual([g['moves'] for g in self.results['opponents'][name]['games']],
                             [g['moves'] for g in again['opponents'][name]['games']])

    def test_training_rngs_untouched(self):
        for name in COMPONENT_RNGS:
            self.assertEqual(getattr(self.state, name).getstate(), self.rng_before[name])
        self.assertEqual(random.getstate(), self.global_before)

    def test_model_modes_restored(self):
        self.assertTrue(self.state.model.training)
        self.assertTrue(self.previous.training)

    def test_search_is_noise_free_argmax(self):
        search = self.results['model_search']
        self.assertFalse(search['noise_enabled'])
        self.assertEqual(search['temperature_moves'], 0)
        self.assertEqual(self.results['opponents']['previous']['opponent_config']['search'],
                         search)
        with self.assertRaises(ValueError):
            PUCTAgent('x', self.previous, evaluation_search_config(self.config).__class__(
                num_simulations=2, noise_enabled=True), 'cpu')

    def test_puct_agent_never_uses_rng(self):
        agent = PUCTAgent('m', self.previous, evaluation_search_config(self.config))
        game = Game()
        for move in ((7, 7), (7, 8), (8, 8)):
            game.play(*move)
        with patch.object(random.Random, 'random', side_effect=AssertionError('rng used')):
            first = agent.select_move(game)
        self.assertEqual(first, agent.select_move(game))
        self.assertIn(first, game.legal_moves())


@unittest.skipIf(torch is None, 'requires torch')
class EvaluationHelpersTest(unittest.TestCase):
    def test_opening_is_legal_and_near_centre(self):
        for seed in range(20):
            opening = make_opening(random.Random(seed), 2, 2)
            self.assertEqual(opening[0], (7, 7))
            self.assertEqual(len(opening), 3)
            game = Game()
            for move in opening:
                game.play(*move)
            self.assertTrue(all(max(abs(r - 7), abs(c - 7)) <= 2 for r, c in opening))
            self.assertEqual(game.to_play, -BLACK)

    def test_summary_aggregation_and_duplicates(self):
        games = [fake_game('black', 'win', [[7, 7], [7, 8]]),
                 fake_game('white', 'loss', [[7, 7], [6, 8]]),
                 fake_game('black', 'draw', [[7, 7], [7, 8]]),
                 fake_game('white', 'win', [[7, 7], [8, 8]])]
        summary = summarize_games(games)
        self.assertEqual((summary['wins'], summary['losses'], summary['draws']), (2, 1, 1))
        self.assertEqual(summary['black'], {'wins': 1, 'losses': 0, 'draws': 1, 'games': 2})
        self.assertEqual(summary['white'], {'wins': 1, 'losses': 1, 'draws': 0, 'games': 2})
        self.assertEqual((summary['unique_games'], summary['duplicate_games']), (3, 1))
        self.assertAlmostEqual(summary['average_model_move_seconds'], 0.3)
        self.assertAlmostEqual(summary['average_game_length'], 2)

    def test_mcts_v6_only_in_final_generation(self):
        config = load_config(ROOT / 'configs' / 'stage6_mvp.yaml')
        self.assertNotIn('mcts_v6', opponents_for_generation(config, 1, 2))
        self.assertIn('mcts_v6', opponents_for_generation(config, 2, 2))
        test_config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        self.assertNotIn('mcts_v6', opponents_for_generation(test_config, 1, 1))

    def test_v6_opponent_uses_frozen_preset(self):
        from agents import MCTSV6Agent
        from search.mcts_v6 import V5_FINAL

        config = load_config(ROOT / 'configs' / 'stage6_mvp.yaml')
        factory, opponent_config = evaluation_module._opponent_factories(config, None)['mcts_v6']
        agent = factory(123)
        self.assertIsInstance(agent, MCTSV6Agent)
        for key, value in V5_FINAL.items():
            self.assertEqual(getattr(agent, key), value)
            self.assertEqual(opponent_config['frozen_config'][key], value)


if __name__ == '__main__':
    unittest.main()
