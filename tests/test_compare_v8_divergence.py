import json
from pathlib import Path
import unittest

from renju import Game
from scripts.compare_v8_divergence import compare, first_divergence

ROOT = Path(__file__).resolve().parents[1]


def _record(ply, route='tree'):
    return {'ply': ply, 'route': route, 'v7_move': [0, 0], 'changed': False, 'seconds': 1.0,
            'root': {'checked': [[[0, 0], 'UNKNOWN']], 'rank': 1, 'switch': '', 'nodes': 5, 'calls': 1,
                     'exhausted': False},
            'vct': {'checked': [], 'nodes': 0, 'calls': 0, 'exhausted': False}}


def _game(pair, color, moves, v8_plies, result='win'):
    return {'pair': pair, 'v8_color': color, 'moves': moves, 'length': len(moves), 'result': result,
            'v8_moves': [_record(p) for p in v8_plies]}


class DivergenceTest(unittest.TestCase):
    def test_first_divergence(self):
        self.assertIsNone(first_divergence({'moves': [[7, 7], [7, 8]]}, {'moves': [[7, 7], [7, 8]]}))
        self.assertEqual(first_divergence({'moves': [[7, 7], [7, 8]]}, {'moves': [[7, 7], [8, 8]]}), 1)
        self.assertEqual(first_divergence({'moves': [[7, 7]]}, {'moves': [[7, 7], [8, 8]]}), 1)

    def test_compare_counts_identical_and_divergent_games(self):
        same = [[7, 7], [7, 8], [8, 8]]
        a = {'seed': 1, 'opponent': 'v8:b_only', 'games': [
            _game(0, 'black', same, [0, 2]), _game(1, 'black', [[7, 7], [7, 8], [8, 8]], [0, 2])]}
        b = {'seed': 1, 'opponent': 'v8:b_only', 'games': [
            _game(0, 'black', same, [0, 2]), _game(1, 'black', [[7, 7], [7, 8], [6, 6]], [0, 2], 'loss')]}
        result = compare(a, b, node_limit=10, node_budget=10, check=False)
        self.assertEqual((result['identical_games'], result['divergent_games']), (1, 1))
        row = result['divergences'][0]
        self.assertEqual((row['ply'], row['by'], row['a']['move'], row['b']['move']), (2, 'v8', [8, 8], [6, 6]))
        self.assertEqual(row['b']['decision']['rank'], 1)
        with self.assertRaises(ValueError):
            compare(a, {**b, 'opponent': 'v7'}, node_limit=10, node_budget=10, check=False)


class DivergenceFixtureTest(unittest.TestCase):
    def test_probes_replay_and_child_order_is_legal(self):
        data = json.loads((ROOT / 'tests/fixtures/v8c_divergence_probes_v1.json').read_text(encoding='utf-8'))
        self.assertEqual(len(data['probes']), 2)
        for probe in data['probes']:
            game = Game()
            for move in probe['moves']:
                game.play(*move)
            legal = set(game.legal_moves())
            self.assertEqual(probe['child_order'][0], probe['v7_move'])
            for move in probe['child_order'] + probe['avoid_moves']:
                self.assertIn(tuple(move), legal)


if __name__ == '__main__':
    unittest.main()
