import unittest
from unittest.mock import patch

from model.config import action_to_coordinate, coordinate_to_action, ModelConfig
from renju import BLACK, WHITE, Game

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from model.encoding import encode_game
    from model.masking import legal_moves_to_mask


class ActionTest(unittest.TestCase):
    def test_all_coordinates(self):
        for row in range(15):
            for col in range(15):
                action = coordinate_to_action(row, col)
                self.assertEqual(action, row * 15 + col)
                self.assertEqual(action_to_coordinate(action), (row, col))

    def test_invalid(self):
        for action in (-1, 225, 1.5, True, '0', None):
            with self.assertRaises(ValueError):
                action_to_coordinate(action)
        for row, col in ((-1, 0), (0, 15), (1.5, 0), (0, True)):
            with self.assertRaises(ValueError):
                coordinate_to_action(row, col)

    def test_config_contract(self):
        for kwargs in ({'board_size': 9}, {'input_planes': 5}, {'blocks': 0},
                       {'channels': True}, {'value_hidden': -1}):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)


@unittest.skipIf(torch is None, 'requires torch')
class EncodingTest(unittest.TestCase):
    def test_opening(self):
        x = encode_game(Game())
        self.assertEqual(tuple(x.shape), (6, 15, 15))
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(x[:3].sum(), 0)
        self.assertTrue((x[3:5] == 1).all())
        self.assertEqual(x[5].sum(), 1)
        self.assertEqual(x[5, 7, 7], 1)

    def test_relative_stones_history_and_mask_reuse(self):
        g = Game()
        g.play(7, 7)
        for player in (WHITE, BLACK):
            g.to_play = player
            mask = legal_moves_to_mask(g.legal_moves())
            before = [row[:] for row in g.board]
            with patch.object(g, 'legal_moves', side_effect=AssertionError('recomputed')):
                x = encode_game(g, mask)
            self.assertEqual(x[0, 7, 7], player == BLACK)
            self.assertEqual(x[1, 7, 7], player == WHITE)
            self.assertEqual(x[2].sum(), 1)
            self.assertEqual(x[2, 7, 7], 1)
            self.assertTrue((x[3] == (player == BLACK)).all())
            self.assertTrue(torch.equal(x[5].bool().flatten(), mask))
            self.assertEqual(before, g.board)

    def test_terminal_and_full_board(self):
        for full in (False, True):
            g = Game()
            if full:
                g.board = [[WHITE] * 15 for _ in range(15)]
            else:
                g.done = True
            self.assertEqual(encode_game(g)[5].sum(), 0)

    def test_invalid_mask(self):
        for mask in (torch.zeros(225), torch.zeros(1, 225, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                encode_game(Game(), mask)
