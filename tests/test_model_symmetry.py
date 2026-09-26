import unittest

from renju import BLACK, WHITE, Game
from renju.rules import forbidden_reason
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from model.encoding import encode_game
    from model.masking import legal_moves_to_mask
    from model.symmetry import (SYMMETRIES, inverse_symmetry, transform_action,
                                transform_coordinate, transform_mask, transform_policy,
                                transform_spatial)


@unittest.skipIf(torch is None, 'install the neural extra')
class SymmetryTest(unittest.TestCase):
    def test_asymmetric_alignment_and_inverse(self):
        game = Game()
        for move in ((7, 7), (2, 11), (9, 3), (1, 4)):
            game.play(*move)
        encoded = encode_game(game)
        action = 2 * 15 + 9
        mask = torch.zeros(225, dtype=torch.bool)
        mask[action] = True
        policy = mask.float()
        # Independent explicit coordinate expectations prevent consistent wrong conventions.
        expected = [(2, 9), (5, 2), (12, 5), (9, 12),
                    (2, 5), (9, 2), (12, 9), (5, 12)]
        for symmetry in SYMMETRIES:
            inv = inverse_symmetry(symmetry)
            row, col = expected[symmetry]
            self.assertEqual(transform_coordinate(2, 9, symmetry), (row, col))
            self.assertEqual(transform_action(action, symmetry), row * 15 + col)
            p = transform_policy(policy, symmetry)
            m = transform_mask(mask, symmetry)
            x = transform_spatial(encoded, symmetry)
            self.assertEqual(p[row * 15 + col], 1)
            self.assertTrue(m[row * 15 + col])
            for plane, original in ((0, (9, 3)), (1, (2, 11)), (2, (1, 4))):
                r, c = transform_coordinate(*original, symmetry)
                self.assertEqual(x[plane, r, c], 1)
            self.assertTrue(torch.equal(x[3:5], encoded[3:5]))
            self.assertTrue(torch.equal(x[5].flatten(), transform_policy(encoded[5].flatten(), symmetry)))
            self.assertTrue(torch.equal(transform_spatial(x, inv), encoded))
            self.assertTrue(torch.equal(transform_policy(p, inv), policy))
            self.assertTrue(torch.equal(transform_mask(m, inv), mask))
            for a in range(225):
                self.assertEqual(transform_action(transform_action(a, symmetry), inv), a)
            board = torch.tensor(game.board)
            self.assertTrue(torch.equal(transform_spatial(transform_spatial(board, symmetry), inv), board))
            batch = torch.stack([policy, policy * .5])
            self.assertTrue(torch.equal(transform_policy(transform_policy(batch, symmetry), inv), batch))

    def test_renju_legal_masks_all_symmetries(self):
        patterns = [
            ([(7, 6), (7, 8), (6, 7), (8, 7)], '삼삼'),
            ([(7, c) for c in (5, 6, 8)] + [(r, 7) for r in (5, 6, 8)], '사사'),
            ([(7, c) for c in (3, 4, 5, 6, 8)], '장목'),
        ]
        for stones, reason in patterns:
            for player in (BLACK, WHITE):
                game = Game()
                game.to_play = player
                for r, c in stones + [(1, 3)]:
                    game.board[r][c] = BLACK
                for r, c in ((2, 11), (12, 4)):
                    game.board[r][c] = WHITE
                game.history = [(12, 4)]
                self.assertEqual(forbidden_reason(game.board, 7, 7), reason)
                original = legal_moves_to_mask(game.legal_moves())
                for symmetry in SYMMETRIES:
                    with self.subTest(reason=reason, player=player, symmetry=symmetry):
                        transformed = Game()
                        transformed.to_play = player
                        transformed.board = transform_spatial(torch.tensor(game.board), symmetry).tolist()
                        transformed.history = [transform_coordinate(12, 4, symmetry)]
                        actual = legal_moves_to_mask(transformed.legal_moves())
                        self.assertTrue(torch.equal(transform_mask(original, symmetry), actual))
                        self.assertTrue(torch.equal(transform_spatial(encode_game(game, original), symmetry),
                                                    encode_game(transformed, actual)))
        opening = legal_moves_to_mask(Game().legal_moves())
        for symmetry in SYMMETRIES:
            self.assertTrue(torch.equal(transform_mask(opening, symmetry), opening))

    def test_invalid(self):
        for symmetry in (-1, 8, True, 1.5):
            with self.assertRaises(ValueError):
                transform_coordinate(0, 0, symmetry)
        with self.assertRaises(ValueError):
            transform_spatial(torch.zeros(14, 15), 0)
        with self.assertRaises(ValueError):
            transform_policy(torch.zeros(224), 0)
        with self.assertRaises(ValueError):
            transform_mask(torch.zeros(225), 0)
