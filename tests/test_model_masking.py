import unittest

from renju import Game, WHITE
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None
if torch is not None:
    from model.masking import legal_moves_to_mask, masked_softmax, policy_loss, value_loss


@unittest.skipIf(torch is None, 'install the neural extra')
class MaskingTest(unittest.TestCase):
    def test_mapping(self):
        moves = [(0, 14), (1, 0), (14, 14)]
        mask = legal_moves_to_mask(moves)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.nonzero().flatten().tolist(), [14, 15, 224])

    def test_distribution(self):
        torch.manual_seed(42)
        for shape in ((225,), (4, 225)):
            logits = torch.randn(shape) * 100
            mask = torch.rand(shape) > .5
            p = masked_softmax(logits, mask)
            self.assertTrue((p[~mask] == 0).all())
            self.assertTrue(((p.sum(-1) - 1).abs() < 1e-5).all())
            self.assertTrue(torch.isfinite(p).all())

    def test_empty_terminal_full_and_mixed_batch(self):
        full = Game()
        full.board = [[WHITE] * 15 for _ in range(15)]
        terminal = Game()
        terminal.done = True
        for game in (full, terminal):
            mask = legal_moves_to_mask(game.legal_moves())
            with self.assertRaises(ValueError):
                masked_softmax(torch.zeros(225), mask)
            with self.assertRaises(ValueError):
                masked_softmax(torch.zeros(2, 225), torch.stack([~mask, mask]))

    def test_policy_loss_matches_legal_only_and_zero_illegal_gradient(self):
        logits = torch.zeros(1, 225, requires_grad=True)
        mask = legal_moves_to_mask([(0, 0), (0, 1)]).unsqueeze(0)
        target = torch.zeros_like(logits)
        target[0, 0] = 1
        loss = policy_loss(logits, target, mask)
        self.assertAlmostEqual(loss.item(), torch.log(torch.tensor(2.)).item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[~mask] == 0).all())

    def test_invalid_targets(self):
        logits = torch.zeros(1, 225)
        mask = legal_moves_to_mask([(0, 0)]).unsqueeze(0)
        target = mask.float()
        bad = []
        for index, val in ((1, .1), (0, -1), (0, float('nan')), (0, float('inf')), (0, .8)):
            t = target.clone()
            t[0, index] = val
            bad.append(t)
        bad.append(target.flatten())
        for t in bad:
            with self.assertRaises(ValueError):
                policy_loss(logits, t, mask)
        for t in (torch.tensor([[1.1]]), torch.tensor([[float('nan')]]), torch.zeros(1)):
            with self.assertRaises(ValueError):
                value_loss(torch.zeros(1, 1), t)

    def test_invalid_logits_and_masks(self):
        for logits, mask in ((torch.zeros(225), torch.ones(225)),
                             (torch.zeros(1, 225), torch.ones(225, dtype=torch.bool)),
                             (torch.full((225,), float('inf')), torch.ones(225, dtype=torch.bool)),
                             (torch.zeros(224), torch.ones(224, dtype=torch.bool))):
            with self.assertRaises(ValueError):
                masked_softmax(logits, mask)
