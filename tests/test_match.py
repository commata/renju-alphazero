import unittest

from agents import RandomAgent, TacticalAgent
from evaluation import play_game, run_match
from renju import BLACK, WHITE, Game, IllegalMove, SIZE


class MatchTest(unittest.TestCase):
    def test_random_match_completes_and_seed_reproduces_histories(self):
        first = run_match(RandomAgent, RandomAgent, games=2, seed=42)
        second = run_match(RandomAgent, RandomAgent, games=2, seed=42)
        for result in (first, second):
            self.assertEqual(result.games, 2)
            self.assertEqual(result.black_wins + result.white_wins + result.draws, 2)
            self.assertEqual(result.total_moves, sum(g.number_of_moves for g in result.results))
            self.assertEqual(result.average_moves, result.total_moves / 2)
            self.assertGreater(result.elapsed_seconds, 0)
            self.assertAlmostEqual(result.games_per_second, 2 / result.elapsed_seconds)
        for left, right in zip(first.results, second.results):
            self.assertEqual(left.winner, right.winner)
            self.assertEqual(left.history, right.history)
            self.assertEqual(left.number_of_moves, right.number_of_moves)
            self.assertEqual(left.black_agent, "Random")
            self.assertEqual(left.white_agent, "Random")
            game = Game()
            for move in left.history:
                game.play(*move)
            self.assertTrue(game.done)
            self.assertEqual(game.winner, left.winner)
            self.assertEqual(game.legal_moves(), [])
            self.assertLessEqual(left.number_of_moves, SIZE * SIZE)

    def test_tactical_matches_both_colors_reproduce(self):
        for black, white in ((RandomAgent, TacticalAgent), (TacticalAgent, RandomAgent)):
            with self.subTest(black=black.name):
                first = run_match(black, white, games=1, seed=17)
                second = run_match(black, white, games=1, seed=17)
                left, right = first.results[0], second.results[0]
                self.assertEqual(left.history, right.history)
                self.assertEqual(left.winner, right.winner)
                self.assertIn(left.winner, (BLACK, WHITE, None))
                self.assertEqual(left.black_agent, black.name)
                self.assertEqual(left.white_agent, white.name)
                self.assertEqual(len(left.history), left.number_of_moves)

    def test_illegal_agent_move_propagates(self):
        class InvalidAgent:
            name = "Invalid"

            def select_move(self, game):
                return (0, 0)

        with self.assertRaisesRegex(IllegalMove, "Invalid returned illegal move"):
            play_game(InvalidAgent(), InvalidAgent())

    def test_malformed_move_raises_clear_error(self):
        class InvalidAgent:
            name = "Malformed"

            def select_move(self, game):
                return (0.5, 0)

        with self.assertRaisesRegex(IllegalMove, "Malformed returned an invalid move"):
            play_game(InvalidAgent(), RandomAgent())

    def test_invalid_games(self):
        for games in (0, -1, 1.5, True):
            with self.subTest(games=games), self.assertRaises(ValueError):
                run_match(RandomAgent, RandomAgent, games=games)


if __name__ == '__main__':
    unittest.main()
