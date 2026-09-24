from copy import deepcopy
import unittest

from renju import BLACK, WHITE, Game
from renju.rules import forbidden_reason
from search.mcts_v5 import (
    _four_completions, _is_unstoppable_four, _unstoppable_four_moves,
    _forced_v5_move, _RootContext, SearchDiagnostics,
)


def position(black=(), white=(), player=BLACK):
    game = Game()
    for color, moves in ((BLACK, black), (WHITE, white)):
        for r, c in moves:
            game.board[r][c] = color
    game.to_play = player
    return game


def trap(player):
    return position([(7,3),(7,4),(7,5),(7,7),(7,8),(3,6)],
                    [(4,6),(5,6),(6,6)], player)


class V5PolicyTest(unittest.TestCase):
    def test_a_forbidden_completion_attack_and_defense(self):
        for player in (WHITE, BLACK):
            game = trap(player)
            before = deepcopy(vars(game))
            self.assertIsNotNone(forbidden_reason(game.board, 7, 6))
            self.assertEqual(_unstoppable_four_moves(game, WHITE), [(8,6)])
            self.assertEqual(_forced_v5_move(game), (8,6))
            self.assertEqual(vars(game), before)

    def test_c_white_double_three(self):
        game = position([(10,10),(10,11),(11,10)], [(7,5),(7,6),(5,7),(6,7)])
        context = _RootContext(game.legal_moves(), SearchDiagnostics())
        self.assertEqual(_forced_v5_move(game, context=context), (7,7))
        self.assertEqual(context.diagnostics.forced_policy_stage, 5)

    def test_e_own_open_four(self):
        game = position([(7,6),(7,7),(7,8)], [(3,3),(3,4),(4,3)])
        self.assertIn(_forced_v5_move(game), {(7,5),(7,9)})

    def test_v4_open_three_defense(self):
        game = position(white=[(7,6),(7,7),(7,8)])
        self.assertIn(_forced_v5_move(game), {(7,5),(7,9)})

    def test_v4_immediate_win_priority(self):
        game = position([(4,c) for c in range(3,7)], [(7,6),(7,7),(7,8)])
        self.assertIn(_forced_v5_move(game), {(4,2),(4,7)})

    def test_opponent_immediate_win_overrides_four(self):
        game = position([(7,6),(7,7),(7,8)], [(3,c) for c in range(3,7)])
        self.assertFalse(_is_unstoppable_four(game, BLACK, (7,5)))
        self.assertIn(_forced_v5_move(game), {(3,2),(3,7)})

    def test_white_double_four_black_creator_illegal(self):
        stones = [(7,4),(7,5),(7,6),(4,7),(5,7),(6,7)]
        white = position(white=stones, player=WHITE)
        self.assertTrue(_is_unstoppable_four(white, WHITE, (7,7)))
        black = position(black=stones)
        self.assertFalse(_is_unstoppable_four(black, BLACK, (7,7)))

    def test_black_completion_cross_overline_rejected(self):
        game = position([(7,3),(7,4),(7,5),(4,7),(5,7),(6,7),(8,7),(9,7)])
        self.assertNotIn((7,7), _four_completions(game, BLACK, (7,6)))
        white = position(white=[(7,3),(7,4),(7,5),(7,8)])
        self.assertIn((7,7), _four_completions(white, WHITE, (7,6)))

    def test_closed_four_is_blockable(self):
        game = position([(7,4),(7,5),(7,6)], [(7,3)])
        self.assertFalse(_is_unstoppable_four(game, BLACK, (7,7)))


if __name__ == '__main__':
    unittest.main()

class V5SearchTest(unittest.TestCase):
    def test_presets_and_explicit_none(self):
        from agents import MCTSV5Agent
        from search.mcts_v5 import V5_PRESETS
        for stage, config in V5_PRESETS.items():
            agent = MCTSV5Agent(stage=stage)
            self.assertEqual(agent.name, f'MCTS-v5{stage}')
            for key, value in config.items():
                self.assertEqual(getattr(agent, key), value)
        self.assertEqual(MCTSV5Agent().stage, 'c')
        self.assertIsNone(MCTSV5Agent(tactical_score_threshold=None).tactical_score_threshold)

    def test_search_determinism_state_and_global_random(self):
        import random
        from agents import MCTSV5Agent
        game = Game()
        game.play(7,7)
        before = deepcopy(vars(game))
        state = random.getstate()
        options = dict(seed=123, simulations=1, tactical_simulations=1)
        first = MCTSV5Agent(**options).select_move(game)
        second = MCTSV5Agent(**options).select_move(game)
        self.assertEqual(first, second)
        self.assertIn(first, game.legal_moves())
        self.assertEqual(vars(game), before)
        self.assertEqual(random.getstate(), state)

    def test_validation_agent_and_search(self):
        from agents import MCTSV5Agent
        from search import mcts_search_v5
        cases = [dict(stage='d')]
        for key in ('simulations','tactical_simulations','candidate_limit','initial_width',
                    'neighborhood_radius','priority_top_k'):
            cases.extend({key: value} for value in (0, -1, 1.5, True))
        cases += [dict(exploration=x) for x in (0,-1,float('nan'),float('inf'),'bad')]
        cases += [dict(tactical_score_threshold=x) for x in (True,1.5,'600')]
        cases += [dict(initial_width=21), dict(priority_top_k=21)]
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    MCTSV5Agent(**config)
                if 'stage' not in config:
                    with self.assertRaises(ValueError):
                        mcts_search_v5(Game(), **config)

    def test_adaptive_threshold_and_score_reuse(self):
        from unittest.mock import patch
        from search.mcts_v5 import mcts_search_v5
        from search import mcts_v321
        for threshold, expected, mode in ((None,1,'normal'), (0,2,'tactical'), (10**9,1,'normal')):
            diag = SearchDiagnostics()
            with patch('search.mcts_v321._v321_priority_score', wraps=mcts_v321._v321_priority_score) as score:
                with patch('search.mcts_v5._search_candidates_v321', return_value=[]), \
                     patch('search.mcts_v5._rollout_v321', return_value=None):
                    mcts_search_v5(Game(), simulations=1, tactical_simulations=2,
                                   tactical_score_threshold=threshold, diagnostics=diag)
                moves = [call.args[1] for call in score.call_args_list]
                self.assertEqual(len(moves), len(set(moves)))
            self.assertEqual(diag.selected_simulations, expected)
            self.assertEqual(diag.simulation_mode, mode)
            self.assertIsInstance(diag.best_root_tactical_score, int)

    def test_multi_injection_exceeds_limit(self):
        from search.mcts_v5 import _double_threat_moves, _root_candidates_v5
        game = position(white=[(4,2),(4,3),(2,4),(3,4),
                               (10,8),(10,9),(8,10),(9,10)])
        danger = _double_threat_moves(game, WHITE)
        self.assertGreaterEqual(len(danger), 2)
        context = _RootContext(game.legal_moves(), SearchDiagnostics())
        self.assertIsNone(_forced_v5_move(game, context=context))
        self.assertTrue(context.diagnostics.stage5_multi_root_injection)
        moves, score = _root_candidates_v5(game, context, 1, 2)
        self.assertGreater(len(moves), 1)
        self.assertEqual(set(moves[:len(danger)]), set(danger))

    def test_forbidden_only_immediate_block_falls_through(self):
        from agents import MCTSV5Agent
        game = trap(BLACK)
        game.board[8][6] = WHITE
        agent = MCTSV5Agent(simulations=1, tactical_simulations=1)
        move = agent.select_move(game)
        self.assertNotEqual(move, (7,6))
        self.assertIn(move, game.legal_moves())

    def test_terminal_no_legal_move(self):
        from agents import MCTSV5Agent
        from renju import IllegalMove
        game = Game()
        game.done = True
        with self.assertRaises(IllegalMove):
            MCTSV5Agent(simulations=1, tactical_simulations=1).select_move(game)

    def test_random_agent_both_colors_two_each(self):
        from agents import MCTSV5Agent, RandomAgent
        from evaluation import run_match
        factory = lambda seed: MCTSV5Agent(seed=seed, simulations=1, tactical_simulations=1)
        for black, white in ((factory, RandomAgent), (RandomAgent, factory)):
            result = run_match(black, white, games=2, seed=42)
            self.assertEqual(result.games, 2)
            self.assertTrue(all(r.number_of_moves > 0 for r in result.results))
