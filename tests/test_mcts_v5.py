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

    def test_black_completion_cross_overline_is_winning_completion(self):
        game = position([(7,3),(7,4),(7,5),(4,7),(5,7),(6,7),(8,7),(9,7)])
        self.assertIn((7,7), _four_completions(game, BLACK, (7,6)))
        white = position(white=[(7,3),(7,4),(7,5),(7,8)])
        self.assertIn((7,7), _four_completions(white, WHITE, (7,6)))

    def test_closed_four_is_blockable(self):
        game = position([(7,4),(7,5),(7,6)], [(7,3)])
        self.assertFalse(_is_unstoppable_four(game, BLACK, (7,7)))


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


class V5BoundaryTest(unittest.TestCase):
    def test_temporary_board_restored_on_exception(self):
        from unittest.mock import patch
        game = position([(7,6),(7,7),(7,8)])
        before = deepcopy(vars(game))
        with patch('search.mcts_v5._placed_completions', side_effect=RuntimeError('probe')):
            with self.assertRaises(RuntimeError):
                _four_completions(game, BLACK, (7,5))
        self.assertEqual(vars(game), before)
        with patch('search.mcts_v5._winning_moves', side_effect=RuntimeError('probe')):
            with self.assertRaises(RuntimeError):
                _is_unstoppable_four(game, BLACK, (7,5))
        self.assertEqual(vars(game), before)

    def test_root_legal_moves_once_and_diagnostics_reset(self):
        from unittest.mock import patch
        from search import mcts_search_v5
        game = Game()
        diag = SearchDiagnostics(forced_policy_stage=5, stage5_multi_root_injection=True)
        with patch.object(game, 'legal_moves', wraps=game.legal_moves) as legal, \
             patch('search.mcts_v5._search_candidates_v321', return_value=[]), \
             patch('search.mcts_v5._rollout_v321', return_value=None):
            mcts_search_v5(game, simulations=1, tactical_simulations=1, diagnostics=diag)
            self.assertEqual(legal.call_count, 1)
        self.assertIsNone(diag.forced_policy_stage)
        self.assertFalse(diag.stage5_multi_root_injection)
        forced = position([(7,6),(7,7),(7,8)])
        mcts_search_v5(forced, simulations=1, tactical_simulations=1, diagnostics=diag)
        self.assertEqual(diag.forced_policy_stage, 3)
        self.assertEqual(diag.selected_simulations, 0)
        self.assertIsNone(diag.best_root_tactical_score)

    def test_real_engine_all_completions_and_defenses(self):
        from search.mcts import _is_legal_for_player, _wins_for_player
        for game, player, creator in [(trap(WHITE),WHITE,(8,6)),
                                      (position([(7,6),(7,7),(7,8)]),BLACK,(7,5))]:
            points = _four_completions(game, player, creator)
            state = deepcopy(game)
            state.to_play = player
            state.play(*creator)
            self.assertTrue(points)
            for point in points:
                self.assertTrue(_is_legal_for_player(state, player, point))
                self.assertTrue(_wins_for_player(state, player, point))
                if _is_legal_for_player(state, -player, point):
                    blocked = deepcopy(state)
                    blocked.play(*point)
                    winning = [p for p in blocked.legal_moves()
                               if _wins_for_player(blocked, player, p)]
                    self.assertTrue(winning)

    def test_script_defaults_and_threshold_cli(self):
        from scripts.run_mcts_v5_vs_v41 import make_parser
        parser = make_parser()
        self.assertEqual(parser.parse_args([]).games, 1)
        args = parser.parse_args(['--stage','b','--tactical-score-threshold','none'])
        self.assertIsNone(args.tactical_score_threshold)



class V5RunnerTest(unittest.TestCase):
    def test_early_draw_disabled_by_default(self):
        from scripts.run_mcts_v5_vs_v41 import make_parser
        self.assertFalse(make_parser().parse_args([]).early_draw)

    def test_early_draw_boundary_streak_reset_and_state(self):
        from scripts.run_mcts_v5_vs_v41 import EarlyDrawTracker
        tracker = EarlyDrawTracker()
        game = Game()
        game.history = [(0,0)] * 99
        self.assertFalse(tracker.update(game))
        for ply in range(100,109):
            game.history = [(0,0)] * ply
            self.assertFalse(tracker.update(game))
        game.history.append((0,0))
        before = deepcopy(vars(game))
        self.assertTrue(tracker.update(game))
        self.assertEqual(vars(game), before)
        for player in (BLACK,WHITE):
            game.board[7][5:8] = [player]*3
            self.assertFalse(tracker.update(game))
            self.assertEqual(tracker.streak, 0)
        game.board[7][5:8] = [0]*3
        game.done = True
        self.assertFalse(tracker.update(game))

    def test_last_threat_classification(self):
        from evaluation.match import GameResult
        from scripts.run_mcts_v5_vs_v41 import last_threat
        history = ((7,5),(0,0),(7,6),(0,1),(7,7),(1,0),(7,8),(7,4),(7,9))
        result = GameResult(BLACK, len(history), 0, 'b', 'w', history)
        self.assertEqual(last_threat(result)['type'], 'open_four')
        self.assertEqual(last_threat(result)['creator_ply'], 7)



class V5OracleTest(unittest.TestCase):
    def test_bounded_wins_match_exhaustive_engine(self):
        from search.mcts import _wins_for_player
        from search.mcts_v5 import _winning_moves
        games = [trap(BLACK), trap(WHITE),
                 position([(7,3),(7,4),(7,5),(7,6),(5,7),(6,7),(8,7),(9,7),(10,7)]),
                 position(white=[(7,3),(7,4),(7,5),(7,7),(7,8)], player=WHITE)]
        for game in games:
            for player in (BLACK,WHITE):
                game.to_play = player
                expected = [m for m in game.legal_moves() if _wins_for_player(game,player,m)]
                self.assertEqual(_winning_moves(game, player), expected)

    def test_forbidden_creator_is_not_double_threat(self):
        from search.mcts_v5 import _double_threat_moves
        game = position([(7,5),(7,6),(5,7),(6,7)])
        self.assertNotIn((7,7), _double_threat_moves(game, BLACK))

    def test_non_creator_defense_minimizes_remaining_threats(self):
        from search.mcts_v5 import _threat_windows
        game = position(white=[(7,6),(7,7),(7,8)])
        creators = _unstoppable_four_moves(game, WHITE)
        legal = set(game.legal_moves())
        defenses = set(creators)
        for window in _threat_windows(game, WHITE):
            if set(creators).intersection(window):
                defenses.update(p for p in window if game.board[p[0]][p[1]] == 0)
        counts = {}
        for move in defenses & legal:
            state = deepcopy(game)
            state.play(*move)
            counts[move] = len(_unstoppable_four_moves(state, WHITE))
        chosen = _forced_v5_move(game)
        self.assertEqual(counts[chosen], min(counts.values()))

    def test_summary_and_log_schema_without_matches(self):
        import json
        import csv
        from dataclasses import asdict
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from evaluation.match import GameResult, MatchResult
        from evaluation import save_match_logs
        from scripts.run_mcts_v5_vs_v41 import summarize
        result = GameResult(None, 1, 0.1, 'MCTS-v5a', 'MCTS-v4.1', ((7,7),))
        match = MatchResult(1,0,0,1,1,1,0.1,10,(result,),42)
        records = [dict(last_threat=dict(type='other'), early_draw_reason=None)]
        decisions = [dict(agent='MCTS-v5a', seconds=0.1,
                          **asdict(SearchDiagnostics(forced_policy_stage=3)))]
        summary = summarize([('test',match)], records, decisions)
        self.assertEqual(summary['forced_stage_3'],1)
        self.assertEqual(summary['draws'],1)
        self.assertIsNone(summary['best_tactical_score']['avg'])
        with TemporaryDirectory() as directory:
            save_match_logs(directory, [('test',match)], {})
            payload = json.loads((Path(directory)/'games.json').read_text(encoding='utf-8'))
            self.assertEqual(set(payload), {'config','matches','games'})
            with (Path(directory)/'moves.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(next(csv.reader(stream)),
                                 ['game_id','matchup','matchup_game','ply','player','row0','col0','row','col'])



class V5AgentTacticsTest(unittest.TestCase):
    def test_agent_end_to_end_required_positions(self):
        from agents import MCTSV5Agent
        cases = [(trap(WHITE), {(8,6)}, 3), (trap(BLACK), {(8,6)}, 4),
                 (position([(10,10),(10,11),(11,10)],[(7,5),(7,6),(5,7),(6,7)]), {(7,7)}, 5),
                 (position([(7,6),(7,7),(7,8)],[(3,3),(3,4),(4,3)]), {(7,5),(7,9)}, 3),
                 (position(white=[(7,6),(7,7),(7,8)]), {(7,5),(7,9)}, 4),
                 (position([(4,c) for c in range(3,7)],[(7,6),(7,7),(7,8)]), {(4,2),(4,7)}, 1)]
        for stage in ('a','b','c'):
            for game, expected, forced_stage in cases:
                with self.subTest(stage=stage, forced_stage=forced_stage):
                    before = deepcopy(vars(game))
                    agent = MCTSV5Agent(stage=stage, simulations=1, tactical_simulations=1)
                    self.assertIn(agent.select_move(game), expected)
                    self.assertEqual(agent.diagnostics.forced_policy_stage, forced_stage)
                    self.assertEqual(vars(game), before)



class V5OptimizationTest(unittest.TestCase):
    def test_direction_prefilter_matches_original_candidates(self):
        from search.mcts_v5 import _window_candidates, _double_threat_moves
        from search.mcts_v321 import _fast_pattern_features_for_move
        from random import Random
        games = [trap(BLACK), trap(WHITE),
                 position(white=[(7,5),(7,6),(5,7),(6,7)]),
                 position(white=[(7,4),(7,5),(7,6),(4,7),(5,7),(6,7)])]
        random = Random(9)
        for _ in range(5):
            cells = random.sample([(r,c) for r in range(15) for c in range(15)], 50)
            games.append(position(cells[:25],cells[25:]))
        for game in games:
            for player in (BLACK,WHITE):
                expected = []
                for move in _window_candidates(game, player, 2):
                    f = _fast_pattern_features_for_move(game, player, move)
                    if f.legal and (f.open_three_directions >= 2 or f.four_directions >= 2 or f.has_four_three):
                        expected.append(move)
                self.assertEqual(_double_threat_moves(game, player), expected)

    def test_cached_counterwin_candidates_recheck_black_legality(self):
        from search.mcts_v5 import _window_candidates, _winning_moves
        game = position([(7,3),(7,4),(7,5),(7,6),(5,7),(6,7),(8,7),(9,7),(10,7)])
        candidates = _window_candidates(game, BLACK, 4)
        self.assertIn((7,7), candidates)
        self.assertNotIn((7,7), _winning_moves(game, BLACK))
        # A white creator can occupy another black candidate; the cached
        # structural pool must skip occupied points and recheck all survivors.
        game.board[7][2] = WHITE
        self.assertEqual(_winning_moves(game, BLACK, candidates), _winning_moves(game, BLACK))


if __name__ == '__main__':
    unittest.main()
