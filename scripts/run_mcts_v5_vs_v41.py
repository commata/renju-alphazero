"""V5 vs V4.1: one game per color by default; diagnostics live in summary.json."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from random import Random
from time import perf_counter

from agents import MCTSV41Agent, MCTSV5Agent
from evaluation import default_log_dir, save_match_logs
from evaluation.match import GameResult, MatchResult
from renju import BLACK, WHITE, Game, IllegalMove
from renju.rules import DIRECTIONS
from search.mcts import _is_legal_for_player
from search.mcts_v321 import _fast_winning_extensions_in_direction
from search.mcts_v5 import V5_PRESETS, _four_completions, _threat_windows


def _replay_for_analysis(history) -> Game:
    """Rebuild historical logs without re-enforcing today's opening rule.

    Old benchmark logs may predate the forced-center opening. This helper is only
    for offline structural analysis; live games still go through Game.play().
    """
    game = Game()
    game.board = [[0] * 15 for _ in range(15)]
    game.history = []
    game.to_play = BLACK
    game.winner = None
    game.done = False
    player = BLACK
    for row, col in history:
        if game.board[row][col] != 0:
            raise IllegalMove(f'historical replay contains occupied move: {(row, col)!r}')
        game.board[row][col] = player
        game.history.append((row, col))
        player = -player
    game.to_play = player
    return game


def last_threat(result: GameResult) -> dict:
    """Classify the winner's previous move before the final winning move.

    Evaluate before the losing reply: a reply can hide one end of an open four.
    Labels are structural, with precedence open four, double four, forbidden four.
    """
    if result.winner is None or len(result.history) < 3:
        return dict(type='other', creator_ply=None)
    index = len(result.history) - 3
    game = _replay_for_analysis(result.history[:index])
    player = result.winner
    move = result.history[index]
    completions = _four_completions(game, player, move)
    game.board[move[0]][move[1]] = player
    try:
        directions = [_fast_winning_extensions_in_direction(game.board, player, move, dr, dc)
                      for dr, dc in DIRECTIONS]
        if any(len(points) >= 2 for points in directions):
            kind = 'open_four'
        elif sum(bool(points) for points in directions) >= 2:
            kind = 'double_four'
        elif completions and all(not _is_legal_for_player(game, -player, p) for p in completions):
            kind = 'forbidden_point_four'
        else:
            kind = 'other'
        return dict(type=kind, creator_ply=index + 1, completions=sorted(completions))
    finally:
        game.board[move[0]][move[1]] = 0


class EarlyDrawTracker:
    """Optional script-only termination; never changes Game terminal flags."""

    def __init__(self):
        self.streak = 0

    def update(self, game: Game) -> bool:
        if game.done or len(game.history) < 100:
            self.streak = 0
        elif any(next(_threat_windows(game, player), None) is not None
                 for player in (BLACK, WHITE)):
            self.streak = 0
        else:
            self.streak += 1
        return self.streak >= 10


def run_color(black_factory, white_factory, games: int, seed: int, records: list,
              decisions: list, early_draw: bool = False) -> MatchResult:
    random = Random(seed)
    results = []
    pairs = [(black_factory(random.getrandbits(64)), white_factory(random.getrandbits(64)))
             for _ in range(games)]
    started = perf_counter()
    for black, white in pairs:
        game = Game()
        tracker = EarlyDrawTracker() if early_draw else None
        early_reason = None
        game_id = len(records) + 1
        game_started = perf_counter()
        while not game.done:
            agent = black if game.to_play == BLACK else white
            move_started = perf_counter()
            move = agent.select_move(game)
            elapsed = perf_counter() - move_started
            if (not isinstance(move, tuple) or len(move) != 2
                    or any(type(value) is not int for value in move)):
                raise IllegalMove(f'{agent.name} returned invalid move: {move!r}')
            record = dict(game_id=game_id, ply=len(game.history) + 1, agent=agent.name,
                          player=game.to_play, seconds=elapsed)
            if isinstance(agent, MCTSV5Agent):
                record.update(asdict(agent.diagnostics))
            decisions.append(record)
            game.play(*move)
            if tracker is not None and tracker.update(game):
                early_reason = 'no_unblocked_three_window_for_10_plies_at_ply_ge_100'
                break
        result = GameResult(game.winner, len(game.history), perf_counter() - game_started,
                            black.name, white.name, tuple(game.history))
        results.append(result)
        records.append(dict(game_id=game_id, winner=game.winner, moves=len(game.history),
                            seconds=result.elapsed_seconds, early_draw_reason=early_reason,
                            last_threat=last_threat(result)))
        print(f'Game {game_id}: {black.name} Black vs {white.name} White; '
              f'winner={game.winner}; moves={len(game.history)}; '
              f'seconds={result.elapsed_seconds:.3f}', flush=True)
    elapsed = perf_counter() - started
    total_moves = sum(r.number_of_moves for r in results)
    return MatchResult(games, sum(r.winner == BLACK for r in results),
                       sum(r.winner == WHITE for r in results),
                       sum(r.winner is None for r in results), total_moves,
                       total_moves / games, elapsed, games / elapsed if elapsed else 0,
                       tuple(results), seed)


def summarize(matches, records, decisions) -> dict:
    results = [r for _, match in matches for r in match.results]
    v5 = [d for d in decisions if d['agent'].startswith('MCTS-v5')]
    stages = Counter(d['forced_policy_stage'] for d in v5)
    scores = [d['best_root_tactical_score'] for d in v5 if d['best_root_tactical_score'] is not None]
    moves = sum(r.number_of_moves for r in results)
    elapsed = sum(r.elapsed_seconds for r in results)
    winners = Counter((r.black_agent if r.winner == BLACK else r.white_agent)
                      for r in results if r.winner is not None)
    times = {}
    for name in sorted({d['agent'] for d in decisions}):
        samples = [d['seconds'] for d in decisions if d['agent'] == name]
        times[name] = dict(moves=len(samples), seconds=sum(samples), seconds_per_move=sum(samples)/len(samples))
    return dict(
        games=len(results), v5_wins=sum(n for name,n in winners.items() if name.startswith('MCTS-v5')),
        v41_wins=winners['MCTS-v4.1'], draws=sum(r.winner is None for r in results),
        black_wins=sum(r.winner == BLACK for r in results),
        white_wins=sum(r.winner == WHITE for r in results),
        total_moves=moves, average_moves=moves/len(results), total_seconds=elapsed,
        seconds_per_game=elapsed/len(results), seconds_per_move=elapsed/moves if moves else 0,
        forced_stage_3=stages[3], forced_stage_4=stages[4], forced_stage_5_single=stages[5],
        forced_stages={str(i): stages[i] for i in range(1,6)},
        stage5_multi_root_injection=sum(d['stage5_multi_root_injection'] for d in v5),
        normal_simulation_decisions=sum(d['simulation_mode'] == 'normal' for d in v5),
        tactical_simulation_decisions=sum(d['simulation_mode'] == 'tactical' for d in v5),
        best_tactical_score=dict(min=min(scores) if scores else None,
                                 max=max(scores) if scores else None,
                                 avg=sum(scores)/len(scores) if scores else None),
        agent_timing=times, last_threat_counts=dict(Counter(r['last_threat']['type'] for r in records)),
        early_draw_count=sum(r['early_draw_reason'] is not None for r in records),
        game_details=records, decisions=decisions,
    )


def threshold_value(value: str) -> int | None:
    return None if value.lower() == 'none' else int(value)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=1, help='games per color; default 1 = 2 total')
    parser.add_argument('--stage', choices=('a','b','c'), default='c')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--early-draw', action='store_true', help='optional draw heuristic; default OFF')
    for key in V5_PRESETS['c']:
        kind = threshold_value if key == 'tactical_score_threshold' else float if key == 'exploration' else int
        parser.add_argument('--' + key.replace('_','-'), type=kind, default=argparse.SUPPRESS)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.games < 1:
        parser.error('--games must be positive')
    overrides = {key: getattr(args, key) for key in V5_PRESETS['c'] if hasattr(args, key)}
    try:
        template = MCTSV5Agent(stage=args.stage, **overrides)
    except ValueError as exc:
        parser.error(str(exc))
    config = dict(seed=args.seed, games_per_color=args.games, stage=args.stage, early_draw=args.early_draw,
                  v5={key: getattr(template, key) for key in V5_PRESETS['c']},
                  v41=dict(simulations=50, exploration=2**0.5, candidate_limit=20,
                           initial_width=8, neighborhood_radius=2, priority_top_k=8))
    log_dir = args.log_dir or default_log_dir('mcts_v5_vs_v41', args.seed)
    print(f'Log directory: {log_dir}; total games: {args.games * 2}', flush=True)
    factory = lambda seed: MCTSV5Agent(seed=seed, stage=args.stage, **overrides)
    records, decisions, matches = [], [], []
    for label, black, white in [('v5_black_vs_v41_white', factory, MCTSV41Agent),
                                ('v41_black_vs_v5_white', MCTSV41Agent, factory)]:
        matches.append((label, run_color(black, white, args.games, args.seed, records, decisions, args.early_draw)))
        save_match_logs(log_dir, matches, config)
        summary = summarize(matches, records, decisions)
        (log_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('decisions','game_details')}, indent=2))
    if args.games == 1:
        print('Two-game smoke/profile only: no playing-strength conclusion.')


if __name__ == '__main__':
    main()
