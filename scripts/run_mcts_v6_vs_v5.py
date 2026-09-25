"""V6 vs V5 FINAL; fixed identical search parameters, games per color."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import json
from pathlib import Path

from agents import MCTSV5Agent, MCTSV6Agent
from evaluation import default_log_dir, save_match_logs
from renju import BLACK, WHITE, Game
from search.mcts_v6 import V5_FINAL, SearchDiagnostics
from search.threat_patterns import compound_at, black_legal_43_moves

if __package__:
    from .run_mcts_v5_vs_v41 import run_color
else:
    from run_mcts_v5_vs_v41 import run_color


def v5_final(seed):
    agent = MCTSV5Agent(seed=seed, **V5_FINAL)
    agent.name = 'MCTS-v5-final'
    return agent


def annotate_defense_outcomes(matches, decisions):
    """Offline replay excludes instrumentation from decision timing."""
    indexed = {(d['game_id'], d['ply']): d for d in decisions}
    results = [r for _, match in matches for r in match.results]
    for game_id, result in enumerate(results, 1):
        game = Game()
        pending = None
        for ply, move in enumerate(result.history, 1):
            record = indexed[(game_id, ply)]
            if pending is not None:
                compound = compound_at(game, BLACK, move)
                pending['black_43_on_next_reply'] = compound is not None and '43' in compound.kinds
                pending = None
            game.play(*move)
            if record['player'] == WHITE and any(
                reason in record.get('v6_selected_reasons', ())
                for reason in ('black_43_defense', 'future_black_43_defense')
            ):
                record['black_43_available_after_defense'] = len(black_legal_43_moves(game))
                record['black_43_on_next_reply'] = None
                pending = record


def summarize(matches, decisions):
    results = [r for _, match in matches for r in match.results]
    by_color = {}
    numeric = [f.name for f in fields(SearchDiagnostics)
               if f.name not in {'best_root_tactical_score', 'selected_simulations',
                                 'forced_policy_stage'} and f.type is int]
    for color, label in ((BLACK, 'black'), (WHITE, 'white')):
        games = [r for r in results
                 if (r.black_agent if color == BLACK else r.white_agent) == 'MCTS-v6']
        records = [d for d in decisions if d['agent'] == 'MCTS-v6' and d['player'] == color]
        selected = Counter(k for d in records for k in d.get('v6_selected_reasons', ()))
        by_color[label] = dict(
            wins=sum(r.winner == color for r in games),
            losses=sum(r.winner == -color for r in games),
            draws=sum(r.winner is None for r in games),
            detector_counts={key: sum(d.get(key, 0) for d in records) for key in numeric},
            selected_counts=dict(selected),
            forced_stages=dict(Counter(str(d['forced_policy_stage']) for d in records)),
            simulation_modes=dict(Counter(str(d['simulation_mode']) for d in records)),
            planner_seconds=sum(d['v6_threat_planner_seconds'] for d in records),
            defense_followups_observed=sum(d.get('black_43_on_next_reply') is not None for d in records),
            black_43_after_selected_defense=sum(d.get('black_43_on_next_reply') is True for d in records),
        )
    timing = {}
    for name in ('MCTS-v6', 'MCTS-v5-final'):
        samples = [d['seconds'] for d in decisions if d['agent'] == name]
        timing[name] = dict(moves=len(samples), seconds=sum(samples),
                            seconds_per_move=sum(samples)/len(samples) if samples else 0)
    baseline = timing['MCTS-v5-final']['seconds_per_move']
    return dict(games=len(results), v6_by_color=by_color, agent_timing=timing,
                observed_overhead=(timing['MCTS-v6']['seconds_per_move']/baseline - 1) if baseline else None)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=1, help='games per color; 1 means 2 total')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-dir', type=Path)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.games < 1:
        parser.error('--games must be positive')
    log_dir = args.log_dir or default_log_dir('mcts_v6_vs_v5', args.seed)
    config = dict(seed=args.seed, games_per_color=args.games,
                  v6=V5_FINAL, v5_final=V5_FINAL, early_draw=False)
    records, decisions, matches = [], [], []
    print(f'Logs: {log_dir}; total games={2*args.games}; fixed V5 FINAL parameters', flush=True)
    for label, black, white in [('v6_black', MCTSV6Agent, v5_final),
                                ('v6_white', v5_final, MCTSV6Agent)]:
        matches.append((label, run_color(black, white, args.games, args.seed, records, decisions)))
        save_match_logs(log_dir, matches, config)
        annotate_defense_outcomes(matches, decisions)
        summary = summarize(matches, decisions)
        (log_dir / 'summary.json').write_text(json.dumps(
            {**summary, 'game_details': records, 'decisions': decisions}, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)
    print('Small smoke: outcomes and timing are observations, not strength or overhead proofs.')


if __name__ == '__main__':
    main()
