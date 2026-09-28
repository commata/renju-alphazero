"""Checkpoint vs checkpoint round robin (Stage 8 Gate 3 candidate ranking).

Every pair of the given training checkpoints plays ``--pairs`` fixed openings twice
(colours swapped), both sides with the same deterministic PUCT (noise off, temperature 0;
default PUCT v1 25 simulations like the external evaluation). Openings depend only on
(--seed, pair index), so every match-up meets the same openings. Reports per pair the
score of A, the result by A's colour, a two-sided exact binomial p (decisive games vs 50%)
and an Elo estimate, plus a per-checkpoint total.

    python scripts/run_stage8_head_to_head.py \
        --checkpoint d16_200=runs/stage8_d16/checkpoints/checkpoint_gen200.pt \
        --checkpoint d16_320=runs/stage8_d16/checkpoints/checkpoint_gen320.pt \
        --pairs 50 --output runs/stage8_h2h/d16_200_vs_320.json
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from math import comb, log10
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from renju import BLACK, WHITE  # noqa: E402
from run_stage7_checkpoint_eval import search_config_for  # noqa: E402
from training.evaluation import PUCTAgent, make_opening, play_evaluation_game  # noqa: E402
from training.probes import load_model_from_training_checkpoint  # noqa: E402
from training.training_state import derive_seed  # noqa: E402

RESULT_FORMAT = 'stage8-head-to-head-v1'


def binomial_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def elo(score: float) -> float | None:
    if score <= 0 or score >= 1:
        return None
    return -400 * log10(1 / score - 1)


def play_match(a: dict, b: dict, *, pairs: int, seed: int, opening_plies: int,
               radius: int, log=print) -> dict:
    started = perf_counter()
    games = []
    for pair in range(pairs):
        opening = make_opening(Random(derive_seed(seed, 'h2h', 'opening', pair)),
                               opening_plies, radius)
        for a_color in (BLACK, WHITE):
            game = play_evaluation_game(a['agent'], b['agent'], a_color, opening)
            game['pair'] = pair
            games.append(game)
    wins = sum(g['result'] == 'win' for g in games)
    losses = sum(g['result'] == 'loss' for g in games)
    draws = len(games) - wins - losses
    score = (wins + 0.5 * draws) / len(games)
    by_color = {color: {'wins': sum(g['result'] == 'win' for g in games
                                    if g['model_color'] == color),
                        'games': sum(g['model_color'] == color for g in games)}
                for color in ('black', 'white')}
    summary = {'a': a['label'], 'b': b['label'], 'games': len(games), 'a_wins': wins,
               'a_losses': losses, 'draws': draws, 'a_score': score,
               'a_by_color': by_color, 'p_two_sided': binomial_two_sided(wins, losses),
               'elo_a_minus_b': elo(score),
               'unique_games': len({tuple(map(tuple, g['moves'])) for g in games}),
               'seconds': perf_counter() - started}
    log(f"{a['label']} vs {b['label']}: {wins}-{losses}-{draws} score {score:.3f} "
        f"(A black {by_color['black']['wins']}/{by_color['black']['games']}, "
        f"A white {by_color['white']['wins']}/{by_color['white']['games']}) "
        f"p={summary['p_two_sided']:.3g} ({summary['seconds']:.0f}s)")
    return {'summary': summary, 'games': games}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', action='append', required=True, metavar='LABEL=PATH',
                        help='repeat for every participant (at least two)')
    parser.add_argument('--pairs', type=int, default=50,
                        help='openings per match-up; each is played with both colours')
    parser.add_argument('--seed', type=int, default=8008)
    parser.add_argument('--simulations', type=int, help='default: checkpoint puct_simulations')
    parser.add_argument('--tactical-rules', choices=('auto', 'on', 'off'), default='off')
    parser.add_argument('--opening-random-plies', type=int, default=2)
    parser.add_argument('--opening-radius', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        parser.error('need at least two --checkpoint LABEL=PATH')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    players = []
    for item in args.checkpoint:
        label, _, path = item.partition('=')
        if not path:
            parser.error(f'--checkpoint must be LABEL=PATH, got {item!r}')
        path = Path(path)
        model, info = load_model_from_training_checkpoint(path)
        search = search_config_for(path, args.simulations, args.tactical_rules)
        players.append({'label': label, 'path': str(path), 'info': info,
                        'search': search.to_dict(), 'agent': PUCTAgent(label, model, search)})
    matches = [play_match(a, b, pairs=args.pairs, seed=args.seed,
                          opening_plies=args.opening_random_plies, radius=args.opening_radius,
                          log=lambda m: print(m, flush=True))
               for a, b in combinations(players, 2)]
    totals = {p['label']: {'points': 0.0, 'games': 0} for p in players}
    for match in matches:
        s = match['summary']
        totals[s['a']]['points'] += s['a_wins'] + 0.5 * s['draws']
        totals[s['b']]['points'] += s['a_losses'] + 0.5 * s['draws']
        for label in (s['a'], s['b']):
            totals[label]['games'] += s['games']
    for label, t in sorted(totals.items(), key=lambda kv: -kv[1]['points']):
        print(f"{label}: {t['points']:.1f}/{t['games']} ({t['points'] / t['games']:.3f})")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'format_version': RESULT_FORMAT, 'seed': args.seed, 'pairs': args.pairs,
            'players': [{k: v for k, v in p.items() if k != 'agent'} for p in players],
            'totals': totals, 'matches': matches}, indent=1), encoding='utf-8')
        print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
