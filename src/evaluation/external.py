"""External-engine anchor: seed blocks and the match summary (docs/external-engine.md).

Games against an external engine are evaluation only. Their seeds are reserved (also in
``hybrid.h6_labels.RESERVED_SEEDS``) so these games never become training data.
"""
from __future__ import annotations

import hashlib
import json
import math
from random import Random

# Fixed before any game was played (docs/external-engine.md §4).
CALIBRATION_SEEDS = tuple(range(8701, 8706))  # X1: strength calibration and the confirmation run
# X2: one per H7 cycle c (seed 8710 + c), record only; 8710 = cycle 0, the pre-RL baseline of the frozen TB-V3
# (registered 2026-10-23, before any game on it)
MONITORING_SEEDS = tuple(range(8710, 8721))
FINAL_ANCHOR_SEEDS = tuple(range(8721, 8726))  # X3: H8 report, not played before H8
EXTERNAL_SEEDS = frozenset(CALIBRATION_SEEDS + MONITORING_SEEDS + FINAL_ANCHOR_SEEDS)

SCORED = {'win': 1.0, 'draw': 0.5, 'loss': 0.0}
EXCLUDED = ('engine_illegal', 'engine_error')


def elo(score: float) -> float | None:
    """Elo difference for a score; None at 0 or 1 (unbounded)."""
    if not 0.0 < score < 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def complete_pairs(games: list[dict]) -> dict[int, list[dict]]:
    """Opening pairs whose two games (one per colour) both have a scored result."""
    pairs: dict[int, list[dict]] = {}
    for game in games:
        pairs.setdefault(game['pair'], []).append(game)
    return {p: g for p, g in sorted(pairs.items())
            if len(g) == 2 and all(x['result'] in SCORED for x in g)}


def pair_bootstrap(games: list[dict], *, samples: int = 10_000, seed: int = 0) -> list[float] | None:
    """95% interval of the score, resampling complete opening pairs with a fixed RNG."""
    values = [sum(SCORED[g['result']] for g in pair) / 2 for pair in complete_pairs(games).values()]
    if len(values) < 2:
        return None
    random = Random(seed)
    means = sorted(sum(random.choice(values) for _ in values) / len(values) for _ in range(samples))
    return [round(means[int(0.025 * samples)], 4), round(means[int(0.975 * samples) - 1], 4)]


def fingerprint(games: list[dict]) -> str:
    """SHA-256 over (key, moves) of every game: equal for a reproducible rerun."""
    payload = [[g['key'], g['moves']] for g in sorted(games, key=lambda g: g['key'])]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


def loss_candidates(games: list[dict], window: int) -> list[dict]:
    """Our moves in the last ``window`` plies of each lost game: input for the offline truth step (§2.4)."""
    out = []
    for game in games:
        if game['result'] != 'loss':
            continue
        moves, length = game['moves'], len(game['moves'])
        for record in game['our_moves']:
            if record['ply'] >= length - window:
                out.append({'key': game['key'], 'ply': record['ply'], 'played': moves[record['ply']],
                            'moves': moves[:record['ply']], 'to_end': length - record['ply']})
    return out


def summarize(games: list[dict], *, loss_window: int = 10) -> dict:
    """Score, Elo, colour split and interval all use the same sample: complete opening pairs (§5).
    A pair with one excluded game (``engine_illegal`` / ``engine_error``) drops out whole; the
    per-game numbers over every scored game stay as ``raw_*`` diagnostics."""
    scored = [g for g in games if g['result'] in SCORED]
    pairs = complete_pairs(games)
    paired = [g for pair in pairs.values() for g in pair]

    def score(subset):
        return round(sum(SCORED[g['result']] for g in subset) / len(subset), 4) if subset else None

    counts = {r: sum(g['result'] == r for g in games) for r in (*SCORED, *EXCLUDED)}
    total = score(paired)
    return {
        'games': len(games), **counts,
        'complete_pairs': len(pairs),
        'incomplete_pairs': sorted({g['pair'] for g in games} - set(pairs)),
        'score': total,
        'elo': round(elo(total), 1) if total is not None and elo(total) is not None else None,
        'score_ci95_pairs': pair_bootstrap(games),
        'by_colour': {c: {'games': len(sub), 'score': score(sub)}
                      for c in ('black', 'white')
                      for sub in [[g for g in paired if g['ours_color'] == c]]},
        'raw_scored_games': len(scored), 'raw_score': score(scored),
        'average_length': round(sum(len(g['moves']) for g in paired) / len(paired), 1) if paired else None,
        'illegal_games': [{'key': g['key'], **g['illegal']} for g in games if g['result'] == 'engine_illegal'],
        'error_games': [{'key': g['key'], 'error': g['error']} for g in games if g['result'] == 'engine_error'],
        'forbid_mismatches': [{'key': g['key'], **m} for g in games for m in g.get('forbid_mismatch', [])],
        'forbid_checked_positions': sum(g.get('forbid_checked', 0) for g in games),
        'loss_candidates': loss_candidates(games, loss_window),
        'fingerprint': fingerprint(games),
    }
