"""Stage 8 self-play colour health (execution-only; ``health`` is not in the critical config).

Two signals (docs/stage8-plan.md §4):

- ``color_imbalance`` (in the training loop, informational): the black win rate over the
  last ``window`` generations of self-play. At or above ``upper`` / at or below ``lower``
  a ``health_warning`` is logged, once per direction per ``window`` generations. It never
  stops training: the successful Stage 7-D D16 run also spent long stretches beyond 90%
  and 10% (stage7-plan §8.7).
- ``color_regression`` (on external evaluations, the actionable signal): per colour, the
  current wins against a fixed opponent are compared with a healthy reference by a
  one-sided Fisher exact test. The first significant drop is a ``candidate``; a second
  consecutive drop against the **same** reference is ``confirmed``. The reference is the
  best healthy result so far, so a gradual slide (17 -> 14 -> 11 -> 8) or a sustained
  collapse (17 -> 3 -> 1) cannot hide by moving the baseline down.

Like ``training.milestones`` this reads the run's own metric events and never touches
training state or RNGs.
"""
from __future__ import annotations

from math import comb

COLORS = ('black', 'white')


def color_window(generation_events: list[dict], generation: int, window: int) -> dict | None:
    """Self-play colour statistics over generations ``generation-window+1 .. generation``."""
    per_generation = {e['generation']: e for e in generation_events}
    generations = range(generation - window + 1, generation + 1)
    if generation - window + 1 < 0 or any(g not in per_generation for g in generations):
        return None
    events = [per_generation[g] for g in generations]
    games = sum(e['self_play_games'] for e in events)
    if not games:
        return None
    black = sum(e['black_wins'] for e in events)
    white = sum(e['white_wins'] for e in events)
    draws = sum(e['draws'] for e in events)
    lengths = sum(e['average_game_length'] * e['self_play_games'] for e in events)
    return {'generations': [generation - window + 1, generation], 'games': games,
            'black_win_rate': black / games, 'white_win_rate': white / games,
            'draw_rate': draws / games, 'average_game_length': lengths / games}


def _direction(rate: float, settings: dict) -> str | None:
    if rate >= settings['upper']:
        return 'black'
    if rate <= settings['lower']:
        return 'white'
    return None


def detect_color_imbalance(generation_events: list[dict], previous_warnings: list[dict],
                           generation: int, settings: dict) -> tuple[dict | None, dict | None]:
    """(health event, warning or None) at ``generation``; (None, None) before a full window."""
    window = settings['window']
    stats = color_window(generation_events, generation, window)
    if stats is None:
        return None, None
    direction = _direction(stats['black_win_rate'], settings)
    streak = 0
    if direction is not None:
        g = generation
        while True:
            earlier = color_window(generation_events, g, window)
            if earlier is None or _direction(earlier['black_win_rate'], settings) != direction:
                break
            streak += 1
            g -= 1
    health = {'kind': 'color_window', 'window': window, **stats,
              'extreme': direction, 'extreme_generations': streak}
    if direction is None:
        return health, None
    recent = [w for w in previous_warnings
              if w.get('kind') == 'color_imbalance' and w.get('dominant') == direction
              and generation - w['generation'] < window]
    if recent:
        return health, None
    warning = {'kind': 'color_imbalance', 'dominant': direction, 'window': window, **stats,
               'extreme_generations': streak,
               'thresholds': {'lower': settings['lower'], 'upper': settings['upper']}}
    return health, warning


def fisher_less_p(wins: int, games: int, ref_wins: int, ref_games: int) -> float:
    """One-sided Fisher exact p that ``wins/games`` is lower than ``ref_wins/ref_games``."""
    total_wins = wins + ref_wins
    total = games + ref_games
    denominator = comb(total, total_wins)
    low = max(0, total_wins - ref_games)
    return sum(comb(games, x) * comb(ref_games, total_wins - x)
               for x in range(low, wins + 1)) / denominator


def update_color_regression(reference: dict | None, current: dict,
                            alpha: float = 0.05) -> tuple[dict, dict]:
    """Advance one colour's regression state with a new external evaluation.

    ``current`` = {'generation', 'wins', 'games'}; ``reference`` is the state returned by
    the previous call (None at the start). Returns (event, new state).
    """
    if reference is None:
        state = {'reference': dict(current), 'pending': None}
        return {'status': 'healthy', **current, 'reference': dict(current), 'p_value': None}, state
    ref = reference['reference']
    p = fisher_less_p(current['wins'], current['games'], ref['wins'], ref['games'])
    if p < alpha:
        status = 'confirmed' if reference['pending'] is not None else 'candidate'
        state = {'reference': ref, 'pending': dict(current)}
    else:
        status = 'healthy'
        best = current if current['wins'] / current['games'] >= ref['wins'] / ref['games'] else ref
        state = {'reference': dict(best), 'pending': None}
    return {'status': status, **current, 'reference': dict(ref), 'p_value': p}, state
