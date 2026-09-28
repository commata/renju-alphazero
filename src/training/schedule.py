"""Stage 8 external-evaluation schedule and colour-regression summary (torch-free).

The orchestrator (``scripts/run_stage8_training.py``) trains in segments that stop at
scheduled checkpoint generations, evaluates those checkpoints outside the training loop
and resumes. Resuming is exact (Stage 6 contract), so a segmented run trains the same
model as an uninterrupted one; evaluation never touches training state.

Schedule points are ``anchor + k * every`` (k >= 0). With 16 games per generation,
``light_every=20`` = 320 games and ``heavy_every=80`` = 1,280 games; ``keep_every: 20``
in the run config keeps exactly those checkpoints.
"""
from __future__ import annotations

from .health import COLORS, update_color_regression


def schedule_points(anchor: int, every: int, upto: int) -> list[int]:
    if every < 1:
        raise ValueError('every must be positive')
    return list(range(anchor, upto + 1, every)) if upto >= anchor else []


def next_stop(current: int, anchor: int, every: int, target: int) -> int:
    """First schedule point after ``current`` (checkpoint generation), capped at ``target``."""
    if current < anchor:
        return min(anchor, target)
    k = (current - anchor) // every + 1
    return min(anchor + k * every, target)


def color_regression_summary(light_results: list[dict], opponent: str = 'mcts_v2',
                             alpha: float = 0.05) -> dict:
    """Replay the per-colour regression state over light evaluations in generation order.

    ``light_results`` are checkpoint-eval result dicts (``generation`` and
    ``opponents[opponent].summary.black/white``). Recomputed from the files each time, so
    the summary is idempotent across orchestrator restarts.
    """
    states = {color: None for color in COLORS}
    rows = []
    for result in sorted(light_results, key=lambda r: r['generation']):
        data = result['opponents'].get(opponent)
        if data is None:
            continue
        row = {'generation': result['generation']}
        for color in COLORS:
            side = data['summary'][color]
            event, states[color] = update_color_regression(
                states[color], {'generation': result['generation'], 'wins': side['wins'],
                                'games': side['games']}, alpha)
            row[color] = event
        rows.append(row)
    latest = rows[-1] if rows else None
    alerts = [{'generation': row['generation'], 'color': color, **row[color]}
              for row in rows for color in COLORS
              if row[color]['status'] in ('candidate', 'confirmed')]
    return {'opponent': opponent, 'alpha': alpha, 'rows': rows, 'alerts': alerts,
            'latest': latest}
