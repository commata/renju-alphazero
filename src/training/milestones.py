"""Stage 7 milestone detection: record sudden in-loop win-rate jumps at any generation.

Execution-only monitoring (``milestones`` is not in the critical config): it reads the
run's own evaluation events, never touches training state or RNGs, and only adds
``milestone`` metric events plus a pinned checkpoint copy.

- ``win_rate_jump``: for each monitored opponent, the score over the last ``window``
  evaluated generations minus the score over the ``window`` generations before it is
  at least ``threshold``. A jump for the same opponent is not reported again within
  ``window`` generations (the sliding windows would otherwise repeat it).
- ``first_win``: the first generation of the run with a win against the opponent.

Evaluation of generation g scores the model saved as ``checkpoint_gen{g+1}``; that
file is copied to ``milestone_gen{g+1}.pt`` so periodic pruning never removes it.
"""
from __future__ import annotations


def _score(events: list[dict]) -> tuple[float, int]:
    games = sum(e['games'] for e in events)
    points = sum(e['wins'] + 0.5 * e['draws'] for e in events)
    return (points / games if games else 0.0), games


def detect_milestones(evaluations: list[dict], previous_milestones: list[dict],
                      generation: int, settings: dict) -> list[dict]:
    """Milestones that occur at ``generation`` given all evaluation events so far."""
    window = settings['window']
    found = []
    for opponent in settings['opponents']:
        per_generation = {e['generation']: e for e in evaluations if e['opponent'] == opponent}
        current = [per_generation[g] for g in range(generation - window + 1, generation + 1)
                   if g in per_generation]
        before = [per_generation[g] for g in range(generation - 2 * window + 1,
                                                    generation - window + 1)
                  if g in per_generation]
        if len(current) < window or len(before) < window:
            continue
        recent = [m for m in previous_milestones
                  if m['kind'] == 'win_rate_jump' and m['opponent'] == opponent
                  and generation - m['generation'] < window]
        if recent:
            continue
        current_rate, current_games = _score(current)
        before_rate, before_games = _score(before)
        if current_rate - before_rate >= settings['threshold']:
            found.append({'kind': 'win_rate_jump', 'opponent': opponent,
                          'generations': [generation - window + 1, generation],
                          'previous_generations': [generation - 2 * window + 1,
                                                   generation - window],
                          'previous_score': before_rate, 'current_score': current_rate,
                          'delta': current_rate - before_rate,
                          'games': [before_games, current_games]})
    for opponent in settings['first_win']:
        now = [e for e in evaluations if e['opponent'] == opponent and e['generation'] == generation]
        earlier = [e for e in evaluations
                   if e['opponent'] == opponent and e['generation'] < generation and e['wins']]
        already = [m for m in previous_milestones
                   if m['kind'] == 'first_win' and m['opponent'] == opponent]
        if now and now[0]['wins'] and not earlier and not already:
            found.append({'kind': 'first_win', 'opponent': opponent,
                          'wins': now[0]['wins'], 'games': now[0]['games']})
    return found
