"""H7-c search retarget (docs/h7-c1.md, "S200 retarget"): pure rules shared by ``scripts/h7c_retarget.py`` and
``scripts/h7c_labels.py``, so the tool that searches and the tool that writes the rows can never disagree.

The actor (the cycle's self-play, frozen TB-V3 S50 / tactical S100) keeps its games; the retarget searches again,
with the same engine, models and per-game agent seeds and only the two tree budgets scaled by the decision's factor,
every self-play position whose actor policy target is of kind ``visits`` (every ply, not only value plies). Its
record of a position is raw: route, engine move, root visits ``[row, col, visits, mean value]``, proven losses and the
E4-v2 step; the policy target is still made only by ``hybrid.h7c_targets.policy_target`` (rule ``h7c-target-v1``,
unchanged), from the retarget's visits and the union of the actor's and the retarget's proven losses (each is a
proof, so neither may get probability).

    eligible       actor plies with ``target.kind == 'visits'`` (forced, proven-win and no-mass plies keep the actor's
                   rule: no retarget); coverage key ``(game_index, ply, canonical_hash)``, ``coverage_sha256`` over
                   the sorted keys
    must match     the retarget's route is ``tree`` and its E4-v2 step (``ran``, ``skip``, ``base_route``) is the
                   actor's: both are independent of the tree budget, so a difference is an error, never data
    proven losses  ``effective = actor | retarget``; per row ``same`` / ``actor_only`` / ``retarget_only`` counts
    row format     ``h7c-policy-row-v2``: the v1 fields (target, kind, kl and ``raw_visits`` from the retarget) plus
                   ``target_source`` (``retarget`` for visits rows, ``actor`` for proof rows), ``retarget`` (the
                   artifact identity), ``actor_raw_visits``, ``proven_loss`` (effective), ``actor_proven_loss``,
                   ``retarget_proven_loss``, ``retarget_move``; ``v1_projection`` drops them again
Agent state (checked in the frozen sources): ``E4v2AttackAgent`` keeps one cooldown bit between moves (set by an
exhausted attack without a WIN, consumed at the next own move) and the base keeps only its random stream, which the
``puct_value`` tree never draws from (no rollouts) and no tree. The ``restored`` mode therefore searches only the
eligible positions with the per-colour agent of the actor's seed, its cooldown reset and set again exactly when the
actor's record says the move was blocked (``e4.skip == 'cooldown'``); the ``sequential`` mode replays every ply of
the game through the agents, as the actor did. Factor 1 in either mode must reproduce the actor (G0).
"""
from __future__ import annotations

import hashlib
import json

FORMAT = 'h7c-retarget-v1'
POLICY_ROW_V1 = 'h7c-policy-row-v1'
POLICY_ROW_V2 = 'h7c-policy-row-v2'
MODES = ('restored', 'sequential')
PURPOSES = ('retarget', 'g0')
BASE_BUDGET = {'simulations': 50, 'tactical_simulations': 100}
E4_KEYS = ('ran', 'skip', 'base_route')
Q_TOLERANCE = 1e-5  # the mean values are stored rounded to 6 decimals
# Cost gate (docs/h7-c1.md): the first projection once both minima are done, then again at these fractions.
GATE_MIN_GAMES = 20
GATE_MIN_ROWS = 256
GATE_FRACTIONS = (0.10, 0.25, 0.50)
V2_FIELDS = ('target_source', 'retarget', 'actor_raw_visits', 'proven_loss', 'actor_proven_loss',
             'retarget_proven_loss', 'retarget_move')
PROVENANCE_FIELDS = ('git_commit', 'h7c_provenance')  # the label run's own commit differs between two label runs


def budget(factor: int) -> dict:
    if not isinstance(factor, int) or factor < 1:
        raise ValueError('the budget factor is an integer of at least 1')
    return {k: v * factor for k, v in BASE_BUDGET.items()}


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


SELFPLAY_KEYS = ('cycle', 'run_seed', 'engine_hash', 'runtime_hash', 'value_device', 'models', 'model_hash',
                 'generator_git')


def selfplay_identity(games: list[dict]) -> dict:
    """What the retarget names of its self-play: the same in every game of the file."""
    if not games:
        raise ValueError('no games')
    ident = {k: games[0].get(k) for k in SELFPLAY_KEYS}
    if any({k: g.get(k) for k in SELFPLAY_KEYS} != ident for g in games):
        raise ValueError('the self-play file mixes runs (cycle, seed, engine, runtime, device, models or commit)')
    return ident


def eligible_plies(game: dict) -> list[dict]:
    """The actor plies the retarget searches: every ply whose actor target is of kind ``visits``."""
    return [p for p in game['plies'] if p['target']['kind'] == 'visits']


def coverage_keys(games: list[dict]) -> list[list]:
    from hybrid.h6_labels import canonical_hash

    keys = [[g['game_index'], p['ply'], canonical_hash(g['moves'][:p['ply']])]
            for g in games for p in eligible_plies(g)]
    return sorted(keys)


def coverage(games: list[dict]) -> dict:
    keys = coverage_keys(games)
    return {'eligible': len(keys), 'games': len(games), 'coverage_sha256': sha256_json(keys)}


def _moves(ms) -> set[tuple[int, int]]:
    return {(int(m[0]), int(m[1])) for m in ms}


def effective_losses(actor: list, retarget: list) -> tuple[list[list[int]], dict]:
    """``actor | retarget`` (sorted) and the per-row counts."""
    a, r = _moves(actor), _moves(retarget)
    union = sorted(a | r)
    return [list(m) for m in union], {'same': len(a & r), 'actor_only': len(a - r), 'retarget_only': len(r - a),
                                      'union': len(union)}


def check_row(game: dict, ply: dict, row: dict) -> None:
    """A retarget record of an eligible actor ply: the same position, route ``tree``, the actor's E4-v2 step."""
    from hybrid.h6_labels import canonical_hash

    where = f"game {game['game_index']} ply {ply['ply']}"
    if row.get('ply') != ply['ply'] or row.get('canonical_hash') != canonical_hash(game['moves'][:ply['ply']]):
        raise ValueError(f'{where}: the retarget record is of another position')
    if row.get('route') != 'tree' or not row.get('visits'):
        raise ValueError(f"{where}: the retarget took route {row.get('route')!r}; the actor's tree position must stay "
                         f'a tree position (only the tree budget changed)')
    actor_e4 = {k: (ply.get('e4') or {}).get(k) for k in E4_KEYS}
    if {k: (row.get('e4') or {}).get(k) for k in E4_KEYS} != actor_e4:
        raise ValueError(f"{where}: the retarget's E4-v2 step {row.get('e4')} is not the actor's {actor_e4}")
    from renju import Game
    position = Game()
    for move in game['moves'][:ply['ply']]:
        position.play(*move)
    legal = set(position.legal_moves())
    for name in ('visits', 'proven_loss'):
        if any((int(m[0]), int(m[1])) not in legal for m in row[name]):
            raise ValueError(f'{where}: a retarget {name} move is not legal in the position')
    if (int(row['engine_move'][0]), int(row['engine_move'][1])) not in legal:
        raise ValueError(f'{where}: the retarget move is not legal')


def index_rows(games: list[dict], records: list[dict], identity: dict, *,
               check: bool = True) -> dict[tuple[int, int], dict]:
    """The retarget rows by (game index, ply), after the coverage checks: every eligible ply exactly once, no other
    ply, every record of this identity; and ``check_row`` on each (G0 counts the differences instead)."""
    by_game = {g['game_index']: g for g in games}
    rows: dict[tuple[int, int], dict] = {}
    for record in records:
        if record.get('identity') != identity:
            raise ValueError(f"retarget game {record.get('game_index')}: another retarget identity in the file")
        g = by_game.get(record['game_index'])
        if g is None:
            raise ValueError(f"retarget game {record['game_index']} is not a game of the self-play file (extra)")
        for row in record['rows']:
            key = (record['game_index'], row.get('ply'))
            if key in rows:
                raise ValueError(f'retarget row {key} is in the file twice')
            rows[key] = row
    wanted = {(g['game_index'], p['ply']): (g, p) for g in games for p in eligible_plies(g)}
    missing = sorted(set(wanted) - set(rows))
    extra = sorted(set(rows) - set(wanted))
    if missing or extra:
        raise ValueError(f'retarget coverage: {len(missing)} missing (first {missing[:3]}), {len(extra)} extra '
                         f'(first {extra[:3]})')
    if check:
        for key, (g, p) in wanted.items():
            check_row(g, p, rows[key])
    return rows


def retarget_target(ply: dict, row: dict) -> tuple[dict, list[list[int]], dict]:
    """The policy target of an eligible ply from its retarget record: (target, effective losses, loss counts)."""
    from hybrid import h7c_targets as targets

    losses, counts = effective_losses(ply['proven_loss'], row['proven_loss'])
    return targets.policy_target(row['route'], row['engine_move'], row['visits'], losses), losses, counts


def v1_projection(row: dict) -> dict:
    """A v2 policy row as the v1 row of the same position would be (the G0 comparison); provenance dropped."""
    out = {k: v for k, v in row.items() if k not in V2_FIELDS and k not in PROVENANCE_FIELDS}
    out['format'] = POLICY_ROW_V1
    return out


def compare_to_actor(ply: dict, row: dict) -> list[str]:
    """G0 at factor 1: the fields where a retarget record differs from the actor's own record (empty: reproduced)."""
    from hybrid import h7c_targets as targets

    diffs = []
    if row['route'] != ply['route']:
        diffs.append('route')
    if list(row['engine_move']) != list(ply['engine_move']):
        diffs.append('engine_move')
    a = sorted(ply['visits'])
    b = sorted(row['visits'])
    if [v[:3] for v in a] != [v[:3] for v in b]:
        diffs.append('visits')
    elif any(abs(x[3] - y[3]) > Q_TOLERANCE for x, y in zip(a, b)):
        diffs.append('mean_value')
    if sorted(map(list, ply['proven_loss'])) != sorted(map(list, row['proven_loss'])):
        diffs.append('proven_loss')
    if {k: (ply.get('e4') or {}).get(k) for k in E4_KEYS} != {k: (row.get('e4') or {}).get(k) for k in E4_KEYS}:
        diffs.append('e4')
    mine = targets.policy_target(row['route'], row['engine_move'], row['visits'],
                                 effective_losses(ply['proven_loss'], row['proven_loss'])[0])
    theirs = ply['target']
    if mine['kind'] != theirs['kind'] or (mine['target'] is None) != (theirs['target'] is None) or (
            mine['target'] is not None and ([m[:2] for m in mine['target']] != [m[:2] for m in theirs['target']] or any(
                abs(x[2] - y[2]) > 1e-9 for x, y in zip(mine['target'], theirs['target'])))):
        diffs.append('target')
    return diffs


def cost_gate(*, elapsed: float, games_done: int, rows_done: int, rows_total: int, cap_seconds: float,
              checked: set) -> dict | None:
    """The projection of the whole run's wall time at this session's rate, when a gate point is reached (the first
    once ``GATE_MIN_GAMES`` games and ``GATE_MIN_ROWS`` rows are done, then each of ``GATE_FRACTIONS``); None when no
    gate point is due. ``exceeded`` stops the run (the finished games stay in the resume file)."""
    if rows_done <= 0 or rows_total <= 0:
        return None
    points = []
    if games_done >= GATE_MIN_GAMES and rows_done >= GATE_MIN_ROWS:
        points.append('first')
        points += [f'{f:.2f}' for f in GATE_FRACTIONS if rows_done >= f * rows_total]
    due = [p for p in points if p not in checked]
    if not due:
        return None
    checked.update(due)
    projected = elapsed / rows_done * rows_total
    return {'point': due[-1], 'rows_done': rows_done, 'rows_total': rows_total, 'elapsed_seconds': round(elapsed, 1),
            'projected_seconds': round(projected, 1), 'cap_seconds': cap_seconds, 'exceeded': projected > cap_seconds}
