"""H7-R opening suites: fixed start positions for evaluation Arenas (docs/h7-rl-plan.md, R1).

``run_mcts_v8_benchmark.make_opening`` (the cycle Arena's and the self-play's rule) is not changed. A suite is a list
of start histories fixed in a file before any game, each with three ids:

    history_id   the moves in order (the engine sees the last move, so order matters): the unit of a duplicate
    position_id  the stones of each colour and the side to move (order ignored)
    class_id     the position under the 8 board symmetries (recorded only: a symmetric opening is a different
                 opening, its games are almost never rotated copies)

Suites (``SUITES``; every move legal by ``renju.Game``, black forbidden points refused by the rules; an opening that
is over or that ``analysis.puct_value_v1.exact_leaf`` settles for the side to move is rejected):
    O0  3 moves: black the centre, white then black within radius 2 of the centre; white to move (the cycle Arena's)
    O1  5 moves: O0, then white and black within radius 3; white to move
    O2  3 moves: the 26 standard renju openings (black the centre, white directly or diagonally next to it, black
        within radius 2; one representative per symmetry of the first two stones) in each of the 8 board
        orientations, exact duplicates removed; white to move. A fixed set of start positions only: no game rule
        changes. O0 and O1 are drawn by a seed (rejection sampling without exact repeats); O2 is listed.
"""
from __future__ import annotations

import hashlib
import json
from random import Random

FORMAT = 'h7r-opening-suite-v1'
CENTRE = (7, 7)
SIZE = 15
SUITES = {
    'O0': {'moves': 3, 'stages': [('white', 2), ('black', 2)], 'next': 'white', 'kind': 'random'},
    'O1': {'moves': 5, 'stages': [('white', 2), ('black', 2), ('white', 3), ('black', 3)], 'next': 'white',
           'kind': 'random'},
    'O2': {'moves': 3, 'next': 'white', 'kind': 'standard26', 'orientations': 8},
}


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def symmetry(move, k: int) -> tuple[int, int]:
    """The k-th of the 8 board symmetries about the centre (k < 4 rotations, k >= 4 the mirrored ones)."""
    r, c = move[0] - CENTRE[0], move[1] - CENTRE[1]
    for _ in range(k % 4):
        r, c = c, -r
    if k >= 4:
        c = -c
    return r + CENTRE[0], c + CENTRE[1]


def position_key(moves) -> tuple:
    black = tuple(sorted(tuple(m) for m in moves[0::2]))
    white = tuple(sorted(tuple(m) for m in moves[1::2]))
    return black, white, len(moves) % 2


def ids(moves) -> dict:
    moves = [list(m) for m in moves]
    canon = min(position_key([symmetry(m, k) for m in moves]) for k in range(8))
    return {'history_id': _sha(moves), 'position_id': _sha(position_key(moves)), 'class_id': _sha(canon)}


def check_opening(moves) -> str | None:
    """None when the history is a usable start: every move legal, not over, not settled at once; else the reason."""
    from analysis.puct_value_v1 import exact_leaf
    from renju import Game, IllegalMove

    game = Game()
    if not moves or tuple(moves[0]) != CENTRE:
        return 'the first move is not the centre'
    for move in moves:
        try:
            game.play(*move)
        except IllegalMove as exc:
            return f'illegal move {list(move)}: {exc}'
        if game.done:
            return 'the game is over'
    if exact_leaf(game) is not None:
        return 'settled at once (exact_leaf)'
    return None


def _within(radius: int):
    return [(CENTRE[0] + dr, CENTRE[1] + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)]


def random_opening(rng: Random, stages) -> list[list[int]]:
    from renju import Game

    game = Game()
    game.play(*CENTRE)
    for _, radius in stages:
        legal = set(game.legal_moves())
        near = sorted(m for m in _within(radius) if m in legal)
        game.play(*rng.choice(near))
        if game.done:
            break
    return [list(m) for m in game.history]


def standard26() -> list[list[list[int]]]:
    """The 26 standard openings as 3-move histories: white directly above the centre (13) or diagonally above-right
    (13), black anywhere within radius 2, one per mirror image that keeps the first two stones."""
    out = []
    for white, mirror in (((6, 7), lambda m: (m[0], 14 - m[1])), ((6, 8), lambda m: (14 - m[1], 14 - m[0]))):
        seen = set()
        for third in sorted(_within(2)):
            if third in (CENTRE, white):
                continue
            key = min(third, mirror(third))
            if key in seen:
                continue
            seen.add(key)
            out.append([list(CENTRE), list(white), list(key)])
    if len(out) != 26:
        raise RuntimeError(f'{len(out)} standard openings, expected 26')
    return out


def build_suite(name: str, count: int | None, seed: int | None, max_attempts: int = 200_000) -> dict:
    """The suite file body: ``count`` openings (O0 / O1 from ``seed``; O2: all, or refused if ``count`` is larger)."""
    spec = SUITES[name]
    rejected, histories, rows = {}, set(), []

    def add(moves):
        reason = check_opening(moves)
        if reason is None and len(moves) != spec['moves']:
            reason = f'{len(moves)} moves, the suite has {spec["moves"]}'
        if reason is not None:
            kind = reason.split(' [')[0].split(':')[0]
            rejected[kind] = rejected.get(kind, 0) + 1
            return
        i = ids(moves)
        if i['history_id'] in histories:
            rejected['exact repeat'] = rejected.get('exact repeat', 0) + 1
            return
        histories.add(i['history_id'])
        rows.append({'index': len(rows), 'moves': moves, **i})

    if spec['kind'] == 'random':
        if count is None or seed is None or count <= 0:
            raise ValueError(f'{name}: a random suite needs a positive count and a seed')
        rng, attempts = Random(seed), 0
        while len(rows) < count:
            attempts += 1
            if attempts > max_attempts:
                raise ValueError(f'{name}: {count} distinct openings not found in {max_attempts} draws')
            add(random_opening(rng, spec['stages']))
    else:
        for moves in standard26():
            for k in range(spec['orientations']):
                add([list(symmetry(m, k)) for m in moves])
        if count is not None and count > len(rows):
            raise ValueError(f'{name} has {len(rows)} distinct openings; {count} asked (no repeats)')
        if count is not None:
            rows = rows[:count]
    body = {'format': FORMAT, 'suite': name, 'rule': {**spec, 'stages': [list(s) for s in spec.get('stages', [])]},
            'next_to_move': spec['next'], 'seed': seed, 'count': len(rows), 'openings': rows,
            'distinct': {k: len({r[k] for r in rows}) for k in ('history_id', 'position_id', 'class_id')},
            'rejected': dict(sorted(rejected.items()))}
    body['openings_sha256'] = _sha([r['moves'] for r in rows])
    return body


def check_suite(body: dict) -> None:
    """A suite file as ``build_suite`` writes it: every opening valid, ids recomputed, no exact repeat."""
    if body.get('format') != FORMAT or body.get('suite') not in SUITES:
        raise ValueError('not an H7-R opening suite')
    spec = SUITES[body['suite']]
    seen = set()
    for i, row in enumerate(body['openings']):
        moves = row['moves']
        if row['index'] != i or len(moves) != spec['moves'] or check_opening(moves) is not None or \
                {k: row[k] for k in ('history_id', 'position_id', 'class_id')} != ids(moves):
            raise ValueError(f'opening {i} is not a valid {body["suite"]} opening of this file')
        if row['history_id'] in seen:
            raise ValueError(f'opening {i} repeats an earlier one exactly')
        seen.add(row['history_id'])
    if body['openings_sha256'] != _sha([r['moves'] for r in body['openings']]) or body['count'] != len(seen):
        raise ValueError('the suite list is not its SHA-256 / count')


def pick(body: dict, pairs: int, seed: int) -> list[dict]:
    """``pairs`` distinct openings of a suite for one arm (a seeded sample without repeats)."""
    if pairs > body['count']:
        raise ValueError(f"{pairs} pairs asked, the suite has {body['count']} openings (no repeats)")
    chosen = Random(seed).sample(range(body['count']), pairs)
    return [body['openings'][i] for i in chosen]


def overlap(openings: list[dict], previous: dict[str, list]) -> dict:
    """How many of these openings an earlier evaluation used: the same history, position or symmetry class.
    ``previous``: name -> a list of move lists (a suite's or an Arena record's openings)."""
    mine = {k: {o[k] for o in openings} for k in ('history_id', 'position_id', 'class_id')}
    out = {}
    for name, histories in previous.items():
        theirs = [ids(m) for m in histories]
        out[name] = {k: len(mine[k] & {t[k] for t in theirs}) for k in mine}
    return out
