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

from collections import Counter
import hashlib
import json
from pathlib import Path
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


def build_suite(name: str, count: int | None, seed: int | None, max_attempts: int = 200_000,
                exclude: set | frozenset | None = None) -> dict:
    """The suite file body: ``count`` openings (O0 / O1 from ``seed``; O2: all, or refused if ``count`` is larger).
    ``exclude``: history ids (recomputed from moves by ``exclusion``) never taken; the draw sequence is the same, an
    excluded draw is skipped, so the same seed and exclusion give the same suite. A shortfall fails: no repeat fills
    it."""
    spec = SUITES[name]
    rejected, histories, rows = {}, set(), []
    given, exclude = exclude is not None, frozenset(exclude or ())
    hit = set()  # distinct excluded openings met (a listed suite can meet one more than once)

    def add(moves):
        reason = check_opening(moves)
        if reason is None and len(moves) != spec['moves']:
            reason = f'{len(moves)} moves, the suite has {spec["moves"]}'
        if reason is not None:
            kind = reason.split(' [')[0].split(':')[0]
            rejected[kind] = rejected.get(kind, 0) + 1
            return
        i = ids(moves)
        if i['history_id'] in exclude:
            rejected['excluded'] = rejected.get('excluded', 0) + 1
            hit.add(i['history_id'])
            return
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
                raise ValueError(f"{name}: {count} distinct openings not found in {max_attempts} draws ({len(rows)} "
                                 f"found; {rejected.get('excluded', 0)} draws excluded, "
                                 f"{rejected.get('exact repeat', 0)} exact repeats); no repeat fills a shortfall")
            add(random_opening(rng, spec['stages']))
    else:
        for moves in standard26():
            for k in range(spec['orientations']):
                add([list(symmetry(m, k)) for m in moves])
        if count is not None and count > len(rows):
            raise ValueError(f'{name} has {len(rows)} distinct openings left after excluding {len(hit)} used '
                             f'ones; {count} asked (no repeats)')
        if count is not None:
            rows = rows[:count]
    body = {'format': FORMAT, 'suite': name, 'rule': {**spec, 'stages': [list(s) for s in spec.get('stages', [])]},
            'next_to_move': spec['next'], 'seed': seed, 'count': len(rows), 'openings': rows,
            'distinct': {k: len({r[k] for r in rows}) for k in ('history_id', 'position_id', 'class_id')},
            'rejected': dict(sorted(rejected.items()))}
    body['openings_sha256'] = _sha([r['moves'] for r in rows])
    if given:
        ordered = sorted(exclude)
        body['exclusion'] = {'history_ids': ordered, 'history_ids_sha256': _sha(ordered),
                             'excluded_draws': rejected.get('excluded', 0)}
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
    if 'exclusion' in body:
        ex = body['exclusion']
        if _sha(ex['history_ids']) != ex['history_ids_sha256'] or seen & set(ex['history_ids']):
            raise ValueError('the suite holds an excluded opening or its exclusion list is not its SHA-256')


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


# --- exclusion of openings already used --------------------------------------------------------------------------

# Records of openings an evaluation played; each ``openings`` entry is a dict with ``moves`` (or a move list).
PLAYED_FORMATS = ('h7r-arena-v1-arm', 'h7c-arena-v1-arm', 'h7c-nmask-arena-v1-arm', 'h7c-drift-v1',
                  'external-match-v1', 'mcts-v7-benchmark-v1', 'mcts-v8-benchmark-v1')
CONTRACT_FORMAT = 'h7r-eval-contract-v1'


def _file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _moves(value, where) -> list[list[int]]:
    if not (isinstance(value, list) and value and all(
            isinstance(m, list) and len(m) == 2 and all(type(x) is int and 0 <= x < SIZE for x in m) for m in value)):
        raise ValueError(f'{where}: an opening that is not a move list')
    return [list(m) for m in value]


def used_openings(path, root) -> dict:
    """The openings an evaluation used. A played record (``PLAYED_FORMATS``): its openings as played. An H7-R contract:
    every arm's drawn sample (``pick``, after the suite file's SHA-256 is checked): the openings its arms play when
    complete. A suite file is refused: it lists candidates, not openings an evaluation used."""
    path = Path(path)
    body = json.loads(path.read_text(encoding='utf-8'))
    fmt = body.get('format') if isinstance(body, dict) else None
    if fmt == FORMAT:
        raise ValueError(f'{path}: a suite file lists candidates, not the openings an evaluation used (give the arm '
                         'records or the contract; --disjoint-from reserves a suite for another sample)')
    if fmt in PLAYED_FORMATS:
        moves = [_moves(o['moves'] if isinstance(o, dict) else o, f'{path} opening {i}')
                 for i, o in enumerate(body.get('openings') or [])]
        kind = 'played openings'
    elif fmt == CONTRACT_FORMAT:
        moves = []
        for arm in body.get('arms', []):
            suite_path = Path(root) / arm['suite']['path']
            if _file_sha(suite_path) != arm['suite']['sha256']:
                raise ValueError(f"{path}: arm {arm['name']}: {suite_path} is not its SHA-256")
            suite = json.loads(suite_path.read_text(encoding='utf-8'))
            moves += [_moves(o['moves'], f"{path} arm {arm['name']}") for o in pick(suite, arm['pairs'], arm['seed'])]
        kind = 'contract arm samples (pick of each arm)'
    else:
        raise ValueError(f'{path}: format {fmt!r} is not a known evaluation record ({PLAYED_FORMATS}, {CONTRACT_FORMAT})')
    return {'kind': kind, 'format': fmt, 'moves': moves}


def _where(path, root) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def exclusion(used_paths, root, disjoint_paths=()) -> dict:
    """History ids to exclude, recomputed from moves: the openings of ``used_paths`` (``used_openings``) and of the
    suites in ``disjoint_paths`` (reserved for another sample of the same design). Every source with its path,
    SHA-256, kind and counts."""
    sources, histories, moves = [], set(), []
    for path in used_paths:
        used = used_openings(path, root)
        mine = {ids(m)['history_id'] for m in used['moves']}
        sources.append({'path': _where(path, root), 'sha256': _file_sha(path), 'kind': used['kind'],
                        'format': used['format'], 'openings': len(used['moves']), 'distinct_histories': len(mine),
                        'lengths': dict(sorted(Counter(len(m) for m in used['moves']).items()))})
        histories |= mine
        moves += used['moves']
    for path in disjoint_paths:
        body = json.loads(Path(path).read_text(encoding='utf-8'))
        check_suite(body)
        mine = {ids(o['moves'])['history_id'] for o in body['openings']}
        sources.append({'path': _where(path, root), 'sha256': _file_sha(path),
                        'kind': 'suite reserved for another sample (disjoint)', 'format': FORMAT,
                        'openings': body['count'], 'distinct_histories': len(mine),
                        'lengths': {str(SUITES[body['suite']]['moves']): body['count']}})
        histories |= mine
        moves += [o['moves'] for o in body['openings']]
    return {'history_ids': histories, 'sources': sources, 'moves': moves}


def prefix_overlap(openings: list[dict], previous: list) -> dict:
    """Openings whose first k moves are a whole earlier opening of length k < theirs (e.g. a 5-move opening that
    continues a 3-move one): shared beginnings, not exact repeats. Counted per k."""
    by_length = {}
    for m in previous:
        by_length.setdefault(len(m), set()).add(tuple(map(tuple, m)))
    out = {}
    for k, earlier in sorted(by_length.items()):
        hits = [o for o in openings if len(o['moves']) > k and tuple(map(tuple, o['moves'][:k])) in earlier]
        if any(len(o['moves']) > k for o in openings):
            out[str(k)] = len(hits)
    return out
