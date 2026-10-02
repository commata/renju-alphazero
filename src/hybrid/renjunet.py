"""RenjuNet RIF database -> validated Track B game set (H2, docs/mcts-v8-teacher.md §12.6, §12.10).

Every game gets exactly one primary outcome, checked in this order (the first that
applies wins, so the counts always add up to the number of ``<game>`` records):

    non_renju_rule       rule category is not 1 (Gomoku rules have no forbidden moves)
    bad_coordinate       a move token is not a1..o15
    occupied_point       a move lands on a stone already on the board
    non_center_first     the first move is not h8 (our rules fix it at the centre)
    midgame_forbidden    a black forbidden move with more moves recorded after it
    moves_after_five     moves recorded after a five ended the game
    too_short            fewer than ``MIN_MOVES`` usable moves (after truncation)
    result_mismatch      the board result contradicts ``bresult`` (five by the other
                         colour, a draw recorded on a five, or a terminal black
                         forbidden move recorded as a black win or draw)
    duplicate_game       the same move sequence up to D4 as an accepted game with a
                         smaller id
    accepted

A terminal black forbidden move (the last recorded move) is a black loss under
renju rules: the game is kept up to the move before it (``truncated_forbidden``)
and that move is never a policy target.

The database has no reliable termination field. ``<info>`` is free text (mostly
opening-procedure notes) on about 9% of games, so the only end type we can derive
is ``five`` (the board shows the win), ``forbidden`` (truncated black forbidden
move) or ``unknown`` (resignation, time, agreement, ... cannot be told apart).

Coordinates: letter a..o is the column, number 1..15 counts rows from the bottom
(h8 is the centre). ``row = 15 - number`` keeps row 0 at the top as in ``renju``;
D4 symmetry makes the orientation irrelevant for training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
import re
import xml.etree.ElementTree as ET

from renju import Game, IllegalMove, SIZE
from renju.game import OPENING_MOVE

MIN_MOVES = 10
POLICY_FROM_PLY = 5  # plies 0..4 (moves 1-5) follow opening procedures: board only, no policy target
RENJU_CATEGORY = '1'

REASONS = ('non_renju_rule', 'bad_coordinate', 'occupied_point', 'non_center_first',
           'midgame_forbidden', 'moves_after_five', 'too_short', 'result_mismatch',
           'duplicate_game', 'accepted')

_TOKEN = re.compile(r'^([a-o])(1[0-5]|[1-9])$')


def parse_move(token: str) -> tuple[int, int] | None:
    match = _TOKEN.match(token)
    if match is None:
        return None
    return SIZE - int(match.group(2)), ord(match.group(1)) - ord('a')


@dataclass
class RawGame:
    id: int
    tournament: int
    rule: int
    bresult: str
    tokens: list[str]
    attrs: dict = field(repr=False, default_factory=dict)


@dataclass
class Database:
    rule_category: dict[int, str]
    tournaments: dict[int, dict]
    games: list[RawGame]


def read_rif(path) -> Database:
    """Parse a RIF XML file. Games are returned in id order."""
    root = ET.parse(path).getroot()
    rules = {int(r.get('id')): r.get('category') for r in root.iter('rule')}
    tournaments = {int(t.get('id')): dict(t.attrib) for t in root.iter('tournament')}
    games = []
    for g in root.iter('game'):
        move = g.find('move')
        tokens = (move.text or '').split() if move is not None else []
        games.append(RawGame(id=int(g.get('id')), tournament=int(g.get('tournament')),
                             rule=int(g.get('rule')), bresult=g.get('bresult', ''),
                             tokens=tokens, attrs=dict(g.attrib)))
    games.sort(key=lambda game: game.id)
    return Database(rules, tournaments, games)


@dataclass
class Replay:
    reason: str            # a REASONS entry other than duplicate_game / accepted, or '' when usable
    moves: list            # usable moves (after truncation)
    end: str = 'unknown'   # five | forbidden | unknown
    winner: str = ''       # board winner: 'black' | 'white' | '' (none on the board)
    detail: str = ''


def replay(raw: RawGame, category: str | None) -> Replay:
    """Steps 1-8 of the module docstring for one game (not the duplicate check)."""
    if category != RENJU_CATEGORY:
        return Replay('non_renju_rule', [])
    moves = []
    for token in raw.tokens:
        move = parse_move(token)
        if move is None:
            return Replay('bad_coordinate', [], detail=token)
        moves.append(move)
    seen = set()
    for move in moves:
        if move in seen:
            return Replay('occupied_point', [], detail=str(move))
        seen.add(move)
    if moves and moves[0] != OPENING_MOVE:
        return Replay('non_center_first', [])
    game = Game()
    end = 'unknown'
    for index, move in enumerate(moves):
        if game.done:
            return Replay('moves_after_five', [], detail=f'ply {index}')
        try:
            game.play(*move)
        except IllegalMove as exc:
            if index != len(moves) - 1:
                return Replay('midgame_forbidden', [], detail=f'ply {index}: {exc}')
            moves = moves[:index]
            end = 'forbidden'
            break
    if game.done and game.winner is not None:
        end = 'five'
    winner = {1: 'black', -1: 'white'}.get(game.winner, '') if end == 'five' else (
        'white' if end == 'forbidden' else '')
    if len(moves) < MIN_MOVES:
        return Replay('too_short', moves, end, winner)
    expected = {'black': '1', 'white': '0'}.get(winner)
    if expected is not None and raw.bresult != expected:
        return Replay('result_mismatch', moves, end, winner, detail=f'board {winner}, bresult {raw.bresult}')
    return Replay('', moves, end, winner)


def transform(move, symmetry: int):
    """D4 as ``model.symmetry``: ids 0..3 rotate CCW, 4..7 mirror columns then rotate."""
    row, col = move
    if symmetry >= 4:
        col = SIZE - 1 - col
    for _ in range(symmetry % 4):
        row, col = SIZE - 1 - col, row
    return row, col


def sequence_key(moves) -> str:
    """Game identity up to D4 (the whole move order)."""
    return min(','.join(f'{r},{c}' for r, c in (transform(m, s) for m in moves)) for s in range(8))


class Zobrist:
    """128-bit Zobrist keys; ``position_keys`` gives the D4-canonical key of every prefix."""

    def __init__(self, seed: int = 20261002):
        rng = random.Random(seed)
        self.table = [[[rng.getrandbits(128) for _ in range(SIZE)] for _ in range(SIZE)] for _ in range(2)]

    def position_keys(self, moves) -> list[int]:
        """Key of the position before ply 0, 1, ..., len(moves) - 1 (the policy states).

        Stone placement only: the side to move follows from the stone count.
        """
        hashes = [0] * 8
        keys = []
        for ply, move in enumerate(moves):
            keys.append(min(hashes))
            colour = ply % 2
            for s in range(8):
                r, c = transform(move, s)
                hashes[s] ^= self.table[colour][r][c]
        return keys


def split_tournaments(games, fractions=(('test', 0.05), ('val', 0.05)), salt: str = 'h2-v1') -> dict[int, str]:
    """Deterministic tournament -> split: tournaments in salted-hash order fill test, then val; rest is train."""
    sizes = {}
    for game in games:
        sizes[game['tournament']] = sizes.get(game['tournament'], 0) + 1
    order = sorted(sizes, key=lambda t: hashlib.sha256(f'{salt}:{t}'.encode()).hexdigest())
    total = sum(sizes.values())
    assignment, index = {}, 0
    for name, fraction in fractions:
        filled = 0
        while index < len(order) and filled < fraction * total:
            assignment[order[index]] = name
            filled += sizes[order[index]]
            index += 1
    for tournament in order[index:]:
        assignment[tournament] = 'train'
    return assignment


SPLIT_ORDER = ('train', 'val', 'test')


def mask_leakage(games, zobrist: Zobrist) -> dict:
    """Mark val/test policy states whose D4-canonical position occurs in an earlier split.

    train keys are never touched; val states found in train, and test states found in
    train or val, get ``masked_plies``. Returns counts per split.
    """
    seen: set[int] = set()
    stats = {}
    for split in SPLIT_ORDER:
        members = [g for g in games if g['split'] == split]
        keys_here = set()
        states = masked = masked_policy = 0
        for game in members:
            keys = zobrist.position_keys(game['moves'])
            game['masked_plies'] = [ply for ply, key in enumerate(keys) if key in seen] if split != 'train' else []
            states += len(keys)
            masked += len(game['masked_plies'])
            masked_policy += sum(ply >= POLICY_FROM_PLY for ply in game['masked_plies'])
            keys_here.update(keys)
        seen |= keys_here
        stats[split] = {'games': len(members), 'states': states, 'masked_states': masked,
                        'policy_states': sum(max(0, len(g['moves']) - POLICY_FROM_PLY) for g in members),
                        'masked_policy_states': masked_policy}
    return stats


def leakage_check(games, zobrist: Zobrist) -> dict:
    """Independent re-check: unmasked states of different splits never share a key."""
    keys = {split: set() for split in SPLIT_ORDER}
    for game in games:
        masked = set(game['masked_plies'])
        for ply, key in enumerate(zobrist.position_keys(game['moves'])):
            if ply not in masked:
                keys[game['split']].add(key)
    return {'train_val': len(keys['train'] & keys['val']), 'train_test': len(keys['train'] & keys['test']),
            'val_test': len(keys['val'] & keys['test'])}


def build(db: Database, *, zobrist: Zobrist | None = None, progress=None) -> tuple[list[dict], dict]:
    """Classify every game, split the accepted ones, mask leakage. Returns (games, report)."""
    zobrist = zobrist or Zobrist()
    counts = {reason: 0 for reason in REASONS}
    by_rule: dict[str, dict] = {}
    examples: dict[str, list] = {}
    ends = {'five': 0, 'forbidden': 0, 'unknown': 0}
    seen_sequences: dict[str, int] = {}
    accepted = []
    for index, raw in enumerate(db.games):
        result = replay(raw, db.rule_category.get(raw.rule))
        reason = result.reason
        if not reason:
            key = sequence_key(result.moves)
            if key in seen_sequences:
                reason = 'duplicate_game'
                result.detail = f'same as game {seen_sequences[key]}'
            else:
                seen_sequences[key] = raw.id
                reason = 'accepted'
        counts[reason] += 1
        rule_row = by_rule.setdefault(str(raw.rule), {'category': db.rule_category.get(raw.rule), 'games': 0,
                                                     'accepted': 0})
        rule_row['games'] += 1
        if reason != 'accepted':
            if len(examples.setdefault(reason, [])) < 5:
                examples[reason].append({'id': raw.id, 'detail': result.detail})
        else:
            rule_row['accepted'] += 1
            ends[result.end] += 1
            accepted.append({'id': raw.id, 'tournament': raw.tournament, 'rule': raw.rule,
                             'bresult': raw.bresult, 'end': result.end, 'winner': result.winner,
                             'moves': [list(m) for m in result.moves]})
        if progress is not None and (index + 1) % 10_000 == 0:
            progress(index + 1, len(db.games))
    assignment = split_tournaments(accepted)
    for game in accepted:
        game['split'] = assignment[game['tournament']]
    leakage = mask_leakage(accepted, zobrist)
    report = {
        'games_in_file': len(db.games),
        'primary_reason': counts,
        'sum_check': sum(counts.values()) == len(db.games),
        'accepted_end': ends,
        'accepted_bresult': {v: sum(g['bresult'] == v for g in accepted) for v in ('1', '0', '0.5')},
        'by_rule': by_rule,
        'examples': examples,
        'splits': leakage,
        'tournaments': {s: len({g['tournament'] for g in accepted if g['split'] == s}) for s in SPLIT_ORDER},
        'leakage_after_masking': leakage_check(accepted, zobrist),
        'policy_from_ply': POLICY_FROM_PLY,
        'min_moves': MIN_MOVES,
    }
    return accepted, report


def verify_games(games) -> int:
    """Replay every output game with ``Game``; returns the number of failures (must be 0)."""
    failures = 0
    for record in games:
        game = Game()
        try:
            for move in record['moves']:
                if game.done:
                    raise IllegalMove('move after the end')
                game.play(*move)
        except IllegalMove:
            failures += 1
    return failures
