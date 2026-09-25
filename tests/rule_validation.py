"""Shared deterministic oracle corpus; never imported by production code."""
from collections import Counter
from copy import deepcopy
from random import Random
import json
from pathlib import Path

from agents import RandomAgent, TacticalAgent
from renju import Game
from renju.rules import forbidden_reason
import reference_rules as reference


def fixtures():
    cross = [(7,6),(7,8),(6,7),(8,7)]
    cases = [
        ('double-three', cross, []),
        ('blocked-three', cross, [(7,4),(7,10)]),
        ('recursive-three', cross + [(6,5),(8,5),(6,4),(8,6)], [(7,10)]),
        ('double-four', [(7,c) for c in (5,6,8)] + [(r,7) for r in (5,6,8)], []),
        ('same-axis-four', [(7,7+i) for i in (-4,-3,-1,1,3,4)], []),
        ('overline', [(7,c) for c in (3,4,5,6,8)] + [(r,7) for r in (3,4,5,6)], []),
        ('exact-five', [(7,c) for c in (3,4,5,6)] + [(6,7),(8,7),(6,6),(8,8)], []),
        ('edge-three', [(1,0),(1,1),(1,2),(0,3),(2,3)], []),
        ('edge-four', [(2,0),(2,1),(2,2),(1,3),(3,3)], []),
        ('cross-overline', [(7,3),(7,4),(7,5),(4,7),(5,7),(6,7),(8,7),(9,7)], []),
        ('forbidden-extension', [(7,4),(7,5),(7,6),(6,7),(8,7)] +
         [(r,c) for r in (5,9) for c in (4,5,6)], [(7,3)]),
        ('recursive-extension', [(7,4),(7,5),(7,6),(6,7),(8,7)] +
         [(r+dr,c) for r in (5,9) for dr,c in ((0,6),(0,8),(-1,6),(1,8))],
         [(7,3),(4,10),(10,4)]),
    ]
    for name, black, white in cases:
        for rotation in range(4):
            game = Game()
            for color, cells in ((1, black), (-1, white)):
                for r,c in cells:
                    for _ in range(rotation):
                        r,c = c,14-r
                    game.board[r][c] = color
            yield f'{name}/rotation={rotation}', game


def positions(seed, count):
    if count < 48:
        raise ValueError('positions must be at least 48 to include all fixtures')
    yield from fixtures()
    # Frozen, complete V6 games generated before any production optimization.
    histories = json.loads((Path(__file__).parent / 'fixtures/rule_v6_smoke.json').read_text())
    replayed = 0
    for record in histories:
        game = Game()
        for move in record['history']:
            if 48 + replayed >= count:
                return
            game.play(*move)
            replayed += 1
            yield 'v6-replay', deepcopy(game)
    rng = Random(seed)
    games = [Game(), Game()]
    agents = [RandomAgent(seed), TacticalAgent(seed+1)]
    for i in range(count - 48 - replayed):
        if i % 3 == 0:
            game = Game()
            # Mix central clusters with full-board density and both stone colors.
            central = i % 2 == 0
            density = rng.uniform(.2, .8)
            for r in range(15):
                for c in range(15):
                    if (not central or 3 <= r <= 11 and 3 <= c <= 11) and rng.random() < density:
                        game.board[r][c] = 1 if rng.random() < .8 else -1
            game.to_play = rng.choice((1,-1))
            yield 'synthetic', game
        else:
            index = i % 3 - 1
            game = games[index]
            if game.done:
                game = games[index] = Game()
            for _ in range(rng.randint(1, 4)):
                game.play(*agents[index].select_move(game))
                if game.done:
                    break
            yield ('random', 'tactical')[index], deepcopy(game)


def validate(corpus, progress=None):
    stats = Counter({'positions': 0, 'cells': 0, 'None': 0, '삼삼': 0, '사사': 0,
                     '장목': 0, 'mismatches': 0, 'legal_mismatches': 0,
                     'restoration_failures': 0})
    for label, game in corpus:
        before = [row[:] for row in game.board]
        empty, legal = [], []
        for r in range(15):
            for c in range(15):
                if before[r][c] != 0:
                    continue
                empty.append((r,c))
                expected = reference.forbidden_reason(game.board, r, c)
                if game.board != before:
                    stats['restoration_failures'] += 1
                    raise AssertionError((label, r, c, dict(stats)))
                actual = forbidden_reason(game.board, r, c)
                stats['cells'] += 1
                stats[str(expected)] += 1
                if game.board != before:
                    stats['restoration_failures'] += 1
                    raise AssertionError((label, r, c, dict(stats)))
                if actual != expected:
                    stats['mismatches'] += 1
                    raise AssertionError((label, r, c, expected, actual, before, dict(stats)))
                if expected is None:
                    legal.append((r,c))
        # Both turns, initial opening, and terminal semantics; ordered lists.
        turn = game.to_play
        for color in (1,-1):
            game.to_play = color
            expected = legal if color == 1 else empty
            if game.done:
                expected = []
            elif color == 1 and not game.history and len(empty) == 225:
                expected = [(7,7)]
            if game.legal_moves() != expected:
                stats['legal_mismatches'] += 1
                raise AssertionError((label, color, dict(stats)))
            if game.board != before:
                stats['restoration_failures'] += 1
                raise AssertionError((label, dict(stats)))
        game.to_play = turn
        stats['positions'] += 1
        stats['source/' + label.split('/')[0]] += 1
        if progress and stats['positions'] % 250 == 0:
            progress(dict(stats))
    return dict(stats)
