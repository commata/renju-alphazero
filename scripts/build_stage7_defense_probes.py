"""Build the Stage 7 open-three defense probe set (measurement only, no search rule).

Kind ``must_defend_open3``: the opponent has a legal move that creates two or more
winning points (an open four or a four-four) - i.e. it wins in two moves - while the
side to move has no winning point and no move that makes a four (so a counter-attack
by fours is impossible). The exact correct set is every legal move after which the
opponent has no such double-threat move left. Labels use only Renju rules
(``search.tactics.winning_points`` + the engine's black forbidden test).

This set exists to track whether self-play learns 2-ply defense WITHOUT a search rule
(Stage 7-C): PUCT v2 blocks fours only, so these positions are outside its rules.
The file uses the ``stage7-probes-v1`` record format so ``run_stage7_probes.py`` can
read it (``--probes ... --suffix _defense``).

    python scripts/build_stage7_defense_probes.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from renju import BLACK, EMPTY, Game  # noqa: E402
from renju.rules import forbidden_reason  # noqa: E402
from search.tactics import WINDOWS, winning_points  # noqa: E402

DEFAULT_OUTPUT = ROOT / 'tests' / 'fixtures' / 'stage7_probes_defense_v1.json'
BENCHMARK_FILES = (
    ROOT / 'docs' / 'mcts-v7-results' / 'v7_vs_v6_seed6507.json',
    ROOT / 'docs' / 'mcts-v7-results' / 'v7_vs_v5_seed6507.json',
)
KIND = 'must_defend_open3'


def _legal_for(board, player: int, move) -> bool:
    r, c = move
    return board[r][c] == EMPTY and (player != BLACK or forbidden_reason(board, r, c) is None)


def _candidate_cells(board, player: int, minimum: int) -> set:
    """Empty cells of five-windows holding >= ``minimum`` player stones and no opponent."""
    cells = set()
    for window in WINDOWS:
        values = [board[r][c] for r, c in window]
        if values.count(player) >= minimum and values.count(-player) == 0:
            cells.update(p for p, v in zip(window, values) if v == EMPTY)
    return cells


def double_threat_moves(board, player: int) -> list:
    """Legal moves after which ``player`` has two or more winning points."""
    found = []
    for move in sorted(_candidate_cells(board, player, 3)):
        if not _legal_for(board, player, move):
            continue
        r, c = move
        board[r][c] = player
        try:
            if len(winning_points(board, player)) >= 2:
                found.append(move)
        finally:
            board[r][c] = EMPTY
    return found


def four_makers(board, player: int) -> list:
    """Legal moves after which ``player`` has at least one winning point."""
    found = []
    for move in sorted(_candidate_cells(board, player, 3)):
        if not _legal_for(board, player, move):
            continue
        r, c = move
        board[r][c] = player
        try:
            if winning_points(board, player):
                found.append(move)
        finally:
            board[r][c] = EMPTY
    return found


def defense_label(game: Game) -> list | None:
    """Correct defensive moves, or None when the position is not a clean probe."""
    board, player = game.board, game.to_play
    opponent = -player
    if winning_points(board, player) or winning_points(board, opponent):
        return None
    if not double_threat_moves(board, opponent) or four_makers(board, player):
        return None
    correct = []
    for r, c in game.legal_moves():
        board[r][c] = player
        try:
            if not double_threat_moves(board, opponent):
                correct.append((r, c))
        finally:
            board[r][c] = EMPTY
    return correct or None


def candidates():
    for path in BENCHMARK_FILES:
        data = json.loads(path.read_text(encoding='utf-8'))
        for record in data['games']:
            game = Game()
            for ply, move in enumerate(record['moves'], 1):
                if ply > 1 and not game.done:
                    board, opponent = game.board, -game.to_play
                    # cheap pre-filter before the exact label
                    if double_threat_moves(board, opponent):
                        yield game, {'file': path.name, 'pair': record['pair'],
                                     'v7_color': record['v7_color'], 'ply': ply}
                game.play(*move)


def build(per_colour: int) -> dict:
    chosen = {'BLACK': {}, 'WHITE': {}}
    for game, source in candidates():
        colour = 'BLACK' if game.to_play == BLACK else 'WHITE'
        key = hashlib.sha256(json.dumps([list(m) for m in game.history]).encode()).hexdigest()
        if key in chosen[colour]:
            continue
        correct = defense_label(game)
        if correct is None:
            continue
        chosen[colour][key] = {
            'kind': KIND, 'source': source, 'to_play': colour,
            'moves': [list(m) for m in game.history],
            'correct_moves': [list(m) for m in correct], 'avoid_moves': [],
            'value_sign': None,
        }
    probes = []
    for colour in ('BLACK', 'WHITE'):
        probes.extend(item for _, item in sorted(chosen[colour].items())[:per_colour])
    for index, probe in enumerate(probes):
        probe['id'] = f'{KIND}-{index:03d}'
    return {
        'format': 'stage7-probes-v1',
        'set_name': 'stage7-defense-v1',
        'coordinates': '0-based [row, col]; moves = full history before the probe position',
        'sources': [str(p.relative_to(ROOT)).replace('\\', '/') for p in BENCHMARK_FILES],
        'counts': {c: sum(p['to_play'] == c for p in probes) for c in ('BLACK', 'WHITE')},
        'probes': probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--per-colour', type=int, default=20)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(args.per_colour)
    args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'counts': payload['counts']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
