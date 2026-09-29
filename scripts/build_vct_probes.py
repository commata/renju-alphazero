"""Build the VCT/open-three probe set from lost human games (measurement only).

For every game the AI lost (``tests/fixtures/web_play_v7_human_games_v1.json`` by
default) this finds the **last AI decision that still had a VCF-safe move** and
proves, with ``analysis.threats.ThreatSolver`` at ``--vct-depth`` (default 1), what
happened around it. Only proven positions become probes:

- ``must_defend_vct``: that AI decision. ``correct_moves`` = moves proven SAFE at the
  VCT depth, ``avoid_moves`` = VCF-safe moves proven UNSAFE (the AI's choice when it
  was one). Emitted only when no candidate is UNKNOWN and at least one is SAFE.
- ``vct_loss``: the same decision when every move is proven UNSAFE (value -1).
- ``vct_attack``: the winner's reply position after the AI's UNSAFE choice
  (winner to move, value +1). ``correct_moves`` = every quiet threat proven to win
  (all replies then lose to VCF).
- ``vcf_loss``: the next AI decision, proven lost to VCF (value -1).

Each base probe is expanded to its 8 D4 images (Renju rules and the center opening
are D4-invariant); ``id`` suffix ``-sN`` names the symmetry. Proof statistics and
witnesses (0-based) are stored per probe under ``proof``. The file uses the
``stage7-probes-v1`` record format, so:

    python scripts/build_vct_probes.py
    python scripts/run_stage7_probes.py --checkpoint <ckpt> \
        --probes tests/fixtures/vct_probes_v1.json --output out.json

The solver is analysis-only and never used by search or training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from analysis.threats import (SAFE, UNKNOWN, UNSAFE, VCF_NONE,  # noqa: E402
                              ThreatSolver, decision_status, ordered_moves)
from renju import BLACK, Game  # noqa: E402

DEFAULT_GAMES = ROOT / 'tests' / 'fixtures' / 'web_play_v7_human_games_v1.json'
DEFAULT_OUTPUT = ROOT / 'tests' / 'fixtures' / 'vct_probes_v1.json'
BOARD = 15


def transform(move, symmetry: int):
    """D4 image of a 0-based coordinate; same convention as ``model.symmetry``."""
    row, col = move
    if symmetry >= 4:
        col = BOARD - 1 - col
    for _ in range(symmetry % 4):
        row, col = BOARD - 1 - col, row
    return [row, col]


def _color(player: int) -> str:
    return 'BLACK' if player == BLACK else 'WHITE'


def _replay(moves) -> Game:
    game = Game()
    for move in moves:
        game.play(*move)
    return game


def winning_threats(solver: ThreatSolver, game: Game, vct_depth: int) -> list:
    """Every quiet move of ``game.to_play`` after which all replies lose (depth - 1)."""
    attacker = game.to_play
    found = []
    for move in ordered_moves(game):
        game.play(*move)
        try:
            if game.done or solver.vcf(game, attacker)[0] == VCF_NONE:
                continue
            counts, _ = solver.decision(game, vct_depth - 1)
            if decision_status(counts) == UNSAFE:
                found.append(move)
        finally:
            game.undo()
    return found


def base_probes(game_record: dict, solver: ThreatSolver, vct_depth: int, log) -> list[dict]:
    moves = [tuple(m) for m in game_record['moves']]
    ai_plies = [i for i, actor in enumerate(game_record['actors']) if actor == 'AI']
    # Walk AI decisions backwards to the last one with a VCF-safe move.
    last_safe = None
    lost_after = None
    for index in reversed(ai_plies):
        game = _replay(moves[:index])
        counts, per_move = solver.decision(game, 0)
        if counts[UNKNOWN]:
            log(f"  ply {index + 1}: VCF UNKNOWN, stop")
            return []
        if counts[SAFE]:
            last_safe = (index, per_move)
            break
        lost_after = (index, counts)
    if last_safe is None or lost_after is None:
        return []

    index, per_move = last_safe
    source = {'game': game_record['name'], 'ply': index + 1,
              'route': game_record['ai_route'][index]}
    game = _replay(moves[:index])
    me = game.to_play
    chosen = moves[index]
    vcf_safe = sorted(m for m, (status, _) in per_move.items() if status == SAFE)
    started = perf_counter()
    deep = solver.classify(game, vcf_safe, vct_depth=vct_depth)
    counts = Counter(status for status, _ in deep.values())
    log(f"  ply {index + 1} ({source['route']}): VCF-safe {len(vcf_safe)} -> "
        f"VCT{vct_depth} {dict(counts)} ({perf_counter() - started:.0f}s)")
    proof = {'vct_depth': vct_depth, 'node_limit': solver.node_limit,
             'vcf_safe': len(vcf_safe), 'vcf_unsafe': len(per_move) - len(vcf_safe),
             'deep': dict(counts),
             'witnesses': {f'{m[0]},{m[1]}': list(w) for m, (s, w) in deep.items()
                           if s == UNSAFE}}
    result = []
    history = [list(m) for m in moves[:index]]
    if not counts[UNKNOWN]:
        safe = sorted(m for m, (s, _) in deep.items() if s == SAFE)
        if safe:
            result.append({'kind': 'must_defend_vct', 'to_play': _color(me), 'moves': history,
                           'correct_moves': [list(m) for m in safe],
                           'avoid_moves': [list(m) for m, (s, _) in sorted(deep.items())
                                           if s == UNSAFE],
                           'value_sign': None, 'source': source, 'proof': proof})
        else:
            result.append({'kind': 'vct_loss', 'to_play': _color(me), 'moves': history,
                           'correct_moves': [], 'avoid_moves': [], 'value_sign': -1,
                           'source': source, 'proof': proof})

    if chosen in deep and deep[chosen][0] == UNSAFE and deep[chosen][1][0] == 'threat':
        game.play(*chosen)
        started = perf_counter()
        threats = winning_threats(solver, game, vct_depth)
        log(f"  ply {index + 2}: {len(threats)} winning threat(s) ({perf_counter() - started:.0f}s)")
        if threats:
            result.append({'kind': 'vct_attack', 'to_play': _color(game.to_play),
                           'moves': history + [list(chosen)],
                           'correct_moves': [list(m) for m in threats], 'avoid_moves': [],
                           'value_sign': 1,
                           'source': {**source, 'ply': index + 2, 'route': 'human'},
                           'proof': {'vct_depth': vct_depth, 'node_limit': solver.node_limit}})
        game.undo()

    lost_index, lost_counts = lost_after
    result.append({'kind': 'vcf_loss', 'to_play': _color(_replay(moves[:lost_index]).to_play),
                   'moves': [list(m) for m in moves[:lost_index]], 'correct_moves': [],
                   'avoid_moves': [], 'value_sign': -1,
                   'source': {'game': game_record['name'], 'ply': lost_index + 1,
                              'route': game_record['ai_route'][lost_index]},
                   'proof': {'vct_depth': 0, 'node_limit': solver.node_limit,
                             'deep': dict(lost_counts)}})
    return result


def expand_d4(probe: dict, number: int) -> list[dict]:
    out = []
    for symmetry in range(8):
        item = dict(probe)
        for key in ('moves', 'correct_moves', 'avoid_moves'):
            item[key] = [transform(m, symmetry) for m in probe[key]]
        item['symmetry'] = symmetry
        item['id'] = f"{probe['kind']}-{number:03d}-s{symmetry}"
        out.append(item)
    return out


def build(games_path: Path, *, vct_depth: int, node_limit: int, log=print) -> dict:
    data = json.loads(games_path.read_bytes())
    solver = ThreatSolver(node_limit=node_limit)
    bases = []
    for record in data['games']:
        if record['result'] != 'HUMAN_WIN':
            continue
        log(f"== {record['name']}")
        bases.extend(base_probes(record, solver, vct_depth, log))
    probes = []
    for number, probe in enumerate(bases):
        probes.extend(expand_d4(probe, number))
    return {
        'format': 'stage7-probes-v1',
        'set_name': f'vct-v1-depth{vct_depth}',
        'coordinates': '0-based [row, col]; moves = full history before the probe position',
        'source_games': str(games_path.relative_to(ROOT)) if games_path.is_relative_to(ROOT)
        else str(games_path),
        'source_sha256': hashlib.sha256(games_path.read_bytes()).hexdigest(),
        'solver': {'module': 'analysis.threats', 'vct_depth': vct_depth,
                   'node_limit': node_limit, 'vcf_calls': solver.vcf_calls,
                   'vcf_exhausted': solver.vcf_exhausted},
        'counts': dict(Counter(p['kind'] for p in probes)),
        'base_probes': len(bases),
        'probes': probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--games', type=Path, default=DEFAULT_GAMES)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--vct-depth', type=int, default=1)
    parser.add_argument('--node-limit', type=int, default=100_000)
    args = parser.parse_args()
    result = build(args.games.resolve(), vct_depth=args.vct_depth, node_limit=args.node_limit)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n',
                           encoding='utf-8')
    print(f"wrote {args.output}: {result['counts']} (base {result['base_probes']})")


if __name__ == '__main__':
    main()
