"""Build the fixed Stage 7 tactical/value probe set (tests/fixtures/stage7_probes_v1.json).

Sources (both already committed, so the build is reproducible):
- the 200 seed-6507 MCTS-v7 benchmark games in docs/mcts-v7-results/ (every position);
- the seed 44 VCF regression fixture tests/fixtures/mcts_v7_vcf_regression.json.

Probe kinds and labels:
- immediate_win: side to move has a winning move. correct = every legal winning move
  (exact, from the rules engine). value = +1 (side to move).
- must_block: side to move cannot win now and the opponent has exactly one winning
  point, which is legal for the side to move. correct = that point. No value label
  (blocking does not decide the game).
- forced_loss: side to move cannot win now and the opponent has two or more distinct
  winning points. No policy label. value = -1.
- vcf: side to move has a victory by continuous fours (conservative solver). correct =
  every first move that wins or starts a VCF. value = +1.
- avoid: the recorded losing move of the regression fixture (it allowed an opponent
  VCF). Reported as probability mass placed on the avoid move, not as a hit rate.

Black forbidden points are illegal in the engine, so they are masked out of the
policy; they are never used as a policy label. See run_stage7_probes.py for the
separate unmasked-logit diagnostic.

Selection is deterministic: candidates are de-duplicated by position and ordered by
SHA-256 of the move history, and at most --per-kind positions are kept per kind, split
evenly between BLACK and WHITE to move where possible.
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

from renju import BLACK, Game  # noqa: E402
from search.mcts import _is_legal_for_player, _wins_for_player  # noqa: E402
from search.mcts_v5 import _window_candidates  # noqa: E402
from search.mcts_v7 import _find_vcf_with_stats, _legal_completions, placed  # noqa: E402

PROBE_FORMAT = 'stage7-probes-v1'
DEFAULT_OUTPUT = ROOT / 'tests' / 'fixtures' / 'stage7_probes_v1.json'
BENCHMARK_FILES = (
    ROOT / 'docs' / 'mcts-v7-results' / 'v7_vs_v6_seed6507.json',
    ROOT / 'docs' / 'mcts-v7-results' / 'v7_vs_v5_seed6507.json',
)
REGRESSION_FILE = ROOT / 'tests' / 'fixtures' / 'mcts_v7_vcf_regression.json'
VCF_MAX_FOURS = 20
VCF_NODE_LIMIT = 20000


def color_name(player: int) -> str:
    return 'BLACK' if player == BLACK else 'WHITE'


def replay(moves) -> Game:
    game = Game()
    for row, col in moves:
        game.play(row, col)
    return game


def winning_points(game: Game, player: int) -> list[tuple[int, int]]:
    """Exact winning points for ``player`` regardless of whose turn it is.

    Any five (or white overline) through a new stone contains a five-cell window
    holding four of ``player``'s stones and that empty cell, so the window
    candidates are exhaustive.
    """
    return sorted(
        move for move in _window_candidates(game, player, 4)
        if _is_legal_for_player(game, player, move)
        and _wins_for_player(game, player, move)
    )


def vcf_first_moves(game: Game) -> list[tuple[int, int]]:
    """Every first move that wins now or starts a (conservative) VCF for the mover."""
    player, defender = game.to_play, -game.to_play
    result = []
    for move in _window_candidates(game, player, 3):
        if not _is_legal_for_player(game, player, move):
            continue
        if _wins_for_player(game, player, move):
            result.append(move)
            continue
        with placed(game, player, move):
            completions = _legal_completions(game, player, move)
            if not completions or winning_points(game, defender):
                continue
            if len(completions) >= 2:
                result.append(move)
                continue
            block = completions[0]
            if not _is_legal_for_player(game, defender, block):
                result.append(move)
                continue
            with placed(game, defender, block):
                if _legal_completions(game, defender, block):
                    continue  # the block itself makes a four for the defender
                found, _, _ = _find_vcf_with_stats(
                    game, player, max_fours=VCF_MAX_FOURS, node_limit=VCF_NODE_LIMIT)
                if found is not None:
                    result.append(move)
    return sorted(result)


def _probe(kind, source, moves, game, *, correct=(), avoid=(), value=None):
    return {
        'kind': kind,
        'source': source,
        'to_play': color_name(game.to_play),
        'moves': [list(m) for m in moves],
        'correct_moves': [list(m) for m in correct],
        'avoid_moves': [list(m) for m in avoid],
        'value_sign': value,
    }


def classify_position(game: Game, source: dict) -> list[dict]:
    if game.done:
        return []
    moves = list(game.history)
    player, opponent = game.to_play, -game.to_play
    own = winning_points(game, player)
    if own:
        return [_probe('immediate_win', source, moves, game, correct=own, value=1)]
    threats = winning_points(game, opponent)
    if len(threats) >= 2:
        return [_probe('forced_loss', source, moves, game, value=-1)]
    if len(threats) == 1 and _is_legal_for_player(game, player, threats[0]):
        return [_probe('must_block', source, moves, game, correct=threats)]
    return []


def benchmark_candidates() -> list[dict]:
    out = []
    for path in BENCHMARK_FILES:
        data = json.loads(path.read_text(encoding='utf-8'))
        for record in data['games']:
            game = Game()
            for ply, move in enumerate(record['moves'], 1):
                if ply > 1:
                    source = {'file': path.name, 'pair': record['pair'],
                              'v7_color': record['v7_color'], 'ply': ply}
                    out.extend(classify_position(game, source))
                game.play(*move)
    return out


def regression_candidates() -> list[dict]:
    data = json.loads(REGRESSION_FILE.read_text(encoding='utf-8'))
    out = []
    for item in data['fixtures']:
        moves = [(r - 1, c - 1) for r, c in item['moves']]
        game = replay(moves)
        source = {'file': REGRESSION_FILE.name, 'id': item['id']}
        if item['kind'] in ('missed_own_vcf', 'vcf_streak_start'):
            first = vcf_first_moves(game)
            if first:
                out.append(_probe('vcf', source, moves, game, correct=first, value=1))
        elif item['kind'] == 'losing_move_allows_vcf':
            losing = tuple(x - 1 for x in item['losing_move'])
            out.append(_probe('avoid', source, moves, game, avoid=[losing]))
    return out


def position_key(probe: dict) -> str:
    return hashlib.sha256(json.dumps(probe['moves']).encode('utf-8')).hexdigest()


def select(candidates: list[dict], per_kind: int) -> list[dict]:
    by_kind: dict[str, dict[str, dict]] = {}
    for probe in candidates:
        by_kind.setdefault(probe['kind'], {}).setdefault(position_key(probe), probe)
    selected = []
    for kind in sorted(by_kind):
        ordered = sorted(by_kind[kind].items())
        black = [p for _, p in ordered if p['to_play'] == 'BLACK']
        white = [p for _, p in ordered if p['to_play'] == 'WHITE']
        half = per_kind // 2
        take_black = min(len(black), max(half, per_kind - len(white)))
        take_white = min(len(white), per_kind - take_black)
        chosen = black[:take_black] + white[:take_white]
        selected.extend(sorted(chosen, key=position_key))
    for index, probe in enumerate(selected):
        probe['id'] = f"{probe['kind']}-{index:03d}"
    return selected


def build(per_kind: int) -> dict:
    candidates = benchmark_candidates() + regression_candidates()
    probes = select(candidates, per_kind)
    counts = {}
    for probe in probes:
        key = f"{probe['kind']}/{probe['to_play']}"
        counts[key] = counts.get(key, 0) + 1
    return {
        'format': PROBE_FORMAT,
        'coordinates': '0-based [row, col]; moves = full history before the probe position',
        'per_kind_limit': per_kind,
        'vcf_solver': {'max_fours': VCF_MAX_FOURS, 'node_limit': VCF_NODE_LIMIT},
        'sources': [str(p.relative_to(ROOT)).replace('\\', '/')
                    for p in (*BENCHMARK_FILES, REGRESSION_FILE)],
        'counts': dict(sorted(counts.items())),
        'probes': probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--per-kind', type=int, default=40)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build(args.per_kind)
    args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'counts': payload['counts'],
                      'probes': len(payload['probes'])}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
