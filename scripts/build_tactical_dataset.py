"""Build the proof-labelled tactical dataset for the teacher arm (docs/v7-nn-integration-review.md).

Positions come from game records; labels come only from rules and VCF proofs
(``analysis.tactical_labels``), never from the agent that played the move, so V7's
stage 4/5 choices and its MCTS moves never become targets.

Sources (any combination):
- ``--benchmark``: MCTS-v7 benchmark result files (``docs/mcts-v7-results/*.json``);
- ``--web-games``: the web-play games fixture (``web-play-games-v1``);
- ``--self-play-run``: a Stage 6+ run directory (``self_play/genNNN.json`` records),
  optionally limited with ``--generations FROM TO``.

Positions are deduplicated by board + side to move under D4, and positions that
appear in a probe set (``--exclude-probes``, default every ``tests/fixtures/*probes*.json``)
are dropped so the probes stay held out.

    python scripts/build_tactical_dataset.py --benchmark docs/mcts-v7-results/*.json \
        --self-play-run runs/stage8_g3_b --generations 200 400 --output runs/teacher/tactical.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from analysis.tactical_labels import label_position  # noqa: E402
from analysis.threats import ThreatSolver  # noqa: E402
from renju import Game  # noqa: E402

DATASET_FORMAT = 'tactical-dataset-v1'
BOARD = 15


def _transform(move, symmetry: int):
    row, col = move
    if symmetry >= 4:
        col = BOARD - 1 - col
    for _ in range(symmetry % 4):
        row, col = BOARD - 1 - col, row
    return row, col


def canonical_key(moves) -> tuple:
    """Board + side to move, minimized over D4 (move order does not matter)."""
    black = [tuple(m) for m in moves[0::2]]
    white = [tuple(m) for m in moves[1::2]]
    to_play = len(moves) % 2
    return min((to_play, tuple(sorted(_transform(m, s) for m in black)),
                tuple(sorted(_transform(m, s) for m in white))) for s in range(8))


def _action_to_move(action: int):
    return divmod(action, BOARD)


def games_from_benchmark(path: Path):
    data = json.loads(path.read_bytes())
    for index, game in enumerate(data['games']):
        yield {'source': f'{path.name}#{index}', 'moves': [tuple(m) for m in game['moves']]}


def games_from_web(path: Path):
    data = json.loads(path.read_bytes())
    for game in data['games']:
        yield {'source': f"{path.name}#{game['name']}",
               'moves': [tuple(m) for m in game['moves']]}


def games_from_self_play(run_dir: Path, generations):
    for path in sorted((run_dir / 'self_play').glob('gen*.json')):
        generation = int(path.stem[3:])
        if generations and not generations[0] <= generation <= generations[1]:
            continue
        data = json.loads(path.read_bytes())
        for index, game in enumerate(data['games']):
            yield {'source': f'{run_dir.name}/gen{generation:03d}#{index}',
                   'moves': [_action_to_move(a) for a in game['record']['moves']]}


def probe_keys(paths) -> set:
    keys = set()
    for path in paths:
        data = json.loads(Path(path).read_bytes())
        for probe in data.get('probes', []):
            keys.add(canonical_key(probe['moves']))
    return keys


def build(games, *, exclude: set, node_limit: int, prove_losses: bool, log=print) -> dict:
    solver = ThreatSolver(node_limit=node_limit)
    seen = set()
    positions = []
    stats = Counter()
    started = perf_counter()
    for game_index, record in enumerate(games):
        game = Game()
        for ply, move in enumerate(record['moves']):
            key = canonical_key(record['moves'][:ply])
            stats['positions'] += 1
            if key in seen:
                stats['duplicates'] += 1
            elif key in exclude:
                stats['excluded_probe'] += 1
                seen.add(key)
            else:
                seen.add(key)
                label = label_position(game, solver, prove_losses=prove_losses)
                if label is not None:
                    positions.append({'moves': [list(m) for m in record['moves'][:ply]],
                                      'kind': label['kind'],
                                      'policy': [list(m) for m in label['policy']],
                                      'value': label['value'], 'source': record['source']})
                    stats[label['kind']] += 1
            game.play(*move)
        if (game_index + 1) % 100 == 0:
            log(f'{game_index + 1} games, {len(positions)} labels '
                f'({perf_counter() - started:.0f}s)')
    return {'format': DATASET_FORMAT, 'coordinates': '0-based [row, col]',
            'labeler': {'module': 'analysis.tactical_labels', 'node_limit': node_limit,
                        'prove_losses': prove_losses, 'vcf_calls': solver.vcf_calls,
                        'vcf_exhausted': solver.vcf_exhausted},
            'stats': dict(stats), 'positions': positions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--benchmark', nargs='*', default=[])
    parser.add_argument('--web-games', nargs='*', default=[])
    parser.add_argument('--self-play-run', nargs='*', default=[], type=Path)
    parser.add_argument('--generations', nargs=2, type=int, metavar=('FROM', 'TO'))
    parser.add_argument('--exclude-probes', nargs='*',
                        default=sorted(glob.glob(str(ROOT / 'tests' / 'fixtures' / '*probes*.json'))))
    parser.add_argument('--node-limit', type=int, default=100_000)
    parser.add_argument('--prove-losses', action='store_true',
                        help='also label vcf_loss positions (slow: a full VCF decision each)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    sources = []
    for pattern in args.benchmark:
        sources += [('benchmark', Path(p)) for p in sorted(glob.glob(pattern))]
    sources += [('web', Path(p)) for p in args.web_games]
    sources += [('self_play', p) for p in args.self_play_run]
    if not sources:
        parser.error('give at least one --benchmark, --web-games or --self-play-run')

    def games():
        for kind, path in sources:
            if kind == 'benchmark':
                yield from games_from_benchmark(path)
            elif kind == 'web':
                yield from games_from_web(path)
            else:
                yield from games_from_self_play(path, args.generations)

    exclude = probe_keys(args.exclude_probes)
    result = build(games(), exclude=exclude, node_limit=args.node_limit,
                   prove_losses=args.prove_losses)
    result['sources'] = [{'kind': kind, 'path': str(path),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()
                          if path.is_file() else None} for kind, path in sources]
    result['generations'] = args.generations
    result['excluded_probe_files'] = args.exclude_probes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + '\n', encoding='utf-8')
    print(f"wrote {args.output}: {result['stats']}")


if __name__ == '__main__':
    main()
