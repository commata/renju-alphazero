"""Build the proof-labelled tactical dataset for the teacher arm (docs/v7-nn-integration-review.md).

Positions come from game records; labels come only from rules and VCF proofs
(``analysis.tactical_labels``), never from the agent that played the move, so V7's
stage 4/5 choices and its MCTS moves never become targets.

Sources (any combination):
- ``--benchmark``: MCTS-v7 benchmark result files (``docs/mcts-v7-results/*.json``);
- ``--web-games``: the web-play games fixture (``web-play-games-v1``);
- ``--self-play-run``: a Stage 6+ run directory (``self_play/genNNN.json`` records),
  optionally limited with ``--generations FROM TO``.

Positions are deduplicated by board + side to move under D4. Held-out probes stay
held out at the **game** level: a game that reaches any probe position
(``--exclude-probes``, default every ``tests/fixtures/*probes*.json``) is dropped
entirely, because its neighbouring plies are near-copies of the probe.

``--vct-depth 1`` (slow; run it on the desktop) adds VCT labels, using the game
continuation to pick candidates: when the side to move at ply t has no VCF-level label
but has a VCF-level win at t+2, its game move h is tested. If every reply to h loses to
VCF, ply t gets ``vct_attack`` (h, +1) and ply t+1 ``vcf_loss`` (-1). The defender's
position at t-1 is then classified at VCT depth 1 when it has at most
``--vct-max-candidates`` VCF-safe moves (``must_defend_vct`` or ``vct_loss``).

``stats``/``balance`` record counts per kind, side to move and game phase.

    python scripts/build_tactical_dataset.py --benchmark "docs/mcts-v7-results/*.json" \
        --self-play-run runs/stage8_g3_b --generations 160 400 --workers 6 \
        --output runs/teacher/tactical_t1.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import glob
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from analysis.tactical_labels import defense_label, label_position, proves_threat  # noqa: E402
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


def unique_positions(games, exclude: set, stats: Counter) -> tuple[list[dict], list[dict]]:
    """Kept games (probe games dropped) and their D4-unique positions, in game order."""
    seen = set()
    tasks = []
    kept = []
    for record in games:
        keys = [canonical_key(record['moves'][:ply]) for ply in range(len(record['moves']))]
        if exclude and any(key in exclude for key in keys):
            stats['excluded_probe_games'] += 1
            continue
        kept.append({**record, 'keys': keys})
        for ply, key in enumerate(keys):
            stats['positions'] += 1
            if key in seen:
                stats['duplicates'] += 1
                continue
            seen.add(key)
            tasks.append({'moves': [list(m) for m in record['moves'][:ply]],
                          'source': record['source'], 'key': key})
    return tasks, kept


_WORKER_SOLVER = None


def _init_worker(node_limit: int) -> None:
    global _WORKER_SOLVER
    _WORKER_SOLVER = ThreatSolver(node_limit=node_limit)


def _label_task(args):
    task, prove_losses = args
    game = Game()
    for move in task['moves']:
        game.play(*move)
    solver = _WORKER_SOLVER
    before = solver.vcf_exhausted
    label = label_position(game, solver, prove_losses=prove_losses)
    # Keep the per-worker caches small: positions rarely repeat across tasks.
    solver._vcf_cache.clear()
    solver._after_cache.clear()
    return label, solver.vcf_exhausted - before


VCF_WINS = ('vcf', 'unstoppable_four', 'immediate_win')


def vct_tasks(kept: list[dict], labels: dict) -> list[dict]:
    """Plies t whose side to move has no VCF-level label but a VCF-level win at t+2."""
    tasks, seen = [], set()
    for record in kept:
        keys, moves = record['keys'], record['moves']
        for t in range(len(moves) - 2):
            if keys[t] in labels or labels.get(keys[t + 2], {}).get('kind') not in VCF_WINS:
                continue
            if keys[t] in seen:
                continue
            seen.add(keys[t])
            defend = t >= 1 and keys[t - 1] not in labels
            tasks.append({'moves': [list(m) for m in moves[:t + 1]], 'defend': defend,
                          'source': record['source']})
    return tasks


def _vct_task(args):
    task, vct_depth, max_candidates = args
    solver = _WORKER_SOLVER
    moves = [tuple(m) for m in task['moves']]
    game = Game()
    for move in moves[:-1]:
        game.play(*move)
    out = []
    if proves_threat(game, moves[-1], solver):
        out.append((moves[:-1], {'kind': 'vct_attack', 'policy': [moves[-1]], 'value': 1}))
        out.append((moves, {'kind': 'vcf_loss', 'policy': [], 'value': -1}))
        if task['defend']:
            game.undo()
            label = defense_label(game, solver, max_candidates=max_candidates,
                                  vct_depth=vct_depth)
            if label is not None:
                out.append((moves[:-2], label))
    solver._vcf_cache.clear()
    solver._after_cache.clear()
    return out


def _run(pool_size, node_limit, fn, jobs):
    if pool_size > 1:
        pool = Pool(pool_size, initializer=_init_worker, initargs=(node_limit,))
        try:
            yield from pool.imap(fn, jobs, chunksize=4)
        finally:
            pool.close()
            pool.join()
    else:
        _init_worker(node_limit)
        yield from map(fn, jobs)


def _phase(ply: int) -> str:
    return 'opening' if ply < 12 else 'middle' if ply < 30 else 'late'


def build(games, *, exclude: set, node_limit: int, prove_losses: bool, workers: int = 1,
          vct_depth: int = 0, vct_max_candidates: int = 12, log=print) -> dict:
    stats = Counter()
    tasks, kept = unique_positions(games, exclude, stats)
    log(f"{len(tasks)} unique positions to label ({stats['duplicates']} duplicates, "
        f"{stats['excluded_probe_games']} probe games dropped)")
    started = perf_counter()
    labels = {}
    positions = []
    jobs = ((task, prove_losses) for task in tasks)
    for index, (task, (label, exhausted)) in enumerate(
            zip(tasks, _run(workers, node_limit, _label_task, jobs))):
        stats['vcf_budget_cut'] += exhausted
        if label is not None:
            labels[task['key']] = label
            positions.append({'moves': task['moves'], 'kind': label['kind'],
                              'policy': [list(m) for m in label['policy']],
                              'value': label['value'], 'source': task['source']})
        if (index + 1) % 2000 == 0:
            log(f'{index + 1}/{len(tasks)} positions, {len(positions)} labels '
                f'({perf_counter() - started:.0f}s)')
    if vct_depth:
        candidates = vct_tasks(kept, labels)
        log(f'VCT phase: {len(candidates)} threat candidates')
        stats['vct_candidates'] = len(candidates)
        jobs = ((task, vct_depth, vct_max_candidates) for task in candidates)
        for index, found in enumerate(_run(workers, node_limit, _vct_task, jobs)):
            for moves, label in found:
                key = canonical_key(moves)
                if key in labels:
                    continue
                labels[key] = label
                positions.append({'moves': [list(m) for m in moves], 'kind': label['kind'],
                                  'policy': [list(m) for m in label['policy']],
                                  'value': label['value'], 'source': 'vct'})
            if (index + 1) % 200 == 0:
                log(f'VCT {index + 1}/{len(candidates)} ({perf_counter() - started:.0f}s)')
    stats.update(Counter(p['kind'] for p in positions))
    balance = Counter(f"{p['kind']}|{'BLACK' if len(p['moves']) % 2 == 0 else 'WHITE'}|"
                      f"{_phase(len(p['moves']))}" for p in positions)
    return {'format': DATASET_FORMAT, 'coordinates': '0-based [row, col]',
            'labeler': {'module': 'analysis.tactical_labels', 'node_limit': node_limit,
                        'prove_losses': prove_losses, 'vct_depth': vct_depth,
                        'vct_max_candidates': vct_max_candidates,
                        'note': 'a VCF probe cut by the node budget gives no label'},
            'stats': dict(stats), 'balance': dict(sorted(balance.items())),
            'positions': positions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--benchmark', nargs='*', default=[])
    parser.add_argument('--web-games', nargs='*', default=[])
    parser.add_argument('--self-play-run', nargs='*', default=[], type=Path)
    parser.add_argument('--generations', nargs=2, type=int, metavar=('FROM', 'TO'))
    parser.add_argument('--exclude-probes', nargs='*',
                        default=sorted(glob.glob(str(ROOT / 'tests' / 'fixtures' / '*probes*.json'))))
    parser.add_argument('--node-limit', type=int, default=20_000,
                        help='VCF node budget; a cut probe gives no label (sound, not complete)')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--vct-depth', type=int, default=0,
                        help='1 adds vct_attack / must_defend_vct / vct_loss labels (slow)')
    parser.add_argument('--vct-max-candidates', type=int, default=12,
                        help='classify a defence only with at most this many VCF-safe moves')
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
                   prove_losses=args.prove_losses, workers=args.workers,
                   vct_depth=args.vct_depth, vct_max_candidates=args.vct_max_candidates)
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
