"""H6 value labels from RenjuNet games (docs/mcts-v8-teacher.md §12.19 H6-1, §12.36). Builder only.

Input: the H2 game set (``data/external/renjunet/games.jsonl.gz``; RenjuNet-derived, never committed).
Each game keeps its H2 tournament split. For every position from ``--from-ply`` on:

- a proof label when one exists (``hybrid.h6_labels.proof_label``: VCF classes, ``--prove-losses`` for
  ``vcf_loss``, ``--vct1`` for proven depth-1 attacks), weight ``--proof-weight``;
- otherwise the game result from the side to move (bresult 1 = black win, 0 = white win), weight
  ``--outcome-weight`` (small: 81% of RenjuNet games end by resignation, time or agreement, §12.6).
  Agreed draws (0.5) are masked; only a full-board draw would be 0.

Masks: H2 ``masked_plies`` (val/test positions already in an earlier split), positions after the first
proof that contradicts the result (``outcome_conflicts_proof``), and cross-split duplicates. Games that
reach a probe / evaluation position are dropped whole (``--probe-files``, default: test fixtures, web
probes and the E1 manifests). The output is written only when ``check_leakage`` and the provenance
check pass.

    python scripts/build_h6_labels.py --games data/external/renjunet/games.jsonl.gz --workers 12 \\
        --output data/tactical/h6/renjunet_labels.jsonl.gz --report runs/h6/h6_labels_report.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hybrid.h6_labels import (  # noqa: E402
    ROW_FORMAT, check_game_provenance, check_leakage, default_probe_files, game_rows, mask_cross_split,
    probe_hashes, proof_label, report, touches_probe,
)
from hybrid.progress import Progress, stage  # noqa: E402
from renju import BLACK, WHITE  # noqa: E402

SOLVER_NODE_LIMIT = 20_000
VCT1_BUDGET = {'node_limit': 20_000, 'call_limit': 10_000, 'node_budget': 200_000}  # V8-B's budget


def read_games(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def renjunet_result(game: dict) -> dict:
    winner = {'1': BLACK, '0': WHITE}.get(str(game['bresult']))
    return {'id': f"renjunet:{game['id']}", 'moves': game['moves'], 'winner': winner, 'real_draw': False}


def _proofs_task(task):
    from analysis.threats import ThreatSolver

    moves, from_ply, prove_losses, vct1 = task
    solver = ThreatSolver(node_limit=SOLVER_NODE_LIMIT)
    out = {}
    for ply in range(from_ply, len(moves)):
        label = proof_label(moves[:ply], solver, prove_losses=prove_losses, vct1_budget=VCT1_BUDGET if vct1 else None)
        if label is not None:
            out[ply] = label
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(games: list[dict], *, splits, from_ply: int, outcome_weight: float, proof_weight: float,
          prove_losses: bool, vct1: bool, probes: set[str], workers: int = 1,
          progress_seconds: float = 60.0) -> tuple[list, dict]:
    kept, dropped = [], {'probe_game': 0, 'other_split': 0}
    for game in games:
        if game['split'] not in splits:
            dropped['other_split'] += 1
            continue
        if probes and touches_probe(game['moves'], probes):
            dropped['probe_game'] += 1
            continue
        kept.append(game)
    tasks = [(g['moves'], from_ply, prove_losses, vct1) for g in kept]
    # progress by positions (games differ in length), one line a minute with the ETA
    progress = Progress('proofs', sum(max(0, len(t[0]) - from_ply) for t in tasks), 'positions',
                        every_seconds=progress_seconds)
    progress.log(f"[proofs] {len(tasks):,} games, {progress.total:,} positions, {workers} worker(s)", flush=True)
    proofs = []
    if workers == 1:
        results = map(_proofs_task, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_proofs_task, tasks, chunksize=16)
    try:
        for task, proof in zip(tasks, results):
            proofs.append(proof)
            progress.update(max(0, len(task[0]) - from_ply), extra=f'{len(proofs):,}/{len(tasks):,} games')
    finally:
        if workers != 1:
            pool.shutdown()
    started = perf_counter()
    solver_class = 'vcf' + ('+vct1' if vct1 else '')
    budget = {'vcf_node_limit': SOLVER_NODE_LIMIT, **({'vct1': VCT1_BUDGET} if vct1 else {})}
    provenance = {'git_commit': _git_commit(), 'engine_config_hash': None, 'model_hash': None}
    rows = []
    for game, proof in zip(kept, proofs):
        game_result = renjunet_result(game)
        these = game_rows(game_result, source='renjunet', split=game['split'], outcome_weight=outcome_weight,
                          proof_weight=proof_weight, proofs=proof, solver_class=solver_class,
                          solver_budget=budget, provenance=provenance, from_ply=from_ply)
        h2_masked = set(game.get('masked_plies', []))
        for row in these:
            if game.get('end') == 'unknown' and row.label_kind == 'outcome':
                row.flags.append('end_unknown')  # resignation / time / agreement: kept, weight already small
            if row.ply in h2_masked and not row.value_mask:
                row.value_mask, row.weight = True, 0.0
                row.flags.append('h2_masked')
        rows.extend(these)
    stage('rows', started)
    started = perf_counter()
    cross = mask_cross_split(rows)
    check_leakage(rows)
    check_game_provenance(rows)
    stage('cross-split mask + leakage check', started)
    return rows, {'games_in': len(games), 'games_kept': len(kept), 'dropped': dropped,
                  'cross_split_masked': cross, **report(rows)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--games', type=Path, required=True, help='H2 games.jsonl.gz')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--from-ply', type=int, default=5, help='first labelled ply (H3 POLICY_FROM_PLY)')
    parser.add_argument('--outcome-weight', type=float, default=0.1)
    parser.add_argument('--proof-weight', type=float, default=1.0)
    parser.add_argument('--prove-losses', action='store_true', help='add vcf_loss (-1) labels (slower)')
    parser.add_argument('--vct1', action='store_true', help='add proven depth-1 attacks (+1) (slow)')
    parser.add_argument('--probe-files', nargs='*', default=None)
    parser.add_argument('--limit', type=int, help='first N games (smoke tests)')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True, help='rows, .jsonl.gz (keep it out of git)')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(ROOT) and not str(args.output.resolve().relative_to(ROOT)).startswith(
            ('data', 'runs')):
        parser.error('RenjuNet-derived rows go under data/ or runs/ (both ignored by git), never into the repo')
    probe_files = default_probe_files(ROOT) if args.probe_files is None else args.probe_files
    games = read_games(args.games)[:args.limit]
    rows, summary = build(games, splits=set(args.splits), from_ply=args.from_ply,
                          outcome_weight=args.outcome_weight, proof_weight=args.proof_weight,
                          prove_losses=args.prove_losses, vct1=args.vct1, probes=probe_hashes(probe_files),
                          workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with gzip.open(args.output, 'wt', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), separators=(',', ':')) + '\n')
    stage(f'write {len(rows):,} rows', started)
    payload = {'format': 'h6-label-report-v1', 'row_format': ROW_FORMAT, 'git_commit': _git_commit(),
               'games': str(args.games), 'restricted': args.limit is not None,
               'settings': {k: getattr(args, k) for k in ('splits', 'from_ply', 'outcome_weight', 'proof_weight',
                                                          'prove_losses', 'vct1')},
               'probe_files': probe_files, 'summary': summary}
    print(json.dumps(payload['summary'], indent=1), flush=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
