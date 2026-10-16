"""H6 value labels from RenjuNet games (docs/mcts-v8-teacher.md §12.19 H6-1, §12.36). Builder only.

Input: the H2 game set (``data/external/renjunet/games.jsonl.gz``; RenjuNet-derived, never committed).
Each game keeps its H2 tournament split. For every position from ``--from-ply`` on:

- a proof label when one exists (``hybrid.h6_labels.proof_label``: VCF classes, ``--prove-losses`` for
  ``vcf_loss``, ``--vct1`` for proven depth-1 attacks), weight ``--proof-weight``;
- otherwise the game result from the side to move (bresult 1 = black win, 0 = white win), weight
  ``--outcome-weight`` (small: 81% of RenjuNet games end by resignation, time or agreement, §12.6).
  Agreed draws (0.5) are masked; only a full-board draw would be 0.

Masks: H2 ``masked_plies`` (val/test positions already in an earlier split), positions after the first
proof that contradicts the result (``outcome_conflicts_proof``), and cross-split duplicates. Probe /
evaluation positions normally drop their whole game. H6-v2 may exempt only 3-stone opening probes
from whole-game exclusion; any later probe must still drop the whole game so its continuation cannot
leak the answer. The output is written only when ``check_leakage`` and the provenance check pass.

    python scripts/build_h6_labels.py --games data/external/renjunet/games.jsonl.gz --workers 12 \\
        --output data/tactical/h6/renjunet_labels.jsonl.gz --report runs/h6/h6_labels_report.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
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
    first_probe, probe_positions, proof_label, report,
)
from hybrid import h6_fast  # noqa: E402
from hybrid.progress import Progress, run_chunks, stage  # noqa: E402
from renju import BLACK, WHITE  # noqa: E402

SOLVER_NODE_LIMIT = 20_000
PROOF_CHUNK = 4  # games per pool task (results are collected as chunks finish; small for visible progress)
# --prove-losses: per-position cap for the all-moves VCF check (the solver runs ~1-2k nodes/s; P94-type
# parent losses need ~20k nodes)
LOSS_BUDGET = {'node_limit': 20_000, 'call_budget': 5_000, 'node_budget': 100_000}  # ~1 min worst case
MAX_ROW_ONLY_PROBE_PLY = 3  # only standard 3-stone openings may keep their continuations in H6-v2
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

    moves, from_ply, prove_losses, vct1, last_n = task
    solver = ThreatSolver(node_limit=SOLVER_NODE_LIMIT)
    out, stats = {}, {}
    for ply in range(from_ply, len(moves)):
        losses = prove_losses and (not last_n or len(moves) - ply <= last_n)
        label = proof_label(moves[:ply], solver, prove_losses=losses, vct1_budget=VCT1_BUDGET if vct1 else None,
                            loss_budget=LOSS_BUDGET, stats=stats)
        if label is not None:
            out[ply] = label
    return out, stats.get('loss_budget_cuts', 0)


def _proofs_chunk(tasks):
    # the worker's bound functions come back with its results, so a worker on another mode is caught
    return h6_fast.state()['active'], [_proofs_task(t) for t in tasks]


def _git_commit() -> str | None:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(games: list[dict], *, splits, from_ply: int, outcome_weight: float, proof_weight: float,
          prove_losses: bool, vct1: bool, probes, workers: int = 1, probe_game_min_ply: int = 0,
          prove_losses_last_n: int = 0, sample_every: int = 0, fast: str = 'off',
          progress_seconds: float = 60.0) -> tuple[list, dict]:
    """Build H6 rows while protecting probe/evaluation positions from continuation leakage.

    ``probes`` is either a hash set or ``probe_positions`` metadata. Normally every probe drops the
    whole game. ``probe_game_min_ply`` may exempt earlier positions, but H6-v2 intentionally permits
    row-only treatment only for probes with at most ``MAX_ROW_ONLY_PROBE_PLY`` stones (the standard
    3-stone RIF opening controls). If a threshold would retain a 4+ stone probe, fail rather than
    silently train on its continuation. ``probe_game_min_ply=0`` preserves the strict v1 behaviour.

    ``fast`` (``hybrid.h6_fast.MODES``) only changes the solver's speed: rows and summary are the same.
    It is installed in this process and in every pool worker; a worker that runs anything else fails.
    """
    if not isinstance(probes, dict) and probe_game_min_ply > 0:
        raise ValueError('probe_game_min_ply needs probe_positions (hash -> ply); a bare hash set has no ply')
    plies = probes if isinstance(probes, dict) else {h: {'ply': 0} for h in probes}
    game_probes = {h for h, p in plies.items() if p['ply'] >= probe_game_min_ply}
    row_probes = set(plies) - game_probes
    unsafe_row_probes = [plies[h] for h in row_probes if plies[h]['ply'] > MAX_ROW_ONLY_PROBE_PLY]
    if unsafe_row_probes:
        examples = ', '.join(
            f"{p.get('file', '?')}:{p.get('label', '?')}@ply{p['ply']}" for p in unsafe_row_probes[:5]
        )
        raise ValueError(
            f'probe_game_min_ply={probe_game_min_ply} would keep continuations after '
            f'{len(unsafe_row_probes)} non-opening probe(s) ({examples}). Only <= '
            f'{MAX_ROW_ONLY_PROBE_PLY}-stone opening probes may be row-only; use '
            f'--probe-game-min-ply {MAX_ROW_ONLY_PROBE_PLY + 1} (current H6-v2 policy) or a stricter value.'
        )
    kept, dropped = [], {'probe_game': 0, 'other_split': 0, 'probe_game_by_ply': {}, 'not_sampled': 0}
    for game in games:
        if game['split'] not in splits:
            dropped['other_split'] += 1
            continue
        if sample_every and not in_sample(game['id'], sample_every):
            dropped['not_sampled'] += 1
            continue
        hit = first_probe(game['moves'], game_probes) if game_probes else None
        if hit is not None:
            dropped['probe_game'] += 1
            key = str(plies[hit[1]]['ply'])
            dropped['probe_game_by_ply'][key] = dropped['probe_game_by_ply'].get(key, 0) + 1
            continue
        kept.append(game)
    tasks = [(g['moves'], from_ply, prove_losses, vct1, prove_losses_last_n) for g in kept]
    # progress by positions (games differ in length), one line a minute with the ETA
    progress = Progress('proofs', sum(max(0, len(t[0]) - from_ply) for t in tasks), 'positions',
                        every_seconds=progress_seconds)
    progress.log(f"[proofs] {len(tasks):,} games, {progress.total:,} positions, {workers} worker(s)", flush=True)
    positions = lambda chunk: sum(max(0, len(t[0]) - from_ply) for t in chunk)  # noqa: E731
    chunks = [tasks[i:i + PROOF_CHUNK] for i in range(0, len(tasks), PROOF_CHUNK)]
    previous = h6_fast.installed_mode()
    expected = h6_fast.install(fast)['active']
    try:
        if workers == 1:
            parts = []
            for chunk in chunks:
                parts.append(_proofs_chunk(chunk))
                progress.update(positions(chunk), extra=f'{len(parts):,}/{len(chunks):,} chunks')
        else:
            # spawn/forkserver workers do not inherit this process's rebinding: each installs on start
            with ProcessPoolExecutor(max_workers=workers, initializer=h6_fast.install, initargs=(fast,)) as pool:
                parts = run_chunks(pool, _proofs_chunk, chunks, progress, positions)
    finally:
        h6_fast.install(previous)
    wrong = {str(active) for active, _ in parts if active != expected}
    if wrong:
        raise RuntimeError(f'--fast {fast}: a worker ran {sorted(wrong)}, expected {expected}')
    parts = [part for _, part in parts]
    proofs = [proof for part in parts for proof, _ in part]
    loss_budget_cuts = sum(cuts for part in parts for _, cuts in part)
    started = perf_counter()
    solver_class = 'vcf' + ('+vct1' if vct1 else '')
    budget = {'vcf_node_limit': SOLVER_NODE_LIMIT, **({'vct1': VCT1_BUDGET} if vct1 else {})}
    provenance = {'git_commit': _git_commit(), 'engine_config_hash': None, 'model_hash': None}
    rows = []
    by_distance = {}
    for game, proof in zip(kept, proofs):
        for ply, label in proof.items():  # distance to the game end, for choosing --prove-losses-last-n
            d = len(game['moves']) - ply
            bucket = by_distance.setdefault(label['kind'], {})
            key = str(d) if d < 40 else '40+'
            bucket[key] = bucket.get(key, 0) + 1
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
            if row.canonical_hash in row_probes:
                row.value_mask, row.weight = True, 0.0
                row.flags.append('probe_position')
        rows.extend(these)
    stage('rows', started)
    started = perf_counter()
    cross = mask_cross_split(rows)
    check_leakage(rows)
    check_game_provenance(rows)
    stage('cross-split mask + leakage check', started)
    return rows, {'games_in': len(games), 'games_kept': len(kept), 'dropped': dropped,
                  'cross_split_masked': cross, **report(rows),
                  'prove_losses_budget_cuts': loss_budget_cuts, 'prove_losses_budget': LOSS_BUDGET if prove_losses else None,
                  'proof_by_distance_to_end': {k: dict(sorted(v.items(), key=lambda x: (len(x[0]), x[0])))
                                               for k, v in sorted(by_distance.items())}}


def in_sample(game_id, every: int) -> bool:
    """Deterministic 1-in-``every`` game sample (the --prove-losses-last-n pilot), independent of results."""
    return int(hashlib.sha256(f'h6-pilot-v1:{game_id}'.encode()).hexdigest(), 16) % every == 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--games', type=Path, required=True, help='H2 games.jsonl.gz')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--from-ply', type=int, default=5, help='first labelled ply (H3 POLICY_FROM_PLY)')
    parser.add_argument('--outcome-weight', type=float, default=0.1)
    parser.add_argument('--proof-weight', type=float, default=1.0)
    parser.add_argument('--prove-losses', action='store_true', help='add vcf_loss (-1) labels (slower)')
    parser.add_argument('--vct1', action='store_true', help='add proven depth-1 attacks (+1) (slow)')
    parser.add_argument('--prove-losses-last-n', type=int, default=0,
                        help='with --prove-losses: only the last N positions of each game (0 = every position)')
    parser.add_argument('--sample-every', type=int, default=0,
                        help='pilot runs: keep a deterministic 1-in-N game sample (0 = all games)')
    parser.add_argument('--probe-files', nargs='*', default=None)
    parser.add_argument('--probe-game-min-ply', type=int, default=0,
                        help='drop whole games for probes with at least this many stones; only <=3-stone opening '
                             'probes may be exempted (0 = strict v1, current H6-v2 recommendation = 4)')
    parser.add_argument('--limit', type=int, help='first N games (smoke tests)')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--fast', choices=h6_fast.MODES, default='off',
                        help='exact solver speed-up (same rows, only faster): off = frozen Python, fast2 = '
                             'rt30.fast, fast3 = fast2 + native window scan (python -m rt30.native.build; '
                             'fails when not built)')
    parser.add_argument('--output', type=Path, required=True, help='rows, .jsonl.gz (keep it out of git)')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(ROOT) and not str(args.output.resolve().relative_to(ROOT)).startswith(
            ('data', 'runs')):
        parser.error('RenjuNet-derived rows go under data/ or runs/ (both ignored by git), never into the repo')
    try:  # fail before the long run when the mode cannot be installed (e.g. fast3 not built)
        h6_fast.install(args.fast)
    except RuntimeError as error:
        parser.error(str(error))
    h6_fast.install('off')
    probe_files = default_probe_files(ROOT) if args.probe_files is None else args.probe_files
    games = read_games(args.games)[:args.limit]
    rows, summary = build(games, splits=set(args.splits), from_ply=args.from_ply,
                          outcome_weight=args.outcome_weight, proof_weight=args.proof_weight,
                          prove_losses=args.prove_losses, vct1=args.vct1,
                          probes=probe_positions(probe_files, ROOT), probe_game_min_ply=args.probe_game_min_ply,
                          prove_losses_last_n=args.prove_losses_last_n, sample_every=args.sample_every,
                          workers=args.workers, fast=args.fast)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with gzip.open(args.output, 'wt', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), separators=(',', ':')) + '\n')
    stage(f'write {len(rows):,} rows', started)
    payload = {'format': 'h6-label-report-v1', 'row_format': ROW_FORMAT, 'git_commit': _git_commit(),
               'games': str(args.games), 'restricted': args.limit is not None or bool(args.sample_every),
               'settings': {k: getattr(args, k) for k in ('splits', 'from_ply', 'outcome_weight', 'proof_weight',
                                                          'prove_losses', 'vct1', 'probe_game_min_ply',
                                                          'prove_losses_last_n', 'sample_every')},
               'acceleration': h6_fast.provenance(args.fast),
               'probe_files': [str(Path(p).resolve().relative_to(ROOT)).replace('\\', '/')
                               if Path(p).resolve().is_relative_to(ROOT) else str(p) for p in probe_files], 'summary': summary}
    print(json.dumps(payload['summary'], indent=1), flush=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
