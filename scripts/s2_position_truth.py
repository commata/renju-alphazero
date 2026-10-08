"""S2: classify every legal move of a probe position up to VCT depth 2 (docs/mcts-v8-teacher.md §12.21).

For each legal move of the side to move (``ThreatSolver.ordered_moves`` order: nearest to
the stones and the last move first), the position after it is checked at depth 0, 1, 2
with a node/call budget per move (``analysis.mcts_v8._BudgetedSolver``, no time cut):

    lost_depth 0-2   the move is a proven loss at that depth (PROVEN_LOSS_VCT<d>)
    SAFE             not lost within depth 2 (NO_VCT2_FOUND_WITHIN_HORIZON; not "has a defence")
    UNKNOWN          the budget ran out

Every finished move is appended to ``--jsonl`` at once, so a rerun resumes where it stopped.
The verdict is PROVEN_LOSS when every legal move is lost within depth 2, otherwise the SAFE
moves form the saving-defence set (depth-2 class). With ``--root-candidates`` (outputs of
``run_s1_probes.py``) it also reports how many saving moves each arm had as root children.

    python scripts/s2_position_truth.py --name P92 --workers 4 \\
        --jsonl runs/s2/p92_truth.jsonl --output runs/s2/p92_truth.json \\
        --root-candidates docs/mcts-v8-results/s1_probes_full.json \\
            docs/mcts-v8-results/s1_probes_puct_heur.json docs/mcts-v8-results/s1_probes_puct_policy.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from analysis.threats import ordered_moves  # noqa: E402
from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256  # noqa: E402
from scripts.run_s1_probes import DEFAULT_PROBES, replay  # noqa: E402
from scripts.s1_loss_analysis import _lost_depth  # noqa: E402

FORMAT = 's2-position-truth-v1'


def classify_move(task) -> dict:
    history, move, budget = task
    started = perf_counter()
    depth, status = _lost_depth([*history, move], len(history), budget)
    return {'move': [move[0] + 1, move[1] + 1], 'lost_depth': depth,
            'status': 'PROVEN_LOSS' if depth is not None else status,
            'seconds': round(perf_counter() - started, 2)}


def load_done(path: Path | None) -> dict:
    done = {}
    if path is not None and path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                done[tuple(row['move'])] = row
    return done


def root_candidates(paths, name: str) -> dict[str, set]:
    """Union of the root children each arm had at probe ``name`` (1-indexed)."""
    out = {}
    for path in paths:
        for run in json.loads(Path(path).read_text(encoding='utf-8'))['runs']:
            if run['probe'] == name:
                out.setdefault(run['arm'], set()).update(tuple(m) for m, _, _ in run['root_children'])
    return out


def summarize(rows: list[dict], legal: int, candidates: dict[str, set], restricted: bool = False) -> dict:
    counts = {}
    for row in rows:
        key = row['status'] if row['lost_depth'] is None else f"PROVEN_LOSS_VCT{row['lost_depth']}"
        counts[key] = counts.get(key, 0) + 1
    saving = sorted(tuple(r['move']) for r in rows if r['status'] == 'SAFE')
    unknown = sorted(tuple(r['move']) for r in rows if r['status'] == 'UNKNOWN')
    complete = len(rows) == legal
    if complete and not saving and not unknown:
        verdict = 'PROVEN_LOSS'
    elif saving:
        verdict = 'SAVING_MOVES_FOUND'
    else:
        verdict = 'UNRESOLVED'
    if restricted:
        verdict = 'RESTRICTED_' + verdict  # only some moves were checked: not a verdict on the position
    recall = {arm: {'saving_in_root': sorted(set(saving) & roots), 'root_children': len(roots),
                    'recall': round(len(set(saving) & roots) / len(saving), 4) if saving else None}
              for arm, roots in sorted(candidates.items())}
    return {'legal_moves': legal, 'classified': len(rows), 'complete': complete, 'counts': counts,
            'verdict': verdict, 'saving_moves': [list(m) for m in saving],
            'unknown_moves': [list(m) for m in unknown], 'root_candidate_recall': recall}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--probes', type=Path, default=DEFAULT_PROBES)
    parser.add_argument('--name', default='P92', help='probe name in the probe file')
    parser.add_argument('--moves', nargs='*', default=None, help="only these moves, 1-indexed 'r,c'")
    parser.add_argument('--limit', type=int, help='smoke tests: first N moves only')
    parser.add_argument('--node-limit', type=int, default=20_000, help='per VCF call')
    parser.add_argument('--call-limit', type=int, default=100_000, help='VCF calls per move')
    parser.add_argument('--node-budget', type=int, default=10_000_000, help='VCF nodes per move')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--jsonl', type=Path, help='one line per finished move (resume)')
    parser.add_argument('--root-candidates', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    probe_file = json.loads(args.probes.read_text(encoding='utf-8'))
    history = [tuple(m) for m in probe_file['moves'][:probe_file['probes'][args.name]['plies_played']]]
    game = replay(history, len(history))
    legal = ordered_moves(game)
    if args.moves:
        wanted = {tuple(int(x) - 1 for x in m.split(',')) for m in args.moves}
        legal = [m for m in legal if m in wanted]
    if args.limit:
        legal = legal[:args.limit]
    budget = {'node_limit': args.node_limit, 'call_limit': args.call_limit, 'node_budget': args.node_budget}
    done = load_done(args.jsonl)
    pending = [m for m in legal if (m[0] + 1, m[1] + 1) not in done]
    print(f'{args.name}: legal={len(legal)} done={len(legal) - len(pending)} pending={len(pending)} '
          f'workers={args.workers}', flush=True)
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)

    def keep(row):
        done[tuple(row['move'])] = row
        print(f"{len(done)}/{len(legal)} {row['move']} {row['status']} depth={row['lost_depth']} "
              f"{row['seconds']:.1f}s", flush=True)
        if args.jsonl is not None:
            with args.jsonl.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')

    tasks = [(history, m, budget) for m in pending]
    if args.workers == 1:
        for task in tasks:
            keep(classify_move(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(classify_move, t) for t in tasks]):
                keep(future.result())
    rows = [done[(m[0] + 1, m[1] + 1)] for m in legal]
    summary = summarize(rows, len(legal), root_candidates(args.root_candidates, args.name),
                        restricted=bool(args.moves or args.limit))
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'probe': args.name,
               'probes_sha256': file_sha256(args.probes), 'budget': budget,
               'restricted': bool(args.moves or args.limit), 'summary': summary, 'moves': rows}
    print(json.dumps(summary, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
