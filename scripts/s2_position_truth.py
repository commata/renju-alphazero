"""S2: classify every legal move of a probe position up to VCT depth 2 (docs/mcts-v8-teacher.md §12.21).

For each legal move of the side to move (``ThreatSolver.ordered_moves`` order: nearest to
the stones and the last move first), the position after it is checked at depth 0, 1, 2
with a node/call budget per move (``analysis.mcts_v8._BudgetedSolver``, no time cut):

    lost_depth 0-2   the move is a proven loss at that depth (PROVEN_LOSS_VCT<d>)
    SAFE             not lost within depth 2 (NO_VCT2_FOUND_WITHIN_HORIZON; not "has a defence")
    UNKNOWN          the budget ran out

Every finished move is appended to ``--jsonl`` at once, so a rerun resumes where it stopped.
Moves are independent, so ``--workers N`` runs N of them at a time (the parent process alone
writes the file). Changing ``--workers`` between runs is safe: only moves without a line are
computed. Each line records its budget; a rerun with a different budget is refused, so one
file never mixes budgets (lines written before the budget was recorded count as the default
budget, the one that run used). A torn last line from an interrupted write is skipped and
that move is computed again.
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
from scripts.s1_loss_analysis import lost_depth_info  # noqa: E402

FORMAT = 's2-position-truth-v3'  # v2: budget per line, ETA, torn-line tolerant resume; v3: UNKNOWN cause
DEFAULT_BUDGET = {'node_limit': 20_000, 'call_limit': 100_000, 'node_budget': 10_000_000}


def classify_move(task) -> dict:
    history, move, budget = task
    started = perf_counter()
    depth, status, info = lost_depth_info([*history, move], len(history), budget)
    return {'move': [move[0] + 1, move[1] + 1], 'lost_depth': depth,
            'status': 'PROVEN_LOSS' if depth is not None else status,
            'seconds': round(perf_counter() - started, 2), 'budget': budget, 'search': info}


def load_done(path: Path | None, budget: dict) -> tuple[dict, dict]:
    """Finished moves from ``path``; raises ValueError if a line used another budget."""
    done, info = {}, {'legacy_rows': 0, 'torn_lines': 0}
    if path is None or not path.exists():
        return done, info
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            info['torn_lines'] += 1  # an interrupted write: the move is simply computed again
            continue
        row_budget = row.get('budget')
        if row_budget is None:
            info['legacy_rows'] += 1
            row_budget = DEFAULT_BUDGET
        if row_budget != budget:
            raise ValueError(f'{path} has a line for {row["move"]} with budget {row_budget}, '
                             f'this run uses {budget}; use another --jsonl file')
        done[tuple(row['move'])] = row
    return done, info


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
    parser.add_argument('--node-limit', type=int, default=DEFAULT_BUDGET['node_limit'], help='per VCF call')
    parser.add_argument('--call-limit', type=int, default=DEFAULT_BUDGET['call_limit'], help='VCF calls per move')
    parser.add_argument('--node-budget', type=int, default=DEFAULT_BUDGET['node_budget'], help='VCF nodes per move')
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
    try:
        done, resume = load_done(args.jsonl, budget)
    except ValueError as exc:
        parser.error(str(exc))
    pending = [m for m in legal if (m[0] + 1, m[1] + 1) not in done]
    print(f'{args.name}: legal={len(legal)} done={len(legal) - len(pending)} pending={len(pending)} '
          f'workers={args.workers} resume={resume}', flush=True)
    started, finished_now = perf_counter(), 0
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)

    def keep(row):
        nonlocal finished_now
        done[tuple(row['move'])] = row
        finished_now += 1
        elapsed = perf_counter() - started
        eta = elapsed / finished_now * (len(pending) - finished_now)  # wall-clock rate incl. parallelism
        print(f"{len(legal) - len(pending) + finished_now}/{len(legal)} {row['move']} {row['status']} "
              f"depth={row['lost_depth']} {row['seconds']:.1f}s | elapsed {elapsed / 60:.1f}m eta {eta / 60:.1f}m",
              flush=True)
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
               'probes_sha256': file_sha256(args.probes), 'budget': budget, 'resume': resume,
               'restricted': bool(args.moves or args.limit), 'summary': summary, 'moves': rows}
    print(json.dumps(summary, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
