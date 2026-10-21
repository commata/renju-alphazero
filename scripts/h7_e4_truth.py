"""H7-E4 truth: does the side to move have a forced win within VCT depth 2 (docs/h6-value.md, H7-E4)?

For each web corpus position (``h7_e4_web_corpus.py``: a human just moved, the AI is to move) the human's
last move is searched at depth 0, 1 and 2 with ``analysis.threats.ThreatSolver`` semantics, full width:
every legal quiet move of the attacker, every legal reply of the defender (the same class as
``scripts/s2_position_truth.py``), one node / call budget for the whole position (``_BudgetedSolver``,
counts only, no time cut, at most 10M VCF nodes). Depths share the solver's caches.
The attacker's quiet moves are the full legal set, threat moves (a four or an open / split three,
``analysis.selective_vct.threat_moves``) first (``TruthSolver``): the same set, so SAFE and UNSAFE mean
what they mean for ``ThreatSolver``; the order only lets a win be found before the budget runs out.

    class        PROVEN_WIN  some depth is UNSAFE for the human (a witness first move of the AI is kept)
                 NOT_FOUND   depth 2 finished SAFE: no win of the searched class (not "no win")
                 UNKNOWN     neither: a budget or a single VCF call cut. Never used as a negative.
    grade        WIN_5 (immediate five), WIN_VCF (depth 0), WIN_VCT1, WIN_VCT2_ONLY (depth 2 UNSAFE with depth 1
                 finished SAFE: only these are E4 targets); WIN_VCT1_OR_LESS / WIN_VCT2_OR_LESS when a
                 shallower depth was UNKNOWN (the smallest depth is not known)

    python scripts/h7_e4_truth.py --corpus docs/h7/h7_e4_web_corpus.json --workers 8 --fast fast4 \\
        --jsonl runs/h7_e4/truth.jsonl --output docs/h7/h7_e4_truth.json

Positions with the same board and side to move are searched once. Every finished board is appended to
``--jsonl`` at once (resume; a torn last line is computed again; a line with another budget is refused).
Progress on stdout once a minute (``hybrid.progress``).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import h7_e4_web_corpus as web  # noqa: E402

FORMAT = 'h7-e4-truth-v1'
DEFAULT_BUDGET = {'node_limit': 100_000, 'call_limit': 1_000_000, 'node_budget': 10_000_000}
CLASSES = ('PROVEN_WIN', 'NOT_FOUND', 'UNKNOWN')
GRADES = ('WIN_5', 'WIN_VCF', 'WIN_VCT1', 'WIN_VCT2_ONLY', 'WIN_VCT1_OR_LESS', 'WIN_VCT2_OR_LESS')
TARGET = 'WIN_VCT2_ONLY'


def _first_move(witness: tuple):
    """The AI's first move of a witness: ('five', m), ('vcf', line), ('threat', m), ('block', m)."""
    kind, data = witness
    move = data[0] if kind == 'vcf' else data
    return [int(move[0]), int(move[1])]


def grade_of(statuses: list[str], witness: tuple | None) -> tuple[str, str | None]:
    """(class, grade) from the depth 0, 1, 2 statuses of the human's last move (UNSAFE = the AI wins)."""
    for depth, status in enumerate(statuses):
        if status != 'UNSAFE':
            continue
        exact = all(s == 'SAFE' for s in statuses[:depth])
        if depth == 0:
            return 'PROVEN_WIN', 'WIN_5' if witness[0] == 'five' else 'WIN_VCF'
        if depth == 1:
            return 'PROVEN_WIN', 'WIN_VCT1' if exact else 'WIN_VCT1_OR_LESS'
        return 'PROVEN_WIN', TARGET if exact else 'WIN_VCT2_OR_LESS'
    return ('NOT_FOUND', None) if statuses[2] == 'SAFE' else ('UNKNOWN', None)


def _truth_solver_class():
    from analysis.mcts_v8 import _BudgetedSolver
    from analysis.selective_vct import threat_moves

    class TruthSolver(_BudgetedSolver):
        """Full width: every legal quiet move of the attacker, threat moves first."""

        def quiet_moves(self, game):
            moves = super().quiet_moves(game)
            threats = threat_moves(game)
            first = set(threats)
            return threats + [m for m in moves if m not in first]

    return TruthSolver


def solve(moves, budget: dict) -> dict:
    """Depth 0, 1, 2 status of the last move of ``moves`` (the opponent is to move), one budget for all."""
    from renju import Game

    game = Game()
    for move in moves[:-1]:
        game.play(*move)
    solver = _truth_solver_class()(node_limit=budget['node_limit'], call_limit=budget['call_limit'],
                             node_budget=budget['node_budget'])
    started = perf_counter()
    statuses, seconds, witness = [], [], None
    for depth in (0, 1, 2):
        found = {}

        def check(g, d=depth):
            found['result'] = solver.after_move(g, d)
            return found['result'][0]

        t = perf_counter()
        status = solver._bounded(game, tuple(moves[-1]), None, None, check)
        statuses.append(status)
        seconds.append(round(perf_counter() - t, 3))
        if status == 'UNSAFE':
            witness = found['result'][1]
            break
    statuses += ['UNSAFE'] * (3 - len(statuses))  # a win at depth d is a win at every deeper depth
    cls, grade = grade_of(statuses, witness)
    return {'class': cls, 'grade': grade, 'statuses': statuses,
            'witness': None if witness is None else [witness[0], _first_move(witness)],
            'witness_line': [list(m) for m in witness[1]] if witness and witness[0] == 'vcf' else None,
            'seconds': round(perf_counter() - started, 3), 'depth_seconds': seconds,
            'search': {'nodes': solver.nodes_used, 'vcf_calls': solver.vcf_calls,
                       'vcf_cut_calls': solver.vcf_exhausted, 'budget_exhausted': solver.exhausted}}


def _init(fast: str) -> None:
    if fast != 'off':
        from hybrid.h6_fast import install
        install(fast)


def _task(task: dict) -> dict:
    return {'key': task['key'], 'budget': task['budget'], **solve(task['moves'], task['budget'])}


def load_done(path: Path | None, budget: dict) -> tuple[dict, int]:
    done, torn = {}, 0
    if path is None or not path.exists():
        return done, torn
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            torn += 1
            continue
        if row['budget'] != budget:
            raise ValueError(f"{path}: board {row['key']} was searched with budget {row['budget']}, "
                             f'this run uses {budget}; use another --jsonl file')
        done[row['key']] = row
    return done, torn


def summarize(corpus: dict, rows: dict) -> dict:
    """Counts per split / set: classes, grades; the targets (WIN_VCT2_ONLY) listed by position id."""
    out = {}
    for p in corpus['positions']:
        r = rows.get(p['key'])
        if r is None:
            continue
        s = out.setdefault(f"{p['split']}/{p['set']}", {'positions': 0, 'class': {c: 0 for c in CLASSES},
                                                       'grade': {g: 0 for g in GRADES}, 'targets': []})
        s['positions'] += 1
        s['class'][r['class']] += 1
        if r['grade'] is not None:
            s['grade'][r['grade']] += 1
        if r['grade'] == TARGET:
            s['targets'].append(p['id'])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--sets', nargs='+', choices=('main', 'undone'), default=['main', 'undone'])
    parser.add_argument('--node-limit', type=int, default=DEFAULT_BUDGET['node_limit'], help='per VCF call')
    parser.add_argument('--call-limit', type=int, default=DEFAULT_BUDGET['call_limit'], help='VCF calls per position')
    parser.add_argument('--node-budget', type=int, default=DEFAULT_BUDGET['node_budget'], help='VCF nodes per position')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--fast', choices=('off', 'fast2', 'fast3', 'fast4'), default='off',
                        help='exact speed-up (same results, hybrid.h6_fast)')
    parser.add_argument('--limit', type=int, help='smoke tests: first N boards only')
    parser.add_argument('--jsonl', type=Path, help='one line per finished board (resume)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from hybrid.progress import Progress, run_chunks
    from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256

    corpus = web.load(args.corpus)
    corpus['positions'] = [p for p in corpus['positions'] if p['set'] in args.sets]
    budget = {'node_limit': args.node_limit, 'call_limit': args.call_limit, 'node_budget': args.node_budget}
    boards = {}
    for p in corpus['positions']:
        boards.setdefault(p['key'], web.history(corpus, p))
    keys = list(boards)[:args.limit]
    try:
        done, torn = load_done(args.jsonl, budget)
    except ValueError as exc:
        parser.error(str(exc))
    todo = [k for k in keys if k not in done]
    print(f'[h7-e4-truth] {len(corpus["positions"])} positions, {len(keys)} boards, {len(keys) - len(todo)} done, '
          f'{len(todo)} to search, workers {args.workers}, fast {args.fast}, budget {budget}', flush=True)

    def keep(_, rows_):
        for row in rows_:
            done[row['key']] = row
            if args.jsonl is not None:
                args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.jsonl.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')

    progress = Progress('h7-e4-truth', len(todo), 'boards')
    chunks = [[{'key': k, 'moves': boards[k], 'budget': budget}] for k in todo]
    chunk = (lambda c: [_task(t) for t in c])
    if args.workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'), initializer=_init,
                                 initargs=(args.fast,)) as pool:
            run_chunks(pool, _chunk, chunks, progress, len, unit_name='boards', on_result=keep)
    else:
        from scripts.h7_timing import restore_fast_mode
        with restore_fast_mode():
            _init(args.fast)
            for i, c in enumerate(chunks):
                keep(i, chunk(c))
                progress.update(1)
    rows = {k: done[k] for k in keys}
    summary = summarize(corpus, rows)
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'corpus_sha256': file_sha256(args.corpus),
               'sets': args.sets, 'budget': budget, 'fast': args.fast, 'restricted': args.limit is not None,
               'torn_lines': torn, 'summary': summary, 'boards': rows}
    print(json.dumps(summary, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


def _chunk(tasks):
    return [_task(t) for t in tasks]


if __name__ == '__main__':
    raise SystemExit(main())
