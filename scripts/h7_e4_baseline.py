"""H7-E4 baseline: does the frozen TB-V2-Value (fast4) punish the web positions it is proven to win?

On every web corpus position whose truth (``h7_e4_truth.py``) is PROVEN_WIN, the frozen runtime profile
(``analysis.tb_v2_runtime``: TB-V2-Value + fast4) chooses a move, and the move is checked:

    PUNISH   after the move every reply of the human still loses within depth 1, i.e. the move keeps a
             proven win of the truth's class (VCT depth <= 2 in total); a five counts. The forced block of
             the human's only four keeps depth 2, as in the truth (``ThreatSolver``: a four chain is free)
    MISS     some reply finished SAFE: the move does not keep a proven win of that class
    UNCLEAR  the check ran out of budget (never counted as MISS)
The check is the truth's solver (full width, threat moves first) with the truth's budget.

The web AI's actual reply (an older engine in most games) is checked the same way, for reference only.
Reported per split (DEV / HOLDOUT) and set (main / undone): per truth grade the counts above. The E4 target
is WIN_VCT2_ONLY; a WIN_5 / WIN_VCF / WIN_VCT1 position B does not punish is flagged (expected: none, B has
an own VCF and the depth-1 attack). HOLDOUT is a non-regression gate for E4 only with >= 10 VCT2-only events
(``HOLDOUT_MIN_EVENTS``), otherwise report-only.

    python scripts/h7_e4_baseline.py --corpus docs/h7/h7_e4_web_corpus.json --truth docs/h7/h7_e4_truth.json \\
        --device cuda --jsonl runs/h7_e4/baseline.jsonl --output docs/h7/h7_e4_baseline.json

``--recheck BASELINE`` keeps the moves of an earlier run (no engine) and redoes only the checks the current
rule changes: v1 checked a forced block at depth 1 and so counted a block that keeps the win as MISS.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import h7_e4_truth as truth  # noqa: E402
from scripts import h7_e4_web_corpus as web  # noqa: E402

FORMAT = 'h7-e4-baseline-v2'  # v2: the forced block of the human's four keeps depth 2
SEED = 8440  # TB-V2-Value's search seed per position: derive_seed(SEED, 'h7-e4', position id)
OUTCOMES = ('PUNISH', 'MISS', 'UNCLEAR')
SHALLOW = ('WIN_5', 'WIN_VCF', 'WIN_VCT1')
HOLDOUT_MIN_EVENTS = 10


def remaining_depth(game, move) -> int:
    """Depth left to the attacker after ``move``: 2 for the forced block of the defender's only four, else 1.

    The defender just moved; in the truth's search (``ThreatSolver.after_move``) the block of its single
    four does not use a depth, every other first move of the attacker does (VCF moves are inside depth 1).
    """
    from search.mcts_v5 import _winning_moves

    fours = _winning_moves(game, -game.to_play)
    return 2 if len(fours) == 1 and tuple(fours[0]) == tuple(move) else 1


def check_move(moves, move, budget: dict) -> dict:
    """PUNISH / MISS / UNCLEAR of ``move`` played after ``moves`` (the mover is the attacker)."""
    from analysis.threats import SAFE, UNSAFE, decision_status
    from renju import Game

    game = Game()
    for m in moves:
        game.play(*m)
    depth = remaining_depth(game, move)
    solver = truth._truth_solver_class()(node_limit=budget['node_limit'], call_limit=budget['call_limit'],
                             node_budget=budget['node_budget'])

    def check(g):
        if g.done:
            return UNSAFE if g.winner is not None else SAFE
        counts, _ = solver.decision(g, depth, stop_at_safe=True)  # every human reply, from the human's side
        return decision_status(counts)

    status = solver._bounded(game, tuple(move), None, None, check)
    return {'outcome': {'UNSAFE': 'PUNISH', 'SAFE': 'MISS'}.get(status, 'UNCLEAR'), 'depth': depth,
            'search': {'nodes': solver.nodes_used, 'vcf_calls': solver.vcf_calls,
                       'vcf_cut_calls': solver.vcf_exhausted, 'budget_exhausted': solver.exhausted}}


def _init(policy_checkpoint, value_checkpoint, device, threads, factories=None):
    from analysis import tb_v2_runtime as rt
    from scripts import h7_timing

    h7_timing._init(policy_checkpoint, value_checkpoint, device, threads, rt.FAST_MODE, factories)
    rt.check_bound()


def position_task(task: dict) -> dict:
    """B's move on one position and the checks of B's move and of the web AI's move."""
    from analysis import tb_v2_runtime as rt
    from scripts import h7_timing
    from scripts.run_mcts_v8_benchmark import derive_seed

    rt.check_bound()  # the frozen profile, as rt.make_agent requires
    moves = task['moves']
    decided = h7_timing.decide(moves, derive_seed(SEED, 'h7-e4', task['id']))
    d = decided['decision']
    row = {'id': task['id'], 'key': task['key'], 'budget': task['budget'],
           'b': {'move': decided['move'], 'route': d['route'], 'tree_move': d['tree_move'], 'changed': d['changed'],
                 'seconds': decided['seconds'], 'times': decided['times'],
                 **check_move(moves, decided['move'], task['budget'])}}
    web_move = task.get('web_move')
    row['web_ai'] = None if web_move is None else {'move': web_move, **check_move(moves, web_move, task['budget'])}
    return row


def _chunk(tasks):
    return [position_task(t) for t in tasks]


def summarize(corpus: dict, boards: dict, rows: dict) -> dict:
    """Per split / set and truth grade: positions and the outcomes of B and of the web AI; the flags."""
    out, flags = {}, []
    for p in corpus['positions']:
        r = rows.get(p['id'])
        if r is None:
            continue
        grade = boards[p['key']]['grade']
        s = out.setdefault(f"{p['split']}/{p['set']}", {})
        g = s.setdefault(grade, {'positions': 0, 'b': {o: 0 for o in OUTCOMES}, 'web_ai': {o: 0 for o in OUTCOMES},
                                 'web_ai_none': 0, 'b_missed': [], 'b_unclear': []})
        g['positions'] += 1
        g['b'][r['b']['outcome']] += 1
        if r['b']['outcome'] == 'MISS':
            g['b_missed'].append(p['id'])
        elif r['b']['outcome'] == 'UNCLEAR':
            g['b_unclear'].append(p['id'])
        if r['web_ai'] is None:
            g['web_ai_none'] += 1
        else:
            g['web_ai'][r['web_ai']['outcome']] += 1
        if grade in SHALLOW and r['b']['outcome'] != 'PUNISH':
            flags.append({'id': p['id'], 'grade': grade, 'outcome': r['b']['outcome'], 'route': r['b']['route']})
    holdout = out.get('HOLDOUT/main', {}).get(truth.TARGET, {}).get('positions', 0)
    return {'by_split': out, 'shallow_flags': flags,
            'target': {k: {'positions': v.get(truth.TARGET, {}).get('positions', 0),
                           'b_punished': v.get(truth.TARGET, {}).get('b', {}).get('PUNISH', 0),
                           'b_missed': v.get(truth.TARGET, {}).get('b', {}).get('MISS', 0),
                           'b_unclear': v.get(truth.TARGET, {}).get('b', {}).get('UNCLEAR', 0)} for k, v in out.items()},
            'holdout_gate': 'non-regression' if holdout >= HOLDOUT_MIN_EVENTS else 'report-only',
            'holdout_target_events': holdout}


def main(argv=None, factories=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True, help='h7_e4_truth.py --output')
    parser.add_argument('--grades', nargs='+', choices=truth.GRADES, default=list(truth.GRADES))
    parser.add_argument('--device', default='cuda', help='value device (cuda is canonical)')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--value-checkpoint', default=str(ROOT / 'runs/h6_value_head_v3/best.pt'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--jsonl', type=Path, help='one line per finished position (resume)')
    parser.add_argument('--recheck', type=Path, help='an earlier baseline output: keep its moves, redo changed checks')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    if args.recheck is not None:
        return recheck(args, parser)
    from analysis import tb_v2_runtime as rt
    from hybrid.progress import Progress, run_chunks
    from scripts.h7_timing import restore_fast_mode
    from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256

    corpus = web.load(args.corpus)
    t = json.loads(args.truth.read_text(encoding='utf-8'))
    if t.get('format') != truth.FORMAT or t.get('corpus_sha256') != file_sha256(args.corpus):
        parser.error(f'{args.truth} is not a truth file of this corpus')
    boards, budget = t['boards'], t['budget']
    corpus['positions'] = [p for p in corpus['positions']
                           if p['key'] in boards and boards[p['key']]['grade'] in args.grades]
    tasks = [{'id': p['id'], 'key': p['key'], 'moves': web.history(corpus, p), 'budget': budget,
              'web_move': p['web_ai']['move'] if p['web_ai'] else None} for p in corpus['positions']][:args.limit]
    done = {}
    if args.jsonl is not None and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line: the position is run again
            done[row['id']] = row
    todo = [x for x in tasks if x['id'] not in done]
    print(f'[h7-e4-baseline] {len(tasks)} proven-win positions, {len(tasks) - len(todo)} done, '
          f'workers {args.workers}, device {args.device}', flush=True)

    def keep(_, rows_):
        for row in rows_:
            done[row['id']] = row
            if args.jsonl is not None:
                args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.jsonl.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')

    progress = Progress('h7-e4-baseline', len(todo), 'positions')
    chunks = [[x] for x in todo]
    init = (args.policy_checkpoint, args.value_checkpoint, args.device, args.threads, factories)
    if args.workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'), initializer=_init,
                                 initargs=init) as pool:
            run_chunks(pool, _chunk, chunks, progress, len, unit_name='positions', on_result=keep)
    else:
        with restore_fast_mode():
            _init(*init)
            for i, c in enumerate(chunks):
                keep(i, _chunk(c))
                progress.update(1)
    rows = {x['id']: done[x['id']] for x in tasks}
    summary = summarize(corpus, boards, rows)
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'corpus_sha256': file_sha256(args.corpus),
               'truth_sha256': file_sha256(args.truth), 'budget': budget, 'seed': SEED,
               'runtime': rt.provenance(args.device) if factories is None else None,
               'restricted': args.limit is not None or set(args.grades) != set(truth.GRADES),
               'summary': summary, 'positions': rows}
    print(json.dumps({k: summary[k] for k in ('target', 'shallow_flags', 'holdout_gate', 'holdout_target_events')},
                     indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


def recheck(args, parser) -> int:
    """``--recheck``: the moves of an earlier run, the checks whose depth the current rule changes redone."""
    from renju import Game
    from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256

    corpus = web.load(args.corpus)
    old = json.loads(args.recheck.read_text(encoding='utf-8'))
    t = json.loads(args.truth.read_text(encoding='utf-8'))
    if old.get('corpus_sha256') != file_sha256(args.corpus) or old.get('truth_sha256') != file_sha256(args.truth):
        parser.error(f'{args.recheck} is not a baseline of this corpus and truth')
    positions = {p['id']: p for p in corpus['positions']}
    rows, changed = {}, []
    for pid, row in old['positions'].items():
        moves = web.history(corpus, positions[pid])
        game = Game()
        for m in moves:
            game.play(*m)
        row = json.loads(json.dumps(row))
        for who in ('b', 'web_ai'):
            if row[who] is None or row[who].get('depth', 1) == remaining_depth(game, row[who]['move']):
                continue
            before = row[who]['outcome']
            row[who].update(check_move(moves, row[who]['move'], old['budget']))
            changed.append({'id': pid, 'who': who, 'move': row[who]['move'], 'before': before,
                            'after': row[who]['outcome'], 'depth': row[who]['depth']})
            print(f"[recheck] {pid} {who} {row[who]['move']}: {before} -> {row[who]['outcome']}", flush=True)
        rows[pid] = row
    corpus['positions'] = [p for p in corpus['positions'] if p['id'] in rows]
    summary = summarize(corpus, t['boards'], rows)
    payload = {**{k: v for k, v in old.items() if k not in ('format', 'summary', 'positions')},
               'format': FORMAT, 'recheck': {'of_sha256': file_sha256(args.recheck), 'of_format': old.get('format'),
                                             'git_commit': _git_commit(), 'changed': changed},
               'summary': summary, 'positions': rows}
    print(json.dumps({k: summary[k] for k in ('target', 'shallow_flags', 'holdout_gate', 'holdout_target_events')},
                     indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
