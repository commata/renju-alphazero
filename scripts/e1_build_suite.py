"""E1 rescue suite builder (docs/mcts-v8-teacher.md §12.26). The evaluator is ``e1_evaluate.py``.

The suite is mined automatically from fixed logs: every tree move of the baseline
``puct_policy`` games of seeds 8401/8402 (H5) and 8411/8412 (S3) whose played move V8-C did
not already prove lost. E2's seeds 8413/8414 and the web-game probes (P92-P94) are never used.
Coordinates in files are 0-indexed unless a key says ``_1idx``.

Steps (each resumes from its ``--jsonl``; ``--workers`` processes):

    screen   full depth 0-2 class (every legal quiet move of the attacker) on the played move,
             ``SCREEN_BUDGET``: PROVEN_LOSS (depth), VCT2_CLEAR (exhausted, no loss within
             depth 2) or UNKNOWN (budget). No policy needed.
    pool     needs the H3 policy: for each selected position (PROVEN_LOSS screen rows, one per
             tactical episode, D4-deduplicated; plus a seeded control sample of VCT2_CLEAR rows)
             the frozen engine runs with the VCT2 check off for each of ``E1_SEEDS`` and its
             root children are recorded (the candidate pool and the veto order). Then every pool
             move of a loss position gets its full-class truth with ``TRUTH_BUDGET``.
             Writes the manifest.

Classes: E1-P (some pool move is VCT2_CLEAR), E1-N (every pool move PROVEN_LOSS),
E1-UNRESOLVED (no clear move, some UNKNOWN), E1-C (control: the played move is VCT2_CLEAR).
VCT2_CLEAR is an evaluation label only ("no loss within depth 2"), never a value label (§12.18).

    python scripts/e1_build_suite.py screen --workers 14 --jsonl runs/e1/screen.jsonl --output runs/e1/screen.json
    python scripts/e1_build_suite.py pool --screen runs/e1/screen.json --workers 14 \\
        --jsonl runs/e1/pool.jsonl --output runs/e1/e1_manifest.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from renju import Game  # noqa: E402
from scripts.run_mcts_v8_benchmark import _git_commit, _git_dirty, derive_seed, file_sha256  # noqa: E402
from scripts.s1_loss_analysis import lost_depth_info  # noqa: E402

RESULTS = ROOT / 'docs' / 'mcts-v8-results'
SOURCES = ('h5_policy_8401.json', 'h5_policy_8402.json', 's3_base_8411.json', 's3_base_8412.json')
SCREEN_BUDGET = {'node_limit': 20_000, 'call_limit': 100_000, 'node_budget': 50_000}
TRUTH_BUDGET = {'node_limit': 20_000, 'call_limit': 100_000, 'node_budget': 10_000_000}  # P92 truth (§12.23)
E1_RUNS = 3          # engine seeds per position
CONTROL = 40         # control positions (seeded sample)
CONTROL_SEED = 2611
PROVEN_LOSS, CLEAR, UNKNOWN = 'PROVEN_LOSS', 'VCT2_CLEAR', 'UNKNOWN'
FORMAT = 'e1-manifest-v1'


def truth(history, move, budget) -> dict:
    """Full depth 0-2 class of ``move`` after ``history``."""
    started = perf_counter()
    depth, status, info = lost_depth_info([*history, list(move)], len(history), budget)
    label = PROVEN_LOSS if depth is not None else CLEAR if status == 'SAFE' else UNKNOWN
    return {'status': label, 'lost_depth': depth, 'nodes': info['nodes_used'],
            'budget_exhausted': info['budget_exhausted'], 'seconds': round(perf_counter() - started, 2)}


def e1_seeds(key: str) -> list[int]:
    return [derive_seed('e1', key, run) % (2 ** 31) for run in range(E1_RUNS)]


def _d4(board):
    n = len(board)
    maps = [lambda r, c: (r, c), lambda r, c: (c, n - 1 - r), lambda r, c: (n - 1 - r, n - 1 - c),
            lambda r, c: (n - 1 - c, r), lambda r, c: (r, n - 1 - c), lambda r, c: (n - 1 - r, c),
            lambda r, c: (c, r), lambda r, c: (n - 1 - c, n - 1 - r)]
    for f in maps:
        out = [[0] * n for _ in range(n)]
        for r in range(n):
            for c in range(n):
                rr, cc = f(r, c)
                out[rr][cc] = board[r][c]
        yield out


def canonical_hash(history) -> str:
    """D4-canonical position (stones only; the side to move follows from the stone count)."""
    game = Game()
    for move in history:
        game.play(*move)
    return min(json.dumps(b, separators=(',', ':')) for b in _d4(game.board))


def screen_items() -> list[dict]:
    items = []
    for name in SOURCES:
        run = json.loads((RESULTS / name).read_text(encoding='utf-8'))
        if run['arm'] != 'puct_policy':
            raise ValueError(f'{name}: arm {run["arm"]} is not the baseline puct_policy')
        for game in run['games']:
            for record in game['v8_moves']:
                if record['route'] != 'tree':
                    continue
                status = dict((tuple(m), s) for m, s in record['root']['checked']).get(tuple(record['played']))
                if status == 'UNSAFE':
                    continue  # V8-C already proved it lost: not the case S3 is for
                items.append({'key': f"{run['seed']}-p{game['pair']}-{game['v8_color']}/{record['ply']}",
                              'source': name, 'seed': run['seed'], 'pair': game['pair'],
                              'v8_color': game['v8_color'], 'ply': record['ply'], 'result': game['result'],
                              'history': game['moves'][:record['ply']], 'played': record['played'],
                              'v8c_status': status})
    return items


def _screen_task(item):
    return {**{k: item[k] for k in ('key', 'source', 'seed', 'pair', 'v8_color', 'ply', 'result',
                                    'history', 'played', 'v8c_status')},
            'screen': truth(item['history'], item['played'], SCREEN_BUDGET), 'budget': SCREEN_BUDGET}


def _run_jobs(tasks, fn, workers, jsonl: Path | None, key, budget):
    done = {}
    if jsonl is not None and jsonl.exists():
        for line in jsonl.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn last line: recomputed
            if row.get('budget') != budget:
                raise ValueError(f'{jsonl} holds rows of another budget; use another --jsonl')
            done[row['key']] = row
    pending = [t for t in tasks if key(t) not in done]
    print(f'items={len(tasks)} done={len(tasks) - len(pending)} pending={len(pending)} workers={workers}', flush=True)
    if jsonl is not None:
        jsonl.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    def keep(row, count):
        done[row['key']] = row
        if jsonl is not None:
            with jsonl.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')
        rate = count / max(perf_counter() - started, 1e-9)
        print(f"{len(done)}/{len(tasks)} {row['key']} eta {(len(pending) - count) / rate / 60:.0f} min", flush=True)

    if workers == 1:
        for n, task in enumerate(pending, 1):
            keep(fn(task), n)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, future in enumerate(as_completed([pool.submit(fn, t) for t in pending]), 1):
                keep(future.result(), n)
    return done


def select(rows: list[dict]) -> dict:
    """Loss positions (one per tactical episode, D4-deduplicated) and a seeded control sample."""
    order = {name: i for i, name in enumerate(SOURCES)}
    rows = sorted(rows, key=lambda r: (order[r['source']], r['pair'], r['v8_color'], r['ply']))
    losses, seen, dropped = [], set(), {'same_episode': 0, 'd4_duplicate': 0}
    last = {}
    for row in rows:
        if row['screen']['status'] != PROVEN_LOSS:
            continue
        game = (row['source'], row['pair'], row['v8_color'])
        if last.get(game) == row['ply'] - 2:  # the next V8 move of the same episode
            last[game] = row['ply']
            dropped['same_episode'] += 1
            continue
        last[game] = row['ply']
        h = canonical_hash(row['history'])
        if h in seen:
            dropped['d4_duplicate'] += 1
            continue
        seen.add(h)
        losses.append(row)
    clear = [r for r in rows if r['screen']['status'] == CLEAR]
    controls, games = [], set()
    for row in Random(CONTROL_SEED).sample(clear, len(clear)):
        game = (row['source'], row['pair'], row['v8_color'])
        h = canonical_hash(row['history'])
        if game in games or h in seen:
            continue
        games.add(game)
        seen.add(h)
        controls.append(row)
        if len(controls) == CONTROL:
            break
    return {'losses': losses, 'controls': controls, 'dropped': dropped}


def _pool_task(task):
    """Engine runs (VCT2 check off) for one position, then truths of its pool moves."""
    from analysis.s3_vct2_v1 import S3_VCT2_V1
    from analysis.mcts_v8_agent import MCTSV8Agent
    from scripts.run_mcts_v8_benchmark import load_policy

    item, checkpoint = task
    policy = load_policy(checkpoint)
    game = Game()
    for move in item['history']:
        game.play(*move)
    runs = []
    for seed in item['e1_seeds']:
        agent = MCTSV8Agent(seed=seed, root_policy=policy, **{**S3_VCT2_V1, 'root_vct2_check': False})
        started = perf_counter()
        played = agent.select_move(game)
        d = agent.diagnostics
        runs.append({'seed': seed, 'route': d.v8_route, 'tree_move': list(d.v8_v7_move) if d.v8_v7_move else None,
                     'played': list(played), 'root_checked': [[list(m), s] for m, s in d.v8_root_checked],
                     'children': [[list(m), v, round(q, 6)] for m, v, q in d.v8_root_visits],
                     'seconds': round(perf_counter() - started, 2)})
    pool = sorted({tuple(m) for r in runs for m, _, _ in r['children']} | {tuple(item['played'])})
    truths = {}
    if item['kind'] == 'loss':
        for move in pool:
            truths[f'{move[0]},{move[1]}'] = truth(item['history'], move, TRUTH_BUDGET)
    else:  # controls: only the played move's screen truth is needed
        truths[f"{item['played'][0]},{item['played'][1]}"] = item['screen']
    return {'key': item['key'], 'kind': item['kind'], 'runs': runs, 'truths': truths, 'budget': TRUTH_BUDGET}


def classify(truths: dict) -> str:
    statuses = [t['status'] for t in truths.values()]
    if CLEAR in statuses:
        return 'E1-P'
    if all(s == PROVEN_LOSS for s in statuses):
        return 'E1-N'
    return 'E1-UNRESOLVED'


def k4_subtype(item: dict, max_children: int = 4) -> list[str]:
    """Per run: P-K4 (a clear move among the first ``max_children`` in veto order) or P-POOL."""
    from analysis.mcts_v8 import _root_order
    from types import SimpleNamespace

    out = []
    for run in item['runs']:
        children = [SimpleNamespace(move=tuple(m), visits=v, mean_value=q) for m, v, q in run['children']]
        order = _root_order(tuple(run['played']), children)[:max_children]
        clear = {k for k, t in item['truths'].items() if t['status'] == CLEAR}
        out.append('P-K4' if any(f'{m[0]},{m[1]}' in clear for m in order) else 'P-POOL')
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('step', choices=('screen', 'pool'))
    parser.add_argument('--screen', type=Path, help='pool: the screen output')
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--limit', type=int, help='smoke tests: first N items only')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--jsonl', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    common = {'git_commit': _git_commit(), 'git_dirty': _git_dirty(), 'restricted': args.limit is not None}
    if args.step == 'screen':
        items = screen_items()[:args.limit]
        done = _run_jobs(items, _screen_task, args.workers, args.jsonl, lambda t: t['key'], SCREEN_BUDGET)
        rows = [done[i['key']] for i in items]
        status = {}
        for r in rows:
            status[r['screen']['status']] = status.get(r['screen']['status'], 0) + 1
        payload = {'format': 'e1-screen-v1', **common, 'budget': SCREEN_BUDGET,
                   'inputs': {n: file_sha256(RESULTS / n) for n in SOURCES},
                   'summary': {'items': len(rows), 'status': status}, 'rows': rows}
        print(json.dumps(payload['summary'], indent=1), flush=True)
    else:
        if args.screen is None:
            parser.error('pool needs --screen')
        from analysis.s3_vct2_v1 import check_checkpoint

        hashes = check_checkpoint(args.policy_checkpoint)
        screen = json.loads(args.screen.read_text(encoding='utf-8'))
        chosen = select(screen['rows'])
        items = ([{**r, 'kind': 'loss'} for r in chosen['losses']]
                 + [{**r, 'kind': 'control'} for r in chosen['controls']])[:args.limit]
        for item in items:
            item['e1_seeds'] = e1_seeds(item['key'])
        done = _run_jobs([(i, args.policy_checkpoint) for i in items], _pool_task, args.workers, args.jsonl,
                         lambda t: t[0]['key'], TRUTH_BUDGET)
        positions = []
        for item in items:
            row = done[item['key']]
            entry = {**{k: item[k] for k in ('key', 'source', 'seed', 'pair', 'v8_color', 'ply', 'result',
                                             'history', 'played', 'v8c_status', 'screen', 'e1_seeds')},
                     'runs': row['runs'], 'truths': row['truths']}
            entry['class'] = classify(row['truths']) if item['kind'] == 'loss' else 'E1-C'
            if entry['class'] == 'E1-P':
                entry['subtype_per_run'] = k4_subtype(entry)
            positions.append(entry)
        classes = {}
        for p in positions:
            classes[p['class']] = classes.get(p['class'], 0) + 1
        payload = {'format': FORMAT, **common, 'split': 'dev', 'policy': hashes,
                   'budgets': {'screen': SCREEN_BUDGET, 'truth': TRUTH_BUDGET}, 'e1_runs': E1_RUNS,
                   'screen_input_sha256': file_sha256(args.screen), 'selection': chosen['dropped'],
                   'summary': {'positions': len(positions), 'classes': classes,
                               'p_subtypes': {s: sum(p.get('subtype_per_run', []).count(s) for p in positions)
                                              for s in ('P-K4', 'P-POOL')}},
                   'positions': positions}
        print(json.dumps(payload['summary'], indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
