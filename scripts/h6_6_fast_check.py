"""H6-6a: the exact solver speed-up gives the off results in the H6 tools (docs/h6-value.md, H6-6).

Compares a ``--fast`` mode (default fast3) with results recorded under off, on the machine that will run
H6-6. Three parts; all must match exactly:

    fixture   TB-Engine-v1 decision fields on the fixed positions (tests/fixtures/tb_engine_v1_puct_regression.json)
    v8c       V8-C status of the recorded H6-5D d2 moves: status, nodes, calls, cut, cut kind, per-call hits
              (every standard check, plus ``--large-limit`` 4M checks)
    engines   both engines' tree move, played move, visits and Q of the tree move on ``--engine-limit`` H6-5D
              d3 positions (needs the H3 policy and the v3 value checkpoints)

Times are reported next to the recorded off times. The off times were taken on the machine that ran H6-5D,
so the ratio mixes the speed-up with any machine / OS difference; it is not a verdict.

    python -u scripts/h6_6_fast_check.py --runs runs/h6_5/value_vs_full_8403.json runs/h6_5/tb1_vs_full_8403.json \\
        --d2 docs/h6/h6_5d_d2.json --d3 docs/h6/h6_5d_d3.json --policy-checkpoint runs/h3_policy_64x4/best.pt \\
        --value-checkpoint runs/h6_value_head_v3/best.pt --workers 8 --output docs/h6/h6_6a_fast_check.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid.progress import Progress, run_chunks  # noqa: E402
from scripts import h6_5d  # noqa: E402

FORMAT = 'h6-6a-fast-check-v1'
V8C_FIELDS = ('status', 'nodes', 'calls', 'budget_cut', 'cut_kind', 'per_call_limit_hits')
ENGINE_FIELDS = ('route', 'tree_move', 'played', 'tree_move_visits', 'tree_move_q', 'visited_children')


def _init(mode, policy_checkpoint=None, value_checkpoint=None, value_device='cpu', factories=None):
    h6_5d._ensure_fast(mode)
    if factories is not None or policy_checkpoint:
        h6_5d._init(policy_checkpoint, value_checkpoint, value_device, factories)


def _v8c(task):
    out = h6_5d.check_status(task['moves'], task['move'], task['budget'])
    return {'id': task['id'], 'which': task['which'], 'got': {k: out.get(k) for k in V8C_FIELDS},
            'want': task['want'], 'seconds': out['seconds'], 'off_seconds': task['off_seconds']}


def _engines(task):
    from scripts.run_mcts_v8_benchmark import derive_seed
    game = h6_5d.replay(task['moves'])
    seed = derive_seed(20261019, 'h6-5d', task['id'])  # the d3 seed of this position
    out = {}
    for engine in h6_5d.ENGINES:
        started = perf_counter()
        r = h6_5d._search(engine, game, seed)
        out[engine] = {'got': {k: r.get(k) for k in ENGINE_FIELDS}, 'want': task['want'][engine],
                       'seconds': round(perf_counter() - started, 2), 'off_seconds': task['want_seconds'][engine]}
    return {'id': task['id'], 'engines': out}


def _chunk_v8c(tasks):
    return [_v8c(t) for t in tasks]


def _chunk_engines(tasks):
    return [_engines(t) for t in tasks]


def _pool(fn, tasks, workers, init, label):
    progress = Progress(label, len(tasks), 'checks')
    chunks = [[t] for t in tasks]
    if workers > 1 and tasks:
        # spawn: a forked worker cannot use CUDA once the parent has (and Windows spawns anyway)
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=init,
                                 mp_context=get_context('spawn')) as pool:
            return [r for c in run_chunks(pool, fn, chunks, progress, len, unit_name='checks') for r in c]
    _init(*init)
    out = []
    for c in chunks:
        out.extend(fn(c))
        progress.update(1)
    return out


def v8c_tasks(positions: dict, d2_rows: list[dict], large_limit: int) -> list[dict]:
    tasks = []
    for r in d2_rows:
        p = positions[r['id']]
        tasks.append({'id': r['id'], 'which': 'standard', 'moves': p['moves'], 'move': r['tree_move'],
                      'budget': h6_5d.STANDARD, 'want': {k: r['standard'].get(k) for k in V8C_FIELDS},
                      'off_seconds': r['standard']['seconds']})
    # 4M checks: the cheapest one of each recorded 4M status, then the cheapest others up to large_limit
    large = sorted((r for r in d2_rows if r['standard']['status'] != 'SAFE'), key=lambda r: (r['large']['seconds'], r['id']))
    picked = []
    for status in ('SAFE', 'UNSAFE', 'UNKNOWN'):
        picked += [r for r in large if r['large']['status'] == status][:1]
    picked += [r for r in large if r not in picked][:max(0, large_limit - len(picked))]
    for r in picked:
        p = positions[r['id']]
        tasks.append({'id': r['id'], 'which': 'large', 'moves': p['moves'], 'move': r['tree_move'],
                      'budget': h6_5d.LARGE, 'want': {k: r['large'].get(k) for k in V8C_FIELDS},
                      'off_seconds': r['large']['seconds']})
    return tasks


def engine_tasks(positions: dict, d3_rows: list[dict], limit: int) -> list[dict]:
    rows = sorted(d3_rows, key=lambda r: (sum(r['engines'][e].get('seconds', 0) for e in h6_5d.ENGINES), r['id']))
    return [{'id': r['id'], 'moves': positions[r['id']]['moves'],
             'want': {e: {k: r['engines'][e].get(k) for k in ENGINE_FIELDS} for e in h6_5d.ENGINES},
             'want_seconds': {e: r['engines'][e].get('seconds') for e in h6_5d.ENGINES}} for r in rows[:limit]]


def main(argv=None, factories=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fast', choices=('fast2', 'fast3'), default='fast3')
    parser.add_argument('--runs', nargs=2, type=Path, required=True, help='H6-5 runs B and C of 8403 (positions)')
    parser.add_argument('--d2', type=Path, required=True)
    parser.add_argument('--d3', type=Path)
    parser.add_argument('--large-limit', type=int, default=3)
    parser.add_argument('--engine-limit', type=int, default=6)
    parser.add_argument('--policy-checkpoint')
    parser.add_argument('--value-checkpoint')
    parser.add_argument('--value-device', default='cpu')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    try:
        h6_5d._ensure_fast(args.fast)
    except RuntimeError as exc:
        parser.error(str(exc))
    from tests.tb_engine_v1_regression import FIXTURE, PLIES, decide
    started = perf_counter()
    fixture_ok = [decide(p) for p in PLIES] == json.loads(FIXTURE.read_text(encoding='utf-8'))
    fixture_seconds = round(perf_counter() - started, 1)
    print(f'[fixture] {"same" if fixture_ok else "DIFFERENT"} ({fixture_seconds}s)', flush=True)

    runs = h6_5d.load_runs(args.runs)
    positions = {p['id']: p for r in runs for p in h6_5d.tree_positions(r)}
    d2_rows = json.loads(args.d2.read_text(encoding='utf-8'))['rows']
    v8c = _pool(_chunk_v8c, v8c_tasks(positions, d2_rows, args.large_limit), args.workers, (args.fast,), 'v8c')
    v8c_bad = [r['id'] + '/' + r['which'] for r in v8c if r['got'] != r['want']]
    print(f'[v8c] {len(v8c) - len(v8c_bad)} / {len(v8c)} same', flush=True)

    engines, engines_bad = [], []
    if args.d3 and args.engine_limit > 0:
        if factories is None and not (args.policy_checkpoint and args.value_checkpoint):
            parser.error('the engines part needs --policy-checkpoint and --value-checkpoint (or --engine-limit 0)')
        d3_rows = json.loads(args.d3.read_text(encoding='utf-8'))['rows']
        init = (args.fast, args.policy_checkpoint, args.value_checkpoint, args.value_device, factories)
        engines = _pool(_chunk_engines, engine_tasks(positions, d3_rows, args.engine_limit),
                        1 if factories is not None else args.workers, init, 'engines')
        engines_bad = [f"{r['id']}/{e}" for r in engines for e, x in r['engines'].items() if x['got'] != x['want']]
        print(f'[engines] {len(engines) * 2 - len(engines_bad)} / {len(engines) * 2} same', flush=True)

    def ratio(rows, key='seconds', off='off_seconds'):
        t = sum(r[key] for r in rows)
        o = sum(r[off] for r in rows if r[off] is not None)
        return {'seconds': round(t, 1), 'off_seconds': round(o, 1), 'off_over_fast': round(o / t, 2) if t else None}

    from scripts.run_mcts_v8_benchmark import _fast_provenance
    result = {
        'format': FORMAT, 'fast': _fast_provenance(args.fast), 'platform': sys.platform,
        'python': sys.version.split()[0],
        'pass': fixture_ok and not v8c_bad and not engines_bad,
        'fixture': {'same': fixture_ok, 'positions': len(PLIES), 'seconds': fixture_seconds},
        'v8c': {'checks': len(v8c), 'different': v8c_bad,
                'time_standard': ratio([r for r in v8c if r['which'] == 'standard']),
                'time_large': ratio([r for r in v8c if r['which'] == 'large'])},
        'engines': {'positions': len(engines), 'different': engines_bad,
                    'time': {e: ratio([r['engines'][e] for r in engines]) for e in h6_5d.ENGINES} if engines else None},
        'rows': {'v8c': v8c, 'engines': engines},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('pass', 'fixture')} | {'v8c': {k: v for k, v in result['v8c'].items()},
                                                                   'engines': result['engines']}, indent=1), flush=True)
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
