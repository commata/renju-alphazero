"""H6-5D: why the 8403 tree not-SAFE rate rose (docs/h6-value.md, H6-5D). Report only; 8403 stays REJECT.

Three steps, all on positions of the recorded H6-5 runs B (TB-V1-Value vs v8:full) and C (TB-Engine-v1 vs
v8:full); no game is played.

``d2``  UNKNOWN resolution. Every tree move whose V8-C status was UNKNOWN (main set) or UNSAFE (control set)
        is checked again, first exactly as V8-C checked it (``STANDARD``: a fresh solver, the tree move's first
        share = half of the root budget; must reproduce the recorded status), then with every V8-C limit x10
        (``LARGE``: 200k nodes per VCF call, 100k calls, 4M nodes). A result is "V8-C@4M SAFE / UNSAFE /
        UNKNOWN": V8-C's own VCT1 definition at that budget, not a game-theoretic truth.
``d3``  Same-position comparison. The not-SAFE positions of B and C plus one matched SAFE control each
        (same game, nearest ply; else same run, same game kind, nearest ply) are searched by both engines
        with the same seed. Both tree moves are checked with the same solver (``STANDARD``, and ``LARGE`` when
        not SAFE), so each engine's move is compared with the other's on the same board. Also recorded per
        engine: prior rank of its tree move, its visits and Q, visited root children, root Q spread.
``judge``  The decision rule fixed before d2 / d3 ran (``RULES``), in this order:
        (sanity: d2 must be consistent with every recorded status (``replay_kind``), and the UNSAFE controls
        must stay UNSAFE)
        A  same positions (d3): value's tree move is V8-C@4M UNSAFE more often than the baseline's,
           paired game-cluster bootstrap 95% lower bound of (value - baseline) > 0
        UNRESOLVED  d2 main: still UNKNOWN at 4M > 1/3                       -> INCONCLUSIVE
        C  d2 main: V8-C@4M SAFE >= 2/3 and V8-C@4M UNSAFE <= 0.10             (proof budget)
        B  d3: standard not-SAFE paired difference 95% interval contains 0  (no same-position excess)
        otherwise INCONCLUSIVE

    python -u scripts/h6_5d.py d2 --runs runs/h6_5/value_vs_full_8403.json runs/h6_5/tb1_vs_full_8403.json \\
        --workers 12 --jsonl runs/h6_5d/d2.jsonl --output docs/h6/h6_5d_d2.json
    python -u scripts/h6_5d.py d3 --runs runs/h6_5/value_vs_full_8403.json runs/h6_5/tb1_vs_full_8403.json \\
        --d2 docs/h6/h6_5d_d2.json --policy-checkpoint ../renju-alphazero/runs/h3_policy_64x4/best.pt \\
        --value-checkpoint runs/h6_value_head_v3/best.pt --value-device cuda --workers 6 \\
        --jsonl runs/h6_5d/d3.jsonl --output docs/h6/h6_5d_d3.json
    python scripts/h6_5d.py judge --d2 docs/h6/h6_5d_d2.json --d3 docs/h6/h6_5d_d3.json \\
        --output docs/h6/h6_5d_verdict.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.tb_engine_v1 import TB_ENGINE_V1  # noqa: E402
from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import Game  # noqa: E402

FORMAT = 'h6-5d-v1'
# V8-C as TB-Engine-v1 runs it; the tree move is checked first with half of the root budget.
STANDARD = {'node_limit': TB_ENGINE_V1['root_vcf_node_limit'], 'call_limit': TB_ENGINE_V1['root_call_limit'],
            'node_budget': TB_ENGINE_V1['root_node_budget'],
            'share': TB_ENGINE_V1['root_node_budget'] // 2, 'call_share': TB_ENGINE_V1['root_call_limit'] // 2}
LARGE = {'node_limit': 10 * STANDARD['node_limit'], 'call_limit': 10 * STANDARD['call_limit'],
         'node_budget': 10 * STANDARD['node_budget'], 'share': None, 'call_share': None}
RULES = {'a_lower_bound_above': 0.0, 'unresolved_max': 1 / 3, 'c_safe_min': 2 / 3, 'c_unsafe_max': 0.10,
         'bootstrap': 10_000, 'seed': 0}
ENGINES = ('TB-Engine-v1', 'TB-V1-Value')
CANDIDATE_ARM, BASELINE_ARM = 'puct_value_vct2_50k', 'puct_policy_vct2_50k'
_WORKER: dict = {}


# ---------- positions ----------

def tree_status(move) -> str:
    return {tuple(m): s for m, s in move['root']['checked']}.get(tuple(move['v7_move']), 'UNCHECKED')


def game_kind(game) -> str:
    if game['result'] == 'draw':
        return 'draw_full_board' if game['length'] >= 225 else 'draw'
    return 'decisive'


def tree_positions(run: dict) -> list[dict]:
    out = []
    for g in sorted(run['games'], key=lambda g: g['key']):
        for m in g['v8_moves']:
            if m['route'] == 'tree':
                out.append({'id': f"{run['arm']}|{g['key']}|{m['ply']}", 'arm': run['arm'], 'game_key': g['key'],
                            'game_kind': game_kind(g), 'ply': m['ply'], 'moves': g['moves'][:m['ply']],
                            'tree_move': list(m['v7_move']), 'recorded_status': tree_status(m)})
    return out


def match_controls(positions: list[dict]) -> list[dict]:
    """One SAFE control per not-SAFE position: same game and nearest ply, else same arm and game kind and
    nearest ply; ties to the earlier ply, then the smaller id. Each control is used once."""
    safe = [p for p in positions if p['recorded_status'] == 'SAFE']
    used, out = set(), []
    for p in [p for p in positions if p['recorded_status'] in ('UNKNOWN', 'UNSAFE')]:
        for level, pool in (('same_game', [s for s in safe if s['game_key'] == p['game_key'] and s['arm'] == p['arm']]),
                            ('same_kind', [s for s in safe if s['arm'] == p['arm'] and s['game_kind'] == p['game_kind']])):
            pool = [s for s in pool if s['id'] not in used]
            if pool:
                best = min(pool, key=lambda s: (abs(s['ply'] - p['ply']), s['ply'], s['id']))
                used.add(best['id'])
                out.append({**best, 'control_for': p['id'], 'match': level})
                break
    return out


def replay(moves) -> Game:
    game = Game()
    for move in moves:
        game.play(*move)
    return game


# ---------- V8-C status ----------

def check_status(moves, move, budget: dict) -> dict:
    from analysis.mcts_v8 import _BudgetedSolver
    game = replay(moves)
    solver = _BudgetedSolver(node_limit=budget['node_limit'], call_limit=budget['call_limit'],
                             node_budget=budget['node_budget'])
    started = perf_counter()
    status = solver.status_after(game, tuple(move), budget['share'], budget['call_share'])
    node_cap = budget['node_budget'] if budget['share'] is None else budget['share']
    call_cap = budget['call_limit'] if budget['call_share'] is None else budget['call_share']
    cut_kind = None
    if solver.last_cut:  # which cap stopped it: the uncached VCF calls or the VCF nodes
        cut_kind = 'calls' if solver.vcf_calls >= call_cap else 'nodes' if solver.nodes_used >= node_cap else 'other'
    elif status == 'UNKNOWN':  # finished inside the budget, but a VCF call hit its own node limit
        cut_kind = 'per_call_limit'
    return {'status': status, 'nodes': solver.nodes_used, 'calls': solver.vcf_calls,
            'budget_cut': solver.last_cut, 'cut_kind': cut_kind, 'exhausted': solver.exhausted,
            'per_call_limit_hits': solver.vcf_exhausted, 'seconds': round(perf_counter() - started, 2)}


def _ensure_fast(mode: str) -> None:
    """Install the exact solver speed-up in this worker (idempotent; same statuses, nodes and cuts)."""
    if mode != 'off':
        from hybrid.h6_fast import install, installed_mode
        if installed_mode() != mode:
            install(mode)


def replay_kind(recorded: str, standard: str, large: str) -> str:
    """How the fresh first-share check (``STANDARD``) relates to the status V8-C recorded.

    The standard check is exactly V8-C's first check of the tree move (fresh solver, half the budget),
    but V8-C does not stop there: a move left UNKNOWN stays pending in ``_proven`` and is checked again
    in later rounds with the same solver and cache, so the recorded status can be decided after an
    UNKNOWN first check. Hence:
        exact      the first check gives the recorded status
        continued  the first check is UNKNOWN, the recorded status is decided (V8-C went on) and the
                   4M recheck agrees with it
        mismatch   anything else (a decided first check that differs, a recorded UNKNOWN the first
                   check decides, or a continued status the 4M recheck contradicts)
    """
    if standard == recorded:
        return 'exact'
    if standard == 'UNKNOWN' and recorded in ('SAFE', 'UNSAFE') and large == recorded:
        return 'continued'
    return 'mismatch'


def d2_task(task: dict) -> dict:
    _ensure_fast(task.get('fast', 'off'))
    std = check_status(task['moves'], task['tree_move'], task['standard'])
    large = check_status(task['moves'], task['tree_move'], task['large']) if std['status'] != 'SAFE' else std
    return {'id': task['id'], 'arm': task['arm'], 'set': task['set'], 'game_key': task['game_key'],
            'game_kind': task['game_kind'], 'ply': task['ply'], 'tree_move': task['tree_move'],
            'recorded_status': task['recorded_status'], 'standard': std,
            'replay': replay_kind(task['recorded_status'], std['status'], large['status']), 'large': large}


# ---------- d3 ----------

def _init(policy_checkpoint, value_checkpoint, value_device, factories=None):
    if factories is not None:  # tests: torch-free stand-ins
        _WORKER.update(factories)
        return
    from hybrid.h4_policy import RootPolicy
    from hybrid.h6_value_leaf import ValueLeaf
    _WORKER['policy'] = RootPolicy(policy_checkpoint)
    _WORKER['leaf'] = ValueLeaf(value_checkpoint, device=value_device)


def _search(engine: str, game: Game, seed: int) -> dict:
    from analysis import tb_engine_v1, tb_v1_value
    policy = _WORKER['policy']
    agent = (tb_engine_v1.make_agent(policy, seed=seed) if engine == ENGINES[0]
             else tb_v1_value.make_agent(policy, _WORKER['leaf'], seed=seed))
    started = perf_counter()
    played = agent.select_move(game)
    d = agent.diagnostics
    if d.v8_route != 'tree' or d.v8_v7_move is None:
        return {'route': d.v8_route, 'tree_move': None, 'played': list(played)}
    tree_move = tuple(d.v8_v7_move)
    children = {tuple(m): (n, q) for m, n, q in d.v8_root_visits}
    scores = policy(game)
    ranked = sorted(scores, key=lambda m: (-scores[m], m))
    visited = [q for n, q in children.values() if n > 0]
    mean = sum(visited) / len(visited) if visited else 0.0
    return {'route': d.v8_route, 'tree_move': list(tree_move), 'played': list(played),
            'seconds': round(perf_counter() - started, 2),
            'tree_move_visits': children.get(tree_move, (0, 0.0))[0],
            'tree_move_q': round(children.get(tree_move, (0, 0.0))[1], 4),
            'tree_move_prior_rank': ranked.index(tree_move) + 1 if tree_move in scores else None,
            'prior_top': tree_move == tuple(d.v8_puct_prior_top) if d.v8_puct_prior_top else None,
            'visited_children': len(visited),
            'root_q_std': round((sum((q - mean) ** 2 for q in visited) / len(visited)) ** 0.5, 4) if visited else None}


def d3_task(task: dict) -> dict:
    _ensure_fast(task.get('fast', 'off'))
    from scripts.run_mcts_v8_benchmark import derive_seed
    game = replay(task['moves'])
    seed = derive_seed(20261019, 'h6-5d', task['id'])
    engines = {e: _search(e, game, seed) for e in ENGINES}
    known = task.get('known', {})
    statuses = {}
    for move in {tuple(r['tree_move']) for r in engines.values() if r['tree_move'] is not None}:
        key = json.dumps(list(move))
        if key in known:
            statuses[key] = known[key]
            continue
        std = check_status(task['moves'], move, task['standard'])
        large = check_status(task['moves'], move, task['large']) if std['status'] != 'SAFE' else std
        statuses[key] = {'standard': std['status'], 'large': large['status'],
                         'standard_nodes': std['nodes'], 'large_nodes': large['nodes']}
    for r in engines.values():
        if r['tree_move'] is not None:
            r.update({f'status_{k}': v for k, v in statuses[json.dumps(r['tree_move'])].items()})
    return {'id': task['id'], 'arm': task['arm'], 'game_key': task['game_key'], 'ply': task['ply'],
            'game_kind': task['game_kind'], 'kind': task['kind'], 'recorded_status': task['recorded_status'],
            'match': task.get('match'), 'control_for': task.get('control_for'), 'engines': engines,
            'same_tree_move': engines[ENGINES[0]]['tree_move'] == engines[ENGINES[1]]['tree_move']}


# ---------- runner plumbing ----------

def _chunk_d2(tasks):
    return [d2_task(t) for t in tasks]


def _chunk_d3(tasks):
    return [d3_task(t) for t in tasks]


def _run(tasks, fn, workers, jsonl: Path | None, label, init=None) -> list[dict]:
    done = {}
    if jsonl is not None and jsonl.exists():
        for line in jsonl.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                done[row['id']] = row
    todo = [t for t in tasks if t['id'] not in done]
    print(f'[{label}] {len(tasks)} tasks, {len(tasks) - len(todo)} already done, workers {workers}', flush=True)

    def keep(i, rows):
        for row in rows:
            done[row['id']] = row
            if jsonl is not None:
                jsonl.parent.mkdir(parents=True, exist_ok=True)
                with jsonl.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')

    progress = Progress(label, len(todo), 'positions')
    chunks = [[t] for t in todo]
    if workers > 1 and todo:
        kwargs = {'initializer': _init, 'initargs': init} if init else {}
        # spawn: a forked worker cannot use CUDA once the parent has (and Windows spawns anyway)
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), **kwargs) as pool:
            run_chunks(pool, fn, chunks, progress, len, unit_name='positions', on_result=keep)
    else:
        if init:
            _init(*init)
        for i, c in enumerate(chunks):
            keep(i, fn(c))
            progress.update(1)
    return [done[t['id']] for t in tasks]


def load_runs(paths) -> list[dict]:
    runs = [json.loads(Path(p).read_text(encoding='utf-8')) for p in paths]
    arms = sorted(r['arm'] for r in runs)
    if arms != sorted([CANDIDATE_ARM, BASELINE_ARM]) or len({(r['seed'], r['opponent']) for r in runs}) != 1:
        raise ValueError(f'need {CANDIDATE_ARM} and {BASELINE_ARM} runs on the same seed and opponent')
    return runs


def _share(rows, key, value):
    return round(sum(r[key] == value for r in rows) / len(rows), 4) if rows else None


def summarize_d2(rows: list[dict]) -> dict:
    # recomputed from the statuses, so files written before ``replay_kind`` read the same way
    kinds = {r['id']: replay_kind(r['recorded_status'], r['standard']['status'], r['large']['status']) for r in rows}
    out = {'replay_mismatches': [i for i, k in kinds.items() if k == 'mismatch'],
           'replay_continued': [i for i, k in kinds.items() if k == 'continued']}
    for name in ('main', 'control'):
        part = [r for r in rows if r['set'] == name]
        by = {}
        for arm in (CANDIDATE_ARM, BASELINE_ARM, 'pooled'):
            sub = [r for r in part if arm in ('pooled', r['arm'])]
            large = [{'s': r['large']['status']} for r in sub]
            by[arm] = {'n': len(sub), **{f'large_{s.lower()}': sum(x['s'] == s for x in large)
                                          for s in ('SAFE', 'UNSAFE', 'UNKNOWN')},
                       'share_safe': _share(large, 's', 'SAFE'), 'share_unsafe': _share(large, 's', 'UNSAFE'),
                       'share_unknown': _share(large, 's', 'UNKNOWN'),
                       'standard_budget_cut': sum(r['standard']['budget_cut'] for r in sub),
                       'standard_cut_kind': {k: sum(r['standard'].get('cut_kind') == k for r in sub)
                                             for k in ('calls', 'nodes', 'per_call_limit', 'other')},
                       'large_cut_kind': {k: sum(r['large'].get('cut_kind') == k for r in sub)
                                          for k in ('calls', 'nodes', 'per_call_limit', 'other')},
                       'standard_per_call_limit_hits': sum(r['standard']['per_call_limit_hits'] > 0 for r in sub)}
        out[name] = by
    return out


def _cluster_bootstrap(rows, value, seed=0, n=10_000) -> dict:
    """Mean over positions of ``value(row)``, 95% bootstrap over source games (positions of a game together)."""
    groups = {}
    for r in rows:
        groups.setdefault(f"{r['arm']}|{r['game_key']}", []).append(value(r))
    keys = sorted(groups)
    if not keys:
        return {'n': 0}

    def mean_of(ks):
        vals = [v for k in ks for v in groups[k]]
        return sum(vals) / len(vals)

    rng = Random(seed)
    draws = sorted(mean_of([rng.choice(keys) for _ in keys]) for _ in range(n))
    return {'n': len(rows), 'games': len(keys), 'mean': round(mean_of(keys), 4),
            'ci95': [round(draws[int(0.025 * n)], 4), round(draws[int(0.975 * n) - 1], 4)]}


def summarize_d3(rows: list[dict]) -> dict:
    base, cand = ENGINES
    both = [r for r in rows if r['engines'][base]['tree_move'] is not None and r['engines'][cand]['tree_move'] is not None]

    def flag(r, engine, key, values):
        return 1.0 if r['engines'][engine][key] in values else 0.0

    out = {'positions': len(rows), 'both_tree': len(both), 'same_tree_move': sum(r['same_tree_move'] for r in both)}
    for kind in ('not_safe', 'control', 'all'):
        sub = [r for r in both if kind == 'all' or r['kind'] == kind]
        out[kind] = {
            'n': len(sub),
            'standard_not_safe': {e: sum(flag(r, e, 'status_standard', ('UNKNOWN', 'UNSAFE')) for r in sub) for e in ENGINES},
            'large_unsafe': {e: sum(flag(r, e, 'status_large', ('UNSAFE',)) for r in sub) for e in ENGINES},
            'large_unknown': {e: sum(flag(r, e, 'status_large', ('UNKNOWN',)) for r in sub) for e in ENGINES},
            'diff_large_unsafe': _cluster_bootstrap(sub, lambda r: flag(r, cand, 'status_large', ('UNSAFE',))
                                                    - flag(r, base, 'status_large', ('UNSAFE',)), RULES['seed'],
                                                    RULES['bootstrap']),
            'diff_standard_not_safe': _cluster_bootstrap(
                sub, lambda r: flag(r, cand, 'status_standard', ('UNKNOWN', 'UNSAFE'))
                - flag(r, base, 'status_standard', ('UNKNOWN', 'UNSAFE')), RULES['seed'], RULES['bootstrap']),
            'prior_top': {e: sum(bool(r['engines'][e].get('prior_top')) for r in sub) for e in ENGINES},
            'visited_children_mean': {e: round(sum(r['engines'][e]['visited_children'] for r in sub) / len(sub), 2)
                                      if sub else None for e in ENGINES},
        }
    return out


def judge(d2: dict, d3: dict) -> dict:
    main, same = d2['main']['pooled'], d3['all']
    a_low = same['diff_large_unsafe'].get('ci95', [None])[0]
    std_ci = same['diff_standard_not_safe'].get('ci95')
    evidence = {'d3_diff_large_unsafe': same['diff_large_unsafe'], 'd3_diff_standard_not_safe': same['diff_standard_not_safe'],
                'd2_main_pooled': main}
    if d2['replay_mismatches']:
        return {'cause': 'INCONCLUSIVE', 'reason': 'd2 did not reproduce the recorded statuses', 'evidence': evidence}
    control = d2.get('control', {}).get('pooled', {})
    if control.get('n') and control.get('share_unsafe') != 1.0:
        return {'cause': 'INCONCLUSIVE', 'reason': 'd2 control: an UNSAFE proof did not stay UNSAFE at 4M',
                'evidence': evidence}
    if a_low is not None and a_low > RULES['a_lower_bound_above']:
        return {'cause': 'A', 'reason': 'same positions: value tree move V8-C@4M UNSAFE more often (95% low > 0)',
                'next': 'deep-loss labels (vct_loss) -> new value candidate', 'evidence': evidence}
    if main['share_unknown'] is not None and main['share_unknown'] > RULES['unresolved_max']:
        return {'cause': 'INCONCLUSIVE', 'reason': 'd2: still UNKNOWN at 4M > 1/3', 'evidence': evidence}
    if main['share_safe'] is not None and main['share_safe'] >= RULES['c_safe_min'] \
            and main['share_unsafe'] <= RULES['c_unsafe_max']:
        return {'cause': 'C', 'reason': 'd2: V8-C@4M SAFE >= 2/3 and UNSAFE <= 0.10 (proof budget)',
                'next': 'no value change; V8-C budget / endgame handling and the guard definition, both engines',
                'evidence': evidence}
    if std_ci is not None and std_ci[0] <= 0 <= std_ci[1]:
        return {'cause': 'B', 'reason': 'd3: no same-position excess at the standard budget',
                'next': 'one new arm: lambda 0.5 (value + rollout)', 'evidence': evidence}
    return {'cause': 'INCONCLUSIVE', 'reason': 'no rule matched', 'evidence': evidence}


def main(argv=None, factories=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='step', required=True)
    p2 = sub.add_parser('d2')
    p3 = sub.add_parser('d3')
    pj = sub.add_parser('judge')
    for p in (p2, p3):
        p.add_argument('--runs', nargs=2, type=Path, required=True, help='runs B and C (same seed, v8:full)')
        p.add_argument('--workers', type=int, default=1)
        p.add_argument('--jsonl', type=Path, help='per-position results, appended as they finish (resume)')
        p.add_argument('--limit', type=int, help='first N positions (smoke runs)')
        p.add_argument('--fast', choices=('off', 'fast2', 'fast3'), default='off',
                       help='exact solver speed-up in every worker (hybrid.h6_fast); results are the same')
        p.add_argument('--output', type=Path, required=True)
    p3.add_argument('--d2', type=Path, help='d2 output: statuses of the same (position, move) are reused')
    p3.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    p3.add_argument('--value-checkpoint')
    p3.add_argument('--value-device', default='cpu')
    pj.add_argument('--d2', type=Path, required=True)
    pj.add_argument('--d3', type=Path, required=True)
    pj.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    if args.step == 'judge':
        d2 = json.loads(args.d2.read_text(encoding='utf-8'))['summary']
        d3 = json.loads(args.d3.read_text(encoding='utf-8'))['summary']
        result = {'format': FORMAT, 'step': 'judge', 'rules': RULES, 'verdict': judge(d2, d3)}
    else:
        runs = load_runs(args.runs)
        positions = [p for r in runs for p in tree_positions(r)]
        sources = {str(p): r['arm'] for p, r in zip(args.runs, runs)}
        if args.fast != 'off':
            try:
                _ensure_fast(args.fast)  # fail before any task (native build missing or stale)
            except RuntimeError as exc:
                parser.error(str(exc))
        if args.step == 'd2':
            tasks = [{**p, 'set': 'main' if p['recorded_status'] == 'UNKNOWN' else 'control',
                      'standard': STANDARD, 'large': LARGE, 'fast': args.fast}
                     for p in positions if p['recorded_status'] in ('UNKNOWN', 'UNSAFE')][:args.limit]
            rows = _run(tasks, _chunk_d2, args.workers, args.jsonl, 'h6-5d d2')
            summary = summarize_d2(rows)
        else:
            if factories is None and not args.value_checkpoint:
                parser.error('d3 needs --value-checkpoint')
            known = {}
            if args.d2:
                for r in json.loads(args.d2.read_text(encoding='utf-8'))['rows']:
                    known.setdefault(r['id'], {})[json.dumps(r['tree_move'])] = {
                        'standard': r['standard']['status'], 'large': r['large']['status'],
                        'standard_nodes': r['standard']['nodes'], 'large_nodes': r['large']['nodes']}
            bad = [{**p, 'kind': 'not_safe'} for p in positions if p['recorded_status'] in ('UNKNOWN', 'UNSAFE')]
            controls = [{**p, 'kind': 'control'} for p in match_controls(positions)]
            tasks = [{**p, 'standard': STANDARD, 'large': LARGE, 'known': known.get(p['id'], {}), 'fast': args.fast}
                     for p in bad + controls][:args.limit]
            init = (args.policy_checkpoint, args.value_checkpoint, args.value_device, factories)
            rows = _run(tasks, _chunk_d3, args.workers, args.jsonl, 'h6-5d d3', init)
            summary = summarize_d3(rows)
        from scripts.run_mcts_v8_benchmark import _fast_provenance
        result = {'format': FORMAT, 'step': args.step, 'standard': STANDARD, 'large': LARGE, 'sources': sources,
                  'fast': _fast_provenance(args.fast),
                  'summary': summary, 'rows': rows}
    text = json.dumps(result, indent=1) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding='utf-8')
    print(json.dumps(result.get('summary', result.get('verdict')), indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
