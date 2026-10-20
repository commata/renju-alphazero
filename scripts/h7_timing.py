"""TB-V2-Value move times and the same-move equivalence gate for runtime optimizations (docs/h6-value.md).

Four steps, the corpus rules fixed before any measurement:
    positions  the base set: every TB-V2-Value move of the E3 A runs (``--runs``, the candidate's own moves),
               the web probes P92 / P93 / P94 and the 298 H6-4b positions (E1-holdout games, both committed)
    measure    play the frozen TB-V2-Value (``analysis.tb_v2_value``) once on each position, fresh agent,
               seed fixed per position, value cache cleared per position (cold): the move, the decision
               fields and the times. ``--device`` / ``--threads`` / ``--fast`` and the machine are recorded.
               Latency runs use ``--workers 1``; more workers are for decision-only runs.
    corpus     from a baseline ``measure``: the equivalence corpus (``RULES['corpus']``), every position of
               a boundary group plus a fixed random sample of the rest
    compare    a baseline and a candidate ``measure`` on the corpus:
                 search  (search / tactical layer changes) every decision field identical: move, route,
                         tree move, root visits, V8-C and VCT2 statuses, VCF / V8 nodes and calls
                 value   (value inference backend, FP32 only) the final move identical
               PASS only with no difference at all on the corpus; times are reported (percentiles, > 30 s)

    python scripts/h7_timing.py positions --runs runs/h7_e3/value_vs_tb2_8407.json runs/h7_e3/value_vs_tb2_8408.json \\
        --output docs/h7/h7_timing_positions.json
    python -u scripts/h7_timing.py measure --positions docs/h7/h7_timing_positions.json --device cpu --threads 1 \\
        --workers 1 --fast fast3 --jsonl runs/h7_timing/base_cpu.jsonl --output runs/h7_timing/base_cpu.json
    python scripts/h7_timing.py corpus --measure runs/h7_timing/base_cpu.json --output docs/h7/h7_equiv_corpus.json
    python scripts/h7_timing.py compare --mode value --corpus docs/h7/h7_equiv_corpus.json \\
        --baseline runs/h7_timing/base_cpu.json --candidate runs/h7_timing/cand.json --output docs/h7/h7_equiv_X.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FORMAT = 'h7-timing-v1'
CANDIDATE_ARM = 'puct_value_vct2_50k_tt'
PROBES = ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json'
H6_4B = ROOT / 'docs' / 'h6' / 'h6_4b_check_cpu.json'
E1H_RUNS = [ROOT / 'docs' / 'mcts-v8-results' / 'e1h' / f'puct_policy_vct2_50k_{s}.json' for s in (8417, 8418)]
SEED = 8707  # agent seeds derive from it and the position id (not a game seed; nothing is trained on these)
DEADLINE = 30.0
RULES = {'corpus': {'near_tie_visit_ratio': 0.9, 'near_tie_q': 0.02, 'slow_top_share': 0.05,
                    'random_rest': 200, 'random_seed': 0,
                    'groups': ['near_tie', 'v8c_not_safe_or_switch', 'vct2_switch', 'probe', 'h6_4b', 'slow']},
         'deadline_seconds': DEADLINE}
_WORKER: dict = {}


# ---------- positions ----------

def build_positions(runs: list[dict]) -> list[dict]:
    out = []
    for run in runs:
        if run['arm'] != CANDIDATE_ARM:
            raise ValueError(f"expected {CANDIDATE_ARM} runs, got {run['arm']}")
        for g in sorted(run['games'], key=lambda g: g['key']):
            for m in g['v8_moves']:
                out.append({'id': f"e3a|{g['key']}|{m['ply']}", 'source': 'e3a', 'moves': g['moves'][:m['ply']],
                            'recorded': {'route': m['route'], 'seconds': m['seconds']}})
    probes = json.loads(PROBES.read_text(encoding='utf-8'))
    for name, spec in sorted(probes['probes'].items()):
        out.append({'id': f'probe|{name}', 'source': 'probe', 'moves': probes['moves'][:spec['plies_played']]})
    games = {g['key']: g for p in E1H_RUNS for g in json.loads(p.read_text(encoding='utf-8'))['games']}
    for p in json.loads(H6_4B.read_text(encoding='utf-8'))['positions']:
        out.append({'id': f"h6_4b|{p['game_key']}|{p['ply']}", 'source': 'h6_4b',
                    'moves': games[p['game_key']]['moves'][:p['ply']]})
    ids = [p['id'] for p in out]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate position ids')
    return out


# ---------- measure ----------

def _init(policy_checkpoint, value_checkpoint, device, threads, fast, factories=None):
    if fast != 'off':
        from hybrid.h6_fast import install
        install(fast)
    if factories is not None:  # tests: torch-free stand-ins
        _WORKER.update(factories)
        return
    from hybrid.h4_policy import RootPolicy
    from hybrid.h6_value_leaf import ValueLeaf
    _WORKER['policy'] = RootPolicy(policy_checkpoint)
    _WORKER['leaf'] = ValueLeaf(value_checkpoint, device=device, threads=threads)


def decide(moves, seed: int) -> dict:
    """The frozen TB-V2-Value on one position; every decision field and the times."""
    from analysis import tb_v2_value
    from renju import Game

    game = Game()
    for move in moves:
        game.play(*move)
    leaf = _WORKER['leaf']
    if hasattr(leaf, '_cache'):
        leaf._cache.clear()  # cold per position: the time does not depend on the order of positions
    agent = tb_v2_value.make_agent(_WORKER['policy'], leaf, seed=seed)
    started = perf_counter()
    move = agent.select_move(game)
    seconds = perf_counter() - started
    d = agent.diagnostics
    one = (lambda m: list(m) if m is not None else None)
    return {
        'move': list(move), 'seconds': round(seconds, 4),
        'decision': {'move': list(move), 'route': d.v8_route, 'tree_move': one(d.v8_v7_move), 'changed': d.v8_changed,
                     'root_visits': [[list(m), n, round(q, 9)] for m, n, q in d.v8_root_visits],
                     'root_checked': [[list(m), s] for m, s in d.v8_root_checked], 'root_switch': d.v8_root_switch,
                     'root_nodes': d.v8_root_nodes, 'root_calls': d.v8_root_calls,
                     'vct_checked': [[list(m), s] for m, s in d.v8_vct_checked], 'vct_nodes': d.v8_vct_nodes,
                     'vct_calls': d.v8_vct_calls, 'attack_status': d.v8_attack_status,
                     'attack_checked': [[list(m), s] for m, s in d.v8_attack_checked],
                     'attack_nodes': d.v8_attack_nodes, 'attack_calls': d.v8_attack_calls,
                     'own_vcf_nodes': d.v7_own_vcf_nodes,
                     'vct2_checked': [[list(m), s] for m, s in d.v8_vct2_checked], 'vct2_switched': d.v8_vct2_switched,
                     'vct2_nodes': d.v8_vct2_nodes, 'value_calls': d.v8_value_calls},
        'times': {'tree': round(d.v8_tree_seconds, 4), 'root': round(d.v8_root_seconds, 4),
                  'vct2': round(d.v8_vct2_seconds, 4), 'value': round(d.v8_value_seconds, 4)}}


def measure_task(task: dict) -> dict:
    from scripts.run_mcts_v8_benchmark import derive_seed

    return {'id': task['id'], 'source': task['source'], **decide(task['moves'], derive_seed(SEED, 'h7-timing', task['id']))}


def _chunk(tasks):
    return [measure_task(t) for t in tasks]


def percentiles(values: list[float]) -> dict | None:
    if not values:
        return None
    v = sorted(values)

    def at(q):
        return round(v[min(len(v) - 1, int(q * len(v)))], 3)

    return {'n': len(v), 'mean': round(sum(v) / len(v), 3), 'p50': at(0.50), 'p90': at(0.90), 'p95': at(0.95),
            'p99': at(0.99), 'max': round(v[-1], 3), 'over_deadline': sum(x > DEADLINE for x in v)}


def summarize_measure(rows: list[dict]) -> dict:
    out = {'all': percentiles([r['seconds'] for r in rows])}
    for source in sorted({r['source'] for r in rows}):
        out[source] = percentiles([r['seconds'] for r in rows if r['source'] == source])
    tree = [r for r in rows if r['decision']['route'] == 'tree']
    out['tree_route'] = percentiles([r['seconds'] for r in tree])
    out['tree_route_parts'] = {k: percentiles([r['times'][k] for r in tree]) for k in ('tree', 'root', 'vct2', 'value')}
    return out


def machine() -> dict:
    model = platform.processor()
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                model = line.split(':', 1)[1].strip()
                break
    except OSError:
        pass
    return {'platform': platform.platform(), 'cpu': model, 'python': platform.python_version()}


# ---------- corpus ----------

def groups_of(row: dict, slow_cut: float) -> list[str]:
    d, rule = row['decision'], RULES['corpus']
    out = []
    if d['route'] == 'tree' and len(d['root_visits']) >= 2:
        (_, v1, q1), (_, v2, q2) = sorted(d['root_visits'], key=lambda x: (-x[1], -x[2]))[:2]
        if (v1 and v2 / v1 >= rule['near_tie_visit_ratio']) or abs(q1 - q2) <= rule['near_tie_q']:
            out.append('near_tie')
    tree_status = {tuple(m): s for m, s in d['root_checked']}.get(tuple(d['tree_move'] or ()), None)
    if d['route'] == 'tree' and (tree_status in ('UNKNOWN', 'UNSAFE') or d['root_switch'] or d['changed']):
        out.append('v8c_not_safe_or_switch')
    if d['vct2_switched']:
        out.append('vct2_switch')
    if row['source'] in ('probe', 'h6_4b'):
        out.append(row['source'])
    if row['seconds'] >= slow_cut:
        out.append('slow')
    return out


def build_corpus(measure: dict) -> dict:
    rows = measure['rows']
    rule = RULES['corpus']
    times = sorted(r['seconds'] for r in rows)
    slow_cut = times[int((1 - rule['slow_top_share']) * (len(times) - 1))]
    picked = {}
    for r in rows:
        g = groups_of(r, slow_cut)
        if g:
            picked[r['id']] = g
    rest = sorted(r['id'] for r in rows if r['id'] not in picked and r['source'] == 'e3a')
    for i in Random(rule['random_seed']).sample(rest, min(rule['random_rest'], len(rest))):
        picked[i] = ['random']
    counts = {}
    for g in picked.values():
        for k in g:
            counts[k] = counts.get(k, 0) + 1
    return {'format': FORMAT, 'step': 'corpus', 'rules': RULES, 'baseline_sha': measure.get('engine'),
            'slow_cut_seconds': slow_cut, 'positions': len(picked), 'groups': counts,
            'ids': {k: picked[k] for k in sorted(picked)}}


# ---------- compare ----------

def compare(corpus: dict, base: dict, cand: dict, mode: str) -> dict:
    ids = sorted(corpus['ids'])
    b, c = {r['id']: r for r in base['rows']}, {r['id']: r for r in cand['rows']}
    missing = [i for i in ids if i not in b or i not in c]
    if missing:
        return {'verdict': 'INCOMPLETE', 'missing': missing[:10], 'n_missing': len(missing)}
    if base.get('engine') != cand.get('engine'):
        return {'verdict': 'INCOMPLETE', 'reason': 'the two measurements are not the same frozen engine'}
    key = (lambda r: r['decision']) if mode == 'search' else (lambda r: r['move'])
    diff = [i for i in ids if key(b[i]) != key(c[i])]
    tb, tc = [b[i]['seconds'] for i in ids], [c[i]['seconds'] for i in ids]
    return {'verdict': 'PASS' if not diff else 'FAIL', 'mode': mode, 'positions': len(ids), 'differences': len(diff),
            'first_differences': diff[:20], 'times': {'baseline': percentiles(tb), 'candidate': percentiles(tc),
                                                      'total_ratio': round(sum(tc) / sum(tb), 4) if sum(tb) else None},
            'settings': {'baseline': base.get('settings'), 'candidate': cand.get('settings')}}


def main(argv=None, factories=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='step', required=True)
    pp = sub.add_parser('positions')
    pp.add_argument('--runs', nargs='+', type=Path, required=True)
    pm = sub.add_parser('measure')
    pm.add_argument('--positions', type=Path, required=True)
    pm.add_argument('--corpus', type=Path, help='only the corpus positions (equivalence runs)')
    pm.add_argument('--device', default='cpu')
    pm.add_argument('--threads', type=int, default=1, help='torch threads per worker')
    pm.add_argument('--workers', type=int, default=1, help='1 for latency; more only for decision-only runs')
    pm.add_argument('--fast', choices=('off', 'fast2', 'fast3'), default='off')
    pm.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    pm.add_argument('--value-checkpoint', default=str(ROOT / 'runs/h6_value_head_v3/best.pt'))
    pm.add_argument('--label', default='frozen', help='what runs: frozen, or the optimization candidate name')
    pm.add_argument('--jsonl', type=Path)
    pm.add_argument('--limit', type=int)
    pc = sub.add_parser('corpus')
    pc.add_argument('--measure', type=Path, required=True)
    pq = sub.add_parser('compare')
    pq.add_argument('--mode', choices=('search', 'value'), required=True)
    pq.add_argument('--corpus', type=Path, required=True)
    pq.add_argument('--baseline', type=Path, required=True)
    pq.add_argument('--candidate', type=Path, required=True)
    for p in (pp, pm, pc, pq):
        p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    load = (lambda p: json.loads(Path(p).read_text(encoding='utf-8')))
    if args.step == 'positions':
        try:
            positions = build_positions([load(p) for p in args.runs])
        except ValueError as exc:
            parser.error(str(exc))
        result = {'format': FORMAT, 'step': 'positions', 'sources': [str(p) for p in args.runs],
                  'counts': {s: sum(p['source'] == s for p in positions) for s in ('e3a', 'probe', 'h6_4b')},
                  'positions': positions}
    elif args.step == 'measure':
        from analysis.tb_v2_value import config_hash
        from scripts.run_mcts_v8_benchmark import _fast_provenance
        tasks = load(args.positions)['positions']
        if args.corpus:
            keep = set(load(args.corpus)['ids'])
            tasks = [t for t in tasks if t['id'] in keep]
        tasks = tasks[:args.limit]
        init = (args.policy_checkpoint, args.value_checkpoint, args.device, args.threads, args.fast, factories)
        rows = _run_measure(tasks, args.workers, args.jsonl, init)
        result = {'format': FORMAT, 'step': 'measure', 'engine': config_hash(), 'label': args.label,
                  'settings': {'device': args.device, 'threads': args.threads, 'workers': args.workers,
                               'fast': _fast_provenance(args.fast) if factories is None else args.fast,
                               'machine': machine()},
                  'summary': summarize_measure(rows), 'rows': rows}
    elif args.step == 'corpus':
        result = build_corpus(load(args.measure))
    else:
        result = compare(load(args.corpus), load(args.baseline), load(args.candidate), args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    shown = {k: v for k, v in result.items() if k not in ('positions', 'rows', 'ids')}
    print(json.dumps(shown, indent=1)[:4000], flush=True)
    return 0 if result.get('verdict', 'PASS') != 'FAIL' else 1


def _run_measure(tasks, workers, jsonl, init) -> list[dict]:
    """Per-position rows, resumable from ``jsonl``; progress every minute (hybrid.progress)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from hybrid.progress import Progress, run_chunks

    done = {}
    if jsonl is not None and jsonl.exists():
        for line in jsonl.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                done[row['id']] = row
    todo = [t for t in tasks if t['id'] not in done]
    print(f'[h7-timing] {len(tasks)} positions, {len(tasks) - len(todo)} already done, workers {workers}', flush=True)

    def keep(_, rows):
        for row in rows:
            done[row['id']] = row
            if jsonl is not None:
                jsonl.parent.mkdir(parents=True, exist_ok=True)
                with jsonl.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')

    progress = Progress('h7-timing', len(todo), 'positions')
    chunks = [[t] for t in todo]
    if workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=_init,
                                 initargs=init) as pool:
            run_chunks(pool, _chunk, chunks, progress, len, unit_name='positions', on_result=keep)
    else:
        _init(*init)
        for i, c in enumerate(chunks):
            keep(i, _chunk(c))
            progress.update(1)
    return [done[t['id']] for t in tasks]


if __name__ == '__main__':
    raise SystemExit(main())
