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
from contextlib import contextmanager
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
         'deadline_seconds': DEADLINE,
         # SER5 deployment certification of an engine (docs/h6-value.md): two independent rounds on the 827
         # corpus, CPU, threads 1, workers 1, fast4; both rounds need no move over 30 s and a maximum <= 25 s
         'ser5': {'over_deadline': 0, 'max_seconds': 25.0}}
ENGINES = ('tb-v2-value', 'tb-v3')
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

@contextmanager
def restore_fast_mode():
    """Put this process back on the fast mode it had before (the runs here rebind names process-wide)."""
    from hybrid.h6_fast import install, installed_mode

    previous = installed_mode()
    try:
        yield
    finally:
        from rt30 import fast5
        fast5.uninstall()  # fast5 (X2-F) rebinds the VCF searches on top of fast4: take that back too
        install(previous)


def _init(policy_checkpoint, value_checkpoint, device, threads, fast, factories=None, runtime='fast4'):
    if fast != 'off':
        from hybrid.h6_fast import install
        install(fast)
    if runtime == 'fast5':  # TB-V3 on fast5: fast4 plus the native VCF core
        from analysis import tb_v3_fast5
        tb_v3_fast5.install()
    if factories is not None:  # tests: torch-free stand-ins
        _WORKER.update(factories)
        return
    from hybrid.h4_policy import RootPolicy
    from hybrid.h6_value_leaf import ValueLeaf
    _WORKER['policy'] = RootPolicy(policy_checkpoint)
    _WORKER['leaf'] = ValueLeaf(value_checkpoint, device=device, threads=threads)
    _WORKER.update(policy_checkpoint=policy_checkpoint, value_checkpoint=value_checkpoint)


def _tb_v3_agent(seed: int, runtime: str = 'fast4'):
    """TB-V3 Resolve by its fail-closed factory on ``runtime`` (tests: the engine after the checks, on stand-in
    models)."""
    from analysis import tb_v3_fast5, tb_v3_runtime

    w = _WORKER
    if w.get('stand_in'):
        if runtime == 'fast5':
            tb_v3_fast5.check_bound()
        return tb_v3_runtime._build(w['policy'], w['leaf'], seed)
    make = tb_v3_fast5.make_tb_v3_fast5_agent if runtime == 'fast5' else tb_v3_runtime.make_tb_v3_agent
    return make(w['policy'], w['leaf'], seed, w['policy_checkpoint'], w['value_checkpoint'])


def decide(moves, seed: int, engine: str = 'tb-v2-value', runtime: str = 'fast4') -> dict:
    """The frozen TB-V2-Value (or TB-V3 Resolve) on one position; every decision field and the times.

    TB-V3 gets a fresh agent per position, so its cooldown never skips an attack here: the worst case for time."""
    from analysis import tb_v2_value
    from renju import Game

    game = Game()
    for move in moves:
        game.play(*move)
    leaf = _WORKER['leaf']
    if hasattr(leaf, '_cache'):
        leaf._cache.clear()  # cold per position: the time does not depend on the order of positions
    if engine == 'tb-v3':
        agent = _tb_v3_agent(seed, runtime)  # built (and checked) before the clock starts
    else:
        agent = tb_v2_value.make_agent(_WORKER['policy'], leaf, seed=seed)
    started = perf_counter()
    move = agent.select_move(game)
    seconds = perf_counter() - started
    d = agent.diagnostics
    one = (lambda m: list(m) if m is not None else None)
    step = {k: v for k, v in getattr(agent, 'attack_info', {}).items() if k != 'seconds'}
    return {**({'e4_attack': step, 'e4_seconds': getattr(agent, 'attack_info', {}).get('seconds')}
               if engine == 'tb-v3' else {}),
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

    engine, runtime = task.get('engine', 'tb-v2-value'), task.get('runtime', 'fast4')
    return {'id': task['id'], 'source': task['source'], **({'engine': engine} if engine != 'tb-v2-value' else {}),
            **({'runtime': runtime} if runtime != 'fast4' else {}),
            **decide(task['moves'], derive_seed(SEED, 'h7-timing', task['id']), engine, runtime)}


def ser5_check(summary: dict) -> dict:
    """One round against the SER5 rule (both rounds must pass)."""
    rule, a = RULES['ser5'], summary['all']
    return {'over_deadline': a['over_deadline'], 'max': a['max'],
            'pass': a['over_deadline'] <= rule['over_deadline'] and a['max'] <= rule['max_seconds']}


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


def _top(visits, k=5):
    return sorted(visits, key=lambda x: (-x[1], -x[2]))[:k]


def explain(base: dict, cand: dict, ids: list[str]) -> dict:
    """Side by side, per position: which decision fields differ, the root's top children (visits, Q), the
    top1-top2 gaps and the checked statuses. Report only (the cause of a ``compare`` difference)."""
    b, c = {r['id']: r for r in base['rows']}, {r['id']: r for r in cand['rows']}
    out = {}
    for i in ids:
        if i not in b or i not in c:
            out[i] = {'missing': True}
            continue
        db, dc = b[i]['decision'], c[i]['decision']
        side = {}
        for name, d in (('baseline', db), ('candidate', dc)):
            top = _top(d['root_visits'])
            gap = ({'visits': top[0][1] - top[1][1], 'q': round(top[0][2] - top[1][2], 9)} if len(top) >= 2 else None)
            side[name] = {'move': d['move'], 'route': d['route'], 'tree_move': d['tree_move'], 'changed': d['changed'],
                          'root_switch': d['root_switch'], 'top_children': top, 'top1_top2_gap': gap,
                          'root_checked': d['root_checked'], 'vct2_checked': d['vct2_checked'],
                          'root_nodes': d['root_nodes'], 'root_calls': d['root_calls'], 'value_calls': d['value_calls']}
        same_children = {tuple(m) for m, _, _ in db['root_visits']} == {tuple(m) for m, _, _ in dc['root_visits']}
        qb = {tuple(m): q for m, _, q in db['root_visits']}
        qd = [abs(qb[tuple(m)] - q) for m, _, q in dc['root_visits'] if tuple(m) in qb]
        out[i] = {'differing_fields': sorted(k for k in db if db[k] != dc[k]), 'same_root_children': same_children,
                  'max_abs_q_diff_common_children': round(max(qd), 9) if qd else None,
                  'tree_move_same': db['tree_move'] == dc['tree_move'], **side}
    return {'format': FORMAT, 'step': 'explain', 'positions': out}


# ---------- paired (P2 speed and semantics in one process) ----------

PAIRED_RULES = {'orders': {'even': 'ABBA', 'odd': 'BAAB'}, 'order_key': 'first byte of sha256(position id)',
                'statistic': 'median over positions of mean(candidate total move s) / mean(baseline total move s)',
                'speed_pass': 'bootstrap 95% upper < 1.0', 'bootstrap': 10_000, 'seed': 0,
                'semantics_pass': 'every run of every position has the same decision fields', 'rounds': 2}


def order_of(position_id: str) -> str:
    import hashlib

    return PAIRED_RULES['orders']['even' if hashlib.sha256(position_id.encode()).digest()[0] % 2 == 0 else 'odd']


def paired_task(task: dict, modes: dict) -> dict:
    """Four runs of one position, ``A`` = baseline mode and ``B`` = candidate mode, in the position's order;
    each run starts from a fresh agent, the same seed and a cold value cache (``decide``)."""
    from hybrid.h6_fast import install
    from scripts.run_mcts_v8_benchmark import derive_seed

    seed = derive_seed(SEED, 'h7-timing', task['id'])
    order = order_of(task['id'])
    runs = {'A': [], 'B': []}
    decisions = []
    with restore_fast_mode():
        for slot in order:
            install(modes[slot])
            out = decide(task['moves'], seed)
            runs[slot].append({'seconds': out['seconds'], 'root': out['times']['root']})
            decisions.append(out['decision'])
    mean = (lambda xs: sum(xs) / len(xs))
    a, b = mean([r['seconds'] for r in runs['A']]), mean([r['seconds'] for r in runs['B']])
    ar, br = mean([r['root'] for r in runs['A']]), mean([r['root'] for r in runs['B']])
    return {'id': task['id'], 'source': task['source'], 'order': order, 'runs': runs, 'baseline_seconds': round(a, 4),
            'candidate_seconds': round(b, 4), 'ratio': round(b / a, 6) if a else None,
            'root_ratio': round(br / ar, 6) if ar else None,
            'identical': all(d == decisions[0] for d in decisions), 'decision': decisions[0]}


def _median_ci(values: list[float]) -> dict:
    from statistics import median

    if not values:
        return {'n': 0}
    rng, b = Random(PAIRED_RULES['seed']), PAIRED_RULES['bootstrap']
    draws = sorted(median(rng.choice(values) for _ in values) for _ in range(b))
    return {'n': len(values), 'median': round(median(values), 4),
            'ci95': [round(draws[int(0.025 * b)], 4), round(draws[int(0.975 * b) - 1], 4)]}


def summarize_paired(rows: list[dict]) -> dict:
    total = _median_ci([r['ratio'] for r in rows if r['ratio'] is not None])
    root = _median_ci([r['root_ratio'] for r in rows if r['root_ratio'] is not None])
    different = [r['id'] for r in rows if not r['identical']]
    speed = 'PASS' if total.get('ci95') and total['ci95'][1] < 1.0 else 'FAIL'
    return {'positions': len(rows), 'semantics': 'PASS' if not different else 'FAIL', 'differences': len(different),
            'first_differences': different[:20], 'total_ratio': total, 'root_ratio_secondary': root, 'speed': speed,
            'times': {'baseline': percentiles([r['baseline_seconds'] for r in rows]),
                      'candidate': percentiles([r['candidate_seconds'] for r in rows]),
                      'sum_ratio': round(sum(r['candidate_seconds'] for r in rows) /
                                         sum(r['baseline_seconds'] for r in rows), 4) if rows else None},
            'orders': {o: sum(r['order'] == o for r in rows) for o in ('ABBA', 'BAAB')}}


def judge_rounds(rounds: list[dict]) -> dict:
    """P2: semantics and speed PASS in every one of the (two) independent rounds, same engine and modes."""
    keys = {(r['engine'], json.dumps(r['modes'], sort_keys=True), r['settings']['device']) for r in rounds}
    if len(rounds) != PAIRED_RULES['rounds'] or len(keys) != 1 or len({r['round'] for r in rounds}) != len(rounds):
        return {'verdict': 'INCOMPLETE', 'reason': 'two rounds (distinct labels) of the same engine, modes and device'}
    s = [r['summary'] for r in rounds]
    semantics = all(x['semantics'] == 'PASS' for x in s)
    speed = all(x['speed'] == 'PASS' for x in s)
    verdict = 'PASS' if semantics and speed else 'SEMANTICS_FAIL' if not semantics else 'SPEED_FAIL'
    return {'verdict': verdict, 'semantics': semantics, 'speed': speed,
            'rounds': {r['round']: {'semantics': x['semantics'], 'speed': x['speed'], 'total_ratio': x['total_ratio'],
                                    'root_ratio_secondary': x['root_ratio_secondary']} for r, x in zip(rounds, s)}}


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
    pm.add_argument('--fast', choices=('off', 'fast2', 'fast3', 'fast4'), default='off')
    pm.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    pm.add_argument('--value-checkpoint', default=str(ROOT / 'runs/h6_value_head_v3/best.pt'))
    pm.add_argument('--label', default='frozen', help='what runs: frozen, or the optimization candidate name')
    pm.add_argument('--engine', choices=ENGINES, default='tb-v2-value', help='tb-v3: TB-V3 Resolve (fast4 only)')
    pm.add_argument('--runtime', choices=('fast4', 'fast5'), default='fast4',
                    help='TB-V3 only: fast5 = fast4 + the native VCF core (analysis.tb_v3_fast5)')
    pm.add_argument('--jsonl', type=Path)
    pm.add_argument('--limit', type=int)
    pc = sub.add_parser('corpus')
    pc.add_argument('--measure', type=Path, required=True)
    pq = sub.add_parser('compare')
    pq.add_argument('--mode', choices=('search', 'value'), required=True)
    pq.add_argument('--corpus', type=Path, required=True)
    pq.add_argument('--baseline', type=Path, required=True)
    pq.add_argument('--candidate', type=Path, required=True)
    pe = sub.add_parser('explain')
    pe.add_argument('--baseline', type=Path, required=True)
    pe.add_argument('--candidate', type=Path, required=True)
    pe.add_argument('--compare', type=Path, required=True, help='a compare output: its first_differences are explained')
    pr = sub.add_parser('paired')
    pr.add_argument('--positions', type=Path, required=True)
    pr.add_argument('--corpus', type=Path, required=True)
    pr.add_argument('--baseline-mode', choices=('off', 'fast2', 'fast3', 'fast4'), required=True)
    pr.add_argument('--candidate-mode', choices=('off', 'fast2', 'fast3', 'fast4'), required=True)
    pr.add_argument('--round', required=True, help='label of this independent round (1, 2)')
    pr.add_argument('--device', default='cpu')
    pr.add_argument('--threads', type=int, default=1)
    pr.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    pr.add_argument('--value-checkpoint', default=str(ROOT / 'runs/h6_value_head_v3/best.pt'))
    pr.add_argument('--jsonl', type=Path)
    pr.add_argument('--limit', type=int)
    pj = sub.add_parser('judge')
    pj.add_argument('--rounds', nargs='+', type=Path, required=True)
    for p in (pp, pm, pc, pq, pe, pr, pj):
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
        if args.runtime == 'fast5' and args.engine != 'tb-v3':
            parser.error('--runtime fast5 is a TB-V3 runtime (--engine tb-v3)')
        if args.engine == 'tb-v3':
            if args.fast != 'fast4':
                parser.error('TB-V3 Resolve runs only on fast4 (fast5 is --runtime fast5 on top of it)')
            tasks = [{**t, 'engine': 'tb-v3', **({'runtime': 'fast5'} if args.runtime == 'fast5' else {})}
                     for t in tasks]
        init = (args.policy_checkpoint, args.value_checkpoint, args.device, args.threads, args.fast, factories,
                args.runtime)
        fp = measure_fingerprint(args, factories)
        try:
            rows = _run_measure(tasks, args.workers, args.jsonl, init, fp)
        except ValueError as exc:
            parser.error(str(exc))
        identity = {}
        if args.engine == 'tb-v3':
            from analysis import tb_v3, tb_v3_fast5, tb_v3_runtime
            runtime = tb_v3_fast5 if args.runtime == 'fast5' else tb_v3_runtime
            identity = {'engine_name': tb_v3.ENGINE_NAME, 'runtime_mode': args.runtime,
                        'runtime_hash': runtime.runtime_hash(args.device),
                        'identity_sha256': tb_v3_runtime.identity_sha256()}
        summary = summarize_measure(rows)
        from scripts.run_mcts_v8_benchmark import _git_commit, _git_dirty
        result = {'format': FORMAT, 'step': 'measure', 'git_commit': _git_commit(), 'git_dirty': _git_dirty(),
                  'fingerprint': fp, 'positions_sha256': fp['positions_sha256'], 'corpus_sha256': fp['corpus_sha256'],
                  'engine': tb_v3.ENGINE_HASH if args.engine == 'tb-v3' else config_hash(), **identity,
                  'label': args.label, 'ser5_check': ser5_check(summary),
                  'settings': {'device': args.device, 'threads': args.threads, 'workers': args.workers,
                               'fast': _fast_provenance(args.fast) if factories is None else args.fast,
                               'machine': machine()},
                  'summary': summary, 'rows': rows}
    elif args.step == 'corpus':
        result = build_corpus(load(args.measure))
    elif args.step == 'paired':
        result = _paired(args, load, factories)
    elif args.step == 'judge':
        result = judge_rounds([load(p) for p in args.rounds])
    elif args.step == 'explain':
        result = explain(load(args.baseline), load(args.candidate), load(args.compare)['first_differences'])
    else:
        result = compare(load(args.corpus), load(args.baseline), load(args.candidate), args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    shown = {k: v for k, v in result.items() if k not in ('positions', 'rows', 'ids')}
    if args.step == 'explain':
        shown = result['positions']
    elif args.step == 'paired':
        shown = result['summary']
    print(json.dumps(shown, indent=1)[:12000], flush=True)
    if args.step == 'paired':
        return 0 if result['summary']['semantics'] == 'PASS' else 1
    return 0 if result.get('verdict', 'PASS') not in ('FAIL', 'SEMANTICS_FAIL', 'SPEED_FAIL') else 1


def _paired(args, load, factories) -> dict:
    """One independent round: every corpus position, both modes, in one process (no pool: timing)."""
    from analysis.tb_v2_value import config_hash
    from hybrid.progress import Progress
    from scripts.run_mcts_v8_benchmark import _fast_provenance

    keep = set(load(args.corpus)['ids'])
    tasks = [t for t in load(args.positions)['positions'] if t['id'] in keep][:args.limit]
    modes = {'A': args.baseline_mode, 'B': args.candidate_mode}
    with restore_fast_mode():
        _init(args.policy_checkpoint, args.value_checkpoint, args.device, args.threads, modes['A'], factories)
        rows = _paired_rows(args, tasks, modes)
    fast = ({m: _fast_provenance(m) for m in set(modes.values())} if factories is None else dict(modes))
    return {'format': FORMAT, 'step': 'paired', 'round': args.round, 'rules': PAIRED_RULES, 'engine': config_hash(),
            'modes': modes, 'settings': {'device': args.device, 'threads': args.threads, 'fast': fast,
                                         'machine': machine()},
            'summary': summarize_paired(rows), 'rows': rows}


def _paired_rows(args, tasks, modes) -> list[dict]:
    from hybrid.progress import Progress

    done = {}
    if args.jsonl is not None and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                done[row['id']] = row
    todo = [t for t in tasks if t['id'] not in done]
    print(f"[h7-paired round {args.round}] {len(tasks)} positions, {len(tasks) - len(todo)} already done, "
          f"A={modes['A']} B={modes['B']}", flush=True)
    progress = Progress(f'h7-paired-{args.round}', len(todo), 'positions')
    for task in todo:
        row = paired_task(task, modes)
        done[row['id']] = row
        if args.jsonl is not None:
            args.jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.jsonl.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')
        progress.update(1)
    return [done[t['id']] for t in tasks]


def measure_fingerprint(args, factories) -> dict:
    """What a measured row depends on: a resumed ``--jsonl`` row is kept only under the same fingerprint."""
    from scripts.run_mcts_v8_benchmark import file_sha256

    if args.engine == 'tb-v3':
        from analysis import tb_v3, tb_v3_fast5, tb_v3_runtime
        engine, identity = tb_v3.ENGINE_HASH, tb_v3_runtime.identity_sha256()
        runtime = (tb_v3_fast5 if args.runtime == 'fast5' else tb_v3_runtime).runtime_hash(args.device)
    else:
        from analysis import tb_v2_runtime
        from analysis.tb_v2_value import config_hash
        engine, identity = config_hash(), None
        runtime = tb_v2_runtime.runtime_hash(args.device) if args.fast == 'fast4' else f'tb-v2-value/{args.fast}'
    if factories is None:
        policy, value = Path(args.policy_checkpoint), Path(args.value_checkpoint)
        models = {'policy': file_sha256(policy), 'policy_metadata': file_sha256(policy.with_suffix('.json')),
                  'value': file_sha256(value), 'value_metadata': file_sha256(value.with_name('best.json'))}
    else:
        models = 'stand-in'
    return {'format': FORMAT, 'tool_sha256': file_sha256(Path(__file__)), 'engine': engine, 'identity_sha256': identity,
            'runtime_mode': args.runtime, 'runtime_hash': runtime, 'device': args.device, 'threads': args.threads,
            'workers': args.workers, 'fast': args.fast, 'positions_sha256': file_sha256(args.positions),
            'corpus_sha256': file_sha256(args.corpus) if args.corpus else None, 'models': models}


def _run_measure(tasks, workers, jsonl, init, fingerprint=None) -> list[dict]:
    """Per-position rows, resumable from ``jsonl`` (only rows of the same ``fingerprint``); progress every minute
    (hybrid.progress)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from hybrid.progress import Progress, run_chunks

    done = {}
    if jsonl is not None and jsonl.exists():
        for line in jsonl.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line: that position is measured again
            if row.get('fingerprint') != fingerprint:
                raise ValueError(f'{jsonl} has rows of another engine / runtime / setting / input; use another file')
            done[row['id']] = row

    def kind(r):
        return r.get('engine', 'tb-v2-value'), r.get('runtime', 'fast4')

    for t in tasks:
        if t['id'] in done and kind(done[t['id']]) != kind(t):
            raise ValueError(f'{jsonl} has rows of another engine / runtime; use another file')
    todo = [t for t in tasks if t['id'] not in done]
    print(f'[h7-timing] {len(tasks)} positions, {len(tasks) - len(todo)} already done, workers {workers}', flush=True)

    def keep(_, rows):
        for row in rows:
            done[row['id']] = row
            if jsonl is not None:
                jsonl.parent.mkdir(parents=True, exist_ok=True)
                with jsonl.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps({**row, 'fingerprint': fingerprint}) + '\n')

    progress = Progress('h7-timing', len(todo), 'positions')
    chunks = [[t] for t in todo]
    if workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=_init,
                                 initargs=init) as pool:
            run_chunks(pool, _chunk, chunks, progress, len, unit_name='positions', on_result=keep)
    else:
        with restore_fast_mode():  # one process: do not leave the measured mode installed
            _init(*init)
            for i, c in enumerate(chunks):
                keep(i, _chunk(c))
                progress.update(1)
    return [{k: v for k, v in done[t['id']].items() if k != 'fingerprint'} for t in tasks]


if __name__ == '__main__':
    raise SystemExit(main())
