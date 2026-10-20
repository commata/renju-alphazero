"""H7-P0: where V8-C's time goes in TB-V2-Value's slow root checks (docs/h6-value.md, P0). Report only.

Boundary timers, not cProfile (which inflates small functions). Each position is decided once per pass by the
frozen TB-V2-Value (``h7_timing.decide``: fresh agent, fixed seed, cold value cache); the timers are installed
around names looked up at call time, so nothing frozen is edited and the decisions must equal an untimed run.

    coarse  timers on V8-C itself (``analysis.mcts_v8._verify_root_with``) and on each VCF call of its solver
            (the ``analysis.vcf_tt_v1.make_find`` closures): the VCF-inside share of V8-C time. Two timers per
            VCF call, so the overhead is small; the P1 branch is decided on this pass.
    fine    also timers on the helpers, split by whether a VCF call is running:
              inside VCF   table key (``vcf_tt_v1._key``), five / candidate / legality / four helpers of
                           ``search.mcts_v7``; the rest of the VCF time is the recursion itself
              outside VCF  ``threats._winning_moves``, ``threats.ordered_moves`` (defence candidates),
                           ``threats._board_key``, ``Game.legal_moves`` / ``play`` / ``undo`` (Renju legality and
                           forbidden points live there); the rest is solver logic (caches, status combination)
            Self time (a timed helper's time minus the timed helpers it calls), so nothing is counted twice.
            Many small calls are timed, so the fine pass is slower; its shares are for reading, not deciding.
Counts per position: V8-C's VCF requests (solver cache hits = requests - calls), calls, nodes per call, cut
calls, table hits / stores, the tree move's status and whether the budget was exhausted.

Decision (fixed before running): aggregate VCF-inside share of V8-C time (coarse) >= 0.6 -> P1a (native VCF
core with the table), else P1b (outside VCF: forbidden points, defence candidates). Per-position median, p75
and p90 shares and the per-group values are reported for interpretation only.

    python -u scripts/h7_p0_profile.py --positions docs/h7/h7_timing_positions.json --corpus docs/h7/h7_equiv_corpus.json \\
        --baseline runs/h7_timing/base_cpu.json --device cpu --threads 1 --fast fast3 \\
        --jsonl runs/h7_timing/p0.jsonl --output docs/h7/h7_p0_profile.json
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import h7_timing  # noqa: E402

FORMAT = 'h7-p0-profile-v1'
RULES = {'groups': {'slow': None, 'v8c_not_safe_or_switch': None, 'random': 50}, 'random_seed': 0,
         'p1a_min_share': 0.6}
INSIDE = ('_winning_moves', '_window_candidates', '_is_legal_for_player', '_wins_for_player', '_legal_completions',
          '_reverse_four_after_block')
OUTSIDE = ('_winning_moves', 'ordered_moves', '_board_key')
GAME_METHODS = ('legal_moves', 'play', 'undo')


class Clock:
    """Totals for one position; ``vcf`` > 0 while a V8-C VCF call runs, ``root`` while V8-C runs."""

    def __init__(self):
        self.root = 0
        self.vcf = 0
        self.root_seconds = 0.0
        self.vcf_seconds = 0.0
        self.requests = 0
        self.calls = 0
        self.nodes = []
        self.cut = 0
        self.table = {'hits': 0, 'stores': 0}
        self.parts = {}
        self.stack = []  # child time of the timed helpers now running (self time = elapsed - children)

    def add(self, name, seconds):
        cell = self.parts.setdefault(name, [0, 0.0])
        cell[0] += 1
        cell[1] += seconds


@contextmanager
def timers(clock: Clock, fine: bool):
    from analysis import mcts_v8, threats, vcf_tt_v1
    from renju import Game
    from search import mcts_v7 as v7

    saved = []

    def swap(owner, name, new):
        saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, new)

    verify = mcts_v8._verify_root_with

    def timed_verify(*args, **kwargs):
        clock.root += 1
        started = perf_counter()
        try:
            return verify(*args, **kwargs)
        finally:
            clock.root_seconds += perf_counter() - started
            clock.root -= 1

    make_find = vcf_tt_v1.make_find

    def timed_make_find(stats=None):
        own = vcf_tt_v1.TableStats()
        find = make_find(own)

        def timed(game, attacker, *, max_fours, node_limit):
            clock.calls += 1
            clock.vcf += 1
            hits, stores = own.hits, own.stores
            started = perf_counter()
            try:
                result = find(game, attacker, max_fours=max_fours, node_limit=node_limit)
            finally:
                clock.vcf_seconds += perf_counter() - started
                clock.vcf -= 1
            clock.nodes.append(result[1])
            clock.cut += bool(result[2])
            clock.table['hits'] += own.hits - hits
            clock.table['stores'] += own.stores - stores
            if stats is not None:
                stats.calls, stats.hits, stats.stores = own.calls, own.hits, own.stores
            return result
        return timed

    solver_vcf = mcts_v8._BudgetedSolver.vcf

    def counted_vcf(self, game, attacker):
        if clock.root:
            clock.requests += 1
        return solver_vcf(self, game, attacker)

    def timer(name, fn, where):
        def wrapped(*args, **kwargs):
            if not clock.root:
                return fn(*args, **kwargs)
            clock.stack.append(0.0)
            started = perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = perf_counter() - started
                children = clock.stack.pop()
                if clock.stack:
                    clock.stack[-1] += elapsed
                inside = clock.vcf > 0
                if where == 'any' or (where == 'inside') == inside:
                    clock.add(f"{'inside' if inside else 'outside'}:{name}", elapsed - children)
        return wrapped

    swap(mcts_v8, '_verify_root_with', timed_verify)
    swap(vcf_tt_v1, 'make_find', timed_make_find)
    swap(mcts_v8._BudgetedSolver, 'vcf', counted_vcf)
    if fine:
        swap(vcf_tt_v1, '_key', timer('table_key', vcf_tt_v1._key, 'inside'))
        for name in INSIDE:
            swap(v7, name, timer(name, getattr(v7, name), 'inside'))
        for name in OUTSIDE:
            swap(threats, name, timer(name, getattr(threats, name), 'outside'))
        for name in GAME_METHODS:
            swap(Game, name, timer(f'Game.{name}', getattr(Game, name), 'any'))
    try:
        yield clock
    finally:
        for owner, name, old in reversed(saved):
            setattr(owner, name, old)


def _share(part, whole):
    return round(part / whole, 4) if whole else None


def profile(task: dict, fine: bool) -> dict:
    from scripts.run_mcts_v8_benchmark import derive_seed

    clock = Clock()
    with timers(clock, fine):
        out = h7_timing.decide(task['moves'], derive_seed(h7_timing.SEED, 'h7-timing', task['id']))
    d = out['decision']
    status = {tuple(m): s for m, s in d['root_checked']}.get(tuple(d['tree_move'] or ()))
    row = {'move': out['move'], 'decision': d, 'seconds': out['seconds'], 'root_seconds': round(clock.root_seconds, 4),
           'vcf_seconds': round(clock.vcf_seconds, 4), 'vcf_share': _share(clock.vcf_seconds, clock.root_seconds),
           'requests': clock.requests, 'calls': clock.calls, 'cache_hits': clock.requests - clock.calls,
           'nodes': sum(clock.nodes), 'max_call_nodes': max(clock.nodes, default=0), 'cut_calls': clock.cut,
           'table': clock.table, 'tree_status': status}
    if fine:
        row['parts'] = {k: [n, round(s, 4)] for k, (n, s) in sorted(clock.parts.items())}
    return row


def p0_task(task: dict) -> dict:
    coarse = profile(task, fine=False)
    fine = profile(task, fine=True)
    return {'id': task['id'], 'groups': task['groups'], 'coarse': coarse, 'fine': fine,
            'identical': coarse['decision'] == fine['decision']}


def _chunk(tasks):
    return [p0_task(t) for t in tasks]


def _quantile(values, q):
    v = sorted(values)
    return round(v[min(len(v) - 1, int(q * len(v)))], 4) if v else None


def shares(rows) -> dict:
    root = sum(r['coarse']['root_seconds'] for r in rows)
    vcf = sum(r['coarse']['vcf_seconds'] for r in rows)
    per = [r['coarse']['vcf_share'] for r in rows if r['coarse']['vcf_share'] is not None]
    return {'positions': len(rows), 'root_seconds': round(root, 2), 'vcf_seconds': round(vcf, 2),
            'aggregate_share': _share(vcf, root), 'median_share': _quantile(per, 0.5), 'p75_share': _quantile(per, 0.75),
            'p90_share': _quantile(per, 0.9)}


def fine_parts(rows) -> dict:
    total = {}
    for r in rows:
        for k, (n, s) in r['fine'].get('parts', {}).items():
            cell = total.setdefault(k, [0, 0.0])
            cell[0] += n
            cell[1] += s
    root = sum(r['fine']['root_seconds'] for r in rows)
    vcf = sum(r['fine']['vcf_seconds'] for r in rows)
    out = {k: {'calls': n, 'seconds': round(s, 2), 'share_of_v8c': _share(s, root)} for k, (n, s) in sorted(total.items())}
    inside = sum(s for k, (_, s) in total.items() if k.startswith('inside:'))
    outside = sum(s for k, (_, s) in total.items() if k.startswith('outside:'))
    out['inside:recursion_rest'] = {'seconds': round(vcf - inside, 2), 'share_of_v8c': _share(vcf - inside, root)}
    out['outside:solver_rest'] = {'seconds': round(root - vcf - outside, 2), 'share_of_v8c': _share(root - vcf - outside, root)}
    coarse_root = sum(r['coarse']['root_seconds'] for r in rows)
    out['timer_overhead_ratio'] = _share(root, coarse_root)
    return out


def summarize(rows: list[dict], baseline: dict | None) -> dict:
    groups = {g: shares([r for r in rows if g in r['groups']]) for g in RULES['groups']}
    pooled = shares(rows)
    counts = {k: sum(r['coarse'][k] for r in rows) for k in ('requests', 'calls', 'cache_hits', 'nodes', 'cut_calls')}
    counts['table_hits'] = sum(r['coarse']['table']['hits'] for r in rows)
    counts['table_stores'] = sum(r['coarse']['table']['stores'] for r in rows)
    status = {}
    for r in rows:
        status[str(r['coarse']['tree_status'])] = status.get(str(r['coarse']['tree_status']), 0) + 1
    out = {'pooled': pooled, 'groups': groups, 'counts': counts, 'tree_status': status, 'fine': fine_parts(rows),
           'timers_transparent': all(r['identical'] for r in rows)}
    if baseline is not None:
        base = {r['id']: r['decision'] for r in baseline['rows']}
        out['same_as_baseline'] = sum(base.get(r['id']) == r['coarse']['decision'] for r in rows)
    share = pooled['aggregate_share'] or 0.0
    out['decision'] = {'branch': 'P1a' if share >= RULES['p1a_min_share'] else 'P1b', 'aggregate_share': share,
                       'rule': f"aggregate VCF-inside share >= {RULES['p1a_min_share']} -> P1a, else P1b"}
    return out


def select(positions: list[dict], corpus: dict) -> list[dict]:
    by_id = {p['id']: p for p in positions}
    picked = {}
    for g in ('slow', 'v8c_not_safe_or_switch'):
        for i, groups in corpus['ids'].items():
            if g in groups:
                picked.setdefault(i, set()).add(g)
    random_ids = sorted(i for i, groups in corpus['ids'].items() if groups == ['random'])
    for i in Random(RULES['random_seed']).sample(random_ids, min(RULES['groups']['random'], len(random_ids))):
        picked.setdefault(i, set()).add('random')
    return [{**by_id[i], 'groups': sorted(g)} for i, g in sorted(picked.items())]


def main(argv=None, factories=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='an h7_timing measure of the same engine: decisions must match')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--fast', choices=('off', 'fast2', 'fast3'), default='off')
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--value-checkpoint', default=str(ROOT / 'runs/h6_value_head_v3/best.pt'))
    parser.add_argument('--jsonl', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    load = (lambda p: json.loads(Path(p).read_text(encoding='utf-8')))
    tasks = select(load(args.positions)['positions'], load(args.corpus))[:args.limit]
    h7_timing._init(args.policy_checkpoint, args.value_checkpoint, args.device, args.threads, args.fast, factories)
    from hybrid.progress import Progress
    done = {}
    if args.jsonl is not None and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                done[row['id']] = row
    todo = [t for t in tasks if t['id'] not in done]
    print(f'[h7-p0] {len(tasks)} positions, {len(tasks) - len(todo)} already done (one process: timing)', flush=True)
    progress = Progress('h7-p0', len(todo), 'positions')
    for task in todo:
        row = p0_task(task)
        done[row['id']] = row
        if args.jsonl is not None:
            args.jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.jsonl.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')
        progress.update(1)
    rows = [done[t['id']] for t in tasks]
    from analysis.tb_v2_value import config_hash
    summary = summarize(rows, load(args.baseline) if args.baseline else None)
    result = {'format': FORMAT, 'report_only': True, 'rules': RULES, 'engine': config_hash(),
              'settings': {'device': args.device, 'threads': args.threads, 'fast': args.fast,
                           'machine': h7_timing.machine()},
              'summary': summary, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=1)[:6000], flush=True)
    return 0 if summary['timers_transparent'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
