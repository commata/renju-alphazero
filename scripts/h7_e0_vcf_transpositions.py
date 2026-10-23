"""H7-E0: how much VCF node budget goes to re-solving the same sub-problem (docs/h6-value.md, H7-E). Report only.

Measures, without changing any result, whether an exact VCF transposition table could free node budget in
the V8-C checks that H6-6 left unresolved. The frozen ``search.mcts_v7._find_vcf_recursive`` keeps no
memory of visited states; this script runs an instrumented copy of it (same moves, same order, same node
accounting) behind the V8-C solver of ``analysis.mcts_v8`` and checks, per position, that status, nodes and
VCF calls equal an uninstrumented run (``identical``).

State key: (board, attacker, remaining fours = max_fours - fours_used): the inputs the recursion depends on.
Exact board only (no D4). A state's first visit is classified when it returns:
    win       a VCF found             (reusable: the same key has the same win)
    no_win    finished without a cut  (reusable: exhaustive for that key)
    cut       stopped by the node limit (not reusable)
A later visit of a key whose first visit was reusable is a duplicate a table would answer at once; the
nodes it spent are counted as reusable, only at the outermost such duplicate (no double counting inside).
Two scopes: ``call`` (one VCF call: the E1 candidate) and ``check`` (all VCF calls of one V8-C check).
Freed nodes would be spent elsewhere, so the saving is an upper-bound style estimate, not a prediction.

Positions (H6-6 aux runs B and C, all seeds, with their d2 files): the standard not-SAFE tree moves grouped
by their 4M status (``resolved_safe``, ``unknown4m``, ``unsafe4m``, the last report only) and one matched
SAFE control each (same run and game, nearest ply). Each is checked like V8-C's first check of the tree
move (``h6_5d.STANDARD``); ``--large`` also runs the 4M check on the non-SAFE groups.

Go / no-go for E1 (call scope, standard check, fixed before this runs; the E1 adoption rule is fixed later):
    strong_go   unknown4m reusable share >= 20% and above the matched controls (paired 95% low > 0)
    weak_go     unknown4m reusable share in [10%, 20%) and above the controls
    no_go       below 10%, or not above the controls

    python -u scripts/h7_e0_vcf_transpositions.py --runs runs/h6_6/value_vs_full_840{4,5,6}.json \\
        runs/h6_6/tb1_vs_full_840{4,5,6}.json --d2 docs/h6/h6_6_d2_840{4,5,6}.json --fast fast3 --workers 12 \\
        --jsonl runs/h7/e0.jsonl --output docs/h7/h7_e0_vcf_transpositions.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import h6_5d  # noqa: E402

FORMAT = 'h7-e0-vcf-transpositions-v1'
GROUPS = ('safe_control', 'resolved_safe', 'unknown4m', 'unsafe4m')
REUSABLE = ('win', 'no_win')
RULES = {'scope': 'call', 'budget': 'standard', 'strong_go': 0.20, 'weak_go': 0.10, 'bootstrap': 10_000, 'seed': 0}


# ---------- instrumented VCF ----------

class _Scope:
    """Visit records of one scope: first-visit class per key, outermost reusable duplicate nesting."""

    def __init__(self):
        self.first: dict = {}
        self.inside = 0
        self.reusable_nodes = 0
        self.duplicates = 0
        self.duplicates_by_first = {'win': 0, 'no_win': 0, 'cut': 0}
        self.duplicate_remaining = []


class Recorder:
    def __init__(self):
        self.check = _Scope()
        self.call = None
        self.totals = {'calls': 0, 'entries': 0, 'unique_call': 0, 'nodes': 0,
                       'reusable_nodes_call': 0, 'reusable_nodes_check': 0, 'duplicates_call': 0,
                       'duplicates_check': 0}
        self.dup_by_first = {'win': 0, 'no_win': 0, 'cut': 0}
        self.dup_remaining = []

    def begin_call(self):
        self.call = _Scope()
        self.totals['calls'] += 1

    def end_call(self, nodes):
        c = self.call
        self.totals['nodes'] += nodes
        self.totals['unique_call'] += len(c.first)
        self.totals['reusable_nodes_call'] += c.reusable_nodes
        self.totals['duplicates_call'] += c.duplicates
        for k, v in c.duplicates_by_first.items():
            self.dup_by_first[k] += v
        self.dup_remaining.extend(c.duplicate_remaining)
        self.call = None

    def finish(self):
        self.totals['reusable_nodes_check'] = self.check.reusable_nodes
        self.totals['duplicates_check'] = self.check.duplicates
        self.totals['unique_check'] = len(self.check.first)


def _key(game, attacker, remaining):
    return bytes(v + 1 for row in game.board for v in row), attacker, remaining


def _instrumented_recursive(game, attacker, *, max_fours, fours_used, state, rec: Recorder):
    """``search.mcts_v7._find_vcf_recursive`` with visit records; the search itself is unchanged."""
    from search import mcts_v7 as v7

    if state.exhausted:
        return None
    rec.totals['entries'] += 1
    remaining = max_fours - fours_used
    key = _key(game, attacker, remaining)
    opened = []
    for scope in (rec.call, rec.check):
        first = scope.first.get(key)
        if first is not None:
            scope.duplicates += 1
            if scope is rec.call:
                scope.duplicates_by_first[first] += 1
                scope.duplicate_remaining.append(remaining)
            if first in REUSABLE and scope.inside == 0:
                opened.append(scope)
            if first in REUSABLE:
                scope.inside += 1
    start = state.nodes

    def body():
        immediate = sorted(v7._winning_moves(game, attacker))
        if immediate:
            return (immediate[0],), (), fours_used
        if fours_used >= max_fours:
            return None
        defender = -attacker
        for move in v7._window_candidates(game, attacker, 3):
            if not state.take_node():
                return None
            if not v7._is_legal_for_player(game, attacker, move):
                continue
            if v7._wins_for_player(game, attacker, move):
                return (move,), (), fours_used
            with v7.placed(game, attacker, move):
                completions = v7._legal_completions(game, attacker, move)
                if not completions:
                    continue
                if v7._winning_moves(game, defender):
                    continue
                if len(completions) >= 2:
                    return (move,), completions, fours_used + 1
                block = completions[0]
                if not v7._is_legal_for_player(game, defender, block):
                    return (move,), completions, fours_used + 1
                if v7._wins_for_player(game, defender, block):
                    continue
                with v7.placed(game, defender, block):
                    if v7._reverse_four_after_block(game, defender, block):
                        continue
                    child = _instrumented_recursive(game, attacker, max_fours=max_fours,
                                                    fours_used=fours_used + 1, state=state, rec=rec)
                    if child is None:
                        continue
                    attack_moves, child_completions, child_fours = child
                    return ((move,) + attack_moves,
                            tuple(sorted(set(completions).union(child_completions))), child_fours)
        return None

    result = body()
    spent = state.nodes - start
    outcome = 'win' if result is not None else 'cut' if state.exhausted else 'no_win'
    for scope in (rec.call, rec.check):
        first = scope.first.get(key)
        if first is None:
            scope.first[key] = outcome
        elif first in REUSABLE:
            scope.inside -= 1
    for scope in opened:
        scope.reusable_nodes += spent
    return result


def instrumented_find_vcf_with_stats(rec: Recorder):
    """A drop-in ``_find_vcf_with_stats`` that records into ``rec`` (same return values as the frozen one)."""
    from search import mcts_v7 as v7

    def find(game, attacker, *, max_fours, node_limit):
        if attacker not in (v7.BLACK, v7.WHITE):
            raise ValueError('attacker must be BLACK or WHITE')
        state = v7._VCFState(node_limit=node_limit)
        rec.begin_call()
        found = _instrumented_recursive(game, attacker, max_fours=max_fours, fours_used=0, state=state, rec=rec)
        rec.end_call(state.nodes)
        if found is None:
            return None, state.nodes, state.exhausted
        attack_moves, completion_points, fours = found
        return (v7.VCFResult(first_move=attack_moves[0], fours=fours, nodes=state.nodes, attack_moves=attack_moves,
                             completion_points=completion_points), state.nodes, state.exhausted)
    return find


def measure(moves, move, budget) -> dict:
    """V8-C check of ``move`` plain and instrumented; the two must agree exactly."""
    from analysis import mcts_v8

    plain = h6_5d.check_status(moves, move, budget)
    rec = Recorder()
    original = mcts_v8._find_vcf_with_stats
    mcts_v8._find_vcf_with_stats = instrumented_find_vcf_with_stats(rec)
    try:
        started = perf_counter()
        traced = h6_5d.check_status(moves, move, budget)
        seconds = perf_counter() - started
    finally:
        mcts_v8._find_vcf_with_stats = original
    rec.finish()
    keys = ('status', 'nodes', 'calls', 'budget_cut', 'cut_kind')
    t = rec.totals
    remaining = sorted(rec.dup_remaining)
    return {'status': plain['status'], 'identical': {k: plain[k] for k in keys} == {k: traced[k] for k in keys},
            'vcf_calls': t['calls'], 'nodes': t['nodes'], 'entries': t['entries'], 'unique_states': t['unique_call'],
            'unique_states_check': t['unique_check'],
            'duplicates': t['duplicates_call'], 'duplicates_check': t['duplicates_check'],
            'duplicate_rate': round(t['duplicates_call'] / t['entries'], 4) if t['entries'] else 0.0,
            'reusable_nodes': t['reusable_nodes_call'], 'reusable_nodes_check': t['reusable_nodes_check'],
            'reusable_share': round(t['reusable_nodes_call'] / t['nodes'], 4) if t['nodes'] else 0.0,
            'reusable_share_check': round(t['reusable_nodes_check'] / t['nodes'], 4) if t['nodes'] else 0.0,
            'duplicates_by_first': rec.dup_by_first,
            'duplicate_remaining': {'median': median(remaining), 'p90': remaining[int(0.9 * (len(remaining) - 1))],
                                    'max': remaining[-1]} if remaining else None,
            'seconds': round(seconds, 2)}


def e0_task(task: dict) -> dict:
    h6_5d._ensure_fast(task.get('fast', 'off'))
    out = {'id': task['id'], 'arm': task['arm'], 'game': f"{task['arm']}|{task['game_key']}", 'group': task['group'],
           'control_for': task.get('control_for'), 'standard': measure(task['moves'], task['tree_move'], h6_5d.STANDARD)}
    if task.get('large'):
        out['large'] = measure(task['moves'], task['tree_move'], h6_5d.LARGE)
    return out


def _chunk(tasks):
    return [e0_task(t) for t in tasks]


# ---------- summary ----------

def _bootstrap(groups: dict[str, list[float]]) -> dict:
    keys = sorted(k for k, v in groups.items() if v)
    if not keys:
        return {'n': 0}

    def mean_of(ks):
        vals = [x for k in ks for x in groups[k]]
        return sum(vals) / len(vals)

    rng, b = Random(RULES['seed']), RULES['bootstrap']
    draws = sorted(mean_of([rng.choice(keys) for _ in keys]) for _ in range(b))
    return {'n': sum(len(groups[k]) for k in keys), 'games': len(keys), 'mean': round(mean_of(keys), 4),
            'ci95': [round(draws[int(0.025 * b)], 4), round(draws[int(0.975 * b) - 1], 4)]}


def group_summary(rows: list[dict], budget: str) -> dict:
    rows = [r for r in rows if budget in r]
    if not rows:
        return {'positions': 0}
    m = [r[budget] for r in rows]
    nodes = sum(x['nodes'] for x in m)
    entries = sum(x['entries'] for x in m)
    by_first = {k: sum(x['duplicates_by_first'][k] for x in m) for k in ('win', 'no_win', 'cut')}
    per_game = {}
    for r in rows:
        per_game.setdefault(r['game'], []).append(r[budget]['reusable_share'])
    return {'positions': len(rows), 'vcf_calls': sum(x['vcf_calls'] for x in m), 'recursive_entries': entries,
            'unique_states': sum(x['unique_states'] for x in m), 'duplicate_entries': sum(x['duplicates'] for x in m),
            'duplicate_rate': round(sum(x['duplicates'] for x in m) / entries, 4) if entries else 0.0,
            'candidate_nodes': nodes, 'estimated_reusable_nodes': sum(x['reusable_nodes'] for x in m),
            'estimated_node_saving_rate': round(sum(x['reusable_nodes'] for x in m) / nodes, 4) if nodes else 0.0,
            'estimated_node_saving_rate_check': round(sum(x['reusable_nodes_check'] for x in m) / nodes, 4) if nodes else 0.0,
            'per_position_saving_share': _bootstrap(per_game), 'duplicate_by_first_result': by_first,
            'identical': sum(x['identical'] for x in m)}


def paired(rows: list[dict], group: str) -> dict:
    """Per-position reusable share, ``group`` position minus its matched SAFE control (standard check)."""
    by_id = {r['id']: r for r in rows}
    groups = {}
    for r in rows:
        if r['group'] == 'safe_control' and r['control_for'] in by_id and by_id[r['control_for']]['group'] == group:
            target = by_id[r['control_for']]
            groups.setdefault(target['game'], []).append(target['standard']['reusable_share'] - r['standard']['reusable_share'])
    return _bootstrap(groups)


def decide(rows: list[dict]) -> dict:
    unk = group_summary([r for r in rows if r['group'] == 'unknown4m'], 'standard')
    diff = paired(rows, 'unknown4m')
    share = unk.get('estimated_node_saving_rate', 0.0)
    above = diff.get('n', 0) > 0 and diff['ci95'][0] > 0
    if not all(r['standard']['identical'] and r.get('large', {'identical': True})['identical'] for r in rows):
        verdict = 'tool_error'
    elif share >= RULES['strong_go'] and above:
        verdict = 'strong_go'
    elif share >= RULES['weak_go'] and above:
        verdict = 'weak_go'
    else:
        verdict = 'no_go'
    return {'verdict': verdict, 'unknown4m_saving_rate': share, 'unknown4m_minus_control': diff}


def summarize(rows: list[dict]) -> dict:
    out = {'groups': {}, 'identical': all(r['standard']['identical'] and r.get('large', {'identical': True})['identical']
                                          for r in rows)}
    for arm in (h6_5d.CANDIDATE_ARM, h6_5d.BASELINE_ARM, 'pooled'):
        sub = [r for r in rows if arm in ('pooled', r['arm'])]
        out['groups'][arm] = {g: {'standard': group_summary([r for r in sub if r['group'] == g], 'standard'),
                                  'large': group_summary([r for r in sub if r['group'] == g], 'large')} for g in GROUPS}
    out['paired_vs_control'] = {g: paired(rows, g) for g in ('resolved_safe', 'unknown4m', 'unsafe4m')}
    out['decision'] = decide(rows)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--d2', nargs='+', type=Path, required=True)
    parser.add_argument('--large', action='store_true', help='also measure the 4M check on the non-SAFE groups')
    parser.add_argument('--fast', choices=('off', 'fast2', 'fast3'), default='off')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--jsonl', type=Path)
    parser.add_argument('--limit', type=int, help='first N positions (smoke runs)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    runs = [json.loads(p.read_text(encoding='utf-8')) for p in args.runs]
    for seed in sorted({r['seed'] for r in runs}):
        h6_5d.load_runs([p for p, r in zip(args.runs, runs) if r['seed'] == seed])
    large = {r['id']: r['large']['status'] for p in args.d2 for r in json.loads(p.read_text(encoding='utf-8'))['rows']}
    if args.fast != 'off':
        try:
            h6_5d._ensure_fast(args.fast)
        except RuntimeError as exc:
            parser.error(str(exc))
    positions = [p for r in runs for p in h6_5d.tree_positions(r)]
    tasks = []
    for p in positions:
        if p['recorded_status'] == 'SAFE':
            continue
        if p['id'] not in large:
            parser.error(f"no 4M recheck for {p['id']}")
        group = {'SAFE': 'resolved_safe', 'UNKNOWN': 'unknown4m', 'UNSAFE': 'unsafe4m'}[large[p['id']]]
        tasks.append({**p, 'group': group, 'large': args.large, 'fast': args.fast})
    tasks += [{**c, 'group': 'safe_control', 'large': False, 'fast': args.fast} for c in h6_5d.match_controls(positions)]
    tasks = tasks[:args.limit]
    rows = h6_5d._run(tasks, _chunk, args.workers, args.jsonl, 'h7-e0')
    summary = summarize(rows)
    from scripts.run_mcts_v8_benchmark import _fast_provenance
    result = {'format': FORMAT, 'report_only': True, 'rules': RULES, 'fast': _fast_provenance(args.fast),
              'standard': h6_5d.STANDARD, 'large': h6_5d.LARGE if args.large else None,
              'sources': {str(p): {'arm': r['arm'], 'seed': r['seed']} for p, r in zip(args.runs, runs)},
              'summary': summary, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    pooled = summary['groups']['pooled']
    print(json.dumps({'identical': summary['identical'], 'decision': summary['decision'],
                      **{g: {k: pooled[g]['standard'].get(k) for k in ('positions', 'duplicate_rate',
                                                                        'estimated_node_saving_rate',
                                                                        'estimated_node_saving_rate_check')}
                         for g in GROUPS}}, indent=1), flush=True)
    return 0 if summary['identical'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
