"""H7-E1: the call-local VCF no-win table (``analysis.vcf_tt_v1``) in the V8-C checks, offline (docs/h6-value.md).

Positions are those of H7-E0: the standard not-SAFE tree moves of H6-6 aux runs B and C (all seeds, with
their d2 files) grouped by 4M status, plus one matched SAFE control each. Every position is checked like
V8-C's first check of the tree move (``h6_5d.STANDARD``) three times:
    off     the frozen solver (must reproduce the d2 standard status)
    on      the table swapped in for the V8-C solver's VCF calls only (timed: the cost check)
    shadow  ``on`` again, with every VCF call also run frozen and compared (must equal ``on``)
``--large`` also checks the unknown4m positions with the table at 4M (``h6_5d.LARGE``), report only.

Checks (fixed before this runs):
    A  call identity: every VCF call the frozen search finishes without a cut gives the same result and
       witness with the table, in no more nodes                                        else tool_error
    B  soundness: no decided status with the table (standard or 4M) contradicts a decided off status
       (standard here, 4M from d2); every table VCF line replays (``verify_line``); for up to
       ``deep_cap`` calls per check that the frozen search cut and the table finished with no VCF, the
       frozen search at ``deep_factor`` x the call's node limit must not find one          else unsound
    C  off reproduction: off equals the d2 standard status on every not-SAFE position; the frozen files
       are untouched (``scripts/check_frozen_baseline.py``, ``tests.test_tb_engine_v1``)        else tool_error
    D  effect: share of the standard-UNKNOWN not-SAFE positions that the table decides at STANDARD
       go >= 30%, weak in [10%, 30%), no_go < 10%
    E  cost: on / off total check seconds <= 1.5 (same worker, same position)              else cost_fail
Reported, not judged: D per arm (value, baseline) and per group, off -> on status transitions, the cut
migration (off cut kind -> resolved or the on cut kind), decided -> UNKNOWN cases, table hits, and the
unknown4m positions decided by the table at STANDARD and at 4M.

    python -u scripts/h7_e1_vcf_table.py --runs runs/h6_6/value_vs_full_840{4,5,6}.json \\
        runs/h6_6/tb1_vs_full_840{4,5,6}.json --d2 docs/h6/h6_6_d2_840{4,5,6}.json --large --fast fast3 \\
        --workers 12 --jsonl runs/h7/e1.jsonl --output docs/h7/h7_e1_vcf_table.json
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import h6_5d  # noqa: E402

FORMAT = 'h7-e1-vcf-table-v1'
GROUPS = ('safe_control', 'resolved_safe', 'unknown4m', 'unsafe4m')
DECIDED = ('SAFE', 'UNSAFE')
RULES = {'go': 0.30, 'weak': 0.10, 'cost_max': 1.5, 'deep_cap': 3, 'deep_factor': 10, 'scope': 'v8c_solver',
         'table': 'call-local exact (board, attacker, remaining fours) no-win'}
CHECK_KEYS = ('status', 'nodes', 'calls', 'budget_cut', 'cut_kind')


@contextmanager
def _swapped(find):
    """Route the V8-C solver's VCF calls (``analysis.mcts_v8._find_vcf_with_stats``) through ``find``."""
    from analysis import mcts_v8

    original = mcts_v8._find_vcf_with_stats
    mcts_v8._find_vcf_with_stats = find
    try:
        yield
    finally:
        mcts_v8._find_vcf_with_stats = original


def _moves(found):
    return None if found is None else (tuple(found.attack_moves), tuple(found.completion_points), found.fours)


class Shadow:
    """Table VCF calls, each also run frozen: call identity (A), line replay and deep checks (B)."""

    def __init__(self, deep_cap=RULES['deep_cap'], deep_factor=RULES['deep_factor']):
        from analysis import vcf_tt_v1

        self.stats = vcf_tt_v1.TableStats()
        self.find_tt = vcf_tt_v1.make_find(self.stats)
        self.deep_cap, self.deep_factor = deep_cap, deep_factor
        self.counts = {'calls': 0, 'uncut_same': 0, 'uncut_mismatch': 0, 'cut_to_win': 0, 'cut_to_none': 0,
                       'cut_to_cut': 0, 'frozen_nodes': 0, 'table_nodes': 0, 'line_fail': 0,
                       'deep_confirmed': 0, 'deep_unconfirmed': 0, 'deep_contradicted': 0}
        self.examples = []

    def __call__(self, game, attacker, *, max_fours, node_limit):
        from analysis.vcf_tt_v1 import verify_line
        from search import mcts_v7 as v7

        got = self.find_tt(game, attacker, max_fours=max_fours, node_limit=node_limit)
        want = v7._find_vcf_with_stats(game, attacker, max_fours=max_fours, node_limit=node_limit)
        c = self.counts
        c['calls'] += 1
        c['frozen_nodes'] += want[1]
        c['table_nodes'] += got[1]
        if got[0] is not None and not verify_line(game, attacker, got[0].attack_moves, max_fours):
            c['line_fail'] += 1
        if not want[2]:
            if _moves(got[0]) == _moves(want[0]) and got[1] <= want[1] and not got[2]:
                c['uncut_same'] += 1
            else:
                c['uncut_mismatch'] += 1
                if len(self.examples) < 5:
                    self.examples.append({'attacker': attacker, 'node_limit': node_limit,
                                          'frozen': [_moves(want[0]), want[1], want[2]],
                                          'table': [_moves(got[0]), got[1], got[2]]})
        elif got[0] is not None:
            c['cut_to_win'] += 1
        elif got[2]:
            c['cut_to_cut'] += 1
        else:
            c['cut_to_none'] += 1
            if c['deep_confirmed'] + c['deep_unconfirmed'] + c['deep_contradicted'] < self.deep_cap:
                deep = v7._find_vcf_with_stats(game, attacker, max_fours=max_fours,
                                               node_limit=self.deep_factor * node_limit)
                c['deep_contradicted' if deep[0] is not None else 'deep_unconfirmed' if deep[2]
                  else 'deep_confirmed'] += 1
        return got


def run_check(moves, move, budget: dict, mode: str) -> dict:
    """One V8-C check: ``off`` (frozen), ``on`` (table), ``shadow`` (table, each call compared with frozen)
    or ``on_verified`` (table, each VCF line replayed: for the 4M checks)."""
    from analysis import vcf_tt_v1

    if mode == 'off':
        return h6_5d.check_status(moves, move, budget)
    if mode == 'on':
        stats = vcf_tt_v1.TableStats()
        with _swapped(vcf_tt_v1.make_find(stats)):
            out = h6_5d.check_status(moves, move, budget)
        return {**out, 'table': vars(stats)}
    if mode == 'shadow':
        shadow = Shadow()
        with _swapped(shadow):
            out = h6_5d.check_status(moves, move, budget)
        return {**out, 'table': vars(shadow.stats), 'shadow': shadow.counts, 'examples': shadow.examples}
    if mode == 'on_verified':
        stats = vcf_tt_v1.TableStats()
        find = vcf_tt_v1.make_find(stats)
        fails = [0]

        def checked(game, attacker, *, max_fours, node_limit):
            got = find(game, attacker, max_fours=max_fours, node_limit=node_limit)
            if got[0] is not None and not vcf_tt_v1.verify_line(game, attacker, got[0].attack_moves, max_fours):
                fails[0] += 1
            return got

        with _swapped(checked):
            out = h6_5d.check_status(moves, move, budget)
        return {**out, 'table': vars(stats), 'line_fail': fails[0]}
    raise ValueError(mode)


def e1_task(task: dict) -> dict:
    h6_5d._ensure_fast(task.get('fast', 'off'))
    moves, move = task['moves'], task['tree_move']
    off = run_check(moves, move, h6_5d.STANDARD, 'off')
    on = run_check(moves, move, h6_5d.STANDARD, 'on')
    shadow = run_check(moves, move, h6_5d.STANDARD, 'shadow')
    out = {'id': task['id'], 'arm': task['arm'], 'game': f"{task['arm']}|{task['game_key']}", 'group': task['group'],
           'control_for': task.get('control_for'), 'd2_standard': task.get('d2_standard'),
           'd2_large': task.get('d2_large'), 'off': off, 'on': on, 'shadow': shadow,
           'consistent': {k: on[k] for k in CHECK_KEYS} == {k: shadow[k] for k in CHECK_KEYS}}
    if task.get('large'):
        out['large_on'] = run_check(moves, move, h6_5d.LARGE, 'on_verified')
    return out


def _chunk(tasks):
    return [e1_task(t) for t in tasks]


# ---------- summary ----------

def _rate(n, d):
    return {'n': n, 'of': d, 'rate': round(n / d, 4) if d else None}


def contradictions(row: dict) -> list[str]:
    """Decided statuses that disagree: table (standard, 4M) against off standard and the d2 4M recheck."""
    out = []
    on, off, large = row['on']['status'], row['off']['status'], row.get('large_on', {}).get('status')
    for name, ours in (('standard', on), ('large', large)):
        for ref_name, ref in (('off_standard', off), ('d2_large', row.get('d2_large'))):
            if ours in DECIDED and ref in DECIDED and ours != ref:
                out.append(f'{name}:{ours} vs {ref_name}:{ref}')
    return out


def effect(rows: list[dict]) -> dict:
    """D on the not-SAFE positions whose off standard check is UNKNOWN, with the cut migration."""
    pool = [r for r in rows if r['group'] != 'safe_control' and r['off']['status'] == 'UNKNOWN']
    out = {'pooled': _rate(sum(r['on']['status'] in DECIDED for r in pool), len(pool))}
    for arm in (h6_5d.CANDIDATE_ARM, h6_5d.BASELINE_ARM):
        sub = [r for r in pool if r['arm'] == arm]
        out[arm] = _rate(sum(r['on']['status'] in DECIDED for r in sub), len(sub))
    out['by_group'] = {}
    for g in GROUPS[1:]:
        sub = [r for r in pool if r['group'] == g]
        out['by_group'][g] = _rate(sum(r['on']['status'] in DECIDED for r in sub), len(sub))
    migration = {}
    for r in pool:
        to = 'resolved' if r['on']['status'] in DECIDED else r['on']['cut_kind'] or 'none'
        k = f"{r['off']['cut_kind'] or 'none'}->{to}"
        migration[k] = migration.get(k, 0) + 1
    out['cut_migration'] = dict(sorted(migration.items()))
    return out


def summarize(rows: list[dict]) -> dict:
    shadow = {}
    for r in rows:
        for k, v in r['shadow']['shadow'].items():
            shadow[k] = shadow.get(k, 0) + v
    line_fail = shadow.get('line_fail', 0) + sum(r.get('large_on', {}).get('line_fail', 0) for r in rows)
    reproduce = [r['id'] for r in rows if r['d2_standard'] is not None and r['off']['status'] != r['d2_standard']]
    inconsistent = [r['id'] for r in rows if not r['consistent']]
    contra = {r['id']: c for r in rows if (c := contradictions(r))}
    off_s = sum(r['off']['seconds'] for r in rows)
    on_s = sum(r['on']['seconds'] for r in rows)
    cost = round(on_s / off_s, 3) if off_s else None
    transitions = {}
    for r in rows:
        k = f"{r['group']}:{r['off']['status']}->{r['on']['status']}"
        transitions[k] = transitions.get(k, 0) + 1
    unk = [r for r in rows if r['group'] == 'unknown4m']
    unk_large = [r for r in unk if 'large_on' in r]
    d = effect(rows)
    checks = {
        'A_call_identity': {'uncut_same': shadow.get('uncut_same', 0), 'uncut_mismatch': shadow.get('uncut_mismatch', 0)},
        'B_soundness': {'contradictions': contra, 'line_fail': line_fail,
                        'deep': {k: shadow.get(f'deep_{k}', 0) for k in ('confirmed', 'unconfirmed', 'contradicted')}},
        'C_off_reproduction': {'mismatches': reproduce, 'shadow_inconsistent': inconsistent},
        'D_effect': d, 'E_cost': {'off_seconds': round(off_s, 1), 'on_seconds': round(on_s, 1), 'ratio': cost}}
    if inconsistent or reproduce or shadow.get('uncut_mismatch', 0):
        verdict = 'tool_error'
    elif contra or line_fail or shadow.get('deep_contradicted', 0):
        verdict = 'unsound'
    elif cost is None or cost > RULES['cost_max']:
        verdict = 'cost_fail'
    elif (d['pooled']['rate'] or 0) >= RULES['go']:
        verdict = 'go'
    elif (d['pooled']['rate'] or 0) >= RULES['weak']:
        verdict = 'weak'
    else:
        verdict = 'no_go'
    return {'verdict': verdict, 'checks': checks, 'transitions': dict(sorted(transitions.items())),
            'decided_to_unknown': [r['id'] for r in rows
                                   if r['off']['status'] in DECIDED and r['on']['status'] == 'UNKNOWN'],
            'calls': {k: shadow.get(k, 0) for k in ('calls', 'cut_to_win', 'cut_to_none', 'cut_to_cut',
                                                     'frozen_nodes', 'table_nodes')},
            'table': {k: sum(r['on']['table'][k] for r in rows) for k in ('calls', 'stores', 'hits')},
            'unknown4m': {'positions': len(unk),
                          'standard_decided': sum(r['on']['status'] in DECIDED for r in unk),
                          'large_checked': len(unk_large),
                          'large_decided': sum(r['large_on']['status'] in DECIDED for r in unk_large),
                          'large_status': {s: sum(r['large_on']['status'] == s for r in unk_large)
                                           for s in ('SAFE', 'UNSAFE', 'UNKNOWN')}}}


def build_tasks(runs: list[dict], d2_rows: dict, large: bool, fast: str) -> list[dict]:
    positions = [p for r in runs for p in h6_5d.tree_positions(r)]
    tasks = []
    for p in positions:
        if p['recorded_status'] == 'SAFE':
            continue
        if p['id'] not in d2_rows:
            raise ValueError(f"no d2 row for {p['id']}")
        d2 = d2_rows[p['id']]
        group = {'SAFE': 'resolved_safe', 'UNKNOWN': 'unknown4m', 'UNSAFE': 'unsafe4m'}[d2['large']['status']]
        tasks.append({**p, 'group': group, 'd2_standard': d2['standard']['status'], 'd2_large': d2['large']['status'],
                      'large': large and group == 'unknown4m', 'fast': fast})
    tasks += [{**c, 'group': 'safe_control', 'd2_standard': None, 'd2_large': None, 'large': False, 'fast': fast}
              for c in h6_5d.match_controls(positions)]
    return tasks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--d2', nargs='+', type=Path, required=True)
    parser.add_argument('--large', action='store_true', help='also check the unknown4m positions with the table at 4M')
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
    d2_rows = {r['id']: r for p in args.d2 for r in json.loads(p.read_text(encoding='utf-8'))['rows']}
    if args.fast != 'off':
        try:
            h6_5d._ensure_fast(args.fast)
        except RuntimeError as exc:
            parser.error(str(exc))
    try:
        tasks = build_tasks(runs, d2_rows, args.large, args.fast)[:args.limit]
    except ValueError as exc:
        parser.error(str(exc))
    # the 4M checks take minutes each: start them first so they do not trail at the end
    order = sorted(range(len(tasks)), key=lambda i: not tasks[i]['large'])
    done = h6_5d._run([tasks[i] for i in order], _chunk, args.workers, args.jsonl, 'h7-e1')
    by_id = {r['id']: r for r in done}
    rows = [by_id[t['id']] for t in tasks]
    summary = summarize(rows)
    from scripts.run_mcts_v8_benchmark import _fast_provenance
    result = {'format': FORMAT, 'rules': RULES, 'fast': _fast_provenance(args.fast), 'standard': h6_5d.STANDARD,
              'large': h6_5d.LARGE if args.large else None,
              'sources': {str(p): {'arm': r['arm'], 'seed': r['seed']} for p, r in zip(args.runs, runs)},
              'summary': summary, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    c = summary['checks']
    print(json.dumps({'verdict': summary['verdict'], 'D': c['D_effect'], 'E_cost': c['E_cost']['ratio'],
                      'A': c['A_call_identity'], 'B_contradictions': len(c['B_soundness']['contradictions']),
                      'B_line_fail': c['B_soundness']['line_fail'], 'B_deep': c['B_soundness']['deep'],
                      'C': {k: len(v) for k, v in c['C_off_reproduction'].items()},
                      'unknown4m': summary['unknown4m']}, indent=1), flush=True)
    return 0 if summary['verdict'] not in ('tool_error', 'unsound') else 1


if __name__ == '__main__':
    raise SystemExit(main())
