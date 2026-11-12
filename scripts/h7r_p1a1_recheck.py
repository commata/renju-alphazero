"""R2-P1-A1 Arena recheck (read only): the committed judge (docs/h7r/r2/R2-P1-POLICY-REPLAY-A1/judge.json) against
the arm records it read, and a post-hoc description of where T's O0 deficit lies (docs/h7r/r2/R2-P1-A1-RECHECK-draft.md).

    1. identity   the A1 contract (arena stage, sources and evidence checked), the committed judge record of the
                  experiment, the committed Arena contract whose SHA the judge names
    2. arm files  every arm record's SHA-256 against ``judge.json`` first (a different file stops the recheck),
                  then each loaded through ``h7r_eval.load_arm`` (its games, openings, identity and summary re-checked);
                  every summary against the judge's per-arm summary
    3. bootstrap  the opening-level and the class-level suite-stratified 50:50 paired bounds of P and D recomputed
                  with the judge's own functions (``h7r_p1.stratified_bounds``), their readings, against the judge
    4. O0 classes post hoc, descriptive only: per symmetry class the pairs, sum and mean of T - 0.5 (and C - 0.5,
                  T - C); leaving each class out, how the O0 score of T, the combined P and D move; how concentrated
                  the deficit is. No reading, threshold or verdict is derived from it.

Nothing is changed: the tool writes one new JSON (``--output``, never over a file) and exits 0 when every check
matches, 2 when a recomputed value differs from the judge (the record says which), 1 on a stop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h7r_p1 as p1  # noqa: E402
import h7r_p1a1 as a1  # noqa: E402
from h7r_p1 import require  # noqa: E402

FORMAT = 'h7r-p1a1-arena-recheck-v1'
OUTPUT = f'{a1.D}/arena-recheck.json'
ARMS_DIR = f'{a1.RUNS}/arena'


def class_analysis(rows: list, other_t_mean: float, other_d_mean: float) -> dict:
    """``rows``: (T pair score, C pair score, class id) of one suite's openings (paired). ``other_*``: the other
    suite's T mean and T - C mean, held fixed, so the combined P and D of a leave-one-class-out are the 50:50 ones."""
    n = len(rows)
    require(n > 0, 'O0 has no pairs')
    t_total = sum(r[0] for r in rows)
    d_total = sum(r[0] - r[1] for r in rows)
    t_mean, d_mean = t_total / n, d_total / n
    p_all = (t_mean + other_t_mean) / 2 - 0.5
    d_all = (d_mean + other_d_mean) / 2
    by = {}
    for t, c, k in rows:
        by.setdefault(k, []).append((t, c))
    classes = []
    require(len(by) > 1, 'O0 needs at least two classes for leave-one-class-out')
    for k in sorted(by):
        g = by[k]
        m = len(g)
        t_sum, c_sum = sum(t - 0.5 for t, _ in g), sum(c - 0.5 for _, c in g)
        rest = n - m
        t_wo = (t_total - sum(t for t, _ in g)) / rest
        d_wo = (d_total - (t_sum - c_sum)) / rest
        classes.append({'class_id': k, 'pairs': m, 'T_minus_half_sum': round(t_sum, 6),
                        'T_minus_half_mean': round(t_sum / m, 6), 'C_minus_half_mean': round(c_sum / m, 6),
                        'T_minus_C_mean': round((t_sum - c_sum) / m, 6),
                        'leave_out': {'T_score': round(t_wo, 6), 'T_score_change': round(t_wo - t_mean, 6),
                                      'P_change': round(((t_wo + other_t_mean) / 2 - 0.5) - p_all, 6),
                                      'D_change': round((d_wo + other_d_mean) / 2 - d_all, 6)}})
    deficit = sorted((c for c in classes if c['T_minus_half_sum'] < 0), key=lambda c: c['T_minus_half_sum'])
    total_neg = sum(c['T_minus_half_sum'] for c in deficit)
    total = sum(c['T_minus_half_sum'] for c in classes)

    def share(k):
        return round(sum(c['T_minus_half_sum'] for c in deficit[:k]) / total_neg, 4) if total_neg else None
    moves = sorted(classes, key=lambda c: abs(c['leave_out']['P_change']), reverse=True)
    return {
        'pairs': n, 'classes': len(classes),
        'pairs_per_class': sorted({c['pairs'] for c in classes}),
        'T_score': round(t_mean, 6), 'T_minus_half_sum': round(total, 6), 'P_combined': round(p_all, 6),
        'D_combined': round(d_all, 6),
        'class_signs': {'negative': sum(c['T_minus_half_mean'] < 0 for c in classes),
                        'zero': sum(c['T_minus_half_mean'] == 0 for c in classes),
                        'positive': sum(c['T_minus_half_mean'] > 0 for c in classes)},
        'deficit_concentration': {'negative_sum': round(total_neg, 6),
                                  'share_of_negative_sum': {f'worst_{k}': share(k) for k in (1, 3, 5, 10)},
                                  'worst_classes': [c['class_id'] for c in deficit[:10]]},
        'leave_one_class_out': {
            'P_change_range': [min(c['leave_out']['P_change'] for c in classes),
                               max(c['leave_out']['P_change'] for c in classes)],
            'D_change_range': [min(c['leave_out']['D_change'] for c in classes),
                               max(c['leave_out']['D_change'] for c in classes)],
            'largest_P_moves': [{'class_id': c['class_id'], **c['leave_out']} for c in moves[:5]]},
        'per_class': classes,
        'note': 'post-hoc description: no reading, threshold or verdict is derived from it'}


def recheck(arms_dir, root=ROOT, *, output=None, log_name=None) -> dict:
    from hybrid import h7r_eval as ev
    import h7r_eval as cli

    root = Path(root)
    start_git = p1.git_state(root)
    c, sha, eid = a1.load_contract(None, root, 'arena')
    judge_path = f'{a1.D}/judge.json'
    tool = str(Path(__file__).resolve())
    apath = root / a1.ARENA_CONTRACT
    require(p1.committed(apath, root), 'the Arena contract is not committed')
    arena, asha = ev.load_contract(apath, root, check_files=False)
    arms_dir = Path(arms_dir).resolve()
    files = {a['name']: arms_dir / f"{a['name']}.json" for a in arena['arms']}
    for path in files.values():
        require(path.is_file(), f'{path}: the arm record is missing')
    guard = p1.Guard(root, a1.contract_path(None, root), c,
                     [judge_path, a1.ARENA_CONTRACT, tool, *(str(p) for p in files.values())])
    require(guard.start['git'] == start_git and guard.start['contract_file_sha256'] == sha
            and guard.start['files'][a1.ARENA_CONTRACT] == asha
            and guard.start['sources'] == {ref['path']: ref['sha256'] for ref in c['sources']},
            'checkout or contract changed before the recheck')
    judge = a1.committed_record(root, judge_path, 'judge record', eid)
    require(judge.get('format') == 'h7r-p1a1-judge-v1' and judge.get('contract_id') == a1.CONTRACT_ID
            and judge.get('contract_file_sha256') == sha, 'the judge record names another A1 contract')
    require(asha == judge['arena_contract_sha256'], 'the judge record names another Arena contract')
    require(set(files) == set(judge['arm_records_sha256']), 'the judge read other arms than the Arena contract has')
    sha_now = {name: guard.start['files'][str(path)] for name, path in files.items()}
    differ = sorted(n for n in files if sha_now[n] != judge['arm_records_sha256'][n])
    require(not differ, f'arm records differ from the ones the judge read: {differ}; nothing is rechecked')
    tables, summaries = {}, {}
    for a in arena['arms']:
        loaded = cli.load_arm(files[a['name']], arena, asha, root)
        require(loaded['table']['valid'] and len(loaded['table']['pairs']) == a['pairs'],
                f"{a['name']}: not a complete valid arm")
        tables[a['name']], summaries[a['name']] = loaded['table'], loaded['summary']
    st = c['reading']['statistics']
    b = st['bootstrap']
    strata = p1.paired_strata(tables)
    main = p1.stratified_bounds(strata, b['samples'], b['seed'], st['alpha_one_sided'])
    sens = p1.stratified_bounds(strata, b['samples'], b['seed'], st['alpha_one_sided'], unit='class')
    checks = {
        'arm_summaries': {n: summaries[n] == judge['per_arm'][n] for n in summaries},
        'opening_bounds': main == judge['bounds'],
        'class_bounds': sens == judge['sensitivity_class_level']['bounds'],
        'readings': p1.labels(main) == judge['readings'],
        'class_readings': p1.labels(sens) == judge['sensitivity_class_level']['readings']}
    equal = all(checks['arm_summaries'].values()) and all(v for k, v in checks.items() if k != 'arm_summaries')
    o1 = strata['O1']
    o1_t = sum(r[0] for r in o1) / len(o1)
    o1_d = sum(r[0] - r[1] for r in o1) / len(o1)
    o0 = class_analysis(strata['O0'], o1_t, o1_d)
    body = {'format': FORMAT, 'contract_id': a1.CONTRACT_ID, 'experiment_sha256': eid, 'contract_file_sha256': sha,
            'git': guard.start['git'], 'tool_sha256': guard.start['files'][tool],
            'judge_sha256': guard.start['files'][judge_path],
            'arena_contract_sha256': asha, 'arm_records_sha256': sha_now, 'arm_records_match_judge': True,
            'recomputed_equal_judge': equal, 'checks': checks,
            'recomputed': {'bounds': main, 'readings': p1.labels(main), 'class_bounds': sens,
                           'class_readings': p1.labels(sens)},
            'o0_classes': o0}
    if log_name is not None:
        body['log_name'] = log_name
    guard.check()
    if output is not None:
        p1.publish(output, body)
    return body


def main(argv=None, root=ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--arms-dir', default=ARMS_DIR)
    parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--log-name', help='the desktop attempt log basename, for records recovery')
    args = parser.parse_args(argv)
    try:
        out = Path(args.output)
        require(not out.exists(), f'{out} exists: never overwritten')
        body = recheck(args.arms_dir, root, output=out, log_name=args.log_name)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', flush=True)
        return 1
    o0 = body['o0_classes']
    print(json.dumps({'recomputed_equal_judge': body['recomputed_equal_judge'], 'checks': body['checks'],
                      'readings': body['recomputed']['readings'],
                      'o0': {k: o0[k] for k in ('pairs', 'classes', 'T_score', 'class_signs', 'deficit_concentration',
                                                'leave_one_class_out')}}, indent=1), flush=True)
    return 0 if body['recomputed_equal_judge'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
