"""One table over several run_rt30_replay.py outputs (same manifest, same baseline): the cap decision view.

Per run: latency (median / p95 / max / over 27.5 s), deadline events per phase, root rescues,
moves equal to the baseline, the change kinds (lost_win, other-move lost wins, kept_tree_move,
played_baseline_unsafe) and, when a check_rt30_changed_moves.py output is given with ``run=recheck``,
its gate (changed moves proven UNSAFE).

    python scripts/compare_rt30_runs.py \\
        cap4=docs/rt30-results/fast2_h3_slow-stratified-v1.json \\
        cap6=runs/rt30/cap6_h3_s60.json:runs/rt30/recheck_cap6_h3_s60.json \\
        cap8=runs/rt30/cap8_h3_s60.json:runs/rt30/recheck_cap8_h3_s60.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def row(label: str, run_path: Path, recheck_path: Path | None) -> dict:
    run = json.loads(run_path.read_text(encoding='utf-8'))
    summary = run['summary']
    arm = next(a for a in ('sup', 'rt2', 'rt', 'fast', 'frozen') if a in summary)
    seconds = summary[arm]['seconds']
    deadline = summary.get('rt_deadline', {})
    vs = summary.get('vs_baseline', {})
    kinds = vs.get('changed_kinds', {})
    out = {
        'run': label, 'timing': {k: v for k, v in (run.get('timing') or {}).items() if k == 'attack_seconds'},
        'manifest': (run.get('manifest') or {}).get('version'), 'n': seconds['n'],
        'median': seconds['median'], 'p95': seconds['p95'], 'max': seconds['max'], 'over_27.5': seconds['over_27.5s'],
        'timeouts': deadline.get('timeouts', {}), 'timeout_moves': deadline.get('moves_with_timeout_or_skip'),
        'rescued': deadline.get('root_rescued'),
        'same_move': vs.get('same_move'), 'lost_win': kinds.get('lost_win', 0),
        'lost_win_other_move': vs.get('lost_own_vct_other_move'), 'kept_tree_move': kinds.get('kept_tree_move', 0),
        'other': kinds.get('other', 0), 'baseline_unsafe': len(vs.get('played_baseline_unsafe', [])),
    }
    if recheck_path is not None:
        recheck = json.loads(recheck_path.read_text(encoding='utf-8'))['summary']
        out['recheck_unsafe'] = recheck['changed_move_proven_unsafe']
        out['recheck_unknown'] = recheck['changed_vct1'].get('UNKNOWN')
        out['gate_pass'] = recheck['gate_pass']
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='LABEL=RUN.json[:RECHECK.json]')
    parser.add_argument('--json', action='store_true', help='print JSON instead of a table')
    args = parser.parse_args(argv)
    rows = []
    for item in args.runs:
        label, _, paths = item.partition('=')
        run_path, _, recheck = paths.partition(':')
        rows.append(row(label, Path(run_path), Path(recheck) if recheck else None))
    manifests = {r['manifest'] for r in rows}
    if len(manifests) > 1:
        print(f'warning: different manifests {manifests}', file=sys.stderr)
    if args.json:
        print(json.dumps(rows, indent=1))
        return 0
    columns = [('run', 'run'), ('n', 'n'), ('median', 'median'), ('p95', 'p95'), ('max', 'max'),
               ('over_27.5', '>27.5'), ('timeout_moves', 't/o moves'), ('rescued', 'rescue'),
               ('same_move', 'same'), ('lost_win', 'lost win'), ('lost_win_other_move', 'lost→other'),
               ('kept_tree_move', 'kept tree'), ('other', 'other'), ('baseline_unsafe', 'base unsafe'),
               ('recheck_unsafe', 'recheck unsafe'), ('gate_pass', 'gate')]
    print('| ' + ' | '.join(h for _, h in columns) + ' | phase timeouts (attack/stage/root/vct2) |')
    print('|' + '---|' * (len(columns) + 1))
    for r in rows:
        t = r['timeouts']
        phases = '/'.join(str(t.get(p, '-')) for p in ('attack', 'stage_vct', 'root', 'vct2'))
        print('| ' + ' | '.join(str(r.get(k, '-')) for k, _ in columns) + f' | {phases} |')
    return 0


if __name__ == '__main__':
    sys.exit(main())
