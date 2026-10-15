"""TB-RT30-v3 checks before A3-v3 (docs/web-ser5.md §9.8), fixed before the runs.

    speedup  TB1's own time at the 34 A3 divergence moves, fast2 against fast3 (the native scan):
             the same moves (34/34 both) and the measured speed-up per phase.
    gate     the A3-v3 entry gate on the v3_check / v3_dev / v3_a1 outputs in --out. Exit 0 when every
             row passes, 1 when one fails, 2 when none fails but one needs a review by hand;
             scripts/ser5/v3_a3.sh runs the 100 games only on 0:

             equivalence   frozen = fast3 = v3 (no deadline) on the 8 H3 positions: 8/8 identical
             X2 in use     no divergence, control or A1 move ran V8-B inline (``attack_helper_failed``):
                           such a move is v1's timing, not the engine being gated
             divergence    v3, 4 searches at once: net points_if_recovered >= +3.0, none over 27.5 s
             control       v3 keeps TB1's late-SAFE tree move on 26/26 control moves. v3 is the exact
                           path, so a changed move is never tolerated silently: one with no timeout
                           and no skipped phase is an exactness failure (FAIL); one cut by the
                           deadline is listed for a review by hand (REVIEW)
             A1            slow60 and critical6: recheck UNSAFE 0 and none over 27.5 s

    python scripts/check_v3_gate.py speedup --fast2 docs/rt30-results/a31_tb1_w4.json --fast3 runs/rt30/a31_tb1f3_w4.json
    python scripts/check_v3_gate.py gate --out runs/rt30
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

NET_POINTS = 3.0


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def speedup(args) -> int:
    a, b = load(args.fast2), load(args.fast3)
    rows_a = {r['key']: r for r in a['rows']}
    pairs = [(rows_a[r['key']], r) for r in b['rows'] if r['key'] in rows_a]
    same = sum(x['move'] == y['move'] for x, y in pairs)
    print(f"| | fast2 | fast3 |\n|---|---|---|\n| moves (TB1's) | {a['summary']['recovered']}/{a['summary']['cases']} | "
          f"{b['summary']['recovered']}/{b['summary']['cases']} |\n| same move as fast2 | | {same}/{len(pairs)} |")
    for phase in ('attack', 'root', 'total'):
        def secs(r):
            return r['seconds'] if phase == 'total' else r['phases'][phase]
        ratios = [secs(x) / secs(y) for x, y in pairs if secs(y) > 0.5 and secs(x) > 0.5]
        if ratios:
            print(f'| speed-up {phase} (median, min-max) | | {statistics.median(ratios):.2f} '
                  f'({min(ratios):.2f}-{max(ratios):.2f}, n={len(ratios)}) |')
    return 0 if same == len(pairs) else 1


def gate(args) -> int:
    out = Path(args.out)
    rows, notes, verdicts = [], [], []

    def row(name, value, verdict):
        verdict = verdict if isinstance(verdict, str) else ('PASS' if verdict else 'FAIL')
        verdicts.append(verdict)
        rows.append(f"| {name} | {value} | {verdict} |")

    try:
        eq = load(out / 'v3_equivalence_h3_8.json')['summary']
        row('equivalence frozen = fast3 = v3', f"{eq['identical']}/{eq['positions']}", eq['identical'] == eq['positions'])
        dv_all = load(out / 'a31_v3_w4.json')
        ct_all = load(out / 'ctrl_v3_w4.json')
        inline = [r['key'] for r in dv_all['rows'] + ct_all['rows'] if r.get('attack_helper_failed')]
        row('X2 in use: divergence + control moves with V8-B inline', len(inline), not inline)
        notes += [f'- V8-B inline (helper not used): `{k}`' for k in inline]
        dv = dv_all['summary']
        net = dv['loss_making']['points_if_recovered'] + dv['gain_making']['points_if_recovered']
        row('divergence w4: net points', f"{net:+.1f} (recovered {dv['recovered']}/{dv['cases']})", net >= NET_POINTS)
        row('divergence w4: over 27.5 s', dv['seconds']['over_27.5s'], dv['seconds']['over_27.5s'] == 0)
        ct = ct_all['summary']
        changed = [r for r in ct_all['rows'] if not r['recovered']]
        unexplained = [r for r in changed if not (r.get('timeouts') or r.get('skipped'))]
        row('control w4: tree move kept', f"{ct['recovered']}/{ct['cases']}",
            'FAIL' if unexplained else ('REVIEW' if changed else 'PASS'))
        notes += [f"- control changed: `{r['key']}` played {r['move']} (TB1 {r['tb1_move']}), "
                  f"timeouts {r.get('timeouts') or '-'}, skipped {r.get('skipped') or '-'}"
                  f"{'' if r.get('timeouts') or r.get('skipped') else ' -> no deadline cut: exactness failure'}"
                  for r in changed]
        for name in ('s60', 'crit'):
            run = load(out / f'v3_h3_{name}.json')['summary']
            recheck = load(out / f'recheck_v3_h3_{name}.json')['summary']
            over = run['rt3']['seconds']['over_27.5s']
            row(f'A1 {name}: recheck UNSAFE / over 27.5 s', f"{recheck['changed_move_proven_unsafe']} / {over}",
                recheck['changed_move_proven_unsafe'] == 0 and over == 0)
            failed = run.get('rt_deadline', {}).get('attack_helper_failed', 0)
            row(f'A1 {name}: moves with V8-B inline', failed, failed == 0)
    except FileNotFoundError as exc:
        row('missing', exc.filename, False)
    verdict = 'FAIL' if 'FAIL' in verdicts else ('REVIEW' if 'REVIEW' in verdicts else 'PASS')
    print('| check | value | |\n|---|---|---|\n' + '\n'.join(rows))
    if notes:
        print('\n' + '\n'.join(notes))
    print(f'\nA3-v3 gate: {verdict}')
    return {'PASS': 0, 'FAIL': 1, 'REVIEW': 2}[verdict]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('speedup')
    p.add_argument('--fast2', type=Path, required=True)
    p.add_argument('--fast3', type=Path, required=True)
    p = sub.add_parser('gate')
    p.add_argument('--out', type=Path, default=Path('runs/rt30'))
    args = parser.parse_args(argv)
    return speedup(args) if args.command == 'speedup' else gate(args)


if __name__ == '__main__':
    sys.exit(main())
