"""--prove-losses-last-n from a pilot report (docs/h6-value.md, H6-2a' step 3), and a rerun check.

    python scripts/h6_last_n.py runs/h6/h6_pilot_fast.json --expect docs/h6/h6_pilot_prove_losses.json

Prints the cumulative ``vcf_loss`` share by distance to the game end and the rule's N (the smallest N
covering at least ``--share`` of the observed ``vcf_loss``). With the budget-cut distances (``prove_losses_budget_cuts_by_distance_to_end``,
reports after the off-mode pilot) it also prints, for each N, the budget cuts farther than N and the share if every one of
them were a hidden loss (the worst case for N; a cut within N adds to both sides and cannot lower it).

``--expect`` compares the settings and every summary key the two reports share (a rerun with ``--fast``
must give the same counts); keys only one report has are listed, not compared. Exit code 1 on a difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _distance(key: str) -> int:
    return 40 if key == '40+' else int(key)


def cumulative(counts: dict) -> dict[int, int]:
    """``{n: count with distance <= n}`` for n = 1..40 (``40+`` counts at 40)."""
    by = {}
    for key, count in counts.items():
        by[_distance(key)] = by.get(_distance(key), 0) + count
    out, total = {}, 0
    for n in range(1, 41):
        total += by.get(n, 0)
        out[n] = total
    return out


def choose_n(summary: dict, share: float = 0.95) -> dict:
    losses = summary['proof_by_distance_to_end'].get('vcf_loss', {})
    total = sum(losses.values())
    within = cumulative(losses)
    cuts = summary.get('prove_losses_budget_cuts_by_distance_to_end')
    cut_within = cumulative(cuts) if cuts is not None else None
    cut_total = sum(cuts.values()) if cuts is not None else None
    table = []
    for n in range(1, 41):
        row = {'n': n, 'vcf_loss_within': within[n], 'share': within[n] / total if total else None}
        if cuts is not None:
            beyond = cut_total - cut_within[n]
            row['cuts_beyond'] = beyond
            row['worst_case_share'] = within[n] / (total + beyond) if total + beyond else None
        table.append(row)
    chosen = next((r['n'] for r in table if r['share'] is not None and r['share'] >= share), None)
    return {'vcf_loss': total, 'budget_cuts': summary.get('prove_losses_budget_cuts'), 'share_rule': share,
            'n': chosen, 'table': table}


def compare(expected: dict, actual: dict) -> list[str]:
    problems = []
    for key in sorted(set(expected['settings']) | set(actual['settings'])):
        if expected['settings'].get(key) != actual['settings'].get(key):
            problems.append(f"settings.{key}: {expected['settings'].get(key)!r} != {actual['settings'].get(key)!r}")
    for key in sorted(set(expected['summary']) & set(actual['summary'])):
        if expected['summary'][key] != actual['summary'][key]:
            problems.append(f'summary.{key} differs')
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('report', type=Path)
    parser.add_argument('--expect', type=Path, help='an earlier report of the same settings (e.g. the off-mode pilot)')
    parser.add_argument('--share', type=float, default=0.95)
    parser.add_argument('--rows', type=int, default=20, help='table rows to print')
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding='utf-8'))
    result = choose_n(report['summary'], args.share)
    print(f"vcf_loss {result['vcf_loss']:,}, budget cuts {result['budget_cuts']}, "
          f"rule: smallest N with share >= {args.share:.0%} -> N = {result['n']}")
    has_cuts = 'cuts_beyond' in result['table'][0]
    print('  N  within   share' + ('  cuts>N  worst case' if has_cuts else ''))
    for row in result['table'][:args.rows]:
        line = f"{row['n']:>3}  {row['vcf_loss_within']:>6}  {row['share']:6.1%}"
        if has_cuts:
            line += f"  {row['cuts_beyond']:>6}  {row['worst_case_share']:9.1%}"
        print(line + ('  <- N' if row['n'] == result['n'] else ''))
    if not has_cuts:
        print('(no budget-cut distances in this report: rerun with the current builder to get them)')
    if args.expect is None:
        return 0
    expected = json.loads(args.expect.read_text(encoding='utf-8'))
    problems = compare(expected, report)
    only = sorted(set(expected['summary']) ^ set(report['summary']))
    print(f"--expect {args.expect}: {'SAME' if not problems else 'DIFFERENT'} "
          f"({len(set(expected['summary']) & set(report['summary']))} shared summary keys; not compared: {only})")
    for problem in problems:
        print('  ' + problem)
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
