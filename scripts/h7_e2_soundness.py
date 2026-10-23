"""H7-E2: offline soundness recheck of TB-Engine-v1-T's tree moves (docs/h6-value.md, H7-E2).

For every tree move of the candidate's aux runs (``puct_policy_vct2_50k_tt`` vs ``v8:full``) the recorded
V8-C status (with the table) is compared with the frozen solver:
    frozen standard   V8-C's first check of the move without the table (``h6_5d.STANDARD``)
    frozen 4M         only where the move is recorded decided and the frozen standard check is UNKNOWN
                      (the table decided what the frozen budget could not): the legacy 4M recheck
A contradiction is a decided status against the opposite decided status (SAFE vs UNSAFE); UNKNOWN on
either side is never one. Every recorded UNSAFE is replayed with the table (standard, then 4M if the
standard check stays UNKNOWN: V8-C may decide after its first check) with each VCF line replayed by
``analysis.vcf_tt_v1.verify_line``; it must come out UNSAFE with every line valid.

    python -u scripts/h7_e2_soundness.py --runs runs/h7_e2/tt_vs_full_8421.json --fast fast3 --workers 12 \\
        --jsonl runs/h7_e2/sound_8421.jsonl --output docs/h7/h7_e2_soundness_8421.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import h6_5d  # noqa: E402

FORMAT = 'h7-e2-soundness-v1'
CANDIDATE = 'puct_policy_vct2_50k_tt'
DECIDED = ('SAFE', 'UNSAFE')


def opposite(a, b) -> bool:
    return a in DECIDED and b in DECIDED and a != b


def sound_task(task: dict) -> dict:
    from scripts.h7_e1_vcf_table import run_check

    h6_5d._ensure_fast(task.get('fast', 'off'))
    moves, move, recorded = task['moves'], task['tree_move'], task['recorded_status']
    frozen = h6_5d.check_status(moves, move, h6_5d.STANDARD)
    row = {'id': task['id'], 'seed': task['seed'], 'recorded': recorded, 'frozen_standard': frozen['status'],
           'frozen_large': None, 'replay': None, 'seconds': frozen['seconds']}
    if recorded in DECIDED and frozen['status'] == 'UNKNOWN':
        large = h6_5d.check_status(moves, move, h6_5d.LARGE)
        row['frozen_large'] = large['status']
        row['seconds'] += large['seconds']
    if recorded == 'UNSAFE':
        replay = run_check(moves, move, h6_5d.STANDARD, 'on_verified')
        line_fail = replay['line_fail']
        if replay['status'] == 'UNKNOWN':
            replay = run_check(moves, move, h6_5d.LARGE, 'on_verified')
            line_fail += replay['line_fail']
        row['replay'] = {'status': replay['status'], 'line_fail': line_fail}
    row['contradiction'] = opposite(recorded, frozen['status']) or opposite(recorded, row['frozen_large'])
    return row


def _chunk(tasks):
    return [sound_task(t) for t in tasks]


def summarize(rows: list[dict]) -> dict:
    unsafe = [r for r in rows if r['recorded'] == 'UNSAFE']
    transitions = {}
    for r in rows:
        k = f"{r['frozen_standard']}->{r['recorded']}"
        transitions[k] = transitions.get(k, 0) + 1
    legacy = {}
    for r in rows:
        if r['frozen_large'] is not None:
            k = f"{r['recorded']}:frozen4M_{r['frozen_large']}"
            legacy[k] = legacy.get(k, 0) + 1
    out = {'tree_moves': len(rows),
           'contradictions': [r['id'] for r in rows if r['contradiction']],
           'unsafe': len(unsafe),
           'unsafe_not_replayed': [r['id'] for r in unsafe if r['replay']['status'] != 'UNSAFE'],
           'line_fail': sum(r['replay']['line_fail'] for r in unsafe),
           'transitions': dict(sorted(transitions.items())), 'legacy_4m': dict(sorted(legacy.items()))}
    out['sound'] = not (out['contradictions'] or out['unsafe_not_replayed'] or out['line_fail'])
    return out


def build_tasks(run: dict, fast: str) -> list[dict]:
    if run['arm'] != CANDIDATE or run['opponent'] != 'v8:full':
        raise ValueError(f"expected {CANDIDATE} vs v8:full, got {run['arm']} vs {run['opponent']}")
    return [{**p, 'seed': run['seed'], 'fast': fast} for p in h6_5d.tree_positions(run)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--fast', choices=('off', 'fast2', 'fast3'), default='off')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--jsonl', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    runs = [json.loads(p.read_text(encoding='utf-8')) for p in args.runs]
    if args.fast != 'off':
        try:
            h6_5d._ensure_fast(args.fast)
        except RuntimeError as exc:
            parser.error(str(exc))
    try:
        tasks = [t for r in runs for t in build_tasks(r, args.fast)]
    except ValueError as exc:
        parser.error(str(exc))
    # the 4M rechecks come from the moves the table decided: start the recorded-decided ones first
    order = sorted(range(len(tasks)), key=lambda i: tasks[i]['recorded_status'] == 'SAFE')
    done = h6_5d._run([tasks[i] for i in order], _chunk, args.workers, args.jsonl, 'h7-e2-sound')
    by_id = {r['id']: r for r in done}
    rows = [by_id[t['id']] for t in tasks]
    summary = {'seeds': sorted({r['seed'] for r in runs}), **summarize(rows)}
    from scripts.run_mcts_v8_benchmark import _fast_provenance
    result = {'format': FORMAT, 'fast': _fast_provenance(args.fast), 'standard': h6_5d.STANDARD, 'large': h6_5d.LARGE,
              'sources': {str(p): {'arm': r['arm'], 'seed': r['seed']} for p, r in zip(args.runs, runs)},
              'summary': summary, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in summary.items()}, indent=1), flush=True)
    return 0 if summary['sound'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
