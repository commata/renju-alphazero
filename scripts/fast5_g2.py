"""fast5 G2 (X2-F): TB-V3 Resolve on fast4 against fast5, the same corpus on the same machine.

Inputs are ``h7_timing.py measure --engine tb-v3`` files with rows (``--output`` or ``--jsonl``), one or more
rounds per runtime. Refused unless every file is TB-V3 (same engine hash and ``tb_v3.py``), the runtimes are
what they claim, and every position has the same decision in every round of both runtimes: the move, every
decision field (routes, root / V8-C / VCT2 statuses, nodes and calls) and the E4-v2 step (times excluded).
Reported (approved before running):
    speed     per position the mean over rounds; totals fast5 / fast4, percentiles, the base parts and the
              E4-v2 solver time, the E4 ran / exhausted groups
    tail      a table of the tracked positions (``--track``, default: the positions over 25 s in the fast4
              rounds; ``--tail`` adds the ``over_rule`` ids of a ``tb_v3_ser5_tail.py`` output): fast4, fast5, ratio
    ser5      for each fast5 round the SER5 rule (no move over 30 s, max <= 25 s); ``--machine ser5`` makes it
              the certification verdict (both rounds needed)

    python scripts/fast5_g2.py --fast4 runs/tb_v3/ser5_r1.json runs/tb_v3/ser5_r2.json \\
        --fast5 runs/tb_v3/ser5_fast5_r1.json runs/tb_v3/ser5_fast5_r2.json --tail docs/h7/tb_v3_ser5_tail.json \\
        --machine ser5 --output docs/h7/tb_v3_fast5_g2_ser5.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from h7_timing import RULES, percentiles, ser5_check  # noqa: E402
from tb_v3_ser5_tail import load_measure  # noqa: E402

FORMAT = 'tb-v3-fast5-g2-v1'


def _untimed(row: dict) -> dict:
    step = {k: v for k, v in (row.get('e4_attack') or {}).items() if k != 'seconds'}
    return {'move': row['move'], 'decision': row['decision'], 'e4_attack': step}


def check_file(measure: dict, runtime: str) -> None:
    from analysis import tb_v3, tb_v3_runtime

    rows = measure['rows']
    if 'engine' in measure and measure['engine'] != tb_v3.ENGINE_HASH:
        raise ValueError('not a TB-V3 Resolve measure')
    if 'identity_sha256' in measure and measure['identity_sha256'] != tb_v3_runtime.identity_sha256():
        raise ValueError('measured with another tb_v3.py')
    if 'runtime_mode' in measure and measure['runtime_mode'] != runtime:
        raise ValueError(f"a {measure['runtime_mode']} measure given as {runtime}")
    kinds = {(r.get('engine', 'tb-v2-value'), r.get('runtime', 'fast4')) for r in rows}
    if kinds != {('tb-v3', runtime)}:
        raise ValueError(f'rows of {sorted(kinds)}, expected TB-V3 on {runtime}')


def per_position(files: list[dict]) -> dict[str, dict]:
    out: dict[str, list] = {}
    for f in files:
        for r in f['rows']:
            out.setdefault(r['id'], []).append(r)
    return {pid: {'rows': rs, 'seconds': sum(r['seconds'] for r in rs) / len(rs),
                  'e4_seconds': sum(r.get('e4_seconds') or 0.0 for r in rs) / len(rs),
                  'times': {k: sum(r['times'][k] for r in rs) / len(rs) for k in rs[0]['times']}}
            for pid, rs in out.items()}


def analyse(fast4: list[dict], fast5: list[dict], track: list[str] | None = None, machine: str = 'desktop') -> dict:
    for f in fast4:
        check_file(f, 'fast4')
    for f in fast5:
        check_file(f, 'fast5')
    a, b = per_position(fast4), per_position(fast5)
    if set(a) != set(b):
        raise ValueError(f'the runtimes measured different positions ({len(set(a) ^ set(b))} differ)')
    differ = []
    for pid in a:
        rows = a[pid]['rows'] + b[pid]['rows']
        first = _untimed(rows[0])
        if any(_untimed(r) != first for r in rows[1:]):
            differ.append(pid)
    if differ:
        raise ValueError(f'{len(differ)} positions decided differently (not the same engine): {differ[:3]}')
    rule = RULES['ser5']['max_seconds']
    track = list(dict.fromkeys((track or []) + [pid for pid in a if a[pid]['seconds'] > rule]))
    ids = sorted(a)

    def side(d):
        ran = [pid for pid in ids if d[pid]['rows'][0].get('e4_attack', {}).get('ran')]
        exhausted = [pid for pid in ran if d[pid]['rows'][0]['e4_attack'].get('exhausted')]
        return {'all': percentiles([d[p]['seconds'] for p in ids]),
                'total_seconds': round(sum(d[p]['seconds'] for p in ids), 1),
                'parts_mean': {k: round(sum(d[p]['times'][k] for p in ids) / len(ids), 3) for k in d[ids[0]]['times']},
                'e4_ran': percentiles([d[p]['e4_seconds'] for p in ran]),
                'e4_exhausted': percentiles([d[p]['e4_seconds'] for p in exhausted]),
                'ran_total_seconds': percentiles([d[p]['seconds'] for p in ran]),
                'exhausted_total_seconds': percentiles([d[p]['seconds'] for p in exhausted])}

    sa, sb = side(a), side(b)
    rounds = [ser5_check(f.get('summary') or {'all': percentiles([r['seconds'] for r in f['rows']])}) for f in fast5]
    out = {'positions': len(ids), 'rounds': {'fast4': len(fast4), 'fast5': len(fast5)}, 'machine': machine,
           'same_decisions': True, 'fast4': sa, 'fast5': sb,
           'ratio_total': round(sb['total_seconds'] / sa['total_seconds'], 4),
           'ratio_by_part': {k: round(sb['parts_mean'][k] / sa['parts_mean'][k], 4) if sa['parts_mean'][k] else None
                             for k in sa['parts_mean']},
           'ratio_e4': round(sb['e4_ran']['mean'] / sa['e4_ran']['mean'], 4) if sa['e4_ran'] else None,
           'tracked': [{'id': pid, 'fast4': round(a[pid]['seconds'], 3), 'fast5': round(b[pid]['seconds'], 3),
                        'ratio': round(b[pid]['seconds'] / a[pid]['seconds'], 4),
                        'e4_fast4': round(a[pid]['e4_seconds'], 3), 'e4_fast5': round(b[pid]['e4_seconds'], 3),
                        'root_fast4': round(a[pid]['times']['root'], 3),
                        'root_fast5': round(b[pid]['times']['root'], 3)}
                       for pid in track if pid in a],
           'ser5_rounds': rounds}
    if machine == 'ser5':
        passed = len(rounds) >= 2 and all(r['pass'] for r in rounds)
        out['verdict'] = 'SER5_CERTIFIED' if passed else 'SER5_NOT_CERTIFIED'
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fast4', nargs='+', type=Path, required=True)
    parser.add_argument('--fast5', nargs='+', type=Path, required=True)
    parser.add_argument('--track', nargs='*', default=[], help='position ids for the tail table')
    parser.add_argument('--tail', type=Path, help='a tb_v3_ser5_tail.py output: its over_rule ids are tracked')
    parser.add_argument('--machine', choices=('desktop', 'ser5'), default='desktop')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    track = list(args.track)
    if args.tail is not None:
        track += json.loads(args.tail.read_text(encoding='utf-8'))['over_rule']
    try:
        result = analyse([load_measure(p) for p in args.fast4], [load_measure(p) for p in args.fast5], track,
                         args.machine)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256
    payload = {'format': FORMAT, 'git_commit': _git_commit(),
               'files': {'fast4': {str(p): file_sha256(p) for p in args.fast4},
                         'fast5': {str(p): file_sha256(p) for p in args.fast5}}, **result}
    print(json.dumps({k: v for k, v in result.items() if k not in ('fast4', 'fast5')}, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
