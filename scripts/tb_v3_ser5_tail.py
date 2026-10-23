"""TB-V3 Resolve on SER5: where the time of the slow moves goes (deployment analysis, no gate).

Reads ``h7_timing.py measure --engine tb-v3`` outputs (the full ``--output`` file with rows, or its ``--jsonl``;
not the committed summaries) and, optionally, the TB-V2-Value fast4 measure of the same machine on the same
positions (its ``--output`` or ``--jsonl``; omit ``--tb-v2`` if neither is kept): the base is the same
engine with the same seed per position, so TB-V3 - TB-V2-Value is the E4-v2 step (gate + attack) there.
Per position: total, the attack solver's own time (``e4_seconds``), the base parts (tree / root / vct2 /
value), the step (skip reason, nodes, exhausted) and the paired TB-V2 time. Reported: the slowest positions,
every position over the 25 s rule, the E4 time distribution, and two what-ifs computed from the rows
(not measured): the attack run beside the base (max of the two instead of the sum) and attack time capped.

    python scripts/tb_v3_ser5_tail.py --tb-v3 runs/tb_v3/ser5_r1.json runs/tb_v3/ser5_r2.json \\
        --tb-v2 runs/h7_timing/mini_fast4_r1.json runs/h7_timing/mini_fast4_r2.json \\
        --output docs/h7/tb_v3_ser5_tail.json
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

from h7_timing import RULES, percentiles  # noqa: E402

TOP = 25
STEP_FIELDS = ('base_route', 'base_move', 'ran', 'skip', 'move', 'exhausted', 'nodes', 'calls')


def decision_of(row: dict) -> dict:
    """What must agree between rounds: the move, the route and the E4-v2 step (TB-V3 rows)."""
    step = row.get('e4_attack')
    out = {'move': row['move'], 'route': row['decision']['route']}
    if step is not None:
        out.update({f'e4.{k}': step.get(k) for k in STEP_FIELDS}, **{'e4.gate': (step.get('gate') or {}).get('status')})
    return out


def _mean_rows(files: list[dict]) -> dict[str, dict]:
    """Rows per position id, times averaged over the rounds (the decisions must agree)."""
    out: dict[str, dict] = {}
    for f in files:
        for r in f['rows']:
            out.setdefault(r['id'], []).append(r)
    merged = {}
    for pid, rs in out.items():
        first = decision_of(rs[0])
        for r in rs[1:]:
            other = decision_of(r)
            if other != first:
                raise ValueError(f'{pid}: the rounds differ in {sorted(k for k in first if first[k] != other.get(k))}')
        n = len(rs)
        merged[pid] = {**rs[0], 'seconds': sum(r['seconds'] for r in rs) / n,
                       'e4_seconds': sum(r.get('e4_seconds') or 0.0 for r in rs) / n,
                       'times': {k: sum(r['times'][k] for r in rs) / n for k in rs[0]['times']}}
    return merged


def engine_of(measure: dict) -> str:
    """'tb-v3' or 'tb-v2-value' from the file's engine hash, or (a jsonl) from its rows."""
    from analysis import tb_v2_value, tb_v3

    if 'engine' in measure:
        return {tb_v3.ENGINE_HASH: 'tb-v3', tb_v2_value.config_hash(): 'tb-v2-value'}.get(measure['engine'], 'other')
    kinds = {r.get('engine', 'tb-v2-value') for r in measure['rows']}
    return kinds.pop() if len(kinds) == 1 else 'mixed'


def load_measure(path: Path) -> dict:
    """A measure ``--output`` file, or its ``--jsonl`` (one row per line; a torn last line is skipped)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding='utf-8')
    if path.suffix != '.jsonl':
        return json.loads(text)
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {'rows': rows}


def analyse(v3_files: list[dict], v2_files: list[dict]) -> dict:
    for f in v3_files:
        if engine_of(f) != 'tb-v3':
            raise ValueError('--tb-v3 takes TB-V3 Resolve measure files (with rows)')
    for f in v2_files:
        if engine_of(f) != 'tb-v2-value':
            raise ValueError('--tb-v2 takes TB-V2-Value measure files (with rows)')
    v3 = _mean_rows(v3_files)
    v2 = _mean_rows(v2_files) if v2_files else {}
    if v2:  # the pairing is valid only if TB-V3's base made TB-V2's decision on every position
        missing = sorted(set(v3) - set(v2))
        if missing:
            raise ValueError(f'{len(missing)} TB-V3 positions have no TB-V2 row: {missing[:3]}')
        bad = [pid for pid, r in v3.items() if (r['e4_attack'].get('base_move'), r['e4_attack'].get('base_route'))
               != (v2[pid]['move'], v2[pid]['decision']['route'])]
        if bad:
            raise ValueError(f'{len(bad)} positions: TB-V3 base move / route != TB-V2 (no paired overhead): {bad[:3]}')
    rule = RULES['ser5']['max_seconds']
    rows = []
    for pid, r in v3.items():
        step = r.get('e4_attack', {})
        base = v2.get(pid)
        rows.append({'id': pid, 'source': r['source'], 'seconds': round(r['seconds'], 3),
                     'e4_seconds': round(r['e4_seconds'], 3), 'route': r['decision']['route'],
                     'times': {k: round(v, 3) for k, v in r['times'].items()},
                     'step': {k: step.get(k) for k in ('base_route', 'ran', 'skip', 'move', 'exhausted', 'nodes',
                                                       'calls', 'candidates')},
                     'gate': (step.get('gate') or {}).get('status'),
                     'tb_v2_seconds': round(base['seconds'], 3) if base else None,
                     'step_seconds': round(r['seconds'] - base['seconds'], 3) if base else None})
    rows.sort(key=lambda x: -x['seconds'])
    paired = [x for x in rows if x['step_seconds'] is not None]

    def what_if(fn):
        values = [fn(x) for x in paired]
        return percentiles(values) if values else None

    return {'positions': len(rows), 'rounds': {'tb_v3': len(v3_files), 'tb_v2': len(v2_files)},
            'rule_max_seconds': rule,
            'tb_v3': percentiles([x['seconds'] for x in rows]),
            'e4_seconds': percentiles([x['e4_seconds'] for x in rows if x['step']['ran']]),
            'step_seconds': percentiles([x['step_seconds'] for x in paired]) if paired else None,
            'exhausted_attacks': sum(bool(x['step']['exhausted']) for x in rows if x['step']['ran']),
            'e4_seconds_exhausted': percentiles([x['e4_seconds'] for x in rows if x['step']['ran']
                                                 and x['step']['exhausted']]),
            'checks': {'rounds_agree': 'move, route and E4-v2 step equal in every round',
                       'paired_base': 'TB-V3 base move and route = TB-V2 move and route' if paired else None},
            'skips': {k: sum(x['step']['skip'] == k for x in rows) for k in ('route', 'cooldown', 'opponent_vcf')},
            'over_rule': [x['id'] for x in rows if x['seconds'] > rule],
            # not measured: what the rows imply if the step ran beside the base (no CPU contention counted),
            # or the paired step overhead were capped (not a solver cut: a cut could change the proof)
            'what_if': {'step_beside_base': what_if(lambda x: max(x['tb_v2_seconds'], x['step_seconds'])),
                        **{f'step_capped_{c}s': what_if(lambda x, c=c: x['tb_v2_seconds'] + min(x['step_seconds'], c))
                           for c in (2, 4, 6)}},
            'slowest': rows[:TOP], 'rows': rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tb-v3', nargs='+', type=Path, required=True)
    parser.add_argument('--tb-v2', nargs='*', type=Path, default=[])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        v3 = [load_measure(p) for p in args.tb_v3]
        v2 = [load_measure(p) for p in args.tb_v2]
    except FileNotFoundError as exc:
        parser.error(f'{exc.args[0]} not found. Measure files keep their rows in the --output .json or the --jsonl '
                     "of that run; look for them with: find ~ -name '*fast4_r*.json*' -path '*runs*'. "
                     'Without a TB-V2-Value file, leave out --tb-v2 (only the paired columns are lost).')
    try:
        result = analyse(v3, v2)
    except ValueError as exc:
        parser.error(str(exc))
    shown = {k: v for k, v in result.items() if k not in ('rows', 'slowest')}
    shown['slowest'] = [{k: x[k] for k in ('id', 'seconds', 'e4_seconds', 'tb_v2_seconds', 'step_seconds', 'route')}
                        for x in result['slowest'][:10]]
    print(json.dumps(shown, indent=1), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=1) + '\n',
                               encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
