"""H7-E4-v2 cost-model check: does the nodes / calls model price the killer attack and the G1 gate right?

The COSTDEV cost model (``h7_e4v2_dev.fit_cost_model``) was fitted on E4-v1's plain attacks. Here, in one
process (timing), on a fixed subset of COSTDEV (the first ``--count`` attacked pre-check positions in
sha256(id) order), the G1 gate and the 200k attack in both modes (plain, killer; their order alternates by
position) are timed and compared with the model:
    factor[mode]  = sum of measured seconds / sum of model seconds
    correction    = factor[mode] / factor['plain'] for 'killer' and 'gate' (relative to the plain attack the
                    model was fitted on, so the machine speed cancels); ``h7_e4v2_dev.py --calibration``
                    multiplies the model seconds of killer attacks and of the gate by these
Run on the desktop (the machine of the pre-check), nothing else running.

    python scripts/h7_e4v2_calibrate.py --precheck docs/h7/h7_e4_precheck.json \\
        --positions docs/h7/h7_timing_positions.json --fast fast4 --output docs/h7/h7_e4v2_calibration.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import h7_e4v2_dev as dev2  # noqa: E402

FORMAT = 'h7-e4v2-calibration-v1'
COUNT = 150


def subset(rows: list[dict], count: int) -> list[dict]:
    attacked = [r for r in rows if r['attacked']]
    return sorted(attacked, key=lambda r: hashlib.sha256(r['id'].encode('utf-8')).hexdigest())[:count]


def measure(moves, first_killer: bool) -> dict:
    from analysis import e4v2_attack as v2
    from renju import Game

    game = Game()
    for m in moves:
        game.play(*m)
    started = perf_counter()
    gate = v2.opponent_vcf(game)
    out = {'gate': {**gate, 'seconds': perf_counter() - started}}
    if gate['status'] == 'WIN':
        return out
    for killer in ((True, False) if first_killer else (False, True)):
        _, info = v2.attack(game, '200k', killer)
        out['killer' if killer else 'plain'] = {k: info[k] for k in ('nodes', 'calls', 'seconds', 'exhausted')}
    return out


def summarize(rows: list[dict], model: dict) -> dict:
    sums = {k: [0.0, 0.0, 0] for k in ('plain', 'killer', 'gate')}
    for r in rows:
        g = r['gate']
        sums['gate'][0] += g['seconds']
        sums['gate'][1] += dev2.model_seconds(model, g['nodes'], g['calls'], False)
        sums['gate'][2] += 1
        for mode in ('plain', 'killer'):
            if mode in r:
                sums[mode][0] += r[mode]['seconds']
                sums[mode][1] += dev2.model_seconds(model, r[mode]['nodes'], r[mode]['calls'], True)
                sums[mode][2] += 1
    factor = {k: round(a / b, 4) if b else None for k, (a, b, _) in sums.items()}
    return {'factor': factor, 'n': {k: v[2] for k, v in sums.items()},
            'measured_seconds': {k: round(v[0], 2) for k, v in sums.items()},
            'model_seconds': {k: round(v[1], 2) for k, v in sums.items()},
            'correction': {k: round(factor[k] / factor['plain'], 4) for k in ('killer', 'gate')
                           if factor[k] and factor['plain']}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--precheck', type=Path, required=True)
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--count', type=int, default=COUNT)
    parser.add_argument('--fast', choices=('off', 'fast4'), default='off')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from hybrid.progress import Progress
    from scripts.h7_timing import machine, restore_fast_mode
    from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256

    pre = json.loads(args.precheck.read_text(encoding='utf-8'))
    if pre.get('positions_sha256') != file_sha256(args.positions):
        parser.error(f'{args.precheck} was not measured on {args.positions}')
    timing = {p['id']: p for p in json.loads(args.positions.read_text(encoding='utf-8'))['positions']}
    model = dev2.fit_cost_model(pre['rows'])
    picked = subset(pre['rows'], args.count)
    progress, rows = Progress('h7-e4v2-calibrate', len(picked), 'positions'), []
    with restore_fast_mode():
        if args.fast != 'off':
            from hybrid.h6_fast import install
            install(args.fast)
        for i, r in enumerate(picked):
            rows.append({'id': r['id'], **measure([tuple(m) for m in timing[r['id']]['moves']], i % 2 == 1)})
            progress.update(1)
    result = summarize(rows, model)
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'precheck_sha256': file_sha256(args.precheck),
               'positions_sha256': file_sha256(args.positions), 'fast': args.fast, 'machine': machine(),
               'count': args.count, 'cost_model': model, **result, 'rows': rows}
    print(json.dumps(result, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
