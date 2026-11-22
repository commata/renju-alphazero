"""Check the unresolved E1 move from TB-V3-BUDGET-DIAG-1 with fast5."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import h7r_budget_diag as original

ROOT = original.ROOT
REFERENCE = '863f808fa494693147cfefe2926563f7ec2f36d3'
POSITION = '8411-p8-white/11'
MOVE = [8, 9]
CALIBRATION_SECONDS = 300
TIMING_SLACK = 2
MAX_DEEPEN_SECONDS = 1800
INPUTS = {'manifest': (original.E1_MANIFEST,
                       'da6c9dbe0fe4eac6809efba7dc86f668ba13b7aa52db9070f47d593c1f378dc4'),
          'deepen': (f'{original.D}/deepen.json',
                     '94f35a318b35eaacb5214344b55343a260c873d7883193ac6ae303e0c50cea2d'),
          'contract': (original.CONTRACT,
                       '9e289f6411d30aaa5cea3d7cd41b3c91529c615a9246087b5fc5321496ba4454')}


def read_pinned(relative_path, expected_sha256):
    current = (ROOT / relative_path).read_bytes()
    original.require(hashlib.sha256(current).hexdigest() == expected_sha256,
                     f'{relative_path} differs from the diagnosis commit')
    return json.loads(current)


def evidence():
    manifest, deepen, contract = (read_pinned(*INPUTS[name]) for name in ('manifest', 'deepen', 'contract'))
    for ref in contract['sources']:
        original.require(original.file_sha256(ROOT / ref['path']) == ref['sha256'],
                         f"{ref['path']} differs from the diagnosis contract")
    position = next(p for p in manifest['positions'] if p['key'] == POSITION and p['class'] == 'E1-P')
    item, = [i for i in deepen['items'] if i['position'] == POSITION and i['move'] == MOVE]
    return position['history'], item['b0'], {'commit': REFERENCE, **{f'{name}_sha256': ref[1]
                                                                    for name, ref in INPUTS.items()}}


def budgets():
    return {'B0': original.B0, **original.BUDGETS}


def worker(budget):
    from analysis import mcts_v8
    from h7c_arena import install_runtime
    from rt30 import fast5
    from rt30.native import build as native_build
    from rt30.vcfcore import build as vcf_build

    history, _, _ = evidence()
    install_runtime()
    original.require(mcts_v8._find_vcf_with_stats is fast5.find_vcf_with_stats,
                     'the truth solver is not bound to fast5')
    return {'result': original.recheck(history, MOVE, budgets()[budget]),
            'runtime': {'platform': platform.platform(), 'python': platform.python_version(),
                        'vcf_binding': 'rt30.fast5.find_vcf_with_stats',
                        'native_build': native_build.build_info(), 'vcf_build': vcf_build.build_info()}}


def check(budget, limit):
    started = time.perf_counter()
    try:
        run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', budget],
                             cwd=ROOT, capture_output=True, text=True, timeout=limit, check=True)
        row = json.loads(run.stdout)
    except subprocess.TimeoutExpired:
        row = {'result': {'status': 'TIMEOUT'}}
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.stderr or exc.stdout or f'{budget}: worker failed') from exc
    return {'budget': budgets()[budget], 'limit_seconds': limit,
            'wall_seconds': round(time.perf_counter() - started, 2), **row}


def write_new(path, body):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(body, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', choices=('calibrate', 'deepen'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--worker', choices=budgets(), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker)))
        return
    if args.stage is None or args.output_dir is None:
        parser.error('a stage and --output-dir are required')

    _, expected, inputs = evidence()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'{args.stage}.json'
    original.require(not output.exists(), f'{output} exists')
    calibration_file = args.output_dir / 'calibrate.json'
    identity = {'diagnostic': 'TB-V3-BUDGET-DIAG-FAST5-1', 'position': POSITION, 'move': MOVE,
                'inputs': inputs, 'expected_b0': expected,
                'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.stage == 'calibrate':
        b0 = check('B0', CALIBRATION_SECONDS)
        matched = all(b0['result'].get(k) == v for k, v in expected.items())
        record = {**identity, 'matched_b0': matched, 'B0': b0}
    else:
        calibration_bytes = calibration_file.read_bytes()
        calibration = json.loads(calibration_bytes)
        original.require(all(calibration.get(k) == v for k, v in identity.items())
                         and calibration['matched_b0']
                         and all(calibration['B0']['result'].get(k) == v for k, v in expected.items()),
                         'B0 must match the original fast5 replay first')
        elapsed = calibration['B0']['wall_seconds']
        # B1 retains B0's total node budget; B2/B3 can use ten times as many nodes.
        limits = {name: min(MAX_DEEPEN_SECONDS, max(60, round(TIMING_SLACK * scale * elapsed)))
                  for name, scale in [('B1', 1), ('B2', 10), ('B3', 10)]}
        plan = {**identity, 'calibration_sha256': hashlib.sha256(calibration_bytes).hexdigest(),
                'limits_seconds': limits}
        write_new(args.output_dir / 'deepen-plan.json', plan)
        outcomes = {}
        for name, limit in limits.items():
            outcomes[name] = check(name, limit)
            if 'runtime' in outcomes[name]:
                original.require(outcomes[name]['runtime'] == calibration['B0']['runtime'],
                                 f'{name}: fast5 runtime differs from B0')
            write_new(args.output_dir / f'{name}.json', {**plan, **outcomes[name]})
            print(f"{name} {outcomes[name]['result']['status']} ({outcomes[name]['wall_seconds']} s)", flush=True)
        statuses = {row['result']['status'] for row in outcomes.values()}
        proven = statuses & {'PROVEN_LOSS', 'VCT2_CLEAR'}
        final = ('CONFLICT' if len(proven) > 1 else next(iter(proven)) if proven else
                 'INCOMPLETE' if 'TIMEOUT' in statuses else 'UNKNOWN')
        record = {**plan, 'results': outcomes, 'final': final}
    write_new(output, record)
    summary = {'matched_b0': record['matched_b0'], **record['B0']['result']} if args.stage == 'calibrate' else {
        'final': record['final']}
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
