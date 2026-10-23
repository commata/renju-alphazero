"""WSL2 gate of the frozen external anchor (docs/external-engine.md §8.3): the record that the Windows Rapfi run from
WSL2 is the X1 anchor, built from the gate runs and checked against the committed X1 record.

    python scripts/external_wsl_gate.py --engine-dir /mnt/c/engines/rapfi \\
        --crosscheck runs/external/wsl/crosscheck_s1.json --repro runs/external/wsl/repro_8703.json \\
        --out-dir docs/external/wsl_gate

Checks (all must hold, else ``result: FAIL`` and exit 1):
    files       the executable and the six engine files have the SHA-256 of ``x1_provenance.json``
    crosscheck  seed 1, radius 1, 300 games: the summary of the desktop Windows record (8,886 positions, 9,517
                forbidden points, 0 mismatches), on the same executable
    repro       seed 8703, 2 pairs: X1's ``run_signature`` (engine bytes, settings, our side) and fingerprint
Writes ``crosscheck_s1.json``, ``repro_8703.json`` (copies) and ``manifest.json`` into ``--out-dir``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
X1 = ROOT / 'docs' / 'external' / 'x1_provenance.json'
EXECUTABLE = 'pbrain-rapfi-windows-avx2.exe'
FORMAT = 'external-wsl-gate-v1'


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def expected() -> dict:
    record = json.loads(X1.read_text(encoding='utf-8'))
    repro = next(r for r in record['runs'] if r['file'] == 'repro_a.json')
    return {'executable_sha256': repro['engine']['executable_sha256'], 'extra_files': repro['engine']['extra_files'],
            'crosscheck': record['rule_crosscheck_desktop'], 'repro': {
                'seed': repro['seed'], 'pairs': repro['pairs'], 'run_signature': repro['run_signature'],
                'fingerprint': repro['fingerprint']}}


def evaluate(engine_dir: Path, crosscheck: dict, repro: dict) -> dict:
    want = expected()
    files = {EXECUTABLE: _sha(engine_dir / EXECUTABLE), **{n: _sha(engine_dir / n) for n in want['extra_files']}}
    checks = {
        'executable': files[EXECUTABLE] == want['executable_sha256'],
        'engine_files': all(files[n] == d for n, d in want['extra_files'].items()),
        'crosscheck': (crosscheck.get('seed'), crosscheck.get('radius'), crosscheck.get('summary')) == (
            want['crosscheck']['seed'], want['crosscheck']['radius'], want['crosscheck']['summary'])
            and crosscheck['engine'].get('executable_sha256') == want['executable_sha256'],
        'repro_signature': repro.get('run_signature') == want['repro']['run_signature']
            and (repro.get('seed'), repro.get('pairs')) == (want['repro']['seed'], want['repro']['pairs']),
        'repro_fingerprint': repro['summary'].get('fingerprint') == want['repro']['fingerprint'],
    }
    return {'format': FORMAT, 'result': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
            'files_sha256': files, 'crosscheck': {'summary': crosscheck.get('summary'), 'about':
                                                   crosscheck['engine'].get('about'),
                                                   'git_commit': crosscheck.get('git_commit'),
                                                   'git_dirty': crosscheck.get('git_dirty')},
            'repro': {'run_signature': repro.get('run_signature'), 'fingerprint': repro['summary'].get('fingerprint'),
                      'score': repro['summary'].get('score'), 'git_commit': repro.get('git_commit'),
                      'git_dirty': repro.get('git_dirty'), 'about': repro['engine'].get('about')},
            'expected': want}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--engine-dir', type=Path, required=True)
    parser.add_argument('--crosscheck', type=Path, required=True)
    parser.add_argument('--repro', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    load = (lambda p: json.loads(p.read_text(encoding='utf-8')))
    manifest = evaluate(args.engine_dir, load(args.crosscheck), load(args.repro))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.crosscheck, args.out_dir / 'crosscheck_s1.json')
    shutil.copyfile(args.repro, args.out_dir / 'repro_8703.json')
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'result': manifest['result'], 'checks': manifest['checks']}, indent=1), flush=True)
    return 0 if manifest['result'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
