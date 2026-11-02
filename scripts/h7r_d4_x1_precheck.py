"""R2-D4 X1 desktop checks (read-only): the plan inputs before the analysis, and the written result after it.

    python scripts/h7r_d4_x1_precheck.py                  # every plan input: OK / BAD / ABSENT, then PRECHECK PASS|FAIL
    python scripts/h7r_d4_x1_precheck.py --verify-output  # the X1 result against the plan, the inputs and HEAD
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'docs/h7r/r2/R2-D4-BASELINE-1/x1-plan.json'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def precheck(root: Path, log=print) -> bool:
    plan = json.loads((root / PLAN).read_text(encoding='utf-8'))
    bad = 0
    for key, ref in plan['inputs'].items():
        path = root / ref['path']
        got = sha256(path) if path.is_file() else 'ABSENT'
        status = 'OK' if got == ref['sha256'] else ('ABSENT' if got == 'ABSENT' else 'BAD')
        bad += status != 'OK'
        log(f'{status:6s} {key:16s} {ref["path"]} {got}', flush=True)
    log(f"PRECHECK {'PASS' if not bad else 'FAIL'} ({len(plan['inputs']) - bad}/{len(plan['inputs'])})", flush=True)
    return not bad


def verify_output(root: Path, log=print) -> list[str]:
    plan = json.loads((root / PLAN).read_text(encoding='utf-8'))
    path = root / plan['output']['summary']
    if not path.is_file():
        return [f'{path} is absent']
    s = json.loads(path.read_text(encoding='utf-8'))
    problems = []

    def need(ok, message):
        if not ok:
            problems.append(message)
    need(s.get('format') == 'h7r-d4-x1-summary-v1', 'not an X1 summary')
    need(s.get('plan', {}).get('sha256') == sha256(root / PLAN), 'the plan SHA differs')
    need(set(s.get('inputs', {})) == set(plan['inputs']), 'the input list differs from the plan')
    for key, ref in plan['inputs'].items():
        p = root / ref['path']
        need(p.is_file() and sha256(p) == ref['sha256'] == s['inputs'].get(key, {}).get('sha256'),
             f'input {key}: SHA differs (plan, result or file)')
    need(s.get('inputs_unchanged') is True, 'inputs_unchanged is not true')
    r = s.get('reconstruction', {})
    need(r.get('summary_match') is True and r.get('reading') == 'NOT_ESTABLISHED',
         'the reconstruction does not match the D4 summary / reading')
    need(s.get('provenance', {}).get('run', {}).get('run_commit') == plan['run_commit'], 'another run commit')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True).stdout.strip()
    need(s.get('x1_git') == {'commit': head, 'dirty': False}, 'the X1 commit is not the current clean HEAD')
    rows = root / plan['output']['rows']
    need(rows.is_file() and s['rows']['path'] == plan['output']['rows'] and sha256(rows) == s['rows']['sha256']
         and len(rows.read_bytes().splitlines()) == s['rows']['count'], 'the observation rows differ from the result')
    tool = root / 'scripts/h7r_d4_x1.py'
    need(s.get('x1_sources') == {'scripts/h7r_d4_x1.py': sha256(tool)}, 'the X1 tool SHA differs')
    for p in problems:
        log(f'FAIL: {p}', flush=True)
    if not problems:
        log(f"X1 commit {head}; tool {s['x1_sources']['scripts/h7r_d4_x1.py']}; rows {s['rows']['count']}", flush=True)
    log(f"VERIFY {'PASS' if not problems else 'FAIL'}", flush=True)
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--verify-output', action='store_true')
    args = parser.parse_args(argv)
    if args.verify_output:
        return 1 if verify_output(ROOT) else 0
    return 0 if precheck(ROOT) else 1


if __name__ == '__main__':
    sys.exit(main())
