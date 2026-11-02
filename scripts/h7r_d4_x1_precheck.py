"""R2-D4 X1 desktop checks (read-only): the plan inputs before the analysis, and the written result after it.

    python scripts/h7r_d4_x1_precheck.py                  # every plan input: OK / BAD / ABSENT, then PRECHECK PASS|FAIL
    python scripts/h7r_d4_x1_precheck.py --verify-output  # the X1 result against the plan, the inputs and HEAD
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'docs/h7r/r2/R2-D4-BASELINE-1/x1-plan.json'
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h7r_d4_x1 as x1  # noqa: E402


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
    """Rebuild the source records and re-run all CPU analyses, not just their success markers."""
    import h7r_d4 as cli
    fc = x1.fc
    plan = fc.strict_json((root / PLAN).read_bytes())
    path = root / plan['output']['summary']
    if not path.is_file():
        return [f'{path} is absent']
    try:
        plan, plan_sha = x1.load_plan(root / PLAN, root)
        state = cli.git_state(root)
        before = x1.input_shas(plan, root)
        s = fc.strict_json(path.read_bytes())
        summary_sha = sha256(path)
        contract, csha = x1.d4.load_contract(root / plan['inputs']['contract']['path'], root)
        x1.need(csha == before['contract'], 'contract SHA differs')
        x1.need(isinstance(s, dict), 'X1 summary is not an object')
        x1.need(s.get('format') == x1.FORMAT, 'not an X1 summary')
        x1.need(s.get('plan') == {'path':PLAN, 'sha256':plan_sha}, 'the plan SHA differs')
        x1.need(s.get('x1_git') == state, 'the X1 commit is not the current clean HEAD')
        tool = root / x1.TOOL
        x1.need(s.get('x1_sources') == {x1.TOOL:sha256(tool)}, 'the X1 tool SHA differs')
        expected_inputs = {k:{'path':plan['inputs'][k]['path'], 'sha256':v} for k,v in before.items()}
        x1.need(s.get('inputs') == expected_inputs and s.get('inputs_unchanged') is True, 'input bindings differ')
        x1.need(s.get('nature') == plan['nature'] and s.get('analysis_spec') == plan['analysis'], 'analysis plan differs')
        provenance, rec, predictions, analysis, rows = x1.compute(plan, contract, before,
                                                               root / plan['hdev_cache'], root, log)
        for key, expected in {'provenance':provenance, 'reconstruction':x1.reconstruction_facts(rec),
                              'predictions':predictions, 'analysis':analysis, 'limits':x1.LIMITS}.items():
            problems = fc.differences(s.get(key), expected, key)
            x1.need(not problems, f'recomputed {key} differs: {problems[:5]}')
        row_path = root / plan['output']['rows']
        raw = row_path.read_bytes()
        expected_rows = [x1.public_row(r) for r in rows]
        x1.need(not raw or raw.endswith(b'\n'), 'row output has a torn last line')
        actual_rows = [fc.strict_json(line) for line in raw.splitlines()]
        problems = fc.differences(actual_rows, expected_rows, 'rows')
        x1.need(not problems, f'recomputed rows differ: {problems[:5]}')
        x1.need(s.get('rows') == {'path':plan['output']['rows'], 'sha256':sha256(row_path),
                                 'count':len(expected_rows)}, 'the observation rows differ from the result')
        x1.need(x1.input_shas(plan, root) == before and cli.git_state(root) == state
                 and sha256(path) == summary_sha, 'inputs, HEAD or summary changed during verification')
    except (x1.Failure, OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        problems = [f'{type(exc).__name__}: {exc}']
        for message in problems: log(f'FAIL: {message}', flush=True)
        log('VERIFY FAIL', flush=True)
        return problems
    log(f"X1 commit {state['commit']}; all analyses and {len(expected_rows)} rows recomputed", flush=True)
    log('VERIFY PASS', flush=True)
    return []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--verify-output', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.verify_output:
            return 1 if verify_output(ROOT) else 0
        return 0 if precheck(ROOT) else 1
    except (x1.Failure, OSError, ValueError, KeyError, TypeError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
