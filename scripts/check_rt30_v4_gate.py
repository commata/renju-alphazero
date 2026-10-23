"""TB-RT30-v4 same-move gate: v4 (v3 with TB-V3's native builder) searches exactly as TB-RT30-v3 did.

v4 is v3's configuration and code; only ``src/rt30/native/build.py`` differs (it also builds fast4's
``_forbidden``). The gate replays positions without a deadline (no time in any decision) and requires the same
move and the same search record as v3 on every one, fixed before the runs:
    A  v3's committed equivalence record (``docs/rt30-results/v3_equivalence_h3_8.json``: the 8 H3 positions,
       arms frozen / fast3 / rt3) replayed on v4: every arm, every position
    B  representative SER5 sets (``--pair V3 V4``, e.g. ``slow-stratified-v1``, ``critical-regressions-v1``):
       arm rt3 on a v3 checkout and on this v4 checkout, the same positions, seed and machine
Compared per position: the move, route, nodes (attack / stage VCT / root / VCT2), the checked statuses and the tree
move. Times are not compared; the helper's other ``rt`` fields are reported only.
Provenance: every v4 run's commit holds exactly v4's lock (``tests/frozen_tb_rt30_v4.sha256``), every v3 run's
commit exactly v3's (``tests/frozen_tb_rt30_v3.sha256``, the old builder included), read through git.

    python scripts/check_rt30_v4_gate.py --v4-equivalence runs/v4_gate/v4_equivalence_h3_8.json \\
        --pair runs/v4_gate/v3_slow.json runs/v4_gate/v4_slow.json \\
        --pair runs/v4_gate/v3_critical.json runs/v4_gate/v4_critical.json --output docs/rt30-results/v4_same_move.json

PASS writes ``verdict: V4_SAME_MOVE_PASS``; ``rt30.legacy.status`` then reads v4 as ``legacy-certified``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'tb-rt30-v4-same-move-v1'
V3_RECORD = ROOT / 'docs' / 'rt30-results' / 'v3_equivalence_h3_8.json'
V3_LOCK = ROOT / 'tests' / 'frozen_tb_rt30_v3.sha256'
V4_LOCK = ROOT / 'tests' / 'frozen_tb_rt30_v4.sha256'
FIELDS = ('move', 'route', 'nodes', 'checked', 'tree_move')
EQUIVALENCE_ARMS = ['frozen', 'fast3', 'rt3']


def lock_entries(path: Path) -> dict[str, str]:
    return dict(reversed(line.split(maxsplit=1)) for line in path.read_text(encoding='utf-8').splitlines()
                if line.strip() and not line.startswith('#'))


def _blob_sha(commit: str, path: str) -> str | None:
    done = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=ROOT, capture_output=True, check=False)
    return hashlib.sha256(done.stdout).hexdigest() if done.returncode == 0 else None


def check_commit(commit: str | None, lock: dict[str, str], what: str, blob=_blob_sha) -> None:
    if not commit:
        raise ValueError(f'{what}: the run records no git commit')
    bad = [p for p, digest in lock.items() if blob(commit, p) != digest]
    if bad:
        raise ValueError(f'{what}: commit {commit[:10]} is not that engine ({len(bad)} locked files differ: {bad[:3]})')


def check_run(run: dict, what: str, arms: list[str]) -> None:
    if (run.get('format') != 'rt30-replay-v1' or run.get('engine') != 'TB-Engine-v1' or list(run.get('arms', []))
            != arms or run.get('deadline') is not None or run.get('native_scan') is not True
            or run.get('root_rescue') is not True):
        raise ValueError(f'{what}: not an H3 replay without a deadline on the native scan with arms {arms}')


def compare(a: dict, b: dict, arm_a: str, arm_b: str) -> dict:
    """Rows of the same positions: the mismatching keys and fields."""
    rows_a, rows_b = {r['key']: r for r in a['rows']}, {r['key']: r for r in b['rows']}
    if set(rows_a) != set(rows_b):
        raise ValueError('the two runs did not replay the same positions')
    mismatches, reported = [], []
    for key in sorted(rows_a):
        x, y = rows_a[key][arm_a], rows_b[key][arm_b]
        fields = [f for f in FIELDS if x.get(f) != y.get(f)]
        if fields:
            mismatches.append({'key': key, 'fields': fields, 'v3': {f: x.get(f) for f in fields},
                               'v4': {f: y.get(f) for f in fields}})
        rt = sorted(k for k in set(x.get('rt', {})) | set(y.get('rt', {}))
                    if 'seconds' not in k and x.get('rt', {}).get(k) != y.get('rt', {}).get(k))
        if rt:
            reported.append({'key': key, 'rt_fields': rt})
    return {'positions': len(rows_a), 'mismatches': mismatches, 'rt_differences': reported}


def evaluate(v4_equivalence: dict, pairs: list[tuple[dict, dict]], *, v3_record: dict, v3_lock: dict, v4_lock: dict,
             blob=_blob_sha) -> dict:
    check_run(v4_equivalence, 'v4 equivalence', EQUIVALENCE_ARMS)
    check_commit(v4_equivalence.get('git_commit'), v4_lock, 'v4 equivalence', blob)
    parts = {'equivalence': {arm: compare(v3_record, v4_equivalence, arm, arm) for arm in EQUIVALENCE_ARMS}}
    if not pairs:
        raise ValueError('pass at least one --pair V3 V4 of representative SER5 positions')
    for v3, v4 in pairs:
        name = (v4.get('manifest') or {}).get('version') or 'sample'
        check_run(v3, f'{name} v3', ['rt3'])
        check_run(v4, f'{name} v4', ['rt3'])
        check_commit(v3.get('git_commit'), v3_lock, f'{name} v3', blob)
        check_commit(v4.get('git_commit'), v4_lock, f'{name} v4', blob)
        same = ('manifest', 'select', 'sample', 'seed', 'timing', 'machine')
        if any(v3.get(k) != v4.get(k) for k in same if k != 'machine') or \
                {k: v3['machine'].get(k) for k in ('platform', 'python')} != \
                {k: v4['machine'].get(k) for k in ('platform', 'python')}:
            raise ValueError(f'{name}: the v3 and v4 runs differ in positions, seed, timing or machine')
        parts[name] = compare(v3, v4, 'rt3', 'rt3')
    counts = [p for part in parts.values() for p in (part.values() if 'positions' not in part else [part])]
    positions = sum(p['positions'] for p in counts)
    mismatched = sum(len(p['mismatches']) for p in counts)
    return {'verdict': 'V4_SAME_MOVE_PASS' if mismatched == 0 else 'V4_SAME_MOVE_FAIL',
            'compared': positions, 'mismatched': mismatched, 'parts': parts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--v4-equivalence', type=Path, required=True,
                        help='run_rt30_replay.py --arms frozen,fast3,rt3 on the 8 keys of v3_equivalence_h3_8, here')
    parser.add_argument('--pair', type=Path, nargs=2, action='append', default=[], metavar=('V3', 'V4'),
                        help='run_rt30_replay.py --arms rt3 --keys-file SET on a v3 checkout and on this one')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    load = (lambda p: json.loads(Path(p).read_text(encoding='utf-8')))
    try:
        result = evaluate(load(args.v4_equivalence), [(load(a), load(b)) for a, b in args.pair],
                          v3_record=load(V3_RECORD), v3_lock=lock_entries(V3_LOCK), v4_lock=lock_entries(V4_LOCK))
    except ValueError as exc:
        parser.error(str(exc))
    sha = (lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    payload = {'format': FORMAT, 'engine': 'TB-RT30-v4', 'base_engine': 'TB-RT30-v3', 'v4_lock': lock_entries(V4_LOCK),
               'v3_builder_sha256': lock_entries(V3_LOCK)['src/rt30/native/build.py'],
               'inputs': {str(p): sha(p) for p in [args.v4_equivalence, *[x for pair in args.pair for x in pair]]},
               **result}
    print(json.dumps({k: v for k, v in payload.items() if k != 'parts'}, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0 if result['verdict'] == 'V4_SAME_MOVE_PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
