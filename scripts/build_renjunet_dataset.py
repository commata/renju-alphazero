"""H2: build the validated RenjuNet game set for Track B (docs/mcts-v8-teacher.md §12.10).

Reads the RenjuNet RIF XML (plain or .gz), classifies every game with one primary
outcome, splits accepted games by tournament (test 5%, val 5%, train rest), masks
val/test policy states whose D4-canonical position already occurs in an earlier
split, re-replays every output game, and writes:

    <out-dir>/games.jsonl.gz   accepted games, one JSON per line, id order (NOT committed:
                               RenjuNet data is licensed for offline, non-commercial use)
    <out-dir>/manifest.json    source and output SHA-256, counts, git commit
    --report                   counts-only validation report (no moves), safe to commit

The gate (exit code 1 if any fails): every game has exactly one outcome and the
counts add up; every output game replays with ``Game``; no unmasked position is
shared between splits; with ``--check-determinism`` a second full build gives the
same output SHA-256.

    python scripts/build_renjunet_dataset.py --rif data/external/renjunet/renjunet_v10_20260930.rif.gz \\
        --out-dir data/external/renjunet --report docs/mcts-v8-results/h2_renjunet_validation.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hybrid.renjunet import build, read_rif, verify_games  # noqa: E402

FORMAT = 'renjunet-h2-v1'


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _open_rif(path: Path):
    return gzip.open(path, 'rb') if path.suffix == '.gz' else path.open('rb')


def _git_commit() -> str | None:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def games_bytes(games) -> bytes:
    keys = ('id', 'tournament', 'rule', 'split', 'bresult', 'end', 'winner', 'moves', 'masked_plies')
    lines = (json.dumps({k: g[k] for k in keys}, separators=(',', ':')) for g in games)
    return ('\n'.join(lines) + '\n').encode('utf-8')


def run_build(rif: Path):
    with _open_rif(rif) as handle:
        db = read_rif(handle)
    games, report = build(db, progress=lambda done, total: print(f'  {done}/{total} games', flush=True))
    return games, report, games_bytes(games)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rif', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'data/external/renjunet')
    parser.add_argument('--report', type=Path, help='counts-only report JSON (commit this, not the games)')
    parser.add_argument('--check-determinism', action='store_true', help='build twice and compare SHA-256')
    args = parser.parse_args(argv)

    started = perf_counter()
    games, report, payload = run_build(args.rif)
    output_sha = hashlib.sha256(payload).hexdigest()
    failures = verify_games(games)
    gate = {
        'sum_check': report['sum_check'],
        'replay_failures': failures,
        'leakage_after_masking': report['leakage_after_masking'],
        'leakage_zero': not any(report['leakage_after_masking'].values()),
    }
    if args.check_determinism:
        _, _, again = run_build(args.rif)
        gate['deterministic'] = hashlib.sha256(again).hexdigest() == output_sha
    gate['passed'] = (gate['sum_check'] and failures == 0 and gate['leakage_zero']
                      and gate.get('deterministic', True))

    manifest = {'format': FORMAT, 'source': args.rif.name, 'source_sha256': _sha256_file(args.rif),
                'output_sha256': output_sha, 'git_commit': _git_commit(),
                'seconds': round(perf_counter() - started, 1), 'gate': gate, 'report': report}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(args.out_dir / 'games.jsonl.gz', 'wb', mtime=0) as handle:  # mtime=0: same bytes every run
        handle.write(payload)
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(json.dumps({'primary_reason': report['primary_reason'], 'splits': report['splits'],
                      'accepted_end': report['accepted_end'], 'gate': gate}, indent=1))
    return 0 if gate['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
