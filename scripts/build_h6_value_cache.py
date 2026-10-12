"""H6-2a value cache from the H6-1 label rows (docs/h6-value.md).

Keeps the labelled rows only (``value_mask`` False), with the split, value (side to move) and weight
H6-1 wrote; the legal mask is computed once. With ``--report`` the H6-1 report's ``labelled`` count
must match and its hash goes into the manifest (the trainer checks it). RenjuNet-derived: the cache
goes under data/ or runs/ and is never committed.

    python scripts/build_h6_value_cache.py --labels data/tactical/h6/renjunet_labels_v1.jsonl.gz \\
        --report docs/h6/h6_1_report.json --out-dir data/tactical/h6/value_cache_v1 --workers 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hybrid.h6_value_cache import build_value_cache  # noqa: E402


def _git_commit() -> str | None:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--labels', type=Path, required=True, help='H6-1 rows (.jsonl.gz)')
    parser.add_argument('--report', type=Path, help='H6-1 report (its labelled count must match)')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    out = args.out_dir.resolve()
    if out.is_relative_to(ROOT) and not str(out.relative_to(ROOT)).startswith(('data', 'runs')):
        parser.error('RenjuNet-derived caches go under data/ or runs/ (both ignored by git), never into the repo')
    if (out / 'manifest.json').exists():
        parser.error(f'{out} already holds a cache; use another --out-dir')
    started = perf_counter()
    manifest = build_value_cache(args.labels, out, report=args.report, workers=args.workers,
                                 git_commit=_git_commit())
    print(json.dumps({'labels_sha256': manifest['labels_sha256'], 'h6_report_sha256': manifest['h6_report_sha256'],
                      'splits': {k: {kk: vv for kk, vv in v.items() if kk != 'sha256'}
                                 for k, v in manifest['splits'].items()},
                      'seconds': round(perf_counter() - started, 1)}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
