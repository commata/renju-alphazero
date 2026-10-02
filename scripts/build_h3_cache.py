"""H3: bit-packed policy-state cache from the H2 RenjuNet games (docs/mcts-v8-teacher.md §12.11).

Runs on CPU (the rules engine computes every legal mask once). Writes
``<out-dir>/{train,val,test}.pt`` + ``manifest.json`` (RenjuNet-derived: never commit),
then checks a random sample: cached planes == ``encode_game`` on the replayed game,
and the cached legal mask is D4-equivariant against ``Game`` legality (all 8 symmetries).

    python scripts/build_h3_cache.py --games data/external/renjunet/games.jsonl.gz \\
        --h2-manifest data/external/renjunet/manifest.json \\
        --out-dir data/external/renjunet/h3_cache --workers 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hybrid.h3_cache import build_cache, load_cache, read_games, verify_d4, verify_encoding  # noqa: E402


def _git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--games', type=Path, default=ROOT / 'data/external/renjunet/games.jsonl.gz')
    parser.add_argument('--h2-manifest', type=Path, default=None,
                        help='H2 manifest.json; the games file must match its output_sha256')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'data/external/renjunet/h3_cache')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--verify-samples', type=int, default=300, help='per split, encode + D4 checks')
    parser.add_argument('--report', type=Path, help='counts-only report JSON (safe to commit)')
    args = parser.parse_args(argv)

    started = perf_counter()
    h2 = json.loads(args.h2_manifest.read_text(encoding='utf-8')) if args.h2_manifest else None
    manifest = build_cache(args.games, args.out_dir, workers=args.workers, git_commit=_git_commit(),
                           h2_manifest=h2)
    build_seconds = perf_counter() - started
    games, _ = read_games(args.games)
    checks = {}
    rng = random.Random(1)
    for split in manifest['splits']:
        data, _ = load_cache(args.out_dir, split, verify_hash=True)
        n = int(data['target'].shape[0])
        sample = rng.sample(range(n), min(args.verify_samples, n))
        checks[split] = {'sampled': len(sample), 'encode_mismatch': verify_encoding(data, games, sample),
                         'd4_mismatch': verify_d4(data, sample)}
    passed = all(c['encode_mismatch'] == 0 and c['d4_mismatch'] == 0 for c in checks.values())
    result = {**manifest, 'build_seconds': round(build_seconds, 1), 'checks': checks, 'passed': passed}
    (args.out_dir / 'manifest.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps({'splits': manifest['splits'], 'checks': checks, 'passed': passed,
                      'build_seconds': result['build_seconds']}, indent=1))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
