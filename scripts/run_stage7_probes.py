"""Run the fixed Stage 7 tactical/value probe set on training checkpoints.

Examples:
    # every checkpoint_genNNN.pt in a run -> <run>/probes/genNNN.json
    python scripts/run_stage7_probes.py --run-dir runs/stage6_mvp_B_stage7a

    # selected generations only
    python scripts/run_stage7_probes.py --run-dir runs/stage6_mvp_B_stage7a --generations 3 5 10

    # one explicit checkpoint
    python scripts/run_stage7_probes.py --checkpoint runs/x/checkpoints/latest.pt --output out.json

The probe measures the raw network (no search), so it runs in seconds per checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import torch  # noqa: E402

from training.probes import run_probe_file  # noqa: E402

DEFAULT_PROBES = ROOT / 'tests' / 'fixtures' / 'stage7_probes_v1.json'
CHECKPOINT_RE = re.compile(r'checkpoint_gen(\d{3})\.pt$')


def run_checkpoints(run_dir: Path, generations: list[int] | None) -> list[tuple[int, Path]]:
    found = []
    for path in sorted((run_dir / 'checkpoints').glob('checkpoint_gen*.pt')):
        match = CHECKPOINT_RE.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    if generations is not None:
        wanted = set(generations)
        missing = wanted - {g for g, _ in found}
        if missing:
            raise SystemExit(f'missing checkpoints for generations: {sorted(missing)}')
        found = [(g, p) for g, p in found if g in wanted]
    if not found:
        raise SystemExit(f'no checkpoint_genNNN.pt files under {run_dir / "checkpoints"}')
    return found


def summary_line(result: dict) -> str:
    parts = [f"gen {result['generation']:3d}"]
    for kind, s in result['summary'].items():
        if 'mass_lift' in s:
            parts.append(f"{kind}: top1 {s['top1']:.2f} lift {s['mass_lift']:.1f}")
        if 'avoid_lift' in s:
            parts.append(f"{kind}: lift {s['avoid_lift']:.1f}")
    v = result['value_overall']
    if 'separation' in v:
        parts.append(f"value sep {v['separation']:+.2f} bal-acc {v['balanced_sign_accuracy']:.2f}")
    f = result['forbidden_diagnostic']
    if f.get('positions'):
        parts.append(f"forbidden raw lift {f['raw_lift']:.1f}")
    return ' | '.join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path)
    source.add_argument('--checkpoint', type=Path)
    parser.add_argument('--generations', type=int, nargs='+',
                        help='with --run-dir: only these checkpoint generations')
    parser.add_argument('--probes', type=Path, default=DEFAULT_PROBES)
    parser.add_argument('--suffix', default='',
                        help='with --run-dir: write probes/genNNN<suffix>.json (e.g. _defense '
                             'for tests/fixtures/stage7_probes_defense_v1.json)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='with --run-dir: skip generations whose output already exists')
    parser.add_argument('--output', type=Path,
                        help='with --checkpoint: output JSON (default: print only)')
    args = parser.parse_args()
    if args.generations is not None and args.run_dir is None:
        parser.error('--generations requires --run-dir')
    if args.output is not None and args.run_dir is not None:
        parser.error('--output is for --checkpoint; --run-dir writes <run>/probes/')

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.checkpoint is not None:
        result = run_probe_file(args.checkpoint, args.probes)
        print(summary_line(result))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=1), encoding='utf-8')
        return 0

    out_dir = args.run_dir / 'probes'
    out_dir.mkdir(parents=True, exist_ok=True)
    for generation, path in run_checkpoints(args.run_dir, args.generations):
        target = out_dir / f'gen{generation:03d}{args.suffix}.json'
        if args.skip_existing and target.exists():
            continue
        result = run_probe_file(path, args.probes)
        target.write_text(
            json.dumps(result, indent=1), encoding='utf-8')
        print(summary_line(result), flush=True)
    print(f'wrote {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
