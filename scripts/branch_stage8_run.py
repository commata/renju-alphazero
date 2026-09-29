"""Branch a Stage 8 run at a generation into a NEW run directory (no stale results).

Copying a run and resuming from an earlier checkpoint is not enough: resume truncation
only covers ``metrics.jsonl``, ``self_play/`` and ``evaluation/``, while
``external_eval/`` and ``probes/`` of later generations would survive and, because the
orchestrator skips existing outputs, get attached to the NEW trajectory's checkpoints.

This copies only what belongs to generations <= N:

- ``checkpoints/checkpoint_genNNN.pt`` (also as ``latest.pt``; model, optimizer, replay,
  RNGs and generation are the training state, so resuming continues exactly from it);
- ``metrics.jsonl`` events with generation < N (the resume truncation rule);
- ``self_play/`` and ``evaluation/`` files with generation < N;
- ``external_eval/`` and ``probes/`` files with generation <= N (they evaluate saved
  checkpoints; ``color_regression.json`` is recomputed by the orchestrator);
- ``metadata.json`` and ``config.yaml``; plus ``BRANCH.json`` with provenance.

The source run is never modified.

    python scripts/branch_stage8_run.py --source runs/stage8_g3_b --generation 400 \
        --dest runs/stage8_b400_long
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

GEN_FILE = re.compile(r'^gen(\d{3,})')


def _generation(name: str) -> int | None:
    match = GEN_FILE.match(name)
    return int(match.group(1)) if match else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def branch(source: Path, generation: int, dest: Path) -> dict:
    checkpoint = source / 'checkpoints' / f'checkpoint_gen{generation:03d}.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'missing {checkpoint}')
    if not (source / 'metadata.json').is_file():
        raise FileNotFoundError(f'{source} is not a Stage 6+ run directory')
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f'destination is not empty: {dest}')
    (dest / 'checkpoints').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, dest / 'checkpoints' / checkpoint.name)
    shutil.copyfile(checkpoint, dest / 'checkpoints' / 'latest.pt')
    for name in ('metadata.json', 'config.yaml'):
        if (source / name).is_file():
            shutil.copyfile(source / name, dest / name)

    kept_events = 0
    metrics = source / 'metrics.jsonl'
    if metrics.is_file():
        lines = [line for line in metrics.read_text(encoding='utf-8').splitlines()
                 if line.strip() and json.loads(line)['generation'] < generation]
        (dest / 'metrics.jsonl').write_text(''.join(line + '\n' for line in lines),
                                            encoding='utf-8')
        kept_events = len(lines)

    copied = {}
    for sub, limit in (('self_play', generation - 1), ('evaluation', generation - 1),
                       ('external_eval', generation), ('probes', generation)):
        directory = source / sub
        if not directory.is_dir():
            continue
        (dest / sub).mkdir(exist_ok=True)
        count = 0
        for path in sorted(directory.iterdir()):
            g = _generation(path.name)
            if path.is_file() and g is not None and g <= limit and '.bak-' not in path.name:
                shutil.copyfile(path, dest / sub / path.name)
                count += 1
        copied[sub] = count

    record = {'source': str(source), 'generation': generation,
              'checkpoint': str(checkpoint), 'checkpoint_sha256': _sha256(checkpoint),
              'metrics_events_kept': kept_events, 'copied_files': copied}
    (dest / 'BRANCH.json').write_text(json.dumps(record, indent=1), encoding='utf-8')
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(branch(args.source, args.generation, args.dest), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
