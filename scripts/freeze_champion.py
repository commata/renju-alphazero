"""Freeze a champion checkpoint: copy it to the anchors and record its provenance.

Writes ``<anchors>/<label>.pt`` (a byte copy) and ``<anchors>/<label>.json`` with the
checkpoint SHA-256, generation, global step, critical config hash, the git commit that
trained it, the self-play search it was trained with, and the SHA-256 of every evidence
file (round robin, heavy evaluations, verdicts) given with ``--evidence``. Refuses to
overwrite an existing anchor with different contents.

    python scripts/freeze_champion.py --checkpoint runs/stage8_s400_c1480/checkpoints/checkpoint_gen1560.pt \\
        --label S400C --anchors-dir runs/anchors \\
        --evidence runs/champion/rr_s400_seed9301.json runs/champion/heavy_S400_*_seed7107.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(checkpoint: Path, label: str, anchors_dir: Path, evidence: list[Path],
           note: str | None = None) -> dict:
    from training.config import self_play_search_config
    from training.training_checkpoint import load_checkpoint_payload

    payload = load_checkpoint_payload(checkpoint)
    digest = sha256(checkpoint)
    target = anchors_dir / f'{label}.pt'
    if target.exists() and sha256(target) != digest:
        raise SystemExit(f'{target} exists with different contents; choose another --label')
    anchors_dir.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(checkpoint, target)
    record = {
        'format': 'champion-freeze-v1', 'label': label, 'frozen_at': datetime.now(timezone.utc)
        .isoformat(timespec='seconds'), 'source': str(checkpoint), 'anchor': str(target),
        'checkpoint_sha256': digest, 'generation': int(payload['generation']),
        'global_step': int(payload['global_step']),
        'critical_config_hash': payload['critical_config_hash'],
        'git_commit': payload.get('git_commit'),
        'self_play_search': self_play_search_config(payload['config']).to_dict(),
        'evidence': [{'path': str(p), 'sha256': sha256(p)} for p in evidence],
        'note': note,
    }
    (anchors_dir / f'{label}.json').write_text(json.dumps(record, indent=1), encoding='utf-8')
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--anchors-dir', type=Path, required=True)
    parser.add_argument('--evidence', nargs='*', default=[], help='files or globs')
    parser.add_argument('--note')
    args = parser.parse_args()
    evidence = sorted({Path(p) for pattern in args.evidence for p in glob.glob(pattern)})
    record = freeze(args.checkpoint, args.label, args.anchors_dir, evidence, args.note)
    print(json.dumps({k: record[k] for k in ('label', 'anchor', 'checkpoint_sha256',
                                             'generation', 'critical_config_hash')}, indent=1))
    print(f"evidence files: {len(record['evidence'])}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
