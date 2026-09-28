"""Export model weights from a Stage 6/7 training checkpoint (Stage 7 plan §7).

Writes a Stage 4 model checkpoint (``model.checkpoint.save_checkpoint`` format) that
``training.init_checkpoint`` accepts, plus a ``<output>.json`` provenance sidecar, and
verifies the round trip: the reloaded ``state_dict`` must equal the source exactly.

Training-critical changes (search, games per generation, ...) cannot resume an old
run, so each Stage 7-B arm is a NEW run initialized from these exported weights.

    python scripts/export_stage7_weights.py runs/stage7a/checkpoints/checkpoint_gen030.pt \
        --output runs/stage7b_init/stage7a_gen030_weights.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import torch  # noqa: E402

from model.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from model.config import ModelConfig  # noqa: E402
from training.probes import load_model_from_training_checkpoint  # noqa: E402
from training.training_checkpoint import load_checkpoint_payload  # noqa: E402

EXPORT_FORMAT = 'stage7-weights-export-v1'


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_weights(source: Path, output: Path) -> dict:
    payload = load_checkpoint_payload(source)
    model, info = load_model_from_training_checkpoint(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, model)
    reloaded = load_checkpoint(output, ModelConfig(**payload['model_config']))
    original = model.state_dict()
    restored = reloaded.state_dict()
    if original.keys() != restored.keys() or not all(
            torch.equal(original[k], restored[k]) for k in original):
        raise RuntimeError('exported weights do not round-trip exactly')
    provenance = {
        'format_version': EXPORT_FORMAT,
        'source_checkpoint': str(source),
        'source_checkpoint_sha256': info['checkpoint_sha256'],
        'source_generation': info['generation'],
        'source_global_step': info['global_step'],
        'source_critical_config_hash': info['critical_config_hash'],
        'source_git_commit': info['git_commit'],
        'model_config': payload['model_config'],
        'model_contract': payload['model_contract'],
        'weights': str(output),
        'weights_sha256': sha256_file(output),
        'round_trip_state_dict_equal': True,
    }
    output.with_suffix(output.suffix + '.json').write_text(
        json.dumps(provenance, indent=2), encoding='utf-8')
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, help='training checkpoint (checkpoint_genNNN.pt)')
    parser.add_argument('--output', type=Path, required=True, help='model checkpoint to write')
    parser.add_argument('--expect-source-sha256',
                        help='refuse to export unless the source checkpoint has this SHA-256')
    args = parser.parse_args()
    if args.expect_source_sha256 is not None:
        actual = sha256_file(args.source)
        if actual != args.expect_source_sha256:
            print(f'source SHA-256 mismatch: expected {args.expect_source_sha256}, '
                  f'got {actual}', file=sys.stderr)
            return 1
    print(json.dumps(export_weights(args.source, args.output), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
