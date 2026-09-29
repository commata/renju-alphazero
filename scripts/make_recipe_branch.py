"""Recipe arm: branch a run at generation N with changed training-critical settings.

Resuming refuses a training-critical config change, and a new run from exported
weights also resets the replay buffer, optimizer and RNGs. For a paired comparison
with the plain continuation (the control arm), this script keeps everything of
generation N except the config:

1. ``branch_stage8_run.branch`` copies generation N and the results of generations
   <= N (the weights are unchanged, so the generation-N results stay valid);
2. the checkpoint's config is replaced by ``--config`` and its critical hash is
   recomputed. The model architecture must not change; every critical difference is
   printed and recorded in ``RECIPE.json``.

``--in-place`` applies step 2 to an existing branch directory instead (for example a
teacher branch from ``make_teacher_branch.py``), so recipe and teacher changes can be
combined.

    python scripts/make_recipe_branch.py --source runs/stage8_g3_b --generation 400 \
        --config configs/stage8_b400_temp4.yaml --dest runs/stage8_b400_temp4
    python scripts/run_stage8_training.py --run-dir runs/stage8_b400_temp4 \
        --config configs/stage8_b400_temp4.yaml --anchor 400 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from branch_stage8_run import branch  # noqa: E402
from training.config import (config_differences, critical_config,  # noqa: E402
                             critical_config_hash, load_config, validate_config)
from training.training_checkpoint import load_checkpoint_payload, save_atomic  # noqa: E402

RECIPE_FORMAT = 'recipe-branch-v1'


def apply_recipe(dest: Path, generation: int, config_path: Path) -> dict:
    new_config = validate_config(load_config(config_path))
    latest = dest / 'checkpoints' / 'latest.pt'
    payload = load_checkpoint_payload(latest)
    if int(payload['generation']) != generation:
        raise ValueError(f"{latest} is generation {payload['generation']}, not {generation}")
    old = payload['config']
    if old['model'] != new_config['model']:
        raise ValueError('the model architecture must not change in a recipe branch')
    diffs = config_differences(critical_config(old), critical_config(new_config))
    if not diffs:
        raise ValueError('the new config has no training-critical difference')
    changes = {}
    for diff in diffs:
        before, after = old, new_config
        for key in diff.split('.'):
            before = before.get(key) if isinstance(before, dict) else None
            after = after.get(key) if isinstance(after, dict) else None
        changes[diff] = {'from': before, 'to': after}
    payload['config'] = new_config
    payload['critical_config_hash'] = critical_config_hash(new_config)
    save_atomic(dest / 'checkpoints' / f'checkpoint_gen{generation:03d}.pt', payload)
    save_atomic(latest, payload)
    return {'config': str(config_path), 'critical_changes': changes,
            'critical_config_hash': payload['critical_config_hash'],
            'checkpoint_sha256': hashlib.sha256(latest.read_bytes()).hexdigest()}


def make_recipe_branch(source: Path | None, generation: int, config_path: Path, dest: Path,
                       *, in_place: bool = False) -> dict:
    record = None if in_place else branch(source, generation, dest)
    result = {'format': RECIPE_FORMAT, 'branch': record, 'in_place': in_place,
              **apply_recipe(dest, generation, config_path)}
    history = []
    recipe_file = dest / 'RECIPE.json'
    if recipe_file.is_file():
        history = json.loads(recipe_file.read_text(encoding='utf-8')).get('history', [])
    result['history'] = history + [result['critical_changes']]
    recipe_file.write_text(json.dumps(result, indent=1), encoding='utf-8')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--in-place', action='store_true',
                        help='change the config of an existing branch in --dest')
    args = parser.parse_args()
    if not args.in_place and args.source is None:
        parser.error('--source is required unless --in-place')
    result = make_recipe_branch(args.source, args.generation, args.config, args.dest,
                                in_place=args.in_place)
    print(json.dumps(result['critical_changes'], indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
