"""Compare two Stage 6 runs (e.g. continuous vs stop/resume) for exact equivalence.

Compares the final latest.pt (model, optimizer, replay buffer, generation/global_step,
component RNGs) with torch.equal / ==, and per-generation self-play record hashes and
evaluation games. Opponents listed in --skip-opponent are excluded from the evaluation
comparison (e.g. mcts_v6 when only one run played it).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import torch  # noqa: E402

from training.training_checkpoint import load_checkpoint_payload  # noqa: E402
from training.training_state import COMPONENT_RNGS  # noqa: E402


def tree_equal(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def per_generation(run: Path, kind: str, skip: set[str]) -> dict:
    result = {}
    for path in sorted((run / kind).glob('gen*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if kind == 'self_play':
            result[path.stem] = [(g['record_sha256'], g['record']['moves']) for g in data['games']]
        else:
            result[path.stem] = {name: [(g['moves'], g['result']) for g in o['games']]
                                 for name, o in data['opponents'].items() if name not in skip}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_a', type=Path)
    parser.add_argument('run_b', type=Path)
    parser.add_argument('--skip-opponent', action='append', default=[])
    args = parser.parse_args()
    a = load_checkpoint_payload(args.run_a / 'checkpoints' / 'latest.pt')
    b = load_checkpoint_payload(args.run_b / 'checkpoints' / 'latest.pt')
    skip = set(args.skip_opponent)
    report = {
        'generation': [a['generation'], b['generation']],
        'global_step': [a['global_step'], b['global_step']],
        'model_state_equal': tree_equal(a['model_state_dict'], b['model_state_dict']),
        'optimizer_state_equal': tree_equal(a['optimizer_state_dict'], b['optimizer_state_dict']),
        'replay_buffer_equal': tree_equal(a['replay_buffer'], b['replay_buffer']),
        'replay_buffer_size': int(a['replay_buffer']['states'].shape[0]),
        'component_rng_equal': {n: a['component_rng'][n] == b['component_rng'][n]
                                for n in COMPONENT_RNGS},
        'critical_config_hash_equal': a['critical_config_hash'] == b['critical_config_hash'],
        'comparison': 'exact (torch.equal / ==)',
    }
    sp_a, sp_b = (per_generation(r, 'self_play', skip) for r in (args.run_a, args.run_b))
    ev_a, ev_b = (per_generation(r, 'evaluation', skip) for r in (args.run_a, args.run_b))
    report['self_play_equal'] = {g: sp_a[g] == sp_b.get(g) for g in sp_a}
    report['self_play_record_sha256'] = {g: [h for h, _ in sp_a[g]] for g in sp_a}
    report['evaluation_equal'] = {g: ev_a[g] == ev_b.get(g) for g in ev_a}
    report['evaluation_skipped_opponents'] = sorted(skip)
    ok = (report['generation'][0] == report['generation'][1]
          and report['global_step'][0] == report['global_step'][1]
          and report['model_state_equal'] and report['optimizer_state_equal']
          and report['replay_buffer_equal'] and all(report['component_rng_equal'].values())
          and report['critical_config_hash_equal'] and all(report['self_play_equal'].values())
          and all(report['evaluation_equal'].values()))
    report['result'] = 'EQUAL' if ok else 'DIFFERENT'
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
