"""H6-3D diagnostics (dev only): D1 D4 decomposition, D2 engine deep losses, D3 D4-averaged inference.

    python scripts/h6_diagnose.py --checkpoint runs/h6_value_head_v1/best.pt \\
        --cache-dir data/tactical/h6/value_cache_v2 --device cuda --output docs/h6/h6_3d_diag.json

Uses test part A (a dev set since the H6-3A FAIL) and the E1 / E1-holdout / E1-S manifests; never part B.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hybrid.h6_diag import FORMAT, d1, d2, d3, deep_loss_children, latency  # noqa: E402
from hybrid.h6_gate import partition  # noqa: E402
from hybrid.h6_value_cache import load_value_cache, planes_from_value_cache  # noqa: E402
from hybrid.h6_value_train import file_sha256, text_sha256, value_forward  # noqa: E402
from model.checkpoint import _git_provenance, load_checkpoint  # noqa: E402
from model.config import ModelConfig  # noqa: E402

PROBE_FILE = ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json'


def default_manifests() -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(ROOT / 'docs' / 'mcts-v8-results' / 'e1*' / '*manifest.json')))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, nargs='*', default=None)
    parser.add_argument('--probe-file', type=Path, default=PROBE_FILE)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, help='first N part-A states (smoke runs)')
    parser.add_argument('--latency-repeats', type=int, default=200)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-dirty', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    commit, dirty = _git_provenance()
    if dirty is not False and not args.allow_dirty:
        parser.error(f'refusing to run from a dirty or unknown worktree (git_dirty={dirty})')
    meta_path = args.metadata or args.checkpoint.with_name('best.json')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('stage') != 'H6-2a':
        parser.error(f'{meta_path} is not an H6-2a checkpoint')
    data, manifest = load_value_cache(args.cache_dir, 'test', verify_hash=True)
    if manifest['labels_sha256'] != meta['labels_sha256']:
        parser.error('the cache was built from other labels than the checkpoint was trained on')
    game_ids = json.loads((args.cache_dir / 'test_games.json').read_text(encoding='utf-8'))
    part, info = partition(data, game_ids, 'A')  # part A only: B stays sealed
    if args.limit:
        part = {k: v[:args.limit] for k, v in part.items()}
    model = load_checkpoint(args.checkpoint, ModelConfig(**meta['model_config']), device=args.device)
    value_fn = lambda planes: value_forward(model, planes)  # noqa: E731
    started = perf_counter()
    print(f"[D1] {part['value'].shape[0]:,} part-A states x 8 symmetries", flush=True)
    d1_result, v8 = d1(model, part, args.device)
    print(f'[D1] done {perf_counter() - started:.0f}s; [D2] engine deep losses', flush=True)
    manifests = args.manifests if args.manifests is not None else default_manifests()
    rows = deep_loss_children(manifests)
    d2_result = d2(value_fn, rows, args.device)
    print('[D3] raw vs D4 average, latency', flush=True)
    d3_result = d3(v8, part, d2_result, args.probe_file, value_fn, args.device)
    import torch
    sample, _, _ = planes_from_value_cache(part, torch.arange(min(256, part['value'].shape[0])), args.device)
    d3_result['latency'] = latency(value_fn, sample, args.device, repeats=args.latency_repeats)
    result = {'format': FORMAT, 'stage': 'H6-3D', 'dev_only': True, 'git_commit': commit, 'git_dirty': dirty,
              'checkpoint_sha256': file_sha256(args.checkpoint), 'metadata_sha256': text_sha256(meta_path),
              'labels_sha256': manifest['labels_sha256'], 'partition': info, 'limit': args.limit,
              'manifests': {str(p.resolve().relative_to(ROOT)).replace('\\', '/') if p.resolve().is_relative_to(ROOT)
                            else str(p): text_sha256(p) for p in manifests},
              'D1': d1_result, 'D2': d2_result, 'D3': d3_result, 'seconds': perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'D1': {k: d1_result[k] for k in ('end_to_end_std', 'trunk_equivariance', 'value_head_only_std',
                                                        'policy_top1_all8_agree', 'sign_agreement_all8')},
                      'D2': {k: d2_result.get('raw', {}).get(k) for k in ('moves', 'unique_children', 'roots',
                                                                         'move_accuracy', 'unique_child_accuracy',
                                                                         'root_clustered_accuracy', 'by_lost_depth')},
                      'D3': d3_result}, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
