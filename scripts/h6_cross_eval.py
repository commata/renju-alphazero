"""H6-2a vs H6-2a' on the same states (dev only; docs/h6-value.md, H6-2a' step 6).

The two models were trained and validated on different labels (v2 and v3), so their best.json metrics
are not on the same footing: v3 turned about 1,600 val outcome rows (weight 0.1) into ``vcf_loss`` proofs
(weight 1.0). This script scores every model on the **v3 cache's states**, val and test part A only
(part B and the fresh probes stay sealed), raw and D4-averaged, and splits them by what v3 changed:

    new_loss   proof -1 in v3 whose (game, ply) is not proof -1 in v2: the vcf_loss labels v3 added
    old_loss   proof -1 in both caches (in the H6 labels: forced_loss)
    proof_win  proof +1
    outcome_win / outcome_loss   outcome rows of the v3 cache
    all        every v3 state

``new_loss`` is where v2 saw an outcome -1 at weight 0.1 and v3 a proven loss at 1.0: the states of G4's
kind. The model's own training labels are not checked against the cache (that check belongs to the gate).

    python -u scripts/h6_cross_eval.py --model v2=runs/h6_value_head_v1/best.pt --model v3=runs/h6_value_head_v3/best.pt \\
        --v2-cache data/tactical/h6/value_cache_v2 --v3-cache data/tactical/h6/value_cache_v3 \\
        --device cuda --output docs/h6/h6_2a_prime_cross_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from hybrid.h6_gate import d4_average, partition, values  # noqa: E402
from hybrid.h6_value_cache import KIND_OUTCOME, KIND_PROOF, load_value_cache  # noqa: E402
from hybrid.h6_value_train import file_sha256, text_sha256, value_forward  # noqa: E402
from model.checkpoint import _git_provenance, load_checkpoint  # noqa: E402
from model.config import ModelConfig  # noqa: E402

FORMAT = 'h6-cross-eval-v1'
SETS = ('val', 'testA')


def load_set(cache: Path, name: str) -> tuple[dict, list[str], dict]:
    split = 'test' if name == 'testA' else name
    data, manifest = load_value_cache(cache, split, verify_hash=True)
    game_ids = json.loads((cache / f'{split}_games.json').read_text(encoding='utf-8'))
    info = {'states': int(data['value'].shape[0])}
    if name == 'testA':
        data, info = partition(data, game_ids, 'A')  # part A only: part B stays sealed
    return data, game_ids, {**info, 'labels_sha256': manifest['labels_sha256']}


def keys(data: dict, game_ids: list[str]) -> list[tuple[str, int]]:
    return [(game_ids[g], p) for g, p in zip(data['game'].tolist(), data['ply'].tolist())]


def subsets(v3: dict, v3_games: list[str], v2: dict, v2_games: list[str]) -> dict[str, torch.Tensor]:
    t, kind = v3['value'], v3['kind']
    proof, outcome = kind == KIND_PROOF, kind == KIND_OUTCOME
    v2_loss = {k for k, is_loss in zip(keys(v2, v2_games), ((v2['kind'] == KIND_PROOF) & (v2['value'] < 0)).tolist())
               if is_loss}
    in_v2_loss = torch.tensor([k in v2_loss for k in keys(v3, v3_games)], dtype=torch.bool)
    loss = proof & (t < 0)
    return {'new_loss': loss & ~in_v2_loss, 'old_loss': loss & in_v2_loss, 'proof_win': proof & (t > 0),
            'outcome_win': outcome & (t > 0), 'outcome_loss': outcome & (t < 0),
            'all': torch.ones_like(t, dtype=torch.bool)}


def metrics(v: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {'states': 0}
    vv, tt = v[mask], t[mask]
    signed = tt != 0
    return {'states': n,
            'sign_acc': float(((vv[signed] > 0) == (tt[signed] > 0)).float().mean()) if bool(signed.any()) else None,
            'mean_value': float(vv.mean()), 'mse': float((vv - tt).square().mean()),
            'saturated': float((vv.abs() > 0.95).float().mean())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', action='append', required=True, metavar='NAME=CHECKPOINT',
                        help='an H6-2a best.pt (its best.json next to it); repeat for each model')
    parser.add_argument('--v2-cache', type=Path, required=True)
    parser.add_argument('--v3-cache', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, help='first N states per set (smoke runs)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-dirty', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    commit, dirty = _git_provenance()
    if dirty is not False and not args.allow_dirty:
        parser.error(f'refusing to run from a dirty or unknown worktree (git_dirty={dirty})')
    models = {}
    for spec in args.model:
        name, _, path = spec.partition('=')
        if not name or not path:
            parser.error(f'--model {spec!r}: use NAME=CHECKPOINT')
        checkpoint = Path(path)
        meta_path = checkpoint.with_name('best.json')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('stage') != 'H6-2a':
            parser.error(f'{meta_path} is not an H6-2a checkpoint')
        models[name] = (checkpoint, meta_path, meta)

    started = perf_counter()
    result = {'format': FORMAT, 'dev_only': True, 'git_commit': commit, 'git_dirty': dirty, 'limit': args.limit,
              'sets': {}, 'models': {}}
    for name, (checkpoint, meta_path, meta) in models.items():
        result['models'][name] = {'checkpoint_sha256': file_sha256(checkpoint), 'metadata_sha256': text_sha256(meta_path),
                                  'trained_on_labels_sha256': meta['labels_sha256'], 'step': meta.get('step'),
                                  'sets': {}}
    for set_name in SETS:
        v3, v3_games, v3_info = load_set(args.v3_cache, set_name)
        v2, v2_games, v2_info = load_set(args.v2_cache, set_name)
        if args.limit:
            v3 = {k: v[:args.limit] for k, v in v3.items()}
        masks = subsets(v3, v3_games, v2, v2_games)
        result['sets'][set_name] = {'v3_cache': v3_info, 'v2_cache': v2_info,
                                    'subsets': {k: int(m.sum()) for k, m in masks.items()}}
        print(f"[{set_name}] {v3['value'].shape[0]:,} states, new_loss {int(masks['new_loss'].sum()):,}, "
              f"old_loss {int(masks['old_loss'].sum()):,}", flush=True)
        for name, (checkpoint, _, meta) in models.items():
            model = load_checkpoint(checkpoint, ModelConfig(**meta['model_config']), device=args.device).eval()
            raw_fn = lambda planes, model=model: value_forward(model, planes)  # noqa: E731
            out = {}
            for mode, fn in (('raw', raw_fn), ('d4', d4_average(raw_fn, args.device))):
                v = values(fn, v3, args.device)
                out[mode] = {k: metrics(v, v3['value'], m) for k, m in masks.items()}
            result['models'][name]['sets'][set_name] = out
            d4 = out['d4']
            print(f"[{set_name}] {name}: d4 new_loss acc {d4['new_loss'].get('sign_acc')}, "
                  f"outcome_loss {d4['outcome_loss'].get('sign_acc')}, outcome_win {d4['outcome_win'].get('sign_acc')}, "
                  f"all {d4['all'].get('sign_acc')}  ({perf_counter() - started:.0f}s)", flush=True)
    result['seconds'] = round(perf_counter() - started, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    print(f'wrote {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
