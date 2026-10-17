"""H6-3 offline gate: run once on one test part (docs/h6-value.md, src/hybrid/h6_gate.py).

    python scripts/h6_offline_gate.py --checkpoint runs/h6_value_head_v1/best.pt \\
        --cache-dir data/tactical/h6/value_cache_v2 --part A --output docs/h6/h6_3a_gate.json

One shot: refuses an existing --output, a dirty worktree, a checkpoint whose best.json is not H6-2a or was
trained on other labels than the cache. Part B is the reserve: use it only for a model changed after part A.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hybrid.h6_gate import (  # noqa: E402
    FORMAT, FRESH, THRESHOLDS, d4_average, d4_metrics, fresh_probes, judge, partition, probes, split_metrics, values,
)
from hybrid.h6_value_cache import load_value_cache  # noqa: E402
from hybrid.h6_value_train import file_sha256, text_sha256, value_forward  # noqa: E402
from model.checkpoint import _git_provenance, load_checkpoint  # noqa: E402
from model.config import ModelConfig  # noqa: E402

PROBE_FILE = ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', type=Path, required=True, help='H6-2a best.pt')
    parser.add_argument('--metadata', type=Path, help='default: best.json next to the checkpoint')
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--part', choices=('A', 'B'), required=True, help='A = first gate, B = reserve')
    parser.add_argument('--probe-file', type=Path, default=PROBE_FILE)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-dirty', action='store_true', help='tests / smoke only, recorded')
    parser.add_argument('--d4-average', action='store_true', help='value = mean over the 8 symmetries (model definition)')
    parser.add_argument('--fresh-probes', type=Path, help='sealed fresh probes: G4 uses them instead of P93/P94')
    args = parser.parse_args(argv)
    if args.part == 'B' and (args.fresh_probes is None or not args.d4_average):
        parser.error('part B is the final gate: it needs --fresh-probes and --d4-average (docs/h6-value.md)')
    if args.fresh_probes is not None and \
            json.loads(args.fresh_probes.read_text(encoding='utf-8')).get('purpose', 'final') != 'final':
        parser.error(f'{args.fresh_probes} is a dev probe set, not the sealed final probes')
    if args.output.exists():
        parser.error(f'{args.output} exists: the gate runs once per part')
    commit, dirty = _git_provenance()
    if dirty is not False and not args.allow_dirty:
        parser.error(f'refusing to run from a dirty or unknown worktree (git_dirty={dirty})')
    meta_path = args.metadata or args.checkpoint.with_name('best.json')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('stage') != 'H6-2a' or meta.get('value_trained') is not True:
        parser.error(f'{meta_path} is not an H6-2a checkpoint')
    data, manifest = load_value_cache(args.cache_dir, 'test', verify_hash=True)
    if manifest['labels_sha256'] != meta['labels_sha256']:
        parser.error('the cache was built from other labels than the checkpoint was trained on')
    game_ids = json.loads((args.cache_dir / 'test_games.json').read_text(encoding='utf-8'))
    part, info = partition(data, game_ids, args.part)
    model = load_checkpoint(args.checkpoint, ModelConfig(**meta['model_config']), device=args.device)
    value_fn = lambda planes: value_forward(model, planes)  # noqa: E731  (eval mode throughout)
    if args.d4_average:
        value_fn = d4_average(value_fn, args.device)
    v = values(value_fn, part, args.device)
    metrics = split_metrics(v, part)
    d4 = d4_metrics(value_fn, part, args.device)
    probe = probes(value_fn, args.probe_file, args.device)
    fresh = fresh_probes(value_fn, args.fresh_probes, args.device) if args.fresh_probes else None
    verdict = judge(metrics, d4, probe, fresh)
    result = {'format': FORMAT, 'stage': 'H6-3', 'git_commit': commit, 'git_dirty': dirty,
              'checkpoint_sha256': file_sha256(args.checkpoint), 'metadata_sha256': text_sha256(meta_path),
              'checkpoint_step': meta.get('step'), 'labels_sha256': manifest['labels_sha256'],
              'test_cache_sha256': manifest['splits']['test']['sha256'],
              'probe_file_sha256': text_sha256(args.probe_file), 'partition': info, 'thresholds': THRESHOLDS,
              'inference': 'd4_average' if args.d4_average else 'raw',
              'fresh_probes': fresh and {'sha256': text_sha256(args.fresh_probes), 'criteria': FRESH, **fresh},
              **verdict, 'metrics': metrics, 'd4': d4, 'probes': probe}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('verdict', 'gates', 'failed', 'next_steps_if_failed', 'lambda',
                                              'outcome_ece', 'partition')}, indent=1))
    print(json.dumps({'proof': metrics['proof'], 'outcome': metrics['outcome'],
                      'outcome_by_side': metrics['outcome_by_side'], 'd4': d4,
                      'P93': probe['P93'], 'P94': probe['P94'], 'P92': {k: probe['P92'][k] for k in
                                                                       ('rank', 'legal_moves', 'played_value')}},
                     indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
