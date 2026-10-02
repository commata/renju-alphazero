"""H3 final gate on the selected weights (docs/mcts-v8-teacher.md §12.11).

Run once on the checkpoint chosen by the validation metric (``best.pt``), never
to pick checkpoints:

1. held-out test split: masked top-1/3/5, cross-entropy, accuracy by ply bucket,
   raw illegal diagnostics (not gated);
2. tactical regression: raw policy (masked, no search) on ``stage7_probes_v1.json``.
   Hard gate: must_block top-1 >= the B400 anchor. The anchor is measured with
   ``--anchor`` (a Track A training checkpoint, e.g. stage8_g3_b gen 400) when given;
   otherwise the recorded B400 value 0.23 of 40 probes -> 9/40 is used.
   The proof-label model (about 0.72, v7-nn-integration-review.md §6.3) is a reference,
   not a gate.

    python scripts/eval_h3_gate.py --run-dir runs/h3_policy_64x4 [--anchor <B400 checkpoint>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hybrid.h3_cache import load_cache  # noqa: E402
from hybrid.h3_train import H3Config, evaluate  # noqa: E402
from model.checkpoint import load_checkpoint  # noqa: E402
from model.config import ModelConfig  # noqa: E402
from training.probes import evaluate_probes, load_model_from_training_checkpoint, load_probe_set  # noqa: E402

PROBES = ROOT / 'tests/fixtures/stage7_probes_v1.json'
B400_MUST_BLOCK = 9  # stage8-plan.md §12.10: B400 raw must_block top-1 0.23 on 40 probes


def must_block_hits(model, probes) -> tuple[int, int, dict]:
    result = evaluate_probes(model, probes)
    rows = [r for r in result['rows'] if r['kind'] == 'must_block']
    return sum(bool(r['top1']) for r in rows), len(rows), result['summary']


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', default='best.pt')
    parser.add_argument('--anchor', type=Path, help='Track A training checkpoint to measure the anchor')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)

    meta = json.loads((args.run_dir / args.checkpoint.replace('.pt', '.json')).read_text(encoding='utf-8'))
    cfg = H3Config.from_dict(meta['config'])
    model = load_checkpoint(args.run_dir / args.checkpoint, ModelConfig(channels=cfg.channels, blocks=cfg.blocks),
                            device=args.device)
    test, manifest = load_cache(Path(cfg.cache_dir), 'test', verify_hash=True)
    if manifest['h2_output_sha256'] != meta['h2_output_sha256']:
        raise SystemExit('the cache differs from the one the model was trained on')
    test_metrics = evaluate(model, test, cfg, args.device)

    probes, probe_sha = load_probe_set(PROBES)
    hits, total, summary = must_block_hits(model, probes)
    if args.anchor:
        anchor_model, anchor_info = load_model_from_training_checkpoint(args.anchor)
        anchor_hits, _, anchor_summary = must_block_hits(anchor_model, probes)
        anchor = {'source': 'measured', 'checkpoint': anchor_info, 'must_block_hits': anchor_hits,
                  'summary': anchor_summary}
    else:
        anchor_hits = B400_MUST_BLOCK
        anchor = {'source': 'recorded B400 (stage8-plan.md §12.10)', 'must_block_hits': anchor_hits}
    passed = hits >= anchor_hits
    report = {'format': 'h3-gate-v1', 'checkpoint': str(args.run_dir / args.checkpoint), 'step': meta['step'],
              'value_trained': meta['value_trained'], 'test': test_metrics,
              'probes': {'file': str(PROBES.relative_to(ROOT)), 'sha256': probe_sha, 'summary': summary},
              'must_block': {'hits': hits, 'total': total, 'anchor': anchor, 'passed': passed,
                             'reference_proof_label_model': 0.72}}
    out = args.output or args.run_dir / 'gate.json'
    out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f"test top1 {test_metrics['top1']:.4f} top3 {test_metrics['top3']:.4f} top5 {test_metrics['top5']:.4f} "
          f"ce {test_metrics['ce']:.4f}; must_block {hits}/{total} vs anchor {anchor_hits}: "
          f"{'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
