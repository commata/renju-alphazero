"""H6-2a' option B: dev deep-probe pre-check before the one-shot final gate (docs/h6-value.md).

Scores the frozen H6-2a' (v3) checkpoint, and v2 for reference, on the dev deep probes
(``make_h6_fresh_probes.py --dev``, seeds 8611-8616) with **D4-averaged** inference only, deep wins and deep
losses separately. The rule was fixed before any result was seen (``RULE``), and only the ``decide`` model
counts:

    PASS   deep-win accuracy >= 0.80 and deep-loss accuracy >= 0.80, each over >= 20 probes
           -> the final gate (part B + sealed fresh probes + D4), with no tuning in between
    FAIL   either sign below 0.80, or fewer than 20 probes in a sign
           -> the final set stays unopened; H6-2b / H7 (better deep labels)

No retuning (LR, epochs, loss weights) on these probes: the frozen checkpoint is checked by its sha256.

    python -u scripts/h6_dev_probe_eval.py --model v3=runs/h6_value_head_v3/best.pt \\
        --model v2=runs/h6_value_head_v1/best.pt --probes docs/h6/h6_fresh_dev_probes_v1.json \\
        --device cuda --output docs/h6/h6_dev_probe_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid.h6_labels import DEV_PROBE_EXTENSION_SEEDS, DEV_PROBE_SEEDS  # noqa: E402

FORMAT = 'h6-dev-probe-eval-v1'
RULE = {'inference': 'd4_average', 'sign_accuracy_min': 0.80, 'min_probes_per_sign': 20, 'decide': 'v3'}
# The H6-2a' (v3) checkpoint, frozen before the dev probes were made.
FROZEN_V3_SHA256 = 'a0fcccc401a833a08994e21cd7e963d5e7adc7a7cff715de14259bbed733ceca'


def verdict(fresh: dict) -> dict:
    """PASS / FAIL of one model's ``fresh_probes`` result under ``RULE``, with n / N per sign."""
    out, ok = {}, True
    for name in ('win', 'loss'):
        s = fresh[name]
        correct = sum(r['correct'] for r in fresh['probes'] if r['expected_sign'] == (1 if name == 'win' else -1))
        enough = s['probes'] >= RULE['min_probes_per_sign']
        good = s['accuracy'] is not None and s['accuracy'] >= RULE['sign_accuracy_min']
        out[name] = {'correct': correct, 'probes': s['probes'], 'games': s['games'], 'accuracy': s['accuracy'],
                     'mean_value': s['mean_value'], 'enough_probes': enough, 'pass': enough and good,
                     'text': f"{correct} / {s['probes']} = {100 * s['accuracy']:.1f}%" if s['probes'] else '0 / 0'}
        ok = ok and enough and good
    out['verdict'] = 'PASS' if ok else 'FAIL'
    return out


def check_probe_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('purpose') != 'dev' or data.get('sealed') is not False:
        raise ValueError(f'{path} is not a dev probe file (the sealed final probes are never opened here)')
    seeds = {int(p['seed']) for p in data['probes']}
    if not seeds <= set(DEV_PROBE_SEEDS + DEV_PROBE_EXTENSION_SEEDS):
        raise ValueError(f'{path}: seeds {sorted(seeds)} are not dev probe seeds')
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', action='append', required=True, metavar='NAME=CHECKPOINT',
                        help='an H6-2a best.pt (its best.json next to it); repeat; v3 decides')
    parser.add_argument('--probes', type=Path, required=True, help='make_h6_fresh_probes.py --dev output')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-dirty', action='store_true', help='tests / smoke only, recorded')
    parser.add_argument('--allow-unfrozen', action='store_true', help='tests only: skip the frozen v3 sha check')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists: the dev pre-check runs once')

    import torch  # noqa: F401  (imported late so the rule above loads without torch)
    from hybrid.h6_gate import d4_average, fresh_probes
    from hybrid.h6_value_train import file_sha256, text_sha256, value_forward
    from model.checkpoint import _git_provenance, load_checkpoint
    from model.config import ModelConfig

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
        models[name] = (checkpoint, meta_path, meta, file_sha256(checkpoint))
    if RULE['decide'] not in models:
        parser.error(f"--model {RULE['decide']}=... is required: it is the model the rule decides on")
    if models[RULE['decide']][3] != FROZEN_V3_SHA256 and not args.allow_unfrozen:
        parser.error(f"{models[RULE['decide']][0]} is not the frozen v3 checkpoint ({FROZEN_V3_SHA256[:8]}...)")
    try:
        data = check_probe_file(args.probes)
    except ValueError as exc:
        parser.error(str(exc))

    result = {'format': FORMAT, 'git_commit': commit, 'git_dirty': dirty, 'rule': RULE,
              'probes': {'file': str(args.probes), 'sha256': text_sha256(args.probes), 'counts': data.get('counts'),
                         'seeds': sorted({int(p['seed']) for p in data['probes']})},
              'models': {}}
    for name, (checkpoint, meta_path, meta, sha) in models.items():
        model = load_checkpoint(checkpoint, ModelConfig(**meta['model_config']), device=args.device).eval()
        fn = d4_average(lambda planes, model=model: value_forward(model, planes), args.device)
        fresh = fresh_probes(fn, args.probes, args.device)
        judged = verdict(fresh)
        result['models'][name] = {'checkpoint_sha256': sha, 'metadata_sha256': text_sha256(meta_path),
                                  'trained_on_labels_sha256': meta['labels_sha256'], 'step': meta.get('step'),
                                  **judged, 'per_probe': fresh['probes']}
        print(f"[{name}] deep win {judged['win']['text']}, deep loss {judged['loss']['text']} -> "
              f"{judged['verdict']}{'  (decides)' if name == RULE['decide'] else '  (reference)'}", flush=True)
    result['decision'] = result['models'][RULE['decide']]['verdict']
    result['next'] = ('final gate: part B + sealed fresh probes + D4, no tuning' if result['decision'] == 'PASS'
                      else 'keep the final set unopened; H6-2b / H7')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    print(f"decision: {result['decision']} -> {result['next']}", flush=True)
    print(f'wrote {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
