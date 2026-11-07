"""R2-D1-HEAD-WIDTH-1, 1,000-row v2: the H-dev evaluation under evaluate amendment A1.

The fixed v2 contract pins ``scripts/h7r_d1.py`` by SHA-256 (and the experiment id the prepare, training and fit
records carry depends on it), so its ``evaluate`` cannot change. This tool is the evaluate stage instead, pinned by
the committed amendment ``d1-1k-v2-evaluate-a1.json`` (its own SHA-256, the v2 experiment id, an approval). It takes
the v2 contract's checks unchanged from ``scripts/h7r_d1.py`` (``load_d1`` with the evaluate approval and the source
pins, the committed prepare PASS, the paired run records, the committed train-fit record equal to its recomputation)
and evaluates the same models (c5 and the six T-val selected checkpoints) on the same verified H-dev with the same
R2 section 7 rule and references; the result is the contract's ``d1_eval.json``.

What changes is the order of the writes, so that an interruption leaves a state that can be completed:

    1. the predictions go to ``predictions.pt.partial``;
    2. a predictions record (``evaluate-predicted.json``, written once) names their SHA-256, the identity of every
       evaluated checkpoint, the H-dev checks and the git state;
    3. the partial file is linked to ``predictions.pt``;
    4. the reading is computed from the predictions and ``d1_eval.json`` is published.

    run       steps 1-4; refuses while any output of an earlier attempt exists.
    --resume  after an interruption at 3 or 4: the recorded predictions (SHA-256, model names, rows, finite values,
              the H-dev rows and checks) and every checkpoint (file hashes and frozen state, re-checked against the
              recorded identities) are verified, then the result is completed from the stored predictions without a
              new inference; ``d1_eval.json`` says it was resumed and from which record. A ``.partial`` file without
              its record (an interruption at 1 or 2) is never used: the tool stops and names it.

    python scripts/h7r_d1_evaluate.py --d1-contract docs/h7r/r2/R2-D1-HEAD-WIDTH-1/d1-1k-v2-contract.json \\
        --cache runs/h7r/R1-HDEV-1/cache --c5 runs/h7c/c5/value/best.pt --device cuda [--resume]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h7r_d1 as cli  # noqa: E402
from hybrid import h7r_d1 as d1  # noqa: E402
from hybrid import h7r_r2 as r2  # noqa: E402

AMENDMENT = 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1/d1-1k-v2-evaluate-a1.json'
AMENDMENT_FORMAT = 'h7r-d1-evaluate-amendment-v1'
AMENDMENT_KEYS = {'format', 'amendment_id', 'approved', 'contract', 'tool', 'predicted_record', 'unchanged', 'reason'}
TOOL_PATH = 'scripts/h7r_d1_evaluate.py'
PREDICTED_FORMAT = 'h7r-d1-eval-predicted-v1'


def tool_sha256() -> str:
    return r2.file_sha256(Path(__file__).resolve())


def load_amendment(path, root: Path, contract_path, eid: str) -> tuple[dict, str]:
    """The committed, approved amendment of this v2 experiment that pins this tool's SHA-256."""
    from hybrid.h7c_nmask import committed

    path = Path(path)
    r2.require(path.is_file(), f'{path}: no evaluate amendment')
    a = json.loads(path.read_text(encoding='utf-8'))
    r2.require(isinstance(a, dict) and set(a) == AMENDMENT_KEYS and a['format'] == AMENDMENT_FORMAT,
               f'{path}: not a D1 evaluate amendment')
    r2.require(committed(path, root), f'{path}: not committed')
    r2.require(a['approved'] is True, 'the D1 evaluate amendment is not approved')
    ref = a['contract']
    r2.require(isinstance(ref, dict) and set(ref) == {'path', 'experiment_sha256'}
               and (root / ref['path']).resolve() == Path(contract_path).resolve()
               and ref['experiment_sha256'] == eid, 'the amendment is not for this D1 contract and experiment')
    r2.require(a['tool'] == {'path': TOOL_PATH, 'sha256': tool_sha256()},
               f'the checkout\'s {TOOL_PATH} is not the tool the amendment pins')
    rec = a['predicted_record']
    r2.require(isinstance(rec, str) and rec.startswith('runs/') and rec.endswith('.json'),
               'the predictions record belongs under runs/')
    return a, r2.file_sha256(path)


def paths(c: dict, a: dict, root: Path) -> dict:
    preds = root / c['output']['predictions']
    return {'result': root / c['output']['result'], 'predictions': preds,
            'partial': preds.with_name(preds.name + '.partial'), 'record': root / a['predicted_record']}


def checked_inputs(c: dict, root: Path, eid: str) -> tuple:
    prep = cli.checked_prepare(c, root, eid)
    recs = cli.run_records(c, root, eid)
    fit_read = cli.checked_fit(c, root, eid, prep, recs)
    r2c = cli.pinned(root, c['r2_contract'], 'R2 contract')
    eps = cli.pinned(root, c['epsilon'], 'epsilon record')
    return prep, recs, fit_read, r2c, eps


def load_models(c: dict, root: Path, recs: dict, r2c: dict, c5_path, device: str) -> tuple[dict, dict]:
    """c5 and the six selected checkpoints, each checked (file hashes, frozen state); their identities."""
    models, identities = {}, {}
    models['c5'], identities['c5'] = cli.load_c5(r2c, c5_path, device)
    for (arm, s) in sorted(recs):
        rec = recs[(arm, s)]
        r2.require(rec['verdict'] == 'CANDIDATE' and rec['selected']['state'] is not None,
                   f'{arm} {s}: no selected checkpoint (NO_CANDIDATE is kept, not evaluated)')
        run_dir = root / c['output']['run_dirs'][arm][s]
        spec = {**r2c['models']['c5'], 'value_checkpoint_sha256': rec['selected']['state']['checkpoint_sha256'],
                'value_metadata_sha256': rec['selected']['state']['metadata_sha256']}
        models[f'{arm}{s}'], identities[f'{arm}{s}'] = r2.check_model(
            f'{arm}{s}', spec, run_dir / 'best.pt', run_dir / 'best.json', r2c['frozen_state_sha256'], device)
    return models, identities


def analyse(c: dict, r2c: dict, pred: dict, data: dict, info: dict, route: list) -> tuple[dict, dict, dict]:
    """The reading of ``scripts/h7r_d1.py evaluate`` on the same predictions: paired wide - narrow per seed, the
    references (against c5, components, the additive kind decomposition) and the R2 section 7 label."""
    from hybrid import h7r_d3 as d3

    target, weight = data['value'].double().tolist(), data['weight'].double().tolist()
    bins = r2c['metrics']['ply_bins']
    groups = r2.row_groups(route, data['kind'].tolist(), info['ply'], target, info['result'], info['label_kind'], bins)
    n_games = len(info['game_ids'])
    metrics = [r2c['metrics']['primary'], *r2c['metrics']['components']]
    sums = {(n, m['prediction'], m['group']): r2.game_sums(pred[n][m['prediction']], target, weight, info['game'],
                                                          groups[m['group']], n_games) for n in pred for m in metrics}
    value = lambda n, m: r2.point_metrics(sums[(n, m['prediction'], m['group'])])[m['aggregate']]  # noqa: E731
    boot = r2c['bootstrap']
    counts = r2.multiplicities(n_games, boot['samples'], boot['seed'])
    primary = metrics[0]
    key = lambda n: (n, primary['prediction'], primary['group'])  # noqa: E731
    kinds = {k: groups[f'network/{k}'] for k in ('outcome', 'proof')}
    paired, references = {}, {}
    for s in cli.SEEDS:
        k, w = f'narrow{s}', f'wide{s}'
        diff = r2.difference(sums[key(w)], sums[key(k)], 'macro')
        paired[s] = {'control': value(k, primary), 'wide': value(w, primary), 'difference': diff, 'improvement': -diff,
                     **r2.bootstrap(counts, sums[key(w)], sums[key(k)], 'macro', boot['level'])}
        references[s] = {
            'wide_vs_c5': {'difference': r2.difference(sums[key(w)], sums[key('c5')], 'macro'),
                           **r2.bootstrap(counts, sums[key(w)], sums[key('c5')], 'macro', boot['level'])},
            'narrow_vs_c5': {'difference': r2.difference(sums[key(k)], sums[key('c5')], 'macro')},
            'components': [{'metric': m, 'difference': value(w, m) - value(k, m)} for m in metrics[1:]],
            'decomposition_kind': d3.decompose(pred[w][primary['prediction']], pred[k][primary['prediction']], target,
                                               weight, info['game'], groups[primary['group']], kinds, n_games, counts,
                                               boot['level'])['parts']}
    reading = d1.hdev_reading(paired, c['ruling']['epsilon'], {s: paired[s]['control'] for s in cli.SEEDS}, cli.SEEDS)
    return paired, references, reading


def publish_result(path: Path, body: dict) -> None:
    from hybrid.h7r_ledger import publish_json
    publish_json(path, body)


def finish(c, csha, eid, a, asha, state, p, hdev, fit_read, r2c, pred, data, info, route, started, resumed) -> dict:
    paired, references, reading = analyse(c, r2c, pred, data, info, route)
    body = {'format': cli.EVAL_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha, 'git': state, 'hdev': hdev,
            'hdev_reading': reading, 'paired': paired, 'references': references,
            'train_fit_reading': fit_read, 'reference_note': c['reference'],
            'predictions': {'path': c['output']['predictions'], 'sha256': r2.file_sha256(p['predictions'])},
            'evaluate_tool': {'amendment': {'path': AMENDMENT, 'sha256': asha, 'id': a['amendment_id']},
                              'tool': a['tool'], 'predicted_record': {'path': a['predicted_record'],
                                                                      'sha256': r2.file_sha256(p['record'])}},
            'resumed': resumed, 'minutes': round((time.monotonic() - started) / 60, 2),
            'scope': 'the 1,000-row D1 diagnosis: no D4 candidate, strength, promotion or V9 ruling'}
    publish_result(p['result'], body)
    return body


def evaluate(args, log=print) -> dict:
    import torch
    import h7r_holdout as ho
    import h7r_r2 as r2cli
    from hybrid.h7r_ledger import publish_json

    root = Path(args.root)
    started = time.monotonic()
    c, csha, eid = cli.load_d1(args.d1_contract, root, 'evaluate')
    a, asha = load_amendment(args.amendment, root, args.d1_contract, eid)
    p = paths(c, a, root)
    if args.resume:
        return resume(args, root, c, csha, eid, a, asha, p, started, log)
    r2.require(not p['result'].exists(), f"{p['result']} exists: never overwritten")
    r2.require(not p['record'].exists() and not p['predictions'].exists(),
               'the predictions of an earlier attempt exist: complete them with --resume (never overwritten)')
    r2.require(not p['partial'].exists(),
               f"{p['partial']}: an unfinished predictions file without its record; it is never used and stays for "
               'inspection, the next step is a separate decision')
    state = cli.git_state(root)
    r2.require(not state['dirty'], 'the evaluation needs a clean tracked tree')
    _, recs, fit_read, r2c, eps = checked_inputs(c, root, eid)
    models, identities = load_models(c, root, recs, r2c, args.c5, args.device)
    log('[d1-eval] H-dev predictions of c5 and the six selected checkpoints', flush=True)
    _, data, info, hdev = r2cli.check_hdev(r2c, Path(args.cache), root)
    r2.require(hdev == eps['hdev'], 'the H-dev checks differ from the R2-0B record')
    route = ho.routes(data)
    pred = {name: r2.predict(models.pop(name), data, args.device) for name in list(models)}
    r2.require(cli.git_state(root) == state, 'HEAD changed during the evaluation')
    p['predictions'].parent.mkdir(parents=True, exist_ok=True)
    tensors = {n: {k: torch.tensor(v, dtype=torch.float64) for k, v in pr.items()} for n, pr in pred.items()}
    torch.save({'experiment_sha256': eid, 'contract_sha256': csha, 'amendment_sha256': asha, 'identities': identities,
                'hdev': hdev, 'game': torch.tensor(info['game']), 'ply': torch.tensor(info['ply']),
                'models': tensors}, p['partial'])
    psha = r2.file_sha256(p['partial'])
    publish_json(p['record'], {
        'format': PREDICTED_FORMAT, 'experiment_sha256': eid, 'contract_sha256': csha, 'amendment_sha256': asha,
        'git': state, 'identities': identities, 'hdev': hdev, 'rows': int(data['value'].shape[0]),
        'models': sorted(pred), 'predictions': {'path': c['output']['predictions'], 'sha256': psha},
        'status': 'predictions complete; the reading is computed from them next'})
    os.link(p['partial'], p['predictions'])
    p['partial'].unlink()
    log(f'[d1-eval] predictions recorded ({psha[:12]}); reading', flush=True)
    return finish(c, csha, eid, a, asha, state, p, hdev, fit_read, r2c, pred, data, info, route, started, None)


def resume(args, root, c, csha, eid, a, asha, p, started, log) -> dict:
    import torch
    import h7r_holdout as ho
    import h7r_r2 as r2cli

    r2.require(not p['result'].exists(), f"{p['result']} exists: nothing to resume")
    r2.require(p['record'].is_file(), 'no predictions record: nothing to resume (a predictions file without its '
                                      'record is never used)')
    rec = json.loads(p['record'].read_text(encoding='utf-8'))
    r2.require(rec.get('format') == PREDICTED_FORMAT and rec.get('experiment_sha256') == eid
               and rec.get('contract_sha256') == csha and rec.get('amendment_sha256') == asha,
               'the predictions record is not of this contract and amendment')
    psha = rec['predictions']['sha256']
    if not p['predictions'].exists() and p['partial'].is_file() and r2.file_sha256(p['partial']) == psha:
        os.link(p['partial'], p['predictions'])  # interrupted between the record and the link
        p['partial'].unlink()
    r2.require(p['predictions'].is_file() and r2.file_sha256(p['predictions']) == psha,
               'the predictions file is not the recorded one')
    state = cli.git_state(root)
    r2.require(not state['dirty'], 'the resumed evaluation needs a clean tracked tree')
    _, recs, fit_read, r2c, eps = checked_inputs(c, root, eid)
    _, identities = load_models(c, root, recs, r2c, args.c5, args.device)
    r2.require(identities == rec['identities'], 'a checkpoint differs from the one the predictions were made with')
    _, data, info, hdev = r2cli.check_hdev(r2c, Path(args.cache), root)
    r2.require(hdev == eps['hdev'] == rec['hdev'], 'the H-dev checks differ from the recorded ones')
    stored = torch.load(p['predictions'], map_location='cpu', weights_only=True)
    n = int(data['value'].shape[0])
    expected = sorted(['c5', *(f'{arm}{s}' for arm in cli.ARMS for s in cli.SEEDS)])
    r2.require(stored['experiment_sha256'] == eid and stored['identities'] == identities and stored['hdev'] == hdev
               and stored['game'].tolist() == list(info['game']) and stored['ply'].tolist() == list(info['ply'])
               and sorted(stored['models']) == expected == rec['models'] and rec['rows'] == n,
               'the stored predictions are not of these models and H-dev rows')
    for name, pr in stored['models'].items():
        r2.require(set(pr) == set(r2.PREDICTIONS) and all(t.dtype == torch.float64 and t.shape == (n,)
                                                          and bool(torch.isfinite(t).all()) for t in pr.values()),
                   f'{name}: incomplete or non-finite stored predictions')
    pred = {name: {k: t.tolist() for k, t in pr.items()} for name, pr in stored['models'].items()}
    route = ho.routes(data)
    log('[d1-eval] resumed from the recorded predictions (no new inference)', flush=True)
    resumed = {'predicted_record_sha256': r2.file_sha256(p['record']), 'prediction_git': rec['git'],
               'resume_git': state}
    return finish(c, csha, eid, a, asha, state, p, hdev, fit_read, r2c, pred, data, info, route, started, resumed)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--d1-contract', type=Path, required=True)
    parser.add_argument('--amendment', type=Path, default=ROOT / AMENDMENT)
    parser.add_argument('--cache', required=True, help='the H-dev cache')
    parser.add_argument('--c5', required=True, help='c5 value checkpoint (best.json beside it)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true', help='complete an interrupted evaluation (no new inference)')
    args = parser.parse_args(argv)
    args.root = ROOT
    try:
        out = evaluate(args)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    r = out['hdev_reading']
    print(f"[d1-eval] H-dev {r['label']}; improvement (narrow - wide) "
          f"{ {s: round(v['improvement'], 6) for s, v in r['seeds'].items()} }; control seed range "
          f"{r['control_seed_range']:.6g}; train fit {out['train_fit_reading']['category']}"
          f"{'; resumed' if out['resumed'] else ''}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
