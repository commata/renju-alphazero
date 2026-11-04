"""R2-D3 preparation (read-only, CPU): H-dev composition and the additive decomposition of the R2 primary metric.

Before the D3 contract (outcome-contribution ablation) is fixed, this reports what the existing records say about
the rows D3 touches. Nothing is trained, searched or ruled; no threshold or label is made.

    composition    rows / games / weight per ply bin x label kind (outcome, proof:<type>) x route (network /
                   exact_leaf) of the verified H-dev cache, and each label kind's share of the primary metric's
                   per-game weight (mean over games of den_g^kind / den_g)
    decomposition  for the R2-0B seed models against c5 and c8, and c8 against c5: the primary metric difference
                   (D4 prediction, network rows, game macro) split into additive contributions by label kind and by
                   label kind x ply bin, each with every game's original denominator (``hybrid.h7r_d3``); the parts
                   add up to the total exactly and, replicate by replicate, in the R2 bootstrap (descriptive
                   intervals only)

Inputs are bound: the fixed R2 contract, the committed R2-0B epsilon record (its SHA-256 pinned here), the
predictions it names (``seed_predictions.pt``, their SHA-256 from that record), the H-dev cache through the R2
contract's checks, and the committed R2-0A record. The recomputed totals must equal the committed paired values
(seed models: the epsilon record; c8:c5: R2-0A within the R2-0B replay tolerance) before anything is written.
Output: one aggregate JSON, new file only; the inputs are re-hashed after the computation.

    python -u scripts/h7r_d3_prep.py --cache runs/h7r/R1-HDEV-1/cache \\
        --output docs/h7r/r2/R2-D3-NO-OUTCOME-1/prep.json
    python -u scripts/h7r_d3_prep.py --cache runs/h7r/R1-HDEV-1/cache \\
        --output docs/h7r/r2/R2-D3-NO-OUTCOME-1/prep.json --verify   # recompute, compare (git must be the same commit)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hybrid import h7r_d3 as d3  # noqa: E402
from hybrid import h7r_r2 as r2  # noqa: E402

FORMAT = 'h7r-d3-prep-v1'
R2_CONTRACT = {'path': 'docs/h7r/r2/R2-DIAG-1.contract.json',
               'sha256': 'b8045043dc9fe081ac11a1a4dc16ca323b0eac55e393d9f8a17390869b2bce2d'}
EPSILON = {'path': 'docs/h7r/r2/R2-DIAG-1/r2_0b/epsilon.json',
           'sha256': 'd46896e99200e3ff6f3d152f1c2028cde7c5a7712e481ad1e6e486ce0cebd4a2'}
R2_0A = {'path': 'docs/h7r/r2/R2-DIAG-1/r2_0a.json',
         'sha256': '316be86c8d9075247c7abe6a793fd4cda57e9233cabe5be87a2e6c522f6b6b07'}
MODELS = ('c5', 'c8', 'seed31001', 'seed31002', 'seed31003')
SEEDS = ('31001', '31002', '31003')
COMPARISONS = tuple(f'seed{s}:{b}' for s in SEEDS for b in ('c5', 'c8')) + ('c8:c5',)
PREDICTION = 'd4'
EXACT = 1e-9  # same predictions file, same arithmetic: only float summation order may differ
R2_0A_MATCH = 1e-5  # c8:c5 against R2-0A (another prediction file): the R2-0B replay tolerance (REPLAY_MATCH)
SOURCES = ('scripts/h7r_d3_prep.py', 'src/hybrid/h7r_d3.py', 'src/hybrid/h7r_r2.py', 'scripts/h7r_r2.py',
           'scripts/h7r_holdout.py')
LIMITS = [
    'descriptive preparation for the D3 contract: no training, ruling, threshold or label',
    'contributions are additive shares of the primary metric difference with each game\'s original denominator; '
    'they are not the label kind\'s own macro and not a causal attribution',
    'the intervals are descriptive bootstrap percentiles; epsilon is not applied to any part',
    'existing model comparisons do not predict the result of an ablation that has not run',
]


def git_state(root: Path) -> dict:
    def git(*a):
        return subprocess.check_output(['git', *a], cwd=root, text=True).strip()
    state = {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}
    r2.require(not state['dirty'], 'the D3 preparation needs a clean tracked tree')
    return state


def bound_inputs(root: Path, predictions: str) -> dict:
    """SHA-256 of every input; the pinned ones must match. Absent or different: stop (never narrowed)."""
    shas = {}
    for name, ref in (('r2_contract', R2_CONTRACT), ('epsilon', EPSILON), ('r2_0a', R2_0A)):
        path = root / ref['path']
        r2.require(path.is_file(), f"{ref['path']} is absent")
        shas[name] = r2.file_sha256(path)
        r2.require(shas[name] == ref['sha256'], f"{ref['path']}: SHA-256 differs from the pinned record")
    eps = json.loads((root / EPSILON['path']).read_text(encoding='utf-8'))
    r2.require(eps.get('predictions', {}).get('path') == predictions,
               f"the epsilon record names {eps.get('predictions')}, not {predictions}")
    path = root / predictions
    r2.require(path.is_file(), f'{predictions} is absent (the preparation stops; it is never narrowed)')
    shas['predictions'] = r2.file_sha256(path)
    r2.require(shas['predictions'] == eps['predictions']['sha256'], f'{predictions}: not the predictions R2-0B wrote')
    return shas


def checked_predictions(pred, info: dict, routes: list, contract_sha: str) -> dict:
    """The D4 predictions of the five models, row-aligned with the verified H-dev cache (dense game index, ply)."""
    import torch
    n = len(info['ply'])
    r2.require(isinstance(pred, dict) and set(pred) == {'contract_sha256', 'game', 'ply', 'route', 'models'},
               'the predictions have another layout')
    r2.require(pred['contract_sha256'] == contract_sha, 'the predictions belong to another R2 contract')
    r2.require(set(pred['models']) == set(MODELS), f"prediction models {sorted(pred['models'])} differ")
    for key in ('game', 'ply'):
        t = pred[key]
        r2.require(isinstance(t, torch.Tensor) and t.dim() == 1 and t.numel() == n
                   and t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64), f'prediction {key}: not {n} ints')
    r2.require(pred['game'].tolist() == list(info['game']) and pred['ply'].tolist() == list(info['ply']),
               'prediction rows are missing, repeated, extra or out of the cache order')
    r2.require(pred['route'] == [r == 'network' for r in routes], 'prediction routes differ from the H-dev rows')
    out = {}
    for name in MODELS:
        t = pred['models'][name].get(PREDICTION) if isinstance(pred['models'][name], dict) else None
        r2.require(isinstance(t, torch.Tensor) and t.dtype == torch.float64 and t.dim() == 1 and t.numel() == n
                   and bool(torch.isfinite(t).all()) and bool((t.abs() <= 1).all()),
                   f'{name}: not {n} finite float64 {PREDICTION} values in [-1, 1]')
        out[name] = t.tolist()
    return out


def analyse(contract: dict, data_kind: list, info: dict, routes: list, target: list, weight: list,
            values: dict, counts=None) -> dict:
    bins = contract['metrics']['ply_bins']
    groups = r2.row_groups(routes, data_kind, info['ply'], target, info['result'], info['label_kind'], bins)
    n_games = len(info['game_ids'])
    rows = groups['network']
    kinds = {k: groups[f'network/{k}'] for k in ('outcome', 'proof')}
    plies = {b: set(groups[f'network/ply:{b}']) for b in r2.ply_bin_names(bins)}
    kind_ply = {f'{k}|{b}': [i for i in part if i in plies[b]] for k, part in kinds.items() for b in plies}
    level = contract['bootstrap']['level']
    out = {'composition': d3.composition(routes, data_kind, info['ply'], info['label_kind'], weight, info['game'],
                                         bins),
           'macro_weight_shares': {'kind': d3.macro_shares(weight, info['game'], rows, kinds, n_games),
                                   'kind_ply': d3.macro_shares(weight, info['game'], rows, kind_ply, n_games)},
           'decomposition': {}}
    for spec in COMPARISONS:
        a, b = spec.split(':')
        by_kind = d3.decompose(values[a], values[b], target, weight, info['game'], rows, kinds, n_games, counts, level)
        by_cell = d3.decompose(values[a], values[b], target, weight, info['game'], rows, kind_ply, n_games, counts,
                               level)
        component = {k: r2.difference(r2.game_sums(values[a], target, weight, info['game'], groups[f'network/{k}'],
                                                   n_games),
                                      r2.game_sums(values[b], target, weight, info['game'], groups[f'network/{k}'],
                                                   n_games), 'macro') for k in kinds}
        out['decomposition'][spec] = {'total': by_kind['total'], 'kind': by_kind['parts'],
                                      'kind_ply': by_cell['parts'], 'component_macro': component}
    return out


def check_against_records(result: dict, eps: dict, r2_0a: dict) -> dict:
    """The recomputed totals and component macros equal the committed R2-0B paired values; c8:c5 equals R2-0A."""
    checks = {}
    for spec in COMPARISONS[:-1]:
        paired = {(m['metric']['group'], m['metric']['aggregate']): m['difference'] for m in eps['paired'][spec]}
        d = result['decomposition'][spec]
        got = {('network', 'macro'): d['total'], ('network/outcome', 'macro'): d['component_macro']['outcome'],
               ('network/proof', 'macro'): d['component_macro']['proof']}
        for key, value in got.items():
            r2.require(abs(value - paired[key]) <= EXACT, f'{spec} {key}: {value} differs from R2-0B {paired[key]}')
        checks[spec] = 'equal to the R2-0B paired record'
    ref = r2_0a['summary']['c8:c5']['primary']['difference']
    got = result['decomposition']['c8:c5']['total']
    r2.require(abs(got - ref) <= R2_0A_MATCH, f'c8:c5 total {got} differs from R2-0A {ref}')
    checks['c8:c5'] = f'within {R2_0A_MATCH} of R2-0A ({got - ref:+.2e})'
    return checks


def compute(args, root: Path, log=print) -> dict:
    """Everything the record holds but the timing: shared by the writer and ``--verify``."""
    import torch
    import h7r_holdout as ho
    import h7r_r2 as r2cli
    from hybrid.h7c_nmask import committed

    state = git_state(root)
    before = bound_inputs(root, args.predictions)
    contract, csha = r2.load_contract(root / R2_CONTRACT['path'])
    r2.require(csha == before['r2_contract'] and contract['status'] == 'fixed'
               and committed(root / R2_CONTRACT['path'], root), 'not the fixed, committed R2 contract')
    eps = json.loads((root / EPSILON['path']).read_text(encoding='utf-8'))
    r2_0a = json.loads((root / R2_0A['path']).read_text(encoding='utf-8'))
    r2.require(eps['contract_sha256'] == csha and r2_0a['contract_sha256'] == csha, 'records of another contract')
    log('[d3-prep] inputs bound; checking the H-dev cache', flush=True)
    _, data, info, hdev = r2cli.check_hdev(contract, Path(args.cache), root)
    r2.require(hdev == eps['hdev'], 'the H-dev checks differ from the R2-0B record')
    routes = ho.routes(data)
    values = checked_predictions(torch.load(root / args.predictions, map_location='cpu', weights_only=True), info,
                                 routes, csha)
    target, weight = data['value'].double().tolist(), data['weight'].double().tolist()
    boot = contract['bootstrap']
    counts = r2.multiplicities(len(info['game_ids']), boot['samples'], boot['seed'])
    log(f"[d3-prep] decomposing {len(COMPARISONS)} comparisons ({boot['samples']} replicates)", flush=True)
    result = analyse(contract, data['kind'].tolist(), info, routes, target, weight, values, counts)
    checks = check_against_records(result, eps, r2_0a)
    r2.require(bound_inputs(root, args.predictions) == before, 'an input changed during the preparation')
    r2.require(git_state(root) == state, 'HEAD changed during the preparation')
    return {'format': FORMAT, 'git': state, 'sources': {s: r2.file_sha256(root / s) for s in SOURCES},
            'inputs': {**{k: {'path': ref['path'], 'sha256': before[k]} for k, ref in
                          (('r2_contract', R2_CONTRACT), ('epsilon', EPSILON), ('r2_0a', R2_0A))},
                       'predictions': {'path': args.predictions, 'sha256': before['predictions']}},
            'hdev': hdev, 'prediction': PREDICTION, 'primary': contract['metrics']['primary'],
            'bootstrap': boot, 'record_checks': checks, **result, 'limits': LIMITS}


def run(args, root=ROOT, log=print) -> dict:
    from hybrid.h7r_ledger import publish_json

    started = time.monotonic()
    root = Path(root)
    output = Path(args.output)
    r2.require(not output.exists(), f'{output} exists: never overwritten')
    body = compute(args, root, log)
    body['seconds'] = round(time.monotonic() - started, 1)
    publish_json(output, body)
    return body


def verify(args, root=ROOT, log=print) -> list[str]:
    """Recompute the record from the bound inputs at the current clean HEAD and compare every field but the timing."""
    from h7r_d4_full_check import Failure, differences, strict_json

    root = Path(root)
    output = Path(args.output)
    r2.require(output.is_file(), f'{output} is absent')
    raw = output.read_bytes()
    before = hashlib.sha256(raw).hexdigest()
    try:
        saved = strict_json(raw)
    except Failure as exc:
        raise ValueError(f'{output}: {exc}') from exc
    r2.require(isinstance(saved, dict), f'{output}: expected a JSON object')
    seconds = saved.get('seconds')
    r2.require(type(seconds) in (int, float) and seconds >= 0, f'{output}: invalid elapsed time')
    actual = compute(args, root, log)
    problems = differences({k: v for k, v in saved.items() if k != 'seconds'}, json.loads(json.dumps(actual)))
    r2.require(r2.file_sha256(output) == before, 'the record changed during verification')
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cache', required=True, help='the H-dev cache')
    parser.add_argument('--predictions', default='runs/h7r/R2-DIAG-1/r2_0b/seed_predictions.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify', action='store_true', help='recompute and compare with the written record')
    args = parser.parse_args(argv)
    try:
        if args.verify:
            problems = verify(args)
            for problem in problems[:20]:
                print(f'FAIL: {problem}', flush=True)
            print(f"VERIFY {'FAIL' if problems else 'PASS'}", flush=True)
            return 1 if problems else 0
        body = run(args)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    print(f"[d3-prep] DONE {body['seconds']} s; record checks: {body['record_checks']}", flush=True)
    for name, share in body['macro_weight_shares']['kind'].items():
        print(f'  macro weight share {name}: {share:.4f}', flush=True)
    for spec, d in body['decomposition'].items():
        parts = ', '.join(f"{k} {v['contribution']:+.6f}" for k, v in d['kind'].items())
        print(f"  {spec:16s} total {d['total']:+.6f} = {parts}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
