"""H7-R R2-0A exploratory addendum X1 (amendment ``docs/h7r/r2/R2-DIAG-1-A1.json``): how much of the R2-0A differences is
a per-side mean shift of the value, from the R2-0A predictions, the H-dev cache and its games, all pinned by SHA-256.

    descriptive  the group's game-macro loss (the contract's weights w_i / (W_g * G)) split exactly by side to move:
                 L = sum_s Omega_s (mu_s^2 + sigma_s^2) (mean residual part and residual variance part)
    cross-fitted the same loss after a constant offset per side to move, each source-game fold's offset estimated on
                 the other folds only (folds int(sha256(game_id), 16) % k); in the bootstrap the offsets are
                 re-estimated in every replicate

Both are post-hoc exploratory readings: no promotion, confirmation or independent evidence, no change of the primary
metric or of the R2-0A readings.

    python -u scripts/h7r_r2_x1.py --contract docs/h7r/r2/R2-DIAG-1.contract.json \\
        --amendment docs/h7r/r2/R2-DIAG-1-A1.json --cache runs/h7r/R1-HDEV-1/cache \\
        --result docs/h7r/r2/R2-DIAG-1/r2_0a.json --predictions runs/h7r/R2-DIAG-1/predictions.pt \\
        --output docs/h7r/r2/R2-DIAG-1/x1_side_offset.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid import h7r_r2 as r2  # noqa: E402

TOOL_SOURCES = ('scripts/h7r_r2_x1.py', 'scripts/h7r_r2.py', 'src/hybrid/h7r_r2.py', 'scripts/h7r_holdout.py')
MATCH = 1e-9  # the reproduced R2-0A macro must agree to this


def checked_inputs(args, root: Path):
    """Contract and amendment (fixed, committed, X1 approved), the R2-0A result and predictions by their pinned SHA-256."""
    import h7r_r2 as cli
    from hybrid.h7c_nmask import committed

    contract, contract_sha = cli.approved_contract(args, root)
    amendment, amendment_sha = r2.load_amendment(args.amendment, contract, contract_sha)
    if not getattr(args, 'allow_draft', False):  # tests only: not a command-line option
        r2.require(committed(args.amendment, root), f'{args.amendment}: not committed')
    r2.require(amendment['x1']['approved'], 'X1 is not approved in the amendment')
    pins = amendment['x1']['inputs']
    for name, path in (('result', args.result), ('predictions', args.predictions)):
        r2.require(r2.file_sha256(path) == pins[name]['sha256'], f'{path}: not the {name} the amendment pins')
    result = json.loads(Path(args.result).read_text(encoding='utf-8'))
    r2.require(result.get('format') == r2.RESULT_FORMAT and result.get('stage') == 'R2-0A'
               and result.get('contract_sha256') == contract_sha, 'the result is not this contract\'s R2-0A record')
    r2.require((result.get('predictions') or {}).get('sha256') == pins['predictions']['sha256'],
               'the result names other predictions')
    return contract, contract_sha, amendment, amendment_sha, result


def run(args, log=print) -> dict:
    import torch

    import h7r_holdout as ho
    import h7r_r2 as cli
    from run_mcts_v8_benchmark import _git_commit, _git_dirty

    root = Path(getattr(args, 'root', ROOT))
    started = time.monotonic()
    contract, contract_sha, amendment, amendment_sha, result = checked_inputs(args, root)
    git = {'commit': _git_commit(), 'dirty': _git_dirty()}
    r2.require(git['dirty'] is False or args.allow_dirty, 'the tree is dirty: X1 records must name a clean commit')
    r2.require(not Path(args.output).exists(), f'{args.output} exists')
    x1 = amendment['x1']

    log('[h7r-r2-x1] checking the H-dev cache and the row alignment of the predictions', flush=True)
    _, data, info, hdev_checks = cli.check_hdev(contract, Path(args.cache), root)
    for key, value in hdev_checks.items():
        r2.require(result['hdev'].get(key) == value, f'H-dev {key} differs from the R2-0A record')
    saved = torch.load(args.predictions, weights_only=False)  # the file is pinned by its SHA-256
    r2.require(saved.get('contract_sha256') == contract_sha, 'the predictions name another contract')
    r2.require(saved['game'].tolist() == info['game'] and saved['ply'].tolist() == info['ply'],
               'the predictions are not aligned with the H-dev rows')
    route = ho.routes(data)
    r2.require([r == 'network' for r in route] == list(saved['route']), 'the routes differ from the R2-0A run')
    r2.require(set(saved['models']) == set(contract['models']), 'the predictions do not hold the contract models')
    target, weight = data['value'].double().tolist(), data['weight'].double().tolist()
    groups = r2.row_groups(route, data['kind'].tolist(), info['ply'], target, info['result'], info['label_kind'],
                           contract['metrics']['ply_bins'])
    n_games = len(info['game_ids'])
    k = x1['folds']['k']
    folds = r2.game_folds(info['game_ids'], k)
    boot = contract['bootstrap']
    log(f"[h7r-r2-x1] {len(contract['models'])} models x {x1['predictions']} x {x1['groups']}; bootstrap "
        f"{boot['samples']:,} replicates over {n_games} source games, {k} folds", flush=True)
    counts = r2.multiplicities(n_games, boot['samples'], boot['seed'])

    sums, models_out, draws = {}, {}, {}
    for name in contract['models']:
        models_out[name] = {}
        for p in x1['predictions']:
            models_out[name][p] = {}
            pred = saved['models'][name][p].tolist()
            for g in x1['groups']:
                s = r2.side_sums(pred, target, weight, info['game'], info['ply'], groups[g], n_games)
                sums[(name, p, g)] = s
                point = r2.decompose(s)
                expected = result['models'][name]['metrics'][p][g]['macro']
                r2.require(abs(point['loss'] - expected) <= MATCH,
                           f'{name} {p} {g}: reproduced macro {point["loss"]} != R2-0A {expected}')
                cf = r2.crossfit(s, folds, k)
                r2.require(not cf['undefined'], f'{name} {p} {g}: a fold side without training weight')
                models_out[name][p][g] = {'descriptive': point, 'crossfit': cf,
                                          'offset_reduction': point['loss'] - cf['loss']}
                rep = r2.decompose(s, counts)
                cfr = r2.crossfit(s, folds, k, counts)
                draws[(name, p, g)] = {'loss': rep['loss'], 'bias': rep['bias'], 'variance': rep['variance'],
                                       'crossfit': cfr['loss'], 'undefined': cfr['undefined']}
    comparisons = {}
    for spec in contract['comparisons']:
        a, b = spec.split(':')
        comparisons[spec] = {}
        for p in x1['predictions']:
            comparisons[spec][p] = {}
            for g in x1['groups']:
                pa, pb = models_out[a][p][g], models_out[b][p][g]
                da, db = draws[(a, p, g)], draws[(b, p, g)]
                undefined = da['undefined'] + db['undefined']
                cell = {}
                for key, va, vb in (('loss', pa['descriptive']['loss'], pb['descriptive']['loss']),
                                    ('bias', pa['descriptive']['bias'], pb['descriptive']['bias']),
                                    ('variance', pa['descriptive']['variance'], pb['descriptive']['variance']),
                                    ('crossfit', pa['crossfit']['loss'], pb['crossfit']['loss'])):
                    cell[key] = {'difference': va - vb,
                                 **r2.interval(da[key] - db[key], boot['level'],
                                               undefined if key == 'crossfit' else 0)}
                comparisons[spec][p][g] = cell
    out = {'format': r2.X1_FORMAT, 'version': r2.X1_VERSION, 'exploratory': True,
           'contract_id': contract['contract_id'], 'contract_sha256': contract_sha,
           'amendment_id': amendment['amendment_id'], 'amendment_sha256': amendment_sha, 'git': git,
           'sources': {s: r2.file_sha256(ROOT / s) for s in TOOL_SOURCES},
           'inputs': {'result_sha256': r2.file_sha256(args.result), 'predictions_sha256': r2.file_sha256(args.predictions),
                      'hdev': hdev_checks},
           'definitions': {k_: x1[k_] for k_ in ('weights', 'decomposition', 'offset', 'folds', 'bootstrap', 'reading')},
           'folds': {'k': k, 'games': [folds.count(f) for f in range(k)]},
           'bootstrap': boot, 'models': models_out, 'comparisons': comparisons,
           'minutes': round((time.monotonic() - started) / 60, 2),
           'scope': x1['reading']}
    from hybrid.h7r_ledger import publish_json
    publish_json(args.output, out)
    return out


def print_summary(out: dict) -> None:
    fmt = lambda x: f'{x:+.6f}'  # noqa: E731
    for spec, comp in out['comparisons'].items():
        cell = comp.get('d4', {}).get('network')
        if cell is None:
            continue
        parts = []
        for key in ('loss', 'bias', 'variance', 'crossfit'):
            ci = cell[key]['ci95']
            parts.append(f"{key} {fmt(cell[key]['difference'])} "
                         f"{'n/a' if ci is None else f'[{ci[0]:+.6f}, {ci[1]:+.6f}]'}")
        print(f'{spec:8s} d4/network  ' + '  '.join(parts), flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--amendment', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-dirty', action='store_true', help='tests only')
    args = parser.parse_args(argv)
    try:
        out = run(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print_summary(out)
    print(f'[h7r-r2-x1] wrote {args.output} (exploratory; no reading of the contract changes)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
