"""H7-R R2-0A: the existing value models on the H-dev cache under a fixed R2 contract (docs/h7r/r2/R2-DIAG-1-draft.md).

    preflight  the checks of eval only (contract, H-dev cache, every model's files and recomputed frozen state);
               evaluates and writes nothing
    eval       checks the contract (fixed, committed, R2-0A approved), the H-dev cache against the contract's hashes
            (contract -> 400 games -> rows -> cache files, ``h7r_holdout.checked_cache``), every model's checkpoint and
            metadata files and the frozen state recomputed from the loaded checkpoint; then the raw and D4 values of
            every row, the metrics of ``hybrid.h7r_r2`` per model / prediction / group, and the contract's paired
            comparisons with one source-game bootstrap shared by everything. Epsilon stays null (R2-0B).

    python -u scripts/h7r_r2.py eval --contract docs/h7r/r2/R2-DIAG-1.contract.json --cache runs/h7r/R1-HDEV-1/cache \\
        --model c5=runs/h7c/c5/value/best.pt,runs/h7c/c5/value/best.json ... --device cuda \\
        --predictions runs/h7r/R2-DIAG-1/predictions.pt --output docs/h7r/r2/R2-DIAG-1/r2_0a.json

The existing ``h7r_holdout.py eval`` record keeps its meaning; this is a new format (``h7r-r2-eval-v1``).
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid import h7r_r2 as r2  # noqa: E402

TOOL_SOURCES = ('scripts/h7r_r2.py', 'src/hybrid/h7r_r2.py', 'scripts/h7r_holdout.py')


def parse_models(specs: list[str], contract: dict) -> dict[str, tuple[Path, Path]]:
    out = {}
    for spec in specs or []:
        name, sep, paths = spec.partition('=')
        parts = paths.split(',')
        r2.require(sep and len(parts) == 2 and all(parts), f'--model {spec!r}: NAME=CHECKPOINT,METADATA')
        r2.require(name not in out, f'--model {name} given twice')
        out[name] = (Path(parts[0]), Path(parts[1]))
    r2.require(set(out) == set(contract['models']),
               f"--model must name exactly the contract models {sorted(contract['models'])}, got {sorted(out)}")
    return out


def check_hdev(contract: dict, cache: Path, root: Path):
    """The cache is the contract's H-dev: binding, games, rows, labels and test tensor hashes, game and row counts.
    The test split, the source game ids, results and label kinds of its rows."""
    import h7r_holdout as ho
    from hybrid.h6_value_cache import iter_games, load_value_cache

    hdev = contract['hdev']
    hcontract, hsha, games, binding = ho.checked_cache(cache, root)
    checks = {'contract_sha256': hsha, 'cache_binding_sha256': r2.file_sha256(cache / 'holdout.json'),
              'games_sha256': binding['games']['sha256'], 'rows_sha256': binding['rows']['sha256']}
    contract_file = root / hdev['contract']
    r2.require(contract_file.is_file() and r2.file_sha256(contract_file) == hsha,
               f"{hdev['contract']}: not the H-dev contract bound to the cache")
    data, manifest = load_value_cache(cache, 'test', verify_hash=True)
    checks['cache_labels_sha256'] = manifest['labels_sha256']
    checks['test_sha256'] = manifest['splits']['test']['sha256']
    for key, value in checks.items():
        r2.require(value == hdev[key], f'H-dev {key} {value[:12]} != contract {hdev[key][:12]}')
    ids = json.loads((cache / 'test_games.json').read_text(encoding='utf-8'))
    index = data['game'].tolist()
    present = sorted(set(index))
    r2.require(len(present) == hdev['games'] and data['value'].shape[0] == hdev['rows'],
               f"H-dev has {len(present)} games / {data['value'].shape[0]} rows, the contract "
               f"{hdev['games']} / {hdev['rows']}")
    import h7c_labels as lab
    result = {lab.game_id(g): g.get('result') for g in games}
    kinds = {}
    for gid, rows in iter_games(ho.read_ref(binding['rows'], cache)):
        for row in rows:
            if not row['value_mask']:
                kinds[(gid, row['ply'])] = row['label_kind']
    plies = data['ply'].tolist()
    row_ids = [ids[g] for g in index]
    missing = [i for i, (gid, p) in enumerate(zip(row_ids, plies)) if (gid, p) not in kinds]
    r2.require(not missing, f'{len(missing)} cache rows have no label row')
    r2.require(all(result.get(gid) in r2.RESULTS for gid in set(row_ids)), 'a source game without a result')
    dense = {g: i for i, g in enumerate(present)}  # the bootstrap unit: the games with rows
    info = {'game': [dense[g] for g in index], 'game_ids': [ids[g] for g in present], 'ply': plies,
            'result': [result[gid] for gid in row_ids], 'label_kind': [kinds[(gid, p)] for gid, p in zip(row_ids, plies)]}
    return hcontract, data, info, checks


def approved_contract(args, root: Path) -> tuple[dict, str]:
    """The contract, fixed, committed and with R2-0A approved (``allow_draft``: tests only, no command-line option)."""
    from hybrid.h7c_nmask import committed

    contract, sha = r2.load_contract(args.contract)
    if not getattr(args, 'allow_draft', False):
        r2.require(contract['status'] == 'fixed' and committed(args.contract, root),
                   f'{args.contract}: not a fixed, committed R2 contract')
        r2.require(contract['stages']['R2-0A']['approved'], 'R2-0A is not approved in the contract')
    return contract, sha


def check_models(contract: dict, models: dict, device: str, log=print) -> tuple[dict, dict]:
    """Every model's files and recomputed frozen state, all of them before any evaluation starts."""
    loaded, identity = {}, {}
    for i, (name, (checkpoint, metadata)) in enumerate(models.items(), 1):
        loaded[name], identity[name] = r2.check_model(name, contract['models'][name], checkpoint, metadata,
                                                      contract['frozen_state_sha256'], device)
        log(f"[h7r-r2] PASS model {i}/{len(models)} {name}: files, frozen state recomputed "
            f"{identity[name]['frozen_state_sha256']['recomputed'][:12]} (declared "
            f"{str(identity[name]['frozen_state_sha256']['declared'])[:12]})", flush=True)
    return loaded, identity


def preflight(args, log=print) -> dict:
    """The checks of ``eval`` without evaluating or writing: contract, H-dev cache, every model."""
    root = Path(getattr(args, 'root', ROOT))
    contract, sha = approved_contract(args, root)
    models = parse_models(args.model, contract)
    hcontract, _, info, hdev_checks = check_hdev(contract, Path(args.cache), root)
    log(f"[h7r-r2] PASS H-dev {hcontract['contract_id']}: {len(info['game_ids'])} games, {len(info['ply']):,} rows, "
        f"binding {hdev_checks['cache_binding_sha256'][:12]}", flush=True)
    _, identity = check_models(contract, models, args.device, log)
    return {'contract_sha256': sha, 'hdev': hdev_checks, 'models': identity}


def evaluate(args, log=print) -> dict:
    import torch

    import h7r_holdout as ho
    from run_mcts_v8_benchmark import _git_commit, _git_dirty

    root = Path(getattr(args, 'root', ROOT))
    started = time.monotonic()
    contract, contract_sha = approved_contract(args, root)
    git = {'commit': _git_commit(), 'dirty': _git_dirty()}
    r2.require(git['dirty'] is False or args.allow_dirty, 'the tree is dirty: R2 records must name a clean commit')
    for path in [args.output] + ([args.predictions] if args.predictions else []):
        r2.require(not Path(path).exists(), f'{path} exists')
    if args.predictions:
        import h7c_labels as lab
        r2.require(lab._under_data_or_runs(Path(args.predictions)), 'predictions go under data/ or runs/')
    models = parse_models(args.model, contract)

    log('[h7r-r2] checking the H-dev cache', flush=True)
    hcontract, data, info, hdev_checks = check_hdev(contract, Path(args.cache), root)
    loaded, identity = check_models(contract, models, args.device, log)
    log('[h7r-r2] routes (frozen exact_leaf)', flush=True)
    route = ho.routes(data)
    target, weight = data['value'].double().tolist(), data['weight'].double().tolist()
    kind = data['kind'].tolist()
    metrics = contract['metrics']
    groups = r2.row_groups(route, kind, info['ply'], target, info['result'], info['label_kind'],
                           metrics['ply_bins'])
    n_games = len(info['game_ids'])

    predictions = {}
    for i, name in enumerate(models, 1):
        log(f'[h7r-r2] predictions {i}/{len(models)} {name}: {len(target):,} rows x (raw, D4)', flush=True)
        predictions[name] = r2.predict(loaded.pop(name), data, args.device)

    boot = contract['bootstrap']
    log(f"[h7r-r2] bootstrap: {boot['samples']:,} replicates over {n_games} source games", flush=True)
    counts = r2.multiplicities(n_games, boot['samples'], boot['seed'])
    sums = {name: {p: {g: r2.game_sums(predictions[name][p], target, weight, info['game'], rows, n_games)
                       for g, rows in groups.items()} for p in r2.PREDICTIONS} for name in models}
    least = metrics['min_contributing_games']
    model_out = {}
    for name in models:
        model_out[name] = {'identity': identity[name], 'metrics': {}}
        for p in r2.PREDICTIONS:
            cells = {}
            for g in groups:
                cell = r2.point_metrics(sums[name][p][g])
                cell['readable'] = cell['games'] >= least
                if cell['games']:
                    cell['macro_ci'] = r2.bootstrap(counts, sums[name][p][g], None, 'macro', boot['level'])
                cells[g] = cell
            model_out[name]['metrics'][p] = cells
    comparisons = {}
    for spec in contract['comparisons']:
        a, b = spec.split(':')
        comparisons[spec] = {}
        for p in r2.PREDICTIONS:
            comparisons[spec][p] = {}
            for g in groups:
                games = r2.point_metrics(sums[a][p][g])['games']
                cell = {'games': games, 'readable': games >= least}
                for agg in r2.AGGREGATES:
                    d = r2.difference(sums[a][p][g], sums[b][p][g], agg)
                    cell[agg] = {'difference': d, **(r2.bootstrap(counts, sums[a][p][g], sums[b][p][g], agg,
                                                                  boot['level']) if games else {'ci95': None})}
                comparisons[spec][p][g] = cell
    primary = metrics['primary']
    summary = {}
    for spec, comp in comparisons.items():
        pick = lambda m: {**comp[m['prediction']][m['group']][m['aggregate']], 'metric': m,  # noqa: E731
                          'games': comp[m['prediction']][m['group']]['games'],
                          'readable': comp[m['prediction']][m['group']]['readable']}
        summary[spec] = {'primary': pick(primary), 'components': [pick(m) for m in metrics.get('components', [])],
                         'reversals': r2.reversals(comp, primary)}
    ref = summary[contract['epsilon']['reference']]['primary']
    eps = {'version': r2.EPSILON_VERSION, 'value': None, 'status': 'AWAITING_R2_0B',
           'reference': contract['epsilon']['reference'],
           'sample': r2.half_width(ref['difference'], ref['ci95']) if ref.get('ci95') and ref['difference'] is not None
           else None, 'seed': None}

    if args.predictions:
        path = Path(args.predictions)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'contract_sha256': contract_sha, 'game': torch.tensor(info['game'], dtype=torch.int32),
                    'ply': torch.tensor(info['ply'], dtype=torch.int16),
                    'route': [r == 'network' for r in route],
                    'models': {n: {p: torch.tensor(v, dtype=torch.float64) for p, v in pr.items()}
                               for n, pr in predictions.items()}}, path)
    out = {'format': r2.RESULT_FORMAT, 'metric_version': r2.METRIC_VERSION, 'stage': 'R2-0A',
           'contract_id': contract['contract_id'], 'contract_sha256': contract_sha,
           'contract_status': contract['status'], 'git': git,
           'sources': {s: r2.file_sha256(ROOT / s) for s in TOOL_SOURCES},
           'environment': {'torch': torch.__version__, 'python': platform.python_version(), 'device': args.device},
           'hdev': {'contract_id': hcontract['contract_id'], **hdev_checks, 'games': n_games, 'rows': len(target)},
           'route_counts': {r: route.count(r) for r in ('network', 'exact_leaf')},
           'groups': {g: {'rows': len(rows)} for g, rows in groups.items()},
           'bootstrap': {**boot, 'generator': 'torch.Generator CPU, randint (samples x games), counts shared by all'},
           'models': model_out, 'comparisons': comparisons, 'summary': summary, 'epsilon': eps,
           'predictions': ({'path': str(args.predictions), 'sha256': r2.file_sha256(args.predictions)}
                           if args.predictions else None),
           'minutes': round((time.monotonic() - started) / 60, 2),
           'scope': 'development measurement on H-dev: no promotion, confirmation or strength claim; epsilon is set '
                    'only after R2-0B (seed spread and the reproduction check)'}
    from hybrid.h7r_ledger import publish_json
    publish_json(args.output, out)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    ev = sub.add_parser('eval')
    ev.add_argument('--contract', type=Path, required=True)
    ev.add_argument('--cache', type=Path, required=True)
    ev.add_argument('--model', action='append', required=True, help='NAME=value checkpoint,value metadata (each model)')
    ev.add_argument('--device', default='cuda')
    ev.add_argument('--predictions', type=Path, help='row-level predictions (under data/ or runs/)')
    ev.add_argument('--output', type=Path, required=True)
    ev.add_argument('--allow-dirty', action='store_true', help='tests only')
    pre = sub.add_parser('preflight', help='the checks of eval only: contract, H-dev cache, every model')
    pre.add_argument('--contract', type=Path, required=True)
    pre.add_argument('--cache', type=Path, required=True)
    pre.add_argument('--model', action='append', required=True, help='NAME=value checkpoint,value metadata')
    pre.add_argument('--device', default='cuda')
    return parser


def print_summary(out: dict) -> None:
    fmt = lambda x: 'n/a' if x is None else f'{x:+.6f}'  # noqa: E731
    for spec, s in out['summary'].items():
        p = s['primary']
        ci = p.get('ci95')
        print(f"{spec:10s} primary {fmt(p['difference'])} CI95 "
              f"{'n/a' if ci is None else f'[{ci[0]:+.6f}, {ci[1]:+.6f}]'} games {p['games']}"
              f"{'' if not s['reversals'] else '  reversals: ' + ', '.join(s['reversals'])}", flush=True)
    print(f"epsilon: null (sample part {fmt(out['epsilon']['sample'])}; R2-0B sets it)", flush=True)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'preflight':
            preflight(args)
            print('[h7r-r2] preflight PASS (nothing evaluated or written)', flush=True)
            return 0
        out = evaluate(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print_summary(out)
    print(f'[h7r-r2] wrote {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
