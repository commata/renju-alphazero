"""H7-R holdout: evaluation labels of holdout self-play and the value evaluation on them (docs/h7-rl-plan.md, R1).

    labels  the games of ``h7c_selfplay.py --holdout-contract`` -> evaluation value rows (``runs/``) and their value
            cache. The label rules are ``h7c_labels.build``'s, through its evaluation entry: the games are checked
            against the holdout contract (purpose, contract SHA-256, reserved seed, actor models, clean generator,
            replay) instead of the training check, every row is split ``test`` and carries the provenance purpose
            ``holdout_eval``, which every training row source refuses (``hybrid.h7c_train.rows_source``), as the
            cycle cache check refuses reserved seeds. The training label command itself refuses holdout games.
    eval    value checkpoints on such a cache: weighted MSE by route (``exact_leaf`` settles the position or the
            network is asked) x label kind (outcome / proof), raw and D4-averaged; paired differences A - B per row
            with a bootstrap over source games (rows of one game move together); groups of models of one recipe
            with different training seeds give the spread between seeds. The noise rule (epsilon) is the R2
            contract's, not this tool's.

    python -u scripts/h7r_holdout.py labels --contract docs/h7r/contracts/<id>.json --games runs/h7r/<id>/games.jsonl
    python -u scripts/h7r_holdout.py eval --cache runs/h7r/<id>/value_cache --model c5=runs/h7c/c5/value/best.pt ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

LABELS_FORMAT = 'h7r-holdout-labels-v1'
EVAL_FORMAT = 'h7r-holdout-eval-v1'
PURPOSE = 'holdout_eval'


def check_holdout_games(games: list[dict], contract: dict, contract_sha: str) -> dict:
    """The games are this contract's holdout run, complete and legal; the identity they share."""
    import h7c_labels as lab

    if not games:
        raise ValueError('no games')
    h = contract['holdout']
    actor = contract['models'][h['actor']]['identity']
    first = {k: games[0].get(k) for k in lab.IDENTITY_KEYS}
    git = first.get('generator_git') or {}
    if not git.get('commit') or git.get('dirty') is not False:
        raise ValueError('holdout games must come from a clean, committed generator')
    seen = set()
    for g in games:
        checks = {'purpose': g.get('purpose') == 'holdout',
                  'contract': (g.get('holdout') or {}).get('contract_sha256') == contract_sha
                  and g['holdout'].get('contract_id') == contract['contract_id'],
                  'seed': g.get('run_seed') == h['seed'], 'cycle': g.get('cycle') == h['cycle'],
                  'actor': g.get('models') == actor, 'identity': {k: g.get(k) for k in lab.IDENTITY_KEYS} == first,
                  'complete': not g.get('truncated') and g.get('result') != 'truncated'}
        failed = sorted(k for k, ok in checks.items() if not ok)
        if failed:
            raise ValueError(f'{lab.game_id(g)}: not a game of this holdout contract ({", ".join(failed)})')
        if lab.game_id(g) in seen:
            raise ValueError(f'{lab.game_id(g)}: listed twice')
        seen.add(lab.game_id(g))
        lab.check_game(g)
    if len(seen) > h['games']:
        raise ValueError(f"{len(seen)} games, the contract allows {h['games']}")
    return first


def labels(args, log=print) -> dict:
    import h7c_labels as lab
    from hybrid import h7r_eval as ev
    from hybrid.h6_value_cache import build_value_cache
    from hybrid.h7c_nmask import committed
    from run_mcts_v8_benchmark import _git_dirty

    git = {'commit': lab.h6b._git_commit(), 'dirty': _git_dirty()}
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty: evaluation labels must name a clean code commit')
    contract, sha = ev.load_contract(args.contract, ROOT, check_files=False, contracts_dir=Path(args.contract).parent)
    if contract['purpose'] != 'holdout' or contract['status'] != 'fixed' or not (
            args.allow_dirty or committed(args.contract, ROOT)):
        raise ValueError(f'{args.contract}: not a fixed, committed holdout contract')
    for path in (args.value_output, args.cache_dir):
        if not lab._under_data_or_runs(Path(path)):
            raise ValueError(f'{path}: evaluation rows go under data/ or runs/')
        if Path(path).exists():
            raise ValueError(f'{path} exists')
    games = lab.read_games([args.games])
    rows, _, summary = lab.build(games, probes=lab.probe_positions(lab.default_probe_files(ROOT), ROOT),
                                 from_ply=5, outcome_weight=0.1, proof_weight=1.0, prove_losses=False, vct1=False,
                                 probe_game_min_ply=lab.h6b.MAX_ROW_ONLY_PROBE_PLY + 1, workers=args.workers,
                                 fast=args.fast, evaluation={'check': lambda g: check_holdout_games(g, contract, sha),
                                                             'purpose': PURPOSE, 'split': 'test'})
    lab.write_atomic({Path(args.value_output): [lab.value_row_dict(r) for r in rows]})
    manifest = build_value_cache(Path(args.value_output), Path(args.cache_dir), progress_seconds=60.0,
                                 git_commit=git['commit'])
    return {'format': LABELS_FORMAT, 'contract_id': contract['contract_id'], 'contract_sha256': sha, 'git': git,
            'games_sha256': ev.file_sha256(args.games), 'rows_sha256': ev.file_sha256(args.value_output),
            'cache_labels_sha256': manifest['labels_sha256'], 'cache_splits': manifest['splits'],
            'purpose': PURPOSE, 'summary': summary['value']}


# --- the evaluation ----------------------------------------------------------------------------------------------

def routes(data) -> list[str]:
    """``exact_leaf`` (the search settles the position without the network) or ``network`` per cache row."""
    import torch

    from analysis.puct_value_v1 import exact_leaf
    from hybrid.h3_cache import unpack
    from renju import BLACK, Game, WHITE

    out = []
    black, white = unpack(data['black']), unpack(data['white'])
    for i in range(black.shape[0]):
        game = Game()
        for a in torch.nonzero(black[i]).flatten().tolist():
            game.board[a // 15][a % 15] = BLACK
        for a in torch.nonzero(white[i]).flatten().tolist():
            game.board[a // 15][a % 15] = WHITE
        game.to_play = BLACK if int(data['ply'][i]) % 2 == 0 else WHITE
        out.append('exact_leaf' if exact_leaf(game) is not None else 'network')
    return out


def predictions(checkpoint, data, device: str, d4: bool) -> dict:
    import torch

    from hybrid.h6_value_cache import planes_from_value_cache
    from hybrid.h6_value_train import freeze, value_forward
    from model.checkpoint import load_checkpoint
    from model.config import ModelConfig

    meta = json.loads(Path(checkpoint).with_name('best.json').read_text(encoding='utf-8'))
    model = load_checkpoint(checkpoint, ModelConfig(**meta['model_config']), device=device)
    freeze(model)
    model.value_head.eval()
    out = {'raw': []}
    if d4:
        from hybrid.h6_gate import d4_average
        average = d4_average(lambda planes: value_forward(model, planes), device)
        out['d4'] = []
    n = data['value'].shape[0]
    with torch.no_grad():
        for start in range(0, n, 2048):
            index = torch.arange(start, min(n, start + 2048))
            planes, _, _ = planes_from_value_cache(data, index, device)
            out['raw'] += value_forward(model, planes).float().cpu().tolist()
            if d4:
                out['d4'] += [float(x) for x in average(planes).float().cpu().tolist()]
    return out


def cells(route: list[str], kind: list[int]) -> dict[str, list[int]]:
    names = {0: 'outcome', 1: 'proof'}
    out = {'all': list(range(len(route)))}
    for i, (r, k) in enumerate(zip(route, kind)):
        out.setdefault(r, []).append(i)
        out.setdefault(f'{r}/{names[k]}', []).append(i)
    return out


def wmse(err: list[float], weight: list[float], rows: list[int]) -> float | None:
    w = sum(weight[i] for i in rows)
    return sum(weight[i] * err[i] for i in rows) / w if w else None


def paired(err_a, err_b, weight, rows, games, samples: int, seed: int) -> dict:
    """WMSE(A) - WMSE(B) on the same rows; the bootstrap resamples source games (their rows together)."""
    by = {}
    for i in rows:
        by.setdefault(games[i], []).append(i)
    ids = sorted(by)
    if not ids:
        return {'rows': 0}
    diff = lambda chosen: wmse(err_a, weight, chosen) - wmse(err_b, weight, chosen)  # noqa: E731
    rng, draws = Random(seed), []
    for _ in range(samples):
        chosen = [i for _ in ids for i in by[ids[rng.randrange(len(ids))]]]
        draws.append(diff(chosen))
    draws.sort()
    q = lambda p: round(draws[int(p * (len(draws) - 1))], 6)  # noqa: E731
    return {'rows': len(rows), 'games': len(ids), 'difference': round(diff(rows), 6), 'ci95': [q(0.025), q(0.975)]}


def evaluate(args, log=print) -> dict:
    from hybrid import h7r_eval as ev
    from hybrid.h6_value_cache import load_value_cache

    data, manifest = load_value_cache(Path(args.cache), 'test', verify_hash=True)
    n = data['value'].shape[0]
    if n == 0:
        raise ValueError(f'{args.cache}: no evaluation rows')
    route = routes(data)
    kind = data['kind'].tolist()
    target, weight, games = data['value'].tolist(), data['weight'].tolist(), data['game'].tolist()
    groups = cells(route, kind)
    models = dict(m.split('=', 1) for m in args.model)
    errors, report = {}, {}
    for name, path in models.items():
        log(f'[h7r-holdout] {name}: {n:,} rows', flush=True)
        pred = predictions(path, data, args.device, args.d4)
        errors[name] = {k: [(v - t) ** 2 for v, t in zip(p, target)] for k, p in pred.items()}
        report[name] = {'checkpoint_sha256': ev.file_sha256(path),
                        **{k: {c: {'rows': len(rows), 'wmse': None if wmse(e, weight, rows) is None
                                   else round(wmse(e, weight, rows), 6)} for c, rows in groups.items()}
                           for k, e in errors[name].items()}}
    pairs = {}
    for spec in args.pair or []:
        a, b = spec.split(':')
        pairs[spec] = {k: {c: paired(errors[a][k], errors[b][k], weight, rows, games, args.bootstrap_samples,
                                     args.bootstrap_seed) for c, rows in groups.items()} for k in errors[a]}
    spread = {}
    for spec in args.group or []:
        name, members = spec.split('=', 1)
        members = members.split(',')
        spread[name] = {}
        for k in errors[members[0]]:
            spread[name][k] = {}
            for c, rows in groups.items():
                values = [wmse(errors[m][k], weight, rows) for m in members]
                if any(v is None for v in values):
                    continue
                mean = sum(values) / len(values)
                sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else None
                spread[name][k][c] = {'seeds': len(values), 'mean': round(mean, 6),
                                      'sd': None if sd is None else round(sd, 6)}
    return {'format': EVAL_FORMAT, 'cache_labels_sha256': manifest['labels_sha256'],
            'cache_sha256': manifest['splits']['test']['sha256'], 'rows': n, 'games': len(set(games)),
            'route_counts': {r: route.count(r) for r in ('network', 'exact_leaf')}, 'models': report,
            'paired': pairs, 'seed_spread': spread,
            'bootstrap': {'samples': args.bootstrap_samples, 'seed': args.bootstrap_seed, 'unit': 'source game'},
            'epsilon': None, 'epsilon_note': 'the minimal detectable change is fixed by the R2 contract from these '
                                             'measurement and seed-spread records'}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    lab = sub.add_parser('labels')
    lab.add_argument('--contract', type=Path, required=True)
    lab.add_argument('--games', type=Path, required=True)
    lab.add_argument('--value-output', type=Path, required=True)
    lab.add_argument('--cache-dir', type=Path, required=True)
    lab.add_argument('--workers', type=int, default=1)
    lab.add_argument('--fast', default='fast4')
    lab.add_argument('--output', type=Path, required=True, help='the labels manifest (JSON)')
    lab.add_argument('--allow-dirty', action='store_true', help='tests only')
    ev = sub.add_parser('eval')
    ev.add_argument('--cache', type=Path, required=True)
    ev.add_argument('--model', action='append', required=True, help='NAME=value checkpoint (repeat)')
    ev.add_argument('--pair', action='append', help='A:B paired differences (repeat)')
    ev.add_argument('--group', action='append', help='NAME=a,b,c: models of one recipe, other training seeds')
    ev.add_argument('--device', default='cuda')
    ev.add_argument('--d4', action='store_true', help='also the D4-averaged value (what the search uses)')
    ev.add_argument('--bootstrap-samples', type=int, default=2000)
    ev.add_argument('--bootstrap-seed', type=int, default=20261028)
    ev.add_argument('--output', type=Path, required=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error(f'{args.output} exists')
    try:
        out = labels(args) if args.command == 'labels' else evaluate(args)
    except ValueError as exc:
        parser.error(str(exc))
    out['output_note'] = 'measurement only'
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    print(hashlib.sha256(Path(args.output).read_bytes()).hexdigest(), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
