"""H6-4b: pre-game sanity check of the value leaf on fixed positions (docs/h6-value.md, H6-4).

Not a verdict: it looks for breakage before H6-5 and changes no setting. Positions are TB-Engine-v1's own
tree-route moves in recorded benchmark games (``--games``, e.g. the E1-holdout runs on reserved seed 8417;
no position is trained on). Each position is searched by TB-Engine-v1 (rollout leaf) and by TB-V1-Value
(value leaf) with the same seed, and the output records:

    tree_route_share   share of TB-Engine-v1's moves in the source games that took the tree route: the
                       upper bound of where the value leaf can act at all
    move_changed       positions where the two engines play different moves (both on the tree route)
    seconds            per-move time of each engine and the ratio (CPU or CUDA: ``--value-device``)
    root_q             C4 of the visited root children (range, std, share with |Q| > 0.95)
    leaf value         mean value of the evaluated leaves by the colour to move there (black bias watch)

The P92 guard is ``scripts/run_s1_probes.py`` (``h6_5_p92_guard``).

    python -u scripts/h6_value_leaf_check.py --games docs/mcts-v8-results/e1h/puct_policy_vct2_50k_8417.json \\
        --per-game 3 --value-checkpoint runs/h6_value_head_v3/best.pt --workers 6 \\
        --output docs/h6/h6_4b_check.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import Game  # noqa: E402
from scripts.run_mcts_v8_benchmark import (  # noqa: E402
    _fast_provenance, _git_commit, _git_dirty, derive_seed, file_sha256, root_q,
)

FORMAT = 'h6-4b-check-v1'
ENGINES = ('TB-Engine-v1', 'TB-V1-Value')
_WORKER: dict = {}


def sample_positions(games: list[dict], per_game: int) -> list[dict]:
    """Up to ``per_game`` tree-route plies of the recorded engine per game, evenly spaced, in game order."""
    out = []
    for g in sorted(games, key=lambda g: g['key']):
        plies = [m['ply'] for m in g['v8_moves'] if m['route'] == 'tree']
        if not plies:
            continue
        step = max(1, len(plies) // per_game)
        for ply in plies[::step][:per_game]:
            out.append({'game_key': g['key'], 'ply': ply, 'moves': g['moves'][:ply]})
    return out


def _init(policy_checkpoint, value_checkpoint, value_device, allow_unfrozen, factories=None, fast='off'):
    if fast != 'off':  # exact solver speed-up, installed in every worker (hybrid.h6_fast)
        from hybrid.h6_fast import install
        install(fast)
    if factories is not None:  # tests: torch-free stand-ins
        _WORKER.update(factories)
        return
    from hybrid.h4_policy import RootPolicy
    from hybrid.h6_value_leaf import ValueLeaf
    _WORKER['policy'] = RootPolicy(policy_checkpoint)
    _WORKER['leaf'] = ValueLeaf(value_checkpoint, device=value_device,
                                **({'expected_sha256': None} if allow_unfrozen else {}))


def _decide(engine: str, game: Game, seed: int) -> dict:
    from analysis import tb_engine_v1, tb_v1_value
    if engine == ENGINES[0]:
        agent = tb_engine_v1.make_agent(_WORKER['policy'], seed=seed)
    else:
        agent = tb_v1_value.make_agent(_WORKER['policy'], _WORKER['leaf'], seed=seed)
    started = perf_counter()
    move = agent.select_move(game)
    d = agent.diagnostics
    out = {'move': list(move), 'route': d.v8_route, 'seconds': round(perf_counter() - started, 4),
           'tree_seconds': round(d.v8_tree_seconds, 4), 'tree_move': list(d.v8_v7_move) if d.v8_v7_move else None,
           'root_q': root_q(d.v8_root_visits)}
    if d.v8_tree_mode == 'puct_value':
        out['value'] = {'calls': d.v8_value_calls, 'seconds': round(d.v8_value_seconds, 4),
                        'exact_leaves': d.v8_value_exact_leaves, 'terminal_leaves': d.v8_value_terminal_leaves,
                        'black_leaves': d.v8_value_black_leaves, 'black_sum': d.v8_value_black_sum,
                        'white_leaves': d.v8_value_white_leaves, 'white_sum': d.v8_value_white_sum}
    return out


def check_position(task: dict) -> dict:
    game = Game()
    for move in task['moves']:
        game.play(*move)
    seed = derive_seed(task['seed'], 'h6-4b', task['game_key'], task['ply'])
    return {'game_key': task['game_key'], 'ply': task['ply'],
            **{engine: _decide(engine, game, seed) for engine in ENGINES}}


def _chunk(tasks):
    return [check_position(t) for t in tasks]


def _dist(values) -> dict:
    values = sorted(values)
    if not values:
        return {'n': 0}
    return {'n': len(values), 'mean': round(sum(values) / len(values), 4), 'median': round(median(values), 4),
            'p95': values[min(len(values) - 1, int(0.95 * len(values)))], 'max': values[-1]}


def summarize(rows: list[dict], games: list[dict]) -> dict:
    base, cand = ENGINES
    moves = [m for g in games for m in g['v8_moves']]
    both_tree = [r for r in rows if r[base]['route'] == 'tree' and r[cand]['route'] == 'tree']
    ratios = [r[cand]['seconds'] / r[base]['seconds'] for r in rows if r[base]['seconds'] > 0]
    values = [r[cand]['value'] for r in rows if 'value' in r[cand]]
    b_n, w_n = sum(v['black_leaves'] for v in values), sum(v['white_leaves'] for v in values)
    q = {e: [r[e]['root_q'] for r in both_tree if r[e]['root_q']] for e in ENGINES}
    return {
        'source_moves': len(moves),
        'tree_route_share': round(sum(m['route'] == 'tree' for m in moves) / len(moves), 4) if moves else None,
        'positions': len(rows),
        'both_tree': len(both_tree),
        'route_differs': sum(r[base]['route'] != r[cand]['route'] for r in rows),
        'move_changed': sum(r[base]['move'] != r[cand]['move'] for r in both_tree),
        'tree_move_changed': sum(r[base]['tree_move'] != r[cand]['tree_move'] for r in both_tree),
        'seconds': {e: _dist([r[e]['seconds'] for r in rows]) for e in ENGINES},
        'tree_seconds': {e: _dist([r[e]['tree_seconds'] for r in rows]) for e in ENGINES},
        'seconds_ratio': _dist(ratios),
        'total_seconds_ratio': round(sum(r[cand]['seconds'] for r in rows)
                                     / max(1e-9, sum(r[base]['seconds'] for r in rows)), 4) if rows else None,
        'root_q': {e: {'range': _dist([x['range'] for x in q[e]]), 'std': _dist([x['std'] for x in q[e]]),
                       'saturated': _dist([x['saturated'] for x in q[e]]),
                       'all_saturated': sum(x['saturated'] == 1.0 for x in q[e])} for e in ENGINES},
        'value': {
            'calls': sum(v['calls'] for v in values),
            'seconds_per_move': _dist([v['seconds'] for v in values]),
            'seconds_per_call': round(sum(v['seconds'] for v in values) / max(1, sum(v['calls'] for v in values)), 6),
            'exact_leaves': sum(v['exact_leaves'] for v in values),
            'terminal_leaves': sum(v['terminal_leaves'] for v in values),
            'black_leaves': b_n, 'black_mean': round(sum(v['black_sum'] for v in values) / b_n, 4) if b_n else None,
            'white_leaves': w_n, 'white_mean': round(sum(v['white_sum'] for v in values) / w_n, 4) if w_n else None,
        },
    }


def main(argv=None, factories=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--games', nargs='+', type=Path, required=True,
                        help='run_mcts_v8_benchmark outputs of TB-Engine-v1 (arm puct_policy_vct2_50k)')
    parser.add_argument('--per-game', type=int, default=3)
    parser.add_argument('--limit', type=int, help='first N positions (smoke runs)')
    parser.add_argument('--seed', type=int, default=20261018)
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--value-checkpoint', required=factories is None)
    parser.add_argument('--value-device', default='cpu')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--fast', choices=('off', 'fast2', 'fast3'), default='off',
                        help='exact solver speed-up (hybrid.h6_fast); results are the same')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-dirty', action='store_true', help='tests / smoke only, recorded')
    parser.add_argument('--allow-unfrozen', action='store_true', help='tests only: skip the value sha check')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    dirty = _git_dirty()
    if dirty is not False and not args.allow_dirty:
        parser.error(f'refusing to run from a dirty or unknown worktree (git_dirty={dirty})')
    games, sources = [], {}
    for path in args.games:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('arm') != 'puct_policy_vct2_50k':
            parser.error(f"{path}: arm {data.get('arm')!r} is not TB-Engine-v1 (puct_policy_vct2_50k)")
        games.extend(data['games'])
        sources[str(path)] = {'sha256': file_sha256(path), 'seed': data.get('seed'), 'games': len(data['games'])}
    positions = sample_positions(games, args.per_game)[:args.limit]
    tasks = [{**p, 'seed': args.seed} for p in positions]
    init = (args.policy_checkpoint, args.value_checkpoint, args.value_device, args.allow_unfrozen, factories, args.fast)
    print(f'[h6-4b] {len(games)} games, {len(tasks)} positions, workers {args.workers}', flush=True)
    progress = Progress('h6-4b', len(tasks), 'positions')
    if args.workers > 1:
        # spawn: a forked worker cannot use CUDA once the parent has (and Windows spawns anyway)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=init,
                                 mp_context=get_context('spawn')) as pool:
            rows = [r for c in run_chunks(pool, _chunk, [[t] for t in tasks], progress, len, unit_name='positions')
                    for r in c]
    else:
        _init(*init)
        rows = []
        for t in tasks:
            rows.append(check_position(t))
            progress.update(1)
    summary = summarize(rows, games)
    payload = {'format': FORMAT, 'verdict': None, 'git_commit': _git_commit(), 'git_dirty': dirty,
               'sources': sources, 'fast': _fast_provenance(args.fast), 'per_game': args.per_game, 'seed': args.seed, 'value_device': args.value_device,
               'policy_checkpoint_sha256': file_sha256(args.policy_checkpoint) if factories is None else None,
               'value_checkpoint_sha256': file_sha256(args.value_checkpoint) if args.value_checkpoint else None,
               'summary': summary, 'positions': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
