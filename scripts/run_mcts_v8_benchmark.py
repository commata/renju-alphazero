"""V8-4: MCTS-v8 against frozen V7 in real games (docs/mcts-v8-teacher.md §11.11).

Paired openings as in ``run_mcts_v7_benchmark.py``: each random opening is played
twice with colours swapped. Openings and game seeds depend only on ``--seed`` and
the pair index, never on ``--arm``, so runs of different arms with the same seed
are paired (ablation).

Arms (V8 configuration):
    full    V8-A + V8-B + V8-C (V8_DEFAULTS)
    ab      V8-A + V8-B        (root_vct_safety=False; the pilot's "full")
    a_only  V8-A only          (the pilot's "a_only")
    b_only  V8-B only
    c_only  V8-C only
    off     none               (V8 == V7; sanity check)

Every V8 move records its route, time and module diagnostics. With
``--counterfactual`` each V8-B move (route ``own_vct``) also records the move V7
would have played there (a separate random stream, so the game itself does not
change) and whether that V7 move is itself a proven VCT1 attack.

Games are written one JSON line each to ``--games-jsonl`` as they finish; a rerun
with the same file skips finished games (resume). ``--workers`` plays games in
parallel processes. The summary (``--output``) is built from every matching line.

    python scripts/run_mcts_v8_benchmark.py --arm full --pairs 10 --seed 8401 \\
        --workers 4 --games-jsonl runs/v8_4/full.jsonl --output runs/v8_4/full.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from random import Random
from statistics import median
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from agents import MCTSV7Agent  # noqa: E402
from analysis.mcts_v8 import V8_DEFAULTS, _BudgetedSolver  # noqa: E402
from analysis.mcts_v8_agent import MCTSV8Agent  # noqa: E402
from renju import BLACK, WHITE, Game, IllegalMove  # noqa: E402
from renju.game import OPENING_MOVE  # noqa: E402
from search.mcts_v7 import V7_FINAL, mcts_search_v7  # noqa: E402

FORMAT = 'mcts-v8-benchmark-v1'
ARMS = {
    'full': {},
    'ab': {'root_vct_safety': False},
    'a_only': {'own_vct_attack': False, 'root_vct_safety': False},
    'b_only': {'stage_vct_safety': False, 'root_vct_safety': False},
    'c_only': {'stage_vct_safety': False, 'own_vct_attack': False},
    'off': {'stage_vct_safety': False, 'own_vct_attack': False, 'root_vct_safety': False},
}
SEARCH_KEYS = ('simulations', 'tactical_simulations')  # smoke-test overrides only


def derive_seed(seed: int, *parts) -> int:
    payload = json.dumps([seed, *parts], separators=(',', ':'), ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest()[:8], 'big')


def make_opening(seed: int, random_plies: int, radius: int):
    rng = Random(seed)
    game = Game()
    game.play(*OPENING_MOVE)
    cr, cc = OPENING_MOVE
    for _ in range(random_plies):
        legal = game.legal_moves()
        near = [m for m in legal if max(abs(m[0] - cr), abs(m[1] - cc)) <= radius]
        game.play(*rng.choice(near or legal))
        if game.done:
            raise RuntimeError('benchmark opening ended the game')
    return tuple(game.history)


def v8_config(arm: str, overrides: dict | None = None) -> dict:
    return {**V8_DEFAULTS, **ARMS[arm], **(overrides or {})}


def _move_record(ply, seconds, diag) -> dict:
    return {
        'ply': ply, 'seconds': round(seconds, 4), 'route': diag.v8_route,
        'changed': diag.v8_changed,
        'v7_move': list(diag.v8_v7_move) if diag.v8_v7_move is not None else None,
        'attack': {
            'status': diag.v8_attack_status, 'rank': diag.v8_attack_rank,
            'candidates': diag.v8_attack_candidates, 'calls': diag.v8_attack_calls,
            'nodes': diag.v8_attack_nodes, 'exhausted': diag.v8_attack_budget_exhausted,
            'seconds': round(diag.v8_attack_seconds, 4),
        },
        'vct': {
            'checked': [[list(m), s] for m, s in diag.v8_vct_checked],
            'widened': diag.v8_vct_widened, 'calls': diag.v8_vct_calls,
            'nodes': diag.v8_vct_nodes, 'exhausted': diag.v8_vct_budget_exhausted,
            'seconds': round(diag.v8_vct_seconds, 4),
        },
        'root': {
            'checked': [[list(m), s] for m, s in diag.v8_root_checked], 'rank': diag.v8_root_rank,
            'calls': diag.v8_root_calls, 'nodes': diag.v8_root_nodes,
            'exhausted': diag.v8_root_budget_exhausted, 'seconds': round(diag.v8_root_seconds, 4),
        },
    }


def _counterfactual(game: Game, seed: int, ply: int, config: dict) -> dict:
    """V7's move in the same position and whether it is itself a proven VCT1 attack."""
    started = perf_counter()
    move = mcts_search_v7(game, **{k: config[k] for k in V7_FINAL},
                          random=Random(derive_seed(seed, 'counterfactual', ply)))
    solver = _BudgetedSolver(node_limit=config['attack_vcf_node_limit'],
                             call_limit=config['attack_call_limit'],
                             node_budget=config['attack_node_budget'])
    status = solver.attack_status(game, move)
    return {'v7_move': list(move), 'v7_move_attack_status': status,
            'seconds': round(perf_counter() - started, 4)}


def play_one(task: dict) -> dict:
    """One game; ``task`` is a plain dict so it can cross process boundaries."""
    config = v8_config(task['arm'], task.get('v8_overrides'))
    search = {k: config[k] for k in SEARCH_KEYS}
    seed, v8_color = task['seed'], task['v8_color']
    v8 = MCTSV8Agent(seed=derive_seed(seed, 'v8'), **{k: v for k, v in config.items() if k in V8_DEFAULTS})
    v7 = MCTSV7Agent(seed=derive_seed(seed, 'v7'), **task.get('v7_overrides', {}))
    game = Game()
    for move in task['opening']:
        game.play(*move)
    v8_moves, v7_seconds = [], []
    started_game = perf_counter()
    while not game.done:
        is_v8 = game.to_play == v8_color
        agent = v8 if is_v8 else v7
        started = perf_counter()
        move = agent.select_move(game)
        elapsed = perf_counter() - started
        if is_v8:
            record = _move_record(len(game.history), elapsed, v8.diagnostics)
            if task.get('counterfactual') and v8.diagnostics.v8_route == 'own_vct':
                record['counterfactual'] = _counterfactual(game, seed, len(game.history), {**config, **search})
            v8_moves.append(record)
        else:
            v7_seconds.append(round(elapsed, 4))
        try:
            game.play(*move)
        except IllegalMove as exc:
            raise RuntimeError(f'{agent.name} returned illegal move {move}: {exc}') from exc
    result = 'draw' if game.winner is None else 'win' if game.winner == v8_color else 'loss'
    return {
        'key': task['key'], 'arm': task['arm'], 'pair': task['pair'], 'seed': seed,
        'v8_color': 'black' if v8_color == BLACK else 'white',
        'result': result, 'winner': game.winner, 'length': len(game.history),
        'moves': [list(m) for m in game.history],
        'v8_moves': v8_moves, 'v7_move_seconds': v7_seconds,
        'game_seconds': round(perf_counter() - started_game, 2),
    }


def build_tasks(args) -> tuple[list[dict], list[dict]]:
    tasks, openings = [], []
    for pair in range(args.pairs):
        opening_seed = derive_seed(args.seed, 'opening', pair)
        opening = make_opening(opening_seed, args.opening_random_plies, args.opening_radius)
        openings.append({'pair': pair, 'seed': opening_seed, 'moves': [list(m) for m in opening]})
        for color in (BLACK, WHITE):
            game_seed = derive_seed(args.seed, 'game', pair, color)
            tasks.append({
                'key': f'{args.arm}/{args.seed}/{pair}/{"black" if color == BLACK else "white"}',
                'arm': args.arm, 'pair': pair, 'seed': game_seed, 'v8_color': color,
                'opening': [list(m) for m in opening], 'counterfactual': args.counterfactual,
                'v8_overrides': args.search_overrides, 'v7_overrides': args.search_overrides,
            })
    return tasks, openings


def _dist(values) -> dict:
    values = sorted(values)
    if not values:
        return {'n': 0}
    return {'n': len(values), 'mean': round(sum(values) / len(values), 3),
            'median': round(median(values), 3),
            'p95': values[min(len(values) - 1, int(0.95 * len(values)))], 'max': values[-1]}


def summarize(games: list[dict]) -> dict:
    def score(subset):
        if not subset:
            return None
        return (sum(g['result'] == 'win' for g in subset) + 0.5 * sum(g['result'] == 'draw' for g in subset)) / len(subset)

    moves = [m for g in games for m in g['v8_moves']]
    routes, changed = {}, {}
    for m in moves:
        routes[m['route']] = routes.get(m['route'], 0) + 1
        if m['changed']:
            changed[m['route']] = changed.get(m['route'], 0) + 1
    ran_attack = [m for m in moves if m['attack']['candidates'] or m['attack']['calls']]
    ran_vct = [m for m in moves if m['route'] in ('stage4', 'stage5') and m['vct']['checked']]
    counterfactual = [m['counterfactual'] for m in moves if 'counterfactual' in m]
    ran_root = [m for m in moves if m.get('root', {}).get('checked')]
    records = [(g['key'], g['winner'], g['moves']) for g in sorted(games, key=lambda g: g['key'])]
    return {
        'games': len(games),
        'wins': sum(g['result'] == 'win' for g in games),
        'draws': sum(g['result'] == 'draw' for g in games),
        'losses': sum(g['result'] == 'loss' for g in games),
        'score': score(games),
        'score_by_v8_color': {c: score([g for g in games if g['v8_color'] == c]) for c in ('black', 'white')},
        'average_game_length': round(sum(g['length'] for g in games) / len(games), 1) if games else None,
        'v8_moves': len(moves),
        'v8_routes': routes,
        'v8_changed_by_route': changed,
        'v8_move_seconds': _dist([m['seconds'] for m in moves]),
        'v7_move_seconds': _dist([s for g in games for s in g['v7_move_seconds']]),
        'v8_b_attack': {
            'ran': len(ran_attack), 'wins': routes.get('own_vct', 0),
            'budget_exhausted': sum(m['attack']['exhausted'] for m in ran_attack),
            'seconds': _dist([m['attack']['seconds'] for m in ran_attack]),
        },
        'v8_a_stage_safety': {
            'ran': len(ran_vct),
            'changed': sum(m['changed'] for m in ran_vct),
            'widened': sum(m['vct']['widened'] for m in ran_vct),
            'budget_exhausted': sum(m['vct']['exhausted'] for m in ran_vct),
            'seconds': _dist([m['vct']['seconds'] for m in ran_vct]),
        },
        'v8_c_root': {
            'ran': len(ran_root),
            'changed': sum(m['changed'] for m in ran_root),
            'budget_exhausted': sum(m['root']['exhausted'] for m in ran_root),
            'seconds': _dist([m['root']['seconds'] for m in ran_root]),
        },
        'counterfactual': {
            'recorded': len(counterfactual),
            'v7_same_move': sum(1 for m in moves if 'counterfactual' in m
                                and m['counterfactual']['v7_move'] == m.get('played')),
            'v7_move_attack_status': {s: sum(c['v7_move_attack_status'] == s for c in counterfactual)
                                      for s in ('WIN', 'REFUTED', 'UNKNOWN')},
        },
        'outcome_history_sha256': hashlib.sha256(
            json.dumps(records, separators=(',', ':')).encode('utf-8')).hexdigest(),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_finished(path: Path | None, keys: set[str]) -> dict[str, dict]:
    finished = {}
    if path is None or not path.exists():
        return finished
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            record = json.loads(line)
            if record['key'] in keys:
                finished[record['key']] = record
    return finished


def _progress(record: dict) -> str:
    moves = record['v8_moves']
    worst = max((m['seconds'] for m in moves), default=0.0)
    routes = {}
    for m in moves:
        if m['route'] in ('own_vct', 'stage4', 'stage5'):
            routes[m['route']] = routes.get(m['route'], 0) + 1
    return (f"{record['key']} result={record['result']} moves={record['length']} "
            f"v8_max_s={worst:.1f} game_s={record['game_seconds']:.0f} routes={routes}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--arm', choices=tuple(ARMS), default='full')
    parser.add_argument('--pairs', type=int, default=10, help='opening pairs; 10 means 20 games')
    parser.add_argument('--seed', type=int, default=8401)
    parser.add_argument('--opening-random-plies', type=int, default=2)
    parser.add_argument('--opening-radius', type=int, default=2)
    parser.add_argument('--workers', type=int, default=1, help='games played in parallel processes')
    parser.add_argument('--counterfactual', action='store_true',
                        help="record V7's move at every V8-B move (adds one V7 search there)")
    parser.add_argument('--games-jsonl', type=Path, help='per-game lines, appended as games finish (resume)')
    parser.add_argument('--output', type=Path, help='summary + all games as one JSON file')
    parser.add_argument('--simulations', type=int, help='smoke tests only: both engines')
    parser.add_argument('--tactical-simulations', type=int, help='smoke tests only: both engines')
    args = parser.parse_args(argv)
    if args.pairs < 1 or args.workers < 1:
        parser.error('--pairs and --workers must be positive')
    args.search_overrides = {k: v for k, v in (('simulations', args.simulations),
                                               ('tactical_simulations', args.tactical_simulations))
                             if v is not None}

    tasks, openings = build_tasks(args)
    finished = load_finished(args.games_jsonl, {t['key'] for t in tasks})
    pending = [t for t in tasks if t['key'] not in finished]
    print(f'arm={args.arm} games={len(tasks)} finished={len(finished)} pending={len(pending)} '
          f'workers={args.workers}', flush=True)
    if args.games_jsonl is not None:
        args.games_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def keep(record):
        finished[record['key']] = record
        print(_progress(record), flush=True)
        if args.games_jsonl is not None:
            with args.games_jsonl.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, separators=(',', ':')) + '\n')

    if args.workers == 1:
        for task in pending:
            keep(play_one(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(play_one, t) for t in pending]):
                keep(future.result())

    games = [finished[t['key']] for t in tasks]
    for game in games:  # the played move is needed for the counterfactual comparison
        for m in game['v8_moves']:
            m['played'] = game['moves'][m['ply']]
    summary = summarize(games)
    payload = {
        'format': FORMAT, 'arm': args.arm, 'seed': args.seed, 'pairs': args.pairs,
        'opening_random_plies': args.opening_random_plies, 'opening_radius': args.opening_radius,
        'counterfactual': args.counterfactual, 'search_overrides': args.search_overrides,
        'git_commit': _git_commit(), 'v8_config': v8_config(args.arm, args.search_overrides),
        'v7_config': {**V7_FINAL, **args.search_overrides},
        'openings': openings, 'summary': summary, 'games': games,
    }
    print(json.dumps({'arm': args.arm, **summary}, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
