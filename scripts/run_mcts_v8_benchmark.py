"""V8-4: MCTS-v8 against frozen V7 (or a V8 arm) in real games (docs/mcts-v8-teacher.md §11.11, §12.4).

Paired openings as in ``run_mcts_v7_benchmark.py``: each random opening is played
twice with colours swapped. Openings and game seeds depend only on ``--seed`` and
the pair index, never on ``--arm``, so runs of different arms with the same seed
are paired (ablation).

Arms (V8 configuration):
    full       V8-A + V8-B + V8-C, V8-C aggressive (V8_DEFAULTS)
    full_veto  V8-A + V8-B + V8-C, V8-C veto (switch only on a proven loss, §12.3, §12.8)
    full_r250  full with root_node_budget 250,000 instead of 400,000 (§12.8)
    full_policy         full + H3 policy order inside the root safety tiers (H4-a, §12.13)
    full_policy_recall  full_policy + up to 8 policy moves added to the root candidates (H4-a+b)

H5 PUCT arms (§12.16): the tree route runs ``analysis.puct_v8`` instead of the V5 tree, on the
same action sets and rollouts; only the prior differs. They need ``--puct-c`` (the value fixed
by ``scripts/h5_calibrate_puct.py``), which applies to the tested arm only.
    puct_uniform  PUCT, uniform prior (the tree-rule control)
    puct_heur     PUCT, 1/rank prior over V8's own candidate order
    puct_policy   PUCT, H3 policy prior (one NN call per expanded node)

Policy arms load ``--policy-checkpoint`` (H3 ``best.pt`` + ``best.json``) in every worker and
fail if it cannot be loaded; they never fall back to the baseline. One NN call per tree move.
    ab      V8-A + V8-B        (root_vct_safety=False; the pilot's "full")
    a_only  V8-A only          (the pilot's "a_only")
    b_only  V8-B only
    c_only  V8-C only
    off     none               (V8 == V7; sanity check)

Opponent (``--opponent``): ``v7`` (frozen V7, default) or ``v8:<arm>`` (V8 with
that arm's configuration, e.g. ``v8:b_only``, which can play VCT1 attacks that
V7 does not see). The opponent does not change openings or seeds, so runs of
different arms against the same opponent are paired. Game keys carry the
opponent unless it is ``v7`` (old JSONL files resume unchanged).

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
    'full_veto': {'root_vct_mode': 'veto'},
    'full_r250': {'root_node_budget': 250_000},
    'full_policy': {'root_policy_order': True},
    'full_policy_recall': {'root_policy_order': True, 'root_policy_extra': 8},
    'puct_uniform': {'tree_mode': 'puct', 'puct_prior': 'uniform'},
    'puct_heur': {'tree_mode': 'puct', 'puct_prior': 'heuristic'},
    'puct_policy': {'tree_mode': 'puct', 'puct_prior': 'policy'},
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


def parse_opponent(value: str) -> str:
    """``v7`` or ``v8:<arm>``; raises ValueError otherwise."""
    if value == 'v7':
        return value
    kind, _, arm = value.partition(':')
    if kind != 'v8' or arm not in ARMS:
        raise ValueError(f"opponent must be 'v7' or 'v8:<arm>' with arm in {sorted(ARMS)}")
    return value


def needs_policy(config: dict) -> bool:
    return bool(config.get('root_policy_order') or config.get('root_policy_extra')
                or (config.get('tree_mode') == 'puct' and config.get('puct_prior') == 'policy'))


def load_policy(checkpoint):
    """H3 root policy (torch); raises if unavailable, so a policy arm never runs as the baseline."""
    if not checkpoint:
        raise ValueError('this arm needs --policy-checkpoint')
    from hybrid.h4_policy import RootPolicy  # Track B adapter: torch is imported only here
    return RootPolicy(checkpoint)


def make_opponent(opponent: str, seed: int, overrides: dict, policy_checkpoint=None):
    """Opponent agent; V8 opponents use the same seed stream V7 would."""
    if opponent == 'v7':
        return MCTSV7Agent(seed=seed, **overrides)
    config = v8_config(opponent.partition(':')[2], overrides)
    policy = load_policy(policy_checkpoint) if needs_policy(config) else None
    return MCTSV8Agent(seed=seed, root_policy=policy, **{k: v for k, v in config.items() if k in V8_DEFAULTS})


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
            'switch': diag.v8_root_switch,
        },
        'policy': {
            'seconds': round(diag.v8_policy_seconds, 4), 'added': [list(m) for m in diag.v8_policy_added],
            'added_kept': [list(m) for m in diag.v8_policy_added_kept],
            'added_opened': diag.v8_policy_added_opened, 'displaced': diag.v8_policy_displaced,
            'rank': diag.v8_policy_rank, 'prob': round(diag.v8_policy_prob, 5),
            'root_order_rank': diag.v8_root_order_rank,
        },
        'tree': {
            'mode': diag.v8_tree_mode, 'seconds': round(diag.v8_tree_seconds, 4),
            'simulations': diag.v8_tree_simulations, 'prior': diag.v8_puct_prior,
            'nn_calls': diag.v8_puct_nn_calls, 'prior_fallbacks': diag.v8_puct_prior_fallbacks,
            'prior_entropy': round(diag.v8_puct_prior_entropy, 4),
            'prior_top': list(diag.v8_puct_prior_top) if diag.v8_puct_prior_top is not None else None,
            'prior_top_prob': round(diag.v8_puct_prior_top_prob, 5),
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
    policy = load_policy(task.get('policy_checkpoint')) if needs_policy(config) else None
    v8 = MCTSV8Agent(seed=derive_seed(seed, 'v8'), root_policy=policy,
                     **{k: v for k, v in config.items() if k in V8_DEFAULTS})
    opponent = task.get('opponent', 'v7')
    opp = make_opponent(opponent, derive_seed(seed, 'v7'), task.get('v7_overrides', {}),
                        task.get('policy_checkpoint'))
    game = Game()
    for move in task['opening']:
        game.play(*move)
    v8_moves, opp_seconds, opp_routes = [], [], {}
    started_game = perf_counter()
    while not game.done:
        is_v8 = game.to_play == v8_color
        agent = v8 if is_v8 else opp
        started = perf_counter()
        move = agent.select_move(game)
        elapsed = perf_counter() - started
        if is_v8:
            record = _move_record(len(game.history), elapsed, v8.diagnostics)
            if task.get('counterfactual') and v8.diagnostics.v8_route == 'own_vct':
                record['counterfactual'] = _counterfactual(game, seed, len(game.history), {**config, **search})
            v8_moves.append(record)
        else:
            opp_seconds.append(round(elapsed, 4))
            route = getattr(opp.diagnostics, 'v8_route', '')
            if route:
                opp_routes[route] = opp_routes.get(route, 0) + 1
        try:
            game.play(*move)
        except IllegalMove as exc:
            raise RuntimeError(f'{agent.name} returned illegal move {move}: {exc}') from exc
    result = 'draw' if game.winner is None else 'win' if game.winner == v8_color else 'loss'
    return {
        'key': task['key'], 'arm': task['arm'], 'opponent': opponent, 'pair': task['pair'], 'seed': seed,
        'v8_color': 'black' if v8_color == BLACK else 'white',
        'result': result, 'winner': game.winner, 'length': len(game.history),
        'moves': [list(m) for m in game.history],
        'v8_moves': v8_moves, 'opponent_move_seconds': opp_seconds, 'opponent_routes': opp_routes,
        'game_seconds': round(perf_counter() - started_game, 2),
    }


def build_tasks(args) -> tuple[list[dict], list[dict]]:
    tasks, openings = [], []
    arm_overrides = getattr(args, 'arm_overrides', {})
    for pair in range(args.pairs):
        opening_seed = derive_seed(args.seed, 'opening', pair)
        opening = make_opening(opening_seed, args.opening_random_plies, args.opening_radius)
        openings.append({'pair': pair, 'seed': opening_seed, 'moves': [list(m) for m in opening]})
        for color in (BLACK, WHITE):
            game_seed = derive_seed(args.seed, 'game', pair, color)
            arm = args.arm + (''.join(f'[{k}={v}]' for k, v in sorted(arm_overrides.items()))
                              if arm_overrides else '')  # a resumed JSONL never mixes settings
            prefix = arm if args.opponent == 'v7' else f'{arm}@{args.opponent}'
            tasks.append({
                'key': f'{prefix}/{args.seed}/{pair}/{"black" if color == BLACK else "white"}',
                'arm': args.arm, 'opponent': args.opponent, 'pair': pair, 'seed': game_seed, 'v8_color': color,
                'opening': [list(m) for m in opening], 'counterfactual': args.counterfactual,
                'v8_overrides': {**args.search_overrides, **arm_overrides},
                'v7_overrides': args.search_overrides,
                'policy_checkpoint': args.policy_checkpoint,
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
    with_policy = [m for m in moves if m.get('policy', {}).get('root_order_rank')]
    with_tree = [m for m in moves if m.get('tree', {}).get('mode')]
    with_puct = [m for m in with_tree if m['tree']['mode'] == 'puct']
    root_first = [m['root']['checked'][0][1] for m in ran_root]  # V7's move is always checked first
    opp_routes = {}
    for g in games:
        for route, n in g.get('opponent_routes', {}).items():
            opp_routes[route] = opp_routes.get(route, 0) + n
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
        'opponent_move_seconds': _dist([s for g in games
                                        for s in g.get('opponent_move_seconds', g.get('v7_move_seconds', []))]),
        'opponent_routes': opp_routes,
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
            'switched_on_proven_loss': sum(m['root'].get('switch') == 'proven_loss' for m in ran_root),
            'switched_on_unknown': sum(m['root'].get('switch') == 'unknown' for m in ran_root),
            'v7_move_status': {s: root_first.count(s) for s in ('SAFE', 'UNSAFE', 'UNKNOWN')},
            'children_checked': _dist([len(m['root']['checked']) for m in ran_root]),
            'budget_exhausted': sum(m['root']['exhausted'] for m in ran_root),
            'seconds': _dist([m['root']['seconds'] for m in ran_root]),
        },
        'policy': {
            'tree_moves': len(with_policy),
            'seconds': _dist([m['policy']['seconds'] for m in with_policy]),
            'added_kept': sum(len(m['policy']['added_kept']) for m in with_policy),
            'added_opened': sum(m['policy']['added_opened'] for m in with_policy),
            'moves_with_added_opened': sum(m['policy']['added_opened'] > 0 for m in with_policy),
            'tree_chose_added': sum(m['v7_move'] in m['policy']['added_kept'] for m in with_policy),
            'displaced': _dist([m['policy']['displaced'] for m in with_policy]),
            'policy_rank': _dist([m['policy']['rank'] for m in with_policy]),
            'root_order_rank': _dist([m['policy']['root_order_rank'] for m in with_policy]),
        },
        'tree': {
            'moves': len(with_tree),
            'seconds': _dist([m['tree']['seconds'] for m in with_tree]),
            'simulations_per_second': round(sum(m['tree']['simulations'] for m in with_tree)
                                            / max(1e-9, sum(m['tree']['seconds'] for m in with_tree)), 1)
            if with_tree else None,
            'puct_moves': len(with_puct),
            'nn_calls': sum(m['tree']['nn_calls'] for m in with_puct),
            'prior_fallbacks': sum(m['tree']['prior_fallbacks'] for m in with_puct),
            'prior_entropy': _dist([m['tree']['prior_entropy'] for m in with_puct]),
            'tree_chose_prior_top': sum(m['v7_move'] == m['tree']['prior_top'] for m in with_puct),
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


def _git_dirty() -> bool | None:
    """True when tracked files differ from HEAD (results then do not match ``git_commit``)."""
    try:
        out = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.strip())


def file_sha256(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def provenance(policy_checkpoint) -> dict:
    """What a result depends on besides the code: tree state and the exact policy bytes (§12.19)."""
    checkpoint = Path(policy_checkpoint) if policy_checkpoint else None
    return {
        'git_dirty': _git_dirty(),
        'policy_checkpoint_sha256': file_sha256(checkpoint) if checkpoint else None,
        'policy_metadata_sha256': file_sha256(checkpoint.with_suffix('.json')) if checkpoint else None,
    }


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
    parser.add_argument('--opponent', default='v7', help="'v7' (default) or 'v8:<arm>', e.g. v8:b_only")
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'),
                        help='H3 best.pt for the policy arms (best.json next to it)')
    parser.add_argument('--pairs', type=int, default=10, help='opening pairs; 10 means 20 games')
    parser.add_argument('--seed', type=int, default=8401)
    parser.add_argument('--opening-random-plies', type=int, default=2)
    parser.add_argument('--opening-radius', type=int, default=2)
    parser.add_argument('--workers', type=int, default=1, help='games played in parallel processes')
    parser.add_argument('--counterfactual', action='store_true',
                        help="record V7's move at every V8-B move (adds one V7 search there)")
    parser.add_argument('--games-jsonl', type=Path, help='per-game lines, appended as games finish (resume)')
    parser.add_argument('--output', type=Path, help='summary + all games as one JSON file')
    parser.add_argument('--puct-c', type=float, help='c_puct for a PUCT arm (required there; tested arm only)')
    parser.add_argument('--arm-simulations', type=int,
                        help='tested arm only: simulations (time-matched check, §12.16)')
    parser.add_argument('--arm-tactical-simulations', type=int,
                        help='tested arm only: tactical simulations (time-matched check, §12.16)')
    parser.add_argument('--simulations', type=int, help='smoke tests only: both engines')
    parser.add_argument('--tactical-simulations', type=int, help='smoke tests only: both engines')
    args = parser.parse_args(argv)
    if args.pairs < 1 or args.workers < 1:
        parser.error('--pairs and --workers must be positive')
    try:
        parse_opponent(args.opponent)
    except ValueError as exc:
        parser.error(str(exc))
    for name in (args.arm, args.opponent.partition(':')[2]):
        if name and needs_policy(v8_config(name)):
            try:
                load_policy(args.policy_checkpoint)  # fail before any game starts
            except (OSError, ValueError, ImportError) as exc:
                parser.error(f'policy arm {name}: {exc}')
    args.search_overrides = {k: v for k, v in (('simulations', args.simulations),
                                               ('tactical_simulations', args.tactical_simulations))
                             if v is not None}
    is_puct = v8_config(args.arm).get('tree_mode') == 'puct'
    if is_puct and args.puct_c is None:
        parser.error(f'{args.arm} needs --puct-c (the calibrated value, §12.16)')
    if not is_puct and args.puct_c is not None:
        parser.error('--puct-c only applies to the puct_* arms')
    if args.opponent != 'v7' and v8_config(args.opponent.partition(':')[2]).get('tree_mode') == 'puct':
        parser.error('a PUCT arm cannot be the opponent (its c_puct would be unset)')
    args.arm_overrides = {k: v for k, v in (('puct_c', args.puct_c), ('simulations', args.arm_simulations),
                                            ('tactical_simulations', args.arm_tactical_simulations))
                          if v is not None}

    tasks, openings = build_tasks(args)
    finished = load_finished(args.games_jsonl, {t['key'] for t in tasks})
    pending = [t for t in tasks if t['key'] not in finished]
    print(f'arm={args.arm} opponent={args.opponent} games={len(tasks)} finished={len(finished)} pending={len(pending)} '
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
    payload_checkpoint = (args.policy_checkpoint if needs_policy(v8_config(args.arm)) or (
        args.opponent != 'v7' and needs_policy(v8_config(args.opponent.partition(':')[2]))) else None)
    payload = {
        'format': FORMAT, 'arm': args.arm, 'opponent': args.opponent, 'seed': args.seed, 'pairs': args.pairs,
        'opening_random_plies': args.opening_random_plies, 'opening_radius': args.opening_radius,
        'counterfactual': args.counterfactual, 'search_overrides': args.search_overrides,
        'arm_overrides': args.arm_overrides,
        'git_commit': _git_commit(), 'v8_config': v8_config(args.arm, {**args.search_overrides,
                                                                        **args.arm_overrides}),
        'policy_checkpoint': payload_checkpoint,
        'provenance': provenance(payload_checkpoint),
        'v7_config': {**V7_FINAL, **args.search_overrides},
        'opponent_config': (None if args.opponent == 'v7' else
                            v8_config(args.opponent.partition(':')[2], args.search_overrides)),
        'openings': openings, 'summary': summary, 'games': games,
    }
    print(json.dumps({'arm': args.arm, 'opponent': args.opponent, **summary}, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
