"""H7-c search-budget ablation (docs/h7-c1.md, SB): is TB-V3's 50-simulation tree too shallow to improve on H3?

Measurement only. The frozen H3 policy and H6 v3 value throughout; the only variable is the tree budget, scaled by a
factor k on both of TB-V3's budgets (``simulations`` 50k, ``tactical_simulations`` 100k: the engine takes the second
when the root's tactical score reaches 1800, so scaling one alone would change two things). k = 1 is the frozen TB-V3
itself (``h7c_arena.make_side``); k > 1 is the same agent with only those two numbers changed (a measurement
configuration with its own config, never an engine of record, a self-play engine or a promotion candidate).

    play       S(50k) vs S50 (the frozen engine), 50 pairs, seed 8741 for every k (``hybrid.h6_labels.RESERVED_SEEDS``:
               pair i has the same opening and game seeds in every arm); per move of the S(50k) side: route, time,
               the tree budget used, the chosen move against the root prior's top, visit top share and visited
               children; per game the time of both sides
    positions  the c1 policy val rows of kind ``visits`` (tree-route self-play positions; the file must be the
               evidence manifest's), each searched once per k by a fresh agent: the k = 1 search must give the recorded
               visits (reproduction count); per k the chosen move vs the root prior top (how far the search departs
               from H3), visit entropy / top share / visited children, the agreement of the chosen move with the
               largest k, Jensen-Shannon to the largest k's visits, seconds
    summary    the play outputs and the positions output: per k the score, the pair bootstrap of (score - 0.5), the
               time ratio, and the pre-registered reading (docs/h7-c1.md): a budget is STRONGER when the 95% lower bound
               of (score - 0.5) is above 0; any STRONGER budget gives BUDGET_HELPS (the smallest such k is the one to
               propose), none gives NOT_DETECTED. The 50-pair reading always uses pairs 0-49; an S400 arm extended to
               150 pairs (``play --factor 8 --pairs 150``, same --jsonl) adds the saturation reading (``SATURATION``):
               STRONGER (lower bound above 0), SATURATED (upper bound below 0.075) or INCONCLUSIVE

    python -u scripts/h7c_budget.py play --factor 4 --policy-checkpoint runs/h3_policy_64x4/best.pt \\
        --value-checkpoint runs/h6_value_head_v3/best.pt --device cuda --workers 6 \\
        --jsonl runs/h7c/sb/s200.jsonl --output runs/h7c/sb/s200.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import BLACK, WHITE, Game, IllegalMove  # noqa: E402

FORMAT = 'h7c-budget-v1'
SEED = 8741
PAIRS = 50
FACTORS = (2, 4, 8)          # S100, S200, S400 against S50
# Saturation stage (pre-registered, docs/h7-c1.md "MCTS-v9 진입 조건"): when no budget is STRONGER at 50 pairs, S400
# is extended to 150 pairs on the same seed (pairs 0-49 are the 50-pair run's). SATURATED: the 95% upper bound of
# (score - 0.5) is below 0.075 (+52 Elo). With a pair-score SD of 0.325 (c1 official) this verdict comes with
# probability 0.81 when S400 is truly no stronger, 0.15 at a true +35 Elo and 0.025 at +52 Elo.
SATURATION = {'factor': 8, 'pairs': 150, 'upper_bound_below': 0.075}
POSITION_FACTORS = (1, 2, 4, 8)
BASE = {'simulations': 50, 'tactical_simulations': 100}
_WORKER: dict = {}


def budget(factor: int) -> dict:
    if factor < 1:
        raise ValueError('the budget factor is at least 1')
    return {k: v * factor for k, v in BASE.items()}


# --- agents ------------------------------------------------------------------------------------------------------

def make_budget_agent(factor: int, policy_path: str, value_path: str, seed: int, device: str):
    """k = 1: the frozen TB-V3 (the candidate factory with these models); k > 1: the same agent and checks, only the
    two tree budgets scaled."""
    import h7c_arena as ar

    if factor == 1:
        return ar.make_side({'policy': policy_path, 'value': value_path}, seed, device)
    from analysis import tb_v2_runtime, tb_v2_value, tb_v3, tb_v3_fast5, tb_v3_net, track_b_guard
    from analysis.mcts_v8_agent import MCTSV8Agent
    from analysis.tb_v2_value_e4v2 import E4v2AttackAgent

    ar.install_runtime()
    policy, leaf = ar._loaded(policy_path, value_path, device)
    tb_v3.check_identity()
    tb_v3_net.check_models(policy, leaf, policy_path, value_path)
    track_b_guard.check_runtime_sources()
    tb_v3_fast5.check_sources()
    tb_v3_fast5.check_bound()
    if tb_v2_value.config_hash() != tb_v2_runtime.ENGINE_HASH:
        raise RuntimeError('analysis.tb_v2_value is not the frozen configuration')
    base = MCTSV8Agent(seed=seed, root_policy=policy, leaf_value=leaf,
                       **{**tb_v2_value.TB_V2_VALUE, **budget(factor)}, **tb_v2_value.ENGINE_OPTIONS)
    agent = E4v2AttackAgent(base, attack=True, config=tb_v3.ATTACK)
    agent.name = f'TB-V3 S{BASE["simulations"] * factor} (measurement)'
    return agent


def decision(agent, game: Game) -> dict:
    from hybrid.h7c_targets import raw_visits

    started = perf_counter()
    move = agent.select_move(game)
    seconds = perf_counter() - started
    d = agent.diagnostics
    visits = raw_visits(d) if d.v8_route == 'tree' else []
    prior_top = list(d.v8_puct_prior_top) if visits and d.v8_puct_prior_top is not None else None
    return {'move': [int(move[0]), int(move[1])], 'route': d.v8_route, 'seconds': round(seconds, 4),
            'tree_budget': d.v8_tree_simulations if visits else None, 'prior_top': prior_top, 'visits': visits}


def visit_stats(visits: list[list]) -> dict:
    counts = [v[2] for v in visits if v[2] > 0]
    total = sum(counts)
    ps = [c / total for c in counts] if total else []
    return {'entropy': -sum(p * math.log(p) for p in ps), 'top_share': max(ps) if ps else None,
            'visited_children': len(counts), 'children': len(visits)}


def js(p: dict, q: dict) -> float:
    def kl(a, b):
        return sum(x * math.log(x / b[m]) for m, x in a.items() if x > 0)
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    return 0.5 * kl({k: p.get(k, 0.0) for k in keys}, m) + 0.5 * kl({k: q.get(k, 0.0) for k in keys}, m)


def distribution(visits: list[list]) -> dict:
    total = sum(v[2] for v in visits)
    return {(v[0], v[1]): v[2] / total for v in visits if v[2] > 0} if total else {}


# --- games -------------------------------------------------------------------------------------------------------

def build_tasks(factor: int, pairs: int = PAIRS) -> list[dict]:
    from run_mcts_v8_benchmark import derive_seed, make_opening
    import h7c_arena as ar

    tasks = []
    for pair in range(pairs):
        opening = make_opening(derive_seed(SEED, 'opening', pair), ar.OPENING['random_plies'], ar.OPENING['radius'])
        for color in (BLACK, WHITE):
            name = 'black' if color == BLACK else 'white'
            tasks.append({'key': f's{BASE["simulations"] * factor}/{SEED}/{pair}/{name}', 'factor': factor,
                          'pair': pair, 'seed': derive_seed(SEED, 'game', pair, color), 'tested_color': color,
                          'opening': [list(m) for m in opening]})
    return tasks


def play_game(task: dict, make=None, max_plies: int | None = None) -> dict:
    """S(50k) (the tested side) against S50 on one opening; ``max_plies`` only for tests (then never scored)."""
    from run_mcts_v8_benchmark import derive_seed

    ctx = task['context']
    make = make or (lambda k, seed: make_budget_agent(k, ctx['policy'], ctx['value'], seed, ctx['device']))
    tested = make(task['factor'], derive_seed(task['seed'], 'tested'))
    opponent = make(1, derive_seed(task['seed'], 'opponent'))
    game = Game()
    for move in task['opening']:
        game.play(*move)
    color, moves, seconds, illegal, truncated = task['tested_color'], [], {'tested': 0.0, 'opponent': 0.0}, None, False
    started = perf_counter()
    while not game.done:
        if max_plies is not None and len(game.history) >= max_plies:
            truncated = True
            break
        is_tested = game.to_play == color
        d = decision(tested if is_tested else opponent, game)
        seconds['tested' if is_tested else 'opponent'] += d['seconds']
        if is_tested:
            v = visit_stats(d['visits']) if d['visits'] else None
            moves.append({'ply': len(game.history), 'route': d['route'], 'seconds': d['seconds'],
                          'tree_budget': d['tree_budget'], 'chose_prior_top': d['prior_top'] == d['move']
                          if d['prior_top'] else None, 'top_share': v and v['top_share'],
                          'visited_children': v and v['visited_children']})
        try:
            game.play(*d['move'])
        except IllegalMove as exc:
            illegal = {'side': 'tested' if is_tested else 'opponent', 'ply': len(game.history), 'reason': str(exc)}
            break
    if illegal is not None:
        result = 'loss' if illegal['side'] == 'tested' else 'opponent_illegal'
    elif truncated:
        result = 'truncated'
    else:
        result = 'draw' if game.winner is None else 'win' if game.winner == color else 'loss'
    return {'key': task['key'], 'signature': ctx['signature'], 'factor': task['factor'], 'pair': task['pair'],
            'seed': task['seed'], 'tested_color': 'black' if color == BLACK else 'white', 'result': result,
            'winner': game.winner, 'length': len(game.history), 'moves': [list(m) for m in game.history],
            'illegal': illegal, 'truncated': truncated, 'tested_moves': moves,
            'seconds': {k: round(v, 3) for k, v in seconds.items()},
            'game_seconds': round(perf_counter() - started, 2)}


def _play_chunk(tasks: list[dict]) -> list[dict]:
    return [play_game(t, max_plies=t.get('max_plies')) for t in tasks]


def identity(args, factor: int, git: dict) -> dict:
    from analysis import tb_v3, tb_v3_fast5, tb_v3_net

    models = tb_v3_net.model_identity(args.policy_checkpoint, args.value_checkpoint)
    if models != tb_v3.MODELS and not args.allow_stand_in:
        raise ValueError('the ablation runs the frozen H3 / H6 v3 models only')
    return {'format': FORMAT, 'factor': factor, 'budget': budget(factor), 'opponent_budget': budget(1),
            'seed': SEED, 'engine_id': tb_v3.ENGINE_ID, 'engine_hash': tb_v3.ENGINE_HASH,
            'runtime_hash': tb_v3_fast5.runtime_hash(args.device), 'value_device': args.device.split(':')[0],
            'models': models, 'git': git}


def play(args, log=print) -> dict:
    import h7c_arena as ar

    if args.factor not in FACTORS:
        raise ValueError(f'--factor is one of {FACTORS}')
    pairs = getattr(args, 'pairs', None) or PAIRS
    if pairs not in (PAIRS, SATURATION['pairs']) and not getattr(args, 'max_plies', None):
        raise ValueError(f'--pairs is {PAIRS}, or {SATURATION["pairs"]} for the saturation stage')
    if pairs == SATURATION['pairs'] and args.factor != SATURATION['factor']:
        raise ValueError(f'the saturation stage extends factor {SATURATION["factor"]} only')
    git = ar._git()
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty (or git is unavailable): the games must name their code commit')
    ident = identity(args, args.factor, git)
    sig = ar.signature(ident)
    tasks = build_tasks(args.factor, pairs)
    finished = ar.load_finished(args.jsonl, {t['key'] for t in tasks}, sig, log=log)
    context = {'policy': str(args.policy_checkpoint), 'value': str(args.value_checkpoint), 'device': args.device,
               'signature': sig}
    todo = [{**t, 'context': context, **({'max_plies': args.max_plies} if getattr(args, 'max_plies', None) else {})}
            for t in tasks if t['key'] not in finished]
    log(f'h7c budget: S{BASE["simulations"] * args.factor} vs S50, seed {SEED}, games {len(tasks)} '
        f'({len(finished)} done), workers {args.workers}, signature {sig[:12]}', flush=True)
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    if todo:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        with args.jsonl.open('a', encoding='utf-8') as out, ProcessPoolExecutor(
                max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            def keep(_, rows):
                for row in rows:
                    out.write(json.dumps(row, separators=(',', ':')) + '\n')
                    finished[row['key']] = row
                out.flush()
            run_chunks(pool, _play_chunk, [[t] for t in todo], Progress(f'h7c-budget-k{args.factor}', len(todo),
                                                                         'games'), len, unit_name='games',
                       on_result=keep)
    games = [finished[t['key']] for t in tasks]
    return {'format': f'{FORMAT}-arm', 'identity': ident, 'signature': sig, 'summary': summarize_arm(games),
            'fingerprint': ar.fingerprint(games), 'games': games}


def summarize_arm(games: list[dict]) -> dict:
    import h7c_arena as ar

    out = ar.summarize_arm(games)
    tested = sum(g['seconds']['tested'] for g in games)
    opponent = sum(g['seconds']['opponent'] for g in games)
    tree = [m for g in games for m in g['tested_moves'] if m['route'] == 'tree']
    out['cost'] = {'tested_seconds': round(tested, 1), 'opponent_seconds': round(opponent, 1),
                   'time_ratio': round(tested / opponent, 3) if opponent else None,
                   'tree_moves': len(tree), 'tree_seconds_mean': round(sum(m['seconds'] for m in tree) / len(tree), 4)
                   if tree else None}
    out['tree'] = {'chose_prior_top': round(sum(bool(m['chose_prior_top']) for m in tree) / len(tree), 4)
                   if tree else None,
                   'top_share': round(sum(m['top_share'] for m in tree) / len(tree), 4) if tree else None,
                   'visited_children': round(sum(m['visited_children'] for m in tree) / len(tree), 3)
                   if tree else None}
    return out


# --- positions ---------------------------------------------------------------------------------------------------

def _init_worker(policy: str, value: str, device: str) -> None:
    _WORKER.update(policy=policy, value=value, device=device)


def search_position(task: dict) -> dict:
    from run_mcts_v8_benchmark import derive_seed

    game = Game()
    for move in task['moves']:
        game.play(*move)
    out = {'index': task['index'], 'side': 'black' if len(task['moves']) % 2 == 0 else 'white', 'by_factor': {}}
    for k in POSITION_FACTORS:
        agent = make_budget_agent(k, _WORKER['policy'], _WORKER['value'], derive_seed(SEED, 'position', task['index']),
                                  _WORKER['device'])
        out['by_factor'][str(k)] = decision(agent, game)
    recorded = sorted(tuple(v[:3]) for v in task['raw_visits'])
    replayed = sorted(tuple(v[:3]) for v in out['by_factor']['1']['visits'])
    out['reproduced'] = recorded == replayed
    return out


def _position_chunk(tasks: list[dict]) -> list[dict]:
    return [search_position(t) for t in tasks]


def positions(args, log=print) -> dict:
    import h7c_arena as ar

    git = ar._git()
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty (or git is unavailable): the record must name its code commit')
    evidence = json.loads(Path(args.evidence).read_text(encoding='utf-8'))
    if evidence.get('files_sha256', {}).get('policy_rows.jsonl.gz') != ar._file_sha(args.policy_rows):
        raise ValueError(f'{args.policy_rows} is not the policy rows of the evidence manifest')
    ident = identity(args, max(POSITION_FACTORS), git)
    with gzip.open(args.policy_rows, 'rt', encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    chosen = [r for r in rows if not r['mask'] and r['split'] == 'val' and r['kind'] == 'visits']
    if args.limit:
        chosen = chosen[:args.limit]
    tasks = [{'index': i, 'moves': r['moves'], 'raw_visits': r['raw_visits']} for i, r in enumerate(chosen)]
    results = []
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'), initializer=_init_worker,
                             initargs=(str(args.policy_checkpoint), str(args.value_checkpoint), args.device)) as pool:
        run_chunks(pool, _position_chunk, [[t] for t in tasks], Progress('h7c-budget-positions', len(tasks),
                                                                         'positions'), len, unit_name='positions',
                   on_result=lambda _, part: results.extend(part))
    results.sort(key=lambda r: r['index'])
    return {'format': f'{FORMAT}-positions', 'identity': {**ident, 'factors': POSITION_FACTORS},
            'policy_rows_sha256': ar._file_sha(args.policy_rows), 'positions': len(results),
            'summary': summarize_positions(results), 'rows': results}


def summarize_positions(results: list[dict]) -> dict:
    """Per k, on the positions every budget searched in the tree (the same set for every k)."""
    top = str(max(POSITION_FACTORS))
    paired = [r for r in results if all(r['by_factor'][str(k)]['visits'] for k in POSITION_FACTORS)]
    out = {'positions': len(results), 'tree_at_every_budget': len(paired),
           'reproduced_at_k1': sum(r['reproduced'] for r in results), 'by_factor': {}}
    for k in map(str, POSITION_FACTORS):
        ds = [r['by_factor'][k] for r in paired]
        stats = [visit_stats(d['visits']) for d in ds]
        n = len(ds)
        out['by_factor'][k] = {
            'budget': budget(int(k)), 'n': n,
            'chose_prior_top': round(sum(d['move'] == d['prior_top'] for d in ds) / n, 4) if n else None,
            'agrees_with_largest': round(sum(d['move'] == r['by_factor'][top]['move'] for d, r in zip(ds, paired)) / n, 4)
            if n else None,
            'js_to_largest': round(sum(js(distribution(d['visits']), distribution(r['by_factor'][top]['visits']))
                                       for d, r in zip(ds, paired)) / n, 5) if n else None,
            'entropy': round(sum(s['entropy'] for s in stats) / n, 4) if n else None,
            'top_share': round(sum(s['top_share'] for s in stats) / n, 4) if n else None,
            'visited_children': round(sum(s['visited_children'] for s in stats) / n, 3) if n else None,
            'seconds': round(sum(d['seconds'] for d in ds) / n, 4) if n else None,
            'by_side': {s: round(sum(d['move'] == d['prior_top'] for d, r in zip(ds, paired) if r['side'] == s)
                                 / max(1, sum(r['side'] == s for r in paired)), 4) for s in ('black', 'white')}}
    return out


# --- summary -----------------------------------------------------------------------------------------------------

def summary(args) -> dict:
    import h7c_arena as ar

    arms, arm_sha = {}, {}
    for path in args.arms:
        record = json.loads(Path(path).read_text(encoding='utf-8'))
        k = record['identity']['factor']
        arm_sha[f'S{BASE["simulations"] * k}'] = ar._file_sha(Path(path))
        if ar.signature(record['identity']) != record['signature'] or any(
                g['signature'] != record['signature'] for g in record['games']):
            raise ValueError(f'{path}: its games do not match its identity')
        if record['summary'] != summarize_arm(record['games']) or record['fingerprint'] != ar.fingerprint(record['games']):
            raise ValueError(f'{path}: its summary or fingerprint is not the one its games give')
        if k in arms:
            raise ValueError(f'factor {k} given twice')
        arms[k] = record
    if sorted(arms) != list(FACTORS):
        raise ValueError(f'the arms must be the factors {FACTORS}')
    shared = {json.dumps({k: v for k, v in r['identity'].items() if k not in ('factor', 'budget')}, sort_keys=True)
              for r in arms.values()}
    if len(shared) != 1:
        raise ValueError('the arms differ in engine, runtime, models, seed or commit')
    rows, saturation = {}, None
    for k, record in sorted(arms.items()):
        full = record['summary']
        played = full['complete_pairs'] + len(full['opponent_illegal_pairs'])  # every pair finished
        if full['truncated'] or played not in ((PAIRS, SATURATION['pairs']) if k == SATURATION['factor'] else (PAIRS,)):
            raise ValueError(f'factor {k}: {full["complete_pairs"]} complete pairs')
        # the 50-pair reading always uses pairs 0-49, also when the arm was extended
        s = summarize_arm([g for g in record['games'] if g['pair'] < PAIRS])
        if s['complete_pairs'] + len(s['opponent_illegal_pairs']) != PAIRS:
            raise ValueError(f'factor {k}: pairs 0-{PAIRS - 1} are not all played')
        if played == SATURATION['pairs']:
            upper, lower150 = full['improvement_ci95'][1], full['improvement_ci95'][0]
            saturation = {'pairs': SATURATION['pairs'], 'score': full['score'], 'score_ci95': full['score_ci95'],
                          'improvement_ci95': full['improvement_ci95'], 'elo': full['elo'],
                          'rule': SATURATION, 'reading': 'STRONGER' if lower150 > 0 else 'SATURATED'
                          if upper < SATURATION['upper_bound_below'] else 'INCONCLUSIVE'}
        lower = s['improvement_ci95'][0]
        rows[f'S{BASE["simulations"] * k}'] = {
            'factor': k, 'score': s['score'], 'score_ci95': s['score_ci95'], 'improvement_ci95': s['improvement_ci95'],
            'elo': s['elo'], 'score_by_color': s['score_by_color'], 'stronger': lower > 0, 'cost': s['cost'],
            'tree': s['tree'], 'opponent_illegal_pairs': s['opponent_illegal_pairs']}
    stronger = [r['factor'] for r in rows.values() if r['stronger']]
    out = {'format': f'{FORMAT}-summary', 'seed': SEED, 'pairs': PAIRS, 'rule': 'STRONGER: the 95% pair-bootstrap '
           'lower bound of (score - 0.5) against S50 is above 0', 'arms': rows,
           'reading': 'BUDGET_HELPS' if stronger else 'NOT_DETECTED',
           'smallest_stronger_budget': budget(min(stronger)) if stronger else None, 'arm_sha256': arm_sha,
           'saturation': saturation}
    if stronger and saturation is not None:
        raise ValueError('the saturation stage runs only when no budget is STRONGER at 50 pairs')
    if args.positions:
        record = json.loads(Path(args.positions).read_text(encoding='utf-8'))
        out['positions'] = {**record['summary'], 'sha256': ar._file_sha(Path(args.positions))}
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('play', 'positions'):
        p = sub.add_parser(mode)
        p.add_argument('--policy-checkpoint', type=Path, required=True)
        p.add_argument('--value-checkpoint', type=Path, required=True)
        p.add_argument('--device', default='cuda')
        p.add_argument('--workers', type=int, default=1)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--allow-dirty', action='store_true', help='tests only')
        p.add_argument('--allow-stand-in', action='store_true', help='tests only: models other than H3 / H6 v3')
    sub.choices['play'].add_argument('--factor', type=int, required=True, choices=FACTORS)
    sub.choices['play'].add_argument('--jsonl', type=Path, required=True)
    sub.choices['play'].add_argument('--pairs', type=int, default=PAIRS,
                                     help=f'{PAIRS}; {SATURATION["pairs"]} extends factor 8 for the saturation stage '
                                          '(same --jsonl: the first 50 pairs are reused)')
    sub.choices['positions'].add_argument('--policy-rows', type=Path, required=True)
    sub.choices['positions'].add_argument('--evidence', type=Path, required=True)
    sub.choices['positions'].add_argument('--limit', type=int, help='first N positions only (tests)')
    s = sub.add_parser('summary')
    s.add_argument('--arms', nargs='+', required=True)
    s.add_argument('--positions', type=Path)
    s.add_argument('--output', type=Path, required=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    try:
        payload = {'play': play, 'positions': positions, 'summary': summary}[args.mode](args)
    except ValueError as exc:
        parser.error(str(exc))
    brief = payload['summary'] if args.mode != 'summary' else {k: v for k, v in payload.items() if k != 'positions'}
    print(json.dumps(brief, indent=1, default=str)[:20000], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
