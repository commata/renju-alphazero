"""H7-c decision sensitivity: how often two model sets choose different moves on the same positions (measurement only).

Positions are a cycle's self-play games (``h7c_selfplay`` JSONL, its SHA-256 named by the cycle's evidence manifest),
a fixed sample of ``--games-sample`` games (sorted by SHA-256 of ``SALT`` + game id, so the sample is fixed before any
search). Each sampled game is replayed ply by ply: at every ply each model set's agent (TB-V3 / fast5, the candidate
factory, the actor's seeds ``derive_seed(game_seed, 'black' | 'white')``, one agent per colour per game) searches the
actor's position, then the actor's played move is played. The model set equal to the self-play's own models is the
actor check: it must reproduce every recorded engine move and route (``actor_check``).

Per comparison ``A:B`` (docs/h7-c1.md, B2 review): on all plies and on plies where both searched a tree, the share
of plies with a different move (direct counts, no model of independence), the share with a different route, the
Jensen-Shannon divergence of the root visit distributions and the change of the chosen child's mean value (tree
plies), the share of games with any different move and the first such ply, and by side to move. 95% intervals are a
cluster bootstrap over games (``BOOTSTRAP``). The result describes these two model sets on this position sample: it
is a diagnosis of their behaviour, not a bound on what another value model, target or recipe could change.

    python -u scripts/h7c_sensitivity.py --games runs/h7c/c8/selfplay.jsonl \\
        --evidence docs/h7c/c8/evidence_manifest.json \\
        --model orig=runs/h3_policy_64x4/best.pt,runs/h6_value_head_v3/best.pt \\
        --model c5=runs/h3_policy_64x4/best.pt,runs/h7c/c5/value/best.pt \\
        --model c8=runs/h3_policy_64x4/best.pt,runs/h7c/c8/value/best.pt \\
        --compare c5:orig --compare c8:c5 --device cuda --workers 6 \\
        --jsonl runs/h7c/sensitivity/c8_sample.jsonl --output docs/h7c/b2/sensitivity_c8.json

Resume: rerun the same command (finished games in ``--jsonl`` are kept; another identity is refused). A dirty tree
is refused (``--allow-dirty`` only for tests).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
from random import Random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import BLACK, Game  # noqa: E402

FORMAT = 'h7c-sensitivity-v1'
SALT = 'h7c-sensitivity-v1'
GAMES_SAMPLE = 200
BOOTSTRAP = {'samples': 2000, 'seed': 20261027}


def _sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sample_games(games: list[dict], n: int) -> list[dict]:
    """The first ``n`` games by SHA-256 of ``SALT`` + game id (fixed before any search)."""
    import h7c_labels as lab

    key = lambda g: hashlib.sha256(f'{SALT}:{lab.game_id(g)}'.encode()).hexdigest()  # noqa: E731
    return sorted(games, key=key)[:n]


def parse_models(items: list[str]) -> dict[str, dict]:
    out = {}
    for item in items:
        name, _, paths = item.partition('=')
        policy, _, value = paths.partition(',')
        if not name or not policy or not value or name in out:
            raise ValueError(f'--model {item!r}: NAME=POLICY,VALUE with a new name')
        out[name] = {'policy': policy, 'value': value}
    return out


def parse_compares(items: list[str], models: dict) -> list[tuple[str, str]]:
    out = []
    for item in items:
        a, _, b = item.partition(':')
        if a not in models or b not in models or a == b:
            raise ValueError(f'--compare {item!r}: two different --model names')
        out.append((a, b))
    return out


# --- the replay --------------------------------------------------------------------------------------------------

def replay_game(game: dict, models: dict, device: str, make=None, search=None) -> dict:
    """Every ply of one self-play game searched by every model set (the actor's position, the actor's seeds)."""
    from run_mcts_v8_benchmark import derive_seed

    if make is None:
        import h7c_arena as ar
        make = lambda side, seed: ar.make_side(side, seed, device)  # noqa: E731
    if search is None:
        import h7c_retarget as rt
        search = rt.search
    seeds = {'black': derive_seed(game['game_seed'], 'black'), 'white': derive_seed(game['game_seed'], 'white')}
    agents = {name: {color: make(side, seed) for color, seed in seeds.items()} for name, side in models.items()}
    position = Game()
    for move in game['opening']:
        position.play(*move)
    plies = []
    for p in game['plies']:
        color = 'black' if position.to_play == BLACK else 'white'
        if p['color'] != color:
            raise ValueError(f"game {game['game_index']} ply {p['ply']}: not the recorded mover")
        sets = {}
        for name in models:
            record = search(agents[name][color], position)
            sets[name] = {k: record[k] for k in ('route', 'engine_move', 'visits')}
        plies.append({'ply': p['ply'], 'color': color, 'actor_route': p['route'], 'actor_move': p['engine_move'],
                      'sets': sets})
        position.play(*p['played'])
    return {'game_index': game['game_index'], 'plies': plies}


def _chunk(tasks: list[dict]) -> list[dict]:
    return [replay_game(t['game'], t['models'], t['device']) for t in tasks]


# --- measures ----------------------------------------------------------------------------------------------------

def _dist(visits) -> dict:
    total = sum(v[2] for v in visits)
    return {(int(v[0]), int(v[1])): v[2] / total for v in visits if v[2] > 0} if total > 0 else {}


def js(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    m = {k: (p.get(k, 0.0) + q.get(k, 0.0)) / 2 for k in keys}
    kl = lambda a: sum(a[k] * math.log(a[k] / m[k]) for k in a if a[k] > 0)  # noqa: E731
    return 0.5 * kl(p) + 0.5 * kl(q)


def _chosen_q(record) -> float | None:
    move = list(record['engine_move'])
    return next((v[3] for v in record['visits'] if [int(v[0]), int(v[1])] == move), None)


def ply_measures(ply: dict, a: str, b: str) -> dict:
    ra, rb = ply['sets'][a], ply['sets'][b]
    tree = ra['route'] == 'tree' and rb['route'] == 'tree' and ra['visits'] and rb['visits']
    out = {'move_diff': list(ra['engine_move']) != list(rb['engine_move']), 'route_diff': ra['route'] != rb['route'],
           'tree': bool(tree), 'color': ply['color'], 'ply': ply['ply']}
    if tree:
        qa, qb = _chosen_q(ra), _chosen_q(rb)
        out['js'] = js(_dist(ra['visits']), _dist(rb['visits']))
        out['dq'] = None if qa is None or qb is None else abs(qa - qb)
    return out


def _rate(groups, pick) -> tuple[int, int]:
    rows = [r for g in groups for r in g if pick(r)]
    return sum(r['move_diff'] for r in rows), len(rows)


def _bootstrap(per_game: list[list[dict]], stat) -> list[float] | None:
    rng = Random(BOOTSTRAP['seed'])
    values = []
    for _ in range(BOOTSTRAP['samples']):
        draw = [per_game[rng.randrange(len(per_game))] for _ in per_game]
        v = stat(draw)
        if v is not None:
            values.append(v)
    if not values:
        return None
    values.sort()
    return [round(values[int(0.025 * (len(values) - 1))], 5), round(values[int(0.975 * (len(values) - 1))], 5)]


def compare(records: list[dict], a: str, b: str) -> dict:
    per_game = [[ply_measures(p, a, b) for p in r['plies']] for r in records]

    def share(pick):
        def stat(groups):
            diff, n = _rate(groups, pick)
            return diff / n if n else None
        diff, n = _rate(per_game, pick)
        return {'plies': n, 'different_moves': diff, 'share': round(diff / n, 5) if n else None,
                'ci95_games': _bootstrap(per_game, stat)}

    tree = [m for g in per_game for m in g if m['tree']]
    any_diff = [next((m['ply'] for m in g if m['move_diff']), None) for g in per_game]
    games_diff = [x for x in any_diff if x is not None]
    out = {'pair': f'{a}:{b}', 'games': len(per_game),
           'all_plies': share(lambda m: True), 'tree_plies': share(lambda m: m['tree']),
           'route_different_share': round(sum(m['route_diff'] for g in per_game for m in g)
                                          / max(1, sum(len(g) for g in per_game)), 5),
           'tree_js_mean': round(statistics.fmean(m['js'] for m in tree), 5) if tree else None,
           'tree_chosen_q_abs_diff_mean': round(statistics.fmean(m['dq'] for m in tree if m['dq'] is not None), 5)
           if any(m['dq'] is not None for m in tree) else None,
           'games_with_a_different_move': {'games': len(games_diff), 'share': round(len(games_diff) / len(per_game), 5),
                                           'first_ply_median': statistics.median(games_diff) if games_diff else None},
           'tree_plies_by_side': {c: share(lambda m, c=c: m['tree'] and m['color'] == c) for c in ('black', 'white')}}
    return out


def actor_check(records: list[dict], games: dict, actor: str) -> dict:
    moves = routes = n = 0
    for r in records:
        recorded = {p['ply']: p for p in games[r['game_index']]['plies']}
        for p in r['plies']:
            n += 1
            mine = p['sets'][actor]
            moves += list(mine['engine_move']) == list(recorded[p['ply']]['engine_move'])
            routes += mine['route'] == recorded[p['ply']]['route']
    return {'model_set': actor, 'plies': n, 'same_move': moves, 'same_route': routes,
            'reproduced': moves == n and routes == n}


# --- the run -----------------------------------------------------------------------------------------------------

def run(args, log=print) -> dict:
    import h7c_labels as lab
    import h7c_selfplay as sp
    from analysis import tb_v3, tb_v3_fast5, tb_v3_net
    from h7c_arena import _git
    from hybrid import h7c_retarget as hr

    git = _git()
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty (or git is unavailable): a measurement must name its code commit')
    if not lab._under_data_or_runs(Path(args.jsonl)):
        raise ValueError(f'{args.jsonl}: the replay records go under data/ or runs/')
    models = parse_models(args.model)
    compares = parse_compares(args.compare, models)
    evidence = json.loads(Path(args.evidence).read_text(encoding='utf-8'))
    games_sha = _sha(args.games)
    named = [v for k, v in evidence.get('files_sha256', {}).items() if k.endswith('.jsonl') and 'selfplay' in k]
    if games_sha not in named:
        raise ValueError(f'{args.games} is not a self-play file the evidence manifest names')
    games = lab.read_games([args.games])
    lab.check_games(games)
    selfplay = hr.selfplay_identity(games)
    if selfplay['engine_hash'] != tb_v3.ENGINE_HASH:
        raise ValueError('the self-play engine is not the frozen TB-V3')
    if args.device.split(':')[0] != selfplay['value_device'] or tb_v3_fast5.runtime_hash(args.device) != \
            selfplay['runtime_hash']:
        raise ValueError(f"--device {args.device}: the self-play ran on {selfplay['value_device']}")
    identities = {name: tb_v3_net.model_identity(m['policy'], m['value']) for name, m in models.items()}
    actors = [name for name, ident in identities.items() if ident == selfplay['models']]
    if len(actors) != 1:
        raise ValueError('exactly one --model must be the self-play actor (the reproduction check)')
    sample = sample_games(games, args.games_sample)
    ids = [lab.game_id(g) for g in sample]
    identity = {'format': FORMAT, 'salt': SALT, 'selfplay_jsonl_sha256': games_sha,
                'evidence_sha256': _sha(args.evidence), 'selfplay': selfplay, 'games_sample': len(sample),
                'sample_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(), 'models': identities,
                'engine_hash': tb_v3.ENGINE_HASH, 'runtime_hash': selfplay['runtime_hash'], 'device': args.device,
                'git': git}
    finished = sp.load_finished(Path(args.jsonl), {'identity': identity}, log=log)
    todo = [g for g in sample if g['game_index'] not in finished]
    plies = sum(len(g['plies']) for g in todo)
    log(f'[h7c-sensitivity] {len(sample)} games ({len(finished)} done), {plies:,} plies x {len(models)} model sets, '
        f'workers {args.workers}', flush=True)
    if todo:
        Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.jsonl).open('a', encoding='utf-8') as out, ProcessPoolExecutor(
                max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            def keep(_, results):
                for result in results:
                    record = {'identity': identity, **result}
                    out.write(json.dumps(record, separators=(',', ':')) + '\n')
                    finished[result['game_index']] = record
                out.flush()
            run_chunks(pool, _chunk, [[{'game': g, 'models': models, 'device': args.device}] for g in todo],
                       Progress('h7c-sensitivity', plies, 'plies'), lambda c: len(c[0]['game']['plies']),
                       unit_name='games', on_result=keep)
    records = [finished[g['game_index']] for g in sample]
    by_index = {g['game_index']: g for g in sample}
    result = {'format': f'{FORMAT}-summary', 'identity': identity, 'jsonl_sha256': _sha(args.jsonl),
              'bootstrap': BOOTSTRAP, 'actor_check': actor_check(records, by_index, actors[0]),
              'comparisons': [compare(records, a, b) for a, b in compares],
              'reading': 'the behaviour of these model sets on this position sample (not a bound for other models)'}
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--games', type=Path, required=True, help='a cycle self-play JSONL')
    parser.add_argument('--evidence', type=Path, required=True, help="the cycle's evidence manifest")
    parser.add_argument('--model', action='append', required=True, help='NAME=POLICY,VALUE (repeat)')
    parser.add_argument('--compare', action='append', required=True, help='A:B (repeat)')
    parser.add_argument('--games-sample', type=int, default=GAMES_SAMPLE)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--jsonl', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-dirty', action='store_true', help='tests only')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    try:
        out = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({k: out[k] for k in ('actor_check', 'comparisons')}, indent=1), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
