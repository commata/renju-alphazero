"""H7-c cycle diagnosis, measurement only (docs/h7-c1.md, D1-D5): why a candidate played the way it did.

Nothing here trains, changes a rule, a verdict or a checkpoint. Every input is checked first and the run fails
closed on a mismatch: the cycle's Arena verdict names the candidate and the parent (their four SHA-256 each), the
evidence manifest names the label files and the P-only arm output by SHA-256, and the P-only output's games carry
its run signature. Inference is the engine's own: ``hybrid.h4_policy.RootPolicy`` (masked softmax over the legal
moves) and ``hybrid.h6_value_leaf.ValueLeaf`` (D4 average).

    D1  policy shift on the P-only games' tree-route positions of the candidate side: candidate policy vs the
        parent policy on the tree's action set (``_search_candidates_v321``, limit 20, radius 2: the inner-node
        set; the root list is V8's own) and on all legal moves; entropy, top-1 agreement, KL(parent || candidate),
        Jensen-Shannon, and the probability on threat-making moves by the engine's own features (five:
        ``_winning_moves``; four: ``_four_completions``; open four: ``_creates_open_four``; window-3: a move in a
        five-cell window with >= 2 own stones and none of the opponent's, ``_window_candidates(minimum=2)``),
        by side to move
    D2  policy validation split: the unmasked val rows, parent and candidate, by kind (visits / proof_one_hot) x
        side to move: N, cross-entropy against the target, top-1 against the target's mode
    D3  first divergence in the P-only losses: the candidate side's moves replayed with the recorded agent seed
        (a check that the record reproduces) next to the parent agent with the same seed on the same positions;
        at the first move where they differ: both routes, moves, root visits / Q, prior top and the plies left
    D4  value split on the unmasked val rows by label kind x side to move (parent and candidate: N, weight, weighted
        and plain MSE, sign accuracy, mean prediction and target); every unmasked row classified by
        ``analysis.puct_value_v1.exact_leaf`` (settled without the network in the search) with row and weight
        shares; the fresh-probe gate and P93 / P94 (``h7c_arena.value_probes``) for both
    D5  visit targets of the policy rows (visits rows only; proof one-hot rows counted apart), by side to move:
        entropy, exp(entropy), top-1 share, visited children

    python -u scripts/h7c_diagnose.py --cycle 1 --verdict docs/h7c/c1/arena_verdict.json \\
        --evidence docs/h7c/c1/evidence_manifest.json --p-only runs/h7c/c1/arena/p_only.json \\
        --candidate-policy runs/h7c/c1/policy/best.pt --candidate-value runs/h7c/c1/value/best.pt \\
        --parent-policy runs/h3_policy_64x4/best.pt --parent-value runs/h6_value_head_v3/best.pt \\
        --policy-rows runs/h7c/c1/policy_rows.jsonl.gz --value-rows runs/h7c/c1/value_rows.jsonl.gz \\
        --device cuda --workers 6 --output docs/h7c/c1/diagnosis.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
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
from renju import BLACK, Game  # noqa: E402

FORMAT = 'h7c-diagnosis-v1'
PARTS = ('d1', 'd2', 'd3', 'd4', 'd5')
TREE_SET = {'limit': 20, 'radius': 2}  # TB-V2-Value candidate_limit / neighborhood_radius
THREATS = ('five', 'four', 'open_four', 'window3')
_WORKER: dict = {}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def side(ply: int) -> str:
    return 'black' if ply % 2 == 0 else 'white'


def position(moves) -> Game:
    game = Game()
    for move in moves:
        game.play(*move)
    return game


# --- inputs ------------------------------------------------------------------------------------------------------

def check_inputs(args) -> dict:
    """Fail closed unless the files are the ones the cycle's records name; returns their identities."""
    from analysis import tb_v3_net
    from h7c_arena import FORMAT as ARENA_FORMAT, signature

    verdict, evidence, p_only = _json(args.verdict), _json(args.evidence), _json(args.p_only)
    candidate = tb_v3_net.model_identity(args.candidate_policy, args.candidate_value)
    parent = tb_v3_net.model_identity(args.parent_policy, args.parent_value)
    if verdict.get('format') != f'{ARENA_FORMAT}-verdict' or verdict.get('cycle') != args.cycle:
        raise ValueError(f'{args.verdict} is not the Arena verdict of cycle {args.cycle}')
    if verdict['candidate'] != candidate or verdict['parent'] != parent:
        raise ValueError('the model files are not the candidate / parent of the verdict')
    files = evidence.get('files_sha256', {})
    for name, path in (('policy_rows.jsonl.gz', args.policy_rows), ('value_rows.jsonl.gz', args.value_rows)):
        if files.get(name) != _sha(path):
            raise ValueError(f'{path} is not the {name} of the evidence manifest')
    p_only_sha = _sha(args.p_only)
    if p_only_sha != verdict['arm_sha256'].get('p_only') or \
            p_only_sha != evidence.get('arena', {}).get('arm_sha256', {}).get('p_only'):
        raise ValueError(f'{args.p_only} is not the P-only output the verdict and the evidence name')
    identity = p_only['identity']
    if identity['arm'] != 'p_only' or identity['cycle'] != args.cycle or signature(identity) != p_only['signature'] \
            or any(g['signature'] != p_only['signature'] for g in p_only['games']):
        raise ValueError(f'{args.p_only}: not a P-only run of cycle {args.cycle} with intact games')
    tested = identity['tested']['models']
    if {k: tested[k] for k in ('policy_checkpoint_sha256', 'policy_metadata_sha256')} != \
            {k: candidate[k] for k in ('policy_checkpoint_sha256', 'policy_metadata_sha256')} or \
            {k: tested[k] for k in ('value_checkpoint_sha256', 'value_metadata_sha256')} != \
            {k: parent[k] for k in ('value_checkpoint_sha256', 'value_metadata_sha256')}:
        raise ValueError('the P-only games were not played by candidate policy + parent value')
    return {'candidate': candidate, 'parent': parent,
            'sha256': {'verdict': _sha(args.verdict), 'evidence': _sha(args.evidence), 'p_only': p_only_sha,
                       'policy_rows': files['policy_rows.jsonl.gz'], 'value_rows': files['value_rows.jsonl.gz']},
            'p_only_signature': p_only['signature']}


# --- distributions -----------------------------------------------------------------------------------------------

def restricted(probs: dict, moves) -> dict:
    """``probs`` on ``moves`` renormalized (uniform if it puts no mass there, as the PUCT prior does)."""
    weights = {m: max(0.0, float(probs.get(m, 0.0))) for m in moves}
    total = sum(weights.values())
    return {m: (w / total if total > 0 else 1.0 / len(moves)) for m, w in weights.items()}


def entropy(dist: dict) -> float:
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


def kl(p: dict, q: dict, eps: float = 1e-12) -> float:
    return sum(pp * math.log(pp / max(q.get(m, 0.0), eps)) for m, pp in p.items() if pp > 0)


def js(p: dict, q: dict) -> float:
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in set(p) | set(q)}
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def top(dist: dict):
    return max(dist, key=lambda m: (dist[m], -m[0], -m[1]))


def threat_classes(game: Game, moves) -> dict[str, set]:
    """The engine's threat features for the side to move: which of ``moves`` make each class."""
    from search.mcts_v4 import _creates_open_four
    from search.mcts_v5 import _four_completions, _window_candidates, _winning_moves

    me, legal = game.to_play, set(moves)
    window3 = set(_window_candidates(game, me, 2))
    return {'five': set(_winning_moves(game, me, list(moves), legal)),
            'four': {m for m in moves if _four_completions(game, me, m)},
            'open_four': {m for m in moves if _creates_open_four(game, me, m, assume_legal=True)},
            'window3': {m for m in moves if m in window3}}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 5) if xs else None


def compare_policies(parent: dict, candidate: dict, classes: dict) -> dict:
    return {'parent_entropy': entropy(parent), 'candidate_entropy': entropy(candidate),
            'top1_agree': top(parent) == top(candidate), 'kl_parent_candidate': kl(parent, candidate),
            'js': js(parent, candidate),
            'threat_mass': {c: {'parent': sum(parent[m] for m in ms), 'candidate': sum(candidate[m] for m in ms)}
                            for c, ms in classes.items()}}


def aggregate(rows: list[dict]) -> dict:
    out = {'n': len(rows)}
    if not rows:
        return out
    for key in ('parent_entropy', 'candidate_entropy', 'kl_parent_candidate', 'js'):
        out[key] = _mean([r[key] for r in rows])
    out['top1_agree'] = _mean([float(r['top1_agree']) for r in rows])
    out['threat_mass'] = {c: {who: _mean([r['threat_mass'][c][who] for r in rows]) for who in ('parent', 'candidate')}
                          for c in THREATS}
    return out


# --- D1 / D5 / D2: policy ----------------------------------------------------------------------------------------

def d1(games: list[dict], parent_policy, candidate_policy, log=print) -> dict:
    from search.mcts_v321 import _search_candidates_v321

    rows = []
    for g in games:
        for m in g['tested_moves']:
            if m['route'] != 'tree':
                continue
            game = position(g['moves'][:m['ply']])
            legal = game.legal_moves()
            tree_set = _search_candidates_v321(game, TREE_SET['limit'], TREE_SET['radius'])
            pp, cp = parent_policy(game), candidate_policy(game)
            on_tree = compare_policies(restricted(pp, tree_set), restricted(cp, tree_set), threat_classes(game, tree_set))
            on_legal = compare_policies(restricted(pp, legal), restricted(cp, legal), threat_classes(game, legal))
            rows.append({'key': g['key'], 'ply': m['ply'], 'side': side(m['ply']), 'tree': on_tree,
                         'legal': on_legal, 'tree_set_size': len(tree_set)})
    log(f'[h7c-diagnose] D1: {len(rows)} tree-route positions', flush=True)
    by = {}
    for support in ('tree', 'legal'):
        by[support] = {s: aggregate([r[support] for r in rows if s in (r['side'], 'all')])
                       for s in ('all', 'black', 'white')}
    return {'positions': len(rows), 'tree_set': TREE_SET, 'threat_classes': THREATS, 'summary': by,
            'rows': [{k: r[k] for k in ('key', 'ply', 'side', 'tree_set_size')} | {
                'tree': {k: v for k, v in r['tree'].items() if k != 'threat_mass'}} for r in rows]}


def read_policy_rows(path) -> list[dict]:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def d5(rows: list[dict]) -> dict:
    stats = {'black': [], 'white': []}
    proof = {'black': 0, 'white': 0}
    for r in rows:
        if r['mask']:
            continue
        s = side(len(r['moves']))
        if r['kind'] == 'proof_one_hot':
            proof[s] += 1
        elif r['kind'] == 'visits':
            ps = [p for _, _, p in r['target'] if p > 0]
            h = -sum(p * math.log(p) for p in ps)
            stats[s].append({'entropy': h, 'support': math.exp(h), 'top_share': max(ps), 'children': len(ps)})
    out = {}
    for s, xs in stats.items():
        out[s] = {'visits_rows': len(xs), 'proof_one_hot_rows': proof[s],
                  **{k: _mean([x[k] for x in xs]) for k in ('entropy', 'support', 'top_share', 'children')}}
    return out


def d2(rows: list[dict], parent_policy, candidate_policy) -> dict:
    strata: dict = {}
    for r in rows:
        if r['mask'] or r['split'] != 'val' or r['kind'] not in ('visits', 'proof_one_hot'):
            continue
        game = position(r['moves'])
        target = {(a, b): p for a, b, p in r['target']}
        mode = top(target)
        cell = strata.setdefault((r['kind'], side(len(r['moves']))), {'n': 0, 'parent': [0.0, 0], 'candidate': [0.0, 0]})
        cell['n'] += 1
        for who, policy in (('parent', parent_policy), ('candidate', candidate_policy)):
            probs = policy(game)
            cell[who][0] += -sum(p * math.log(max(probs.get(m, 0.0), 1e-12)) for m, p in target.items())
            cell[who][1] += int(top(probs) == mode)
    out = {}
    for (kind, s), cell in sorted(strata.items()):
        n = cell['n']
        out[f'{kind}/{s}'] = {'n': n, **{who: {'ce': round(cell[who][0] / n, 5), 'top1': round(cell[who][1] / n, 5)}
                                        for who in ('parent', 'candidate')},
                              'delta_ce': round((cell['candidate'][0] - cell['parent'][0]) / n, 5),
                              'delta_top1': round((cell['candidate'][1] - cell['parent'][1]) / n, 5)}
    return out


# --- D3: first divergence ----------------------------------------------------------------------------------------

def _init_worker(paths: dict, device: str) -> None:
    _WORKER.update(paths=paths, device=device)


def _agent(policy, value, seed):
    from h7c_arena import make_side

    return make_side({'policy': policy, 'value': value}, seed, _WORKER['device'])


def _decision(agent, game: Game) -> dict:
    from hybrid.h7c_targets import raw_visits
    from run_mcts_v8_benchmark import _move_record

    started = perf_counter()
    move = agent.select_move(game)
    record = _move_record(len(game.history), perf_counter() - started, agent.diagnostics)
    return {'move': [int(move[0]), int(move[1])], 'route': record['route'], 'visits': raw_visits(agent.diagnostics),
            'prior_top': record['tree']['prior_top'], 'prior_top_prob': record['tree']['prior_top_prob'],
            'root_q': record['tree']['root_q']}


def divergence(game_record: dict) -> dict:
    """Replay the candidate side of one lost game with the candidate agent and the parent agent (same seed)."""
    from h7c_arena import OPENING
    from run_mcts_v8_benchmark import derive_seed

    paths = _WORKER['paths']
    seed = derive_seed(game_record['seed'], 'tested')
    candidate = _agent(paths['candidate_policy'], paths['parent_value'], seed)
    parent = _agent(paths['parent_policy'], paths['parent_value'], seed)
    moves, color = game_record['moves'], BLACK if game_record['tested_color'] == 'black' else -BLACK
    opening, game = OPENING['random_plies'] + 1, Game()  # the centre and the random plies: no agent moved
    out = {'key': game_record['key'], 'length': game_record['length'], 'result': game_record['result']}
    for ply, move in enumerate(moves):
        if game.to_play == color and ply >= opening:
            c, p = _decision(candidate, game), _decision(parent, game)
            if c['move'] != list(move):
                return {**out, 'status': 'NOT_REPRODUCED', 'ply': ply, 'recorded': list(move), 'candidate': c}
            if p['move'] != c['move']:
                return {**out, 'status': 'DIVERGED', 'ply': ply, 'side': side(ply), 'plies_left': len(moves) - ply,
                        'candidate': c, 'parent': p}
        game.play(*move)
    return {**out, 'status': 'NO_DIVERGENCE'}


def _divergence_chunk(games: list[dict]) -> list[dict]:
    return [divergence(g) for g in games]


def d3(games: list[dict], paths: dict, device: str, workers: int, log=print) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    losses = [g for g in games if g['result'] == 'loss']
    rows = []
    progress = Progress('h7c-diagnose-d3', len(losses), 'games')
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=_init_worker,
                             initargs=(paths, device)) as pool:
        run_chunks(pool, _divergence_chunk, [[g] for g in losses], progress, len, unit_name='games',
                   on_result=lambda _, part: rows.extend(part))
    rows.sort(key=lambda r: r['key'])
    diverged = [r for r in rows if r['status'] == 'DIVERGED']
    routes = {}
    for r in diverged:
        pair = f"{r['candidate']['route']} / {r['parent']['route']}"
        routes[pair] = routes.get(pair, 0) + 1
    return {'losses': len(losses), 'status': {s: sum(r['status'] == s for r in rows)
                                              for s in ('DIVERGED', 'NO_DIVERGENCE', 'NOT_REPRODUCED')},
            'routes_candidate_parent': routes, 'by_side': {s: sum(r['side'] == s for r in diverged)
                                                           for s in ('black', 'white')},
            'plies_left': {'mean': _mean([r['plies_left'] for r in diverged]),
                           'values': sorted(r['plies_left'] for r in diverged)},
            'rows': rows}


# --- D4: value ---------------------------------------------------------------------------------------------------

def d4(value_rows_path, parent_value, candidate_value, args, log=print) -> dict:
    from analysis.puct_value_v1 import exact_leaf
    from h7c_arena import value_probes
    from hybrid.h6_value_cache import iter_games

    bypass, cells = {}, {}
    for _, rows in iter_games(Path(value_rows_path)):
        game = Game()
        for r in sorted(rows, key=lambda r: r['ply']):
            while len(game.history) < r['ply']:  # one replay per game: the rows are its prefixes
                game.play(*r['moves'][len(game.history)])
            if r['value_mask']:
                continue
            route = 'exact_leaf' if exact_leaf(game) is not None else 'network'
            b = bypass.setdefault(r['label_kind'], {'exact_leaf': [0, 0.0], 'network': [0, 0.0]})
            b[route][0] += 1
            b[route][1] += r['weight']
            if r['split'] != 'val':
                continue
            s = side(r['ply'])
            cell = cells.setdefault((r['label_kind'], s), [])
            cell.append((r['value_target'], r['weight'], parent_value(game), candidate_value(game), route))
    total_w = sum(v[1] for b in bypass.values() for v in b.values())
    total_n = sum(v[0] for b in bypass.values() for v in b.values())
    exact = {'by_kind': {k: {route: {'rows': v[0], 'weight': round(v[1], 3)} for route, v in b.items()}
                         for k, b in sorted(bypass.items())},
             'exact_leaf_rows_share': round(sum(b['exact_leaf'][0] for b in bypass.values()) / total_n, 5),
             'exact_leaf_weight_share': round(sum(b['exact_leaf'][1] for b in bypass.values()) / total_w, 5)}

    def stats(items, who):
        idx = 2 if who == 'parent' else 3
        w = sum(i[1] for i in items)
        nonzero = [i for i in items if i[0] != 0]
        return {'wmse': round(sum(i[1] * (i[idx] - i[0]) ** 2 for i in items) / w, 5) if w else None,
                'mse': round(sum((i[idx] - i[0]) ** 2 for i in items) / len(items), 5),
                'sign_acc': round(sum((i[idx] > 0) == (i[0] > 0) for i in nonzero) / len(nonzero), 5) if nonzero else None,
                'mean_prediction': round(sum(i[idx] for i in items) / len(items), 5)}
    strata = {}
    for (kind, s), items in sorted(cells.items()):
        strata[f'{kind}/{s}'] = {'n': len(items), 'weight': round(sum(i[1] for i in items), 3),
                                 'exact_leaf_rows': sum(i[4] == 'exact_leaf' for i in items),
                                 'mean_target': round(sum(i[0] for i in items) / len(items), 5),
                                 'parent': stats(items, 'parent'), 'candidate': stats(items, 'candidate')}
    log('[h7c-diagnose] D4: probes', flush=True)
    probes = {'parent': value_probes(args.parent_value, args.device),
              'candidate': value_probes(args.candidate_value, args.device)}
    for p in probes.values():
        p['fresh'].pop('rows', None)
    return {'note': 'val rows of the label step (the trainer additionally dropped 49 union-masked rows)',
            'exact_leaf': exact, 'val_strata': strata, 'probes': probes}


# --- main --------------------------------------------------------------------------------------------------------

def run(args, log=print) -> dict:
    from h7c_arena import _git
    from hybrid.h4_policy import RootPolicy
    from hybrid.h6_value_leaf import ValueLeaf

    git = _git()
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty (or git is unavailable): a diagnosis must name its code commit')
    inputs = check_inputs(args)
    games = _json(args.p_only)['games']
    out = {'format': FORMAT, 'cycle': args.cycle, 'git': git, 'device': args.device, 'inputs': inputs, 'parts': {}}
    parts = args.parts or PARTS
    policies = {}
    if {'d1', 'd2'} & set(parts):
        policies = {'parent': RootPolicy(args.parent_policy, device=args.device),
                    'candidate': RootPolicy(args.candidate_policy, device=args.device)}
    policy_rows = read_policy_rows(args.policy_rows) if {'d2', 'd5'} & set(parts) else []
    for part in parts:
        started = perf_counter()
        log(f'[h7c-diagnose] {part} ...', flush=True)
        if part == 'd1':
            result = d1(games, policies['parent'], policies['candidate'], log=log)
        elif part == 'd2':
            result = d2(policy_rows, policies['parent'], policies['candidate'])
        elif part == 'd3':
            paths = {k: str(getattr(args, k)) for k in ('candidate_policy', 'parent_policy', 'parent_value')}
            result = d3(games, paths, args.device, args.workers, log=log)
        elif part == 'd4':
            values = (ValueLeaf(args.parent_value, device=args.device, expected_sha256=None),
                      ValueLeaf(args.candidate_value, device=args.device, expected_sha256=None))
            result = d4(args.value_rows, *values, args, log=log)
        else:
            result = d5(policy_rows)
        out['parts'][part] = {**result, 'seconds': round(perf_counter() - started, 1)}
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cycle', type=int, required=True)
    parser.add_argument('--verdict', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--p-only', type=Path, required=True, help="the P-only arm's play output (games included)")
    for name in ('candidate-policy', 'candidate-value', 'parent-policy', 'parent-value', 'policy-rows', 'value-rows'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--parts', nargs='+', choices=PARTS)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=1, help='D3 replay processes')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-dirty', action='store_true', help='tests only')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    try:
        payload = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    brief = {part: {k: v for k, v in result.items() if k not in ('rows',)} for part, result in payload['parts'].items()}
    print(json.dumps(brief, indent=1, default=str)[:20000], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
