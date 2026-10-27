"""H7-c cycle diagnosis, measurement only (docs/h7-c1.md, D1-D5): why a candidate played the way it did.

Nothing here trains, changes a rule, a verdict or a checkpoint. Every input is checked first and the run fails
closed on a mismatch: the cycle's Arena verdict names the candidate and the parent (four SHA-256 each); the evidence
manifest names the label files, the cycle value labels and the P-only arm output; the value training record names the
candidate, the parent and both caches; the P-only games carry their run signature, and ``--device`` must be the
P-only run's value device and runtime. Engine inference is the engine's own: ``hybrid.h4_policy.RootPolicy`` (masked
softmax over the legal moves) and ``hybrid.h6_value_leaf.ValueLeaf`` (D4 average).

    D1  policy shift where the engine used the prior: every P-only game is replayed with the candidate-side agent
        (recorded seed; each move must reproduce the record) and, at its tree-route moves, the parent and candidate
        policies are compared on the root's actual children (``diagnostics.v8_root_visits``, the set the PUCT prior
        is renormalized over) and on all legal moves: entropy, top-1 agreement, KL(parent || candidate),
        Jensen-Shannon, and the mass on threat-making moves by the engine's own features (five: ``_winning_moves``;
        four: ``_four_completions``; open four: ``_creates_open_four``; window-3: a move in a five-cell window with
        >= 2 own stones and none of the opponent's, ``_window_candidates(minimum=2)``), by side to move. The
        candidate's root-prior entropy is checked against the engine's own (``v8_puct_prior_entropy``, normalized)
    D2  policy validation split as the trainer scores it: the unmasked val rows, parent and candidate, by kind
        (visits / proof_one_hot) x side to move: N, cross-entropy against the target, top-1 (the predicted move's
        target equals the target's maximum: tied maxima all count)
    D3  first divergence in the P-only losses, in the same replay: the parent agent (same seed) on the same
        positions until its move differs; both routes, moves, root visits / Q, prior top and the plies left
    D4  value, two metrics on one position set, the trainer's effective cycle val (its union masking re-run):
        training_metric  the trainer's own evaluation (``h6_value_train.evaluate``: raw value, no D4); its totals
                         must reproduce ``value_record.json`` (parent and best cycle / replay WMSE, the dropped
                         counts) or the run fails; then split by label kind x side to move
        engine_metric    the same positions through ``ValueLeaf`` (D4 average, what the search uses): sign
                         accuracy, mean prediction and bias by label kind x side to move
        and every unmasked label row classified by ``analysis.puct_value_v1.exact_leaf`` (settled without the network
        in the search; row and weight shares), the fresh-probe gate and P93 / P94 (``h7c_arena.value_probes``)
    D5  visit targets of the policy rows by side to move (visits rows only; proof one-hot rows counted apart): raw
        visits (visited children, entropy, top share) and the target after proven losses are removed (support,
        entropy, exp(entropy), top share), and the visited children the target dropped

VALUE_ONLY cycles (``--candidate-mode VALUE_ONLY``, B2 c4+ and c3): D4 only. There is no P-only arm and no trained
policy: the candidate policy must be the parent policy byte for byte, the verdict must be a VALUE_ONLY verdict naming
both model sets, and the label file and the cycle cache manifest must be the ones the cycle's evidence manifest names
(B2 ``h7c-b2-evidence-v1`` by repository path, c3 ``h7c-phase-a-evidence-v1`` by file name).

    python -u scripts/h7c_diagnose.py --cycle 1 --verdict docs/h7c/c1/arena_verdict.json \\
        --evidence docs/h7c/c1/evidence_manifest.json --p-only runs/h7c/c1/arena/p_only.json \\
        --value-record docs/h7c/c1/value_record.json --cycle-cache runs/h7c/c1/value_cache \\
        --replay-cache data/tactical/h6/value_cache_v3 \\
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

FORMAT = 'h7c-diagnosis-v2'
PARTS = ('d1', 'd2', 'd3', 'd4', 'd5')
THREATS = ('five', 'four', 'open_four', 'window3')
WMSE_TOLERANCE = 1e-4  # the trainer's numbers, recomputed: any larger difference means another position set or model
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

VALUE_ONLY = 'VALUE_ONLY'
EVIDENCE_FORMATS = ('h7c-b2-evidence-v1', 'h7c-phase-a-evidence-v1')


def _evidence_sha(files: dict, name: str):
    """The one manifest entry for ``name`` (a file name, or a path ending in it)."""
    hits = [v for k, v in files.items() if k == name or k.endswith('/' + name)]
    return hits[0] if len(hits) == 1 else None


def check_value_only_inputs(args) -> dict:
    """D4 of a VALUE_ONLY cycle: the verdict's two model sets, the same policy, the evidence's label file and cache."""
    from analysis import tb_v3_net
    from h7c_arena import FORMAT as ARENA_FORMAT

    if list(args.parts or ['d4']) != ['d4']:
        raise ValueError(f'{VALUE_ONLY}: D4 only (no P-only arm, no trained policy)')
    verdict, evidence = _json(args.verdict), _json(args.evidence)
    candidate = tb_v3_net.model_identity(args.candidate_policy, args.candidate_value)
    parent = tb_v3_net.model_identity(args.parent_policy, args.parent_value)
    policy_keys = ('policy_checkpoint_sha256', 'policy_metadata_sha256')
    files = evidence.get('files_sha256', {})
    checks = {
        'verdict of this cycle': verdict.get('format') == f'{ARENA_FORMAT}-verdict' and verdict.get('cycle') == args.cycle,
        'VALUE_ONLY verdict': verdict.get('candidate_mode') == VALUE_ONLY,
        'verdict models': verdict.get('candidate') == candidate and verdict.get('parent') == parent,
        'same policy': all(candidate[k] == parent[k] for k in policy_keys),
        'evidence of this cycle': evidence.get('format') in EVIDENCE_FORMATS and evidence.get('cycle') == args.cycle,
        'evidence verdict': (evidence.get('verdict_sha256') == _sha(args.verdict)
                             if evidence.get('format') == 'h7c-b2-evidence-v1'
                             else evidence.get('verdict_sha256') in (None, _sha(args.verdict))),
        'value rows': _evidence_sha(files, 'value_rows.jsonl.gz') == _sha(args.value_rows),
        'cycle cache': _evidence_sha(files, 'value_cache/manifest.json') == _sha(Path(args.cycle_cache) / 'manifest.json'),
        'value record': _evidence_sha(files, 'value/record.json') == _sha(args.value_record),
        'value metadata': _evidence_sha(files, 'value/best.json') == candidate['value_metadata_sha256'],
    }
    # Tensor hashes do not bind the game-index -> game-id mapping used to join D4 predictions to label rows.
    # Both cache tensors and game lists must be the original evidence files before this join is trusted.
    for split in ('train', 'val'):
        for suffix in ('.pt', '_games.json'):
            name = f'value_cache/{split}{suffix}'
            checks[name] = _evidence_sha(files, name) == _sha(Path(args.cycle_cache) / f'{split}{suffix}')
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        raise ValueError(f'{VALUE_ONLY} D4 inputs are not the ones the cycle records name ({", ".join(failed)})')
    out = {'candidate_mode': VALUE_ONLY, 'candidate': candidate, 'parent': parent,
           'sha256': {'verdict': _sha(args.verdict), 'evidence': _sha(args.evidence),
                      'value_rows': _sha(args.value_rows)}}
    out['value'] = check_value_inputs(args, evidence, candidate, parent,
                                      labels=_json(Path(args.cycle_cache) / 'manifest.json')['labels_sha256'])
    return out


def check_inputs(args) -> dict:
    """Fail closed unless the files are the ones the cycle's records name; returns their identities."""
    from analysis import tb_v3_fast5, tb_v3_net
    from h7c_arena import FORMAT as ARENA_FORMAT, signature

    if getattr(args, 'candidate_mode', 'JOINT') == VALUE_ONLY:
        return check_value_only_inputs(args)
    if args.p_only is None or args.policy_rows is None:
        raise ValueError('a joint cycle needs --p-only and --policy-rows')

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
    policy_keys, value_keys = ('policy_checkpoint_sha256', 'policy_metadata_sha256'), ('value_checkpoint_sha256',
                                                                                         'value_metadata_sha256')
    if any(tested[k] != candidate[k] for k in policy_keys) or any(tested[k] != parent[k] for k in value_keys):
        raise ValueError('the P-only games were not played by candidate policy + parent value')
    if identity['value_device'] != args.device.split(':')[0] or \
            identity['runtime_hash'] != tb_v3_fast5.runtime_hash(args.device):
        raise ValueError(f"--device {args.device}: the P-only games ran on {identity['value_device']} "
                         f"(runtime {identity['runtime_hash'][:12]}); a replay elsewhere is not the same engine")
    out = {'candidate': candidate, 'parent': parent, 'p_only_signature': p_only['signature'],
           'sha256': {'verdict': _sha(args.verdict), 'evidence': _sha(args.evidence), 'p_only': p_only_sha,
                      'policy_rows': files['policy_rows.jsonl.gz'], 'value_rows': files['value_rows.jsonl.gz']}}
    if 'd4' in (args.parts or PARTS):
        out['value'] = check_value_inputs(args, evidence, candidate, parent)
    return out


def check_value_inputs(args, evidence: dict, candidate: dict, parent: dict, labels: str | None = None) -> dict:
    """The value record and both caches are the ones the training used (``labels``: the cycle labels SHA when the
    evidence has no training section, already tied to the evidence by the cache manifest file)."""
    record = _json(args.value_record)
    inputs = record.get('inputs', {})
    cycle_manifest, replay_manifest = _json(Path(args.cycle_cache) / 'manifest.json'), \
        _json(Path(args.replay_cache) / 'manifest.json')
    checks = {
        'candidate': (record.get('candidate') or {}).get('checkpoint_sha256') == candidate['value_checkpoint_sha256'],
        'parent': inputs.get('parent_checkpoint_sha256') == parent['value_checkpoint_sha256'],
        'cycle labels': inputs.get('cycle_labels_sha256') == cycle_manifest['labels_sha256']
        == (labels if labels is not None else evidence.get('training', {}).get('cycle_value_labels_sha256')),
        'replay labels': inputs.get('replay_labels_sha256') == replay_manifest['labels_sha256'],
        'cycle splits': inputs.get('cycle_cache_splits') == {k: v['sha256'] for k, v in cycle_manifest['splits'].items()},
        'replay splits': inputs.get('replay_cache_splits') == {k: v['sha256'] for k, v in replay_manifest['splits'].items()},
        'cycle': record.get('cycle') == args.cycle,
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        raise ValueError(f'{args.value_record} does not name these value inputs ({", ".join(failed)})')
    return {'value_record': _sha(args.value_record), 'cycle_labels': cycle_manifest['labels_sha256'],
            'replay_labels': replay_manifest['labels_sha256']}


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


def top1_correct(probs: dict, target: dict) -> bool:
    """The trainer's top-1: the predicted move's target is the target's maximum (tied maxima all count)."""
    return target.get(top(probs), 0.0) == max(target.values())


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


# --- D1 / D3: one replay of the P-only games ---------------------------------------------------------------------

def _init_worker(paths: dict, device: str) -> None:
    _WORKER.update(paths=paths, device=device)


def _agent(policy, value, seed):
    from h7c_arena import make_side

    return make_side({'policy': policy, 'value': value}, seed, _WORKER['device'])


def _policies():
    """The parent and candidate ``RootPolicy`` objects the agents use (``h7c_arena``'s per-process model cache)."""
    from h7c_arena import _loaded

    paths, device = _WORKER['paths'], _WORKER['device']
    return (_loaded(paths['parent_policy'], paths['parent_value'], device)[0],
            _loaded(paths['candidate_policy'], paths['parent_value'], device)[0])


def _decision(agent, game: Game) -> dict:
    from hybrid.h7c_targets import raw_visits
    from run_mcts_v8_benchmark import _move_record

    started = perf_counter()
    move = agent.select_move(game)
    diag = agent.diagnostics
    record = _move_record(len(game.history), perf_counter() - started, diag)
    return {'move': [int(move[0]), int(move[1])], 'route': record['route'], 'visits': raw_visits(diag),
            'prior_top': record['tree']['prior_top'], 'prior_top_prob': record['tree']['prior_top_prob'],
            'prior_entropy': record['tree']['prior_entropy'], 'root_q': record['tree']['root_q']}


def d1_row(game: Game, ply: int, decision: dict, parent_policy, candidate_policy) -> dict:
    root = [(r, c) for r, c, _, _ in decision['visits']]
    legal = game.legal_moves()
    pp, cp = parent_policy(game), candidate_policy(game)
    on_root = compare_policies(restricted(pp, root), restricted(cp, root), threat_classes(game, root))
    on_legal = compare_policies(restricted(pp, legal), restricted(cp, legal), threat_classes(game, legal))
    # the engine records the root prior's entropy normalized by log(children) (puct_v8.prior_entropy), 4 decimals
    normalized = on_root['candidate_entropy'] / math.log(len(root)) if len(root) > 1 else 0.0
    return {'ply': ply, 'side': side(ply), 'root_children': len(root), 'root': on_root, 'legal': on_legal,
            'engine_prior_entropy_diff': abs(round(normalized, 4) - decision['prior_entropy'])}


def replay(task: dict) -> dict:
    """The candidate side of one P-only game replayed: D1 rows at its tree moves; for a loss, the parent agent (same
    seed) beside it until the first move it would play differently (D3)."""
    from h7c_arena import OPENING
    from run_mcts_v8_benchmark import derive_seed

    g, with_parent = task['game'], task['d3']
    paths = _WORKER['paths']
    seed = derive_seed(g['seed'], 'tested')
    candidate = _agent(paths['candidate_policy'], paths['parent_value'], seed)
    parent = _agent(paths['parent_policy'], paths['parent_value'], seed) if with_parent else None
    parent_policy, candidate_policy = _policies()
    color = BLACK if g['tested_color'] == 'black' else -BLACK
    opening, game = OPENING['random_plies'] + 1, Game()
    rows, d3, status = [], None, 'REPRODUCED'
    for ply, move in enumerate(g['moves']):
        if game.to_play == color and ply >= opening:
            c = _decision(candidate, game)
            if c['move'] != list(move):
                status = {'status': 'NOT_REPRODUCED', 'ply': ply, 'recorded': list(move), 'replayed': c['move']}
                break
            if c['route'] == 'tree' and c['visits']:
                rows.append(d1_row(game, ply, c, parent_policy, candidate_policy))
            if parent is not None and d3 is None:
                p = _decision(parent, game)
                if p['move'] != c['move']:
                    d3 = {'status': 'DIVERGED', 'ply': ply, 'side': side(ply), 'plies_left': len(g['moves']) - ply,
                          'candidate': c, 'parent': p}
        game.play(*move)
    if parent is not None and d3 is None:
        d3 = {'status': 'NOT_REPRODUCED'} if status != 'REPRODUCED' else {'status': 'NO_DIVERGENCE'}
    return {'key': g['key'], 'result': g['result'], 'length': g['length'], 'replay': status, 'd1': rows,
            'd3': d3 and {'key': g['key'], 'length': g['length'], **d3}}


def _replay_chunk(tasks: list[dict]) -> list[dict]:
    return [replay(t) for t in tasks]


def run_replay(games: list[dict], paths: dict, device: str, workers: int, with_d3: bool) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    tasks = [{'game': g, 'd3': with_d3 and g['result'] == 'loss'} for g in games]
    out = []
    progress = Progress('h7c-diagnose-replay', len(tasks), 'games')
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=_init_worker,
                             initargs=(paths, device)) as pool:
        run_chunks(pool, _replay_chunk, [[t] for t in tasks], progress, len, unit_name='games',
                   on_result=lambda _, part: out.extend(part))
    return sorted(out, key=lambda r: r['key'])


def d1(replayed: list[dict]) -> dict:
    rows = [r for g in replayed for r in g['d1']]
    summary = {support: {s: aggregate([r[support] for r in rows if s in (r['side'], 'all')])
                         for s in ('all', 'black', 'white')} for support in ('root', 'legal')}
    diffs = [r['engine_prior_entropy_diff'] for r in rows]
    return {'positions': len(rows), 'threat_classes': THREATS,
            'replay': {'games': len(replayed),
                       'not_reproduced': [g['replay'] for g in replayed if g['replay'] != 'REPRODUCED']},
            'root_children': _mean([r['root_children'] for r in rows]),
            'engine_prior_entropy_max_diff': max(diffs) if diffs else None, 'summary': summary,
            'rows': [{'key': g['key'], 'ply': r['ply'], 'side': r['side'], 'root_children': r['root_children'],
                      'root': {k: v for k, v in r['root'].items() if k != 'threat_mass'}}
                     for g in replayed for r in g['d1']]}


def d3(replayed: list[dict]) -> dict:
    rows = [g['d3'] for g in replayed if g['d3'] is not None]
    diverged = [r for r in rows if r['status'] == 'DIVERGED']
    routes = {}
    for r in diverged:
        pair = f"{r['candidate']['route']} / {r['parent']['route']}"
        routes[pair] = routes.get(pair, 0) + 1
    return {'losses': len(rows), 'status': {s: sum(r['status'] == s for r in rows)
                                            for s in ('DIVERGED', 'NO_DIVERGENCE', 'NOT_REPRODUCED')},
            'routes_candidate_parent': routes,
            'by_side': {s: sum(r['side'] == s for r in diverged) for s in ('black', 'white')},
            'plies_left': {'mean': _mean([r['plies_left'] for r in diverged]),
                           'values': sorted(r['plies_left'] for r in diverged)},
            'rows': rows}


# --- D2 / D5: policy rows ----------------------------------------------------------------------------------------

def read_policy_rows(path) -> list[dict]:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def d2(rows: list[dict], parent_policy, candidate_policy) -> dict:
    strata: dict = {}
    for r in rows:
        if r['mask'] or r['split'] != 'val' or r['kind'] not in ('visits', 'proof_one_hot'):
            continue
        game = position(r['moves'])
        target = {(a, b): p for a, b, p in r['target']}
        cell = strata.setdefault((r['kind'], side(len(r['moves']))), {'n': 0, 'parent': [0.0, 0], 'candidate': [0.0, 0]})
        cell['n'] += 1
        for who, policy in (('parent', parent_policy), ('candidate', candidate_policy)):
            probs = policy(game)
            cell[who][0] += -sum(p * math.log(max(probs.get(m, 0.0), 1e-12)) for m, p in target.items())
            cell[who][1] += int(top1_correct(probs, target))
    out = {}
    for (kind, s), cell in sorted(strata.items()):
        n = cell['n']
        out[f'{kind}/{s}'] = {'n': n, **{who: {'ce': round(cell[who][0] / n, 5), 'top1': round(cell[who][1] / n, 5)}
                                        for who in ('parent', 'candidate')},
                              'delta_ce': round((cell['candidate'][0] - cell['parent'][0]) / n, 5),
                              'delta_top1': round((cell['candidate'][1] - cell['parent'][1]) / n, 5)}
    return out


def _dist_stats(ps: list[float]) -> dict:
    total = sum(ps)
    ps = [p / total for p in ps if p > 0]
    h = -sum(p * math.log(p) for p in ps)
    return {'entropy': h, 'support': math.exp(h), 'top_share': max(ps), 'children': len(ps)}


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
            raw = _dist_stats([v[2] for v in r['raw_visits']])
            target = _dist_stats([p for _, _, p in r['target']])
            stats[s].append({'raw': raw, 'target': target, 'filtered_children': raw['children'] - target['children']})
    out = {}
    for s, xs in stats.items():
        out[s] = {'visits_rows': len(xs), 'proof_one_hot_rows': proof[s],
                  'raw_visits': {k: _mean([x['raw'][k] for x in xs]) for k in ('children', 'entropy', 'top_share')},
                  'target': {k: _mean([x['target'][k] for x in xs])
                             for k in ('children', 'entropy', 'support', 'top_share')},
                  'filtered_children': _mean([x['filtered_children'] for x in xs]),
                  'rows_with_filtered_children': sum(x['filtered_children'] > 0 for x in xs)}
    return out


# --- D4: value ---------------------------------------------------------------------------------------------------

def _value_model(checkpoint, device):
    from model.checkpoint import load_checkpoint
    from model.config import ModelConfig

    meta = _json(Path(checkpoint).with_name('best.json'))
    return load_checkpoint(checkpoint, ModelConfig(**meta['model_config']), device=device).eval()


def _value_stats(items: list[tuple], idx: int) -> dict:
    """items: (target, weight, value...) rows; ``idx`` the value column."""
    w = sum(i[1] for i in items)
    nonzero = [i for i in items if i[0] != 0]
    return {'wmse': round(sum(i[1] * (i[idx] - i[0]) ** 2 for i in items) / w, 6) if w else None,
            'mse': round(sum((i[idx] - i[0]) ** 2 for i in items) / len(items), 6),
            'sign_acc': round(sum((i[idx] > 0) == (i[0] > 0) for i in nonzero) / len(nonzero), 5) if nonzero else None,
            'mean_prediction': round(sum(i[idx] for i in items) / len(items), 5),
            'bias': round(sum(i[idx] - i[0] for i in items) / len(items), 5)}


def training_metric(args, record: dict, log=print):
    """The trainer's effective cycle val and its own evaluation, reproduced (fails on a mismatch); per-row raw values
    keyed by (game id, ply)."""
    import torch

    from hybrid import h7c_train as tr
    from hybrid.h6_value_cache import load_value_cache, planes_from_value_cache
    from hybrid.h6_value_train import evaluate, value_forward

    load = lambda cache: {s: load_value_cache(Path(cache), s, verify_hash=True)[0] for s in ('train', 'val')}  # noqa: E731
    cycle, replay, dropped = tr.union_masks(load(args.cycle_cache), load(args.replay_cache))
    if dropped != record['union_masking']:
        raise ValueError(f"union masking gives {dropped}, the training recorded {record['union_masking']}")
    recorded = {'parent': {'cycle': record['parent_val']['cycle_wmse'], 'replay': record['parent_val']['replay_wmse']},
                'candidate': {'cycle': record['best']['cycle_wmse'], 'replay': record['best']['replay_wmse']}}
    games = json.loads((Path(args.cycle_cache) / 'val_games.json').read_text(encoding='utf-8'))
    val = cycle['val']
    keys = [(games[int(g)], int(p)) for g, p in zip(val['game'], val['ply'])]
    raw, totals = {}, {}
    for who, checkpoint in (('parent', args.parent_value), ('candidate', args.candidate_value)):
        model = _value_model(checkpoint, args.device)
        totals[who] = {'cycle': evaluate(model, cycle['val'], args.device)['weighted_mse'],
                       'replay': evaluate(model, replay['val'], args.device)['weighted_mse']}
        for part in ('cycle', 'replay'):
            if abs(totals[who][part] - recorded[who][part]) > WMSE_TOLERANCE:
                raise ValueError(f'{who} {part} WMSE {totals[who][part]:.6f} does not reproduce the recorded '
                                 f'{recorded[who][part]:.6f}: not the training position set or model')
        with torch.no_grad():
            planes, _, _ = planes_from_value_cache(val, torch.arange(len(keys)), args.device)
            raw[who] = value_forward(model, planes).float().cpu().tolist()
        log(f'[h7c-diagnose] D4 training metric {who}: reproduced {totals[who]}', flush=True)
    rows = {k: {'target': float(t), 'weight': float(w), 'parent_raw': raw['parent'][i],
                'candidate_raw': raw['candidate'][i]}
            for i, (k, t, w) in enumerate(zip(keys, val['value'].tolist(), val['weight'].tolist()))}
    return rows, {'dropped': dropped, 'reproduced': totals, 'recorded': recorded, 'tolerance': WMSE_TOLERANCE}


def d4(args, log=print) -> dict:
    from analysis.puct_value_v1 import exact_leaf
    from h7c_arena import value_probes
    from hybrid.h6_value_cache import iter_games
    from hybrid.h6_value_leaf import ValueLeaf

    record = _json(args.value_record)
    effective, reproduction = training_metric(args, record, log=log)
    leaves = {'parent': ValueLeaf(args.parent_value, device=args.device, expected_sha256=None),
              'candidate': ValueLeaf(args.candidate_value, device=args.device, expected_sha256=None)}
    bypass, cells, digest = {}, {}, hashlib.sha256()
    for game_id, rows in iter_games(Path(args.value_rows), digest):
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
            row = effective.get((game_id, r['ply']))
            if row is None:
                continue
            if (row['target'], round(row['weight'], 6)) != (r['value_target'], round(r['weight'], 6)):
                raise ValueError(f"{game_id} ply {r['ply']}: the cache and the label row differ")
            row.update(kind=r['label_kind'], side=side(r['ply']), route=route,
                       parent_d4=leaves['parent'](game), candidate_d4=leaves['candidate'](game))
            cells.setdefault((r['label_kind'], side(r['ply'])), []).append(row)
    if digest.hexdigest() != reproduction_labels(args):
        raise ValueError(f'{args.value_rows} is not the label file of the cycle cache')
    if any('kind' not in row for row in effective.values()):
        raise ValueError('an effective val position has no unmasked label row')
    total_w = sum(v[1] for b in bypass.values() for v in b.values())
    total_n = sum(v[0] for b in bypass.values() for v in b.values())
    exact = {'by_kind': {k: {route: {'rows': v[0], 'weight': round(v[1], 3)} for route, v in b.items()}
                         for k, b in sorted(bypass.items())},
             'exact_leaf_rows_share': round(sum(b['exact_leaf'][0] for b in bypass.values()) / total_n, 5),
             'exact_leaf_weight_share': round(sum(b['exact_leaf'][1] for b in bypass.values()) / total_w, 5)}

    def split(columns):
        out = {}
        for (kind, s), rows in sorted(cells.items()):
            items = [(r['target'], r['weight'], r[columns[0]], r[columns[1]]) for r in rows]
            out[f'{kind}/{s}'] = {'n': len(rows), 'weight': round(sum(r['weight'] for r in rows), 3),
                                  'exact_leaf_rows': sum(r['route'] == 'exact_leaf' for r in rows),
                                  'mean_target': round(sum(r['target'] for r in rows) / len(rows), 5),
                                  'parent': _value_stats(items, 2), 'candidate': _value_stats(items, 3)}
        return out

    def by_route(columns):
        """Totals over the effective val rows: all, those the search sends to the network, those exact_leaf settles."""
        rows = [r for cell in cells.values() for r in cell]
        out = {}
        for name, keep in (('all', lambda r: True), ('network', lambda r: r['route'] == 'network'),
                           ('exact_leaf', lambda r: r['route'] == 'exact_leaf')):
            items = [(r['target'], r['weight'], r[columns[0]], r[columns[1]]) for r in rows if keep(r)]
            if not items:
                out[name] = {'n': 0}
                continue
            parent, candidate = _value_stats(items, 2), _value_stats(items, 3)
            out[name] = {'n': len(items), 'weight': round(sum(i[1] for i in items), 3), 'parent': parent,
                         'candidate': candidate, 'wmse_relative_improvement':
                         round((parent['wmse'] - candidate['wmse']) / parent['wmse'], 5) if parent['wmse'] else None}
        return out
    log('[h7c-diagnose] D4 probes', flush=True)
    probes = {'parent': value_probes(args.parent_value, args.device),
              'candidate': value_probes(args.candidate_value, args.device)}
    for p in probes.values():
        p['fresh'].pop('rows', None)
    return {'positions': len(effective),
            'training_metric': {'inference': 'raw (h6_value_train.evaluate)', 'reproduction': reproduction,
                                'by_route': by_route(('parent_raw', 'candidate_raw')),
                                'strata': split(('parent_raw', 'candidate_raw'))},
            'engine_metric': {'inference': 'ValueLeaf D4 average', 'by_route': by_route(('parent_d4', 'candidate_d4')),
                              'strata': split(('parent_d4', 'candidate_d4'))},
            'exact_leaf': exact, 'probes': probes}


def reproduction_labels(args) -> str:
    return _json(Path(args.cycle_cache) / 'manifest.json')['labels_sha256']


# --- main --------------------------------------------------------------------------------------------------------

def run(args, log=print) -> dict:
    from h7c_arena import _git
    from hybrid.h4_policy import RootPolicy

    git = _git()
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty (or git is unavailable): a diagnosis must name its code commit')
    inputs = check_inputs(args)
    parts = args.parts or (['d4'] if getattr(args, 'candidate_mode', 'JOINT') == VALUE_ONLY else PARTS)
    games = _json(args.p_only)['games'] if {'d1', 'd3'} & set(parts) else None
    out = {'format': FORMAT, 'cycle': args.cycle, 'candidate_mode': getattr(args, 'candidate_mode', 'JOINT'),
           'git': git, 'device': args.device, 'inputs': inputs,
           'inference': {'policy': 'RootPolicy (CPU, as in the Arena workers)', 'value_engine': 'ValueLeaf D4',
                         'value_training': 'h6_value_train.evaluate (raw)'}, 'parts': {}}
    policy_rows = read_policy_rows(args.policy_rows) if {'d2', 'd5'} & set(parts) else []
    replayed = None
    if {'d1', 'd3'} & set(parts):
        started = perf_counter()
        paths = {k: str(getattr(args, k)) for k in ('candidate_policy', 'parent_policy', 'parent_value')}
        replayed = run_replay(games, paths, args.device, args.workers, 'd3' in parts)
        log(f'[h7c-diagnose] replay: {len(replayed)} games in {perf_counter() - started:.0f} s', flush=True)
    for part in parts:
        started = perf_counter()
        if part == 'd1':
            result = d1(replayed)
        elif part == 'd2':
            result = d2(policy_rows, RootPolicy(args.parent_policy), RootPolicy(args.candidate_policy))
        elif part == 'd3':
            result = d3(replayed)
        elif part == 'd4':
            result = d4(args, log=log)
        else:
            result = d5(policy_rows)
        out['parts'][part] = {**result, 'seconds': round(perf_counter() - started, 1)}
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cycle', type=int, required=True)
    parser.add_argument('--verdict', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--candidate-mode', choices=('JOINT', VALUE_ONLY), default='JOINT',
                        help=f'{VALUE_ONLY}: D4 only, no --p-only / --policy-rows')
    parser.add_argument('--p-only', type=Path, help="the P-only arm's play output (games included; joint cycles)")
    parser.add_argument('--policy-rows', type=Path, help='the policy label rows (joint cycles)')
    for name in ('candidate-policy', 'candidate-value', 'parent-policy', 'parent-value', 'value-rows'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--value-record', type=Path, help="D4: the value training's record.json")
    parser.add_argument('--cycle-cache', type=Path, help='D4: the cycle value cache the training used')
    parser.add_argument('--replay-cache', type=Path, help='D4: the replay-anchor cache the training used')
    parser.add_argument('--parts', nargs='+', choices=PARTS)
    parser.add_argument('--device', default='cuda', help="the P-only run's value device")
    parser.add_argument('--workers', type=int, default=1, help='replay processes (D1 / D3)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-dirty', action='store_true', help='tests only')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    if args.candidate_mode != VALUE_ONLY and None in (args.p_only, args.policy_rows):
        parser.error('a joint cycle needs --p-only and --policy-rows')
    if 'd4' in (args.parts or PARTS) and None in (args.value_record, args.cycle_cache, args.replay_cache):
        parser.error('D4 needs --value-record, --cycle-cache and --replay-cache')
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
