"""R2-S400-VT-MEASURE-1: on the S50 search-contest positions of H-dev, is the S400 result of the same H3 + c5 worth a
value target? Measurement only: nothing here trains, plays an Arena, or changes a champion.

The sealed H-dev of R2-D4 (396 games, the 15,985 eligible positions of the committed D4 manifest) is replayed with the
actor's seeds by two searches of the same models: S50 (the frozen TB-V3 / fast5 S50; it must reproduce every actor
move, route and root visit) and S400 (the same agent with both tree budgets x8, ``h7c_budget.make_budget_agent``). Each
replayed ply keeps both searches' route, final move, root visits / Q, the tree's own choice before the root checks
(``v8_v7_move``) and the root prior top (``v8_puct_prior_top``), and c5's raw network value of the position.

    classes  E: the eligible positions; T: both searches on the tree route with root visits; V (S50 search contest): T
             positions where the S50 final move is not the root prior top (V-A) or the S50 top two root visit counts
             are within 20 % of the root's total visits (V-B; fewer than two visited children: not V-B). V says the
             S50 search was contested there, not that the value changed the move: the tree's choice and the final move
             differ in some positions (root checks) and are counted apart.
    Q1       where the S400 and S50 final moves differ: both moves continued by a fresh Rapfi to the end (R2-D4 branches,
             replay checks, INVALID policy). delta = score(S400 move) - score(S50 move). Primary: V; also T and E.
    Q2       on every position: a fresh Rapfi plays the position itself from its first move to the end (a position
             branch: no fixed move, its key apart from every move branch), r = the result for the side to move.
             d = (v_c5 - r)^2 - (v_S400 - r)^2 with v_c5 = (c5 network value + 1) / 2 and v_S400 = (Q + 1) / 2, Q the
             visit-weighted mean of the visited S400 root children's Q (both from the side to move). Positive d: the
             S400 value has the smaller squared error against that fixed Rapfi result. Primary: V; also T.
    reading  per primary metric, a source-game cluster bootstrap (20,000 replicates, seed 20261151) of the
             conservative bound (unresolved branches against S400): BETTER when the 95 % interval of the lower bound
             is above 0, WORSE when that of the upper bound is below 0, else INCONCLUSIVE; full needs per metric 100
             observations from 30 games with at most 5 % unresolved. Both BETTER opens a value-target training
             contract draft; anything else opens none.

    prepare --log-name NAME     the inputs (H-dev, D4 manifest, models, runtime, Rapfi) checked; the prepare record
    run --mode smoke|full ...   replay, branches, statistics (full), summary; resumable journals; HEAD and every
                                input checked again just before the summary is published
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h7r_d4 as base  # noqa: E402
from hybrid import h7r_d4 as d4  # noqa: E402
from hybrid.h7c_nmask import committed  # noqa: E402
from hybrid.h7r_ledger import file_lock, publish_json  # noqa: E402

FORMAT = 'h7r-vt-measure-contract-v1'
CONTRACT_ID = 'R2-S400-VT-MEASURE-1'
D = 'docs/h7r/r2/R2-S400-VT-MEASURE-1'
CONTRACT = f'{D}/vt-measure-contract.json'
STAGES = ('prepare', 'smoke', 'full')
SEARCHES = {'S50': 1, 'S400': 8}  # the budget factor of each search (h7c_budget.make_budget_agent)
ACTOR = 'S50'
CLASSIFICATION = {'contest_visit_gap': '1/5', 'min_visited_children': 2,
                  'prior_top': 'v8_puct_prior_top (the root child with the highest prior)',
                  'move': 'the final engine move (after the root checks)'}
STATS = {'samples': 20000, 'seed': 20261151, 'level': 0.95, 'min_observations': 100, 'min_games': 30,
         'max_unresolved_share': '1/20'}
SMOKE = {'games': 2, 'max_games': 10, 'move_observations': 4, 'positions': 4}
VALUE = {'c5': '(network value + 1) / 2, side to move',
         'S400': '(Q + 1) / 2, Q the visit-weighted mean Q of the visited root children, side to move',
         'reference': 'r: a fresh Rapfi plays the position to the end; win 1, draw 0.5, loss 0 for the side to move',
         'd': '(v_c5 - r)^2 - (v_S400 - r)^2'}
KEYS = {'format', 'contract_id', 'status', 'approved', 'purpose', 'draft', 'baseline', 'hdev', 'sample',
        'frozen_state_sha256', 'model', 'searches', 'device', 'runtime_identity', 'rapfi_program', 'rapfi_identity',
        'classification', 'value', 'stats', 'invalid_policy', 'smoke', 'entry', 'output', 'sources', 'scope'}
ENTRY_OPEN = 'OPEN_VALUE_TARGET_TRAINING_DRAFT'


BASELINE = 'docs/h7r/r2/R2-D4-BASELINE-1.contract.json'
SAMPLE = 'docs/h7r/r2/R2-D4-BASELINE-1/eligible.json'
SOURCES = ('scripts/h7r_vt_measure.py', 'scripts/h7r_d4.py', 'src/hybrid/h7r_d4.py', 'scripts/h7c_budget.py',
           'scripts/h7c_adjudicate.py', 'scripts/h7c_sensitivity.py', 'scripts/h7c_retarget.py',
           'src/hybrid/h7r_evidence.py', 'src/hybrid/h7r_eval.py')


def require(condition, message):
    d4.require(condition, message)


def contract_body(root=ROOT, sources=SOURCES) -> dict:
    """The fixed contract (every approval false) from the pinned R2-D4 contract and the checkout's sources."""
    root = Path(root)
    b = json.loads((root / BASELINE).read_text(encoding='utf-8'))
    ident = {k: b['models']['c5'][k] for k in ('policy_checkpoint_sha256', 'policy_metadata_sha256',
                                              'value_checkpoint_sha256', 'value_metadata_sha256')}
    return {
        'format': FORMAT, 'contract_id': CONTRACT_ID, 'status': 'fixed',
        'approved': {s: False for s in STAGES},
        'purpose': 'development measurement: on the S50 search-contest positions of H-dev, are the S400 moves better '
                   '(Q1) and is the S400 root value closer to a fixed Rapfi result than c5\'s value (Q2)?',
        'draft': {'path': f'{D}-draft.md'},
        'baseline': {'path': BASELINE, 'sha256': d4.file_sha256(root / BASELINE)},
        'hdev': b['hdev'], 'sample': {'path': SAMPLE, 'sha256': d4.file_sha256(root / SAMPLE)},
        'frozen_state_sha256': b['frozen_state_sha256'],
        'model': {'policy': 'runs/h3_policy_64x4/best.pt', 'value': 'runs/h7c/c5/value/best.pt', 'identity': ident},
        'searches': dict(SEARCHES), 'device': 'cuda', 'runtime_identity': b['runtime_identity'],
        'rapfi_program': b['rapfi_program'], 'rapfi_identity': b['rapfi_identity'],
        'classification': dict(CLASSIFICATION), 'value': dict(VALUE), 'stats': dict(STATS),
        'invalid_policy': dict(d4.POLICY), 'smoke': dict(SMOKE),
        'entry': {'rule': 'Q1 (V, move disagreements) BETTER and Q2 (V, every position) BETTER: a value-target training '
                          'contract draft may be opened; anything else opens none',
                  'never': 'no change of V, thresholds, sample or settings after any result (a new contract instead)'},
        'output': {'prepare': f'{D}/prepare.json', 'raw': 'runs/h7r/R2-S400-VT-MEASURE-1/{mode}',
                   'records': D + '/{mode}'},
        'sources': [{'path': rel, 'sha256': d4.file_sha256(root / rel)} for rel in sources],
        'scope': 'development measurement only: no training, Arena, strength, promotion, champion, canonical or V9 '
                 'ruling'}


# --- contract -----------------------------------------------------------------------------------------------------

def experiment_sha256(c: dict) -> str:
    return d4.digest({k: v for k, v in c.items() if k != 'approved'})


def load_contract(path=None, root=ROOT, stage=None, *, check_sources=True) -> tuple[dict, str, str]:
    root = Path(root)
    path = Path(path) if path else root / CONTRACT
    c = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(c, dict) and set(c) == KEYS, f'{path}: the measurement contract keys differ')
    require(c['format'] == FORMAT and c['contract_id'] == CONTRACT_ID and c['status'] == 'fixed',
            f'{path}: not the fixed {CONTRACT_ID} contract')
    a = c['approved']
    require(isinstance(a, dict) and list(a) == list(STAGES) and all(type(v) is bool for v in a.values())
            and not (a['smoke'] and not a['prepare']) and not (a['full'] and not a['smoke']),
            'approvals: prepare, smoke, full in that order')
    require(c['searches'] == SEARCHES and c['classification'] == CLASSIFICATION and c['stats'] == STATS
            and c['smoke'] == SMOKE and c['value'] == VALUE and c['invalid_policy'] == d4.POLICY
            and c['device'] == 'cuda', 'the searches, classes, values, statistics, smoke or INVALID policy differ')
    baseline = json.loads((root / c['baseline']['path']).read_text(encoding='utf-8'))
    require(d4.file_sha256(root / c['baseline']['path']) == c['baseline']['sha256'],
            'the R2-D4 contract is not the pinned one')
    for key in ('hdev', 'frozen_state_sha256', 'runtime_identity', 'rapfi_program', 'rapfi_identity'):
        require(c[key] == baseline[key], f'{key}: not the R2-D4 value')
    require(c['model']['identity'] == {k: baseline['models']['c5'][k] for k in c['model']['identity']}
            and len(c['model']['identity']) == 4, 'the model is not the R2-D4 c5 (H3 policy + c5 value)')
    require(d4.file_sha256(root / c['sample']['path']) == c['sample']['sha256'], 'the D4 eligible manifest differs')
    if check_sources:
        changed = [r['path'] for r in c['sources'] if d4.file_sha256(root / r['path']) != r['sha256']]
        require(not changed, f'the checkout differs from the sources the contract was fixed for: {changed}')
    require(committed(path, root), f'{path}: not committed')
    if stage is not None:
        require(c['approved'][stage] is True, f'the {stage} stage is not approved')
    return c, d4.file_sha256(path), experiment_sha256(c)


def committed_json(root, rel, what):
    path = Path(root) / rel
    require(path.is_file() and committed(path, root), f'{rel}: the {what} is not committed')
    return json.loads(path.read_text(encoding='utf-8'))


# --- inputs -------------------------------------------------------------------------------------------------------

def inputs(c: dict, cache, root=ROOT):
    """The sealed H-dev checks, the D4 eligible manifest rebuilt (it must be the committed one), the selected
    games and the value labels of the eligible rows ((game id, ply) -> label, side to move)."""
    import h7r_holdout as ho
    import h7r_r2 as r2cli
    root = Path(root)
    baseline_sha = c['baseline']['sha256']
    _, data, info, checks = r2cli.check_hdev(c, Path(cache), root)
    hc, _, games, _ = ho.checked_cache(cache, root)
    require(hc['models'][hc['holdout']['actor']]['identity'] == c['model']['identity'], 'the H-dev actor is not c5')
    manifest = base.eligible_manifest({'hdev': c['hdev']}, baseline_sha, games, info, data)
    sample = json.loads((root / c['sample']['path']).read_text(encoding='utf-8'))
    require(manifest == sample, 'the sealed H-dev does not give the committed D4 eligible manifest')
    by_index = {g['game_index']: g for g in games}
    selected = {e['game_index']: by_index[e['game_index']] for e in manifest['games']}
    labels = {(info['game_ids'][g], p): float(v) for g, p, v in zip(info['game'], info['ply'], data['value'].tolist())}
    return manifest, selected, checks, labels


def model_specs(c: dict, root=ROOT) -> dict:
    m = c['model']
    return {name: {'policy': str(Path(root) / m['policy']), 'value': str(Path(root) / m['value']), 'budget': k}
            for name, k in SEARCHES.items()}


def checked_models(c: dict, root=ROOT, device='cuda') -> dict:
    """The model files are the contract's (the four SHA-256, the frozen value state)."""
    from analysis.tb_v3_net import model_identity
    from hybrid.h7r_r2 import check_model
    m = c['model']
    actual = model_identity(Path(root) / m['policy'], Path(root) / m['value'])
    require(actual == m['identity'], f'the model files differ from the contract: {actual}')
    value = Path(root) / m['value']
    loaded, identity = check_model('c5', m['identity'], value, value.with_name('best.json'),
                                   c['frozen_state_sha256'], device)
    del loaded
    return identity


def runtime(device):
    from analysis import tb_v3, tb_v3_fast5
    return {'engine_hash': tb_v3.ENGINE_HASH, 'runtime_hash': tb_v3_fast5.runtime_hash(device)}


def rapfi_spec(c: dict, rapfi_dir, root=ROOT) -> dict:
    import h7c_adjudicate as ad
    spec = ad.engine_spec(Path(root) / c['rapfi_program'], rapfi_dir)
    require(spec['identity'] == c['rapfi_identity'], 'the Rapfi identity differs from the R2-D4 contract')
    return spec


def preflight(c: dict, args, root=ROOT) -> dict:
    """Every input of a stage (tests replace this whole step)."""
    manifest, games, checks, labels = inputs(c, args.cache, root)
    rt = runtime(args.device)
    require(rt == c['runtime_identity'], 'the engine / runtime differs from the contract')
    require(all(all(g.get(k) == v for k, v in rt.items()) and g['value_device'] == 'cuda' for g in games.values()),
            'the source actor engine / runtime / device differs')
    return {'manifest': manifest, 'games': games, 'hdev': checks, 'labels': labels,
            'model': checked_models(c, root, args.device), 'runtime': rt,
            'rapfi': rapfi_spec(c, args.rapfi_dir, root)}


# --- the replay ---------------------------------------------------------------------------------------------------

def search(agent, game) -> dict:
    """The actor-format record (route, final move, raw root visits) with the tree's own choice and the prior top."""
    import h7c_retarget as rt
    record = rt.search(agent, game)
    d = agent.diagnostics
    tree = record['route'] == 'tree'
    record['tree_move'] = list(map(int, d.v8_v7_move)) if tree and d.v8_v7_move is not None else None
    record['prior_top'] = list(map(int, d.v8_puct_prior_top)) if tree and d.v8_puct_prior_top is not None else None
    return record


SET_KEYS = ('route', 'engine_move', 'visits', 'tree_move', 'prior_top', 'seconds')


def replay_game(game: dict, specs: dict, device: str, make=None, value=None) -> dict:
    """``h7c_sensitivity.replay_game`` for the two searches (the actor's positions and seeds) with each search's
    tree choice and prior top, and c5's raw network value of the position (computed after both searches)."""
    from renju import BLACK, Game
    from run_mcts_v8_benchmark import derive_seed
    if make is None:
        import h7c_budget as sb
        make = lambda side, seed: sb.make_budget_agent(side['budget'], side['policy'], side['value'], seed,  # noqa: E731
                                                       device)
    if value is None:
        import h7c_arena as ar
        spec = next(iter(specs.values()))
        value = lambda position: float(ar._loaded(spec['policy'], spec['value'], device)[1](position))  # noqa: E731
    seeds = {'black': derive_seed(game['game_seed'], 'black'), 'white': derive_seed(game['game_seed'], 'white')}
    agents = {name: {color: make(side, seed) for color, seed in seeds.items()} for name, side in specs.items()}
    position = Game()
    for move in game['opening']:
        position.play(*move)
    plies = []
    for p in game['plies']:
        color = 'black' if position.to_play == BLACK else 'white'
        require(p['color'] == color, f"game {game['game_index']} ply {p['ply']}: not the recorded mover")
        sets = {}
        for name in specs:
            record = search(agents[name][color], position)
            sets[name] = {k: record[k] for k in SET_KEYS}
        plies.append({'ply': p['ply'], 'color': color, 'actor_route': p['route'], 'actor_move': p['engine_move'],
                      'sets': sets, 'c5_value': value(position)})
        position.play(*p['played'])
    return {'game_index': game['game_index'], 'plies': plies}


def replay_chunk(tasks):
    return [{**replay_game(t['game'], t['specs'], t['device']), 'identity_sha256': t['identity_sha256']}
            for t in tasks]


def _move_or_none(value) -> bool:
    return value is None or (isinstance(value, list) and len(value) == 2
                             and all(type(x) is int and 0 <= x < 15 for x in value))


def check_replay(record: dict, games: dict) -> None:
    """The baseline replay checks (coverage, order, actor fields, legal moves, visits) and the added fields."""
    import h7c_sensitivity as sn
    sn.check_replay(record, games, SEARCHES, ACTOR)
    for row in record['plies']:
        v = row.get('c5_value')
        require(isinstance(v, float) and -1.0 <= v <= 1.0, f"replay game {record['game_index']}: c5 value")
        for result in row['sets'].values():
            require(set(result) == set(SET_KEYS) and _move_or_none(result['tree_move'])
                    and _move_or_none(result['prior_top']), f"replay game {record['game_index']}: search fields")


# --- classes ------------------------------------------------------------------------------------------------------

def visited(visits) -> list:
    return [v for v in visits if v[2] > 0]


def root_value(visits) -> float:
    """(Q + 1) / 2 for the side to move: Q the visit-weighted mean Q of the visited root children (their mean values
    are from the root mover's view, ``puct_value_v1.backup``)."""
    rows = visited(visits)
    total = sum(v[2] for v in rows)
    require(total > 0, 'a tree record without root visits')
    return (sum(v[2] * v[3] for v in rows) / total + 1.0) / 2.0


def network_value(v: float) -> float:
    """c5's raw network value (side to move, [-1, 1]) on the score scale."""
    return (v + 1.0) / 2.0


def classify(row: dict) -> dict:
    """The classes of one replayed eligible ply (module docstring)."""
    s50, s400 = row['sets']['S50'], row['sets']['S400']
    tree = all(r['route'] == 'tree' and visited(r['visits']) for r in (s50, s400))
    out = {'tree': tree, 'move_diff': list(s50['engine_move']) != list(s400['engine_move']),
           'contest_a': False, 'contest_b': False,
           'tree_final_diff': {name: r['tree_move'] is not None and list(r['tree_move']) != list(r['engine_move'])
                               for name, r in (('S50', s50), ('S400', s400))}}
    if tree:
        out['contest_a'] = s50['prior_top'] is not None and list(s50['engine_move']) != list(s50['prior_top'])
        counts = sorted((v[2] for v in visited(s50['visits'])), reverse=True)
        total = sum(v[2] for v in s50['visits'])
        gap = Fraction(CLASSIFICATION['contest_visit_gap'])
        out['contest_b'] = (len(counts) >= CLASSIFICATION['min_visited_children']
                            and Fraction(counts[0] - counts[1]) <= gap * total)
    out['contest'] = out['contest_a'] or out['contest_b']
    return out


def table(records: list[dict], games: dict, manifest: dict, labels: dict) -> list[dict]:
    """One row per eligible ply (warm-up plies dropped), with its classes, values and branch keys."""
    import h7c_adjudicate as ad
    allowed = {e['game_index']: set(e['plies']) for e in manifest['games']}
    ids = {e['game_index']: e['game_id'] for e in manifest['games']}
    rows = []
    for r in records:
        g = games[r['game_index']]
        for p in r['plies']:
            if p['ply'] not in allowed[r['game_index']]:
                continue
            moves = [list(m) for m in g['moves'][:p['ply']]]
            k = classify(p)
            s50, s400 = p['sets']['S50'], p['sets']['S400']
            row = {'game_index': r['game_index'], 'ply': p['ply'], 'color': p['color'], 'moves': moves, **k,
                   'move_s400': list(s400['engine_move']), 'move_s50': list(s50['engine_move']),
                   'position_key': position_key(moves), 'label': labels.get((ids[r['game_index']], p['ply']))}
            if k['move_diff']:
                row['key_a'] = ad.branch_key(moves, row['move_s400'])
                row['key_b'] = ad.branch_key(moves, row['move_s50'])
            if k['tree']:
                row['v_c5'] = network_value(p['c5_value'])
                row['v_s400'] = root_value(s400['visits'])
                row['v_s50'] = root_value(s50['visits'])
            rows.append(row)
    return rows


def class_counts(rows: list[dict]) -> dict:
    out = {}
    for name, pick in (('E', lambda r: True), ('T', lambda r: r['tree']), ('V', lambda r: r['contest']),
                       ('V-A', lambda r: r['contest_a']), ('V-B', lambda r: r['contest_b']),
                       ('V-A and V-B', lambda r: r['contest_a'] and r['contest_b'])):
        mine = [r for r in rows if pick(r)]
        out[name] = {'positions': len(mine), 'games': len({r['game_index'] for r in mine}),
                     'move_diff': sum(r['move_diff'] for r in mine),
                     'move_diff_share': sum(r['move_diff'] for r in mine) / len(mine) if mine else None,
                     'tree_final_diff': {s: sum(r['tree_final_diff'][s] for r in mine) for s in SEARCHES}}
    return out


# --- branches -----------------------------------------------------------------------------------------------------

def position_key(moves) -> str:
    """A position branch's key: never a move branch's (``branch_key`` hashes [moves, move])."""
    import h7c_adjudicate as ad
    return ad.sha_json({'position_branch': [list(m) for m in moves]})


def move_tasks(rows: list[dict]) -> dict:
    """Every move branch of the disagreements (shared keys run once)."""
    out = {}
    for r in rows:
        if r['move_diff']:
            for key, move in ((r['key_a'], r['move_s400']), (r['key_b'], r['move_s50'])):
                out.setdefault(key, {'key': key, 'moves': r['moves'], 'move': move})
    return out


def position_tasks(rows: list[dict]) -> dict:
    return {r['position_key']: {'key': r['position_key'], 'moves': r['moves']} for r in rows if r['tree']}


def play_position(task: dict) -> dict:
    """A position branch: a fresh Rapfi process plays both colours from the position itself (no fixed move) to the end;
    the score is for the side to move there (``h7c_adjudicate.play_branch`` without the fixed move)."""
    from time import perf_counter
    import h7c_adjudicate as ad
    from agents.external_engine import EngineError, EngineIllegalMove, EngineTimeout, ExternalEngineAgent, \
        our_forbidden_points
    from renju import BLACK, Game

    started = perf_counter()
    game = Game()
    for move in task['moves']:
        game.play(*move)
    mover = game.to_play
    out = {'key': task['key'], 'order': task['order'], 'identity_sha256': task['identity_sha256'], 'kind': 'position',
           'ply': len(task['moves']), 'mover': 'black' if mover == BLACK else 'white', 'continuation': [],
           'forbid_checked': 0}

    def invalid(reason, **extra):
        return {**out, 'status': 'INVALID', 'reason': reason, 'score': None, 'length': len(game.history),
                'seconds': round(perf_counter() - started, 2), **extra}

    require(not game.done, 'a position branch of a finished game')
    try:
        with ExternalEngineAgent(ad._config(task['spec'])) as engine:  # a new process for this branch only
            while not game.done:
                if game.to_play == BLACK:
                    ours, theirs = our_forbidden_points(game), engine.forbidden_points(game)
                    out['forbid_checked'] += 1
                    if ours != theirs:
                        return invalid('forbid_mismatch', ply_at=len(game.history),
                                       ours_only=sorted(map(list, ours - theirs)),
                                       engine_only=sorted(map(list, theirs - ours)))
                try:
                    move = engine.select_move(game)
                except EngineIllegalMove as exc:
                    return invalid('engine_illegal', move_at=list(exc.move), detail=exc.reason)
                game.play(*move)
                out['continuation'].append([int(move[0]), int(move[1])])
    except EngineTimeout as exc:
        return invalid('timeout', detail=str(exc)[:500])
    except EngineError as exc:
        return invalid('engine_error', detail=str(exc)[:500])
    if game.winner is None and any(0 in row for row in game.board):
        return invalid('abnormal_end')
    winner = None if game.winner is None else 'black' if game.winner == BLACK else 'white'
    return {**out, 'status': 'VALID', 'reason': None, 'score': ad.score_for(mover, game), 'winner': winner,
            'length': len(game.history), 'seconds': round(perf_counter() - started, 2)}


def position_chunk(tasks):
    return [play_position(t) for t in tasks]


def move_chunk(tasks):
    import h7c_adjudicate as ad
    return ad._chunk(tasks)


def validate_position(record: dict, task: dict, policy: dict) -> dict:
    """A position branch replayed: its first continuation move stands in for the fixed move of
    ``h7r_evidence.branch_outcome`` (the mover is the side to move before it in both); a failure before Rapfi's first
    move is checked here (an execution failure on an unfinished position, its details against the board)."""
    from hybrid import h7r_eval as ev
    from hybrid.h7r_evidence import branch_outcome
    from renju import BLACK, Game
    require(all(record.get(k) == task[k] for k in ('key', 'order', 'identity_sha256')) and record.get('kind') == 'position',
            'wrong position branch identity / order')
    require(record.get('ply') == len(task['moves']), 'wrong position branch ply')
    require(record.get('mover') == ('black' if len(task['moves']) % 2 == 0 else 'white'), 'wrong position mover')
    cont = record.get('continuation')
    require(isinstance(cont, list), 'position continuation is not a move list')
    if cont:
        view = {**record, 'move': cont[0], 'continuation': cont[1:]}
        out = branch_outcome(view, {'moves': task['moves'], 'move': cont[0]}, policy)
    else:
        board = Game()
        for move in task['moves']:
            board.play(*move)
        status, reason = record.get('status'), record.get('reason')
        require(status == 'INVALID' and record.get('score') is None and record.get('length') == len(board.history)
                and not board.done, 'a position branch without a move must be an execution failure')
        require(reason in ev.EXECUTION_FAILURES, f'INVALID reason {reason!r} before the first move is a data error')
        if reason == 'engine_illegal':
            m = record.get('move_at')
            require(isinstance(m, list) and len(m) == 2 and tuple(m) not in set(board.legal_moves()),
                    'engine_illegal names a legal move')
        elif reason == 'forbid_mismatch':
            from agents.external_engine import our_forbidden_points
            ours = our_forbidden_points(board)
            mine, theirs = ({tuple(m) for m in record.get(k) or []} for k in ('ours_only', 'engine_only'))
            require(board.to_play == BLACK and record.get('ply_at') == len(board.history) and (mine or theirs)
                    and mine <= ours and not theirs & ours, 'forbid_mismatch differs from our forbidden points')
        out = {'kind': 'unresolved' if reason in policy['unresolved_reasons'] else 'unhandled', 'score': None,
               'reason': reason}
    require(out['kind'] != 'unhandled', 'unhandled INVALID')
    return out


def validate_move(record: dict, task: dict, policy: dict) -> dict:
    return base.validate_branch(record, task, policy)


# --- statistics ---------------------------------------------------------------------------------------------------

def _minimum(failures, total, games, unresolved):
    if total < STATS['min_observations']:
        failures.append('TOO_FEW_OBSERVATIONS')
    if games < STATS['min_games']:
        failures.append('TOO_FEW_GAMES')
    if total and Fraction(unresolved, total) > Fraction(STATS['max_unresolved_share']):
        failures.append('UNRESOLVED_OVER_5_PERCENT')


def move_stats(rows: list[dict], outcomes: dict, game_indices, counts) -> dict:
    """Q1 on the disagreements in ``rows``: the R2-D4 statistics (``selection_stats``: ties counted, both bounds of
    unresolved branches, the common bootstrap) at 95 %, with this contract's minimum conditions."""
    obs = [{'game_index': r['game_index'], 'key_a': r['key_a'], 'key_b': r['key_b']} for r in rows if r['move_diff']]
    stats = {'min_observations': 0, 'min_resolved': 0, 'min_resolved_games': 0,
             'joint_level': STATS['level'], 'individual_level': STATS['level']}
    out = d4.selection_stats(obs, outcomes, game_indices, counts, stats)
    out['games'] = len({o['game_index'] for o in obs})
    _minimum(out['failures'], out['observations'], out['games'], out['unresolved'])
    out['label'] = d4.reading(out, STATS['level'])
    for k in ('label95', 'label_individual', 'upper95_descriptive'):
        out.pop(k)
    return out


def _d(a, b, r):
    return (a - r) ** 2 - (b - r) ** 2


def value_stats(rows: list[dict], outcomes: dict, game_indices, counts) -> dict:
    """Q2 on the positions in ``rows``: d = (v_c5 - r)^2 - (v_S400 - r)^2 against each position's fixed Rapfi result.
    d is linear in r, so an unresolved r gives the bounds d(0) and d(1); a structural end scores 0.5."""
    indices = {g: i for i, g in enumerate(game_indices)}
    k = len(game_indices)
    n, low, high, rn = ([0.] * k for _ in range(4))
    unresolved = structural = 0
    sums = {'c5': 0., 'S400': 0., 'S50': 0., 'd_s50': 0., 'abs_s400_c5': 0.}
    label = {'rows': 0, 'c5': 0., 'S400': 0., 'S50': 0.}
    for r in rows:
        i = indices[r['game_index']]
        o = outcomes[r['position_key']]
        require(o['kind'] in ('resolved', 'structural', 'unresolved'), 'unhandled position branch')
        a, b = r['v_c5'], r['v_s400']
        n[i] += 1
        sums['abs_s400_c5'] += abs(b - a)
        if o['kind'] == 'unresolved':
            unresolved += 1
            ends = (_d(a, b, 0.0), _d(a, b, 1.0))
            low[i] += min(ends)
            high[i] += max(ends)
        else:
            require(o['score'] in (0, .5, 1), 'bad position score')
            structural += o['kind'] == 'structural'
            d = _d(a, b, o['score'])
            low[i] += d
            high[i] += d
            rn[i] += 1
            for name, v in (('c5', a), ('S400', b), ('S50', r['v_s50'])):
                sums[name] += (v - o['score']) ** 2
            sums['d_s50'] += _d(a, r['v_s50'], o['score'])
        if r.get('label') is not None:
            target = network_value(r['label'])
            label['rows'] += 1
            for name, v in (('c5', a), ('S400', b), ('S50', r['v_s50'])):
                label[name] += (v - target) ** 2
    total, resolved = int(sum(n)), int(sum(rn))
    games = sum(x > 0 for x in n)
    failures = []
    _minimum(failures, total, games, unresolved)
    out = {'positions': total, 'games': games, 'resolved': resolved, 'unresolved': unresolved,
           'structural': structural, 'failures': failures,
           'mean_d_bounds': [sum(low) / total, sum(high) / total] if total else None,
           'resolved_mse': {name: sums[name] / resolved for name in ('c5', 'S400', 'S50')} if resolved else None,
           'resolved_mean_d_s50': sums['d_s50'] / resolved if resolved else None,
           'mean_abs_s400_minus_c5': sums['abs_s400_c5'] / total if total else None,
           'hdev_label_mse': ({name: label[name] / label['rows'] for name in ('c5', 'S400', 'S50')} | {
               'rows': label['rows'], 'note': 'reference only: the H-dev labels come from the c5 actor games'})
           if label['rows'] else None,
           'intervals': {str(STATS['level']): {'lower': d4.interval(d4.ratio(counts, low, n), STATS['level']),
                                               'upper': d4.interval(d4.ratio(counts, high, n), STATS['level'])}}}
    if any(cell['undefined'] for cell in out['intervals'][str(STATS['level'])].values()):
        failures.append('UNDEFINED_BOOTSTRAP')
    out['label'] = d4.reading(out, STATS['level'])
    return out


def entry(q1: str, q2: str) -> dict:
    """The contract's entry table (fixed before any result): both BETTER opens a training contract draft."""
    worse = [name for name, v in (('Q1', q1), ('Q2', q2)) if v == 'WORSE']
    if q1 == 'BETTER' and q2 == 'BETTER':
        return {'decision': ENTRY_OPEN, 'worse': worse,
                'note': 'a value-target training contract draft may be opened: rows, policy, structure and S50 budget '
                        'as c5, only the value target changed; its strength is decided later by a confirmation Arena'}
    if q1 == 'BETTER':
        note = 'the deeper search picks better moves there, but its value is not shown to be a better target'
    elif q2 == 'BETTER':
        note = 'the value is closer to the Rapfi results, but nothing shows the deeper search picks better moves there'
    else:
        note = 'the S400 value-target direction stops'
    return {'decision': 'NO_TRAINING_CONTRACT', 'worse': worse, 'note': note}


# --- stages -------------------------------------------------------------------------------------------------------

def pool_of(args):
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from multiprocessing import get_context
    if getattr(args, 'threads_for_tests', False):  # in-process fakes only
        return ThreadPoolExecutor(max_workers=1)
    return ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'))


def record_path(c: dict, mode: str, root=ROOT) -> Path:
    return Path(root) / c['output']['records'].format(mode=mode) / 'summary.json'


def prior_record(c: dict, eid: str, mode: str, root=ROOT) -> dict:
    """The committed record a stage needs: prepare before smoke, a reproduced smoke with both branch kinds before
    full. Its SHA-256 goes into the run identity (checked again before publishing)."""
    if mode == 'smoke':
        rel = c['output']['prepare']
        rec = committed_json(root, rel, 'prepare record')
        require(rec.get('experiment_sha256') == eid and rec.get('passed') is True,
                'the committed prepare record is not a passed prepare of this experiment')
    else:
        rel = c['output']['records'].format(mode='smoke') + '/summary.json'
        rec = committed_json(root, rel, 'smoke summary')
        b = rec.get('branches') or {}
        require(rec.get('identity', {}).get('experiment_sha256') == eid and rec.get('reading') == 'SMOKE'
                and rec.get('actor_check', {}).get('reproduced') is True and b.get('move', 0) > 0
                and b.get('position', 0) > 0,
                'the committed smoke is not a reproduced smoke of this experiment with both branch kinds')
    return {'path': rel, 'sha256': d4.file_sha256(Path(root) / rel)}


def prepare(args, root=ROOT) -> dict:
    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'prepare')
    state = base.git_state(root)
    out = root / c['output']['prepare']
    require(not out.exists(), f'{out} exists: never overwritten')
    pre = preflight(c, args, root)
    m = pre['manifest']
    record = {'format': 'h7r-vt-measure-prepare-v1', 'contract_id': CONTRACT_ID, 'experiment_sha256': eid,
              'contract_file_sha256': sha, 'git': state, 'log_name': args.log_name, 'hdev': pre['hdev'],
              'sample_sha256': c['sample']['sha256'],
              'eligible': {'games': len(m['games']), 'rows': m['rows'], 'warmup_plies': m['warmup_plies']},
              'model': pre['model'], 'runtime': pre['runtime'], 'rapfi': pre['rapfi']['identity'], 'passed': True}
    require(base.git_state(root) == state, 'HEAD changed during prepare; nothing is published')
    publish_json(out, record)
    return record


def projection(rows: list[dict], eligible: int) -> dict:
    """Smoke only: each class's share of the smoke positions and the full counts it would give (recorded, never a
    criterion)."""
    counts = class_counts(rows)
    total = counts['E']['positions']
    out = {}
    for name in ('T', 'V'):
        share = counts[name]['positions'] / total if total else None
        diff = counts[name]['move_diff'] / total if total else None
        out[name] = {'share': share, 'projected_positions': None if share is None else round(share * eligible),
                     'projected_move_diffs': None if diff is None else round(diff * eligible)}
    return out


def run(args, root=ROOT) -> dict:
    import torch
    started = time.monotonic()
    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, args.mode)
    state = base.git_state(root)
    prior = prior_record(c, eid, args.mode, root)
    require(args.device == c['device'] and args.workers > 0, 'device / workers differ from the contract')
    out_path = record_path(c, args.mode, root)
    require(not out_path.exists(), f'{out_path} exists: never overwritten')
    print('[vt-preflight] H-dev, D4 manifest, model, runtime, Rapfi', flush=True)
    pre = preflight(c, args, root)
    manifest, games, labels, spec = pre['manifest'], pre['games'], pre['labels'], pre['rapfi']
    folder = (root / c['output']['raw'].format(mode=args.mode)).resolve()
    require(Path(args.out_dir).resolve() == folder, f'use the isolated output directory {folder}')
    identity = {'format': 'h7r-vt-measure-run-v1', 'experiment_sha256': eid, 'contract_sha256': sha, 'git': state,
                'prior_record': prior, 'hdev': pre['hdev'], 'sample_sha256': c['sample']['sha256'],
                'model': pre['model'], 'runtime': pre['runtime'], 'rapfi': spec['identity'], 'mode': args.mode,
                'device': args.device,
                'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                                'cuda': torch.version.cuda,
                                'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}}
    identity_sha = d4.digest(identity)
    print(f"[vt-preflight] PASS {len(games)} games / {manifest['rows']} rows; identity {identity_sha[:12]}", flush=True)
    specs = model_specs(c, root)
    order = sorted(games)
    folder.mkdir(parents=True, exist_ok=True)
    with file_lock(folder / 'run.lock', nonblocking=True):
        ipath = folder / 'identity.json'
        if ipath.exists():
            require(json.loads(ipath.read_text(encoding='utf-8')) == identity, 'resume identity differs')
        else:
            require(not any((folder / n).exists() for n in ('replay.jsonl', 'move_branches.jsonl',
                                                             'position_branches.jsonl')), 'unbound journals')
            publish_json(ipath, identity)
        paths = {n: folder / f'{n}.jsonl' for n in ('replay', 'move_branches', 'position_branches')}
        replayed = base.read_journal(paths['replay'], 'game_index', identity_sha, set(games))
        chosen = order[:SMOKE['games']] if args.mode == 'smoke' else order
        with pool_of(args) as pool:
            while True:  # smoke: games in index order until V has a disagreement (fixed before any Rapfi result)
                tasks = [{'game': games[i], 'game_index': i, 'specs': specs, 'device': args.device,
                          'identity_sha256': identity_sha} for i in chosen]
                mine = {i: replayed[i] for i in chosen if i in replayed}
                mine = base.execute(pool, replay_chunk, tasks, paths['replay'], lambda r: check_replay(r, games), mine,
                                    'game_index', 'vt-replay', lambda t: len(t['game']['plies']))
                replayed.update(mine)
                records = [replayed[i] for i in chosen]
                part = {**manifest, 'games': [e for e in manifest['games'] if e['game_index'] in set(chosen)]}
                rows = table(records, games, part, labels)
                if args.mode == 'full' or any(r['contest'] and r['move_diff'] for r in rows):
                    break
                require(len(chosen) < SMOKE['max_games'],
                        f"smoke: no S400 / S50 disagreement in V within {SMOKE['max_games']} games")
                chosen = order[:len(chosen) + 1]
            import h7c_sensitivity as sn
            actor = sn.actor_check(records, games, ACTOR)
            require(actor['reproduced'], f'S50 does not reproduce the actor: {actor}')
            if args.mode == 'smoke':
                q1_rows = sorted((r for r in rows if r['contest'] and r['move_diff']),
                                 key=lambda r: (r['game_index'], r['ply']))[:SMOKE['move_observations']]
                pos_rows = sorted((r for r in rows if r['contest']),
                                  key=lambda r: (r['game_index'], r['ply']))[:SMOKE['positions']]
            else:
                q1_rows, pos_rows = rows, rows
            mtable, ptable = move_tasks(q1_rows), position_tasks(pos_rows)
            mt = [{**mtable[k], 'order': i, 'identity_sha256': identity_sha, 'spec': spec}
                  for i, k in enumerate(sorted(mtable))]
            pt = [{**ptable[k], 'order': i, 'identity_sha256': identity_sha, 'spec': spec}
                  for i, k in enumerate(sorted(ptable))]
            mby, pby = {t['key']: t for t in mt}, {t['key']: t for t in pt}
            policy = c['invalid_policy']
            mdone = base.read_journal(paths['move_branches'], 'key', identity_sha, set(mby))
            mdone = base.execute(pool, move_chunk, mt, paths['move_branches'],
                                 lambda r: validate_move(r, mby[r['key']], policy), mdone, 'key', 'vt-move-branches',
                                 lambda _: 1)
            pdone = base.read_journal(paths['position_branches'], 'key', identity_sha, set(pby))
            pdone = base.execute(pool, position_chunk, pt, paths['position_branches'],
                                 lambda r: validate_position(r, pby[r['key']], policy), pdone, 'key',
                                 'vt-position-branches', lambda _: 1)
        # everything again just before publishing: HEAD, contract, prior record, inputs, journals
        require(base.git_state(root) == state, 'HEAD changed during the measurement; nothing is published')
        c2, sha2, _ = load_contract(args.contract, root, args.mode)
        require(sha2 == sha and prior_record(c2, eid, args.mode, root) == prior,
                'the contract or the prior record changed; nothing is published')
        again = preflight(c, args, root)
        require(again['manifest'] == manifest and again['hdev'] == pre['hdev'] and again['model'] == pre['model']
                and again['runtime'] == pre['runtime'] and again['rapfi']['identity'] == spec['identity'],
                'an input changed during the measurement (H-dev, manifest, model, runtime or Rapfi); nothing is published')
        outcomes = {k: validate_move(r, mby[k], policy) for k, r in mdone.items()}
        positions = {k: validate_position(r, pby[k], policy) for k, r in pdone.items()}
        kinds = lambda done: {kind: sum(o['kind'] == kind for o in done.values())  # noqa: E731
                              for kind in ('resolved', 'structural', 'unresolved')}
        result = {'format': 'h7r-vt-measure-summary-v1', 'identity': identity, 'identity_sha256': identity_sha,
                  'log_name': args.log_name, 'actor_check': actor, 'games': chosen if args.mode == 'smoke' else len(chosen),
                  'eligible_plies': len(rows),
                  'warmup_plies': sum(len(r['plies']) for r in records) - len(rows),
                  'classes': class_counts(rows),
                  'branches': {'move': len(mdone), 'position': len(pdone)},
                  'branch_outcomes': {'move': kinds(outcomes), 'position': kinds(positions)},
                  'files': {p.name: d4.file_sha256(p) for p in paths.values() if p.exists()},
                  'seconds': round(time.monotonic() - started, 1),
                  'scope': 'development measurement: no training, Arena, strength, promotion, champion or V9 ruling'}
        if args.mode == 'smoke':
            result['reading'] = 'SMOKE'
            result['projection'] = projection(rows, manifest['rows'])
        else:
            counts = d4.multiplicities(len(order), STATS['samples'], STATS['seed'])
            pick = {'E': lambda r: True, 'T': lambda r: r['tree'], 'V': lambda r: r['contest']}
            result['Q1'] = {name: move_stats([r for r in rows if pick[name](r)], outcomes, order, counts)
                            for name in ('V', 'T', 'E')}
            result['Q2'] = {name: value_stats([r for r in rows if pick[name](r) and r['tree']], positions, order,
                                              counts) for name in ('V', 'T')}
            result['readings'] = {'Q1': result['Q1']['V']['label'], 'Q2': result['Q2']['V']['label']}
            result['reading'] = f"Q1 {result['readings']['Q1']} / Q2 {result['readings']['Q2']}"
            result['entry'] = entry(result['readings']['Q1'], result['readings']['Q2'])
        require(base.git_state(root) == state, 'HEAD changed before publishing; nothing is published')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        publish_json(out_path, result)
        print(json.dumps({'reading': result['reading'], 'summary': str(out_path)}, ensure_ascii=False), flush=True)
        return result


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--contract', default=None)
    sub = p.add_subparsers(dest='command', required=True)
    for cmd in ('prepare', 'run'):
        s = sub.add_parser(cmd)
        s.add_argument('--cache', default='runs/h7r/R1-HDEV-1/cache')
        s.add_argument('--device', default='cuda')
        s.add_argument('--rapfi-dir', required=True)
        s.add_argument('--log-name', default=None, help='the desktop log basename, kept in the record for a recovery')
        if cmd == 'run':
            s.add_argument('--mode', choices=('smoke', 'full'), required=True)
            s.add_argument('--workers', type=int, default=6)
            s.add_argument('--out-dir', required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        out = (prepare if args.command == 'prepare' else run)(args)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', flush=True)
        return 1
    if args.command == 'prepare':
        print(json.dumps({'passed': out['passed'], 'eligible': out['eligible']}, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
