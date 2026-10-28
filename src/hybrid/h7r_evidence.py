"""Validate gate evidence from source records. Status strings alone are never evidence.

Imports of existing script readers/judges are lazy; no candidate factory is used
here. This module never starts an engine, writes the confirmation ledger or changes a
champion: verification is read-only, and the caller binds the ledger afterwards.

Boundary: recorded measurements are trusted as measured. Fresh-probe values, must_block
hits and timing samples are checked for identity, completeness, form and the verdict
they imply, but re-running the models is not part of the offline verification.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from . import h7r_eval as ev

LEGACY_TACTICAL = ('h7c-arena-v1-tactical', 'h7c-nmask-arena-v1-tactical')  # read, never confirmation evidence


def role_models(contract):
    return {role: contract['models'][name]['identity'] for role, name in contract['roles'].items()}


def verified(contract, **result):
    return {'evidence_verified': True, 'models': role_models(contract), **result}


def tactical(record, contract, sha=None, commit=None, confirmation=False):
    """A tactical record's verdict recomputed from its raw checks.

    ``h7r-tactical-v1`` (``scripts/h7r_eval.py tactical``): the contract's SHA and roles, the candidate / champion
    identities. The cycle and NMASK formats stay readable (candidate / parent) but are refused when
    ``confirmation``: a past record is never re-bound to a new contract. ``commit``: the Arena arms' clean commit,
    which the record's code commit must equal."""
    import h7c_arena as ar
    from training.probes import load_probe_set
    from e1_evaluate import summarize, outcome, CLEAR

    ev.require(isinstance(record, dict), 'tactical record is not a JSON object')
    expected_files = {name: ev.file_sha256(path) for name, path in {
        'value_probes': ar.PROBE_FILE, 'fresh_probes': ar.FRESH_PROBES,
        'policy_probes': ar.POLICY_PROBES, 'e1_manifest': ar.E1_MANIFEST}.items()}
    models = role_models(contract)
    fmt = record.get('format')
    if fmt == ev.TACTICAL_FORMAT:
        ev.require(sha is not None and record.get('contract_sha256') == sha
                   and record.get('contract_id') == contract.get('contract_id'), 'tactical record of another contract')
        ev.require(record.get('roles') == {r: contract['roles'][r] for r in ('candidate', 'champion')}
                   and record.get('candidate') == models['candidate'] and record.get('champion') == models['champion'],
                   'tactical record is not of the candidate and champion')
        result_keys = ('candidate_result', 'champion_result')
    else:
        ev.require(not confirmation, f'a confirmation takes only {ev.TACTICAL_FORMAT} records made for its contract '
                                     '(cycle and NMASK records are read, never re-bound)')
        ev.require(fmt in LEGACY_TACTICAL and record.get('candidate') == models['candidate']
                   and record.get('parent') == models['champion'],
                   'tactical record is not of the candidate and champion')
        result_keys = ('candidate_result', 'parent_result')
    ev.require(record.get('rule_version') == contract['gates']['tactical_rule'] == ar.TACTICAL_RULE,
               'tactical rule differs')
    git = record.get('git') or {}
    ev.require(record.get('files') == expected_files and git.get('commit') and git.get('dirty') is False,
               'tactical probes or code provenance differ')
    if confirmation:
        ev.require(commit is not None and git['commit'] == commit,
                   'tactical code commit differs from the Arena arms\' commit')
    fresh_probes = json.loads(ar.FRESH_PROBES.read_text())['probes']
    e1_positions = json.loads(ar.E1_MANIFEST.read_text())['positions']
    policy_probes, policy_sha = load_probe_set(ar.POLICY_PROBES)
    sides = []
    for side in result_keys:
        ev.require(isinstance(record.get(side), dict), f'tactical record has no {side}')
        source = deepcopy(record[side])
        vp = source['value_probes']
        fresh = vp['fresh']
        ev.require(vp['inference'] == ar.FRESH_RULE['inference'] and fresh['file_sha256'] == ar.FRESH_PROBES_SHA256,
                   'wrong tactical inference or fresh probe identity')
        rows = fresh['rows']
        ev.require(len(rows) == len(fresh_probes) and len({r['name'] for r in rows}) == len(rows),
                   'missing or duplicate fresh probes')
        by_name = {r['name']: r for r in rows}
        for probe in fresh_probes:
            row = by_name.get(probe['name'])
            ev.require(row is not None and all(row[k] == probe[k] for k in ('game_key', 'expected_sign', 'kind', 'ply')),
                       'fresh probe source differs')
            ev.require(isinstance(row['value'], (int, float)) and math.isfinite(row['value']), 'invalid probe value')
            row['correct'] = row['value'] * probe['expected_sign'] > 0
        for sign, name in ((1, 'win'), (-1, 'loss')):
            mine = [r for r in rows if r['expected_sign'] == sign]
            fresh[name] = {'games': len({r['game_key'] for r in mine}), 'mean_value': sum(r['value'] for r in mine) / len(mine)}
        recomputed = ar.fresh_rule({'probes': rows, **{name: fresh[name] for name in ('win', 'loss')}})
        source['value_probes']['fresh'] = {**recomputed, 'file_sha256': fresh['file_sha256']}
        block = source['must_block']
        from eval_h3_gate import B400_MUST_BLOCK
        count = sum(p.kind == 'must_block' for p in policy_probes)
        ev.require(block['probe_sha256'] == policy_sha and block['total'] == count
                   and type(block['hits']) is int and 0 <= block['hits'] <= count and block['anchor'] == B400_MUST_BLOCK,
                   'must_block source or count differs')
        e1 = source['e1']
        ev.require(len(e1['rows']) == len(e1_positions)
                   and {r['key'] for r in e1['rows']} == {p['key'] for p in e1_positions}, 'E1 rows incomplete')
        truths = {p['key']: p['truths'] for p in e1_positions}
        ev.require(all(not (set(r['truths_added']) & set(truths[r['key']])) for r in e1['rows']),
                   'E1 added truths overwrite the manifest')
        truths = {r['key']: {**truths[r['key']], **r['truths_added']} for r in e1['rows']}
        pmap = {p['key']: p for p in e1_positions}
        for row in e1['rows']:
            ev.require(row['class'] == pmap[row['key']]['class']
                       and [run['seed'] for run in row['runs']] == pmap[row['key']]['e1_seeds'],
                       'missing or wrong E1 runs')
            for run in row['runs']:
                expected_unsound = [m for m, status in run['checked'] if status == 'UNSAFE'
                                    and truths[row['key']].get(f'{m[0]},{m[1]}', {}).get('status') == CLEAR]
                ev.require(run['outcome'] == outcome(row['class'], truths[row['key']], run)
                           and run['unsound'] == expected_unsound, 'E1 run verdict differs from raw checks')
        e1['metrics'] = summarize(e1['rows'], truths)['metrics']
        sides.append(source)
    result = ar.tactical_judge(*sides)
    ev.require(record.get('judge') == result, 'tactical verdict does not match raw checks')
    return verified(contract, **result, source_format=fmt, contract_sha256=record.get('contract_sha256'),
                    code_commit=git['commit'])


def time_statistics(values):
    ev.require(isinstance(values, list) and values and all(type(v) in (int, float) and math.isfinite(v) and v > 0
                                                         for v in values), 'invalid or empty timing samples')
    ordered = sorted(values)
    return {'mean': sum(values) / len(values), 'p95': ordered[min(len(ordered) - 1, int(.95 * (len(ordered) - 1) + .5))]}


def timing(record, contract, sha, root):
    ev.require(record.get('format') == 'h7r-timing-v2' and record.get('contract_sha256') == sha,
               'timing source format or contract differs (v2 samples required)')
    ev.require((record.get('git') or {}).get('commit') and (record.get('git') or {}).get('dirty') is False,
               'timing must name clean committed code')
    roles = contract['roles']
    names = record.get('comparison')
    ev.require(names == [roles['candidate'], roles['champion']], 'timing must be candidate / champion in that order')
    measured = {}
    for name in names:
        side = record['sides'][name]
        ev.require(side.get('model') == name and side.get('identity') == contract['models'][name]['identity']
                   and side.get('budget') == 1, 'timing model identity or budget differs')
        ev.require(len(side['samples']) == record['positions'], 'timing sample count differs')
        measured[name] = time_statistics(side['samples'])
    specs = [arm['suite']['sha256'] for arm in contract['arms'] if arm['name'] == 'candidate_vs_champion']
    ev.require(record.get('suite_sha256') in specs and record.get('position_ids')
               and len(record['position_ids']) == record['positions']
               and len(set(record['position_ids'])) == record['positions'], 'timing suite or positions differ')
    suite_ref = next(a['suite'] for a in contract['arms'] if a['name'] == 'candidate_vs_champion')
    suite = json.loads((Path(root) / suite_ref['path']).read_text())
    expected_ids = [o['history_id'] for o in suite['openings'][record['warmup']:record['warmup'] + record['positions']]]
    ev.require(record['position_ids'] == expected_ids, 'timing did not measure the contracted positions')
    ev.require(record['warmup'] > 0 and record.get('environment', {}).get('torch_threads') == 1,
               'timing warm-up/threads not controlled')
    ev.require(all(record.get('environment', {}).get(k) for k in ('platform', 'processor', 'cpu_count', 'torch', 'python', 'device')),
               'timing environment incomplete')
    limits = contract['gates']['time']
    ratios = {k: measured[names[0]][k] / measured[names[1]][k] for k in ('mean', 'p95')}
    gate = {'limits': limits, 'pass': all(ratios[k] <= limits[f'{k}_ratio_max'] for k in ratios)}
    ev.require(record.get('gate', {}).get('limits') == limits and record.get('gate', {}).get('pass') is gate['pass'],
               'timing gate does not match samples')
    return verified(contract, ratio=ratios, gate=gate)


# --- selection branches ------------------------------------------------------------------------------------------

def structural_failures(board) -> list[str]:
    """The conditions of a structural ``abnormal_end`` that this final position does not meet (empty: it is one):
    over without a winner, black to move, exactly one empty point, no legal move, that point black-forbidden."""
    from renju import BLACK
    from renju.rules import EMPTY, SIZE, forbidden_reason, legal_black_points

    empty = [(r, c) for r in range(SIZE) for c in range(SIZE) if board.board[r][c] == EMPTY]
    out = []
    if not board.done:
        out.append('the game is not over')
    if board.winner is not None:
        out.append('the game has a winner')
    if board.to_play != BLACK:
        out.append('white is to move')
    if len(empty) != 1:
        out.append(f'{len(empty)} empty points, not one')
    if legal_black_points(board.board):
        out.append('black has a legal move')
    if len(empty) == 1 and not forbidden_reason(board.board, *empty[0]):
        out.append('the last empty point is not black-forbidden')
    return out


def _replay_branch(result: dict, task: dict):
    from renju import Game, IllegalMove

    ev.require(result.get('ply') == len(task['moves']) and result.get('move') == task['move'], 'wrong fixed branch')
    continuation = result.get('continuation')
    ev.require(isinstance(continuation, list) and all(
        isinstance(m, list) and len(m) == 2 and all(type(x) is int and 0 <= x < 15 for x in m) for m in continuation),
        'branch continuation is not a move list')
    board = Game()
    try:
        for move in task['moves']:
            board.play(*move)
        mover = board.to_play
        for move in [task['move'], *continuation]:
            board.play(*move)
    except IllegalMove as exc:
        raise ValueError(f"branch {result['key'][:12]}: its replay has an illegal move ({exc})") from exc
    ev.require(result.get('length') == len(board.history), 'branch length differs from its replay')
    return board, mover


def branch_outcome(result: dict, task: dict, policy: dict | None) -> dict:
    """One branch record replayed (prefix, fixed move, continuation): ``{'kind', 'score', 'reason'}`` for
    ``h7r_eval.policy_selection``. Data errors (refused, never unresolved): a record its replay contradicts, an
    INVALID reason that is no execution failure, an ``abnormal_end`` that is not structural, an execution failure
    after the game was over. Without a policy (the strict rule) every INVALID branch is unresolved."""
    import h7c_adjudicate as ad
    from renju import BLACK

    board, mover = _replay_branch(result, task)
    status, reason = result.get('status'), result.get('reason')
    if status == 'VALID':
        score = ad.score_for(mover, board)
        winner = None if board.winner is None else 'black' if board.winner == BLACK else 'white'
        ev.require(board.done and (board.winner is not None or len(board.history) == 225)
                   and result.get('winner') == winner and result.get('score') == score,
                   'adjudication outcome differs from replay')
        return {'kind': 'resolved', 'score': score, 'reason': None}
    ev.require(status == 'INVALID' and result.get('score') is None, 'branch status / score malformed')
    if reason == 'abnormal_end':
        failures = structural_failures(board)
        ev.require(not failures, f"branch {result['key'][:12]}: abnormal_end is not structural ({'; '.join(failures)})")
        if policy is not None and policy['structural_abnormal_end'] == 'evaluation_draw':
            return {'kind': 'structural', 'score': 0.5, 'reason': reason}
        return {'kind': 'unresolved', 'score': None, 'reason': reason}
    ev.require(reason in ev.EXECUTION_FAILURES,
               f'INVALID reason {reason!r} is a data error in a confirmation, not an execution failure')
    ev.require(not board.done, f'{reason} recorded after the game was over')
    if reason == 'engine_illegal':
        move = result.get('move_at')
        ev.require(isinstance(move, list) and len(move) == 2 and tuple(move) not in set(board.legal_moves()),
                   'engine_illegal names a legal move')
    elif reason == 'forbid_mismatch':
        from agents.external_engine import our_forbidden_points
        ours = our_forbidden_points(board)
        mine, theirs = ({tuple(m) for m in result.get(k) or []} for k in ('ours_only', 'engine_only'))
        ev.require(board.to_play == BLACK and result.get('ply_at') == len(board.history) and (mine or theirs)
                   and mine <= ours and not theirs & ours, 'forbid_mismatch differs from our forbidden points')
    handled = policy is None or reason in policy['unresolved_reasons']
    return {'kind': 'unresolved' if handled else 'unhandled', 'score': None, 'reason': reason}


def selection(path, contract, sha, root):
    """Rebuild deltas from complete holdout sensitivity and replayed Rapfi branch records.

    The manifest references a sealed H-conf cache, sensitivity JSONL and branch
    JSONL. Branch identities bind their source hashes and the registered engine;
    all observations are derived, never selected from a user-supplied delta list.
    INVALID branches follow ``selection.invalid_policy`` (``h7r_eval.policy_selection``)
    or, without one, the strict rule (any INVALID branch: INCONCLUSIVE).
    Read-only: an H-conf already bound to another confirmation is refused here, and
    the caller binds it.
    """
    import h7r_holdout as ho
    import h7c_labels as lab
    import h7c_adjudicate as ad
    from renju import Game
    from .h7r_ledger import conflicts

    path = Path(path)
    body = json.loads(path.read_text())
    ev.require(body.get('format') == 'h7r-selection-evidence-v1' and body.get('contract_sha256') == sha,
               'selection evidence contract/format differs')
    base = path.parent
    cache_binding = ho.read_ref(body['holdout_binding'], base)
    hc, hsha, games, _ = ho.checked_cache(cache_binding.parent, root)
    ev.require(hc['holdout']['split'] == 'conf' and hsha == contract['selection'].get('holdout_contract_sha256'),
               'selection requires the contracted H-conf, not development/training data')
    clash = conflicts(root, [('holdout-use', hsha, {'confirmation': sha})])
    ev.require(not clash, f'this H-conf is already bound to another confirmation ({clash})')
    engine = contract['selection'].get('engine')
    ev.require(isinstance(engine, dict) and engine.get('registered_engine') is True
               and engine.get('registered') == engine.get('actual') and body.get('engine') == engine,
               'selection engine is not the contracted registered Rapfi')
    models = role_models(contract)
    ev.require(hc['models'][hc['holdout']['actor']]['identity'] == models['champion'], 'holdout actor is not champion')
    spath = ho.read_ref(body['sensitivity'], base)
    bpath = ho.read_ref(body['branches'], base)
    expected_identity = {'format': 'h7r-selection-source-v1', 'contract_sha256': sha,
                         'models': {k: models[k] for k in ('candidate', 'champion')}, 'git': body.get('source_git'),
                         'holdout_contract_sha256': hsha,
                         'selfplay_sha256': ho.file_ref(ho.read_ref(json.loads(cache_binding.read_text())['games'],
                                                                  cache_binding.parent), base)['sha256']}
    ev.require(isinstance(body.get('source_git'), dict) and body['source_git'].get('commit')
               and body['source_git'].get('dirty') is False, 'selection source must name clean committed code')
    sources = [json.loads(line) for line in spath.read_text().splitlines() if line.strip()]
    by_index = {g['game_index']: g for g in games}
    ev.require(len(sources) == len(games) and {r['game_index'] for r in sources} == set(by_index),
               'selection sensitivity has missing/duplicate/extra source games')
    observations = []
    for source in sources:
        ev.require(source.get('identity') == expected_identity, 'selection sensitivity identity differs')
        import h7c_sensitivity as sn
        sn.check_replay(source, by_index, {'candidate': None, 'champion': None}, 'champion')
        original = by_index[source['game_index']]
        ev.require(len(source['plies']) == len(original['plies']), 'sensitivity plies incomplete')
        for p, old in zip(source['plies'], original['plies']):
            ev.require((p['ply'], p['color']) == (old['ply'], old['color']) and set(p['sets']) == {'candidate', 'champion'},
                       'sensitivity position/model sets differ')
            ma, mb = (p['sets'][k]['engine_move'] for k in ('candidate', 'champion'))
            ev.require(mb == old['engine_move'], 'champion actor move not reproduced')
            prefix = original['moves'][:p['ply']]
            position = Game()
            for move in prefix:
                position.play(*move)
            legal = set(position.legal_moves())
            ev.require(tuple(ma) in legal and tuple(mb) in legal, 'illegal sensitivity choice')
            if ma != mb:
                observations.append({'game': lab.game_id(original), 'ply': p['ply'], 'moves': prefix,
                                     'move_a': ma, 'move_b': mb})
    branch_identity = {'source': expected_identity, 'sensitivity_sha256': ev.file_sha256(spath),
                       'holdout_binding_sha256': ev.file_sha256(cache_binding), 'engine': engine}
    identity_sha = ad.sha_json(branch_identity)
    results = [json.loads(line) for line in bpath.read_text().splitlines() if line.strip()]
    table = ad.branch_table(observations)
    ev.require(len(results) == len(table) and {r.get('key') for r in results} == set(table),
               'missing/duplicate/extra adjudication branches')
    spec = contract['selection']
    policy = spec.get('invalid_policy')
    outcomes = {}
    for result in results:
        ev.require(result.get('identity_sha256') == identity_sha, 'adjudication identity differs')
        outcomes[result['key']] = branch_outcome(result, table[result['key']], policy)
    keyed = [{'game': o['game'], **{f'key_{s}': ad.branch_key(o['moves'], o[f'move_{s}']) for s in ('a', 'b')}}
             for o in observations]
    if policy is None:
        rows = [{'game': o['game'], 'delta': outcomes[o['key_a']]['score'] - outcomes[o['key_b']]['score']}
                for o in keyed if outcomes[o['key_a']]['kind'] == outcomes[o['key_b']]['kind'] == 'resolved']
        invalid = len(keyed) - len(rows)
        result = ev.cluster_noninferiority(rows, spec['margin'], spec['alpha'], contract['stats']['bootstrap'],
                                         spec.get('min_observations', 1))
        # No outcome-dependent dropping: without a policy, invalid branches give no confirmation certificate.
        if invalid:
            result.update(status='INCONCLUSIVE', reason='INVALID_BRANCHES')
        result.update(invalid_policy=None, observations_total=len(keyed), invalid=invalid)
    else:
        result = ev.policy_selection(keyed, outcomes, policy, spec, contract['stats']['bootstrap'])
        result['invalid'] = result['unresolved_observations']
    return verified(contract, **result, contract_sha256=sha, holdout_contract_sha256=hsha,
                    evidence_sha256=ev.file_sha256(path))
