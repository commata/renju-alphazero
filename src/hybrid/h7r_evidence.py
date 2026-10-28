"""Validate gate evidence from source records. Status strings alone are never evidence.

Imports of existing script readers/judges are lazy; no candidate factory is used
here. This module never starts an engine or changes a champion.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from . import h7r_eval as ev


def role_models(contract):
    return {role: contract['models'][name]['identity'] for role, name in contract['roles'].items()}


def verified(contract, **result):
    return {'evidence_verified': True, 'models': role_models(contract), **result}


def tactical(record, contract):
    import h7c_arena as ar
    from training.probes import load_probe_set
    from e1_evaluate import summarize, outcome, CLEAR

    expected_files = {name: ev.file_sha256(path) for name, path in {
        'value_probes': ar.PROBE_FILE, 'fresh_probes': ar.FRESH_PROBES,
        'policy_probes': ar.POLICY_PROBES, 'e1_manifest': ar.E1_MANIFEST}.items()}
    models = role_models(contract)
    ev.require(record.get('format') in (f'{ar.FORMAT}-tactical', 'h7c-nmask-arena-v1-tactical')
               and record.get('candidate') == models['candidate'] and record.get('parent') == models['champion'],
               'tactical record is not of the candidate and champion')
    ev.require(record.get('rule_version') == contract['gates']['tactical_rule'] == ar.TACTICAL_RULE,
               'tactical rule differs')
    ev.require(record.get('files') == expected_files and (record.get('git') or {}).get('commit')
               and (record.get('git') or {}).get('dirty') is False,
               'tactical probes or code provenance differ')
    fresh_probes = json.loads(ar.FRESH_PROBES.read_text())['probes']
    e1_positions = json.loads(ar.E1_MANIFEST.read_text())['positions']
    policy_probes, policy_sha = load_probe_set(ar.POLICY_PROBES)
    sides = []
    for side in ('candidate_result', 'parent_result'):
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
    return verified(contract, **result)


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


def selection(path, contract, sha, root):
    """Rebuild deltas from complete holdout sensitivity and replayed Rapfi branch records.

    The manifest references a sealed H-conf cache, sensitivity JSONL and branch
    JSONL. Branch identities bind their source hashes and the registered engine;
    all observations are derived, never selected from a user-supplied delta list.
    """
    import h7r_holdout as ho
    import h7c_labels as lab
    import h7c_adjudicate as ad
    from renju import Game, BLACK

    path = Path(path)
    body = json.loads(path.read_text())
    ev.require(body.get('format') == 'h7r-selection-evidence-v1' and body.get('contract_sha256') == sha,
               'selection evidence contract/format differs')
    base = path.parent
    cache_binding = ho.read_ref(body['holdout_binding'], base)
    hc, hsha, games, _ = ho.checked_cache(cache_binding.parent, root)
    ev.require(hc['holdout']['split'] == 'conf' and hsha == contract['selection'].get('holdout_contract_sha256'),
               'selection requires the contracted H-conf, not development/training data')
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
    ev.require(len(results) == len(table) and {r['key'] for r in results} == set(table),
               'missing/duplicate/extra adjudication branches')
    valid = {}
    for result in results:
        task = table[result['key']]
        ev.require(result.get('identity_sha256') == identity_sha, 'adjudication identity differs')
        if result.get('status') != 'VALID':
            continue
        board = Game()
        for move in task['moves']:
            board.play(*move)
        mover = board.to_play
        ev.require(result['ply'] == len(task['moves']) and result['move'] == task['move'], 'wrong fixed branch')
        for move in [task['move'], *result['continuation']]:
            board.play(*move)
        score = ad.score_for(mover, board)
        winner = None if board.winner is None else 'black' if board.winner == BLACK else 'white'
        ev.require(board.done and (board.winner is not None or len(board.history) == 225) and result['length'] == len(board.history) and result['winner'] == winner
                   and result['score'] == score, 'adjudication outcome differs from replay')
        valid[result['key']] = score
    rows = []
    invalid = 0
    for o in observations:
        ka, kb = (ad.branch_key(o['moves'], o[f'move_{side}']) for side in ('a', 'b'))
        if ka not in valid or kb not in valid:
            invalid += 1
        else:
            rows.append({'game': o['game'], 'delta': valid[ka] - valid[kb]})
    spec = contract['selection']
    result = ev.cluster_noninferiority(rows, spec['margin'], spec['alpha'], contract['stats']['bootstrap'],
                                     spec.get('min_observations', 1))
    # No outcome-dependent dropping: invalid branches cannot provide a confirmation certificate.
    if invalid:
        result.update(status='INCONCLUSIVE', reason='INVALID_BRANCHES')
    from .h7r_ledger import bind
    bind(root, 'holdout-use', hsha, {'confirmation': sha})
    return verified(contract, **result, observations_total=len(observations), invalid=invalid,
                    contract_sha256=sha, evidence_sha256=ev.file_sha256(path))
