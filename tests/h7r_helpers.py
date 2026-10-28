"""Replayable fixture games and bound evidence; never runs a learning experiment."""
from copy import deepcopy
import json
from pathlib import Path


def winning_moves(opening, winner):
    from renju import Game
    game = Game()
    for move in opening:
        game.play(*move)
    target = iter([[0, c] for c in range(5)])
    quiet = iter([[r, 14] for r in (14, 12, 10, 8, 6, 4)])
    while not game.done:
        game.play(*(next(target) if game.to_play == winner else next(quiet)))
    assert game.winner == winner
    return [list(m) for m in game.history]


def reserve(root, body, sha):
    """The pre-registration ``selection-source`` makes before evaluating (contract, seeds, H-conf use)."""
    from hybrid.h7r_ledger import bind_all, confirmation_registration
    return bind_all(root, confirmation_registration(body, sha))


def timing_record(body, sha, root, commit='test'):
    from hybrid.h7r_evidence import time_statistics
    names = [body['roles']['candidate'], body['roles']['champion']]
    return {'format': 'h7r-timing-v2', 'contract_sha256': sha, 'comparison': names, 'git': {'commit': commit, 'dirty': False},
            'environment': {'torch_threads': 1, 'platform': 'test', 'processor': 'test', 'cpu_count': 1, 'torch': 'test', 'python': 'test', 'device': 'cpu'}, 'warmup': 1, 'positions': 2,
            'position_ids': [o['history_id'] for o in json.loads((Path(root) / body['arms'][0]['suite']['path']).read_text())['openings'][1:3]], 'suite_sha256': body['arms'][0]['suite']['sha256'],
            'sides': {n: {'model': n, 'budget': 1, 'identity': body['models'][n]['identity'],
                          'samples': [1., 1.], **time_statistics([1., 1.])} for n in names},
            'gate': {'pass': True, 'limits': body['gates']['time']}}


def tactical_results():
    """One side's raw tactical results that pass the v2 rule (fixture values, no model run)."""
    import h7c_arena as ar
    from training.probes import load_probe_set
    from eval_h3_gate import B400_MUST_BLOCK
    from e1_evaluate import summarize, outcome
    probes = json.loads(ar.FRESH_PROBES.read_text())['probes']
    rows = [{**{k: p[k] for k in ('name', 'game_key', 'expected_sign', 'kind', 'ply')},
             'value': float(p['expected_sign']), 'correct': True} for p in probes]
    fresh = {'rows': rows, 'file_sha256': ar.FRESH_PROBES_SHA256}
    aggregates = {name: {'games': len({r['game_key'] for r in rows if r['expected_sign'] == sign}),
                         'mean_value': float(sign)} for sign, name in ((1, 'win'), (-1, 'loss'))}
    fresh.update(ar.fresh_rule({'probes': rows, **aggregates}))
    policy_probes, psha = load_probe_set(ar.POLICY_PROBES)
    total = sum(p.kind == 'must_block' for p in policy_probes)
    e1positions = json.loads(ar.E1_MANIFEST.read_text())['positions']
    # Complete zero-error runs with no tactical switch; every contracted seed represented.
    e1rows = []
    for p in e1positions:
        run = {'route': 'tree', 'unsound': [], 'consistent': True, 'vct2_seconds': 0., 'seconds': .1,
               'pre': p['history'][-1], 'played': p['history'][-1], 'switched': False,
               'checked': [], 'children': []}
        run['outcome'] = outcome(p['class'], p['truths'], run)
        e1rows.append({'key': p['key'], 'class': p['class'], 'truths_added': {},
                       'runs': [{**run, 'seed': seed} for seed in p['e1_seeds']]})
    metrics = summarize(e1rows, {p['key']: p['truths'] for p in e1positions})['metrics']
    return {'value_probes': {'device': 'cpu', 'inference': ar.FRESH_RULE['inference'], 'fresh': fresh,
                             'p93_p94': {}},
            'must_block': {'hits': total, 'total': total, 'anchor': B400_MUST_BLOCK, 'probe_sha256': psha},
            'e1': {'rows': e1rows, 'metrics': metrics}}


def tactical_record(body, sha=None, commit='test'):
    """``sha`` given: an ``h7r-tactical-v1`` record of that contract (as ``h7r_eval.py tactical`` writes it);
    otherwise a cycle-format record (candidate / parent), readable but never confirmation evidence."""
    import h7c_arena as ar
    from hybrid.h7r_eval import TACTICAL_FORMAT, file_sha256
    result = tactical_results()
    files = {k: file_sha256(p) for k, p in {'value_probes': ar.PROBE_FILE, 'fresh_probes': ar.FRESH_PROBES,
                                           'policy_probes': ar.POLICY_PROBES, 'e1_manifest': ar.E1_MANIFEST}.items()}
    ident = {r: body['models'][body['roles'][r]]['identity'] for r in ('candidate', 'champion')}
    common = {'git': {'commit': commit, 'dirty': False}, 'rule_version': ar.TACTICAL_RULE, 'files': files,
              'candidate_result': deepcopy(result), 'judge': ar.tactical_judge(result, result)}
    if sha is None:
        return {'format': f'{ar.FORMAT}-tactical', 'candidate': ident['candidate'], 'parent': ident['champion'],
                'parent_result': deepcopy(result), **common}
    return {'format': TACTICAL_FORMAT, 'contract_id': body['contract_id'], 'contract_sha256': sha,
            'roles': {r: body['roles'][r] for r in ('candidate', 'champion')}, 'device': 'cpu', **ident,
            'champion_result': deepcopy(result), **common}


def selection_source(root, body, confirm_sha, candidate_moves=None, branch_records=None, source_commit='test',
                     folder_name='evidence'):
    """Bound selection evidence over 25 H-conf fixture games; the disagreement is at ply 8 (the champion's five).
    ``candidate_moves``: game index -> the candidate's ply-8 move (default [7, 6], also a five); ``branch_records``:
    fixed move (tuple) -> the branch record's fields (default: a VALID five, which only a winning move is)."""
    import h7c_labels as lab
    import h7c_selfplay as sp
    import h7r_holdout as ho
    import h7c_adjudicate as ad
    from hybrid import h7r_eval as ev
    from hybrid.h6_value_cache import build_value_cache
    from tests.test_h7c_labels import game, build
    from tests.test_h7r_holdout import holdout_contract
    models = {role: body['models'][name]['identity'] for role, name in body['roles'].items()}
    hcpath = holdout_contract(root / 'contracts', models={'c5': {**body['models'][body['roles']['champion']]}},
                             holdout={'seed': 8785, 'split': 'conf', 'games': 25, 'cycle': 1, 'actor': 'c5'})
    hc, hsha = ev.load_contract(hcpath, root, check_files=False)
    body['selection']['holdout_contract_sha256'] = hsha
    engine = {'registered_engine': True, 'registered': {'executable_sha256': 'e'*64, 'files': {}, 'info': []},
              'actual': {'executable_sha256': 'e'*64, 'files': {}, 'info': []}, 'rule': 'renju', 'move_timeout': 30}
    body['selection']['engine'] = engine
    # Caller saves this changed contract, then calls again with the final SHA.
    raw = [game(i, **sp.run_identity(1, 8785, models['champion'], 'cuda'), purpose='holdout',
                holdout={'contract_id': hc['contract_id'], 'contract_sha256': hsha, 'split': 'conf'}) for i in range(25)]
    folder = root / folder_name; folder.mkdir(exist_ok=True)
    gpath = folder / 'games.jsonl'
    gpath.write_text(''.join(json.dumps(g)+'\n' for g in raw))
    rows, _, _ = build(raw, evaluation={'check': lambda gs: ho.check_holdout_games(gs, hc, hsha),
                                      'purpose': ho.PURPOSE, 'split': 'test'})
    rpath = folder / 'rows.jsonl.gz'
    lab.write_atomic({rpath: [lab.value_row_dict(r) for r in rows]})
    cache = folder / 'cache'
    if cache.exists():
        import shutil
        shutil.rmtree(cache)
    build_value_cache(rpath, cache, progress_seconds=1e9)
    ho.seal_cache(cache, hcpath, gpath, rpath)
    ident = {'format': 'h7r-selection-source-v1', 'contract_sha256': confirm_sha,
             'models': {k: models[k] for k in ('candidate', 'champion')},
             'git': {'commit': source_commit, 'dirty': False},
             'holdout_contract_sha256': hsha, 'selfplay_sha256': ev.file_sha256(gpath)}
    records, obs = [], []
    for g in raw:
        plies = []
        for p in g['plies']:
            move = (candidate_moves or {}).get(g['game_index'], [7, 6]) if p['ply'] == 8 else p['engine_move']
            plies.append({'ply': p['ply'], 'color': p['color'], 'actor_route': p['route'], 'actor_move': p['engine_move'],
                          'sets': {'candidate': {'engine_move': move, 'route': p['route'], 'visits': p['visits']},
                                   'champion': {'engine_move': p['engine_move'], 'route': p['route'], 'visits': p['visits']}}})
            if p['ply'] == 8:
                obs.append({'moves': g['moves'][:8], 'move_a': move, 'move_b': p['engine_move']})
        records.append({'identity': ident, 'game_index': g['game_index'], 'plies': plies})
    spath = folder / 'sensitivity.jsonl'
    spath.write_text(''.join(json.dumps(r)+'\n' for r in records))
    bid = {'source': ident, 'sensitivity_sha256': ev.file_sha256(spath),
           'holdout_binding_sha256': ev.file_sha256(cache/'holdout.json'), 'engine': engine}
    five = {'status': 'VALID', 'score': 1., 'length': 9, 'winner': 'black', 'continuation': []}
    branches = [{'key': k, 'identity_sha256': ad.sha_json(bid), 'ply': 8, 'move': v['move'],
                 **(branch_records or {}).get(tuple(v['move']), five)} for k, v in ad.branch_table(obs).items()]
    bpath = folder / 'branches.jsonl'
    bpath.write_text(''.join(json.dumps(r)+'\n' for r in branches))
    evidence = {'format': 'h7r-selection-evidence-v1', 'contract_sha256': confirm_sha, 'engine': engine, 'source_git': ident['git'],
                'holdout_binding': ho.file_ref(cache/'holdout.json', folder),
                'sensitivity': ho.file_ref(spath, folder), 'branches': ho.file_ref(bpath, folder)}
    path = folder / 'selection-evidence.json'; path.write_text(json.dumps(evidence))
    return path
