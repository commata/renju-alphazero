"""H7-R R1 evaluation tools (docs/h7-rl-plan.md): opening suites, contract Arenas, sizing, the time gate, the
selection-quality record and the promotion judgment v3. Measurement only: nothing here trains, and nothing changes a
champion, the canonical engine or the web model; a judgment is a record.

    suite      build an opening suite file (``hybrid.h7r_openings``: O0 / O1 / O2), with its overlap with earlier
               evaluations' openings (suite files or Arena records)
    play       one arm of a contract: every opening of the arm twice (colours swapped), games appended as they finish
               (resume; another setup refused); the games are the cycle Arena's (``h7c_arena.play_game``), each
               side TB-V3 / fast5 with its model pair at its budget (``h7c_budget.make_budget_agent``: 1 = S50,
               8 = S400)
    summarize  an arm record re-checked (signature, contract, suite, every opening once per colour) and its summary
    size       a sizing proposal from pilot arm records (pairs for delta / power, the chance of m informative pairs,
               the cost); it starts nothing
    timing     the time gate: the same positions, one process, alternating order, warm-up discarded
    selection-source  complete H-conf sensitivity and registered Rapfi branches, with resumable raw records
    selection  confirmation from bound source evidence; delta rows are development-only
    tactical   the tactical gate v2 of a confirm contract's candidate and champion (``h7c_arena`` probes, must_block
               and E1 on the contract's model files; no cycle lineage asked)
    judge      promotion rule v3; every evidence record required, verified before the ledger is bound; identical
               audit recomputation allowed, changed inputs refused

A pilot contract or a confirm contract is played only when ``status`` is ``fixed`` and the file is committed; a
confirm contract also needs every rule value set. A dirty tree is refused (``--allow-dirty`` only for tests).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid import h7r_eval as ev  # noqa: E402
from hybrid import h7r_openings as op  # noqa: E402
from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import BLACK, WHITE, Game  # noqa: E402

ARENA_FORMAT = 'h7r-arena-v1'
TIMING_FORMAT = 'h7r-timing-v2'
SELECTION_FORMAT = 'h7r-selection-v1'
TACTICAL_FORMAT = ev.TACTICAL_FORMAT
VERDICT_FORMAT = 'h7r-verdict-v1'


def _git(allow_dirty: bool) -> dict:
    from h7c_arena import _git as state

    git = state()
    ev.require(git['dirty'] is False or allow_dirty, 'the tree is dirty: every H7-R record names its code commit')
    return git


def _json(path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _contract(args, root: Path = ROOT, playing: bool = False) -> tuple[dict, str]:
    from hybrid.h7c_nmask import committed

    root = Path(getattr(args, 'root', None) or root)  # tests: a repository-shaped temporary tree
    path = Path(args.contract)
    contract, sha = ev.load_contract(path, root, check_files=playing, contracts_dir=path.parent)
    if playing:
        ev.require(contract['status'] == 'fixed', f'{path}: a draft contract is not played')
        ev.require(args.allow_dirty or committed(path, root), f'{path}: commit the contract before playing it')
        if contract['purpose'] == 'confirm':
            ev.require(not ev.unset(contract), f'{path}: confirm rule values unset: {ev.unset(contract)}')
    return contract, sha


def _arm(contract: dict, name: str) -> dict:
    arms = {a['name']: a for a in contract['arms']}
    ev.require(name in arms, f"no arm {name} in {contract['contract_id']}")
    return arms[name]


def _suite(arm: dict, root: Path = ROOT) -> dict:
    body = _json(root / arm['suite']['path'])
    op.check_suite(body)
    return body


# --- suite -------------------------------------------------------------------------------------------------------

def previous_openings(paths) -> dict[str, list]:
    out = {}
    for path in paths or []:
        body = _json(path)
        if body.get('format') == op.FORMAT:
            out[str(path)] = [o['moves'] for o in body['openings']]
        else:  # an Arena record (h7c_arena / h7c_nmask_arena / this tool): its openings
            out[str(path)] = [o['moves'] for o in body.get('openings', [])]
    return out


def suite(args, log=print) -> dict:
    git = _git(args.allow_dirty)
    body = op.build_suite(args.name, args.count, args.seed)
    body['git'] = git
    body['overlap_with_earlier'] = op.overlap(body['openings'], previous_openings(args.compare_with))
    log(f"[h7r-suite] {args.name}: {body['count']} openings, distinct {body['distinct']}, rejected "
        f"{body['rejected']}", flush=True)
    return body


# --- play --------------------------------------------------------------------------------------------------------

def arm_identity(contract: dict, sha: str, arm: dict, device: str, git: dict) -> dict:
    from analysis import tb_v3, tb_v3_fast5

    side = lambda s: {'model': arm[s]['model'], 'budget': arm[s]['budget'],  # noqa: E731
                      'identity': contract['models'][arm[s]['model']]['identity']}
    return {'format': ARENA_FORMAT, 'contract_id': contract['contract_id'], 'contract_sha256': sha,
            'purpose': contract['purpose'], 'arm': arm['name'], 'pairing': arm.get('pairing'),
            'tested': side('tested'), 'opponent': side('opponent'), 'pairs': arm['pairs'], 'seed': arm['seed'],
            'suite_sha256': arm['suite']['sha256'], 'engine_id': tb_v3.ENGINE_ID, 'engine_hash': tb_v3.ENGINE_HASH,
            'runtime_hash': tb_v3_fast5.runtime_hash(device), 'value_device': device.split(':')[0], 'git': git}


def build_tasks(arm: dict, openings: list[dict]) -> list[dict]:
    from run_mcts_v8_benchmark import derive_seed

    tasks = []
    for i, o in enumerate(openings):
        for color in (BLACK, WHITE):
            name = 'black' if color == BLACK else 'white'
            tasks.append({'key': f"{arm['name']}/{o['history_id'][:20]}/{name}", 'arm': 'official', 'pair': i,
                          'seed': derive_seed(arm['seed'], 'h7r-game', i, color), 'tested_color': color,
                          'opening': o['moves'], 'opening_id': o['history_id']})
    return tasks


def _play_chunk(tasks: list[dict]) -> list[dict]:
    """In a worker: the cycle Arena's game with each side at its budget."""
    import h7c_arena as ar
    import h7c_budget as sb

    out = []
    for t in tasks:
        ctx = t['context']
        tested, opponent = ctx['sides']['tested'], ctx['sides']['opponent']
        make = lambda side, seed: sb.make_budget_agent(side['budget'], side['policy'], side['value'],  # noqa: E731
                                                       seed, ctx['device'])
        game = ar.play_game(t, make_tested=lambda _, seed: make(tested, seed),
                            make_opponent=lambda seed: make(opponent, seed), max_plies=t.get('max_plies'))
        out.append({**game, 'arm': ctx['arm'], 'opening_id': t['opening_id']})
    return out


def play(args, log=print, root: Path = ROOT) -> dict:
    import h7c_arena as ar

    git = _git(args.allow_dirty)
    contract, sha = _contract(args, root, playing=True)
    arm = _arm(contract, args.arm)
    openings = op.pick(_suite(arm, root), arm['pairs'], arm['seed'])
    identity = arm_identity(contract, sha, arm, args.device, git)
    sig = _sha_json(identity)
    if contract['purpose'] == 'confirm':
        from hybrid.h7r_ledger import bind_contract
        bind_contract(root, contract, sha, args.allow_dirty)
    tasks = build_tasks(arm, openings)
    finished = ar.load_finished(Path(args.jsonl), {t['key'] for t in tasks}, sig, log=log)
    check_games(list(finished.values()), tasks, arm['name'], sig)
    side = lambda s: {'policy': str(root / contract['models'][arm[s]['model']]['policy']),  # noqa: E731
                      'value': str(root / contract['models'][arm[s]['model']]['value']), 'budget': arm[s]['budget']}
    sides = {'tested': side('tested'), 'opponent': side('opponent')}
    context = {'models': {'candidate': sides['tested'], 'parent': sides['opponent']}, 'sides': sides,
               'device': args.device, 'signature': sig, 'arm': arm['name']}
    todo = [{**t, 'context': context, **({'max_plies': args.max_plies} if getattr(args, 'max_plies', None) else {})}
            for t in tasks if t['key'] not in finished]
    log(f"[h7r-play] {contract['contract_id']} {arm['name']}: {len(tasks)} games ({len(finished)} done), "
        f"workers {args.workers}, signature {sig[:12]}", flush=True)
    Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if todo:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        from multiprocessing import get_context

        # spawn: a forked worker cannot re-initialise CUDA; threads only for tests (in-process fakes)
        pool_of = (lambda: ThreadPoolExecutor(max_workers=1)) if getattr(args, 'threads_for_tests', False) else (
            lambda: ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')))
        with Path(args.jsonl).open('a', encoding='utf-8') as out, pool_of() as pool:
            def keep(_, rows):
                check_games(rows, tasks, arm['name'], sig)
                for row in rows:
                    out.write(json.dumps(row, separators=(',', ':')) + '\n')
                    finished[row['key']] = row
                out.flush()
            run_chunks(pool, _play_chunk, [[t] for t in todo], Progress('h7r-play', len(todo), 'games'), len,
                       unit_name='games', on_result=keep)
    games = [finished[t['key']] for t in tasks]
    record = {'format': f'{ARENA_FORMAT}-arm', 'identity': identity, 'signature': sig, 'openings': openings,
              'wall_seconds': round(perf_counter() - started, 1), 'workers': args.workers,
              'games_played_now': len(todo), 'games': games}
    record['summary'] = ev.arm_summary(ev.pair_table(games, openings), contract['stats'])
    return record


def check_games(games, tasks, arm_name, signature):
    wanted = {task['key']: task for task in tasks}
    seen = set()
    for game in games:
        task = wanted.get(game.get('key'))
        ev.require(task is not None and game['key'] not in seen, 'unexpected or duplicate game key')
        seen.add(game['key'])
        color = 'black' if task['tested_color'] == BLACK else 'white'
        ev.require((game.get('signature'), game.get('arm'), game.get('pair'), game.get('seed'),
                    game.get('tested_color'), game.get('opening_id')) ==
                   (signature, arm_name, task['pair'], task['seed'], color, task['opening_id']),
                   'game task identity differs (seed, pair, colour, opening or signature)')
        ev.require(not game.get('truncated'), 'truncated game')
        ev.validate_game(game, task['opening'])


def load_arm(path, contract: dict, sha: str, root: Path = ROOT) -> dict:
    """An arm record of this contract, re-checked; its pair table and recomputed summary."""
    record = _json(path)
    ev.require(isinstance(record, dict) and record.get('format') == f'{ARENA_FORMAT}-arm'
               and isinstance(record.get('identity'), dict) and isinstance(record.get('games'), list)
               and isinstance(record.get('openings'), list), f'{path}: not an {ARENA_FORMAT} arm record')
    identity = record['identity']
    arm = _arm(contract, identity.get('arm'))
    ev.require(identity.get('contract_sha256') == sha and _sha_json(identity) == record.get('signature'),
               f'{path}: not an arm record of this contract')
    ev.require(all(isinstance(g, dict) for g in record['games']), f'{path}: a game is not a JSON object')
    ev.require(all(g['signature'] == record['signature'] and g['arm'] == arm['name'] for g in record['games']),
               f'{path}: a game of another setup or arm')
    ev.require((identity['tested']['model'], identity['tested']['budget'], identity['opponent']['model'],
                identity['opponent']['budget'], identity['pairs'], identity['seed'], identity['suite_sha256']) == (
        arm['tested']['model'], arm['tested']['budget'], arm['opponent']['model'], arm['opponent']['budget'],
        arm['pairs'], arm['seed'], arm['suite']['sha256']), f'{path}: another model, budget, sample or suite')
    openings = op.pick(_suite(arm, root), arm['pairs'], arm['seed'])
    ev.require([o['history_id'] for o in record['openings']] == [o['history_id'] for o in openings],
               f'{path}: other openings than the arm\'s')
    expected = arm_identity(contract, sha, arm, identity['value_device'], identity['git'])
    ev.require(expected == identity, 'arm engine/runtime/model identity differs')
    ev.require(record['openings'] == openings, 'arm opening payload differs')
    check_games(record['games'], build_tasks(arm, openings), arm['name'], record['signature'])
    table = ev.pair_table(record['games'], openings)
    summary = ev.arm_summary(table, contract['stats'])
    ev.require(summary == record['summary'], f'{path}: its summary is not the one its games give')
    return {'record': record, 'table': table, 'summary': summary}


def summarize(args) -> dict:
    contract, sha = _contract(args)
    arm = load_arm(args.arm_record, contract, sha, Path(getattr(args, 'root', None) or ROOT))
    return {'contract_id': contract['contract_id'], 'arm': arm['record']['identity']['arm'], **arm['summary']}


# --- size --------------------------------------------------------------------------------------------------------

def size(args) -> dict:
    contract, sha = _contract(args)
    stats = contract['stats']
    out = {'format': 'h7r-sizing-v1', 'contract_id': contract['contract_id'], 'contract_sha256': sha,
           'delta': args.delta, 'power': args.power, 'm_candidates': args.m, 'arms': {}}
    for path in args.arm_record:
        arm = load_arm(path, contract, sha, Path(getattr(args, 'root', None) or ROOT))
        rec = arm['record']
        pairs = arm['summary']['pairs']
        per_pair = rec['wall_seconds'] / max(1, len(rec['games']) // 2) if rec.get('games_played_now') == len(
            rec['games']) else None  # wall time only of a run that played the whole arm
        worker = sum(g['game_seconds'] for g in rec['games']) / max(1, pairs)
        result = ev.sizing(arm['table']['pairs'], stats, args.delta, args.power, args.m, per_pair)
        out['arms'][rec['identity']['arm']] = {**result, 'workers': rec.get('workers'),
                                                'worker_seconds_per_pair': round(worker, 1)}
    return out


# --- timing ------------------------------------------------------------------------------------------------------

def environment(device: str) -> dict:
    import torch

    return {'platform': platform.platform(), 'processor': platform.processor() or platform.machine(),
            'cpu_count': os.cpu_count(), 'torch': torch.__version__, 'torch_threads': torch.get_num_threads(),
            'python': platform.python_version(), 'device': device,
            'cuda': torch.cuda.get_device_name(0) if device.startswith('cuda') and torch.cuda.is_available() else None}


def timing(args, log=print, root: Path = ROOT) -> dict:
    """Seconds per move of each side on the same positions, in one process: for every position a fresh agent of
    each side (the E4-v2 cooldown is per agent), the order alternating; ``warmup`` positions discarded."""
    import h7c_budget as sb
    from random import Random

    git = _git(args.allow_dirty)
    contract, sha = _contract(args, root)
    sides = {}
    for spec in args.side:
        name, budget = spec.split(':') if ':' in spec else (spec, '1')
        m = contract['models'][name]
        sides[spec] = {'policy': str(root / m['policy']), 'value': str(root / m['value']), 'budget': int(budget)}
    ev.require(len(sides) == 2, 'timing compares two sides: --side CANDIDATE[:budget] --side CHAMPION[:budget]')
    ev.require(args.positions > 0 and args.warmup > 0 and args.bootstrap_samples > 0, 'timing needs positive sample counts and warm-up')
    for name, side in sides.items():
        ev.require(ev.model_identity(side['policy'], side['value']) == contract['models'][name.split(':')[0]]['identity'],
                   'timing model files differ')
    if contract['purpose'] == 'confirm':
        ev.require([(n.split(':')[0], side['budget']) for n, side in sides.items()] ==
                   [(contract['roles']['candidate'], 1), (contract['roles']['champion'], 1)],
                   'confirmation timing must be candidate / champion at S50')
        sides = {n.split(':')[0]: side for n, side in sides.items()}
    suite_body = _json(args.suite)
    op.check_suite(suite_body)
    ev.require(suite_body['count'] >= args.warmup + args.positions, 'timing suite too small')
    positions = [o['moves'] for o in suite_body['openings'][:args.warmup + args.positions]]
    names = list(sides)
    seconds = {n: [] for n in names}
    rng = Random(args.seed)
    for i, moves in enumerate(positions):
        order = names if i % 2 == 0 else names[::-1]
        for n in order:
            agent = sb.make_budget_agent(sides[n]['budget'], sides[n]['policy'], sides[n]['value'],
                                         rng.randrange(1 << 30), args.device)
            game = Game()
            for m in moves:
                game.play(*m)
            started = perf_counter()
            agent.select_move(game)
            if i >= args.warmup:
                seconds[n].append(perf_counter() - started)
        if i == args.warmup - 1:
            log(f'[h7r-timing] warm-up done ({args.warmup} positions)', flush=True)
    a, b = names

    def stat(values):
        s = sorted(values)
        return {'mean': sum(s) / len(s), 'p95': s[min(len(s) - 1, int(0.95 * (len(s) - 1) + 0.5))]}

    ratios = {k: stat(seconds[a])[k] / stat(seconds[b])[k] for k in ('mean', 'p95')}
    draws = []
    for _ in range(args.bootstrap_samples):
        idx = [rng.randrange(len(seconds[a])) for _ in seconds[a]]
        draws.append(stat([seconds[a][j] for j in idx])['mean'] / stat([seconds[b][j] for j in idx])['mean'])
    draws.sort()
    limits = (contract.get('gates') or {}).get('time') or {'mean_ratio_max': args.mean_ratio_max,
                                                           'p95_ratio_max': args.p95_ratio_max}
    return {'format': TIMING_FORMAT, 'contract_id': contract['contract_id'], 'contract_sha256': sha, 'git': git,
            'environment': environment(args.device), 'suite_sha256': ev.file_sha256(args.suite),
            'positions': len(seconds[a]), 'warmup': args.warmup,
            'comparison': names, 'position_ids': [o['history_id'] for o in suite_body['openings'][args.warmup:args.warmup + args.positions]], 'order': 'alternating, a fresh agent per move',
            'sides': {n: {**sides[n], 'model': n.split(':')[0], 'identity': contract['models'][n.split(':')[0]]['identity'],
                         'samples': seconds[n], **{k: round(v, 5) for k, v in stat(seconds[n]).items()}} for n in names},
            'ratio': {k: round(v, 4) for k, v in ratios.items()},
            'mean_ratio_ci95': [round(draws[int(0.025 * (len(draws) - 1))], 4),
                                round(draws[int(0.975 * (len(draws) - 1))], 4)],
            'gate': {'limits': limits, 'pass': ratios['mean'] <= limits['mean_ratio_max']
                     and ratios['p95'] <= limits['p95_ratio_max'],
                     'note': 'provisional limits on the point ratios; the interval is the measurement error'}}


# --- selection and judge -----------------------------------------------------------------------------------------

def _sensitivity_chunk(tasks):
    import h7c_sensitivity as sn
    return [{**sn.replay_game(t['game'], t['models'], t['device']), 'identity': t['identity']} for t in tasks]


def _branch_chunk(tasks):
    import h7c_adjudicate as ad
    return [ad.play_branch(task) for task in tasks]


def selection_source(args, log=print, root=ROOT):
    """Produce bound selection evidence using the existing sensitivity/Rapfi runners, not arbitrary deltas."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from multiprocessing import get_context
    import h7r_holdout as ho
    import h7c_sensitivity as sn
    import h7c_adjudicate as ad
    import h7c_labels as lab
    from hybrid.h7r_ledger import bind_all, contract_bindings, guard
    import h7c_selfplay as sp

    root = Path(getattr(args, 'root', None) or root)
    guard(root, args.allow_dirty)
    git = _git(args.allow_dirty)
    contract, sha = _contract(args, root, playing=True)
    ev.require(contract['purpose'] == 'confirm', 'selection-source requires a confirmation contract')
    hc, hsha, games, binding = ho.checked_cache(args.cache, root)
    ev.require(hc['holdout']['split'] == 'conf' and hsha == contract['selection'].get('holdout_contract_sha256'),
               'source requires the contracted H-conf')
    models = {role: contract['models'][contract['roles'][role]] for role in ('candidate', 'champion')}
    ev.require(hc['models'][hc['holdout']['actor']]['identity'] == models['champion']['identity'], 'holdout actor differs')
    ev.require(all(g['value_device'] == args.device.split(':')[0] for g in games), 'holdout device differs')
    spec = ad.engine_spec(args.program, args.rapfi_dir)
    ev.require(spec['identity'] == contract['selection'].get('engine'), 'Rapfi engine differs from contract')
    folder = Path(args.source_dir)
    ev.require(lab._under_data_or_runs(folder), 'selection source records go under data/ or runs/')
    # Every input checked: reserve the H-conf for this confirmation before any evaluation (a resume with the same
    # inputs finds the same entries; another confirmation of this H-conf is refused, the ledger unchanged).
    bind_all(root, [*contract_bindings(contract, sha), ('holdout-use', hsha, {'confirmation': sha})],
             args.allow_dirty)
    folder.mkdir(parents=True, exist_ok=True)
    ident = {'format': 'h7r-selection-source-v1', 'contract_sha256': sha, 'git': git,
             'models': {k: m['identity'] for k, m in models.items()}, 'holdout_contract_sha256': hsha,
             'selfplay_sha256': binding['games']['sha256']}
    by_index = {g['game_index']: g for g in games}
    source = folder / 'sensitivity.jsonl'
    # Hold a source-directory lock across resume, append, and publication. Never run two writers.
    from hybrid.h7r_ledger import file_lock
    with file_lock(folder / '.lock', nonblocking=True):
        done = sp.load_finished(source, {'identity': ident}, log=log)
        for row in done.values():
            sn.check_replay(row, by_index, models, 'champion')
        if source.exists() and source.stat().st_size and not source.read_bytes().endswith(b'\n'):
            with source.open('ab') as stream:
                stream.write(b'\n')
        paths = {name: {k: str(root / model[k]) for k in ('policy', 'value')} for name, model in models.items()}
        tasks = [{'game': g, 'models': paths, 'device': args.device, 'identity': ident}
                 for g in games if g['game_index'] not in done]
        pool_of = (lambda: ThreadPoolExecutor(max_workers=1)) if getattr(args, 'threads_for_tests', False) else (
            lambda: ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')))
        with source.open('a', encoding='utf-8') as stream, pool_of() as pool:
            def keep(_, rows):
                for row in rows:
                    sn.check_replay(row, by_index, models, 'champion')
                    stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                    done[row['game_index']] = row
                stream.flush()
            run_chunks(pool, _sensitivity_chunk, [[t] for t in tasks], Progress('h7r-source', len(tasks), 'games'), len,
                       unit_name='games', on_result=keep)
        ordered = [done[g['game_index']] for g in games]
        observations = ad.observations(ordered, by_index, {'comparisons': [{'pair': 'candidate:champion'}]})
        table = ad.branch_table(observations)
        cache_path = Path(args.cache) / 'holdout.json'
        branch_ident = {'source': ident, 'sensitivity_sha256': ev.file_sha256(source),
                        'holdout_binding_sha256': ev.file_sha256(cache_path), 'engine': spec['identity']}
        bsha = ad.sha_json(branch_ident)
        bpath = folder / 'branches.jsonl'
        finished = ad.load_finished(bpath, bsha, set(table))
        todo = [{**table[key], 'order': i, 'identity_sha256': bsha, 'spec': spec}
                for i, key in enumerate(sorted(table)) if key not in finished]
        with bpath.open('a', encoding='utf-8') as stream, pool_of() as pool:
            def keep_branch(_, rows):
                for row in rows:
                    stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                stream.flush()
            run_chunks(pool, _branch_chunk, [[t] for t in todo], Progress('h7r-rapfi', len(todo), 'branches'), len,
                       unit_name='branches', on_result=keep_branch)
        base = Path(args.output).parent
        return {'format': 'h7r-selection-evidence-v1', 'contract_sha256': sha, 'engine': spec['identity'],
                'source_git': git, 'holdout_binding': ho.file_ref(cache_path, base),
                'sensitivity': ho.file_ref(source, base), 'branches': ho.file_ref(bpath, base)}


def selection(args) -> dict:
    contract, sha = _contract(args)
    from hybrid import h7r_evidence as evidence
    import h7r_holdout as ho
    if contract['purpose'] == 'confirm':
        ev.require(getattr(args, 'evidence', None) is not None and contract['status'] == 'fixed',
                   'confirm selection requires fixed contract and --evidence, never arbitrary delta rows')
        root = Path(getattr(args, 'root', None) or ROOT)
        result = evidence.selection(args.evidence, contract, sha, root)
        base = Path(getattr(args, 'output', args.evidence)).parent
        return {'format': SELECTION_FORMAT, 'contract_id': contract['contract_id'],
                'source': ho.file_ref(args.evidence, base), **result}
    ev.require(getattr(args, 'rows', None) is not None, 'development selection requires --rows')
    rows = [json.loads(x) for x in Path(args.rows).read_text().splitlines() if x.strip()]
    sel = contract.get('selection') or {}
    out = ev.cluster_noninferiority(rows, sel.get('margin'), sel.get('alpha') or contract['stats']['alpha_one_sided'],
                                    contract['stats']['bootstrap'], sel.get('min_observations') or 1)
    return {'format': SELECTION_FORMAT, 'contract_id': contract['contract_id'], 'contract_sha256': sha,
            'development_only': True, 'rows_sha256': ev.file_sha256(args.rows), **out}


EVIDENCE = ('selection', 'tactical', 'timing')


def tactical(args, log=print, root: Path = ROOT) -> dict:
    """The tactical gate v2 of a confirm contract's candidate and champion (``TACTICAL_FORMAT``): the cycle Arena's
    value probes, must_block and E1 suite (``h7c_arena``) on the contract's model files. No cycle lineage or
    ``h7c_stage`` is asked of the candidate: the contract names it by its four SHA-256."""
    import h7c_arena as ar

    root = Path(getattr(args, 'root', None) or root)
    git = _git(args.allow_dirty)
    contract, sha = _contract(args, root, playing=True)  # fixed, committed, rule values set, model files = SHA
    ev.require(contract['purpose'] == 'confirm', 'the tactical record is made for a confirm contract')
    roles = {r: contract['roles'][r] for r in ('candidate', 'champion')}
    sides, identities = {}, {}
    for role, name in roles.items():
        model = contract['models'][name]
        sides[role] = {'policy': str(root / model['policy']), 'value': str(root / model['value'])}
        identities[role] = ev.model_identity(sides[role]['policy'], sides[role]['value'])
        ev.require(identities[role] == model['identity'], f'{role} model files are not its contract identity')
    ev.require(identities['candidate'] != identities['champion'], 'the candidate is the champion')
    out = {'format': TACTICAL_FORMAT, 'contract_id': contract['contract_id'], 'contract_sha256': sha, 'git': git,
           'device': args.device, 'roles': roles, **identities, 'rule_version': ar.TACTICAL_RULE,
           'files': {'value_probes': ev.file_sha256(ar.PROBE_FILE), 'fresh_probes': ev.file_sha256(ar.FRESH_PROBES),
                     'policy_probes': ev.file_sha256(ar.POLICY_PROBES), 'e1_manifest': ev.file_sha256(ar.E1_MANIFEST)}}
    for role, side in sides.items():
        log(f'[h7r-tactical] {role}: value probes, must_block, E1 ({roles[role]})', flush=True)
        out[f'{role}_result'] = {'value_probes': ar.value_probes(side['value'], args.device),
                                 'must_block': ar.must_block(side['policy']),
                                 'e1': ar.e1_suite(side, args.device, args.workers, role, log=log)}
    out['judge'] = ar.tactical_judge(out['candidate_result'], out['champion_result'])
    log(f"[h7r-tactical] verdict {out['judge']['verdict']}", flush=True)
    return out


def judge(args, root: Path = ROOT) -> dict:
    """Promotion rule v3 of a confirm contract. Order: inputs checked, source evidence verified, judgment computed,
    then the ledger bound (contract, seeds, H-conf use, judgment inputs, one lock, every conflict checked first);
    ``main`` publishes. Any refusal before the ledger step leaves the ledger unchanged; an identical recomputation is
    allowed, changed inputs for a judged contract are refused. The judge itself must run from the arms' clean code
    commit, which is recorded in the verdict and bound to this confirmation in the ledger."""
    from hybrid import h7r_evidence as evidence
    from hybrid.h7r_ledger import FOLDER, bind_all, contract_bindings, guard
    from hybrid.h7c_nmask import committed
    import h7r_holdout as ho

    root = Path(getattr(args, 'root', None) or root)
    guard(root, args.allow_dirty)
    judge_git = _git(args.allow_dirty)
    ev.require(judge_git.get('commit') and judge_git.get('dirty') is False,
               'a judgment requires clean committed code (nothing was recorded)')
    contract, sha = _contract(args, root)
    ev.require(contract['purpose'] == 'confirm', 'only a confirm contract is judged')
    ev.require(contract['status'] == 'fixed' and not ev.unset(contract), 'confirm rule values unset or draft')
    ev.require(args.allow_dirty or committed(args.contract, root), 'commit the confirm contract before judging')
    missing = [f'--{name}' for name in EVIDENCE if getattr(args, name, None) is None]
    ev.require(not missing, f'a confirmation judgment needs every evidence record, missing {missing} '
                            '(nothing was recorded)')
    for path in [*args.arm_record, *(getattr(args, name) for name in EVIDENCE)]:
        ev.require(Path(path).is_file(), f'{path}: no such evidence file (nothing was recorded)')
    roles = contract['roles']
    expected = {'candidate_vs_champion': ('candidate', 'champion'), 'candidate_vs_anchor': ('candidate', 'anchor'),
                'champion_vs_anchor': ('champion', 'anchor')}
    arms, arm_files = {}, {}
    for path in args.arm_record:
        arm = load_arm(path, contract, sha, root)
        name = arm['record']['identity']['arm']
        ev.require(name in expected and name not in arms, f'{path}: arm {name} unexpected or twice')
        t, o = expected[name]
        ev.require((arm['record']['identity']['tested']['model'], arm['record']['identity']['opponent']['model'])
                   == (roles[t], roles[o]) and arm['record']['identity']['tested']['budget'] == 1
                   and arm['record']['identity']['opponent']['budget'] == 1, f'{path}: not the {name} arm')
        arms[name], arm_files[name] = arm, Path(path)
    ev.require(set(arms) == set(expected), f'the judgment needs the arms {sorted(expected)}')
    gits = [arm['record']['identity']['git'] for arm in arms.values()]
    commits = {g.get('commit') for g in gits}
    ev.require(len(commits) == 1 and None not in commits, 'the three Arena arms were played from different code commits')
    ev.require(args.allow_dirty or all(g.get('dirty') is False for g in gits), 'an Arena arm names a dirty tree')
    commit = commits.pop()
    ev.require(judge_git['commit'] == commit,
               'judge code commit differs from the Arena arms (nothing was recorded)')
    records = {name: _json(getattr(args, name)) for name in EVIDENCE}
    for name, record in records.items():
        ev.require(isinstance(record, dict) and record.get('contract_sha256') == sha,
                   f'{getattr(args, name)}: a record of another contract')
    tact = evidence.tactical(records['tactical'], contract, sha, commit, confirmation=True)
    time = evidence.timing(records['timing'], contract, sha, root)
    record = records['selection']
    ev.require(record.get('format') == SELECTION_FORMAT and not record.get('development_only'),
               'selection record is not confirm evidence')
    source = ho.read_ref(record.get('source'), Path(args.selection).parent)
    sel = evidence.selection(source, contract, sha, root)
    ev.require(all(record.get(k) == v for k, v in sel.items()), 'selection verdict differs from raw evidence')
    out = ev.judge(contract, arms, sel, tact, time)
    inputs = {name: ev.file_sha256(getattr(args, name)) for name in EVIDENCE}
    inputs['arms'] = {name: ev.file_sha256(path) for name, path in arm_files.items()}
    entries = bind_all(root, [*contract_bindings(contract, sha),
                              ('holdout-use', sel['holdout_contract_sha256'], {'confirmation': sha}),
                              ('judgment', sha, inputs), ('judge-code', sha, judge_git)], args.allow_dirty)
    return {'format': VERDICT_FORMAT, 'contract_id': contract['contract_id'], 'contract_sha256': sha,
            'roles': roles, 'code_commit': commit, 'judge_git': judge_git, 'evidence_records_sha256': inputs,
            'arm_records_sha256': inputs['arms'], **out,
            'ledger': {'folder': FOLDER, 'entries': entries,
                       'scope': 'local to this checkout: keep the folder with the source records '
                                '(a fresh clone or another machine starts empty)'},
            'meaning': 'a record of the promotion rule v3; it changes no champion, canonical engine or web model'}


# --- command line ------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('suite')
    s.add_argument('--name', choices=tuple(op.SUITES), required=True)
    s.add_argument('--count', type=int)
    s.add_argument('--seed', type=int)
    s.add_argument('--compare-with', nargs='*', default=[], help='earlier suite files or Arena records')
    p = sub.add_parser('play')
    p.add_argument('--arm', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--jsonl', type=Path, required=True)
    m = sub.add_parser('summarize')
    m.add_argument('--arm-record', type=Path, required=True)
    z = sub.add_parser('size')
    z.add_argument('--arm-record', type=Path, nargs='+', required=True)
    z.add_argument('--delta', type=float, default=0.03)
    z.add_argument('--power', type=float, default=0.8)
    z.add_argument('--m', type=int, nargs='+', default=[20, 30, 40])
    t = sub.add_parser('timing')
    t.add_argument('--side', action='append', required=True, help='MODEL[:budget] of the contract (twice)')
    t.add_argument('--suite', type=Path, required=True)
    t.add_argument('--positions', type=int, default=60)
    t.add_argument('--warmup', type=int, default=5)
    t.add_argument('--device', default='cuda')
    t.add_argument('--seed', type=int, default=20261028)
    t.add_argument('--bootstrap-samples', type=int, default=2000)
    t.add_argument('--mean-ratio-max', type=float, default=1.10, help='used when the contract sets none')
    t.add_argument('--p95-ratio-max', type=float, default=1.20)
    src = sub.add_parser('selection-source')
    src.add_argument('--cache', type=Path, required=True)
    src.add_argument('--source-dir', type=Path, required=True)
    src.add_argument('--program', type=Path, required=True, help='registered Rapfi program manifest')
    src.add_argument('--rapfi-dir', type=Path, required=True)
    src.add_argument('--device', default='cuda')
    src.add_argument('--workers', type=int, default=4)
    c = sub.add_parser('tactical')
    c.add_argument('--device', default='cuda')
    c.add_argument('--workers', type=int, default=6)
    e = sub.add_parser('selection')
    e.add_argument('--evidence', type=Path, help='source manifest for a confirmation (H-conf + sensitivity + Rapfi branches)')
    e.add_argument('--rows', type=Path, help='JSONL rows {"game": ..., "delta": ...}')
    j = sub.add_parser('judge')
    j.add_argument('--arm-record', type=Path, nargs=3, required=True)
    for name in EVIDENCE:
        j.add_argument('--' + name, type=Path, help='required (checked before anything is recorded)')
    for q in sub.choices.values():
        if q is not s:
            q.add_argument('--contract', type=Path, required=True)
        q.add_argument('--output', type=Path, required=True)
        q.add_argument('--allow-dirty', action='store_true', help='tests only (never the real ledger)')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error(f'{args.output} exists: an H7-R record is never overwritten')
    try:
        out = {'suite': suite, 'play': play, 'summarize': summarize, 'size': size, 'timing': timing,
               'selection': selection, 'selection-source': selection_source, 'tactical': tactical,
               'judge': judge}[args.command](args)
    except (ValueError, FileNotFoundError) as exc:  # input errors; anything else is a bug and keeps its traceback
        parser.error(str(exc))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    from hybrid.h7r_ledger import publish_json
    publish_json(args.output, out)
    brief = out.get('summary') or {k: v for k, v in out.items() if k not in ('openings', 'games', 'arms')}
    print(json.dumps(brief, indent=1, default=str)[:3000], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
