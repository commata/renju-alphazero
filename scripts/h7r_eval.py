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
    selection  selection-quality non-inferiority from per-observation rows (source-game bootstrap)
    judge      the promotion rule v3 on a confirm contract's records (never overwritten, never re-judged)

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
TIMING_FORMAT = 'h7r-timing-v1'
SELECTION_FORMAT = 'h7r-selection-v1'
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
    tasks = build_tasks(arm, openings)
    finished = ar.load_finished(Path(args.jsonl), {t['key'] for t in tasks}, sig, log=log)
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


def load_arm(path, contract: dict, sha: str, root: Path = ROOT) -> dict:
    """An arm record of this contract, re-checked; its pair table and recomputed summary."""
    record = _json(path)
    identity = record['identity']
    arm = _arm(contract, identity['arm'])
    ev.require(record.get('format') == f'{ARENA_FORMAT}-arm' and identity['contract_sha256'] == sha
               and _sha_json(identity) == record['signature'], f'{path}: not an arm record of this contract')
    ev.require(all(g['signature'] == record['signature'] and g['arm'] == arm['name'] for g in record['games']),
               f'{path}: a game of another setup or arm')
    ev.require((identity['tested']['model'], identity['tested']['budget'], identity['opponent']['model'],
                identity['opponent']['budget'], identity['pairs'], identity['seed'], identity['suite_sha256']) == (
        arm['tested']['model'], arm['tested']['budget'], arm['opponent']['model'], arm['opponent']['budget'],
        arm['pairs'], arm['seed'], arm['suite']['sha256']), f'{path}: another model, budget, sample or suite')
    openings = op.pick(_suite(arm, root), arm['pairs'], arm['seed'])
    ev.require([o['history_id'] for o in record['openings']] == [o['history_id'] for o in openings],
               f'{path}: other openings than the arm\'s')
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
    suite_body = _json(args.suite)
    op.check_suite(suite_body)
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
            'positions': len(seconds[a]), 'warmup': args.warmup, 'order': 'alternating, a fresh agent per move',
            'sides': {n: {**sides[n], **{k: round(v, 5) for k, v in stat(seconds[n]).items()}} for n in names},
            'ratio': {k: round(v, 4) for k, v in ratios.items()},
            'mean_ratio_ci95': [round(draws[int(0.025 * (len(draws) - 1))], 4),
                                round(draws[int(0.975 * (len(draws) - 1))], 4)],
            'gate': {'limits': limits, 'pass': ratios['mean'] <= limits['mean_ratio_max']
                     and ratios['p95'] <= limits['p95_ratio_max'],
                     'note': 'provisional limits on the point ratios; the interval is the measurement error'}}


# --- selection and judge -----------------------------------------------------------------------------------------

def selection(args) -> dict:
    contract, sha = _contract(args)
    rows = [json.loads(x) for x in Path(args.rows).read_text(encoding='utf-8').splitlines() if x.strip()]
    sel = contract.get('selection') or {}
    out = ev.cluster_noninferiority(rows, sel.get('margin'), sel.get('alpha') or contract['stats']['alpha_one_sided'],
                                    contract['stats']['bootstrap'], sel.get('min_observations') or 1)
    return {'format': SELECTION_FORMAT, 'contract_id': contract['contract_id'], 'contract_sha256': sha,
            'rows_sha256': ev.file_sha256(args.rows), **out}


def judge(args, root: Path = ROOT) -> dict:
    contract, sha = _contract(args, root)
    ev.require(contract['purpose'] == 'confirm', 'only a confirm contract is judged')
    roles = contract['roles']
    expected = {'candidate_vs_champion': ('candidate', 'champion'), 'candidate_vs_anchor': ('candidate', 'anchor'),
                'champion_vs_anchor': ('champion', 'anchor')}
    arms = {}
    for path in args.arm_record:
        arm = load_arm(path, contract, sha, root)
        name = arm['record']['identity']['arm']
        ev.require(name in expected and name not in arms, f'{path}: arm {name} unexpected or twice')
        t, o = expected[name]
        ev.require((arm['record']['identity']['tested']['model'], arm['record']['identity']['opponent']['model'])
                   == (roles[t], roles[o]) and arm['record']['identity']['tested']['budget'] == 1
                   and arm['record']['identity']['opponent']['budget'] == 1, f'{path}: not the {name} arm')
        arms[name] = arm
    ev.require(set(arms) == set(expected), f'the judgment needs the arms {sorted(expected)}')
    records = {}
    for name in ('selection', 'tactical', 'timing'):
        path = getattr(args, name)
        records[name] = None if path is None else _json(path)
        if records[name] is not None and name in ('selection', 'timing'):
            ev.require(records[name].get('contract_sha256') == sha, f'{path}: a record of another contract')
    tactical = records['tactical']
    tact = None if tactical is None else {'verdict': (tactical.get('judge') or tactical).get('verdict')}
    out = ev.judge(contract, arms, records['selection'], tact, records['timing'])
    return {'format': VERDICT_FORMAT, 'contract_id': contract['contract_id'], 'contract_sha256': sha,
            'roles': roles, 'arm_records_sha256': {n: ev.file_sha256(p) for n, p in zip(
                [_json(p)['identity']['arm'] for p in args.arm_record], args.arm_record)}, **out,
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
    e = sub.add_parser('selection')
    e.add_argument('--rows', type=Path, required=True, help='JSONL rows {"game": ..., "delta": ...}')
    j = sub.add_parser('judge')
    j.add_argument('--arm-record', type=Path, nargs=3, required=True)
    for name in ('selection', 'tactical', 'timing'):
        j.add_argument('--' + name, type=Path)
    for q in sub.choices.values():
        if q is not s:
            q.add_argument('--contract', type=Path, required=True)
        q.add_argument('--output', type=Path, required=True)
        q.add_argument('--allow-dirty', action='store_true', help='tests only')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error(f'{args.output} exists: an H7-R record is never overwritten')
    try:
        out = {'suite': suite, 'play': play, 'summarize': summarize, 'size': size, 'timing': timing,
               'selection': selection, 'judge': judge}[args.command](args)
    except ValueError as exc:
        parser.error(str(exc))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    brief = out.get('summary') or {k: v for k, v in out.items() if k not in ('openings', 'games', 'arms')}
    print(json.dumps(brief, indent=1, default=str)[:3000], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
