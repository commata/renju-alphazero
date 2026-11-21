"""TB-V3-BUDGET-1: with the same canonical models, does a larger search budget within 5.5x the per-move time of S50
beat the current canonical S50 on fresh openings? (docs/h7r/budget/TB-V3-BUDGET-1-draft.md)

Models: the canonical pair (original H3 policy + H6 v3 value, ``analysis.tb_v3.MODELS``) on both sides; only the two
tree budgets of the tested side are scaled (``h7c_budget.make_budget_agent``). Desktop GPU.

    prepare   (approved.prepare) the model files are the canonical ones; the suites are their committed SHA-256 and
              pass ``check_suite``; the confirmation sample (``h7r_openings.pick`` of O1-confirm-main, 300, seed
              9111) is computed and recorded with its classes and their overlap with O1-confirm-anchor and the
              development suites; no contract uses the seeds; an empty runs folder
    timing    (approved.timing) S400, then S200, then S100 against S50 for 20 pairs each on O1-dev-c (a used
              development suite) until one keeps both per-move ratios within 5.5x: mean = mean seconds of the
              tested moves / mean seconds of the S50 moves of the same games, p95 the same with the 95th
              percentiles; every move of every game must carry its time and side (else nothing is selected). The
              scores of these games are not read. The record fixes the selected budget and the confirmation sample
              (opening ids, seed, model SHA-256) before any confirmation game; none fits: END_NO_BUDGET
    confirm   (approved.confirm) the selected budget against S50 on exactly the recorded sample, 300 pairs; the E1
              tactical suite with the budget agent and the S50 agent; the judge (section 6): strength (one-sided
              lower bound of score - 0.5 above 0), no illegal move, the E1 gate, the time cap measured again in
              these games
    verify    a stage record against its files (recomputed from the games and the E1 runs)

Games keep every move's side, budget factor, route and seconds (``play_game``: the cycle Arena's game loop with the
opponent's moves timed too). A pass names a canonical candidate only; it does not say which of the three budgets is
strongest (they never play each other). Promotion is a separate step.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h7r_p1 as p1  # noqa: E402
from h7r_p1 import Guard, committed, digest, file_sha256, require  # noqa: E402

FORMAT = 'h7r-budget-contract-v1'
CONTRACT_ID = 'TB-V3-BUDGET-1'
D = 'docs/h7r/budget/TB-V3-BUDGET-1'
CONTRACT = f'{D}/budget-contract.json'
RUNS = 'runs/h7r/TB-V3-BUDGET-1'
STAGES = ('prepare', 'timing', 'confirm')
FACTORS = (8, 4, 2)  # S400, S200, S100: measured in this order
CAP = 5.5
TIMING_PAIRS = 20
TIMING_SUITE = ('O1-dev-c', 'docs/h7r/suites/O1-dev-c.json')
TIMING_SEEDS = {8: 9112, 4: 9113, 2: 9114}
CONFIRM_SUITE = ('O1-confirm-main', 'docs/h7r/suites/O1-confirm-main.json')
ANCHOR_SUITE = ('O1-confirm-anchor', 'docs/h7r/suites/O1-confirm-anchor.json')
DEV_SUITES = tuple((f'O1-dev-{x}', f'docs/h7r/suites/O1-dev-{x}.json') for x in 'abcd')
CONFIRM_SEED = 9111
PAIRS = 300
WORKERS = 6
MODELS = {'policy': 'runs/h3_policy_64x4/best.pt', 'value': 'runs/h6_value_head_v3/best.pt'}
STATS = {'alpha_one_sided': 0.025,
         'bootstrap': {'samples': 20000, 'seed': 20261221, 'method': 'percentile', 'unit': 'opening pair'}}
E1_GATE = {'unsound_witness': 'the budget agent: exactly 0',
           'rescue_rate': 'not lower than S50', 'veto_recall': 'not lower than S50',
           'false_veto': 'not higher than S50'}
E1_HIGHER_IS_BETTER = ('rescue_rate', 'veto_recall')
E1_LOWER_IS_BETTER = ('false_veto',)
SOURCES = ('scripts/h7r_budget.py', 'scripts/h7c_budget.py', 'scripts/h7c_arena.py', 'scripts/h7r_eval.py',
           'src/hybrid/h7r_eval.py', 'src/hybrid/h7r_openings.py', 'scripts/e1_evaluate.py', 'scripts/h7r_p1.py')
KEYS = {'format', 'contract_id', 'status', 'approved', 'purpose', 'draft', 'models', 'budgets', 'timing', 'confirm',
        'tactical', 'reading', 'seeds', 'stages', 'output', 'sources', 'scope'}


# --- contract -----------------------------------------------------------------------------------------------------

def contract_path(path, root=ROOT) -> Path:
    return Path(path) if path else Path(root) / CONTRACT


def suite_ref(root, name_path) -> dict:
    name, path = name_path
    return {'name': name, 'path': path, 'sha256': file_sha256(Path(root) / path)}


def contract_body(root=ROOT, sources=SOURCES) -> dict:
    from analysis import tb_v3

    root = Path(root)
    return {
        'format': FORMAT, 'contract_id': CONTRACT_ID, 'status': 'fixed', 'approved': {s: False for s in STAGES},
        'purpose': 'confirmation: within 5.5x the per-move time of S50 on the desktop GPU, does the largest budget '
                   'beat the canonical S50 with the same canonical models on fresh openings? A pass names a '
                   'canonical candidate only; no training, no promotion',
        'draft': {'path': f'{D}-draft.md'},
        'models': {'policy': MODELS['policy'], 'value': MODELS['value'], 'identity': dict(tb_v3.MODELS),
                   'note': 'the canonical pair (original H3 + H6 v3) on both sides'},
        'budgets': {'factors_in_order': list(FACTORS), 'names': {str(f): f'S{50 * f}' for f in FACTORS},
                    'opponent_factor': 1, 'scaled': 'simulations and tactical_simulations (h7c_budget.budget)'},
        'timing': {'suite': suite_ref(root, TIMING_SUITE), 'pairs': TIMING_PAIRS,
                   'seeds': {str(f): s for f, s in TIMING_SEEDS.items()}, 'cap': CAP, 'workers': WORKERS,
                   'device': 'cuda',
                   'ratios': {'mean': 'mean seconds of the tested moves / mean seconds of the S50 moves, the same '
                                      'games',
                              'p95': '95th percentile of the tested moves / that of the S50 moves, the same games'},
                   'rule': 'the first of S400, S200, S100 with both ratios <= 5.5 is selected; none: END_NO_BUDGET; '
                           'every move must carry its time and side; the scores are not read'},
        'confirm': {'suite': suite_ref(root, CONFIRM_SUITE), 'pairs': PAIRS, 'seed': CONFIRM_SEED, 'workers': WORKERS,
                    'device': 'cuda', 'anchor_suite': suite_ref(root, ANCHOR_SUITE),
                    'sample_rule': 'h7r_openings.pick(O1-confirm-main, 300, 9111), fixed in the timing record before '
                                   'any confirmation game; a resumed confirmation plays only that sample'},
        'tactical': {'suite': 'the E1 dev manifest (h7c_arena.E1_MANIFEST), the budget agent and the S50 agent',
                     'gate': E1_GATE,
                     'comparison': 'exact fractions (num/den); a rate with no denominator on both sides is no '
                                   'regression; on one side only it is not comparable and the gate fails',
                     'not_measured': 'value and policy probes: the models are the same on both sides',
                     'why_not_tactical_judge': 'h7c_arena.tactical_judge reports E1 regressions without failing '
                                               'and its command refuses two sides with the same models'},
        'reading': {'statistics': STATS, 'P': 'score - 0.5 over the 300 pairs: STRONGER (lower > 0) / WEAKER '
                    '(upper < 0) / NOT_ESTABLISHED; the class level reported',
                    'table': ['all of P STRONGER, no illegal move (tested 0, opponent illegal pairs 0), the E1 gate, '
                              'both time ratios <= 5.5 in the confirmation games: PASS (a canonical candidate)',
                              'otherwise: END (the failed items recorded), canonical unchanged',
                              'no budget within the cap: END_NO_BUDGET'],
                    'scope': 'the selected budget against S50 only: never which budget is strongest'},
        'seeds': {'confirm': CONFIRM_SEED, 'timing': {str(f): s for f, s in TIMING_SEEDS.items()},
                  'bootstrap': STATS['bootstrap']['seed'],
                  'registry': 'v2 Arena seeds (hybrid.h7r_seeds.ARENA_SEEDS 9101-9140; 9101-9110 used before)'},
        'stages': {'prepare': 'inputs and the confirmation sample', 'timing': 'the budget selected by time only',
                   'confirm': 'the confirmation games, the E1 gate and the judge'},
        'output': {'prepare': f'{D}/prepare.json', 'timing': f'{D}/timing.json', 'confirm': f'{D}/confirm.json',
                   'runs': RUNS},
        'sources': [{'path': rel, 'sha256': file_sha256(root / rel)} for rel in sources],
        'scope': 'desktop GPU; the web SER5 CPU time, certification and the canonical change are a separate promotion'}


def experiment_sha256(c: dict) -> str:
    return p1.experiment_sha256(c)


def load_contract(path=None, root=ROOT, stage=None, *, check_sources=True) -> tuple[dict, str, str]:
    root = Path(root)
    path = contract_path(path, root)
    c = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(c, dict) and set(c) == KEYS, f'{path}: the contract keys differ')
    require(c['format'] == FORMAT and c['contract_id'] == CONTRACT_ID and c['status'] == 'fixed',
            f'{path}: not the fixed {CONTRACT_ID} contract')
    a = c['approved']
    require(isinstance(a, dict) and list(a) == list(STAGES) and all(type(v) is bool for v in a.values()),
            f'{path}: approved flags')
    for i, s in enumerate(STAGES):
        if a[s]:
            require(all(a[x] for x in STAGES[:i]), f'{path}: {s} approved before {list(STAGES[:i])}')
    t, cf = c['timing'], c['confirm']
    require(c['budgets']['factors_in_order'] == list(FACTORS) and t['cap'] == CAP and t['pairs'] == TIMING_PAIRS
            and t['seeds'] == {str(f): s for f, s in TIMING_SEEDS.items()} and cf['pairs'] == PAIRS
            and cf['seed'] == CONFIRM_SEED and c['reading']['statistics'] == STATS and c['tactical']['gate'] == E1_GATE
            and t['workers'] == cf['workers'] == WORKERS, f'{path}: the budgets, cap, samples, seeds or rules differ')
    if check_sources:
        changed = [r['path'] for r in c['sources'] if file_sha256(root / r['path']) != r['sha256']]
        require(not changed, f'the checkout differs from the sources the contract was fixed for: {changed}')
    require(committed(path, root), f'{path}: not committed')
    if stage is not None:
        require(a[stage] is True, f'{path}: the {stage} stage is not approved')
    return c, file_sha256(path), experiment_sha256(c)


def stamp(sha, eid, git) -> dict:
    return {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'contract_file_sha256': sha, 'git': git}


def log_field(args, prefix: str) -> dict:
    name = getattr(args, 'log_name', None)
    if name is None:
        return {}
    require(re.fullmatch(rf'{prefix}-[A-Za-z0-9]+\.log', name) is not None, f'--log-name {name!r}')
    return {'log_name': name}


def committed_record(root, rel, what, eid) -> dict:
    path = Path(root) / rel
    require(path.is_file() and committed(path, root), f'{rel}: the {what} is not committed')
    rec = json.loads(path.read_text(encoding='utf-8'))
    require(rec.get('experiment_sha256') == eid, f'{rel}: the {what} is of another experiment')
    return rec


def check_models(c: dict, root) -> dict:
    from analysis import tb_v3, tb_v3_net

    root = Path(root)
    for rel in (MODELS['policy'], Path(MODELS['policy']).with_suffix('.json').as_posix(), MODELS['value'],
                Path(MODELS['value']).with_name('best.json').as_posix()):
        require((root / rel).is_file(), f'{rel} is missing')
    ident = tb_v3_net.model_identity(root / MODELS['policy'], root / MODELS['value'])
    require(ident == c['models']['identity'] == dict(tb_v3.MODELS), 'the model files are not the canonical pair')
    return ident


def load_suite(root, ref: dict) -> dict:
    from hybrid import h7r_openings as op
    path = Path(root) / ref['path']
    require(file_sha256(path) == ref['sha256'] and committed(path, root), f"{ref['path']}: not the committed suite")
    body = json.loads(path.read_text(encoding='utf-8'))
    op.check_suite(body)
    return body


def no_cublas() -> None:
    import os
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None, 'CUBLAS_WORKSPACE_CONFIG is set: the engine games keep '
                                                               'the conditions of every earlier engine run')


# --- prepare ------------------------------------------------------------------------------------------------------

SEED_KEY = re.compile(r'"(?:seed|run_seed|base_seed)": (\d+)')


def used_seeds(root) -> dict:
    """The contract's Arena seeds anywhere as a seed in a docs JSON other than this contract's own files."""
    root = Path(root)
    want = {CONFIRM_SEED, *TIMING_SEEDS.values()}
    hits = {}
    for path in (root / 'docs').rglob('*.json'):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(D + '/'):
            continue
        for s in sorted({int(x) for x in SEED_KEY.findall(path.read_text(errors='ignore'))} & want):
            hits.setdefault(s, []).append(rel)
    return hits


def confirmation_sample(c: dict, root) -> dict:
    from hybrid import h7r_openings as op

    body = load_suite(root, c['confirm']['suite'])
    openings = op.pick(body, PAIRS, CONFIRM_SEED)
    anchor = load_suite(root, c['confirm']['anchor_suite'])
    classes = [o['class_id'] for o in openings]
    dev = {}
    for name, path in DEV_SUITES:
        if (Path(root) / path).is_file():
            theirs = json.loads((Path(root) / path).read_text(encoding='utf-8'))['openings']
            dev[name] = len(set(classes) & {o['class_id'] for o in theirs})
    return {'suite': c['confirm']['suite']['name'], 'seed': CONFIRM_SEED, 'pairs': PAIRS,
            'opening_ids': [o['history_id'] for o in openings],
            'opening_ids_sha256': digest([o['history_id'] for o in openings]),
            'distinct_classes': len(set(classes)), 'class_ids_sha256': digest(sorted(set(classes))),
            'shared_classes_with': {ANCHOR_SUITE[0]: sorted(set(classes) & {o['class_id'] for o in anchor['openings']}),
                                    **{k: v for k, v in dev.items()}},
            'note': 'later confirmations that claim independence leave the shared anchor classes out'}


def prepare(args, root=ROOT) -> dict:
    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'prepare')
    out = root / c['output']['prepare']
    require(not out.exists(), f'{out} exists: never overwritten')
    guard = Guard(root, contract_path(args.contract, root), c,
                  [MODELS['policy'], MODELS['value'], c['timing']['suite']['path'], c['confirm']['suite']['path'],
                   c['confirm']['anchor_suite']['path']])
    models = check_models(c, root)
    load_suite(root, c['timing']['suite'])
    sample = confirmation_sample(c, root)
    hits = used_seeds(root)
    require(not hits, f'the contract seeds are already used: {hits}')
    left = [p.relative_to(root).as_posix() for p in (root / RUNS).rglob('*') if p.is_file() and 'logs' not in p.parts] \
        if (root / RUNS).exists() else []
    require(not left, f'{RUNS} holds files of another attempt: {left[:5]}')
    record = {'format': 'h7r-budget-prepare-v1', **log_field(args, 'prepare'), **stamp(sha, eid, guard.check()),
              'passed': True, 'models': models, 'confirmation_sample': sample, 'seeds_unused': True}
    p1.publish(out, record)
    return record


# --- games --------------------------------------------------------------------------------------------------------

def play_game(task: dict, make_agent=None) -> dict:
    """One game of the tested budget against S50: ``h7c_arena.play_game``'s loop, every move's side, factor, route
    and seconds kept (the opponent's too)."""
    from renju import BLACK, Game, IllegalMove
    from run_mcts_v8_benchmark import derive_seed

    ctx = task['context']
    if make_agent is None:
        import h7c_budget as hb
        make_agent = lambda factor, seed: hb.make_budget_agent(factor, ctx['policy'], ctx['value'], seed,  # noqa: E731
                                                               ctx['device'])
    seed, tested_color = task['seed'], task['tested_color']
    agents = {'tested': (make_agent(ctx['factor'], derive_seed(seed, 'tested')), ctx['factor']),
              'opponent': (make_agent(1, derive_seed(seed, 'opponent')), 1)}
    game = Game()
    for move in task['opening']:
        game.play(*move)
    times, illegal = [], None
    started_game = perf_counter()
    while not game.done:
        side = 'tested' if game.to_play == tested_color else 'opponent'
        agent, factor = agents[side]
        started = perf_counter()
        move = agent.select_move(game)
        seconds = perf_counter() - started
        route = getattr(getattr(agent, 'diagnostics', None), 'v8_route', '') or ''
        times.append({'ply': len(game.history), 'side': side, 'factor': factor, 'route': route,
                      'seconds': round(seconds, 5)})
        try:
            game.play(*move)
        except IllegalMove as exc:
            illegal = {'side': side, 'ply': len(game.history), 'color': 'black' if game.to_play == BLACK else 'white',
                       'move': [int(move[0]), int(move[1])], 'reason': str(exc)}
            break
    if illegal is not None:
        result = 'loss' if illegal['side'] == 'tested' else 'opponent_illegal'
    else:
        result = 'draw' if game.winner is None else 'win' if game.winner == tested_color else 'loss'
    return {'key': task['key'], 'signature': ctx['signature'], 'arm': ctx['arm'], 'pair': task['pair'], 'seed': seed,
            'tested_color': 'black' if tested_color == BLACK else 'white', 'opening_id': task['opening_id'],
            'result': result, 'winner': game.winner, 'length': len(game.history),
            'moves': [list(m) for m in game.history], 'illegal': illegal, 'truncated': False, 'move_times': times,
            'game_seconds': round(perf_counter() - started_game, 2)}


def _chunk(tasks: list[dict]) -> list[dict]:
    return [play_game(t) for t in tasks]


def identity(c: dict, eid: str, stage: str, factor: int, suite: dict, pairs: int, seed: int, device: str,
             git: dict) -> dict:
    from analysis import tb_v3, tb_v3_fast5
    return {'format': 'h7r-budget-games-v1', 'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'stage': stage,
            'tested_factor': factor, 'opponent_factor': 1, 'models': c['models']['identity'], 'suite': suite,
            'pairs': pairs, 'seed': seed, 'engine_hash': tb_v3.ENGINE_HASH,
            'runtime_hash': tb_v3_fast5.runtime_hash(device), 'device': device, 'git': git}


def arm_of(stage: str, factor: int, seed: int) -> dict:
    return {'name': f'{stage}_S{50 * factor}_vs_S50', 'seed': seed}


def play_games(c: dict, eid: str, root, stage: str, factor: int, suite_ref_: dict, openings: list[dict], seed: int,
               jsonl: Path, device: str, log, threads_for_tests: bool = False) -> tuple[list[dict], dict]:
    """Every game of the sample (resumed from ``jsonl``: the same identity, git commit included)."""
    import h7c_arena as ar
    import h7r_eval as cli
    from hybrid.progress import Progress, run_chunks

    git = cli._git(False)
    ident = identity(c, eid, stage, factor, suite_ref_, len(openings), seed, device, git)
    sig = digest(ident)
    arm = arm_of(stage, factor, seed)
    tasks = cli.build_tasks(arm, openings)
    finished = ar.load_finished(jsonl, {t['key'] for t in tasks}, sig, log=log)
    root = Path(root)
    ctx = {'policy': str(root / MODELS['policy']), 'value': str(root / MODELS['value']), 'factor': factor,
           'device': device, 'signature': sig, 'arm': arm['name']}
    todo = [{**t, 'context': ctx} for t in tasks if t['key'] not in finished]
    log(f"[budget] {arm['name']}: {len(tasks)} games ({len(finished)} done), workers {WORKERS}", flush=True)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    if todo:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        from multiprocessing import get_context

        pool_of = (lambda: ThreadPoolExecutor(max_workers=1)) if threads_for_tests else (
            lambda: ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context('spawn')))
        with jsonl.open('a', encoding='utf-8') as out, pool_of() as pool:
            def keep(_, rows):
                for row in rows:
                    check_game(row, tasks, arm['name'], sig, factor)
                    out.write(json.dumps(row, separators=(',', ':')) + '\n')
                    finished[row['key']] = row
                out.flush()
            run_chunks(pool, _chunk, [[t] for t in todo], Progress('budget', len(todo), 'games'), len,
                       unit_name='games', on_result=keep)
    games = [finished[t['key']] for t in tasks]
    for g in games:
        check_game(g, tasks, arm['name'], sig, factor)
    return games, ident


def check_game(game: dict, tasks: list[dict], arm_name: str, sig: str, factor: int) -> None:
    """The game is its task's, legal from its opening, and every move after the opening carries its time, its side
    (by colour: the tested side's moves are its colour's) and that side's factor."""
    import h7r_eval as cli
    from renju import BLACK

    task = next((t for t in tasks if t['key'] == game.get('key')), None)
    require(task is not None, f"{game.get('key')}: not a game of this sample")
    cli.check_games([game], [task], arm_name, sig)
    times = game.get('move_times')
    first = len(task['opening'])
    end = game['length'] + (1 if game.get('illegal') else 0)
    require(isinstance(times, list) and [t.get('ply') for t in times] == list(range(first, end)),
            f"{game['key']}: a move without its time (the times do not cover plies {first}-{end - 1})")
    tested_parity = 0 if task['tested_color'] == BLACK else 1
    for t in times:
        side = 'tested' if t['ply'] % 2 == tested_parity else 'opponent'
        require(t.get('side') == side and t.get('factor') == (factor if side == 'tested' else 1),
                f"{game['key']} ply {t['ply']}: the time is not the {side} side's")
        s = t.get('seconds')
        require(isinstance(s, (int, float)) and math.isfinite(s) and s >= 0, f"{game['key']} ply {t['ply']}: seconds")


def percentile(values: list[float], q: float) -> float:
    """The nearest-rank percentile (``h7r_eval``'s order-statistic rule)."""
    v = sorted(values)
    require(v, 'no times')
    return v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))]


def times_of(games: list[dict]) -> dict:
    tested = [t['seconds'] for g in games for t in g['move_times'] if t['side'] == 'tested']
    opponent = [t['seconds'] for g in games for t in g['move_times'] if t['side'] == 'opponent']
    require(tested and opponent, 'no timed moves')
    mean_t, mean_o = sum(tested) / len(tested), sum(opponent) / len(opponent)
    p95_t, p95_o = percentile(tested, 0.95), percentile(opponent, 0.95)
    require(mean_o > 0 and p95_o > 0, 'the S50 moves took no time')
    return {'tested': {'moves': len(tested), 'mean_seconds': round(mean_t, 4), 'p95_seconds': round(p95_t, 4)},
            'opponent_s50': {'moves': len(opponent), 'mean_seconds': round(mean_o, 4), 'p95_seconds': round(p95_o, 4)},
            'ratio': {'mean': round(mean_t / mean_o, 4), 'p95': round(p95_t / p95_o, 4)}}


def within_cap(t: dict) -> bool:
    return t['ratio']['mean'] <= CAP and t['ratio']['p95'] <= CAP


def select(measured: dict) -> int | None:
    """Section 4: the first factor of S400, S200, S100 (in that order) whose two ratios are within the cap."""
    for f in FACTORS:
        if f not in measured:
            return None
        if within_cap(measured[f]):
            return f
    return None


# --- timing -------------------------------------------------------------------------------------------------------

def timing(args, root=ROOT, log=print) -> dict:
    from hybrid import h7r_openings as op

    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'timing')
    prep = committed_record(root, c['output']['prepare'], 'prepare record', eid)
    out = root / c['output']['timing']
    require(not out.exists(), f'{out} exists: never overwritten')
    no_cublas()
    guard = Guard(root, contract_path(args.contract, root), c, [MODELS['policy'], MODELS['value'],
                                                                c['timing']['suite']['path'], c['output']['prepare']])
    check_models(c, root)
    suite = load_suite(root, c['timing']['suite'])
    measured, files = {}, {}
    for f in FACTORS:
        seed = TIMING_SEEDS[f]
        openings = op.pick(suite, TIMING_PAIRS, seed)
        jsonl = root / RUNS / f'timing-S{50 * f}.jsonl'
        games, ident = play_games(c, eid, root, 'timing', f, c['timing']['suite'], openings, seed, jsonl,
                                  args.device, log, getattr(args, 'threads_for_tests', False))
        measured[f] = times_of(games)
        files[str(f)] = {'jsonl': jsonl.relative_to(root).as_posix(), 'sha256': file_sha256(jsonl),
                         'identity_sha256': digest(ident), 'games': len(games)}
        log(f"[budget] S{50 * f}: mean x{measured[f]['ratio']['mean']}, p95 x{measured[f]['ratio']['p95']}", flush=True)
        if within_cap(measured[f]):
            break
    chosen = select(measured)
    sample = confirmation_sample(c, root)
    require(sample == prep['confirmation_sample'], 'the confirmation sample is not the one prepare recorded')
    record = {'format': 'h7r-budget-timing-v1', **log_field(args, 'timing'), **stamp(sha, eid, guard.check()),
              'measured': {f'S{50 * f}': {**m, 'factor': f, 'within_cap': within_cap(m)} for f, m in measured.items()},
              'files': files, 'cap': CAP, 'selected_factor': chosen,
              'selected': None if chosen is None else f'S{50 * chosen}',
              'decision': 'SELECTED' if chosen is not None else 'END_NO_BUDGET',
              'confirmation': None if chosen is None else {
                  'factor': chosen, 'models': c['models']['identity'], 'seed': CONFIRM_SEED, 'pairs': PAIRS,
                  'suite': c['confirm']['suite'], 'opening_ids': sample['opening_ids'],
                  'opening_ids_sha256': sample['opening_ids_sha256']},
              'note': 'the scores of the timing games are not read'}
    p1.publish(out, record)
    return record


def timing_again(c: dict, eid: str, rec: dict, root) -> dict:
    """The timing record recomputed from its game files (the selection included)."""
    from hybrid import h7r_openings as op
    import h7r_eval as cli

    root = Path(root)
    suite = load_suite(root, c['timing']['suite'])
    measured = {}
    for key, ref in rec['files'].items():
        f = int(key)
        path = root / ref['jsonl']
        require(path.is_file() and file_sha256(path) == ref['sha256'], f"{ref['jsonl']}: not the recorded games")
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        openings = op.pick(suite, TIMING_PAIRS, TIMING_SEEDS[f])
        arm = arm_of('timing', f, TIMING_SEEDS[f])
        tasks = cli.build_tasks(arm, openings)
        require({r['signature'] for r in rows} == {ref['identity_sha256']}, f'S{50 * f}: games of another setup')
        by = {r['key']: r for r in rows}
        require(set(by) == {t['key'] for t in tasks} and len(rows) == len(tasks), f'S{50 * f}: not every timing game')
        games = [by[t['key']] for t in tasks]
        for g in games:
            check_game(g, tasks, arm['name'], ref['identity_sha256'], f)
        measured[f] = times_of(games)
    chosen = select(measured)
    require({f'S{50 * f}': {**m, 'factor': f, 'within_cap': within_cap(m)} for f, m in measured.items()}
            == rec['measured'] and chosen == rec['selected_factor'], 'the timing record is not what its games give')
    return rec


# --- tactical -----------------------------------------------------------------------------------------------------

def _e1_task(task):
    from e1_evaluate import evaluate_position
    import h7c_budget as hb

    position, factor, policy, value, device = task
    return evaluate_position(position, lambda seed: hb.make_budget_agent(factor, policy, value, seed, device), 'h7c')


def e1_suite(factor: int, root, device: str, log=print, workers: int = WORKERS,
             threads_for_tests: bool = False) -> dict:
    """The E1 dev manifest with the budget agent (``h7c_arena.e1_suite`` with ``h7c_budget.make_budget_agent``)."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from multiprocessing import get_context

    import h7c_arena as ar
    from e1_evaluate import summarize as e1_summarize
    from hybrid.progress import Progress, run_chunks

    root = Path(root)
    manifest = json.loads(ar.E1_MANIFEST.read_text(encoding='utf-8'))
    positions = manifest['positions']
    rows = {}
    pool_of = (lambda: ThreadPoolExecutor(max_workers=1)) if threads_for_tests else (
        lambda: ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')))
    with pool_of() as pool:
        def keep(_, result):
            rows[result['key']] = result
        run_chunks(pool, _e1_task, [(p, factor, str(root / MODELS['policy']), str(root / MODELS['value']), device)
                                    for p in positions], Progress(f'budget-e1-S{50 * factor}', len(positions),
                                                                  'positions'), lambda _: 1,
                   unit_name='positions', on_result=keep)
    ordered = [rows[p['key']] for p in positions]
    truths_of = {p['key']: {**p['truths'], **rows[p['key']]['truths_added']} for p in positions}
    summary = e1_summarize(ordered, truths_of)
    return {'factor': factor, 'manifest_sha256': file_sha256(ar.E1_MANIFEST), 'positions': len(positions),
            'metrics': summary['metrics'], 'outcomes': summary['outcomes']}


def fraction(metric) -> Fraction | None:
    """A rate's exact value from its ``num/den`` (None: no denominator), a count as itself."""
    if isinstance(metric, dict):
        num, den = (int(x) for x in metric['n'].split('/'))
        return None if den == 0 else Fraction(num, den)
    return Fraction(int(metric))


def tactical_gate(budget: dict, s50: dict) -> dict:
    """Section 6 (fixed before any result): the budget agent's unsound witnesses are 0, its rescue rate and veto
    recall are not lower than S50's and its false vetoes not more. A rate with no denominator on both sides is no
    regression; on one side only it is not comparable, which fails the gate."""
    b, s = budget['metrics'], s50['metrics']
    items = {'unsound_witness == 0': fraction(b['unsound_witness']) == 0}
    compare = {}
    for name in (*E1_HIGHER_IS_BETTER, *E1_LOWER_IS_BETTER):
        x, y = fraction(b[name]), fraction(s[name])
        compare[name] = {'budget': b[name], 's50': s[name]}
        if x is None and y is None:
            items[f'{name}: no data on either side'] = True
        elif x is None or y is None:
            items[f'{name}: not comparable (a denominator on one side only)'] = False
        elif name in E1_HIGHER_IS_BETTER:
            items[f'{name} >= S50'] = x >= y
        else:
            items[f'{name} <= S50'] = x <= y
    return {'items': items, 'verdict': 'PASS' if all(items.values()) else 'FAIL', 'compare': compare,
            'reported': {k: {'budget': b[k], 's50': s[k]} for k in b if k not in compare}}


# --- confirm ------------------------------------------------------------------------------------------------------

def label(b: dict) -> str:
    return 'STRONGER' if b['lower_one_sided'] > 0 else 'WEAKER' if b['upper_one_sided'] < 0 else 'NOT_ESTABLISHED'


def judge_of(c: dict, games: list[dict], openings: list[dict], tactical: dict) -> dict:
    from hybrid import h7r_eval as ev

    table = ev.pair_table(games, openings)
    strata = {c['confirm']['suite']['name']: [(p['pair_score'], 0.5, p['class_id']) for p in table['pairs']]}
    main = p1.stratified_bounds(strata, STATS['bootstrap']['samples'], STATS['bootstrap']['seed'],
                                STATS['alpha_one_sided'], unit='opening')
    sens = p1.stratified_bounds(strata, STATS['bootstrap']['samples'], STATS['bootstrap']['seed'],
                                STATS['alpha_one_sided'], unit='class')
    t = times_of(games)
    items = {'P STRONGER': label(main['P']) == 'STRONGER',
             'no illegal move (tested 0, opponent illegal pairs 0)': table['tested_illegal'] == 0 and table['valid'],
             'E1 gate': tactical['gate']['verdict'] == 'PASS',
             f'time ratios <= {CAP} (mean and p95)': within_cap(t)}
    summary = ev.arm_summary(table, STATS)
    return {'pairs': len(table['pairs']), 'score': summary['score'], 'P': main['P'], 'reading': label(main['P']),
            'sensitivity_class_level': {'P': sens['P'], 'reading': label(sens['P'])},
            'summary': summary, 'times': t, 'tested_illegal': table['tested_illegal'],
            'opponent_illegal_pairs': table['opponent_illegal_pairs'], 'items': items,
            'verdict': 'PASS' if all(items.values()) else 'END', 'failed': [k for k, v in items.items() if not v]}


def confirm_openings(c: dict, trec: dict, root) -> list[dict]:
    """Exactly the sample the timing record fixed (and ``pick`` gives it again)."""
    from hybrid import h7r_openings as op
    body = load_suite(root, c['confirm']['suite'])
    openings = op.pick(body, PAIRS, CONFIRM_SEED)
    fixed = trec['confirmation']
    require([o['history_id'] for o in openings] == fixed['opening_ids'] and fixed['seed'] == CONFIRM_SEED
            and fixed['pairs'] == PAIRS and fixed['models'] == c['models']['identity']
            and fixed['suite'] == c['confirm']['suite'],
            'the confirmation sample is not the one the timing record fixed')
    return openings


def e1_cached(factor: int, root, device: str, log, threads_for_tests: bool) -> dict:
    """The E1 run of a factor, kept under runs/ (a rerun reads it back when its manifest and models are the same)."""
    import h7c_arena as ar
    path = Path(root) / RUNS / f'e1-S{50 * factor}.json'
    if path.is_file():
        rec = json.loads(path.read_text(encoding='utf-8'))
        if rec.get('manifest_sha256') == file_sha256(ar.E1_MANIFEST) and rec.get('factor') == factor:
            return rec
        raise ValueError(f'{path}: an E1 run of another manifest or budget: report it (nothing is overwritten)')
    rec = e1_suite(factor, root, device, log, threads_for_tests=threads_for_tests)
    path.parent.mkdir(parents=True, exist_ok=True)
    from hybrid.h7r_ledger import publish_json
    publish_json(path, rec)
    return rec


def confirm(args, root=ROOT, log=print) -> dict:
    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'confirm')
    committed_record(root, c['output']['prepare'], 'prepare record', eid)
    trec = committed_record(root, c['output']['timing'], 'timing record', eid)
    require(trec['decision'] == 'SELECTED', f"the timing stage ended {trec['decision']}: nothing to confirm")
    out = root / c['output']['confirm']
    require(not out.exists(), f'{out} exists: never overwritten')
    no_cublas()
    guard = Guard(root, contract_path(args.contract, root), c, [MODELS['policy'], MODELS['value'],
                                                                c['confirm']['suite']['path'], c['output']['timing']])
    check_models(c, root)
    factor = trec['selected_factor']
    openings = confirm_openings(c, trec, root)
    jsonl = root / RUNS / 'confirm.jsonl'
    threads = getattr(args, 'threads_for_tests', False)
    games, ident = play_games(c, eid, root, 'confirm', factor, c['confirm']['suite'], openings, CONFIRM_SEED, jsonl,
                              args.device, log, threads)
    e1 = {'budget': e1_cached(factor, root, args.device, log, threads), 's50': e1_cached(1, root, args.device, log,
                                                                                         threads)}
    tactical = {'budget': e1['budget']['metrics'], 's50': e1['s50']['metrics'],
                'gate': tactical_gate(e1['budget'], e1['s50'])}
    body = judge_of(c, games, openings, tactical)
    record = {'format': 'h7r-budget-confirm-v1', **log_field(args, 'confirm'), **stamp(sha, eid, guard.check()),
              'selected': trec['selected'], 'factor': factor,
              'timing_record_sha256': file_sha256(root / c['output']['timing']),
              'games': {'jsonl': jsonl.relative_to(root).as_posix(), 'sha256': file_sha256(jsonl),
                        'identity_sha256': digest(ident), 'games': len(games)},
              'e1_files': {k: {'path': f'{RUNS}/e1-S{50 * v["factor"]}.json',
                               'sha256': file_sha256(root / RUNS / f'e1-S{50 * v["factor"]}.json')}
                           for k, v in e1.items()},
              'tactical': tactical, **body,
              'scope': 'the selected budget against S50 with the canonical models on the desktop GPU; never which '
                       'budget is strongest; a PASS names a canonical candidate only'}
    p1.publish(out, record)
    return record


def confirm_again(c: dict, eid: str, rec: dict, root) -> dict:
    import h7r_eval as cli

    root = Path(root)
    trec = committed_record(root, c['output']['timing'], 'timing record', eid)
    openings = confirm_openings(c, trec, root)
    path = root / rec['games']['jsonl']
    require(file_sha256(path) == rec['games']['sha256'], 'the confirmation games changed')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    arm = arm_of('confirm', rec['factor'], CONFIRM_SEED)
    tasks = cli.build_tasks(arm, openings)
    by = {r['key']: r for r in rows}
    require(set(by) == {t['key'] for t in tasks} and len(rows) == len(tasks), 'not every confirmation game')
    require({r['signature'] for r in rows} == {rec['games']['identity_sha256']}, 'confirmation games of another setup')
    games = [by[t['key']] for t in tasks]
    for g in games:
        check_game(g, tasks, arm['name'], rec['games']['identity_sha256'], rec['factor'])
    e1 = {}
    for k, ref in rec['e1_files'].items():
        require(file_sha256(root / ref['path']) == ref['sha256'], f"{ref['path']} changed")
        e1[k] = json.loads((root / ref['path']).read_text(encoding='utf-8'))
    tactical = {'budget': e1['budget']['metrics'], 's50': e1['s50']['metrics'],
                'gate': tactical_gate(e1['budget'], e1['s50'])}
    body = judge_of(c, games, openings, tactical)
    require(tactical == rec['tactical'] and all(rec[k] == v for k, v in json.loads(json.dumps(body)).items()),
            'the confirmation record is not what its games and E1 runs give')
    return rec


# --- verify / main ------------------------------------------------------------------------------------------------

def verify(args, root=ROOT) -> dict:
    root = Path(root)
    c, sha, eid = load_contract(args.contract, root)
    rel = c['output'][args.stage]
    rec = json.loads((root / rel).read_text(encoding='utf-8'))
    require(rec.get('experiment_sha256') == eid, f'{rel}: another experiment')
    if args.stage == 'timing':
        timing_again(c, eid, rec, root)
    elif args.stage == 'confirm':
        confirm_again(c, eid, rec, root)
    code = 0 if rec.get('passed', True) else 2
    return {'stage': args.stage, 'log_name': rec.get('log_name'), 'exit_code': code, 'stage_files': [rel]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--contract', default=None)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in STAGES:
        s = sub.add_parser(name)
        s.add_argument('--device', default='cuda')
        s.add_argument('--log-name', default=None)
    v = sub.add_parser('verify')
    v.add_argument('--stage', required=True, choices=STAGES)
    args = parser.parse_args(argv)
    try:
        if args.command == 'verify':
            print(json.dumps(verify(args), separators=(',', ':')), flush=True)
            return 0
        rec = {'prepare': prepare, 'timing': timing, 'confirm': confirm}[args.command](args)
        keys = ('passed', 'measured', 'selected', 'decision', 'score', 'P', 'reading', 'times', 'items', 'verdict',
                'failed')
        print(json.dumps({k: rec[k] for k in keys if k in rec}, ensure_ascii=False, indent=1, default=str), flush=True)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
