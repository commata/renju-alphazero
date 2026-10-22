"""H7-E4-v2 arena: TB-V2-Value+E4v2 against TB-V2-Value on seeds 8441 / 8442 (docs/h6-value.md, E4-v2).

    python scripts/summarize_h7_e4.py --runs runs/h7_e4v2/b_8441.json runs/h7_e4v2/b_8442.json \\
        runs/h7_e4v2/e_8441.json runs/h7_e4v2/e_8442.json --workers 4 --fast fast4 \\
        --jsonl runs/h7_e4v2/arena_positions.jsonl --output docs/h7/h7_e4v2_arena.json

Runs (fixed before the games): B ``puct_value_vct2_50k_tt`` and E ``puct_value_vct2_50k_tt_e4v2``, both vs
``v8:full``, ``--puct-c 1.5 --fast fast4``, CUDA value, 50 pairs on each of the seeds 8441 and 8442 (both, no
further seed), built by the canonical factory (B: the frozen runtime profile; E: the frozen candidate on it),
the same openings and colours. One verdict on both seeds together, every gate needed:

    strength   score difference E - B per opening (both colours), bootstrap over openings: 95% lower >= -0.05
    cost       total V8 move seconds E / B over the paired games <= 1.5
    soundness  every move the E4-v2 attack proved (route ``own_vct2``) is rechecked by the truth's
               ``check_move`` (``h7_e4_truth.DEFAULT_BUDGET``, 10M): no MISS (UNCLEAR reported); no safety
               violation in either arm
    frequency  events = the positions right after a ``v8:full`` move (the arm to move; not the position after
               the opening) where the frozen reference solver (``analysis.e4_reference``, budget ``'4m'``,
               ``docs/h7/h7_e4v2_reference_4m.json``) proves a WIN; punished = the arm's move there is a
               ``check_move`` PUNISH. Openings (both colours, both arms) are the clusters: every bootstrap
               sample recomputes rate = sum punished / sum events per arm (never a mean of per-opening rates;
               an opening without events stays in); 95% lower bound of rate_E - rate_B > 0
    isolation  B is the frozen runtime profile and E the frozen candidate (configuration, runtime identities,
               canonical factory, candidate sources checked now); B never ran the attack, E recorded the
               E4-v2 step at every move; the off path is the unit tests (tests.test_tb_v2_value_e4v2)
    verdict    ADOPT if every gate passes, else REJECT (no retuning, no further seed)
Reported only: per-colour scores, game minutes, the E4-v2 steps (skips by reason, attacks, wins, switches),
punish checks UNCLEAR, games identical to B and the openings that left B's line before E's first switch.

The positions are solved once per distinct history (both arms, both seeds), in a worker pool; each finished
position is appended to ``--jsonl`` (resume; a fingerprint binds the reference, the checker and the budget).
The solvers count nodes, not seconds, so the result of a position does not depend on the machine: the work can
be split, ``--shard K/N`` on each machine (``--jsonl`` only), then one summary run with ``--merge`` of the files:

    python scripts/summarize_h7_e4.py --runs ... --workers 8 --shard 0/2 --jsonl runs/h7_e4v2/arena_0.jsonl
    python scripts/summarize_h7_e4.py --runs ... --workers 6 --shard 1/2 --jsonl runs/h7_e4v2/arena_1.jsonl
    python scripts/summarize_h7_e4.py --runs ... --merge runs/h7_e4v2/arena_0.jsonl runs/h7_e4v2/arena_1.jsonl \
        --jsonl runs/h7_e4v2/arena_positions.jsonl --output docs/h7/h7_e4v2_arena.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from summarize_h5 import SCORE, merge, summarize  # noqa: E402
import summarize_h7_e2 as e2  # noqa: E402

FORMAT = 'h7-e4v2-arena-v1'
BASELINE, CANDIDATE = 'puct_value_vct2_50k_tt', 'puct_value_vct2_50k_tt_e4v2'
OPPONENT = 'v8:full'
SEEDS = (8441, 8442)
DEVICE = 'cuda'
REFERENCE_BUDGET = '4m'
REFERENCE_RESULT = 'docs/h7/h7_e4v2_reference_4m.json'
RULES = {'strength_low_min': -0.05, 'cost_max': 1.5, 'frequency_low_gt': 0.0, 'soundness_miss_max': 0,
         'fast': 'fast4', 'value_device': DEVICE, 'pairs_per_seed': 50, 'seeds': list(SEEDS),
         'reference_budget': REFERENCE_BUDGET, 'bootstrap': e2.RULES['bootstrap'], 'seed': e2.RULES['seed'],
         'cluster': 'opening (run seed, pair), both colours, both arms'}
# what a solved position depends on: the reference solver, the checker and the code under both
SOURCES = ('src/analysis/e4_reference.py', 'scripts/h7_e4_truth.py', 'scripts/h7_e4_baseline.py',
           'src/analysis/e4_attack.py', 'src/analysis/vct2_attack.py', 'src/analysis/selective_vct.py',
           'tests/track_b_frozen.sha256')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_run(run: dict) -> None:
    from analysis import tb_v2_runtime as rt
    from analysis import tb_v2_value_e4v2 as cand
    from analysis.tb_v2_value import ENGINE_OPTIONS, TB_V2_VALUE

    where = f"{run.get('arm')} seed {run.get('seed')}"
    if run.get('arm') not in (BASELINE, CANDIDATE) or run.get('opponent') != OPPONENT or run.get('seed') not in SEEDS:
        raise ValueError(f'{where}: expected {BASELINE} / {CANDIDATE} vs {OPPONENT} on seeds {SEEDS}')
    is_e = run['arm'] == CANDIDATE
    want = {**TB_V2_VALUE, **ENGINE_OPTIONS, **({'e4v2_attack': True} if is_e else {})}
    if run.get('v8_config') != want:
        raise ValueError(f'{where}: not the frozen configuration')
    if run.get('fast', {}).get('requested') != RULES['fast']:
        raise ValueError(f"{where}: not run with --fast {RULES['fast']}")
    if str(run.get('value_device', '')).split(':')[0] != DEVICE:
        raise ValueError(f"{where}: value device {run.get('value_device')}, expected {DEVICE}")
    if run.get('pairs') != RULES['pairs_per_seed']:
        raise ValueError(f"{where}: {run.get('pairs')} pairs, the protocol fixes {RULES['pairs_per_seed']}")
    if run.get('canonical_factory') is not True:
        raise ValueError(f'{where}: not built by the canonical factory')
    r = run.get('runtime', {})
    if r.get('engine_hash') != rt.ENGINE_HASH:
        raise ValueError(f'{where}: the base is not the frozen engine')
    if is_e:
        if (r.get('candidate_hash') != cand.candidate_hash() or r.get('runtime_hash') != cand.runtime_hash(DEVICE)
                or r.get('base_runtime_hash') != rt.runtime_hash(DEVICE)):
            raise ValueError(f'{where}: not the frozen candidate {cand.NAME}')
    elif r.get('runtime_hash') != rt.runtime_hash(DEVICE) or 'candidate_hash' in r:
        raise ValueError(f'{where}: not the frozen runtime profile')
    steps = [('e4_attack' in m) for g in run['games'] for m in g['v8_moves']]
    if is_e and not all(steps):
        raise ValueError(f'{where}: a move without its E4-v2 step record')
    if not is_e and any(steps):
        raise ValueError(f'{where}: the baseline recorded an attack step')


def check_reference(result: dict) -> None:
    """The arena's detector is the frozen one: the 4M DEV check passed on this very file."""
    from scripts.run_mcts_v8_benchmark import file_sha256

    if result.get('verdict') != 'FREEZE' or result.get('budget') != REFERENCE_BUDGET:
        raise ValueError(f'the reference DEV check is not FREEZE at {REFERENCE_BUDGET}')
    if result.get('reference_sha256') != file_sha256(ROOT / 'src/analysis/e4_reference.py'):
        raise ValueError('src/analysis/e4_reference.py is not the file of the reference DEV check')


def fingerprint(fast: str) -> dict:
    from scripts.h7_e4_truth import DEFAULT_BUDGET

    return {'budget': REFERENCE_BUDGET, 'check_budget': DEFAULT_BUDGET, 'fast': fast,
            'sources': {name: _sha(ROOT / name) for name in SOURCES}}


def history_key(moves) -> str:
    return hashlib.sha256(json.dumps([list(m) for m in moves], separators=(',', ':')).encode()).hexdigest()


def _opening_length(run: dict, pair: int) -> int:
    return len(next(o for o in run['openings'] if o['pair'] == pair)['moves'])


def positions(runs: list[dict]) -> list[dict]:
    """Every own move of both arms: the history before it, the played move, whether it is a frequency position
    (right after a ``v8:full`` move) and whether the E4-v2 attack proved it."""
    out = []
    for run in runs:
        for g in run['games']:
            opening = _opening_length(run, g['pair'])
            for m in g['v8_moves']:
                ply = m['ply']
                played = list(m.get('played') or g['moves'][ply])
                step = m.get('e4_attack') or {}
                proved = step.get('move') is not None
                if proved and (m['route'] != 'own_vct2' or list(step['move']) != played):
                    raise ValueError(f"{g['key']} ply {ply}: an attack WIN that is not the move played")
                out.append({'arm': run['arm'], 'cluster': (run['seed'], g['pair']), 'game': g['key'], 'ply': ply,
                            'history': g['moves'][:ply], 'played': played, 'event_position': ply - 1 >= opening,
                            'attack': proved, 'changed': proved and bool(m.get('changed'))})
    return out


def tasks_of(rows: list[dict]) -> list[dict]:
    """One task per distinct history: the reference always; every played move there is checked when the
    reference proves a WIN, every attack move always (soundness)."""
    tasks: dict[str, dict] = {}
    for r in rows:
        key = history_key(r['history'])
        t = tasks.setdefault(key, {'key': key, 'moves': r['history'], 'played': [], 'attack': []})
        for name, want in (('played', True), ('attack', r['attack'])):
            if want and r['played'] not in t[name]:
                t[name].append(r['played'])
    for t in tasks.values():
        t['played'].sort()
        t['attack'].sort()
    return sorted(tasks.values(), key=lambda t: t['key'])


def _init(fast: str) -> None:
    if fast != 'off':
        from hybrid.h6_fast import install
        install(fast)


def position_task(task: dict) -> dict:
    from analysis.e4_reference import reference_attack
    from renju import Game
    from scripts.h7_e4_baseline import check_move
    from scripts.h7_e4_truth import DEFAULT_BUDGET

    game = Game()
    for m in task['moves']:
        game.play(*m)
    move, info = reference_attack(game, REFERENCE_BUDGET)
    want = (task['played'] if move is not None else []) + task['attack']
    checks = {}
    for m in want:
        name = json.dumps(list(m))
        if name not in checks:
            checks[name] = check_move(task['moves'], m, DEFAULT_BUDGET)
    return {'key': task['key'], 'played': task['played'], 'attack': task['attack'], 'reference': info,
            'checks': checks}


def _chunk(tasks):
    return [position_task(t) for t in tasks]


def _outcome(solved: dict, row: dict) -> str:
    c = solved[history_key(row['history'])]['checks'].get(json.dumps(row['played']))
    if c is None:
        raise ValueError(f"{row['game']} ply {row['ply']}: the played move was not checked")
    return c['outcome']


def frequency_counts(rows: list[dict], solved: dict) -> dict:
    """Per cluster and arm: events, punished, punish checks UNCLEAR."""
    out: dict[tuple, dict] = {}
    for r in rows:
        c = out.setdefault(r['cluster'], {a: {'events': 0, 'punished': 0, 'unclear': 0} for a in (BASELINE, CANDIDATE)})
        if not r['event_position'] or solved[history_key(r['history'])]['reference']['move'] is None:
            continue
        a, outcome = c[r['arm']], _outcome(solved, r)
        a['events'] += 1
        a['punished'] += outcome == 'PUNISH'
        a['unclear'] += outcome == 'UNCLEAR'
    return out


def frequency(counts: dict) -> dict:
    keys = sorted(counts)

    def rates(ks):
        tot = {a: [sum(counts[k][a][f] for k in ks) for f in ('punished', 'events')] for a in (BASELINE, CANDIDATE)}
        if not tot[BASELINE][1] or not tot[CANDIDATE][1]:
            return None
        return {a: p / e for a, (p, e) in tot.items()}

    total = {a: {f: sum(counts[k][a][f] for k in keys) for f in ('events', 'punished', 'unclear')}
             for a in (BASELINE, CANDIDATE)}
    point = rates(keys) if keys else None
    out = {'openings': len(keys), 'baseline': total[BASELINE], 'candidate': total[CANDIDATE]}
    if point is None:
        return {**out, 'rate': None, 'diff': None, 'ci95': None, 'degenerate_samples': None, 'verdict': 'INCOMPLETE'}
    rng, b, draws, degenerate = Random(RULES['seed']), RULES['bootstrap'], [], 0
    for _ in range(b):
        r = rates([rng.choice(keys) for _ in keys])
        if r is None:  # a sample without events in one arm: its rate is undefined
            degenerate += 1
        else:
            draws.append(r[CANDIDATE] - r[BASELINE])
    draws.sort()
    n = len(draws)
    ci = [round(draws[int(0.025 * n)], 4), round(draws[int(0.975 * n) - 1], 4)] if n else None
    diff = point[CANDIDATE] - point[BASELINE]
    return {**out, 'rate': {'baseline': round(point[BASELINE], 4), 'candidate': round(point[CANDIDATE], 4)},
            'diff': round(diff, 4), 'ci95': ci, 'degenerate_samples': degenerate,
            'verdict': 'PASS' if ci and ci[0] > RULES['frequency_low_gt'] else 'FAIL'}


def strength(cand: dict, base: dict) -> dict:
    clusters = {}
    for key in sorted(cand):
        clusters.setdefault(key[:2], []).append(SCORE[cand[key]['result']] - SCORE[base[key]['result']])
    out = e2._bootstrap(clusters)
    out['verdict'] = 'PASS' if out['ci95'][0] >= RULES['strength_low_min'] else 'FAIL'
    return out


def soundness(rows: list[dict], solved: dict, safety: dict) -> dict:
    attacks = [r for r in rows if r['attack']]
    outcomes = {r['game'] + f"|{r['ply']}": _outcome(solved, r) for r in attacks}
    miss = sorted(k for k, o in outcomes.items() if o == 'MISS')
    unclear = sorted(k for k, o in outcomes.items() if o == 'UNCLEAR')
    bad = [f'{arm}: {n} safety violations' for arm, n in safety.items() if n]
    return {'attack_wins': len(attacks), 'switched': sum(r['changed'] for r in attacks),
            'punish': sum(o == 'PUNISH' for o in outcomes.values()), 'miss': miss, 'unclear': unclear,
            'safety_violations': safety,
            'verdict': 'PASS' if len(miss) <= RULES['soundness_miss_max'] and not bad else 'FAIL',
            'reasons': ([f'{len(miss)} attack moves MISS'] if miss else []) + bad}


def steps(run_games: list[dict]) -> dict:
    out = {'moves': 0, 'ran': 0, 'wins': 0, 'switched': 0, 'exhausted': 0, 'skips': {}, 'attack_seconds': 0.0}
    for g in run_games:
        for m in g['v8_moves']:
            s = m['e4_attack']
            out['moves'] += 1
            if s.get('skip'):
                out['skips'][s['skip']] = out['skips'].get(s['skip'], 0) + 1
            if s.get('ran'):
                out['ran'] += 1
                out['exhausted'] += bool(s.get('exhausted'))
                out['attack_seconds'] += s.get('seconds', 0.0)
            if s.get('move') is not None:
                out['wins'] += 1
                out['switched'] += bool(m.get('changed'))
    out['attack_seconds'] = round(out['attack_seconds'], 1)
    return out


def isolation_report(cand: dict, base: dict) -> dict:
    """Games equal to B, and openings whose E game left B's line before E's first switched move."""
    identical, early = 0, []
    for key in sorted(cand):
        c, b = cand[key]['moves'], base[key]['moves']
        if c == b:
            identical += 1
            continue
        first = min((m['ply'] for m in cand[key]['v8_moves'] if m['e4_attack'].get('move') is not None
                     and m.get('changed')), default=len(c))
        split = next((i for i, (x, y) in enumerate(zip(c, b)) if x != y), min(len(c), len(b)))
        if split < first:
            early.append(f'{key[0]}/{key[1]}/{key[2]}@{split}')
    return {'identical_games': identical, 'games': len(cand), 'left_before_first_switch': early}


def evaluate(runs: list[dict], solved: dict) -> dict:
    for run in runs:
        check_run(run)
    by_arm = {a: sorted(r['seed'] for r in runs if r['arm'] == a) for a in (BASELINE, CANDIDATE)}
    if by_arm[BASELINE] != list(SEEDS) or by_arm[CANDIDATE] != list(SEEDS):
        raise ValueError(f'both arms must cover exactly the seeds {SEEDS}, once each')
    for seed in SEEDS:
        openings = [r['openings'] for r in runs if r['seed'] == seed]
        if openings[0] != openings[1]:
            raise ValueError(f'seed {seed}: the two arms did not play the same openings')
    arms = merge(runs, OPPONENT)
    cand, base = e2._games(arms[CANDIDATE]), e2._games(arms[BASELINE])
    if set(cand) != set(base):
        raise ValueError('the two arms did not play the same openings and colours')
    rows = positions(runs)
    missing = {history_key(r['history']) for r in rows} - set(solved)
    if missing:
        raise ValueError(f'{len(missing)} positions are not solved')
    tables = {arm: summarize(a) for arm, a in arms.items()}
    safety = {arm: len(t['safety_violations']) for arm, t in tables.items()}
    counts = frequency_counts(rows, solved)
    gates = {'strength': strength(cand, base), 'cost': e2.cost(cand, base),
             'soundness': soundness(rows, solved, safety), 'frequency': frequency(counts),
             'isolation': {'verdict': 'PASS', 'checked': ['configuration', 'runtime identities', 'canonical factory',
                                                         'candidate sources', 'attack step records'],
                           'off_path': 'tests.test_tb_v2_value_e4v2'}}
    failed = [k for k, g in gates.items() if g['verdict'] != 'PASS']
    verdict = {'decision': 'REJECT' if failed else 'ADOPT',
               'reasons': [f'{k} {gates[k]["verdict"]}' for k in failed] or ['every gate PASS']}
    minutes = {arm: round(sum(g['game_seconds'] for g in a['games']) / 60, 1) for arm, a in arms.items()}
    return {'seeds': list(SEEDS), 'arms': tables, 'gates': gates, 'verdict': verdict,
            'report': {'game_minutes': minutes, 'e4v2_steps': steps(arms[CANDIDATE]['games']),
                       'isolation': isolation_report(cand, base),
                       'frequency_per_opening': {f'{k[0]}/{k[1]}': v for k, v in sorted(counts.items())}}}


def read_solved(paths: list[Path], fp: dict) -> dict:
    done = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line: that position is solved again
            if row.get('fingerprint') != fp:
                raise ValueError(f'{path} has positions of another reference / checker / budget; use another file')
            done[row['key']] = row
    return done


def solve_all(tasks: list[dict], fast: str, workers: int, jsonl: Path | None, fp: dict, merge_from=(),
              log=print) -> dict:
    """Every task solved: the ones in ``jsonl`` / ``merge_from`` (same fingerprint and checks) kept, the others
    run and appended to ``jsonl``."""
    from hybrid.progress import Progress, run_chunks
    from scripts.h7_timing import restore_fast_mode

    done = read_solved([*merge_from, *([jsonl] if jsonl is not None else [])], fp)
    todo = [t for t in tasks if t['key'] not in done or (done[t['key']]['played'], done[t['key']]['attack'])
            != (t['played'], t['attack'])]
    log(f'[h7-e4v2-arena] {len(tasks)} positions, {len(tasks) - len(todo)} solved, workers {workers}', flush=True)

    def keep(_, rows):
        for row in rows:
            row = {**row, 'fingerprint': fp}
            done[row['key']] = row
            if jsonl is not None:
                jsonl.parent.mkdir(parents=True, exist_ok=True)
                with jsonl.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')

    progress = Progress('h7-e4v2-arena', len(todo), 'positions', log=log)
    chunks = [[t] for t in todo]
    if workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=_init,
                                 initargs=(fast,)) as pool:
            run_chunks(pool, _chunk, chunks, progress, len, unit_name='positions', on_result=keep)
    else:
        with restore_fast_mode():
            _init(fast)
            for i, c in enumerate(chunks):
                keep(i, _chunk(c))
                progress.update(1)
    return {t['key']: done[t['key']] for t in tasks}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--reference-result', type=Path, default=ROOT / REFERENCE_RESULT)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--fast', choices=('off', 'fast4'), default='fast4')
    parser.add_argument('--jsonl', type=Path, help='one line per solved position (resume)')
    parser.add_argument('--merge', nargs='+', type=Path, default=[], help='solved positions of other machines (read)')
    parser.add_argument('--shard', help='K/N: solve only every N-th distinct position from the K-th, no summary')
    parser.add_argument('--plan', action='store_true', help='check the runs and count the positions only')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from analysis import tb_v2_value_e4v2 as cand
    from analysis.track_b_guard import check_code
    from scripts.run_mcts_v8_benchmark import _git_commit, _git_dirty, file_sha256

    runs = [json.loads(p.read_text(encoding='utf-8')) for p in args.runs]
    try:
        check_reference(json.loads(args.reference_result.read_text(encoding='utf-8')))
        cand.check_sources()  # the candidate files, the code manifest, the base engine: still the frozen ones
        bad = check_code()
        if bad:
            raise ValueError(f'the canonical code differs from its manifest: {bad}')
        for run in runs:
            check_run(run)
        rows = positions(runs)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    tasks = tasks_of(rows)
    shard = None
    if args.shard is not None:
        try:
            k, n = (int(x) for x in args.shard.split('/'))
            assert 0 <= k < n
        except (ValueError, AssertionError):
            parser.error('--shard is K/N with 0 <= K < N')
        if args.jsonl is None or args.output is not None:
            parser.error('--shard writes --jsonl only (no --output)')
        shard = tasks[k::n]
    print(f"[h7-e4v2-arena] {len(runs)} runs, {sum(len(r['games']) for r in runs)} games, {len(rows)} own moves "
          f"({sum(r['event_position'] for r in rows)} after a {OPPONENT} move, {sum(r['attack'] for r in rows)} "
          f'attack wins), {len(tasks)} distinct positions', flush=True)
    if args.plan:
        return 0
    fp = fingerprint(args.fast)
    try:
        if shard is not None:
            solve_all(shard, args.fast, args.workers, args.jsonl, fp, args.merge)
            print(f'[h7-e4v2-arena] shard {args.shard}: {len(shard)} positions solved into {args.jsonl}', flush=True)
            return 0
        solved = solve_all(tasks, args.fast, args.workers, args.jsonl, fp, args.merge)
        result = evaluate(runs, solved)
    except ValueError as exc:
        parser.error(str(exc))
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'git_dirty': _git_dirty(), 'rules': RULES,
               'runs': {str(p): file_sha256(p) for p in args.runs},
               'reference_result_sha256': file_sha256(args.reference_result), 'fingerprint': fp,
               'candidate_hash': cand.candidate_hash(), **result,
               'positions': {k: {f: v[f] for f in ('reference', 'checks')} for k, v in sorted(solved.items())}}
    print(json.dumps({'gates': {k: {f: g.get(f) for f in ('verdict', 'diff', 'ci95', 'ratio', 'rate', 'miss')
                                    if f in g} for k, g in result['gates'].items()},
                      'verdict': result['verdict']}, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
