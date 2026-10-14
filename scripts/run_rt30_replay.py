"""Replay recorded E3-A positions with TB-Engine-v1: TB1 vs TB1-fast equivalence and timing (docs/web-ser5.md §5).

Each selected position (the board before a V8 move of the E3-A games) is searched by a fresh
agent with a fixed seed, once per arm:

    frozen  TB-Engine-v1 as frozen
    fast    the same engine with ``rt30.fast_windows`` installed (exact speed-up)
    rt      the TB-RT30 search (``rt30.engine``) with the fast windows; ``--deadline S`` gives every
            move an absolute deadline S seconds after its start, else it runs without one
    sup     TB-RT30 through ``rt30.supervisor`` (engine worker process, IPC, watchdog), as the web
            server runs it: ``seconds`` is the wall time from the request; needs ``--deadline`` and
            is the only arm of its run (``--watchdog``, default 29)

With two or more arms the run is the equivalence check: every arm must give the first arm's move
and diagnostics (every field except the ``*_seconds`` timings); ``rt`` without ``--deadline`` must
match ``frozen``. With one arm it is a timing benchmark (the mini-PC baseline; ``rt --deadline
27.5`` is the latency run). A replay does not reproduce the recorded move: the recorded
agent's random stream had advanced through the game, the replay starts a fresh one.

The tree route needs the H3 policy (``--checkpoint``, checked against the frozen hashes). Without
it, ``--uniform-prior`` runs the TB1 configuration with ``puct_prior='uniform'``: a valid
equivalence check of the tactical modules, but not TB1 itself (the output says so).

    python scripts/run_rt30_replay.py --checkpoint runs/h3_policy_64x4/best.pt --select slow `
        --workers 4 --output runs/rt30/equivalence_slow.json 2>&1 | Tee-Object runs/rt30/equivalence_slow.log
    python scripts/run_rt30_replay.py --checkpoint runs/h3_policy_64x4/best.pt --arms frozen `
        --select sample --sample 300 --workers 1 --output runs/rt30/baseline_frozen.json
    python scripts/run_rt30_replay.py --checkpoint runs/h3_policy_64x4/best.pt --arms rt --deadline 27.5 `
        --select slow --workers 1 --output runs/rt30/rt_deadline_slow.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import Random
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from hybrid.progress import Progress, run_chunks  # noqa: E402

E3 = ROOT / 'docs' / 'mcts-v8-results' / 'e3'
DEFAULT_JSONL = [E3 / 'puct_policy_vct2_50k_8415.jsonl', E3 / 'puct_policy_vct2_50k_8416.jsonl']
ARMS = ('frozen', 'fast', 'rt', 'sup')
PHASES = {'attack': 'v8_attack_seconds', 'stage_vct': 'v8_vct_seconds', 'tree': 'v8_tree_seconds',
          'root': 'v8_root_seconds', 'vct2': 'v8_vct2_seconds', 'v7_modules': 'v7_module_seconds'}
# ``v7_module_seconds`` runs from before the own VCF to the end of the V7 root preparation, so it
# contains V8-B; ``v7_only`` (= v7_modules - attack) is the frozen V7 work that no deadline cuts.
DEADLINE = 27.5

_WORKER: dict = {}


def load_positions(paths, select: str, slow_seconds: float, sample: int, seed: int, limit: int | None, keys=None):
    rows = []
    for path in paths:
        with Path(path).open(encoding='utf-8') as handle:
            for line_no, line in enumerate(handle):
                record = json.loads(line)
                for move in record['v8_moves']:
                    rows.append({'key': f"{Path(path).stem}/{line_no}/{move['ply']}", 'moves': record['moves'][:move['ply']],
                                 'recorded_seconds': move['seconds'], 'recorded_route': move['route']})
    if keys:
        wanted = set(keys)
        rows = [r for r in rows if r['key'] in wanted]
        if len(rows) != len(wanted):
            raise SystemExit(f'--keys not found: {sorted(wanted - {r["key"] for r in rows})}')
    elif select == 'slow':
        rows = [r for r in rows if r['recorded_seconds'] > slow_seconds]
    elif select == 'sample':
        rows = Random(seed).sample(rows, min(sample, len(rows)))
    return rows[:limit] if limit else rows


def _init_worker(checkpoint, uniform_prior, deadline, timing):
    from analysis.tb_engine_v1 import TB_ENGINE_V1

    config = dict(TB_ENGINE_V1)
    policy = None
    if checkpoint:
        from analysis.tb_engine_v1 import check_checkpoint
        from hybrid.h4_policy import RootPolicy

        check_checkpoint(checkpoint)
        policy = RootPolicy(checkpoint, threads=1)
    elif uniform_prior:
        config['puct_prior'] = 'uniform'
    else:
        raise ValueError('pass --checkpoint, or --uniform-prior for a policy-free equivalence check')
    _WORKER.update(config=config, policy=policy, deadline=deadline, timing=timing)


def _diagnostics(diag) -> dict:
    return {k: v for k, v in vars(diag).items() if not k.endswith('_seconds')}


def _search(moves, arm: str, seed: int):
    from analysis.mcts_v8_agent import MCTSV8Agent
    from renju import Game
    from rt30 import fast_windows
    from rt30.deadline import Deadline
    from rt30.engine import RT30Agent

    if arm == 'frozen':
        fast_windows.uninstall()
    else:
        fast_windows.install()
    game = Game()
    for move in moves:
        game.play(*move)
    started = perf_counter()
    extra = {}
    if arm == 'rt':
        agent = RT30Agent(seed=seed, root_policy=_WORKER['policy'], timing=_WORKER['timing'], **_WORKER['config'])
        seconds = _WORKER['deadline']
        deadline = Deadline.after(seconds) if seconds else None
        move = agent.select_move(game, deadline=deadline)
        info = agent.rt_info
        extra = {'rt': {'timeouts': info.timeouts, 'skipped': info.skipped, 'emergency': info.emergency,
                        'best': [[stage, list(m)] for stage, m in info.best]}}
    else:
        agent = MCTSV8Agent(seed=seed, root_policy=_WORKER['policy'], **_WORKER['config'])
        move = agent.select_move(game)
    seconds = perf_counter() - started
    if game.board[move[0]][move[1]] != 0 or tuple(move) not in game.legal_moves():
        raise RuntimeError(f'{arm}: illegal move {move}')
    diag = agent.diagnostics
    phases = {name: round(getattr(diag, field), 4) for name, field in PHASES.items()}
    phases['v7_only'] = round(max(0.0, phases['v7_modules'] - phases['attack']), 4) if phases['v7_modules'] else 0.0
    return {'move': list(move), 'seconds': round(seconds, 4), 'route': diag.v8_route, 'phases': phases,
            'nodes': {'attack': diag.v8_attack_nodes, 'stage_vct': diag.v8_vct_nodes, 'root': diag.v8_root_nodes,
                      'vct2': diag.v8_vct2_nodes},
            # Statuses the search proved or left open, per checked move: [[row, col], status]
            'checked': {'stage_vct': _checked(diag.v8_vct_checked), 'root': _checked(diag.v8_root_checked),
                        'vct2': _checked(diag.v8_vct2_checked)},
            'tree_move': list(diag.v8_v7_move) if diag.v8_v7_move else None, **extra}, _diagnostics(diag)


def _checked(pairs) -> list:
    return [[list(m), status] for m, status in pairs or ()]


def run_supervised(positions, args, timing, progress):
    """The sup arm: one supervisor, positions one after another (concurrency 1, as in the server)."""
    from time import monotonic

    from rt30.supervisor import AgentSpec, EngineSupervisor

    kwargs = {'checkpoint': str(args.checkpoint)} if args.checkpoint else {'uniform_prior': True}
    if timing:
        kwargs['timing'] = timing
    sup = EngineSupervisor(AgentSpec('rt30.supervisor:make_rt30_agent', kwargs), engine_deadline=args.deadline,
                           watchdog=args.watchdog)
    rows = []
    try:
        for index, row in enumerate(positions):
            # A fresh game key per position: a fresh agent with the run's seed, as the other arms.
            result = sup.select(game_key=f'{index}:{row["key"]}', seed=args.seed, history=row['moves'],
                                received_at=monotonic())
            diag, rt = result.diagnostics, result.rt
            phases = {name: round(float(diag.get(field) or 0.0), 4) for name, field in PHASES.items()}
            phases['v7_only'] = round(max(0.0, phases['v7_modules'] - phases['attack']), 4) if phases['v7_modules'] else 0.0
            rows.append({'key': row['key'], 'recorded_seconds': row['recorded_seconds'],
                         'recorded_route': row['recorded_route'],
                         'sup': {'move': list(result.move), 'seconds': round(result.wall_seconds, 4),
                                 'route': diag.get('v8_route'), 'phases': phases,
                                 'source': result.source, 'watchdog': result.watchdog,
                                 'worker_restarted': result.worker_restarted, 'error': result.error,
                                 'queue_seconds': result.queue_seconds, 'engine_seconds': result.engine_seconds,
                                 'checked': {k: diag.get(f) or [] for k, f in (('stage_vct', 'v8_vct_checked'),
                                                                                ('root', 'v8_root_checked'),
                                                                                ('vct2', 'v8_vct2_checked'))},
                                 'tree_move': diag.get('v8_v7_move'),
                                 'rt': {'timeouts': rt.get('timeouts', []), 'skipped': rt.get('skipped', []),
                                        'emergency': rt.get('emergency'), 'best': result.best}}})
            progress.update(1)
    finally:
        sup.close()
    return rows


def _first_difference(a: dict, b: dict):
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            return key
    return None


def run_chunk(chunk):
    rows_out = []
    for row in chunk['rows']:
        out = {'key': row['key'], 'recorded_seconds': row['recorded_seconds'], 'recorded_route': row['recorded_route']}
        diags = {}
        for arm in chunk['arms']:
            out[arm], diags[arm] = _search(row['moves'], arm, chunk['seed'])
        if len(chunk['arms']) >= 2:
            first = chunk['arms'][0]
            out['identical'] = True
            for arm in chunk['arms'][1:]:
                if out[arm]['move'] != out[first]['move'] or diags[arm] != diags[first]:
                    out['identical'] = False
                    out['first_difference'] = (arm, 'move' if out[arm]['move'] != out[first]['move']
                                               else _first_difference(diags[first], diags[arm]))
                    break
        rows_out.append(out)
    return rows_out


def _dist(values):
    values = sorted(values)
    if not values:
        return {}
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {'n': len(values), 'median': round(statistics.median(values), 3), 'p95': round(pick(0.95), 3),
            'p99': round(pick(0.99), 3), 'max': round(values[-1], 3), 'sum': round(sum(values), 1),
            f'over_{DEADLINE}s': sum(v > DEADLINE for v in values)}


def summarize(rows, arms):
    summary = {'positions': len(rows)}
    for arm in arms:
        summary[arm] = {'seconds': _dist([r[arm]['seconds'] for r in rows]),
                        'routes': dict(Counter(r[arm]['route'] for r in rows)),
                        'phases': {name: _dist([r[arm]['phases'][name] for r in rows if r[arm]['phases'][name] > 0])
                                   for name in (*PHASES, 'v7_only')}}
    if 'sup' in arms:
        sup = [r['sup'] for r in rows]
        summary['sup_wall'] = {'sources': dict(Counter(x['source'] for x in sup)),
                               'watchdogs': sum(x['watchdog'] for x in sup),
                               'over_30s': sum(x['seconds'] >= 30.0 for x in sup),
                               'ipc_overhead': _dist([x['seconds'] - x['engine_seconds'] for x in sup
                                                      if x['engine_seconds'] is not None])}
    for arm in ('rt', 'sup'):
        if arm not in arms:
            continue
        rt = [r[arm]['rt'] for r in rows]
        # Not summary['rt']: that key holds the rt arm's seconds and phases.
        summary['rt_deadline'] = {'timeouts': {p: sum(p in x['timeouts'] for x in rt) for p in ('attack', 'stage_vct', 'root', 'vct2')},
                         'skipped': {p: sum(p in x['skipped'] for x in rt) for p in ('puct', 'vct2')},
                         'moves_with_timeout_or_skip': sum(bool(x['timeouts'] or x['skipped']) for x in rt)}
    if len(arms) >= 2:
        summary['identical'] = sum(r['identical'] for r in rows)
        summary['different'] = [{'key': r['key'], 'first_difference': r['first_difference']}
                                for r in rows if not r['identical']]
        total = {arm: sum(r[arm]['seconds'] for r in rows) for arm in arms}
        summary['speedup_vs_first'] = {arm: round(total[arms[0]] / total[arm], 3) if total[arm] else None
                                       for arm in arms[1:]}
    return summary


def compare_baseline(rows, arm: str, path: Path, baseline_arm: str, seed: int) -> dict:
    """Moves and routes of ``arm`` against ``baseline_arm`` of an earlier run on the same positions.

    ``lost_own_vct``: the baseline proved a VCT1 win (route own_vct) and this run played another
    route; ``same_move`` tells whether the move still was the proven one."""
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    base = {r['key']: r[baseline_arm] for r in report['rows'] if baseline_arm in r}
    if report.get('seed') != seed:
        # The PUCT rollouts follow the agent seed: other seeds give other tree moves.
        raise SystemExit(f"--baseline was run with --seed {report.get('seed')}, this run uses {seed}")
    common = [r for r in rows if r['key'] in base]
    changed, lost = [], []
    for r in common:
        a, b = base[r['key']], r[arm]
        if a['move'] != b['move'] or a['route'] != b['route']:
            changed.append({'key': r['key'], 'baseline': [a['route'], a['move']], 'run': [b['route'], b['move']],
                            'kind': _change_kind(a, b),
                            'played_in_baseline': _baseline_status(a, b['move'])})
        if a['route'] == 'own_vct' and b['route'] != 'own_vct':
            lost.append({'key': r['key'], 'win': a['move'], 'played': b['move'], 'same_move': a['move'] == b['move']})
    kinds = Counter(c['kind'] for c in changed)
    return {'baseline': str(path), 'baseline_arm': baseline_arm, 'positions': len(common),
            'missing': len(rows) - len(common), 'same_move': len(common) - sum(c['baseline'][1] != c['run'][1]
                                                                                for c in changed),
            'changed_kinds': dict(kinds), 'changed': changed, 'lost_own_vct': lost,
            'lost_own_vct_other_move': sum(not x['same_move'] for x in lost),
            'played_baseline_unsafe': [c['key'] for c in changed if c['played_in_baseline'] == 'UNSAFE']}


def _baseline_status(base: dict, move) -> str | None:
    """The worst status the baseline's own checks gave ``move`` (UNSAFE beats UNKNOWN beats SAFE)."""
    found = [status for phase in ('stage_vct', 'root', 'vct2') for m, status in base.get('checked', {}).get(phase, [])
             if m == list(move)]
    for status in ('UNSAFE', 'UNKNOWN', 'SAFE', 'NO_TARGETED_VCT2_FOUND'):
        if status in found:
            return status
    return None


def _change_kind(base: dict, run: dict) -> str:
    """Why a move changed against the baseline (first matching rule).

    lost_win             baseline proved a VCT1 win, this run did not
    played_baseline_unsafe  this run played a move the baseline's checks proved lost
    kept_tree_move       this run played the tree move the baseline's checks replaced
    route_only           same move, other route
    other                anything else (another tree move, another fallback)
    """
    if base['route'] == 'own_vct' and run['route'] != 'own_vct':
        return 'lost_win'
    if base['move'] == run['move']:
        return 'route_only'
    if _baseline_status(base, run['move']) == 'UNSAFE':
        return 'played_baseline_unsafe'
    if base.get('tree_move') == run['move']:
        return 'kept_tree_move'
    return 'other'


def _commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--jsonl', type=Path, nargs='+', default=DEFAULT_JSONL)
    parser.add_argument('--select', choices=('all', 'slow', 'sample'), default='sample')
    parser.add_argument('--slow-seconds', type=float, default=30.0, help='--select slow: recorded seconds above this')
    parser.add_argument('--sample', type=int, default=200)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--keys', nargs='+', help='replay exactly these positions (keys as in the output rows)')
    parser.add_argument('--keys-file', type=Path,
                        help='replay the positions of a committed manifest (docs/rt30-eval/*.json, or a JSON list)')
    parser.add_argument('--seed', type=int, default=42, help='sample seed and agent seed')
    parser.add_argument('--arms', default='frozen,fast', help="comma list of frozen, fast, rt (2+: equivalence)")
    parser.add_argument('--deadline', type=float, help='rt arm: seconds from the start of each move (none: no deadline)')
    parser.add_argument('--watchdog', type=float, default=29.0, help='sup arm: watchdog seconds')
    parser.add_argument('--timing', nargs='*', default=[], metavar='KEY=SECONDS',
                        help='rt arm: override rt30.engine.RT30_TIMING, e.g. attack_seconds=8')
    parser.add_argument('--checkpoint', type=Path, help='H3 policy checkpoint of TB-Engine-v1')
    parser.add_argument('--uniform-prior', action='store_true', help='no checkpoint: uniform PUCT prior (not TB1)')
    parser.add_argument('--baseline', type=Path,
                        help='earlier replay JSON on the same positions: report move/route changes against it')
    parser.add_argument('--baseline-arm', default='fast', help='arm of --baseline to compare with (default fast)')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--chunk', type=int, default=1, help='positions per task')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    arms = tuple(a for a in args.arms.split(',') if a)
    if 'sup' in arms and (arms != ('sup',) or not args.deadline):
        parser.error("--arms sup runs alone and needs --deadline")
    if not arms or set(arms) - set(ARMS):
        parser.error(f'--arms: one or both of {ARMS}')
    if args.checkpoint is None and not args.uniform_prior:
        parser.error('pass --checkpoint, or --uniform-prior for a policy-free equivalence check')

    if args.baseline is not None:
        base_seed = json.loads(args.baseline.read_text(encoding='utf-8')).get('seed')
        if base_seed != args.seed:
            parser.error(f'--baseline was run with --seed {base_seed}; pass the same --seed')
    from rt30.engine import RT30_TIMING

    timing = {}
    for item in args.timing:
        key, _, value = item.partition('=')
        if key not in RT30_TIMING or not value:
            parser.error(f'--timing {item!r}: use KEY=SECONDS with KEY in {sorted(RT30_TIMING)}')
        timing[key] = float(value)
    manifest = None
    if args.keys_file is not None:
        if args.keys:
            parser.error('--keys and --keys-file are exclusive')
        sys.path.insert(0, str(ROOT / 'scripts'))
        from build_rt30_eval_set import load_keys

        args.keys, version = load_keys(args.keys_file)
        manifest = {'path': str(args.keys_file), 'version': version, 'keys': len(args.keys)}
    positions = load_positions(args.jsonl, args.select, args.slow_seconds, args.sample, args.seed, args.limit,
                               args.keys)
    if args.keys:
        order = {key: i for i, key in enumerate(args.keys)}  # the manifest's order, not the files'
        positions.sort(key=lambda r: order[r['key']])
    chunks = [{'rows': positions[i:i + args.chunk], 'arms': arms, 'seed': args.seed}
              for i in range(0, len(positions), args.chunk)]
    print(f'positions={len(positions)} arms={",".join(arms)} workers={args.workers} '
          f'engine={"TB-Engine-v1" if args.checkpoint else "TB1 config, uniform prior"}', flush=True)
    progress = Progress('rt30-replay', len(positions), 'positions')
    init = (str(args.checkpoint) if args.checkpoint else None, args.uniform_prior, args.deadline, timing)
    if arms == ('sup',):
        rows = run_supervised(positions, args, timing, progress)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=init) as pool:
            results = run_chunks(pool, run_chunk, chunks, progress, lambda c: len(c['rows']))
        rows = [row for chunk in results for row in chunk]
    summary = summarize(rows, arms)
    if args.baseline is not None:
        summary['vs_baseline'] = compare_baseline(rows, arms[-1], args.baseline, args.baseline_arm, args.seed)
    report = {'format': 'rt30-replay-v1', 'engine': 'TB-Engine-v1' if args.checkpoint else 'tb1-uniform-prior',
              'arms': arms, 'manifest': manifest, 'deadline': args.deadline, 'timing': {**RT30_TIMING, **timing}, 'select': args.select, 'slow_seconds': args.slow_seconds, 'seed': args.seed,
              'workers': args.workers, 'git_commit': _commit(), 'jsonl': [str(p) for p in args.jsonl],
              'machine': {'platform': platform.platform(), 'processor': platform.processor(),
                          'cpu_count': os.cpu_count(), 'python': platform.python_version()},
              'summary': summary, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)
    if len(arms) >= 2 and summary['identical'] != len(rows):
        print(f'NOT IDENTICAL: {len(rows) - summary["identical"]} position(s)', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
