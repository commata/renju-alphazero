"""TB-V3 Resolve promotion replay: the new factory must replay the adopted arena candidate move for move.

Approved before running (docs/h6-value.md, "TB-V3 Resolve 승격"):
    input       the E4-v2 arena (``docs/h7/h7_e4v2_arena.json``, the ADOPT of ``tb_v3.ADOPTION``) and its run
                files, recognised by their SHA-256 in that summary: E (``puct_value_vct2_50k_tt_e4v2``, required)
                and B (``puct_value_vct2_50k_tt``, optional)
    replay      per game one agent from the versioned factory with the arena's seed (``derive_seed(seed, 'v8')``):
                E by ``tb_v3_runtime.make_tb_v3_agent``, B by ``track_b_guard.make_tb_v2_agent``. The recorded
                moves are played in order; at every own ply the agent chooses on the recorded history
    compare     move      the chosen move = the recorded move
                decision  route, changed, v7_move (the V8 decision), and of the E4-v2 step base_route, base_move,
                          ran, skip, gate status, proved move, exhausted (the cooldown follows from these)
                record    every other field of the move record except times (reported only)
    gate        E: all 200 games, all 3,201 moves (``report.e4v2_steps.moves`` of the arena), 0 move and 0
                decision mismatches -> PROMOTION_PASS (the canonical alias may move to TB-V3); else
                PROMOTION_FAIL: no promotion, find the cause (the arena's ADOPT stands; the factory is not the
                candidate). B: a reproducibility cross-check, reported (REPRODUCED / DIFFERS), not a gate
G1 of the fast5 runtime (``--runtime fast5``, ``analysis.tb_v3_fast5``, approved before running): the same replay
on fast5, and every record field but times must match too (charged VCF nodes, uncached VCF calls, cuts,
statuses, V8-C table hits / stores, the E4-v2 step): G1_PASS / G1_FAIL. B then runs TB-V2-Value's factory
with the native VCF core bound (a cross-check of the V8-C table path, reported).
CUDA only (the arena's value device). One game per task in a worker pool; each finished game is appended to
``--jsonl`` (resume; a fingerprint binds the gate version, this tool's SHA-256, the arena summary, the runtime hash,
the run file and the device, so a row compared by other rules or code is never reused). An official run needs a
clean tree (``--allow-dirty``: a smoke run, verdict NOT_OFFICIAL).

    python scripts/tb_v3_promotion.py --runs runs/h7_e4v2/e_8441.json runs/h7_e4v2/e_8442.json \\
        --baseline-runs runs/h7_e4v2/b_8441.json runs/h7_e4v2/b_8442.json --workers 8 --device cuda \\
        --jsonl runs/tb_v3/promotion.jsonl --output docs/h7/tb_v3_promotion.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FORMAT = 'tb-v3-promotion-v1'
FAST5_FORMAT = 'tb-v3-fast5-g1-v1'
GATE_VERSION = 2  # 2: G1 for fast5 (record fields), fingerprint with the tool and the arena
ARENA = 'docs/h7/h7_e4v2_arena.json'
ENGINES = {'puct_value_vct2_50k_tt_e4v2': 'TB-V3', 'puct_value_vct2_50k_tt': 'TB-V2-Value'}
MOVE_FIELDS = ('route', 'changed', 'v7_move')
STEP_FIELDS = ('base_route', 'base_move', 'ran', 'skip', 'move', 'exhausted')
_WORKER: dict = {}


def _untimed(value):
    if isinstance(value, dict):
        return {k: _untimed(v) for k, v in value.items() if k != 'seconds'}
    if isinstance(value, list):
        return [_untimed(v) for v in value]
    return value


def decision(record: dict) -> dict:
    out = {k: record.get(k) for k in MOVE_FIELDS}
    step = record.get('e4_attack')
    if step is not None:
        out.update({f'e4.{k}': step.get(k) for k in STEP_FIELDS})
        out['e4.gate'] = (step.get('gate') or {}).get('status')
    return out


def compare(recorded: dict, got: dict) -> dict:
    """The three comparisons of one move: names of the fields that differ."""
    want, have = decision(recorded), decision(got)
    full_want = {k: v for k, v in _untimed(recorded).items() if k != 'ply'}
    full_have = {k: v for k, v in _untimed(got).items() if k != 'ply'}
    return {'move': list(got['played']) != list(recorded['played']),
            'decision': sorted(k for k in want if want[k] != have.get(k)),
            'record': sorted(k for k in set(full_want) | set(full_have) if full_want.get(k) != full_have.get(k))}


def replay_game(task: dict, make_agent) -> dict:
    """Replay one recorded game: the agent chooses at every own ply, the recorded move is then played."""
    from time import perf_counter

    from renju import Game
    from scripts.run_mcts_v8_benchmark import _move_record, derive_seed

    agent = make_agent(task['engine'], derive_seed(task['seed'], 'v8'))
    records = {m['ply']: m for m in task['v8_moves']}
    game, rows, started = Game(), {'move': [], 'decision': [], 'record': []}, perf_counter()
    for ply, move in enumerate(task['moves']):
        recorded = records.get(ply)
        if recorded is not None:
            if list(recorded.get('played') or move) != list(move):
                raise ValueError(f"{task['key']} ply {ply}: the record's move is not the game's move")
            chosen = agent.select_move(game)
            got = _move_record(ply, 0.0, agent.diagnostics)
            if getattr(agent, 'attack_info', None):
                got['e4_attack'] = dict(agent.attack_info)
            got['played'] = list(chosen)
            diff = compare({**recorded, 'played': list(move)}, got)
            if diff['move']:
                rows['move'].append(ply)
            if diff['decision']:
                rows['decision'].append({'ply': ply, 'fields': diff['decision']})
            if diff['record']:
                rows['record'].append({'ply': ply, 'fields': diff['record']})
        game.play(*move)
    return {'key': task['key'], 'engine': task['engine'], 'run_sha256': task['run_sha256'], 'moves': len(records),
            'move_mismatch': rows['move'], 'decision_mismatch': rows['decision'],
            'record_mismatch': rows['record'], 'seconds': round(perf_counter() - started, 1)}


def install_runtime(runtime: str) -> None:
    from analysis import tb_v3_fast5, tb_v3_runtime

    tb_v3_fast5.install() if runtime == 'fast5' else tb_v3_runtime.install()


def _init(policy_checkpoint, value_checkpoint, device, runtime='fast4') -> None:
    from scripts.run_mcts_v8_benchmark import load_policy, load_value

    install_runtime(runtime)
    _WORKER.update(policy=load_policy(policy_checkpoint), leaf=load_value(value_checkpoint, device),
                   policy_checkpoint=policy_checkpoint, value_checkpoint=value_checkpoint, runtime=runtime)


def _factory_agent(engine: str, seed: int):
    from analysis.canonical import factory

    w = _WORKER
    runtime = w.get('runtime', 'fast4') if engine == 'TB-V3' else None
    return factory(engine, runtime)(w['policy'], w['leaf'], seed, w['policy_checkpoint'], w['value_checkpoint'])


def _chunk(tasks):
    return [replay_game(t, _factory_agent) for t in tasks]


def tasks_of(runs: list[tuple[dict, str]]) -> list[dict]:
    out = []
    for run, sha in runs:
        for g in run['games']:
            out.append({'key': g['key'], 'engine': ENGINES[run['arm']], 'seed': g['seed'], 'moves': g['moves'],
                        'v8_moves': g['v8_moves'], 'run_sha256': sha})
    return out


def summarize(rows: list[dict], engine: str) -> dict:
    part = [r for r in rows if r['engine'] == engine]
    return {'games': len(part), 'moves': sum(r['moves'] for r in part),
            'moves_matched': sum(r['moves'] - len(r['move_mismatch']) for r in part),
            'move_mismatches': [f"{r['key']}@{p}" for r in part for p in r['move_mismatch']],
            'decision_mismatches': [{'key': r['key'], **d} for r in part for d in r['decision_mismatch']],
            'record_mismatch_moves': sum(len(r['record_mismatch']) for r in part),
            'record_mismatch_fields': sorted({f for r in part for d in r['record_mismatch'] for f in d['fields']}),
            'seconds': round(sum(r['seconds'] for r in part), 1)}


def verdict(e: dict, expected_games: int, expected_moves: int, b: dict | None, runtime: str = 'fast4') -> dict:
    """fast4: the promotion gate. fast5: G1, which also needs every record field but times equal."""
    reasons = []
    if (e['games'], e['moves']) != (expected_games, expected_moves):
        reasons.append(f"{e['games']} games / {e['moves']} moves replayed, the arena has "
                       f'{expected_games} / {expected_moves}')
    if e['move_mismatches']:
        reasons.append(f"{len(e['move_mismatches'])} moves differ")
    if e['decision_mismatches']:
        reasons.append(f"{len(e['decision_mismatches'])} decision records differ")
    strict = runtime != 'fast4'
    if strict and e['record_mismatch_moves']:
        reasons.append(f"{e['record_mismatch_moves']} move records differ in {e['record_mismatch_fields']}")
    passed, failed = ('G1_PASS', 'G1_FAIL') if strict else ('PROMOTION_PASS', 'PROMOTION_FAIL')
    out = {'verdict': failed if reasons else passed,
           'reasons': reasons or [f'{expected_moves} / {expected_moves} moves and every E4-v2 step equal'
                                  + (', every record field but times equal' if strict else '')]}
    if b is not None:
        clean = b['games'] and not b['move_mismatches'] and not b['decision_mismatches'] \
            and not (strict and b['record_mismatch_moves'])
        out['baseline'] = 'REPRODUCED' if clean else 'DIFFERS'
    return out


def main(argv=None, factories=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs', nargs='+', type=Path, required=True, help='the arena E run files')
    parser.add_argument('--baseline-runs', nargs='*', type=Path, default=[], help='the arena B run files (check)')
    parser.add_argument('--arena', type=Path, default=ROOT / ARENA)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--value-checkpoint', default=str(ROOT / 'runs/h6_value_head_v3/best.pt'))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--runtime', choices=('fast4', 'fast5'), default='fast4',
                        help='fast4: the promotion replay; fast5: G1 of the fast5 runtime')
    parser.add_argument('--jsonl', type=Path, help='one line per replayed game (resume)')
    parser.add_argument('--allow-dirty', action='store_true', help='smoke runs only: the verdict is NOT_OFFICIAL')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from analysis import tb_v2_runtime, tb_v3, tb_v3_fast5, tb_v3_runtime
    from hybrid.progress import Progress, run_chunks
    from scripts.h7_timing import restore_fast_mode
    from scripts.run_mcts_v8_benchmark import _git_commit, _git_dirty, file_sha256
    import summarize_h7_e4 as arena_rules

    arena = json.loads(args.arena.read_text(encoding='utf-8'))
    if file_sha256(args.arena) != tb_v3.ADOPTION['arena_sha256'] or arena.get('verdict', {}).get('decision') != 'ADOPT':
        parser.error(f'{args.arena} is not the adopting arena of {tb_v3.ENGINE_NAME}')
    if factories is None and tb_v3_runtime.device_kind(args.device) != 'cuda':
        parser.error('the arena ran its value on CUDA: the replay must too')
    dirty = _git_dirty()
    if factories is None and dirty is not False and not args.allow_dirty:
        parser.error('the tree has uncommitted changes (or git is unavailable): an official replay needs a clean tree')
    loaded = []
    for path, arm in [(p, arena_rules.CANDIDATE) for p in args.runs] + \
                     [(p, arena_rules.BASELINE) for p in args.baseline_runs]:
        sha, run = file_sha256(path), json.loads(path.read_text(encoding='utf-8'))
        if sha not in arena['runs'].values() or run.get('arm') != arm:
            parser.error(f'{path} is not a {arm} run of {args.arena}')
        try:
            arena_rules.check_run(run)
        except ValueError as exc:
            parser.error(str(exc))
        loaded.append((run, sha))
    if sorted(r['seed'] for r, _ in loaded if r['arm'] == arena_rules.CANDIDATE) != list(arena_rules.SEEDS):
        parser.error(f'--runs must be the E runs of both seeds {arena_rules.SEEDS}')
    tasks = tasks_of(loaded)
    device = tb_v3_runtime.device_kind(args.device)
    if args.runtime == 'fast5':  # B runs its own factory on the native VCF core: marked, not its runtime hash
        hashes = {'TB-V3': tb_v3_fast5.runtime_hash(device),
                  'TB-V2-Value': f"{tb_v2_runtime.runtime_hash(device)}+vcfcore:{tb_v3_fast5.runtime_hash(device)}"}
    else:
        hashes = {'TB-V3': tb_v3_runtime.runtime_hash(device), 'TB-V2-Value': tb_v2_runtime.runtime_hash(device)}
    tool_sha, arena_sha = file_sha256(Path(__file__)), file_sha256(args.arena)

    def fingerprint(engine):
        return {'gate_version': GATE_VERSION, 'tool_sha256': tool_sha, 'arena_sha256': arena_sha,
                'runtime_mode': args.runtime, 'runtime_hash': hashes[engine], 'device': device}

    done = {}
    if args.jsonl is not None and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line: that game is replayed again
            if row.get('fingerprint') != fingerprint(row['engine']):
                parser.error(f'{args.jsonl} has games of another runtime / device; use another file')
            done[(row['key'], row['run_sha256'])] = row
    todo = [t for t in tasks if (t['key'], t['run_sha256']) not in done]
    print(f'[tb-v3-promotion] {len(tasks)} games ({sum(len(t["v8_moves"]) for t in tasks)} own moves), '
          f'{len(tasks) - len(todo)} done, workers {args.workers}, device {args.device}', flush=True)

    def keep(_, rows):
        for row in rows:
            row = {**row, 'fingerprint': fingerprint(row['engine'])}
            done[(row['key'], row['run_sha256'])] = row
            if args.jsonl is not None:
                args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.jsonl.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')

    progress = Progress('tb-v3-promotion', len(todo), 'games')
    if factories is not None:  # tests: stand-in models and an unchecked builder, in this process
        with restore_fast_mode():
            try:
                install_runtime(args.runtime)
                for i, t in enumerate(todo):
                    keep(i, [replay_game(t, factories['build'])])
                    progress.update(1)
            finally:
                _uninstall_fast5()
    elif args.workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'), initializer=_init,
                                 initargs=(args.policy_checkpoint, args.value_checkpoint, args.device,
                                           args.runtime)) as pool:
            run_chunks(pool, _chunk, [[t] for t in todo], progress, len, unit_name='games', on_result=keep)
    else:
        with restore_fast_mode():
            try:
                _init(args.policy_checkpoint, args.value_checkpoint, args.device, args.runtime)
                for i, t in enumerate(todo):
                    keep(i, _chunk([t]))
                    progress.update(1)
            finally:
                _uninstall_fast5()
    rows = [done[(t['key'], t['run_sha256'])] for t in tasks]
    e = summarize(rows, 'TB-V3')
    b = summarize(rows, 'TB-V2-Value') if args.baseline_runs else None
    steps = arena['report']['e4v2_steps']['moves']
    result = verdict(e, arena['arms'][arena_rules.CANDIDATE]['games'], steps, b, args.runtime)
    if dirty is not False and factories is None:
        result = {**result, 'verdict': 'NOT_OFFICIAL', 'reasons': result['reasons'] + ['dirty tree (--allow-dirty)']}
    payload = {'format': FAST5_FORMAT if args.runtime == 'fast5' else FORMAT, 'gate_version': GATE_VERSION,
               'git_commit': _git_commit(), 'git_dirty': dirty, 'engine': tb_v3.ENGINE_NAME,
               'engine_hash': tb_v3.ENGINE_HASH, 'identity_sha256': tb_v3_runtime.identity_sha256(),
               'code_manifest_sha256': tb_v3.CODE_MANIFEST_SHA256, 'runtime_hash': hashes['TB-V3'],
               'baseline_runtime_hash': hashes['TB-V2-Value'], 'device': device, 'runtime_mode': args.runtime,
               'arena_sha256': file_sha256(args.arena), 'runs': {str(p): s for p, (_, s) in
                                                                 zip(args.runs + args.baseline_runs, loaded)},
               'runtime': ((tb_v3_fast5 if args.runtime == 'fast5' else tb_v3_runtime).provenance(args.device)
                           if factories is None else None),
               'rules': {'move_fields': list(MOVE_FIELDS), 'step_fields': list(STEP_FIELDS) + ['gate.status'],
                         'gate': 'E: every game and move of the arena, 0 move and 0 decision mismatches'
                                 + ('; fast5 (G1): and 0 record mismatches (times excluded)'
                                    if args.runtime == 'fast5' else '')},
               **result, 'tb_v3': e, 'tb_v2_value': b, 'rows': rows}
    print(json.dumps({k: payload[k] for k in ('verdict', 'reasons')} | {'baseline': result.get('baseline'),
                     'tb_v3': {k: e[k] for k in ('games', 'moves', 'moves_matched', 'record_mismatch_moves')}},
                     indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0 if result['verdict'] in ('PROMOTION_PASS', 'G1_PASS') else 1


def _uninstall_fast5() -> None:
    from rt30 import fast5

    fast5.uninstall()


if __name__ == '__main__':
    raise SystemExit(main())
