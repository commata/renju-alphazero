"""fast5 G0 (X2-F): every VCF call of real solver runs, answered by the frozen search and checked against the C core.

Shadow mode: the frozen Python searches (on fast4) keep answering the engine; at every call the native core
(``rt30.fast5``) is run on the same board and must return the same result, witness, nodes and cut, and for the
V8-C table the same hits and stores; the board must be unchanged. A mismatch is recorded with its input.

Workloads (per position, in order; positions from h7_timing / COSTTEST style files: ``{'positions': [{'id',
'moves'}]}``):
    e4      the E4-v2 step (``e4v2_attack.decide`` with route 'tree': the opponent-VCF gate and the attack)
    base    the TB-V3 move (TB-V2-Value's V7 / V8-A / V8-B / V8-C with its table / VCT2, then the E4-v2 step):
            needs the models (``--stand-in``: the torch-free test models, for a smoke run without them)
    ref     the frozen 4M reference solver (``e4_reference.reference_attack``; slow, a few positions)

    python -u scripts/fast5_g0.py --positions docs/h7/h7_timing_positions.json --parts e4 base --workers 8 \\
        --jsonl runs/tb_v3/fast5_g0.jsonl --output docs/h7/tb_v3_fast5_g0.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FORMAT = 'tb-v3-fast5-g0-v1'
PARTS = ('e4', 'base', 'ref')
SEED = 8707
_WORKER: dict = {}


def _shadow_find(stats):
    from rt30 import fast5

    def find(game, attacker, *, max_fours, node_limit):
        board = [row[:] for row in game.board]
        frozen = fast5.FROZEN(game, attacker, max_fours=max_fours, node_limit=node_limit)
        native = fast5.find_vcf_with_stats(game, attacker, max_fours=max_fours, node_limit=node_limit)
        _compare(stats, 'plain', board, game, attacker, max_fours, node_limit, frozen, native)
        return frozen
    return find


def _shadow_make_find(stats):
    from analysis import vcf_tt_v1
    from rt30 import fast5

    def make_find(table_stats=None):
        frozen_find = fast5.FROZEN_MAKE_FIND(table_stats)

        def find(game, attacker, *, max_fours, node_limit):
            board = [row[:] for row in game.board]
            before = (table_stats.hits, table_stats.stores) if table_stats is not None else None
            frozen = frozen_find(game, attacker, max_fours=max_fours, node_limit=node_limit)
            mine = vcf_tt_v1.TableStats()
            native = fast5.make_find(mine)(game, attacker, max_fours=max_fours, node_limit=node_limit)
            extra = None
            if before is not None:
                extra = ((table_stats.hits - before[0], table_stats.stores - before[1]), (mine.hits, mine.stores))
            _compare(stats, 'table', board, game, attacker, max_fours, node_limit, frozen, native, extra)
            return frozen
        return find
    return make_find


def _compare(stats, kind, board, game, attacker, max_fours, node_limit, frozen, native, extra=None):
    stats['calls'][kind] += 1
    stats['nodes'][kind] += frozen[1]
    stats['found'][kind] += frozen[0] is not None
    stats['cut'][kind] += bool(frozen[2])
    ok = frozen == native and game.board == board and (extra is None or extra[0] == extra[1])
    if not ok:
        stats['mismatches'].append({'kind': kind, 'board': board, 'attacker': attacker, 'max_fours': max_fours,
                                    'node_limit': node_limit, 'frozen': repr(frozen), 'native': repr(native),
                                    'table': extra})


def install_shadow(stats) -> None:
    """Rebind the two searches to the shadow pair in every loaded module (fast4 stays underneath)."""
    from rt30 import fast5

    find, make_find = _shadow_find(stats), _shadow_make_find(stats)
    for module, name in fast5._bindings(fast5.FROZEN):
        setattr(module, name, find)
    for module, name in fast5._bindings(fast5.FROZEN_MAKE_FIND):
        setattr(module, name, make_find)
    _WORKER['shadow'] = (find, make_find)


def uninstall_shadow() -> None:
    from rt30 import fast5

    find, make_find = _WORKER.pop('shadow')
    for module, name in fast5._bindings(find):
        setattr(module, name, fast5.FROZEN)
    for module, name in fast5._bindings(make_find):
        setattr(module, name, fast5.FROZEN_MAKE_FIND)


def _init(parts, policy_checkpoint, value_checkpoint, device, stand_in) -> None:
    from analysis import tb_v3_runtime

    tb_v3_runtime.install()  # fast4 underneath, as the engine runs
    _WORKER.update(parts=parts, policy_checkpoint=policy_checkpoint, value_checkpoint=value_checkpoint,
                   stand_in=stand_in)
    if 'base' in parts:
        if stand_in:
            from tests.test_h7_timing import FACTORIES
            _WORKER.update(policy=FACTORIES['policy'], leaf=FACTORIES['leaf'])
        else:
            from scripts.run_mcts_v8_benchmark import load_policy, load_value
            _WORKER.update(policy=load_policy(policy_checkpoint), leaf=load_value(value_checkpoint, device))


def position_task(task: dict) -> dict:
    from analysis import e4v2_attack, tb_v3, tb_v3_runtime
    from renju import Game
    from scripts.run_mcts_v8_benchmark import derive_seed

    stats = {'calls': {'plain': 0, 'table': 0}, 'nodes': {'plain': 0, 'table': 0}, 'found': {'plain': 0, 'table': 0},
             'cut': {'plain': 0, 'table': 0}, 'mismatches': []}
    install_shadow(stats)
    try:
        game = Game()
        for m in task['moves']:
            game.play(*m)
        w = _WORKER
        if 'e4' in w['parts']:
            e4v2_attack.decide(game, tb_v3.ATTACK, 'tree', None)
        if 'base' in w['parts']:
            seed = derive_seed(SEED, 'fast5-g0', task['id'])
            if w['stand_in']:
                agent = tb_v3_runtime._build(w['policy'], w['leaf'], seed)
            else:
                agent = tb_v3_runtime.make_tb_v3_agent(w['policy'], w['leaf'], seed, w['policy_checkpoint'],
                                                       w['value_checkpoint'])
            agent.select_move(game)
        if 'ref' in w['parts']:
            from analysis.e4_reference import reference_attack
            reference_attack(game, '4m')
    finally:
        uninstall_shadow()
    return {'id': task['id'], **stats}


def _chunk(tasks):
    return [position_task(t) for t in tasks]


def summarize(rows: list[dict]) -> dict:
    total = {k: {kind: sum(r[k][kind] for r in rows) for kind in ('plain', 'table')}
             for k in ('calls', 'nodes', 'found', 'cut')}
    mismatches = [m for r in rows for m in r['mismatches']]
    return {'positions': len(rows), **total, 'mismatches': len(mismatches),
            'verdict': 'G0_PASS' if not mismatches and total['calls']['plain'] else 'G0_FAIL'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--positions', nargs='+', type=Path, required=True)
    parser.add_argument('--parts', nargs='+', choices=PARTS, default=['e4'])
    parser.add_argument('--limit', type=int, help='first N positions of each file')
    parser.add_argument('--stride', type=int, default=1, help='every N-th position')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--value-checkpoint', default=str(ROOT / 'runs/h6_value_head_v3/best.pt'))
    parser.add_argument('--stand-in', action='store_true', help='base with the torch-free test models (smoke)')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--jsonl', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from analysis import tb_v3_fast5
    from hybrid.progress import Progress, run_chunks
    from rt30 import fast5
    from scripts.h7_timing import restore_fast_mode
    from scripts.run_mcts_v8_benchmark import _git_commit, _git_dirty, file_sha256

    if not fast5.available():
        parser.error('python -m rt30.vcfcore.build first (run in src/)')
    tb_v3_fast5.check_sources()
    tasks = []
    for path in args.positions:
        ps = json.loads(path.read_text(encoding='utf-8'))['positions'][::args.stride][:args.limit]
        tasks += [{'id': p['id'], 'moves': p['moves']} for p in ps]
    fp = {'parts': sorted(args.parts), 'stand_in': args.stand_in, 'fast5_sources': tb_v3_fast5.FAST5_SOURCES}
    done = {}
    if args.jsonl is not None and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('fingerprint') != fp:
                parser.error(f'{args.jsonl} has rows of other parts / sources; use another file')
            done[row['id']] = row
    todo = [t for t in tasks if t['id'] not in done]
    print(f'[fast5-g0] {len(tasks)} positions, {len(tasks) - len(todo)} done, parts {args.parts}, '
          f'workers {args.workers}', flush=True)

    def keep(_, rows):
        for row in rows:
            row = {**row, 'fingerprint': fp}
            done[row['id']] = row
            if row['mismatches']:
                print(f"[fast5-g0] MISMATCH at {row['id']}: {len(row['mismatches'])}", flush=True)
            if args.jsonl is not None:
                args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.jsonl.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')

    progress = Progress('fast5-g0', len(todo), 'positions')
    init = (args.parts, args.policy_checkpoint, args.value_checkpoint, args.device, args.stand_in)
    if args.workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'), initializer=_init,
                                 initargs=init) as pool:
            run_chunks(pool, _chunk, [[t] for t in todo], progress, len, unit_name='positions', on_result=keep)
    else:
        with restore_fast_mode():
            _init(*init)
            for i, t in enumerate(todo):
                keep(i, _chunk([t]))
                progress.update(1)
    rows = [done[t['id']] for t in tasks]
    result = summarize(rows)
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'git_dirty': _git_dirty(),
               'positions_files': {str(p): file_sha256(p) for p in args.positions}, 'fingerprint': fp,
               **result, 'mismatch_rows': [m for r in rows for m in r['mismatches']][:50],
               'per_position': [{k: r[k] for k in ('id', 'calls', 'nodes')} for r in rows]}
    print(json.dumps(result, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0 if result['verdict'] == 'G0_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
