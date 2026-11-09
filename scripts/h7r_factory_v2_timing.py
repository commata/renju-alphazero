"""Candidate factory v2, time gate (R2-D1-AUTOPILOT-1): the widened value head through factory v2 against the
1-channel build (factory v1), on the same positions and under the same conditions.

    reference (A)  factory v1 (``tb_v3_net``) with the 1-channel value checkpoint
    candidate (B)  factory v2 (``tb_v3_net_v2``) with the widened value checkpoint
    positions      the first ``OPENINGS`` openings of the committed O2 suite; from each, ``PLIES`` positions that
                   follow A's own moves, so A and B search exactly the same positions (each on a copy)
    order          A first on even (opening + ply), B first otherwise; one warm-up move per agent, not counted
    gate           mean(B) <= 1.10 x mean(A) and p95(B) <= 1.20 x p95(A) (per-move wall-clock seconds)

The protocol (suite, counts, seeds, thresholds) is fixed here before any full-scale result is read; the record
(``h7r-factory-v2-timing-v1``) is written once with the runtime settings, the model identities and every timing.

    python scripts/h7r_factory_v2_timing.py --policy runs/h3_policy_64x4/best.pt \\
        --narrow-value runs/h7r/R2-D1-HEAD-WIDTH-1/full/narrow-31001/best.pt \\
        --wide-value runs/h7r/R2-D1-HEAD-WIDTH-1/full/wide-31001/best.pt --device cuda \\
        --output docs/h7r/r2/R2-D1-HEAD-WIDTH-1/factory-v2/timing.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FORMAT = 'h7r-factory-v2-timing-v1'
SUITE = 'docs/h7r/suites/O2.json'
OPENINGS = 24
PLIES = 6
SEED_BASE = 20261109
MEAN_RATIO = 1.10
P95_RATIO = 1.20


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def measure(make_a, make_b, openings: list[dict], plies: int, seed_base: int, clock=time.perf_counter,
            log=print) -> list[dict]:
    """Per position: both agents' wall-clock seconds and moves; positions follow A's moves."""
    from renju import Game

    rows = []
    warmed = False
    for k, opening in enumerate(openings):
        seed = seed_base + opening['index']
        a, b = make_a(seed), make_b(seed)
        game = Game()
        for move in opening['moves']:
            game.play(*move)
        if not warmed:  # first CUDA / cache work, not counted
            a.select_move(copy.deepcopy(game))
            b.select_move(copy.deepcopy(game))
            a, b = make_a(seed), make_b(seed)
            warmed = True
        for ply in range(plies):
            if game.winner is not None or not game.legal_moves():
                break
            order = (('a', a), ('b', b)) if (k + ply) % 2 == 0 else (('b', b), ('a', a))
            out = {}
            for name, agent in order:
                position = copy.deepcopy(game)
                started = clock()
                move = agent.select_move(position)
                out[name] = (clock() - started, [int(move[0]), int(move[1])])
            rows.append({'opening': opening['index'], 'ply': ply, 'first': order[0][0],
                         'a_seconds': out['a'][0], 'b_seconds': out['b'][0],
                         'a_move': out['a'][1], 'b_move': out['b'][1]})
            game.play(*out['a'][1])
        log(f'[factory-v2-timing] opening {k + 1}/{len(openings)}: {len(rows)} positions', flush=True)
    return rows


def gate(rows: list[dict]) -> dict:
    a = [r['a_seconds'] for r in rows]
    b = [r['b_seconds'] for r in rows]
    stats = {'positions': len(rows), 'mean_a': sum(a) / len(a), 'mean_b': sum(b) / len(b),
             'p95_a': p95(a), 'p95_b': p95(b)}
    stats['mean_ratio'] = stats['mean_b'] / stats['mean_a']
    stats['p95_ratio'] = stats['p95_b'] / stats['p95_a']
    stats['thresholds'] = {'mean_ratio': MEAN_RATIO, 'p95_ratio': P95_RATIO}
    stats['passed'] = stats['mean_ratio'] <= MEAN_RATIO and stats['p95_ratio'] <= P95_RATIO
    stats['same_move_share'] = sum(r['a_move'] == r['b_move'] for r in rows) / len(rows)
    return stats


def run(args, log=print) -> dict:
    import h7c_arena as ar
    from analysis import tb_v3_net, tb_v3_net_v2 as v2
    from hybrid import h7r_d1 as d1
    from hybrid.h7r_ledger import publish_json

    out = Path(args.output)
    if out.exists():
        raise ValueError(f'{out} exists: never overwritten')
    git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()  # noqa: E731
    state = {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}
    if state['dirty']:
        raise ValueError('the time gate needs a clean tracked tree')
    suite = json.loads((ROOT / SUITE).read_text(encoding='utf-8'))
    openings = suite['openings'][:OPENINGS]
    a_side = {'policy': args.policy, 'value': args.narrow_value}
    b_side = {'policy': args.policy, 'value': args.wide_value}
    ar.install_runtime()
    started = time.monotonic()
    rows = measure(lambda seed: ar.make_side(a_side, seed, args.device),
                   lambda seed: v2.make_side(b_side, seed, args.device), openings, PLIES, SEED_BASE, log=log)
    result = gate(rows)
    if git('rev-parse', 'HEAD') != state['commit']:
        raise ValueError('HEAD changed during the time gate')
    record = {'format': FORMAT, 'git': state, 'settings': d1.numeric_settings(args.device),
              'protocol': {'suite': SUITE, 'suite_sha256': _sha(ROOT / SUITE), 'openings': OPENINGS, 'plies': PLIES,
                           'seed_base': SEED_BASE, 'order': 'alternating per position', 'warm_up': 'one move each'},
              'reference': {'factory': 'tb-v3-net (v1)', 'models': tb_v3_net.model_identity(*a_side.values())},
              'candidate': {'factory': v2.FACTORY_VERSION, 'models': tb_v3_net.model_identity(*b_side.values())},
              'result': result, 'rows': rows, 'minutes': round((time.monotonic() - started) / 60, 2)}
    publish_json(out, record)
    return record


def _sha(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--policy', required=True)
    parser.add_argument('--narrow-value', required=True)
    parser.add_argument('--wide-value', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        out = run(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    r = out['result']
    print(f"[factory-v2-timing] {'PASS' if r['passed'] else 'FAIL'}: mean x{r['mean_ratio']:.3f} (<= {MEAN_RATIO}), "
          f"p95 x{r['p95_ratio']:.3f} (<= {P95_RATIO}); {r['positions']} positions; same move "
          f"{r['same_move_share']:.0%}", flush=True)
    return 0 if r['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
