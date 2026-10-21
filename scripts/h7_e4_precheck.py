"""H7-E4 cost pre-check: would the candidate pass the arena cost gate (total move time ratio <= 1.5)?

Run once, after the candidate (``analysis.tb_v2_value_e4``) and the reference solver are frozen, before the
seed 8441 / 8442 arena (docs/h6-value.md, "H7-E4 다음 단계 확정"). It reads no target metric.

    positions   ``docs/h7/h7_timing_positions.json`` (E3 / H6-4b games and probes), minus every position whose
                D4-canonical board (``hybrid.h6_labels.canonical_hash``) is a web corpus position (DEV or
                HOLDOUT); the rest is asserted disjoint from the web corpus
    measure     per position, one process, fresh agents, the canonical CUDA value: the candidate on the frozen
                fast4 profile chooses; its base (TB-V2-Value) move time and the attack time are measured inside
                the same call (the wrapper runs the base first, unchanged), so B = base seconds and
                E = base + attack seconds are paired exactly
    statistic   ratio = sum E / sum B; 95% interval by bootstrap over games (positions of one game resampled
                together, ``BOOTSTRAP`` resamples, ratio of sums recomputed in each)
    rule        lower bound > 1.5  -> PRECHECK_FAIL: the candidate is REJECTED without the arena
                otherwise          -> PROCEED: the arena's cost gate is the verdict

    python scripts/h7_e4_precheck.py --positions docs/h7/h7_timing_positions.json \\
        --corpus docs/h7/h7_e4_web_corpus.json --device cuda --jsonl runs/h7_e4/precheck.jsonl \\
        --output docs/h7/h7_e4_precheck.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import h7_e4_web_corpus as web  # noqa: E402

FORMAT = 'h7-e4-precheck-v1'
RATIO_MAX = 1.5
BOOTSTRAP = 10_000
SEED = 8440


def game_of(position_id: str) -> str:
    """The game a timing position comes from (the cluster): its id without the ply."""
    return position_id.rsplit('|', 1)[0] if position_id.count('|') >= 2 else position_id.split('|', 1)[0]


def split_positions(positions: list[dict], corpus: dict) -> tuple[list[dict], list[str]]:
    """(positions kept, ids dropped): drop every timing position that is a web corpus board (D4)."""
    from hybrid.h6_labels import canonical_hash

    webkeys = {canonical_hash(web.history(corpus, p)) for p in corpus['positions']}
    kept, dropped = [], []
    for p in positions:
        (dropped.append(p['id']) if canonical_hash(p['moves']) in webkeys else kept.append(p))
    assert not any(canonical_hash(p['moves']) in webkeys for p in kept), 'a web position is in the pre-check set'
    return kept, dropped


def measure(task: dict, policy, leaf) -> dict:
    """The candidate on one position: base seconds, attack seconds, moves."""
    from analysis import tb_v2_value_e4 as cand
    from renju import Game
    from scripts.run_mcts_v8_benchmark import derive_seed

    game = Game()
    for m in task['moves']:
        game.play(*m)
    if hasattr(leaf, '_cache'):
        leaf._cache.clear()  # cold per position, as in h7_timing
    agent = cand.make_runtime_agent(policy, leaf, seed=derive_seed(SEED, 'h7-e4-precheck', task['id']))
    base = agent.base
    timed = {}
    inner = base.select_move

    def timed_base(g):
        started = perf_counter()
        move = inner(g)
        timed['base'] = perf_counter() - started
        return move

    base.select_move = timed_base
    started = perf_counter()
    move = agent.select_move(game)
    total = perf_counter() - started
    info = agent.attack_info
    return {'id': task['id'], 'game': game_of(task['id']), 'b_seconds': round(timed['base'], 4),
            'e_seconds': round(total, 4), 'attack_seconds': round(total - timed['base'], 4),
            'route': agent.diagnostics.v8_route, 'move': list(move),
            'base_move': info.get('base_move') or list(move), 'attacked': bool(info),
            'attack_nodes': info.get('nodes', 0), 'attack_calls': info.get('calls', 0)}


def ratio_ci(rows: list[dict]) -> dict:
    games: dict[str, list[dict]] = {}
    for r in rows:
        games.setdefault(r['game'], []).append(r)
    sums = {g: (sum(r['e_seconds'] for r in rs), sum(r['b_seconds'] for r in rs)) for g, rs in games.items()}
    keys = sorted(sums)
    total_e, total_b = sum(e for e, _ in sums.values()), sum(b for _, b in sums.values())
    rng, samples = Random(SEED), []
    for _ in range(BOOTSTRAP):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        b = sum(sums[k][1] for k in pick)
        samples.append(sum(sums[k][0] for k in pick) / b if b else float('inf'))
    samples.sort()
    low, high = samples[int(0.025 * BOOTSTRAP)], samples[int(0.975 * BOOTSTRAP) - 1]
    ratio = total_e / total_b if total_b else None
    return {'ratio': round(ratio, 4) if ratio else None, 'ci95': [round(low, 4), round(high, 4)],
            'games': len(keys), 'positions': len(rows), 'sum_e': round(total_e, 1), 'sum_b': round(total_b, 1),
            'verdict': 'PRECHECK_FAIL' if low > RATIO_MAX else 'PROCEED'}


def report(rows: list[dict]) -> dict:
    def pct(values, q):
        values = sorted(values)
        return values[min(len(values) - 1, int(q * len(values)))] if values else None

    attacked = [r for r in rows if r['attacked']]
    return {'attacked': len(attacked), 'switched': sum(r['move'] != r['base_move'] for r in rows),
            'attack_seconds': {'mean': round(sum(r['attack_seconds'] for r in attacked) / len(attacked), 3)
                               if attacked else None, 'p95': pct([r['attack_seconds'] for r in attacked], 0.95),
                               'max': max((r['attack_seconds'] for r in attacked), default=None)},
            'b_seconds': {'mean': round(sum(r['b_seconds'] for r in rows) / len(rows), 3) if rows else None,
                          'p95': pct([r['b_seconds'] for r in rows], 0.95)},
            'median_ratio': pct([r['e_seconds'] / r['b_seconds'] for r in rows if r['b_seconds'] > 0], 0.5)}


def main(argv=None, factories=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--value-checkpoint', default=str(ROOT / 'runs/h6_value_head_v3/best.pt'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--jsonl', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from analysis import tb_v2_runtime as rt
    from analysis import tb_v2_value_e4 as cand
    from hybrid.progress import Progress
    from scripts.h7_timing import machine, restore_fast_mode
    from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256

    positions = json.loads(args.positions.read_text(encoding='utf-8'))['positions']
    corpus = web.load(args.corpus)
    kept, dropped = split_positions(positions, corpus)
    kept = kept[:args.limit]
    done = {}
    if args.jsonl is not None and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[row['id']] = row
    todo = [p for p in kept if p['id'] not in done]
    print(f'[h7-e4-precheck] {len(positions)} timing positions, {len(dropped)} dropped (web boards), '
          f'{len(kept)} kept, {len(kept) - len(todo)} done, device {args.device}', flush=True)
    progress = Progress('h7-e4-precheck', len(todo), 'positions')
    with restore_fast_mode():
        rt.install()
        if factories is not None:
            policy, leaf = factories['policy'], factories['leaf']
        else:
            from hybrid.h4_policy import RootPolicy
            from hybrid.h6_value_leaf import ValueLeaf
            policy, leaf = RootPolicy(args.policy_checkpoint), ValueLeaf(args.value_checkpoint, device=args.device)
        for p in todo:
            row = measure(p, policy, leaf)
            done[row['id']] = row
            if args.jsonl is not None:
                args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.jsonl.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')
            progress.update(1)
    rows = [done[p['id']] for p in kept]
    result = ratio_ci(rows)
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'positions_sha256': file_sha256(args.positions),
               'corpus_sha256': file_sha256(args.corpus), 'candidate_hash': cand.candidate_hash(),
               'runtime': cand.provenance(args.device) if factories is None else None, 'machine': machine(),
               'rule': {'ratio_max': RATIO_MAX, 'bootstrap': BOOTSTRAP, 'seed': SEED, 'cluster': 'game'},
               'restricted': args.limit is not None, 'dropped_web_boards': dropped, **result,
               'report': report(rows), 'rows': rows}
    print(json.dumps({k: payload[k] for k in ('ratio', 'ci95', 'games', 'positions', 'verdict', 'report')},
                     indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
