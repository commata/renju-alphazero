"""Review a web game the AI lost: which AI move first left the human a proven win, and were there safe moves?

For each web corpus game given (``h7_e4_web_corpus.py``), the AI moves are walked backwards with the E4
truth solver (``h7_e4_truth.solve``: full-width VCT2 of the position after the AI move, the human to move):

    walk          per AI move: PROVEN_WIN (the human wins: the move is lost, with its grade), NOT_FOUND or
                  UNKNOWN. The walk stops at the first move that is not PROVEN_WIN (``--max-steps`` caps it).
    first loss    the earliest AI move of the proven-lost tail. An earlier move may be lost too beyond the
                  budget: the move before the tail is reported with its statuses.
    alternatives  every legal AI move at the first loss, lost within depth 1 (VCF / VCT1) or not
                  (``ALT_BUDGET`` per move). No SAFE move: the position was already lost before it.
    engine        ``--engine``: the frozen TB-V2-Value + fast4 (CUDA value) chooses at the first loss, and its
                  move gets the same depth <= 1 status (needs the checkpoints: desktop).

    python scripts/h7_web_loss_review.py --corpus docs/h7/h7_e4_web_corpus.json --games 8fcbd265 0bd2800b \\
        --workers 3 --output docs/h7/h7_web_losses_20261021.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import h7_e4_truth as truth  # noqa: E402
from scripts import h7_e4_web_corpus as web  # noqa: E402

FORMAT = 'h7-web-loss-review-v1'
ALT_BUDGET = {'node_limit': 100_000, 'call_limit': 100_000, 'node_budget': 2_000_000}


def one(m) -> list[int]:
    return [int(m[0]) + 1, int(m[1]) + 1]  # 1-indexed (row, col), as on the web board


def ai_plies(game: dict) -> list[int]:
    """Plies (stones after the move) of the AI's own decisions; the opening centre stone is not one."""
    ai = 0 if game['human_color'] == 'WHITE' else 1  # move index i is Black's when i is even
    return [i + 1 for i in range(1, len(game['moves'])) if i % 2 == ai]


def depth1_status(moves, move, budget=ALT_BUDGET) -> str:
    """SAFE, LOST_D0 / LOST_D1, or UNKNOWN_D<d> of ``move`` after ``moves``."""
    from renju import Game

    game = Game()
    for m in moves:
        game.play(*m)
    solver = truth._truth_solver_class()(node_limit=budget['node_limit'], call_limit=budget['call_limit'],
                                         node_budget=budget['node_budget'])
    for depth in (0, 1):
        status = solver._bounded(game, tuple(move), None, None, lambda g, d=depth: solver.after_move(g, d)[0])
        if status != 'SAFE':
            return f'LOST_D{depth}' if status == 'UNSAFE' else f'UNKNOWN_D{depth}'
    return 'SAFE'


def review(task: dict) -> dict:
    from analysis.threats import ordered_moves
    from renju import Game

    if task['fast'] != 'off':
        from hybrid.h6_fast import install
        install(task['fast'])
    moves, started = [tuple(m) for m in task['moves']], perf_counter()
    walk = []
    for ply in list(reversed(task['plies']))[:task['max_steps']]:
        r = truth.solve(moves[:ply], truth.DEFAULT_BUDGET)
        walk.append({'ply': ply, 'move': one(moves[ply - 1]), 'class': r['class'], 'grade': r['grade'],
                     'statuses': r['statuses'], 'witness': r['witness'] and [r['witness'][0], one(r['witness'][1])],
                     'seconds': r['seconds'], 'search': r['search']})
        if r['class'] != 'PROVEN_WIN':
            break
    tail = [w for w in walk if w['class'] == 'PROVEN_WIN']
    out = {'game_id': task['game_id'], 'walk': walk, 'first_loss': None}
    if tail and (len(tail) < len(walk) or len(walk) == len(task['plies'])):
        first = tail[-1]
        before = moves[:first['ply'] - 1]
        game = Game()
        for m in before:
            game.play(*m)
        alts = {'{},{}'.format(*one(m)): depth1_status(before, m) for m in ordered_moves(game)}
        counts = {}
        for s in alts.values():
            counts[s] = counts.get(s, 0) + 1
        out['first_loss'] = {**first, 'previous_ai_move': walk[len(tail)] if len(tail) < len(walk) else None,
                             'alternatives': counts, 'safe_moves': [k for k, s in alts.items() if s == 'SAFE'],
                             'already_lost_within_depth1': not any(s == 'SAFE' or s.startswith('UNKNOWN')
                                                                   for s in alts.values())}
    out['seconds'] = round(perf_counter() - started, 1)
    return out


def engine_check(rows: list[dict], corpus: dict, args) -> None:
    """``--engine``: TB-V2-Value + fast4 at each first loss (one process, the canonical CUDA value)."""
    from analysis import tb_v2_runtime as rt
    from scripts import h7_timing
    from scripts.run_mcts_v8_benchmark import derive_seed

    with h7_timing.restore_fast_mode():
        h7_timing._init(args.policy_checkpoint, args.value_checkpoint, args.device, 1, rt.FAST_MODE)
        rt.check_bound()
        for row in rows:
            first = row['first_loss']
            if first is None:
                continue
            before = [tuple(m) for m in corpus['games'][row['game_id']]['moves'][:first['ply'] - 1]]
            decided = h7_timing.decide(before, derive_seed(8440, 'h7-web-loss', f"{row['game_id']}:{first['ply']}"))
            first['engine'] = {'name': rt.NAME, 'engine_hash': rt.ENGINE_HASH, 'move': one(decided['move']),
                               'route': decided['decision']['route'], 'seconds': decided['seconds'],
                               'status': depth1_status(before, decided['move'])}
            print(f"[engine] {row['game_id'][:8]} ply {first['ply']}: {first['engine']}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--games', nargs='+', required=True, help='game ids (a unique prefix is enough)')
    parser.add_argument('--max-steps', type=int, default=12, help='AI moves walked back at most')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--fast', choices=('off', 'fast2', 'fast3', 'fast4'), default='fast4')
    parser.add_argument('--engine', action='store_true', help='also run TB-V2-Value + fast4 at each first loss')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--value-checkpoint', default=str(ROOT / 'runs/h6_value_head_v3/best.pt'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256

    corpus = web.load(args.corpus)
    tasks = []
    for prefix in args.games:
        hits = [g for g in corpus['games'] if g.startswith(prefix)]
        if len(hits) != 1:
            parser.error(f'{prefix}: {len(hits)} games match')
        g = corpus['games'][hits[0]]
        tasks.append({'game_id': hits[0], 'moves': g['moves'], 'plies': ai_plies(g), 'max_steps': args.max_steps,
                      'fast': args.fast})
    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(review, t) for t in tasks]):
                rows.append(future.result())
                print(f"[review] {rows[-1]['game_id'][:8]} done, {rows[-1]['seconds']}s", flush=True)
    else:
        from scripts.h7_timing import restore_fast_mode
        with restore_fast_mode():
            for t in tasks:
                rows.append(review(t))
                print(f"[review] {rows[-1]['game_id'][:8]} done, {rows[-1]['seconds']}s", flush=True)
    rows.sort(key=lambda r: args.games.index(next(p for p in args.games if r['game_id'].startswith(p))))
    if args.engine:
        engine_check(rows, corpus, args)
    for r in rows:
        g = corpus['games'][r['game_id']]
        r.update({'split': g['split'], 'human_color': g['human_color'], 'agent_name': g['agent_name'],
                  'result': g['result'], 'moves': [one(m) for m in g['moves']]})
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'corpus_sha256': file_sha256(args.corpus),
               'truth_budget': truth.DEFAULT_BUDGET, 'alt_budget': ALT_BUDGET, 'fast': args.fast,
               'coordinates': '1-indexed (row, col)', 'games': rows}
    shown = [{k: r[k] for k in ('game_id', 'first_loss')} for r in rows]
    for s in shown:
        if s['first_loss']:
            s['first_loss'] = {k: v for k, v in s['first_loss'].items() if k in
                               ('ply', 'move', 'grade', 'alternatives', 'safe_moves', 'already_lost_within_depth1',
                                'engine')}
    print(json.dumps(shown, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
