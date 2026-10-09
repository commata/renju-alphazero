"""S3-VCT2: re-check every veto event of the benchmark runs (docs/mcts-v8-teacher.md §12.25).

A veto event is a tree move whose played move the in-engine check (selective depth 2,
10k budget) proved lost. For every move checked at that event, three results are recorded:

    engine     the in-engine status (UNSAFE / UNKNOWN / SAFE, 10k selective budget)
    selective  ``selective_vct.classify`` with its default budget, and ``verify_witness`` on a
               depth-2 PROVEN_LOSS (False would be an unsound proof)
    full       the full depth 0-2 class (every legal quiet move of the attacker,
               ``s1_loss_analysis.lost_depth_info``) with the P92 truth budget

From these: veto precision (engine UNSAFE that the full class confirms), and for a switch
whether the replacement escaped (not lost within depth 2 under the full class). Coordinates
are 1-indexed.

    python scripts/s3_vct2_recheck.py --runs runs/s3/vct2_8411.json runs/s3/vct2_8412.json \\
        --workers 4 --output docs/mcts-v8-results/s3_vct2_full_recheck.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from analysis.selective_vct import DEFAULT_BUDGET as SELECTIVE_BUDGET, PROVEN_LOSS, classify, verify_witness  # noqa: E402
from renju import Game  # noqa: E402
from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256  # noqa: E402
from scripts.s1_loss_analysis import lost_depth_info  # noqa: E402
from scripts.s2_position_truth import DEFAULT_BUDGET as FULL_BUDGET  # noqa: E402

FORMAT = 's3-vct2-full-recheck-v1'


def veto_events(run: dict) -> list[dict]:
    events = []
    for game in run['games']:
        for record in game['v8_moves']:
            checked = record.get('vct2', {}).get('checked') or []
            if checked and checked[0][1] == 'UNSAFE':
                events.append({'seed': run['seed'], 'pair': game['pair'], 'v8_color': game['v8_color'],
                               'ply': record['ply'], 'result': game['result'], 'game_length': len(game['moves']),
                               'switched': record['vct2']['switched'], 'played': record['played'],
                               'engine_seconds': record['vct2']['seconds'], 'engine_nodes': record['vct2']['nodes'],
                               'checked': checked, 'history': game['moves'][:record['ply']]})
    return events


def recheck(task) -> dict:
    history, move = task
    game = Game()
    for m in history:
        game.play(*m)
    started = perf_counter()
    selective = classify(game, tuple(move))
    verified = verify_witness(game, tuple(move), selective) if selective['status'] == PROVEN_LOSS else None
    selective_seconds = perf_counter() - started
    started = perf_counter()
    depth, status, info = lost_depth_info([*history, move], len(history), FULL_BUDGET)
    return {'selective': {'status': selective['status'], 'depth': selective.get('depth'),
                          'cause': selective.get('cause'), 'witness': selective.get('witness'),
                          'nodes': selective['nodes_used'], 'witness_verified': verified,
                          'seconds': round(selective_seconds, 2)},
            'full': {'status': 'PROVEN_LOSS' if depth is not None else status, 'lost_depth': depth,
                     'nodes': info['nodes_used'], 'budget_exhausted': info['budget_exhausted'],
                     'vcf_cut_calls': info['vcf_cut_calls'], 'seconds': round(perf_counter() - started, 2)}}


def summarize(events: list[dict]) -> dict:
    vetoes = [e['moves'][0] for e in events]
    switches = [e for e in events if e['switched']]
    replacements = [next(m for m in e['moves'] if m['move_0idx'] == e['played_0idx']) for e in switches]
    return {
        'veto_events': len(events), 'switches': len(switches),
        'veto_precision': f"{sum(m['full']['status'] == 'PROVEN_LOSS' for m in vetoes)}/{len(vetoes)}",
        'engine_unsafe_moves': sum(m['engine'] == 'UNSAFE' for e in events for m in e['moves']),
        'engine_unsafe_confirmed_by_full': sum(m['engine'] == 'UNSAFE' and m['full']['status'] == 'PROVEN_LOSS'
                                               for e in events for m in e['moves']),
        'unsound_witness': sum(m['selective']['witness_verified'] is False for e in events for m in e['moves']),
        'replacement_engine_status': {s: sum(m['engine'] == s for m in replacements) for s in ('SAFE', 'UNKNOWN')},
        'replacement_full_status': {s: sum(m['full']['status'] == s for m in replacements)
                                    for s in ('PROVEN_LOSS', 'SAFE', 'UNKNOWN')},
        'replacement_escape_rate': (f"{sum(m['full']['status'] == 'SAFE' for m in replacements)}/{len(replacements)}"
                                    if replacements else None),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs', type=Path, nargs='+', required=True, help='puct_policy_vct2 benchmark outputs')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    runs = [json.loads(p.read_text(encoding='utf-8')) for p in args.runs]
    events = [e for run in runs for e in veto_events(run)]
    tasks = [(e['history'], m) for e in events for m, _ in e['checked']]
    print(f'veto events={len(events)} moves to re-check={len(tasks)} workers={args.workers}', flush=True)
    if args.workers == 1:
        results = [recheck(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(recheck, tasks))
    it = iter(results)
    out = []
    for e in events:
        moves = [{'move': [m[0] + 1, m[1] + 1], 'move_0idx': m, 'engine': s, **next(it)} for m, s in e['checked']]
        out.append({**{k: v for k, v in e.items() if k not in ('checked', 'history')},
                    'played': [e['played'][0] + 1, e['played'][1] + 1], 'played_0idx': e['played'], 'moves': moves})
    summary = summarize(out)
    payload = {'format': FORMAT, 'git_commit': _git_commit(),
               'budgets': {'engine_selective': {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': 10_000},
                           'selective': SELECTIVE_BUDGET, 'full': FULL_BUDGET},
               'inputs': {p.name: file_sha256(p) for p in args.runs},
               'run_git_commits': sorted({r['git_commit'] for r in runs}),
               'summary': summary, 'events': out}
    print(json.dumps(summary, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
