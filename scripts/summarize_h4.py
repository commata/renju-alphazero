"""H4: one table for several paired V8 benchmark runs (docs/mcts-v8-teacher.md §12.14).

    python scripts/summarize_h4.py --baseline runs/h4/full_vs_b.json \\
        runs/h4/policy_vs_b.json runs/h4/recall_vs_b.json --output runs/h4/summary.json

For each run: score with a 95% Wilson interval, score by V8 colour, wins/draws/losses,
per-game comparison with the baseline (same opening, colour and seed), the safety
invariant, losses in games where the opponent played a VCT1 attack, policy diagnostics
and thinking time. All runs must share --seed and --opponent.

Safety invariant (must be 0): a move V8 played that its own check proved UNSAFE while
some other checked move was not UNSAFE. A proven loss played when every checked move was
proven UNSAFE (a lost position) is counted separately.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCORE = {'win': 1.0, 'draw': 0.5, 'loss': 0.0}


def wilson(score: float, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = score / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def safety(run) -> dict:
    violations, lost_positions = [], 0
    for game in run['games']:
        for move in game['v8_moves']:
            checked = move['root']['checked'] if move['route'] == 'tree' else (
                move['vct']['checked'] if move['route'] in ('stage4', 'stage5') else [])
            if not checked:
                continue
            statuses = {tuple(m): s for m, s in checked}
            if statuses.get(tuple(move['played'])) != 'UNSAFE':
                continue
            if all(s == 'UNSAFE' for s in statuses.values()):
                lost_positions += 1
            else:
                violations.append({'key': game['key'], 'ply': move['ply'], 'route': move['route']})
    return {'violations': violations, 'proven_loss_in_lost_position': lost_positions}


def summarize(run, baseline=None) -> dict:
    games = run['games']
    total = sum(SCORE[g['result']] for g in games)
    by_colour = {}
    for colour in ('black', 'white'):
        part = [g for g in games if g['v8_color'] == colour]
        s = sum(SCORE[g['result']] for g in part)
        by_colour[colour] = {'games': len(part), 'score': round(s / len(part), 3) if part else None,
                             'ci95': wilson(s, len(part))}
    out = {
        'arm': run['arm'], 'games': len(games),
        'wins': sum(g['result'] == 'win' for g in games), 'draws': sum(g['result'] == 'draw' for g in games),
        'losses': sum(g['result'] == 'loss' for g in games),
        'score': round(total / len(games), 3) if games else None, 'ci95': wilson(total, len(games)),
        'by_colour': by_colour, 'safety': safety(run),
        'losses_with_opponent_vct': sum(g['result'] == 'loss' and g.get('opponent_routes', {}).get('own_vct', 0) > 0
                                        for g in games),
        'policy': run['summary'].get('policy'), 'v8_move_seconds': run['summary']['v8_move_seconds'],
        'total_game_minutes': round(sum(g['game_seconds'] for g in games) / 60, 1),
        'git_commit': run.get('git_commit'),
    }
    if baseline is not None:
        base = {(g['pair'], g['v8_color']): g for g in baseline['games']}
        rows = [(SCORE[g['result']], SCORE[base[(g['pair'], g['v8_color'])]['result']])
                for g in games if (g['pair'], g['v8_color']) in base]
        out['vs_baseline'] = {'common_games': len(rows), 'better': sum(a > b for a, b in rows),
                              'worse': sum(a < b for a, b in rows), 'same': sum(a == b for a, b in rows),
                              'same_moves': sum(g['moves'] == base[(g['pair'], g['v8_color'])]['moves']
                                                for g in games if (g['pair'], g['v8_color']) in base)}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    runs = [json.loads(p.read_text(encoding='utf-8')) for p in args.runs]
    for run in runs:
        if (run['seed'], run.get('opponent', 'v7')) != (baseline['seed'], baseline.get('opponent', 'v7')):
            parser.error(f"{run['arm']}: seed/opponent differ from the baseline")
    rows = [summarize(baseline)] + [summarize(run, baseline) for run in runs]
    print('| arm | games | W/D/L | score (95% CI) | black | white | vs baseline +/-/= | safety violations | '
          'losses with opp. VCT | V8 s median / p95 |')
    print('|---|---:|---|---|---:|---:|---|---:|---:|---|')
    for r in rows:
        vs = r.get('vs_baseline')
        vs_text = '-' if vs is None else f"{vs['better']}/{vs['worse']}/{vs['same']}"
        print(f"| {r['arm']} | {r['games']} | {r['wins']}/{r['draws']}/{r['losses']} | {r['score']} {r['ci95']} | "
              f"{r['by_colour']['black']['score']} | {r['by_colour']['white']['score']} | "
              f"{vs_text} | "
              f"{len(r['safety']['violations'])} | {r['losses_with_opponent_vct']} | "
              f"{r['v8_move_seconds'].get('median')} / {r['v8_move_seconds'].get('p95')} |")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'baseline': str(args.baseline), 'rows': rows}, indent=1), encoding='utf-8')
    return 1 if any(r['safety']['violations'] for r in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
