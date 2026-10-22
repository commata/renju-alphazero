"""E4-v2 design input: where the E4-v1 attack time goes, and what cheap gates would have saved (analysis only).

Reads the E4-v1 DEV result (``h7_e4_dev.json``, DEV positions only) and the cost pre-check
(``h7_e4_precheck.json``, timing positions, desktop seconds). Runs no engine; the one new computation is
the opponent's VCF if we passed (``ThreatSolver.vcf``, 20k nodes) on every attacked position.

    classes      attack WIN / no WIN and budget exhausted / no WIN and finished; their share of attack time
    gates        projected on both sets (no rerun): skip the attack when the opponent has a VCF if we pass;
                 cap the attack budget (time scaled by cap / nodes, a WIN needing more nodes is lost: an
                 approximation, since the fair-share schedule depends on the budget); tree route only
    persistence  timing set: an exhausted attack followed by another one two plies later in the same game

The timing positions used here become design data for E4-v2: its own cost pre-check needs a fresh corpus.

    python scripts/h7_e4v2_cost_analysis.py --corpus docs/h7/h7_e4_web_corpus.json --dev docs/h7/h7_e4_dev.json \\
        --precheck docs/h7/h7_e4_precheck.json --positions docs/h7/h7_timing_positions.json \\
        --output docs/h7/h7_e4v2_cost_analysis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import h7_e4_web_corpus as web  # noqa: E402
from scripts.h7_e4_dev import dev_only  # noqa: E402

FORMAT = 'h7-e4v2-cost-analysis-v1'
EXHAUSTED_NODES = 199_000  # the pre-check rows keep nodes, not the exhausted flag: 200k budget minus slack
CAPS = (200_000, 150_000, 100_000, 75_000, 50_000)


def opponent_vcf(moves) -> str:
    from analysis.threats import ThreatSolver
    from renju import Game

    game = Game()
    for m in moves:
        game.play(*m)
    return ThreatSolver(node_limit=20_000).vcf(game, -game.to_play)[0]


def classes(rows: list[dict]) -> dict:
    total = sum(r['seconds'] for r in rows) or 1.0
    out = {}
    for name in ('win', 'exhausted', 'finished'):
        part = [r for r in rows if r['class'] == name]
        out[name] = {'attacks': len(part), 'time_share': round(sum(r['seconds'] for r in part) / total, 3),
                     'seconds_mean': round(sum(r['seconds'] for r in part) / len(part), 3) if part else None,
                     'opponent_vcf': sum(r['opp_vcf'] == 'WIN' for r in part)}
    out['win_nodes'] = sorted(r['nodes'] for r in rows if r['class'] == 'win')
    return out


def project(rows: list[dict], base_seconds: float, cap: int, gate: bool, tree_only: bool = False) -> dict:
    extra, wins, switches = 0.0, 0, 0
    for r in rows:
        if gate and r['opp_vcf'] == 'WIN':
            continue
        if tree_only and r.get('route') in ('stage4', 'stage5'):
            continue
        nodes = max(r['nodes'], 1)
        extra += r['seconds'] if nodes <= cap else r['seconds'] * cap / nodes
        wins += r['class'] == 'win' and nodes <= cap
        switches += bool(r.get('switched')) and nodes <= cap
    return {'cap': cap, 'opponent_vcf_gate': gate, 'tree_only': tree_only, 'wins_kept': wins,
            'switches_kept': switches, 'ratio': round(1 + extra / base_seconds, 3) if base_seconds else None,
            'attack_time_kept': round(extra / (sum(r['seconds'] for r in rows) or 1.0), 3)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--precheck', type=Path, required=True)
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--fast', choices=('off', 'fast4'), default='off')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from scripts.h7_timing import restore_fast_mode
    from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256

    corpus = web.load(args.corpus)
    dev = json.loads(args.dev.read_text(encoding='utf-8'))
    pre = json.loads(args.precheck.read_text(encoding='utf-8'))
    timing = {p['id']: p for p in json.loads(args.positions.read_text(encoding='utf-8'))['positions']}
    selected = dev['selection']['selected']
    positions = {p['id']: p for p in corpus['positions'] if p['split'] == 'DEV'}
    with restore_fast_mode():
        if args.fast != 'off':
            from hybrid.h6_fast import install
            install(args.fast)
        dev_rows = []
        for a in dev['attacks']:
            if a['budget_name'] != selected['budget'] or a['candidates'] != selected['candidates']:
                continue
            p = dev_only(positions[a['id']])
            at = a['attack']
            dev_rows.append({'id': a['id'], 'seconds': at['seconds'], 'nodes': at['nodes'],
                             'class': 'win' if at['move'] else 'exhausted' if at['exhausted'] else 'finished',
                             'opp_vcf': opponent_vcf(web.history(corpus, p))})
        pre_rows = []
        for r in pre['rows']:
            if not r['attacked']:
                continue
            win = r['route'] == 'own_vct2' or r['move'] != r['base_move']
            pre_rows.append({'id': r['id'], 'game': r['game'], 'seconds': r['attack_seconds'], 'nodes': r['attack_nodes'],
                             'route': r['route'], 'switched': r['move'] != r['base_move'],
                             'class': 'win' if win else 'exhausted' if r['attack_nodes'] >= EXHAUSTED_NODES else 'finished',
                             'opp_vcf': opponent_vcf(timing[r['id']]['moves'])})
    base_seconds = sum(r['b_seconds'] for r in pre['rows'])
    dev_base = None  # DEV has no base timing of all positions: projections there report the attack time kept
    by_game = {}
    for r in pre['rows']:
        if r['id'].startswith('e3a|'):
            by_game.setdefault(r['game'], []).append((int(r['id'].rsplit('|', 1)[1]), r))
    after, follow, follow_switch = 0, 0, 0
    for seq in by_game.values():
        seq.sort()
        for (p1, a), (p2, b) in zip(seq, seq[1:]):
            if p2 - p1 == 2 and a['attacked'] and a['attack_nodes'] >= EXHAUSTED_NODES:
                after += 1
                follow += b['attacked'] and b['attack_nodes'] >= EXHAUSTED_NODES
                follow_switch += b['move'] != b['base_move']
    payload = {
        'format': FORMAT, 'git_commit': _git_commit(), 'dev_sha256': file_sha256(args.dev),
        'precheck_sha256': file_sha256(args.precheck), 'selected': selected,
        'dev': {'attacks': len(dev_rows), 'classes': classes(dev_rows),
                'projections': [project(dev_rows, dev_base or 0.0, c, g) for c in CAPS for g in (False, True)]},
        'timing': {'attacks': len(pre_rows), 'base_seconds': round(base_seconds, 1), 'classes': classes(pre_rows),
                   'projections': [project(pre_rows, base_seconds, c, g, t) for c in CAPS for g in (False, True)
                                   for t in (False, True)],
                   'persistence': {'exhausted_then_next': after, 'next_also_exhausted': follow,
                                   'next_switched': follow_switch}},
    }
    print(json.dumps({k: payload[k] for k in ('dev', 'timing')}, indent=1)[:6000], flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
