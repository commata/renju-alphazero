"""V8 functional gates on the fixed probe sets (docs/mcts-v8-teacher.md §4.1, §4.2.5).

``--gate defend`` (V8-A, ``must_defend_vct``, run with V8-B off as the design's
"A on, B off" regression): a probe passes only if the V8 move is in
``correct_moves``, outside ``avoid_moves`` AND V8 itself proved it VCT1-SAFE.

``--gate attack`` (V8-B, ``vct_attack``, V8 defaults): three stages per probe.
1. candidate recall: the V8-B candidate list meets ``correct_moves``;
2. bounded proof: ``v8_route == 'own_vct'``, the move is in ``correct_moves``
   and V8's own bounded solver marked it WIN;
3. independent proof: every WIN V8 declares (inside the fixture set or not) is
   re-proved by ``tactical_labels.proves_threat`` with a fresh
   ``ThreatSolver(node_limit=100_000)`` (every reply classified). A declared WIN
   outside the complete fixture set or rejected there counts as a false positive.

``--gate root`` (V8-C, ``tests/fixtures/v8c_root_probes_v1.json``, V8 defaults), two checks:
1. engine: the move is played on the tree route, is not the recorded losing move,
   V8-C proved it VCT1-SAFE and a fresh ``ThreatSolver(node_limit=100_000)`` agrees;
2. forced: the tree's choice depends on the random state of the original game,
   so the recorded losing move is handed to V8-C as "V7's move" directly (root
   children in V6 root order). V8-C must prove it UNSAFE and replace it with a
   move it proved SAFE that the fresh solver also confirms SAFE.

A fixture-correct move reached without V8's own proof never passes. Budget
exhaustion and the time distribution are reported separately; exits 1 on any
failure.

    python scripts/run_v8_gates.py [--gate defend|attack|root|all] [--symmetries 0 1 ...]
                                   [--output runs/v8_gates.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from analysis.mcts_v8 import (  # noqa: E402
    V8_DEFAULTS, WIN, SearchDiagnostics, _attack_candidates, _verify_root_choice, mcts_search_v8,
)
from analysis.tactical_labels import proves_threat  # noqa: E402
from analysis.threats import ThreatSolver  # noqa: E402
from renju import Game  # noqa: E402
from search.mcts_v5 import _RootContext  # noqa: E402
from search.mcts_v6 import _root_candidates_v6  # noqa: E402
from types import SimpleNamespace  # noqa: E402

PROBES = ROOT / 'tests/fixtures/vct_probes_v1.json'
ROOT_PROBES = ROOT / 'tests/fixtures/v8c_root_probes_v1.json'
DEFEND_CONFIG = {**V8_DEFAULTS, 'own_vct_attack': False, 'root_vct_safety': False}
ROOT_CONFIG = dict(V8_DEFAULTS)


def set_root_config(**overrides):
    ROOT_CONFIG.update(overrides)
ATTACK_CONFIG = dict(V8_DEFAULTS)


def _game(moves) -> Game:
    game = Game()
    for move in moves:
        game.play(*move)
    return game


def _summary(name, rows, config, keys, extra):
    seconds = sorted(r['seconds'] for r in rows)
    nodes = sum(r['nodes'] for r in rows)
    failures = sum(not r['ok'] for r in rows)
    summary = {'gate': name, 'config': {k: config[k] for k in keys},
               'passed': len(rows) - failures, 'total': len(rows), **extra,
               'budget_exhausted': sum(r['budget_exhausted'] for r in rows),
               'mean_seconds': round(sum(seconds) / len(seconds), 2) if seconds else 0.0,
               'median_seconds': round(median(seconds), 2) if seconds else 0.0,
               'p95_seconds': seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))] if seconds else 0.0,
               'max_seconds': seconds[-1] if seconds else 0.0,
               'max_nodes': max((r['nodes'] for r in rows), default=0),
               'max_calls': max((r['calls'] for r in rows), default=0),
               'nodes_per_second': round(nodes / sum(seconds)) if seconds and sum(seconds) else 0,
               'rows': rows}
    counts = ', '.join(f'{k} {v}' for k, v in extra.items())
    print(f"{name}: {summary['passed']}/{summary['total']} passed, {counts}, budget exhausted "
          f"{summary['budget_exhausted']}; seconds mean {summary['mean_seconds']} median "
          f"{summary['median_seconds']} p95 {summary['p95_seconds']} max {summary['max_seconds']}; "
          f"max nodes {summary['max_nodes']}, max calls {summary['max_calls']}; "
          f"{summary['nodes_per_second']} VCF nodes/s", flush=True)
    return summary


def defend_gate(probes, seed):
    rows = []
    for probe in probes:
        game = _game(probe['moves'])
        diag = SearchDiagnostics()
        started = perf_counter()
        move = list(mcts_search_v8(game, **DEFEND_CONFIG, random=Random(seed), diagnostics=diag))
        checked = {tuple(m): status for m, status in diag.v8_vct_checked}
        proven = checked.get(tuple(move)) == 'SAFE'
        ok = move in probe['correct_moves'] and move not in probe['avoid_moves'] and proven
        row = {'id': probe['id'], 'ok': ok, 'proven_safe': proven, 'move': move,
               'v7_move': list(diag.v8_v7_move), 'route': diag.v8_route,
               'calls': diag.v8_vct_calls, 'nodes': diag.v8_vct_nodes,
               'budget_exhausted': diag.v8_vct_budget_exhausted,
               'checked': [[list(m), s] for m, s in diag.v8_vct_checked],
               'seconds': round(perf_counter() - started, 2)}
        rows.append(row)
        print(f"{'PASS' if ok else 'FAIL'} {row['id']:<24} v8 {move} v7 {row['v7_move']} "
              f"{row['route']} calls {row['calls']} nodes {row['nodes']} "
              f"{'EXHAUSTED ' if row['budget_exhausted'] else ''}{row['seconds']}s", flush=True)
    return _summary('must_defend_vct', rows, DEFEND_CONFIG,
                    ('stage_vct_safety', 'own_vct_attack', 'vct_vcf_node_limit',
                     'vct_call_limit', 'vct_node_budget'),
                    {'proven_safe': sum(r['proven_safe'] for r in rows)})


def attack_gate(probes, seed):
    rows = []
    for probe in probes:
        game = _game(probe['moves'])
        correct = [tuple(m) for m in probe['correct_moves']]
        candidates = _attack_candidates(game, _RootContext(game.legal_moves(), SearchDiagnostics()))
        recall = bool(set(candidates) & set(correct))
        diag = SearchDiagnostics()
        started = perf_counter()
        move = mcts_search_v8(game, **ATTACK_CONFIG, random=Random(seed), diagnostics=diag)
        seconds = round(perf_counter() - started, 2)
        checked = dict(diag.v8_attack_checked)
        declared = diag.v8_route == 'own_vct' and checked.get(move) == WIN
        bounded = declared and move in correct
        # Every declared WIN is re-proved, inside the fixture set or not: a WIN outside the
        # complete fixture set, or one the fresh solver rejects, is a false positive.
        independent = declared and proves_threat(game, move, ThreatSolver(node_limit=100_000))
        false_positive = declared and not (independent and move in correct)
        ok = recall and bounded and independent
        row = {'id': probe['id'], 'ok': ok, 'recall': recall, 'bounded_proof': bounded,
               'independent_proof': independent, 'false_positive': false_positive,
               'move': list(move), 'route': diag.v8_route,
               'rank': diag.v8_attack_rank, 'candidates': len(candidates),
               'calls': diag.v8_attack_calls, 'nodes': diag.v8_attack_nodes,
               'budget_exhausted': diag.v8_attack_budget_exhausted,
               'checked': [[list(m), s] for m, s in diag.v8_attack_checked], 'seconds': seconds}
        rows.append(row)
        print(f"{'PASS' if ok else 'FAIL'} {row['id']:<20} v8 {list(move)} {row['route']} "
              f"rank {row['rank']}/{row['candidates']} recall {recall} bounded {bounded} "
              f"independent {independent} {'FALSE-POSITIVE ' if false_positive else ''}"
              f"calls {row['calls']} nodes {row['nodes']} "
              f"{'EXHAUSTED ' if row['budget_exhausted'] else ''}{seconds}s", flush=True)
    return _summary('vct_attack', rows, ATTACK_CONFIG,
                    ('own_vct_attack', 'attack_vcf_node_limit', 'attack_call_limit',
                     'attack_node_budget'),
                    {'recall': sum(r['recall'] for r in rows),
                     'bounded_proof': sum(r['bounded_proof'] for r in rows),
                     'independent_proof': sum(r['independent_proof'] for r in rows),
                     'false_positive': sum(r['false_positive'] for r in rows)})


def _forced_root_check(game, losing) -> dict:
    root, _, _ = _root_candidates_v6(game, _RootContext(game.legal_moves(), SearchDiagnostics()), 20, 2)
    others = [m for m in root if m != losing]
    children = [SimpleNamespace(move=losing, visits=len(others) + 1, mean_value=0.0)]
    children += [SimpleNamespace(move=m, visits=len(others) - i, mean_value=0.0) for i, m in enumerate(others)]
    diag = SearchDiagnostics()
    started = perf_counter()
    final = _verify_root_choice(game, losing, children, diag, node_limit=ROOT_CONFIG['root_vcf_node_limit'],
                                call_limit=ROOT_CONFIG['root_call_limit'], node_budget=ROOT_CONFIG['root_node_budget'],
                                max_children=ROOT_CONFIG['root_max_children'])
    statuses = dict(diag.v8_root_checked)
    replaced = final != losing and statuses.get(losing) == 'UNSAFE' and statuses.get(final) == 'SAFE'
    independent = replaced and ThreatSolver(node_limit=100_000).classify(game, [final], vct_depth=1)[final][0] == 'SAFE'
    return {'ok': replaced and independent, 'final': list(final), 'losing_status': statuses.get(losing),
            'final_status': statuses.get(final), 'independent_safe': independent, 'rank': diag.v8_root_rank,
            'calls': diag.v8_root_calls, 'nodes': diag.v8_root_nodes, 'exhausted': diag.v8_root_budget_exhausted,
            'seconds': round(perf_counter() - started, 2)}


def root_gate(probes, seed):
    rows = []
    for probe in probes:
        game = _game(probe['moves'])
        diag = SearchDiagnostics()
        started = perf_counter()
        move = mcts_search_v8(game, **ROOT_CONFIG, random=Random(seed), diagnostics=diag)
        seconds = round(perf_counter() - started, 2)
        proven = dict(diag.v8_root_checked).get(move) == 'SAFE'
        avoided = list(move) not in probe['avoid_moves']
        independent = proven and ThreatSolver(node_limit=100_000).classify(game, [move], vct_depth=1)[move][0] == 'SAFE'
        engine_ok = diag.v8_route == 'tree' and avoided and proven and independent
        forced = _forced_root_check(game, tuple(probe['avoid_moves'][0]))
        ok = engine_ok and forced['ok']
        row = {'id': probe['id'], 'ok': ok, 'engine_ok': engine_ok, 'forced': forced,
               'move': list(move), 'route': diag.v8_route, 'avoided': avoided,
               'proven_safe': proven, 'independent_safe': independent,
               'rank': diag.v8_root_rank, 'v7_move': list(diag.v8_v7_move),
               'calls': diag.v8_root_calls, 'nodes': diag.v8_root_nodes,
               'budget_exhausted': diag.v8_root_budget_exhausted,
               'root_seconds': round(diag.v8_root_seconds, 2), 'attack_seconds': round(diag.v8_attack_seconds, 2),
               'checked': [[list(m), s] for m, s in diag.v8_root_checked], 'seconds': seconds}
        rows.append(row)
        print(f"  forced {probe['avoid_moves'][0]}: {forced}", flush=True)
        print(f"{'PASS' if ok else 'FAIL'} {row['id']:<22} v8 {list(move)} {row['route']} avoided {avoided} "
              f"proven {proven} independent {independent} rank {row['rank']} v7 {row['v7_move']} "
              f"calls {row['calls']} nodes {row['nodes']} {'EXHAUSTED ' if row['budget_exhausted'] else ''}"
              f"root {row['root_seconds']}s total {seconds}s", flush=True)
    return _summary('v8c_root', rows, ROOT_CONFIG,
                    ('root_vct_safety', 'root_vcf_node_limit', 'root_call_limit', 'root_node_budget',
                     'root_max_children'),
                    {'proven_safe': sum(r['proven_safe'] for r in rows),
                     'independent_safe': sum(r['independent_safe'] for r in rows)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--gate', choices=('defend', 'attack', 'root', 'all'), default='all')
    parser.add_argument('--symmetries', type=int, nargs='*', default=None,
                        help='D4 indices to run (default: all 8)')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    probes = [p for p in json.loads(PROBES.read_text(encoding='utf-8'))['probes']
              if args.symmetries is None or p['symmetry'] in args.symmetries]
    summaries = []
    if args.gate in ('defend', 'all'):
        summaries.append(defend_gate([p for p in probes if p['kind'] == 'must_defend_vct'], args.seed))
    if args.gate in ('attack', 'all'):
        summaries.append(attack_gate([p for p in probes if p['kind'] == 'vct_attack'], args.seed))
    if args.gate in ('root', 'all'):
        summaries.append(root_gate(json.loads(ROOT_PROBES.read_text(encoding='utf-8'))['probes'], args.seed))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summaries, indent=1), encoding='utf-8')
    return 1 if any(s['passed'] < s['total'] for s in summaries) else 0


if __name__ == '__main__':
    raise SystemExit(main())
