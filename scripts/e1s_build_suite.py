"""E1-S: the stage-route layer of the E1 suite (docs/mcts-v8-teacher.md §12.27, ROUTE_COVERAGE_MISS).

S3-VCT2-v1 checks only tree moves. E1-S asks how often a Stage 4/5 forced defense (V8-A, no
VCT2 check) was a depth-2 loss while another move V8-A could have played was not, and whether
the S3 rule applied there (a 10k selective check of the played move, then alternatives) would
have found it. It is analysis only: the engine is not changed and nothing is played.

Same fixed logs and budgets as E1-dev (``e1_build_suite``: ``SOURCES``, ``SCREEN_BUDGET``,
``TRUTH_BUDGET``), separate steps and files, so it never mixes with an E1-dev run. No policy
is needed (the stage routes do not use the tree). Coordinates are 0-indexed.

    screen   every stage4/stage5 V8 move whose played move V8-A did not already prove lost
             (VCT1 UNSAFE): full depth 0-2 class of the played move with ``SCREEN_BUDGET``.
    defense  for the PROVEN_LOSS rows (one per tactical episode, D4-deduplicated, as E1-dev):
             rebuild the defense set V8-A could play (``forced`` tier: the Stage 4 order, or the
             single Stage 5 point; ``widened`` tier: the V6 root candidates V8-A widens to), give
             every move its full-class truth with ``TRUTH_BUDGET``, and run the S3 rule offline
             on that set (``counterfactual``). Writes the manifest.

Classes (by truth on the defense set):
    ROUTE_COVERAGE_MISS        some defense move is VCT2_CLEAR: a check there could have rescued
    ROUTE_COVERAGE_UNRESOLVED  none is clear, some are UNKNOWN (not refuted)
    ROUTE_NO_RESCUE            every defense move is PROVEN_LOSS (the loss is earlier)

Counterfactual (the S3 rule with S3-VCT2-v1's selective budget, the order forced -> widened,
skipping moves V8-A proved VCT1-UNSAFE and moves that lose at once): ``k4`` checks at most
``vct2_max_children`` moves in all (the played one included, as on the tree route); ``all``
checks the whole set. Outcomes: DETECT_MISS (the 10k check did not prove the played move lost),
RESCUED (the final move is VCT2_CLEAR), KEPT (no alternative qualified), SWITCH_TO_LOSS,
SWITCH_TO_UNRESOLVED.

    python scripts/e1s_build_suite.py screen --workers 14 --jsonl runs/e1s/screen.jsonl --output runs/e1s/screen.json
    python scripts/e1s_build_suite.py defense --screen runs/e1s/screen.json --workers 14 \\
        --jsonl runs/e1s/defense.jsonl --output runs/e1s/e1s_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from renju import Game  # noqa: E402
from scripts.e1_build_suite import (  # noqa: E402
    CLEAR, PROVEN_LOSS, RESULTS, SCREEN_BUDGET, SOURCES, TRUTH_BUDGET, UNKNOWN, _run_jobs, select, truth,
)
from scripts.run_mcts_v8_benchmark import _git_commit, _git_dirty, file_sha256  # noqa: E402

FORMAT = 'e1s-manifest-v1'
STAGE_ROUTES = ('stage4', 'stage5')
CLASSES = ('ROUTE_COVERAGE_MISS', 'ROUTE_COVERAGE_UNRESOLVED', 'ROUTE_NO_RESCUE')
OUTCOMES = ('DETECT_MISS', 'RESCUED', 'KEPT', 'SWITCH_TO_LOSS', 'SWITCH_TO_UNRESOLVED')


def _engine_budget():
    from analysis.s3_vct2_v1 import S3_VCT2_V1
    return {'node_limit': S3_VCT2_V1['vct2_node_limit'], 'call_limit': S3_VCT2_V1['vct2_call_limit'],
            'node_budget': S3_VCT2_V1['vct2_node_budget'], 'max_children': S3_VCT2_V1['vct2_max_children']}


def screen_items() -> tuple[list[dict], dict]:
    """Stage 4/5 V8 moves of the E1 logs, minus those V8-A already proved lost (VCT1)."""
    items, skipped = [], {'v8a_proven_loss': 0}
    for name in SOURCES:
        run = json.loads((RESULTS / name).read_text(encoding='utf-8'))
        if run['arm'] != 'puct_policy':
            raise ValueError(f'{name}: arm {run["arm"]} is not the baseline puct_policy')
        for game in run['games']:
            for record in game['v8_moves']:
                if record['route'] not in STAGE_ROUTES:
                    continue
                checked = [[list(m), s] for m, s in record['vct']['checked']]
                status = dict((tuple(m), s) for m, s in checked).get(tuple(record['played']))
                if status == 'UNSAFE':
                    skipped['v8a_proven_loss'] += 1  # every option V8-A saw was a VCT1 loss
                    continue
                items.append({'key': f"{run['seed']}-p{game['pair']}-{game['v8_color']}/{record['ply']}",
                              'source': name, 'seed': run['seed'], 'pair': game['pair'],
                              'v8_color': game['v8_color'], 'ply': record['ply'], 'result': game['result'],
                              'history': game['moves'][:record['ply']], 'played': record['played'],
                              'source_route': record['route'], 'v7_move': record['v7_move'],
                              'v8a_checked': checked, 'v8a_widened': record['vct']['widened'],
                              'v8a_status': status, 'vct2_eligible': False, 'vct2_checked': False})
    return items, skipped


def _screen_task(item):
    return {**item, 'screen': truth(item['history'], item['played'], SCREEN_BUDGET), 'budget': SCREEN_BUDGET}


def defense_set(history, route: str, v7_move, candidate_limit=20, neighborhood_radius=2) -> list[dict]:
    """Moves V8-A could play in this position: [{'move', 'tier'}], forced tier first.

    ``forced``: the Stage 4 defenses in V8-A's order (``_stage4_order``, V7's choice first) or the
    single Stage 5 point; ``widened``: the V6 root candidates V8-A widens to when every forced
    defense is lost (``_root_candidates_v6``), minus the forced ones. Raises if the position does
    not reproduce the logged route.
    """
    from analysis.mcts_v8 import SearchDiagnostics, _stage4_order
    from search.mcts_v5 import _RootContext, _forced_v5_move
    from search.mcts_v6 import _root_candidates_v6

    game = Game()
    for move in history:
        game.play(*move)
    diag = SearchDiagnostics()
    context = _RootContext(game.legal_moves(), diag)
    _forced_v5_move(game, context=context)
    if f'stage{diag.forced_policy_stage}' != route:
        raise ValueError(f'position gives stage {diag.forced_policy_stage}, the log says {route}')
    v7 = tuple(v7_move)
    forced = _stage4_order(game, context, v7) if route == 'stage4' else [v7]
    root, _, _ = _root_candidates_v6(game, context, candidate_limit, neighborhood_radius)
    out = [{'move': list(m), 'tier': 'forced'} for m in forced]
    out += [{'move': list(m), 'tier': 'widened'} for m in root if m not in set(forced)]
    return out


def selective_lost(game: Game, move, budget) -> tuple[str, int]:
    """S3's check: PROVEN_LOSS if the selective depth-2 search proves ``move`` lost (else UNKNOWN/NO_TARGETED)."""
    from analysis.selective_vct import SelectiveSolver
    from analysis.threats import UNSAFE

    solver = SelectiveSolver(node_limit=budget['node_limit'], call_limit=budget['call_limit'],
                             node_budget=budget['node_budget'])
    status = solver._bounded(game, tuple(move), None, None, lambda g: solver.after_move(g, 2)[0])
    return (PROVEN_LOSS if status == UNSAFE else status), solver.nodes_used


def counterfactual(history, played, defenses, v8a_checked, budget, limit: int | None) -> dict:
    """The S3 rule on the stage route: keep ``played`` unless the selective check proves it lost,
    then the first alternative (defense order) that is not proven lost, not V8-A UNSAFE and not
    lost at once. ``limit`` caps the number of checked moves, the played one included."""
    from analysis.mcts_v8 import _not_immediately_lost

    game = Game()
    for move in history:
        game.play(*move)
    started = perf_counter()
    v8a = dict((tuple(m), s) for m, s in v8a_checked)
    checked, nodes = [], 0
    status, used = selective_lost(game, played, budget)
    checked.append([list(played), status])
    nodes += used
    final, switched = list(played), False
    if status == PROVEN_LOSS:
        for entry in defenses:
            move = tuple(entry['move'])
            if move == tuple(played):
                continue
            if limit is not None and len(checked) >= limit:
                break
            if v8a.get(move) == 'UNSAFE' or not _not_immediately_lost(game, move):
                continue
            status, used = selective_lost(game, move, budget)
            checked.append([list(move), status])
            nodes += used
            if status != PROVEN_LOSS:
                final, switched = list(move), True
                break
    return {'final': final, 'switched': switched, 'checked': checked, 'nodes': nodes,
            'seconds': round(perf_counter() - started, 2)}


def outcome(cf: dict, truths: dict) -> str:
    if cf['checked'][0][1] != PROVEN_LOSS:
        return 'DETECT_MISS'
    final = truths[f"{cf['final'][0]},{cf['final'][1]}"]['status']
    if final == CLEAR:
        return 'RESCUED'
    if not cf['switched']:
        return 'KEPT'
    return 'SWITCH_TO_LOSS' if final == PROVEN_LOSS else 'SWITCH_TO_UNRESOLVED'


def classify(truths: dict) -> str:
    statuses = [t['status'] for t in truths.values()]
    if CLEAR in statuses:
        return 'ROUTE_COVERAGE_MISS'
    if UNKNOWN in statuses:
        return 'ROUTE_COVERAGE_UNRESOLVED'
    return 'ROUTE_NO_RESCUE'


def _defense_task(item):
    budget = _engine_budget()
    defenses = defense_set(item['history'], item['source_route'], item['v7_move'])
    if tuple(item['played']) not in {tuple(d['move']) for d in defenses}:
        raise ValueError(f"{item['key']}: played move {item['played']} is not in the rebuilt defense set")
    truths = {}
    for entry in defenses:
        move = entry['move']
        truths[f'{move[0]},{move[1]}'] = truth(item['history'], move, TRUTH_BUDGET)
    cfs = {name: counterfactual(item['history'], item['played'], defenses, item['v8a_checked'], budget, limit)
           for name, limit in (('k4', budget['max_children']), ('all', None))}
    return {'key': item['key'], 'defenses': defenses, 'truths': truths, 'counterfactual': cfs,
            'budget': {'truth': TRUTH_BUDGET, 'engine': budget}}


def summarize(positions: list[dict]) -> dict:
    classes = {c: sum(p['class'] == c for p in positions) for c in CLASSES}
    cf = {name: {o: sum(p['counterfactual'][name]['outcome'] == o for p in positions) for o in OUTCOMES}
          for name in ('k4', 'all')}
    miss = [p for p in positions if p['class'] == 'ROUTE_COVERAGE_MISS']
    return {
        'positions': len(positions), 'classes': classes,
        'by_route': {r: {c: sum(p['class'] == c and p['source_route'] == r for p in positions) for c in CLASSES}
                     for r in STAGE_ROUTES},
        'clear_in_forced_tier': sum(any(p['truths'][f"{d['move'][0]},{d['move'][1]}"]['status'] == CLEAR
                                        and d['tier'] == 'forced' for d in p['defenses']) for p in miss),
        'counterfactual': cf,
        'counterfactual_rescue_rate_on_miss': {
            name: (round(sum(p['counterfactual'][name]['outcome'] == 'RESCUED' for p in miss) / len(miss), 3)
                   if miss else None) for name in ('k4', 'all')},
        'counterfactual_seconds_max': {name: max((p['counterfactual'][name]['seconds'] for p in positions), default=0)
                                       for name in ('k4', 'all')},
        'lost_games': sum(p['result'] == 'loss' for p in positions),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('step', choices=('screen', 'defense'))
    parser.add_argument('--screen', type=Path, help='defense: the screen output')
    parser.add_argument('--limit', type=int, help='smoke tests: first N items only')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--jsonl', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    common = {'git_commit': _git_commit(), 'git_dirty': _git_dirty(), 'restricted': args.limit is not None}
    if args.step == 'screen':
        items, skipped = screen_items()
        items = items[:args.limit]
        done = _run_jobs(items, _screen_task, args.workers, args.jsonl, lambda t: t['key'], SCREEN_BUDGET)
        rows = [done[i['key']] for i in items]
        status = {}
        for r in rows:
            status[r['screen']['status']] = status.get(r['screen']['status'], 0) + 1
        payload = {'format': 'e1s-screen-v1', **common, 'budget': SCREEN_BUDGET,
                   'inputs': {n: file_sha256(RESULTS / n) for n in SOURCES},
                   'summary': {'items': len(rows), 'skipped': skipped, 'status': status,
                               'by_route': {r: sum(x['source_route'] == r for x in rows) for r in STAGE_ROUTES}},
                   'rows': rows}
        print(json.dumps(payload['summary'], indent=1), flush=True)
    else:
        if args.screen is None:
            parser.error('defense needs --screen')
        screen = json.loads(args.screen.read_text(encoding='utf-8'))
        if screen.get('format') != 'e1s-screen-v1':
            parser.error(f'{args.screen} is not an E1-S screen output')
        chosen = select(screen['rows'])  # the E1-dev episode / D4 rule; controls are not used here
        items = chosen['losses'][:args.limit]
        budget = {'truth': TRUTH_BUDGET, 'engine': _engine_budget()}
        done = _run_jobs(items, _defense_task, args.workers, args.jsonl, lambda t: t['key'], budget)
        positions = []
        for item in items:
            row = done[item['key']]
            entry = {**{k: item[k] for k in ('key', 'source', 'seed', 'pair', 'v8_color', 'ply', 'result',
                                             'history', 'played', 'source_route', 'v7_move', 'v8a_checked',
                                             'v8a_widened', 'v8a_status', 'vct2_eligible', 'vct2_checked',
                                             'screen')},
                     'defenses': row['defenses'], 'truths': row['truths'], 'counterfactual': row['counterfactual']}
            entry['class'] = classify(row['truths'])
            for cf in entry['counterfactual'].values():
                cf['outcome'] = outcome(cf, row['truths'])
            positions.append(entry)
        payload = {'format': FORMAT, **common, 'split': 'dev', 'budgets': budget,
                   'screen_input_sha256': file_sha256(args.screen), 'selection': chosen['dropped'],
                   'summary': summarize(positions), 'positions': positions}
        print(json.dumps(payload['summary'], indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
