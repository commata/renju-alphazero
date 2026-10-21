"""H7-E4 DEV: the eight attack configurations on the DEV web positions, and the DEV selection.

DEV only. Every position this tool reads, scores or ranks must have ``split == 'DEV'`` (``dev_only``); a
HOLDOUT position anywhere on that path is an error, so no HOLDOUT number can reach the selection.

The E4 wrapper never changes the base engine's move, and the base (frozen TB-V2-Value + fast4) is
deterministic per position. So on a DEV position the E4 move is, without running the base again:
    the attack's move   if the base's route is one of ``e4_attack.ROUTES``, the trigger holds and the attack
                        (``e4_attack.e4_attack``) proves a WIN
    the base's move     otherwise (as recorded by ``h7_e4_baseline.py``, v2)
The attack depends on (budget, first-move candidates) only, so it runs once per such pair and position;
the trigger is applied when scoring. It runs on every DEV position except proven wins whose recorded base
route is outside the routes; where the base route is not recorded (not a proven win), the attack is
assumed eligible (an upper bound for the cost, and a wider soundness check).

Per configuration (``e4_attack.GRID``):
    punish         DEV WIN_VCT2_ONLY positions (main + undone) where the E4 move is PUNISH (``check_move`` v2)
    shallow        DEV WIN_5 / WIN_VCF / WIN_VCT1 positions where the base punished and E4 does not
    false proofs   attack WINs whose ``check_move`` is MISS (any DEV position); contradictions: attack WINs
                   on positions the truth searched to NOT_FOUND. UNCLEAR checks are reported (unverified).
    cost           over the positions where the configuration runs the attack: mean / p95 / max VCF nodes,
                   calls and seconds (nodes are machine independent; seconds depend on the machine and load)
Selection (fixed before DEV): among configurations with no false proof, no contradiction and no shallow
regression, the most punished positions; configurations within one position of the best are tied, and the
tie goes to the fewest attack VCF nodes per DEV position (the added work per move; then the grid order).

    python scripts/h7_e4_dev.py --corpus docs/h7/h7_e4_web_corpus.json --truth docs/h7/h7_e4_truth.json \\
        --baseline docs/h7/h7_e4_baseline_v2.json --workers 12 --fast fast4 \\
        --jsonl runs/h7_e4/dev.jsonl --output docs/h7/h7_e4_dev.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import h7_e4_baseline as base  # noqa: E402
from scripts import h7_e4_truth as truth  # noqa: E402
from scripts import h7_e4_web_corpus as web  # noqa: E402

FORMAT = 'h7-e4-dev-v1'
SHALLOW = ('WIN_5', 'WIN_VCF', 'WIN_VCT1')
TIE = 1  # configurations within this many punished positions of the best are tied


def dev_only(position: dict) -> dict:
    if position.get('split') != 'DEV':
        raise AssertionError(f"{position.get('id')}: a {position.get('split')} position reached the DEV tool")
    return position


def combos():
    from analysis.e4_attack import GRID

    seen = []
    for config in GRID:
        pair = (config['budget'], config['candidates'])
        if pair not in seen:
            seen.append(pair)
    return seen


def _init(fast: str) -> None:
    if fast != 'off':
        from hybrid.h6_fast import install
        install(fast)


def attack_task(task: dict) -> dict:
    """One position, one (budget, candidates) pair: the attack, and the check of its WIN."""
    from analysis.e4_attack import e4_attack, opponent_threat
    from renju import Game

    game = Game()
    for m in task['moves']:
        game.play(*m)
    threat = opponent_threat(game)
    move, info = e4_attack(game, {'budget': task['budget_name'], 'candidates': task['candidates']})
    row = {'key': task['key'], 'id': task['id'], 'budget_name': task['budget_name'], 'candidates': task['candidates'],
           'check_budget': task['check_budget'], 'opponent_threat': threat, 'attack': info, 'check': None}
    if move is not None:
        row['check'] = base.check_move(task['moves'], move, task['check_budget'])
    return row


def _chunk(tasks):
    return [attack_task(t) for t in tasks]


def _pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def score(config: dict, positions: list[dict], boards: dict, baseline: dict, attacks: dict) -> dict:
    """The metrics of one configuration (DEV positions only)."""
    from analysis.e4_attack import ROUTES

    out = {'config': config, 'punish': 0, 'targets': 0, 'punished': [], 'missed': [], 'unclear': [],
           'shallow_regressions': [], 'false_proofs': [], 'contradictions': [], 'unverified': [],
           'attack_wins': 0, 'switched': 0, 'attacked': 0}
    nodes, calls, seconds = [], [], []
    for p in positions:
        dev_only(p)
        a = attacks.get((p['id'], config['budget'], config['candidates']))  # None: the route never attacks
        b = baseline.get(p['id'])
        route = b['b']['route'] if b else 'tree'  # not a proven win: no recorded route, assumed eligible
        runs = a is not None and route in ROUTES and (config['trigger'] == 'all' or a['opponent_threat'])
        win = a is not None and a['attack']['move'] is not None
        if win:  # soundness of the attack itself, whatever the trigger
            outcome = a['check']['outcome']
            if outcome == 'MISS':
                out['false_proofs'].append(p['id'])
            elif outcome == 'UNCLEAR':
                out['unverified'].append(p['id'])
            if boards[p['key']]['class'] == 'NOT_FOUND':
                out['contradictions'].append(p['id'])
        if runs:
            out['attacked'] += 1
            nodes.append(a['attack']['nodes'])
            calls.append(a['attack']['calls'])
            seconds.append(a['attack']['seconds'])
        if b is None:
            continue
        grade = boards[p['key']]['grade']
        e_move, e_outcome = b['b']['move'], b['b']['outcome']
        if runs and win:
            out['attack_wins'] += 1
            if a['attack']['move'] != b['b']['move']:
                out['switched'] += 1
                e_move, e_outcome = a['attack']['move'], a['check']['outcome']
        if grade == truth.TARGET:
            out['targets'] += 1
            key = {'PUNISH': 'punished', 'MISS': 'missed'}.get(e_outcome, 'unclear')
            out[key].append(p['id'])
            out['punish'] += e_outcome == 'PUNISH'
        elif grade in SHALLOW and b['b']['outcome'] == 'PUNISH' and e_outcome != 'PUNISH':
            out['shallow_regressions'].append(p['id'])
    mean = (lambda v: round(sum(v) / len(v), 1) if v else None)
    per_position = round(sum(nodes) / len(positions), 1) if positions else None
    out['cost'] = {'attacked': len(nodes), 'nodes_per_position': per_position, 'nodes_mean': mean(nodes), 'nodes_p95': _pct(nodes, 0.95),
                   'nodes_max': max(nodes, default=None), 'calls_mean': mean(calls), 'calls_p95': _pct(calls, 0.95),
                   'seconds_mean': mean(seconds), 'seconds_p95': _pct(seconds, 0.95),
                   'seconds_max': max(seconds, default=None), 'seconds_total': round(sum(seconds), 1)}
    out['eligible'] = not (out['false_proofs'] or out['contradictions'] or out['shallow_regressions'])
    return out


def select(scores: list[dict]) -> dict:
    """The fixed DEV rule: eligible, most punished (ties within ``TIE``), then fewest mean VCF nodes."""
    eligible = [s for s in scores if s['eligible']]
    if not eligible:
        return {'selected': None, 'reason': 'no configuration without false proofs and shallow regressions'}
    best = max(s['punish'] for s in eligible)
    tied = [s for s in eligible if s['punish'] >= best - TIE]
    order = {s['config']['name']: i for i, s in enumerate(scores)}
    chosen = min(tied, key=lambda s: (s['cost']['nodes_per_position'] or 0, order[s['config']['name']]))
    return {'selected': chosen['config'], 'best_punish': best, 'tied': [s['config']['name'] for s in tied],
            'rule': f'eligible, punish >= best - {TIE}, then the fewest attack VCF nodes per DEV position'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='h7_e4_baseline.py output, format v2')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--fast', choices=('off', 'fast2', 'fast3', 'fast4'), default='off')
    parser.add_argument('--limit', type=int, help='smoke tests: first N DEV positions only')
    parser.add_argument('--jsonl', type=Path, help='one line per finished (position, budget, candidates)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from analysis.e4_attack import GRID, ROUTES
    from hybrid.progress import Progress, run_chunks
    from scripts.h7_timing import restore_fast_mode
    from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256

    corpus = web.load(args.corpus)
    t = json.loads(args.truth.read_text(encoding='utf-8'))
    b = json.loads(args.baseline.read_text(encoding='utf-8'))
    if t.get('corpus_sha256') != file_sha256(args.corpus) or b.get('truth_sha256') != file_sha256(args.truth):
        parser.error('the truth and the baseline must be of this corpus')
    if b.get('format') != base.FORMAT:
        parser.error(f'{args.baseline}: format {b.get("format")}, this tool needs {base.FORMAT}')
    positions = [dev_only(p) for p in corpus['positions'] if p['split'] == 'DEV'][:args.limit]
    baseline = {p['id']: b['positions'][p['id']] for p in positions if p['id'] in b['positions']}
    budget = t['budget']
    tasks, skipped = [], set()
    for p in positions:
        recorded = baseline.get(p['id'])
        if recorded is not None and recorded['b']['route'] not in ROUTES:
            skipped.add(p['id'])  # the wrapper never runs here: E4 plays the base move
            continue
        for budget_name, candidates in combos():
            tasks.append({'key': p['key'], 'id': p['id'], 'moves': web.history(corpus, p), 'budget_name': budget_name,
                          'candidates': candidates, 'check_budget': budget})
    done = {}
    if args.jsonl is not None and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line: computed again
            if row['check_budget'] != budget:
                parser.error(f'{args.jsonl} was checked with another budget')
            done[(row['id'], row['budget_name'], row['candidates'])] = row
    todo = [x for x in tasks if (x['id'], x['budget_name'], x['candidates']) not in done]
    print(f'[h7-e4-dev] {len(positions)} DEV positions, {len(tasks)} attacks, {len(tasks) - len(todo)} done, '
          f'workers {args.workers}, fast {args.fast}', flush=True)

    def keep(_, rows):
        for row in rows:
            done[(row['id'], row['budget_name'], row['candidates'])] = row
            if args.jsonl is not None:
                args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.jsonl.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')

    progress = Progress('h7-e4-dev', len(todo), 'attacks')
    chunks = [[x] for x in todo]
    if args.workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'), initializer=_init,
                                 initargs=(args.fast,)) as pool:
            run_chunks(pool, _chunk, chunks, progress, len, unit_name='attacks', on_result=keep)
    else:
        with restore_fast_mode():
            _init(args.fast)
            for i, c in enumerate(chunks):
                keep(i, _chunk(c))
                progress.update(1)
    scored = [p for p in positions
              if p['id'] in skipped or all((p['id'], *pair) in done for pair in combos())]
    scores = [score(config, scored, t['boards'], baseline, done) for config in GRID]
    selection = select(scores)
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'corpus_sha256': file_sha256(args.corpus),
               'truth_sha256': file_sha256(args.truth), 'baseline_sha256': file_sha256(args.baseline),
               'split': 'DEV', 'routes': list(ROUTES), 'fast': args.fast, 'restricted': args.limit is not None,
               'positions': len(scored), 'not_attacked_by_route': len(skipped), 'selection': selection, 'scores': scores,
               'attacks': [done[k] for k in sorted(done) if k[0] in {p['id'] for p in scored}]}
    table = [{'name': s['config']['name'], 'punish': f"{s['punish']}/{s['targets']}",
              'shallow': len(s['shallow_regressions']), 'false': len(s['false_proofs']),
              'contra': len(s['contradictions']), 'unverified': len(s['unverified']),
              'attacked': s['attacked'], 'nodes_per_position': s['cost']['nodes_per_position'],
              'seconds_mean': s['cost']['seconds_mean']} for s in scores]
    print(json.dumps({'table': table, 'selection': selection}, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
