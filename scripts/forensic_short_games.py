"""Why do self-play games end at ply 9-10? Locate and classify the losing decision.

For each short self-play game (``--max-length``, default 10) of a run:

1. **Opening cluster:** plies 2..6 (the forced centre move skipped), canonical under
   D4. Many games in few clusters points at opening-exploration collapse.
2. **Losing decision:** the loser's last decision that still had a VCF-safe move
   (``analysis.threats``, VCF level). The move played there was unsafe.
3. **Network and search at that position** (``--checkpoint``):
   - ``prior_safe``: raw policy mass on the safe moves, ``prior_played``;
   - ``value``: network value for the side to move;
   - deterministic search (noise off, temperature 0) at the run's self-play budget
     and at ``--deep-simulations``: share of visits on safe moves, whether the chosen
     move is safe, ``KL(search || prior)``, and the root children of the safe moves:
     visits N, the best safe Q and the Q of the chosen move (mover's view, so a safe Q
     above the chosen Q that still gets few visits points at the prior, a safe Q at
     about -1 at the value head);
   - with self-play noise on, over ``--noise-trials`` seeds: how often the search
     still picks a safe move.

Category of each loss:

- ``noise``: the deterministic self-play search picks a safe move but the noisy one
  often does not -> root noise / temperature made the losing move;
- ``search_budget``: wrong at the self-play budget, right with the deep search;
- ``prior_blind``: wrong even with the deep search and prior_safe < 0.05;
- ``value_blind``: wrong with the deep search although the prior saw the safe moves.

Rows record the losing colour (``loser``); ``by_loser`` in the output and the printed
summary split everything by it (white failing to defend and black failing to defend
are different problems).

Read-only. The VCF part is cheap for short games; the deep search dominates the time.

    python scripts/forensic_short_games.py runs/stage8_s640_s2 --from 1040 --to 1120 \
        --checkpoint runs/stage8_s640_s2/checkpoints/checkpoint_gen1120.pt --limit 200 \
        --output runs/forensics/s2_1040_1120.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from math import log
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.threats import SAFE, UNKNOWN, UNSAFE, ThreatSolver  # noqa: E402
from renju import Game  # noqa: E402

BOARD = 15


def _transform(move, symmetry: int):
    row, col = move
    if symmetry >= 4:
        col = BOARD - 1 - col
    for _ in range(symmetry % 4):
        row, col = BOARD - 1 - col, row
    return row, col


def opening_cluster(moves, plies: int = 6) -> str:
    """D4-canonical key of plies 2..plies (move order kept)."""
    head = [divmod(a, BOARD) for a in moves[1:plies]]
    best = min(tuple(_transform(m, s) for m in head) for s in range(8))
    return ' '.join(f'{r},{c}' for r, c in best)


def losing_decision(moves, winner: int, solver: ThreatSolver):
    """(ply, safe moves, played move) of the loser's last decision with a VCF-safe move."""
    game = Game()
    found = None
    for ply, action in enumerate(moves):
        move = divmod(action, BOARD)
        if game.to_play == -winner:
            _, per_move = solver.decision(game, 0)
            safe = sorted(m for m, (status, _) in per_move.items() if status == SAFE)
            if safe:
                found = (ply, safe, move)
        game.play(*move)
    return found


def vct_after_safe(moves_before, safe, solver: ThreatSolver, depth: int) -> dict:
    """Do the VCF-safe moves also survive a depth-limited VCT of the attacker?

    If none does, the position was already lost at this decision (the value near -1 is
    right) and the real mistake was earlier; if some does, the value is too pessimistic.
    """
    game = Game()
    for m in moves_before:
        game.play(*m)
    statuses = Counter()
    for move in safe:
        game.play(*move)
        try:
            statuses[solver.after_move(game, depth)[0]] += 1
        finally:
            game.undo()
    return {'vct_depth': depth, 'vct_safe': statuses[SAFE], 'vct_unsafe': statuses[UNSAFE],
            'vct_unknown': statuses[UNKNOWN]}


def kl(p: list[float], q: list[float]) -> float:
    return sum(pi * log(pi / max(qi, 1e-9)) for pi, qi in zip(p, q) if pi > 0)


def analyze_position(moves_before, safe, played, evaluator, base_config, deep: int,
                     noise_trials: int, seed: int) -> dict:
    from dataclasses import replace

    from model.config import coordinate_to_action
    from search.alphazero import (argmax_action, run_search, search_with_tree,
                                  select_action, visit_policy)
    from search.evaluator import EvaluationSnapshot

    game = Game()
    for m in moves_before:
        game.play(*m)
    legal = game.legal_moves()
    result = evaluator.evaluate(EvaluationSnapshot.from_game(game, legal))
    prior = list(result.priors)
    safe_actions = {coordinate_to_action(*m) for m in safe}
    row = {'loser': 'BLACK' if game.to_play == 1 else 'WHITE',
           'prior_safe': sum(prior[a] for a in safe_actions),
           'prior_safe_max': max(prior[a] for a in safe_actions),
           'prior_played': prior[coordinate_to_action(*played)], 'value': result.value,
           'safe_moves': len(safe), 'legal_moves': len(legal)}
    for label, sims in (('base', base_config.num_simulations), ('deep', deep)):
        config = replace(base_config, num_simulations=sims, noise_enabled=False,
                         temperature_moves=0)
        search, root = search_with_tree(game, evaluator, config, None)
        pi = list(visit_policy(search.visit_counts))
        chosen = argmax_action(search)
        row[f'{label}_visit_safe'] = sum(pi[a] for a in safe_actions)
        row[f'{label}_chosen_safe'] = chosen in safe_actions
        row[f'{label}_kl'] = kl(pi, prior)
        # Root children: N = visits, Q = mean value for the side to move (the mover).
        children = root.children if root is not None else {}
        safe_q = [c.q for a, c in children.items() if a in safe_actions and c.visit_count]
        row[f'{label}_n_safe'] = sum(search.visit_counts[a] for a in safe_actions)
        row[f'{label}_q_safe_best'] = max(safe_q) if safe_q else None
        row[f'{label}_q_chosen'] = children[chosen].q if chosen in children else None
        # Would a Q-based target pick a safe move? Rank of the best safe child by Q among
        # the visited children (1 = highest), and the best Q of the unsafe ones.
        other_q = [c.q for a, c in children.items() if a not in safe_actions and c.visit_count]
        row[f'{label}_q_best_unsafe'] = max(other_q) if other_q else None
        row[f'{label}_q_rank_safe'] = (1 + sum(q > max(safe_q) for q in other_q)
                                       if safe_q else None)
        row[f'{label}_visited_children'] = sum(1 for c in children.values() if c.visit_count)
    rng = Random(seed)
    safe_picks = 0
    for _ in range(noise_trials):
        search = run_search(game, evaluator, base_config, rng)
        action = select_action(search, len(game.history), base_config, rng)
        safe_picks += action in safe_actions
    row['noisy_safe_rate'] = safe_picks / noise_trials if noise_trials else None
    if row['base_chosen_safe'] and row['noisy_safe_rate'] is not None and row['noisy_safe_rate'] < 0.8:
        row['category'] = 'noise'
    elif row['base_chosen_safe']:
        row['category'] = 'base_safe'
    elif row['deep_chosen_safe']:
        row['category'] = 'search_budget'
    elif row['prior_safe'] < 0.05:
        row['category'] = 'prior_blind'
    else:
        row['category'] = 'value_blind'
    return row


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _pct(value) -> str:
    return '-' if value is None else f'{value:.0%}'


def _f(value) -> str:
    return '-' if value is None else f'{value:+.2f}'


def _share(rows: list[dict], key: str):
    """Share of rows where the best safe move has the highest Q of the visited children."""
    ranked = [r[key] for r in rows if r.get(key) is not None]
    return sum(rank == 1 for rank in ranked) / len(ranked) if ranked else None


def summarize_by_loser(rows: list[dict]) -> dict:
    """Per losing colour: categories and mean prior / value / search diagnostics."""
    out = {}
    for colour in ('BLACK', 'WHITE'):
        group = [r for r in rows if r.get('loser') == colour]
        if not group:
            continue
        out[colour] = {'n': len(group), 'categories': dict(Counter(r['category'] for r in group)),
                       **{key: _mean(float(r[key]) if isinstance(r[key], bool) else r[key]
                                     for r in group)
                          for key in ('prior_safe', 'prior_safe_max', 'value',
                                      'base_chosen_safe', 'base_visit_safe', 'base_q_safe_best',
                                      'base_q_chosen', 'deep_chosen_safe', 'deep_visit_safe',
                                      'deep_q_safe_best', 'deep_q_chosen', 'noisy_safe_rate',
                                      'base_visited_children', 'deep_visited_children')},
                       'base_q_rank1_safe': _share(group, 'base_q_rank_safe'),
                       'deep_q_rank1_safe': _share(group, 'deep_q_rank_safe'),
                       **({'vct_any_safe': _mean(float(r['vct_safe'] > 0) for r in group),
                           'vct_all_unsafe': _mean(float(r['vct_safe'] == 0
                                                         and r['vct_unknown'] == 0)
                                                   for r in group)}
                          if all('vct_safe' in r for r in group) else {})}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('run', type=Path)
    parser.add_argument('--from', dest='start', type=int, default=0)
    parser.add_argument('--to', dest='end', type=int, default=10**9)
    parser.add_argument('--max-length', type=int, default=10)
    parser.add_argument('--checkpoint', type=Path,
                        help='training checkpoint for the network part (omit for clusters only)')
    parser.add_argument('--deep-simulations', type=int, default=400)
    parser.add_argument('--noise-trials', type=int, default=8)
    parser.add_argument('--limit', type=int, default=200, help='short games analysed in depth')
    parser.add_argument('--node-limit', type=int, default=20_000)
    parser.add_argument('--fpu-reduction', type=float,
                        help='override the search FPU: unvisited Q = parent value - this '
                             '(default: the run\'s rule, unvisited Q = 0)')
    parser.add_argument('--check-vct-depth', type=int, default=0,
                        help='also test the VCF-safe moves against a VCT of this depth '
                             '(0 = skip; 1-2 is slow but tells whether the loss was earlier)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    short, total = [], 0
    for path in sorted((args.run / 'self_play').glob('gen*.json')):
        generation = int(path.stem[3:])
        if not args.start <= generation < args.end:
            continue
        for index, item in enumerate(json.loads(path.read_text(encoding='utf-8'))['games']):
            total += 1
            record = item['record']
            if len(record['moves']) <= args.max_length and record['winner'] is not None:
                short.append({'generation': generation, 'index': index,
                              'moves': record['moves'], 'winner': record['winner']})
    clusters = Counter(opening_cluster(g['moves']) for g in short)
    print(f'{len(short)}/{total} games <= {args.max_length} plies; '
          f'{len(clusters)} distinct 6-ply openings among them')
    for key, count in clusters.most_common(5):
        print(f'  {count:5d} ({count / max(1, len(short)):.0%})  {key}')

    rows = []
    if args.checkpoint is not None:
        import torch

        from model.evaluator import PolicyValueEvaluator
        from training.config import self_play_search_config
        from training.probes import load_model_from_training_checkpoint
        from training.training_checkpoint import load_checkpoint_payload

        torch.set_num_threads(1)
        model, _ = load_model_from_training_checkpoint(args.checkpoint)
        evaluator = PolicyValueEvaluator(model)
        base = self_play_search_config(load_checkpoint_payload(args.checkpoint)['config'])
        if args.fpu_reduction is not None:
            from dataclasses import replace
            base = replace(base, fpu_reduction=args.fpu_reduction)
        solver = ThreatSolver(node_limit=args.node_limit)
        sample = Random(args.seed).sample(short, min(args.limit, len(short)))
        for number, g in enumerate(sample):
            decision = losing_decision(g['moves'], g['winner'], solver)
            if decision is None:
                rows.append({**{k: g[k] for k in ('generation', 'index')},
                             'category': 'lost_before_any_choice'})
                continue
            ply, safe, played = decision
            before = [divmod(a, BOARD) for a in g['moves'][:ply]]
            row = analyze_position(before, safe, played, evaluator, base,
                                   args.deep_simulations, args.noise_trials, args.seed + number)
            if args.check_vct_depth:
                row.update(vct_after_safe(before, safe, solver, args.check_vct_depth))
            rows.append({'generation': g['generation'], 'index': g['index'], 'ply': ply + 1,
                         'cluster': opening_cluster(g['moves']), **row})
            if (number + 1) % 20 == 0:
                print(f'  analysed {number + 1}/{len(sample)}', flush=True)
        categories = Counter(r['category'] for r in rows)
        print('categories:', dict(categories))
        for colour, summary in summarize_by_loser(rows).items():
            print(f"  loser {colour}: n={summary['n']} categories {summary['categories']} "
                  f"prior_safe {summary['prior_safe']:.3f} value {summary['value']:+.2f} "
                  f"base chosen {summary['base_chosen_safe']:.0%} (q safe "
                  f"{_f(summary['base_q_safe_best'])} vs chosen {_f(summary['base_q_chosen'])}) "
                  f"deep chosen {summary['deep_chosen_safe']:.0%} visit {summary['deep_visit_safe']:.2f} "
                  f"(q safe {_f(summary['deep_q_safe_best'])} vs chosen {_f(summary['deep_q_chosen'])})")
            print(f"    visited children base {summary['base_visited_children']:.0f} / deep "
                  f"{summary['deep_visited_children']:.0f}; safe move has the top Q: base "
                  f"{_pct(summary['base_q_rank1_safe'])}, deep {_pct(summary['deep_q_rank1_safe'])}"
                  + (f"; VCT-safe block exists {_pct(summary['vct_any_safe'])}, all blocks "
                     f"VCT-lost {_pct(summary['vct_all_unsafe'])}" if 'vct_any_safe' in summary
                     else ''))
        decided = [r for r in rows if 'prior_safe' in r]
        if decided:
            mean = lambda key: sum(r[key] for r in decided) / len(decided)  # noqa: E731
            print(f"mean prior_safe {mean('prior_safe'):.3f}  base_visit_safe "
                  f"{mean('base_visit_safe'):.3f}  deep_visit_safe {mean('deep_visit_safe'):.3f}  "
                  f"noisy_safe_rate {mean('noisy_safe_rate'):.2f}  losing ply "
                  f"{Counter(r['ply'] for r in decided).most_common(4)}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'run': str(args.run), 'range': [args.start, args.end], 'games': total,
            'short_games': len(short), 'clusters': clusters.most_common(50),
            'distinct_clusters': len(clusters), 'checkpoint': str(args.checkpoint),
            'deep_simulations': args.deep_simulations, 'noise_trials': args.noise_trials,
            'fpu_reduction': args.fpu_reduction, 'check_vct_depth': args.check_vct_depth,
            'by_loser': summarize_by_loser([r for r in rows if 'prior_safe' in r]),
            'rows': rows}, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
