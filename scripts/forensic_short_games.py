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
     move is safe, ``KL(search || prior)``;
   - with self-play noise on, over ``--noise-trials`` seeds: how often the search
     still picks a safe move.

Category of each loss:

- ``noise``: the deterministic self-play search picks a safe move but the noisy one
  often does not -> root noise / temperature made the losing move;
- ``search_budget``: wrong at the self-play budget, right with the deep search;
- ``prior_blind``: wrong even with the deep search and prior_safe < 0.05;
- ``value_blind``: wrong with the deep search although the prior saw the safe moves.

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

from analysis.threats import SAFE, ThreatSolver  # noqa: E402
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


def kl(p: list[float], q: list[float]) -> float:
    return sum(pi * log(pi / max(qi, 1e-9)) for pi, qi in zip(p, q) if pi > 0)


def analyze_position(moves_before, safe, played, evaluator, base_config, deep: int,
                     noise_trials: int, seed: int) -> dict:
    from dataclasses import replace

    from model.config import coordinate_to_action
    from search.alphazero import argmax_action, run_search, select_action, visit_policy
    from search.evaluator import EvaluationSnapshot

    game = Game()
    for m in moves_before:
        game.play(*m)
    legal = game.legal_moves()
    result = evaluator.evaluate(EvaluationSnapshot.from_game(game, legal))
    prior = list(result.priors)
    safe_actions = {coordinate_to_action(*m) for m in safe}
    row = {'prior_safe': sum(prior[a] for a in safe_actions),
           'prior_played': prior[coordinate_to_action(*played)], 'value': result.value,
           'safe_moves': len(safe), 'legal_moves': len(legal)}
    for label, sims in (('base', base_config.num_simulations), ('deep', deep)):
        config = replace(base_config, num_simulations=sims, noise_enabled=False,
                         temperature_moves=0)
        search = run_search(game, evaluator, config, None)
        pi = list(visit_policy(search.visit_counts))
        row[f'{label}_visit_safe'] = sum(pi[a] for a in safe_actions)
        row[f'{label}_chosen_safe'] = argmax_action(search) in safe_actions
        row[f'{label}_kl'] = kl(pi, prior)
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
            rows.append({'generation': g['generation'], 'index': g['index'], 'ply': ply + 1,
                         'cluster': opening_cluster(g['moves']), **row})
            if (number + 1) % 20 == 0:
                print(f'  analysed {number + 1}/{len(sample)}', flush=True)
        categories = Counter(r['category'] for r in rows)
        print('categories:', dict(categories))
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
            'rows': rows}, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
