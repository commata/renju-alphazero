"""H4 offline diagnostic: does the H3 policy put the human move where V8's tree looks?
(docs/mcts-v8-teacher.md §12.13)

For sampled RenjuNet policy states (ply >= 5, not masked) that would reach V8's tree route
(no Stage 1-5 forced move, no own VCF; V8-B is not run, so a few V8-B positions remain), the
root move list is built exactly as V8 does (V6 candidates -> V7 VCF safety tiers -> M3), for

    baseline   V8 as is
    order      H4-a: policy order inside the safety tiers
    union      H4-a+b: + up to ``--extra`` policy moves, same safety tiers

and the human move is looked up:

- candidate recall: in the root list at all;
- opened recall: the probability that the tree opens it as a root child. The V5 tree opens
  ``k = root_opening_count(budget)`` children (15 at 50 simulations, 18 at 100), each popped
  from the top-``priority_top_k`` window with rank weights; this is simulated ``--draws`` times;
- incremental recall: union candidate recall - baseline candidate recall (what H4-b recovers);
- policy top-8 / top-20 recall among all legal moves (no V8 involved).

Human moves are not "best moves": these are diagnostics of where the search looks, not of
strength (that is the paired benchmark). Parameters are fixed in advance (extra 8); tune on
``val`` only, report ``test`` once.

    python scripts/h4_offline_recall.py --games data/external/renjunet/games.jsonl.gz \\
        --policy-checkpoint runs/h3_policy_64x4/best.pt --split val --samples 2000 \\
        --output runs/h4_recall_val.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from analysis.mcts_v8 import V8_DEFAULTS, SearchDiagnostics, prepare_root_moves, root_opening_count  # noqa: E402
from hybrid.h4_policy import RootPolicy  # noqa: E402
from hybrid.renjunet import POLICY_FROM_PLY  # noqa: E402
from renju import Game  # noqa: E402
from search.mcts_v5 import _RootContext, _forced_v5_move  # noqa: E402
from search.mcts_v7 import _find_vcf_with_stats  # noqa: E402

ARMS = ('baseline', 'order', 'union')


def opened_probability(moves, target, k, top_k, rng: Random, draws: int) -> float:
    """P(target is among the k root children the V5 tree opens), rank-weighted pops from the top-k window."""
    if target not in moves:
        return 0.0
    hits = 0
    for _ in range(draws):
        pool = list(moves)
        for _ in range(min(k, len(pool))):
            width = min(len(pool), top_k)
            index = rng.choices(range(width), weights=range(width, 0, -1), k=1)[0]
            if pool.pop(index) == target:
                hits += 1
                break
    return hits / draws


def sample_states(games_path: Path, split: str, samples: int, seed: int):
    games = [json.loads(line) for line in gzip.decompress(games_path.read_bytes()).decode('utf-8').splitlines()
             if line]
    states = [(g, ply) for g in games if g['split'] == split
              for ply in range(POLICY_FROM_PLY, len(g['moves'])) if ply not in set(g['masked_plies'])]
    rng = Random(seed)
    return rng.sample(states, min(samples, len(states))), len(states)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--games', type=Path, default=ROOT / 'data/external/renjunet/games.jsonl.gz')
    parser.add_argument('--policy-checkpoint', type=Path, default=ROOT / 'runs/h3_policy_64x4/best.pt')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--extra', type=int, default=8)
    parser.add_argument('--draws', type=int, default=200)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)

    cfg = V8_DEFAULTS
    policy = RootPolicy(args.policy_checkpoint, threads=args.threads)
    sampled, population = sample_states(args.games, args.split, args.samples, args.seed)
    safety = dict(candidate_limit=cfg['candidate_limit'], neighborhood_radius=cfg['neighborhood_radius'],
                  safety_vcf_max_fours=cfg['safety_vcf_max_fours'], safety_vcf_node_limit=cfg['safety_vcf_node_limit'],
                  safety_precheck_node_limit=cfg['safety_precheck_node_limit'],
                  safety_total_node_limit=cfg['safety_total_node_limit'],
                  self_forbidden_min_white=cfg['self_forbidden_min_white'])
    rng = Random(args.seed)
    routes = {'tree': 0, 'forced': 0, 'own_vcf': 0}
    sums = {arm: {'candidate': 0, 'window': 0.0, 'opened': 0.0} for arm in ARMS}
    policy_top = {'top8': 0, 'top20': 0}
    added_total = displaced_total = recovered = 0
    started = perf_counter()
    for index, (record, ply) in enumerate(sampled):
        game = Game()
        for move in record['moves'][:ply]:
            game.play(*move)
        target = tuple(record['moves'][ply])
        context = _RootContext(game.legal_moves(), SearchDiagnostics())
        if _forced_v5_move(game, context=context) is not None:
            routes['forced'] += 1
            continue
        found, _, _ = _find_vcf_with_stats(game, game.to_play, max_fours=cfg['own_vcf_max_fours'],
                                           node_limit=cfg['own_vcf_node_limit'])
        if found is not None:
            routes['own_vcf'] += 1
            continue
        routes['tree'] += 1
        scores = policy(game)
        ranked = sorted(scores, key=lambda m: (-scores[m], m))
        policy_top['top8'] += target in ranked[:8]
        policy_top['top20'] += target in ranked[:20]
        lists = {}
        for arm, kwargs in (('baseline', {}), ('order', {'policy_scores': scores, 'order_by_policy': True}),
                            ('union', {'policy_scores': scores, 'order_by_policy': True, 'extra': args.extra})):
            moves, score, _, added = prepare_root_moves(
                game, _RootContext(game.legal_moves(), SearchDiagnostics()), SearchDiagnostics(), **safety, **kwargs)
            lists[arm] = moves
            budget = cfg['tactical_simulations'] if score >= cfg['tactical_score_threshold'] else cfg['simulations']
            k = root_opening_count(budget, cfg['initial_width'], len(moves))
            sums[arm]['candidate'] += target in moves
            sums[arm]['window'] += target in moves[:cfg['priority_top_k']]
            sums[arm]['opened'] += opened_probability(moves, target, k, cfg['priority_top_k'], rng, args.draws)
            if arm == 'union':
                added_total += sum(m in moves for m in added)
                displaced_total += len(set(lists['baseline'][:k]) - set(moves[:k]))
                recovered += target in added and target in moves
        if (index + 1) % 100 == 0:
            print(f'  {index + 1}/{len(sampled)} sampled, {routes["tree"]} tree positions, '
                  f'{perf_counter() - started:.0f}s', flush=True)
    n = routes['tree']
    rates = {arm: {key: value / n for key, value in sums[arm].items()} for arm in ARMS} if n else {}
    report = {
        'format': 'h4-offline-recall-v1', 'split': args.split, 'population_states': population,
        'sampled': len(sampled), 'routes': routes, 'tree_positions': n, 'extra': args.extra, 'draws': args.draws,
        'policy': policy.describe(), 'recall': rates,
        'incremental_candidate_recall': (rates['union']['candidate'] - rates['baseline']['candidate']) if n else None,
        'recovered_by_added': recovered / n if n else None,
        'policy_top8_recall': policy_top['top8'] / n if n else None,
        'policy_top20_recall': policy_top['top20'] / n if n else None,
        'added_kept_per_position': added_total / n if n else None,
        'displaced_from_opening_per_position': displaced_total / n if n else None,
        'seconds': round(perf_counter() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'policy'}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
