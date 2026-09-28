"""Evaluate Stage 6/7 training checkpoints against fixed classical opponents.

This runs outside the training loop, so it never changes a run's training-critical
config or its resume/evaluation reproducibility. The model plays the same
deterministic PUCT as the in-loop evaluation (noise OFF, temperature 0, simulations
from the checkpoint's ``evaluation.puct_simulations`` unless overridden).

Openings and opponent seeds depend only on (--seed, opponent, pair), NOT on the
checkpoint, so every checkpoint meets the same openings and opponents and results are
comparable across generations. Each opening is played with colours swapped.

Opponent ladder (weak -> strong): random, tactical, mcts_v2 (pure MCTS),
mcts_v321, mcts_v5 (V5 FINAL), mcts_v6, mcts_v7 (frozen Stage 6.5 benchmark).

Examples:
    python scripts/run_stage7_checkpoint_eval.py --run-dir runs/stage6_mvp_B_stage7a \
        --generations 10 20 30 --opponents mcts_v2 mcts_v321 mcts_v7 --pairs 5
    python scripts/run_stage7_checkpoint_eval.py --checkpoint runs/x/checkpoints/latest.pt \
        --opponents mcts_v7 --pairs 1 --output out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from renju import BLACK, WHITE  # noqa: E402
from search.alphazero import SearchConfig  # noqa: E402
from training.evaluation import (PUCTAgent, make_opening, play_evaluation_game,  # noqa: E402
                                 summarize_games)
from training.probes import load_model_from_training_checkpoint  # noqa: E402
from training.training_checkpoint import load_checkpoint_payload  # noqa: E402
from training.training_state import derive_seed  # noqa: E402

RESULT_FORMAT = 'stage7-checkpoint-eval-v1'
OPPONENTS = ('random', 'tactical', 'mcts_v2', 'mcts_v321', 'mcts_v5', 'mcts_v6', 'mcts_v7')
DEFAULT_OPPONENTS = ('mcts_v2', 'mcts_v321', 'mcts_v7')


def make_opponent(name: str, seed: int):
    from agents import (MCTSV2Agent, MCTSV321Agent, MCTSV5Agent, MCTSV6Agent,
                        MCTSV7Agent, RandomAgent, TacticalAgent)
    from search.mcts_v6 import V5_FINAL

    if name == 'random':
        return RandomAgent(seed)
    if name == 'tactical':
        return TacticalAgent(seed)
    if name == 'mcts_v2':
        return MCTSV2Agent(seed=seed)
    if name == 'mcts_v321':
        return MCTSV321Agent(seed=seed)
    if name == 'mcts_v5':
        agent = MCTSV5Agent(seed=seed, **V5_FINAL)
        agent.name = 'MCTS-v5-final'
        return agent
    if name == 'mcts_v6':
        return MCTSV6Agent(seed=seed)
    if name == 'mcts_v7':
        return MCTSV7Agent(seed=seed)
    raise ValueError(f'unknown opponent: {name}')


def search_config_for(checkpoint: Path, simulations: int | None,
                      tactical_rules: str = 'auto') -> SearchConfig:
    """``tactical_rules``: 'auto' = the checkpoint's evaluation setting, or 'on'/'off'."""
    evaluation = load_checkpoint_payload(checkpoint)['config']['evaluation']
    rules = (evaluation.get('tactical_rules', False) if tactical_rules == 'auto'
             else tactical_rules == 'on')
    return SearchConfig(
        num_simulations=simulations if simulations is not None else evaluation['puct_simulations'],
        c_puct=evaluation['c_puct'], temperature_moves=0, noise_enabled=False,
        tactical_rules=rules)


def evaluate_checkpoint(checkpoint: Path, opponents, *, pairs: int, seed: int,
                        simulations: int | None = None, tactical_rules: str = 'auto',
                        opening_random_plies: int = 2,
                        opening_radius: int = 2, log=print) -> dict:
    model, info = load_model_from_training_checkpoint(checkpoint)
    search = search_config_for(checkpoint, simulations, tactical_rules)
    model_agent = PUCTAgent('model', model, search)
    result = {'format_version': RESULT_FORMAT, **info, 'model_search': search.to_dict(),
              'seed': seed, 'pairs': pairs, 'opening_random_plies': opening_random_plies,
              'opening_radius': opening_radius, 'opponents': {}}
    for name in opponents:
        started = perf_counter()
        games = []
        for pair in range(pairs):
            opening = make_opening(Random(derive_seed(seed, name, 'opening', pair)),
                                   opening_random_plies, opening_radius)
            for offset, color in enumerate((BLACK, WHITE)):
                index = 2 * pair + offset
                opponent = make_opponent(name, derive_seed(seed, name, 'agent', index))
                game = play_evaluation_game(model_agent, opponent, color, opening)
                game['index'] = index
                games.append(game)
        summary = summarize_games(games)
        summary['score'] = (summary['wins'] + 0.5 * summary['draws']) / summary['games']
        summary['seconds'] = perf_counter() - started
        result['opponents'][name] = {'summary': summary, 'games': games}
        log(f"gen {info['generation']:3d} [{'v2' if search.tactical_rules else 'v1'}] vs {name}: "
            f"W{summary['wins']} L{summary['losses']} "
            f"D{summary['draws']} score {summary['score']:.2f} "
            f"({summary['seconds']:.0f}s)")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path)
    source.add_argument('--checkpoint', type=Path)
    parser.add_argument('--generations', type=int, nargs='+',
                        help='with --run-dir: checkpoint generations to evaluate (required)')
    parser.add_argument('--opponents', nargs='+', choices=OPPONENTS,
                        default=list(DEFAULT_OPPONENTS))
    parser.add_argument('--pairs', type=int, default=5,
                        help='openings per opponent; each is played twice (colours swapped)')
    parser.add_argument('--seed', type=int, default=7007)
    parser.add_argument('--simulations', type=int,
                        help='override the checkpoint evaluation.puct_simulations')
    parser.add_argument('--tactical-rules', choices=('auto', 'on', 'off'), default='auto',
                        help="model search: 'auto' = checkpoint's evaluation.tactical_rules "
                             "(PUCT v2 when on); use on/off to compare arms under one search")
    parser.add_argument('--suffix', default='',
                        help='with --run-dir: output name genNNN<suffix>.json')
    parser.add_argument('--output', type=Path, help='with --checkpoint: output JSON path')
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error('--pairs must be positive')
    if args.run_dir is not None and not args.generations:
        parser.error('--run-dir requires --generations')
    if args.checkpoint is not None and args.generations:
        parser.error('--generations requires --run-dir')

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.checkpoint is not None:
        targets = [(args.checkpoint, args.output)]
    else:
        out_dir = args.run_dir / 'external_eval'
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = []
        for generation in args.generations:
            path = args.run_dir / 'checkpoints' / f'checkpoint_gen{generation:03d}.pt'
            if not path.is_file():  # pruned: fall back to a milestone pin
                path = path.with_name(f'milestone_gen{generation:03d}.pt')
            if not path.is_file():
                raise SystemExit(f'missing checkpoint: {path}')
            targets.append((path, out_dir / f'gen{generation:03d}{args.suffix}.json'))

    for checkpoint, output in targets:
        result = evaluate_checkpoint(checkpoint, args.opponents, pairs=args.pairs,
                                     seed=args.seed, simulations=args.simulations,
                                     tactical_rules=args.tactical_rules)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=1), encoding='utf-8')
            print(f'wrote {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
