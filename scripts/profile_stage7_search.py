"""Stage 7 PUCT cost breakdown per searched move (plan §4, completion criterion 3).

Searches a fixed set of mid-game positions (the Stage 7 probe positions) with the
production evaluator path and reports mean milliseconds per move for: legal_moves
(black / white to move), snapshot+validation, NN encode / stack / forward / softmax /
CPU transfer / result objects, PUCT v2 rule filter, Game.play, Game.undo and the
remaining tree work. Search results are identical to an uninstrumented run
(tests/test_stage7_profiling.py). Run it alone: concurrent training distorts timings.

    python scripts/profile_stage7_search.py --checkpoint runs/stage7d_b32/checkpoints/checkpoint_gen080.pt \
        --simulations 25 50 --tactical-rules off on --output runs/stage7_profile.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import torch  # noqa: E402

from model.config import ModelConfig  # noqa: E402
from model.network import PolicyValueNet  # noqa: E402
from renju import Game  # noqa: E402
from search.alphazero import SearchConfig  # noqa: E402
from training.probes import load_model_from_training_checkpoint  # noqa: E402
from training.profiling import profile_searches  # noqa: E402

PROBE_FILES = (ROOT / 'tests' / 'fixtures' / 'stage7_probes_v1.json',
               ROOT / 'tests' / 'fixtures' / 'stage7_probes_defense_v1.json')
ROWS = ('legal_moves_black', 'legal_moves_white', 'snapshot_and_validation', 'nn_encode',
        'nn_stack', 'nn_forward', 'nn_softmax', 'nn_transfer', 'nn_results', 'rule_filter',
        'play', 'undo', 'tree_other')


def load_positions(limit: int | None) -> list[Game]:
    games = []
    for path in PROBE_FILES:
        for probe in json.loads(path.read_text(encoding='utf-8'))['probes']:
            game = Game()
            for move in probe['moves']:
                game.play(*move)
            if not game.done:
                games.append(game)
    return games[:limit] if limit else games


def render(result: dict) -> str:
    search = result['search']
    head = (f"sims {search['num_simulations']:4d} rules {'on ' if search.get('tactical_rules') else 'off'}"
            f" | {result['total_ms_per_move']:7.1f} ms/move | "
            f"{result['evaluator_calls_per_move']:.1f} evals/move | "
            f"NN {result['per_call_us']['nn_total']:.0f} us/eval")
    lines = [head]
    for name in ROWS:
        ms = result['per_move_ms'][name]
        lines.append(f"    {name:<24} {ms:8.2f} ms  {100 * result['share'][name]:5.1f}%")
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path, help='Stage 6/7 training checkpoint')
    source.add_argument('--random-init', type=int, metavar='SEED')
    parser.add_argument('--simulations', type=int, nargs='+', default=[50])
    parser.add_argument('--tactical-rules', choices=('off', 'on'), nargs='+', default=['on'])
    parser.add_argument('--positions', type=int, default=80,
                        help='number of fixed probe positions to search (default 80)')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=5,
                        help='positions searched once before measuring (not reported)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    if args.checkpoint is not None:
        model, info = load_model_from_training_checkpoint(args.checkpoint)
    else:
        torch.manual_seed(args.random_init)
        model = PolicyValueNet(ModelConfig()).eval()
        info = {'random_init_seed': args.random_init}
    games = load_positions(args.positions)
    if args.warmup:
        profile_searches(model, games[:args.warmup],
                         SearchConfig(num_simulations=min(args.simulations), temperature_moves=0,
                                      noise_enabled=False))

    results = []
    for rules in args.tactical_rules:
        for sims in args.simulations:
            config = SearchConfig(num_simulations=sims, temperature_moves=0, noise_enabled=False,
                                  tactical_rules=rules == 'on')
            result = profile_searches(model, games, config)
            results.append(result)
            print(render(result), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'format_version': 'stage7-search-profile-v1', **info,
            'torch_threads': args.threads, 'torch_version': str(torch.__version__),
            'python_version': platform.python_version(), 'platform': platform.platform(),
            'processor': platform.processor(), 'positions': len(games),
            'results': results}, indent=1), encoding='utf-8')
        print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
