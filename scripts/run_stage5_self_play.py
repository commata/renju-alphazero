"""Stage 5 AlphaZero self-play smoke: play, replay-validate, hash and time full games.

Uniform mode is torch-free. Neural mode uses a checkpoint *file* (optionally created once
from a seeded random init) and records its SHA256. Checkpoints belong in checkpoints/,
outputs in logs/ (both git-ignored).
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from search.alphazero import SearchConfig
from search.evaluator import UniformEvaluator
from training.provenance import base_runtime_env
from training.self_play import (DIVERSITY_START_PLY, canonical_json_bytes, game_hash,
                                opening_diversity, play_self_play_game, record_hash,
                                replay_record, summarize_timing)

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluator', choices=('uniform', 'neural'), default='uniform')
    parser.add_argument('--checkpoint', type=Path, help='existing checkpoint file (neural)')
    parser.add_argument('--create-random-checkpoint', type=Path, metavar='PATH',
                        help='save a seeded random-init checkpoint here, then load and use it')
    parser.add_argument('--model-seed', type=int, help='torch seed for --create-random-checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='self-play RNG seed of the first game')
    parser.add_argument('--games', type=int, default=1, help='games with seeds seed, seed+1, ...')
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--c-puct', type=float, default=1.5)
    parser.add_argument('--temperature', type=float, default=1.0, help='tau')
    parser.add_argument('--temperature-moves', type=int, default=10)
    parser.add_argument('--dirichlet-alpha', type=float, default=0.05)
    parser.add_argument('--dirichlet-epsilon', type=float, default=0.25)
    parser.add_argument('--no-noise', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1, help='evaluator batch size (1)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, help='JSON report path (default: logs/stage5/...)')
    return parser


def make_neural(args, parser):
    import torch
    from model.evaluator import PolicyValueEvaluator, create_random_checkpoint, file_sha256

    if (args.checkpoint is None) == (args.create_random_checkpoint is None):
        parser.error('neural mode needs exactly one of --checkpoint / --create-random-checkpoint')
    if args.threads <= 0:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.create_random_checkpoint is not None:
        if args.model_seed is None:
            parser.error('--create-random-checkpoint requires --model-seed')
        path = args.create_random_checkpoint
        if path.exists():
            parser.error(f'{path} exists; checkpoints are immutable inputs, use --checkpoint')
        create_random_checkpoint(path, args.model_seed)
    else:
        path = args.checkpoint
    checkpoint_hash = file_sha256(path)  # hash of the exact bytes that are loaded
    evaluator = PolicyValueEvaluator.from_checkpoint(path, device=args.device)
    runtime = {
        **base_runtime_env(), 'evaluator': 'neural', 'torch': str(torch.__version__),
        'device': str(evaluator.device), 'torch_threads': torch.get_num_threads(),
        'checkpoint_path': str(path),
    }
    return evaluator, checkpoint_hash, args.model_seed, runtime


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.games <= 0:
        parser.error('--games must be positive')
    config = SearchConfig(
        num_simulations=args.simulations, c_puct=args.c_puct, tau=args.temperature,
        temperature_moves=args.temperature_moves, dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon, noise_enabled=not args.no_noise,
        evaluator_batch_size=args.batch_size,
    )
    if args.evaluator == 'neural':
        evaluator, checkpoint_hash, model_seed, runtime = make_neural(args, parser)
    else:
        if args.checkpoint or args.create_random_checkpoint:
            parser.error('checkpoints are only used with --evaluator neural')
        evaluator, checkpoint_hash, model_seed = UniformEvaluator(), None, None
        runtime = {**base_runtime_env(), 'evaluator': 'uniform'}

    games = []
    records = []
    all_stats = []
    for index in range(args.games):
        seed = args.seed + index
        played = play_self_play_game(evaluator, config, seed, model_seed=model_seed,
                                     checkpoint_hash=checkpoint_hash, runtime_env=runtime)
        record = played.record
        replay_record(record, played.final_game)  # raises on any illegal action/count/state
        records.append(record)
        all_stats.extend(played.move_stats)
        games.append({
            'seed': seed, 'winner': record.winner, 'move_count': len(record.moves),
            'illegal_moves': 0, 'replay': 'PASS', 'sample_count': len(record.samples),
            'fast_path_count': sum(s.fast_path for s in played.move_stats),
            'game_sha256': game_hash(record), 'record_sha256': record_hash(record),
            'timing': summarize_timing(played.move_stats),
        })
    report = {
        'format': 'stage5-smoke-v1', 'evaluator': args.evaluator,
        'search_config': config.to_dict(), 'config_hash': records[0].config_hash,
        'checkpoint_hash': checkpoint_hash, 'model_seed': model_seed,
        'git_commit': records[0].git_commit, 'git_dirty': records[0].git_dirty,
        'runtime_env': runtime, 'games': games, 'timing_all_games': summarize_timing(all_stats),
        'diversity': opening_diversity(records, max(config.temperature_moves, 2)),
        'diversity_note': f'ply 0 is the forced centre move; measured from ply {DIVERSITY_START_PLY}',
        'records': [r.to_dict() for r in records],
    }
    output = args.output or ROOT / 'logs' / 'stage5' / (
        f"{datetime.now():%Y%m%d-%H%M%S}_{args.evaluator}_seed{args.seed}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding='utf-8')

    print(f'evaluator={args.evaluator} config_hash={report["config_hash"]}')
    print(f'checkpoint_hash={checkpoint_hash} model_seed={model_seed}')
    print(f'runtime_env={canonical_json_bytes(runtime).decode("utf-8")}')
    for game in games:
        searched, fast = game['timing']['searched'], game['timing']['fast_path']
        print(f"seed={game['seed']} winner={game['winner']} moves={game['move_count']} "
              f"samples={game['sample_count']} illegal={game['illegal_moves']} "
              f"fast_paths={game['fast_path_count']} replay={game['replay']}")
        print(f"  game_sha256={game['game_sha256']}")
        print(f"  record_sha256={game['record_sha256']}")
        print(f"  searched ms/move: legal={searched['legal_moves_ms']:.2f} "
              f"inference={searched['inference_ms']:.2f} tree={searched['tree_ms']:.2f} "
              f"total={searched['total_move_ms']:.2f} (n={searched['count']})")
        print(f"  fast path: count={fast['count']} total_ms={fast['total_ms']:.3f}")
    total = report['timing_all_games']['searched']
    print(f"all games searched ms/move: legal={total['legal_moves_ms']:.2f} "
          f"inference={total['inference_ms']:.2f} tree={total['tree_ms']:.2f} "
          f"total={total['total_move_ms']:.2f} (n={total['count']})")
    print(f"diversity (from ply {DIVERSITY_START_PLY}): {report['diversity']}")
    print(f'report: {output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
