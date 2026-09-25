"""CPU leaf-inference pipeline timing; warm-up/setup excluded, no search."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import statistics
from time import perf_counter

import torch

from benchmark_engine import prepare_midgame
from model.encoding import encode_game
from model.masking import legal_moves_to_mask, masked_softmax
from model.network import PolicyValueNet


def measure(operation, iterations: int, warmup: int, batch: int = 1):
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(iterations):
        started = perf_counter()
        operation()
        samples.append(perf_counter() - started)
    mean = statistics.mean(samples)
    return dict(mean_ms=mean * 1000, median_ms=statistics.median(samples) * 1000,
                min_ms=min(samples) * 1000, max_ms=max(samples) * 1000,
                ops_sec=1 / mean, samples_sec=batch / mean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.threads, args.iterations, args.warmup) <= 0:
        parser.error('threads, iterations and warmup must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = PolicyValueNet().eval()
    game, _, position_seed = prepare_midgame(args.seed)
    moves = game.legal_moves()
    mask = legal_moves_to_mask(moves)
    encoded = encode_game(game, mask).unsqueeze(0)

    def pipeline():
        current_mask = legal_moves_to_mask(game.legal_moves())
        x = encode_game(game, current_mask).unsqueeze(0)
        logits, value = model(x)
        return masked_softmax(logits, current_mask.unsqueeze(0)), value

    results = {}
    with torch.inference_mode():
        logits, _ = model(encoded)
        operations = {
            'legal_moves': game.legal_moves,
            'mask_construction': lambda: legal_moves_to_mask(moves),
            'encode_reusing_mask': lambda: encode_game(game, mask),
            'masking_softmax': lambda: masked_softmax(logits, mask.unsqueeze(0)),
            'full_pipeline': pipeline,
        }
        for name, operation in operations.items():
            results[name] = measure(operation, args.iterations, args.warmup)
        for batch in (1, 8, 32):
            x = encoded.repeat(batch, 1, 1, 1)
            results[f'forward_B{batch}'] = measure(lambda: model(x), args.iterations,
                                                  args.warmup, batch)
    report = dict(python=platform.python_version(), torch=str(torch.__version__),
                  threads=torch.get_num_threads(), device='cpu', cpu=platform.processor(),
                  platform=platform.platform(), model_config=asdict(model.config),
                  parameters=sum(p.numel() for p in model.parameters()),
                  seed=args.seed, position_seed=position_seed, plies=len(game.history),
                  iterations=args.iterations, warmup=args.warmup, results=results)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
