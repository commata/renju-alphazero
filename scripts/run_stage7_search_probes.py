"""Search-only Stage 7 probe: how well does PUCT solve tactical probes? (no training)

Runs the fixed probe set through deterministic PUCT (noise OFF, temperature 0) for a
grid of simulation budgets and FPU settings, so the search budget / FPU choice can be
measured on a fixed network before any training run is spent on it.

    python scripts/run_stage7_search_probes.py --checkpoint runs/stage7a/checkpoints/checkpoint_gen030.pt \
        --simulations 25 50 100 200 400 --fpu none 0.25 --output runs/stage7a/search_probes_gen030.json

``--fpu none`` is the Stage 5 rule (unvisited Q = 0); a number r is parent-relative FPU
(parent value - r). ``--random-init`` measures an untrained network instead.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

import torch  # noqa: E402

from model.config import ModelConfig  # noqa: E402
from model.network import PolicyValueNet  # noqa: E402
from search.alphazero import SearchConfig  # noqa: E402
from training.probes import (POLICY_KINDS, evaluate_search_probes,  # noqa: E402
                             load_model_from_training_checkpoint, load_probe_set)

DEFAULT_PROBES = ROOT / 'tests' / 'fixtures' / 'stage7_probes_v1.json'


def parse_fpu(text: str) -> float | None:
    return None if text.lower() == 'none' else float(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path, help='Stage 6/7 training checkpoint')
    source.add_argument('--random-init', type=int, metavar='SEED',
                        help='untrained default-architecture network with this seed')
    parser.add_argument('--simulations', type=int, nargs='+', default=[25, 50, 100, 200])
    parser.add_argument('--fpu', type=parse_fpu, nargs='+', default=[None],
                        help="'none' (Stage 5, Q=0) and/or parent-relative reductions")
    parser.add_argument('--tactical-rules', choices=('off', 'on'), nargs='+', default=['off'],
                        help='PUCT v1 (off) and/or the Stage 7-B PUCT v2 teacher (on)')
    parser.add_argument('--c-puct', type=float, default=1.5)
    parser.add_argument('--kinds', nargs='+', choices=POLICY_KINDS, default=list(POLICY_KINDS))
    parser.add_argument('--probes', type=Path, default=DEFAULT_PROBES)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    probes, probe_sha = load_probe_set(args.probes)
    if args.checkpoint is not None:
        model, info = load_model_from_training_checkpoint(args.checkpoint)
    else:
        torch.manual_seed(args.random_init)
        model = PolicyValueNet(ModelConfig()).eval()
        info = {'random_init_seed': args.random_init}

    grid = []
    for rules, fpu, sims in ((r, f, n) for r in args.tactical_rules for f in args.fpu
                             for n in args.simulations):
        config = SearchConfig(num_simulations=sims, c_puct=args.c_puct, temperature_moves=0,
                              noise_enabled=False, fpu_reduction=fpu,
                              tactical_rules=rules == 'on')
        result = evaluate_search_probes(model, probes, config, tuple(args.kinds))
        grid.append(result)
        cells = ' | '.join(
            f"{k} {s['solved']:.2f} (share {s['visit_share']:.2f}, "
            f"children {s['visited_children']:.0f})"
            for k, s in result['summary'].items())
        seconds = sum(s['seconds'] * s['probes'] for s in result['summary'].values()) \
            / sum(s['probes'] for s in result['summary'].values())
        print(f"{'v2' if rules == 'on' else 'v1'} fpu {'none' if fpu is None else fpu:>5} "
              f"sims {sims:4d} | {cells} | "
              f"{seconds * 1000:.0f} ms/probe", flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'format_version': 'stage7-search-probes-v1', 'probe_set_sha256': probe_sha,
            **info, 'grid': grid}, indent=1), encoding='utf-8')
        print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
