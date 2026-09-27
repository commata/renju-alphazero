"""Compact per-block summary of Stage 6/7 runs from ``metrics.jsonl`` (shareable text).

For each block of generations: self-play games, black/white/draw results, mean game
length, new samples, sample reuse, mean policy/value loss over the block's training
steps, self-play seconds per generation, and in-loop evaluation W-L per opponent.

    python scripts/summarize_stage7_runs.py runs/stage7c_b_rules runs/stage7c_a_scale --block 10 \
        --output runs/stage7c_summary.json

The text output is small enough to paste; the JSON holds the same numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from training.metrics import read_metrics  # noqa: E402


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize_run(run_dir: Path, block: int) -> dict:
    rows = read_metrics(run_dir / 'metrics.jsonl')
    generations = [r for r in rows if r['type'] == 'generation']
    train = [r for r in rows if r['type'] == 'train']
    evaluations = [r for r in rows if r['type'] == 'evaluation']
    if not generations:
        raise SystemExit(f'no generation records in {run_dir}')
    last = max(g['generation'] for g in generations)
    blocks = []
    for start in range(0, last + 1, block):
        end = min(start + block - 1, last)
        gens = [g for g in generations if start <= g['generation'] <= end]
        if not gens:
            continue
        steps = [t for t in train if start <= t['generation'] <= end]
        games = sum(g['self_play_games'] for g in gens)
        entry = {
            'generations': [start, end],
            'games': games,
            'black_wins': sum(g['black_wins'] for g in gens),
            'white_wins': sum(g['white_wins'] for g in gens),
            'draws': sum(g['draws'] for g in gens),
            'mean_game_length': _mean(g['average_game_length'] for g in gens),
            'new_samples_per_generation': _mean(g['new_samples'] for g in gens),
            'sample_reuse_ratio': _mean(g['sample_reuse_ratio'] for g in gens),
            'policy_loss': _mean(t['policy_loss'] for t in steps),
            'value_loss': _mean(t['value_loss'] for t in steps),
            'self_play_seconds_per_generation': _mean(g['self_play_seconds'] for g in gens),
            'evaluation': {},
        }
        entry['black_win_rate'] = entry['black_wins'] / games if games else None
        for e in (e for e in evaluations if start <= e['generation'] <= end):
            slot = entry['evaluation'].setdefault(e['opponent'], {'wins': 0, 'losses': 0,
                                                                 'draws': 0})
            for key in ('wins', 'losses', 'draws'):
                slot[key] += e[key]
        blocks.append(entry)
    return {'run_dir': str(run_dir), 'last_generation': last, 'block': block,
            'total_games': sum(g['self_play_games'] for g in generations), 'blocks': blocks}


def render(summary: dict) -> str:
    lines = [f"== {summary['run_dir']}  (last gen {summary['last_generation']}, "
             f"{summary['total_games']} self-play games)"]
    lines.append('gens     | games | black% |  len | new/gen | reuse | pol   | val   | '
                 'sp s/gen | eval W-L')
    for b in summary['blocks']:
        evaluation = ' '.join(f"{name} {v['wins']}-{v['losses']}"
                              for name, v in sorted(b['evaluation'].items()))
        lines.append(
            f"{b['generations'][0]:3d}-{b['generations'][1]:<4d} | {b['games']:5d} | "
            f"{100 * b['black_win_rate']:5.1f}% | {b['mean_game_length']:4.0f} | "
            f"{b['new_samples_per_generation']:7.0f} | {b['sample_reuse_ratio']:5.2f} | "
            f"{b['policy_loss']:.3f} | {b['value_loss']:.3f} | "
            f"{b['self_play_seconds_per_generation']:8.0f} | {evaluation}")
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dirs', type=Path, nargs='+')
    parser.add_argument('--block', type=int, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summaries = [summarize_run(run_dir, args.block) for run_dir in args.run_dirs]
    for summary in summaries:
        print(render(summary))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summaries, indent=1), encoding='utf-8')
        print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
