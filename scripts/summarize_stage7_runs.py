"""Compact per-block summary of Stage 6/7 runs from ``metrics.jsonl`` (shareable text).

For each block of generations: self-play games, black/white/draw results, mean game
length, new samples, sample reuse, mean policy/value loss over the block's training
steps, self-play seconds per generation, and in-loop evaluation W-L per opponent.

Independently of the blocks it lists, at exact generations:
- milestones the training loop logged (``milestones`` config, pinned checkpoints);
- win-rate jumps recomputed post hoc with ``--window``/``--threshold`` for every
  in-loop opponent (same rule as ``training.milestones``);
- self-play black-win-rate shifts of at least ``--threshold`` between consecutive
  windows (tracks the Stage 7-B black bias).

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
from training.milestones import detect_milestones  # noqa: E402


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def black_rate_shifts(generations: list[dict], window: int, threshold: float) -> list[dict]:
    by_gen = {g['generation']: g for g in generations}
    shifts, last_reported = [], None
    for gen in sorted(by_gen):
        current = [by_gen[g] for g in range(gen - window + 1, gen + 1) if g in by_gen]
        before = [by_gen[g] for g in range(gen - 2 * window + 1, gen - window + 1) if g in by_gen]
        if len(current) < window or len(before) < window:
            continue
        if last_reported is not None and gen - last_reported < window:
            continue
        rate = lambda gs: sum(g['black_wins'] for g in gs) / sum(g['self_play_games'] for g in gs)
        delta = rate(current) - rate(before)
        if abs(delta) >= threshold:
            last_reported = gen
            shifts.append({'generations': [gen - window + 1, gen], 'previous_rate': rate(before),
                           'current_rate': rate(current), 'delta': delta})
    return shifts


def posthoc_jumps(evaluations: list[dict], window: int, threshold: float) -> list[dict]:
    settings = {'window': window, 'threshold': threshold,
                'opponents': sorted({e['opponent'] for e in evaluations}), 'first_win': []}
    found = []
    for gen in sorted({e['generation'] for e in evaluations}):
        for event in detect_milestones(evaluations, found, gen, settings):
            found.append({'generation': gen, **event})
    return found


def summarize_run(run_dir: Path, block: int, window: int = 5, threshold: float = 0.25) -> dict:
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
            'total_games': sum(g['self_play_games'] for g in generations), 'blocks': blocks,
            'logged_milestones': [r for r in rows if r['type'] == 'milestone'],
            'posthoc_jumps': posthoc_jumps(evaluations, window, threshold),
            'black_rate_shifts': black_rate_shifts(generations, window, threshold),
            'jump_rule': {'window': window, 'threshold': threshold}}


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
    lines.append('logged milestones (pinned checkpoints):')
    for m in summary['logged_milestones'] or []:
        detail = (f"{m['previous_score']:.2f} -> {m['current_score']:.2f} (gens "
                  f"{m['generations'][0]}-{m['generations'][1]})" if m['kind'] == 'win_rate_jump'
                  else f"{m['wins']}/{m['games']}")
        lines.append(f"  gen {m['generation']:3d} {m['kind']:<13} {m['opponent']:<9} {detail} "
                     f"-> {m['checkpoint']}")
    if not summary['logged_milestones']:
        lines.append('  (none)')
    rule = summary['jump_rule']
    lines.append(f"post-hoc win-rate jumps (window {rule['window']}, "
                 f"threshold +{rule['threshold']}):")
    for j in summary['posthoc_jumps'] or []:
        lines.append(f"  gen {j['generation']:3d} {j['opponent']:<9} "
                     f"{j['previous_score']:.2f} -> {j['current_score']:.2f} "
                     f"(gens {j['previous_generations'][0]}-{j['previous_generations'][1]} -> "
                     f"{j['generations'][0]}-{j['generations'][1]}, games {j['games']})")
    if not summary['posthoc_jumps']:
        lines.append('  (none)')
    lines.append('self-play black-win-rate shifts:')
    for s in summary['black_rate_shifts'] or []:
        lines.append(f"  gens {s['generations'][0]}-{s['generations'][1]}: "
                     f"{100 * s['previous_rate']:.0f}% -> {100 * s['current_rate']:.0f}%")
    if not summary['black_rate_shifts']:
        lines.append('  (none)')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dirs', type=Path, nargs='+')
    parser.add_argument('--block', type=int, default=10)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summaries = [summarize_run(run_dir, args.block, args.window, args.threshold)
                 for run_dir in args.run_dirs]
    for summary in summaries:
        print(render(summary))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summaries, indent=1), encoding='utf-8')
        print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
