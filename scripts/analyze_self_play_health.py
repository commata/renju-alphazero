"""Self-play health per window and a stability gate (read-only, torch-free).

Reads ``<run>/self_play/genNNN.json`` and ``<run>/metrics.jsonl`` and reports, per
window of ``--window`` generations:

- game length: mean, share of games <= 10 plies, exact 9 / 10 plies (the shortest
  black / white wins, i.e. games without any defence);
- colour: black win share, the min / max of the 10-generation rolling black share, and
  the share of one-sided generations (one colour won >= 15/16 of the games) and the mean
  per-generation colour margin |black wins - white wins| / games (a window can average
  57 % black while every generation is lopsided);
- data: fresh samples per generation, replay reuse (samples drawn / fresh samples),
  SGD steps per generation;
- opening diversity: distinct prefixes and Shannon entropy (bits) of plies 2..k
  (the first, forced centre move is skipped) for k = 4, 6, 8, and the share of the
  most common 6-ply prefix.

Gate on the last window (thresholds are flags; see docs/stage8-plan.md §12.14):

- STOP: <= 10-ply share >= 40 % in the last two windows, or reuse >= 1.3 x reference
  in the last two windows;
- WARN: <= 10-ply share >= 30 %, reuse >= 1.2 x reference, one-sided generations
  >= 30 %, or 6-ply prefix entropy below 0.8 x the reference window's;
- OK otherwise.

The reference window is ``--reference-from``..+window (default: the first window).

    python scripts/analyze_self_play_health.py runs/stage8_s640_s2 --from 880 --window 40
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from math import log2
from pathlib import Path

SHORT = 10
ONE_SIDED = 15 / 16
ONE_SIDED_WARN = 0.30   # Stage 8 §12.24: colour-mode switching watch (WARN only, never STOP)
PREFIXES = (4, 6, 8)


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    return -sum(c / total * log2(c / total) for c in counter.values()) if total else 0.0


def load_games(run: Path, start: int, end: int) -> dict[int, list[dict]]:
    games = {}
    for path in sorted((run / 'self_play').glob('gen*.json')):
        generation = int(path.stem[3:])
        if start <= generation < end:
            data = json.loads(path.read_text(encoding='utf-8'))
            games[generation] = [g['record'] for g in data['games']]
    return games


def load_generation_events(run: Path) -> dict[int, dict]:
    events = {}
    metrics = run / 'metrics.jsonl'
    if metrics.is_file():
        for line in metrics.read_text(encoding='utf-8').splitlines():
            if line.strip():
                event = json.loads(line)
                if event.get('type') == 'generation':
                    events[event['generation']] = event
    return events


def _steps_from_global(events: dict, generation: int):
    if generation - 1 in events:
        return events[generation]['global_step'] - events[generation - 1]['global_step']
    return None


def window_stats(generations: list[int], games: dict, events: dict) -> dict:
    records = [r for g in generations for r in games.get(g, [])]
    lengths = Counter(len(r['moves']) for r in records)
    total = sum(lengths.values())
    black = sum(r['winner'] == 1 for r in records)
    prefix = {k: Counter(tuple(r['moves'][1:k]) for r in records if len(r['moves']) >= k)
              for k in PREFIXES}
    per_gen_black = []
    for g in generations:
        if games.get(g):
            per_gen_black.append((sum(r['winner'] == 1 for r in games[g]), len(games[g])))
    one_sided = [max(b, n - b) / n >= ONE_SIDED for b, n in per_gen_black]
    margins = [abs(2 * b - n) / n for b, n in per_gen_black]
    rolling = [sum(b for b, _ in per_gen_black[i:i + 10]) / sum(n for _, n in per_gen_black[i:i + 10])
               for i in range(max(0, len(per_gen_black) - 9))]
    ev = [events[g] for g in generations if g in events]
    fresh = [e['new_samples'] for e in ev]
    reuse = [e['sample_reuse_ratio'] for e in ev if e.get('sample_reuse_ratio') is not None]
    # global_step advances once per SGD step; older events have no train_steps.
    steps = [e.get('train_steps') or _steps_from_global(events, e['generation']) for e in ev]
    mean = (lambda xs: sum(xs) / len(xs) if xs else None)
    top6 = prefix[6].most_common(1)
    return {
        'from': generations[0], 'to': generations[-1], 'games': total,
        'mean_length': mean([len(r['moves']) for r in records]),
        'short_share': sum(c for k, c in lengths.items() if k <= SHORT) / total if total else None,
        'ply9_share': lengths[9] / total if total else None,
        'ply10_share': lengths[10] / total if total else None,
        'black_share': black / total if total else None,
        'rolling10_black_min': min(rolling) if rolling else None,
        'rolling10_black_max': max(rolling) if rolling else None,
        'one_sided_share': sum(one_sided) / len(one_sided) if one_sided else None,
        'mean_abs_color_margin': sum(margins) / len(margins) if margins else None,
        'fresh_per_gen': mean(fresh), 'reuse': mean(reuse),
        'steps_per_gen': mean([s for s in steps if s is not None]),
        **{f'prefix{k}_distinct': len(prefix[k]) for k in PREFIXES},
        **{f'prefix{k}_entropy': entropy(prefix[k]) for k in PREFIXES},
        'prefix6_top_share': (top6[0][1] / sum(prefix[6].values())) if top6 else None,
    }


def gate(windows: list[dict], reference: dict, reference_reuse: float | None) -> dict:
    last = windows[-1]
    prev = windows[-2] if len(windows) > 1 else None
    ref_reuse = reference_reuse or reference.get('reuse')
    flags, level = [], 'OK'

    def high_short(w):
        return w['short_share'] is not None and w['short_share'] >= 0.40

    def high_reuse(w, factor):
        return ref_reuse and w['reuse'] is not None and w['reuse'] >= factor * ref_reuse

    if prev is not None and high_short(last) and high_short(prev):
        flags.append('short games >= 40% for two windows')
        level = 'STOP'
    if prev is not None and high_reuse(last, 1.3) and high_reuse(prev, 1.3):
        flags.append(f'reuse >= 1.3 x {ref_reuse:.2f} for two windows')
        level = 'STOP'
    if level != 'STOP':
        if last['short_share'] is not None and last['short_share'] >= 0.30:
            flags.append(f"short games {last['short_share']:.0%} >= 30%")
            level = 'WARN'
        if high_reuse(last, 1.2):
            flags.append(f"reuse {last['reuse']:.2f} >= 1.2 x {ref_reuse:.2f}")
            level = 'WARN'
        if last.get('one_sided_share') is not None and last['one_sided_share'] >= ONE_SIDED_WARN:
            flags.append(f"one-sided generations {last['one_sided_share']:.0%} >= "
                         f"{ONE_SIDED_WARN:.0%}")
            level = 'WARN'
        if reference['prefix6_entropy'] and last['prefix6_entropy'] < 0.8 * reference['prefix6_entropy']:
            flags.append(f"6-ply prefix entropy {last['prefix6_entropy']:.2f} < 0.8 x "
                         f"{reference['prefix6_entropy']:.2f}")
            level = 'WARN'
    return {'level': level, 'flags': flags, 'reference_reuse': ref_reuse,
            'reference_window': [reference['from'], reference['to']]}


def analyze(run: Path, start: int, end: int, window: int, reference_from: int | None,
            reference_reuse: float | None) -> dict:
    games = load_games(run, start, end)
    events = load_generation_events(run)
    generations = sorted(games)
    if not generations:
        raise SystemExit(f'no self-play records in {run}/self_play for [{start}, {end})')
    windows = []
    for i in range(0, len(generations), window):
        chunk = generations[i:i + window]
        if len(chunk) == window or not windows:
            windows.append(window_stats(chunk, games, events))
    ref_start = reference_from if reference_from is not None else generations[0]
    ref_gens = [g for g in generations if ref_start <= g < ref_start + window]
    reference = window_stats(ref_gens, games, events) if ref_gens else windows[0]
    return {'run': str(run), 'window': window, 'windows': windows, 'reference': reference,
            'gate': gate(windows, reference, reference_reuse)}


def _fmt(value, spec):
    return '-' if value is None else format(value, spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('run', type=Path)
    parser.add_argument('--from', dest='start', type=int, default=0)
    parser.add_argument('--to', dest='end', type=int, default=10**9)
    parser.add_argument('--window', type=int, default=40)
    parser.add_argument('--reference-from', type=int,
                        help='first generation of the reference window (default: first window)')
    parser.add_argument('--reference-reuse', type=float,
                        help='stable replay reuse (default: the reference window mean)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.run, args.start, args.end, args.window, args.reference_from,
                     args.reference_reuse)
    print('window      games  mean  <=10   9ply  10ply black roll10(min-max) fresh reuse steps '
          '1side margin H4   H6   H8   top6')
    for w in result['windows']:
        print(f"{w['from']:>5}-{w['to']:<5} {w['games']:>5} {_fmt(w['mean_length'], '5.1f')} "
              f"{_fmt(w['short_share'], '5.0%')} {_fmt(w['ply9_share'], '5.0%')} "
              f"{_fmt(w['ply10_share'], '5.0%')} {_fmt(w['black_share'], '5.0%')} "
              f"{_fmt(w['rolling10_black_min'], '4.2f')}-{_fmt(w['rolling10_black_max'], '4.2f')} "
              f"{_fmt(w['fresh_per_gen'], '6.0f')} {_fmt(w['reuse'], '5.2f')} "
              f"{_fmt(w['steps_per_gen'], '5.0f')} {_fmt(w['one_sided_share'], '5.0%')} "
              f"{_fmt(w['mean_abs_color_margin'], '6.2f')} "
              f"{w['prefix4_entropy']:4.1f} "
              f"{w['prefix6_entropy']:4.1f} {w['prefix8_entropy']:4.1f} "
              f"{_fmt(w['prefix6_top_share'], '4.0%')}")
    g = result['gate']
    print(f"gate: {g['level']} {'; '.join(g['flags'])} (reference window {g['reference_window']}, "
          f"reference reuse {_fmt(g['reference_reuse'], '.2f')})")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
