"""Score difference of two external-anchor runs on independent opening samples (docs/h7-c1.md, Pre-RL review).

Each run is a set of opening pairs (both colours against the same fixed engine); the pair scores of each run are
resampled with replacement on their own (the two runs used different openings, so they are not paired with each
other), and the 95% interval of the difference of the means is reported. Not a hypothesis test: the share of
resamples at or below 0 is not a p-value and is not reported.

    python scripts/external_pair_bootstrap.py --a docs/external/x1_calibration.json:confirmation \\
        --b docs/external/x2_c0_prerl.json --output docs/external/tb1_tbv3_bootstrap.json

A run is an ``external-match-v1`` result (its ``games``), or ``<x1_calibration.json>:confirmation``: X1 kept only the
summary, so its per-game results are recovered from the loss candidates (one key per lost game; allowed only when
the run had no draw and no excluded game, and the recovered counts match the summary).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from random import Random

FORMAT = 'external-pair-bootstrap-v1'
SAMPLES, RNG_SEED = 20_000, 0


def pair_scores_from_games(games: list[dict]) -> dict[int, float]:
    score = {'win': 1.0, 'draw': 0.5, 'loss': 0.0}
    pairs: dict[int, list[float]] = {}
    for g in games:
        if g['result'] not in score:
            raise ValueError(f"game {g['key']}: {g['result']} (an excluded game: use complete pairs only)")
        pairs.setdefault(g['pair'], []).append(score[g['result']])
    if any(len(v) != 2 for v in pairs.values()):
        raise ValueError('an opening pair without both colours')
    return {p: sum(v) / 2 for p, v in sorted(pairs.items())}


def pair_scores_from_x1(summary: dict) -> dict[int, float]:
    if summary['draw'] or summary['engine_illegal'] or summary['engine_error'] or summary['incomplete_pairs']:
        raise ValueError('X1 results can be recovered only without draws or excluded games')
    lost: dict[int, set] = {}
    for c in summary['loss_candidates']:
        _, seed, pair, colour = c['key'].rsplit('/', 3)
        lost.setdefault(int(pair), set()).add(colour)
    if sum(len(v) for v in lost.values()) != summary['loss']:
        raise ValueError('the loss candidates do not account for every lost game')
    pairs = {p: (2 - len(lost.get(p, ()))) / 2 for p in range(summary['complete_pairs'])}
    if round(sum(pairs.values()) / len(pairs), 4) != summary['score']:
        raise ValueError('the recovered pair scores do not give the recorded score')
    return pairs


def load_run(spec: str) -> tuple[dict[int, float], dict]:
    path, _, part = spec.partition(':')
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if part == 'confirmation':
        summary = data['confirmation']['summary']
        return pair_scores_from_x1(summary), {'file': path, 'part': part, 'sha256': sha, 'score': summary['score']}
    return pair_scores_from_games(data['games']), {'file': path, 'sha256': sha, 'seed': data.get('seed'),
                                                   'score': data['summary']['score'], 'ours': data.get('ours')}


def bootstrap_difference(a: dict[int, float], b: dict[int, float], samples: int = SAMPLES, seed: int = RNG_SEED):
    rng = Random(seed)
    va, vb = list(a.values()), list(b.values())
    diffs = sorted(sum(rng.choice(vb) for _ in vb) / len(vb) - sum(rng.choice(va) for _ in va) / len(va)
                   for _ in range(samples))
    lo, hi = diffs[int(0.025 * samples)], diffs[int(0.975 * samples) - 1]
    return {'difference': round(sum(vb) / len(vb) - sum(va) / len(va), 4), 'ci95': [round(lo, 4), round(hi, 4)]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--a', required=True, help='the earlier run (the difference is b - a)')
    parser.add_argument('--b', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    a, a_info = load_run(args.a)
    b, b_info = load_run(args.b)
    result = {'format': FORMAT, 'method': 'independent opening-pair bootstrap of each run, difference b - a',
              'samples': SAMPLES, 'rng_seed': RNG_SEED, 'a': {**a_info, 'pair_scores': a},
              'b': {**b_info, 'pair_scores': b}, **bootstrap_difference(a, b)}
    print(json.dumps({k: result[k] for k in ('difference', 'ci95', 'samples', 'rng_seed')}), flush=True)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
