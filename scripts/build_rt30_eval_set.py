"""Build the fixed RT30 evaluation manifest from the E3-A records (docs/web-ser5.md §5.2).

Stratified over the record files (seeds) and games, so the set is not a run of consecutive moves of
a few games (``--select slow --limit 40`` took 40 moves of 13 games of one seed):

- the pool: V8 moves whose recorded time is above ``--threshold`` seconds;
- per record file ``--per-file`` positions; inside a file, games are visited in a shuffled order
  and each gives at most ``--per-game`` positions (random plies), round after round until the
  quota is met;
- everything is drawn from ``random.Random(--seed)``, so the same arguments give the same keys.

The output is committed and then used by ``run_rt30_replay.py --keys-file``; it is never redrawn
for a comparison. Known failure positions are kept apart in a regression manifest
(``docs/rt30-eval/critical-regressions-v1.json``), not mixed into this set.

    python scripts/build_rt30_eval_set.py --output docs/rt30-eval/slow-stratified-v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / 'docs' / 'mcts-v8-results' / 'e3'
DEFAULT_JSONL = [E3 / 'puct_policy_vct2_50k_8415.jsonl', E3 / 'puct_policy_vct2_50k_8416.jsonl']


def pool(path: Path, threshold: float) -> dict[int, list[dict]]:
    """{game line: [position, ...]} of the moves slower than ``threshold`` in one record file."""
    games: dict[int, list[dict]] = defaultdict(list)
    with Path(path).open(encoding='utf-8') as handle:
        for line_no, line in enumerate(handle):
            for move in json.loads(line)['v8_moves']:
                if move['seconds'] > threshold:
                    games[line_no].append({'key': f'{Path(path).stem}/{line_no}/{move["ply"]}', 'game': line_no,
                                           'ply': move['ply'], 'recorded_seconds': move['seconds'],
                                           'recorded_route': move['route']})
    return games


def stratified(games: dict[int, list[dict]], quota: int, per_game: int, rng: Random) -> list[dict]:
    order = sorted(games)
    rng.shuffle(order)
    left = {g: rng.sample(games[g], len(games[g])) for g in order}  # each game's positions, shuffled
    taken: dict[int, int] = Counter()
    chosen: list[dict] = []
    while len(chosen) < quota:
        progressed = False
        for game in order:
            if len(chosen) >= quota:
                break
            if taken[game] < per_game and left[game]:
                chosen.append(left[game].pop())
                taken[game] += 1
                progressed = True
        if not progressed:
            break  # the pool is smaller than the quota under the per-game cap
    return chosen


def build(paths, *, threshold: float = 30.0, per_file: int = 30, per_game: int = 2, seed: int = 42,
          version: str = 'slow-stratified-v1') -> dict:
    rng = Random(seed)
    positions, sources = [], []
    for path in paths:
        games = pool(path, threshold)
        chosen = stratified(games, per_file, per_game, rng)
        if len(chosen) < per_file:
            raise SystemExit(f'{path}: only {len(chosen)} positions under the per-game cap, need {per_file}')
        positions += sorted(chosen, key=lambda p: (p['game'], p['ply']))
        sources.append({'file': str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else str(path),
                        'pool_positions': sum(len(v) for v in games.values()), 'pool_games': len(games),
                        'chosen': len(chosen), 'chosen_games': len({p['game'] for p in chosen})})
    return {
        'version': version, 'kind': 'rt30-eval-manifest',
        'description': 'Recorded E3-A V8 moves slower than the threshold, stratified by record file and game.',
        'source_threshold_seconds': threshold, 'per_file': per_file, 'max_positions_per_game': per_game,
        'sampling_seed': seed, 'sources': sources,
        'routes': dict(Counter(p['recorded_route'] for p in positions)),
        'keys': [p['key'] for p in positions], 'positions': positions,
    }


def load_keys(path) -> tuple[list[str], str | None]:
    """Keys of a manifest (``{"keys": [...]}``) or of a plain JSON list; and the manifest version."""
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(value, list):
        return [str(k) for k in value], None
    return [str(k) for k in value['keys']], value.get('version')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--jsonl', type=Path, nargs='+', default=DEFAULT_JSONL)
    parser.add_argument('--threshold', type=float, default=30.0)
    parser.add_argument('--per-file', type=int, default=30)
    parser.add_argument('--per-game', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--version', default='slow-stratified-v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build(args.jsonl, threshold=args.threshold, per_file=args.per_file, per_game=args.per_game,
                     seed=args.seed, version=args.version)
    if args.output.exists():
        old = json.loads(args.output.read_text(encoding='utf-8'))
        if old.get('keys') != manifest['keys']:
            raise SystemExit(f'{args.output} exists with other keys: a committed manifest is never redrawn; '
                             f'write a new version instead')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('version', 'sources', 'routes')}, indent=1), flush=True)
    print(f'{len(manifest["keys"])} keys -> {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
