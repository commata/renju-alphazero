"""H7-E4-v2 COSTTEST: the fresh positions of the official E4-v2 cost pre-check (fixed before any measurement).

Source: the E2 holdout games of TB-Engine-v2 against ``v8:full`` (``runs/h7_e2/tt_vs_full_8421.json`` and
``_8422.json``: arm ``puct_policy_vct2_50k_tt``), never used for timing or E4 design. Rule:
    1. every position before a tested-arm move: id ``e2h|<game key>|<ply>``, the moves before it
    2. drop every position whose D4-canonical board (``hybrid.h6_labels.canonical_hash``) is a web corpus
       position (DEV or HOLDOUT) or a COSTDEV position (``docs/h7/h7_timing_positions.json``); assert none left
    3. whole games in sha256(game key) order until the kept positions reach ``--count`` (2,500): a game's
       positions stay together and in order, because the cooldown works along the moves of one game
The output is the position file of the E4-v2 pre-check (same shape as ``h7_timing.py positions``).

    python scripts/h7_e4v2_costtest.py --runs runs/h7_e2/tt_vs_full_8421.json runs/h7_e2/tt_vs_full_8422.json \\
        --corpus docs/h7/h7_e4_web_corpus.json --costdev docs/h7/h7_timing_positions.json \\
        --output docs/h7/h7_e4v2_costtest_positions.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import h7_e4_web_corpus as web  # noqa: E402

FORMAT = 'h7-e4v2-costtest-v1'
ARM, OPPONENT, SEEDS = 'puct_policy_vct2_50k_tt', 'v8:full', (8421, 8422)
COUNT = 2500


def build(runs: list[dict], corpus: dict, costdev: list[dict], count: int = COUNT) -> dict:
    from hybrid.h6_labels import canonical_hash

    seeds = sorted(r['seed'] for r in runs)
    for r in runs:
        if r['arm'] != ARM or r['opponent'] != OPPONENT or r['seed'] not in SEEDS:
            raise ValueError(f"expected {ARM} vs {OPPONENT} on seeds {SEEDS}, got {r['arm']} vs {r['opponent']} "
                             f"seed {r['seed']}")
    if seeds != list(SEEDS):
        raise ValueError(f'expected one run per seed {SEEDS}, got {seeds}')
    banned = {canonical_hash(web.history(corpus, p)) for p in corpus['positions']}
    banned |= {canonical_hash(p['moves']) for p in costdev}
    games = {}
    for run in runs:
        for g in run['games']:
            key = f"{run['seed']}/{g['key']}"
            games[key] = [{'id': f"e2h|{key}|{m['ply']}", 'source': 'e2h', 'moves': g['moves'][:m['ply']],
                           'recorded': {'route': m['route'], 'seconds': m['seconds']}}
                          for m in sorted(g['v8_moves'], key=lambda m: m['ply'])]
    kept, dropped, used = [], [], []
    for key in sorted(games, key=lambda k: hashlib.sha256(k.encode('utf-8')).hexdigest()):
        if len(kept) >= count:
            break
        used.append(key)
        for p in games[key]:
            (dropped if canonical_hash(p['moves']) in banned else kept).append(p)
    assert not any(canonical_hash(p['moves']) in banned for p in kept), 'a web or COSTDEV board is in COSTTEST'
    return {'games': used, 'positions': kept, 'dropped': [p['id'] for p in dropped]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--costdev', type=Path, required=True)
    parser.add_argument('--count', type=int, default=COUNT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    from scripts.run_mcts_v8_benchmark import file_sha256

    load = (lambda p: json.loads(Path(p).read_text(encoding='utf-8')))
    try:
        built = build([load(p) for p in args.runs], web.load(args.corpus), load(args.costdev)['positions'],
                      args.count)
    except ValueError as exc:
        parser.error(str(exc))
    payload = {'format': FORMAT, 'step': 'positions', 'sources': [str(p) for p in args.runs],
               'source_sha256': [file_sha256(p) for p in args.runs], 'corpus_sha256': file_sha256(args.corpus),
               'costdev_sha256': file_sha256(args.costdev), 'rule': {'count': args.count, 'arm': ARM,
                                                                     'opponent': OPPONENT, 'seeds': list(SEEDS)},
               'counts': {'games': len(built['games']), 'positions': len(built['positions']),
                          'dropped': len(built['dropped'])}, **built}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload) + '\n', encoding='utf-8')
    print(json.dumps(payload['counts']), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
