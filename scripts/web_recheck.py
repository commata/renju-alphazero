"""Full-solver recheck of moves from a web-play game log (docs/mcts-v8-teacher.md §12.27).

The position is the game after its first ``--plies`` moves (``game.json`` records, the
opening included); each ``--move`` (1-indexed ``row,col``) is classified by the full depth
0-2 class (``s1_loss_analysis.lost_depth_info``, every legal quiet move of the attacker)
with ``--node-budget`` (default: the P92 truth budget, 10M), and by the selective detector
with its default budget plus ``verify_witness``. Rows are appended to ``--jsonl``.

    python scripts/web_recheck.py --game docs/mcts-v8-results/web_stress_20261009/<game>/game.json \\
        --plies 13 --move 7,11 --jsonl runs/web/recheck.jsonl
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from analysis.selective_vct import PROVEN_LOSS, classify, verify_witness  # noqa: E402
from renju import Game  # noqa: E402
from scripts.e1_build_suite import TRUTH_BUDGET, truth  # noqa: E402
from scripts.run_mcts_v8_benchmark import _git_commit  # noqa: E402


def recheck(task) -> dict:
    game_path, plies, move, budget = task
    log = json.loads(Path(game_path).read_text(encoding='utf-8'))
    history = [[m['row0'], m['col0']] for m in log['moves'][:plies]]
    game = Game()
    for m in history:
        game.play(*m)
    move0 = (move[0] - 1, move[1] - 1)
    selective = classify(game, move0)
    verified = verify_witness(game, move0, selective) if selective['status'] == PROVEN_LOSS else None
    return {'game': Path(game_path).parent.name, 'plies': plies, 'move': list(move),
            'selective': {'status': selective['status'], 'depth': selective.get('depth'), 'verified': verified,
                          'seconds': selective['seconds']},
            'full': truth(history, list(move0), budget), 'budget': budget, 'git_commit': _git_commit()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--game', type=Path, required=True)
    parser.add_argument('--plies', type=int, required=True, help='moves already played (opening included)')
    parser.add_argument('--move', action='append', required=True, help='1-indexed row,col; repeatable')
    parser.add_argument('--node-budget', type=int, default=TRUTH_BUDGET['node_budget'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--jsonl', type=Path, required=True)
    args = parser.parse_args(argv)
    budget = {**TRUTH_BUDGET, 'node_budget': args.node_budget}
    tasks = [(str(args.game), args.plies, tuple(int(x) for x in m.split(',')), budget) for m in args.move]
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(recheck, tasks):
            print(json.dumps(row), flush=True)
            with args.jsonl.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
