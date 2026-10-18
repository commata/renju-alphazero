"""Decision fields of TB-Engine-v1 (tree_mode 'puct') on fixed positions, recorded before H6-4 added
tree_mode 'puct_value' to analysis.mcts_v8 (the engine-version step must not change this path).

    python tests/tb_engine_v1_regression.py   # rewrites tests/fixtures/tb_engine_v1_puct_regression.json
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from renju import Game  # noqa: E402

FIXTURE = ROOT / 'tests' / 'fixtures' / 'tb_engine_v1_puct_regression.json'
WEB = ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json'
PLIES = (8, 20, 30, 40, 50, 60)
SEED = 7


def hashed_policy(game):
    """Deterministic, non-uniform, torch-free prior (so the fixture needs no checkpoint)."""
    legal = game.legal_moves()
    weights = {m: 1 + int(hashlib.sha256(f'{m}{len(game.history)}'.encode()).hexdigest()[:4], 16) for m in legal}
    total = sum(weights.values())
    return {m: w / total for m, w in weights.items()}


def decide(ply: int, **agent_kwargs) -> dict:
    from analysis.tb_engine_v1 import make_agent
    moves = json.loads(WEB.read_text(encoding='utf-8'))['moves']
    game = Game()
    for move in moves[:ply]:
        game.play(*move)
    agent = make_agent(hashed_policy, seed=SEED)
    move = agent.select_move(game)
    d = agent.diagnostics
    return {'ply': ply, 'move': list(move), 'route': d.v8_route,
            'v7_move': list(d.v8_v7_move) if d.v8_v7_move is not None else None,
            'root_visits': [[list(m), n, round(q, 9)] for m, n, q in d.v8_root_visits],
            'nn_calls': d.v8_puct_nn_calls, 'tree_mode': d.v8_tree_mode,
            'root_checked': [[list(m), s] for m, s in d.v8_root_checked],
            'vct2_checked': [[list(m), s] for m, s in d.v8_vct2_checked]}


if __name__ == '__main__':
    FIXTURE.write_text('[\n' + ',\n'.join(json.dumps(decide(p)) for p in PLIES) + '\n]\n', encoding='utf-8')
    print(f'wrote {FIXTURE}')
