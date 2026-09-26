import subprocess
import sys
import unittest

CORE_IMPORTS = 'import search.alphazero\nimport search.evaluator\nimport training.self_play\n'
FORBIDDEN = ('search.mcts', 'search.mcts_v3', 'search.mcts_v32', 'search.mcts_v321',
             'search.mcts_v4', 'search.mcts_v5', 'search.mcts_v6', 'search.threat_planning',
             'search.threat_patterns', 'agents', 'torch', 'model.checkpoint', 'model.network',
             'model.encoding', 'model.masking', 'model.evaluator', 'numpy')

BLOCK_TORCH = '''
import importlib.abc, sys
class _NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'torch' or name.startswith('torch.'):
            raise ModuleNotFoundError("No module named 'torch'", name='torch')
        return None
sys.meta_path.insert(0, _NoTorch())
'''


def run(code: str) -> str:
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                          check=True).stdout


class CoreIsolationTest(unittest.TestCase):
    def test_core_does_not_import_v5_v6_threat_agents_torch(self):
        output = run(CORE_IMPORTS + 'import sys\nfor name in sorted(sys.modules):\n    print(name)\n')
        loaded = set(output.split())
        self.assertIn('search.alphazero', loaded)
        self.assertIn('training.self_play', loaded)
        leaked = [name for name in FORBIDDEN
                  if name in loaded or any(m.startswith(name + '.') for m in loaded)]
        self.assertEqual(leaked, [])

    def test_core_runs_without_torch(self):
        code = BLOCK_TORCH + CORE_IMPORTS + '''
from search.alphazero import SearchConfig
from search.evaluator import UniformEvaluator
from training.self_play import game_hash, play_self_play_game, replay_record
game = play_self_play_game(UniformEvaluator(), SearchConfig(num_simulations=2), 3)
replay_record(game.record, game.final_game)
try:
    import torch
except ModuleNotFoundError:
    print('torch-blocked')
print(game_hash(game.record))
'''
        output = run(code).split()
        self.assertEqual(output[0], 'torch-blocked')
        self.assertEqual(len(output[1]), 64)


if __name__ == '__main__':
    unittest.main()
