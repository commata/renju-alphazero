"""Track B (Strong Hybrid) code: external data and hybrid engines (docs/mcts-v8-teacher.md §12).

Track A (self-play-only AZ: ``search``, ``training``, ``model``, ``agents``) must never
import this package (``tests/test_renjunet.py``).
"""
import warnings

# The desktop CPU venv has no NumPy, which Track B code never needs; torch warns about it
# when it is imported. The package is imported before any of its modules import torch.
warnings.filterwarnings('ignore', message='Failed to initialize NumPy')
