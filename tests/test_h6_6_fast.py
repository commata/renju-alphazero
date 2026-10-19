"""H6-6a: the H6 tools give the same V8-C results with the exact speed-up (skipped without the native build)."""
import json
from pathlib import Path
import unittest

from scripts import h6_5d as d

ROOT = Path(__file__).resolve().parents[1]
WEB = json.loads((ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json').read_text(encoding='utf-8'))
BUDGETS = ({'node_limit': 20_000, 'call_limit': 10_000, 'node_budget': 400_000, 'share': 100, 'call_share': 5_000},
           {'node_limit': 20_000, 'call_limit': 10_000, 'node_budget': 400_000, 'share': None, 'call_share': None})


def native_available() -> bool:
    try:
        from rt30 import fast3_windows
        from rt30.native import build
        return fast3_windows.available() and build.stale() is None
    except Exception:  # noqa: BLE001  (no native build: nothing to compare)
        return False


class FastEquivalenceTest(unittest.TestCase):
    def tearDown(self):
        from hybrid.h6_fast import install
        install('off')

    def _statuses(self):
        moves, move = WEB['moves'][:94], WEB['moves'][94]
        keys = ('status', 'nodes', 'calls', 'budget_cut', 'cut_kind', 'per_call_limit_hits')
        return [{k: d.check_status(moves, move, b)[k] for k in keys} for b in BUDGETS]

    def test_fast2_same_v8c(self):
        off = self._statuses()
        d._ensure_fast('fast2')
        self.assertEqual(self._statuses(), off)

    @unittest.skipUnless(native_available(), 'native fast3 build missing or stale')
    def test_fast3_same_v8c(self):
        off = self._statuses()
        d._ensure_fast('fast3')
        self.assertEqual(self._statuses(), off)


if __name__ == '__main__':
    unittest.main()
