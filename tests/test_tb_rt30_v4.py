"""TB-RT30-v4 candidate: v3 unchanged but for the native builder of TB-V3 (src/rt30/tb_rt30_v4.py)."""
import hashlib
from pathlib import Path
import unittest

from rt30 import tb_rt30_v3, tb_rt30_v4

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'tests' / 'frozen_tb_rt30_v4.sha256'
V3_LOCK = ROOT / 'tests' / 'frozen_tb_rt30_v3.sha256'


def lock_entries(path):
    return dict(reversed(line.split(maxsplit=1)) for line in path.read_text(encoding='utf-8').splitlines()
                if line.strip() and not line.startswith('#'))


class V4Test(unittest.TestCase):
    def test_candidate_until_its_gate(self):
        self.assertEqual((tb_rt30_v4.NAME, tb_rt30_v4.STATUS, tb_rt30_v4.BASE_ENGINE),
                         ('TB-RT30-v4', 'candidate', 'TB-RT30-v3'))

    def test_v3_configuration_but_for_the_builder(self):
        v3, v4 = tb_rt30_v3.TB_RT30_V3, tb_rt30_v4.TB_RT30_V4
        self.assertEqual({k for k in v4 if v4[k] != v3.get(k)}, {'native_builder'})
        self.assertNotEqual(tb_rt30_v4.config_hash(), tb_rt30_v3.config_hash())

    def test_locked_files(self):
        v3, v4 = lock_entries(V3_LOCK), lock_entries(LOCK)
        self.assertEqual(set(v4), set(v3) | {'src/rt30/tb_rt30_v4.py'})
        self.assertEqual({p for p in v3 if v3[p] != v4[p]}, {'src/rt30/native/build.py'})
        for relative, digest in v4.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest, relative)


if __name__ == '__main__':
    unittest.main()
