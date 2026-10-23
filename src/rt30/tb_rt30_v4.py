"""TB-RT30-v4 candidate: TB-RT30-v3 unchanged, built by the TB-V3 native builder (docs/web-ser5.md, TB-V3 deployment).

The web deployment branch carries TB-V3 / fast5, whose frozen ``src/rt30/native/build.py`` also builds fast4's
forbidden-point module (``MODULES`` gains ``'_forbidden'``). That file is locked by TB-RT30-v3
(``tests/frozen_tb_rt30_v3.sha256``), and a change to a locked file makes a new engine: this one. Nothing else
differs: v3's configuration, code and threat-window source (``threat_windows.c``, same flags) are v4's, and
``tests/test_tb_rt30_v3.py`` checks that the builder differs from v3's by that one line.

``STATUS`` stays ``'candidate'`` until v4's same-move gate (v3's equivalence positions replayed on SER5, the same
moves) passes; until then the web offers it only as a legacy / rollback opponent, never the default.
"""
from __future__ import annotations

from .tb_rt30_v3 import FREEZE_EVIDENCE as V3_FREEZE_EVIDENCE
from .tb_rt30_v3 import TB_RT30_V3

NAME = 'TB-RT30-v4'
STATUS = 'candidate'
BASE_ENGINE = 'TB-RT30-v3'
FREEZE_EVIDENCE = dict(V3_FREEZE_EVIDENCE)  # v3's, until v4's own gate record is added
TB_RT30_V4 = {**TB_RT30_V3, 'native_builder': 'tb-v3'}  # rt30/native/build.py of TB-V3 (+ _forbidden)


def config_hash() -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(TB_RT30_V4, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_agent(policy, seed: int = 42, **kwargs):
    """TB-RT30-v3's agent under v4's name (same configuration and code)."""
    from . import tb_rt30_v3

    agent = tb_rt30_v3.make_agent(policy, seed=seed, **kwargs)
    agent.name = agent.name.replace(tb_rt30_v3.NAME, NAME)
    return agent
