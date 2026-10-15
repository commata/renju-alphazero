"""TB-RT30-v3 candidate: TB-RT30-v1 + X2 (V8-B in a parallel helper) + X1 (exact speed-ups) (docs/web-ser5.md §9.8).

Both changes are exact: whenever V8-B, and the rest of the move, finish before the deadline the
move is TB-Engine-v1's. Every other value is v1's, read from v1's frozen configuration. The native
threat-window scan (X1-A) must be built on the machine (``python -m rt30.native.build``);
``make_agent`` refuses to run without it unless ``allow_python=True`` (same moves, about half the
speed: not the engine being evaluated). ``STATUS`` stays 'candidate' until v3 passes the gates of
§9.8; it is then frozen with its own lock file, and v1's lock is never touched.
"""
from __future__ import annotations

from .tb_rt30_v1 import TB_RT30_V1

NAME = 'TB-RT30-v3'
STATUS = 'candidate'
BASE_ENGINE = 'TB-RT30-v1'
TB_RT30_V3 = {
    **TB_RT30_V1,
    'parallel_attack': True,   # X2: V8-B in the resident helper, the whole move's time, no cap
    'exact_speedups': ['threat_windows_native', 'winning_extensions_in_direction'],  # X1-A
}


def config_hash() -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(TB_RT30_V3, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_agent(policy, seed: int = 42, *, uniform_prior: bool = False, allow_python: bool = False):
    """``RT30AgentV3`` with this configuration; installs the exact speed-ups in this process."""
    from . import fast3
    from .engine_v3 import RT30AgentV3

    config = dict(TB_RT30_V3['base_config'])
    if uniform_prior:
        config['puct_prior'] = 'uniform'
        policy = None
    elif policy is None:
        raise ValueError('TB-RT30-v3 needs the H3 policy (or uniform_prior=True for a test engine)')
    active = fast3.install_all()
    if active != TB_RT30_V3['exact_speedups'] and not allow_python:
        raise RuntimeError(f'TB-RT30-v3 needs {TB_RT30_V3["exact_speedups"]}, active {active}: '
                           'build the native scan with python -m rt30.native.build')
    agent = RT30AgentV3(seed=seed, root_policy=policy, timing=dict(TB_RT30_V3['timing']),
                        attack_order=TB_RT30_V3['attack_order'], root_rescue=TB_RT30_V3['root_rescue'],
                        parallel_attack=TB_RT30_V3['parallel_attack'], **config)
    agent.name = NAME if not uniform_prior else f'{NAME} (uniform prior)'
    agent.exact_speedups = active
    return agent
