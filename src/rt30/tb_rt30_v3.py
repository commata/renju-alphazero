"""TB-RT30-v3 candidate: TB-RT30-v1 + X2 (V8-B in a parallel helper) + X1 (exact speed-ups) (docs/web-ser5.md §9.8).

The two changes are of different kinds. X1-A is an exact computational speed-up: the same windows
in the same order, so the same nodes and moves, only faster. X2 is TB1-order-preserving parallel
scheduling: whenever V8-B and the rest of the move finish before the deadline the move is
TB-Engine-v1's, but under a deadline it spends the time differently from v1 (V8-B gets the whole
move instead of an 8 s cap), so it may decide differently from v1 on purpose. Every other value is
v1's, read from v1's frozen configuration.

``make_agent`` refuses to build an engine that is not this one: without the native threat-window
scan (``python -m rt30.native.build``) unless ``allow_python=True`` (same moves, about half the
speed), and without the V8-B helper process (for example inside a daemonic worker, which may not
have children) unless ``allow_inline_attack=True`` (then V8-B runs inline with v1's cap: v1's
timing, not X2). ``STATUS`` stays 'candidate' until v3 passes the gates of
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


def make_agent(policy, seed: int = 42, *, uniform_prior: bool = False, allow_python: bool = False,
               allow_inline_attack: bool = False):
    """``RT30AgentV3`` with this configuration; installs the exact speed-ups in this process."""
    from . import attack_helper, fast3
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
    if TB_RT30_V3['parallel_attack'] and not allow_inline_attack and attack_helper.get_helper() is None:
        raise RuntimeError(f'TB-RT30-v3 needs the V8-B helper process (X2), unavailable here: '
                           f'{attack_helper.unavailable_reason()}; run the engine in a non-daemonic process')
    agent = RT30AgentV3(seed=seed, root_policy=policy, timing=dict(TB_RT30_V3['timing']),
                        attack_order=TB_RT30_V3['attack_order'], root_rescue=TB_RT30_V3['root_rescue'],
                        parallel_attack=TB_RT30_V3['parallel_attack'], **config)
    agent.name = NAME if not uniform_prior else f'{NAME} (uniform prior)'
    agent.exact_speedups = active
    return agent
