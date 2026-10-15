"""TB-RT30-v2 candidate: TB-RT30-v1 plus the V8-B slack retry (docs/web-ser5.md §9.5).

v2-A of the A3 plan: the only change from TB-RT30-v1 (``rt30.tb_rt30_v1``, HARM in A3) is
``attack_retry`` (``rt30.engine_v2``). Every other value is v1's, read from v1's frozen
configuration so the two cannot drift apart. ``STATUS`` stays 'candidate' until v2 passes the
gates of §9.5; it is then frozen with its own lock file (``tests/frozen_tb_rt30_v2.sha256``),
and v1's lock is never touched.
"""
from __future__ import annotations

from .tb_rt30_v1 import TB_RT30_V1

NAME = 'TB-RT30-v2'
STATUS = 'candidate'
BASE_ENGINE = 'TB-RT30-v1'
TB_RT30_V2 = {
    **TB_RT30_V1,
    'timing': {**TB_RT30_V1['timing'], 'attack_retry_min': 1.0},
    'attack_retry': True,   # resume a soft-capped V8-B in the time left before the final margin
}


def config_hash() -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(TB_RT30_V2, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_agent(policy, seed: int = 42, *, uniform_prior: bool = False):
    """``RT30AgentV2`` with this configuration; installs the exact speed-ups in this process."""
    from . import fast
    from .engine_v2 import RT30AgentV2

    if TB_RT30_V2['attack_scheduler'] != 'fair':
        raise ValueError('the V8-B retry resumes the fair scheduler only')
    config = dict(TB_RT30_V2['base_config'])
    if uniform_prior:
        config['puct_prior'] = 'uniform'
        policy = None
    elif policy is None:
        raise ValueError('TB-RT30-v2 needs the H3 policy (or uniform_prior=True for a test engine)')
    fast.install_all()
    agent = RT30AgentV2(seed=seed, root_policy=policy, timing=dict(TB_RT30_V2['timing']),
                        attack_order=TB_RT30_V2['attack_order'], root_rescue=TB_RT30_V2['root_rescue'],
                        attack_retry=TB_RT30_V2['attack_retry'], **config)
    agent.name = NAME if not uniform_prior else f'{NAME} (uniform prior)'
    return agent
