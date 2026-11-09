"""TB-V3 Resolve / fast5 with candidate models, factory v2: the widened value head (never canonical).

``analysis.tb_v3_net`` (factory v1) stays as it is: it accepts only the frozen architecture (``value_channels`` 1)
and a value metadata naming the frozen H3 parent. This factory is the H7-R plan's "candidate factory v2" step
(docs/h7-rl-plan.md section 4): a versioned list of value architectures, everything else unchanged.

    engine / runtime   the frozen TB-V3 engine and fast5 runtime, checked exactly as v1 does (``tb_v3.check_identity``,
                       runtime sources, the fast5 sources and binding); the agent is v1's ``_build`` (the frozen search
                       configuration), so a 1-channel model pair plays as under v1
    loader             the frozen ``hybrid.h6_value_leaf.ValueLeaf`` (it reads the metadata ``model_config``)
    policy             the frozen architecture (``tb_v3_net.ARCHITECTURE``): an H3-style policy-only network
    value              one of ``VALUE_ARCHITECTURES``: the frozen one or the D1 widened head (``value_channels`` 4)
    frozen trunk       the value network's trunk and policy head (``frozen_state``) recomputed from the loaded model
                       must be ``FROZEN_STATE_SHA256``, the frozen H3 trunk and policy head every H6 / H7-c value
                       checkpoint carries (pinned and recomputed by the R2-DIAG-1 and R2-D4-BASELINE-1 contracts);
                       a metadata H3 parent, if named, must be the frozen H3 checkpoint (v1's check), and a declared
                       frozen state must be the recomputed one
    identity           v1's model identity (the four SHA-256) and model hash; the provenance adds the factory version
                       and the value architecture

The time cost of a widened head is a separate gate (``scripts/h7r_factory_v2_timing.py``).
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from . import tb_v3_net as v1

FACTORY_VERSION = 'tb-v3-net-v2'
NAME = v1.NAME
CANONICAL = False
POLICY_ARCHITECTURE = dict(v1.ARCHITECTURE)
FROZEN_STATE_SHA256 = 'c2fbaac3597185a6ab751f08819ddc9e44644e873f68361036f7cbdb22e1b41d'
VALUE_ARCHITECTURES = (dict(v1.ARCHITECTURE), {**v1.ARCHITECTURE, 'value_channels': 4})


def value_architecture(leaf) -> dict:
    return asdict(leaf.model.config)


def check_models(policy, leaf, policy_checkpoint, value_checkpoint) -> dict:
    """Candidate models compatible with the frozen engine under factory v2, loaded from exactly these paths; their
    identity (v1's)."""
    from hybrid.h4_policy import RootPolicy
    from hybrid.h6_value_leaf import ValueLeaf
    from hybrid.h6_value_train import frozen_state, state_sha256

    from . import tb_v3

    if policy_checkpoint is None or value_checkpoint is None:
        raise ValueError(f'{NAME} needs both checkpoint paths to record its models')
    if not isinstance(policy, RootPolicy) or not isinstance(leaf, ValueLeaf):
        raise TypeError(f'{NAME} needs a RootPolicy and a ValueLeaf, got {type(policy).__name__} and '
                        f'{type(leaf).__name__}')
    for model, path, what in ((policy, policy_checkpoint, 'policy'), (leaf, value_checkpoint, 'value')):
        if Path(model.checkpoint).resolve() != Path(path).resolve():
            raise ValueError(f'the {what} was not loaded from {path} (it reports {model.checkpoint})')
    if asdict(policy.model.config) != POLICY_ARCHITECTURE:
        raise ValueError(f'the policy network is not the frozen architecture: {asdict(policy.model.config)}')
    if value_architecture(leaf) not in VALUE_ARCHITECTURES:
        raise ValueError(f'the value network is not an architecture of {FACTORY_VERSION}: {value_architecture(leaf)}')
    identity = v1.model_identity(policy_checkpoint, value_checkpoint)
    if leaf.sha256 != identity['value_checkpoint_sha256']:
        raise ValueError(f'the value leaf reports {leaf.sha256[:12]}..., the file is '
                         f"{identity['value_checkpoint_sha256'][:12]}...")
    meta = leaf.meta
    if meta.get('trunk_frozen') is not True or meta.get('policy_head_frozen') is not True:
        raise ValueError(f'{value_checkpoint}: not a value head trained on a frozen trunk')
    parent = meta.get('parent_h3_checkpoint_sha256')
    if parent is not None and parent != tb_v3.MODELS['policy_checkpoint_sha256']:
        raise ValueError(f'{value_checkpoint}: its metadata names another H3 parent ({str(parent)[:12]})')
    value_frozen = state_sha256(frozen_state(leaf.model))
    if value_frozen != FROZEN_STATE_SHA256:
        raise ValueError(f'{value_checkpoint}: its trunk / policy head ({value_frozen[:12]}) are not the frozen ones '
                         f'({FROZEN_STATE_SHA256[:12]})')
    declared = meta.get('frozen_state_sha256')
    if declared is not None and declared != value_frozen:
        raise ValueError(f'{value_checkpoint}: the metadata frozen state {str(declared)[:12]} is not the loaded one '
                         f'{value_frozen[:12]}')
    return identity


def make_agent(policy, leaf, seed: int, policy_checkpoint, value_checkpoint):
    """TB-V3 Resolve on fast5 with these candidate models for one game, after every check (v1's engine checks and
    build, v2's model checks)."""
    from . import tb_v3, tb_v3_fast5, track_b_guard

    tb_v3.check_identity()
    identity = check_models(policy, leaf, policy_checkpoint, value_checkpoint)
    track_b_guard.check_runtime_sources()
    tb_v3_fast5.check_sources()
    tb_v3_fast5.check_bound()
    agent = v1._build(policy, leaf, seed)
    agent.name = v1.agent_name(identity)
    agent.models = identity
    agent.factory = FACTORY_VERSION
    return agent


def provenance(device: str, identity: dict, value_config: dict) -> dict:
    """v1's provenance plus the factory version and the value architecture; never canonical."""
    return {**v1.provenance(device, identity), 'factory': FACTORY_VERSION, 'value_architecture': dict(value_config)}


def load(policy_checkpoint, value_checkpoint, device: str = 'cpu'):
    """The policy and value leaf loaded as the H7-c tools load them (``ValueLeaf`` without the frozen-SHA pin)."""
    from hybrid.h4_policy import RootPolicy
    from hybrid.h6_value_leaf import ValueLeaf

    return RootPolicy(policy_checkpoint, threads=1), ValueLeaf(value_checkpoint, device=device, threads=1,
                                                               expected_sha256=None)


def make_side(side: dict, seed: int, device: str):
    """The v2 counterpart of ``h7c_arena.make_side``: the exact runtime installed and the models loaded (and cached)
    by the arena's own loader, then this factory's checks and build."""
    import h7c_arena as ar

    ar.install_runtime()
    policy, leaf = ar._loaded(side['policy'], side['value'], device)
    return make_agent(policy, leaf, seed, side['policy'], side['value'])
