"""TB-V3 Resolve / fast5 with candidate models: the H7-c self-play and promotion-arena factory (never canonical).

The engine and its runtime are the frozen ones: ``tb_v3.check_identity`` (engine hash, 63-file manifest), the fast4
and fast5 sources and a fresh, bound fast5 build. Only the models change: they are
not compared with ``tb_v3.MODELS`` but get their own identity (the SHA-256 of both checkpoints and both metadata
files), which every self-play and arena record carries. fast5's runtime hash holds no model, so its G0 / G1 / G2 and
CPU certification records stay those of this runtime.

Not a canonical path: ``analysis.canonical`` never returns this factory, and the web server, the frozen benchmarks
and the certification records keep the frozen H3 policy / H6 value. A candidate model replaces them only through
its own promotion step after the H7-c gates (docs/h6-value.md, H7-c1).

The agent is built here, not by ``tb_v3_runtime._build``: the frozen ``tb_v2_value.make_agent`` refuses any value
checkpoint but the frozen H6 one (its SHA is a term of TB-V2-Value's ``config_hash``, so of TB-V3's engine hash).
``_build`` makes the same agent from the same frozen configuration (``MCTSV8Agent`` with ``TB_V2_VALUE`` and
``ENGINE_OPTIONS``, wrapped in ``E4v2AttackAgent`` with ``tb_v3.ATTACK``); ``tests/test_tb_v3_net.py`` replays an
arena record through both builds. The engine hash in its records therefore names the frozen search configuration
(whose value term is the frozen H6 model); the models actually used are in ``models`` / ``model_hash``.

Compatibility checked before building (fails closed):
    policy   an H3-style policy-only ``RootPolicy`` (policy_trained true, value_trained false) loaded from the path
    value    an H6-2a ``ValueLeaf`` loaded from the path, its file SHA as the leaf reports it; trunk and policy head
             frozen on the frozen H3 network (c1: value head retrained only)
    both     the frozen architecture (``ARCHITECTURE``); the encoder and action index versions are enforced by
             ``model.checkpoint.load_checkpoint``
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

ENGINE_ID = 'TB-V3'
RUNTIME = 'fast5'
NAME = 'TB-V3 Resolve/fast5 + net'
CANONICAL = False
ARCHITECTURE = {'board_size': 15, 'input_planes': 6, 'channels': 64, 'blocks': 4, 'policy_channels': 2,
                'value_channels': 1, 'value_hidden': 64}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_identity(policy_checkpoint, value_checkpoint) -> dict:
    """The four SHA-256 of a model pair, keyed as ``tb_v3.MODELS``."""
    policy_checkpoint, value_checkpoint = Path(policy_checkpoint), Path(value_checkpoint)
    return {'policy_checkpoint_sha256': _sha(policy_checkpoint),
            'policy_metadata_sha256': _sha(policy_checkpoint.with_suffix('.json')),
            'value_checkpoint_sha256': _sha(value_checkpoint),
            'value_metadata_sha256': _sha(value_checkpoint.with_name('best.json'))}


def model_hash(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def agent_name(identity: dict) -> str:
    return f"{NAME} {identity['policy_checkpoint_sha256'][:8]}/{identity['value_checkpoint_sha256'][:8]}"


def check_models(policy, leaf, policy_checkpoint, value_checkpoint) -> dict:
    """Candidate models compatible with the frozen engine, loaded from exactly these paths; their identity."""
    from hybrid.h4_policy import RootPolicy
    from hybrid.h6_value_leaf import ValueLeaf

    from . import tb_v3

    if policy_checkpoint is None or value_checkpoint is None:
        raise ValueError(f'{NAME} needs both checkpoint paths to record its models')
    if not isinstance(policy, RootPolicy) or not isinstance(leaf, ValueLeaf):
        raise TypeError(f'{NAME} needs a RootPolicy and a ValueLeaf, got {type(policy).__name__} and '
                        f'{type(leaf).__name__}')
    for model, path, what in ((policy, policy_checkpoint, 'policy'), (leaf, value_checkpoint, 'value')):
        if Path(model.checkpoint).resolve() != Path(path).resolve():
            raise ValueError(f'the {what} was not loaded from {path} (it reports {model.checkpoint})')
        if asdict(model.model.config) != ARCHITECTURE:
            raise ValueError(f'the {what} network is not the frozen architecture: {asdict(model.model.config)}')
    identity = model_identity(policy_checkpoint, value_checkpoint)
    if leaf.sha256 != identity['value_checkpoint_sha256']:
        raise ValueError(f'the value leaf reports {leaf.sha256[:12]}..., the file is '
                         f"{identity['value_checkpoint_sha256'][:12]}...")
    meta = leaf.meta
    if (meta.get('trunk_frozen') is not True or meta.get('policy_head_frozen') is not True
            or meta.get('parent_h3_checkpoint_sha256') != tb_v3.MODELS['policy_checkpoint_sha256']):
        raise ValueError(f'{value_checkpoint}: not a value head trained on the frozen H3 trunk')
    return identity


def _build(policy, leaf, seed: int):
    """``tb_v3_runtime._build`` without the frozen value-checkpoint check: the same agent, any compatible models."""
    from . import tb_v2_runtime, tb_v2_value, tb_v3
    from .mcts_v8_agent import MCTSV8Agent
    from .tb_v2_value_e4v2 import E4v2AttackAgent

    if tb_v2_value.config_hash() != tb_v2_runtime.ENGINE_HASH:
        raise RuntimeError(f'{NAME}: analysis.tb_v2_value is not the frozen configuration')
    tb_v2_runtime.check_bound()
    base = MCTSV8Agent(seed=seed, root_policy=policy, leaf_value=leaf, **tb_v2_value.TB_V2_VALUE,
                       **tb_v2_value.ENGINE_OPTIONS)
    base.name = tb_v2_runtime.NAME
    return E4v2AttackAgent(base, attack=True, config=tb_v3.ATTACK)


def make_tb_v3_net_agent(policy, leaf, seed: int, policy_checkpoint, value_checkpoint):
    """TB-V3 Resolve on fast5 with these candidate models for one game, after every check."""
    from . import tb_v3, tb_v3_fast5, track_b_guard

    tb_v3.check_identity()
    identity = check_models(policy, leaf, policy_checkpoint, value_checkpoint)
    track_b_guard.check_runtime_sources()
    tb_v3_fast5.check_sources()
    tb_v3_fast5.check_bound()
    agent = _build(policy, leaf, seed)
    agent.name = agent_name(identity)
    agent.models = identity
    return agent


def provenance(device: str, identity: dict) -> dict:
    """fast5's runtime provenance plus the candidate models; never canonical."""
    from . import tb_v3, tb_v3_fast5

    return {**tb_v3_fast5.provenance(device), 'runtime': agent_name(identity), 'canonical': CANONICAL,
            'models': identity, 'model_hash': model_hash(identity), 'frozen_models': identity == tb_v3.MODELS}
