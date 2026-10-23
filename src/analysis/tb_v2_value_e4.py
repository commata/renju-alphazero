"""TB-V2-Value + E4: the H7-E4 candidate (docs/h6-value.md, "H7-E4 DEV grid 결과와 후보 선택").

A candidate identity, not an adopted engine: it is what the seed 8441 / 8442 arena judges (ADOPT / REJECT).

    base    the frozen TB-V2-Value (``analysis.tb_v2_value``, ``BASE_ENGINE_HASH``), unchanged
    attack  the DEV selection ``200k/vcf_threat/all`` of ``analysis.e4_attack`` (``ATTACK``): after the base
            chooses, on its routes stage4 / stage5 / tree, the selective depth-2 attack runs (first moves:
            fours, open threes and the local VCF-threat moves; budget 200k VCF nodes / 20k calls / 20k
            nodes per call); a proven WIN replaces the base move (route ``own_vct2``), otherwise the base
            move stands
The attack code is part of the identity: the SHA-256 of ``e4_attack.py``, ``vct2_attack.py`` and
``selective_vct.py`` (``ATTACK_SOURCES``, checked by ``check_sources``) and ``ATTACK`` are hashed with the base
engine hash (``candidate_hash``). ``attack=False`` gives the base agent's moves unchanged (isolation).
The runtime profile is the frozen fast4 one (``tb_v2_runtime``: same native and binding sources), with its
own hash (``runtime_hash``) because the engine differs.
"""
from __future__ import annotations

from pathlib import Path

NAME = 'TB-V2-Value+E4'
BASE = 'TB-V2-Value'
BASE_ENGINE_HASH = '256e5abe028c694891d33b9be1b9f58287fe18b6d4af278822b03cf09b7ec5c1'
ATTACK = {'name': '200k/vcf_threat/all', 'budget': '200k', 'candidates': 'vcf_threat', 'trigger': 'all',
          'budget_values': {'node_limit': 20_000, 'call_limit': 20_000, 'node_budget': 200_000},
          'routes': ['stage4', 'stage5', 'tree']}
ATTACK_SOURCES = {
    'src/analysis/e4_attack.py': '73b0902d9d479065b9279e43ca356509dfdec4daa1a759f881aa695009b7415f',
    'src/analysis/vct2_attack.py': 'ad263e3fc981f186f3196cdd914f646dc6d864038f412cb1e728c5a254515c15',
    'src/analysis/selective_vct.py': 'b0b3a0637ea91a6808a60521dfdda447958020f52cf2ec17ac29c10be369889b',
}
SELECTION = {'dev_result': 'docs/h7/h7_e4_dev.json', 'dev_commit': 'c65cd62', 'dev_tool_commit': '1ec482c'}
ROUTE = 'own_vct2'
_ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources() -> None:
    """Raise unless the attack code is the frozen one and its configuration is the DEV selection."""
    from . import e4_attack, tb_v2_value

    for name, digest in ATTACK_SOURCES.items():
        actual = _sha(_ROOT / name)
        if actual != digest:
            raise RuntimeError(f'{NAME}: {name} is not the frozen attack code ({actual[:12]}...)')
    if tb_v2_value.config_hash() != BASE_ENGINE_HASH:
        raise RuntimeError(f'{NAME}: analysis.tb_v2_value is not the frozen base engine')
    if (e4_attack.BUDGETS[ATTACK['budget']] != ATTACK['budget_values'] or list(e4_attack.ROUTES) != ATTACK['routes']
            or ATTACK['candidates'] not in e4_attack.CANDIDATES or ATTACK['trigger'] not in e4_attack.TRIGGERS):
        raise RuntimeError(f'{NAME}: analysis.e4_attack does not define the selected configuration')


def candidate_hash() -> str:
    """SHA-256 of the canonical JSON of the base engine hash, the attack configuration and its sources."""
    import hashlib
    import json

    payload = {'base_engine_hash': BASE_ENGINE_HASH, 'attack': ATTACK, 'attack_sources': ATTACK_SOURCES}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class E4AttackAgent:
    """The base agent, then the E4 attack on the base's attack routes (a proven WIN replaces its move)."""

    def __init__(self, base, attack: bool = True):
        self.base = base
        self.attack = attack
        self.name = NAME if attack else base.name
        self.attack_info: dict = {}

    @property
    def diagnostics(self):
        return self.base.diagnostics

    def select_move(self, game):
        from .e4_attack import e4_attack, triggered

        move = self.base.select_move(game)
        self.attack_info = {}
        if not self.attack:
            return move
        route = self.base.diagnostics.v8_route
        if not triggered(game, ATTACK, route):
            return move
        attack, info = e4_attack(game, ATTACK)
        self.attack_info = {'base_route': route, 'base_move': list(move), **info}
        if attack is None:
            return move
        diag = self.base.diagnostics
        diag.v8_route = ROUTE
        diag.v8_changed = attack != move
        return attack


def wrap(base_agent, attack: bool = True) -> E4AttackAgent:
    check_sources()
    return E4AttackAgent(base_agent, attack)


def make_agent(policy, leaf_value, seed: int = 42, attack: bool = True) -> E4AttackAgent:
    """The candidate on the frozen TB-V2-Value agent (``attack=False``: the base agent's moves)."""
    from . import tb_v2_value

    return wrap(tb_v2_value.make_agent(policy, leaf_value, seed=seed), attack)


# -- runtime profile ---------------------------------------------------------------------------------------

def runtime_hash(device: str) -> str:
    """The fast4 runtime of this candidate: ``tb_v2_runtime``'s sources with the candidate hash and device."""
    import hashlib
    import json

    from . import tb_v2_runtime as rt

    payload = {'candidate_hash': candidate_hash(), 'fast_mode': rt.FAST_MODE, 'native_sources': rt.NATIVE_SOURCES,
               'binding_sources': rt.BINDING_SOURCES, 'value_device': rt.device_kind(device)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_runtime_agent(policy, leaf_value, seed: int = 42) -> E4AttackAgent:
    """The candidate on the frozen fast4 profile (refused on any other binding)."""
    from . import tb_v2_runtime as rt

    rt.check_bound()
    agent = make_agent(policy, leaf_value, seed=seed)
    agent.name = f'{NAME}/{rt.FAST_MODE}'
    return agent


def provenance(device: str) -> dict:
    from . import tb_v2_runtime as rt

    base = rt.provenance(device)
    return {**base, 'engine': NAME, 'candidate': True, 'candidate_hash': candidate_hash(),
            'base_engine_hash': BASE_ENGINE_HASH, 'attack': ATTACK, 'attack_sources': ATTACK_SOURCES,
            'runtime': f'{NAME}/{rt.FAST_MODE}', 'runtime_hash': runtime_hash(device),
            'base_runtime_hash': base['runtime_hash'], 'certified': None}
