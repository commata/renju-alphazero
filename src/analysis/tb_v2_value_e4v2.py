"""TB-V2-Value + E4-v2: the H7-E4-v2 candidate (docs/h6-value.md, "E4-v2 후보 동결").

A candidate identity, not an adopted engine: COSTTEST and the seed 8441 / 8442 arena judge it.

    base     the frozen TB-V2-Value, built only by ``track_b_guard.make_canonical_agent`` (policy / value /
             runtime sources / code manifest checked, engine hash and fast4 binding checked)
    attack   the DEV selection ``150k/cd/killer`` of ``analysis.e4v2_attack`` (``ATTACK``): after the base
             chooses, ``e4v2_attack.decide`` (route stage4 / stage5 / tree, the cooldown, the opponent-VCF gate,
             then the E4 attack with killer replies at 150k nodes / 15k calls); a proven WIN replaces the base
             move (route ``own_vct2``)
State: the only thing kept from one move to the next is the cooldown bit (``E4v2AttackAgent.cooldown``, one
per agent, i.e. per game; a move of another game clears it, ``new_game`` resets it). The attack builds a new
solver at every call; the base keeps what the frozen TB-V2-Value keeps (its random stream; no tree).
Identity (``candidate_hash``): the base engine hash, ``ATTACK``, the SHA-256 of the attack sources
(``ATTACK_SOURCES``) and of the canonical code manifest (``MANIFEST_SHA256``), all checked by ``check_sources``.
These are the files the DEV run measured (``docs/h7/h7_e4v2_dev.json`` fingerprint, code ``7caf38a``).
"""
from __future__ import annotations

from pathlib import Path

NAME = 'TB-V2-Value+E4v2'
BASE = 'TB-V2-Value'
BASE_ENGINE_HASH = '256e5abe028c694891d33b9be1b9f58287fe18b6d4af278822b03cf09b7ec5c1'
ATTACK = {'name': '150k/cd/killer', 'budget': '150k', 'cooldown': True, 'killer': True,
          'budget_values': {'node_limit': 20_000, 'call_limit': 15_000, 'node_budget': 150_000},
          'gate_node_limit': 20_000, 'routes': ['stage4', 'stage5', 'tree']}
ATTACK_SOURCES = {
    'src/analysis/e4v2_attack.py': 'dd61f5af34d4ed26fa4e58fa6d43f3b09902136914610673c2a226cbca12de47',
    'src/analysis/e4_attack.py': '73b0902d9d479065b9279e43ca356509dfdec4daa1a759f881aa695009b7415f',
    'src/analysis/vct2_attack.py': 'ad263e3fc981f186f3196cdd914f646dc6d864038f412cb1e728c5a254515c15',
    'src/analysis/selective_vct.py': 'b0b3a0637ea91a6808a60521dfdda447958020f52cf2ec17ac29c10be369889b',
}
MANIFEST = 'tests/track_b_frozen.sha256'
MANIFEST_SHA256 = '7b73005da72a39fc0a4c184e1667a882fc92a08bc109950adac0ba198bab43ee'
SELECTION = {'dev_result': 'docs/h7/h7_e4v2_dev.json', 'dev_commit': '77901c7', 'dev_tool_commit': '7caf38a'}
ROUTE = 'own_vct2'
_ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources() -> None:
    """Raise unless the attack code, the code manifest, the base engine and the configuration are the frozen ones."""
    from . import e4v2_attack, tb_v2_value
    from .track_b_guard import check_code

    for name, digest in {**ATTACK_SOURCES, MANIFEST: MANIFEST_SHA256}.items():
        actual = _sha(_ROOT / name)
        if actual != digest:
            raise RuntimeError(f'{NAME}: {name} is not the frozen file ({actual[:12]}...)')
    bad = check_code()
    if bad:
        raise RuntimeError(f'{NAME}: the canonical code differs from its manifest: {bad}')
    if tb_v2_value.config_hash() != BASE_ENGINE_HASH:
        raise RuntimeError(f'{NAME}: analysis.tb_v2_value is not the frozen base engine')
    selected = {k: ATTACK[k] for k in ('name', 'budget', 'cooldown', 'killer')}
    if (selected not in e4v2_attack.GRID or e4v2_attack.BUDGETS[ATTACK['budget']] != ATTACK['budget_values']
            or e4v2_attack.GATE_NODE_LIMIT != ATTACK['gate_node_limit']
            or list(e4v2_attack.ROUTES) != ATTACK['routes']):
        raise RuntimeError(f'{NAME}: analysis.e4v2_attack does not define the selected configuration')


def candidate_hash() -> str:
    import hashlib
    import json

    payload = {'base_engine_hash': BASE_ENGINE_HASH, 'attack': ATTACK, 'attack_sources': ATTACK_SOURCES,
               'manifest_sha256': MANIFEST_SHA256}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class E4v2AttackAgent:
    """The base agent, then the E4-v2 step; one cooldown bit per agent (one agent per game)."""

    def __init__(self, base, attack: bool = True, config: dict | None = None):
        from .e4v2_attack import Cooldown

        self.base = base
        self.attack = attack
        self.config = dict(config or ATTACK)
        self.name = NAME if attack else base.name
        self.cooldown = Cooldown()
        self.attack_info: dict = {}

    @property
    def diagnostics(self):
        return self.base.diagnostics

    def new_game(self) -> None:
        """Reset the only state kept between moves (a move of another game also clears it on its own)."""
        from .e4v2_attack import Cooldown

        self.cooldown = Cooldown()

    def select_move(self, game):
        from .e4v2_attack import decide

        move = self.base.select_move(game)
        self.attack_info = {}
        if not self.attack:
            return move
        route = self.base.diagnostics.v8_route
        attack, info = decide(game, self.config, route, self.cooldown)
        self.attack_info = {'base_route': route, 'base_move': list(move), **info}
        if attack is None:
            return move
        diag = self.base.diagnostics
        diag.v8_route = ROUTE
        diag.v8_changed = attack != move
        return attack


def wrap(base_agent, attack: bool = True) -> E4v2AttackAgent:
    check_sources()
    return E4v2AttackAgent(base_agent, attack)


def make_agent(policy, leaf, seed: int, policy_checkpoint, value_checkpoint, attack: bool = True) -> E4v2AttackAgent:
    """The candidate on the canonical TB-V2-Value + fast4 (``attack=False``: the canonical's moves)."""
    from . import tb_v2_runtime as rt
    from .track_b_guard import make_canonical_agent

    agent = wrap(make_canonical_agent(policy, leaf, seed, policy_checkpoint, value_checkpoint), attack)
    if attack:
        agent.name = f'{NAME}/{rt.FAST_MODE}'
    return agent


def runtime_hash(device: str) -> str:
    import hashlib
    import json

    from . import tb_v2_runtime as rt

    payload = {'candidate_hash': candidate_hash(), 'fast_mode': rt.FAST_MODE, 'native_sources': rt.NATIVE_SOURCES,
               'binding_sources': rt.BINDING_SOURCES, 'value_device': rt.device_kind(device)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def provenance(device: str) -> dict:
    from . import tb_v2_runtime as rt

    base = rt.provenance(device)
    return {**base, 'engine': NAME, 'candidate': True, 'candidate_hash': candidate_hash(),
            'base_engine_hash': BASE_ENGINE_HASH, 'attack': ATTACK, 'attack_sources': ATTACK_SOURCES,
            'manifest_sha256': MANIFEST_SHA256, 'runtime': f'{NAME}/{rt.FAST_MODE}',
            'runtime_hash': runtime_hash(device), 'base_runtime_hash': base['runtime_hash'], 'certified': None}
