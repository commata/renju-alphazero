"""H7-R evaluation contracts v2: the v1 contract checks with the v2 seed reservation (``hybrid.h7r_seeds``).

``hybrid.h7r_eval`` is not changed and keeps checking v1 contracts against its own block; nothing here rebinds its
names. A v2 contract (``CONTRACT_FORMAT``) carries ``seed_policy: 'h7r-seeds-v2'``; its checks are the v1 checks of a
development contract (purpose ``pilot``): models and their identities, arms (names, models, budgets, pairs, suite file
SHA-256, pairs within the suite's openings), the pairing rule for shared seeds (one pairing draws one sample:
the same seed, pairs and suite) and the statistics block, with the seeds held to the v2 Arena block
instead of the v1 one. A v1 checker refuses a v2 contract (its format), and a v2 checker refuses a v1 one.

Seeds are unique across every contract of ``docs/h7r/contracts`` (v1) and ``CONTRACTS_DIR`` (v2). Playing and arm-record
checks reuse ``hybrid.h7r_eval`` / ``scripts/h7r_eval.py`` functions; the v2 play entry is built with the Arena stage.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import h7r_eval as ev
from . import h7r_openings as op
from . import h7r_seeds as seeds

CONTRACT_FORMAT = 'h7r-eval-contract-v2'
CONTRACTS_DIR = 'docs/h7r/contracts-v2'
PURPOSES = ('pilot',)
require = ev.require


def check_contract(contract: dict, path, root, check_files: bool = True) -> None:
    root = Path(root)
    require(contract.get('format') == CONTRACT_FORMAT, f'{path}: not an {CONTRACT_FORMAT} contract')
    require(contract.get('seed_policy') == seeds.POLICY, f'{path}: seed policy is not {seeds.POLICY}')
    require(isinstance(contract.get('contract_id'), str) and contract['contract_id'], f'{path}: no contract id')
    require(contract.get('purpose') in PURPOSES and contract.get('status') in ('draft', 'fixed'),
            f'{path}: purpose / status (v2 contracts are development contracts: purpose pilot)')
    models = contract.get('models') or {}
    for name, m in models.items():
        require(set(m) >= {'policy', 'value', 'identity'}, f'{path}: model {name} needs policy, value, identity')
        ev.check_model_identity(m['identity'])
        if check_files:
            actual = ev.model_identity(root / m['policy'], root / m['value'])
            require(actual is not None and actual == m['identity'],
                    f'{path}: model {name} files are missing or not its recorded identity')
    arms = contract.get('arms') or []
    require(arms, f'{path}: no arms')
    names, used = set(), []
    for arm in arms:
        require(arm.get('name') not in names, f"{path}: arm {arm.get('name')} twice")
        names.add(arm['name'])
        for side in ('tested', 'opponent'):
            require(arm[side]['model'] in models and arm[side]['budget'] in ev.BUDGETS,
                    f"{path}: arm {arm['name']} {side} model / budget")
        require(isinstance(arm.get('pairs'), int) and arm['pairs'] > 0, f"{path}: arm {arm['name']} pairs")
        require(ev.file_sha256(root / arm['suite']['path']) == arm['suite']['sha256'],
                f"{path}: arm {arm['name']} suite file is not its SHA-256")
        suite = json.loads((root / arm['suite']['path']).read_text(encoding='utf-8'))
        op.check_suite(suite)
        require(arm['pairs'] <= suite['count'], f"{path}: arm {arm['name']} asks {arm['pairs']} pairs of a suite of "
                                                f"{suite['count']} openings (no repeats)")
        seeds.check_arena_seed(arm.get('seed'))
        used.append(arm['seed'])
    require(len(set(used)) == len(used) or ev._paired_seeds(contract), f'{path}: a seed used twice outside a pairing')
    by_pairing = {}
    for arm in arms:  # one pairing = one sample: the same seed, pairs and suite (D' is paired opening by opening)
        if arm.get('pairing') is not None:
            by_pairing.setdefault(arm['pairing'], set()).add(
                (arm['seed'], arm['pairs'], arm['suite']['path'], arm['suite']['sha256']))
    unequal = sorted(p for p, samples in by_pairing.items() if len(samples) > 1)
    require(not unequal, f'{path}: pairings {unequal} draw different samples (seed, pairs or suite differ)')
    stats = contract.get('stats') or {}
    b = stats.get('bootstrap') or {}
    require(0 < stats.get('alpha_one_sided', 0) < 0.5 and b.get('method') == 'percentile'
            and b.get('unit') == 'opening pair' and isinstance(b.get('samples'), int) and b['samples'] >= 1000
            and isinstance(b.get('seed'), int), f'{path}: stats')


def _contracts(root) -> list[Path]:
    root = Path(root)
    return sorted(p.resolve() for folder in (root / 'docs/h7r/contracts', root / CONTRACTS_DIR) if folder.exists()
                  for p in folder.glob('*.json'))


def load_contract(path, root, check_files: bool = True) -> tuple[dict, str]:
    """A valid v2 contract and its SHA-256; its seeds used by no other contract (v1 or v2) and its id unique."""
    path = Path(path)
    contract = json.loads(path.read_text(encoding='utf-8'))
    check_contract(contract, path, root, check_files)
    mine = ev.contract_seeds(contract)
    for other in _contracts(root):
        if other == path.resolve():
            continue
        theirs = json.loads(other.read_text(encoding='utf-8'))
        require(theirs.get('contract_id') != contract['contract_id'], f'{other}: the same contract id')
        shared = mine & ev.contract_seeds(theirs)
        require(not shared, f'{path}: seeds {sorted(shared)} are already used by {other.name}')
    return contract, ev.file_sha256(path)
