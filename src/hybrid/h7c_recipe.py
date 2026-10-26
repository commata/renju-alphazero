"""H7-c recipe decisions: a cycle's recipe fixed before its self-play, read by the tools that must obey it.

``docs/h7c/c<c>/decision.json`` (``FORMAT``) for a ``VALUE_ONLY`` cycle (docs/h7-c1.md, c3 approval): the policy is
frozen at the parent's, only the value is trained (the c1 / c2 value recipe), no search retarget. The value trainer
(``--recipe-decision``) checks its config, parent, replay anchor and self-play run seed against it and writes its
SHA-256 into ``record.json`` and ``best.json``; the Arena and X2 (``--candidate-mode VALUE_ONLY --recipe-decision``)
accept a candidate only with that SHA-256, the parent policy byte for byte and the seeds the decision names.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FORMAT = 'h7c-recipe-decision-v1'
VALUE_ONLY = 'VALUE_ONLY'
VALUE_ONLY_FROM_CYCLE = 3  # approved after the c2 Phase B review; c1 / c2 are not retroactive
MODEL_KEYS = ('policy_checkpoint_sha256', 'policy_metadata_sha256', 'value_checkpoint_sha256',
              'value_metadata_sha256')


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def load(path, cycle: int) -> tuple[dict, str]:
    """A ``VALUE_ONLY`` recipe decision of this cycle (every field the tools rely on) and its SHA-256."""
    decision = json.loads(Path(path).read_text(encoding='utf-8'))
    problems = {
        'format': decision.get('format') == FORMAT,
        'cycle': decision.get('cycle') == cycle,
        'from cycle 3': cycle >= VALUE_ONLY_FROM_CYCLE,
        'recipe / candidate mode': (decision.get('recipe'), decision.get('candidate_mode')) == (VALUE_ONLY, VALUE_ONLY),
        'policy frozen at the parent': decision.get('policy') == {'action': 'freeze_parent', 'training': False},
        'no retarget': decision.get('retarget') is False,
        'parent models': isinstance(decision.get('parent'), dict) and set(decision['parent']) == set(MODEL_KEYS)
        and all(_sha(v) for v in decision['parent'].values()),
        'champion record': _sha(decision.get('champion_record_sha256')),
        'value recipe': _sha((decision.get('value') or {}).get('config_sha256'))
        and _sha((decision.get('value') or {}).get('replay_anchor_labels_sha256')),
        'seeds': all(isinstance((decision.get('seeds') or {}).get(k), int) for k in ('selfplay_run_seed', 'arena', 'x2')),
    }
    failed = sorted(k for k, ok in problems.items() if not ok)
    if failed:
        raise ValueError(f'{path}: not a {VALUE_ONLY} recipe decision of cycle {cycle} ({", ".join(failed)})')
    return decision, file_sha256(path)
