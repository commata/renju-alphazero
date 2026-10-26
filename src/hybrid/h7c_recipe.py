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
    if 'program' in decision:
        from .h7c_b2 import validate_decision
        validate_decision(decision)
    return decision, file_sha256(path)


def check_value_no_candidate(record: dict, decision: dict, decision_sha: str, path) -> None:
    """Require a clean, complete value-training failure from this decision's parent, data and recipe."""
    inputs = record.get('inputs') or {}
    source = inputs.get('source') or {}
    git = record.get('git') or {}
    problems = {
        'format': record.get('format') == 'h7c-value-train-v1',
        'cycle / verdict': (record.get('cycle'), record.get('verdict')) == (decision['cycle'], 'NO_CANDIDATE'),
        'clean training tree': git.get('dirty') is False and bool(git.get('commit')),
        'parent value pair': (inputs.get('parent_checkpoint_sha256'), inputs.get('parent_metadata_sha256')) == (
            decision['parent']['value_checkpoint_sha256'], decision['parent']['value_metadata_sha256']),
        'self-play models': source.get('models') == decision['parent'],
        'self-play commit': isinstance(source.get('selfplay_git_commit'), str) and bool(source['selfplay_git_commit']),
        'runtime': _sha(source.get('runtime_hash')),
        'source rows': type(source.get('rows')) is int and source['rows'] > 0,
        'cycle labels': _sha(inputs.get('cycle_labels_sha256')),
        'replay anchor': inputs.get('replay_labels_sha256') == decision['value']['replay_anchor_labels_sha256'],
        'value config': inputs.get('config_sha256') == decision['value']['config_sha256'],
        'self-play seed': inputs.get('selfplay_run_seed') == decision['seeds']['selfplay_run_seed'],
        'recipe decision': inputs.get('recipe_decision_sha256') == decision_sha,
    }
    failed = sorted(k for k, ok in problems.items() if not ok)
    if failed:
        raise ValueError(f'{path}: not the value NO_CANDIDATE record of this recipe decision ({", ".join(failed)})')
