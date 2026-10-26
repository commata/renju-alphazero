"""Pre-registered B2 VALUE_ONLY program, deterministic decisions and stop rules.

B1 verdicts keep their original promotion/convergence meaning. A drift alarm stops the executor AFTER
recording that verdict; it never rewrites a promotion or silently rolls back a champion.
"""
from __future__ import annotations

import json
from pathlib import Path

from .h7c_recipe import FORMAT as DECISION_FORMAT, MODEL_KEYS, file_sha256

ROOT = Path(__file__).resolve().parents[2]
FORMAT = 'h7c-b2-program-v1'
DRIFT_SEEDS = frozenset(range(8751, 8761))  # evaluation only, reserved before B2
ARMS = ('official', 'mon_v8_full', 'mon_puct_policy', 'mon_e2')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def contained(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), 'expected a relative path')
    result = (Path(root) / relative).resolve()
    require(result.is_relative_to(Path(root).resolve()), f'path outside repository: {relative}')
    return result


def load_program(path, root=None):
    root = ROOT if root is None else Path(root)
    p = read(path)
    require(p.get('format') == FORMAT and p.get('recipe') == 'VALUE_ONLY', 'not a B2 VALUE_ONLY program')
    require(type(p.get('first_cycle')) is int and p['first_cycle'] == 4
            and type(p.get('last_cycle')) is int and 4 <= p['last_cycle'] <= 10, 'B2 covers c4..c10 only')
    require(p.get('seeds') == {'selfplay_base': 8800, 'arena_base': 8730, 'x2_base': 8710, 'drift_base': 8750},
            'B2 seed registration differs')
    require(p.get('stops') == {'consecutive_no_promotion': 3, 'consecutive_tactical_fail': 2,
                             'integrity_error': 'immediate'}, 'B2 stop rules differ')
    require(p.get('drift') == {'pairs': 50, 'improvement_upper_below': 0.0,
                             'reference': 'frozen_H3_H6_v3', 'invalid': 'stop'}, 'B2 drift rule differs')
    require(p.get('execution') == {'games': 1000, 'workers': 6, 'label_workers': 12,
                                 'label_fast': 'fast4', 'x2_workers': 4}, 'B2 execution recipe differs')
    require(p.get('promotion') == {'score_above': 0.5, 'improvement_lower_min': -0.05,
                                 'tactical_rule': 'h7c-tactical-v2'}, 'B1 promotion rule must be preserved')
    start = p['start']
    champion_path = contained(root, start['verdict'])
    require(file_sha256(champion_path) == start['verdict_sha256'], 'B2 start verdict SHA mismatch')
    champion = read(champion_path)
    require(champion.get('cycle') == 3 and champion.get('format') == 'h7c-arena-v1-verdict'
            and champion.get('next_champion') == start['models'], 'B2 start champion mismatch')
    require(set(p['frozen_models']) == set(MODEL_KEYS), 'invalid frozen model identity')
    from analysis import tb_v3
    require(p['frozen_models'] == tb_v3.MODELS, 'drift reference is not the frozen H3/H6 v3')
    require(all(start['models'][k] == p['frozen_models'][k] for k in MODEL_KEYS[:2]), 'B2 freezes H3 policy')
    require(file_sha256(contained(root, p['value']['config'])) == p['value']['config_sha256'],
            'B2 value config SHA mismatch')
    require(p['value'].get('h6_guard_ratio') == 1.05 and p['value'].get('mix') == 'cycle:replay = 50:50 per epoch',
            'B2 value recipe must be c3 unchanged')
    return p, file_sha256(path)


def seeds(program, cycle):
    require(program['first_cycle'] <= cycle <= program['last_cycle'], 'cycle outside approved B2 range')
    return {'selfplay_run_seed': 8800 + cycle, 'arena': 8730 + cycle, 'x2': 8710 + cycle,
            'drift': 8750 + cycle}


def decision(program, program_sha, champion, champion_sha, cycle, program_path='docs/h7c/b2_program.json'):
    require(champion.get('format') == 'h7c-arena-v1-verdict' and champion.get('cycle') == cycle - 1,
            'decision needs the previous cycle verdict')
    parent = champion['next_champion']
    require(set(parent) == set(MODEL_KEYS), 'invalid parent identity')
    require(all(parent[k] == program['frozen_models'][k] for k in MODEL_KEYS[:2]), 'parent policy changed')
    if cycle == program['first_cycle']:
        require(champion_sha == program['start']['verdict_sha256'], 'wrong B2 initial champion record')
    else:
        require(champion.get('b2_program_sha256') == program_sha, 'previous verdict belongs to another program')
    return {'format': DECISION_FORMAT, 'cycle': cycle, 'recipe': 'VALUE_ONLY', 'candidate_mode': 'VALUE_ONLY',
            'parent': parent, 'champion_record_sha256': champion_sha,
            'policy': {'action': 'freeze_parent', 'training': False}, 'retarget': False,
            'value': program['value'], 'seeds': seeds(program, cycle),
            'arena': {'official_pairs': 50, 'promotion': program['promotion'],
                      'diagnosis': {'p_only': 'NOT_APPLICABLE', 'v_only': 'SAME_AS_OFFICIAL'},
                      'monitoring': list(ARMS[1:]), 'x2': 'record only, never a gate'},
            'program': {'path': program_path, 'sha256': program_sha},
            'approved_by': 'pre_registered_program'}


def validate_decision(d, root=None):
    root = ROOT if root is None else Path(root)
    ref = d['program']
    program_path = contained(root, ref['path'])
    p, sha = load_program(program_path, root)
    require(ref['sha256'] == sha, 'recipe program SHA mismatch')
    previous_path = contained(root, f"docs/h7c/c{d['cycle'] - 1}/arena_verdict.json")
    expected = decision(p, sha, read(previous_path), file_sha256(previous_path), d['cycle'], ref['path'])
    require(d == expected, 'decision is not the deterministic result of its B2 program and champion')
    return p, sha


def drift_judge(summary):
    valid = (summary['games'] == 100 and summary['complete_pairs'] == 50
             and not summary['incomplete_pairs'] and not summary['truncated']
             and not summary['opponent_illegal_pairs'] and not any(summary['illegal'].values())
             and summary['improvement_ci95'] is not None)
    alarm = valid and summary['improvement_ci95'][1] < 0
    return {'valid': valid, 'alarm': alarm, 'stop': not valid or alarm,
            'reason': 'DRIFT_INVALID' if not valid else 'DRIFT_DETECTED' if alarm else 'DRIFT_NOT_DETECTED',
            'rule': '50 complete pairs; stop if improvement 95% upper < 0; invalid stops'}


def stop_reason(program, records, drift=None):
    if drift and drift['stop']:
        return drift['reason']
    if not records:
        return None
    if records[-1].get('official') is not None and records[-1]['official'].get('valid') is False:
        return 'OFFICIAL_INVALID'
    n = program['stops']['consecutive_no_promotion']
    if len(records) >= n and all(v['promotion']['decision'] != 'PROMOTE' for v in records[-n:]):
        return 'NO_PROMOTION_STREAK'
    n = program['stops']['consecutive_tactical_fail']
    if len(records) >= n and all(v['tactical']['verdict'] == 'FAIL' for v in records[-n:]):
        return 'TACTICAL_FAIL_STREAK'
    if records[-1]['cycle'] >= program['last_cycle']:
        return 'MAX_CYCLE'
    return None
