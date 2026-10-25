"""H7-c automation, Auto-v1 (docs/h7c-auto.md): pure state transitions from cycle evidence, no file or git access.

``scripts/h7c_auto.py`` loads the evidence, checks its SHA-256 chain and passes normalized values here. Nothing here
recomputes a verdict: the Arena verdict (``promotion``), the ablation summary (``arms[*].stronger``,
``saturation.reading``) and the decision files are read as recorded. The only numbers are the operational rules
R1-R6 (docs/h7c-auto.md), each named below.

    cycle_stage            the first step of cycle c whose evidence is missing, and the action it needs
    next_cycle_decision    after a cycle's verdict: MANUAL_REVIEW / E2 / BLOCKED_SEED_RANGE / *_TOOLING / NEXT_CYCLE
    sb_branch, e2_state    the E2 episode (search-budget ablation with the champion, saturation, B prescriptions)
    cost_preflight         the retarget's projected time before it starts (R4; the run's own gate is authoritative)
    evidence_only          the paths a commit may touch after the cycle's ``code_base_commit``
"""
from __future__ import annotations

import fnmatch

JOINT, FALLBACK = 'JOINT', 'VALUE_ONLY_FALLBACK'
PROMOTE, KEEP = 'PROMOTE', 'KEEP_CHAMPION'
NO_CANDIDATE, OFFICIAL_INVALID = 'NO_CANDIDATE', 'OFFICIAL_INVALID'

FIRST_CYCLE = 2                  # the automation starts at C2_READY
SELFPLAY_SEED_BASE = 8800        # h7c_selfplay RUN_SEEDS 8801-8999: cycle c = 8800 + c
ARENA_CYCLES = range(1, 11)      # h7c_arena: seeds 8731-8740 = cycles 1-10 (R5)
E2_SEEDS = range(8742, 8750)     # hybrid.h6_labels.RESERVED_SEEDS, SB re-runs: episode n = 8741 + n (R5)
HOLD_WINDOW = 3                  # a promotion in the last 3 evaluated cycles shelves V9 (docs/h7-c1.md)
STALL_CYCLES = 2                 # E2: the last 2 cycles evaluated, KEEP, improvement lower bound <= 0
NO_CANDIDATE_LIMIT = 2           # R2
PREFLIGHT_WARN = 0.95            # R4: warning share of the cap
TEACHER_FACTOR = 4               # the c2 recipe (docs/h7c/c1/cost_review.json): S200 / tactical S400
SATURATION_FACTOR = 8            # the S400 150-pair stage
TOOLING = ('c3_retarget_recipe', 'e2_sb_champion', 'fallback_diagnose', 'retarget_cost_ledger')
ALLOWED_PATHS = ('docs/h7c/c{c}/*', 'docs/external/x2_c{c}.json')
ALLOWED_EPISODE_PATHS = ('docs/h7c/e2/e{n}/*',)


def selfplay_seed(cycle: int) -> int:
    return SELFPLAY_SEED_BASE + cycle


def e2_seed(episode: int) -> int | None:
    seed = E2_SEEDS.start - 1 + episode
    return seed if episode >= 1 and seed in E2_SEEDS else None


def _step(state: str, action: str | None = None, **why) -> dict:
    return {'state': state, 'action': action, 'why': why}


# --- one cycle --------------------------------------------------------------------------------------------------

def train_artifacts_ok(record: dict | None, best_sha256: str | None, cycle: int, parent_sha256: str) -> str | None:
    """A training record and its candidate: None when consistent, else the reason (BLOCKED_TRAIN). NO_CANDIDATE is a
    result, not a failure: then no candidate may be recorded."""
    if record is None:
        return 'no record'
    if record.get('cycle') != cycle:
        return f"record of cycle {record.get('cycle')}"
    if (record.get('inputs') or {}).get('parent_checkpoint_sha256') != parent_sha256:
        return 'training did not start from the parent'
    verdict = record.get('verdict')
    if verdict == NO_CANDIDATE:
        return None if best_sha256 is None else 'NO_CANDIDATE with a candidate file'
    if verdict != 'CANDIDATE':
        return f'verdict {verdict!r}'
    if best_sha256 is None:
        return 'CANDIDATE without its metadata file'
    if (record.get('candidate') or {}).get('metadata_sha256') != best_sha256:
        return 'candidate metadata SHA-256 differs from the record'
    return None


def resolve_candidate(policy_verdict: str, value_verdict: str, cycle: int) -> str:
    """h7c_arena: both candidates -> JOINT; policy NO_CANDIDATE + value candidate -> the value-only fallback
    (cycle >= 2); otherwise the no-candidate record."""
    if policy_verdict == 'CANDIDATE' and value_verdict == 'CANDIDATE':
        return JOINT
    if cycle >= 2 and policy_verdict == NO_CANDIDATE and value_verdict == 'CANDIDATE':
        return FALLBACK
    return NO_CANDIDATE


def cost_preflight(eligible_rows: int, tree_seconds: float, workers: int, cap_seconds: float) -> dict:
    """R4: rows x the ablation's seconds per tree move / workers against the cap. Advisory: the retarget's own gate
    (this session's rate) decides BLOCKED_COST_CAP."""
    projected = eligible_rows * tree_seconds / workers
    status = ('PASS' if projected <= PREFLIGHT_WARN * cap_seconds else 'WARN' if projected <= cap_seconds
              else 'MANUAL_COST_PREFLIGHT_REVIEW_REQUIRED')
    return {'eligible_rows': eligible_rows, 'projected_seconds': round(projected, 1), 'cap_seconds': cap_seconds,
            'share_of_cap': round(projected / cap_seconds, 4), 'status': status}


def cycle_stage(ev: dict) -> dict:
    """The first missing step of cycle ``ev['cycle']``. Keys (all optional but ``cycle``): ``ready`` (cycle 2: the
    G0 / G0b evidence and the c1 chain), ``tooling``, ``recipe``, ``selfplay``, ``preflight`` (cost_preflight's
    status), ``retarget``, ``retarget_partial``, ``labels``, ``train_error``, ``policy_verdict`` / ``value_verdict``,
    ``verdict``, ``x2`` (the X2 record exists; not needed for a no-candidate cycle), ``cycle_index``."""
    c, tooling = ev['cycle'], ev.get('tooling') or {}
    if c < FIRST_CYCLE:
        return _step('BLOCKED', None, reason=f'the automation starts at cycle {FIRST_CYCLE}')
    if c not in ARENA_CYCLES or not 8801 <= selfplay_seed(c) <= 8999:
        return _step('BLOCKED_SEED_RANGE', None, cycle=c)
    if c == FIRST_CYCLE and ev.get('ready') is not True:
        return _step('NOT_C2_READY', None, reason=ev.get('ready') or 'G0 / G0b evidence missing')
    if c > FIRST_CYCLE and not tooling.get('c3_retarget_recipe'):
        return _step('C3_RECIPE_TOOLING_REQUIRED', None, reason='h7c_retarget run takes no recipe carry-forward')
    if not ev.get('recipe'):
        return _step('RECIPE_REQUIRED', 'WRITE_RECIPE')
    if not ev.get('selfplay'):
        return _step('SELFPLAY_READY', 'RUN_SELFPLAY', seed=selfplay_seed(c))
    if not ev.get('retarget'):
        if ev.get('retarget_partial') and not tooling.get('retarget_cost_ledger'):
            return _step('MANUAL_COST_RESUME_REVIEW_REQUIRED', None,
                         reason='a partial retarget without its summary (running or stopped): the cost gate restarts '
                                'per session, so a resume is not automatic')
        if ev.get('preflight') == 'MANUAL_COST_PREFLIGHT_REVIEW_REQUIRED':
            return _step('MANUAL_COST_PREFLIGHT_REVIEW_REQUIRED', None)
        return _step('RETARGET_READY', 'RUN_RETARGET', preflight=ev.get('preflight'))
    if not ev.get('labels'):
        return _step('LABELS_READY', 'RUN_LABELS')
    if not ev.get('policy_verdict'):
        return _step('POLICY_TRAIN_READY', 'RUN_POLICY_TRAIN')
    if not ev.get('value_verdict'):
        return _step('VALUE_TRAIN_READY', 'RUN_VALUE_TRAIN')
    if ev.get('train_error'):
        return _step('BLOCKED_TRAIN', None, reason=ev['train_error'])
    mode = resolve_candidate(ev['policy_verdict'], ev['value_verdict'], c)
    if not ev.get('verdict'):
        if mode == NO_CANDIDATE:
            return _step('NO_CANDIDATE_VERDICT_READY', 'RUN_NO_CANDIDATE', candidate_mode=mode)
        return _step('ARENA_BUNDLE_READY', 'RUN_ARENA_BUNDLE', candidate_mode=mode,
                     note='every arm, the tactical record and the verdict on one HEAD: no commit in between')
    if mode != NO_CANDIDATE and not ev.get('x2'):
        return _step('X2_MONITOR_READY', 'RUN_X2')
    if not ev.get('cycle_index'):
        return _step('CYCLE_INDEX_READY', 'WRITE_CYCLE_INDEX', x2='NOT_APPLICABLE' if mode == NO_CANDIDATE else 'DONE')
    return _step('CYCLE_COMPLETE', 'NEXT_CYCLE_DECISION')


# --- across cycles ----------------------------------------------------------------------------------------------

def cycle_entry(verdict: dict) -> dict:
    """A verdict as the history needs it (``h7c_arena`` verdict or no-candidate record)."""
    p = verdict['promotion']
    return {'cycle': verdict['cycle'], 'decision': p['decision'], 'reason': p.get('reason'),
            'lower': p.get('improvement_lower'), 'candidate_mode': verdict.get('candidate_mode') or JOINT}


def evaluated(entry: dict) -> bool:
    """A cycle with a valid official arm (an improvement lower bound): not a no-candidate or invalid one."""
    return entry['lower'] is not None and entry['reason'] not in (NO_CANDIDATE, OFFICIAL_INVALID)


def history_signals(cycles: list[dict]) -> dict:
    ev = [e for e in cycles if evaluated(e)]
    window = ev[-HOLD_WINDOW:]
    last = cycles[-STALL_CYCLES:]
    stagnant = len(last) == STALL_CYCLES and all(evaluated(e) and e['decision'] == KEEP and e['lower'] <= 0
                                                 for e in last)
    streak = 0
    for e in reversed(cycles):
        if e['reason'] != NO_CANDIDATE:
            break
        streak += 1
    return {'evaluated_cycles': [e['cycle'] for e in ev], 'hold_window': [e['cycle'] for e in window],
            'hold': any(e['decision'] == PROMOTE for e in window), 'stagnant': stagnant,
            'stall_cycles': [e['cycle'] for e in last] if stagnant else [], 'no_candidate_streak': streak}


def next_cycle_decision(cycles: list[dict], tooling: dict | None = None) -> dict:
    """After the last cycle's verdict (``cycles``: cycle_entry of cycles 1..c in order), in this order: an invalid
    official arm (R6), two no-candidate cycles (R2), E2 (two stalled evaluated cycles and no promotion in the last
    three evaluated ones), the seed range (R5), missing tools, else the next cycle."""
    tooling = tooling or {}
    if not cycles:
        raise ValueError('no cycle verdicts')
    c = cycles[-1]['cycle']
    if [e['cycle'] for e in cycles] != list(range(1, c + 1)):
        raise ValueError('the verdicts are not cycles 1..c in order')
    sig = history_signals(cycles)
    if cycles[-1]['reason'] == OFFICIAL_INVALID:
        return _step('MANUAL_REVIEW_REQUIRED', None, rule='R6', reason='the official arm is invalid', **sig)
    if sig['no_candidate_streak'] >= NO_CANDIDATE_LIMIT:
        return _step('MANUAL_REVIEW_REQUIRED', None, rule='R2', reason='consecutive no-candidate cycles', **sig)
    if sig['stagnant'] and not sig['hold']:
        if cycles[-1]['candidate_mode'] == FALLBACK and not tooling.get('fallback_diagnose'):
            return _step('E2_FALLBACK_DIAG_TOOLING_REQUIRED', None, reason='h7c_diagnose needs the p_only arm', **sig)
        if not tooling.get('e2_sb_champion'):
            return _step('E2_SB_TOOLING_REQUIRED', None, reason='h7c_budget runs the frozen H3 / H6 v3 only', **sig)
        return _step('E2_REQUIRED', 'START_E2_EPISODE', **sig)
    nxt = c + 1
    if nxt not in ARENA_CYCLES:
        return _step('BLOCKED_SEED_RANGE', None, rule='R5', cycle=nxt, **sig)
    if nxt > FIRST_CYCLE and not tooling.get('c3_retarget_recipe'):
        return _step('C3_RECIPE_TOOLING_REQUIRED', None, cycle=nxt, **sig)
    return _step('V9_HOLD' if sig['hold'] else 'NEXT_CYCLE', 'NEXT_CYCLE', cycle=nxt, **sig)


# --- E2 ----------------------------------------------------------------------------------------------------------

def sb_branch(stronger: set, teacher_factor: int = TEACHER_FACTOR) -> dict:
    """R3 on the factors the ablation read STRONGER (50 pairs). Budget strength need not be monotone: STRONGER on
    both sides of the teacher but not at it is a review, never a new teacher."""
    t, s = teacher_factor, set(stronger)
    lower, higher = {f for f in s if f < t}, {f for f in s if f > t}
    if not s:
        return _step('E2_SATURATION_REQUIRED', 'RUN_SATURATION', factor=SATURATION_FACTOR)
    if t not in s and lower and higher:
        return _step('E2_BUDGET_NONMONOTONIC_REVIEW', None, stronger=sorted(s))
    if higher:
        return _step('NEW_TEACHER_REVIEW', None, factor=min(higher), stronger=sorted(s))
    if t in s:
        return _step('E2_CURRENT_TEACHER_STILL_STRONG_REVIEW', None, stronger=sorted(s))
    return _step('E2_BUDGET_NONMONOTONIC_REVIEW', None, stronger=sorted(s))


def exhausted(entry: dict | None) -> bool | None:
    """A B prescription's cycle: True when it was evaluated and stayed (KEEP, lower <= 0), False when it promoted
    (the episode ends), None when it proves nothing yet (no cycle, no candidate, invalid, or a positive bound)."""
    if entry is None or not evaluated(entry):
        return None
    if entry['decision'] == PROMOTE:
        return False
    return True if entry['lower'] <= 0 else None


def e2_state(stronger: set | None, saturation: str | None, triggers: dict | None, prescriptions: dict | None = None,
             teacher_factor: int = TEACHER_FACTOR) -> dict:
    """An E2 episode from its recorded readings: ``stronger`` (None until the ablation summary exists),
    ``saturation`` (None until the 150-pair stage), ``triggers`` ({'B_POLICY': bool, 'B_VALUE': bool}, fixed by the
    episode's diagnosis), ``prescriptions`` ({trigger: cycle_entry of the cycle whose recipe applied it})."""
    if triggers is None:
        return _step('E2_DIAGNOSIS_REQUIRED', 'RUN_E2_DIAGNOSIS')
    if stronger is None:
        return _step('E2_SB_REQUIRED', 'RUN_E2_SB')
    if stronger:
        return sb_branch(stronger, teacher_factor)
    if saturation is None:
        return sb_branch(set(), teacher_factor)
    if saturation == 'STRONGER':
        return _step('NEW_TEACHER_REVIEW', None, factor=SATURATION_FACTOR, saturation=saturation)
    if saturation == 'INCONCLUSIVE':
        return _step('REPLICATE_REQUIRED', 'NEXT_CYCLE', saturation=saturation, note='recipe unchanged')
    if saturation != 'SATURATED':
        raise ValueError(f'saturation reading {saturation!r}')
    prescriptions = prescriptions or {}
    for trigger in ('B_POLICY', 'B_VALUE'):
        if not triggers.get(trigger):
            continue
        done = exhausted(prescriptions.get(trigger))
        if done is False:
            return _step('E2_EPISODE_CLOSED', 'NEXT_CYCLE', trigger=trigger, reason='the prescription promoted')
        if done is None:
            return _step('MANUAL_RECIPE_CHANGE_REQUIRED', None, trigger=trigger,
                         pending=prescriptions.get(trigger) is not None)
    return _step('V9_RESEARCH_READY', 'WRITE_V9_ENTRY', then='MANUAL_APPROVAL_REQUIRED', saturation=saturation,
                 triggers=triggers)


# --- commits ------------------------------------------------------------------------------------------------------

def evidence_only(paths, cycle: int, episode: int | None = None) -> list[str]:
    """The paths changed since the cycle's ``code_base_commit`` that are not this cycle's evidence (or this E2
    episode's): any of them stops the automation (MANUAL_CODE_CHANGE_REVIEW_REQUIRED)."""
    allowed = [p.format(c=cycle) for p in ALLOWED_PATHS]
    if episode is not None:
        allowed += [p.format(n=episode) for p in ALLOWED_EPISODE_PATHS]
    return sorted(p for p in paths if not any(fnmatch.fnmatchcase(p, a) for a in allowed))


RECIPE_LINEAGE = ('cycle', 'parent_recipe_sha256', 'source_verdict_sha256', 'code_base_commit', 'source_e2_plan_sha256',
                  'intervention', 'trigger')


def recipe_fingerprint(recipe: dict) -> dict:
    """The recipe without its lineage: a REPLICATE cycle's must equal the previous one's."""
    return {k: v for k, v in recipe.items() if k not in RECIPE_LINEAGE and not k.startswith('source_')}
