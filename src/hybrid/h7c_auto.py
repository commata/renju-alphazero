"""H7-c automation, Auto-v1 (docs/h7c-auto.md): pure state transitions from cycle evidence, no file or git access.

``scripts/h7c_auto.py`` loads the evidence, checks its SHA-256 chain and passes normalized values here. Nothing here
recomputes a verdict: the Arena verdict (``promotion``), the ablation summary (``arms[*].stronger``,
``saturation.reading``) and the decision files are read as recorded. The only numbers are the operational rules
R1-R6 (docs/h7c-auto.md), each named below.

    cycle_stage            the first step of cycle c whose evidence is missing, and the action it needs
    next_cycle_decision    after a cycle's verdict: MANUAL_REVIEW / E2 / BLOCKED_SEED_RANGE / *_TOOLING / NEXT_CYCLE
    sb_branch, e2_state    the E2 episode (search-budget ablation with the champion, saturation, B prescriptions)
    g0_errors, *_errors    each stage's evidence against the previous one (the SHA-256 chain); [] when consistent
    commit_gate            before any step: uncommitted evidence, or any other change since the code base
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
    G0 / G0b contract and the c1 chain), ``tooling``, ``recipe``, ``champion_error`` (the parent files on this machine
    are not the previous verdict's ``next_champion``), ``selfplay``, ``preflight`` (cost_preflight's status),
    ``retarget``, ``retarget_partial``, ``labels``, ``policy_verdict`` / ``value_verdict``, ``verdict``, ``x2``,
    ``x2_config``, ``cycle_index``, and ``errors`` ({stage: [reasons]} for evidence that exists but fails its
    checks: a stage with errors blocks, it never counts as done)."""
    c, tooling, errors = ev['cycle'], ev.get('tooling') or {}, ev.get('errors') or {}

    def blocked(stage: str):
        return _step(f'BLOCKED_{stage.upper()}', None, errors=errors[stage]) if errors.get(stage) else None

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
    if b := blocked('recipe'):
        return b
    if ev.get('champion_error') and not ev.get('verdict'):
        return _step('BLOCKED_CHAMPION_FILES', None, reason=ev['champion_error'])
    if not ev.get('selfplay'):
        return _step('SELFPLAY_READY', 'RUN_SELFPLAY', seed=selfplay_seed(c))
    if b := blocked('selfplay'):
        return b
    if not ev.get('retarget'):
        if ev.get('retarget_partial') and not tooling.get('retarget_cost_ledger'):
            return _step('MANUAL_COST_RESUME_REVIEW_REQUIRED', None,
                         reason='a partial retarget without its summary (running or stopped): the cost gate restarts '
                                'per session, so a resume is not automatic')
        if ev.get('preflight') == 'MANUAL_COST_PREFLIGHT_REVIEW_REQUIRED':
            return _step('MANUAL_COST_PREFLIGHT_REVIEW_REQUIRED', None)
        return _step('RETARGET_READY', 'RUN_RETARGET', preflight=ev.get('preflight'))
    if b := blocked('retarget'):
        return b
    if not ev.get('labels'):
        return _step('LABELS_READY', 'RUN_LABELS')
    if b := blocked('labels'):
        return b
    if not ev.get('policy_verdict'):
        return _step('POLICY_TRAIN_READY', 'RUN_POLICY_TRAIN')
    if not ev.get('value_verdict'):
        return _step('VALUE_TRAIN_READY', 'RUN_VALUE_TRAIN')
    if b := blocked('train'):
        return b
    mode = resolve_candidate(ev['policy_verdict'], ev['value_verdict'], c)
    if not ev.get('verdict'):
        if mode == NO_CANDIDATE:
            return _step('NO_CANDIDATE_VERDICT_READY', 'RUN_NO_CANDIDATE', candidate_mode=mode)
        return _step('ARENA_BUNDLE_READY', 'RUN_ARENA_BUNDLE', candidate_mode=mode,
                     note='every arm, the tactical record and the verdict on one HEAD: no commit in between')
    if b := blocked('arena'):
        return b
    if mode != NO_CANDIDATE:
        if not ev.get('x2'):
            if not ev.get('x2_config'):
                return _step('X2_CONFIG_REQUIRED', None, reason='the Rapfi engine files of this machine '
                                                                '(runs/h7c/automation/x2_config.json)')
            return _step('X2_MONITOR_READY', 'RUN_X2')
        if b := blocked('x2'):
            return b
    if not ev.get('cycle_index'):
        return _step('CYCLE_INDEX_READY', 'WRITE_CYCLE_INDEX', x2='NOT_APPLICABLE' if mode == NO_CANDIDATE else 'DONE')
    if b := blocked('cycle_index'):
        return b
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


# --- evidence checks (one stage each; [] when consistent) ---------------------------------------------------------

MODEL_KEYS = ('policy_checkpoint_sha256', 'policy_metadata_sha256', 'value_checkpoint_sha256', 'value_metadata_sha256')
SELFPLAY_KEYS = ('cycle', 'run_seed', 'engine_hash', 'runtime_hash', 'value_device', 'models', 'model_hash',
                 'generator_git')   # hybrid.h7c_retarget.SELFPLAY_KEYS: what a retarget names of its self-play
SELFPLAY_GAMES = 1000


def _diff(errors: list, label: str, got, want) -> None:
    if got != want:
        errors.append(f'{label}: {got!r} != {want!r}' if len(repr(got)) + len(repr(want)) < 160 else f'{label} differs')


def g0_errors(g0: dict | None, g0b: dict | None, selfplay: dict, selfplay_jsonl_sha256: str) -> list[str]:
    """The approved G0 / G0b contract (docs/h7-c1.md), not just a PASS: G0 = factor 1 over all 1,000 c1 games with
    the v1 projection compared and equal; G0b = factor 4, both modes, the first 20 games, no difference. Both name
    the c1 self-play file and identity and a clean commit."""
    errors = []
    for name, rec, factor, games, modes in (('G0', g0, 1, SELFPLAY_GAMES, None),
                                            ('G0b', g0b, TEACHER_FACTOR, 20, ['restored', 'sequential'])):
        if rec is None:
            errors.append(f'{name} missing')
            continue
        ident = rec.get('identity') or {}
        _diff(errors, f'{name} format', rec.get('format'), 'h7c-retarget-v1-check')
        _diff(errors, f'{name} verdict', rec.get('verdict'), 'PASS')
        _diff(errors, f'{name} purpose', ident.get('purpose'), 'g0')
        _diff(errors, f'{name} factor', ident.get('factor'), factor)
        _diff(errors, f'{name} games', rec.get('games'), games)
        _diff(errors, f'{name} coverage games', (ident.get('coverage') or {}).get('games'), games)
        _diff(errors, f'{name} rows', rec.get('rows'), (ident.get('coverage') or {}).get('eligible'))
        _diff(errors, f'{name} self-play file', ident.get('selfplay_jsonl_sha256'), selfplay_jsonl_sha256)
        _diff(errors, f'{name} self-play identity', ident.get('selfplay'), {k: selfplay.get(k) for k in SELFPLAY_KEYS})
        _diff(errors, f'{name} clean commit', (ident.get('git') or {}).get('dirty'), False)
        if rec.get('failures'):
            errors.append(f'{name} failures {rec["failures"]}')
        if modes is None:
            proj = rec.get('projection')
            if not proj:
                errors.append('G0 has no v1 projection (--policy-rows)')
            elif proj.get('differ') or proj.get('missing') or not proj.get('compared'):
                errors.append(f'G0 projection {proj}')
            if any((rec.get('actor') or {}).values()) or rec.get('actor') is None:
                errors.append('G0 actor records not reproduced')
        else:
            _diff(errors, 'G0b modes', sorted(ident.get('modes') or []), modes)
            _diff(errors, 'G0b modes_differ', rec.get('modes_differ'), {})
    return errors


def selfplay_errors(summary: dict, cycle: int, parent: dict, recipe: dict) -> list[str]:
    errors, ident = [], summary.get('identity') or {}
    _diff(errors, 'format', summary.get('format'), 'h7c-selfplay-v1-summary')
    _diff(errors, 'cycle', ident.get('cycle'), cycle)
    _diff(errors, 'run seed', ident.get('run_seed'), selfplay_seed(cycle))
    _diff(errors, 'models', ident.get('models'), parent)
    for key in ('engine_hash', 'runtime_hash', 'value_device'):
        _diff(errors, key, ident.get(key), recipe.get(key))
    _diff(errors, 'games', (summary.get('start'), summary.get('games'), (summary.get('summary') or {}).get('games')),
          (0, SELFPLAY_GAMES, SELFPLAY_GAMES))
    dirty = ((summary.get('git') or {}).get('dirty'), (ident.get('generator_git') or {}).get('dirty'))
    _diff(errors, 'clean commit', dirty, (False, False))
    return errors


def retarget_errors(summary: dict, selfplay: dict, recipe: dict) -> list[str]:
    errors, ident = [], summary.get('identity') or {}
    _diff(errors, 'format', summary.get('format'), 'h7c-retarget-v1-summary')
    _diff(errors, 'purpose', ident.get('purpose'), 'retarget')
    for key, want in (('factor', recipe.get('teacher_factor')), ('budget', recipe.get('budget')),
                      ('mode', recipe.get('mode')), ('decision_sha256', recipe.get('source_decision_sha256')),
                      ('cost_review_sha256', recipe.get('source_cost_review_sha256'))):
        _diff(errors, key, ident.get(key), want)
    _diff(errors, 'cap', summary.get('cap_seconds'), recipe.get('hard_cap_seconds'))
    _diff(errors, 'self-play identity', ident.get('selfplay'),
          {k: (selfplay.get('identity') or {}).get(k) for k in SELFPLAY_KEYS})
    cov = summary.get('coverage') or {}
    eligible = (selfplay.get('summary') or {}).get('target_kinds', {}).get('visits')
    _diff(errors, 'coverage', (cov.get('eligible'), cov.get('completed'), cov.get('missing'), cov.get('duplicate'),
                               cov.get('extra')), (eligible, eligible, 0, 0, 0))
    _diff(errors, 'clean commit', (ident.get('git') or {}).get('dirty'), False)
    return errors


def labels_errors(report: dict, selfplay: dict, retarget: dict) -> list[str]:
    errors, summary = [], report.get('summary') or {}
    _diff(errors, 'format', (report.get('format'), report.get('policy_row_format')),
          ('h7c-labels-v1', 'h7c-policy-row-v2'))
    _diff(errors, 'clean commit', report.get('git_dirty'), False)
    ident, sp = summary.get('identity') or {}, selfplay.get('identity') or {}
    for key in ('cycle', 'engine_hash', 'runtime_hash', 'model_hash'):
        _diff(errors, key, ident.get(key), sp.get(key))
    _diff(errors, 'games in', summary.get('games_in'), SELFPLAY_GAMES)
    info = ((summary.get('policy') or {}).get('retarget') or {}).get('info') or {}
    _diff(errors, 'retarget file', info.get('sha256'), retarget.get('jsonl_sha256'))
    _diff(errors, 'retarget coverage', info.get('coverage_sha256'),
          ((retarget.get('identity') or {}).get('coverage') or {}).get('coverage_sha256'))
    return errors


def train_errors(name: str, record: dict | None, cycle: int, parent: dict, metadata_sha256: str | None,
                 checkpoint_sha256: str | None, check_checkpoint: bool, source: dict) -> list[str]:
    """One training record and its candidate. NO_CANDIDATE is a result (no candidate file may exist); a CANDIDATE
    needs its metadata (committed) and, before its Arena, its checkpoint on this machine, both as the record names
    them. ``source``: the fields the record's inputs must repeat (parent models, the retarget, the value labels)."""
    if record is None:
        return [f'{name}: no record']
    errors = []
    _diff(errors, f'{name} format', record.get('format'), f'h7c-{name}-train-v1')
    _diff(errors, f'{name} cycle', record.get('cycle'), cycle)
    _diff(errors, f'{name} clean commit', (record.get('git') or {}).get('dirty'), False)
    inputs = record.get('inputs') or {}
    for kind in ('checkpoint', 'metadata'):
        _diff(errors, f'{name} parent {kind}', inputs.get(f'parent_{kind}_sha256'), parent.get(f'{name}_{kind}_sha256'))
    for key, want in source.items():
        got = inputs
        for part in key.split('.'):
            got = (got or {}).get(part)
        _diff(errors, f'{name} {key}', got, want)
    verdict = record.get('verdict')
    if verdict == NO_CANDIDATE:
        if metadata_sha256 is not None:
            errors.append(f'{name}: NO_CANDIDATE with a candidate metadata file')
        return errors
    if verdict != 'CANDIDATE':
        return errors + [f'{name}: verdict {verdict!r}']
    cand = record.get('candidate') or {}
    if metadata_sha256 is None:
        errors.append(f'{name}: CANDIDATE without its metadata file')
    else:
        _diff(errors, f'{name} candidate metadata', metadata_sha256, cand.get('metadata_sha256'))
    if check_checkpoint:
        if checkpoint_sha256 is None:
            errors.append(f'{name}: the candidate checkpoint is not on this machine')
        else:
            _diff(errors, f'{name} candidate checkpoint', checkpoint_sha256, cand.get('checkpoint_sha256'))
    return errors


def expected_candidate(mode: str, policy_record: dict, value_record: dict, parent: dict) -> dict | None:
    """The four SHA-256 the Arena must name: JOINT both new; the fallback the parent policy and the new value."""
    if mode == NO_CANDIDATE:
        return None
    pol = ({'policy_checkpoint_sha256': parent['policy_checkpoint_sha256'],
            'policy_metadata_sha256': parent['policy_metadata_sha256']} if mode == FALLBACK else
           {'policy_checkpoint_sha256': policy_record['candidate']['checkpoint_sha256'],
            'policy_metadata_sha256': policy_record['candidate']['metadata_sha256']})
    return {**pol, 'value_checkpoint_sha256': value_record['candidate']['checkpoint_sha256'],
            'value_metadata_sha256': value_record['candidate']['metadata_sha256']}


def verdict_errors(verdict: dict, cycle: int, parent: dict, champion_sha256: str, mode: str, candidate: dict | None,
                   record_sha256: dict) -> list[str]:
    errors, promotion = [], verdict.get('promotion') or {}
    _diff(errors, 'format', verdict.get('format'), 'h7c-arena-v1-verdict')
    _diff(errors, 'cycle', verdict.get('cycle'), cycle)
    _diff(errors, 'seed', verdict.get('seed'), 8730 + cycle)
    _diff(errors, 'parent', verdict.get('parent'), parent)
    _diff(errors, 'champion record', verdict.get('champion_record_sha256'), champion_sha256)
    _diff(errors, 'candidate', verdict.get('candidate'), candidate)
    if mode == NO_CANDIDATE:
        _diff(errors, 'reason', promotion.get('reason'), NO_CANDIDATE)
        _diff(errors, 'training records', (verdict.get('training') or {}).get('record_sha256'), record_sha256)
    else:
        _diff(errors, 'candidate mode', verdict.get('candidate_mode') or JOINT, mode)
    if promotion.get('decision') not in (PROMOTE, KEEP):
        errors.append(f"decision {promotion.get('decision')!r}")
    want_next = candidate if promotion.get('decision') == PROMOTE else parent
    _diff(errors, 'next champion', verdict.get('next_champion'), want_next)
    return errors


def x2_errors(x2: dict, cycle: int, candidate: dict) -> list[str]:
    errors, h7c = [], (x2.get('ours_provenance') or {}).get('h7c') or {}
    _diff(errors, 'format', x2.get('format'), 'external-match-v1')
    _diff(errors, 'seed', x2.get('seed'), 8710 + cycle)
    _diff(errors, 'clean commit', x2.get('git_dirty'), False)
    _diff(errors, 'cycle', h7c.get('cycle'), cycle)
    _diff(errors, 'models', h7c.get('models'), candidate)
    return errors


def commit_gate(dirty: list[str], changed: list[str], cycle: int, extra: tuple = ()) -> dict | None:
    """Before any step: uncommitted or committed changes since the cycle's ``code_base_commit``. Only this cycle's
    evidence (or the previous cycle's, while its index waits for a commit; ``extra`` before the first recipe) may
    change; it must be committed before the next tool runs (they refuse a dirty tree)."""
    previous = cycle - 1 if cycle - 1 >= FIRST_CYCLE else None

    def outside(path: str, committed: bool) -> bool:
        if path in extra or not evidence_only([path], cycle):
            return False
        return committed or previous is None or bool(evidence_only([path], previous))

    bad = sorted({p for p in changed if outside(p, True)} | {p for p in dirty if outside(p, False)})
    if bad:
        return _step('MANUAL_CODE_CHANGE_REVIEW_REQUIRED', None, paths=bad)
    if dirty:
        return _step('EVIDENCE_COMMIT_REQUIRED', 'COMMIT_EVIDENCE', paths=sorted(dirty))
    return None


# --- commits ------------------------------------------------------------------------------------------------------

def evidence_only(paths, cycle: int, episode: int | None = None) -> list[str]:
    """The paths changed since the cycle's ``code_base_commit`` that are not this cycle's evidence (or this E2
    episode's): any of them stops the automation (MANUAL_CODE_CHANGE_REVIEW_REQUIRED)."""
    allowed = [p.format(c=cycle) for p in ALLOWED_PATHS]
    if episode is not None:
        allowed += [p.format(n=episode) for p in ALLOWED_EPISODE_PATHS]
    return sorted(p for p in paths if not any(fnmatch.fnmatchcase(p, a) for a in allowed))


RECIPE_SEMANTICS = ('recipe', 'teacher_factor', 'budget', 'mode', 'hard_cap_seconds', 'tree_seconds_mean', 'workers',
                    'actor', 'engine_hash', 'runtime_hash', 'value_device', 'policy_config_sha256',
                    'value_config_sha256', 'intervention')


def recipe_fingerprint(recipe: dict) -> dict:
    """What a recipe trains with (a whitelist: seeds, lineage and provenance fields are left out): a REPLICATE
    cycle's must equal the previous one's."""
    return {k: recipe.get(k) for k in RECIPE_SEMANTICS}
