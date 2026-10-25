"""H7-c automation, Auto-v1 (docs/h7c-auto.md): status, verify, next step and its reasons from the committed evidence.

It runs nothing: it reads ``docs/h7c/c<c>/`` (and the cycle's partial retarget file under ``runs/``), checks the
SHA-256 chain, and prints the one step the evidence allows next, with its command when the tool for it exists. The
transitions are ``hybrid.h7c_auto`` (pure); the verdicts are read as recorded, and ``verify`` recomputes only the c1
decision with ``h7c_decide.decide`` to compare it with the committed file. ``runs/h7c/automation/state.json`` is a
cache: ``status`` rewrites it from the evidence every time.

    python scripts/h7c_auto.py status | next | explain
    python scripts/h7c_auto.py verify [--skip-frozen]
    python scripts/h7c_auto.py recipe --cycle 2      # docs/h7c/c2/recipe.json, after the G0 / G0b evidence commit
    python scripts/h7c_auto.py manifest --cycle 2    # docs/h7c/c2/evidence_manifest.json: raw files under runs/
    python scripts/h7c_auto.py index --cycle 2       # docs/h7c/c2/cycle.json, from the checked evidence only
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid import h7c_auto as auto  # noqa: E402

FORMAT = 'h7c-automation-v1'
RECIPE_FORMAT = 'h7c-recipe-v1'
DOCS = ROOT / 'docs' / 'h7c'
RUNS = Path('runs/h7c')
FROZEN = {'policy': 'runs/h3_policy_64x4/best.pt', 'value': 'runs/h6_value_head_v3/best.pt'}
C1 = {'decision': DOCS / 'c1' / 'decision.json', 'cost_review': DOCS / 'c1' / 'cost_review.json',
      'g0': DOCS / 'c1' / 'retarget_g0.json', 'g0b': DOCS / 'c1' / 'retarget_g0b.json',
      'diagnosis': DOCS / 'c1' / 'diagnosis.json', 'sb': DOCS / 'sb_summary.json',
      'selfplay': DOCS / 'c1' / 'selfplay_c1_seed8801.json'}
CONFIGS = {'policy': ROOT / 'configs' / 'h7c' / 'c1_policy.json', 'value': ROOT / 'configs' / 'h7c' / 'c1_value.json'}
WORKERS = 6
TOOLING = {name: False for name in auto.TOOLING}   # Auto-v1: none of the extensions exists yet
INDEX_FORMAT = 'h7c-cycle-v1'
X2_CONFIG = ROOT / RUNS / 'automation' / 'x2_config.json'   # this machine's Rapfi files (never committed)


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_absolute() else str(path)


def _git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def cycle_files(c: int, recipe: dict | None = None) -> dict:
    d = DOCS / f'c{c}'
    factor = (recipe or {}).get('teacher_factor', auto.TEACHER_FACTOR)
    return {'recipe': d / 'recipe.json', 'selfplay': d / f'selfplay_c{c}_seed{auto.selfplay_seed(c)}.json',
            'retarget': d / f'retarget_s{50 * factor}.json',
            'retarget_partial': ROOT / RUNS / f'c{c}' / f'retarget_s{50 * factor}.jsonl',
            'labels': d / f'labels_c{c}_report.json', 'policy_record': d / 'policy_record.json',
            'policy_best': d / 'policy_best.json', 'value_record': d / 'value_record.json',
            'value_best': d / 'value_best.json', 'value_cache_manifest': d / 'value_cache_manifest.json',
            'manifest': d / 'evidence_manifest.json',
            'verdict': d / 'arena_verdict.json', 'x2': ROOT / 'docs' / 'external' / f'x2_c{c}.json',
            'cycle_index': d / 'cycle.json'}


def candidate_checkpoint(c: int, name: str) -> Path:
    return ROOT / RUNS / f'c{c}' / name / 'best.pt'


# --- readiness and the chain ---------------------------------------------------------------------------------------

def c1_selfplay() -> tuple[dict, str | None]:
    """The c1 self-play identity and the SHA-256 of its JSONL file (docs/h7c/c1/evidence_manifest.json)."""
    manifest = _json(DOCS / 'c1' / 'evidence_manifest.json') or {}
    return _json(C1['selfplay'])['identity'], (manifest.get('files_sha256') or {}).get('selfplay_8801.jsonl')


def c2_ready() -> str | bool:
    """Cycle 2's starting point: the c1 decision and its cost review, the full G0 / G0b contract, the c1 verdict.
    True, or what is wrong."""
    missing = [k for k in ('decision', 'cost_review') if not C1[k].exists()]
    if not (DOCS / 'c1' / 'arena_verdict.json').exists():
        missing.append('c1 verdict')
    if missing:
        return f'missing: {", ".join(missing)}'
    review = _json(C1['cost_review'])
    if review.get('decision_sha256') != _sha(C1['decision']) or review.get('applies_to_cycle') != 2:
        return 'the cost review does not approve this decision for cycle 2'
    identity, jsonl_sha = c1_selfplay()
    g0, g0b = _json(C1['g0']), _json(C1['g0b'])
    errors = auto.g0_errors(g0, g0b, identity, jsonl_sha)
    for name, rec in (('G0', g0), ('G0b', g0b)):   # measured on code this checkout still contains
        commit = (((rec or {}).get('identity') or {}).get('git') or {}).get('commit')
        if rec is not None and (not commit or subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'],
                                                             cwd=ROOT, capture_output=True).returncode != 0):
            errors.append(f'{name} commit {str(commit)[:12]} is not an ancestor of HEAD')
    return True if not errors else '; '.join(errors)


def verdicts() -> list[dict]:
    """The verdicts of cycles 1..c in order (stops at the first missing one)."""
    out, c = [], 1
    while (path := DOCS / f'c{c}' / 'arena_verdict.json').exists():
        out.append({'path': path, 'record': _json(path)})
        c += 1
    return out


def chain_errors(found: list[dict]) -> list[str]:
    """Each verdict names its cycle, and cycle c's parent is cycle c - 1's ``next_champion`` with that verdict's
    SHA-256 as its champion record (h7c_arena.check_parent)."""
    errors = []
    for i, v in enumerate(found):
        rec = v['record']
        if rec.get('cycle') != i + 1:
            errors.append(f"{_rel(v['path'])}: cycle {rec.get('cycle')}")
        if i and (rec.get('parent') != found[i - 1]['record']['next_champion'] or
                  rec.get('champion_record_sha256') != _sha(found[i - 1]['path'])):
            errors.append(f"{_rel(v['path'])}: the parent is not cycle {i}'s next_champion")
    return errors


def champion_paths(history: list[dict], c: int) -> dict:
    """The parent models of cycle c: the frozen ones until a cycle promoted, then that cycle's candidate files (the
    fallback keeps the policy it was given)."""
    paths = dict(FROZEN)
    for e in history[:c - 1]:
        if e['decision'] == auto.PROMOTE:
            paths = {'policy': paths['policy'] if e['candidate_mode'] == auto.FALLBACK else
                     f'{RUNS}/c{e["cycle"]}/policy/best.pt', 'value': f'{RUNS}/c{e["cycle"]}/value/best.pt'}
    return paths


def champion_error(history: list[dict], c: int, parent: dict) -> str | None:
    """The files the commands pass as the parent are the previous verdict's ``next_champion`` (four SHA-256)."""
    from analysis.tb_v3_net import model_identity

    paths = champion_paths(history, c)
    try:
        found = model_identity(ROOT / paths['policy'], ROOT / paths['value'])
    except FileNotFoundError as exc:
        return f'not on this machine: {exc.filename}'
    bad = [k for k in auto.MODEL_KEYS if found[k] != parent.get(k)]
    return f"{paths} differ from cycle {c - 1}'s next_champion in {bad}" if bad else None


def commit_errors(label: str, commit: str | None, base: str) -> list[str]:
    """An evidence file's code commit lies between the cycle's ``code_base_commit`` and HEAD (evidence-only commits
    in between, checked by the commit gate)."""
    if not commit:
        return [f'{label}: no commit recorded']
    ok = all(subprocess.run(['git', 'merge-base', '--is-ancestor', a, b], cwd=ROOT, capture_output=True).returncode == 0
             for a, b in ((base, commit), (commit, 'HEAD')))
    return [] if ok else [f'{label}: commit {commit[:12]} is not between the code base {base[:12]} and HEAD']


def evidence(c: int, history: list[dict], found: list[dict], local: bool = True) -> dict:
    """What exists of cycle c and whether each piece checks against the previous one, for ``auto.cycle_stage``.
    ``local``: also the files only this machine has (the parent and the candidate checkpoints)."""
    recipe = _json(cycle_files(c)['recipe'])
    f = cycle_files(c, recipe)
    prev = found[c - 2] if len(found) >= c - 1 else None
    parent = (prev or {}).get('record', {}).get('next_champion') or {}
    errors = {}
    ev = {'cycle': c, 'tooling': TOOLING if local else {k: True for k in TOOLING},
          'ready': c2_ready() if c == auto.FIRST_CYCLE else True, 'recipe': recipe is not None, 'errors': errors,
          'x2_config': X2_CONFIG.exists()}
    if ev['x2_config']:
        try:
            errors['x2_config'] = auto.x2_config_errors(_json(X2_CONFIG))
        except ValueError as exc:
            errors['x2_config'] = [f'not JSON: {exc}']
    if recipe is None:
        return ev
    base = recipe.get('code_base_commit')
    errors['recipe'] = recipe_errors(recipe) if c == auto.FIRST_CYCLE else ['only the cycle-2 recipe is checked']
    ev['verdict'], ev['x2'] = f['verdict'].exists(), f['x2'].exists()
    verdicts_of = {n: (_json(f[f'{n}_record']) or {}).get('verdict') for n in ('policy', 'value')}
    mode = auto.resolve_candidate(verdicts_of['policy'], verdicts_of['value'], c) if all(verdicts_of.values()) else None
    need = local and auto.checkpoints_needed(ev['verdict'], ev['x2'], mode)   # the parent and candidate files
    if need:
        ev['champion_error'] = champion_error(history, c, parent)
    selfplay = _json(f['selfplay'])
    ev['selfplay'] = selfplay is not None
    if selfplay is None:
        return ev
    errors['selfplay'] = auto.selfplay_errors(selfplay, c, parent, recipe) + \
        commit_errors('self-play', (selfplay.get('git') or {}).get('commit'), base)
    rows = selfplay['summary']['target_kinds'].get('visits', 0)
    ev['preflight_detail'] = auto.cost_preflight(rows, recipe['tree_seconds_mean'], recipe['workers'],
                                                 recipe['hard_cap_seconds'])
    ev['preflight'] = ev['preflight_detail']['status']
    retarget = _json(f['retarget'])
    ev['retarget'], ev['retarget_partial'] = retarget is not None, f['retarget_partial'].exists()
    if retarget is None:
        return ev
    errors['retarget'] = auto.retarget_errors(retarget, selfplay, recipe) + \
        commit_errors('retarget', ((retarget.get('identity') or {}).get('git') or {}).get('commit'), base)
    labels = _json(f['labels'])
    ev['labels'] = labels is not None
    if labels is None:
        return ev
    errors['labels'] = auto.labels_errors(labels, selfplay, retarget) + \
        commit_errors('labels', labels.get('git_commit'), base)
    raw = _json(f['manifest'])
    ev['manifest'] = raw is not None
    if raw is None:
        return ev
    errors['manifest'] = auto.manifest_errors(raw, c, retarget, _sha(f['labels']))
    records = {name: _json(f[f'{name}_record']) for name in ('policy', 'value')}
    ev['policy_verdict'] = (records['policy'] or {}).get('verdict')
    ev['value_verdict'] = (records['value'] or {}).get('verdict')
    if not (ev['policy_verdict'] and ev['value_verdict']):
        return ev
    retarget_info = ((labels.get('summary') or {}).get('policy') or {}).get('retarget', {}).get('info')
    cache = _json(f['value_cache_manifest']) or {}
    sources = {'policy': {'source.models': parent, 'source.retarget': retarget_info,
                          'policy_rows_sha256': raw.get('policy_rows_sha256')},
               'value': {'source.models': parent, 'cycle_labels_sha256': cache.get('labels_sha256') or 'missing'}}
    errors['train'] = []
    for name, record in records.items():
        errors['train'] += auto.train_errors(name, record, c, parent, _sha(f[f'{name}_best']),
                                             _sha(candidate_checkpoint(c, name)), need,
                                             sources[name])
        errors['train'] += commit_errors(f'{name} training', (record.get('git') or {}).get('commit'), base)
    if errors['train']:
        return ev
    candidate = auto.expected_candidate(mode, records['policy'], records['value'], parent)
    ev['candidate'] = candidate
    verdict = _json(f['verdict'])
    if verdict is None:
        return ev
    errors['arena'] = auto.verdict_errors(verdict, c, parent, _sha(prev['path']), mode, candidate,
                                          {'policy': _sha(f['policy_record']), 'value': _sha(f['value_record'])}) + \
        commit_errors('verdict', verdict.get('git_commit'), base)
    x2 = _json(f['x2'])
    if x2 is not None and mode != auto.NO_CANDIDATE:
        errors['x2'] = auto.x2_errors(x2, c, candidate) + commit_errors('X2', x2.get('git_commit'), base)
    index = _json(f['cycle_index'])
    ev['cycle_index'] = index is not None
    if index is not None:
        errors['cycle_index'] = [] if index == build_index(c, f, verdict, mode) else \
            ['cycle.json is not the index of the checked evidence (h7c_auto.py index)']
    return ev


def raw_manifest(c: int) -> dict:
    """The SHA-256 of the cycle's raw files under runs/ (they are never committed; docs/h7c/c1/evidence_manifest.json
    did the same for c1)."""
    C, seed = ROOT / RUNS / f'c{c}', auto.selfplay_seed(c)
    files = {'selfplay_jsonl_sha256': C / f'selfplay_{seed}.jsonl', 'retarget_jsonl_sha256': C / 'retarget_s200.jsonl',
             'labels_report_sha256': C / 'labels_report.json', 'policy_rows_sha256': C / 'policy_rows.jsonl.gz',
             'value_rows_sha256': C / 'value_rows.jsonl.gz'}
    out = {'format': auto.MANIFEST_FORMAT, 'cycle': c}
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(2, 'missing', str(path))
        out[key] = _sha(path)
    out['files'] = {k: _rel(v) for k, v in files.items()}
    return out


def build_index(c: int, f: dict, verdict: dict, mode: str) -> dict:
    names = ['recipe', 'selfplay', 'retarget', 'labels', 'manifest', 'policy_record', 'policy_best', 'value_record',
             'value_best',
             'value_cache_manifest', 'verdict'] + ([] if mode == auto.NO_CANDIDATE else ['x2'])
    p = verdict['promotion']
    return {'format': INDEX_FORMAT, 'cycle': c, 'candidate_mode': mode,
            'files_sha256': {_rel(f[n]): _sha(f[n]) for n in names if f[n].exists()},
            'promotion': {'decision': p['decision'], 'reason': p.get('reason'),
                          'improvement_lower': p.get('improvement_lower')},
            'x2': 'NOT_APPLICABLE' if mode == auto.NO_CANDIDATE else 'RECORDED',
            'next_champion': verdict['next_champion']}


def dirty_paths() -> list[str]:
    out = []
    for line in _git('status', '--porcelain', '--untracked-files=all').splitlines():
        path = line[3:]
        out.append(path.split(' -> ')[-1].strip('"'))
    return out


def current() -> dict:
    """The cycle the evidence is at and its one allowed step. In order: the verdict chain, every indexed cycle's
    evidence (an index counts only when it is the index of checked evidence), the commit gate, the previous cycle's
    decision, then this cycle's first missing or failing step."""
    found = verdicts()
    history = [auto.cycle_entry(v['record']) for v in found]
    out = {'history': history, 'chain_errors': chain_errors(found)}
    if out['chain_errors']:
        return {**out, 'cycle': len(found), 'step': auto._step('BLOCKED_EVIDENCE', None, errors=out['chain_errors'])}
    c = auto.FIRST_CYCLE
    while cycle_files(c)['cycle_index'].exists():
        step = auto.cycle_stage(evidence(c, history, found, local=False))
        if step['state'] != 'CYCLE_COMPLETE':
            return {**out, 'cycle': c, 'step': step}
        c += 1
    recipe = _json(cycle_files(c)['recipe'])
    changed = [p for p in _git('diff', '--name-only', recipe['code_base_commit'], 'HEAD').split() if p] \
        if recipe and recipe.get('code_base_commit') else []
    extra = tuple(_rel(C1[k]) for k in ('g0', 'g0b')) if c == auto.FIRST_CYCLE and recipe is None else ()
    gate = auto.commit_gate(dirty_paths(), changed, c, extra)
    if gate is not None:
        return {**out, 'cycle': c, 'step': gate}
    decision = auto.next_cycle_decision(history[:c - 1], TOOLING) if len(history) >= c - 1 else None
    if decision is not None and decision['state'] not in ('NEXT_CYCLE', 'V9_HOLD'):
        return {**out, 'cycle': c - 1, 'step': decision}
    ev = evidence(c, history, found)
    step = auto.cycle_stage(ev)
    if decision is not None:
        step['why']['previous_cycle'] = {k: decision['why'][k] for k in ('hold', 'stagnant', 'hold_window',
                                                                          'no_candidate_streak')}
    if ev.get('preflight_detail'):
        step['why']['preflight'] = ev['preflight_detail']
    return {**out, 'cycle': c, 'step': step, 'evidence': ev}


# --- commands (only for tools that exist) --------------------------------------------------------------------------

def command(state: dict) -> list[str]:
    c, step, history = state['cycle'], state['step'], state['history']
    action, C = step['action'], f'{RUNS}/c{c}'
    par = champion_paths(history, c)
    pol, val, champ = par['policy'], par['value'], f'docs/h7c/c{c - 1}/arena_verdict.json'
    hist = ' '.join(f'docs/h7c/c{k}/arena_verdict.json' for k in range(1, c))
    if action == 'COMMIT_EVIDENCE':
        paths = ' '.join(step['why']['paths'])
        return [f'git add {paths} && git commit -m "H7-c c{c}: evidence"']
    if action == 'WRITE_RECIPE':
        return [f'python scripts/h7c_auto.py recipe --cycle {c}', f'git add docs/h7c/c{c}/recipe.json && git commit']
    if action == 'RUN_SELFPLAY':
        seed = auto.selfplay_seed(c)
        return [f'python -u scripts/h7c_selfplay.py --cycle {c} --run-seed {seed} --games 1000 --workers {WORKERS} '
                f'--policy-checkpoint {pol} --value-checkpoint {val} --device cuda --jsonl {C}/selfplay_{seed}.jsonl '
                f'--output {C}/selfplay_{seed}.json 2>&1 | tee {C}/selfplay.log',
                f'cp {C}/selfplay_{seed}.json docs/h7c/c{c}/selfplay_c{c}_seed{seed}.json  # then commit']
    if action == 'RUN_RETARGET':
        seed = auto.selfplay_seed(c)
        return [f'python -u scripts/h7c_retarget.py run --games {C}/selfplay_{seed}.jsonl '
                f'--decision {_rel(C1["decision"])} --cost-review {_rel(C1["cost_review"])} '
                f'--selfplay-summary docs/h7c/c{c}/selfplay_c{c}_seed{seed}.json --policy-checkpoint {pol} '
                f'--value-checkpoint {val} --device cuda --workers {WORKERS} --jsonl {C}/retarget_s200.jsonl '
                f'--output docs/h7c/c{c}/retarget_s200.json 2>&1 | tee {C}/retarget.log']
    if action == 'RUN_LABELS':
        seed = auto.selfplay_seed(c)
        return [f'python -u scripts/h7c_labels.py --games {C}/selfplay_{seed}.jsonl --retarget {C}/retarget_s200.jsonl '
                f'--workers 12 --fast fast4 --value-output {C}/value_rows.jsonl.gz --policy-output '
                f'{C}/policy_rows.jsonl.gz --report {C}/labels_report.json',
                f'cp {C}/labels_report.json docs/h7c/c{c}/labels_c{c}_report.json  # then commit']
    if action == 'RUN_POLICY_TRAIN':
        return [f'python -u scripts/h7c_train_policy.py --cycle {c} --config configs/h7c/c1_policy.json --policy-rows '
                f'{C}/policy_rows.jsonl.gz --parent {pol} --h3-cache data/external/renjunet/h3_cache --device cuda '
                f'--out-dir {C}/policy 2>&1 | tee {C}/policy_train.log  # exit 1 with a record = NO_CANDIDATE',
                f'cp {C}/policy/record.json docs/h7c/c{c}/policy_record.json; '
                f'[ -f {C}/policy/best.json ] && cp {C}/policy/best.json docs/h7c/c{c}/policy_best.json']
    if action == 'RUN_VALUE_TRAIN':
        return [f'python scripts/build_h6_value_cache.py --labels {C}/value_rows.jsonl.gz --out-dir {C}/value_cache '
                f'--workers 12',
                f'python -u scripts/h7c_train_value.py --cycle {c} --config configs/h7c/c1_value.json --cycle-cache '
                f'{C}/value_cache --cycle-rows {C}/value_rows.jsonl.gz --replay-cache data/tactical/h6/value_cache_v3 '
                f'--parent {val} --device cuda --out-dir {C}/value 2>&1 | tee {C}/value_train.log',
                f'cp {C}/value_cache/manifest.json docs/h7c/c{c}/value_cache_manifest.json; '
                f'cp {C}/value/record.json docs/h7c/c{c}/value_record.json; '
                f'[ -f {C}/value/best.json ] && cp {C}/value/best.json docs/h7c/c{c}/value_best.json']
    if action == 'RUN_NO_CANDIDATE':
        return [f'python scripts/h7c_arena.py no-candidate --cycle {c} --policy-record {C}/policy/record.json '
                f'--value-record {C}/value/record.json --parent-policy {pol} --parent-value {val} --champion {champ} '
                f'--history {hist} --output docs/h7c/c{c}/arena_verdict.json']
    if action == 'RUN_ARENA_BUNDLE':
        fallback = step['why']['candidate_mode'] == auto.FALLBACK
        cand_pol = pol if fallback else f'{C}/policy/best.pt'
        extra = f' --candidate-mode {auto.FALLBACK} --policy-record {C}/policy/record.json' if fallback else ''
        arms = ('official', 'mon_v8_full', 'mon_puct_policy', 'mon_e2') if fallback else \
            ('official', 'p_only', 'v_only', 'mon_v8_full', 'mon_puct_policy', 'mon_e2')
        a = (f'--cycle {c} --candidate-policy {cand_pol} --candidate-value {C}/value/best.pt --parent-policy {pol} '
             f'--parent-value {val} --frozen-policy {FROZEN["policy"]} --champion {champ} --device cuda '
             f'--workers {WORKERS}{extra}')
        return ['# one HEAD for the whole bundle: no commit until the verdict exists'] + \
            [f'python -u scripts/h7c_arena.py play {a} --arm {arm} --jsonl {C}/arena/{arm}.jsonl '
             f'--output {C}/arena/{arm}.json' for arm in arms] + \
            [f'python -u scripts/h7c_arena.py tactical {a} --output {C}/arena/tactical.json',
             f'python scripts/h7c_arena.py verdict --arms {" ".join(f"{C}/arena/{arm}.json" for arm in arms)} '
             f'--tactical {C}/arena/tactical.json --history {hist} --output docs/h7c/c{c}/arena_verdict.json']
    if action == 'RUN_X2':
        cfg = _json(X2_CONFIG)
        fallback = step['why'].get('candidate_mode') == auto.FALLBACK
        cand_pol = pol if fallback else f'{C}/policy/best.pt'
        engine = ' '.join([f"--engine {cfg['engine']} --engine-name {cfg.get('engine_name', 'rapfi')}"] +
                          [f'--engine-info {i}' for i in cfg.get('engine_info', [])] +
                          [f'--engine-file {f}' for f in cfg.get('engine_files', [])])
        extra = f' --candidate-mode {auto.FALLBACK} --policy-record {C}/policy/record.json' if fallback else ''
        return [f'python -u scripts/h7c_external.py --cycle {c} --policy-checkpoint {cand_pol} --value-checkpoint '
                f'{C}/value/best.pt --value-device cuda {engine} --workers {cfg["workers"]} --forbid-check '
                f'--games-jsonl runs/external/x2/c{c}.jsonl --output docs/external/x2_c{c}.json{extra}']
    if action == 'WRITE_EVIDENCE_MANIFEST':
        return [f'python scripts/h7c_auto.py manifest --cycle {c}', f'git add docs/h7c/c{c}/evidence_manifest.json '
                f'&& git commit -m "H7-c c{c}: raw-file manifest"']
    if action == 'WRITE_CYCLE_INDEX':
        return [f'python scripts/h7c_auto.py index --cycle {c}']
    return []


# --- verify --------------------------------------------------------------------------------------------------------

def verify(skip_frozen: bool) -> dict:
    checks = {}
    if not skip_frozen:
        rc = subprocess.run([sys.executable, 'scripts/check_track_b_frozen.py'], cwd=ROOT).returncode
        checks['frozen'] = 'PASS' if rc == 0 else f'FAIL (rc {rc})'
    import h7c_decide
    stored = _json(C1['decision'])
    inputs = {k: _json(C1[k]) for k in ('diagnosis', 'sb', 'selfplay')}
    again = h7c_decide.decide(inputs['diagnosis'], inputs['sb'], inputs['selfplay'])
    same = all(again[k] == stored[k] for k in ('case', 'budget', 'policy_trigger', 'value_trigger'))
    checks['c1_decision_recomputed'] = 'PASS' if same else 'FAIL'
    errors = chain_errors(verdicts())
    checks['verdict_chain'] = 'PASS' if not errors else 'FAIL: ' + '; '.join(errors)
    ready = c2_ready()
    checks['c2_ready'] = 'PASS' if ready is True else f'NOT_C2_READY ({ready})'
    state = current()
    recipe = _json(cycle_files(state['cycle'])['recipe'])
    if recipe is not None:
        bad = recipe_errors(recipe)
        checks['recipe_sources'] = 'PASS' if not bad else 'FAIL: ' + '; '.join(bad)
    checks['state'] = f"c{state['cycle']} {state['step']['state']}"
    return checks


def recipe_errors(recipe: dict) -> list[str]:
    errors = []
    for key, path in (('source_decision_sha256', C1['decision']), ('source_cost_review_sha256', C1['cost_review']),
                      ('g0_sha256', C1['g0']), ('g0b_sha256', C1['g0b']),
                      ('policy_config_sha256', CONFIGS['policy']), ('value_config_sha256', CONFIGS['value'])):
        if recipe.get(key) != _sha(path):
            errors.append(key)
    identity, _ = c1_selfplay()
    errors += [k for k in ('engine_hash', 'runtime_hash', 'value_device') if recipe.get(k) != identity[k]]
    if recipe.get('format') != RECIPE_FORMAT or recipe.get('cycle') != auto.FIRST_CYCLE or \
            recipe.get('workers') != WORKERS or not recipe.get('code_base_commit'):
        errors.append('format / cycle / workers / code_base_commit')
    if recipe.get('parent_verdict_sha256') != _sha(DOCS / f'c{recipe["cycle"] - 1}' / 'arena_verdict.json'):
        errors.append('parent_verdict_sha256')
    return errors


def make_recipe(cycle: int) -> dict:
    if cycle != auto.FIRST_CYCLE:
        raise ValueError('Auto-v1 writes the cycle-2 recipe only (C3_RECIPE_TOOLING_REQUIRED)')
    ready = c2_ready()
    if ready is not True:
        raise ValueError(f'NOT_C2_READY: {ready}')
    if _git('status', '--porcelain', '--untracked-files=all').strip():
        raise ValueError('the working tree is not clean')
    review = _json(C1['cost_review'])
    decision = _json(C1['decision'])
    identity, _ = c1_selfplay()
    return {'format': RECIPE_FORMAT, 'cycle': cycle, 'recipe': 'S200_RETARGET',
            'teacher_factor': review['approved']['factor'], 'budget': review['approved']['budget'],
            'mode': review['approved']['mode'], 'hard_cap_seconds': review['hard_cap_seconds'],
            'tree_seconds_mean': decision['budget']['tree_seconds_mean'], 'workers': WORKERS,
            'actor': {'engine': 'TB-V3', 'runtime': 'fast5', 'simulations': 50, 'tactical_simulations': 100},
            'arena_seed': 8730 + cycle, 'selfplay_seed': auto.selfplay_seed(cycle),
            'source_decision_sha256': _sha(C1['decision']), 'source_cost_review_sha256': _sha(C1['cost_review']),
            'g0_sha256': _sha(C1['g0']), 'g0b_sha256': _sha(C1['g0b']),
            'parent_verdict_sha256': _sha(DOCS / f'c{cycle - 1}' / 'arena_verdict.json'),
            'policy_config_sha256': _sha(CONFIGS['policy']), 'value_config_sha256': _sha(CONFIGS['value']),
            **{k: identity[k] for k in ('engine_hash', 'runtime_hash', 'value_device')},
            'code_base_commit': _git('rev-parse', 'HEAD').strip()}


# --- CLI -----------------------------------------------------------------------------------------------------------

def _cache(state: dict) -> None:
    path = ROOT / RUNS / 'automation' / 'state.json'
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'format': FORMAT, 'cycle': state['cycle'], 'state': state['step']['state'],
                                    'next_action': state['step']['action']}, indent=1) + '\n', encoding='utf-8')
    except OSError:
        pass  # a cache only


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('status', 'next', 'explain'):
        sub.add_parser(mode)
    sub.add_parser('verify').add_argument('--skip-frozen', action='store_true',
                                          help='skip scripts/check_track_b_frozen.py (no models on this machine)')
    sub.add_parser('recipe').add_argument('--cycle', type=int, required=True)
    sub.add_parser('index').add_argument('--cycle', type=int, required=True)
    sub.add_parser('manifest').add_argument('--cycle', type=int, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'verify':
        checks = verify(args.skip_frozen)
        print(json.dumps(checks, indent=1), flush=True)
        return 0 if all(v == 'PASS' or k in ('c2_ready', 'state') for k, v in checks.items()) else 1
    if args.mode == 'recipe':
        path = cycle_files(args.cycle)['recipe']
        if path.exists():
            parser.error(f'{_rel(path)} exists')
        try:
            recipe = make_recipe(args.cycle)
        except ValueError as exc:
            parser.error(str(exc))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(recipe, indent=1) + '\n', encoding='utf-8')
        print(f'wrote {_rel(path)} (code_base_commit {recipe["code_base_commit"][:12]})', flush=True)
        return 0
    if args.mode == 'manifest':
        state = current()
        if state['cycle'] != args.cycle or state['step']['state'] != 'EVIDENCE_MANIFEST_READY':
            parser.error(f"cycle {state['cycle']} is at {state['step']['state']}, not EVIDENCE_MANIFEST_READY")
        try:
            out = raw_manifest(args.cycle)
        except FileNotFoundError as exc:
            parser.error(f'missing raw file {exc.filename}')
        path = cycle_files(args.cycle)['manifest']
        path.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
        print(f'wrote {_rel(path)}', flush=True)
        return 0
    if args.mode == 'index':
        state = current()
        if state['cycle'] != args.cycle or state['step']['state'] != 'CYCLE_INDEX_READY':
            parser.error(f"cycle {state['cycle']} is at {state['step']['state']}, not CYCLE_INDEX_READY")
        found = verdicts()
        f = cycle_files(args.cycle, _json(cycle_files(args.cycle)['recipe']))
        mode = auto.resolve_candidate(state['evidence']['policy_verdict'], state['evidence']['value_verdict'],
                                      args.cycle)
        index = build_index(args.cycle, f, found[args.cycle - 1]['record'], mode)
        f['cycle_index'].write_text(json.dumps(index, indent=1) + '\n', encoding='utf-8')
        print(f"wrote {_rel(f['cycle_index'])}: {index['promotion']['decision']}", flush=True)
        return 0
    state = current()
    _cache(state)
    step = state['step']
    if args.mode == 'next':
        print(f"NEXT_ACTION = {step['action'] or step['state']}", flush=True)
        for line in command(state):
            print(line, flush=True)
    elif args.mode == 'status':
        h = ', '.join('c{cycle} {decision} (lower {lower})'.format(**e) for e in state['history']) or '-'
        print(f"Cycle: {state['cycle']}\nState: {step['state']}\nNext: {step['action'] or '-'}\nVerdicts: {h}\n"
              f"Chain: {'PASS' if not state['chain_errors'] else state['chain_errors']}", flush=True)
    else:
        print(json.dumps({'cycle': state['cycle'], 'state': step['state'], 'action': step['action'],
                          'why': step['why'], 'history': state['history']}, indent=1, default=str), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
