"""H7-c automation, Auto-v1 (docs/h7c-auto.md): status, verify, next step and its reasons from the committed evidence.

It runs nothing: it reads ``docs/h7c/c<c>/`` (and the cycle's partial retarget file under ``runs/``), checks the
SHA-256 chain, and prints the one step the evidence allows next, with its command when the tool for it exists. The
transitions are ``hybrid.h7c_auto`` (pure); the verdicts are read as recorded, and ``verify`` recomputes only the c1
decision with ``h7c_decide.decide`` to compare it with the committed file. ``runs/h7c/automation/state.json`` is a
cache: ``status`` rewrites it from the evidence every time.

    python scripts/h7c_auto.py status | next | explain
    python scripts/h7c_auto.py verify [--skip-frozen]
    python scripts/h7c_auto.py recipe --cycle 2      # docs/h7c/c2/recipe.json, after the G0 / G0b evidence commit
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
            'value_best': d / 'value_best.json', 'verdict': d / 'arena_verdict.json',
            'x2': ROOT / 'docs' / 'external' / f'x2_c{c}.json', 'cycle_index': d / 'cycle.json'}


# --- readiness and the chain ---------------------------------------------------------------------------------------

def c2_ready() -> str | bool:
    """Cycle 2's starting point: the c1 decision and its cost review, G0 (factor 1) and G0b (factor 4, both modes)
    PASS, the c1 verdict. True, or what is missing."""
    missing = [k for k in ('decision', 'cost_review', 'g0', 'g0b') if not C1[k].exists()]
    if not (DOCS / 'c1' / 'arena_verdict.json').exists():
        missing.append('c1 verdict')
    if missing:
        return f'missing: {", ".join(missing)}'
    review, decision_sha = _json(C1['cost_review']), _sha(C1['decision'])
    if review.get('decision_sha256') != decision_sha or review.get('applies_to_cycle') != 2:
        return 'the cost review does not approve this decision for cycle 2'
    for key, factor in (('g0', 1), ('g0b', auto.TEACHER_FACTOR)):
        rec = _json(C1[key])
        if rec.get('format') != 'h7c-retarget-v1-check' or rec.get('verdict') != 'PASS' or \
                rec['identity'].get('factor') != factor:
            return f'{key} is not a PASS check at factor {factor}'
    return True


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


def evidence(c: int) -> dict:
    """What exists of cycle c, normalized for ``auto.cycle_stage``."""
    recipe = _json(cycle_files(c)['recipe'])
    f = cycle_files(c, recipe)
    ev = {'cycle': c, 'tooling': TOOLING, 'ready': c2_ready() if c == auto.FIRST_CYCLE else True,
          'recipe': recipe is not None}
    selfplay = _json(f['selfplay'])
    ev['selfplay'] = selfplay is not None
    if selfplay is not None and recipe is not None:
        rows = selfplay['summary']['target_kinds'].get('visits', 0)
        ev['preflight_detail'] = auto.cost_preflight(rows, recipe['tree_seconds_mean'], recipe['workers'],
                                                     recipe['hard_cap_seconds'])
        ev['preflight'] = ev['preflight_detail']['status']
    ev['retarget'] = f['retarget'].exists()
    ev['retarget_partial'] = f['retarget_partial'].exists()
    ev['labels'] = f['labels'].exists()
    previous = DOCS / f'c{c - 1}' / 'arena_verdict.json'
    parent = (_json(previous) or {}).get('next_champion') or {}
    errors = []
    for name in ('policy', 'value'):
        record = _json(f[f'{name}_record'])
        if record is None:
            continue
        ev[f'{name}_verdict'] = record.get('verdict')
        err = auto.train_artifacts_ok(record, _sha(f[f'{name}_best']), c, parent.get(f'{name}_checkpoint_sha256'))
        if err:
            errors.append(f'{name}: {err}')
    ev['train_error'] = '; '.join(errors) or None
    ev['verdict'] = f['verdict'].exists()
    ev['x2'] = f['x2'].exists()
    ev['cycle_index'] = f['cycle_index'].exists()
    return ev


def current() -> dict:
    """The cycle the evidence is at and its step: after a complete cycle, the next-cycle decision decides whether a
    next cycle starts at all."""
    found = verdicts()
    history = [auto.cycle_entry(v['record']) for v in found]
    c = auto.FIRST_CYCLE
    while (DOCS / f'c{c}' / 'cycle.json').exists():
        c += 1
    decision = auto.next_cycle_decision(history[:c - 1], TOOLING) if len(history) >= c - 1 else None
    if decision is not None and decision['state'] not in ('NEXT_CYCLE', 'V9_HOLD'):
        return {'cycle': c - 1, 'step': decision, 'history': history, 'chain_errors': chain_errors(found)}
    ev = evidence(c)
    step = auto.cycle_stage(ev)
    if decision is not None:
        step['why']['previous_cycle'] = {k: decision['why'][k] for k in ('hold', 'stagnant', 'hold_window',
                                                                          'no_candidate_streak')}
    if ev.get('preflight_detail'):
        step['why']['preflight'] = ev['preflight_detail']
    return {'cycle': c, 'step': step, 'evidence': ev, 'history': history, 'chain_errors': chain_errors(found)}


# --- commands (only for tools that exist) --------------------------------------------------------------------------

def champion_paths(history: list[dict], c: int) -> dict:
    """The parent models of cycle c: the frozen ones until a cycle promoted, then that cycle's candidate files."""
    paths = dict(FROZEN)
    for e in history[:c - 1]:
        if e['decision'] == auto.PROMOTE:
            paths = {'policy': f'{RUNS}/c{e["cycle"]}/policy/best.pt', 'value': f'{RUNS}/c{e["cycle"]}/value/best.pt'}
            if e['candidate_mode'] == auto.FALLBACK:
                paths['policy'] = champion_paths(history, e['cycle'])['policy']
    return paths


def command(state: dict) -> list[str]:
    c, step, history = state['cycle'], state['step'], state['history']
    action, C = step['action'], f'{RUNS}/c{c}'
    par = champion_paths(history, c)
    pol, val, champ = par['policy'], par['value'], f'docs/h7c/c{c - 1}/arena_verdict.json'
    hist = ' '.join(f'docs/h7c/c{k}/arena_verdict.json' for k in range(1, c))
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
        return [f'python -u scripts/h7c_retarget.py run --games {C}/selfplay_{seed}.jsonl --decision {_rel(C1["decision"])} '
                f'--cost-review {_rel(C1["cost_review"])} --selfplay-summary docs/h7c/c{c}/selfplay_c{c}_seed{seed}.json '
                f'--policy-checkpoint {pol} --value-checkpoint {val} --device cuda --workers {WORKERS} '
                f'--jsonl {C}/retarget_s200.jsonl --output docs/h7c/c{c}/retarget_s200.json 2>&1 | tee {C}/retarget.log']
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
        return [f'python -u scripts/h7c_external.py --cycle {c} ...  # the Rapfi settings of docs/h7-c1.md (X2), '
                f'--value-device cuda, --output docs/external/x2_c{c}.json']
    if action == 'WRITE_CYCLE_INDEX':
        return [f'# docs/h7c/c{c}/cycle.json (h7c-cycle-v1): recipe, self-play, retarget, labels, records, verdict and '
                'X2 SHA-256, the verdict decision; then commit']
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
    c, recipe = state['cycle'], _json(cycle_files(state['cycle'])['recipe'])
    if recipe is not None:
        checks['recipe_sources'] = 'PASS' if recipe_errors(recipe) == [] else 'FAIL: ' + '; '.join(recipe_errors(recipe))
        changed = [p for p in _git('diff', '--name-only', recipe['code_base_commit'], 'HEAD').split() if p]
        bad = auto.evidence_only(changed, c)
        checks['evidence_only'] = 'PASS' if not bad else 'MANUAL_CODE_CHANGE_REVIEW_REQUIRED: ' + ', '.join(bad)
    porcelain = _git('status', '--porcelain', '--untracked-files=all').splitlines()
    dirty = [line[3:] for line in porcelain]
    bad = auto.evidence_only(dirty, c)
    checks['working_tree'] = ('PASS' if not dirty else 'EVIDENCE_COMMIT_REQUIRED: ' + ', '.join(dirty) if not bad
                              else 'MANUAL_CODE_CHANGE_REVIEW_REQUIRED: ' + ', '.join(bad))
    return checks


def recipe_errors(recipe: dict) -> list[str]:
    errors = []
    for key, path in (('source_decision_sha256', C1['decision']), ('source_cost_review_sha256', C1['cost_review']),
                      ('g0_sha256', C1['g0']), ('g0b_sha256', C1['g0b']),
                      ('policy_config_sha256', CONFIGS['policy']), ('value_config_sha256', CONFIGS['value'])):
        if recipe.get(key) != _sha(path):
            errors.append(key)
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
    args = parser.parse_args(argv)
    if args.mode == 'verify':
        checks = verify(args.skip_frozen)
        print(json.dumps(checks, indent=1), flush=True)
        return 0 if all(v == 'PASS' or v.startswith('NOT_C2_READY') for v in checks.values()) else 1
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
