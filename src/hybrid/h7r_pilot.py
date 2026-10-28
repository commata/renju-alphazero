"""H7-R R1 pilot rules (docs/h7-rl-plan.md 9.9): the fixed rules of a pilot contract with an ``execution_plan``,
the runner's session log and soft wall-time budget, and the suite ranking by the opening-cluster bootstrap.

A calibration of evaluation suites only: nothing here selects a model, promotes or deploys.

Session log (``sessions.jsonl``, append-only, one JSON event per line): ``session_start`` (commit, contract SHA),
``heartbeat`` / ``arm_start`` / ``arm_end`` / ``budget_stop`` / ``session_end`` (each with the session's monotonic
``elapsed`` seconds), ``extension`` (extra hours and the approval text). A session's time is its largest recorded
``elapsed``: a session that ended without ``session_end`` (killed) counts up to its last event, at most one heartbeat
interval short. Time between sessions is not counted.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from random import Random
from statistics import fmean

from .h7r_eval import require

RANK_VERSION = 'h7r-suite-rank-v1'
BUDGET_VERSION = 'h7r-pilot-budget-v1'
LOG_FORMAT = 'h7r-pilot-session-v1'
EVENTS = ('session_start', 'heartbeat', 'arm_start', 'arm_end', 'budget_stop', 'session_end', 'extension')
DIGITS = 6  # ranking comparisons (lower bound, then estimate) at this many decimals


# --- contract rules ------------------------------------------------------------------------------------------------

def check_rules(contract: dict, path) -> dict:
    """The fixed rules of a pilot contract with an ``execution_plan``; returns them in one dict. Every rule value is
    required (a draft may leave them unset, a fixed contract may not)."""
    arms = {a['name']: a for a in contract.get('arms', [])}
    plan, selection = contract.get('execution_plan') or {}, contract.get('suite_selection') or {}
    require(sorted(plan.get('arm_order') or []) == sorted(arms), f'{path}: execution_plan.arm_order is not every arm once')
    require(type(plan.get('workers')) is int and plan['workers'] > 0, f'{path}: execution_plan.workers')
    budget = plan.get('budget') or {}
    require(budget.get('version') == BUDGET_VERSION, f'{path}: execution_plan.budget version {BUDGET_VERSION}')
    hours = budget.get('soft_wall_hours')
    require(type(hours) in (int, float) and math.isfinite(hours) and hours > 0, f'{path}: soft_wall_hours')
    require(type(budget.get('heartbeat_seconds')) in (int, float) and 0 < budget['heartbeat_seconds'] <= 60,
            f'{path}: heartbeat_seconds (at most 60)')
    for key in ('accounting', 'checks', 'on_reach', 'after_reach', 'errors'):
        require(isinstance(budget.get(key), str) and budget[key], f'{path}: execution_plan.budget.{key}')
    rank = selection.get('rank_by') or {}
    require(rank.get('version') == RANK_VERSION, f'{path}: suite_selection.rank_by version {RANK_VERSION}')
    boot = rank.get('bootstrap') or {}
    require(boot.get('unit') == 'history_id' and boot.get('method') == 'percentile'
            and type(boot.get('samples')) is int and boot['samples'] >= 1000 and type(boot.get('seed')) is int
            and type(boot.get('quantile')) is float and 0 < boot['quantile'] < 0.5, f'{path}: rank_by.bootstrap')
    require(rank.get('order') == ['lower', 'estimate'] and rank.get('tie') == 'joint', f'{path}: rank_by order / tie')
    suites = rank.get('suites') or {}
    require(suites and isinstance(suites, dict), f'{path}: rank_by.suites')
    used = []
    for suite, spec in suites.items():
        small, control = spec.get('small_arms') or [], spec.get('control_arm')
        require(len(small) == 3 and all(n in arms for n in small) and control in arms,
                f'{path}: suite {suite}: three small arms and a control arm of this contract')
        samples = {(arms[n]['seed'], arms[n]['pairs'], arms[n]['pairing'], arms[n]['suite']['sha256']) for n in small}
        require(len(samples) == 1 and arms[small[0]]['pairing'],
                f'{path}: suite {suite}: the small arms must play one paired sample')
        require(arms[control]['suite']['sha256'] == arms[small[0]]['suite']['sha256']
                and arms[control]['tested']['budget'] > arms[control]['opponent']['budget'],
                f'{path}: suite {suite}: the control is a larger budget on the same suite')
        used += [*small, control]
    require(sorted(used) == sorted(arms), f'{path}: rank_by.suites must cover every arm once')
    colour = selection.get('colour_bias') or {}
    require(colour.get('gate') == 'none' and isinstance(colour.get('report'), str),
            f'{path}: suite_selection.colour_bias (gate none, a per-arm report)')
    control = contract.get('positive_control') or {}
    require(sorted(control.get('arms') or []) == sorted(s['control_arm'] for s in suites.values())
            and control.get('detection') == 'lower_one_sided > 0', f'{path}: positive_control arms / detection')
    return {'order': list(plan['arm_order']), 'workers': plan['workers'], 'soft_wall_hours': float(hours),
            'heartbeat_seconds': float(budget['heartbeat_seconds']), 'suites': suites,
            'bootstrap': boot, 'planned_pairs': {n: a['pairs'] for n, a in arms.items()}}


def is_run_by_plan(contract: dict) -> bool:
    """A fixed pilot contract whose arms are played only through the pilot runner."""
    return contract.get('purpose') == 'pilot' and contract.get('status') == 'fixed' and 'execution_plan' in contract


# --- session log ---------------------------------------------------------------------------------------------------

def load_log(path, repair: bool = True) -> list[dict]:
    """Every event; an incomplete last line (an interrupted append) is cut (``repair``, only before a session's
    writers start) or skipped (``repair=False``, while it runs), any other damage refused."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines, offset, events = [], 0, []
    for line in raw.split(b'\n'):
        lines.append((offset, line))
        offset += len(line) + 1
    content = [(o, line) for o, line in lines if line.strip()]
    for n, (start, line) in enumerate(content):
        try:
            event = json.loads(line)
            ok = isinstance(event, dict) and event.get('format') == LOG_FORMAT and event.get('event') in EVENTS
        except (json.JSONDecodeError, UnicodeDecodeError):
            event, ok = None, False
        if not ok:
            require(n == len(content) - 1, f'{path}: a damaged event at byte {start} before the end')
            if not repair:
                break
            with path.open('r+b') as handle:
                handle.truncate(start)
            break
        events.append(event)
    if not repair:
        return events
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):  # a complete last event cut before its newline
        with path.open('ab') as handle:
            handle.write(b'\n')
    return events


def append(path, event: dict) -> None:
    require(event.get('event') in EVENTS, f"unknown event {event.get('event')}")
    with Path(path).open('a', encoding='utf-8') as out:
        out.write(json.dumps({'format': LOG_FORMAT, **event}, sort_keys=True, separators=(',', ':')) + '\n')
        out.flush()
        os.fsync(out.fileno())


def check_identity(events: list[dict], commit: str, contract_sha256: str) -> None:
    """Every session of one pilot ran one commit and one contract (HEAD never changes mid-pilot)."""
    for e in events:
        if e['event'] == 'session_start':
            require((e.get('commit'), e.get('contract_sha256')) == (commit, contract_sha256),
                    f"session {e.get('session')} ran commit {e.get('commit')} / contract "
                    f"{str(e.get('contract_sha256'))[:12]}: a pilot runs from one commit and one contract")


def budget_state(events: list[dict], soft_wall_hours: float, current: tuple[str, float] | None = None) -> dict:
    """Cumulative seconds over sessions (each its largest recorded ``elapsed``; ``current``: the running session and
    its elapsed now), the limit (soft hours plus approved extensions) and whether it is reached."""
    per = {}
    for e in events:
        if e['event'] != 'extension' and 'session' in e:
            per[e['session']] = max(per.get(e['session'], 0.0), float(e.get('elapsed', 0.0)))
    if current is not None:
        per[current[0]] = max(per.get(current[0], 0.0), current[1])
    extensions = [{'hours': e['hours'], 'approval': e['approval']} for e in events if e['event'] == 'extension']
    total = sum(per.values())
    limit = 3600 * (soft_wall_hours + sum(x['hours'] for x in extensions))
    return {'cumulative_seconds': round(total, 1), 'cumulative_hours': round(total / 3600, 3),
            'limit_hours': round(limit / 3600, 3), 'reached': total >= limit, 'sessions': len(per),
            'extensions': extensions, 'budget_stops': sum(e['event'] == 'budget_stop' for e in events)}


# --- suite ranking -------------------------------------------------------------------------------------------------

def opening_rows(tables: dict, arms: list[str], planned: dict) -> tuple[list[dict] | None, list[str]]:
    """Per opening, the three small arms together: ``q`` (share of the arms whose pair score is not 0.5) and ``m``
    (share whose two games' moves differ). ``None`` with the reasons when an arm is missing, invalid, short of its
    planned valid pairs, has an opening twice, or the arms do not play the same openings (no pair is ever added)."""
    problems = []
    by_arm = {}
    for name in arms:
        table = tables.get(name)
        if table is None:
            problems.append(f'{name}: no verified record')
            continue
        rows = table['pairs']
        ids = [r['opening_id'] for r in rows]
        if len(set(ids)) != len(ids):
            problems.append(f'{name}: an opening twice')
        if not table['valid']:
            problems.append(f"{name}: {len(table['opponent_illegal_pairs'])} pairs invalid (opponent illegal)")
        if len(rows) != planned[name]:
            problems.append(f'{name}: {len(rows)} valid pairs, {planned[name]} planned')
        by_arm[name] = {r['opening_id']: r for r in rows}
    if not problems and len({frozenset(v) for v in by_arm.values()}) != 1:
        problems.append('the small arms do not play the same openings')
    if problems:
        return None, problems
    ids = sorted(by_arm[arms[0]])
    return [{'opening_id': i, 'q': fmean(bool(by_arm[a][i]['score_split']) for a in arms),
             'm': fmean(bool(by_arm[a][i]['move_split']) for a in arms)} for i in ids], []


def cluster_bootstrap(values: list[float], bootstrap: dict) -> dict:
    """The mean of ``values`` (one per opening) and the ``quantile`` of its percentile bootstrap over openings."""
    rng, n = Random(bootstrap['seed']), len(values)
    means = sorted(fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(bootstrap['samples']))
    q = bootstrap['quantile']
    return {'estimate': round(fmean(values), DIGITS), 'lower': round(means[int(q * (len(means) - 1))], DIGITS),
            'openings': n, 'zero_variance': len(set(values)) == 1}


def rank(stats: dict) -> list[dict]:
    """Suites by the lower bound, then the estimate (both at ``DIGITS`` decimals); equal on both: a joint rank."""
    key = lambda s: (stats[s]['lower'], stats[s]['estimate'])  # noqa: E731
    order = sorted(stats, key=lambda s: (-key(s)[0], -key(s)[1], s))
    out, previous = [], None
    for i, suite in enumerate(order):
        place = out[-1]['rank'] if previous is not None and key(suite) == previous else i + 1
        out.append({'suite': suite, 'rank': place, 'lower': stats[suite]['lower'],
                    'estimate': stats[suite]['estimate']})
        previous = key(suite)
    return out


def colour_report(summary: dict) -> dict:
    by = summary.get('score_by_tested_colour') or {}
    if not by:
        return {'tested_black_score': None, 'tested_white_score': None, 'absolute_gap': None}
    return {'tested_black_score': by['black'], 'tested_white_score': by['white'],
            'absolute_gap': round(abs(by['black'] - by['white']), 5)}


def control_report(summary: dict) -> dict:
    lower = summary.get('lower_one_sided')
    return {'pairs': summary['pairs'], 'score': summary.get('score'), 'improvement': summary.get('improvement'),
            'ci95': summary.get('ci95'), 'lower_one_sided': lower, 'alpha_one_sided': summary.get('alpha_one_sided'),
            'detected': lower is not None and lower > 0,
            'reading': None if lower is None else 'DETECTED' if lower > 0 else
            'NOT_DETECTED: the one-sided lower bound is not above 0 (report the effect and interval; no rerun)'}


def report(contract: dict, rules: dict, arms: dict, events: list[dict]) -> dict:
    """``arms``: name -> {'table', 'summary'} for every verified arm record (missing arms absent)."""
    suites, problems = {}, {}
    for suite, spec in rules['suites'].items():
        rows, issues = opening_rows({n: arms[n]['table'] for n in spec['small_arms'] if n in arms},
                                    spec['small_arms'], rules['planned_pairs'])
        control = arms.get(spec['control_arm'])
        if control is None:
            issues.append(f"{spec['control_arm']}: no verified record")
        elif not control['table']['valid'] or control['summary']['pairs'] != rules['planned_pairs'][spec['control_arm']]:
            issues.append(f"{spec['control_arm']}: {control['summary']['pairs']} valid pairs, "
                          f"{rules['planned_pairs'][spec['control_arm']]} planned")
        entry = {'complete': not issues, 'problems': issues}
        if rows is not None:
            entry['score_split'] = cluster_bootstrap([r['q'] for r in rows], rules['bootstrap'])
            entry['move_split'] = cluster_bootstrap([r['m'] for r in rows], rules['bootstrap'])
        if control is not None:
            entry['positive_control'] = control_report(control['summary'])
        suites[suite] = entry
        if issues:
            problems[suite] = issues
    complete = not problems
    per_arm = {n: {'pairs': a['summary']['pairs'], 'tested_illegal': a['summary']['tested_illegal'],
                   'opponent_illegal_pairs': len(a['summary']['opponent_illegal_pairs']),
                   'move_split': a['summary']['move_split'], 'score_split': a['summary']['score_split'],
                   'colour': colour_report(a['summary'])} for n, a in arms.items()}
    return {'status': 'COMPLETE' if complete else 'INCOMPLETE',
            'ranking': rank({s: suites[s]['score_split'] for s in suites}) if complete else None,
            'ranking_note': None if complete else 'no final suite choice until every required arm is complete',
            'suites': suites, 'arms': per_arm, 'missing_arms': sorted(set(rules['order']) - set(arms)),
            'budget': budget_state(events, rules['soft_wall_hours']),
            'meaning': 'a calibration of evaluation suites; no model is selected, promoted or deployed'}
