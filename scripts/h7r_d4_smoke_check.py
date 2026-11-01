"""R2-D4 smoke operation check (read-only): is the smoke run of ``scripts/h7r_d4.py`` sound, re-derived from sources.

Nothing of the run is trusted as a status string. The check re-verifies the contract, the smoke approval, the
eligible list and the run identity (SHA-256 links), the summary against its identity and journals, the full replay
of the smoke games against the verified H-dev source (the c5 actor reproduced with ``h7c_sensitivity.check_replay``),
the eligible / warm-up split, the disagreement observations and the fixed-order branch set (at most 8), and every
branch through the Game replay of ``h7r_evidence.branch_outcome``. No GPU inference and no Rapfi game is run.

The run files (identity, summary, journals, logs, cache) are only read: the journals have their own strict reader
(the runner's ``read_journal`` cuts a torn last line and appends a newline, which a check must never do). A torn,
duplicated, missing, unknown or non-finite record is a FAIL. The result is written once to a new path.

Verdict (a smoke operation check; it changes no D4 reading and is no approval of the full run):
    FAIL                integrity or actor reproduction fails, an undefined INVALID, or any execution failure
                        (timeout / engine_error / engine_illegal / forbid_mismatch)
    PASS_WITHOUT_RAPFI  none of these, but no branch ended (VALID or verified structural draw) with a legal
                        continuation by the external engine: the normal Rapfi continuation path is unconfirmed
                        (this does not mean the engine was never called)
    PASS                none of these and at least one such continuation

Timing comes from the timestamped desktop logs (``<epoch seconds> <line>``, ``__START__`` / ``__EXIT__ <code>``
lines added by the wrapper); a boundary that cannot be read is null, never guessed. Branch ``seconds`` sum to worker
time, not elapsed time. Lock and resume tests are reported only from their evidence; otherwise NOT_TESTED.

    python -u scripts/h7r_d4_smoke_check.py --approval docs/h7r/r2/R2-D4-BASELINE-1/execution-smoke.json \\
        --log runs/h7r/R2-D4-BASELINE-1/logs/smoke-<stamp>.log \\
        --output docs/h7r/r2/R2-D4-BASELINE-1/records/smoke/smoke_check.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from random import Random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hybrid import h7r_d4 as d4  # noqa: E402

FORMAT = 'h7r-d4-smoke-check-v1'
CONTRACT = 'docs/h7r/r2/R2-D4-BASELINE-1.contract.json'
SAMPLE = 'docs/h7r/r2/R2-D4-BASELINE-1/eligible.json'
SMOKE_DIR = 'runs/h7r/R2-D4-BASELINE-1/smoke'
SMOKE_GAMES, SMOKE_BRANCHES = 2, 8
EXPECTED = {'replay_plies': 43, 'eligible_plies': 39, 'warmup_plies': 4}  # games 0 and 1 of the fixed list
FULL_ENGINE_PLIES = 16777  # replay plies of the 396 games (eligible 15,985 + warm-up 792)
EXECUTION_REASONS = ('timeout', 'engine_error', 'engine_illegal', 'forbid_mismatch')
JOURNALS = ('replay.jsonl', 'branches.jsonl')


class Failure(Exception):
    """A failed check (the message names it)."""


def need(ok, message: str) -> None:
    if not ok:
        raise Failure(message)


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def strict_journal(path: Path, field: str, identity_sha: str, expected: set) -> dict:
    """Every line a complete JSON record of this identity, each key once, exactly the expected keys. Read only."""
    raw = path.read_bytes()
    need(raw.endswith(b'\n'), f'{path.name}: torn or unterminated last line')
    done = {}
    for number, line in enumerate(raw.split(b'\n')[:-1], 1):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise Failure(f'{path.name}: line {number} is not a JSON record') from None
        need(isinstance(record, dict), f'{path.name}: line {number} is not an object')
        need(record.get('identity_sha256') == identity_sha, f'{path.name}: line {number} has another identity')
        key = record.get(field)
        need(key in expected, f'{path.name}: line {number} is an unknown record ({key!r})')
        need(key not in done, f'{path.name}: {key!r} recorded twice')
        done[key] = record
    need(set(done) == set(expected), f'{path.name}: missing records {sorted(set(expected) - set(done), key=str)[:5]}')
    return done


# --- logs and timing -----------------------------------------------------------------------------------------------

_LINE = re.compile(r'^(\d+(?:\.\d+)?) (.*)$')
_RESUME = re.compile(r'^\[(d4-replay|d4-rapfi)\] resume (\d+)/(\d+), remaining (\d+)')
_PROGRESS = re.compile(r'^\[(d4-replay|d4-rapfi)\] ([\d,]+)/([\d,]+) units')


def read_session(path: Path) -> dict:
    """One run invocation from its timestamped log: stage marks (epoch seconds) or None."""
    marks = {k: None for k in ('start', 'preflight_start', 'preflight_end', 'replay_start', 'replay_done',
                               'rapfi_start', 'rapfi_done', 'final', 'exit')}
    out = {'log': str(path), 'exit_code': None, 'resume': {}, 'lines': 0, 'untimed_lines': 0}
    for raw in path.read_text(encoding='utf-8', errors='replace').splitlines():
        out['lines'] += 1
        m = _LINE.match(raw)
        if not m:
            out['untimed_lines'] += 1
            continue
        t, text = float(m.group(1)), m.group(2)
        if text == '__START__' and marks['start'] is None:
            marks['start'] = t
        elif text.startswith('__EXIT__'):
            marks['exit'] = t
            code = text.split()[1:2]
            out['exit_code'] = int(code[0]) if code and code[0].lstrip('-').isdigit() else None
        elif text.startswith('[d4-preflight] checking') and marks['preflight_start'] is None:
            marks['preflight_start'] = t
        elif text.startswith('[d4-preflight] PASS'):
            marks['preflight_end'] = t
        elif (r := _RESUME.match(text)):
            stage = 'replay' if r.group(1) == 'd4-replay' else 'rapfi'
            marks[f'{stage}_start'] = t
            out['resume'][stage] = {'done': int(r.group(2)), 'total': int(r.group(3)), 'remaining': int(r.group(4))}
            if int(r.group(4)) == 0:
                marks[f'{stage}_done'] = t
        elif (pr := _PROGRESS.match(text)):
            stage = 'replay' if pr.group(1) == 'd4-replay' else 'rapfi'
            if pr.group(2) == pr.group(3):
                marks[f'{stage}_done'] = t
        elif text.startswith('{') and '"reading"' in text:
            marks['final'] = t
    out['marks'] = marks
    return out


def span(a, b):
    return None if a is None or b is None or b < a else round(b - a, 3)


def timing(sessions: list[dict], branch_records: dict, workers: int | None, longest_game: int | None) -> dict:
    out = {'sessions': [], 'branch_worker_seconds': None, 'reference': None}
    for s in sessions:
        m = s['marks']
        out['sessions'].append({
            'log': s['log'], 'exit_code': s['exit_code'], 'resume': s['resume'],
            'seconds': {'total': span(m['start'], m['exit']),
                        'preflight': span(m['preflight_start'], m['preflight_end']),
                        'identity_and_lock': span(m['preflight_end'], m['replay_start']),
                        'replay': span(m['replay_start'], m['replay_done']),
                        'observations': span(m['replay_done'], m['rapfi_start']),
                        'rapfi_elapsed': span(m['rapfi_start'], m['rapfi_done']),
                        'revalidation_and_summary': span(m['rapfi_done'], m['final'])},
            'untimed_lines': s['untimed_lines']})
    seconds = [r.get('seconds') for r in branch_records.values()]
    if seconds and all(finite(x) and x >= 0 for x in seconds):
        out['branch_worker_seconds'] = {'branches': len(seconds), 'sum': round(sum(seconds), 3),
                                        'mean': round(sum(seconds) / len(seconds), 3),
                                        'note': 'worker time (branches run in parallel), not elapsed time'}
    last = out['sessions'][-1] if out['sessions'] else None
    replay = last and last['seconds']['replay']
    resumed = last and (last['resume'].get('replay') or {}).get('done', 0) > 0
    if replay and not resumed and workers and longest_game:
        # the smoke games run side by side: the longest game bounds the elapsed replay
        per_ply = replay / longest_game
        out['reference'] = {
            'kind': 'reference only: two games, per-worker model loading and CUDA start inside the replay span',
            'replay_seconds_per_engine_ply_all_models': round(per_ply, 3),
            'full_replay_hours_at_workers': round(per_ply * FULL_ENGINE_PLIES / workers / 3600, 2),
            'workers': workers, 'full_rapfi': 'branches x mean branch worker seconds / workers (branches known only '
                                              'after the full replay)'}
    return out


def lock_test(path: Path | None, sessions: list[dict]) -> dict:
    """``LOCK_REFUSED <epoch>`` inside a run's locked span (after its replay started, before it ended)."""
    if path is None:
        return {'status': 'NOT_TESTED'}
    events = []
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        m = re.match(r'^(LOCK_REFUSED|LOCK_ACQUIRED) (\d+(?:\.\d+)?)', line.strip())
        if m:
            events.append((m.group(1), float(m.group(2))))
    if not events:
        return {'status': 'UNKNOWN', 'reason': 'no LOCK_REFUSED / LOCK_ACQUIRED line'}
    windows = [(s['marks']['replay_start'], s['marks']['final'] or s['marks']['exit']) for s in sessions]
    windows = [(a, b) for a, b in windows if a is not None and b is not None]
    inside = [(kind, t) for kind, t in events if any(a <= t <= b for a, b in windows)]
    if not inside:
        return {'status': 'UNKNOWN', 'reason': 'the probe is outside every readable locked span', 'events': events}
    if any(kind == 'LOCK_ACQUIRED' for kind, _ in inside):
        return {'status': 'TESTED_FAIL', 'events': inside}
    return {'status': 'TESTED_PASS', 'events': inside}


def resume_test(sessions: list[dict]) -> dict:
    if len(sessions) < 2:
        return {'status': 'NOT_TESTED'}
    interrupted = [s for s in sessions[:-1] if s['exit_code'] not in (0, None) or s['marks']['final'] is None]
    resumed = [s for s in sessions[1:] if any(v.get('done', 0) > 0 for v in s['resume'].values())]
    if interrupted and resumed and sessions[-1]['exit_code'] == 0:
        return {'status': 'TESTED', 'interrupted_logs': [s['log'] for s in interrupted],
                'resumed': [s['resume'] for s in resumed]}
    return {'status': 'UNKNOWN', 'reason': 'several logs without an interrupted session followed by a resumed one'}


# --- the check -------------------------------------------------------------------------------------------------------

def verify(args, root: Path) -> dict:
    """Every integrity step; raises ``Failure`` at the first break (later steps depend on earlier ones)."""
    import h7c_adjudicate as ad
    import h7c_sensitivity as sn
    import h7r_d4 as cli
    from hybrid.h7c_nmask import committed

    facts = {}
    contract_path, sample_path, approval_path = root / CONTRACT, root / SAMPLE, Path(args.approval)
    contract, csha = d4.load_contract(contract_path, root)
    need(committed(contract_path, root), 'the contract is not committed')
    approval = json.loads(approval_path.read_text(encoding='utf-8'))
    need(approval.get('format') == 'h7r-d4-execution-v1' and approval.get('contract_sha256') == csha
         and approval.get('mode') == 'smoke' and approval.get('approved') is True, 'not the smoke approval')
    need(committed(approval_path, root), 'the smoke approval is not committed')
    asha = d4.file_sha256(approval_path)
    sample = json.loads(sample_path.read_text(encoding='utf-8'))
    need(sample.get('contract_sha256') == csha and committed(sample_path, root), 'the eligible list is not bound')
    esha = d4.file_sha256(sample_path)
    facts['shas'] = {'contract': csha, 'approval': asha, 'eligible': esha}

    folder = Path(args.smoke_dir)
    need(folder.resolve() == (root / SMOKE_DIR).resolve(), f'the smoke directory is {SMOKE_DIR}')
    identity = json.loads((folder / 'identity.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    identity_sha = d4.digest(identity)
    need(summary.get('format') == 'h7r-d4-summary-v1' and summary.get('reading') == 'SMOKE', 'not a smoke summary')
    need(summary.get('identity') == identity and summary.get('identity_sha256') == identity_sha,
         'the summary is not bound to identity.json')
    need(identity.get('format') == 'h7r-d4-run-v1' and identity.get('mode') == 'smoke'
         and identity.get('contract_sha256') == csha and identity.get('approval_sha256') == asha
         and identity.get('sample_sha256') == esha and identity.get('device') == contract['device'],
         'the run identity names another contract, approval, list, mode or device')
    need(identity.get('runtime') == contract['runtime_identity'] and identity.get('rapfi') == contract['rapfi_identity'],
         'the run identity names another engine runtime or Rapfi')
    for name, spec in contract['models'].items():
        got = (identity.get('models') or {}).get(name) or {}
        need(all(got.get(k) == spec[k] for k in ('value_checkpoint_sha256', 'value_metadata_sha256'))
             and (got.get('frozen_state_sha256') or {}).get('recomputed') == contract['frozen_state_sha256'],
             f'the run identity of model {name} differs from the contract')
    head = cli.git_state(root)
    need(identity.get('git') == head, f"the run commit {str((identity.get('git') or {}).get('commit'))[:12]} is not "
                                      f"the checked commit {head['commit'][:12]}")
    need(finite(summary.get('seconds')) and summary['seconds'] >= 0, 'summary seconds are not finite')
    facts['identity_sha256'] = identity_sha

    files = summary.get('files') or {}
    need(set(files) <= set(JOURNALS) and 'replay.jsonl' in files, 'summary journal list differs')
    for name in JOURNALS:
        path = folder / name
        need((name in files) == path.exists(), f'{name}: present / listed mismatch')
        if path.exists():
            need(d4.file_sha256(path) == files[name], f'{name}: SHA differs from the summary')

    manifest, games, hdev = cli.inputs(contract, csha, args.cache, root)
    need(manifest == sample, 'the committed eligible list differs from the verified H-dev source')
    need(hdev == identity.get('hdev'), 'the H-dev checks differ from the run identity')
    keep = sorted(games)[:SMOKE_GAMES]
    smoke_games = {i: games[i] for i in keep}
    replay = strict_journal(folder / 'replay.jsonl', 'game_index', identity_sha, set(keep))
    models = {name: {} for name in contract['models']}
    records = [replay[i] for i in keep]
    for record in records:
        sn.check_replay(record, smoke_games, models, 'c5')  # coverage, legality, the c5 actor reproduced
    actor = sn.actor_check(records, smoke_games, 'c5')
    need(actor == summary.get('actor_check') and actor['reproduced'], 'the c5 actor check differs or failed')
    smoke_manifest = {**manifest, 'games': [e for e in manifest['games'] if e['game_index'] in keep]}
    filtered = cli.eligible_replays(records, smoke_manifest)
    counts = {'replay_plies': sum(len(r['plies']) for r in records),
              'eligible_plies': sum(len(r['plies']) for r in filtered)}
    counts['warmup_plies'] = counts['replay_plies'] - counts['eligible_plies']
    need(counts == EXPECTED, f'smoke plies {counts} != expected {EXPECTED}')
    need(summary.get('eligible_plies') == counts['eligible_plies']
         and summary.get('warmup_plies') == counts['warmup_plies'], 'summary ply counts differ')
    facts.update(games=keep, plies=counts, actor_check=actor,
                 longest_game_plies=max(len(r['plies']) for r in records))

    obs = cli.observations_for(filtered, smoke_games)
    need(summary.get('observations') == len(obs), 'summary observation count differs')
    table = ad.branch_table(obs)
    keys = sorted(table)
    Random(contract['stats']['order_seed']).shuffle(keys)
    keys = keys[:SMOKE_BRANCHES]
    tasks = {k: {**table[k], 'order': i, 'identity_sha256': identity_sha} for i, k in enumerate(keys)}
    need(summary.get('branches') == len(keys), 'summary branch count differs')
    branches = {}
    if keys:
        need((folder / 'branches.jsonl').exists(), 'branches.jsonl is missing')
        branches = strict_journal(folder / 'branches.jsonl', 'key', identity_sha, set(keys))
    else:  # no disagreement in the smoke games: the runner never opens the branch journal
        need(not (folder / 'branches.jsonl').exists(), 'branches.jsonl exists although no branch is expected')
    tally = {'VALID': 0, 'structural_draw': 0, 'execution_failures': {r: 0 for r in EXECUTION_REASONS},
             'fixed_move_terminal': 0, 'continuation': 0}
    for key, record in branches.items():
        need(finite(record.get('seconds')) and record['seconds'] >= 0, f'branch {key[:12]}: seconds not finite')
        outcome = cli.validate_branch(record, tasks[key], contract['invalid_policy'])  # Game replay, policy
        moves = record.get('continuation') or []
        if outcome['kind'] == 'resolved':
            tally['VALID'] += 1
        elif outcome['kind'] == 'structural':
            need(record.get('status') == 'INVALID' and record.get('reason') == 'abnormal_end',
                 f'branch {key[:12]}: structural draw without its original INVALID mark')
            tally['structural_draw'] += 1
        else:
            reason = outcome['reason']
            need(reason in EXECUTION_REASONS, f'branch {key[:12]}: undefined INVALID {reason!r}')
            tally['execution_failures'][reason] += 1
        if outcome['kind'] in ('resolved', 'structural'):
            if moves:
                tally['continuation'] += 1
            elif outcome['kind'] == 'resolved':
                tally['fixed_move_terminal'] += 1
    facts.update(observations=len(obs), branches_expected=len(keys), branch_tally=tally,
                 unique_branches_in_smoke_games=len(table))
    facts['branch_records'] = branches
    return facts


def check(args, root=ROOT) -> dict:
    output = Path(args.output)
    need(not output.exists(), f'{output} exists')
    report = {'format': FORMAT, 'scope': 'smoke operation check; no D4 reading and no approval of the full run',
              'integrity': None, 'failures': [], 'verdict': None}
    branches = {}
    try:
        facts = verify(args, root)
        branches = facts.pop('branch_records')
        report.update(facts)
        report['integrity'] = 'PASS'
    except (Failure, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        report['integrity'] = 'FAIL'
        report['failures'].append(f'{type(exc).__name__}: {exc}')
    sessions = [read_session(Path(p)) for p in (args.log or [])]
    report['timing'] = (timing(sessions, branches, args.workers, report.get('longest_game_plies')) if sessions
                        else {'sessions': [], 'status': 'UNKNOWN', 'reason': 'no timestamped log given'})
    report['lock_test'] = lock_test(Path(args.lock_log) if args.lock_log else None, sessions)
    report['resume_test'] = resume_test(sessions)
    tally = report.get('branch_tally')
    if report['integrity'] != 'PASS':
        report['verdict'] = 'FAIL'
    elif any(tally['execution_failures'].values()):
        report['verdict'] = 'FAIL'
        report['failures'].append(f"execution failures {tally['execution_failures']}")
    elif tally['continuation'] == 0:
        report['verdict'] = 'PASS_WITHOUT_RAPFI'
    else:
        report['verdict'] = 'PASS'
    from hybrid.h7r_ledger import publish_json
    publish_json(output, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--approval', required=True, help='the committed smoke approval (execution-smoke.json)')
    parser.add_argument('--cache', default='runs/h7r/R1-HDEV-1/cache')
    parser.add_argument('--smoke-dir', default=SMOKE_DIR)
    parser.add_argument('--log', action='append', help='timestamped run log, one per invocation, oldest first')
    parser.add_argument('--lock-log', help='output of the optional lock probe')
    parser.add_argument('--workers', type=int, default=6, help='workers of the planned full run (reference ETA)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        report = check(args)
    except Failure as exc:
        print(f'STOP: {exc}', file=sys.stderr, flush=True)
        return 1
    tally = report.get('branch_tally') or {}
    print(f"[d4-smoke-check] {report['verdict']} (integrity {report['integrity']}); actor "
          f"{(report.get('actor_check') or {}).get('reproduced')}; plies {report.get('plies')}; "
          f"branches {report.get('branches_expected')}; tally {tally}", flush=True)
    for failure in report['failures']:
        print(f'  failure: {failure}', flush=True)
    print(f"  lock {report['lock_test']['status']}, resume {report['resume_test']['status']}", flush=True)
    return {'PASS': 0, 'PASS_WITHOUT_RAPFI': 3}.get(report['verdict'], 1)


if __name__ == '__main__':
    raise SystemExit(main())
