"""R2-D4 full run check (read-only): the full run of ``scripts/h7r_d4.py`` re-verified and its reading recomputed.

Nothing of the run is taken on trust. The check re-verifies the contract, the full approval, the eligible list and
the run identity (SHA-256 links, model identities, commit), the summary against its identity and journals, the
replay of every game of the list against the verified H-dev source (``h7c_sensitivity.check_replay``: coverage,
legality and the c5 actor reproduced), the eligible / warm-up split, the disagreement observations and the complete
fixed-order branch set (nothing missing, repeated or added), and every branch through the Game replay and INVALID
policy of the contract (``h7r_d4.validate_branch``; known execution failures stay unresolved branches as the contract
says). It then recomputes, with the runner's own functions, the same game order, bootstrap seed and 20,000
replicates: sensitivity, the two-sided bounds, every comparison's statistics and labels, the joint reading.

Integrity and research reading are separate: NOT_ESTABLISHED or INCONCLUSIVE is a reading, not a failed check.
Labels, counts, keys and strings must match exactly; floats within ``FLOAT_TOLERANCE`` (the same code on the same
inputs gives the same numbers; the tolerance only absorbs JSON round trips). No GPU search, no Rapfi game. The run
files are only read (own strict journal reader); the result is written once to a new path.

    python -u scripts/h7r_d4_full_check.py --approval docs/h7r/r2/R2-D4-BASELINE-1/execution-full.json \\
        --log runs/h7r/R2-D4-BASELINE-1/logs/full-<stamp>.log --output docs/h7r/r2/R2-D4-BASELINE-1/records/full/check.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hybrid import h7r_d4 as d4  # noqa: E402
import h7r_d4_smoke_check as smoke  # noqa: E402

FORMAT = 'h7r-d4-full-check-v1'
CONTRACT = smoke.CONTRACT
SAMPLE = smoke.SAMPLE
FULL_DIR = 'runs/h7r/R2-D4-BASELINE-1/full'
FLOAT_TOLERANCE = {'rel': 1e-9, 'abs': 1e-12}
RECOMPUTED = ('actor_check', 'eligible_plies', 'warmup_plies', 'branches', 'observations', 'branch_outcomes',
              'sensitivity', 'comparisons', 'reading', 'shared_branch_uses', 'shared_branches_by_comparison',
              'prediction_context')
need, Failure = smoke.need, smoke.Failure


def differences(saved, actual, path='') -> list[str]:
    """Paths where ``saved`` (from JSON) and ``actual`` differ: exact for everything but floats."""
    if isinstance(saved, dict) and isinstance(actual, dict):
        if set(saved) != set(actual):
            return [f'{path}: keys {sorted(set(saved) ^ set(actual))[:5]}']
        return [d for k in sorted(saved) for d in differences(saved[k], actual[k], f'{path}/{k}')]
    if isinstance(saved, list) and isinstance(actual, list):
        if len(saved) != len(actual):
            return [f'{path}: length {len(saved)} != {len(actual)}']
        return [d for i, (a, b) in enumerate(zip(saved, actual)) for d in differences(a, b, f'{path}[{i}]')]
    if isinstance(saved, bool) or isinstance(actual, bool) or not all(isinstance(x, (int, float)) for x in (saved, actual)):
        return [] if saved == actual and type(saved) is type(actual) else [f'{path}: {saved!r} != {actual!r}']
    if isinstance(saved, int) and isinstance(actual, int):
        return [] if saved == actual else [f'{path}: {saved} != {actual}']
    ok = (math.isfinite(saved) and math.isfinite(actual)
          and math.isclose(saved, actual, rel_tol=FLOAT_TOLERANCE['rel'], abs_tol=FLOAT_TOLERANCE['abs']))
    return [] if ok else [f'{path}: {saved!r} != {actual!r}']


def recompute(contract, games, records, filtered, obs, table, outcomes, cli, sn, ad):
    """The run's own summary computation (``h7r_d4.run``, full mode) on the verified records."""
    counts = d4.multiplicities(len(games), contract['stats']['samples'], contract['stats']['seed'])
    out = {'actor_check': sn.actor_check(records, games, 'c5'),
           'eligible_plies': sum(len(r['plies']) for r in filtered),
           'warmup_plies': sum(len(r['plies']) for r in records) - sum(len(r['plies']) for r in filtered),
           'branches': len(outcomes), 'observations': len(obs),
           'branch_outcomes': {kind: sum(o['kind'] == kind for o in outcomes.values())
                               for kind in ('resolved', 'structural', 'unresolved')},
           'sensitivity': cli.sensitivity(filtered, counts),
           'comparisons': {p: d4.selection_stats([o for o in obs if o['compare'] == p], outcomes, sorted(games), counts)
                           for p in d4.COMPARISONS}}
    out['reading'] = d4.joint_reading(out['comparisons'])
    for k in ('label95', 'label_individual'):
        out['comparisons']['c8:c5'].pop(k)
    out['shared_branch_uses'] = 2 * len(obs) - len(table)
    used = {p: {o[k] for o in obs if o['compare'] == p for k in ('key_a', 'key_b')} for p in d4.COMPARISONS}
    out['shared_branches_by_comparison'] = {a: {b: len(used[a] & used[b]) for b in used} for a in used}
    eps = json.loads((ROOT / contract['epsilon_result']).read_text())
    out['prediction_context'] = {'epsilon': eps['epsilon'], 'seed_metrics': eps['seeds']}
    return json.loads(json.dumps(out, allow_nan=False))  # the JSON form the summary was written in


def verify(args, root: Path) -> dict:
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
         and approval.get('mode') == 'full' and approval.get('approved') is True, 'not the full approval')
    need(committed(approval_path, root), 'the full approval is not committed')
    asha = d4.file_sha256(approval_path)
    sample = json.loads(sample_path.read_text(encoding='utf-8'))
    need(sample.get('contract_sha256') == csha and committed(sample_path, root), 'the eligible list is not bound')
    need(sample.get('hdev') == contract['hdev'], 'the eligible list names another H-dev')
    esha = d4.file_sha256(sample_path)
    facts['shas'] = {'contract': csha, 'approval': asha, 'eligible': esha}

    folder = Path(args.full_dir)
    need(folder.resolve() == (root / FULL_DIR).resolve(), f'the full directory is {FULL_DIR}')
    identity = json.loads((folder / 'identity.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    identity_sha = d4.digest(identity)
    need(summary.get('format') == 'h7r-d4-summary-v1' and summary.get('reading') != 'SMOKE', 'not a full summary')
    need(summary.get('identity') == identity and summary.get('identity_sha256') == identity_sha,
         'the summary is not bound to identity.json')
    need(identity.get('format') == 'h7r-d4-run-v1' and identity.get('mode') == 'full'
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
    need(isinstance(summary.get('seconds'), (int, float)) and math.isfinite(summary['seconds'])
         and summary['seconds'] >= 0, 'summary seconds are not finite')
    files = summary.get('files') or {}
    need(set(files) <= set(smoke.JOURNALS) and 'replay.jsonl' in files, 'summary journal list differs')
    for name in smoke.JOURNALS:
        path = folder / name
        need((name in files) == path.exists(), f'{name}: present / listed mismatch')
        if path.exists():
            need(d4.file_sha256(path) == files[name], f'{name}: SHA differs from the summary')
    facts['identity_sha256'] = identity_sha

    manifest, games, hdev = cli.inputs(contract, csha, args.cache, root)
    need(manifest == sample, 'the committed eligible list differs from the verified H-dev source')
    need(hdev == identity.get('hdev'), 'the H-dev checks differ from the run identity')
    replay = smoke.strict_journal(folder / 'replay.jsonl', 'game_index', identity_sha, set(games))
    models = {name: {} for name in contract['models']}
    records = [replay[i] for i in sorted(games)]
    for record in records:
        sn.check_replay(record, games, models, 'c5')  # coverage, legality, the c5 actor reproduced
    filtered = cli.eligible_replays(records, manifest)
    need(sum(len(r['plies']) for r in filtered) == manifest['rows'], 'eligible plies differ from the list')
    obs = cli.observations_for(filtered, games)
    table = ad.branch_table(obs)
    keys = sorted(table)
    Random(contract['stats']['order_seed']).shuffle(keys)
    tasks = {k: {**table[k], 'order': i, 'identity_sha256': identity_sha} for i, k in enumerate(keys)}
    branches = {}
    if keys:
        need((folder / 'branches.jsonl').exists(), 'branches.jsonl is missing')
        branches = smoke.strict_journal(folder / 'branches.jsonl', 'key', identity_sha, set(keys))
    else:
        need(not (folder / 'branches.jsonl').exists(), 'branches.jsonl exists although no branch is expected')
    outcomes, reasons = {}, {}
    for key, record in branches.items():
        need(isinstance(record.get('seconds'), (int, float)) and math.isfinite(record['seconds'])
             and record['seconds'] >= 0, f'branch {key[:12]}: seconds not finite')
        outcomes[key] = cli.validate_branch(record, tasks[key], contract['invalid_policy'])  # Game replay, policy
        if outcomes[key]['kind'] == 'structural':
            need(record.get('status') == 'INVALID' and record.get('reason') == 'abnormal_end',
                 f'branch {key[:12]}: structural draw without its original INVALID mark')
        if outcomes[key]['kind'] != 'resolved':
            reasons[outcomes[key]['reason']] = reasons.get(outcomes[key]['reason'], 0) + 1
    actual = recompute(contract, games, records, filtered, obs, table, outcomes, cli, sn, ad)
    mismatches = [d for key in RECOMPUTED for d in differences(summary.get(key), actual[key], key)]
    need(not mismatches, f'the summary differs from the recomputation: {mismatches[:5]}'
                         + (f' (+{len(mismatches) - 5} more)' if len(mismatches) > 5 else ''))
    need(actual['actor_check']['reproduced'], 'the c5 actor was not reproduced')
    facts.update(games=len(games), plies={'eligible': actual['eligible_plies'], 'warmup': actual['warmup_plies']},
                 actor_check=actual['actor_check'], observations=len(obs), branches=len(keys),
                 branch_outcomes=actual['branch_outcomes'], not_resolved_by_reason=reasons,
                 branch_worker_seconds=round(sum(r['seconds'] for r in branches.values()), 3))
    facts['reading'] = {'joint': actual['reading'],
                        'comparisons': {p: {k: c.get(k) for k in ('label95', 'label_individual', 'observations',
                                                                  'resolved', 'decisive', 'decisive_games',
                                                                  'unresolved', 'mean_delta_bounds', 'failures',
                                                                  'upper95_descriptive')}
                                        for p, c in actual['comparisons'].items()}}
    return facts


def check(args, root=ROOT) -> dict:
    output = Path(args.output)
    need(not output.exists(), f'{output} exists')
    report = {'format': FORMAT, 'float_tolerance': FLOAT_TOLERANCE,
              'scope': 'record check of the D4 full run and its recomputed development reading; no strength, '
                       'promotion, V9 or champion ruling',
              'integrity': None, 'failures': []}
    try:
        report.update(verify(args, root))
        report['integrity'] = 'PASS'
    except (Failure, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        report['integrity'] = 'FAIL'
        report['failures'].append(f'{type(exc).__name__}: {exc}')
    sessions = [smoke.read_session(Path(p)) for p in (args.log or [])]
    report['timing'] = {'sessions': [{'log': s['log'], 'exit_code': s['exit_code'], 'resume': s['resume'],
                                      'seconds': {'total': smoke.span(s['marks']['start'], s['marks']['exit']),
                                                  'preflight': smoke.span(s['marks']['preflight_start'],
                                                                          s['marks']['preflight_end']),
                                                  'replay': smoke.span(s['marks']['replay_start'],
                                                                       s['marks']['replay_done']),
                                                  'rapfi_elapsed': smoke.span(s['marks']['rapfi_start'],
                                                                              s['marks']['rapfi_done']),
                                                  'revalidation_and_summary': smoke.span(s['marks']['rapfi_done'],
                                                                                         s['marks']['final'])},
                                      'untimed_lines': s['untimed_lines']} for s in sessions]}
    report['resume_test'] = smoke.resume_test(sessions)
    report['lock_test'] = smoke.lock_test(Path(args.lock_log) if args.lock_log else None, sessions)
    from hybrid.h7r_ledger import publish_json
    publish_json(output, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--approval', required=True, help='the committed full approval (execution-full.json)')
    parser.add_argument('--cache', default='runs/h7r/R1-HDEV-1/cache')
    parser.add_argument('--full-dir', default=FULL_DIR)
    parser.add_argument('--log', action='append', help='timestamped run log, one per invocation, oldest first')
    parser.add_argument('--lock-log', help='output of an optional lock probe')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        report = check(args)
    except Failure as exc:
        print(f'STOP: {exc}', file=sys.stderr, flush=True)
        return 1
    reading = report.get('reading') or {}
    print(f"[d4-full-check] integrity {report['integrity']}; reading {reading.get('joint')}; "
          f"games {report.get('games')}; observations {report.get('observations')}; branches {report.get('branches')}; "
          f"outcomes {report.get('branch_outcomes')}", flush=True)
    for name, c in (reading.get('comparisons') or {}).items():
        print(f"  {name:14s} label95 {c.get('label95')} individual {c.get('label_individual')} "
              f"bounds {c.get('mean_delta_bounds')} failures {c.get('failures')}", flush=True)
    for failure in report['failures']:
        print(f'  failure: {failure}', flush=True)
    print(f"  resume {report['resume_test']['status']}, lock {report['lock_test']['status']}", flush=True)
    return 0 if report['integrity'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
