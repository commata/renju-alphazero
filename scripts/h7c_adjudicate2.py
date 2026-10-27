"""H7C-NMASK-1 confirmatory adjudication (v2): disagreement observations of the confirmatory sample, judged by Rapfi
continuations as v1 does (``scripts/h7c_adjudicate.py``, unchanged), with the contract the decision fixes.

Measurement only (docs/h7-c1.md, "VALUE_ONLY_NETWORK_MASK 계약"): choice quality on fixed disagreement positions under
Rapfi continuations, not playing strength, not a minimax answer, never a promotion gate.

    inputs       the committed decision (``hybrid.h7c_nmask``), the candidate's training record, the confirmatory
                 sample file, the sensitivity summary and JSONL of that sample (``scripts/h7c_sensitivity.py
                 --sample-file``, model sets ``mask`` / ``c8`` / ``c5``, actor ``c5``), the c8 self-play: chained by
                 SHA-256; every record of the sample, every ply = the self-play's, the actor's move reproduced
    comparisons  the decision's (``mask:c8`` primary, ``mask:c5``, ``c8:c5``), each read on its own, never pooled;
                 A is the first set; observations and source games are never merged or deduplicated
    branches     v1's ``play_branch`` (a fresh registered X2 Rapfi per branch, ``Game`` referee, the same INVALID
                 reasons, ``abnormal_end`` INVALID); order ``Random(decision order seed)`` over the sorted keys
    statistics   v1's measures; the source-game cluster bootstrap with the decision's seed and repetitions; 95%
                 reported, the reading on the Bonferroni interval for three comparisons (1 - 0.05 / 3) with v1's
                 INCONCLUSIVE reasons; ``abnormal_end`` scored 0.5 as a separate sensitivity table (never a reading)

    python -u scripts/h7c_adjudicate2.py --rapfi-dir /mnt/c/engines/rapfi --workers 4 [--limit 8]

Resume: rerun the same command (v1's record rules). One executor per ``--jsonl`` (OS lock). A dirty tree is refused.
``--limit`` (smoke) writes under ``runs/h7c/nmask/adjudicate/smoke/`` and never reads.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from random import Random
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h7c_adjudicate as ad  # noqa: E402
from hybrid import h7c_nmask as nm  # noqa: E402
from hybrid.progress import Progress, run_chunks  # noqa: E402

FORMAT = 'h7c-adjudicate-v2'
DEFAULTS = {
    'candidate_value': ROOT / 'runs/h7c/nmask/value/best.pt',
    'summary': ROOT / 'docs/h7c/nmask/sensitivity_confirm.json',
    'sensitivity_jsonl': ROOT / 'runs/h7c/nmask/sensitivity/confirm.jsonl',
    'selfplay': ROOT / nm.PATHS['selfplay'], 'sample': ROOT / nm.RECORDS['sample'],
    'program': ROOT / nm.RECORDS['b2_program'],
    'jsonl': ROOT / 'runs/h7c/nmask/adjudicate/branches.jsonl',
    'output': ROOT / 'docs/h7c/nmask/adjudication.json', 'manifest_out': ROOT / 'docs/h7c/nmask/adjudication_manifest.json',
}
SMOKE_DIR = ROOT / 'runs/h7c/nmask/adjudicate/smoke'


# --- inputs ------------------------------------------------------------------------------------------------------

def load_inputs(decision: dict, candidate: dict, summary_path, sensitivity_path, selfplay_path, sample_path) -> dict:
    import h7c_labels as lab

    confirm = decision['evaluation']['confirm']
    summary = json.loads(Path(summary_path).read_text(encoding='utf-8'))
    identity = summary.get('identity') or {}
    shas = {'summary': ad.file_sha(summary_path), 'sensitivity_jsonl': ad.file_sha(sensitivity_path),
            'selfplay': ad.file_sha(selfplay_path), 'sample': ad.file_sha(sample_path)}
    sample = json.loads(Path(sample_path).read_text(encoding='utf-8'))
    nm.check_sample(sample)
    roles = {'parent': decision['parent'], 'control': decision['control'], 'candidate': candidate}
    expected_models = {name: roles[role] for name, role in confirm['models'].items()}
    pairs = sorted(c['pair'] for c in summary.get('comparisons', []))
    checks = {
        'sensitivity JSONL = the summary\'s': summary.get('jsonl_sha256') == shas['sensitivity_jsonl'],
        'self-play = the summary\'s = the decision\'s': identity.get('selfplay_jsonl_sha256') == shas['selfplay']
        == decision['source']['selfplay_jsonl_sha256'],
        'sample file = the decision\'s': shas['sample'] == confirm['sample_sha256']
        and (identity.get('sample_file') or {}).get('sha256') == shas['sample'],
        'sample = the listed games': identity.get('sample_sha256') == confirm['sample_ids_sha256']
        and identity.get('games_sample') == confirm['games'],
        'model sets = mask / c8 / c5': identity.get('models') == expected_models,
        'actor c5 reproduced': (summary.get('actor_check') or {}).get('model_set') == confirm['actor']
        and summary['actor_check'].get('reproduced') is True
        and (identity.get('selfplay') or {}).get('models') == decision['parent'],
        'comparisons = the decision\'s': pairs == sorted(f'{a}:{b}' for a, b in confirm['comparisons']),
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    nm.require(not failed, f'the confirmatory inputs do not chain ({", ".join(failed)})')
    records = [json.loads(line) for line in Path(sensitivity_path).read_text(encoding='utf-8').splitlines()
               if line.strip()]
    nm.require(all(r.get('identity') == identity for r in records), 'a record of another identity than the summary')
    games = {lab.game_id(g): g for g in lab.read_games([selfplay_path])}
    by_index = {g['game_index']: g for g in games.values()}
    listed = {by_index_of(games, i) for i in sample['game_ids']}
    indices = [r['game_index'] for r in records]
    nm.require(len(set(indices)) == len(indices) and set(indices) == listed,
               'the sensitivity records are not the confirmatory sample (missing, extra or duplicate games)')
    actor = confirm['actor']
    for r in records:
        expected_plies = by_index[r['game_index']]['plies']
        nm.require(len(r['plies']) == len(expected_plies), f"game {r['game_index']}: other plies than the self-play")
        for p, original in zip(r['plies'], expected_plies):
            nm.require((p['ply'], p['color'], p['actor_route'], p['actor_move']) == (
                original['ply'], original['color'], original['route'], original['engine_move'])
                and p['sets'][actor]['engine_move'] == original['engine_move'] and set(p['sets']) == set(
                expected_models), f"game {r['game_index']} ply {p['ply']}: not the self-play's position / actor move")
    obs = ad.observations(records, by_index, summary)
    counts = {}
    for o in obs:
        counts[o['compare']] = counts.get(o['compare'], 0) + 1
    recorded = {c['pair']: c['all_plies']['different_moves'] for c in summary['comparisons']}
    nm.require(all(counts.get(k, 0) == v for k, v in recorded.items()), f'recounted {counts} != the summary {recorded}')
    return {'summary': summary, 'records': records, 'observations': obs, 'shas': shas, 'counts': counts}


def by_index_of(games: dict, game_id: str) -> int:
    nm.require(game_id in games, f'{game_id} is not in the self-play file')
    return games[game_id]['game_index']


# --- statistics --------------------------------------------------------------------------------------------------

def bootstrap(groups: dict, statistic, rules: dict) -> dict:
    """v1's cluster bootstrap with the decision's seed, repetitions and levels; an undefined resample is counted."""
    ids = sorted(groups)
    rng = Random(rules['bootstrap']['seed'])
    values, undefined = [], 0
    for _ in range(rules['bootstrap']['samples']):
        v = statistic([o for _ in ids for o in groups[ids[rng.randrange(len(ids))]]])
        if v is None:
            undefined += 1
        else:
            values.append(v)
    if undefined or not values:
        return {'undefined_resamples': undefined, 'ci95': None, 'ci_reading': None}
    return {'undefined_resamples': 0, 'ci95': ad._quantiles(values, rules['report_level']),
            'ci_reading': ad._quantiles(values, rules['reading_level']), 'reading_level': rules['reading_level']}


def compare_stats(name: str, obs: list[dict], results: dict, rules: dict) -> dict:
    """v1's measures and reasons for one comparison, with the decision's bootstrap and reading level."""
    rows, invalid, missing = [], 0, 0
    for o in obs:
        ra, rb = (results.get(ad.branch_key(o['moves'], o[f'move_{s}'])) for s in ('a', 'b'))
        if ra is None or rb is None:
            missing += 1
        elif ra['status'] != 'VALID' or rb['status'] != 'VALID':
            invalid += 1
        else:
            rows.append({**o, 'score_a': ra['score'], 'score_b': rb['score'], 'delta': ra['score'] - rb['score']})
    n = len(obs)
    decisive = [r for r in rows if r['delta'] != 0]
    same = [r for r in rows if r['delta'] == 0]
    groups = {}
    for r in rows:
        groups.setdefault(r['game_index'], []).append(r)
    out = {'comparison': name, 'primary': name == rules['primary'],
           'candidate': obs[0]['a'] if obs else name.split(':')[0], 'reference': obs[0]['b'] if obs else
           name.split(':')[1], 'observations': n, 'source_games': len({o['game_index'] for o in obs}),
           'valid': len(rows), 'invalid': invalid, 'missing': missing,
           'invalid_share': round(invalid / n, 5) if n else None,
           'decisive': len(decisive), 'decisive_games': len({r['game_index'] for r in decisive}),
           'better': sum(r['delta'] > 0 for r in decisive), 'worse': sum(r['delta'] < 0 for r in decisive),
           'better_share': None if not decisive else round(ad._better_share(rows), 5),
           'same_result': {'share': round(len(same) / len(rows), 5) if rows else None,
                           'both_win': sum(r['score_a'] == 1 for r in same),
                           'both_draw': sum(r['score_a'] == 0.5 for r in same),
                           'both_loss': sum(r['score_a'] == 0 for r in same)},
           'half_point_deltas': sum(abs(r['delta']) == 0.5 for r in rows),
           'mean_delta': None if not rows else round(ad._mean_delta(rows), 5)}
    reasons = []
    if missing:
        reasons.append('INCOMPLETE')
    if n and invalid / n > rules['max_invalid_share']:
        reasons.append('INVALID_RATE')
    if not n or len(decisive) < rules['min_decisive'] or out['decisive_games'] < rules['min_decisive_games']:
        reasons.append('TOO_FEW_DECISIVE')
    out['better_share_bootstrap'] = None
    out['mean_delta_bootstrap'] = bootstrap(groups, ad._mean_delta, rules) if rows and not missing else None
    if not reasons:
        b = bootstrap(groups, ad._better_share, rules)
        out['better_share_bootstrap'] = b
        if b['undefined_resamples']:
            reasons.append('DEGENERATE_BOOTSTRAP')
    if reasons:
        out['reading'], out['reasons'] = 'INCONCLUSIVE', reasons
    else:
        lo, hi = out['better_share_bootstrap']['ci_reading']
        out['reading'] = 'BETTER' if lo > 0.5 else 'WORSE' if hi < 0.5 else 'INCONCLUSIVE'
        out['reasons'] = [] if out['reading'] != 'INCONCLUSIVE' else ['INTERVAL_CONTAINS_HALF']
    return out


def order(table: dict, seed: int) -> list[str]:
    keys = sorted(table)
    Random(seed).shuffle(keys)
    return keys


# --- the run -----------------------------------------------------------------------------------------------------

def run(args, log=print) -> dict:
    from h7c_arena import _git
    from h7c_loop import exclusive
    import h7c_nmask as cmd

    git = _git()
    nm.require(git['dirty'] is False or args.allow_dirty, 'the tree is dirty: the adjudication must name its commit')
    decision, dsha = nm.load_decision(args.decision, require_committed=not args.allow_uncommitted)
    confirm = decision['evaluation']['confirm']
    from analysis import tb_v3_net
    record = cmd.check_candidate(args.candidate_value, decision, dsha)
    candidate = tb_v3_net.model_identity(ROOT / nm.PATHS['policy'] if args.policy is None else args.policy,
                                         args.candidate_value)
    nm.require(candidate['value_checkpoint_sha256'] == record['checkpoint_sha256'], 'not the candidate value')
    inputs = load_inputs(decision, candidate, args.summary, args.sensitivity_jsonl, args.selfplay, args.sample)
    spec = ad.engine_spec(args.program, args.rapfi_dir, args.engine_command, args.allow_unregistered_engine)
    nm.require(spec['identity']['registered'] == {k: confirm['engine']['registered'][k] for k in (
        'executable_sha256', 'files', 'info')}, 'the program\'s engine is not the decision\'s')
    obs = inputs['observations']
    table = ad.branch_table(obs)
    keys = order(table, confirm['order_seed'])
    if args.limit is not None:
        keys = keys[:args.limit]
    identity = {'format': FORMAT, 'experiment': nm.EXPERIMENT, 'git': git, 'decision_sha256': dsha,
                'candidate': candidate, 'inputs': inputs['shas'], 'engine': spec['identity'],
                'order_seed': confirm['order_seed'], 'branches_sha256': ad.sha_json(sorted(table)),
                'limit': args.limit, 'max_plies': args.max_plies}
    identity_sha = ad.sha_json(identity)
    jsonl = Path(args.jsonl)
    with exclusive(jsonl.with_name(jsonl.name + '.lock')):
        finished = ad.load_finished(jsonl, identity_sha, set(keys))
        position = {k: i for i, k in enumerate(order(table, confirm['order_seed']))}
        todo = [{'key': k, 'order': position[k], 'moves': table[k]['moves'], 'move': table[k]['move'],
                 'spec': spec, 'identity_sha256': identity_sha, 'max_plies': args.max_plies}
                for k in keys if k not in finished]
        log(f'[h7c-adjudicate2] {len(obs)} observations ({inputs["counts"]}), {len(table)} unique branches'
            f'{"" if args.limit is None else f", limit {args.limit}"}; {len(finished)} done, {len(todo)} to play, '
            f'workers {args.workers}', flush=True)
        if todo:
            jsonl.parent.mkdir(parents=True, exist_ok=True)
            with jsonl.open('a', encoding='utf-8') as out, ProcessPoolExecutor(max_workers=args.workers) as pool:
                def keep(_, rows):
                    for row in rows:
                        out.write(json.dumps(row, separators=(',', ':')) + '\n')
                        finished[row['key']] = row
                    out.flush()
                run_chunks(pool, ad._chunk, [[t] for t in todo], Progress('h7c-adjudicate2', len(todo), 'branches'),
                           len, unit_name='branches', on_result=keep)
    names = [f'{a}:{b}' for a, b in confirm['comparisons']]
    stats = [compare_stats(n, [o for o in obs if o['compare'] == n], finished, confirm) for n in names]
    as_draw = {k: ({**r, 'status': 'VALID', 'score': 0.5} if r.get('reason') == 'abnormal_end' else r)
               for k, r in finished.items()}
    sensitivity = []
    for n in names:
        s = compare_stats(n, [o for o in obs if o['compare'] == n], as_draw, confirm)
        s['sensitivity_reading'], s['sensitivity_reasons'] = s.pop('reading'), s.pop('reasons')
        sensitivity.append(s)
    invalid = {}
    for r in finished.values():
        if r['status'] == 'INVALID':
            invalid[r['reason']] = invalid.get(r['reason'], 0) + 1
    result = {'format': f'{FORMAT}-summary', 'experiment': nm.EXPERIMENT, 'identity': identity,
              'identity_sha256': identity_sha, 'jsonl_sha256': ad.file_sha(jsonl) if jsonl.exists() else None,
              'meaning': 'choice quality on fixed disagreement positions of the confirmatory sample under Rapfi '
                         'continuations; each comparison read on its own; not playing strength, never a gate',
              'rules': {k: confirm[k] for k in ('bootstrap', 'report_level', 'reading_level', 'min_decisive',
                                                'min_decisive_games', 'max_invalid_share', 'primary')},
              'branches': {'unique': len(table), 'played': len(finished), 'invalid_by_reason': invalid},
              'overlap': ad.overlap(obs),
              'comparisons': stats,
              'sensitivity_abnormal_end_as_draw': {'not_a_reading': True, 'comparisons': sensitivity}}
    no_reading = (['LIMITED_RUN'] if args.limit is not None else []) + \
        ([] if spec['identity']['registered_engine'] else ['UNREGISTERED_ENGINE'])
    if no_reading:
        for s in result['comparisons']:
            s['reading'], s['reasons'] = 'SMOKE', no_reading
    return result


def resolve_paths(args) -> None:
    smoke = args.limit is not None
    for attr in ('jsonl', 'output', 'manifest_out'):
        if getattr(args, attr) is None:
            setattr(args, attr, SMOKE_DIR / f'limit{args.limit}_{DEFAULTS[attr].name}' if smoke else DEFAULTS[attr])
        elif smoke and Path(getattr(args, attr)).resolve() == DEFAULTS[attr].resolve():
            raise ValueError(f'--limit: {attr} must not be the full run\'s')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--decision', type=Path, default=ROOT / nm.RECORDS['decision'])
    parser.add_argument('--candidate-value', type=Path, default=DEFAULTS['candidate_value'])
    parser.add_argument('--policy', type=Path, help=f"default {nm.PATHS['policy']}")
    for key in ('summary', 'sensitivity_jsonl', 'selfplay', 'sample', 'program'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path, default=DEFAULTS[key])
    parser.add_argument('--rapfi-dir', type=Path, default=Path('/mnt/c/engines/rapfi'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, help='smoke: the first N branches of the order, no reading')
    for key in ('jsonl', 'output', 'manifest_out'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path)
    parser.add_argument('--engine-command', type=shlex.split, help='tests only')
    parser.add_argument('--allow-unregistered-engine', action='store_true', help='tests only')
    parser.add_argument('--max-plies', type=int, help='tests only')
    parser.add_argument('--allow-dirty', action='store_true', help='tests only')
    parser.add_argument('--allow-uncommitted', action='store_true', help='tests only')
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        resolve_paths(args)
    except ValueError as exc:
        parser.error(str(exc))
    for path in (args.output, args.manifest_out):
        if Path(path).exists():
            parser.error(f'{path} exists: never overwritten')
    try:
        result = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    manifest = {'format': f'{FORMAT}-manifest', 'identity_sha256': result['identity_sha256'],
                'code_git': result['identity']['git'], 'inputs_sha256': result['identity']['inputs'],
                'decision_sha256': result['identity']['decision_sha256'],
                'files_sha256': {ad._rel(args.jsonl): ad.file_sha(args.jsonl)} if Path(args.jsonl).exists() else {},
                'document_files_sha256': {ad._rel(args.output): ad.file_sha(args.output)}}
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    print(json.dumps([{k: s[k] for k in ('comparison', 'primary', 'observations', 'valid', 'invalid', 'decisive',
                                         'better_share', 'mean_delta', 'reading', 'reasons')}
                      for s in result['comparisons']], indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
