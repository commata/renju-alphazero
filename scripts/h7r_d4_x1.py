"""R2-D4-BASELINE-1-X1: post-hoc exploratory description of the D4 full records (read-only, CPU only).

Written after the D4 full result was seen: descriptive statistics of existing records, not a confirmatory test.
Nothing is searched, inferred on a GPU, played against Rapfi or trained; the D4 contract, list, approvals, runner,
records and reading are only read. The fixed plan (``x1-plan.json``) names every input by its full SHA-256.

Provenance is checked without pretending that the current HEAD is the run commit: the run identity must name the run
commit of the plan (clean), the contract / approval / list must be the bytes of that commit, and every reused source
file must be byte-identical between the run commit and this checkout. The X1 tool's own commit and sources are
recorded separately. The run journals are read with the strict reader of the full check (never the repairing resume
reader of the runner). Observations, branches, the c5 actor reproduction, branch outcomes and the reading are rebuilt
with the runner's / checker's own functions and must equal the summary before anything is described.

Analyses (plan ``analysis``): A ply bins x side, B concentration and repeated histories, C seed overlap on
(original game ID, ply), D within-model root Q of the two moves (visits > 0 only; list-missing, visits-0 and non-tree
cases counted, never set to 0), and quartile cells of within-model dQ and of |dv| of the H-dev root network values.
Output: the aggregate JSON under ``records/x1/`` (new file only); the per-observation rows under ``runs/``.

    python -u scripts/h7r_d4_x1.py --plan docs/h7r/r2/R2-D4-BASELINE-1/x1-plan.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
from random import Random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hybrid import h7r_d4 as d4  # noqa: E402
import h7r_d4_full_check as fc  # noqa: E402
import h7r_d4_smoke_check as smoke  # noqa: E402

FORMAT = 'h7r-d4-x1-summary-v1'
PLAN_FORMAT = 'h7r-d4-x1-plan-v1'
PLAN = 'docs/h7r/r2/R2-D4-BASELINE-1/x1-plan.json'
TOOL = 'scripts/h7r_d4_x1.py'
need, Failure = smoke.need, smoke.Failure

ANALYSIS = {
    'comparisons': list(d4.COMPARISONS),
    'seed_comparisons': list(d4.PRIMARY),
    'ply_bins': [[5, 20], [20, 40], [40, 80], [80, None]],
    'sides': ['black', 'white'],
    'outcomes': ['better', 'worse', 'equal', 'unresolved'],
    'top_games': 10,
    'top_order': 'decisive count descending, then original game_index ascending',
    'repetition': 'identical full move history before the observed ply',
    'prediction': 'd4',
    'q_decimals': 6,
    'q_validity': 'both moves in that model\'s root visit list with visits > 0 and route tree',
    'visit_share_denominator': 'sum of the positive visits of that model\'s root list',
    'quartiles': {'probabilities': [0.25, 0.5, 0.75],
                  'method': 'linear interpolation at position (n - 1) p of the sorted values',
                  'cells': ['x<=q25', 'q25<x<=q50', 'q50<x<=q75', 'x>q75'],
                  'ties': 'no tie splitting or jitter; duplicate boundaries and empty cells are reported',
                  'dq_population': 'valid disagreement observations of each comparison x model',
                  'dv_population': 'all eligible rows of each comparison'},
}
INPUTS = ('contract', 'approval', 'eligible', 'identity', 'summary', 'replay', 'branches', 'records_identity',
          'records_summary', 'check', 'check_records', 'sha256sums', 'records_manifest', 'log_manifest',
          'r2_contract', 'epsilon', 'predictions')
PREDICTION_MODELS = ('c5', 'c8', 'seed31001', 'seed31002', 'seed31003')
LIMITS = [
    'post-hoc exploratory description written after the D4 result was seen; not a confirmatory test',
    'descriptive statistics only: no correlation, p-value, per-bin interval or new label',
    'root Q is a search estimate in the root mover\'s perspective, not ground truth and not proof of a cause',
    'Q values are recorded to 6 decimals: differences below that precision are not read',
    'the prediction is the D4-averaged root-position network value of the H-dev cache, not a record of the leaf '
    'values the search used',
    'shared branches make seed outcomes dependent: agreement through a shared branch is not an independent reproduction',
    'the D4 reading (NOT_ESTABLISHED), c5 kept and V9 on hold are unchanged',
]


# --- plan and provenance --------------------------------------------------------------------------------------------

def git_blob_sha256(root: Path, commit: str, path: str) -> str:
    """SHA-256 of a file's bytes at ``commit`` (never the working tree)."""
    import hashlib
    out = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=root, capture_output=True, check=False)
    need(out.returncode == 0, f'{path} is not in the run commit {commit[:12]}')
    return hashlib.sha256(out.stdout).hexdigest()


def git_is_ancestor(root: Path, commit: str) -> bool:
    return subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=root,
                          capture_output=True).returncode == 0


def load_plan(path, root: Path) -> tuple[dict, str]:
    from hybrid.h7c_nmask import committed

    path = Path(path)
    need(path.is_file(), f'{path}: no X1 plan')
    plan = fc.strict_json(path.read_bytes())
    need(isinstance(plan, dict) and plan.get('format') == PLAN_FORMAT and plan.get('status') == 'fixed'
         and plan.get('nature') == 'post_hoc_exploratory', 'not the fixed post-hoc X1 plan')
    need(plan.get('analysis') == ANALYSIS, 'the plan analysis differs from the tool (neither may drift)')
    need(set(plan.get('inputs') or {}) == set(INPUTS), 'the plan input list differs from the tool')
    for key, ref in plan['inputs'].items():
        need(isinstance(ref, dict) and isinstance(ref.get('path'), str) and d4.is_sha(ref.get('sha256')),
             f'plan input {key}: a path and a full SHA-256 are required')
    need(isinstance(plan.get('run_commit'), str) and len(plan['run_commit']) == 40, 'the run commit must be full')
    need(committed(path, root), 'the X1 plan is not committed')
    return plan, d4.file_sha256(path)


def input_shas(plan: dict, root: Path) -> dict:
    """SHA-256 of every plan input, equal to the plan; an absent input stops the analysis (it is never narrowed)."""
    absent = [plan['inputs'][k]['path'] for k in INPUTS if not (root / plan['inputs'][k]['path']).is_file()]
    need(not absent, f'inputs absent: {absent} (the analysis stops; it is never narrowed to the inputs present)')
    out = {}
    for key in INPUTS:
        out[key] = d4.file_sha256(root / plan['inputs'][key]['path'])
        need(out[key] == plan['inputs'][key]['sha256'], f"{plan['inputs'][key]['path']}: SHA-256 differs from the plan")
    return out


def check_run_links(plan: dict, root: Path) -> dict:
    """The run commit is in this history; contract, approval, list and reused sources are its bytes."""
    commit = plan['run_commit']
    need(git_is_ancestor(root, commit), f'the run commit {commit[:12]} is not an ancestor of HEAD')
    links = {}
    for key in ('contract', 'approval', 'eligible'):
        ref = plan['inputs'][key]
        need(git_blob_sha256(root, commit, ref['path']) == ref['sha256'],
             f"{ref['path']}: not the bytes of the run commit")
        links[key] = ref['sha256']
    sources = plan.get('run_sources') or {}
    need(sources, 'the plan names no reused source')
    for path, sha in sorted(sources.items()):
        need(d4.is_sha(sha) and (root / path).is_file() and d4.file_sha256(root / path) == sha,
             f'{path}: the checkout differs from the plan')
        need(git_blob_sha256(root, commit, path) == sha, f'{path}: the run commit has another version')
    return {'run_commit': commit, 'run_commit_is_ancestor': True, 'bound_inputs': links,
            'run_sources': dict(sorted(sources.items()))}


def check_records(plan: dict, root: Path, shas: dict) -> dict:
    """The committed records of the full run name exactly the pinned raw files and passed integrity."""
    inp = {k: root / v['path'] for k, v in plan['inputs'].items()}
    need(inp['check'].read_bytes() == inp['check_records'].read_bytes(),
         'check.json and check-records.json differ')
    check = fc.strict_json(inp['check'].read_bytes())
    raw = {'identity.json': shas['identity'], 'summary.json': shas['summary'], 'replay.jsonl': shas['replay'],
           'branches.jsonl': shas['branches']}
    need(check.get('format') == fc.FORMAT and check.get('integrity') == 'PASS' and not check.get('failures'),
         'the full check did not pass')
    need(check.get('artifacts') == raw, 'the full check names other raw files')
    need(check.get('shas') == {'contract': shas['contract'], 'approval': shas['approval'], 'eligible': shas['eligible']},
         'the full check names another contract, approval or list')
    sums = {}
    for line in inp['sha256sums'].read_text(encoding='utf-8').splitlines():
        sha, name = line.split(None, 1)
        need(name.startswith('./') and name[2:] not in sums, 'SHA256SUMS is malformed')
        sums[name[2:]] = sha
    need(sums == raw, 'SHA256SUMS differs from the pinned raw files')
    need(shas['records_identity'] == shas['identity'] and shas['records_summary'] == shas['summary'],
         'the committed identity / summary copies differ from the raw files')
    manifest = fc.strict_json(inp['records_manifest'].read_bytes())
    folder = inp['records_manifest'].parent
    listed = {f['path']: f['sha256'] for f in manifest.get('files') or []}
    present = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()} - {'records_manifest.json'}
    need(set(listed) == present, 'the records manifest does not list exactly the record files')
    need(all(d4.file_sha256(folder / k) == v for k, v in listed.items()), 'a record file differs from its manifest')
    return {'check_integrity': check['integrity'], 'check_reading': (check.get('reading') or {}).get('joint'),
            'raw': raw, 'records_files': len(listed)}


def check_prediction_source(plan: dict, root: Path, shas: dict, contract: dict) -> dict:
    """seed_predictions.pt is the one R2-0B wrote (epsilon record), for the models and H-dev of the D4 contract."""
    eps = fc.strict_json((root / plan['inputs']['epsilon']['path']).read_bytes())
    r2c = fc.strict_json((root / plan['inputs']['r2_contract']['path']).read_bytes())
    need(eps.get('stage') == 'R2-0B' and eps.get('contract_sha256') == shas['r2_contract'],
         'the epsilon record is not the R2-0B record of the R2 contract')
    need(eps.get('predictions') == {'path': plan['inputs']['predictions']['path'], 'sha256': shas['predictions']},
         'the epsilon record names other predictions')
    for key in ('epsilon', 'r2_contract'):  # both pinned by the D4 contract's sources
        need({'path': plan['inputs'][key]['path'], 'sha256': shas[key]} in contract['sources'],
             f"{plan['inputs'][key]['path']}: not the version the D4 contract pins")
    need(set(eps['hdev']) <= set(contract['hdev']) and all(contract['hdev'][k] == v for k, v in eps['hdev'].items()),
         'the predictions were made on another H-dev')
    for name in ('c5', 'c8'):
        need(r2c['models'][name]['value_checkpoint_sha256'] == contract['models'][name]['value_checkpoint_sha256'],
             f'{name}: the R2 and D4 contracts name other models')
    for seed in ('31001', '31002', '31003'):
        got = eps['seeds'][seed]['identity']
        spec = contract['models'][f'seed{seed}']
        need(all(got.get(k) == spec[k] for k in ('value_checkpoint_sha256', 'value_metadata_sha256')),
             f'seed{seed}: the predictions were made with another model')
    return {'epsilon_sha256': shas['epsilon'], 'predictions_sha256': shas['predictions'],
            'r2_contract_sha256': shas['r2_contract'], 'epsilon_git': eps.get('git')}


def check_identity(identity: dict, summary: dict, contract: dict, shas: dict, run_commit: str) -> str:
    """The full checker's identity checks, with the run commit of the plan in place of the current HEAD."""
    identity_sha = d4.digest(identity)
    need(summary.get('format') == 'h7r-d4-summary-v1' and summary.get('reading') != 'SMOKE', 'not a full summary')
    need(summary.get('identity') == identity and summary.get('identity_sha256') == identity_sha,
         'the summary is not bound to identity.json')
    need(identity.get('format') == 'h7r-d4-run-v1' and identity.get('mode') == 'full'
         and identity.get('contract_sha256') == shas['contract'] and identity.get('approval_sha256') == shas['approval']
         and identity.get('sample_sha256') == shas['eligible'] and identity.get('device') == contract['device'],
         'the run identity names another contract, approval, list, mode or device')
    need(identity.get('runtime') == contract['runtime_identity'] and identity.get('rapfi') == contract['rapfi_identity'],
         'the run identity names another engine runtime or Rapfi')
    need(set(identity.get('models') or {}) == set(contract['models']), 'model coverage differs from the contract')
    for name, spec in contract['models'].items():
        got = identity['models'][name]
        need(isinstance(got, dict) and isinstance(got.get('frozen_state_sha256'), dict)
             and all(got.get(k) == spec[k] for k in ('value_checkpoint_sha256', 'value_metadata_sha256',
                                                   'policy_checkpoint_sha256', 'policy_metadata_sha256'))
             and all(got['frozen_state_sha256'].get(k) == contract['frozen_state_sha256']
                     for k in ('recomputed', 'declared', 'contract')),
             f'the run identity of model {name} differs from the contract')
    need(identity.get('git') == {'commit': run_commit, 'dirty': False},
         f"the run identity names commit {str((identity.get('git') or {}).get('commit'))[:12]}, "
         f'not the clean run commit {run_commit[:12]} of the plan')
    files = summary.get('files') or {}
    need(files == {'replay.jsonl': shas['replay'], 'branches.jsonl': shas['branches']},
         'the summary names other journals')
    return identity_sha


# --- reconstruction (the runner's and the full checker's functions) --------------------------------------------------

def reconstruct(plan: dict, contract: dict, shas: dict, cache, root: Path, log=print) -> dict:
    import h7c_adjudicate as ad
    import h7c_sensitivity as sn
    import h7r_d4 as cli
    from hybrid.progress import Progress

    inp = {k: root / v['path'] for k, v in plan['inputs'].items()}
    identity = fc.strict_json(inp['identity'].read_bytes())
    summary = fc.strict_json(inp['summary'].read_bytes())
    identity_sha = check_identity(identity, summary, contract, shas, plan['run_commit'])
    manifest, games, hdev = cli.inputs(contract, shas['contract'], cache, root)
    sample = fc.strict_json(inp['eligible'].read_bytes())
    need(manifest == sample, 'the committed eligible list differs from the verified H-dev source')
    need(hdev == identity.get('hdev'), 'the H-dev checks differ from the run identity')
    replay = fc.strict_journal(inp['replay'], 'game_index', identity_sha, set(games))
    models = {name: {} for name in contract['models']}
    records = [replay[i] for i in sorted(games)]
    progress = Progress('d4-x1-replay-check', len(records), 'games', log=log)
    for record in records:
        sn.check_replay(record, games, models, 'c5')  # coverage, legality, the c5 actor reproduced
        progress.update()
    filtered = cli.eligible_replays(records, manifest)
    need(sum(len(r['plies']) for r in filtered) == manifest['rows'], 'eligible plies differ from the list')
    obs = cli.observations_for(filtered, games)
    table = ad.branch_table(obs)
    keys = sorted(table)
    Random(contract['stats']['order_seed']).shuffle(keys)
    tasks = {k: {**table[k], 'order': i, 'identity_sha256': identity_sha} for i, k in enumerate(keys)}
    branches = fc.strict_journal(inp['branches'], 'key', identity_sha, set(keys)) if keys else {}
    outcomes = {}
    progress = Progress('d4-x1-branch-check', len(branches), 'branches', log=log)
    for key, record in branches.items():
        outcomes[key] = cli.validate_branch(record, tasks[key], contract['invalid_policy'])
        if outcomes[key]['kind'] == 'structural':
            need(record.get('status') == 'INVALID' and record.get('reason') == 'abnormal_end',
                 f'branch {key[:12]}: structural draw without its original INVALID mark')
        progress.update()
    log('[d4-x1] recomputing the summary (bootstrap of the runner)', flush=True)
    actual = fc.recompute(contract, games, records, filtered, obs, table, outcomes, cli, sn, ad)
    mismatches = [m for key in fc.RECOMPUTED for m in fc.differences(summary.get(key), actual[key], key)]
    need(not mismatches, f'the summary differs from the recomputation: {mismatches[:5]}')
    need(actual['actor_check']['reproduced'], 'the c5 actor was not reproduced')
    return {'identity_sha256': identity_sha, 'manifest': manifest, 'games': games, 'records': records,
            'filtered': filtered, 'observations': obs, 'outcomes': outcomes, 'summary': summary, 'actual': actual}


# --- predictions --------------------------------------------------------------------------------------------------

def hdev_rows(contract: dict, cache, root: Path) -> tuple[dict, list]:
    """The verified H-dev rows (dense game index, original game IDs, plies) and their routes."""
    import h7r_holdout as ho
    import h7r_r2 as r2cli
    _, data, info, _ = r2cli.check_hdev(contract, Path(cache), root)
    return info, ho.routes(data)


def load_predictions(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def checked_predictions(pred, info: dict, routes: list, manifest: dict, r2_sha: str) -> tuple[dict, dict]:
    """Prediction rows joined 1:1 to the eligible rows on (original game ID, ply) through the cache's dense index."""
    import torch

    need(isinstance(pred, dict) and set(pred) == {'contract_sha256', 'game', 'ply', 'route', 'models'},
         'seed_predictions.pt has another layout')
    need(pred['contract_sha256'] == r2_sha, 'the predictions belong to another R2 contract')
    n = len(info['ply'])
    need(n == manifest['rows'] and len(info['game']) == n, 'the H-dev rows differ from the eligible list')
    need(set(pred['models']) == set(PREDICTION_MODELS), f"prediction models {sorted(pred['models'])} differ")
    for name, kinds in pred['models'].items():
        need(isinstance(kinds, dict) and set(kinds) == {'raw', 'd4'}, f'{name}: prediction kinds differ')
        for kind, t in kinds.items():
            need(isinstance(t, torch.Tensor) and t.dtype == torch.float64 and t.dim() == 1 and t.numel() == n,
                 f'{name} {kind}: {getattr(t, "shape", None)} is not {n} float64 values')
            need(bool(torch.isfinite(t).all()), f'{name} {kind}: non-finite prediction')
    for key in ('game', 'ply'):
        t = pred[key]
        need(isinstance(t, torch.Tensor) and t.dim() == 1 and t.numel() == n and not t.is_floating_point(),
             f'prediction {key}: not {n} integers')
    need(isinstance(pred['route'], list) and len(pred['route']) == n
         and all(type(r) is bool for r in pred['route']), f'prediction route: not {n} booleans')
    game, ply = pred['game'].tolist(), pred['ply'].tolist()
    need(all(0 <= g < len(info['game_ids']) for g in game), 'a prediction row has no cache game')
    need(game == list(info['game']) and ply == list(info['ply']),
         'prediction rows are missing, repeated, extra or out of the cache order')
    need(pred['route'] == [r == 'network' for r in routes], 'prediction routes differ from the H-dev rows')
    keys = [(info['game_ids'][g], p) for g, p in zip(game, ply)]
    need(len(set(keys)) == n, 'a (game ID, ply) row is predicted twice')
    eligible = {(e['game_id'], p) for e in manifest['games'] for p in e['plies']}
    need(set(keys) == eligible, 'prediction rows and eligible rows differ: '
                                f'{len(set(keys) - eligible)} extra, {len(eligible - set(keys))} missing')
    d4v = {name: pred['models'][name][ANALYSIS['prediction']].tolist() for name in PREDICTION_MODELS}
    values = {k: {name: d4v[name][i] for name in PREDICTION_MODELS} for i, k in enumerate(keys)}
    return values, {'rows': n, 'network_rows': sum(pred['route']), 'models': list(PREDICTION_MODELS),
                    'prediction': ANALYSIS['prediction'], 'join': '(original game ID, ply) through the cache dense index'}


# --- descriptive analyses ----------------------------------------------------------------------------------------

def outcome_rows(obs: list[dict], outcomes: dict, game_ids: dict) -> list[dict]:
    """One row per disagreement observation: the D4 outcome class (unresolved branches kept as their own class)."""
    rows = []
    for o in obs:
        a, b = outcomes[o['key_a']], outcomes[o['key_b']]
        if 'unresolved' in (a['kind'], b['kind']):
            outcome, delta = 'unresolved', None
        else:
            delta = a['score'] - b['score']
            outcome = 'better' if delta > 0 else 'worse' if delta < 0 else 'equal'
        rows.append({'compare': o['compare'], 'model': o['a'], 'game_index': o['game_index'],
                     'game_id': game_ids[o['game_index']], 'ply': o['ply'], 'color': o['color'],
                     'history': tuple(tuple(m) for m in o['moves']), 'move_a': list(o['move_a']),
                     'move_b': list(o['move_b']), 'key_a': o['key_a'], 'key_b': o['key_b'],
                     'outcome': outcome, 'delta': delta})
    return rows


def tally(rows: list[dict]) -> dict:
    out = {k: 0 for k in ANALYSIS['outcomes']}
    deltas = []
    for r in rows:
        out[r['outcome']] += 1
        if r['delta'] is not None:
            deltas.append(r['delta'])
    out.update(observations=len(rows), decisive=out['better'] + out['worse'], resolved=len(deltas),
               delta_sum=sum(deltas), mean_delta=sum(deltas) / len(deltas) if deltas else None)
    return out


def ply_bin(ply: int) -> str:
    for lo, hi in ANALYSIS['ply_bins']:
        if ply >= lo and (hi is None or ply < hi):
            return f"[{lo},{'inf' if hi is None else hi})"
    raise Failure(f'ply {ply} is in no bin')


def ply_side(eligible: list[tuple], rows: list[dict]) -> dict:
    """A: eligible denominator, disagreements and outcomes per ply bin x side, each comparison on its own."""
    bins = [ply_bin(lo) for lo, _ in ANALYSIS['ply_bins']]
    den = Counter((ply_bin(p), c) for _, p, c in eligible)
    out = {}
    for compare in ANALYSIS['comparisons']:
        mine = [r for r in rows if r['compare'] == compare]
        cells = {}
        for b in bins:
            for side in ANALYSIS['sides']:
                part = [r for r in mine if ply_bin(r['ply']) == b and r['color'] == side]
                e = den[(b, side)]
                cells[f'{b}|{side}'] = {'eligible': e, 'disagreements': len(part),
                                        'rate': len(part) / e if e else None, **tally(part)}
        total = {'eligible': len(eligible), 'disagreements': len(mine),
                 'rate': len(mine) / len(eligible) if eligible else None, **tally(mine)}
        for k in ('eligible', 'disagreements', *ANALYSIS['outcomes'], 'decisive', 'resolved'):
            need(sum(c[k] for c in cells.values()) == total[k], f'{compare}: ply x side cells do not add up ({k})')
        out[compare] = {'cells': cells, 'total': total}
    return out


def concentration(rows: list[dict]) -> dict:
    """B: decisive observations per game, the top games by count, and repeated histories."""
    out = {}
    for compare in ANALYSIS['comparisons']:
        mine = [r for r in rows if r['compare'] == compare]
        games = {}
        for r in mine:
            g = games.setdefault(r['game_index'], {'game_index': r['game_index'], 'game_id': r['game_id'],
                                                   'better': 0, 'worse': 0, 'delta_sum': 0.0})
            g['better'] += r['outcome'] == 'better'
            g['worse'] += r['outcome'] == 'worse'
            g['delta_sum'] += r['delta'] or 0.0
        for g in games.values():
            g['decisive'], g['net'] = g['better'] + g['worse'], g['better'] - g['worse']
        decisive = [g for g in games.values() if g['decisive'] > 0]
        top = sorted(decisive, key=lambda g: (-g['decisive'], g['game_index']))[:ANALYSIS['top_games']]
        total = tally(mine)

        def block(gs):
            d = sum(g['decisive'] for g in gs)
            return {'games': len(gs), 'decisive': d, 'better': sum(g['better'] for g in gs),
                    'worse': sum(g['worse'] for g in gs), 'net': sum(g['net'] for g in gs),
                    'delta_sum': sum(g['delta_sum'] for g in gs)}
        top_block = block(top)
        top_block['decisive_share'] = top_block['decisive'] / total['decisive'] if total['decisive'] else None
        histories = Counter(r['history'] for r in mine)
        repeated = [r for r in mine if histories[r['history']] > 1]
        same_pair = Counter((r['history'], tuple(r['move_a'])) for r in mine)
        out[compare] = {
            'observations': len(mine), 'games_with_observations': len(games), 'decisive_games': len(decisive),
            'overall': {'decisive': total['decisive'], 'better': total['better'], 'worse': total['worse'],
                        'net': total['better'] - total['worse'], 'delta_sum': total['delta_sum']},
            'per_game_max': {'decisive': max((g['decisive'] for g in games.values()), default=0),
                             'better': max((g['better'] for g in games.values()), default=0),
                             'worse': max((g['worse'] for g in games.values()), default=0),
                             'abs_net': max((abs(g['net']) for g in games.values()), default=0)},
            'top_games': {**top_block, 'list': [{k: g[k] for k in ('game_index', 'game_id', 'decisive', 'better',
                                                                     'worse', 'net', 'delta_sum')} for g in top]},
            'repeated_history': {'distinct_histories': len(histories),
                                 'repeated_histories': sum(n > 1 for n in histories.values()),
                                 'observations': len(repeated), 'decisive': tally(repeated)['decisive'],
                                 'max_repeats': max(histories.values(), default=0),
                                 'observations_with_identical_branch_pair': sum(n for n in same_pair.values() if n > 1)}}
    return out


def seed_overlap(rows: list[dict]) -> dict:
    """C: overlap of the seed comparisons on (original game ID, ply); shared branches are not independent."""
    seeds = ANALYSIS['seed_comparisons']
    by = {p: {} for p in seeds}
    for r in rows:
        if r['compare'] in by:
            k = (r['game_id'], r['ply'])
            need(k not in by[r['compare']], f"{r['compare']}: position {k} observed twice")
            by[r['compare']][k] = r
    pairs = {}
    for a, b in combinations(seeds, 2):
        common = sorted(set(by[a]) & set(by[b]))
        for k in common:
            need(by[a][k]['key_b'] == by[b][k]['key_b'], f'{k}: the c5 branch differs between comparisons')
        same = [k for k in common if by[a][k]['move_a'] == by[b][k]['move_a']]
        diff = [k for k in common if by[a][k]['move_a'] != by[b][k]['move_a']]
        agree_same = sum(by[a][k]['outcome'] == by[b][k]['outcome'] for k in same)
        need(agree_same == len(same), 'the same branch pair gave two outcomes')
        table = Counter(f"{by[a][k]['outcome']}|{by[b][k]['outcome']}" for k in diff)
        pairs[f'{a}|{b}'] = {
            'disagreements': {a: len(by[a]), b: len(by[b])}, 'common_positions': len(common),
            'same_candidate_move': len(same), 'different_candidate_move': len(diff),
            'shared_branches': {'c5_branch': len(common), 'candidate_branch': len(same)},
            'outcome_agreement': {
                'same_candidate_move': {'positions': len(same), 'agree': agree_same,
                                        'note': 'both branches shared: agreement is automatic, not a reproduction'},
                'different_candidate_move': {'positions': len(diff),
                                             'agree': sum(by[a][k]['outcome'] == by[b][k]['outcome'] for k in diff),
                                             'table': dict(sorted(table.items())),
                                             'note': 'the c5 branch is shared, the candidate branches differ'}}}
    union = set().union(*(set(v) for v in by.values()))
    count = Counter(sum(k in by[p] for p in seeds) for k in union)
    all_three = [k for k in union if all(k in by[p] for p in seeds)]
    return {'pairs': pairs,
            'union': {'positions': len(union), 'by_seed_count': {str(n): count.get(n, 0) for n in (1, 2, 3)},
                      'all_three_same_candidate_move': sum(len({tuple(by[p][k]['move_a']) for p in seeds}) == 1
                                                           for k in all_three),
                      'candidate_branches': len({by[p][k]['key_a'] for p in seeds for k in by[p]}),
                      'c5_branches': len({by[p][k]['key_b'] for p in seeds for k in by[p]})}}


def root_entry(visits: list, move) -> list | None:
    return next((v for v in visits if [int(v[0]), int(v[1])] == list(move)), None)


def within_model_q(record: dict, cand, c5) -> dict:
    """Q(candidate move) - Q(c5 move) inside one model's root list (the root mover's perspective; no colour flip)."""
    if record['route'] != 'tree':
        return {'status': 'non_tree'}
    a, b = root_entry(record['visits'], cand), root_entry(record['visits'], c5)
    if a is None or b is None:
        return {'status': 'missing', 'which': 'both' if a is None and b is None else
                ('candidate_move' if a is None else 'c5_move')}
    if a[2] <= 0 or b[2] <= 0:  # PUCTNode.mean_value is 0.0 with no visit: never a value
        return {'status': 'visits_zero', 'which': 'both' if a[2] <= 0 and b[2] <= 0 else
                ('candidate_move' if a[2] <= 0 else 'c5_move')}
    total = sum(v[2] for v in record['visits'] if v[2] > 0)
    d = ANALYSIS['q_decimals']
    return {'status': 'valid', 'dq': round(a[3] - b[3], d), 'q_candidate': a[3], 'q_c5': b[3],
            'share_diff': (a[2] - b[2]) / total}


def own_q(record: dict, move) -> float | None:
    if record['route'] != 'tree':
        return None
    e = root_entry(record['visits'], move)
    return e[3] if e is not None and e[2] > 0 else None


def quantiles(values: list[float]) -> list[float] | None:
    v = sorted(values)
    if not v:
        return None
    out = []
    for p in ANALYSIS['quartiles']['probabilities']:
        h = (len(v) - 1) * p
        lo = math.floor(h)
        hi = min(lo + 1, len(v) - 1)
        out.append(v[lo] + (h - lo) * (v[hi] - v[lo]))
    return out


def cell_of(x: float, b: list[float]) -> int:
    return 0 if x <= b[0] else 1 if x <= b[1] else 2 if x <= b[2] else 3


def boundaries(b: list[float] | None) -> dict:
    names = ('q25', 'q50', 'q75')
    if b is None:
        return {**{n: None for n in names}, 'duplicates': []}
    return {**dict(zip(names, b)), 'duplicates': [f'{names[i]}={names[i + 1]}' for i in range(2) if b[i] == b[i + 1]]}


def distribution(values: list[float]) -> dict:
    b = quantiles(values)
    return {'count': len(values), 'min': min(values) if values else None, 'max': max(values) if values else None,
            'mean': sum(values) / len(values) if values else None, **boundaries(b)}


def quartile_cells(items: list[tuple[float, dict]]) -> dict:
    """Cells of (value, observation row) pairs by the quartiles of the values; empty cells and ties reported."""
    b = quantiles([x for x, _ in items])
    cells = [[] for _ in ANALYSIS['quartiles']['cells']]
    if b is not None:
        for x, r in items:
            cells[cell_of(x, b)].append(r)
    names = ANALYSIS['quartiles']['cells']
    return {'boundaries': boundaries(b), 'values': len(items),
            'cells': {n: {'n': len(c), **tally(c)} for n, c in zip(names, cells)},
            'empty_cells': [n for n, c in zip(names, cells) if not c]}


def q_analysis(rows: list[dict], plies: dict) -> dict:
    """D: within-model dQ of both models of each comparison, coverage and exclusions, and the dQ quartile cells."""
    out = {}
    for compare in ANALYSIS['comparisons']:
        cand_model = compare.split(':')[0]
        mine = [r for r in rows if r['compare'] == compare]
        per = {}
        for role, model in (('candidate', cand_model), ('c5', 'c5')):
            valid, excluded = [], Counter()
            for r in mine:
                res = within_model_q(plies[(r['game_index'], r['ply'])]['sets'][model], r['move_a'], r['move_b'])
                r.setdefault('q', {})[role] = res
                if res['status'] == 'valid':
                    valid.append((res['dq'], r))
                else:
                    excluded[res['status'] + (f":{res['which']}" if 'which' in res else '')] += 1
            need(len(valid) + sum(excluded.values()) == len(mine), f'{compare} {role}: coverage does not add up')
            dqs = [x for x, _ in valid]
            per[role] = {'model': model, 'observations': len(mine), 'valid': len(valid),
                         'excluded': dict(sorted(excluded.items())),
                         'dq': {**distribution(dqs), 'positive': sum(x > 0 for x in dqs),
                                'zero': sum(x == 0 for x in dqs), 'negative': sum(x < 0 for x in dqs)},
                         'visit_share_diff': distribution([r['q'][role]['share_diff'] for _, r in valid]),
                         'dq_quartiles': quartile_cells(valid)}
        cross = []
        for r in mine:
            rec = plies[(r['game_index'], r['ply'])]['sets']
            qa, qb = own_q(rec[cand_model], r['move_a']), own_q(rec['c5'], r['move_b'])
            if qa is not None and qb is not None:
                cross.append(round(qa - qb, ANALYSIS['q_decimals']))
        per['cross_model_q_diff'] = {**distribution(cross), 'excluded': len(mine) - len(cross),
                                     'note': 'models differ in value calibration: distribution only'}
        out[compare] = per
    return out


def value_change(eligible: list[tuple], rows: list[dict], values: dict, game_ids: dict) -> dict:
    """|dv| = |v_candidate - v_c5| of the root network values: quartiles over all eligible rows of a comparison."""
    out = {}
    for compare in ANALYSIS['comparisons']:
        cand_model = compare.split(':')[0]
        mine = {(r['game_id'], r['ply']): r for r in rows if r['compare'] == compare}
        items = []
        for gi, ply, _ in eligible:
            k = (game_ids[gi], ply)
            v = values[k]
            items.append((abs(v[cand_model] - v['c5']), k))
        b = quantiles([x for x, _ in items])
        names = ANALYSIS['quartiles']['cells']
        cells = [{'eligible': 0, 'obs': []} for _ in names]
        for x, k in items:
            c = cells[cell_of(x, b)]
            c['eligible'] += 1
            if k in mine:
                c['obs'].append(mine[k])
                mine[k]['dv_abs'] = x
        need(sum(len(c['obs']) for c in cells) == len(mine), f'{compare}: an observation has no prediction row')
        out[compare] = {
            'boundaries': boundaries(b), 'values': len(items),
            'cells': {n: {'eligible': c['eligible'], 'disagreements': len(c['obs']),
                          'rate': len(c['obs']) / c['eligible'] if c['eligible'] else None, **tally(c['obs'])}
                      for n, c in zip(names, cells)},
            'empty_cells': [n for n, c in zip(names, cells) if not c['eligible']],
            'agree': distribution([x for x, k in items if k not in mine]),
            'disagree': distribution([x for x, k in items if k in mine])}
    return out


def check_against_summary(rows: list[dict], summary: dict) -> None:
    for compare in ANALYSIS['comparisons']:
        t, s = tally([r for r in rows if r['compare'] == compare]), summary['comparisons'][compare]
        need((t['observations'], t['better'], t['worse'], t['equal'], t['decisive'])
             == (s['observations'], s['better'], s['worse'], s['equal'], s['decisive'])
             and t['unresolved'] == s['unresolved'], f'{compare}: the outcome rows differ from the summary')


def analyse(rec: dict, values: dict) -> tuple[dict, list[dict]]:
    game_ids = {e['game_index']: e['game_id'] for e in rec['manifest']['games']}
    eligible = [(r['game_index'], p['ply'], p['color']) for r in rec['filtered'] for p in r['plies']]
    plies = {(r['game_index'], p['ply']): p for r in rec['filtered'] for p in r['plies']}
    rows = outcome_rows(rec['observations'], rec['outcomes'], game_ids)
    check_against_summary(rows, rec['summary'])
    conc = concentration(rows)
    for compare, c in conc.items():
        need(c['decisive_games'] == rec['summary']['comparisons'][compare]['decisive_games'],
             f'{compare}: decisive games differ from the summary')
    result = {'ply_side': ply_side(eligible, rows), 'concentration': conc, 'seed_overlap': seed_overlap(rows),
              'within_model_q': q_analysis(rows, plies), 'value_change': value_change(eligible, rows, values, game_ids)}
    return result, rows


# --- the command -------------------------------------------------------------------------------------------------

def public_row(r: dict) -> dict:
    return {k: r[k] for k in ('compare', 'game_index', 'game_id', 'ply', 'color', 'move_a', 'move_b', 'key_a',
                              'key_b', 'outcome', 'delta')} | {'q': r.get('q'), 'dv_abs': r.get('dv_abs')}


def run(args, root=ROOT, log=print) -> dict:
    import h7r_d4 as cli
    from hybrid.h7c_nmask import committed
    from hybrid.h7r_ledger import publish_json

    started = time.monotonic()
    root = Path(root)
    need(Path(args.plan).resolve() == (root / PLAN).resolve(), f'the X1 plan is {PLAN}')
    plan, plan_sha = load_plan(args.plan, root)
    out_summary, out_rows = root / plan['output']['summary'], root / plan['output']['rows']
    need(not out_summary.exists() and not out_rows.exists(), 'an X1 output exists: never overwritten')
    need(out_rows.resolve().is_relative_to((root / 'runs').resolve()), 'per-observation rows go under runs/')
    state = cli.git_state(root)  # the X1 tool's own clean commit (not the run commit)
    tool = Path(__file__).resolve()
    need(committed(tool, root), 'the X1 tool is not committed')
    log(f"[d4-x1] plan {plan_sha[:12]}; X1 commit {state['commit'][:12]}; run commit {plan['run_commit'][:12]}",
        flush=True)
    before = input_shas(plan, root)
    contract, csha = d4.load_contract(root / plan['inputs']['contract']['path'], root)
    need(csha == before['contract'], 'the contract SHA differs')
    provenance = {'run': check_run_links(plan, root), 'records': check_records(plan, root, before),
                  'predictions': check_prediction_source(plan, root, before, contract)}
    log('[d4-x1] provenance PASS; rebuilding the run with the runner\'s functions', flush=True)
    cache = Path(args.cache) if args.cache else root / plan['hdev_cache']
    rec = reconstruct(plan, contract, before, cache, root, log)
    log('[d4-x1] reconstruction equals the summary; joining the H-dev predictions', flush=True)
    info, routes = hdev_rows(contract, cache, root)
    values, pred_facts = checked_predictions(load_predictions(root / plan['inputs']['predictions']['path']), info,
                                             routes, rec['manifest'], before['r2_contract'])
    result, rows = analyse(rec, values)
    after = input_shas(plan, root)
    need(after == before, 'an input changed during the analysis')
    need(cli.git_state(root) == state, 'HEAD changed during the analysis')
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    with out_rows.open('x', encoding='utf-8', newline='\n') as handle:
        for r in rows:
            handle.write(json.dumps(public_row(r), allow_nan=False, separators=(',', ':')) + '\n')
    a = rec['actual']
    body = {'format': FORMAT, 'plan': {'path': PLAN, 'sha256': plan_sha},
            'nature': plan['nature'], 'x1_git': state,
            'x1_sources': {TOOL: d4.file_sha256(tool)},
            'inputs': {k: {'path': plan['inputs'][k]['path'], 'sha256': before[k]} for k in INPUTS},
            'inputs_unchanged': True, 'provenance': provenance,
            'reconstruction': {'identity_sha256': rec['identity_sha256'], 'games': len(rec['games']),
                               'eligible_plies': a['eligible_plies'], 'warmup_plies': a['warmup_plies'],
                               'observations': a['observations'], 'branches': a['branches'],
                               'branch_outcomes': a['branch_outcomes'], 'actor_check': a['actor_check'],
                               'reading': a['reading'], 'summary_match': True},
            'predictions': pred_facts, 'analysis_spec': ANALYSIS, 'analysis': result,
            'rows': {'path': plan['output']['rows'], 'sha256': d4.file_sha256(out_rows), 'count': len(rows)},
            'limits': LIMITS, 'seconds': round(time.monotonic() - started, 1),
            'scope': 'post-hoc exploratory description of the D4 full records; no strength, promotion, V9 or '
                     'champion ruling, no change to the D4 reading'}
    publish_json(out_summary, body)
    return body


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--plan', default=PLAN)
    parser.add_argument('--cache', help='the H-dev cache (default: the plan hdev_cache)')
    args = parser.parse_args(argv)
    try:
        body = run(args)
    except (Failure, ValueError, KeyError, TypeError, IndexError, OSError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    r = body['reconstruction']
    print(f"[d4-x1] DONE reading {r['reading']} (unchanged); observations {r['observations']}, branches "
          f"{r['branches']}; rows {body['rows']['count']}; {body['seconds']} s", flush=True)
    for compare, c in body['analysis']['within_model_q'].items():
        print(f"  {compare:14s} dQ valid candidate {c['candidate']['valid']}/{c['candidate']['observations']}, "
              f"c5 {c['c5']['valid']}/{c['c5']['observations']}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
