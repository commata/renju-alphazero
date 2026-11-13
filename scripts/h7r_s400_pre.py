"""R2-S400-PRE: how much an S400 re-search changes the c2 policy targets against S200, and what a full re-search
costs (docs/h7r/r2/R2-S400-PRE-draft.md). A measurement only: no training, no Arena, no record of another experiment
is changed; ``scripts/h7c_retarget.py``, ``hybrid/h7c_retarget.py`` and ``hybrid/h7c_targets.py`` are used as they are.

    prepare  (approved.prepare) every input against the contract (the c2 self-play, the c2 S200 re-search records,
             the c2 policy rows, the parent checkpoints, the engine / runtime / device); the fixed sample: the c2
             train-split games of the policy rows (games the label step kept), ordered by
             sha256("<salt>:<game id>"), the first ``games``; their eligible plies (coverage key list SHA); the
             counts of both time-estimate scopes. Writes prepare.json.
    measure  (approved.measure) the committed sample re-derived first, then, through ``h7c_retarget._execute``
             (the same agents, budgets, restored mode, per-row checks and resume):
               1. S200 on the first ``early_check_games`` sample games: every record must reproduce the c2 S200
                  record (route, move, visit counts, proven losses, E4-v2 step, tree budget, cooldown exact; mean
                  values within ``h7c_retarget.Q_TOLERANCE``, the G0 rule); a difference stops before anything else
               2. S200 on the rest of the sample: the same check over the whole sample; a difference stops
               3. S400 on the whole sample (only after 2)
             then the metrics (raw visit distributions and the policy targets of the c2 rule, S50 -> S200 beside
             S200 -> S400, on the positions of each pair and on the common positions), the game-level bootstrap
             intervals and the time estimates for both scopes. Writes measure.json (also on a stop).

No reading is derived: the measurement has no pass rule. Records stay under runs/; the committed outputs carry
counts, hashes and aggregates only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h7r_p1 as p1  # noqa: E402  (Guard, publication, committed checks: unchanged)
from h7r_p1 import Guard, committed, digest, file_sha256, require  # noqa: E402

FORMAT = 'h7r-s400-pre-contract-v1'
CONTRACT_ID = 'R2-S400-PRE'
D = 'docs/h7r/r2/R2-S400-PRE'
CONTRACT = f'{D}/s400-pre-contract.json'
RUNS = 'runs/h7r/R2-S400-PRE'
STAGES = ('prepare', 'measure')
INPUTS = ('selfplay', 'retarget_s200', 'policy_rows', 'policy_checkpoint', 'policy_metadata', 'value_checkpoint',
          'value_metadata')
PURPOSE = 'r2-s400-pre'
REPRO_EXACT = ('ply', 'color', 'seed', 'canonical_hash', 'route', 'engine_move', 'proven_loss', 'e4', 'tree_budget',
               'cooldown_restored')
PAIRS = (('s50', 's200'), ('s200', 's400'))
BUDGETS = ('s50', 's200', 's400')


# --- contract ---------------------------------------------------------------------------------------------------

def contract_path(path, root) -> Path:
    return Path(path) if path else Path(root) / CONTRACT


def load_contract(path=None, root=ROOT, stage=None, *, check_sources=True) -> tuple[dict, str, str]:
    root = Path(root)
    path = contract_path(path, root)
    c = json.loads(path.read_text(encoding='utf-8'))
    require(c.get('format') == FORMAT and c.get('contract_id') == CONTRACT_ID, f'{path}: not the {CONTRACT_ID} contract')
    require(c.get('status') == 'fixed', f'{path}: the contract is not fixed')
    approved = c.get('approved')
    require(isinstance(approved, dict) and set(approved) == set(STAGES)
            and all(isinstance(v, bool) for v in approved.values()), f'{path}: approved flags')
    require(not approved['measure'] or approved['prepare'], f'{path}: measure approved before prepare')
    require(set(c['inputs']) == set(INPUTS), f'{path}: inputs')
    require({k: v['factor'] for k, v in c['budgets'].items()} == {'s200': 4, 's400': 8}, f'{path}: budgets')
    s = c['sample']
    require(0 < s['early_check_games'] < s['games'], f'{path}: sample sizes')
    if stage is not None:
        require(approved[stage], f'{path}: the {stage} stage is not approved')
    if check_sources:
        for ref in c['sources']:
            require(file_sha256(root / ref['path']) == ref['sha256'], f"{ref['path']}: differs from the contract")
        for ref in c['evidence'].values():
            require(file_sha256(root / ref['path']) == ref['sha256'], f"{ref['path']}: evidence differs")
        require(committed(path, root), f'{path}: not committed')
    return c, file_sha256(path), p1.experiment_sha256(c)


def log_field(args) -> dict:
    """The desktop attempt log's basename (``--log-name``), so a recovery can find the log of this result."""
    import re

    name = getattr(args, 'log_name', None)
    if name is None:
        return {}
    require(re.fullmatch(rf'{args.command}-[A-Za-z0-9]+\.log', name) is not None, f'--log-name {name!r}')
    return {'log_name': name}


def stamp(sha, eid, git) -> dict:
    return {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'contract_file_sha256': sha, 'git': git}


def input_files(c) -> list[str]:
    return [c['inputs'][k]['path'] for k in INPUTS]


def check_input_files(c, root) -> None:
    for k in INPUTS:
        ref = c['inputs'][k]
        require(file_sha256(Path(root) / ref['path']) == ref['sha256'], f"input {k} ({ref['path']}) differs")


# --- inputs and the sample --------------------------------------------------------------------------------------

def read_policy_rows(path) -> list[dict]:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def game_key(salt: str, game_id: str) -> str:
    return hashlib.sha256(f'{salt}:{game_id}'.encode()).hexdigest()


def select_sample(train_ids: set, salt: str, games: int) -> list[str]:
    """The fixed sample: the train game ids ordered by sha256("<salt>:<id>"), the first ``games``."""
    require(len(train_ids) >= games, f'{len(train_ids)} train games, {games} wanted')
    return sorted(train_ids, key=lambda g: (game_key(salt, g), g))[:games]


def load_inputs(c: dict, root, device: str, *, engine: bool = True) -> dict:
    """Every input checked against the contract; the games, the c2 S200 rows, the policy rows and the sample."""
    import h7c_labels as lab
    import h7c_retarget as rt
    from hybrid import h7c_retarget as hr

    root = Path(root)
    check_input_files(c, root)
    path = lambda k: root / c['inputs'][k]['path']  # noqa: E731
    games, selfplay = rt.load_games(path('selfplay'))
    require(selfplay == c['selfplay_identity'], 'the self-play is not the c2 self-play of the contract')
    if engine:
        rt.check_engine(selfplay, path('policy_checkpoint'), path('value_checkpoint'), device)
    records = [json.loads(line) for line in path('retarget_s200').read_text(encoding='utf-8').splitlines()
               if line.strip()]
    require(records, 'the c2 S200 re-search file is empty')
    ident = records[0]['identity']
    require(ident.get('purpose') == 'retarget' and ident.get('factor') == c['budgets']['s200']['factor']
            and ident.get('budget') == hr.budget(ident['factor']) and ident.get('mode') == c['mode']
            and ident.get('selfplay') == selfplay and ident.get('coverage') == hr.coverage(games),
            'the c2 S200 re-search file is not the S200 re-search of this self-play')
    s200 = hr.index_rows(games, records, ident)
    rows = read_policy_rows(path('policy_rows'))
    by_id = {lab.game_id(g): g for g in games}
    require(all(r['game_id'] in by_id for r in rows), 'a policy row of a game outside the self-play')
    train_ids = {r['game_id'] for r in rows if r['split'] == 'train'}
    s = c['sample']
    sample_ids = select_sample(train_ids, s['salt'], s['games'])
    sample = [by_id[g] for g in sample_ids]
    row_mask = {(r['game_id'], r['ply']): bool(r['mask']) for r in rows if r['kind'] == 'visits'}
    full = sum(len(hr.eligible_plies(g)) for g in games)
    training = sum(len(hr.eligible_plies(by_id[g])) for g in train_ids)
    return {'games': games, 'selfplay': selfplay, 's200_c2': s200, 'sample_ids': sample_ids, 'sample': sample,
            'row_mask': row_mask, 'by_id': by_id,
            'scopes': {'full_eligible': full, 'training_eligible': training, 'train_games': len(train_ids),
                       'games': len(games)}}


def sample_record(c: dict, inp: dict) -> dict:
    from hybrid import h7c_retarget as hr

    return {'game_ids': inp['sample_ids'], 'game_indices': [g['game_index'] for g in inp['sample']],
            'sample_sha256': digest(inp['sample_ids']), 'coverage': hr.coverage(inp['sample']),
            'early_check_games': c['sample']['early_check_games'], 'scopes': inp['scopes']}


def prepare(args, root=ROOT) -> dict:
    c, sha, eid = load_contract(args.contract, root, 'prepare')
    guard = Guard(root, contract_path(args.contract, root), c, input_files(c))
    inp = load_inputs(c, root, args.device)
    require(inp['scopes']['full_eligible'] == c['time_reference']['eligible'],
            f"{inp['scopes']['full_eligible']} eligible plies, the time reference has {c['time_reference']['eligible']}")
    record = {'format': 'h7r-s400-pre-prepare-v1', **stamp(sha, eid, guard.check()), 'passed': True,
              'inputs_sha256': {k: c['inputs'][k]['sha256'] for k in INPUTS}, 'sample': sample_record(c, inp),
              **log_field(args)}
    p1.publish(Path(root) / D / 'prepare.json', record)
    return record


# --- the search ---------------------------------------------------------------------------------------------------

def search_identity(c: dict, eid: str, budget: str, sample_sha: str, selfplay: dict) -> dict:
    from hybrid import h7c_retarget as hr

    factor = c['budgets'][budget]['factor']
    return {'format': hr.FORMAT, 'purpose': PURPOSE, 'factor': factor, 'budget': hr.budget(factor),
            'mode': c['mode'], 'selfplay_jsonl_sha256': c['inputs']['selfplay']['sha256'], 'selfplay': selfplay,
            'experiment_sha256': eid, 'sample_sha256': sample_sha}


def search_games(c: dict, root, games: list, identity: dict, jsonl: Path, device: str, workers: int, log) -> dict:
    """``h7c_retarget._execute`` on these games (resumed from ``jsonl``): their rows by game index, the wall seconds
    of this call and how many of them were already in the file."""
    import h7c_retarget as rt
    import h7c_selfplay as sp

    root = Path(root)
    before = set(sp.load_finished(jsonl, {'identity': identity}, log=log)) if jsonl.exists() else set()
    args = SimpleNamespace(jsonl=jsonl, workers=workers, device=device,
                           policy_checkpoint=root / c['inputs']['policy_checkpoint']['path'],
                           value_checkpoint=root / c['inputs']['value_checkpoint']['path'])
    started = perf_counter()
    finished = rt._execute(args, games, identity, (c['mode'],), identity['factor'], log, validate=True)
    wall = perf_counter() - started
    wanted = {g['game_index'] for g in games}
    rows = {gi: rec['rows'] for gi, rec in finished.items() if gi in wanted}
    for g in games:  # resumed records too: _execute checks only the games it searched now
        check_rows(g, rows.get(g['game_index']))
    return {'rows': rows, 'wall_seconds': round(wall, 1), 'resumed_games': len(wanted & before), 'games': len(wanted)}


def finite(row: dict) -> bool:
    return all(isinstance(v[2], int) and v[2] >= 0 and math.isfinite(float(v[3])) for v in row.get('visits') or [])


def check_rows(game: dict, rows) -> None:
    """A game's search rows, searched now or read back from the record file: exactly its eligible plies in order,
    each passing ``h7c_retarget.check_row`` (the same position, route tree, the actor's E4-v2 step, legal moves), with
    non-negative integer visit counts and finite mean values."""
    from hybrid import h7c_retarget as hr

    where = f"game {game['game_index']}"
    plies = hr.eligible_plies(game)
    require(isinstance(rows, list) and [r.get('ply') for r in rows] == [p['ply'] for p in plies],
            f'{where}: the search rows are not its eligible plies (missing, empty or extra rows)')
    for p, row in zip(plies, rows):
        hr.check_row(game, p, row)
        require(finite(row), f"{where} ply {p['ply']}: a visit count or mean value is not a finite number")


def reproduction_diffs(new: dict, old: dict) -> list[str]:
    """The fields where a new S200 record differs from the c2 S200 record (the G0 rule: mean values within
    ``h7c_retarget.Q_TOLERANCE``, everything else exact)."""
    from hybrid import h7c_retarget as hr

    out = [k for k in REPRO_EXACT if new.get(k) != old.get(k)]
    if not finite(new) or not finite(old):
        return out + ['non_finite']  # a NaN would pass any tolerance comparison
    a, b = sorted(new['visits']), sorted(old['visits'])
    if [v[:3] for v in a] != [v[:3] for v in b]:
        out.append('visits')
    elif any(abs(x[3] - y[3]) > hr.Q_TOLERANCE for x, y in zip(a, b)):
        out.append('mean_value')
    return out


def check_reproduction(games: list, rows: dict, c2: dict) -> dict:
    """Every eligible ply of ``games`` against its c2 S200 record; reproduced only when every one was compared
    (the count is the eligible count) and none differs."""
    from hybrid import h7c_retarget as hr

    counts, examples, compared = {}, [], 0
    expected = sum(len(hr.eligible_plies(g)) for g in games)
    for g in games:
        for row in rows.get(g['game_index']) or []:
            key = (g['game_index'], row.get('ply'))
            if key not in c2:
                counts['not_a_c2_position'] = counts.get('not_a_c2_position', 0) + 1
                continue
            compared += 1
            for field in reproduction_diffs(row, c2[key]):
                counts[field] = counts.get(field, 0) + 1
                if len(examples) < 20:
                    examples.append({'game_index': g['game_index'], 'ply': row['ply'], 'field': field})
    if compared != expected:
        counts['not_compared'] = expected - compared
    return {'expected': expected, 'compared': compared, 'differing_fields': counts, 'examples': examples,
            'reproduced': not counts and compared == expected and expected > 0}


# --- metrics ------------------------------------------------------------------------------------------------------

def distribution(entries) -> dict:
    """``[[row, col, x, ...], ...]`` -> {(row, col): x / total} over x > 0 (visit counts or target mass)."""
    kept = {(int(e[0]), int(e[1])): float(e[2]) for e in entries if e[2] > 0}
    total = sum(kept.values())
    return {m: x / total for m, x in kept.items()} if total > 0 else {}


def top1(dist: dict):
    best = max(dist.values())
    return min(m for m, x in dist.items() if x == best)  # ties: the first move in row-major order


def tv(a: dict, b: dict) -> float:
    return 0.5 * sum(abs(a.get(m, 0.0) - b.get(m, 0.0)) for m in set(a) | set(b))


def entropy(a: dict) -> float:
    return -sum(x * math.log2(x) for x in a.values() if x > 0)


def js(a: dict, b: dict) -> float:
    moves = set(a) | set(b)
    m = {k: (a.get(k, 0.0) + b.get(k, 0.0)) / 2 for k in moves}
    kl = lambda p: sum(x * math.log2(x / m[k]) for k, x in p.items() if x > 0)  # noqa: E731
    return max(0.0, 0.5 * kl(a) + 0.5 * kl(b))


def pair_metrics(a: dict, b: dict) -> dict:
    return {'top1_changed': float(top1(a) != top1(b)), 'tv': tv(a, b), 'js': js(a, b)}


def loss_relation(a, b) -> str:
    sa, sb = {tuple(m) for m in a}, {tuple(m) for m in b}
    if sa == sb:
        return 'same'
    if sa > sb:
        return 'first_only'
    if sb > sa:
        return 'second_only'
    return 'both_unique'


def bucket(ply: int, edges) -> str:
    move = ply + 1  # ``ply`` counts the stones already on the board
    for lo, hi in edges:
        if lo <= move <= (hi if hi is not None else move):
            return f'{lo}-{hi if hi is not None else ""}'
    raise ValueError(f'move {move}: no bucket')


def position_metrics(c: dict, inp: dict, s200: dict, s400: dict) -> list[dict]:
    """One record per sampled eligible ply (S50 = the actor's own search)."""
    import h7c_labels as lab
    from hybrid import h7c_retarget as hr

    out = []
    for g in inp['sample']:
        gid = lab.game_id(g)
        new200 = {r['ply']: r for r in s200[g['game_index']]}
        new400 = {r['ply']: r for r in s400[g['game_index']]}
        for p in hr.eligible_plies(g):
            rec = {'s50': p, 's200': new200[p['ply']], 's400': new400[p['ply']]}
            raw = {b: distribution(rec[b]['visits']) for b in BUDGETS}
            tgt = {'s50': p['target']['target'] if p['target']['kind'] == 'visits' else None}
            for b in ('s200', 's400'):
                t, _, _ = hr.retarget_target(p, rec[b])
                tgt[b] = t['target'] if t['kind'] == 'visits' else None
            dist = {b: (distribution(t) if t else None) for b, t in tgt.items()}
            out.append({
                'game_index': g['game_index'], 'game_id': gid, 'ply': p['ply'],
                'bucket': bucket(p['ply'], c['metrics']['move_buckets']),
                'mask': inp['row_mask'].get((gid, p['ply'])),  # None: no c2 row (S200 gave no target)
                'raw': {f'{x}_{y}': pair_metrics(raw[x], raw[y]) for x, y in PAIRS},
                'raw_entropy': {b: entropy(raw[b]) for b in BUDGETS},
                'raw_top_share': {b: max(raw[b].values()) for b in BUDGETS},
                'children': {b: len(raw[b]) for b in BUDGETS},
                'loss_sets': {f'{x}_{y}': loss_relation(rec[x]['proven_loss'], rec[y]['proven_loss'])
                              for x, y in PAIRS},
                'has_target': {b: dist[b] is not None for b in BUDGETS},
                'target': {f'{x}_{y}': (pair_metrics(dist[x], dist[y]) if dist[x] and dist[y] else None)
                           for x, y in PAIRS},
                'target_entropy': {b: (entropy(dist[b]) if dist[b] else None) for b in BUDGETS},
                'targets': tgt, 'moves': g['moves'][:p['ply']]})
    return out


def kl_to_h3(c: dict, root, positions: list[dict], device: str) -> None:
    """KL(target || H3) on the unmasked positions, with the training encoding and legal mask; in place."""
    import torch

    from hybrid import h7c_train as tr
    from hybrid.h3_train import policy_logits
    from h7c_train_policy import load_policy_net

    net, _ = load_policy_net(Path(root) / c['inputs']['policy_checkpoint']['path'], device)
    net = net.to(device).eval()
    for pos in positions:
        pos['kl_h3'] = {b: None for b in BUDGETS}
    jobs = [(pos, b) for pos in positions if pos['mask'] is False for b in BUDGETS if pos['targets'][b]]
    for start in range(0, len(jobs), 512):
        chunk = jobs[start:start + 512]
        rows = [{'moves': pos['moves'], 'target': pos['targets'][b], 'kl': True, 'game_id': pos['game_id'],
                 'ply': pos['ply']} for pos, b in chunk]
        data = tr.policy_tensors(rows)
        planes, legal = tr.planes_from_states(data, torch.arange(len(rows)), device)
        with torch.no_grad():
            log_q = policy_logits(net, planes).float().masked_fill(~legal, -torch.inf).log_softmax(1)
        t = data['target'].to(device)
        terms = torch.where(t > 0, t * (t.clamp_min(1e-30).log() - log_q), torch.zeros_like(t))
        for (pos, b), v in zip(chunk, terms.sum(1).tolist()):
            pos['kl_h3'][b] = v


# --- aggregation --------------------------------------------------------------------------------------------------

def quantile(sorted_values: list, q: float) -> float:
    from hybrid.h7r_eval import _q
    return _q(sorted_values, q)


def describe(values: list) -> dict:
    v = sorted(values)
    n = len(v)
    if not n:
        return {'n': 0}
    return {'n': n, 'mean': round(sum(v) / n, 6), 'median': round(quantile(v, 0.5), 6),
            'p90': round(quantile(v, 0.9), 6)}


def bootstrap_means(series: dict, game_order: list, samples: int, seed: int) -> dict:
    """``series``: name -> [(game_index, value)]. Games resampled with replacement (the same draws for every
    series); per draw the position-weighted mean over the drawn games' positions; 95% percentile interval."""
    import torch

    g_index = {g: i for i, g in enumerate(game_order)}
    k = len(game_order)
    draws = torch.randint(0, k, (samples, k), generator=torch.Generator().manual_seed(seed))
    out = {}
    for name, pairs in series.items():
        if not pairs:
            out[name] = None
            continue
        sums, counts = torch.zeros(k, dtype=torch.float64), torch.zeros(k, dtype=torch.float64)
        for g, v in pairs:
            sums[g_index[g]] += v
            counts[g_index[g]] += 1
        s, n = sums[draws].sum(1), counts[draws].sum(1)
        means = sorted((s[n > 0] / n[n > 0]).tolist())
        out[name] = {'ci95': [round(quantile(means, 0.025), 6), round(quantile(means, 0.975), 6)],
                     'draws_used': len(means)}
    return out


def aggregate(c: dict, positions: list[dict], game_order: list) -> dict:
    """The tables: raw visit metrics over every sampled position; target metrics on each pair's positions and on the
    common positions (unmasked, a target at all three budgets); the denominators beside them."""
    series, populations = {}, {}

    def add(name, pos, value):
        series.setdefault(name, []).append((pos['game_index'], value))
        populations.setdefault(name, {}).setdefault(pos['bucket'], []).append(value)

    unmasked = [x for x in positions if x['mask'] is False]
    common = [x for x in unmasked if all(x['has_target'].values())]
    for pos in positions:
        for pair, m in pos['raw'].items():
            for k, v in m.items():
                add(f'raw.{pair}.{k}', pos, v)
        for b in BUDGETS:
            add(f'raw.entropy.{b}', pos, pos['raw_entropy'][b])
            add(f'raw.top_share.{b}', pos, pos['raw_top_share'][b])
            add(f'raw.children.{b}', pos, pos['children'][b])
    for scope, pop in (('pair', unmasked), ('common', common)):
        for pos in pop:
            for pair, m in pos['target'].items():
                if m is not None:
                    for k, v in m.items():
                        add(f'target.{scope}.{pair}.{k}', pos, v)
            for b in BUDGETS:
                if pos['target_entropy'][b] is not None:
                    add(f'target.{scope}.entropy.{b}', pos, pos['target_entropy'][b])
                if pos.get('kl_h3', {}).get(b) is not None:
                    add(f'target.{scope}.kl_h3.{b}', pos, pos['kl_h3'][b])
    boot = bootstrap_means(series, game_order, c['metrics']['bootstrap']['samples'], c['metrics']['bootstrap']['seed'])
    tables = {name: {**describe([v for _, v in pairs]), 'ci95_game_bootstrap': (boot[name] or {}).get('ci95'),
                     'by_bucket': {b: describe(v) for b, v in sorted(populations[name].items())}}
              for name, pairs in sorted(series.items())}
    loss = {pair: {} for pair in ('s50_s200', 's200_s400')}
    for pos in positions:
        for pair, rel in pos['loss_sets'].items():
            loss[pair][rel] = loss[pair].get(rel, 0) + 1
    has = lambda pos, b: pos['has_target'][b]  # noqa: E731
    denominators = {
        'positions': len(positions), 'masked': sum(x['mask'] is True for x in positions),
        'no_c2_row': sum(x['mask'] is None for x in positions), 'unmasked': len(unmasked),
        'target_none': {b: sum(not has(x, b) for x in unmasked) for b in BUDGETS},
        'pair_positions': {f'{x}_{y}': sum(has(p, x) and has(p, y) for p in unmasked) for x, y in PAIRS},
        's200_only_target': sum(has(x, 's200') and not has(x, 's400') for x in unmasked),
        's400_only_target_unmasked': sum(has(x, 's400') and not has(x, 's200') for x in unmasked),
        's400_only_target_mask_unknown': sum(has(x, 's400') for x in positions if x['mask'] is None),
        'common_positions': len(common),
        'note': 'no_c2_row: positions without a c2 row (S200 gave no target), so their mask is unknown; they are '
                'counted only in s400_only_target_mask_unknown and the raw tables, never as rows a training '
                'set would gain'}
    return {'tables': tables, 'denominators': denominators, 'proven_loss_sets': loss}


def timing(c: dict, prep: dict, runs: dict, sample_rows: int) -> dict:
    """Per budget the rows, worker search seconds and wall time; the two estimates for both scopes."""
    ref = c['time_reference']
    per = {}
    for b in ('s200', 's400'):
        rows = [r for game in runs[b]['rows'].values() for r in game]
        per[b] = {'rows': len(rows), 'search_seconds': round(sum(r['seconds'] for r in rows), 1),
                  'search_seconds_mean': round(sum(r['seconds'] for r in rows) / len(rows), 4),
                  'wall_seconds_this_session': runs[b]['wall_seconds'], 'resumed_games': runs[b]['resumed_games']}
    ratio = per['s400']['search_seconds'] / per['s200']['search_seconds']
    scopes = {'full_re_search': prep['scopes']['full_eligible'], 'training_data': prep['scopes']['training_eligible']}
    whole = not runs['s200']['resumed_games'] and not runs['s400']['resumed_games']
    out = {'per_budget': per, 's400_over_s200_search_seconds': round(ratio, 4), 'workers': c['workers'],
           'reference': ref, 'sample_rows': sample_rows, 'scopes': scopes, 'estimates': {}}
    for scope, n in scopes.items():
        est = {'positions': n,
               'ratio_method_seconds': round(ref['c2_s200_wall_seconds'] * ratio * n / ref['eligible'], 1),
               'search_only_seconds': round(per['s400']['search_seconds'] / sample_rows * n / c['workers'], 1)}
        if whole:
            est['extrapolated_seconds'] = round(per['s400']['wall_seconds_this_session'] * n / sample_rows, 1)
        out['estimates'][scope] = est
    if whole:
        s200_full = per['s200']['wall_seconds_this_session'] * ref['eligible'] / sample_rows
        out['extrapolation_check'] = {'s200_full_extrapolated_seconds': round(s200_full, 1),
                                      'c2_s200_actual_seconds': ref['c2_s200_wall_seconds'],
                                      'ratio': round(s200_full / ref['c2_s200_wall_seconds'], 4)}
    else:
        out['extrapolation_check'] = 'not available: a search phase resumed earlier games (wall time incomplete)'
    return out


# --- measure ------------------------------------------------------------------------------------------------------

def measure(args, root=ROOT, log=print) -> dict:
    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'measure')
    prep = _committed(root, f'{D}/prepare.json', 'prepare record', eid)
    out_path = root / D / 'measure.json'
    require(not out_path.exists(), f'{out_path} exists: never overwritten')
    guard = Guard(root, contract_path(args.contract, root), c, [*input_files(c), f'{D}/prepare.json'])
    inp = load_inputs(c, root, args.device)
    sample = sample_record(c, inp)
    require(sample == prep['sample'], 'the sample is not the committed prepare record\'s')
    k = c['sample']['early_check_games']
    early, rest = inp['sample'][:k], inp['sample'][k:]
    runs_dir = root / RUNS
    ident = {b: search_identity(c, eid, b, sample['sample_sha256'], inp['selfplay']) for b in ('s200', 's400')}
    body = {'format': 'h7r-s400-pre-measure-v1', **log_field(args), 'sample_sha256': sample['sample_sha256'],
            'prepare_sha256': file_sha256(root / D / 'prepare.json'), 'search_identity_sha256':
                {b: digest(v) for b, v in ident.items()}}

    def finish(status, **parts):
        record = {**body, **stamp(sha, eid, guard.check()), 'status': status, **parts}
        p1.publish(out_path, record)
        return record

    log(f'[s400-pre] S200 early reproduction check: {len(early)} games', flush=True)
    s200_early = search_games(c, root, early, ident['s200'], runs_dir / 's200.jsonl', args.device, c['workers'], log)
    repro_early = check_reproduction(early, s200_early['rows'], inp['s200_c2'])
    if not repro_early['reproduced']:
        return finish('S200_EARLY_NOT_REPRODUCED', s200_reproduction={'early': repro_early})
    log(f'[s400-pre] S200 rest of the sample: {len(rest)} games', flush=True)
    s200_all = search_games(c, root, inp['sample'], ident['s200'], runs_dir / 's200.jsonl', args.device,
                            c['workers'], log)
    s200_all['wall_seconds'] = round(s200_early['wall_seconds'] + s200_all['wall_seconds'], 1)
    s200_all['resumed_games'] = s200_early['resumed_games'] + max(0, s200_all['resumed_games'] - len(early))
    repro = check_reproduction(inp['sample'], s200_all['rows'], inp['s200_c2'])
    if not repro['reproduced']:
        return finish('S200_NOT_REPRODUCED', s200_reproduction={'early': repro_early, 'sample': repro})
    log(f"[s400-pre] S400 on the sample: {len(inp['sample'])} games", flush=True)
    s400 = search_games(c, root, inp['sample'], ident['s400'], runs_dir / 's400.jsonl', args.device, c['workers'], log)
    positions = position_metrics(c, inp, s200_all['rows'], s400['rows'])
    kl_to_h3(c, root, positions, args.device)
    agg = aggregate(c, positions, [g['game_index'] for g in inp['sample']])
    times = timing(c, prep['sample'], {'s200': s200_all, 's400': s400}, len(positions))
    runs_dir.mkdir(parents=True, exist_ok=True)
    detail = runs_dir / 'positions.jsonl.gz'
    with gzip.open(detail, 'wt', encoding='utf-8') as handle:  # under runs/: per-position values, never committed
        for pos in positions:
            handle.write(json.dumps({k: v for k, v in pos.items() if k not in ('targets', 'moves')}) + '\n')
    return finish('MEASURED', s200_reproduction={'early': repro_early, 'sample': repro}, **agg, timing=times,
                  positions_file_sha256=file_sha256(detail),
                  note='a measurement: no pass rule, no reading; the decision on a full re-search is the user\'s')


def _committed(root, rel, what, eid) -> dict:
    path = Path(root) / rel
    require(path.is_file() and committed(path, root), f'{rel}: the {what} is not committed')
    rec = json.loads(path.read_text(encoding='utf-8'))
    require(rec.get('experiment_sha256') == eid, f'{rel}: the {what} is of another experiment')
    return rec


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--contract', default=None)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'measure'):
        s = sub.add_parser(name)
        s.add_argument('--device', default='cuda')
        s.add_argument('--log-name', help='the desktop attempt log basename, kept in the record for a recovery')
    args = parser.parse_args(argv)
    try:
        out = (prepare if args.command == 'prepare' else measure)(args)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', flush=True)
        return 1
    keys = ('passed', 'status', 'sample', 's200_reproduction', 'denominators', 'timing')
    print(json.dumps({k: out[k] for k in keys if k in out}, indent=1, default=str), flush=True)
    if args.command == 'measure' and out['status'] != 'MEASURED':
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
