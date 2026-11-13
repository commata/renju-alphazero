"""R2-S400-TRAIN: the A1 T policy method with the train visit targets re-searched at S400
(docs/h7r/r2/R2-S400-TRAIN-draft.md). Control: A1's T checkpoints, unchanged. Candidate S: the same training (A1's
``h7r_p1a1.execute``: deterministic settings, the P1 trainer with replay, mu 0.5) on a rows file that differs from
the c2 rows only in the train visit targets.

    prepare        (approved.prepare) every input against the contract: the A1 inputs (``h7r_p1.check_inputs``), the
                   c2 self-play and S200 re-search records, the search models, the A1 T checkpoints of the committed
                   A1 seed records
    gate           (approved.gate) T(20261024) trained again through ``h7r_p1a1.execute`` with the A1 gate's numeric
                   settings must equal its committed A1 seed record exactly (fingerprints, trainer record, P1
                   records, initial / selected state hashes, fixed-position log-probabilities); before any search
    search         (approved.search) the S200 reproduction on the first 5 games of the S400-PRE sample (the G0
                   rule, ``h7r_s400_pre``), then S400 on every eligible ply of the kept train-split games
                   (``h7r_s400_pre.search_games``: the same agents, checks and resume); records under runs/
    rows           (approved.search) the candidate rows: the c2 rows with each train visit row's target replaced by
                   its S400 target (``h7c_retarget.retarget_target``); no S400 mass keeps the S200 target (counted);
                   no row is added; then the same check ``train`` repeats
    train          (approved.train) one candidate run ``--seed S``: the committed gate and rows judged again; the rows
                   file is its recorded SHA-256 and passes the row check; trained through ``h7r_p1a1.execute`` (kind
                   T) with the gate's numeric settings; the record and the checkpoint metadata carry the c2 rows, the
                   candidate rows and the S400 record SHA-256
    arena-contract (approved.arena) the v2 Arena contract (``hybrid.h7r_eval_v2``): per development suite (O1-dev-a,
                   O1-dev-b; 300 pairs; v2 seeds 9101 / 9102) S(20261024) vs champion and T(20261024) vs champion,
                   one pairing per suite

Playing and judging the v2 Arena come with the Arena stage. Invariance and identity as in A1: HEAD (clean), the
contract file, the sources and the read files checked again right before publishing; every record carries the
experiment SHA-256 and its contract file SHA-256; records are written once.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h7r_p1 as p1  # noqa: E402
import h7r_p1a1 as a1  # noqa: E402
import h7r_s400_pre as pre  # noqa: E402
from h7r_p1 import Guard, committed, digest, file_sha256, require  # noqa: E402

FORMAT = 'h7r-s400-train-contract-v1'
CONTRACT_ID = 'R2-S400-TRAIN'
D = 'docs/h7r/r2/R2-S400-TRAIN'
CONTRACT = f'{D}/s400-train-contract.json'
RUNS = 'runs/h7r/R2-S400-TRAIN'
SEED_RECORDS = f'{D}/seed-records'
ARENA_CONTRACT = 'docs/h7r/contracts-v2/R2-S400-TRAIN-ARENA-1.json'
ARENA_ID = 'R2-S400-TRAIN-ARENA-1'
STAGES = ('prepare', 'gate', 'search', 'train', 'arena')
GATE_SEED = 20261024
PURPOSE = 'r2-s400-train'  # S400 search records: never a label-step input (it accepts purpose retarget only)
ROW_CHANGED = ('target', 's400')
SEARCH_KEYS = ('selfplay', 'retarget_s200', 'policy_checkpoint', 'policy_metadata', 'value_checkpoint',
               'value_metadata')


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
    for i, s in enumerate(STAGES):
        if approved[s]:
            require(all(approved[x] for x in STAGES[:i]), f'{path}: {s} approved before {STAGES[:i]}')
    require(set(c['search']['inputs']) == set(SEARCH_KEYS), f'{path}: search inputs')
    require(c['search']['budgets']['s400']['factor'] == 8 and c['search']['budgets']['s200']['factor'] == 4,
            f'{path}: search budgets')
    require(c['gate']['seed'] == GATE_SEED and GATE_SEED in c['seeds']['training']
            and c['seeds']['arena_candidate_training_seed'] == GATE_SEED, f'{path}: gate / candidate seed')
    if stage is not None:
        require(approved[stage], f'{path}: the {stage} stage is not approved')
    if check_sources:
        for ref in c['sources']:
            require(file_sha256(root / ref['path']) == ref['sha256'], f"{ref['path']}: differs from the contract")
        for ref in c['evidence'].values():
            require(file_sha256(root / ref['path']) == ref['sha256'], f"{ref['path']}: evidence differs")
        require(committed(path, root), f'{path}: not committed')
    return c, file_sha256(path), p1.experiment_sha256(c)


def stamp(sha, eid, git) -> dict:
    return {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'contract_file_sha256': sha, 'git': git}


def committed_record(root, rel, what, eid) -> dict:
    path = Path(root) / rel
    require(path.is_file() and committed(path, root), f'{rel}: the {what} is not committed')
    rec = json.loads(path.read_text(encoding='utf-8'))
    require(rec.get('experiment_sha256') == eid, f'{rel}: the {what} is of another experiment')
    return rec


def fresh_dir(out: Path):
    require(not (out / 'record.json').exists(), f'{out} already holds a run')
    require(not out.exists() or not any(out.iterdir()),
            f'{out} holds files of an interrupted run: report it (nothing is overwritten)')


def search_view(c: dict) -> dict:
    """The S400-PRE loader's contract shape for the c2 search inputs (``h7r_s400_pre.load_inputs``)."""
    s = c['search']
    return {'inputs': {**s['inputs'], 'policy_rows': c['inputs']['cycle_rows']}, 'selfplay_identity': s['selfplay_identity'],
            'budgets': s['budgets'], 'mode': s['mode'], 'sample': s['reproduction_sample'], 'workers': s['workers'],
            'time_reference': s['time_reference']}


def control_paths(c: dict) -> list[str]:
    return [ref['checkpoint'] for ref in c['control']['models'].values()] + \
        [ref['metadata'] for ref in c['control']['models'].values()]


def input_files(c: dict) -> list[str]:
    return [*p1.input_files(), *(c['search']['inputs'][k]['path'] for k in SEARCH_KEYS), *control_paths(c)]


def check_control(c: dict, root) -> dict:
    """The A1 T checkpoints of the committed A1 seed records (their candidate SHA-256), on disk."""
    out = {}
    for seed, ref in c['control']['models'].items():
        rec = json.loads((Path(root) / ref['seed_record']).read_text(encoding='utf-8'))
        require(file_sha256(Path(root) / ref['seed_record']) == ref['seed_record_sha256'],
                f"{ref['seed_record']}: not the A1 seed record of the contract")
        require((rec['arm'], rec['seed']) == ('T', int(seed)) and rec['candidate'] == {
            'checkpoint_sha256': ref['checkpoint_sha256'], 'metadata_sha256': ref['metadata_sha256']},
            f'A1 T {seed}: the contract does not name its checkpoint')
        require(file_sha256(Path(root) / ref['checkpoint']) == ref['checkpoint_sha256']
                and file_sha256(Path(root) / ref['metadata']) == ref['metadata_sha256'], f'A1 T {seed}: files differ')
        out[seed] = rec
    return out


# --- prepare ------------------------------------------------------------------------------------------------------

def prepare(args, root=ROOT) -> dict:
    c, sha, eid = load_contract(args.contract, root, 'prepare')
    guard = Guard(root, contract_path(args.contract, root), c, input_files(c))
    checked = p1.check_inputs(c, root)
    inp = pre.load_inputs(search_view(c), root, args.device)
    control = check_control(c, root)
    train_ids = {r['game_id'] for r in read_rows(Path(root) / p1.PATHS['cycle_rows']) if r['split'] == 'train'}
    record = {'format': 'h7r-s400-train-prepare-v1', **stamp(sha, eid, guard.check()), 'passed': True,
              'cycle_rows': {'sha256': c['inputs']['cycle_rows']['sha256'], 'train': len(checked['rows']['train']),
                             'val': len(checked['rows'].get('val', []))},
              'search_scope': {'train_games': len(train_ids), 'eligible': inp['scopes']['training_eligible']},
              'reproduction_sample': {'game_ids': inp['sample_ids'][:c['search']['reproduction_games']]},
              'control': {s: r['candidate'] for s, r in control.items()},
              'input_files_sha256': guard.start['files']}
    p1.publish(Path(root) / D / 'prepare.json', record)
    return record


# --- gate: T reproduced -------------------------------------------------------------------------------------------

GATE_COMPARED = ('fingerprints', 'initial_state_sha256', 'selected_state_sha256', 'fixed_logits', 'p1',
                 'total_steps', 'steps_per_epoch', 'warmup_steps', 'train_rows', 'val_rows', 'teacher_guard',
                 'parent_guard_top1', 'parent_val', 'evaluations', 'best', 'verdict', 'config', 'settings')


def gate_view(rec: dict) -> dict:
    return {k: rec.get(k) for k in GATE_COMPARED}


def a1_gate_settings(c: dict, root) -> dict:
    ref = c['gate']['a1_gate']
    require(file_sha256(Path(root) / ref['path']) == ref['sha256'], f"{ref['path']}: not the A1 gate of the contract")
    return json.loads((Path(root) / ref['path']).read_text(encoding='utf-8'))['settings']


def gate(args, root=ROOT, log=print) -> dict:
    from hybrid import h7c_train as tr

    c, sha, eid = load_contract(args.contract, root, 'gate')
    committed_record(root, f'{D}/prepare.json', 'prepare record', eid)
    out_path = Path(root) / D / 'gate.json'
    require(not out_path.exists(), f'{out_path} exists: never overwritten')
    a1_rec = check_control(c, root)[str(GATE_SEED)]
    guard = Guard(root, contract_path(args.contract, root), c, [*input_files(c), f'{D}/prepare.json',
                                                                c['gate']['a1_gate']['path']])
    checked = p1.check_inputs(c, root)
    settings = a1_gate_settings(c, root)
    run = a1.execute(c, 'T', GATE_SEED, args.device, checked, root, log, expect_settings=settings)
    result = run['result']
    mine = {**run['summary'], **result['record'], 'p1': result['p1'], 'config': tr.config_dict(run['cfg']),
            'settings': run['settings']}
    diffs = a1.differences(gate_view(mine), gate_view(a1_rec))
    record = {'format': 'h7r-s400-train-gate-v1', **stamp(sha, eid, guard.check()), 'seed': GATE_SEED,
              'passed': not diffs, 'differences': diffs, 'a1_seed_record_sha256':
                  c['control']['models'][str(GATE_SEED)]['seed_record_sha256'], 'settings': run['settings'],
              'run': gate_view(mine), 'seconds': run['seconds']}
    p1.publish(out_path, record)
    return record


def verified_gate(c: dict, eid: str, root) -> dict:
    rec = committed_record(root, f'{D}/gate.json', 'gate record', eid)
    a1_rec = check_control(c, root)[str(GATE_SEED)]
    again = a1.differences(rec['run'], gate_view(a1_rec))
    require(rec.get('passed') is True and rec.get('differences') == [] and not again,
            f"the T reproduction gate did not pass: {again or rec.get('differences')}")
    return rec


# --- search -------------------------------------------------------------------------------------------------------

def search_identity(c: dict, eid: str, budget: str, selfplay: dict, scope_sha: str) -> dict:
    from hybrid import h7c_retarget as hr

    factor = c['search']['budgets'][budget]['factor']
    return {'format': hr.FORMAT, 'purpose': PURPOSE, 'factor': factor, 'budget': hr.budget(factor),
            'mode': c['search']['mode'], 'selfplay_jsonl_sha256': c['search']['inputs']['selfplay']['sha256'],
            'selfplay': selfplay, 'experiment_sha256': eid, 'scope_sha256': scope_sha}


def train_scope(inp: dict, root) -> list[dict]:
    """The kept train-split games of the c2 rows (the S400 search scope), in game-index order."""
    train_ids = {r['game_id'] for r in read_rows(Path(root) / p1.PATHS['cycle_rows']) if r['split'] == 'train'}
    return sorted((inp['by_id'][g] for g in train_ids), key=lambda g: g['game_index'])


def search(args, root=ROOT, log=print) -> dict:
    from hybrid import h7c_retarget as hr

    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'search')
    committed_record(root, f'{D}/prepare.json', 'prepare record', eid)
    verified_gate(c, eid, root)
    out_path = root / D / 'search.json'
    require(not out_path.exists(), f'{out_path} exists: never overwritten')
    guard = Guard(root, contract_path(args.contract, root), c, [*input_files(c), f'{D}/prepare.json', f'{D}/gate.json'])
    view = search_view(c)
    inp = pre.load_inputs(view, root, args.device)
    k = c['search']['reproduction_games']
    early = inp['sample'][:k]
    games = train_scope(inp, root)
    scope = {'games': len(games), **hr.coverage(games)}
    require(scope['eligible'] == c['search']['scope_eligible'], f"{scope['eligible']} eligible plies in the train "
                                                                 f"scope, the contract has {c['search']['scope_eligible']}")
    body = {'format': 'h7r-s400-train-search-v1', 'scope': scope}

    def finish(status, **parts):
        record = {**body, **stamp(sha, eid, guard.check()), 'status': status, **parts}
        p1.publish(out_path, record)
        return record

    runs = root / RUNS
    ident200 = search_identity(c, eid, 's200', inp['selfplay'], digest([g['game_index'] for g in early]))
    log(f'[s400-train] S200 reproduction check: {len(early)} games', flush=True)
    s200 = pre.search_games(view, root, early, ident200, runs / 's200-check.jsonl', args.device,
                            c['search']['workers'], log)
    repro = pre.check_reproduction(early, s200['rows'], inp['s200_c2'])
    if not repro['reproduced']:
        return finish('S200_NOT_REPRODUCED', s200_reproduction=repro)
    ident400 = search_identity(c, eid, 's400', inp['selfplay'], scope['coverage_sha256'])
    log(f"[s400-train] S400 on the train scope: {len(games)} games, {scope['eligible']:,} positions", flush=True)
    s400 = pre.search_games(view, root, games, ident400, runs / 's400.jsonl', args.device, c['search']['workers'], log)
    rows = [r for g in s400['rows'].values() for r in g]
    return finish('SEARCHED', s200_reproduction=repro, s400={
        'records_sha256': file_sha256(runs / 's400.jsonl'), 'identity_sha256': digest(ident400), 'rows': len(rows),
        'search_seconds': round(sum(r['seconds'] for r in rows), 1), 'wall_seconds_this_session': s400['wall_seconds'],
        'resumed_games': s400['resumed_games']})


# --- rows ---------------------------------------------------------------------------------------------------------

def read_rows(path) -> list[dict]:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def s400_records(path, games: list[dict], identity_sha: str) -> dict:
    """The S400 rows by (game index, ply), every game of the scope present and checked (``h7r_s400_pre.check_rows``)."""
    by = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            rec = json.loads(line)
            require(digest(rec['identity']) == identity_sha, 'an S400 record of another search identity')
            by[rec['game_index']] = rec['rows']
    for g in games:
        pre.check_rows(g, by.get(g['game_index']))
    return {(gi, r['ply']): r for gi, rows in by.items() for r in rows}


def candidate_rows(c2_rows: list[dict], games_by_id: dict, s400: dict) -> tuple[list[dict], dict]:
    """The candidate rows (module docstring) and the replacement counts."""
    from hybrid import h7c_retarget as hr

    out, counts = [], {'train_visit_rows': 0, 's400': 0, 's200_kept_no_s400_mass': 0}
    for row in c2_rows:
        if row['split'] == 'train' and row['kind'] == 'visits':
            g = games_by_id[row['game_id']]
            ply = next(p for p in g['plies'] if p['ply'] == row['ply'])
            target, _, _ = hr.retarget_target(ply, s400[(g['game_index'], row['ply'])])
            counts['train_visit_rows'] += 1
            if target['kind'] == 'visits':
                out.append({**row, 'target': target['target'], 's400': {'target': 's400'}})
                counts['s400'] += 1
            else:
                out.append({**row, 's400': {'target': 's200_kept', 'reason': 'no S400 mass'}})
                counts['s200_kept_no_s400_mass'] += 1
        else:
            out.append(row)
    n = counts['train_visit_rows']
    counts['s400_share'] = round(counts['s400'] / n, 6) if n else None
    counts['s200_kept_share'] = round(counts['s200_kept_no_s400_mass'] / n, 6) if n else None
    return out, counts


def s400_only_targets(c2_rows: list[dict], games: list[dict], s400: dict) -> int:
    """Eligible train plies without a c2 visit row (S200 left no mass) where S400 does: never added, counted."""
    from hybrid import h7c_retarget as hr
    import h7c_labels as lab

    have = {(r['game_id'], r['ply']) for r in c2_rows if r['kind'] == 'visits'}
    n = 0
    for g in games:
        gid = lab.game_id(g)
        for p in hr.eligible_plies(g):
            if (gid, p['ply']) not in have and hr.retarget_target(p, s400[(g['game_index'], p['ply'])])[0]['kind'] == 'visits':
                n += 1
    return n


def check_candidate(c2_rows: list[dict], cand: list[dict], games_by_id: dict, s400: dict) -> list[str]:
    """Why the candidate rows are not the c2 rows with only the train visit targets replaced (empty when they are)."""
    from hybrid import h7c_retarget as hr

    bad = []
    if len(cand) != len(c2_rows):
        return [f'{len(cand)} rows, the c2 file has {len(c2_rows)}']
    for i, (a, b) in enumerate(zip(c2_rows, cand)):
        where = f"row {i} ({a.get('game_id')} ply {a.get('ply')})"
        if {k: v for k, v in a.items() if k not in ROW_CHANGED} != {k: v for k, v in b.items() if k not in ROW_CHANGED}:
            bad.append(f'{where}: a field other than the target differs')
        elif a['split'] == 'train' and a['kind'] == 'visits':
            g = games_by_id[a['game_id']]
            ply = next(p for p in g['plies'] if p['ply'] == a['ply'])
            target, _, _ = hr.retarget_target(ply, s400[(g['game_index'], a['ply'])])
            want = ({'target': 's400'}, target['target']) if target['kind'] == 'visits' else \
                ({'target': 's200_kept', 'reason': 'no S400 mass'}, a['target'])
            if (b.get('s400'), b['target']) != want:
                bad.append(f'{where}: the target is not the S400 rule\'s')
        elif a != b:
            bad.append(f'{where}: a val, proof or masked row differs')
        if len(bad) >= 20:
            break
    return bad


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with gzip.GzipFile(tmp, 'wb', mtime=0) as raw:  # mtime 0: the same rows give the same file SHA-256
        for row in rows:
            raw.write((json.dumps(row, separators=(',', ':')) + '\n').encode())
    tmp.replace(path)


def rows(args, root=ROOT, log=print) -> dict:
    import h7c_labels as lab

    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'search')
    srec = committed_record(root, f'{D}/search.json', 'search record', eid)
    require(srec['status'] == 'SEARCHED', f"the search ended {srec['status']}: no candidate rows")
    out_path = root / D / 'rows.json'
    require(not out_path.exists(), f'{out_path} exists: never overwritten')
    cand_path = root / RUNS / 'policy_rows_s400.jsonl.gz'
    require(not cand_path.exists(), f'{cand_path} exists: never overwritten')
    guard = Guard(root, contract_path(args.contract, root), c, [*input_files(c), f'{D}/search.json',
                                                                f'{RUNS}/s400.jsonl'])
    require(file_sha256(root / RUNS / 's400.jsonl') == srec['s400']['records_sha256'], 'the S400 records changed')
    inp = pre.load_inputs(search_view(c), root, args.device, engine=False)
    games = train_scope(inp, root)
    s400 = s400_records(root / RUNS / 's400.jsonl', games, srec['s400']['identity_sha256'])
    c2_rows = read_rows(root / p1.PATHS['cycle_rows'])
    by_id = {lab.game_id(g): g for g in inp['games']}
    cand, counts = candidate_rows(c2_rows, by_id, s400)
    bad = check_candidate(c2_rows, cand, by_id, s400)
    require(not bad, f'the candidate rows fail their check: {bad}')
    write_rows(cand_path, cand)
    record = {'format': 'h7r-s400-train-rows-v1', **stamp(sha, eid, guard.check()),
              'c2_rows_sha256': c['inputs']['cycle_rows']['sha256'], 'candidate_rows_sha256': file_sha256(cand_path),
              's400_records_sha256': srec['s400']['records_sha256'], 'rows': len(cand),
              'replacement': counts, 's400_only_targets_not_added': s400_only_targets(c2_rows, games, s400),
              'check': 'passed'}
    p1.publish(out_path, record)
    return record


def verified_rows(c: dict, eid: str, root) -> tuple[dict, list[dict]]:
    """The committed rows record; the candidate file is its SHA-256 and passes the row check again."""
    import h7c_labels as lab

    root = Path(root)
    rec = committed_record(root, f'{D}/rows.json', 'rows record', eid)
    cand_path = root / RUNS / 'policy_rows_s400.jsonl.gz'
    require(file_sha256(cand_path) == rec['candidate_rows_sha256'], 'the candidate rows file is not the recorded one')
    require(file_sha256(root / RUNS / 's400.jsonl') == rec['s400_records_sha256'], 'the S400 records changed')
    srec = committed_record(root, f'{D}/search.json', 'search record', eid)
    inp = pre.load_inputs(search_view(c), root, 'cpu', engine=False)
    games = train_scope(inp, root)
    s400 = s400_records(root / RUNS / 's400.jsonl', games, srec['s400']['identity_sha256'])
    c2_rows, cand = read_rows(root / p1.PATHS['cycle_rows']), read_rows(cand_path)
    bad = check_candidate(c2_rows, cand, {lab.game_id(g): g for g in inp['games']}, s400)
    require(not bad, f'the candidate rows fail their check: {bad}')
    return rec, cand


# --- train --------------------------------------------------------------------------------------------------------

def run_record_path(seed) -> str:
    return f'{SEED_RECORDS}/S-{seed}/record.json'


def train(args, root=ROOT, log=print) -> dict:
    import torch
    from hybrid import h7c_train as tr
    from model.checkpoint import save_checkpoint

    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'train')
    require(args.seed in c['seeds']['training'], 'seed not in the contract')
    committed_record(root, f'{D}/prepare.json', 'prepare record', eid)
    g = verified_gate(c, eid, root)
    rows_rec, _ = verified_rows(c, eid, root)
    out = root / RUNS / f'S-{args.seed}'
    fresh_dir(out)
    cand_rel = f'{RUNS}/policy_rows_s400.jsonl.gz'
    guard = Guard(root, contract_path(args.contract, root), c,
                  [*input_files(c), f'{D}/prepare.json', f'{D}/gate.json', f'{D}/rows.json', cand_rel,
                   f'{RUNS}/s400.jsonl'])
    checked = p1.check_inputs(c, root)
    checked['rows'] = tr.load_policy_rows(root / cand_rel)  # the candidate rows (checked just above)
    run = a1.execute(c, 'T', args.seed, args.device, checked, root, log, expect_settings=g['settings'])
    result, cfg, parent = run['result'], run['cfg'], run['parent']
    r = result['record']
    exp = c['training_structure']
    require([r[k] for k in a1.STRUCTURE] == [exp[k] for k in a1.STRUCTURE]
            and [e['step'] for e in r['evaluations']] == exp['evaluation_steps'],
            'the step structure is not the contract\'s; nothing is published')
    inputs = {**run['inputs'], 'policy_rows': cand_rel, 'policy_rows_sha256': rows_rec['candidate_rows_sha256'],
              'c2_policy_rows_sha256': rows_rec['c2_rows_sha256'], 's400_records_sha256': rows_rec['s400_records_sha256']}
    git = guard.start['git']
    record = {'format': 'h7r-s400-train-run-v1', **stamp(sha, eid, git), 'arm': 'S', 'seed': args.seed,
              'config': tr.config_dict(cfg), 'inputs': inputs, 'settings': run['settings'],
              'gate_sha256': file_sha256(root / D / 'gate.json'), 'rows_sha256': file_sha256(root / D / 'rows.json'),
              **run['summary'], **r, 'p1': result['p1'], 'seconds': run['seconds']}
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    if result['state'] is not None:
        parent_meta = json.loads((root / p1.h3_meta_path()).read_text(encoding='utf-8'))
        meta = {'stage': tr.POLICY_STAGE, 'track': 'B', 'cycle': 2, 'policy_trained': True, 'value_trained': False,
                'config': {'channels': parent_meta['config']['channels'], 'blocks': parent_meta['config']['blocks'],
                           **tr.config_dict(cfg)},
                'parent_checkpoint_sha256': inputs['parent_checkpoint_sha256'],
                'teacher_checkpoint_sha256': inputs['teacher_checkpoint_sha256'],
                'policy_rows_sha256': inputs['policy_rows_sha256'],
                'c2_policy_rows_sha256': inputs['c2_policy_rows_sha256'],
                's400_records_sha256': inputs['s400_records_sha256'], 'source_models': inputs['source']['models'],
                'best': r['best'], 'teacher_guard_top1': r['teacher_guard']['top1'],
                's400_train': {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'arm': 'S', 'seed': args.seed,
                               'replay': result['p1']['replay']},
                'git': git}
        files['best.pt'] = lambda f: save_checkpoint(f, parent)
        files['best.json'] = lambda f: f.write_text(json.dumps(meta, indent=1) + '\n', encoding='utf-8')
        files['logits.pt'] = lambda f: torch.save(run['logits'], f)
    a1.write_temporary(out, files, guard)
    if result['state'] is not None:
        record['candidate'] = {'checkpoint_sha256': file_sha256(out / 'best.pt'),
                               'metadata_sha256': file_sha256(out / 'best.json')}
    p1.publish(out / 'record.json', record)
    return record


# --- Arena contract -----------------------------------------------------------------------------------------------

def arena_contract_body(c: dict, eid: str, root) -> dict:
    from hybrid import h7r_seeds as seeds

    s = c['seeds']['arena_candidate_training_seed']
    rec = committed_record(root, run_record_path(s), f'S {s} run record', eid)
    require(rec['verdict'] == 'CANDIDATE' and 'candidate' in rec, f'S {s}: no candidate (no other seed substituted)')
    t = c['control']['models'][str(s)]
    value = p1.PATHS['c5_value']  # the champion's value for every model: S50 with c5, as in A1
    models = {'champion': {'policy': p1.PATHS['h3'], 'value': value,
                           'identity': {**c['inputs']['parent_and_teacher'], **c['inputs']['value_c5']}},
              'S': {'policy': f'{RUNS}/S-{s}/best.pt', 'value': value,
                    'identity': {'policy_checkpoint_sha256': rec['candidate']['checkpoint_sha256'],
                                 'policy_metadata_sha256': rec['candidate']['metadata_sha256'], **c['inputs']['value_c5']}},
              'T': {'policy': t['checkpoint'], 'value': value,
                    'identity': {'policy_checkpoint_sha256': t['checkpoint_sha256'],
                                 'policy_metadata_sha256': t['metadata_sha256'], **c['inputs']['value_c5']}}}
    arms = []
    for spec in c['arena']['suites']:
        suite = {'path': spec['path'], 'sha256': file_sha256(Path(root) / spec['path'])}
        require(suite['sha256'] == spec['sha256'], f"{spec['path']}: not the suite of the contract")
        for tested in ('S', 'T'):
            arms.append({'name': f"{spec['name']}_{tested}_vs_champion", 'tested': {'model': tested, 'budget': 1},
                         'opponent': {'model': 'champion', 'budget': 1}, 'pairs': spec['pairs'], 'seed': spec['seed'],
                         'pairing': spec['name'], 'suite': suite})
    st = c['reading']['statistics']
    return {'format': 'h7r-eval-contract-v2', 'seed_policy': seeds.POLICY, 'contract_id': ARENA_ID,
            'purpose': 'pilot', 'status': 'fixed',
            'meaning': f'the Arena of {CONTRACT_ID} (experiment {eid[:12]}): development reading; never a promotion',
            'models': models, 'arms': arms,
            'stats': {'alpha_one_sided': st['alpha_one_sided'], 'bootstrap': st['bootstrap']},
            's400_train': {'contract_id': CONTRACT_ID, 'experiment_sha256': eid,
                           'candidate_record_sha256': file_sha256(Path(root) / run_record_path(s))}}


def arena_contract(args, root=ROOT) -> dict:
    from hybrid import h7r_eval_v2 as ev2

    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'arena')
    guard = Guard(root, contract_path(args.contract, root), c, [run_record_path(c['seeds']['arena_candidate_training_seed'])])
    body = arena_contract_body(c, eid, root)
    path = root / ARENA_CONTRACT
    ev2.check_contract(body, path, root, check_files=False)
    guard.check()
    p1.publish(path, body)
    ev2.load_contract(path, root, check_files=False)  # seeds and id unique across v1 and v2
    return body


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--contract', default=None)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'gate', 'search', 'rows'):
        s = sub.add_parser(name)
        s.add_argument('--device', default='cuda')
    t = sub.add_parser('train')
    t.add_argument('--seed', required=True, type=int)
    t.add_argument('--device', default='cuda')
    sub.add_parser('arena-contract')
    args = parser.parse_args(argv)
    fn = {'prepare': prepare, 'gate': gate, 'search': search, 'rows': rows, 'train': train,
          'arena-contract': arena_contract}[args.command]
    try:
        out = fn(args)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', flush=True)
        return 1
    keys = ('passed', 'differences', 'status', 'scope', 's200_reproduction', 'replacement',
            's400_only_targets_not_added', 'verdict', 'best', 'selected_state_sha256', 'seconds')
    print(json.dumps({k: out[k] for k in keys if k in out} or {'written': True}, indent=1, default=str), flush=True)
    if args.command == 'gate' and not out['passed']:
        return 2
    if args.command == 'search' and out['status'] != 'SEARCHED':
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
