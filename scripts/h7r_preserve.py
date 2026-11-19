"""R2-S400-PRESERVE-1: on the same S400 training rows, does a stronger H3 preservation (replay coefficient mu 2.0 against
the R2-S400-TRAIN 0.5) give a better candidate, and does it beat the champion? (docs/h7r/r2/R2-S400-PRESERVE-1-draft.md)

Nothing pinned is edited: the R2-S400-TRAIN contract, its records, rows and tool (``scripts/h7r_s400_train.py``) are read
and reused as they are. A run of mu 2.0 is ``h7r_p1a1.execute`` (kind T) on the R2-S400-TRAIN contract with the replay
coefficient replaced in a copy (``with_mu``): the same rows, seeds, replay generator and draws, steps, selection rule and
numeric settings. The mu 0.5 candidates are the committed R2-S400-TRAIN runs (never retrained for the Arena).

    prepare           the R2-S400-TRAIN contract (pinned file SHA-256 and experiment id) and its committed records, the
                      candidate rows (SHA-256 and the row check), the mu 0.5 seed records and the 20261024 checkpoint
                      files, the A1 inputs; the prepare record
    gate              the committed suites rechecked (``h7r_dev_suites_2.check``); S(mu 0.5, 20261024) trained again
                      and compared exactly with its committed R2-S400-TRAIN record (fingerprints, every evaluation,
                      selection, selected state, fixed-position log-probabilities, settings, inputs); a reproduction
                      check only, its checkpoint is never used
    train --seed S    S(mu 2.0) for one contract seed after the gate, kept only when its step count and batch,
                      symmetry and learning-rate fingerprints are those of the mu 0.5 run of the seed; the run record
                      (log name, mu, inputs, the mu 0.5 record of the seed) and the checkpoint files under runs/
    verify-run        a committed run record's checkpoint files are the ones it names (CANDIDATE) or absent
    arena-contract    the v2 development Arena contract on the committed new suites (O1-dev-c / d, game seeds 9103 /
                      9104): S(mu 2.0) and S(mu 0.5) of seed 20261024 each against the champion, one pairing per suite
    check-contract / check-models / play / judge / verify-judge
                      the Arena: the contract as the S400 contract and the records give it, every model file before
                      any game, the v2 play loop of ``scripts/h7r_s400_arena.py``, and the judge: D'' = S(2.0) - S(0.5)
                      and P'' = S(2.0) - 0.5, read only when both candidates played the same openings in each suite

Readings (section 7): P'' STRONGER may open a separate confirmation contract; D'' BETTER / WORSE records the stronger
preservation as an improvement over / worse than the old mu 0.5 candidate (otherwise its effect stays undetermined);
without P'' STRONGER the contract ends. The scope is the pre-designated training seed 20261024.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h7r_eval as cli  # noqa: E402
import h7r_p1 as p1  # noqa: E402
import h7r_p1a1 as a1  # noqa: E402
import h7r_s400_arena as sa  # noqa: E402
import h7r_s400_pre as pre  # noqa: E402
import h7r_s400_train as st  # noqa: E402
from h7r_p1 import Guard, committed, file_sha256, require  # noqa: E402

FORMAT = 'h7r-preserve-contract-v1'
CONTRACT_ID = 'R2-S400-PRESERVE-1'
D = 'docs/h7r/r2/R2-S400-PRESERVE-1'
CONTRACT = f'{D}/preserve-contract.json'
RUNS = 'runs/h7r/R2-S400-PRESERVE-1'
SEED_RECORDS = f'{D}/seed-records'
ARENA_CONTRACT = 'docs/h7r/contracts-v2/R2-S400-PRESERVE-ARENA-1.json'
ARENA_ID = 'R2-S400-PRESERVE-ARENA-1'
STAGES = ('prepare', 'suites', 'gate', 'train', 'arena')
MU, BASE_MU = 2.0, 0.5
SEEDS = (20261024, 20261111, 20261112)
GATE_SEED = ARENA_SEED = 20261024
SUITES = (('O1-dev-c', 'docs/h7r/suites/O1-dev-c.json', 9103), ('O1-dev-d', 'docs/h7r/suites/O1-dev-d.json', 9104))
PAIRS = 300
STATS = {'alpha_one_sided': 0.025,
         'bootstrap': {'samples': 20000, 'seed': 20261161, 'method': 'percentile', 'unit': 'opening pair'}}
ARMS = ('S2', 'S05')  # S(mu 2.0), S(mu 0.5)
JUDGE = f'{D}/judge.json'
ARMS_DIR = f'{RUNS}/arena'
SOURCES = ('scripts/h7r_preserve.py', 'scripts/h7r_dev_suites_2.py', 'src/hybrid/h7r_suite_seeds_dev2.py',
           'scripts/h7r_dev_suites.py', 'scripts/h7r_s400_train.py', 'scripts/h7r_s400_arena.py', 'scripts/h7r_eval.py',
           'scripts/h7r_p1a1.py', 'scripts/h7r_p1.py', 'src/hybrid/h7r_p1_train.py', 'src/hybrid/h7r_eval_v2.py',
           'src/hybrid/h7r_openings.py')
KEYS = {'format', 'contract_id', 'status', 'approved', 'purpose', 'draft', 's400_train', 'rows', 'base_runs', 'replay',
        'seeds', 'suites', 'arena', 'reading', 'stages', 'output', 'sources', 'scope'}


# --- contract -----------------------------------------------------------------------------------------------------

def contract_path(path, root=ROOT) -> Path:
    return Path(path) if path else Path(root) / CONTRACT


def contract_body(root=ROOT, sources=SOURCES) -> dict:
    """The fixed contract (every approval false) from the committed R2-S400-TRAIN contract, rows record and seed records."""
    root = Path(root)
    s4, s4sha, s4eid = st.load_contract(None, root, check_sources=False)
    rows = json.loads((root / st.D / 'rows.json').read_text(encoding='utf-8'))
    base = {}
    for s in SEEDS:
        rel = st.run_record_path(s)
        rec = json.loads((root / rel).read_text(encoding='utf-8'))
        base[str(s)] = {'record': rel, 'record_sha256': file_sha256(root / rel), 'verdict': rec['verdict'],
                        'checkpoint': f'{st.RUNS}/S-{s}/best.pt', 'metadata': f'{st.RUNS}/S-{s}/best.json',
                        'checkpoint_sha256': rec['candidate']['checkpoint_sha256'],
                        'metadata_sha256': rec['candidate']['metadata_sha256']}
    return {
        'format': FORMAT, 'contract_id': CONTRACT_ID, 'status': 'fixed', 'approved': {s: False for s in STAGES},
        'purpose': 'development: on the same S400 rows, is a stronger H3 preservation (mu 2.0 against 0.5) a better '
                   'candidate (D\'\') and does it beat the champion (P\'\')? Seed 20261024; never a promotion',
        'draft': {'path': f'{D}-draft.md'},
        's400_train': {'contract': st.CONTRACT, 'contract_file_sha256': s4sha, 'experiment_sha256': s4eid,
                       'gate': f'{st.D}/gate.json', 'rows_record': f'{st.D}/rows.json',
                       'rows_record_sha256': file_sha256(root / st.D / 'rows.json')},
        'rows': {'path': f'{st.RUNS}/policy_rows_s400.jsonl.gz', 'sha256': rows['candidate_rows_sha256']},
        'base_runs': base,
        'replay': {'mu': MU, 'base_mu': BASE_MU, 'note': 'mu is the replay KL coefficient (loss = cycle batch mean + '
                   'mu x replay KL mean); 2.0 is a pre-chosen value, four times the old coefficient, tested once; '
                   'every other replay setting (batch 512, generator seed = training seed + 1000003, draws) is the '
                   'R2-S400-TRAIN one'},
        'seeds': {'training': list(SEEDS), 'gate': GATE_SEED, 'arena_candidate_training_seed': ARENA_SEED,
                  'arena_games': {name: seed for name, _, seed in SUITES},
                  'suite_generation': {'O1-dev-c': 202611191, 'O1-dev-d': 202611192},
                  'bootstrap': STATS['bootstrap']['seed']},
        'suites': {'tool': 'scripts/h7r_dev_suites_2.py', 'record': 'docs/h7r/suites/O1-dev-cd-record.json',
                   'registry': 'src/hybrid/h7r_suite_seeds_dev2.py', 'rule': 'O1', 'count': PAIRS,
                   'specs': [{'name': n, 'path': path, 'seed': seed, 'pairs': PAIRS} for n, path, seed in SUITES]},
        'arena': {'contract': ARENA_CONTRACT, 'id': ARENA_ID, 'models': 'champion (H3 policy + c5 value), S2 = S(mu 2.0, '
                  '20261024), S05 = S(mu 0.5, 20261024) of R2-S400-TRAIN; all with c5 value at S50',
                  'arms': 'per suite S2 vs champion and S05 vs champion, the same openings and seed (one pairing)',
                  'stats': STATS},
        'reading': {'D\'\'': 'S2 - S05 pair scores on the same openings: BETTER (lower > 0) / WORSE (upper < 0) / '
                             'NOT_ESTABLISHED',
                    'P\'\'': 'S2 vs champion - 0.5: STRONGER / WEAKER / NOT_ESTABLISHED',
                    'combined': '0.5 x each suite (fixed weights), stratified opening bootstrap; class level reported',
                    'use': {'P\'\' STRONGER and D\'\' BETTER': 'a separate confirmation contract may open; the stronger '
                            'preservation is recorded as an improvement over the old candidate',
                            'P\'\' STRONGER and D\'\' WORSE': 'a separate confirmation contract may open; the '
                            'stronger preservation is recorded as worse than the old mu 0.5 candidate',
                            'P\'\' STRONGER otherwise': 'a separate confirmation contract may open; the effect of the '
                            'stronger preservation stays undetermined',
                            'D\'\' BETTER otherwise': 'recorded as an improvement over the old candidate; the contract '
                            'ends', 'D\'\' WORSE otherwise': 'recorded as worse than the old mu 0.5 candidate; the '
                            'contract ends', 'otherwise': 'the contract ends'},
                    'scope': 'the pre-designated training seed 20261024; training-seed variation is not in the bounds'},
        'stages': {'prepare': 'every input and record checked', 'suites': 'the two new suites built (after this '
                   'approval only) and committed with their record',
                   'gate': 'the committed suites rechecked, then S(mu 0.5, 20261024) reproduced exactly',
                   'train': 'S(mu 2.0) for the three seeds, kept only with the step count and the batch, symmetry and '
                   'learning-rate fingerprints of the mu 0.5 run of the seed; one commit each, its checkpoint files '
                   'checked before the record is staged', 'arena': 'the Arena contract, the models checked, four arms, the judge'},
        'output': {'prepare': f'{D}/prepare.json', 'gate': f'{D}/gate.json', 'runs': RUNS + '/S2-{seed}',
                   'seed_records': SEED_RECORDS + '/S2-{seed}/record.json', 'judge': JUDGE, 'arms': ARMS_DIR},
        'sources': [{'path': rel, 'sha256': file_sha256(root / rel)} for rel in sources],
        'scope': 'development reading only: no champion change, confirmation, promotion or loop restart'}


def experiment_sha256(c: dict) -> str:
    return p1.experiment_sha256(c)


def load_contract(path=None, root=ROOT, stage=None, *, check_sources=True) -> tuple[dict, str, str]:
    root = Path(root)
    path = contract_path(path, root)
    c = json.loads(path.read_text(encoding='utf-8'))
    require(isinstance(c, dict) and set(c) == KEYS, f'{path}: the contract keys differ')
    require(c['format'] == FORMAT and c['contract_id'] == CONTRACT_ID and c['status'] == 'fixed',
            f'{path}: not the fixed {CONTRACT_ID} contract')
    a = c['approved']
    require(isinstance(a, dict) and list(a) == list(STAGES) and all(type(v) is bool for v in a.values()),
            f'{path}: approved flags')
    for i, s in enumerate(STAGES):
        if a[s]:
            require(all(a[x] for x in STAGES[:i]), f'{path}: {s} approved before {list(STAGES[:i])}')
    require(c['replay']['mu'] == MU and c['replay']['base_mu'] == BASE_MU and c['seeds']['training'] == list(SEEDS)
            and c['seeds']['gate'] == GATE_SEED and c['seeds']['arena_candidate_training_seed'] == ARENA_SEED
            and c['arena']['stats'] == STATS, f'{path}: mu, seeds or statistics differ')
    if check_sources:
        changed = [r['path'] for r in c['sources'] if file_sha256(root / r['path']) != r['sha256']]
        require(not changed, f'the checkout differs from the sources the contract was fixed for: {changed}')
    require(committed(path, root), f'{path}: not committed')
    if stage is not None:
        require(a[stage] is True, f'{path}: the {stage} stage is not approved')
    return c, file_sha256(path), experiment_sha256(c)


def s400(c: dict, root=ROOT) -> tuple[dict, str, str]:
    """The R2-S400-TRAIN contract, exactly the pinned file and experiment (its sources checked)."""
    s4, sha, eid = st.load_contract(None, root)
    ref = c['s400_train']
    require(sha == ref['contract_file_sha256'] and eid == ref['experiment_sha256'],
            'the R2-S400-TRAIN contract is not the pinned one')
    require(file_sha256(Path(root) / ref['rows_record']) == ref['rows_record_sha256'], 'the S400 rows record differs')
    return s4, sha, eid


def with_mu(s4: dict, mu: float) -> dict:
    """The R2-S400-TRAIN contract with only the replay coefficient set to ``mu`` (``h7r_p1.replay_of`` reads it)."""
    out = copy.deepcopy(s4)
    out['arms']['T']['replay']['weight_mu'] = mu
    return out


def base_record(c: dict, seed: int, root=ROOT) -> dict:
    """The committed R2-S400-TRAIN run record of a seed (mu 0.5), its pinned SHA-256."""
    ref = c['base_runs'][str(seed)]
    rec = st.committed_record(root, ref['record'], f'S(mu 0.5) {seed} run record', c['s400_train']['experiment_sha256'])
    require(file_sha256(Path(root) / ref['record']) == ref['record_sha256'], f"{ref['record']}: not the pinned record")
    require(rec.get('arm') == 'S' and rec.get('seed') == seed and rec['p1']['replay']['mu'] == BASE_MU,
            f"{ref['record']}: not an S run of mu {BASE_MU} and seed {seed}")
    return rec


def stamp(sha, eid, git) -> dict:
    return {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'contract_file_sha256': sha, 'git': git}


def log_field(args, command) -> dict:
    return pre.log_field(argparse.Namespace(log_name=getattr(args, 'log_name', None), command=command))


def committed_record(root, rel, what, eid) -> dict:
    return st.committed_record(root, rel, what, eid)


def run_record_path(seed) -> str:
    return f'{SEED_RECORDS}/S2-{seed}/record.json'


# --- prepare ------------------------------------------------------------------------------------------------------

def prepare(args, root=ROOT) -> dict:
    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'prepare')
    out = root / c['output']['prepare']
    require(not out.exists(), f'{out} exists: never overwritten')
    s4, s4sha, s4eid = s400(c, root)
    guard = Guard(root, contract_path(args.contract, root), c, [*st.input_files(s4), c['rows']['path'],
                                                                 *(r['record'] for r in c['base_runs'].values())])
    st.verified_gate(s4, s4eid, root)
    rows_rec, cand = st.verified_rows(s4, s4eid, root)
    require(rows_rec['candidate_rows_sha256'] == c['rows']['sha256'], 'the candidate rows are not the pinned ones')
    bases = {s: base_record(c, s, root) for s in SEEDS}
    ref = c['base_runs'][str(ARENA_SEED)]
    files = {k: file_sha256(root / ref[k]) if (root / ref[k]).is_file() else None for k in ('checkpoint', 'metadata')}
    require(files == {'checkpoint': ref['checkpoint_sha256'], 'metadata': ref['metadata_sha256']},
            f'S(mu 0.5) {ARENA_SEED}: the checkpoint files differ from its record: {files}')
    checked = p1.check_inputs(s4, root)
    record = {'format': 'h7r-preserve-prepare-v1', **log_field(args, 'prepare'), **stamp(sha, eid, guard.check()),
              'passed': True, 's400_train': c['s400_train'],
              'rows': {'sha256': c['rows']['sha256'], 'rows': len(cand), 'train': len(checked['rows']['train'])},
              'base_runs': {str(s): {'verdict': r['verdict'], 'selected_state_sha256': r['selected_state_sha256'],
                                     'best_step': (r['best'] or {}).get('step')} for s, r in bases.items()},
              'arena_base_checkpoint': files}
    p1.publish(out, record)
    return record


# --- gate ---------------------------------------------------------------------------------------------------------

def run_view(run: dict) -> dict:
    """The compared view of one training run (``h7r_s400_train.gate_view`` of its record fields)."""
    from hybrid import h7c_train as tr
    result = run['result']
    return st.gate_view({**run['summary'], **result['record'], 'p1': result['p1'], 'config': tr.config_dict(run['cfg']),
                         'settings': run['settings']})


def run_inputs(run: dict, rows_rec: dict) -> dict:
    """The inputs block an S400-rows run records (``h7r_s400_train.train``)."""
    return {**run['inputs'], 'policy_rows': f'{st.RUNS}/policy_rows_s400.jsonl.gz',
            'policy_rows_sha256': rows_rec['candidate_rows_sha256'], 'c2_policy_rows_sha256': rows_rec['c2_rows_sha256'],
            's400_records_sha256': rows_rec['s400_records_sha256']}


def checked_inputs(c, s4, s4eid, root):
    rows_rec, cand = st.verified_rows(s4, s4eid, root)
    require(rows_rec['candidate_rows_sha256'] == c['rows']['sha256'], 'the candidate rows are not the pinned ones')
    checked = p1.check_inputs(s4, root)
    checked['rows'] = st.rows_by_split(cand)  # exactly the rows verified_rows checked (no second read)
    return rows_rec, checked


def gate(args, root=ROOT, log=print) -> dict:
    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'gate')
    committed_record(root, c['output']['prepare'], 'prepare record', eid)
    committed_suites(c, root)  # the suites stage comes before the gate
    out = root / c['output']['gate']
    require(not out.exists(), f'{out} exists: never overwritten')
    s4, s4sha, s4eid = s400(c, root)
    base = base_record(c, GATE_SEED, root)
    guard = Guard(root, contract_path(args.contract, root), c, [*st.input_files(s4), c['rows']['path'],
                                                                 c['base_runs'][str(GATE_SEED)]['record']])
    settings = st.verified_gate(s4, s4eid, root)['settings']
    rows_rec, checked = checked_inputs(c, s4, s4eid, root)
    run = a1.execute(with_mu(s4, BASE_MU), 'T', GATE_SEED, args.device, checked, root, log, expect_settings=settings)
    mine = run_view(run)
    diffs = a1.differences(mine, st.gate_view(base))
    diffs += [f'inputs.{d}' for d in a1.differences(run_inputs(run, rows_rec), base['inputs'])]
    record = {'format': 'h7r-preserve-gate-v1', **log_field(args, 'gate'), **stamp(sha, eid, guard.check()),
              'seed': GATE_SEED, 'mu': BASE_MU, 'passed': not diffs, 'differences': diffs,
              'base_record_sha256': c['base_runs'][str(GATE_SEED)]['record_sha256'], 'settings': run['settings'],
              'run': mine, 'inputs': run_inputs(run, rows_rec), 'seconds': run['seconds'],
              'note': 'a reproduction check: the checkpoint of this run is never used'}
    p1.publish(out, record)
    return record


def verified_gate(c: dict, eid: str, root=ROOT) -> dict:
    """The committed gate record, compared again with the committed mu 0.5 record (never its ``passed`` alone)."""
    rec = committed_record(root, c['output']['gate'], 'gate record', eid)
    base = base_record(c, GATE_SEED, root)
    again = a1.differences(rec['run'], st.gate_view(base)) + \
        [f'inputs.{d}' for d in a1.differences(rec['inputs'], base['inputs'])]
    require(rec.get('passed') is True and rec.get('differences') == [] and not again,
            f"the reproduction gate did not pass: {again or rec.get('differences')}")
    return rec


# --- train --------------------------------------------------------------------------------------------------------

DRAWS = ('steps', 'batch_sha256', 'sym_sha256', 'lr_sha256')


def same_draws(fp: dict, base_fp: dict) -> dict:
    """Step count, batch, symmetry and learning-rate fingerprints of a mu 2.0 run against the mu 0.5 run of its seed:
    each must be equal (and recorded in the base) for the run to be kept (the losses and the selected state move)."""
    return {k: base_fp.get(k) is not None and fp.get(k) == base_fp.get(k) for k in DRAWS}


def train(args, root=ROOT, log=print) -> dict:
    import torch
    from hybrid import h7c_train as tr
    from model.checkpoint import save_checkpoint

    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'train')
    require(args.seed in SEEDS, 'seed not in the contract')
    committed_record(root, c['output']['prepare'], 'prepare record', eid)
    g = verified_gate(c, eid, root)
    s4, s4sha, s4eid = s400(c, root)
    base = base_record(c, args.seed, root)
    out = root / RUNS / f'S2-{args.seed}'
    st.fresh_dir(out)
    guard = Guard(root, contract_path(args.contract, root), c,
                  [*st.input_files(s4), c['rows']['path'], c['output']['gate'], c['base_runs'][str(args.seed)]['record']])
    rows_rec, checked = checked_inputs(c, s4, s4eid, root)
    require(guard.start['files'][c['rows']['path']] == rows_rec['candidate_rows_sha256'],
            'the candidate rows changed between the start of the run and their check')
    run = a1.execute(with_mu(s4, MU), 'T', args.seed, args.device, checked, root, log, expect_settings=g['settings'])
    result, cfg, parent = run['result'], run['cfg'], run['parent']
    r = result['record']
    exp = s4['training_structure']
    require([r[k] for k in a1.STRUCTURE] == [exp[k] for k in a1.STRUCTURE]
            and [e['step'] for e in r['evaluations']] == exp['evaluation_steps'],
            'the step structure is not the contract\'s; nothing is published')
    require(result['p1']['replay']['mu'] == MU, 'the run did not use the contract mu')
    same = same_draws(run['summary']['fingerprints'], base['fingerprints'])
    require(all(same.values()), f'S2 {args.seed}: {[k for k, v in same.items() if not v]} differ from the mu 0.5 run of '
                                f'the seed (only mu may change); nothing is published')
    inputs = run_inputs(run, rows_rec)
    git = guard.start['git']
    record = {'format': 'h7r-preserve-run-v1', **log_field(args, 'train'), **stamp(sha, eid, git), 'arm': 'S2',
              'mu': MU, 'seed': args.seed, 'config': tr.config_dict(cfg), 'inputs': inputs, 'settings': run['settings'],
              'gate_sha256': file_sha256(root / c['output']['gate']),
              'base_record_sha256': c['base_runs'][str(args.seed)]['record_sha256'],
              'base_fingerprints_same': same,
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
                'policy_rows_sha256': inputs['policy_rows_sha256'], 'c2_policy_rows_sha256': inputs['c2_policy_rows_sha256'],
                's400_records_sha256': inputs['s400_records_sha256'], 'source_models': inputs['source']['models'],
                'best': r['best'], 'teacher_guard_top1': r['teacher_guard']['top1'],
                'preserve': {'contract_id': CONTRACT_ID, 'experiment_sha256': eid, 'arm': 'S2', 'mu': MU,
                             'seed': args.seed, 'replay': result['p1']['replay']},
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


def verify_run(args, root=ROOT) -> dict:
    """A committed run record's files: a CANDIDATE names its checkpoint and metadata, which must be those files now;
    a NO_CANDIDATE wrote none (``h7r_attempts.check_train_files`` on this contract's paths)."""
    root = Path(root)
    rel = run_record_path(args.seed)
    rec = json.loads((root / rel).read_text(encoding='utf-8'))
    require(rec.get('format') == 'h7r-preserve-run-v1' and rec.get('seed') == args.seed, f'{rel}: not the run of '
                                                                                         f'seed {args.seed}')
    out = root / RUNS / f'S2-{args.seed}'
    if rec.get('verdict') == 'CANDIDATE':
        cand = rec.get('candidate') or {}
        for name, key in (('best.pt', 'checkpoint_sha256'), ('best.json', 'metadata_sha256')):
            f = out / name
            require(f.is_file(), f'{RUNS}/S2-{args.seed}/{name} is missing: {rel} records {str(cand.get(key))[:12]}...')
            require(file_sha256(f) == cand.get(key), f"{RUNS}/S2-{args.seed}/{name}: its SHA-256 is not the one {rel} "
                                                     f"records")
    else:
        require(rec.get('verdict') == 'NO_CANDIDATE' and 'candidate' not in rec, f"{rel}: verdict {rec.get('verdict')!r}")
        left = [n for n in ('best.pt', 'best.json', 'logits.pt') if (out / n).exists()]
        require(not left, f'{rel} is NO_CANDIDATE, yet {left} exist')
    return {'seed': args.seed, 'verdict': rec['verdict']}


# --- the Arena ----------------------------------------------------------------------------------------------------

def arm_name(suite: str, model: str) -> str:
    return f'{suite}_{model}_vs_champion'


def arena_contract_body(c: dict, eid: str, root=ROOT) -> dict:
    from hybrid import h7r_seeds as seeds
    import h7r_dev_suites_2 as ds2

    root = Path(root)
    s4, _, _ = s400(c, root)
    rec = committed_record(root, run_record_path(ARENA_SEED), f'S2 {ARENA_SEED} run record', eid)
    require(rec['verdict'] == 'CANDIDATE' and 'candidate' in rec,
            f'S2 {ARENA_SEED}: no candidate (no other seed substituted)')
    ref = c['base_runs'][str(ARENA_SEED)]
    value = p1.PATHS['c5_value']
    v = s4['inputs']['value_c5']
    models = {'champion': {'policy': p1.PATHS['h3'], 'value': value,
                           'identity': {**s4['inputs']['parent_and_teacher'], **v}},
              'S2': {'policy': f'{RUNS}/S2-{ARENA_SEED}/best.pt', 'value': value,
                     'identity': {'policy_checkpoint_sha256': rec['candidate']['checkpoint_sha256'],
                                  'policy_metadata_sha256': rec['candidate']['metadata_sha256'], **v}},
              'S05': {'policy': ref['checkpoint'], 'value': value,
                      'identity': {'policy_checkpoint_sha256': ref['checkpoint_sha256'],
                                   'policy_metadata_sha256': ref['metadata_sha256'], **v}}}
    record = json.loads((root / c['suites']['record']).read_text(encoding='utf-8'))
    require(record.get('format') == ds2.FORMAT, 'the suites record is not the dev-suites v2 record')
    arms = []
    for spec in c['suites']['specs']:
        suite = {'path': spec['path'], 'sha256': file_sha256(root / spec['path'])}
        label = spec['name'].rsplit('-', 1)[1]
        require(record['suite_files_sha256'][label] == suite['sha256'], f"{spec['path']}: not the recorded suite")
        for model in ARMS:
            arms.append({'name': arm_name(spec['name'], model), 'tested': {'model': model, 'budget': 1},
                         'opponent': {'model': 'champion', 'budget': 1}, 'pairs': spec['pairs'],
                         'seed': c['seeds']['arena_games'][spec['name']], 'pairing': spec['name'], 'suite': suite})
    return {'format': 'h7r-eval-contract-v2', 'seed_policy': seeds.POLICY, 'contract_id': ARENA_ID, 'purpose': 'pilot',
            'status': 'fixed',
            'meaning': f'the Arena of {CONTRACT_ID} (experiment {eid[:12]}): development reading; never a promotion',
            'models': models, 'arms': arms, 'stats': STATS,
            'preserve': {'contract_id': CONTRACT_ID, 'experiment_sha256': eid,
                         'candidate_record_sha256': file_sha256(root / run_record_path(ARENA_SEED)),
                         'base_record_sha256': ref['record_sha256']}}


def committed_suites(c: dict, root=ROOT) -> dict:
    """The new suites, committed and recomputed from their sources (``h7r_dev_suites_2.check``)."""
    import h7r_dev_suites_2 as ds2
    root = Path(root)
    for rel in [c['suites']['record'], *(s['path'] for s in c['suites']['specs'])]:
        require((root / rel).is_file() and committed(root / rel, root), f'{rel}: not committed')
    return ds2.check(None, root, log=lambda *a, **k: None)


def arena_contract(args, root=ROOT) -> dict:
    from hybrid import h7r_eval_v2 as ev2
    root = Path(root)
    c, sha, eid = load_contract(args.contract, root, 'arena')
    committed_suites(c, root)
    path = root / ARENA_CONTRACT
    require(not path.exists(), f'{path} exists: never overwritten')
    guard = Guard(root, contract_path(args.contract, root), c, [run_record_path(ARENA_SEED), c['suites']['record']])
    body = arena_contract_body(c, eid, root)
    ev2.check_contract(body, path, root, check_files=False)
    ev2.check_unique(body, path, root)  # seeds and id unique across v1 and v2, before anything is written
    guard.check()
    p1.publish(path, body)
    return body


def load(args, root=ROOT) -> dict:
    """This contract (``approved.arena``) and the committed v2 Arena contract it gives, exactly."""
    from hybrid import h7r_eval_v2 as ev2
    root = Path(root)
    c, sha, eid = load_contract(getattr(args, 'contract', None), root, 'arena')
    apath = root / ARENA_CONTRACT
    require(apath.is_file() and committed(apath, root), f'{ARENA_CONTRACT}: the Arena contract is not committed')
    arena, asha = ev2.load_contract(apath, root, check_files=False)
    require(arena == json.loads(json.dumps(arena_contract_body(c, eid, root))),
            'the Arena contract is not the one this contract and the committed records give')
    return {'c': c, 'sha': sha, 'eid': eid, 'arena': arena, 'asha': asha}


def check_contract_file(args, root=ROOT) -> dict:
    """The Arena contract file as ``arena-contract`` wrote it, before or after its commit."""
    from hybrid import h7r_eval_v2 as ev2
    root = Path(root)
    c, sha, eid = load_contract(getattr(args, 'contract', None), root, 'arena')
    apath = root / ARENA_CONTRACT
    require(apath.is_file(), f'{ARENA_CONTRACT}: no Arena contract')
    arena = json.loads(apath.read_text(encoding='utf-8'))
    require(arena == json.loads(json.dumps(arena_contract_body(c, eid, root))),
            f'{ARENA_CONTRACT}: not the one this contract and the committed records give')
    ev2.check_contract(arena, apath, root, check_files=False)
    ev2.check_unique(arena, apath, root)
    return {'arena_contract_sha256': file_sha256(apath), 'committed': committed(apath, root)}


def models_command(args, root=ROOT, log=print) -> dict:
    a = load(args, root)
    files = sa.check_models(a['arena'], root)
    for name, sums in files.items():
        log(f"[preserve-arena] {name}: policy {sums['policy_checkpoint_sha256'][:12]} / "
            f"{sums['policy_metadata_sha256'][:12]}, value {sums['value_checkpoint_sha256'][:12]}: as contracted",
            flush=True)
    return {'models_checked': sorted(files), 'arena_contract_sha256': a['asha']}


_play_chunk = cli._play_chunk


def play(args, log=print, root=ROOT) -> dict:
    """``scripts/h7r_eval.py play`` on the v2 Arena contract (the loop of ``h7r_s400_arena.play``)."""
    import h7c_arena as ar
    from hybrid import h7r_eval as ev
    from hybrid import h7r_openings as op
    from hybrid.progress import Progress, run_chunks

    root = Path(root)
    git = cli._git(args.allow_dirty)
    a = load(args, root)
    arena, asha = a['arena'], a['asha']
    sa.check_models(arena, root)
    arm = cli._arm(arena, args.arm)
    openings = op.pick(cli._suite(arm, root), arm['pairs'], arm['seed'])
    identity = cli.arm_identity(arena, asha, arm, args.device, git)
    sig = cli._sha_json(identity)
    tasks = cli.build_tasks(arm, openings)
    jsonl = Path(args.jsonl)
    finished = ar.load_finished(jsonl, {t['key'] for t in tasks}, sig, log=log)
    cli.check_games(list(finished.values()), tasks, arm['name'], sig)
    side = lambda s: {'policy': str(root / arena['models'][arm[s]['model']]['policy']),  # noqa: E731
                      'value': str(root / arena['models'][arm[s]['model']]['value']), 'budget': arm[s]['budget']}
    sides = {'tested': side('tested'), 'opponent': side('opponent')}
    context = {'models': {'candidate': sides['tested'], 'parent': sides['opponent']}, 'sides': sides,
               'device': args.device, 'signature': sig, 'arm': arm['name']}
    todo = [{**t, 'context': context} for t in tasks if t['key'] not in finished]
    log(f"[preserve-arena] {arena['contract_id']} {arm['name']}: {len(tasks)} games ({len(finished)} done), "
        f"workers {args.workers}, signature {sig[:12]}", flush=True)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if todo:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        from multiprocessing import get_context

        pool_of = (lambda: ThreadPoolExecutor(max_workers=1)) if getattr(args, 'threads_for_tests', False) else (
            lambda: ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')))
        with jsonl.open('a', encoding='utf-8') as out, pool_of() as pool:
            def keep(_, rows):
                cli.check_games(rows, tasks, arm['name'], sig)
                for row in rows:
                    out.write(json.dumps(row, separators=(',', ':')) + '\n')
                    finished[row['key']] = row
                out.flush()
            run_chunks(pool, _play_chunk, [[t] for t in todo], Progress('preserve-arena', len(todo), 'games'), len,
                       unit_name='games', on_result=keep)
    games = [finished[t['key']] for t in tasks]
    record = {'format': f'{cli.ARENA_FORMAT}-arm', 'identity': identity, 'signature': sig, 'openings': openings,
              'wall_seconds': round(perf_counter() - started, 1), 'workers': args.workers,
              'games_played_now': len(todo), 'games': games}
    record['summary'] = ev.arm_summary(ev.pair_table(games, openings), arena['stats'])
    return record


def suites_of(arena: dict) -> list[str]:
    names = []
    for arm in arena['arms']:
        if arm['pairing'] not in names:
            names.append(arm['pairing'])
    for s in names:
        got = sorted(a['name'] for a in arena['arms'] if a['pairing'] == s)
        require(got == sorted(arm_name(s, m) for m in ARMS), f'{s}: the arms are not S2 and S05: {got}')
    return names


def paired_strata(arena: dict, tables: dict) -> dict:
    """suite -> [(S2 pair score, S05 pair score, class id)]: only when both candidates played the same openings in the
    same order, with the same classes and the arm's full number of pairs."""
    pairs = {a['name']: a['pairs'] for a in arena['arms']}
    out = {}
    for s in suites_of(arena):
        an, bn = arm_name(s, 'S2'), arm_name(s, 'S05')
        ap, bp = tables[an]['pairs'], tables[bn]['pairs']
        require(len(ap) == pairs[an] and len(bp) == pairs[bn],
                f'{s}: {len(ap)} S2 and {len(bp)} S05 pairs of {pairs[an]} (the sample is never reduced)')
        require([p['opening_id'] for p in ap] == [p['opening_id'] for p in bp],
                f'{s}: S2 and S05 were not played on the same openings in the same order')
        require([p['class_id'] for p in ap] == [p['class_id'] for p in bp], f'{s}: S2 and S05 openings differ in class')
        out[s] = [(x['pair_score'], y['pair_score'], x['class_id']) for x, y in zip(ap, bp)]
    return out


def readings(bounds: dict) -> dict:
    lab = sa.readings({"P'": bounds["P''"], "D'": bounds["D''"]})
    return {"P''": lab["P'"], "D''": lab["D'"]}


def bounds_of(strata: dict, unit: str) -> dict:
    b = sa.bounds(strata, {'alpha_one_sided': STATS['alpha_one_sided'], 'bootstrap': STATS['bootstrap']}, unit)
    return {"P''": b["P'"], "D''": b["D'"], 'unit': b['unit'], 'units': b['units']}


def use(lab: dict) -> dict:
    """The section 7 table (fixed before any result)."""
    p = lab["P''"] == 'STRONGER'
    effect = {'BETTER': 'the stronger preservation is recorded as an improvement over the old mu 0.5 candidate',
              'WORSE': 'the stronger preservation is recorded as worse than the old mu 0.5 candidate'}.get(lab["D''"])
    worse = [k for k, v in lab.items() if v in ('WEAKER', 'WORSE')]
    if p:
        return {'decision': 'MAY_OPEN_CONFIRMATION', 'worse': worse,
                'note': 'a separate confirmation contract (new sample and seeds) may open; ' +
                        (effect or 'the effect of the stronger preservation stays undetermined')}
    return {'decision': 'END', 'worse': worse,
            'note': (effect + '; ' if effect else '') + 'the contract ends'}


def compute(a: dict, arms_dir, root) -> dict:
    arena, asha = a['arena'], a['asha']
    files = {arm['name']: Path(arms_dir) / f"{arm['name']}.json" for arm in arena['arms']}
    arm_sha = {name: file_sha256(path) for name, path in files.items()}
    gits = {name: (json.loads(path.read_text(encoding='utf-8')).get('identity') or {}).get('git') or {}
            for name, path in files.items()}
    commits = {g.get('commit') for g in gits.values()}
    require(len(commits) == 1 and None not in commits and all(g.get('dirty') is False for g in gits.values()),
            f'the arms were not played from one clean code commit: {gits}')
    arms = {}
    for arm in arena['arms']:
        loaded = cli.load_arm(files[arm['name']], arena, asha, Path(root))
        require(loaded['table']['valid'], f"{arm['name']}: invalid pairs (an opponent illegal move)")
        require(loaded['record']['identity']['git'] == gits[arm['name']], f"{arm['name']}: changed during the judge")
        arms[arm['name']] = loaded
    strata = paired_strata(arena, {n: v['table'] for n, v in arms.items()})
    main, sens = bounds_of(strata, 'opening'), bounds_of(strata, 'class')
    lab, sens_lab = readings(main), readings(sens)
    require({name: file_sha256(path) for name, path in files.items()} == arm_sha,
            'an arm record changed during the judge; nothing is published')
    return {'format': 'h7r-preserve-judge-v1', 'contract_id': CONTRACT_ID, 'experiment_sha256': a['eid'],
            'contract_file_sha256': a['sha'], 'arena_contract_id': arena['contract_id'], 'arena_contract_sha256': asha,
            'arm_records_sha256': arm_sha, 'arms_commit': commits.pop(), 'readings': lab, 'bounds': main,
            'definitions': {"D''": 'S(mu 2.0) - S(mu 0.5) pair scores on the same openings, suites 50:50',
                            "P''": 'S(mu 2.0) vs champion pair score - 0.5, suites 50:50'},
            'sensitivity_class_level': {'bounds': sens, 'readings': sens_lab, 'same_as_main': sens_lab == lab},
            'per_suite': {s: {m: arms[arm_name(s, m)]['summary'] for m in ARMS} for s in strata},
            'use': use(lab), 'scope': 'the pre-designated training seed 20261024; never a promotion'}


def judge(args, root=ROOT) -> dict:
    root = Path(root)
    require(not (root / JUDGE).exists(), f'{JUDGE} exists: never overwritten (verify-judge records an uncommitted one)')
    a = load(args, root)
    guard = Guard(root, contract_path(getattr(args, 'contract', None), root), a['c'],
                  [run_record_path(ARENA_SEED), ARENA_CONTRACT])
    body = compute(a, args.arms_dir, root)
    git = guard.check()
    require(git.get('commit') == body['arms_commit'], f"the judge runs at {git.get('commit')}, the arms were played "
                                                      f"at {body['arms_commit']}: judge from the arms' commit")
    record = {**body, 'git': git}
    p1.publish(root / JUDGE, record)
    return record


def verify_judge(args, root=ROOT) -> dict:
    root = Path(root)
    path = root / JUDGE
    require(path.is_file(), f'{JUDGE}: no judge record')
    require(not committed(path, root), f'{JUDGE}: already committed')
    found = json.loads(path.read_text(encoding='utf-8'))
    again = compute(load(args, root), args.arms_dir, root)
    git = found.get('git') or {}
    require({k: v for k, v in found.items() if k != 'git'} == again,
            f'{JUDGE}: not the judgment its arm records give (recomputed)')
    require(git.get('commit') == again['arms_commit'] and git.get('dirty') is False,
            f"{JUDGE}: not judged from a clean tree at the arms' commit")
    return found


# --- main ---------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--contract', default=None)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'gate'):
        s = sub.add_parser(name)
        s.add_argument('--device', default='cuda')
        s.add_argument('--log-name', default=None)
    t = sub.add_parser('train')
    t.add_argument('--seed', required=True, type=int)
    t.add_argument('--device', default='cuda')
    t.add_argument('--log-name', default=None)
    v = sub.add_parser('verify-run')
    v.add_argument('--seed', required=True, type=int)
    for name in ('arena-contract', 'check-contract', 'check-models'):
        sub.add_parser(name)
    p = sub.add_parser('play')
    p.add_argument('--arm', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--jsonl', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--allow-dirty', action='store_true', help='tests only')
    for name in ('judge', 'verify-judge'):
        j = sub.add_parser(name)
        j.add_argument('--arms-dir', default=ARMS_DIR)
    args = parser.parse_args(argv)
    try:
        if args.command == 'play':
            out_path = Path(args.output)
            require(not out_path.exists(), f'{out_path} exists: an arm record is never overwritten')
            record = play(args)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            from hybrid.h7r_ledger import publish_json
            publish_json(out_path, record)
            out = {'arm': args.arm, 'summary': {k: record['summary'][k] for k in ('pairs', 'score', 'valid')}}
        else:
            fn = {'prepare': prepare, 'gate': gate, 'train': train, 'verify-run': verify_run,
                  'arena-contract': arena_contract, 'check-contract': check_contract_file,
                  'check-models': models_command, 'judge': judge, 'verify-judge': verify_judge}[args.command]
            rec = fn(args)
            keys = ('passed', 'differences', 'verdict', 'best', 'selected_state_sha256', 'seconds', 'readings',
                    'bounds', 'use', 'models_checked', 'arena_contract_sha256', 'committed', 'seed')
            out = {k: rec[k] for k in keys if k in rec} or {'written': True}
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', flush=True)
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str), flush=True)
    if args.command == 'gate' and not out.get('passed'):
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
