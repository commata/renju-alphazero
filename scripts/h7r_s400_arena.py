"""R2-S400-TRAIN Arena: the v2 play entry, the model-file check and the D' / P' judge.

The training tool (``scripts/h7r_s400_train.py``, pinned by the contract's ``sources``) writes the v2 Arena contract
(``arena-contract``); this tool plays and judges it, so nothing pinned changes. ``scripts/h7r_eval.py play`` loads v1
contracts only (A1 pins that file), so the play here is its loop on a v2 contract: the same task list, games, journal
and arm record (``build_tasks``, ``_play_chunk``, ``check_games``, ``arm_identity``, ``load_arm`` of that file).

Every command loads the S400 contract with ``approved.arena`` and the committed Arena contract, which must be exactly
the one the contract and the committed S 20261024 run record give (``arena_contract_body``).

    check-contract  the Arena contract file (committed or not yet) is exactly the one the S400 contract and the
                  committed S run record give, a valid v2 contract whose seeds and id no other contract uses
    check-models  the four files of every model (policy checkpoint and metadata, value checkpoint and metadata) are
                  the SHA-256 the Arena contract records; run before any game
    play          one arm (``--arm``): models checked first, every opening of the arm twice (colours swapped), games
                  appended to ``--jsonl`` as they finish (resume: the same setup only), the arm record to ``--output``
                  (never overwritten)
    judge         the four arm records re-checked (``load_arm``: signature, contract, models, suite, openings, games,
                  summary), every pair valid and the full sample, one clean code commit for all arms; then per suite
                  the S and T pairs must be the same openings in the same order (and classes): D' = S - T and
                  P' = S - 0.5, suites weighted 50:50, stratified bootstrap (openings within a suite; classes as the
                  sensitivity), one-sided bounds at the contract's alpha; ``docs/.../judge.json``
    verify-judge  an uncommitted judge.json computed again from the arm records: equal (but for its ``git``) or refused

Readings (contract ``reading``): D' BETTER (lower bound > 0) / WORSE (upper bound < 0) / NOT_ESTABLISHED; P' STRONGER /
WEAKER / NOT_ESTABLISHED. P' STRONGER sends the candidate to the independent confirmation stage (no champion change);
otherwise the contract ends. The planning values in the contract are never thresholds.
"""
from __future__ import annotations

import argparse
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
import h7r_s400_train as st  # noqa: E402
from h7r_p1 import committed, file_sha256, require  # noqa: E402
from hybrid import h7r_eval as ev  # noqa: E402
from hybrid import h7r_eval_v2 as ev2  # noqa: E402
from hybrid import h7r_openings as op  # noqa: E402
from hybrid.progress import Progress, run_chunks  # noqa: E402

JUDGE = f'{st.D}/judge.json'
ARMS_DIR = f'{st.RUNS}/arena'
JUDGE_FORMAT = 'h7r-s400-train-judge-v1'
MODEL_FILES = ('policy_checkpoint_sha256', 'policy_metadata_sha256', 'value_checkpoint_sha256', 'value_metadata_sha256')
NEXT = {'STRONGER': "P' STRONGER: the candidate goes to the independent confirmation stage; no champion change here",
        None: "P' is not STRONGER: the contract ends"}


# --- contracts ----------------------------------------------------------------------------------------------------

def load(args, root) -> dict:
    """The S400 contract (``approved.arena``) and the committed v2 Arena contract it gives, exactly."""
    root = Path(root)
    c, sha, eid = st.load_contract(getattr(args, 'contract', None), root, 'arena')
    apath = root / st.ARENA_CONTRACT
    require(apath.is_file() and committed(apath, root), f'{st.ARENA_CONTRACT}: the Arena contract is not committed')
    arena, asha = ev2.load_contract(apath, root, check_files=False)
    expected = json.loads(json.dumps(st.arena_contract_body(c, eid, root)))
    require(arena == expected, 'the Arena contract is not the one the S400 contract and the committed S run record give '
                               '(models, arms, suites, pairs, seeds, pairings, budgets or statistics differ)')
    return {'c': c, 'sha': sha, 'eid': eid, 'arena': arena, 'asha': asha}


def check_contract_file(args, root=ROOT) -> dict:
    """The Arena contract file as ``arena-contract`` wrote it, before or after its commit."""
    root = Path(root)
    c, sha, eid = st.load_contract(getattr(args, 'contract', None), root, 'arena')
    apath = root / st.ARENA_CONTRACT
    require(apath.is_file(), f'{st.ARENA_CONTRACT}: no Arena contract')
    arena = json.loads(apath.read_text(encoding='utf-8'))
    require(arena == json.loads(json.dumps(st.arena_contract_body(c, eid, root))),
            f'{st.ARENA_CONTRACT}: not the one the S400 contract and the committed S run record give')
    ev2.check_contract(arena, apath, root, check_files=False)
    ev2.check_unique(arena, apath, root)
    return {'arena_contract_sha256': file_sha256(apath), 'committed': committed(apath, root)}


def model_files(model: dict) -> dict:
    """The four files of a model pair, keyed like its identity (``h7r_eval.model_identity``)."""
    policy, value = Path(model['policy']), Path(model['value'])
    return {'policy_checkpoint_sha256': policy, 'policy_metadata_sha256': policy.with_suffix('.json'),
            'value_checkpoint_sha256': value, 'value_metadata_sha256': value.with_name('best.json')}


def check_models(arena: dict, root) -> dict:
    """Every model file is the SHA-256 the Arena contract records (a missing file is a difference)."""
    root = Path(root)
    bad, out = [], {}
    for name, model in sorted(arena['models'].items()):
        out[name] = {}
        for key, rel in model_files(model).items():
            path = root / rel
            actual = file_sha256(path) if path.is_file() else None
            out[name][key] = actual
            if actual != model['identity'][key]:
                bad.append(f"{name} {rel.as_posix()}: {'missing' if actual is None else actual[:12] + '...'} "
                           f"(the contract records {model['identity'][key][:12]}...)")
    require(not bad, 'model files differ from the Arena contract: ' + '; '.join(bad))
    return out


def models_command(args, root=ROOT, log=print) -> dict:
    a = load(args, root)
    files = check_models(a['arena'], root)
    for name, sums in files.items():
        log(f"[s400-arena] {name}: policy {sums['policy_checkpoint_sha256'][:12]} / "
            f"{sums['policy_metadata_sha256'][:12]}, value {sums['value_checkpoint_sha256'][:12]} / "
            f"{sums['value_metadata_sha256'][:12]}: as contracted", flush=True)
    return {'models_checked': sorted(files), 'arena_contract_sha256': a['asha']}


# --- play ---------------------------------------------------------------------------------------------------------

_play_chunk = cli._play_chunk  # the worker of scripts/h7r_eval.py (tests replace the game, not this)


def play(args, log=print, root=ROOT) -> dict:
    """``scripts/h7r_eval.py play`` on the v2 Arena contract (its loop, tasks, games, journal and record)."""
    import h7c_arena as ar

    root = Path(root)
    git = cli._git(args.allow_dirty)
    a = load(args, root)
    arena, asha = a['arena'], a['asha']
    check_models(arena, root)
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
    log(f"[s400-arena] {arena['contract_id']} {arm['name']}: {len(tasks)} games ({len(finished)} done), "
        f"workers {args.workers}, signature {sig[:12]}", flush=True)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if todo:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        from multiprocessing import get_context

        # spawn: a forked worker cannot re-initialise CUDA; threads only for tests (in-process fakes)
        pool_of = (lambda: ThreadPoolExecutor(max_workers=1)) if getattr(args, 'threads_for_tests', False) else (
            lambda: ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')))
        with jsonl.open('a', encoding='utf-8') as out, pool_of() as pool:
            def keep(_, rows):
                cli.check_games(rows, tasks, arm['name'], sig)
                for row in rows:
                    out.write(json.dumps(row, separators=(',', ':')) + '\n')
                    finished[row['key']] = row
                out.flush()
            run_chunks(pool, _play_chunk, [[t] for t in todo], Progress('s400-arena', len(todo), 'games'), len,
                       unit_name='games', on_result=keep)
    games = [finished[t['key']] for t in tasks]
    record = {'format': f'{cli.ARENA_FORMAT}-arm', 'identity': identity, 'signature': sig, 'openings': openings,
              'wall_seconds': round(perf_counter() - started, 1), 'workers': args.workers,
              'games_played_now': len(todo), 'games': games}
    record['summary'] = ev.arm_summary(ev.pair_table(games, openings), arena['stats'])
    return record


# --- judge --------------------------------------------------------------------------------------------------------

def suites(arena: dict) -> list[str]:
    """The pairings of the Arena contract in their order; each has exactly one S arm and one T arm."""
    names = []
    for arm in arena['arms']:
        if arm['pairing'] not in names:
            names.append(arm['pairing'])
    for s in names:
        got = sorted(a['name'] for a in arena['arms'] if a['pairing'] == s)
        require(got == sorted(f'{s}_{m}_vs_champion' for m in ('S', 'T')), f'{s}: the arms are not S and T: {got}')
    return names


def paired_strata(arena: dict, tables: dict) -> dict:
    """suite -> [(S pair score, T pair score, class id)]: only when S and T played the same openings in the same
    order, with the same classes, and the arm's full number of pairs."""
    pairs = {a['name']: a['pairs'] for a in arena['arms']}
    out = {}
    for s in suites(arena):
        sn, tn = f'{s}_S_vs_champion', f'{s}_T_vs_champion'
        sp, tp = tables[sn]['pairs'], tables[tn]['pairs']
        require(len(sp) == pairs[sn] and len(tp) == pairs[tn],
                f'{s}: {len(sp)} S and {len(tp)} T pairs of {pairs[sn]} (the sample is never reduced)')
        require([p['opening_id'] for p in sp] == [p['opening_id'] for p in tp],
                f'{s}: S and T were not played on the same openings in the same order')
        require([p['class_id'] for p in sp] == [p['class_id'] for p in tp], f'{s}: S and T openings differ in class')
        out[s] = [(x['pair_score'], y['pair_score'], x['class_id']) for x, y in zip(sp, tp)]
    return out


def bounds(strata: dict, statistics: dict, unit: str) -> dict:
    b = statistics['bootstrap']
    raw = p1.stratified_bounds(strata, b['samples'], b['seed'], statistics['alpha_one_sided'], unit=unit)
    return {"P'": raw['P'], "D'": raw['D'], 'unit': raw['unit'], 'units': raw['units']}


def readings(b: dict) -> dict:
    lab = lambda x, hi, lo: hi if x['lower_one_sided'] > 0 else lo if x['upper_one_sided'] < 0 \
        else 'NOT_ESTABLISHED'  # noqa: E731
    return {"P'": lab(b["P'"], 'STRONGER', 'WEAKER'), "D'": lab(b["D'"], 'BETTER', 'WORSE')}


def compute(a: dict, arms_dir, root) -> dict:
    """The judgment of the arm records in ``arms_dir`` (everything but the git stamp)."""
    arena, asha, c = a['arena'], a['asha'], a['c']
    files = {arm['name']: Path(arms_dir) / f"{arm['name']}.json" for arm in arena['arms']}
    arm_sha = {name: file_sha256(path) for name, path in files.items()}
    # one clean code commit for every arm (read first; load_arm then checks each identity against its signature)
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
    tables = {name: v['table'] for name, v in arms.items()}
    strata = paired_strata(arena, tables)
    statistics = c['reading']['statistics']
    main, sens = bounds(strata, statistics, 'opening'), bounds(strata, statistics, 'class')
    lab, sens_lab = readings(main), readings(sens)
    require({name: file_sha256(path) for name, path in files.items()} == arm_sha,
            'an arm record changed during the judge; nothing is published')
    return {'format': JUDGE_FORMAT, 'contract_id': st.CONTRACT_ID, 'experiment_sha256': a['eid'],
            'contract_file_sha256': a['sha'], 'arena_contract_id': arena['contract_id'], 'arena_contract_sha256': asha,
            'arm_records_sha256': arm_sha, 'arms_commit': commits.pop(),
            'readings': lab, 'bounds': main,
            'definitions': {"D'": 'S - T pair scores on the same openings, suites weighted 50:50',
                            "P'": 'S vs champion pair score - 0.5, suites weighted 50:50'},
            'sensitivity_class_level': {'bounds': sens, 'readings': sens_lab, 'same_as_main': sens_lab == lab},
            'per_suite': {s: {m: arms[f'{s}_{m}_vs_champion']['summary'] for m in ('S', 'T')} for s in strata},
            'next': NEXT.get(lab["P'"], NEXT[None]),
            'note': 'development reading of one contract seed (20261024); never a promotion'}


def judge(args, root=ROOT) -> dict:
    root = Path(root)
    require(not (root / JUDGE).exists(), f'{JUDGE} exists: never overwritten (verify-judge records an uncommitted one)')
    a = load(args, root)
    guard = p1.Guard(root, st.contract_path(getattr(args, 'contract', None), root), a['c'],
                     [st.run_record_path(a['c']['seeds']['arena_candidate_training_seed']), st.ARENA_CONTRACT])
    body = compute(a, args.arms_dir, root)
    git = guard.check()
    require(git.get('commit') == body['arms_commit'], f"the judge runs at {git.get('commit')}, the arms were played "
                                                      f"at {body['arms_commit']}: judge from the arms' commit")
    record = {**body, 'git': git}
    p1.publish(root / JUDGE, record)
    return record


def verify_judge(args, root=ROOT) -> dict:
    """An uncommitted judge.json is recorded only when the arm records give it again (its git stamp aside, which
    must be a clean tree at the arms' commit)."""
    root = Path(root)
    path = root / JUDGE
    require(path.is_file(), f'{JUDGE}: no judge record')
    require(not committed(path, root), f'{JUDGE}: already committed')
    found = json.loads(path.read_text(encoding='utf-8'))
    a = load(args, root)
    again = compute(a, args.arms_dir, root)
    git = found.get('git') or {}
    require({k: v for k, v in found.items() if k != 'git'} == again,
            f'{JUDGE}: not the judgment its arm records give (recomputed)')
    require(git.get('commit') == again['arms_commit'] and git.get('dirty') is False,
            f'{JUDGE}: not judged from a clean tree at the arms\' commit')
    return found


# --- main ---------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--contract', default=None)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check-contract')
    sub.add_parser('check-models')
    p = sub.add_parser('play')
    p.add_argument('--arm', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--jsonl', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--allow-dirty', action='store_true', help='tests only: every H7-R record names a clean commit')
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
        elif args.command == 'check-models':
            out = models_command(args)
        elif args.command == 'check-contract':
            out = check_contract_file(args)
        elif args.command == 'judge':
            rec = judge(args)
            out = {k: rec[k] for k in ('readings', 'bounds', 'next')}
        else:
            rec = verify_judge(args)
            out = {'verified': True, 'readings': rec['readings']}
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', flush=True)
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
