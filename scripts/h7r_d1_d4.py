"""R2-D1-D4-1: does the full-scale D1 widened head's H-dev prediction gain carry over to move selection?

The D4 method of R2-D4-BASELINE-1 (the sealed H-dev self-play of c5, every eligible ply searched by every model with
the actor's seeds and budget, disagreements continued by Rapfi, the continuation-score statistics, the INVALID policy)
on the D1 direct-control pairs: wide_s : narrow_s for the three seeds (primary), wide_s : c5 (reference only).

Differences from ``scripts/h7r_d4.py`` (whose functions are reused unchanged where they are generic):
    models       c5 from the R2 contract; the six full-scale D1 runs linked at run time from their committed run records
                 (the T-val selected checkpoint of each run): the contract is fixed before training ends and never
                 edited for the SHAs, and the linked identities go into the run identity
    factory      every model searches through candidate factory v2 (``analysis.tb_v3_net_v2``); c5 must still
                 reproduce every actor move (the actor check), which tests v2 on the real H-dev
    entry        the committed full-scale fit record is comparable, the committed full-scale H-dev reading is
                 DEVELOPMENT_SIGNAL and the committed time gate passed for the linked seed-31001 pair; full also needs
                 the committed smoke record
    approvals    in this contract (``approved.smoke`` / ``approved.full``), linked by the approval-independent id
    reading      ALL_THREE_BETTER / ALL_THREE_WORSE / MIXED_DIRECTION / PARTIAL_SIGNAL / NOT_ESTABLISHED on the three
                 primary pairs (the baseline rule); only ALL_THREE_BETTER opens the confirmation phase

    prepare                       the eligible manifest of the sealed H-dev (no outcome-dependent selection)
    run --mode smoke|full         replay, Rapfi continuations, statistics (full), summary; resumable journals
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
import platform
from pathlib import Path
from random import Random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import h7r_d4 as base  # noqa: E402
from hybrid import h7r_d4 as d4  # noqa: E402
from hybrid.h7c_nmask import committed  # noqa: E402
from hybrid.h7r_ledger import file_lock, publish_json  # noqa: E402

FORMAT = 'h7r-d1-d4-contract-v1'
CONTRACT_ID = 'R2-D1-D4-1'
SEEDS = ('31001', '31002', '31003')
PRIMARY = tuple(f'wide{s}:narrow{s}' for s in SEEDS)
REFERENCE = tuple(f'wide{s}:c5' for s in SEEDS)
COMPARISONS = (*PRIMARY, *REFERENCE)
MODELS = ('c5', *(f'{arm}{s}' for arm in ('narrow', 'wide') for s in SEEDS))
MODES = ('smoke', 'full')
TOOL_PATH = 'scripts/h7r_d1_d4.py'
KEYS = {'format', 'contract_id', 'status', 'approved', 'purpose', 'delegation', 'baseline', 'd1_full', 'hdev',
        'frozen_state_sha256', 'c5', 'model_link', 'factory', 'comparisons', 'selection', 'budget', 'device',
        'stats', 'invalid_policy', 'runtime_identity', 'rapfi_program', 'rapfi_identity', 'smoke', 'entry',
        'reading', 'sources', 'output', 'scope'}


def experiment_sha256(c: dict) -> str:
    return d4.digest({k: v for k, v in c.items() if k != 'approved'})


def load_contract(path, root=ROOT, mode=None, *, check_sources=True) -> tuple[dict, str, str]:
    path = Path(path)
    c = json.loads(path.read_text(encoding='utf-8'))
    d4.require(isinstance(c, dict) and set(c) == KEYS, f'{path}: the D1 D4 contract keys differ')
    d4.require(c['format'] == FORMAT and c['contract_id'] == CONTRACT_ID and c['status'] == 'fixed',
               f'{path}: not the fixed D1 D4 contract')
    d4.require(set(c['approved']) == set(MODES) and all(type(v) is bool for v in c['approved'].values())
               and not (c['approved']['full'] and not c['approved']['smoke']), 'D1 D4 approvals: smoke before full')
    d4.require(c['comparisons'] == {'primary': list(PRIMARY), 'reference': list(REFERENCE)},
               'D1 D4 comparisons: wide:narrow (primary) and wide:c5 (reference) for the three seeds')
    d4.require(c['stats'] == d4.STATS and c['invalid_policy'] == d4.POLICY, 'D1 D4 statistics / INVALID policy differ '
                                                                             'from the baseline')
    d4.require(c['selection'] == 'all_cache_games_and_rows' and c['budget'] == 'TB-V3-fast5-S50-original-actor-seeds'
               and c['device'] == 'cuda', 'D1 D4 sample, budget or device differ from the baseline')
    d4.require(c['factory'] == {'module': 'analysis.tb_v3_net_v2', 'version': 'tb-v3-net-v2'},
               'D1 D4 factory: candidate factory v2')
    d4.require(c['smoke'] == {'games': 2, 'branches': 8}, 'D1 D4 smoke: the baseline smoke size')
    baseline = json.loads((Path(root) / c['baseline']['path']).read_text(encoding='utf-8'))
    d4.require(d4.file_sha256(Path(root) / c['baseline']['path']) == c['baseline']['sha256'],
               'the baseline D4 contract is not the pinned one')
    for key in ('hdev', 'frozen_state_sha256', 'runtime_identity', 'rapfi_program', 'rapfi_identity'):
        d4.require(c[key] == baseline[key], f'D1 D4 {key}: not the baseline value')
    d4.require(c['c5'] == baseline['models']['c5'], 'D1 D4 c5: not the baseline c5')
    d4.require(d4.file_sha256(Path(root) / c['delegation']['path']) == c['delegation']['sha256'],
               'the delegation record is not the pinned one')
    import h7r_d1 as cli
    full = json.loads((Path(root) / c['d1_full']['path']).read_text(encoding='utf-8'))
    d4.require(cli.experiment_sha256(full) == c['d1_full']['experiment_sha256'] == c['model_link']['experiment_sha256'],
               'the D1 full-scale contract is not the pinned experiment (approvals aside)')
    if check_sources:
        changed = [r['path'] for r in c['sources'] if d4.file_sha256(Path(root) / r['path']) != r['sha256']]
        d4.require(not changed, f'the checkout differs from the sources D1 D4 was fixed for: {changed}')
    d4.require(committed(path, root), f'{path}: not committed')
    if mode is not None:
        d4.require(c['approved'][mode] is True, f'the D1 D4 {mode} stage is not approved')
    return c, d4.file_sha256(path), experiment_sha256(c)


def committed_json(root, rel, what):
    path = Path(root) / rel
    d4.require(path.is_file() and committed(path, root), f'{rel}: the {what} is not committed')
    return json.loads(path.read_text(encoding='utf-8'))


def linked_models(c: dict, root=ROOT) -> dict:
    """c5 and the six full-scale runs' T-val selected checkpoints, from the committed run records."""
    link = c['model_link']
    models = {'c5': dict(c['c5'])}
    policy = {k: c['c5'][k] for k in ('policy_checkpoint_sha256', 'policy_metadata_sha256')}
    for arm in ('narrow', 'wide'):
        for s in SEEDS:
            rec = committed_json(root, link['run_record'].format(arm=arm, seed=s), f'{arm} {s} run record')
            d4.require(rec.get('experiment_sha256') == link['experiment_sha256']
                       and rec['arm'] == arm and rec['seed'] == s and rec['verdict'] == 'CANDIDATE'
                       and rec['selected']['state'] is not None, f'{arm} {s}: not a selected run of the D1 full stage')
            st = rec['selected']['state']
            models[f'{arm}{s}'] = {**policy, 'value_checkpoint_sha256': st['checkpoint_sha256'],
                                   'value_metadata_sha256': st['metadata_sha256']}
    return models


def entry_checks(c: dict, root=ROOT) -> dict:
    """The committed full-scale fit (comparable), H-dev (DEVELOPMENT_SIGNAL) and time gate (passed, for the linked
    seed-31001 pair)."""
    e = c['entry']
    fit = committed_json(root, e['fit'], 'full-scale train-fit record')
    d4.require(fit.get('comparable') is True and fit.get('reading') is not None, 'the full-scale runs are not '
                                                                                 'comparable')
    res = committed_json(root, e['hdev'], 'full-scale H-dev record')
    d4.require(res['hdev_reading']['label'] == 'DEVELOPMENT_SIGNAL',
               f"the full-scale H-dev reading is {res['hdev_reading']['label']}: no D4 (delegation P3 entry)")
    timing = committed_json(root, e['timing'], 'factory v2 time gate record')
    models = linked_models(c, root)
    d4.require(timing['result']['passed'] is True, 'the factory v2 time gate did not pass')
    d4.require(timing['reference']['models']['value_checkpoint_sha256'] == models['narrow31001']['value_checkpoint_sha256']
               and timing['candidate']['models']['value_checkpoint_sha256']
               == models['wide31001']['value_checkpoint_sha256'], 'the time gate was not measured on the linked pair')
    return {'fit_category': fit['reading']['category'], 'hdev_label': res['hdev_reading']['label'],
            'timing': timing['result']}


def checked_models(c: dict, specs, device):
    return base.checked_models({'models': linked_models(c), 'frozen_state_sha256': c['frozen_state_sha256']},
                               specs, device)


def replay_chunk(tasks):
    """The baseline replay with every agent built by candidate factory v2."""
    import h7c_sensitivity as sn
    from analysis import tb_v3_net_v2 as v2
    out = []
    for t in tasks:
        make = lambda side, seed, device=t['device']: v2.make_side(side, seed, device)  # noqa: E731
        out.append({**sn.replay_game(t['game'], t['models'], t['device'], make=make),
                    'identity_sha256': t['identity_sha256']})
    return out


def observations_for(records, games):
    import h7c_adjudicate as ad
    obs = ad.observations(records, games, {'comparisons': [{'pair': p} for p in COMPARISONS]})
    for o in obs:
        o['key_a'], o['key_b'] = (ad.branch_key(o['moves'], o[k]) for k in ('move_a', 'move_b'))
    return obs


def sensitivity(records, counts):
    import statistics
    import h7c_sensitivity as sn
    result = {}
    for pair in COMPARISONS:
        a, b = pair.split(':')
        groups = [[sn.ply_measures(p, a, b) for p in r['plies']] for r in records]
        tree = [m for g in groups for m in g if m['tree']]
        num, den = [sum(m['move_diff'] for m in g) for g in groups], [len(g) for g in groups]
        firsts = [next(m['ply'] for m in g if m['move_diff']) for g in groups if any(m['move_diff'] for m in g)]
        result[pair] = {'different': sum(num), 'plies': sum(den), 'share': sum(num) / sum(den) if sum(den) else None,
                        'ci95': d4.interval(d4.ratio(counts, num, den), .95),
                        'tree_plies': len(tree), 'tree_different': sum(m['move_diff'] for m in tree),
                        'games_with_difference': len(firsts),
                        'first_difference_ply_median': statistics.median(firsts) if firsts else None,
                        'route_differences': sum(m['route_diff'] for g in groups for m in g)}
    return result


def joint_reading(comparisons: dict) -> str:
    """The baseline joint rule on the three D1 primary pairs."""
    rows = [comparisons[name] for name in PRIMARY]
    if all(r['label95'] == 'BETTER' for r in rows):
        return 'ALL_THREE_BETTER'
    if all(r['label95'] == 'WORSE' for r in rows):
        return 'ALL_THREE_WORSE'
    labels = {r['label_individual'] for r in rows}
    if {'BETTER', 'WORSE'} <= labels:
        return 'MIXED_DIRECTION'
    if 'BETTER' in labels and 'WORSE' not in labels:
        return 'PARTIAL_SIGNAL'
    return 'NOT_ESTABLISHED'


def prepare(args):
    base.git_state()
    c, sha, eid = load_contract(args.contract)
    out = ROOT / c['output']['sample']
    d4.require(not out.exists(), f'{out} exists: never overwritten')
    manifest, _, _ = inputs(c, sha, args.cache)
    publish_json(out, manifest)
    print(f"D1 D4 PREPARED {len(manifest['games'])} games / {manifest['rows']} rows; warm-up "
          f"{manifest['warmup_plies']}; SHA {d4.file_sha256(out)}", flush=True)
    return manifest


def inputs(c: dict, sha: str, cache):
    """The baseline's sealed-H-dev checks and eligible manifest, with this contract's c5 as the actor."""
    return base.inputs({**c, 'models': {'c5': c['c5']}}, sha, cache)


def run(args):
    import h7c_adjudicate as ad
    import h7c_sensitivity as sn
    from analysis import tb_v3, tb_v3_fast5
    import torch

    started = time.monotonic()
    c, sha, eid = load_contract(args.contract, mode=args.mode)
    state = base.git_state()
    gates = entry_checks(c)
    d4.require(args.device == c['device'] and args.workers > 0, 'device / workers differ from the contract')
    if args.mode == 'full':
        smoke = committed_json(ROOT, c['output']['records'].format(mode='smoke') + '/summary.json', 'smoke summary')
        d4.require(smoke['identity']['experiment_sha256'] == eid and smoke['reading'] == 'SMOKE'
                   and smoke['actor_check']['reproduced'] is True,
                   'the committed smoke record is not a reproduced smoke of this experiment')
    print('[d1-d4-preflight] sealed H-dev and eligible positions', flush=True)
    manifest, games, checks = inputs(c, sha, args.cache)
    full_manifest = manifest
    sample = committed_json(ROOT, c['output']['sample'], 'eligible manifest')
    d4.require(sample == manifest, 'the committed eligible manifest differs from the sealed H-dev')
    models, identities = checked_models(c, args.model, args.device)
    runtime = {'engine_hash': tb_v3.ENGINE_HASH, 'runtime_hash': tb_v3_fast5.runtime_hash(args.device)}
    d4.require(runtime == c['runtime_identity'], 'engine / runtime differs from the contract')
    d4.require(all(all(g.get(k) == v for k, v in runtime.items()) and g['value_device'] == 'cuda'
                   for g in games.values()), 'source actor engine / runtime / device differs')
    spec = ad.engine_spec(ROOT / c['rapfi_program'], args.rapfi_dir)
    d4.require(spec['identity'] == c['rapfi_identity'], 'Rapfi identity differs')
    folder = (ROOT / c['output']['raw'].format(mode=args.mode)).resolve()
    d4.require(Path(args.out_dir).resolve() == folder, f'use the isolated output directory {folder}')
    print(f"[d1-d4-preflight] PASS {len(games)} games / {manifest['rows']} rows; seven models; entry {gates}",
          flush=True)
    identity = {'format': 'h7r-d1-d4-run-v1', 'experiment_sha256': eid, 'contract_sha256': sha, 'git': state,
                'hdev': checks, 'sample_sha256': d4.file_sha256(ROOT / c['output']['sample']), 'models': identities,
                'factory': c['factory'], 'runtime': runtime, 'rapfi': spec['identity'], 'mode': args.mode,
                'device': args.device, 'entry': gates,
                'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                                'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name()}}
    identity_sha = d4.digest(identity)
    if args.mode == 'smoke':
        keep = set(sorted(games)[:c['smoke']['games']])
        games = {i: g for i, g in games.items() if i in keep}
        manifest = {**manifest, 'games': [e for e in manifest['games'] if e['game_index'] in keep]}
    folder.mkdir(parents=True, exist_ok=True)
    with file_lock(folder / 'run.lock', nonblocking=True):
        d4.require(not (folder / 'summary.json').exists(), 'a completed summary exists; never overwritten')
        ipath = folder / 'identity.json'
        if ipath.exists():
            d4.require(json.loads(ipath.read_text()) == identity, 'resume identity differs')
        else:
            d4.require(not any((folder / n).exists() for n in ('replay.jsonl', 'branches.jsonl')), 'unbound journals')
            publish_json(ipath, identity)
        replay_path, branch_path = folder / 'replay.jsonl', folder / 'branches.jsonl'
        replay_done = base.read_journal(replay_path, 'game_index', identity_sha, set(games))
        tasks = [{'game': g, 'game_index': i, 'models': models, 'device': args.device,
                  'identity_sha256': identity_sha} for i, g in sorted(games.items())]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            replay_done = base.execute(pool, replay_chunk, tasks, replay_path,
                                       lambda r: sn.check_replay(r, games, models, 'c5'), replay_done,
                                       'game_index', 'd1-d4-replay', lambda t: len(t['game']['plies']))
            records = [replay_done[i] for i in sorted(games)]
            filtered = base.eligible_replays(records, manifest)
            obs = observations_for(filtered, games)
            table = ad.branch_table(obs)
            keys = sorted(table)
            Random(c['stats']['order_seed']).shuffle(keys)
            if args.mode == 'smoke':
                keys = keys[:c['smoke']['branches']]
            branch_tasks = [{**table[k], 'order': i, 'identity_sha256': identity_sha, 'spec': spec}
                            for i, k in enumerate(keys)]
            by_key = {t['key']: t for t in branch_tasks}
            done = base.read_journal(branch_path, 'key', identity_sha, set(keys))
            done = base.execute(pool, ad._chunk, branch_tasks, branch_path,
                                lambda r: base.validate_branch(r, by_key[r['key']], c['invalid_policy']),
                                done, 'key', 'd1-d4-rapfi', lambda _: 1)
        d4.require(base.git_state() == state, 'HEAD changed during D4')
        load_contract(args.contract, mode=args.mode)
        d4.require(inputs(c, sha, args.cache)[0] == full_manifest, 'H-dev changed during D4')
        checked_models(c, args.model, args.device)
        d4.require(ad.engine_spec(ROOT / c['rapfi_program'], args.rapfi_dir)['identity'] == spec['identity'],
                   'Rapfi files changed during D4')
        actor = sn.actor_check(records, games, 'c5')
        d4.require(actor['reproduced'], f'c5 does not reproduce the actor through factory v2: {actor}')
        result = {'format': 'h7r-d1-d4-summary-v1', 'identity': identity, 'identity_sha256': identity_sha,
                  'actor_check': actor,
                  'eligible_plies': sum(len(r['plies']) for r in filtered),
                  'warmup_plies': sum(len(r['plies']) for r in records) - sum(len(r['plies']) for r in filtered),
                  'branches': len(done), 'observations': len(obs),
                  'files': {p.name: d4.file_sha256(p) for p in (replay_path, branch_path) if p.exists()},
                  'seconds': time.monotonic() - started,
                  'scope': 'development only; no strength, promotion, champion or V9 ruling'}
        if args.mode == 'smoke':
            result['reading'] = 'SMOKE'
        else:
            counts = d4.multiplicities(len(games), c['stats']['samples'], c['stats']['seed'])
            outcomes = {k: base.validate_branch(r, by_key[k], c['invalid_policy']) for k, r in done.items()}
            result['branch_outcomes'] = {kind: sum(o['kind'] == kind for o in outcomes.values())
                                         for kind in ('resolved', 'structural', 'unresolved')}
            result['sensitivity'] = sensitivity(filtered, counts)
            result['comparisons'] = {p: d4.selection_stats([o for o in obs if o['compare'] == p], outcomes,
                                                          sorted(games), counts, c['stats']) for p in COMPARISONS}
            result['reading'] = joint_reading(result['comparisons'])
            for p in REFERENCE:  # references carry intervals only, no formal label
                for k in ('label95', 'label_individual'):
                    result['comparisons'][p].pop(k)
            result['next'] = ('confirmation (delegation P5)' if result['reading'] == 'ALL_THREE_BETTER' else
                              'stop: record and report (delegation P3)')
        d4.require(base.git_state() == state, 'HEAD changed before D4 publication')
        publish_json(folder / 'summary.json', result)
        print(json.dumps({'reading': result['reading'], 'summary': str(folder / 'summary.json')}, ensure_ascii=False),
              flush=True)
        return result


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='command', required=True)
    for cmd in ('prepare', 'run'):
        s = sub.add_parser(cmd)
        s.add_argument('--contract', default='docs/h7r/r2/R2-D1-HEAD-WIDTH-1/d4/d1-d4-contract.json')
        s.add_argument('--cache', default='runs/h7r/R1-HDEV-1/cache')
        if cmd == 'run':
            s.add_argument('--mode', choices=MODES, required=True)
            s.add_argument('--model', action='append', required=True, metavar='NAME=POLICY,VALUE')
            s.add_argument('--device', default='cuda')
            s.add_argument('--workers', type=int, default=6)
            s.add_argument('--rapfi-dir', required=True)
            s.add_argument('--out-dir', required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        (prepare if args.command == 'prepare' else run)(args)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
