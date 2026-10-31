"""D4 H-dev development wrapper. prepare is read-only input verification; run needs separate committed approval.

Reuses the H7-c candidate factory, deterministic actor replay and fresh-process Rapfi runner.
Never trains, consumes a confirmation ledger, changes a champion or alters earlier evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
import os
import platform
from pathlib import Path
from random import Random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

from hybrid import h7r_d4 as d4
from hybrid.h7r_ledger import file_lock, publish_json
from hybrid.h7c_nmask import committed
from hybrid.progress import Progress, run_chunks


def git_state(root=ROOT):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    state = {'commit': git('rev-parse', 'HEAD'),
             'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}
    d4.require(not state['dirty'], 'D4 needs a clean tracked tree')
    return state


def contract_at(path, root=ROOT):
    c, sha = d4.load_contract(path, root)
    d4.require(committed(path, root), 'D4 contract must be committed')
    return c, sha


def approval(path, sha, mode, root=ROOT):
    d4.require(path is not None, 'smoke/full needs a separately approved, committed execution file')
    a = json.loads(Path(path).read_text(encoding='utf-8'))
    d4.require(a.get('format') == 'h7r-d4-execution-v1' and a.get('contract_sha256') == sha
               and a.get('mode') == mode and a.get('approved') is True and committed(path, root),
               'execution approval absent, uncommitted or of another contract/mode')
    return d4.file_sha256(path)


def eligible_manifest(contract, sha, games, info, data=None):
    """All cache rows, with no outcome-dependent selection. Replay warm-up is not filtered out here."""
    import h7c_labels as lab
    by_id = {lab.game_id(g): g for g in games}
    d4.require(len(by_id) == len(games), 'duplicate source game IDs')
    d4.require(len(info['game_ids']) == len(set(info['game_ids'])) and
               len(info['game']) == len(info['ply']) and
               all(type(i) is int and 0 <= i < len(info['game_ids']) for i in info['game']),
               'invalid cache game-to-row mapping')
    eligible = {gid: [] for gid in info['game_ids']}
    for index, ply in zip(info['game'], info['ply']):
        gid = info['game_ids'][index]
        d4.require(type(ply) is int and ply >= 5 and ply not in eligible[gid], 'duplicate/invalid eligible row')
        eligible[gid].append(ply)
    listed = []
    indices = set()
    for gid in sorted(eligible):
        d4.require(eligible[gid] and gid in by_id, 'an eligible game is empty or missing from the source')
        g = by_id[gid]
        idx = g['game_index']
        d4.require(idx not in indices, 'duplicate source game index')
        indices.add(idx)
        plies = [p['ply'] for p in g['plies']]
        d4.require(len(plies) == len(set(plies)) and set(eligible[gid]) <= set(plies),
                   'an eligible cache row has no unique source engine ply')
        listed.append({'game_id': gid, 'game_index': idx, 'plies': sorted(eligible[gid])})
    if data is not None:
        from hybrid.h3_cache import pack
        from renju import Game
        row_index = {(info['game_ids'][g], p): i for i, (g, p) in enumerate(zip(info['game'], info['ply']))}
        for e in listed:
            g, board, black, white = by_id[e['game_id']], Game(), [], []
            engine = {p['ply']: p for p in g['plies']}
            for ply, move in enumerate(g['moves']):
                if ply in e['plies']:
                    i = row_index[e['game_id'], ply]
                    expected = {'black': pack(black), 'white': pack(white),
                                'legal': pack([r*15+c for r, c in board.legal_moves()])}
                    d4.require(all(data[k][i].tolist() == list(v) for k, v in expected.items()),
                               'cache stones/legal mask differ from the original self-play position')
                    d4.require(int(data['last'][i]) == g['moves'][ply-1][0]*15+g['moves'][ply-1][1]
                               and engine[ply]['played'] == move
                               and engine[ply]['color'] == ('black' if ply % 2 == 0 else 'white'),
                               'cache last move/source engine ply differs')
                board.play(*move)
                (black if ply % 2 == 0 else white).append(move[0]*15+move[1])
    return {'format': 'h7r-d4-eligible-v1', 'contract_sha256': sha, 'hdev': contract['hdev'],
            'games': listed, 'rows': len(info['ply']),
            'warmup_plies': sum(len(by_id[e['game_id']]['plies']) - len(e['plies']) for e in listed)}


def inputs(contract, sha, cache, root=ROOT):
    import h7r_holdout as ho
    import h7r_r2 as r2cli
    _, data, info, checks = r2cli.check_hdev(contract, Path(cache), root)
    hc, _, games, _ = ho.checked_cache(cache, root)
    d4.require(hc['models'][hc['holdout']['actor']]['identity'] == {
        k: contract['models']['c5'][k] for k in ('policy_checkpoint_sha256', 'policy_metadata_sha256',
                                               'value_checkpoint_sha256', 'value_metadata_sha256')}, 'H-dev actor not c5')
    manifest = eligible_manifest(contract, sha, games, info, data)
    by_index = {g['game_index']: g for g in games}
    selected = {e['game_index']: by_index[e['game_index']] for e in manifest['games']}
    return manifest, selected, checks


def prepare(args):
    git_state()
    c, sha = contract_at(args.contract)
    d4.require(not Path(args.output).exists(), 'prepare output exists')
    manifest, _, _ = inputs(c, sha, args.cache)
    publish_json(args.output, manifest)
    print(f"D4 PREPARED {len(manifest['games'])} games / {manifest['rows']} rows; "
          f"warm-up {manifest['warmup_plies']}; SHA {d4.file_sha256(args.output)}", flush=True)
    return manifest


def checked_models(contract, specs, device):
    from analysis.tb_v3_net import model_identity
    from hybrid.h7r_r2 import check_model
    import h7c_sensitivity as sn
    models = sn.parse_models(specs)
    d4.require(set(models) == set(contract['models']), 'model coverage differs')
    identities = {}
    for name, side in models.items():
        cp = Path(side['value'])
        metadata = cp.with_name('best.json')  # exactly what the existing candidate factory reads
        actual = model_identity(side['policy'], side['value'])
        d4.require(isinstance(actual, dict), f'{name}: missing model identity')
        expected = {k: contract['models'][name][k] for k in actual}
        d4.require(actual == expected, f'{name}: checkpoint/adjacent metadata identity differs')
        loaded, identities[name] = check_model(name, contract['models'][name], cp, metadata,
                                               contract['frozen_state_sha256'], device)
        del loaded
    return models, identities


def read_journal(path, field, identity, expected):
    """Strict completed records; preserve a torn final write before recovering it. Never drop a complete bad row."""
    path = Path(path)
    done = {}
    if not path.exists(): return done
    raw, offset = path.read_bytes(), 0
    lines = raw.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            d4.require(i == len(lines)-1 and not line.endswith(b'\n'), f'{path}: corrupt journal')
            tail = path.with_name(path.name + '.partial-' + d4.digest(list(line))[:12])
            if not tail.exists(): tail.write_bytes(line)
            with path.open('r+b') as handle: handle.truncate(offset)
            return done
        d4.require(record.get('identity_sha256') == identity, f'{path}: resume identity differs')
        key = record.get(field)
        d4.require(key in expected and key not in done, f'{path}: duplicate/extra record')
        done[key] = record
        offset += len(line)
    if raw and not raw.endswith(b'\n'):
        with path.open('ab') as handle: handle.write(b'\n')
    return done


def append_record(path, value):
    with Path(path).open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def replay_chunk(tasks):
    import h7c_sensitivity as sn
    return [{**sn.replay_game(t['game'], t['models'], t['device']),
             'identity_sha256': t['identity_sha256']} for t in tasks]


def execute(pool, fn, tasks, path, validate, done, field, label, amount):
    for record in done.values(): validate(record)
    todo = [t for t in tasks if t[field] not in done]
    print(f"[{label}] resume {len(done)}/{len(tasks)}, remaining {len(todo)}", flush=True)
    if not todo: return done
    def keep(_, records):
        for record in records:
            validate(record)
            append_record(path, record)
            done[record[field]] = record
    run_chunks(pool, fn, [[t] for t in todo], Progress(label, sum(amount(t) for t in todo), 'units'),
               lambda chunk: sum(amount(t) for t in chunk), on_result=keep)
    d4.require(set(done) == {t[field] for t in tasks}, 'incomplete results')
    return done


def eligible_replays(records, manifest):
    allowed = {e['game_index']: set(e['plies']) for e in manifest['games']}
    return [{**r, 'plies': [p for p in r['plies'] if p['ply'] in allowed[r['game_index']]]} for r in records]


def observations_for(records, games):
    import h7c_adjudicate as ad
    obs = ad.observations(records, games, {'comparisons': [{'pair': p} for p in d4.COMPARISONS]})
    for o in obs:
        o['key_a'], o['key_b'] = (ad.branch_key(o['moves'], o[k]) for k in ('move_a', 'move_b'))
    return obs


def validate_branch(record, task, policy):
    from hybrid.h7r_evidence import branch_outcome
    d4.require(all(record.get(k) == task[k] for k in ('key', 'order', 'identity_sha256')), 'wrong branch identity/order')
    d4.require(record.get('mover') == ('black' if len(task['moves']) % 2 == 0 else 'white'), 'wrong fixed mover')
    out = branch_outcome(record, task, policy)
    d4.require(out['kind'] != 'unhandled', 'unhandled INVALID')
    return out


def sensitivity(records, counts):
    import h7c_sensitivity as sn
    result = {}
    for pair in d4.COMPARISONS:
        a, b = pair.split(':')
        groups = [[sn.ply_measures(p, a, b) for p in r['plies']] for r in records]
        one = {}
        for name, pick in [('all', lambda m: True), ('tree', lambda m: m['tree']),
                           ('tree_black', lambda m: m['tree'] and m['color'] == 'black'),
                           ('tree_white', lambda m: m['tree'] and m['color'] == 'white')]:
            rows = [[m for m in g if pick(m)] for g in groups]
            num, den = [sum(m['move_diff'] for m in g) for g in rows], [len(g) for g in rows]
            one[name] = {'different': sum(num), 'plies': sum(den), 'share': sum(num)/sum(den) if sum(den) else None,
                         'ci95': d4.interval(d4.ratio(counts, num, den), .95)}
        firsts = [next(m['ply'] for m in g if m['move_diff']) for g in groups if any(m['move_diff'] for m in g)]
        one['first_difference_ply'] = ({'games': len(firsts), 'median': statistics.median(firsts),
                                        'min': min(firsts), 'max': max(firsts)} if firsts else
                                       {'games': 0, 'median': None, 'min': None, 'max': None})
        tree = [m for g in groups for m in g if m['tree']]
        qs = [m['dq'] for m in tree if m['dq'] is not None]
        all_rows = [m for g in groups for m in g]
        one.update(games_with_difference=sum(any(m['move_diff'] for m in g) for g in groups),
                   route_differences=sum(m['route_diff'] for m in all_rows),
                   tree_js_mean=sum(m['js'] for m in tree)/len(tree) if tree else None,
                   tree_q_abs_diff_mean=sum(qs)/len(qs) if qs else None)
        result[pair] = one
    return result


def run(args):
    import h7c_adjudicate as ad
    import h7c_sensitivity as sn
    from analysis import tb_v3, tb_v3_fast5
    import torch
    started = time.monotonic()
    c, sha = contract_at(args.contract)
    state = git_state()
    approval_sha = approval(args.approval, sha, args.mode)
    d4.require(args.device == c['device'] and args.workers > 0, 'device/workers differ from contract')
    print('[d4-preflight] checking sealed H-dev and eligible positions', flush=True)
    manifest, games, checks = inputs(c, sha, args.cache)
    full_manifest = manifest
    d4.require(committed(args.sample, ROOT) and json.loads(Path(args.sample).read_text()) == manifest,
               'eligible manifest is not committed or differs from the sealed H-dev')
    models, identities = checked_models(c, args.model, args.device)
    print(f"[d4-preflight] PASS {len(games)} games / {manifest['rows']} eligible rows; five model identities", flush=True)
    runtime = {'engine_hash': tb_v3.ENGINE_HASH, 'runtime_hash': tb_v3_fast5.runtime_hash(args.device)}
    d4.require(runtime == c['runtime_identity'], 'engine/runtime differs from fixed contract')
    d4.require(all(all(g.get(k) == v for k, v in runtime.items()) and g['value_device'] == 'cuda'
                   for g in games.values()), 'source actor engine/runtime/device differs')
    spec = ad.engine_spec(ROOT / c['rapfi_program'], args.rapfi_dir)
    d4.require(spec['identity'] == c['rapfi_identity'], 'Rapfi identity differs')
    folder = Path(args.out_dir).resolve()
    required_folder = (ROOT / 'runs/h7r/R2-D4-BASELINE-1' / args.mode).resolve()
    d4.require(folder == required_folder, f'use isolated output directory {required_folder}')
    identity = {'format': 'h7r-d4-run-v1', 'contract_sha256': sha, 'approval_sha256': approval_sha,
                'git': state, 'hdev': checks, 'sample_sha256': d4.file_sha256(args.sample),
                'models': identities, 'runtime': runtime, 'rapfi': spec['identity'], 'mode': args.mode,
                'device': args.device, 'environment': {'python': platform.python_version(),
                 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                 'gpu': torch.cuda.get_device_name()}}
    identity_sha = d4.digest(identity)
    if args.mode == 'smoke':
        keep = set(sorted(games)[:2])
        games = {i: g for i, g in games.items() if i in keep}
        manifest = {**manifest, 'games': [e for e in manifest['games'] if e['game_index'] in keep]}
    folder.mkdir(parents=True, exist_ok=True)
    with file_lock(folder / 'run.lock', nonblocking=True):
        d4.require(not (folder / 'summary.json').exists(), 'completed summary exists; never overwrite')
        ipath = folder / 'identity.json'
        if ipath.exists():
            d4.require(json.loads(ipath.read_text()) == identity, 'resume identity differs')
        else:
            d4.require(not any((folder / n).exists() for n in ('replay.jsonl', 'branches.jsonl')), 'unbound journals')
            publish_json(ipath, identity)
        replay_path, branch_path = folder / 'replay.jsonl', folder / 'branches.jsonl'
        replay_done = read_journal(replay_path, 'game_index', identity_sha, set(games))
        tasks = [{'game': g, 'game_index': i, 'models': models, 'device': args.device,
                  'identity_sha256': identity_sha} for i, g in sorted(games.items())]
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            replay_done = execute(pool, replay_chunk, tasks, replay_path,
                                  lambda r: sn.check_replay(r, games, models, 'c5'), replay_done,
                                  'game_index', 'd4-replay', lambda t: len(t['game']['plies']))
            records = [replay_done[i] for i in sorted(games)]
            filtered = eligible_replays(records, manifest)
            obs = observations_for(filtered, games)
            table = ad.branch_table(obs)
            keys = sorted(table)
            Random(c['stats']['order_seed']).shuffle(keys)
            if args.mode == 'smoke': keys = keys[:8]
            branch_tasks = [{**table[k], 'order': i, 'identity_sha256': identity_sha, 'spec': spec}
                            for i, k in enumerate(keys)]
            by_key = {t['key']: t for t in branch_tasks}
            done = read_journal(branch_path, 'key', identity_sha, set(keys))
            done = execute(pool, ad._chunk, branch_tasks, branch_path,
                           lambda r: validate_branch(r, by_key[r['key']], c['invalid_policy']),
                           done, 'key', 'd4-rapfi', lambda _: 1)
        d4.require(git_state() == state, 'HEAD changed during D4')
        # Revalidate files and source pins before publishing a summary.
        contract_at(args.contract)
        d4.require(inputs(c, sha, args.cache)[0] == full_manifest, 'H-dev changed during D4')
        d4.require(d4.file_sha256(args.sample) == identity['sample_sha256']
                   and approval(args.approval, sha, args.mode) == approval_sha, 'sample/approval changed')
        checked_models(c, args.model, args.device)
        d4.require(ad.engine_spec(ROOT / c['rapfi_program'], args.rapfi_dir)['identity'] == spec['identity'],
                   'Rapfi files changed during D4')
        result = {'format': 'h7r-d4-summary-v1', 'identity': identity, 'identity_sha256': identity_sha,
                  'actor_check': sn.actor_check(records, games, 'c5'),
                  'eligible_plies': sum(len(r['plies']) for r in filtered),
                  'warmup_plies': sum(len(r['plies']) for r in records)-sum(len(r['plies']) for r in filtered),
                  'branches': len(done), 'observations': len(obs),
                  'raw_directory': str(folder),
                  'files': {p.name: d4.file_sha256(p) for p in (replay_path, branch_path) if p.exists()},
                  'seconds': time.monotonic()-started, 'scope': 'development only; no strength/promotion/V9 ruling'}
        if args.mode == 'smoke':
            result['reading'] = 'SMOKE'
        else:
            counts = d4.multiplicities(len(games), c['stats']['samples'], c['stats']['seed'])
            outcomes = {k: validate_branch(r, by_key[k], c['invalid_policy']) for k, r in done.items()}
            result['branch_outcomes'] = {kind: sum(o['kind'] == kind for o in outcomes.values())
                                         for kind in ('resolved', 'structural', 'unresolved')}
            result['sensitivity'] = sensitivity(filtered, counts)
            result['comparisons'] = {p: d4.selection_stats([o for o in obs if o['compare'] == p], outcomes,
                                                          sorted(games), counts) for p in d4.COMPARISONS}
            # The reference c8 has intervals only, not a formal seed label.
            result['reading'] = d4.joint_reading(result['comparisons'])
            for k in ('label95', 'label_individual'): result['comparisons']['c8:c5'].pop(k)
            result['shared_branch_uses'] = 2*len(obs)-len(table)
            used = {p: {o[k] for o in obs if o['compare'] == p for k in ('key_a', 'key_b')}
                    for p in d4.COMPARISONS}
            result['shared_branches_by_comparison'] = {a: {b: len(used[a] & used[b]) for b in used} for a in used}
            eps = json.loads((ROOT / c['epsilon_result']).read_text())
            result['prediction_context'] = {'epsilon': eps['epsilon'], 'seed_metrics': eps['seeds']}
        d4.require(git_state() == state, 'HEAD changed before D4 publication')
        publish_json(folder / 'summary.json', result)
        print(json.dumps({'reading': result['reading'], 'summary': str(folder / 'summary.json')}, ensure_ascii=False), flush=True)
        return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for cmd in ('prepare', 'run'):
        s = sub.add_parser(cmd)
        s.add_argument('--contract', default='docs/h7r/r2/R2-D4-BASELINE-1.contract.json')
        s.add_argument('--cache', default='runs/h7r/R1-HDEV-1/cache')
        if cmd == 'prepare': s.add_argument('--output', required=True)
        else:
            s.add_argument('--sample', required=True)
            s.add_argument('--approval', required=True)
            s.add_argument('--mode', choices=('smoke', 'full'), required=True)
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
    except (ValueError, KeyError, OSError) as exc:
        print(f'STOP: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
