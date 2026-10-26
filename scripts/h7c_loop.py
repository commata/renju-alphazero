"""Execute the pre-registered B2 c4..c10 VALUE_ONLY program. No model API, tuning, deployment or git writes.

Receipts bind every stage to its argv, input bytes, code commit and output bytes. Existing unreceipted
completed/partial outputs are never silently reused or overwritten. Only game JSONLs have automatic partial
resume; an interrupted trainer/cache export that already wrote final files requires human inspection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from hybrid import h7c_b2 as b2


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=1) + '\n').encode('utf-8')


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as handle:
        handle.write(encoded(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_once(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        b2.require(path.read_bytes() == content, f'protected output already differs: {path}')
    else:
        with path.open('xb') as handle:
            handle.write(content)


@contextmanager
def exclusive(path):
    """OS lock is released on process death. The lock file itself is not evidence and may remain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            handle.write(b'0')
            handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError('another B2 executor holds the lock') from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError('another B2 executor holds the lock') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def git_identity(root):
    def git(*args):
        return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    b2.require(not git('status', '--porcelain', '--untracked-files=no'), 'tracked tree is dirty')
    b2.require(git('branch', '--show-current') == 'feat/h7-c1', 'B2 must run on feat/h7-c1')
    b2.require(not git('ls-files', '--others', '--exclude-standard', 'src', 'scripts', 'configs'),
               'untracked source/config files could shadow registered code')
    git('ls-files', '--error-unmatch', 'docs/h7c/b2_program.json')
    return {'commit': git('rev-parse', 'HEAD'), 'dirty': False}


def alive(pid):
    if not pid:
        return False
    if os.name == 'nt':
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            status = ctypes.c_ulong()
            return bool(ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(status))) and status.value == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def last_progress(log, since=0):
    """Read a bounded tail from this attempt only; avoid dumping JSON and stale retry output to stdout."""
    path = Path(log)
    if not path.exists():
        return 'child progress unavailable (done/total, % and ETA unknown)'
    with path.open('rb') as handle:
        handle.seek(max(since, path.stat().st_size - 16384))
        lines = handle.read().decode('utf-8', errors='replace').splitlines()
    for pattern in (r'\bETA\b', r'^\[[^]]+\].*\bstep\s+\d', r'^\[[^]]+\].*\d[\d,.]*\s*/\s*\d'):
        for line in reversed(lines):
            if re.search(pattern, line):
                return line.strip()[:1000]
    return 'child progress unavailable (done/total, % and ETA unknown)'


def execute(argv, log, root, running, *, heartbeat_seconds=60.0, log_progress=print):
    env = {**os.environ, 'PYTHONPATH': str(root / 'src'), 'PYTHONUNBUFFERED': '1'}
    started = perf_counter()
    offset = log.stat().st_size if log.exists() else 0
    with log.open('ab') as handle:
        process = subprocess.Popen(argv, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
        running(process.pid)
        try:
            while True:
                try:
                    return process.wait(timeout=heartbeat_seconds)
                except subprocess.TimeoutExpired:
                    elapsed = (perf_counter() - started) / 60
                    log_progress(f'[b2:{log.stem}] running, elapsed {elapsed:.1f} min | '
                                 f'{last_progress(log, offset)}', flush=True)
        except BaseException:
            process.terminate()
            process.wait()
            raise


class Stages:
    def __init__(self, root, out, identity, executor=execute):
        self.root, self.out, self.identity, self.executor = Path(root), Path(out), identity, executor
        self.path = self.out / 'state.json'
        self.out.mkdir(parents=True, exist_ok=True)
        self.state = b2.read(self.path) if self.path.exists() else {
            'format': 'h7c-b2-state-v1', 'identity': identity, 'stages': {}, 'cycles': [], 'status': 'READY'}
        b2.require(self.state.get('identity') == identity, 'resume identity changed (code/program/environment)')
        for name, receipt in self.state['stages'].items():
            if receipt['status'] == 'COMPLETE':
                self.verify(receipt)
            else:
                b2.require(not alive(receipt.get('pid')), f'{name}: its child process is still alive')
        self.save()

    def save(self):
        atomic(self.path, self.state)

    def hashes(self, paths):
        return {str(Path(p).resolve()): digest(p) for p in paths}

    def verify(self, receipt):
        for path, sha in receipt['outputs'].items():
            b2.require(Path(path).is_file() and digest(path) == sha, f'completed output changed: {path}')

    def step(self, name, argv, inputs, outputs, validator, *, resume_jsonl=None, codes=(0,)):
        outputs = [Path(p) for p in outputs]
        binding = {'argv': list(map(str, argv)), 'inputs': self.hashes(inputs)}
        old = self.state['stages'].get(name)
        if old:
            b2.require(old['binding'] == binding, f'{name}: argv or input bytes changed')
            if old['status'] == 'COMPLETE':
                self.verify(old)
                validator()
                print(f'RESUME_SKIP {name}', flush=True)
                return
        # Files left behind before the receipt commit are preserved. Game JSONLs alone can resume.
        for path in outputs:
            if path.exists():
                b2.require(old is not None and resume_jsonl is not None and path == Path(resume_jsonl),
                           f'{name}: unreceipted output exists, inspect before recovery: {path}')
        receipt = {'status': 'RUNNING', 'binding': binding, 'pid': None}
        self.state['stages'][name] = receipt
        self.state['status'], self.state['stage'] = 'RUNNING', name
        self.save()
        def running(pid):
            receipt['pid'] = pid
            self.save()
        print(f'START {name}; log={self.out / (name + ".log")}', flush=True)
        code = self.executor(binding['argv'], self.out / (name + '.log'), self.root, running)
        receipt['pid'], receipt['exit_code'] = None, code
        self.save()
        b2.require(code in codes, f'{name}: command failed ({code}); see log')
        validator()
        if name.endswith('-value'):
            r = b2.read(outputs[0])
            b2.require((code == 0) == (r['verdict'] == 'CANDIDATE'), 'trainer exit code/verdict mismatch')
            if r['verdict'] == 'CANDIDATE':
                outputs += [outputs[0].parent / 'best.pt', outputs[0].parent / 'best.json']
            else:
                b2.require(not (outputs[0].parent / 'best.pt').exists(), 'NO_CANDIDATE with best.pt')
        receipt['outputs'], receipt['status'] = self.hashes(outputs), 'COMPLETE'
        self.save()
        print(f'COMPLETE {name}', flush=True)


def model_paths(root, identity, upto):
    policy = root / 'runs/h3_policy_64x4/best.pt'
    b2.require(digest(policy) == identity['policy_checkpoint_sha256']
               and digest(policy.with_suffix('.json')) == identity['policy_metadata_sha256'], 'parent policy bytes changed')
    values = [root / 'runs/h6_value_head_v3/best.pt'] + [root / f'runs/h7c/c{c}/value/best.pt' for c in range(3, upto)]
    for value in values:
        if value.is_file() and digest(value) == identity['value_checkpoint_sha256']:
            b2.require(digest(value.with_name('best.json')) == identity['value_metadata_sha256'],
                       'parent value metadata mismatch')
            return policy, value
    raise ValueError('parent value bytes not found at a registered checkpoint path')


def validate_arm(path, cycle, candidate, parent, decision_sha, *, drift=False):
    from scripts import h7c_arena as ar
    r = b2.read(path)
    i = r['identity']
    b2.require(i['cycle'] == cycle and i['candidate'] == candidate
               and i['recipe_decision_sha256'] == decision_sha, f'{path}: candidate/decision mismatch')
    b2.require(i['training_parent' if drift else 'parent'] == parent, f'{path}: parent mismatch')
    b2.require(i['seed'] == (8750 if drift else 8730) + cycle, f'{path}: seed mismatch')
    b2.require(ar.signature(i) == r['signature'] and all(g['signature'] == r['signature'] for g in r['games']),
               f'{path}: signature mismatch')
    pairs = 50 if drift or i.get('arm') == 'official' else 25
    expected = {(p, c) for p in range(pairs) for c in ('black', 'white')}
    b2.require(len(r['games']) == 2 * pairs and {(g['pair'], g['tested_color']) for g in r['games']} == expected
               and len({g['key'] for g in r['games']}) == 2 * pairs, f'{path}: missing or duplicate games')
    b2.require(r['summary'] == ar.summarize_arm(r['games']) and r['fingerprint'] == ar.fingerprint(r['games']),
               f'{path}: summary/fingerprint mismatch')
    if drift:
        b2.require(r['judge'] == b2.drift_judge(r['summary']), 'drift judgment differs')


def validate_x2(path, cycle, candidate, decision_sha):
    from evaluation.external import summarize
    r = b2.read(path)
    p = r['ours_provenance']['h7c']  # the decision SHA is nested here, not at ours_provenance's top level
    b2.require(r['seed'] == 8710 + cycle and r['pairs'] == 25 and r['git_dirty'] is False
               and p['candidate_mode'] == 'VALUE_ONLY' and p['models'] == candidate
               and p['recipe_decision_sha256'] == decision_sha, 'X2 provenance mismatch')
    b2.require(len(r['games']) == 50 and len({g['key'] for g in r['games']}) == 50
               and {(g['pair'], g['ours_color']) for g in r['games']}
               == {(p, c) for p in range(25) for c in ('black', 'white')}
               and all(g['run_signature'] == r['run_signature'] for g in r['games']), 'X2 games/signature mismatch')
    b2.require(r['summary'] == summarize(r['games']), 'X2 summary mismatch')
    s = r['summary']
    b2.require(s['complete_pairs'] == 25 and not s['incomplete_pairs'] and not s['engine_error']
               and not s['engine_illegal'] and not s['forbid_mismatches'], 'X2 execution/integrity failure')


def export_cycle(root, cycle, d, verdict, store, skipped):
    c, docs = root / f'runs/h7c/c{cycle}', root / f'docs/h7c/c{cycle}'
    mapping = {'selfplay.json': f'selfplay_c{cycle}_seed{8800 + cycle}.json', 'labels_report.json': f'labels_c{cycle}_report.json',
               'value/record.json': 'value_record.json', 'value_cache/manifest.json': 'value_cache_manifest.json'}
    if verdict['candidate'] is not None:
        mapping['value/best.json'] = 'value_best.json'
    for src, dst in mapping.items():
        write_once(docs / dst, (c / src).read_bytes())
    final = {**verdict, 'b2_program_sha256': d['program']['sha256'], 'b2_skipped': skipped}
    write_once(docs / 'arena_verdict.json', encoded(final))
    x2 = c / 'x2.json'
    if x2.exists():
        write_once(root / f'docs/external/x2_c{cycle}.json', x2.read_bytes())
    files = {}
    for name, receipt in store.state['stages'].items():
        if name.startswith(f'c{cycle}-') and receipt['status'] == 'COMPLETE':
            store.verify(receipt)
            for path, sha in receipt['outputs'].items():
                files[Path(path).relative_to(root).as_posix()] = sha
    manifest = {'format': 'h7c-b2-evidence-v1', 'cycle': cycle, 'phase': 'AB',
                'b2_program_sha256': d['program']['sha256'], 'recipe_decision_sha256': digest(docs / 'decision.json'),
                'code_git': store.identity['git'], 'files_sha256': files, 'document_copies': mapping,
                'verdict_sha256': digest(docs / 'arena_verdict.json'), 'skipped': skipped}
    write_once(docs / 'evidence_manifest.json', encoded(manifest))
    return final


def run_cycle(root, program, program_sha, cycle, args, store):
    from scripts import h7c_arena as ar
    from hybrid import h7c_recipe
    parent_record = root / f'docs/h7c/c{cycle - 1}/arena_verdict.json'
    parent = b2.read(parent_record)['next_champion']
    policy, value = model_paths(root, parent, cycle)
    frozen_value = root / 'runs/h6_value_head_v3/best.pt'
    b2.require(digest(frozen_value) == program['frozen_models']['value_checkpoint_sha256']
               and digest(frozen_value.with_name('best.json')) == program['frozen_models']['value_metadata_sha256'],
               'frozen drift model bytes changed')
    docs, c = root / f'docs/h7c/c{cycle}', root / f'runs/h7c/c{cycle}'
    d = b2.decision(program, program_sha, b2.read(parent_record), digest(parent_record), cycle)
    dp = docs / 'decision.json'
    write_once(dp, encoded(d))
    h7c_recipe.load(dp, cycle)
    ds = digest(dp)
    common_inputs = [dp, args.program, parent_record, policy, policy.with_suffix('.json'), value, value.with_name('best.json')]
    def command(script, *items):
        return [sys.executable, '-u', str(root / 'scripts' / script), *map(str, items)]
    def step(name, argv, inputs, outputs, validator, **kw):
        b2.require(git_identity(root) == store.identity['git'], 'code commit changed during B2')
        def validated():
            b2.require(git_identity(root) == store.identity['git'], 'code changed while a B2 stage was running')
            validator()
        store.step(f'c{cycle}-{name}', argv, common_inputs + inputs, outputs, validated, **kw)
    sp, games = c / 'selfplay.json', c / 'selfplay.jsonl'
    def selfplay_ok():
        r = b2.read(sp)
        b2.require(r['identity']['models'] == parent and r['identity']['cycle'] == cycle
                   and r['identity']['run_seed'] == 8800 + cycle and r['games'] == 1000
                   and r['summary']['games'] == 1000 and r['git'] == store.identity['git'], 'self-play summary mismatch')
    step('selfplay', command('h7c_selfplay.py', '--cycle', cycle, '--run-seed', 8800+cycle, '--games', 1000,
         '--workers', 6, '--policy-checkpoint', policy, '--value-checkpoint', value, '--device', args.device,
         '--jsonl', games, '--output', sp), [], [sp, games], selfplay_ok, resume_jsonl=games)
    vr, pr, lr = c / 'value_rows.jsonl.gz', c / 'policy_rows.jsonl.gz', c / 'labels_report.json'
    def labels_ok():
        r = b2.read(lr)
        b2.require(r['summary']['identity']['cycle'] == cycle and r['summary']['games_in'] == 1000
                   and r['git_dirty'] is False and r['git_commit'] == store.identity['git']['commit'], 'labels mismatch')
    step('labels', command('h7c_labels.py', '--games', games, '--workers', 12, '--fast', 'fast4',
         '--value-output', vr, '--policy-output', pr, '--report', lr), [games], [lr, vr, pr], labels_ok)
    cache = c / 'value_cache'
    cache_files = [cache / 'manifest.json'] + [cache / (s + suffix) for s in ('train', 'val', 'test')
                                               for suffix in ('.pt', '_games.json')]
    def cache_ok():
        from scripts.h7c_train_value import check_cycle_cache, load_cache_splits
        from hybrid.h7c_train import value_rows_source
        check_cycle_cache(cache, cycle, 8800+cycle, rows=vr)
        _, manifest = load_cache_splits(cache, ('train', 'val', 'test'))
        b2.require(value_rows_source(vr, cycle, manifest['labels_sha256'])['models'] == parent, 'cache source mismatch')
    step('cache', command('build_h6_value_cache.py', '--labels', vr, '--out-dir', cache), [vr], cache_files, cache_ok)
    value_record, best = c / 'value/record.json', c / 'value/best.pt'
    config = b2.contained(root, program['value']['config'])
    replay = root / 'data/tactical/h6/value_cache_v3'
    replay_files = [replay / 'manifest.json'] + [replay / (s + '.pt') for s in ('train', 'val', 'test')]
    def value_ok():
        r = b2.read(value_record)
        b2.require(r['cycle'] == cycle and r['git'] == store.identity['git']
                   and r['inputs']['recipe_decision_sha256'] == ds, 'value record mismatch')
        if r['verdict'] == 'NO_CANDIDATE':
            h7c_recipe.check_value_no_candidate(r, d, ds, value_record)
        else:
            b2.require(r['verdict'] == 'CANDIDATE' and digest(best) == r['candidate']['checkpoint_sha256']
                       and digest(best.with_name('best.json')) == r['candidate']['metadata_sha256'], 'value candidate bytes mismatch')
            ar.check_candidate(policy, best, cycle, 'VALUE_ONLY', None, dp)
    step('value', command('h7c_train_value.py', '--cycle', cycle, '--config', config, '--cycle-cache', cache,
         '--cycle-rows', vr, '--replay-cache', replay, '--parent', value, '--device', args.device,
         '--recipe-decision', dp, '--out-dir', c / 'value'), [config, vr, *cache_files, *replay_files],
         [value_record], value_ok, codes=(0, 1))
    history = [root / f'docs/h7c/c{i}/arena_verdict.json' for i in range(1, cycle)]
    output = c / 'arena/verdict.json'
    hist_args = ['--history', *map(str, history)]
    def verdict_ok():
        v = b2.read(output)
        b2.require(v['cycle'] == cycle and v['parent'] == parent and v['recipe_decision_sha256'] == ds,
                   'verdict provenance mismatch')
        ar.load_history(history, cycle, parent, digest(parent_record))
    if b2.read(value_record)['verdict'] == 'NO_CANDIDATE':
        step('verdict', command('h7c_arena.py', 'no-candidate', '--cycle', cycle, '--recipe-decision', dp,
             '--value-record', value_record, '--parent-policy', policy, '--parent-value', value,
             '--champion', parent_record, *hist_args, '--output', output), [value_record, *history], [output], verdict_ok)
        return export_cycle(root, cycle, d, b2.read(output), store,
                            {'arena': 'NO_CANDIDATE', 'drift': 'NO_CANDIDATE', 'x2': 'NO_CANDIDATE'}), None
    candidate = {**parent, 'value_checkpoint_sha256': digest(best), 'value_metadata_sha256': digest(best.with_name('best.json'))}
    a = ['--cycle', cycle, '--candidate-policy', policy, '--candidate-value', best, '--parent-policy', policy,
         '--parent-value', value, '--frozen-policy', policy, '--candidate-mode', 'VALUE_ONLY',
         '--recipe-decision', dp, '--champion', parent_record, '--device', args.device, '--workers', 6]
    arm_files = []
    for arm in b2.ARMS:
        path, jsonl = c / f'arena/{arm}.json', c / f'arena/{arm}.jsonl'
        step(arm, command('h7c_arena.py', 'play', *a, '--arm', arm, '--jsonl', jsonl, '--output', path),
             [best, best.with_name('best.json')], [path, jsonl],
             lambda path=path: validate_arm(path, cycle, candidate, parent, ds), resume_jsonl=jsonl)
        arm_files.append(path)
    tactical = c / 'arena/tactical.json'
    def tactical_ok():
        r = b2.read(tactical)
        b2.require(r['cycle'] == cycle and r['candidate'] == candidate and r['parent'] == parent
                   and r['recipe_decision_sha256'] == ds and r['rule_version'] == ar.TACTICAL_RULE
                   and r['git'] == store.identity['git'], 'tactical provenance mismatch')
    step('tactical', command('h7c_arena.py', 'tactical', *a, '--output', tactical),
         [best, best.with_name('best.json')], [tactical], tactical_ok)
    step('verdict', command('h7c_arena.py', 'verdict', '--arms', *arm_files, '--tactical', tactical,
         *hist_args, '--output', output), [*arm_files, tactical, *history], [output], verdict_ok)
    v = b2.read(output)
    drift = None
    skipped = {}
    if v['promotion']['decision'] == 'PROMOTE':
        path, jsonl = c / 'arena/drift.json', c / 'arena/drift.jsonl'
        da = [x for x in a]  # drift has a fixed mode; omit the mode option accepted only by arena
        index = da.index('--candidate-mode')
        del da[index:index+2]
        step('drift', command('h7c_drift.py', *da, '--frozen-value', frozen_value, '--verdict', output,
             '--jsonl', jsonl, '--output', path), [output, best, best.with_name('best.json'), frozen_value,
             frozen_value.with_name('best.json')], [path, jsonl],
             lambda: validate_arm(path, cycle, candidate, parent, ds, drift=True), resume_jsonl=jsonl)
        drift = b2.read(path)['judge']
    else:
        skipped['drift'] = 'NOT_PROMOTED'
    if (drift and drift['stop']) or v['official'].get('valid') is False:
        skipped['x2'] = drift['reason'] if drift and drift['stop'] else 'OFFICIAL_INVALID'
    else:
        x2, jsonl = c / 'x2.json', c / 'x2.jsonl'
        external = program['external']
        engine_args = ['--engine', args.rapfi_dir / external['executable'], '--engine-name', 'rapfi']
        engine_files = [args.rapfi_dir / f for f in external['files']]
        for f in engine_files:
            engine_args += ['--engine-file', f]
        for info in external['info']:
            engine_args += ['--engine-info', info]
        step('x2', command('h7c_external.py', '--cycle', cycle, '--policy-checkpoint', policy,
             '--value-checkpoint', best, '--candidate-mode', 'VALUE_ONLY', '--recipe-decision', dp,
             '--value-device', args.device, *engine_args, '--workers', 4, '--forbid-check',
             '--games-jsonl', jsonl, '--output', x2), [best, best.with_name('best.json'),
             args.rapfi_dir / external['executable'], *engine_files], [x2, jsonl],
             lambda: validate_x2(x2, cycle, candidate, ds), resume_jsonl=jsonl)
    return export_cycle(root, cycle, d, v, store, skipped), drift


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--program', type=Path, default=ROOT / 'docs/h7c/b2_program.json')
    parser.add_argument('--rapfi-dir', type=Path, default=Path('/mnt/c/engines/rapfi'))
    parser.add_argument('--device', choices=('cuda',), default='cuda', help='B2 keeps the approved CUDA recipe')
    parser.add_argument('--dry-run', action='store_true', help='show approved cycle/seed plan; no games or output writes')
    args = parser.parse_args(argv)
    args.program, args.rapfi_dir = args.program.resolve(), args.rapfi_dir.resolve()
    store = None
    try:
        p, sha = b2.load_program(args.program)
        b2.require(args.program == ROOT / 'docs/h7c/b2_program.json', 'use the canonical tracked B2 program path')
        if args.dry_run:
            print(json.dumps({'program_sha256': sha, 'cycles': {c: b2.seeds(p, c)
                             for c in range(p['first_cycle'], p['last_cycle'] + 1)}}, indent=1))
            return 0
        identity = {'program_sha256': sha, 'git': git_identity(ROOT), 'device': args.device,
                    'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
                    'repository': str(ROOT), 'rapfi_dir': str(args.rapfi_dir)}
        for file, expected in {p['external']['executable']: p['external']['executable_sha256'],
                               **p['external']['files']}.items():
            b2.require(digest(args.rapfi_dir / file) == expected, f'Rapfi identity mismatch: {file}')
        out = ROOT / 'runs/h7c/b2'
        with exclusive(out / 'executor.lock'):
            store = Stages(ROOT, out, identity)
            # Re-enter each completed cycle through the receipts: do not trust state.cyles or file existence.
            if store.state.get('status') == 'STOPPED':
                print(json.dumps({'status': 'STOPPED', 'reason': store.state['reason']}))
                return 0
            frozen_log = out / 'frozen_check.log'
            with frozen_log.open('ab') as log:
                subprocess.run([sys.executable, str(ROOT / 'scripts/check_track_b_frozen.py')], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            records = []
            for cycle in range(p['first_cycle'], p['last_cycle'] + 1):
                verdict, drift = run_cycle(ROOT, p, sha, cycle, args, store)
                records.append(verdict)
                store.state['cycles'] = [v['cycle'] for v in records]
                reason = b2.stop_reason(p, records, drift)
                if reason:
                    store.state.update(status='STOPPED', reason=reason, last_cycle=cycle)
                    store.save()
                    print(json.dumps({'status': 'STOPPED', 'reason': reason, 'last_cycle': cycle}), flush=True)
                    return 0
                store.save()
    except Exception as exc:
        if store is not None:
            store.state.update(status='ERROR', reason='EXECUTION_OR_INTEGRITY_ERROR', error=str(exc))
            store.save()
        print(f'B2 STOP: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
