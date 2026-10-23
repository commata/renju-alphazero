"""Check that this checkout runs the frozen Track B engines: TB-V2-Value + fast4 and TB-V3 Resolve, with models.

``scripts/check_frozen_baseline.py`` covers the MCTS-v6 / v7 benchmark files only. This covers the canonical:
    code       every file of the canonical import closure against ``tests/track_b_frozen.sha256``
    engine     ``tb_v2_value.config_hash`` == ``tb_v2_runtime.ENGINE_HASH``; the VCF table file SHA
    runtime    the native C and Python binding sources (``track_b_guard.check_runtime_sources``); the native build
               is fresh; fast4 installs and binds (``tb_v2_runtime.install`` / ``check_bound``)
    models     the H3 policy checkpoint and metadata (``s3_vct2_v1.check_checkpoint``); the value checkpoint and
               its ``best.json`` (``track_b_guard.check_value_checkpoint``)
    e4         the E4-v1 candidate's locked attack sources (``tb_v2_value_e4.check_sources``)
    tb-v3      its code manifest ``tests/tb_v3_frozen.sha256`` against its import closure, its identity
               (``tb_v3.check_identity``), and, once it is the canonical (``analysis.canonical``), the promotion
               record ``docs/h7/tb_v3_promotion.json`` (PROMOTION_PASS on this runtime hash and identity file)
    fast5      the exact runtime candidate's sources (``tb_v3_fast5.check_sources``) and, when built, its native
               VCF core (fresh, bound by ``tb_v3_fast5.install``); not built is reported, not a failure
Exit status 1 on any FAIL. Run it before benchmark, web or self-play work on a machine with the models.

    python scripts/check_track_b_frozen.py
    python scripts/check_track_b_frozen.py --no-models   # code / engine / runtime only (no checkpoints here)

``--write-manifest tb-v3`` regenerates TB-V3's manifest from its import closure: only in an explicit engine or
runtime freeze step, after reviewing the behaviour change. TB-V2-Value's manifest is never rewritten (the
E4-v2 candidate and TB-V3 lock its SHA-256).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

# The modules the canonical runs (some import others only inside functions, hence listed here).
ENTRY = ('analysis.tb_v2_runtime', 'analysis.tb_v2_value', 'analysis.mcts_v8', 'analysis.mcts_v8_agent',
         'analysis.puct_v8', 'analysis.puct_value_v1', 'analysis.vcf_tt_v1', 'analysis.threats',
         'analysis.s3_vct2_v1', 'hybrid.h4_policy', 'hybrid.h6_value_leaf', 'hybrid.h6_fast',
         'rt30.fast4_forbidden', 'rt30.fast3_windows', 'rt30.native.build')
# Imported on the way but not part of what the engine decides (tooling and data loaders).
EXCLUDE = ('src/hybrid/h6_labels.py', 'src/hybrid/progress.py', 'src/hybrid/renjunet.py', 'src/hybrid/h3_cache.py',
           'src/hybrid/h6_value_cache.py')
# TB-V3: the factory and the attack on top; its identity file holds the manifest's hash, so it is pinned by the
# runtime hash instead (``tb_v3_runtime.identity_sha256``).
ENTRY_V3 = ENTRY + ('analysis.tb_v3_runtime', 'analysis.tb_v2_value_e4v2', 'analysis.e4v2_attack',
                    'analysis.e4_attack', 'analysis.vct2_attack', 'analysis.selective_vct', 'analysis.track_b_guard')
EXCLUDE_V3 = EXCLUDE + ('src/analysis/tb_v3.py',)
MANIFEST_V3 = ROOT / 'tests' / 'tb_v3_frozen.sha256'
PROMOTION = ROOT / 'docs' / 'h7' / 'tb_v3_promotion.json'
HEADER_V3 = """# TB-V3 Resolve (TB-V2-Value + E4-v2 150k/cd/killer, fast4): every file of the import closure of
# its factory (scripts/check_track_b_frozen.py --write-manifest tb-v3), except src/analysis/tb_v3.py, which
# holds this file's SHA-256 in the engine hash. Written at the TB-V3 freeze; do not update these hashes only
# to make a check pass.
"""
HEADER = """# Track B canonical (TB-V2-Value + fast4): every file of its import closure (scripts/check_track_b_frozen.py).
# Written at the review of the frozen guarantees; the code is unchanged since the TB-V2-Value freeze (9c94676)
# and the fast4 runtime freeze (46ebcf7). Do not update these hashes only to make a check pass.
"""


def closure(engine: str = 'tb-v2') -> list[str]:
    """Repository files under src/ imported by the entry modules (and the fast4 install), computed in a fresh
    interpreter so that nothing this process imported for other reasons is counted."""
    import json
    import subprocess

    out = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--print-closure', engine], check=True,
                         capture_output=True, text=True, cwd=ROOT)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _closure_here(engine: str = 'tb-v2') -> list[str]:
    entry, exclude = (ENTRY_V3, EXCLUDE_V3) if engine == 'tb-v3' else (ENTRY, EXCLUDE)
    for name in entry:
        importlib.import_module(name)
    from hybrid import h6_fast
    previous = h6_fast.installed_mode()
    h6_fast.install('fast4')
    h6_fast.install(previous)
    src = (ROOT / 'src').resolve()
    files = set()
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and path.endswith('.py') and Path(path).resolve().is_relative_to(src):
            files.add(Path(path).resolve().relative_to(ROOT).as_posix())
    return sorted(f for f in files if f not in exclude)


def write_manifest_v3(path: Path = MANIFEST_V3) -> int:
    lines = [f'{hashlib.sha256((ROOT / f).read_bytes()).hexdigest()}  {f}' for f in closure('tb-v3')]
    path.write_text(HEADER_V3 + '\n'.join(lines) + '\n', encoding='utf-8')
    return len(lines)


def checks(models: bool, policy_checkpoint: Path, value_checkpoint: Path) -> list[tuple[str, str, str]]:
    from analysis import track_b_guard as guard

    out = []

    def run(name, fn):
        try:
            detail = fn()
            out.append((name, 'PASS', detail or ''))
        except Exception as exc:  # every check reports, none stops the others
            out.append((name, 'FAIL', f'{type(exc).__name__}: {exc}'))

    def code():
        bad = guard.check_code()
        missing = sorted(set(closure()) - set(guard.read_manifest()))
        if bad or missing:
            raise RuntimeError(f'changed: {bad}; imported but not in the manifest: {missing}')
        return f'{len(guard.read_manifest())} files'

    def engine():
        from analysis import tb_v2_runtime as rt, tb_v2_value as tv
        if tv.config_hash() != rt.ENGINE_HASH:
            raise RuntimeError(f'config_hash {tv.config_hash()[:12]} != ENGINE_HASH {rt.ENGINE_HASH[:12]}')
        table = hashlib.sha256((ROOT / 'src/analysis/vcf_tt_v1.py').read_bytes()).hexdigest()
        if table != tv.VCF_TABLE_SHA256:
            raise RuntimeError('src/analysis/vcf_tt_v1.py is not the frozen table')
        return rt.ENGINE_HASH[:12]

    def binding():
        from analysis import tb_v2_runtime as rt
        from hybrid import h6_fast
        previous = h6_fast.installed_mode()
        try:
            rt.install()
        finally:
            h6_fast.install(previous)
        return 'fast4 bound'

    def policy():
        return guard.check_policy_checkpoint(policy_checkpoint)['policy_checkpoint_sha256'][:12]

    def value():
        found = guard.check_value_checkpoint(value_checkpoint)
        return f"{found['value_checkpoint_sha256'][:12]} + best.json {found['value_metadata_sha256'][:12]}"


    def e4():
        from analysis import tb_v2_value_e4
        tb_v2_value_e4.check_sources()
        return tb_v2_value_e4.candidate_hash()[:12]

    run('code', code)
    run('engine', engine)
    run('runtime sources', guard.check_runtime_sources)
    run('runtime binding', binding)
    if models:
        run('policy checkpoint', policy)
        run('value checkpoint', value)
    def v3_code():
        bad = guard.check_code(MANIFEST_V3)
        missing = sorted(set(closure('tb-v3')) - set(guard.read_manifest(MANIFEST_V3)))
        if bad or missing:
            raise RuntimeError(f'changed: {bad}; imported but not in the manifest: {missing}')
        return f'{len(guard.read_manifest(MANIFEST_V3))} files'

    def v3_identity():
        from analysis import tb_v3
        tb_v3.check_identity()
        return f'{tb_v3.ENGINE_NAME} {tb_v3.ENGINE_HASH[:12]}'

    def v3_promotion():
        import json
        from analysis import canonical, tb_v3_runtime
        if canonical.CANONICAL_ENGINE_ID != 'TB-V3':
            return f'pending: the canonical is {canonical.CANONICAL_ENGINE_ID}'
        record = json.loads(PROMOTION.read_text(encoding='utf-8'))
        if (record.get('verdict') != 'PROMOTION_PASS'
                or record.get('runtime_hash') != tb_v3_runtime.runtime_hash(tb_v3_runtime.CANONICAL_DEVICE)
                or record.get('identity_sha256') != tb_v3_runtime.identity_sha256()):
            raise RuntimeError(f'{PROMOTION.name} is not a PROMOTION_PASS of this TB-V3 runtime')
        return 'canonical, promotion replay PASS'

    run('e4-v1 candidate sources', e4)
    run('tb-v3 code', v3_code)
    run('tb-v3 identity', v3_identity)
    def v3_fast5():
        from analysis import tb_v3_fast5
        from hybrid import h6_fast
        from rt30 import fast5
        tb_v3_fast5.check_sources()
        if not fast5.available():
            return 'sources PASS; native VCF core not built (python -m rt30.vcfcore.build)'
        previous = h6_fast.installed_mode()
        try:
            tb_v3_fast5.install()
        finally:
            fast5.uninstall()
            h6_fast.install(previous)
        return f'bound, runtime {tb_v3_fast5.runtime_hash("cuda")[:12]} (cuda)'

    run('tb-v3 promotion', v3_promotion)
    run('tb-v3 fast5 runtime', v3_fast5)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--policy-checkpoint', type=Path, default=ROOT / 'runs/h3_policy_64x4/best.pt')
    parser.add_argument('--value-checkpoint', type=Path, default=ROOT / 'runs/h6_value_head_v3/best.pt')
    parser.add_argument('--no-models', action='store_true', help='skip the checkpoint checks')
    parser.add_argument('--write-manifest', choices=('tb-v3',), help='freeze step only (TB-V3 manifest)')
    parser.add_argument('--print-closure', choices=('tb-v2', 'tb-v3'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.print_closure:
        import json
        print(json.dumps(_closure_here(args.print_closure)), flush=True)
        return 0
    if args.write_manifest:
        print(f'wrote {write_manifest_v3()} files to {MANIFEST_V3}', flush=True)
        return 0
    results = checks(not args.no_models, args.policy_checkpoint, args.value_checkpoint)
    for name, status, detail in results:
        print(f'{status:4}  {name:24}  {detail}', flush=True)
    failed = [r for r in results if r[1] == 'FAIL']
    print('Track B frozen checks ' + ('FAIL' if failed else 'PASS'), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
