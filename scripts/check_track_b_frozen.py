"""Check that this checkout runs the frozen Track B canonical: TB-V2-Value + fast4, with its models.

``scripts/check_frozen_baseline.py`` covers the MCTS-v6 / v7 benchmark files only. This covers the canonical:
    code       every file of the canonical import closure against ``tests/track_b_frozen.sha256``
    engine     ``tb_v2_value.config_hash`` == ``tb_v2_runtime.ENGINE_HASH``; the VCF table file SHA
    runtime    the native C and Python binding sources (``track_b_guard.check_runtime_sources``); the native build
               is fresh; fast4 installs and binds (``tb_v2_runtime.install`` / ``check_bound``)
    models     the H3 policy checkpoint and metadata (``s3_vct2_v1.check_checkpoint``); the value checkpoint and
               its ``best.json`` (``track_b_guard.check_value_checkpoint``)
    e4         the E4-v1 candidate's locked attack sources (``tb_v2_value_e4.check_sources``)
Exit status 1 on any FAIL. Run it before benchmark, web or self-play work on a machine with the models.

    python scripts/check_track_b_frozen.py
    python scripts/check_track_b_frozen.py --no-models   # code / engine / runtime only (no checkpoints here)

``--write-manifest`` regenerates the manifest from the import closure: only in an explicit engine or runtime
freeze step, after reviewing the behaviour change.
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
HEADER = """# Track B canonical (TB-V2-Value + fast4): every file of its import closure (scripts/check_track_b_frozen.py).
# Written at the review of the frozen guarantees; the code is unchanged since the TB-V2-Value freeze (9c94676)
# and the fast4 runtime freeze (46ebcf7). Do not update these hashes only to make a check pass.
"""


def closure() -> list[str]:
    """Repository files under src/ imported by the entry modules (and the fast4 install), computed in a fresh
    interpreter so that nothing this process imported for other reasons is counted."""
    import json
    import subprocess

    out = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--print-closure'], check=True,
                         capture_output=True, text=True, cwd=ROOT)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _closure_here() -> list[str]:
    for name in ENTRY:
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
    return sorted(f for f in files if f not in EXCLUDE)


def write_manifest(path: Path) -> int:
    lines = [f'{hashlib.sha256((ROOT / f).read_bytes()).hexdigest()}  {f}' for f in closure()]
    path.write_text(HEADER + '\n'.join(lines) + '\n', encoding='utf-8')
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
    run('e4-v1 candidate sources', e4)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--policy-checkpoint', type=Path, default=ROOT / 'runs/h3_policy_64x4/best.pt')
    parser.add_argument('--value-checkpoint', type=Path, default=ROOT / 'runs/h6_value_head_v3/best.pt')
    parser.add_argument('--no-models', action='store_true', help='skip the checkpoint checks')
    parser.add_argument('--write-manifest', action='store_true', help='freeze step only')
    parser.add_argument('--print-closure', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.print_closure:
        import json
        print(json.dumps(_closure_here()), flush=True)
        return 0
    if args.write_manifest:
        from analysis.track_b_guard import MANIFEST
        print(f'wrote {write_manifest(MANIFEST)} files to {MANIFEST}', flush=True)
        return 0
    results = checks(not args.no_models, args.policy_checkpoint, args.value_checkpoint)
    for name, status, detail in results:
        print(f'{status:4}  {name:24}  {detail}', flush=True)
    failed = [r for r in results if r[1] == 'FAIL']
    print('Track B canonical frozen checks ' + ('FAIL' if failed else 'PASS'), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
