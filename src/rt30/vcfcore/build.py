"""Build the native VCF core in place (X2-F; same rules as ``rt30.native.build``, its own build record).

    python -m rt30.vcfcore.build             # needs a C compiler and the headers of *this* Python
    python -m rt30.vcfcore.build --if-stale  # only when the sources, the Python or the build changed

``vcf_core.c`` includes the frozen ``rt30/native/forbidden.c`` (the black forbidden classifier), so both
sources are recorded and either change makes the build stale. The module lands next to its source
(``src/rt30/vcfcore/_vcf_core*.so``, not committed) with ``build_info.json`` beside it. ``rt30.native``'s own
build and record are untouched (they are part of the frozen fast4 runtime).
"""
from __future__ import annotations

import json
import platform
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

from rt30.native.build import _compiler, _compiler_version, _sha256

HERE = Path(__file__).resolve().parent
MODULE, SOURCE = '_vcf_core', 'vcf_core.c'
SOURCES = {SOURCE: HERE / SOURCE, 'forbidden.c': HERE.parent / 'native' / 'forbidden.c'}
FLAGS = ['-O2', '-shared', '-fPIC']
INFO = HERE / 'build_info.json'


def _expected() -> dict:
    return {'python': platform.python_version(), 'soabi': sysconfig.get_config_var('SOABI'),
            'ext_suffix': sysconfig.get_config_var('EXT_SUFFIX') or '.so', 'flags': FLAGS,
            'sources': {name: _sha256(path) for name, path in SOURCES.items()}}


def build_info() -> dict | None:
    try:
        return json.loads(INFO.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def stale() -> str | None:
    """Why the built module does not match this checkout and interpreter (None: it does)."""
    info, expected = build_info(), _expected()
    if info is None:
        return 'no build_info.json'
    for key, value in expected.items():
        if info.get(key) != value:
            return f'{key} changed'
    for name, digest in info.get('modules', {}).items():
        path = HERE / name
        if not path.exists() or _sha256(path) != digest:
            return f'{name} missing or changed'
    return None


def build(verbose: bool = True) -> Path:
    include = sysconfig.get_paths()['include']
    if not (Path(include) / 'Python.h').exists():
        raise SystemExit(f'Python.h not found in {include}: install the headers of this Python '
                         f'({platform.python_version()}, {sys.executable})')
    expected, compiler = _expected(), _compiler()
    target = HERE / f'{MODULE}{expected["ext_suffix"]}'
    command = [*compiler, *FLAGS, f'-I{include}', str(HERE / SOURCE), '-o', str(target)]
    if verbose:
        print(' '.join(shlex.quote(c) for c in command), flush=True)
    subprocess.run(command, check=True)
    info = {**expected, 'executable': sys.executable, 'platform': platform.platform(),
            'compiler': ' '.join(compiler), 'compiler_version': _compiler_version(compiler),
            'modules': {target.name: _sha256(target)}}
    INFO.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    return target


if __name__ == '__main__':
    if '--if-stale' in sys.argv[1:]:
        reason = stale()
        if reason is None:
            print(f'native VCF core up to date ({INFO})', flush=True)
            sys.exit(0)
        print(f'rebuilding: {reason}', flush=True)
    print(f'built {build()}', flush=True)
    print(f'build info {INFO}', flush=True)
    sys.exit(0)
