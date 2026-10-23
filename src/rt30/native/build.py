"""Build the native exact speed-ups in place (docs/web-ser5.md §9.8).

    python -m rt30.native.build             # needs a C compiler and the headers of *this* Python
    python -m rt30.native.build --if-stale  # only when the sources, the Python or the build changed

The headers must be the running interpreter's (``sysconfig`` include directory, checked below):
for a Python 3.14 venv that is the 3.14 development package, which ``apt install python3-dev`` does
not install when the system Python is another version.

The extension modules land next to their sources (``src/rt30/native/_*.so``, not committed), with
``build_info.json`` beside them: Python version, SOABI, compiler, flags and the SHA256 of every
source and built module (the provenance a frozen engine records). Without the modules the engines
run the Python speed-ups, with the same moves, only slower.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = {'_threat_windows': 'threat_windows.c', '_forbidden': 'forbidden.c'}  # fast3 X1-A; fast4 (H7 P1a-1)
FLAGS = ['-O2', '-shared', '-fPIC']
INFO = HERE / 'build_info.json'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compiler() -> list[str]:
    return os.environ.get('CC', sysconfig.get_config_var('CC') or 'cc').split()


def _compiler_version(compiler: list[str]) -> str:
    try:
        out = subprocess.run([*compiler, '--version'], capture_output=True, text=True, check=False).stdout
    except OSError:
        return 'unknown'
    return out.splitlines()[0].strip() if out else 'unknown'


def _expected() -> dict:
    """What a current build must match: the interpreter, the flags and the sources."""
    return {'python': platform.python_version(), 'soabi': sysconfig.get_config_var('SOABI'),
            'ext_suffix': sysconfig.get_config_var('EXT_SUFFIX') or '.so', 'flags': FLAGS,
            'sources': {source: _sha256(HERE / source) for source in MODULES.values()}}


def build_info() -> dict | None:
    try:
        return json.loads(INFO.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def stale() -> str | None:
    """Why the built modules do not match this checkout and interpreter (None: they do)."""
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


def build(verbose: bool = True) -> list[Path]:
    include = sysconfig.get_paths()['include']
    if not (Path(include) / 'Python.h').exists():
        raise SystemExit(f'Python.h not found in {include}: install the headers of this Python '
                         f'({platform.python_version()}, {sys.executable}), e.g. python{sys.version_info[0]}.'
                         f'{sys.version_info[1]}-dev')
    expected = _expected()
    compiler = _compiler()
    built = []
    for name, source in MODULES.items():
        target = HERE / f'{name}{expected["ext_suffix"]}'
        command = [*compiler, *FLAGS, f'-I{include}', str(HERE / source), '-o', str(target)]
        if verbose:
            print(' '.join(shlex.quote(c) for c in command), flush=True)
        subprocess.run(command, check=True)
        built.append(target)
    info = {**expected, 'executable': sys.executable, 'platform': platform.platform(),
            'compiler': ' '.join(compiler), 'compiler_version': _compiler_version(compiler),
            'modules': {path.name: _sha256(path) for path in built}}
    INFO.write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    return built


if __name__ == '__main__':
    if '--if-stale' in sys.argv[1:]:
        reason = stale()
        if reason is None:
            print(f'native modules up to date ({INFO})', flush=True)
            sys.exit(0)
        print(f'rebuilding: {reason}', flush=True)
    for path in build():
        print(f'built {path}', flush=True)
    print(f'build info {INFO}', flush=True)
    sys.exit(0)
