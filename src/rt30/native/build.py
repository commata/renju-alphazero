"""Build the native exact speed-ups in place (docs/web-ser5.md §9.8).

    python -m rt30.native.build          # needs a C compiler and the Python headers
                                         # (Ubuntu: sudo apt install build-essential python3-dev)

The extension modules land next to their sources (``src/rt30/native/_*.so``, not committed).
Without them the engines run the Python speed-ups, with the same moves, only slower.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = {'_threat_windows': 'threat_windows.c'}


def build(verbose: bool = True) -> list[Path]:
    include = sysconfig.get_paths()['include']
    if not (Path(include) / 'Python.h').exists():
        raise SystemExit(f'Python.h not found in {include}: install the Python headers (python3-dev)')
    suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
    compiler = os.environ.get('CC', sysconfig.get_config_var('CC') or 'cc').split()
    built = []
    for name, source in MODULES.items():
        target = HERE / f'{name}{suffix}'
        command = [*compiler, '-O2', '-shared', '-fPIC', f'-I{include}', str(HERE / source), '-o', str(target)]
        if verbose:
            print(' '.join(shlex.quote(c) for c in command), flush=True)
        subprocess.run(command, check=True)
        built.append(target)
    return built


if __name__ == '__main__':
    for path in build():
        print(f'built {path}', flush=True)
    sys.exit(0)
