"""All exact speed-ups of TB-Engine-v1 at once ("TB1-fast", docs/web-ser5.md §3.1, §3.9).

Each one returns the frozen function's result for every input (tests), so with no deadline the
engine makes the same moves with the same diagnostics, nodes and calls; under a deadline it only
gets further in the same time. Import the engine modules before installing.
"""
from __future__ import annotations

from contextlib import contextmanager

from . import fast_extensions, fast_windows


def install_all() -> None:
    fast_windows.install()
    fast_extensions.install()


def uninstall_all() -> None:
    fast_windows.uninstall()
    fast_extensions.uninstall()


def installed() -> dict:
    return {'windows': fast_windows.installed(), 'extensions': fast_extensions.installed()}


@contextmanager
def all_fast():
    install_all()
    try:
        yield
    finally:
        uninstall_all()
