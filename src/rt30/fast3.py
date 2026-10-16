"""Every exact speed-up of TB-RT30-v3 (docs/web-ser5.md §9.8): fast2 (``rt30.fast``) plus the native
threat-window scan when it is built. ``install_all`` returns the names of what is active, recorded
with each engine result (the moves are the same either way; only the speed differs)."""
from __future__ import annotations

from . import fast, fast3_windows


def install_all() -> list[str]:
    fast.install_all()
    active = ['threat_windows_codes', 'winning_extensions_in_direction']
    if fast3_windows.available():
        fast3_windows.install()
        active[0] = 'threat_windows_native'
    return active


def uninstall_all() -> None:
    fast3_windows.uninstall()
    fast.uninstall_all()


def active() -> list[str]:
    if not fast.fast_windows.installed():
        return []
    return ['threat_windows_native' if fast3_windows.installed() else 'threat_windows_codes',
            'winning_extensions_in_direction']
