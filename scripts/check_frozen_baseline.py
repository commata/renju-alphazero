"""Verify SHA-256 locks for the frozen Stage 3 benchmark implementation files."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import string
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "tests" / "frozen_baseline.sha256"


def parse_lock(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: expected '<sha256> <path>'")
        digest, relative = parts
        if len(digest) != 64 or any(ch not in string.hexdigits for ch in digest):
            raise ValueError(f"{path}:{line_number}: invalid SHA-256")
        entries.append((digest.lower(), relative))
    if not entries:
        raise ValueError(f"{path}: no frozen files recorded")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()

    try:
        entries = parse_lock(args.lock_file)
    except (OSError, ValueError) as exc:
        print(f"frozen baseline lock error: {exc}", file=sys.stderr)
        return 2

    failed = False
    for expected, relative in entries:
        target = ROOT / relative
        if not target.is_file():
            print(f"MISSING {relative}", file=sys.stderr)
            failed = True
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            print(f"CHANGED {relative}\n  expected {expected}\n  actual   {actual}",
                  file=sys.stderr)
            failed = True
        else:
            print(f"OK {relative} {actual}")

    if failed:
        print("Frozen benchmark changed. Do not update the lock blindly; review and "
              "re-baseline the behavior fingerprint first.", file=sys.stderr)
        return 1
    print("Frozen baseline file hashes PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
