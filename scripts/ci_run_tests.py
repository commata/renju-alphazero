"""Strict unittest runner for GitHub Actions.

Policies intentionally avoid hard-coded test counts:
- none: no skipped tests are allowed.
- torch-only: skips are allowed only with the exact reason "requires torch", and at
  least one such skip must exist so a torch-free job cannot silently stop exercising
  the neural-test skip path.
"""
from __future__ import annotations

import argparse
import sys
import unittest

TORCH_SKIP_REASON = "requires torch"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-policy", choices=("none", "torch-only"), required=True)
    parser.add_argument("--start-dir", default="tests")
    parser.add_argument("--pattern", default="test*.py")
    parser.add_argument("--verbosity", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    suite = unittest.defaultTestLoader.discover(args.start_dir, pattern=args.pattern)
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)

    if not result.wasSuccessful():
        return 1

    skipped = list(result.skipped)
    if args.skip_policy == "none":
        if skipped:
            print("\nCI skip policy violation: full test job must have zero skips.", file=sys.stderr)
            for test, reason in skipped:
                print(f"  {test.id()}: {reason}", file=sys.stderr)
            return 2
    else:
        if not skipped:
            print("\nCI skip policy violation: torch-free job expected torch-only skips.",
                  file=sys.stderr)
            return 2
        unexpected = [(test, reason) for test, reason in skipped
                      if reason != TORCH_SKIP_REASON]
        if unexpected:
            print("\nCI skip policy violation: unexpected skip reasons.", file=sys.stderr)
            for test, reason in unexpected:
                print(f"  {test.id()}: {reason!r}", file=sys.stderr)
            return 2

    print(f"\nCI unittest gate PASS: skip_policy={args.skip_policy}, skips={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
