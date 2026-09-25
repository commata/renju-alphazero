"""Compare every empty cell and both ordered legal lists against the frozen oracle."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from rule_validation import positions, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--positions', type=int, default=10000,
                        help='total states, including 48 rotated fixtures (minimum 48)')
    args = parser.parse_args()
    if args.positions < 48:
        parser.error('--positions must be at least 48')
    stats = validate(positions(args.seed, args.positions),
                     lambda s: print(json.dumps(s, ensure_ascii=False), flush=True))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    for reason in ('삼삼', '사사', '장목'):
        if stats[reason] < 100:
            print(f'WARNING: only {stats[reason]} {reason} classifications')


if __name__ == '__main__':
    main()
