"""R2-S400-TRAIN development suites: two new O1-rule opening suites of 300, class-disjoint from every earlier suite and
evaluation sample and from each other, one opening per symmetry class (docs/h7r/r2/R2-S400-TRAIN-draft.md, section 5).

    build  the exclusion (symmetry classes of the H7-R suites O0, O1, O2, O1-confirm-main, O1-confirm-anchor and of
           every arm sample of the committed v1 / v2 evaluation contracts), then suite a (seed 202611131) and suite b
           (seed 202611132, also excluding suite a's classes) with ``hybrid.h7r_openings.build_suite``
           (``exclude_classes``, ``distinct_classes``); the overlap of each suite with every source (history, position,
           class) and the prefix overlap with 3-move openings; writes the two suites and the record (never over a file)
    check  everything recomputed from the committed files: the sources' SHA-256, the exclusion, both suites rebuilt
           from their seeds and equal to the files, no class shared with any source or between the suites

The generation seeds are the v2 suite seeds (``hybrid.h7r_seeds.SUITE_SEEDS``), never Arena game seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hybrid import h7r_openings as op  # noqa: E402
from hybrid import h7r_seeds as seeds  # noqa: E402

FORMAT = 'h7r-dev-suites-v1'
SUITE_DIR = 'docs/h7r/suites'
SOURCE_SUITES = ('O0', 'O1', 'O2', 'O1-confirm-main', 'O1-confirm-anchor')
CONTRACT_DIRS = ('docs/h7r/contracts', 'docs/h7r/contracts-v2')
SUITES = (('a', 202611131), ('b', 202611132))
COUNT = 300
RULE = 'O1'
RECORD = f'{SUITE_DIR}/O1-dev-record.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def suite_path(label: str) -> str:
    return f'{SUITE_DIR}/O1-dev-{label}.json'


def git_state(root) -> dict:
    run = lambda *a: subprocess.check_output(['git', *a], cwd=root, text=True).strip()  # noqa: E731
    return {'commit': run('rev-parse', 'HEAD'), 'dirty': bool(run('status', '--porcelain', '--untracked-files=no'))}


def sources(root) -> list[dict]:
    """Every earlier suite and evaluation sample: path, SHA-256, kind and its openings (move lists)."""
    root = Path(root)
    out = []
    for name in SOURCE_SUITES:
        path = root / SUITE_DIR / f'{name}.json'
        body = json.loads(path.read_text(encoding='utf-8'))
        op.check_suite(body)
        out.append({'path': f'{SUITE_DIR}/{name}.json', 'sha256': op._file_sha(path), 'kind': 'suite',
                    'moves': [o['moves'] for o in body['openings']]})
    for folder in CONTRACT_DIRS:
        for path in sorted((root / folder).glob('*.json')) if (root / folder).exists() else []:
            body = json.loads(path.read_text(encoding='utf-8'))
            rel = path.relative_to(root).as_posix()
            if not body.get('arms'):  # e.g. a holdout contract: self-play games, no opening sample
                out.append({'path': rel, 'sha256': op._file_sha(path), 'kind': 'contract without arm samples',
                            'moves': []})
                continue
            moves = []
            for arm in body['arms']:
                spath = root / arm['suite']['path']
                require(op._file_sha(spath) == arm['suite']['sha256'], f"{rel}: arm {arm['name']} suite SHA-256")
                suite = json.loads(spath.read_text(encoding='utf-8'))
                moves += [o['moves'] for o in op.pick(suite, arm['pairs'], arm['seed'])]
            out.append({'path': rel, 'sha256': op._file_sha(path), 'kind': 'contract arm samples', 'moves': moves})
    return out


def excluded_classes(srcs: list[dict]) -> set:
    return {op.ids(m)['class_id'] for s in srcs for m in s['moves']}


def build_suites(srcs: list[dict]) -> dict:
    """Suite a, then suite b excluding a's classes too (the build is deterministic: the same inputs, the same suites)."""
    base = excluded_classes(srcs)
    out, extra = {}, set()
    for label, seed in SUITES:
        require(seed in seeds.SUITE_SEEDS, f'{seed}: not a v2 suite seed')
        body = op.build_suite(RULE, COUNT, seed, exclude_classes=base | extra, distinct_classes=True)
        body['seed_policy'] = seeds.POLICY
        out[label] = body
        extra |= {o['class_id'] for o in body['openings']}
    return {'suites': out, 'base_classes': base}


def overlaps(srcs: list[dict], suites: dict) -> dict:
    three = [m for s in srcs for m in s['moves'] if len(m) == 3]
    out = {}
    for label, body in suites.items():
        out[label] = {'by_source': op.overlap(body['openings'], {s['path']: s['moves'] for s in srcs if s['moves']}),
                      'prefix_of_3_move_openings': op.prefix_overlap(body['openings'], three)}
    a, b = ({o['class_id'] for o in suites[x]['openings']} for x in ('a', 'b'))
    out['between_a_and_b'] = {'class_id': len(a & b),
                              'history_id': len({o['history_id'] for o in suites['a']['openings']}
                                                & {o['history_id'] for o in suites['b']['openings']})}
    return out


def checks(record_overlaps: dict, suites: dict) -> list[str]:
    bad = []
    for label in ('a', 'b'):
        body = suites[label]
        if body['count'] != COUNT or len({o['class_id'] for o in body['openings']}) != COUNT:
            bad.append(f'suite {label}: not {COUNT} openings of {COUNT} distinct classes')
        for src, counts in record_overlaps[label]['by_source'].items():
            if counts['class_id']:
                bad.append(f'suite {label}: {counts["class_id"]} classes shared with {src}')
    if any(record_overlaps['between_a_and_b'].values()):
        bad.append('suites a and b share classes or openings')
    found = seeds.conflicts()
    if any(found.values()):
        bad.append(f'seed registry conflicts: {found}')
    return bad


def body_of(srcs, built, ov, git) -> dict:
    base = sorted(built['base_classes'])
    return {'format': FORMAT, 'git': git, 'rule': RULE, 'count': COUNT, 'seed_policy': seeds.POLICY,
            'sources': [{k: s[k] for k in ('path', 'sha256', 'kind')} | {'openings': len(s['moves']),
                        'classes': len({op.ids(m)['class_id'] for m in s['moves']})} for s in srcs],
            'excluded_classes': {'count': len(base), 'sha256': op._sha(base)},
            'suites': {label: {'path': suite_path(label), 'seed': seed,
                               'openings_sha256': built['suites'][label]['openings_sha256'],
                               'count': built['suites'][label]['count'],
                               'distinct': built['suites'][label]['distinct'],
                               'rejected': built['suites'][label]['rejected']} for label, seed in SUITES},
            'overlaps': ov, 'seed_registry_conflicts': seeds.conflicts(),
            'note': 'the excluded class list is recomputed from the sources by `check`; 3-move and 5-move openings '
                    'cannot share a class, so prefix overlap with 3-move openings is counted, not excluded'}


def build(args, root=ROOT, log=print) -> dict:
    root = Path(root)
    git = git_state(root)
    require(not git['dirty'] or args.allow_dirty, 'the tree is dirty: suites must name a clean commit')
    targets = [root / suite_path(label) for label, _ in SUITES] + [root / RECORD]
    require(not any(t.exists() for t in targets), f'{[str(t) for t in targets if t.exists()]} exist: never overwritten')
    srcs = sources(root)
    built = build_suites(srcs)
    ov = overlaps(srcs, built['suites'])
    bad = checks(ov, built['suites'])
    require(not bad, f'the suites fail their checks: {bad}')
    for label, _ in SUITES:
        body = {**built['suites'][label], 'git': git}
        (root / suite_path(label)).write_text(json.dumps(body, indent=1) + '\n', encoding='utf-8')
    record = body_of(srcs, built, ov, git)
    record['suite_files_sha256'] = {label: op._file_sha(root / suite_path(label)) for label, _ in SUITES}
    (root / RECORD).write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
    log(json.dumps({'suites': record['suites'], 'excluded_classes': record['excluded_classes']['count']}, indent=1),
        flush=True)
    return record


def check(args, root=ROOT, log=print) -> dict:
    """Recompute everything from the committed files and compare (module docstring)."""
    root = Path(root)
    record = json.loads((root / RECORD).read_text(encoding='utf-8'))
    srcs = sources(root)
    require([{k: s[k] for k in ('path', 'sha256')} for s in srcs] ==
            [{k: s[k] for k in ('path', 'sha256')} for s in record['sources']],
            'the sources are not the record\'s (a suite or contract changed, appeared or disappeared)')
    built = build_suites(srcs)
    for label, _ in SUITES:
        body = json.loads((root / suite_path(label)).read_text(encoding='utf-8'))
        op.check_suite(body)
        require(op._file_sha(root / suite_path(label)) == record['suite_files_sha256'][label],
                f'suite {label}: the file is not the recorded one')
        require({k: v for k, v in body.items() if k != 'git'} == built['suites'][label],
                f'suite {label}: not what its seed and the exclusion give')
    ov = overlaps(srcs, {label: json.loads((root / suite_path(label)).read_text(encoding='utf-8'))
                         for label, _ in SUITES})
    bad = checks(ov, built['suites'])
    require(ov == record['overlaps'], 'the overlaps are not the record\'s')
    require(op._sha(sorted(built['base_classes'])) == record['excluded_classes']['sha256'], 'the exclusion differs')
    require(not bad, f'the suites fail their checks: {bad}')
    out = {'passed': True, 'suites': record['suites'], 'excluded_classes': record['excluded_classes']}
    log(json.dumps(out, indent=1), flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--allow-dirty', action='store_true', help='tests only')
    sub.add_parser('check')
    args = parser.parse_args(argv)
    try:
        (build if args.command == 'build' else check)(args)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
