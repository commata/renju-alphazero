"""R2-S400-PRESERVE-1 development suites: two new O1-rule opening suites of 300 (O1-dev-c, O1-dev-d), class-disjoint
from every earlier suite and evaluation sample and from each other, one opening per symmetry class
(docs/h7r/r2/R2-S400-PRESERVE-1-draft.md, section 5).

``scripts/h7r_dev_suites.py`` (O1-dev-a / b) is kept as it is and keeps checking its suites; this tool reuses its
source reading and class functions with its own suites, generation seeds (``hybrid.h7r_suite_seeds_dev2``) and record.

    build  only as the approved ``suites`` stage of R2-S400-PRESERVE-1 (its committed contract, the prepare record
           committed); the exclusion (symmetry classes of the H7-R suites O0, O1, O2, O1-confirm-main,
           O1-confirm-anchor, O1-dev-a, O1-dev-b and of every arm sample of the committed v1 / v2 evaluation
           contracts), then suite c (seed 202611191) and suite d (seed 202611192, also excluding suite c's classes);
           the overlaps with every source (history, position, class) and the prefix overlap with 3-move openings;
           writes the two suites and the record (never over a file)
    check  everything recomputed from the committed files: the build's sources read again from their recorded paths
           and SHA-256 (contracts added later, such as the Arena that plays these suites, are not build sources), the
           exclusion, both suites rebuilt from their seeds and equal to the files, no class shared with any source or
           between the suites, a sound seed registry
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

import h7r_dev_suites as ds  # noqa: E402
from hybrid import h7r_openings as op  # noqa: E402
from hybrid import h7r_suite_seeds_dev2 as seeds  # noqa: E402

FORMAT = 'h7r-dev-suites-v2'
SUITE_DIR = ds.SUITE_DIR
SOURCE_SUITES = (*ds.SOURCE_SUITES, 'O1-dev-a', 'O1-dev-b')
SUITES = (('c', 202611191), ('d', 202611192))
COUNT = 300
RULE = 'O1'
RECORD = f'{SUITE_DIR}/O1-dev-cd-record.json'
require = ds.require


def suite_path(label: str) -> str:
    return f'{SUITE_DIR}/O1-dev-{label}.json'


def sources(root) -> list[dict]:
    """Every earlier suite (``SOURCE_SUITES``) and evaluation sample (every contract of the v1 / v2 folders)."""
    root = Path(root)
    rels = [f'{SUITE_DIR}/{name}.json' for name in SOURCE_SUITES]
    for folder in ds.CONTRACT_DIRS:
        if (root / folder).exists():
            rels += [p.relative_to(root).as_posix() for p in sorted((root / folder).glob('*.json'))]
    return [ds.source(root, rel) for rel in rels]


def build_suites(srcs: list[dict]) -> dict:
    """Suite c, then suite d excluding c's classes too (deterministic: the same inputs, the same suites)."""
    base = ds.excluded_classes(srcs)
    out, extra = {}, set()
    for label, seed in SUITES:
        require(seed in seeds.SUITE_SEEDS, f'{seed}: not a registered {seeds.POLICY} suite seed')
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
    c, d = ({o['class_id'] for o in suites[x]['openings']} for x in ('c', 'd'))
    out['between_c_and_d'] = {'class_id': len(c & d),
                              'history_id': len({o['history_id'] for o in suites['c']['openings']}
                                                & {o['history_id'] for o in suites['d']['openings']})}
    return out


def checks(record_overlaps: dict, suites: dict) -> list[str]:
    bad = []
    for label, _ in SUITES:
        body = suites[label]
        if body['count'] != COUNT or len({o['class_id'] for o in body['openings']}) != COUNT:
            bad.append(f'suite {label}: not {COUNT} openings of {COUNT} distinct classes')
        for src, counts in record_overlaps[label]['by_source'].items():
            if counts['class_id']:
                bad.append(f'suite {label}: {counts["class_id"]} classes shared with {src}')
    if any(record_overlaps['between_c_and_d'].values()):
        bad.append('suites c and d share classes or openings')
    found = seeds.conflicts()
    if any(found.values()):
        bad.append(f'seed registry conflicts: {found}')
    return bad


def body_of(srcs, built, ov, git) -> dict:
    base = sorted(built['base_classes'])
    return {'format': FORMAT, 'git': git, 'rule': RULE, 'count': COUNT, 'seed_policy': seeds.POLICY,
            'seed_registry': {'module': 'src/hybrid/h7r_suite_seeds_dev2.py', 'registered': seeds.REGISTERED},
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


def approved(args, root) -> dict:
    """The committed R2-S400-PRESERVE-1 contract with its ``suites`` stage approved and its prepare record committed:
    the suites are built only as that stage."""
    import h7r_preserve as pv
    c, _, eid = pv.load_contract(getattr(args, 'contract', None), root, 'suites')
    pv.committed_record(root, c['output']['prepare'], 'prepare record', eid)
    return c


def build(args, root=ROOT, log=print) -> dict:
    root = Path(root)
    approved(args, root)
    git = ds.git_state(root)
    require(not git['dirty'] or getattr(args, 'allow_dirty', False), 'the tree is dirty: suites must name a clean commit')
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


def check(args=None, root=ROOT, log=print) -> dict:
    """Recompute everything from the committed files and compare (module docstring)."""
    root = Path(root)
    record = json.loads((root / RECORD).read_text(encoding='utf-8'))
    require(record.get('format') == FORMAT, f'{RECORD}: not an {FORMAT} record')
    srcs = ds.recorded_sources(root, record)
    built = build_suites(srcs)
    files = {}
    for label, _ in SUITES:
        body = json.loads((root / suite_path(label)).read_text(encoding='utf-8'))
        op.check_suite(body)
        require(op._file_sha(root / suite_path(label)) == record['suite_files_sha256'][label],
                f'suite {label}: the file is not the recorded one')
        require({k: v for k, v in body.items() if k != 'git'} == built['suites'][label],
                f'suite {label}: not what its seed and the exclusion give')
        files[label] = body
    ov = overlaps(srcs, files)
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
    b.add_argument('--contract', default=None, help='the R2-S400-PRESERVE-1 contract (default: its committed path)')
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
