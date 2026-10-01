"""Compare the control and teacher arms and apply the Stage 9 entry rule (review §3.7).

Reads what ``run_stage8_training.py`` already writes in each run directory:

- ``external_eval/genNNN_h2h.json``   heavy points: the checkpoint vs the B400 anchor;
- ``external_eval/genNNN_heavy.json`` / ``_light.json``: v321 / v5 / v6 / v7, tactical, MCTS-v2;
- ``probes/genNNN{,_defense,_vct}.json``: raw-network probes;

plus optional direct teacher-vs-control matches (``run_stage8_head_to_head.py
--output``, given with ``--direct``). Only matches between a ``--teacher-label-prefix``
label and a ``--control-label-prefix`` label are used, so a round robin that also
contains another teacher arm is safe to pass.

Rules (a heavy point "improves" when its anchor score > 55 % and p < 0.05):

- plateau(arm): the last three heavy points of the arm do not improve over the anchor;
- teacher stronger: the latest direct match gives the teacher > 55 % with p < 0.05
  (without direct matches: the teacher's anchor score beats the control's by more than
  5 points at the latest common heavy point -- reported as weak evidence);
- probes still rising: any key probe metric rose by >= 0.05 over the last heavy interval.

Verdict:

Roles: the fixed anchor measures absolute progress of each arm; the same-generation
direct match compares the two recipes.

- ``teacher_better``   -> the arm beats the control directly AND still improves over the
                          anchor: adopt the arm's recipe;
- ``relative_only``    -> the arm beats the control directly but has no anchor progress over
                          its last three heavy points: the control regressed, the arm did
                          not improve (do not adopt on this evidence);
- ``capacity``         -> both plateau, no difference, probes flat AND the raw must_block
                          top-1 of both arms >= MUST_BLOCK_CEILING (the network is near
                          what 64x4 is known to learn): start Stage 9 (bigger net);
- ``signal``           -> both plateau and flat, but raw tactics stay far below that
                          ceiling: fix the learning signal (self-play recipe / teacher)
                          before a bigger network;
- ``undecided``        -> no difference yet but probes still rising, or too few points.

    python scripts/compare_teacher_arms.py --control runs/stage8_b400_long \
        --teacher runs/stage8_b400_teacher --direct runs/teacher_h2h/*.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re

GEN = re.compile(r'gen(\d{3,})')
IMPROVE_SCORE = 0.55
IMPROVE_P = 0.05
PROBE_RISE = 0.05
# A 64x4 network trained only on 1,443 proof labels reaches held-out must_block top-1
# 0.72 (review §6.3). Below this the network is not saturated, so a plateau there is a
# learning-signal problem, not a capacity limit.
MUST_BLOCK_CEILING = 0.6
# Stage 8 §12.14: 'capacity' needs EVERY key tactical probe high, not must_block alone
# (S1120 had must_block 0.75 but VCT defence top-1 0.25 and forced_loss value 0.53).
SATURATION_GATES = (
    ('', 'must_block', 'top1', MUST_BLOCK_CEILING),
    ('', 'vcf', 'top1', 0.6),
    ('', 'forced_loss', 'value_sign_accuracy', 0.8),
    ('_vct', 'must_defend_vct', 'top1', 0.5),
    ('_vct', 'vct_attack', 'top1', 0.5),
)
KEY_PROBES = (
    ('', 'must_block', 'top1'),
    ('', None, 'separation'),
    ('_defense', 'must_defend_open3', 'top3'),
    ('_vct', 'must_defend_vct', 'top3'),
    ('_vct', 'vct_attack', 'top3'),
    ('_vct', None, 'balanced_sign_accuracy'),
)


def _load(path: Path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def _gens(directory: Path, suffix: str) -> list[int]:
    return sorted(int(GEN.match(p.name).group(1))
                  for p in directory.glob(f'gen*{suffix}.json') if GEN.match(p.name))


def _p(summary: dict) -> float:
    """Pair-level p when recorded (robust to opening correlation), else game-level."""
    return summary.get('p_pairs_two_sided', summary['p_two_sided'])


def improves(summary: dict) -> bool:
    return summary['a_score'] > IMPROVE_SCORE and _p(summary) < IMPROVE_P


def probe_metric(run: Path, generation: int, suffix: str, kind: str | None, metric: str):
    data = _load(run / 'probes' / f'gen{generation:03d}{suffix}.json')
    if data is None:
        return None
    if kind is None:
        return (data.get('value_overall') or {}).get(metric)
    return (data.get('summary', {}).get(kind) or {}).get(metric)


def arm_table(run: Path) -> dict:
    ext = run / 'external_eval'
    rows = {}
    for generation in _gens(ext, '_h2h'):
        h2h = _load(ext / f'gen{generation:03d}_h2h.json')['summary']
        row = {'anchor_score': h2h['a_score'], 'anchor_p': _p(h2h),
               'improves': improves(h2h)}
        heavy = _load(ext / f'gen{generation:03d}_heavy.json')
        if heavy:
            row['heavy'] = {name: o['summary']['score'] for name, o in heavy['opponents'].items()}
        light = _load(ext / f'gen{generation:03d}_light.json')
        if light:
            row['light'] = {name: o['summary']['score'] for name, o in light['opponents'].items()}
        row['probes'] = {f'{s or "main"}:{k or "value"}:{m}': probe_metric(run, generation, s, k, m)
                         for s, k, m in KEY_PROBES}
        rows[generation] = row
    return rows


def plateau(rows: dict) -> bool | None:
    heavy = sorted(rows)
    if len(heavy) < 3:
        return None
    return not any(rows[g]['improves'] for g in heavy[-3:])


def probes_rising(run: Path, rows: dict) -> bool | None:
    heavy = sorted(rows)
    if len(heavy) < 2:
        return None
    last, previous = heavy[-1], heavy[-2]
    for suffix, kind, metric in KEY_PROBES:
        now = probe_metric(run, last, suffix, kind, metric)
        before = probe_metric(run, previous, suffix, kind, metric)
        if now is not None and before is not None and now - before >= PROBE_RISE:
            return True
    return False


def direct_result(paths: list[Path], teacher_prefix: str, control_prefix: str) -> dict | None:
    """Latest-generation match between a teacher label and a control label.

    Only teacher-vs-control matches count: a round robin that also contains e.g.
    T1_480 vs T2_480 must not be read as the teacher's result against the control.
    """
    best = None
    for path in paths:
        data = _load(path)
        for match in data.get('matches', []):
            s = match['summary']
            if s['a'].startswith(teacher_prefix) and s['b'].startswith(control_prefix):
                score = s['a_score']
            elif s['b'].startswith(teacher_prefix) and s['a'].startswith(control_prefix):
                score = 1 - s['a_score']
            else:
                continue
            generation = max((int(m) for m in re.findall(r'(\d{3,})', s['a'] + s['b'])),
                             default=-1)
            record = {'file': str(path), 'match': f"{s['a']} vs {s['b']}", 'generation': generation,
                      'teacher_score': score, 'p': _p(s), 'games': s['games']}
            if best is None or record['generation'] >= best['generation']:
                best = record
    return best


def verdict(control: dict, teacher: dict, control_run: Path, teacher_run: Path,
            direct: dict | None) -> dict:
    common = sorted(set(control) & set(teacher))
    reasons = []
    stronger = None
    if direct is not None:
        stronger = direct['teacher_score'] > IMPROVE_SCORE and direct['p'] < IMPROVE_P
        reasons.append(f"direct {direct['match']}: teacher {direct['teacher_score']:.3f}, "
                       f"p={direct['p']:.3g} ({direct['games']} games)")
    elif common:
        g = common[-1]
        gap = teacher[g]['anchor_score'] - control[g]['anchor_score']
        stronger = gap > 0.05
        reasons.append(f'no direct match; anchor score gap at gen {g}: {gap:+.3f} (weak evidence)')
    plateaus = {'control': plateau(control), 'teacher': plateau(teacher)}
    rising = {'control': probes_rising(control_run, control),
              'teacher': probes_rising(teacher_run, teacher)}
    reasons.append(f'plateau {plateaus}, probes rising {rising}')
    saturated, gaps = {}, {}
    for name, rows, run in (('control', control, control_run), ('teacher', teacher, teacher_run)):
        values = [(f'{kind}.{metric}', probe_metric(run, max(rows), suffix, kind, metric), floor)
                  for suffix, kind, metric, floor in SATURATION_GATES] if rows else []
        missing = [label for label, value, _ in values if value is None]
        below = [f'{label}={value:.2f}<{floor}' for label, value, floor in values
                 if value is not None and value < floor]
        saturated[name] = None if (not rows or missing) else not below
        gaps[name] = below + [f'{label}=missing' for label in missing]
    reasons.append(f'tactical probes saturated (all gates): {saturated}; below: {gaps}')
    flat = (plateaus['control'] and plateaus['teacher']
            and rising['control'] is False and rising['teacher'] is False)
    if stronger and plateaus['teacher']:
        # Beats the same-generation control but made no absolute progress over the
        # fixed anchor: the control got weaker (the T1 case), not the arm stronger.
        decision = 'relative_only'
    elif stronger:
        decision = 'teacher_better'
    elif flat and saturated['control'] and saturated['teacher']:
        decision = 'capacity'
    elif flat and saturated['control'] is False and saturated['teacher'] is False:
        decision = 'signal'
    else:
        decision = 'undecided'
    return {'decision': decision, 'common_heavy_points': common, 'reasons': reasons,
            'plateau': plateaus, 'probes_rising': rising, 'saturated': saturated,
            'direct': direct}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--direct', nargs='*', default=[],
                        help='run_stage8_head_to_head.py outputs (globs allowed)')
    parser.add_argument('--teacher-label-prefix', default='teacher',
                        help='label prefix of the teacher checkpoint in direct matches')
    parser.add_argument('--control-label-prefix', default='control',
                        help='label prefix of the control checkpoint in direct matches')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    control, teacher = arm_table(args.control), arm_table(args.teacher)
    paths = [Path(p) for pattern in args.direct for p in sorted(glob.glob(pattern))]
    direct = (direct_result(paths, args.teacher_label_prefix, args.control_label_prefix)
              if paths else None)
    result = verdict(control, teacher, args.control, args.teacher, direct)
    print(f"{'gen':>5} {'control vs B400':>18} {'teacher vs B400':>18}")
    for g in sorted(set(control) | set(teacher)):
        cells = []
        for rows in (control, teacher):
            row = rows.get(g)
            cells.append('-' if row is None else
                         f"{row['anchor_score']:.2f} p={row['anchor_p']:.2g}"
                         f"{' *' if row['improves'] else ''}")
        print(f'{g:>5} {cells[0]:>18} {cells[1]:>18}')
    for reason in result['reasons']:
        print(reason)
    print(f"decision: {result['decision']}")
    if args.output:
        args.output.write_text(json.dumps({'control': control, 'teacher': teacher, **result},
                                          indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
