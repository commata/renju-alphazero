"""CPU-backend certification of TB-V3 Resolve / fast5 (docs/h6-value.md, after the fast5 G2). Fixed before the games.

The question is only whether running the value on CPU (the SER5 web machine) instead of CUDA (canonical) loses
strength. Not an engine or runtime change: TB-V3 and fast5 are frozen; a fast4 CPU certification would not carry
over to fast5, so this certifies the runtime that is used.

    python scripts/summarize_tb_v3_cpu_cert.py --cpu runs/tb_v3_cert/cpu_8451.json runs/tb_v3_cert/cpu_8452.json \\
        --cuda runs/tb_v3_cert/cuda_8451.json runs/tb_v3_cert/cuda_8452.json --output docs/h7/tb_v3_fast5_cpu_cert.json

Runs: the ``tb_v3`` arm (TB-V3 Resolve, ``--runtime fast5``) vs ``v8:full``, ``--puct-c 1.5 --fast fast4``, 50 pairs,
seeds 8451 and 8452 (both, reserved before running; no further seeds), ``--value-device cpu`` and ``cuda`` on the same
openings and colours, from a clean tree. Every run is checked: the frozen configuration, fast5 recorded on every
game, the canonical factory, the TB-V3 engine hash and the fast5 runtime hash of its device.
    PASS   score difference CPU - CUDA per opening (both colours), bootstrap over openings: 95% lower >= -0.05,
           and no safety violation in either backend
    FAIL   otherwise (CPU stays uncertified for TB-V3 / fast5; the web runs its value on a CUDA machine instead)
Reported only: identical games, per-colour scores, game minutes per backend.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from summarize_h5 import SCORE, summarize  # noqa: E402
import summarize_h7_e2 as e2  # noqa: E402

FORMAT = 'tb-v3-fast5-cpu-cert-v1'
ARM, OPPONENT, RUNTIME = 'tb_v3', 'v8:full', 'fast5'
SEEDS = (8451, 8452)
RULES = {'low_min': -0.05, 'pairs_per_seed': 50, 'fast': 'fast4', 'runtime': RUNTIME, 'bootstrap': e2.RULES['bootstrap'],
         'seed': e2.RULES['seed']}


def check_run(run: dict, device: str) -> None:
    from analysis import tb_v3, tb_v3_fast5
    from analysis.tb_v2_value import ENGINE_OPTIONS, TB_V2_VALUE

    where = f"{run.get('arm')} seed {run.get('seed')} ({device})"
    if run.get('arm') != ARM or run.get('opponent') != OPPONENT or run.get('seed') not in SEEDS:
        raise ValueError(f'{where}: expected {ARM} vs {OPPONENT} on seeds {SEEDS}')
    if run.get('v8_config') != {**TB_V2_VALUE, **ENGINE_OPTIONS, 'tb_v3': True}:
        raise ValueError(f'{where}: not the frozen TB-V3 configuration')
    if run.get('fast', {}).get('requested') != RULES['fast'] or run.get('runtime_mode') != RUNTIME:
        raise ValueError(f'{where}: not run with --fast fast4 --runtime fast5')
    if str(run.get('value_device', '')).split(':')[0] != device:
        raise ValueError(f"{where}: value device {run.get('value_device')}, expected {device}")
    r = run.get('runtime', {})
    if (run.get('canonical_factory') is not True or r.get('engine_hash') != tb_v3.ENGINE_HASH
            or r.get('runtime_hash') != tb_v3_fast5.runtime_hash(device)):
        raise ValueError(f'{where}: not TB-V3 Resolve on this fast5 runtime')
    if run.get('pairs') != RULES['pairs_per_seed']:
        raise ValueError(f"{where}: {run.get('pairs')} pairs, the protocol fixes {RULES['pairs_per_seed']}")
    if any(g.get('runtime') != RUNTIME for g in run['games']):
        raise ValueError(f'{where}: a game not recorded on {RUNTIME}')
    if run.get('provenance', {}).get('git_dirty') is not False:
        raise ValueError(f'{where}: not run from a clean tree')


def _arm(runs: list[dict]) -> dict:
    return {'arm': ARM, 'seeds': sorted(r['seed'] for r in runs),
            'games': [{**g, 'run_seed': r['seed']} for r in runs for g in r['games']]}


def evaluate(cpu_runs: list[dict], cuda_runs: list[dict]) -> dict:
    for r in cpu_runs:
        check_run(r, 'cpu')
    for r in cuda_runs:
        check_run(r, 'cuda')
    cpu, cuda = _arm(cpu_runs), _arm(cuda_runs)
    if cpu['seeds'] != list(SEEDS) or cuda['seeds'] != list(SEEDS):
        raise ValueError(f'both backends must cover exactly the seeds {SEEDS}')
    for seed in SEEDS:
        openings = [r['openings'] for r in cpu_runs + cuda_runs if r['seed'] == seed]
        if any(o != openings[0] for o in openings):
            raise ValueError(f'seed {seed}: the backends did not play the same openings')
    a, b = e2._games(cpu), e2._games(cuda)
    if set(a) != set(b):
        raise ValueError('the two backends did not play the same openings and colours')
    clusters = {}
    for key in sorted(a):
        clusters.setdefault(key[:2], []).append(SCORE[a[key]['result']] - SCORE[b[key]['result']])
    strength = e2._bootstrap(clusters)
    rows = {'cpu': summarize(cpu), 'cuda': summarize(cuda)}
    violations = {k: len(v['safety_violations']) for k, v in rows.items()}
    passed = strength['ci95'][0] >= RULES['low_min'] and not any(violations.values())
    return {'verdict': 'PASS' if passed else 'FAIL', 'strength': strength, 'safety_violations': violations,
            'report': {'identical_games': sum(a[k]['moves'] == b[k]['moves'] for k in a), 'games': len(a),
                       'score': {k: v['score'] for k, v in rows.items()},
                       'by_colour': {k: v['by_colour'] for k, v in rows.items()},
                       'game_minutes': {k: v['total_game_minutes'] for k, v in rows.items()}}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cpu', nargs='+', type=Path, required=True)
    parser.add_argument('--cuda', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from analysis import tb_v3_fast5
    from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256

    load = (lambda p: json.loads(p.read_text(encoding='utf-8')))
    try:
        result = evaluate([load(p) for p in args.cpu], [load(p) for p in args.cuda])
    except ValueError as exc:
        parser.error(str(exc))
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'rules': RULES,
               'runtime_hash': {d: tb_v3_fast5.runtime_hash(d) for d in ('cpu', 'cuda')},
               'runs': {str(p): file_sha256(p) for p in args.cpu + args.cuda}, **result}
    print(json.dumps(payload, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
