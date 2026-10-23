"""CPU-backend certification of TB-V2-Value + fast4 (docs/h6-value.md, after P3). Fixed before the games.

The question is only whether running the value on CPU (the SER5 web machine) instead of CUDA (canonical,
E3) loses strength. Not an engine adoption: the engine and the runtime profile are frozen.

    python scripts/summarize_h7_cpu_cert.py --cpu runs/h7_cert/cpu_8431.json runs/h7_cert/cpu_8432.json \\
        --cuda runs/h7_cert/cuda_8431.json runs/h7_cert/cuda_8432.json --output docs/h7/h7_cpu_cert.json

Runs: ``puct_value_vct2_50k_tt`` vs ``v8:full``, ``--puct-c 1.5 --fast fast4``, 50 pairs, seeds 8431 and 8432
(both, reserved before running; no further seeds), ``--value-device cpu`` and ``cuda`` on the same openings
and colours.
    PASS   score difference CPU - CUDA per opening (both colours), bootstrap over openings: 95% lower >= -0.05,
           and no safety violation in either backend
    FAIL   otherwise (CPU stays uncertified; the web runs on a CUDA machine instead)
Reported only: identical games, per-colour scores, game seconds per backend.
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

ARM, OPPONENT = 'puct_value_vct2_50k_tt', 'v8:full'
SEEDS = (8431, 8432)
RULES = {'low_min': -0.05, 'pairs_per_seed': 50, 'fast': 'fast4', 'bootstrap': e2.RULES['bootstrap'],
         'seed': e2.RULES['seed']}


def check_run(run: dict, device: str) -> None:
    from analysis.tb_v2_runtime import ENGINE_HASH, runtime_hash
    from analysis.tb_v2_value import ENGINE_OPTIONS, TB_V2_VALUE

    where = f"{run.get('arm')} seed {run.get('seed')} ({device})"
    if run['arm'] != ARM or run['opponent'] != OPPONENT or run['seed'] not in SEEDS:
        raise ValueError(f'{where}: expected {ARM} vs {OPPONENT} on seeds {SEEDS}')
    if run['v8_config'] != {**TB_V2_VALUE, **ENGINE_OPTIONS}:
        raise ValueError(f'{where}: not the frozen TB-V2-Value configuration')
    if run.get('fast', {}).get('requested') != RULES['fast']:
        raise ValueError(f'{where}: not run with --fast fast4')
    if str(run.get('value_device', '')).split(':')[0] != device:
        raise ValueError(f"{where}: value device {run.get('value_device')}, expected {device}")
    r = run.get('runtime', {})
    if r.get('engine_hash') != ENGINE_HASH or r.get('runtime_hash') != runtime_hash(device):
        raise ValueError(f'{where}: not the frozen runtime profile')
    if run.get('pairs') != RULES['pairs_per_seed']:
        raise ValueError(f"{where}: {run.get('pairs')} pairs, the protocol fixes {RULES['pairs_per_seed']}")


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
    load = (lambda p: json.loads(p.read_text(encoding='utf-8')))
    try:
        result = {'format': 'h7-cpu-cert-v1', 'rules': RULES,
                  **evaluate([load(p) for p in args.cpu], [load(p) for p in args.cuda])}
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
