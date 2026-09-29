"""Stage 8 training with scheduled external evaluation (docs/stage8-plan.md §11).

Resumes ``<run-dir>/checkpoints/latest.pt`` and trains in segments that stop at the
scheduled generations. At every scheduled checkpoint it runs, outside the training loop:

- light (every ``--light-every`` generations, default 20 = 320 games): tactical and
  MCTS-v2 with ``--light-pairs`` openings (x2 colours), PUCT v1 25 sims (rules off),
  plus the tactical/value probes and the open-three defense probes;
- heavy (every ``--heavy-every`` generations, default 80 = 1,280 games): MCTS-v3.2.1 and
  MCTS-v7 with ``--heavy-pairs`` openings;
- the per-colour ``color_regression`` summary over all light results so far
  (``training.health``; candidate -> confirmed against the best healthy result).

Outputs go to ``<run-dir>/external_eval/`` and ``<run-dir>/probes/``; existing files are
kept, so the command can be re-run or interrupted at any point. Resuming is exact, so
the segmented run trains the same model as an uninterrupted one.

    python scripts/run_stage8_training.py --run-dir runs/stage8_d16 --config configs/stage8_d16.yaml
    python scripts/run_stage8_training.py --run-dir runs/stage8_d16 --config configs/stage8_d16.yaml \
        --target-generation 288            # gate 2
    python scripts/run_stage8_training.py --run-dir runs/stage8_d16 --eval-only   # catch up only
    # Gate 3 arm: a NEW run from exported weights, schedule from its own generation 0
    python scripts/run_stage8_training.py --run-dir runs/stage8_g3_b --config configs/stage8_g3_b.yaml \
        --new-run --anchor 0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from run_stage7_checkpoint_eval import OPPONENTS, evaluate_checkpoint, search_config_for  # noqa: E402
from run_stage8_head_to_head import play_match  # noqa: E402
from training.evaluation import PUCTAgent  # noqa: E402
from training.config import load_config, validate_config  # noqa: E402
from training.health import replay_color_stats  # noqa: E402
from training.loop import run_training  # noqa: E402
from training.probes import load_model_from_training_checkpoint, run_probe_file  # noqa: E402
from training.schedule import color_regression_summary, next_stop, schedule_points  # noqa: E402
from training.training_checkpoint import (INIT_NAME, load_checkpoint_payload,  # noqa: E402
                                          load_training_state)

PROBE_SETS = (('', ROOT / 'tests' / 'fixtures' / 'stage7_probes_v1.json'),
              ('_defense', ROOT / 'tests' / 'fixtures' / 'stage7_probes_defense_v1.json'))


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=1), encoding='utf-8')
    os.replace(tmp, path)


def checkpoint_for(run_dir: Path, generation: int) -> Path | None:
    names = [f'checkpoint_gen{generation:03d}.pt', f'milestone_gen{generation:03d}.pt']
    if generation == 0:
        names.append(INIT_NAME)  # a new run's starting weights (e.g. an exported checkpoint)
    for name in names:
        path = run_dir / 'checkpoints' / name
        if path.is_file():
            return path
    return None


def current_generation(run_dir: Path) -> int:
    return int(load_checkpoint_payload(run_dir / 'checkpoints' / 'latest.pt')['generation'])


def evaluate_point(run_dir: Path, generation: int, kinds: list[str], args, log=print) -> None:
    checkpoint = checkpoint_for(run_dir, generation)
    if checkpoint is None:
        log(f'gen {generation}: no checkpoint kept, skipped (keep_every must divide the schedule)')
        return
    plans = {'light': (args.light_opponents, args.light_pairs),
             'heavy': (args.heavy_opponents, args.heavy_pairs)}
    for kind in kinds:
        output = run_dir / 'external_eval' / f'gen{generation:03d}_{kind}.json'
        if output.exists():
            continue
        opponents, pairs = plans[kind]
        result = evaluate_checkpoint(checkpoint, opponents, pairs=pairs, seed=args.seed,
                                     simulations=args.simulations,
                                     tactical_rules=args.tactical_rules, log=log)
        result['schedule'] = {'kind': kind, 'generation': generation}
        _write_json(output, result)
    if 'heavy' in kinds and getattr(args, 'h2h_anchor', None):
        output = run_dir / 'external_eval' / f'gen{generation:03d}_h2h.json'
        if not output.exists():
            _write_json(output, head_to_head_vs_anchor(checkpoint, generation, args, log))
    if 'light' in kinds and not args.skip_probes:
        for suffix, probes in PROBE_SETS:
            output = run_dir / 'probes' / f'gen{generation:03d}{suffix}.json'
            if not output.exists():
                _write_json(output, run_probe_file(checkpoint, probes))
        output = run_dir / 'external_eval' / f'gen{generation:03d}_replay.json'
        if not output.exists():
            state = load_training_state(checkpoint)
            _write_json(output, {'generation': generation, 'checkpoint': str(checkpoint),
                                 **replay_color_stats(state.buffer)})


_ANCHOR_CACHE: dict = {}


def _player(label: str, path: Path, args) -> dict:
    model, info = load_model_from_training_checkpoint(path)
    search = search_config_for(path, args.simulations, args.tactical_rules)
    return {'label': label, 'path': str(path), 'info': info, 'search': search.to_dict(),
            'agent': PUCTAgent(label, model, search)}


def head_to_head_vs_anchor(checkpoint: Path, generation: int, args, log=print) -> dict:
    """Heavy points: the checkpoint vs a fixed anchor checkpoint (saturation check)."""
    label, _, path = args.h2h_anchor.partition('=')
    key = (label, path)
    if key not in _ANCHOR_CACHE:
        _ANCHOR_CACHE[key] = _player(label, Path(path), args)
    anchor = _ANCHOR_CACHE[key]
    current = _player(f'gen{generation:03d}', checkpoint, args)
    match = play_match(current, anchor, pairs=args.h2h_pairs, seed=args.h2h_seed,
                       opening_plies=2, radius=2, log=log)
    return {'format_version': 'stage8-h2h-anchor-v1', 'generation': generation,
            'checkpoint': str(checkpoint), 'anchor': {k: v for k, v in anchor.items()
                                                      if k != 'agent'},
            'seed': args.h2h_seed, 'pairs': args.h2h_pairs, **match}


def update_color_summary(run_dir: Path, log=print) -> dict:
    results = [json.loads(p.read_text(encoding='utf-8'))
               for p in sorted((run_dir / 'external_eval').glob('gen*_light.json'))]
    summary = color_regression_summary(results)
    _write_json(run_dir / 'external_eval' / 'color_regression.json', summary)
    latest = summary['latest']
    if latest is not None:
        parts = [f"{c} {latest[c]['wins']}/{latest[c]['games']} {latest[c]['status']}"
                 for c in ('black', 'white')]
        log(f"color_regression vs mcts_v2 at gen {latest['generation']}: " + ', '.join(parts))
        for color in ('black', 'white'):
            if latest[color]['status'] == 'confirmed':
                log(f"*** CONFIRMED {color} regression: {latest[color]['wins']}/"
                    f"{latest[color]['games']} vs reference {latest[color]['reference']['wins']}/"
                    f"{latest[color]['reference']['games']} (gen "
                    f"{latest[color]['reference']['generation']}). Review before continuing.")
    return summary


def catch_up(run_dir: Path, upto: int, args, log=print) -> None:
    light = set(schedule_points(args.anchor, args.light_every, upto))
    heavy = set(schedule_points(args.anchor, args.heavy_every, upto))
    for generation in sorted(light | heavy):
        kinds = [k for k, points in (('light', light), ('heavy', heavy)) if generation in points]
        evaluate_point(run_dir, generation, kinds, args, log)
    update_color_summary(run_dir, log)


def orchestrate(run_dir: Path, config: dict | None, args, log=print) -> int:
    if getattr(args, 'new_run', False) and not (run_dir / 'checkpoints' / 'latest.pt').is_file():
        # Start a new run (training.init_checkpoint = exported weights) and stop at the
        # first schedule point; generation 0 (init.pt) is evaluated by the catch-up below.
        target = config['training']['generations']
        stop = next_stop(0, args.anchor, args.light_every, target)
        stop = stop if stop > 0 else min(args.light_every, target)
        log(f'new run {run_dir}: training gen 0 -> {stop} (target {target})')
        run_training(config, run_dir=run_dir, stop_after=stop, log=log)
    generation = current_generation(run_dir)
    catch_up(run_dir, generation, args, log)
    if args.eval_only:
        return generation
    target = config['training']['generations']
    while generation < target:
        stop = next_stop(generation, args.anchor, args.light_every, target)
        log(f'training gen {generation} -> {stop} (target {target})')
        state = run_training(config, resume=run_dir / 'checkpoints' / 'latest.pt',
                             stop_after=stop, log=log)
        generation = state.generation
        catch_up(run_dir, generation, args, log)
    return generation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='existing run directory (a copy, e.g. runs/stage8_d16)')
    parser.add_argument('--config', type=Path,
                        help='run config (execution-control values may differ, e.g. generations)')
    parser.add_argument('--target-generation', type=int,
                        help='override training.generations (gate target: 192 / 288 / 480)')
    parser.add_argument('--new-run', action='store_true',
                        help='start a NEW run in --run-dir (must not exist or be empty) from '
                             '--config; later invocations resume it (use --anchor 0)')
    parser.add_argument('--eval-only', action='store_true',
                        help='only evaluate scheduled checkpoints that already exist')
    parser.add_argument('--anchor', type=int, default=160,
                        help='first scheduled generation (Stage 8 start: D16 gen 160)')
    parser.add_argument('--light-every', type=int, default=20)
    parser.add_argument('--heavy-every', type=int, default=80)
    parser.add_argument('--light-opponents', nargs='+', choices=OPPONENTS,
                        default=['tactical', 'mcts_v2'])
    parser.add_argument('--light-pairs', type=int, default=25)
    parser.add_argument('--heavy-opponents', nargs='+', choices=OPPONENTS,
                        default=['mcts_v321', 'mcts_v7'])
    parser.add_argument('--heavy-pairs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=7007, help='same openings as Stage 7')
    parser.add_argument('--simulations', type=int, help='default: checkpoint puct_simulations')
    parser.add_argument('--tactical-rules', choices=('auto', 'on', 'off'), default='off',
                        help='model search in external eval (Stage 7 comparisons used off)')
    parser.add_argument('--skip-probes', action='store_true')
    parser.add_argument('--h2h-anchor', metavar='LABEL=PATH',
                        help='at heavy points also play a fixed anchor checkpoint '
                             '(genNNN_h2h.json; e.g. B400=runs/.../checkpoint_gen400.pt)')
    parser.add_argument('--h2h-pairs', type=int, default=50)
    parser.add_argument('--h2h-seed', type=int, default=8008, help='same openings as round robins')
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    has_latest = (args.run_dir / 'checkpoints' / 'latest.pt').is_file()
    if not has_latest and not args.new_run:
        parser.error(f'{args.run_dir} has no checkpoints/latest.pt (use --new-run to start one)')
    if args.new_run and args.eval_only:
        parser.error('--new-run cannot be combined with --eval-only')
    if args.heavy_every % args.light_every:
        parser.error('--heavy-every must be a multiple of --light-every')
    config = None
    if not args.eval_only:
        if args.config is None:
            parser.error('--config is required unless --eval-only')
        config = load_config(args.config)
        if args.target_generation is not None:
            config['training']['generations'] = args.target_generation
            validate_config(config)
        keep_every = config['training'].get('keep_every')
        if not keep_every or args.light_every % keep_every or args.anchor % keep_every:
            parser.error('training.keep_every must divide --light-every and --anchor so '
                         'scheduled checkpoints are kept')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    generation = orchestrate(args.run_dir, config, args,
                             log=lambda message: print(message, flush=True))
    print(json.dumps({'run_dir': str(args.run_dir), 'generation': generation,
                      'external_eval': str(args.run_dir / 'external_eval')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
