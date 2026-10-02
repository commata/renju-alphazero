"""Resumable champion loop: train in segments, gate each one, promote champions (C1).

One command both starts and resumes. Every segment of ``--segment`` generations:

1. ``run_stage8_training.py`` trains to the segment end with the current champion as
   ``--h2h-anchor`` (it resumes from ``latest.pt`` and skips evaluations already on disk);
2. ``segment_gate.py`` writes ``<gates>/<run name>_<end>.json`` (PROMOTE / HOLD / STOP);
3. on PROMOTE the segment-end checkpoint is copied to ``<anchors>/<prefix><end>.pt`` and
   becomes the champion of the next segment.

All loop state is rebuilt from the gate files, so after a crash, a reboot or Ctrl+C the
same command continues where it stopped: segments with a gate file are replayed (the
champion follows their PROMOTE decisions, a missing anchor copy is redone) and the first
segment without one is trained / evaluated / gated. Stops on STOP, on ``--max-holds``
consecutive HOLDs (plateau), or at ``--end``.

Exit code: 0 finished (end reached), 12 STOP, 13 plateau, 1 a child step failed.

    python scripts/run_champion_loop.py --run-dir runs/stage8_ada_c1120 \\
        --config configs/stage8_s2_adaptive.yaml --start 1360 --end 1600 \\
        --champion ADA1360=runs/stage8_ada_c1120/checkpoints/checkpoint_gen1360.pt \\
        --anchors-dir runs/anchors --gates-dir runs/gates
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXIT_CODES = {10: 'PROMOTE', 11: 'HOLD', 12: 'STOP'}


def gate_path(gates_dir: Path, run_dir: Path, end: int) -> Path:
    return gates_dir / f'{run_dir.name}_{end}.json'


def checkpoint_path(run_dir: Path, generation: int) -> Path:
    return run_dir / 'checkpoints' / f'checkpoint_gen{generation:03d}.pt'


def replay_state(args) -> dict:
    """Champion, consecutive HOLDs and the next segment end, from the gate files."""
    label, _, path = args.champion.partition('=')
    champion, holds = (label, Path(path)), 0
    for end in range(args.start + args.segment, args.end + 1, args.segment):
        gate = gate_path(args.gates_dir, args.run_dir, end)
        if not gate.is_file():
            return {'champion': champion, 'holds': holds, 'next': end, 'stopped': None}
        decision = json.loads(gate.read_text(encoding='utf-8'))['decision']
        if decision == 'STOP':
            return {'champion': champion, 'holds': holds, 'next': None, 'stopped': ('STOP', end)}
        if decision == 'PROMOTE':
            champion, holds = promote(args, end), 0
        else:
            holds += 1
        if holds >= args.max_holds:
            return {'champion': champion, 'holds': holds, 'next': None,
                    'stopped': ('PLATEAU', end)}
    return {'champion': champion, 'holds': holds, 'next': None, 'stopped': None}


def promote(args, end: int) -> tuple[str, Path]:
    label = f'{args.prefix}{end}'
    anchor = args.anchors_dir / f'{label}.pt'
    if not anchor.is_file():
        args.anchors_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path(args.run_dir, end), anchor)
    return label, anchor


def check_h2h_anchor(args, end: int, champion: tuple[str, Path]) -> None:
    """Refuse to gate on a champion match played against a different anchor."""
    path = args.run_dir / 'external_eval' / f'gen{end:03d}_h2h.json'
    if path.is_file():
        found = json.loads(path.read_text(encoding='utf-8')).get('anchor', {}).get('label')
        if found != champion[0]:
            raise SystemExit(f'{path} was played against {found}, expected {champion[0]}; '
                             'move it away and rerun')


def run_logged(command: list[str], log_path: Path | None) -> int:
    """Run a child, echo its output and append it to the log (like Tee-Object -Append)."""
    print('>', ' '.join(command), flush=True)
    log = open(log_path, 'ab') if log_path else None
    try:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 cwd=ROOT)
        for line in child.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            if log:
                log.write(line)
                log.flush()
        return child.wait()
    finally:
        if log:
            log.close()


def training_command(args, end: int, champion: tuple[str, Path]) -> list[str]:
    return [sys.executable, str(ROOT / 'scripts' / 'run_stage8_training.py'),
            '--run-dir', str(args.run_dir), '--config', str(args.config),
            '--anchor', str(args.anchor), '--target-generation', str(end),
            '--light-opponents', *args.light_opponents,
            '--heavy-every', str(args.segment), '--heavy-opponents', *args.heavy_opponents,
            '--heavy-pairs', str(args.heavy_pairs),
            '--h2h-anchor', f'{champion[0]}={champion[1]}']


def gate_command(args, end: int) -> list[str]:
    return [sys.executable, str(ROOT / 'scripts' / 'segment_gate.py'), str(args.run_dir),
            '--from', str(end - args.segment), '--to', str(end),
            '--reference-reuse', str(args.reference_reuse),
            '--output', str(gate_path(args.gates_dir, args.run_dir, end))]


def loop(args, run=run_logged) -> int:
    state = replay_state(args)
    champion = state['champion']
    print(f'champion {champion[0]} ({champion[1]}), consecutive HOLDs {state["holds"]}',
          flush=True)
    if state['stopped']:
        kind, end = state['stopped']
        print(f'already stopped: {kind} at {end}', flush=True)
        return 12 if kind == 'STOP' else 13
    if state['next'] is None:
        print(f'finished: gates exist up to {args.end}', flush=True)
        return 0
    holds = state['holds']
    for end in range(state['next'], args.end + 1, args.segment):
        print(f'=== segment {end - args.segment} -> {end}, champion {champion[0]} ===', flush=True)
        check_h2h_anchor(args, end, champion)
        code = run(training_command(args, end, champion), args.log)
        if code != 0:
            print(f'TRAINING FAILED at {end} (exit {code}); rerun the same command to resume',
                  flush=True)
            return 1
        code = run(gate_command(args, end), args.log)
        decision = EXIT_CODES.get(code)
        if decision is None:
            print(f'GATE FAILED at {end} (exit {code}); rerun the same command', flush=True)
            return 1
        if decision == 'STOP':
            print(f'GATE STOP at {end}: run forensic_short_games.py on this segment', flush=True)
            return 12
        if decision == 'PROMOTE':
            champion, holds = promote(args, end), 0
            print(f'NEW CHAMPION {champion[0]} -> {champion[1]}', flush=True)
        else:
            holds += 1
            if holds >= args.max_holds:
                print(f'PLATEAU: {holds} segments without PROMOTE (last {end})', flush=True)
                return 13
    print(f'finished at {args.end}, champion {champion[0]}', flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--start', type=int, required=True,
                        help='generation of the starting champion checkpoint in --run-dir')
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--segment', type=int, default=40)
    parser.add_argument('--champion', required=True, metavar='LABEL=PATH',
                        help='starting champion (the anchor of the first segment)')
    parser.add_argument('--prefix', default='ADA', help='label prefix of promoted champions')
    parser.add_argument('--anchors-dir', type=Path, required=True)
    parser.add_argument('--gates-dir', type=Path, required=True)
    parser.add_argument('--max-holds', type=int, default=3,
                        help='stop after this many consecutive HOLD segments (plateau)')
    parser.add_argument('--reference-reuse', type=float, default=6.4)
    parser.add_argument('--anchor', type=int, default=400, help='run_stage8_training --anchor')
    parser.add_argument('--light-opponents', nargs='+', default=['tactical', 'mcts_v2'])
    parser.add_argument('--heavy-opponents', nargs='+',
                        default=['mcts_v5', 'mcts_v6', 'mcts_v7'])
    parser.add_argument('--heavy-pairs', type=int, default=25)
    parser.add_argument('--log', type=Path, help='append all child output here')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (args.end - args.start) % args.segment:
        raise SystemExit('--end - --start must be a multiple of --segment')
    if not Path(args.champion.partition('=')[2]).is_file():
        raise SystemExit(f'champion checkpoint not found: {args.champion}')
    return loop(args)


if __name__ == '__main__':
    raise SystemExit(main())
