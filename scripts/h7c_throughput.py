"""H7-c1 worker reproducibility and throughput (docs/h7-c1.md): an operating value, not a gate on the engine.

    repro   the same 4 games (run seed 8800, cycle 1, indices 0-3) with 1 worker and with ``--repro-workers``
            (default 12): every record field but times must be equal, since no decision of TB-V3 depends on time
    sweep   the same 20 games (indices 0-19) with each worker count (default 6, 8, 10, 12, 14): games per hour,
            move latency mean / p99, GPU utilisation and memory (nvidia-smi, sampled each second), CPU utilisation
            (/proc/stat); the choice is the fewest workers within 95% of the best games per hour

    python -u scripts/h7c_throughput.py repro --policy-checkpoint runs/h3_policy_64x4/best.pt \\
        --value-checkpoint runs/h6_value_head_v3/best.pt --device cuda --out runs/h7c/throughput
    python -u scripts/h7c_throughput.py sweep --policy-checkpoint runs/h3_policy_64x4/best.pt \\
        --value-checkpoint runs/h6_value_head_v3/best.pt --device cuda --out runs/h7c/throughput \\
        --output docs/h7c/h7c1_throughput.json

Its games use run seed 8800, reserved for this measurement: they are never training data (the generator refuses
8800 for a training run). Every run writes its own JSONL under ``--out`` (a fresh directory per command).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h7c_selfplay as sp  # noqa: E402

FORMAT = 'h7c-throughput-v1'
REPRO_GAMES, SWEEP_GAMES = 4, 20
SWEEP_WORKERS = (6, 8, 10, 12, 14)
CHOICE_SHARE = 0.95
TIME_FIELDS = {'seconds', 'game_seconds'}


def untimed(value):
    if isinstance(value, dict):
        return {k: untimed(v) for k, v in value.items() if k not in TIME_FIELDS}
    if isinstance(value, list):
        return [untimed(v) for v in value]
    return value


def first_difference(a, b, path='') -> str | None:
    if type(a) is not type(b):
        return path or '.'
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f'{path}.{key}'
            found = first_difference(a[key], b[key], f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f'{path}[len]'
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f'{path}[{i}]')
            if found:
                return found
        return None
    return None if a == b else path or '.'


def compare_runs(a: list[dict], b: list[dict]) -> dict:
    """Games of two runs by index: the first differing field of each mismatching game."""
    by_a, by_b = {r['game_index']: untimed(r) for r in a}, {r['game_index']: untimed(r) for r in b}
    if set(by_a) != set(by_b):
        raise ValueError('the two runs did not play the same game indices')
    diffs = {i: first_difference(by_a[i], by_b[i]) for i in sorted(by_a)}
    return {'games': len(by_a), 'identical': sum(d is None for d in diffs.values()),
            'differences': {str(i): d for i, d in diffs.items() if d is not None}}


def choose(rows: list[dict]) -> int:
    best = max(r['games_per_hour'] for r in rows)
    return min(r['workers'] for r in rows if r['games_per_hour'] >= CHOICE_SHARE * best)


class Sampler:
    """GPU (nvidia-smi) and CPU (/proc/stat) utilisation, once a second, in a thread."""

    def __init__(self, every: float = 1.0):
        self.every, self.gpu, self.cpu = every, [], []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.smi = shutil.which('nvidia-smi')

    @staticmethod
    def _cpu_times():
        try:
            fields = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
        except (OSError, ValueError, IndexError):
            return None
        return sum(fields), fields[3] + (fields[4] if len(fields) > 4 else 0)

    def _loop(self):
        last = self._cpu_times()
        while not self._stop.wait(self.every):
            now = self._cpu_times()
            if last and now and now[0] > last[0]:
                self.cpu.append(1 - (now[1] - last[1]) / (now[0] - last[0]))
            last = now
            if self.smi:
                try:
                    out = subprocess.run([self.smi, '--query-gpu=utilization.gpu,memory.used',
                                          '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                         timeout=5, check=True).stdout.splitlines()[0]
                    util, memory = (float(x) for x in out.split(','))
                    self.gpu.append((util, memory))
                except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                    pass

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(5)

    def summary(self) -> dict:
        mean = (lambda xs: round(sum(xs) / len(xs), 2) if xs else None)
        return {'gpu_util_mean': mean([u for u, _ in self.gpu]), 'gpu_memory_max_mib': max(
                    (m for _, m in self.gpu), default=None), 'cpu_util_mean': mean([100 * c for c in self.cpu]),
                'samples': len(self.cpu)}


def _run(args, workers: int, games: int, name: str, log) -> tuple[list[dict], float, dict]:
    run_args = argparse.Namespace(cycle=1, run_seed=sp.MEASURE_SEED, measurement=True, games=games, start=0,
                                  workers=workers, policy_checkpoint=args.policy_checkpoint,
                                  value_checkpoint=args.value_checkpoint, device=args.device,
                                  jsonl=args.out / f'{name}.jsonl', allow_dirty=args.allow_dirty,
                                  max_plies=getattr(args, 'max_plies', None))  # tests only
    if run_args.jsonl.exists():
        raise ValueError(f'{run_args.jsonl} exists: use a fresh --out directory')
    started = perf_counter()
    payload = sp.run(run_args, log=log)
    wall = perf_counter() - started
    rows = [json.loads(x) for x in run_args.jsonl.read_text(encoding='utf-8').splitlines() if x.strip()]
    return rows, wall, payload


def repro(args, log=print) -> dict:
    one, _, payload = _run(args, 1, REPRO_GAMES, 'repro_w1', log)
    many, _, _ = _run(args, args.repro_workers, REPRO_GAMES, f'repro_w{args.repro_workers}', log)
    result = compare_runs(one, many)
    return {'format': FORMAT, 'mode': 'repro', 'identity': payload['identity'], 'git': payload['git'],
            'workers': [1, args.repro_workers], **result,
            'verdict': 'REPRO_PASS' if result['identical'] == result['games'] else 'REPRO_FAIL'}


def sweep(args, log=print) -> dict:
    rows, identity, git, reference = [], None, None, None
    for workers in args.workers:
        with Sampler() as sampler:
            games, wall, payload = _run(args, workers, SWEEP_GAMES, f'sweep_w{workers}', log)
        identity, git = payload['identity'], payload['git']
        reference = reference or games
        seconds = sorted(p['seconds'] for g in games for p in g['plies'])
        rows.append({'workers': workers, 'wall_seconds': round(wall, 1),
                     'games_per_hour': round(len(games) * 3600 / wall, 1), 'moves': len(seconds),
                     'move_seconds_mean': round(sum(seconds) / len(seconds), 4),
                     'move_seconds_p99': seconds[int(0.99 * (len(seconds) - 1))],
                     'same_games_as_first': compare_runs(reference, games)['identical'], **sampler.summary()})
        log(f"[h7c-throughput] workers {workers}: {rows[-1]['games_per_hour']} games/h, "
            f"move mean {rows[-1]['move_seconds_mean']} s, p99 {rows[-1]['move_seconds_p99']} s", flush=True)
    return {'format': FORMAT, 'mode': 'sweep', 'identity': identity, 'git': git, 'games': SWEEP_GAMES,
            'rule': f'fewest workers within {CHOICE_SHARE:.0%} of the best games per hour', 'rows': rows,
            'chosen_workers': choose(rows)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('mode', choices=('repro', 'sweep'))
    parser.add_argument('--policy-checkpoint', type=Path, required=True)
    parser.add_argument('--value-checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', type=Path, required=True, help='a fresh directory for the run JSONL files')
    parser.add_argument('--repro-workers', type=int, default=12)
    parser.add_argument('--workers', type=int, nargs='+', default=list(SWEEP_WORKERS))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-dirty', action='store_true', help='tests only')
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        payload = repro(args) if args.mode == 'repro' else sweep(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({k: v for k, v in payload.items() if k != 'identity'}, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0 if payload.get('verdict', 'REPRO_PASS') == 'REPRO_PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
