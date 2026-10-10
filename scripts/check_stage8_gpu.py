"""Stage 8-C accelerator smoke gate (docs/stage8-plan.md §5.4).

Runs the checks that must pass before any GPU self-play work, on a real training
checkpoint (default: the Stage 7 D16 gen 160 copy), comparing ``--device`` with the CPU:

1. backend: torch / HIP versions, device name
2. inference parity: B=1 and B=16 priors/values vs CPU B=1 (max abs difference)
3. repeatability: the same device batch evaluated twice
4. training: ``--steps`` real training steps (replay buffer, augmentation, Adam) from the
   checkpoint's own state on the device, all finite; ms/step vs the CPU
5. checkpoint save on the device -> reload on the CPU, weights identical
6. ``torch.use_deterministic_algorithms(True)`` forward+backward on the device
7. evaluator batch benchmark B=1..64 on the device and the CPU (stage7 batch benchmark)

Nothing is written to the run directory. ROCm (TheRock) PyTorch uses the ``cuda``
device API, so ``--device cuda`` selects the RX 6600. ``--device cpu`` is a dry run of
the script itself.

    python scripts/check_stage8_gpu.py --checkpoint runs/stage8_d16/checkpoints/checkpoint_gen160.pt \
        --device cuda --output runs/stage8_gpu_smoke.json
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from benchmark_stage7_batch import benchmark, snapshots_for  # noqa: E402
from model.evaluator import PolicyValueEvaluator  # noqa: E402
from training.dataset import build_batch  # noqa: E402
from training.probes import load_model_from_training_checkpoint  # noqa: E402
from training.trainer import train_step  # noqa: E402
from training.training_checkpoint import (build_checkpoint, load_checkpoint_payload,  # noqa: E402
                                          load_training_state, save_atomic)

PARITY_TOLERANCE = 1e-3   # float32 kernels differ between devices; bit-exactness not required


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def backend_info(device: torch.device) -> dict:
    info = {'torch': str(torch.__version__), 'hip': getattr(torch.version, 'hip', None),
            'cuda': getattr(torch.version, 'cuda', None), 'python': platform.python_version(),
            'platform': platform.platform(), 'device': str(device)}
    if device.type == 'cuda':
        info.update(available=torch.cuda.is_available(), count=torch.cuda.device_count())
        if torch.cuda.is_available():
            info['name'] = torch.cuda.get_device_name(device)
    return info


def _max_diff(results, reference) -> dict:
    return {'prior': max(max(abs(a - b) for a, b in zip(r.priors, s.priors))
                         for r, s in zip(results, reference)),
            'value': max(abs(r.value - s.value) for r, s in zip(results, reference))}


def check_inference(model_cpu, device, snapshots) -> dict:
    reference = [PolicyValueEvaluator(model_cpu).evaluate_batch([s])[0] for s in snapshots]
    evaluator = PolicyValueEvaluator(deepcopy(model_cpu), device=device)
    single = [evaluator.evaluate_batch([s])[0] for s in snapshots]
    batched = [r for i in range(0, len(snapshots), 16)
               for r in evaluator.evaluate_batch(snapshots[i:i + 16])]
    again = [r for i in range(0, len(snapshots), 16)
             for r in evaluator.evaluate_batch(snapshots[i:i + 16])]
    b1, b16 = _max_diff(single, reference), _max_diff(batched, reference)
    repeat = _max_diff(again, batched)
    return {'b1_vs_cpu': b1, 'b16_vs_cpu': b16, 'repeat_b16': repeat,
            'tolerance': PARITY_TOLERANCE,
            'ok': max(b1['prior'], b1['value'], b16['prior'], b16['value']) < PARITY_TOLERANCE}


def _train(checkpoint: Path, device: torch.device, steps: int) -> tuple[dict, object]:
    requested = load_checkpoint_payload(checkpoint)['config']
    requested = {**requested, 'device': str(device)}
    state = load_training_state(checkpoint, requested)
    t = state.config['training']
    losses = []
    started = None
    for step in range(steps):
        batch = build_batch(state.buffer, t['batch_size'], sample_rng=state.sample_rng,
                            augment_rng=state.augment_rng,
                            augment=state.config['augmentation']['enabled'])
        if step == 1:  # exclude the first step (kernel selection / allocator warm-up)
            _sync(device)
            started = perf_counter()
        losses.append(train_step(state.model, state.optimizer, batch,
                                 state.config['loss']['value_weight'], t['grad_clip'],
                                 state.config['loss']['l2_coeff']))
    _sync(device)
    elapsed = perf_counter() - started if started is not None else 0.0
    finite = all(isfinite(v) for step in losses for v in step.values())
    return {'steps': steps, 'finite': finite, 'first_loss': losses[0]['total_loss'],
            'last_loss': losses[-1]['total_loss'],
            'ms_per_step': 1000 * elapsed / max(steps - 1, 1)}, state


def check_training(checkpoint: Path, device: torch.device, steps: int) -> tuple[dict, object]:
    on_device, state = _train(checkpoint, device, steps)
    on_cpu, _ = _train(checkpoint, torch.device('cpu'), steps)
    return {'device': on_device, 'cpu': on_cpu,
            'first_loss_diff_vs_cpu': abs(on_device['first_loss'] - on_cpu['first_loss']),
            'ok': on_device['finite']}, state


def check_checkpoint_roundtrip(state) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'gpu_roundtrip.pt'
        save_atomic(path, build_checkpoint(state))
        payload = load_checkpoint_payload(path, device='cpu')
    same = all(torch.equal(value.detach().cpu(), payload['model_state_dict'][key])
               for key, value in state.model.state_dict().items())
    return {'ok': same}


def check_deterministic(model_cpu, device, snapshots) -> dict:
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        model = deepcopy(model_cpu).to(device).train()
        evaluator = PolicyValueEvaluator(deepcopy(model_cpu))
        planes = torch.stack([evaluator.encode(s)[0] for s in snapshots[:16]]).to(device)
        logits, values = model(planes)
        (logits.sum() + values.sum()).backward()
        _sync(device)
        return {'ok': True}
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        return {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    finally:
        torch.use_deterministic_algorithms(previous)


def run_checks(checkpoint: Path, device_name: str, *, steps: int, positions: int,
               batch_sizes: list[int], repeats: int, log=print) -> dict:
    device = torch.device(device_name)
    report = {'format_version': 'stage8-gpu-smoke-v1', 'checkpoint': str(checkpoint),
              'backend': backend_info(device), 'checks': {}}
    log(f"backend: {report['backend']}")
    model_cpu, info = load_model_from_training_checkpoint(checkpoint)
    report['checkpoint_info'] = info
    snapshots = snapshots_for(positions)

    def run(name, fn):
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - every check reports, the gate continues
            result = {'ok': False, 'error': f'{type(exc).__name__}: {exc}',
                      'traceback': traceback.format_exc(limit=5)}
        report['checks'][name] = result
        log(f"{'PASS' if result.get('ok') else 'FAIL'} {name}: "
            f"{ {k: v for k, v in result.items() if k not in ('traceback', 'rows')} }")
        return result

    run('inference_parity', lambda: check_inference(model_cpu, device, snapshots))
    holder = {}

    def training():
        result, holder['state'] = check_training(checkpoint, device, steps)
        return result

    run('training', training)
    run('checkpoint_roundtrip', lambda: check_checkpoint_roundtrip(holder['state']))
    run('deterministic_algorithms', lambda: check_deterministic(model_cpu, device, snapshots))

    def bench():
        rows = {'device': benchmark(deepcopy(model_cpu), snapshots, batch_sizes, repeats, 1,
                                    device=str(device)),
                'device_batched': benchmark(deepcopy(model_cpu), snapshots, batch_sizes, repeats,
                                            1, device=str(device), batched_evaluator=True),
                'cpu': benchmark(deepcopy(model_cpu), snapshots, batch_sizes, repeats, 1)}
        for label, table in rows.items():
            log(f'  {label}: ' + ', '.join(f"B{r['batch']} {r['full_ms_per_position']:.2f}"
                                           for r in table) + ' ms/pos (full)')
        return {'ok': True, 'rows': rows}

    run('batch_benchmark', bench)
    report['ok'] = all(c.get('ok') for name, c in report['checks'].items()
                       if name != 'deterministic_algorithms')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cuda', help="'cuda' (ROCm/CUDA) or 'cpu' (dry run)")
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--positions', type=int, default=128)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error('--steps must be at least 2')
    torch.set_num_threads(args.threads)
    report = run_checks(args.checkpoint, args.device, steps=args.steps, positions=args.positions,
                        batch_sizes=args.batch_sizes, repeats=args.repeats,
                        log=lambda message: print(message, flush=True))
    print(f"GPU smoke gate: {'PASS' if report['ok'] else 'FAIL'}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
        print(f'wrote {args.output}')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
