"""Bounded, deterministic Stage 4 plumbing gate; synthetic labels imply no strength."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
from time import perf_counter

import torch

from model.config import coordinate_to_action
from model.encoding import encode_game
from model.masking import legal_moves_to_mask, masked_softmax, policy_loss, value_loss
from model.network import PolicyValueNet
from renju import Game


def make_dataset(seed: int = 42, samples: int = 32):
    """Independent legal prefixes, random legal one-hot actions, cyclic -1/0/+1.

    Labels are fixed synthetic memorization targets, not game outcomes. Every
    sample is non-terminal. A local RNG isolates the corpus from model init.
    """
    if samples < 3:
        raise ValueError('at least three samples are required for all value labels')
    rng = random.Random(seed)
    encoded, masks, policies, values, records = [], [], [], [], []
    for index in range(samples):
        for _ in range(100):
            game = Game()
            for _ in range(6 + index % 19):
                game.play(*rng.choice(game.legal_moves()))
                if game.done:
                    break
            moves = game.legal_moves()
            if not game.done and moves:
                break
        else:
            raise RuntimeError('could not generate a non-terminal sample')
        action = coordinate_to_action(*rng.choice(moves))
        mask = legal_moves_to_mask(moves)
        target = torch.zeros_like(mask, dtype=torch.float32)
        target[action] = 1
        value = float(index % 3 - 1)
        encoded.append(encode_game(game, mask))
        masks.append(mask)
        policies.append(target)
        values.append([value])
        records.append(dict(history=game.history, action=action, value=value))
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return (torch.stack(encoded), torch.stack(masks), torch.stack(policies),
            torch.tensor(values), digest)


def metrics(logits, value, mask, target, target_value):
    probabilities = masked_softmax(logits, mask)
    pl = policy_loss(logits, target, mask)
    vl = value_loss(value, target_value)
    return dict(policy_loss=pl.item(), value_mse=vl.item(), loss=(pl + vl).item(),
                masked_top1=(probabilities.argmax(-1) == target.argmax(-1)).float().mean().item(),
                nonfinite=int((~torch.isfinite(probabilities)).sum() + (~torch.isfinite(value)).sum()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, help='optional JSON report path')
    args = parser.parse_args()
    if min(args.max_steps, args.eval_every, args.threads) <= 0 or args.samples < 3:
        parser.error('positive steps/threads and at least 3 samples required')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    x, mask, target, target_value, digest = make_dataset(args.seed, args.samples)
    model = PolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    curve = []
    started = perf_counter()
    passed = False
    for step in range(args.max_steps + 1):
        if step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, value = model(x)
            loss = policy_loss(logits, target, mask) + value_loss(value, target_value)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f'nonfinite/missing gradient: {name}')
            optimizer.step()
        if step % args.eval_every == 0 or step == args.max_steps:
            # Preserve BN buffers: train-mode diagnostics must not calibrate eval.
            buffers = {name: buf.clone() for name, buf in model.named_buffers()}
            model.train()
            with torch.no_grad():
                train = metrics(*model(x), mask, target, target_value)
                for name, buf in model.named_buffers():
                    buf.copy_(buffers[name])
            model.eval()
            with torch.inference_mode():
                evaluation = metrics(*model(x), mask, target, target_value)
            record = dict(step=step, train=train, eval=evaluation)
            curve.append(record)
            print(json.dumps(record), flush=True)
            passed = (evaluation['masked_top1'] >= .95 and evaluation['value_mse'] <= .01
                      and evaluation['nonfinite'] == 0)
            if passed:
                break
    report = dict(seed=args.seed, samples=args.samples, steps=step,
                  max_steps=args.max_steps, eval_every=args.eval_every,
                  learning_rate=.001, optimizer='Adam', dataset_sha256=digest,
                  model_config=asdict(model.config),
                  parameters=sum(p.numel() for p in model.parameters()),
                  python=platform.python_version(), torch=str(torch.__version__),
                  threads=torch.get_num_threads(), device='cpu',
                  elapsed_seconds=perf_counter() - started, curve=curve,
                  result='PASS' if passed else 'FAIL')
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'curve'}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
