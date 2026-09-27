"""Stage 7 supervised sanity checks: can the Stage 4 network learn tactics at all?

Separates "implementation bug" from "no learning signal in self-play targets":

A. memorize: train a fresh network on the policy/value-labelled probe set. Uses the
   production encoder, coordinates and ``model.masking`` policy/value losses. Failing
   to reach ~100% top-1 would point at an implementation bug.
B. generalize: train on every tactical position (immediate_win / must_block /
   forced_loss) of the V7-vs-V5 benchmark games with D4 augmentation, then test on the
   probes taken from the disjoint V7-vs-V6 games.

Recorded result (CPU, seed 0, 4 threads; small float differences across machines are
possible): A reaches top-1 1.00 / value 1.00 for every kind by step 100. B (1,098 labels:
913 must_block, 98 immediate_win, 87 forced_loss) reaches held-out must_block top-1
0.78 / 0.67 and forced_loss value accuracy 0.85 / 0.75 at steps 1,000 / 1,500, versus
0.00 top-1 and 0.45 balanced value accuracy for the Stage 7-A gen 30 self-play network.
Held-out immediate_win top-1 stays <= 0.11 (only 98 training labels).

    python scripts/run_stage7_supervised_sanity.py --output runs/stage7_supervised_sanity.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from build_stage7_probes import classify_position  # noqa: E402
from model.config import ModelConfig, coordinate_to_action  # noqa: E402
from model.encoding import encode_game  # noqa: E402
from model.masking import (legal_moves_to_mask, masked_softmax, policy_loss,  # noqa: E402
                           value_loss)
from model.network import PolicyValueNet  # noqa: E402
from model.symmetry import transform_mask, transform_policy, transform_spatial  # noqa: E402
from renju import Game  # noqa: E402

PROBES = ROOT / 'tests' / 'fixtures' / 'stage7_probes_v1.json'
TRAIN_GAMES = ROOT / 'docs' / 'mcts-v7-results' / 'v7_vs_v5_seed6507.json'
HELDOUT_SOURCE = 'v7_vs_v6_seed6507.json'


def to_tensors(items):
    states, policies, values, masks, kinds = [], [], [], [], []
    for item in items:
        game = Game()
        for move in item['moves']:
            game.play(*move)
        mask = legal_moves_to_mask(game.legal_moves())
        policy = torch.zeros(225)
        correct = [coordinate_to_action(*m) for m in item['correct_moves']]
        if correct:
            policy[correct] = 1 / len(correct)
        states.append(encode_game(game, mask))
        policies.append(policy)
        masks.append(mask)
        sign = item['value_sign']
        values.append(float(sign) if sign is not None else float('nan'))
        kinds.append(item['kind'])
    return (torch.stack(states), torch.stack(policies), torch.tensor(values),
            torch.stack(masks), kinds)


def evaluate(model, data) -> dict:
    states, policies, values, masks, kinds = data
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits, predicted = model(states)
        policy = masked_softmax(logits, masks)
        predicted = predicted.reshape(-1)
    model.train(was_training)
    out = {}
    for kind in sorted(set(kinds)):
        index = [i for i, k in enumerate(kinds) if k == kind]
        summary = {'positions': len(index)}
        with_policy = [i for i in index if policies[i].sum() > 0]
        if with_policy:
            summary['top1'] = sum(float(policies[i][policy[i].argmax()] > 0)
                                  for i in with_policy) / len(with_policy)
        with_value = [i for i in index if not torch.isnan(values[i])]
        if with_value:
            summary['value_sign_accuracy'] = sum(float(predicted[i] * values[i] > 0)
                                                 for i in with_value) / len(with_value)
        out[kind] = summary
    return out


def train(data, *, steps: int, augment: bool, seed: int, evals: dict, log_every: int):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = PolicyValueNet(ModelConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    states, policies, values, masks, _ = data
    has_policy = policies.sum(1) > 0
    has_value = ~torch.isnan(values)
    history = []
    for step in range(1, steps + 1):
        batch = torch.randint(0, len(states), (min(64, len(states)),))
        x, p, v, m = states[batch], policies[batch], values[batch], masks[batch]
        if augment:
            symmetry = rng.randrange(8)
            x = transform_spatial(x, symmetry)
            p = transform_policy(p, symmetry)
            m = transform_mask(m, symmetry)
        logits, predicted = model(x)
        loss = torch.zeros(())
        hp, hv = has_policy[batch], has_value[batch]
        if hp.any():
            loss = loss + policy_loss(logits[hp], p[hp], m[hp])
        if hv.any():
            loss = loss + value_loss(predicted[hv], v[hv].unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == steps:
            record = {'step': step, 'loss': float(loss.detach()),
                      **{name: evaluate(model, d) for name, d in evals.items()}}
            history.append(record)
            print(json.dumps(record), flush=True)
    return history


def benchmark_positions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding='utf-8'))
    found = {}
    for record in data['games']:
        game = Game()
        for ply, move in enumerate(record['moves'], 1):
            if ply > 1:
                for item in classify_position(game, {}):
                    found.setdefault(json.dumps(item['moves']) + item['kind'], item)
            game.play(*move)
    return list(found.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--memorize-steps', type=int, default=400)
    parser.add_argument('--generalize-steps', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    probes = [p for p in json.loads(PROBES.read_text(encoding='utf-8'))['probes']
              if p['kind'] != 'avoid']
    print(f'A) memorize {len(probes)} probes', flush=True)
    started = perf_counter()
    probe_data = to_tensors(probes)
    memorize = train(probe_data, steps=args.memorize_steps, augment=False, seed=args.seed,
                     evals={'train': probe_data}, log_every=100)
    memorize_seconds = perf_counter() - started

    print('B) generalize: V7-vs-V5 positions -> V7-vs-V6 probes', flush=True)
    started = perf_counter()
    train_items = benchmark_positions(TRAIN_GAMES)
    heldout = [p for p in probes if p['source'].get('file') == HELDOUT_SOURCE]
    print(json.dumps({'train': Counter(i['kind'] for i in train_items),
                      'heldout': Counter(p['kind'] for p in heldout)}), flush=True)
    train_data = to_tensors(train_items)
    heldout_data = to_tensors(heldout)
    generalize = train(train_data, steps=args.generalize_steps, augment=True, seed=args.seed,
                       evals={'train': train_data, 'heldout': heldout_data}, log_every=500)
    generalize_seconds = perf_counter() - started

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            'format_version': 'stage7-supervised-sanity-v1', 'seed': args.seed,
            'memorize': {'history': memorize, 'seconds': memorize_seconds},
            'generalize': {'train_positions': dict(Counter(i['kind'] for i in train_items)),
                           'heldout_positions': dict(Counter(p['kind'] for p in heldout)),
                           'history': generalize, 'seconds': generalize_seconds},
        }, indent=1), encoding='utf-8')
        print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
