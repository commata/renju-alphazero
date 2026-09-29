"""Teacher arm: branch a run at generation N and fine-tune its weights on proven tactics.

The control arm is the plain continuation of the same run from generation N. This
script makes the teacher arm so that the ONLY difference is the network weights:

1. ``branch_stage8_run.branch`` copies generation N (model, optimizer, replay, RNGs)
   and the results of generations <= N into ``--dest``;
2. the copied results of generation N itself are removed (they evaluated the old
   weights; the orchestrator recomputes them for the new ones);
3. the weights are fine-tuned on the proof-labelled tactical dataset
   (``build_tactical_dataset.py``), each batch mixed with samples from the run's own
   replay buffer (colour-balanced when the run uses balanced sampling) so the
   self-play knowledge is not overwritten. Rows without a policy label (``forced_loss``,
   ``vcf_loss``) train only the value head; ``must_block`` trains only the policy;
4. the fine-tuned ``model_state_dict`` replaces it in ``checkpoint_genNNN.pt`` and
   ``latest.pt``. Config, optimizer state, replay buffer, RNGs and generation are
   untouched, so the branch resumes like the control arm with the same seeds.

``TEACHER.json`` records the dataset hash, settings and before/after metrics: per-kind
top-1 / value-sign accuracy on the tactical set and policy/value loss on a fixed
replay sample (to see how much self-play fit the fine-tune costs).

    python scripts/make_teacher_branch.py --source runs/stage8_g3_b --generation 400 \
        --dataset runs/teacher/tactical.json --dest runs/stage8_b400_teacher
    python scripts/run_stage8_training.py --run-dir runs/stage8_b400_teacher \
        --config configs/stage8_g3_b.yaml --anchor 400 \
        --h2h-anchor B400=runs/stage8_g3_b/checkpoints/checkpoint_gen400.pt
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from branch_stage8_run import branch  # noqa: E402
from model.config import ModelConfig, coordinate_to_action  # noqa: E402
from model.encoding import encode_game  # noqa: E402
from model.masking import legal_moves_to_mask, mask_policy_logits  # noqa: E402
from model.network import PolicyValueNet  # noqa: E402
from model.symmetry import transform_mask, transform_policy, transform_spatial  # noqa: E402
from renju import Game  # noqa: E402
from training.dataset import build_batch  # noqa: E402
from training.replay_buffer import ReplayBuffer  # noqa: E402
from training.training_checkpoint import load_checkpoint_payload, save_atomic  # noqa: E402

TEACHER_FORMAT = 'teacher-branch-v1'


def load_tactical(path: Path) -> dict:
    data = json.loads(path.read_bytes())
    if data.get('format') != 'tactical-dataset-v1':
        raise ValueError(f"unsupported dataset format: {data.get('format')!r}")
    states, policies, masks, values, has_policy, has_value, kinds = [], [], [], [], [], [], []
    for item in data['positions']:
        game = Game()
        for move in item['moves']:
            game.play(*move)
        mask = legal_moves_to_mask(game.legal_moves())
        policy = torch.zeros(225)
        if item['policy']:
            for move in item['policy']:
                policy[coordinate_to_action(*move)] = 1.0 / len(item['policy'])
        else:
            policy[mask.nonzero()[0]] = 1.0  # placeholder; weight 0 below
        states.append(encode_game(game, mask))
        policies.append(policy)
        masks.append(mask)
        values.append(float(item['value'] or 0))
        has_policy.append(bool(item['policy']))
        has_value.append(item['value'] is not None)
        kinds.append(item['kind'])
    return {'states': torch.stack(states), 'policies': torch.stack(policies),
            'masks': torch.stack(masks), 'values': torch.tensor(values).unsqueeze(1),
            'has_policy': torch.tensor(has_policy), 'has_value': torch.tensor(has_value),
            'kinds': kinds, 'size': len(kinds)}


def _augment(states, policies, masks, rng: Random):
    out = ([], [], [])
    for x, p, m in zip(states, policies, masks):
        k = rng.randrange(8)
        out[0].append(transform_spatial(x, k))
        out[1].append(transform_policy(p, k))
        out[2].append(transform_mask(m, k))
    return tuple(torch.stack(t) for t in out)


def weighted_losses(model, states, policies, masks, values, policy_w, value_w):
    logits, predicted = model(states)
    log_probs = F.log_softmax(mask_policy_logits(logits, masks), dim=-1).masked_fill(~masks, 0)
    policy_rows = -(policies * log_probs).sum(-1)
    value_rows = (predicted - values).pow(2).squeeze(1)
    policy = (policy_rows * policy_w).sum() / policy_w.sum().clamp(min=1)
    value = (value_rows * value_w).sum() / value_w.sum().clamp(min=1)
    return policy, value, logits, predicted


@torch.no_grad()
def tactical_metrics(model, data: dict) -> dict:
    model.eval()
    logits, predicted = model(data['states'])
    top1 = mask_policy_logits(logits, data['masks']).argmax(-1)
    hit = data['policies'][torch.arange(data['size']), top1] > 0
    result = {}
    for kind in sorted(set(data['kinds'])):
        rows = torch.tensor([k == kind for k in data['kinds']])
        entry = {'count': int(rows.sum())}
        policy_rows = rows & data['has_policy']
        if policy_rows.any():
            entry['top1'] = float(hit[policy_rows].float().mean())
        value_rows = rows & data['has_value']
        if value_rows.any():
            signs = (predicted[value_rows] * data['values'][value_rows]) > 0
            entry['value_sign_accuracy'] = float(signs.float().mean())
        result[kind] = entry
    model.train()
    return result


@torch.no_grad()
def replay_losses(model, batch) -> dict:
    model.eval()
    ones = torch.ones(batch.states.shape[0])
    policy, value, _, _ = weighted_losses(model, batch.states, batch.policies, batch.legal_masks,
                                          batch.values, ones, ones)
    model.train()
    return {'policy_loss': float(policy), 'value_loss': float(value)}


def fine_tune(model, data: dict, buffer: ReplayBuffer, *, steps: int, batch_size: int,
              teacher_fraction: float, lr: float, value_weight: float, balanced: bool,
              seed: int, balance_kinds: bool = False, kind_weights: dict | None = None,
              log=print) -> list[dict]:
    rng = Random(seed)
    by_kind = {}
    for position, kind in enumerate(data['kinds']):
        by_kind.setdefault(kind, []).append(position)
    kind_names = sorted(by_kind)
    weights = [kind_weights.get(kind, 1.0) for kind in kind_names] if kind_weights else None

    def draw() -> int:
        if weights is not None:  # kind by the given weights, then a position of that kind
            return rng.choice(by_kind[rng.choices(kind_names, weights)[0]])
        if balance_kinds:  # kind uniformly, then a position of that kind
            return rng.choice(by_kind[rng.choice(kind_names)])
        return rng.randrange(data['size'])

    sample_rng, augment_rng = Random(seed + 1), Random(seed + 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    teacher_rows = max(1, round(batch_size * teacher_fraction))
    replay_rows = batch_size - teacher_rows if len(buffer) else 0
    history = []
    model.train()
    for step in range(steps):
        index = torch.tensor([draw() for _ in range(teacher_rows)])
        states, policies, masks = _augment(data['states'][index], data['policies'][index],
                                           data['masks'][index], augment_rng)
        values = data['values'][index]
        policy_w = data['has_policy'][index].float()
        value_w = data['has_value'][index].float()
        if replay_rows:
            batch = build_batch(buffer, replay_rows, sample_rng=sample_rng,
                                augment_rng=augment_rng, augment=True, balanced=balanced)
            states = torch.cat([states, batch.states])
            policies = torch.cat([policies, batch.policies])
            masks = torch.cat([masks, batch.legal_masks])
            values = torch.cat([values, batch.values])
            policy_w = torch.cat([policy_w, torch.ones(replay_rows)])
            value_w = torch.cat([value_w, torch.ones(replay_rows)])
        policy, value, _, _ = weighted_losses(model, states, policies, masks, values,
                                              policy_w, value_w)
        total = policy + value_weight * value
        if not torch.isfinite(total):
            raise RuntimeError(f'non-finite fine-tune loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            history.append({'step': step, 'policy_loss': policy.item(),
                            'value_loss': value.item()})
            log(f'step {step}: policy {policy.item():.4f} value {value.item():.4f}')
    return history


def kind_shares(kinds: list[str], balance_kinds: bool, kind_weights: dict | None) -> dict:
    """Expected share of teacher rows per kind (recorded so T1/T2 recipes can be compared)."""
    counts = Counter(kinds)
    if kind_weights:
        raw = {kind: kind_weights.get(kind, 1.0) for kind in counts}
    elif balance_kinds:
        raw = {kind: 1.0 for kind in counts}
    else:
        raw = dict(counts)
    total = sum(raw.values())
    return {kind: round(value / total, 4) for kind, value in sorted(raw.items())}


def parse_kind_weights(text: str | None) -> dict | None:
    """``kind=w,kind=w`` -> dict; unlisted kinds weigh 1."""
    if not text:
        return None
    weights = {}
    for item in text.split(','):
        kind, _, value = item.partition('=')
        weights[kind.strip()] = float(value)
    if any(w < 0 for w in weights.values()) or not any(w > 0 for w in weights.values()):
        raise ValueError('kind weights must be >= 0 with at least one > 0')
    return weights


def make_teacher_branch(source: Path, generation: int, dataset: Path, dest: Path, *,
                        steps: int = 1000, batch_size: int = 64, teacher_fraction: float = 0.5,
                        lr: float = 2e-4, seed: int = 0, balance_kinds: bool = False,
                        kind_weights: dict | None = None, log=print) -> dict:
    record = branch(source, generation, dest)
    removed = []
    for sub in ('external_eval', 'probes'):
        for path in sorted((dest / sub).glob(f'gen{generation:03d}*')) if (dest / sub).is_dir() else []:
            path.unlink()
            removed.append(f'{sub}/{path.name}')

    latest = dest / 'checkpoints' / 'latest.pt'
    payload = load_checkpoint_payload(latest)
    config = payload['config']
    model = PolicyValueNet(ModelConfig(**payload['model_config']))
    model.load_state_dict(payload['model_state_dict'])
    buffer = ReplayBuffer(config['training']['replay_capacity'])
    buffer.load_state_dict(payload['replay_buffer'])
    data = load_tactical(dataset)
    balanced = bool(config['training'].get('balanced_sampling', False))
    probe_batch = (build_batch(buffer, min(1024, len(buffer)), sample_rng=Random(seed + 7),
                               augment_rng=Random(seed + 8), augment=False)
                   if len(buffer) else None)

    before = {'tactical': tactical_metrics(model, data),
              'replay': replay_losses(model, probe_batch) if probe_batch else None}
    history = fine_tune(model, data, buffer, steps=steps, batch_size=batch_size,
                        teacher_fraction=teacher_fraction, lr=lr,
                        value_weight=config['loss']['value_weight'], balanced=balanced,
                        seed=seed, balance_kinds=balance_kinds, kind_weights=kind_weights,
                        log=log)
    after = {'tactical': tactical_metrics(model, data),
             'replay': replay_losses(model, probe_batch) if probe_batch else None}

    payload['model_state_dict'] = model.state_dict()
    save_atomic(dest / 'checkpoints' / f'checkpoint_gen{generation:03d}.pt', payload)
    save_atomic(latest, payload)
    result = {
        'format': TEACHER_FORMAT, 'branch': record, 'removed_results': removed,
        'dataset': str(dataset), 'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'dataset_kinds': dict(Counter(data['kinds'])),
        'settings': {'steps': steps, 'batch_size': batch_size,
                     'teacher_fraction': teacher_fraction, 'lr': lr, 'seed': seed,
                     'balance_kinds': balance_kinds, 'kind_weights': kind_weights,
                     'kind_shares': kind_shares(data['kinds'], balance_kinds, kind_weights),
                     'teacher_rows_drawn': steps * max(1, round(batch_size * teacher_fraction)),
                     'replay_balanced': balanced, 'optimizer': 'fresh Adam (fine-tune only); '
                     'the run optimizer state is kept unchanged'},
        'before': before, 'after': after, 'history': history,
        'checkpoint_sha256': hashlib.sha256(latest.read_bytes()).hexdigest(),
    }
    (dest / 'TEACHER.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--teacher-fraction', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--balance-kinds', action='store_true',
                        help='draw each teacher row from a uniformly chosen kind')
    parser.add_argument('--kind-weights',
                        help='kind=w,... relative kind weights (unlisted kinds weigh 1); '
                             'overrides --balance-kinds')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = make_teacher_branch(args.source, args.generation, args.dataset, args.dest,
                                 steps=args.steps, batch_size=args.batch_size,
                                 teacher_fraction=args.teacher_fraction, lr=args.lr,
                                 seed=args.seed, balance_kinds=args.balance_kinds,
                                 kind_weights=parse_kind_weights(args.kind_weights))
    print(json.dumps({'dataset_kinds': result['dataset_kinds'],
                      'kind_shares': result['settings']['kind_shares'],
                      'before': result['before'], 'after': result['after']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
