"""H3 policy-only supervised pretraining on RenjuNet (docs/mcts-v8-teacher.md §12.11).

- Model: ``model.network.PolicyValueNet`` (64x4 by default), random init.
- Policy only: the forward pass is ``policy_head(trunk(x))``; the value head is never
  called, so its weights AND its BatchNorm running statistics stay at their initial
  values (``value_trained: false`` in the checkpoint metadata).
- Data: ``hybrid.h3_cache``; a random D4 symmetry per sample during training.
- Loss: masked policy cross-entropy against the one-hot human move.
- Resume: a training snapshot (model, optimizer, scheduler, step, data order, RNG) is
  written atomically every ``checkpoint_every`` steps; a resumed run continues the
  same sample order and augmentation stream.
- Selection: the validation metric (masked top-1, ties by lower cross-entropy) picks
  ``best.pt``; the must_block probe gate is run once on the selected weights
  (``scripts/eval_h3_gate.py``), never for checkpoint selection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from model.checkpoint import save_checkpoint
from model.config import ACTION_COUNT, ModelConfig
from model.network import PolicyValueNet
from model.symmetry import transform_action

from .h3_cache import load_cache, planes_from_cache

SNAPSHOT_FORMAT = 'h3-train-snapshot-v1'
PLY_BUCKETS = ((5, 9), (10, 14), (15, 19), (20, 29), (30, 49), (50, 1000))


@dataclass
class H3Config:
    cache_dir: str = 'data/external/renjunet/h3_cache'
    out_dir: str = 'runs/h3_policy'
    device: str = 'cpu'
    torch_threads: int = 0  # 0 = PyTorch default
    seed: int = 20261003
    channels: int = 64
    blocks: int = 4
    batch_size: int = 512
    epochs: float = 10.0
    max_steps: int = 0  # 0 = epochs decide; >0 stops earlier (smoke tests)
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.02
    augment_d4: bool = True
    amp_bf16: bool = False  # CUDA only
    eval_every: int = 2000
    eval_max_states: int = 0  # 0 = the whole validation split
    checkpoint_every: int = 2000
    log_every: int = 100

    @classmethod
    def from_dict(cls, data: dict) -> 'H3Config':
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown H3 config key(s): {', '.join(sorted(unknown))}")
        return cls(**data)


def load_config(path: Path, overrides: dict | None = None) -> H3Config:
    import yaml
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    return H3Config.from_dict({**data, **(overrides or {})})


def policy_logits(model: PolicyValueNet, x: torch.Tensor) -> torch.Tensor:
    """Policy-only forward: the value head is not run (its BatchNorm stays untouched)."""
    return model.policy_head(model.trunk(x))


def d4_tables(device) -> tuple[torch.Tensor, torch.Tensor]:
    """dest[s, a] = action a after symmetry s; src[s, b] = the action that lands on b."""
    dest = torch.tensor([[transform_action(a, s) for a in range(ACTION_COUNT)] for s in range(8)], dtype=torch.long)
    src = torch.empty_like(dest)
    for s in range(8):
        src[s, dest[s]] = torch.arange(ACTION_COUNT)
    return dest.to(device), src.to(device)


def augment(planes, legal, target, symmetry, dest, src):
    """Apply a per-sample D4 symmetry to planes [B,6,15,15], legal [B,225] and target [B]."""
    b, c = planes.shape[:2]
    index = src[symmetry]  # [B,225]
    flat = planes.reshape(b, c, ACTION_COUNT).gather(2, index[:, None, :].expand(b, c, ACTION_COUNT))
    return (flat.reshape_as(planes), legal.gather(1, index),
            dest[symmetry, target])


def masked_ce(logits, legal, target):
    masked = logits.masked_fill(~legal, -torch.inf)
    return F.cross_entropy(masked, target)


def lr_factor(step: int, total: int, cfg: H3Config) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, data, cfg: H3Config, device, *, max_states: int = 0, batch_size: int = 2048) -> dict:
    """Masked top-1/3/5 and cross-entropy (main), raw illegal diagnostics, accuracy by ply bucket."""
    was_training = model.training
    model.eval()
    n = data['target'].shape[0]
    order = torch.arange(n)
    if max_states and max_states < n:
        order = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed))[:max_states].sort().values
    sums = {'ce': 0.0, 'top1': 0, 'top3': 0, 'top5': 0, 'raw_top1_legal': 0, 'raw_illegal_mass': 0.0}
    buckets = {f'{lo}-{hi}': [0, 0] for lo, hi in PLY_BUCKETS}
    for start in range(0, order.shape[0], batch_size):
        index = order[start:start + batch_size]
        planes, legal, target = planes_from_cache(data, index, device)
        logits = policy_logits(model, planes).float()
        masked = logits.masked_fill(~legal, -torch.inf)
        sums['ce'] += float(F.cross_entropy(masked, target, reduction='sum'))
        top = masked.topk(5, dim=1).indices
        hit = top == target[:, None]
        sums['top1'] += int(hit[:, 0].sum())
        sums['top3'] += int(hit[:, :3].any(1).sum())
        sums['top5'] += int(hit.any(1).sum())
        raw = torch.softmax(logits, dim=1)
        sums['raw_top1_legal'] += int(legal.gather(1, logits.argmax(1, keepdim=True)).sum())
        sums['raw_illegal_mass'] += float((raw * (~legal)).sum())
        ply = data['ply'][index]
        for lo, hi in PLY_BUCKETS:
            inside = ((ply >= lo) & (ply <= hi)).to(device)
            buckets[f'{lo}-{hi}'][0] += int(inside.sum())
            buckets[f'{lo}-{hi}'][1] += int((hit[:, 0] & inside).sum())
    model.train(was_training)
    m = order.shape[0]
    return {'states': m, 'ce': sums['ce'] / m, 'top1': sums['top1'] / m, 'top3': sums['top3'] / m,
            'top5': sums['top5'] / m,
            'diagnostic': {'raw_top1_legal_rate': sums['raw_top1_legal'] / m,
                           'raw_illegal_mass': sums['raw_illegal_mass'] / m},
            'top1_by_ply': {k: (v[1] / v[0] if v[0] else None) for k, v in buckets.items()},
            'states_by_ply': {k: v[0] for k, v in buckets.items()}}


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _metadata(cfg: H3Config, cache_manifest: dict, step: int, metrics: dict | None) -> dict:
    return {'stage': 'H3', 'track': 'B', 'policy_trained': True, 'value_trained': False,
            'source': 'RenjuNet (offline, non-commercial; do not publish online)',
            'h2_output_sha256': cache_manifest['h2_output_sha256'],
            'cache_sha256': {k: v['sha256'] for k, v in cache_manifest['splits'].items()},
            'rules_sha256': cache_manifest['rules_sha256'], 'step': step, 'config': asdict(cfg),
            'val': metrics}


class Trainer:
    def __init__(self, cfg: H3Config, *, log=print):
        self.cfg, self.log = cfg, log
        if cfg.torch_threads:
            torch.set_num_threads(cfg.torch_threads)
        if cfg.device == 'cpu':
            torch.set_flush_denormal(True)  # Stage 8: subnormal Adam state slowed CPU training 3x
        self.device = torch.device(cfg.device)
        cache = Path(cfg.cache_dir)
        self.train_data, self.manifest = load_cache(cache, 'train')
        self.val_data, _ = load_cache(cache, 'val')
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        self.model = PolicyValueNet(ModelConfig(channels=cfg.channels, blocks=cfg.blocks)).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate,
                                           weight_decay=cfg.weight_decay)
        n = self.train_data['target'].shape[0]
        self.steps_per_epoch = n // cfg.batch_size
        if self.steps_per_epoch < 1:
            raise ValueError('batch_size is larger than the training split')
        total = int(cfg.epochs * self.steps_per_epoch)
        self.total_steps = min(total, cfg.max_steps) if cfg.max_steps else total
        schedule_steps = self.total_steps  # fixed here: stopping early must not change the schedule
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: lr_factor(s, schedule_steps, cfg))
        self.dest, self.src = d4_tables(self.device)
        self.step = 0
        self.augment_rng = torch.Generator().manual_seed(cfg.seed + 1)
        self.best = None

    def _order(self, epoch: int) -> torch.Tensor:
        n = self.train_data['target'].shape[0]
        return torch.randperm(n, generator=torch.Generator().manual_seed(self.cfg.seed * 1000 + epoch))

    def snapshot(self) -> dict:
        return {'format': SNAPSHOT_FORMAT, 'config': asdict(self.cfg), 'step': self.step,
                'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(), 'augment_rng': self.augment_rng.get_state(),
                'torch_rng': torch.get_rng_state(), 'best': self.best,
                'h2_output_sha256': self.manifest['h2_output_sha256'],
                'cache_sha256': {k: v['sha256'] for k, v in self.manifest['splits'].items()}}

    def resume(self, path: Path) -> None:
        snap = torch.load(path, map_location=self.device, weights_only=False)
        if snap.get('format') != SNAPSHOT_FORMAT:
            raise ValueError('not an H3 training snapshot')
        if snap['h2_output_sha256'] != self.manifest['h2_output_sha256'] or snap['cache_sha256'] != {
                k: v['sha256'] for k, v in self.manifest['splits'].items()}:
            raise ValueError('snapshot was trained on a different cache')
        critical = ('seed', 'channels', 'blocks', 'batch_size', 'epochs', 'max_steps', 'learning_rate',
                    'weight_decay', 'warmup_steps', 'min_lr_ratio', 'augment_d4')
        changed = [k for k in critical if snap['config'][k] != getattr(self.cfg, k)]
        if changed:
            raise ValueError(f'training-critical config changed since the snapshot: {changed}')
        self.model.load_state_dict(snap['model'])
        self.optimizer.load_state_dict(snap['optimizer'])
        self.scheduler.load_state_dict(snap['scheduler'])
        self.augment_rng.set_state(snap['augment_rng'])
        torch.set_rng_state(snap['torch_rng'])
        self.step, self.best = snap['step'], snap['best']

    def _validate(self) -> dict:
        metrics = evaluate(self.model, self.val_data, self.cfg, self.device, max_states=self.cfg.eval_max_states)
        key = (metrics['top1'], -metrics['ce'])
        if self.best is None or key > tuple(self.best['key']):
            self.best = {'key': list(key), 'step': self.step, 'metrics': metrics}
            save_checkpoint(self.out / 'best.pt', self.model)
            (self.out / 'best.json').write_text(json.dumps(_metadata(self.cfg, self.manifest, self.step, metrics),
                                                           indent=1), encoding='utf-8')
        with (self.out / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'step': self.step, 'val': metrics}) + '\n')
        self.log(f"step {self.step} val top1 {metrics['top1']:.4f} top3 {metrics['top3']:.4f} "
                 f"top5 {metrics['top5']:.4f} ce {metrics['ce']:.4f}")
        return metrics

    def train(self) -> dict:
        cfg = self.cfg
        self.model.train()
        autocast = (torch.autocast('cuda', dtype=torch.bfloat16) if cfg.amp_bf16 and self.device.type == 'cuda'
                    else torch.autocast('cpu', enabled=False))
        started, seen, loss_sum, loss_n = perf_counter(), 0, 0.0, 0
        window_started, window_seen = started, 0  # samples/s per log window, not since start (CUDA warm-up)
        while self.step < self.total_steps:
            epoch, offset = divmod(self.step, self.steps_per_epoch)
            index = self._order(epoch)[offset * cfg.batch_size:(offset + 1) * cfg.batch_size]
            planes, legal, target = planes_from_cache(self.train_data, index, self.device)
            symmetry = torch.randint(0, 8 if cfg.augment_d4 else 1, (index.shape[0],), generator=self.augment_rng)
            if cfg.augment_d4:
                planes, legal, target = augment(planes, legal, target, symmetry.to(self.device), self.dest, self.src)
            with autocast:
                logits = policy_logits(self.model, planes)
            loss = masked_ce(logits.float(), legal, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss at step {self.step}')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            seen += index.shape[0]
            window_seen += index.shape[0]
            loss_sum, loss_n = loss_sum + float(loss.detach()), loss_n + 1
            if self.step % cfg.log_every == 0 or self.step == self.total_steps:
                now = perf_counter()
                rate = window_seen / max(1e-9, now - window_started)
                window_started, window_seen = now, 0
                record = {'step': self.step, 'epoch': self.step / self.steps_per_epoch, 'loss': loss_sum / loss_n,
                          'lr': self.scheduler.get_last_lr()[0], 'samples_per_second': rate}
                with (self.out / 'train.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record) + '\n')
                self.log(f"step {self.step}/{self.total_steps} loss {record['loss']:.4f} "
                         f"lr {record['lr']:.2e} {rate:.0f} samples/s")
                loss_sum, loss_n = 0.0, 0
            if self.step % cfg.eval_every == 0 or self.step == self.total_steps:
                self._validate()
            if self.step % cfg.checkpoint_every == 0 or self.step == self.total_steps:
                _atomic_save(self.snapshot(), self.out / 'snapshot.pt')
        save_checkpoint(self.out / 'last.pt', self.model)
        (self.out / 'last.json').write_text(json.dumps(_metadata(cfg, self.manifest, self.step, None), indent=1),
                                            encoding='utf-8')
        return {'step': self.step, 'best': self.best,
                'samples_per_second': seen / max(1e-9, perf_counter() - started)}
