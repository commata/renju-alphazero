"""H6-2a: value head only, on a frozen H3 network (docs/h6-value.md, docs/mcts-v8-teacher.md §12.19).

- Start: the H3 ``best.pt`` loaded strictly (``model.checkpoint.load_checkpoint``) with its ``best.json``,
  which must say stage H3, policy_trained true, value_trained false.
- Frozen: trunk and policy head, parameters AND BatchNorm running statistics. The whole model stays in
  eval mode; only ``value_head`` is switched to train mode (``model.train()`` is never called), the trunk
  runs under ``no_grad``, and the optimizer holds the value-head parameters only. The policy logits of the
  saved model are therefore bit-identical to H3's.
- Data: ``hybrid.h6_value_cache`` (labelled rows only; value from the side to move as H6-1 wrote it, no
  sign change). ``train`` gives the gradient, ``val`` picks ``best.pt``; ``test`` is never loaded (H6-3).
- Loss: weighted MSE of the tanh value, ``sum(w * (v - t)^2) / sum(w)`` (proof 1.0, outcome 0.1 by H6-1).
- D4: a random symmetry per sample on the six planes (the legal plane included); the target is unchanged.
- Resume: an atomic snapshot (value head, optimizer, scheduler, step, RNG); a resumed run refuses another
  H3 checkpoint, another label cache or a changed training-critical setting.
- Provenance: ``best.json`` names the parent H3 checkpoint and metadata, the label rows and the H6-1 report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch

from model.checkpoint import _git_provenance, load_checkpoint, save_checkpoint
from model.config import ACTION_INDEX_VERSION, ENCODER_VERSION, ModelConfig

from .h3_train import d4_tables
from .h6_value_cache import KIND_OUTCOME, KIND_PROOF, load_value_cache, planes_from_value_cache

STAGE = 'H6-2a'
SNAPSHOT_FORMAT = 'h6-value-snapshot-v1'
OPTIMIZER = 'adamw'
PLY_BUCKETS = ((5, 9), (10, 14), (15, 19), (20, 29), (30, 49), (50, 1000))
# A resumed run must match these; anything else (device, threads, logging) may change.
CRITICAL = ('seed', 'channels', 'blocks', 'batch_size', 'epochs', 'max_steps', 'learning_rate', 'weight_decay',
            'warmup_steps', 'min_lr_ratio', 'augment_d4', 'optimizer')


@dataclass
class H6ValueConfig:
    h3_checkpoint: str = 'runs/h3_policy_64x4/best.pt'
    h3_metadata: str = ''  # '' = best.json next to the checkpoint
    cache_dir: str = 'data/tactical/h6/value_cache'
    h6_report: str = 'docs/h6/h6_1_report.json'
    out_dir: str = 'runs/h6_value_head'
    device: str = 'cpu'
    torch_threads: int = 0
    seed: int = 20261012
    channels: int = 64
    blocks: int = 4
    optimizer: str = OPTIMIZER
    batch_size: int = 1024
    epochs: float = 3.0
    max_steps: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    min_lr_ratio: float = 0.05
    augment_d4: bool = True
    eval_every: int = 2000
    eval_max_states: int = 0
    checkpoint_every: int = 1000
    log_every: int = 100
    require_clean_git: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> 'H6ValueConfig':
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown H6 value config key(s): {', '.join(sorted(unknown))}")
        cfg = cls(**data)
        if cfg.optimizer != OPTIMIZER:
            raise ValueError(f'only optimizer {OPTIMIZER!r} is supported')
        return cfg

    @property
    def metadata_path(self) -> Path:
        return Path(self.h3_metadata) if self.h3_metadata else Path(self.h3_checkpoint).with_name('best.json')


def load_config(path: Path, overrides: dict | None = None) -> H6ValueConfig:
    import yaml
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    return H6ValueConfig.from_dict({**data, **(overrides or {})})


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def state_sha256(state: dict) -> str:
    """Content hash of a state_dict subset (names, dtypes, shapes, bytes)."""
    from .h3_cache import tensor_bytes
    digest = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        digest.update(f'{name}|{t.dtype}|{tuple(t.shape)}'.encode())
        digest.update(tensor_bytes(t))
    return digest.hexdigest()


def frozen_state(model) -> dict:
    """Trunk + policy head parameters and buffers (BatchNorm running stats included)."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith('value_head.')}


def freeze(model) -> list:
    """Freeze trunk and policy head (parameters and BN statistics); return the trainable parameters."""
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('value_head.'))
    model.eval()
    model.value_head.train()
    return list(model.value_head.parameters())


def value_forward(model, planes: torch.Tensor) -> torch.Tensor:
    """Frozen trunk under no_grad, value head with gradient; [B] in [-1, 1]."""
    if model.trunk.training or model.policy_head.training:
        raise RuntimeError('trunk / policy head must stay in eval mode (frozen BatchNorm)')
    with torch.no_grad():
        features = model.trunk(planes)
    return model.value_head(features).squeeze(1)


def weighted_mse(value: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (weight * (value - target).square()).sum() / weight.sum()


def augment_planes(planes: torch.Tensor, symmetry: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Per-sample D4 symmetry of [B,6,15,15] (all planes are spatial; the value target does not change)."""
    b, c = planes.shape[:2]
    index = src[symmetry]
    flat = planes.reshape(b, c, -1)
    return flat.gather(2, index[:, None, :].expand(b, c, flat.shape[2])).reshape_as(planes)


def lr_factor(step: int, total: int, cfg: H6ValueConfig) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, data, device, *, seed: int = 0, max_states: int = 0, batch_size: int = 4096) -> dict:
    """Weighted MSE (selection), and per label kind: MSE, sign accuracy, mean value; by side and ply."""
    was = model.value_head.training
    model.value_head.eval()
    n = data['value'].shape[0]
    order = torch.arange(n)
    if max_states and max_states < n:
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:max_states].sort().values
    values = []
    for start in range(0, order.shape[0], batch_size):
        index = order[start:start + batch_size]
        planes, _, _ = planes_from_value_cache(data, index, device)
        values.append(value_forward(model, planes).float().cpu())
    model.value_head.train(was)
    v = torch.cat(values) if values else torch.zeros(0)
    t, w = data['value'][order], data['weight'][order]
    kind, ply = data['kind'][order], data['ply'][order].long()

    def stats(mask):
        m = int(mask.sum())
        if not m:
            return {'states': 0}
        vv, tt = v[mask], t[mask]
        nonzero = tt != 0
        return {'states': m, 'mse': float((vv - tt).square().mean()),
                'sign_acc': float(((vv > 0) == (tt > 0))[nonzero].float().mean()) if nonzero.any() else None,
                'mean_value': float(vv.mean()), 'mean_target': float(tt.mean()),
                'saturated': float((vv.abs() > 0.95).float().mean())}

    every = torch.ones_like(t, dtype=torch.bool)
    return {'states': int(order.shape[0]),
            'weighted_mse': float((w * (v - t).square()).sum() / w.sum()) if order.shape[0] else None,
            'all': stats(every),
            'by_kind': {'proof': stats(kind == KIND_PROOF), 'outcome': stats(kind == KIND_OUTCOME)},
            'by_side': {'black': stats(ply % 2 == 0), 'white': stats(ply % 2 == 1)},
            'by_ply': {f'{lo}-{hi}': stats((ply >= lo) & (ply <= hi)) for lo, hi in PLY_BUCKETS}}


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_text(text: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def check_h3_metadata(meta: dict, cfg: H6ValueConfig) -> None:
    expected = {'stage': 'H3', 'track': 'B', 'policy_trained': True, 'value_trained': False}
    wrong = {k: meta.get(k) for k, v in expected.items() if meta.get(k) != v}
    if wrong:
        raise ValueError(f'parent is not an H3 policy-only checkpoint: {wrong}')
    arch = {k: meta.get('config', {}).get(k) for k in ('channels', 'blocks')}
    if arch != {'channels': cfg.channels, 'blocks': cfg.blocks}:
        raise ValueError(f'H3 architecture {arch} != config channels {cfg.channels} blocks {cfg.blocks}')


class Trainer:
    def __init__(self, cfg: H6ValueConfig, *, log=print):
        self.cfg, self.log = cfg, log
        git_commit, git_dirty = _git_provenance()
        if cfg.require_clean_git and git_dirty is not False:
            raise RuntimeError(f'refusing to train from a dirty or unknown worktree (git_dirty={git_dirty}); '
                               'commit first (require_clean_git: false only for smoke runs)')
        self.git = {'git_commit': git_commit, 'git_dirty': git_dirty}
        if cfg.torch_threads:
            torch.set_num_threads(cfg.torch_threads)
        if cfg.device == 'cpu':
            torch.set_flush_denormal(True)
        self.device = torch.device(cfg.device)
        meta_path = cfg.metadata_path
        self.h3_meta = json.loads(meta_path.read_text(encoding='utf-8'))
        check_h3_metadata(self.h3_meta, cfg)
        self.parent = {'h3_checkpoint': str(cfg.h3_checkpoint),
                       'parent_h3_checkpoint_sha256': file_sha256(cfg.h3_checkpoint),
                       'parent_h3_metadata_sha256': text_sha256(meta_path)}
        model_config = ModelConfig(channels=cfg.channels, blocks=cfg.blocks)
        self.model = load_checkpoint(cfg.h3_checkpoint, model_config, device=self.device)
        self.frozen_sha256 = state_sha256(frozen_state(self.model))
        cache = Path(cfg.cache_dir)
        self.train_data, self.manifest = load_value_cache(cache, 'train', verify_hash=True)
        self.val_data, _ = load_value_cache(cache, 'val', verify_hash=True)
        report = Path(cfg.h6_report) if cfg.h6_report else None
        self.report_sha256 = text_sha256(report) if report else None
        if self.manifest.get('h6_report_sha256') != self.report_sha256:
            raise ValueError(f"the value cache was built with H6-1 report {self.manifest.get('h6_report_sha256')}, "
                             f'config gives {self.report_sha256}')
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        self.params = freeze(self.model)
        self.optimizer = torch.optim.AdamW(self.params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        n = self.train_data['value'].shape[0]
        self.steps_per_epoch = n // cfg.batch_size
        if self.steps_per_epoch < 1:
            raise ValueError('batch_size is larger than the training split')
        total = int(cfg.epochs * self.steps_per_epoch)
        self.total_steps = min(total, cfg.max_steps) if cfg.max_steps else total
        schedule_steps = self.total_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lambda s: lr_factor(s, schedule_steps, cfg))
        _, self.src = d4_tables(self.device)
        self.step = 0
        self.augment_rng = torch.Generator().manual_seed(cfg.seed + 1)
        self.best = None

    def _identity(self) -> dict:
        return {**self.parent, 'labels_sha256': self.manifest['labels_sha256'],
                'h6_report_sha256': self.report_sha256,
                'cache_sha256': {k: v['sha256'] for k, v in self.manifest['splits'].items() if k != 'test'}}

    def _order(self, epoch: int) -> torch.Tensor:
        n = self.train_data['value'].shape[0]
        return torch.randperm(n, generator=torch.Generator().manual_seed(self.cfg.seed * 1000 + epoch))

    def snapshot(self) -> dict:
        return {'format': SNAPSHOT_FORMAT, 'config': asdict(self.cfg), 'step': self.step,
                'value_head': self.model.value_head.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(), 'augment_rng': self.augment_rng.get_state(),
                'torch_rng': torch.get_rng_state(), 'best': self.best, 'identity': self._identity(),
                'frozen_sha256': self.frozen_sha256}

    def resume(self, path: Path) -> None:
        snap = torch.load(path, map_location=self.device, weights_only=False)
        if snap.get('format') != SNAPSHOT_FORMAT:
            raise ValueError('not an H6-2a training snapshot')
        identity = self._identity()
        changed = [k for k in identity if snap['identity'].get(k) != identity[k]]
        if changed:
            raise ValueError(f'snapshot was trained from other inputs: {changed}')
        if snap['frozen_sha256'] != self.frozen_sha256:
            raise ValueError('the frozen trunk / policy head differ from the snapshot')
        changed = [k for k in CRITICAL if snap['config'][k] != getattr(self.cfg, k)]
        if changed:
            raise ValueError(f'training-critical config changed since the snapshot: {changed}')
        self.model.value_head.load_state_dict(snap['value_head'])
        self.optimizer.load_state_dict(snap['optimizer'])
        self.scheduler.load_state_dict(snap['scheduler'])
        self.augment_rng.set_state(snap['augment_rng'])
        torch.set_rng_state(snap['torch_rng'])
        self.step, self.best = snap['step'], snap['best']

    def metadata(self, step: int, metrics: dict | None) -> dict:
        return {'stage': STAGE, 'track': 'B', 'policy_trained': True, 'value_trained': True,
                'trunk_frozen': True, 'policy_head_frozen': True,
                'source': 'RenjuNet (offline, non-commercial; do not publish online)',
                **self._identity(), 'parent_h3_metadata': self.h3_meta,
                'frozen_state_sha256': self.frozen_sha256,
                'model_config': asdict(self.model.config), 'encoder_version': ENCODER_VERSION,
                'action_index_version': ACTION_INDEX_VERSION, **self.git,
                'config': asdict(self.cfg), 'step': step,
                'best_step': self.best and self.best['step'], 'val': metrics,
                'cache_summary': {k: {kk: vv for kk, vv in v.items() if kk != 'sha256'}
                                  for k, v in self.manifest['splits'].items() if k != 'test'}}

    def _check_frozen(self) -> None:
        if state_sha256(frozen_state(self.model)) != self.frozen_sha256:
            raise RuntimeError('the trunk or policy head changed during H6-2a training')

    def _validate(self) -> dict:
        metrics = evaluate(self.model, self.val_data, self.device, seed=self.cfg.seed,
                           max_states=self.cfg.eval_max_states)
        if metrics['weighted_mse'] is None or not math.isfinite(metrics['weighted_mse']):
            raise FloatingPointError(f'non-finite validation loss at step {self.step}')
        if self.best is None or metrics['weighted_mse'] < self.best['weighted_mse']:
            self._check_frozen()
            self.best = {'weighted_mse': metrics['weighted_mse'], 'step': self.step, 'metrics': metrics}
            save_checkpoint(self.out / 'best.pt.tmp', self.model)
            os.replace(self.out / 'best.pt.tmp', self.out / 'best.pt')
            _atomic_text(json.dumps(self.metadata(self.step, metrics), indent=1), self.out / 'best.json')
        with (self.out / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'step': self.step, 'val': metrics}) + '\n')
        proof = metrics['by_kind']['proof']
        self.log(f"step {self.step} val wmse {metrics['weighted_mse']:.4f} proof sign {proof.get('sign_acc')} "
                 f"outcome mse {metrics['by_kind']['outcome'].get('mse')}")
        return metrics

    def train(self) -> dict:
        cfg = self.cfg
        started, seen, loss_sum, loss_n = perf_counter(), 0, 0.0, 0
        window_started, window_seen = started, 0
        first_step = self.step  # ETA from this run's own pace (a resumed run starts mid-way)
        while self.step < self.total_steps:
            epoch, offset = divmod(self.step, self.steps_per_epoch)
            index = self._order(epoch)[offset * cfg.batch_size:(offset + 1) * cfg.batch_size]
            planes, target, weight = planes_from_value_cache(self.train_data, index, self.device)
            symmetry = torch.randint(0, 8 if cfg.augment_d4 else 1, (index.shape[0],), generator=self.augment_rng)
            if cfg.augment_d4:
                planes = augment_planes(planes, symmetry.to(self.device), self.src)
            loss = weighted_mse(value_forward(self.model, planes).float(), target, weight)
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
                done = self.step - first_step
                eta = (now - started) / done * (self.total_steps - self.step) if done else float('nan')
                self.log(f"step {self.step}/{self.total_steps} ({100 * self.step / self.total_steps:.1f}%) "
                         f"loss {record['loss']:.4f} lr {record['lr']:.2e} {rate:.0f} samples/s "
                         f"elapsed {(now - started) / 60:.1f} min ETA {eta / 60:.0f} min")
                loss_sum, loss_n = 0.0, 0
            if self.step % cfg.eval_every == 0 or self.step == self.total_steps:
                self._validate()
            if self.step % cfg.checkpoint_every == 0 or self.step == self.total_steps:
                _atomic_save(self.snapshot(), self.out / 'snapshot.pt')
        self._check_frozen()
        save_checkpoint(self.out / 'last.pt', self.model)
        _atomic_text(json.dumps(self.metadata(self.step, None), indent=1), self.out / 'last.json')
        return {'step': self.step, 'best': self.best,
                'samples_per_second': seen / max(1e-9, perf_counter() - started)}
