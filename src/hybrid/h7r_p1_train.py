"""R2-P1-POLICY-REPLAY-1 trainer: the H7-c policy fine-tune (``h7c_train.train_policy``) with an optional
H3-preserving replay term (docs/h7r/r2/R2-P1-POLICY-REPLAY-1-draft.md, section 2).

``h7c_train`` is not changed: its row loading, planes, per-row terms, val objective, schedule and evaluation steps are
used as they are. Only the step loop is restated here, so the replay term can be added:

    cycle      exactly ``train_policy``'s: the same rows, batches, order, generator (seeded ``cfg.seed``), D4
               symmetries, learning-rate schedule, per-row loss (CE + lambda [kl] KL(H3 || student)) and batch mean
    replay     (arm T only) every step, after the cycle batch: ``batch`` RenjuNet train positions drawn uniformly
               with replacement and their D4 symmetries, both from a separate generator seeded
               ``cfg.seed + seed_offset`` (the cycle generator is never touched); the same transformed planes and
               legal mask go to H3 and the student; logits masked to -inf off the legal moves, log-softmax; per
               position sum over the legal moves of p_H3 (log p_H3 - log p_student); mean over the batch; H3 in eval
               mode without gradient. loss = cycle batch mean + ``mu`` * replay mean
    records    the cycle KL is the existing val metric; the replay KL is kept apart: its mean over the steps between
               two evaluations and, at every evaluation, on the whole RenjuNet val split (``diag``; both arms, under
               ``no_grad`` in eval mode, outside the training path and its generators). Neither is a selection
               metric
    selection  ``train_policy``'s rule: the RenjuNet val top-1 guard within ``guard_drop`` of H3, then the lowest
               cycle val objective; the replay term is not in it

With ``replay=None`` and ``diag=None`` the run is ``train_policy``'s, record and state (tests/test_h7r_p1.py).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math

import torch

from . import h7c_train as tr

REPLAY_SEED_OFFSET = 1000003


@dataclass(frozen=True)
class Replay:
    mu: float = 0.5
    batch: int = 512
    seed_offset: int = REPLAY_SEED_OFFSET


def legal_log_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return logits.float().masked_fill(~legal, -torch.inf).log_softmax(1)


def replay_kl(student, teacher, planes: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """Per position KL(p_H3 || p_student) over the legal moves (both renormalised there); [B]."""
    from .h3_train import policy_logits

    log_s = legal_log_softmax(policy_logits(student, planes), legal)
    with torch.no_grad():
        log_t = legal_log_softmax(policy_logits(teacher, planes), legal)
    p_t = log_t.exp()
    return torch.where(legal, p_t * (log_t - log_s), torch.zeros_like(log_s)).sum(1)


def replay_batch(data: dict, generator: torch.Generator, batch: int, device, dest, src, augment_d4: bool):
    """``batch`` positions of ``data`` (an H3 cache split) and their D4 symmetries, all drawn from ``generator``."""
    from .h3_cache import planes_from_cache
    from .h3_train import augment

    n = data['target'].shape[0]
    index = torch.randint(0, n, (batch,), generator=generator)
    planes, legal, target = planes_from_cache(data, index, device)
    if augment_d4:
        sym = torch.randint(0, 8, (batch,), generator=generator).to(device)
        planes, legal, _ = augment(planes, legal, target, sym, dest, src)
    return planes, legal


@torch.no_grad()
def replay_kl_metric(student, teacher, data: dict, device, batch_size: int = 2048) -> dict:
    """The replay KL on a whole split (no augmentation): its mean over the positions."""
    from .h3_cache import planes_from_cache

    student.eval()
    n = data['target'].shape[0]
    total = 0.0
    for start in range(0, n, batch_size):
        index = torch.arange(start, min(n, start + batch_size))
        planes, legal, _ = planes_from_cache(data, index, device)
        total += float(replay_kl(student, teacher, planes, legal).sum())
    return {'positions': n, 'kl': total / n if n else None}


def train_policy(cfg: tr.PolicyConfig, parent, teacher, train: dict, val: dict, guard_data: dict, guard_eval,
                 replay: Replay | None = None, replay_data: dict | None = None, diag: dict | None = None,
                 log=print) -> dict:
    """``h7c_train.train_policy`` plus the replay term and the separate KL records (module docstring)."""
    from .h3_train import augment, d4_tables

    if (replay is None) != (replay_data is None):
        raise ValueError('the replay term needs both its settings and its positions')
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    teacher = copy.deepcopy(teacher).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = copy.deepcopy(parent).to(device).eval()  # eval: BatchNorm statistics stay the parent's
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    n = train['target'].shape[0]
    if n == 0:
        raise ValueError('no policy train rows')
    per_epoch = math.ceil(n / cfg.batch_size)
    total = max(1, round(cfg.epochs * per_epoch))
    warm = tr.warmup_steps(total)
    evals = set(tr.evaluation_steps(total, cfg.evaluations))
    dest, src = d4_tables(device)
    base_guard = guard_eval(teacher, guard_data)
    record = {'total_steps': total, 'steps_per_epoch': per_epoch, 'warmup_steps': warm, 'train_rows': n,
              'val_rows': int(val['target'].shape[0]), 'teacher_guard': base_guard,
              'parent_guard_top1': guard_eval(student, guard_data)['top1'],
              'parent_val': tr.policy_objective(student, teacher, val, cfg.kl_lambda, device), 'evaluations': []}
    p1 = {'replay': None if replay is None else {**asdict(replay), 'generator_seed': cfg.seed + replay.seed_offset},
          'parent_diag': None if diag is None else replay_kl_metric(student, teacher, diag, device)}
    best, best_state = None, None
    generator = torch.Generator().manual_seed(cfg.seed)
    replay_gen = None if replay is None else torch.Generator().manual_seed(cfg.seed + replay.seed_offset)
    step, replay_sum, replay_steps = 0, 0.0, 0
    while step < total:
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, cfg.batch_size):
            if step >= total:
                break
            index = order[start:start + cfg.batch_size]
            planes, legal = tr.planes_from_states(train, index, device)
            target, kl = train['target'][index].to(device), train['kl'][index].to(device)
            if cfg.augment_d4:
                sym = torch.randint(0, 8, (index.shape[0],), generator=generator).to(device)
                planes, legal, _ = augment(planes, legal, torch.zeros(index.shape[0], dtype=torch.long,
                                                                         device=device), sym, dest, src)
                target = target.gather(1, src[sym])
            for group in optimizer.param_groups:
                group['lr'] = cfg.learning_rate * tr.lr_factor(step, total, warm, cfg.min_lr_ratio)
            _, _, loss = tr._policy_terms(student, teacher, planes, legal, target, kl, cfg.kl_lambda)
            objective = loss.mean()
            if replay is not None:
                r_planes, r_legal = replay_batch(replay_data, replay_gen, replay.batch, device, dest, src,
                                                 cfg.augment_d4)
                r_kl = replay_kl(student, teacher, r_planes, r_legal).mean()
                objective = objective + replay.mu * r_kl
                replay_sum += float(r_kl.detach())
                replay_steps += 1
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            step += 1
            if step in evals:
                guard = guard_eval(student, guard_data)
                metrics = tr.policy_objective(student, teacher, val, cfg.kl_lambda, device)
                eligible = guard['top1'] >= base_guard['top1'] - cfg.guard_drop
                entry = {'step': step, 'val': metrics, 'guard_top1': guard['top1'],
                         'guard_drop': base_guard['top1'] - guard['top1'], 'eligible': eligible}
                record['evaluations'].append(entry)
                p1.setdefault('evaluations', []).append({
                    'step': step,
                    'replay_kl_train_mean': replay_sum / replay_steps if replay_steps else None,
                    'renjunet_val_kl': None if diag is None else replay_kl_metric(student, teacher, diag, device)})
                replay_sum, replay_steps = 0.0, 0
                log(f"[h7r-p1-policy] step {step}/{total} val {metrics['objective']:.4f} guard top-1 "
                    f"{guard['top1']:.4f} (H3 {base_guard['top1']:.4f}){'' if eligible else ' NOT eligible'}",
                    flush=True)
                if eligible and (best is None or metrics['objective'] < best['val']['objective']):
                    best, best_state = entry, copy.deepcopy(student.state_dict())
    record['best'] = best
    record['verdict'] = 'CANDIDATE' if best is not None else 'NO_CANDIDATE'
    return {'record': record, 'state': best_state, 'p1': p1}
