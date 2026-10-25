"""H7-c training: the policy fine-tune and the value-head continuation of one cycle (docs/h7-c1.md, fixed before c1).

Policy (``train_policy``), from the parent policy (c1: the frozen H3; the KL teacher and the guard baseline are always
the frozen H3):
    data       ``scripts/h7c_labels.py`` policy rows (unmasked; ``visits`` targets with KL, ``proof_one_hot`` without)
    loss       per row: CE(target, student) + lambda * [kl] * KL(teacher || student), mean over the batch; teacher =
               the frozen H3, student = the candidate, both masked softmax over the legal moves at T = 1, the
               teacher evaluated on the same (D4-augmented) planes
    network    every weight trains; BatchNorm running statistics stay the parent's (eval mode): a few hundred
               steps on one cycle must not re-estimate the statistics the RenjuNet guard is measured with
    schedule   AdamW, linear warmup over ``warmup_steps(total)`` (5% of the steps, at least 5, at most 100), cosine
               down to ``min_lr_ratio``
    selection  at ``evaluations`` evenly spaced steps: the c1 val objective (same loss, no augmentation) and the
               H3 RenjuNet val top-1 (``h3_train.evaluate``, the whole split); eligible = top-1 within
               ``guard_drop`` (0.02) of the parent's; best = the eligible evaluation with the lowest c1 val
               objective. No eligible evaluation: no candidate (``verdict: NO_CANDIDATE``).
Value (``train_value``), from the parent value checkpoint (c1: the frozen H6 v3), H6-2a's contract kept:
    network    trunk and policy head frozen (parameters and BatchNorm, checked by hash before and after); only
               the value head trains; weighted MSE of the tanh value (proof 1.0, outcome 0.1 from the rows)
    data       the cycle's value cache and the replay anchor: the frozen H6 v3 value cache in every cycle (its
               labels SHA is the frozen H6 v3's, carried by every H7-c value candidate as
               ``training_sources.replay_anchor_labels_sha256``);
               every epoch all cycle train rows plus as many H6 train rows drawn without replacement (seeded per
               epoch): 50 : 50 by source
    leakage    D4 canonical hashes over the union: a val row whose position is in either train split is dropped
               from its val set (train < val), and the check is repeated after
    selection  objective = mean of the two val weighted MSE (only the H6 one if the cycle val set is empty); eligible = H6 val weighted MSE at most
               ``h6_guard_ratio`` (1.05) times the parent's; best = the eligible evaluation with the lowest objective
Source (``policy_rows_source`` / ``value_rows_source``): the label rows of one cycle carry the four model SHA-256 of
the self-play that made them (``h7c_provenance``); both trainers require them to be one model pair whose policy
(value) is the trainer's ``--parent``, and every game id to be of this cycle.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import gzip
import hashlib
import json
import math
from pathlib import Path

import torch

from model.config import BOARD_SIZE
from renju import Game

from .h3_cache import FIELD_BYTES, pack, unpack

ACTIONS = BOARD_SIZE * BOARD_SIZE
POLICY_STAGE = 'H7-c-policy'
VALUE_STAGE = 'H7-c-value'
VALUE_LOADER_STAGE = 'H6-2a'  # hybrid.h6_value_leaf's (frozen) loader contract: a value head on the frozen H3 trunk


SOURCE_MODEL_KEYS = ('policy_checkpoint_sha256', 'policy_metadata_sha256', 'value_checkpoint_sha256',
                     'value_metadata_sha256')


def rows_source(provenances, game_ids, cycle: int) -> dict:
    """The one self-play source of a cycle's label rows: its four model SHA-256, runtime and generator commit."""
    prefix, sources, rows = f'h7c:c{cycle}:', set(), 0
    for detail, game_id in zip(provenances, game_ids):
        rows += 1
        if not str(game_id).startswith(prefix):
            raise ValueError(f'a row of game {game_id} is not of cycle {cycle}')
        if not isinstance(detail, dict) or any(not detail.get(k) for k in SOURCE_MODEL_KEYS):
            raise ValueError(f'a row of game {game_id} has no H7-c self-play provenance')
        sources.add(json.dumps({k: detail[k] for k in (*SOURCE_MODEL_KEYS, 'runtime_hash', 'selfplay_git_commit')},
                               sort_keys=True))
    if rows == 0 or len(sources) != 1:
        raise ValueError(f'the rows come from {len(sources)} self-play sources (one is required)')
    source = json.loads(sources.pop())
    return {'models': {k: source[k] for k in SOURCE_MODEL_KEYS}, 'runtime_hash': source['runtime_hash'],
            'selfplay_git_commit': source['selfplay_git_commit'], 'rows': rows}


def policy_rows_source(path: Path, cycle: int) -> dict:
    """``rows_source`` over every policy row (masked ones too), plus the one search retarget of the file (v2 rows,
    ``scripts/h7c_retarget.py``) or None (v1 rows)."""
    provenances, ids, retargets = [], [], set()
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                provenances.append(row.get('h7c_provenance'))
                ids.append(row.get('game_id'))
                retargets.add(json.dumps(row.get('retarget'), sort_keys=True))
    source = rows_source(provenances, ids, cycle)
    if len(retargets) != 1:
        raise ValueError(f'the policy rows name {len(retargets)} search retargets (one, or none for v1 rows)')
    return {**source, 'retarget': json.loads(retargets.pop())}


def value_rows_source(path: Path, cycle: int, labels_sha256: str) -> dict:
    """``rows_source`` over the value rows a cache was built from (their SHA-256 as ``h6_value_cache`` computes it
    must be the cache's ``labels_sha256``)."""
    from .h6_value_cache import iter_games

    digest, provenances, ids = hashlib.sha256(), [], []
    for game_id, rows in iter_games(Path(path), digest):
        for row in rows:
            provenances.append(row.get('h7c_provenance'))
            ids.append(game_id)
    if digest.hexdigest() != labels_sha256:
        raise ValueError(f'{path} is not the label file of the cycle cache ({digest.hexdigest()[:12]} != '
                         f'{labels_sha256[:12]})')
    return rows_source(provenances, ids, cycle)


def warmup_steps(total: int) -> int:
    return min(100, max(5, round(total * 0.05)))


def lr_factor(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def evaluation_steps(total: int, count: int) -> list[int]:
    """``count`` evenly spaced steps in (0, total], the last one ``total``."""
    return sorted({max(1, round(total * (i + 1) / count)) for i in range(count)})


def _config(cls, data: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} key(s): {', '.join(sorted(unknown))}")
    return cls(**data)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


# -- positions --------------------------------------------------------------------------------------------------

def planes_from_states(data: dict[str, torch.Tensor], index: torch.Tensor, device='cpu') -> tuple:
    """[B,6,15,15] planes (``model.encoding.encode_game``'s) and the legal mask [B,225] of packed states."""
    black = unpack(data['black'][index]).to(device)
    white = unpack(data['white'][index]).to(device)
    legal = unpack(data['legal'][index]).to(device)
    is_black = (data['ply'][index].remainder(2) == 0).to(device)
    last = data['last'][index].long().to(device)
    b = index.shape[0]
    planes = torch.zeros((b, 6, ACTIONS), dtype=torch.float32, device=device)
    planes[:, 0] = torch.where(is_black[:, None], black, white)
    planes[:, 1] = torch.where(is_black[:, None], white, black)
    has_last = last >= 0
    planes[torch.arange(b, device=device)[has_last], 2, last[has_last]] = 1.0
    planes[:, 3] = is_black[:, None].float()
    planes[:, 4] = 1.0
    planes[:, 5] = legal
    return planes.reshape(b, 6, BOARD_SIZE, BOARD_SIZE), legal


def state_hashes(data: dict[str, torch.Tensor]) -> list[str]:
    """``h6_labels.canonical_hash`` of every packed state (stones and side to move; move order is irrelevant)."""
    from .h6_labels import canonical_hash

    out = []
    if data['ply'].shape[0] == 0:
        return out
    black, white = unpack(data['black']), unpack(data['white'])
    for i in range(black.shape[0]):
        b = [divmod(int(a), BOARD_SIZE) for a in torch.nonzero(black[i]).flatten()]
        w = [divmod(int(a), BOARD_SIZE) for a in torch.nonzero(white[i]).flatten()]
        ply = int(data['ply'][i])
        if len(b) != (ply + 1) // 2 or len(w) != ply // 2:
            raise ValueError(f'state {i}: {len(b)} black / {len(w)} white stones at ply {ply}')
        moves = [None] * ply
        moves[0::2], moves[1::2] = b, w
        out.append(canonical_hash(moves))
    return out


# -- policy -----------------------------------------------------------------------------------------------------

@dataclass
class PolicyConfig:
    seed: int = 20261024
    batch_size: int = 512
    epochs: float = 2.0
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    min_lr_ratio: float = 0.05
    kl_lambda: float = 0.5
    augment_d4: bool = True
    evaluations: int = 10
    guard_drop: float = 0.02
    device: str = 'cpu'

    @classmethod
    def from_dict(cls, data: dict) -> 'PolicyConfig':
        return _config(cls, data)


POLICY_ROW_FORMATS = ('h7c-policy-row-v1', 'h7c-policy-row-v2')  # v2: visits targets from a search retarget


def load_policy_rows(path: Path) -> dict[str, list[dict]]:
    """Unmasked policy rows of ``scripts/h7c_labels.py`` by split (one row format per file)."""
    out: dict[str, list[dict]] = {'train': [], 'val': []}
    formats = set()
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            formats.add(row.get('format'))
            if row.get('format') not in POLICY_ROW_FORMATS or len(formats) > 1:
                raise ValueError(f"policy row format {row.get('format')!r} (one of {POLICY_ROW_FORMATS} per file)")
            if not row['mask']:
                out.setdefault(row['split'], []).append(row)
    return out


def policy_tensors(rows: list[dict]) -> dict[str, torch.Tensor]:
    """Packed states, the dense target over the 225 actions and the KL flag; every target move must be legal."""
    n = len(rows)
    black, white, legal = bytearray(), bytearray(), bytearray()
    last, ply = [], []
    target = torch.zeros((n, ACTIONS), dtype=torch.float32)
    for i, row in enumerate(rows):
        game = Game()
        for move in row['moves']:
            game.play(*move)
        moves = [tuple(m) for m in row['moves']]
        black += pack(r * BOARD_SIZE + c for r, c in moves[0::2])
        white += pack(r * BOARD_SIZE + c for r, c in moves[1::2])
        legal_moves = set(game.legal_moves())
        legal += pack(r * BOARD_SIZE + c for r, c in legal_moves)
        last.append(moves[-1][0] * BOARD_SIZE + moves[-1][1] if moves else -1)
        ply.append(len(moves))
        for r, c, p in row['target']:
            if (r, c) not in legal_moves:
                raise ValueError(f"{row['game_id']} ply {row['ply']}: target move {(r, c)} is not legal")
            target[i, r * BOARD_SIZE + c] = p
    frame = (lambda b: torch.frombuffer(b, dtype=torch.uint8).reshape(n, FIELD_BYTES).clone()
             if n else torch.zeros((0, FIELD_BYTES), dtype=torch.uint8))
    return {'black': frame(black), 'white': frame(white), 'legal': frame(legal),
            'last': torch.tensor(last, dtype=torch.int16), 'ply': torch.tensor(ply, dtype=torch.int16),
            'target': target, 'kl': torch.tensor([bool(r['kl']) for r in rows], dtype=torch.bool)}


def _policy_terms(student, teacher, planes, legal, target, kl, lam):
    """Per-row CE and KL(teacher || student) over the legal moves; the loss per row."""
    from .h3_train import policy_logits

    log_s = policy_logits(student, planes).float().masked_fill(~legal, -torch.inf).log_softmax(1)
    ce = -torch.where(legal, target * log_s, torch.zeros_like(log_s)).sum(1)
    with torch.no_grad():
        log_t = policy_logits(teacher, planes).float().masked_fill(~legal, -torch.inf).log_softmax(1)
    p_t = log_t.exp()
    kld = torch.where(legal, p_t * (log_t - log_s), torch.zeros_like(log_s)).sum(1)
    return ce, kld, ce + lam * kl.float() * kld


@torch.no_grad()
def policy_objective(student, teacher, data, lam: float, device, batch_size: int = 2048) -> dict:
    from .h3_train import policy_logits

    n = data['target'].shape[0]
    sums = {'loss': 0.0, 'ce': 0.0, 'kl': 0.0, 'kl_rows': 0, 'top1': 0}
    for start in range(0, n, batch_size):
        index = torch.arange(start, min(n, start + batch_size))
        planes, legal = planes_from_states(data, index, device)
        target, kl = data['target'][index].to(device), data['kl'][index].to(device)
        ce, kld, loss = _policy_terms(student, teacher, planes, legal, target, kl, lam)
        best = policy_logits(student, planes).float().masked_fill(~legal, -torch.inf).argmax(1)
        sums['loss'] += float(loss.sum())
        sums['ce'] += float(ce.sum())
        sums['kl'] += float((kld * kl.float()).sum())
        sums['kl_rows'] += int(kl.sum())
        sums['top1'] += int((target.gather(1, best[:, None]).squeeze(1) == target.max(1).values).sum())
    return {'rows': n, 'objective': sums['loss'] / n if n else None, 'ce': sums['ce'] / n if n else None,
            'kl': sums['kl'] / sums['kl_rows'] if sums['kl_rows'] else None, 'top1_vs_target_mode':
            sums['top1'] / n if n else None}


def train_policy(cfg: PolicyConfig, parent, teacher, train: dict, val: dict, guard_data: dict, guard_eval,
                 log=print) -> dict:
    """Fine-tune ``parent`` (a PolicyValueNet; c1: H3) against the frozen ``teacher`` (always H3); the guard
    baseline is the teacher's RenjuNet top-1. Returns the run record and the best state (or none)."""
    from .h3_train import augment, d4_tables

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
    warm = warmup_steps(total)
    evals = set(evaluation_steps(total, cfg.evaluations))
    dest, src = d4_tables(device)
    base_guard = guard_eval(teacher, guard_data)
    record = {'total_steps': total, 'steps_per_epoch': per_epoch, 'warmup_steps': warm, 'train_rows': n,
              'val_rows': int(val['target'].shape[0]), 'teacher_guard': base_guard,
              'parent_guard_top1': guard_eval(student, guard_data)['top1'],
              'parent_val': policy_objective(student, teacher, val, cfg.kl_lambda, device), 'evaluations': []}
    best, best_state = None, None
    generator = torch.Generator().manual_seed(cfg.seed)
    step = 0
    while step < total:
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, cfg.batch_size):
            if step >= total:
                break
            index = order[start:start + cfg.batch_size]
            planes, legal = planes_from_states(train, index, device)
            target, kl = train['target'][index].to(device), train['kl'][index].to(device)
            if cfg.augment_d4:
                sym = torch.randint(0, 8, (index.shape[0],), generator=generator).to(device)
                planes, legal, _ = augment(planes, legal, torch.zeros(index.shape[0], dtype=torch.long,
                                                                         device=device), sym, dest, src)
                target = target.gather(1, src[sym])
            for group in optimizer.param_groups:
                group['lr'] = cfg.learning_rate * lr_factor(step, total, warm, cfg.min_lr_ratio)
            _, _, loss = _policy_terms(student, teacher, planes, legal, target, kl, cfg.kl_lambda)
            optimizer.zero_grad(set_to_none=True)
            loss.mean().backward()
            optimizer.step()
            step += 1
            if step in evals:
                guard = guard_eval(student, guard_data)
                metrics = policy_objective(student, teacher, val, cfg.kl_lambda, device)
                eligible = guard['top1'] >= base_guard['top1'] - cfg.guard_drop
                entry = {'step': step, 'val': metrics, 'guard_top1': guard['top1'],
                         'guard_drop': base_guard['top1'] - guard['top1'], 'eligible': eligible}
                record['evaluations'].append(entry)
                log(f"[h7c-policy] step {step}/{total} val {metrics['objective']:.4f} guard top-1 "
                    f"{guard['top1']:.4f} (H3 {base_guard['top1']:.4f}){'' if eligible else ' NOT eligible'}",
                    flush=True)
                if eligible and (best is None or metrics['objective'] < best['val']['objective']):
                    best, best_state = entry, copy.deepcopy(student.state_dict())
    record['best'] = best
    record['verdict'] = 'CANDIDATE' if best is not None else 'NO_CANDIDATE'
    return {'record': record, 'state': best_state}


# -- value ------------------------------------------------------------------------------------------------------

@dataclass
class ValueConfig:
    seed: int = 20261025
    batch_size: int = 1024
    epochs: float = 3.0
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    min_lr_ratio: float = 0.05
    augment_d4: bool = True
    evaluations: int = 10
    h6_guard_ratio: float = 1.05
    device: str = 'cpu'

    @classmethod
    def from_dict(cls, data: dict) -> 'ValueConfig':
        return _config(cls, data)


def select_rows(data: dict[str, torch.Tensor], keep: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v[keep] for k, v in data.items()}


def union_masks(cycle: dict[str, dict], replay: dict[str, dict]) -> tuple[dict, dict, dict]:
    """Drop val rows (of either source) whose position is in either train split; check nothing is left."""
    hashes = {(src, split): state_hashes(d) for src, splits in (('cycle', cycle), ('replay', replay))
              for split, d in splits.items()}
    train = set(hashes[('cycle', 'train')]) | set(hashes[('replay', 'train')])
    out, counts = {'cycle': {}, 'replay': {}}, {}
    for src, splits in (('cycle', cycle), ('replay', replay)):
        out[src]['train'] = splits['train']
        keep = [h not in train for h in hashes[(src, 'val')]]
        out[src]['val'] = select_rows(splits['val'], torch.tensor(keep, dtype=torch.bool))
        counts[f'{src}_val_dropped'] = keep.count(False)
        kept = {h for h, k in zip(hashes[(src, 'val')], keep) if k}
        if kept & train:  # the check, repeated on what is left
            raise ValueError(f'a {src} val position is still in a train split')
    return out['cycle'], out['replay'], counts


def mixed_epoch(cycle_train: dict, replay_train: dict, epoch: int, seed: int) -> dict[str, torch.Tensor]:
    """All cycle train rows and as many replay train rows (without replacement), shuffled: 50 : 50 by source."""
    n = cycle_train['value'].shape[0]
    m = replay_train['value'].shape[0]
    if m < n:
        raise ValueError(f'the replay train split ({m}) is smaller than the cycle train split ({n})')
    generator = torch.Generator().manual_seed(seed * 1000 + epoch)
    pick = torch.randperm(m, generator=generator)[:n].sort().values
    keys = ('black', 'white', 'legal', 'last', 'ply', 'value', 'weight', 'kind')
    merged = {k: torch.cat([cycle_train[k], replay_train[k][pick]]) for k in keys}
    merged['source'] = torch.cat([torch.zeros(n, dtype=torch.int8), torch.ones(n, dtype=torch.int8)])
    order = torch.randperm(2 * n, generator=generator)
    return {k: v[order] for k, v in merged.items()}


def train_value(cfg: ValueConfig, parent, cycle: dict[str, dict], replay: dict[str, dict], log=print) -> dict:
    """Continue the value head of ``parent`` (trunk and policy head frozen); the run record and the best state."""
    from .h3_train import d4_tables
    from .h6_value_train import (augment_planes, evaluate, freeze, frozen_state, state_sha256, value_forward,
                                 weighted_mse)

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    model = copy.deepcopy(parent).to(device)
    params = freeze(model)
    frozen = state_sha256(frozen_state(model))
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    cycle, replay, dropped = union_masks(cycle, replay)
    n = 2 * cycle['train']['value'].shape[0]
    if n == 0:
        raise ValueError('no cycle value train rows')
    per_epoch = math.ceil(n / cfg.batch_size)
    total = max(1, round(cfg.epochs * per_epoch))
    warm = warmup_steps(total)
    evals = set(evaluation_steps(total, cfg.evaluations))
    _, src = d4_tables(device)

    if replay['val']['value'].shape[0] == 0:
        raise ValueError('the replay val split is empty after the union masking: no guard')

    def val_metrics():
        a, b = evaluate(model, cycle['val'], device), evaluate(model, replay['val'], device)
        parts = [x['weighted_mse'] for x in (a, b) if x['weighted_mse'] is not None]  # an empty cycle val: H6 only
        return {'cycle_val': a, 'replay_val': b, 'objective': sum(parts) / len(parts)}

    base = val_metrics()
    record = {'total_steps': total, 'steps_per_epoch': per_epoch, 'warmup_steps': warm, 'train_rows_per_epoch': n,
              # the parent's diagnostics in the evaluations' format (measurement only: selection and guard unchanged)
              'union_masking': dropped, 'parent_val': {'objective': base['objective'],
                                                       'cycle_wmse': base['cycle_val']['weighted_mse'],
                                                       'replay_wmse': base['replay_val']['weighted_mse'],
                                                       'cycle_val': base['cycle_val']['all'],
                                                       'replay_val': base['replay_val']['all']},
              'frozen_state_sha256': frozen, 'evaluations': []}
    best, best_state, step, epoch = None, None, 0, 0
    generator = torch.Generator().manual_seed(cfg.seed)
    while step < total:
        data = mixed_epoch(cycle['train'], replay['train'], epoch, cfg.seed)
        epoch += 1
        for start in range(0, n, cfg.batch_size):
            if step >= total:
                break
            index = torch.arange(start, min(n, start + cfg.batch_size))
            planes, _ = planes_from_states(data, index, device)
            if cfg.augment_d4:
                sym = torch.randint(0, 8, (index.shape[0],), generator=generator).to(device)
                planes = augment_planes(planes, sym, src)
            for group in optimizer.param_groups:
                group['lr'] = cfg.learning_rate * lr_factor(step, total, warm, cfg.min_lr_ratio)
            loss = weighted_mse(value_forward(model, planes), data['value'][index].to(device),
                                data['weight'][index].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step in evals:
                m = val_metrics()
                eligible = m['replay_val']['weighted_mse'] <= cfg.h6_guard_ratio * base['replay_val']['weighted_mse']
                entry = {'step': step, 'objective': m['objective'], 'cycle_wmse': m['cycle_val']['weighted_mse'],
                         'replay_wmse': m['replay_val']['weighted_mse'], 'eligible': eligible,
                         'cycle_val': m['cycle_val']['all'], 'replay_val': m['replay_val']['all']}
                record['evaluations'].append(entry)
                log(f"[h7c-value] step {step}/{total} objective {m['objective']:.5f} (cycle "
                    f"{entry['cycle_wmse']}, H6 {entry['replay_wmse']:.5f}){'' if eligible else ' NOT eligible'}",
                    flush=True)
                if eligible and (best is None or m['objective'] < best['objective']):
                    best, best_state = entry, copy.deepcopy(model.state_dict())
    if state_sha256(frozen_state(model)) != frozen:
        raise RuntimeError('the trunk or policy head changed during value training')
    record['best'] = best
    record['verdict'] = 'CANDIDATE' if best is not None else 'NO_CANDIDATE'
    return {'record': record, 'state': best_state}


def config_dict(cfg) -> dict:
    return asdict(cfg)


def file_sha256(path: Path) -> str:
    return _sha(path)
