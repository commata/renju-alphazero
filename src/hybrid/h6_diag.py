"""H6-3D diagnostics for the H6-2a value head after the H6-3A FAIL (docs/h6-value.md). Dev data only.

D1  D4 decomposition on all of test part A (a dev set since H6-3A):
    - end-to-end: per-state std of the 8 symmetric values, broken down by kind / side / ply / |mean v|,
      and per symmetry the mean |v_s - v_identity|;
    - trunk equivariance: F = trunk features; for each symmetry s, T_s^-1 F(T_s x) against F(x)
      (relative L2 and cosine similarity per state);
    - value-head-only anisotropy: the value head on T_s F(x) for the 8 s (features exactly equivariant by
      construction): per-state std;
    - auxiliary: policy top-1 agreement across the 8 symmetries (the policy head is not D4-invariant either).
D2  engine-domain deep losses: every move whose 10M truth in the E1 / E1-holdout / E1-S manifests is
    PROVEN_LOSS; the child position (opponent to move) should have value > 0. Counted per move, per unique
    child (D4 canonical), and clustered per root; split by lost depth, side and ply. This tests recognising a
    deep win for the side to move, not the parent-level judgement P94 needs (all moves lose -> v < 0).
    The manifests hold no PROVEN_WIN truths, so there is no opposite-sign counterpart.
D3  D4-averaged inference V_D4(x) = mean_s V(T_s x) against the raw value, on A (G1/G2/G5 metrics, ECE,
    bias) and on D2, with latency / throughput / peak memory of both.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from model.encoding import encode_game
from renju import Game

from .h3_train import d4_tables
from .h6_gate import _position, probe_values, split_metrics
from .h6_labels import canonical_hash
from .h6_value_cache import KIND_OUTCOME, KIND_PROOF, planes_from_value_cache
from .h6_value_train import PLY_BUCKETS, augment_planes

FORMAT = 'h6-diag-v1'
ABS_V_BUCKETS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01))


def inverse_spatial(x: torch.Tensor, symmetry: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
    """Undo ``augment_planes``: out[a] = in[dest[s, a]] for [B,C,15,15]."""
    b, c = x.shape[:2]
    index = dest[symmetry]
    flat = x.reshape(b, c, -1)
    return flat.gather(2, index[:, None, :].expand(b, c, flat.shape[2])).reshape_as(x)


@torch.no_grad()
def d4_forward(model, planes: torch.Tensor, dest: torch.Tensor, src: torch.Tensor) -> dict:
    """For B states: values [B,8], trunk equivariance [B,8] (rel L2, cosine), head-only values [B,8],
    policy top-1 mapped back to the original frame [B,8]."""
    b = planes.shape[0]
    device = planes.device
    sym = torch.arange(8, device=device).repeat(b)
    moved = augment_planes(planes.repeat_interleave(8, dim=0), sym, src)
    feats = model.trunk(moved)                                      # [8B,C,15,15]
    values = model.value_head(feats).squeeze(1).reshape(b, 8)
    logits = model.policy_head(feats).masked_fill(moved[:, 5].reshape(8 * b, -1) == 0, -torch.inf)
    top = logits.argmax(1)
    top_orig = src[sym, top].reshape(b, 8)                         # transformed action -> original action
    back = inverse_spatial(feats, sym, dest).reshape(b, 8, *feats.shape[1:])
    base = back[:, :1]
    diff = (back - base).flatten(2)
    rel = diff.norm(dim=2) / base.flatten(2).norm(dim=2).clamp_min(1e-8)
    cos = torch.nn.functional.cosine_similarity(back.flatten(2), base.flatten(2).expand_as(back.flatten(2)), dim=2)
    f0 = back[:, 0]                                                 # F(x) for the identity
    head = model.value_head(augment_planes(f0.repeat_interleave(8, dim=0), sym, src)).squeeze(1).reshape(b, 8)
    return {'values': values, 'rel_l2': rel, 'cosine': cos, 'head_values': head, 'top1': top_orig}


def _summ(x: torch.Tensor) -> dict | None:
    if x.numel() == 0:
        return None
    x = x.float()
    return {'n': int(x.numel()), 'mean': float(x.mean()), 'p50': float(x.median()),
            'p95': float(torch.quantile(x, 0.95)) if x.numel() > 1 else float(x[0]), 'max': float(x.max())}


@torch.no_grad()
def d1(model, data: dict, device, batch_size: int = 256) -> tuple[dict, torch.Tensor]:
    dest, src = d4_tables(device)
    keys = ('values', 'rel_l2', 'cosine', 'head_values', 'top1')
    parts = {k: [] for k in keys}
    n = data['value'].shape[0]
    for start in range(0, n, batch_size):
        index = torch.arange(start, min(n, start + batch_size))
        planes, _, _ = planes_from_value_cache(data, index, device)
        out = d4_forward(model, planes, dest, src)
        for k in keys:
            parts[k].append(out[k].float().cpu() if k != 'top1' else out[k].cpu())
    r = {k: torch.cat(v) for k, v in parts.items()}
    v8 = r['values']
    std = v8.std(dim=1, unbiased=False)
    mean = v8.mean(dim=1)
    kind, ply = data['kind'], data['ply'].long()
    groups = {'kind': {'outcome': kind == KIND_OUTCOME, 'proof': kind == KIND_PROOF},
              'side': {'black': ply % 2 == 0, 'white': ply % 2 == 1},
              'ply': {f'{lo}-{hi}': (ply >= lo) & (ply <= hi) for lo, hi in PLY_BUCKETS},
              'abs_mean_v': {f'{lo}-{min(hi, 1.0)}': (mean.abs() >= lo) & (mean.abs() < hi) for lo, hi in ABS_V_BUCKETS}}
    head_std = r['head_values'].std(dim=1, unbiased=False)
    agree = (r['top1'] == r['top1'][:, :1]).float()
    result = {
        'states': int(n),
        'end_to_end_std': _summ(std),
        'end_to_end_std_by': {g: {k: _summ(std[m]) for k, m in d.items()} for g, d in groups.items()},
        'per_symmetry_abs_diff_from_identity': [float((v8[:, s] - v8[:, 0]).abs().mean()) for s in range(8)],
        'sign_agreement_all8': float(((v8 > 0).all(1) | (v8 < 0).all(1)).float().mean()),
        'trunk_equivariance': {
            'rel_l2_per_symmetry': [float(r['rel_l2'][:, s].mean()) for s in range(8)],
            'rel_l2_nonidentity': _summ(r['rel_l2'][:, 1:].flatten()),
            'cosine_per_symmetry': [float(r['cosine'][:, s].mean()) for s in range(8)],
            'cosine_nonidentity': _summ(r['cosine'][:, 1:].flatten()),
        },
        'value_head_only_std': _summ(head_std),
        'value_head_only_std_by_kind': {k: _summ(head_std[m]) for k, m in groups['kind'].items()},
        'policy_top1_agreement_per_symmetry': [float(agree[:, s].mean()) for s in range(8)],
        'policy_top1_all8_agree': float((agree.sum(1) == 8).float().mean()),
    }
    return result, v8


def deep_loss_children(manifests: list[Path]) -> list[dict]:
    """Children of every PROVEN_LOSS truth (opponent to move; its value should be > 0)."""
    out = []
    for path in manifests:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        for p in data.get('positions', []):
            for key, truth in (p.get('truths') or {}).items():
                if (truth or {}).get('status') != 'PROVEN_LOSS':
                    continue
                move = [int(x) for x in key.split(',')]
                game = _position(p['history'])
                if tuple(move) not in set(map(tuple, game.legal_moves())):
                    continue
                game.play(*move)
                if game.done:
                    continue
                child = p['history'] + [move]
                out.append({'source': Path(path).parent.name, 'root': f"{Path(path).parent.name}:{p['key']}",
                            'move': move, 'lost_depth': truth.get('lost_depth'), 'ply': len(child),
                            'side_to_move': 'black' if len(child) % 2 == 0 else 'white',
                            'canonical': canonical_hash(child), 'planes': encode_game(game)})
    return out


def _d2_scores(rows: list[dict], v: torch.Tensor) -> dict:
    ok = (v > 0).float()
    by = lambda key: {str(k): {'n': int(sum(1 for r in rows if r[key] == k)),  # noqa: E731
                               'accuracy': float(ok[[i for i, r in enumerate(rows) if r[key] == k]].mean()),
                               'mean_value': float(v[[i for i, r in enumerate(rows) if r[key] == k]].mean())}
                      for k in sorted({r[key] for r in rows}, key=str)}
    first = {}
    for i, r in enumerate(rows):
        first.setdefault(r['canonical'], i)
    uniq = torch.tensor(sorted(first.values()))
    roots = {}
    for i, r in enumerate(rows):
        roots.setdefault(r['root'], []).append(i)
    clustered = sum(float(ok[ix].mean()) for ix in roots.values()) / len(roots)
    return {'moves': len(rows), 'unique_children': len(first), 'roots': len(roots),
            'move_accuracy': float(ok.mean()), 'unique_child_accuracy': float(ok[uniq].mean()),
            'root_clustered_accuracy': clustered,
            'mean_value': float(v.mean()), 'median_value': float(v.median()),
            'by_lost_depth': by('lost_depth'), 'by_source': by('source'), 'by_side': by('side_to_move'),
            'by_ply': {f'{lo}-{hi}': {'n': int(m.sum()), 'accuracy': float(ok[m].mean())}
                       for lo, hi in PLY_BUCKETS
                       for m in [torch.tensor([lo <= r['ply'] <= hi for r in rows])] if m.any()}}


@torch.no_grad()
def d2(value_fn, rows: list[dict], device) -> dict:
    if not rows:
        return {'moves': 0}
    _, src = d4_tables(device)
    planes = torch.stack([r['planes'] for r in rows]).to(device)
    raw = value_fn(planes).float().cpu()
    b = planes.shape[0]
    sym = torch.arange(8, device=device).repeat(b)
    avg = value_fn(augment_planes(planes.repeat_interleave(8, dim=0), sym, src)).float().cpu().reshape(b, 8).mean(1)
    return {'raw': _d2_scores(rows, raw), 'd4_average': _d2_scores(rows, avg),
            'note': 'child-level deep-win recognition (opponent to move after a proven losing move); '
                    'not the parent-level all-moves-lose judgement of P94; no PROVEN_WIN truths exist'}


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def latency(value_fn, planes: torch.Tensor, device, repeats: int = 200) -> dict:
    """Single-position latency raw vs D4 (8 in one batch), throughput at batch 256, peak memory."""
    _, src = d4_tables(device)
    one = planes[:1].to(device)
    eight = augment_planes(one.repeat(8, 1, 1, 1), torch.arange(8, device=device), src)

    def timed(x, n):
        for _ in range(min(10, n)):
            value_fn(x)
        _sync(device)
        times = []
        for _ in range(n):
            t0 = perf_counter()
            value_fn(x)
            _sync(device)
            times.append(perf_counter() - t0)
        return median(times)

    out = {'device': str(device), 'single_raw_ms': 1000 * timed(one, repeats),
           'single_d4_batch8_ms': 1000 * timed(eight, repeats)}
    batch = planes[:256].to(device)
    if batch.shape[0] < 256:
        batch = batch.repeat((256 + batch.shape[0] - 1) // batch.shape[0], 1, 1, 1)[:256]
    batch8 = augment_planes(batch.repeat_interleave(8, dim=0), torch.arange(8, device=device).repeat(256), src)
    for name, x in (('raw', batch), ('d4', batch8)):
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        t = timed(x, max(3, repeats // 7))
        out[f'positions_per_second_{name}'] = 256 / t
        out[f'peak_memory_mb_{name}'] = (torch.cuda.max_memory_allocated() / 2 ** 20
                                         if torch.device(device).type == 'cuda' else None)
    out['slowdown_single'] = out['single_d4_batch8_ms'] / out['single_raw_ms']
    out['slowdown_throughput'] = out['positions_per_second_raw'] / out['positions_per_second_d4']
    return out


def d3(v8: torch.Tensor, data: dict, d2_result: dict, probe_file: Path, value_fn, device) -> dict:
    raw, avg = split_metrics(v8[:, 0], data), split_metrics(v8.mean(1), data)

    def pick(m):
        return {'proof_balanced_accuracy': m['proof']['balanced_accuracy'],
                'proof_minus_accuracy': m['proof']['minus_accuracy'],
                'outcome_balanced_accuracy': m['outcome']['balanced_accuracy'], 'outcome_mse': m['outcome']['mse'],
                'black_outcome_ba': m['outcome_by_side']['black']['balanced_accuracy'],
                'white_outcome_ba': m['outcome_by_side']['white']['balanced_accuracy'],
                'outcome_ece': m['report']['outcome_ece'], 'outcome_bias': m['report']['bias']['outcome']}

    a, b = pick(raw), pick(avg)
    table = {k: {'raw': a[k], 'd4_average': b[k],
                 'delta': None if a[k] is None or b[k] is None else b[k] - a[k]} for k in a}
    for k in ('move_accuracy', 'unique_child_accuracy', 'root_clustered_accuracy', 'mean_value'):
        if d2_result.get('moves'):
            x, y = d2_result['raw'][k], d2_result['d4_average'][k]
            table[f'd2_{k}'] = {'raw': x, 'd4_average': y, 'delta': y - x}
    data_p = json.loads(Path(probe_file).read_text(encoding='utf-8'))
    probes = {}
    for name in ('P93', 'P94'):
        p = data_p['probes'][name]
        probes[name] = probe_values(value_fn, encode_game(_position(data_p['moves'][:p['plies_played']])), device)
    return {'table': table, 'dev_probes_P93_P94': probes}
