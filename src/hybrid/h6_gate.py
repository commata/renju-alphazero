"""H6-3 offline gate for the H6-2a value head (docs/h6-value.md, "H6-3 오프라인 gate").

The thresholds below were fixed before the test split was first opened; do not change them after a run.

Test partition: the test games are split once, by game id, into H6-3A (first gate) and H6-3B (reserve):
``sha256(PARTITION_SALT + game_id)`` even -> A, odd -> B. A model changed after seeing A (``--prove-losses``
v3, H6-2b, ...) is gated on B; A is then a dev set. P93/P94 likewise become dev probes once a model was
changed because of them.

Gates (all must hold):
    G1 proof sign      proof balanced accuracy >= 0.90 and proof -1 accuracy >= 0.85
    G2 outcome         outcome MSE < 1.000 and outcome balanced accuracy >= 0.60
    G3 D4              over 1,000 outcome + 1,000 proof states (fixed seed): mean of the per-state std of the
                       8 symmetric values <= 0.05, its 95th percentile <= 0.10, and among states with
                       |mean of the 8| >= 0.10 the share whose 8 signs all agree >= 0.95
    G4 probes          P93 (white to move) original > 0 and D4 mean > 0; P94 (black to move) original < 0 and
                       D4 mean < 0
    G5 colours         outcome balanced accuracy >= 0.55 for black to move and for white to move, and their
                       difference <= 0.05
Report only: outcome ECE (10 equal-width bins of p = (v+1)/2 against y = (t+1)/2, unweighted) and the
lambda it implies (ECE <= 0.10 -> 1.0, else 0.5), bias (mean v - mean t) by side / ply / kind, the P92
ranking of the move actually played, bias-corrected outcome sign accuracy, sign agreement over all D4 states.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from model.config import BOARD_SIZE
from model.encoding import encode_game
from renju import Game

from .h3_train import d4_tables
from .h6_value_cache import KIND_OUTCOME, KIND_PROOF, planes_from_value_cache
from .h6_value_train import PLY_BUCKETS, augment_planes

FORMAT = 'h6-offline-gate-v1'
PARTITION_SALT = 'h6-3-v1:'
D4_SAMPLE = {'outcome': 1000, 'proof': 1000}
D4_SEED = 20261014
THRESHOLDS = {
    'G1': {'proof_balanced_accuracy_min': 0.90, 'proof_minus_accuracy_min': 0.85},
    'G2': {'outcome_mse_max_exclusive': 1.0, 'outcome_balanced_accuracy_min': 0.60},
    'G3': {'mean_std_max': 0.05, 'p95_std_max': 0.10, 'confident_abs_mean_min': 0.10,
           'confident_sign_agreement_min': 0.95},
    'G4': {'P93': 1, 'P94': -1},
    'G5': {'outcome_balanced_accuracy_min': 0.55, 'max_difference': 0.05},
}
ECE_BINS = 10
ECE_LAMBDA = {'ece_max_for_lambda_1': 0.10, 'lambda_low_ece': 1.0, 'lambda_high_ece': 0.5}
FAILURE_PATHS = {
    'G1': 'shallow proof / negative generalisation: review --prove-losses and the negative label mix',
    'G2': 'frozen H3 trunk may lack long-range outcome features: consider H6-2b',
    'G3': 'check the D4 transform, encoder, BatchNorm and augmentation before any retraining',
    'G4_P94_only': 'deep forced-loss representation: --prove-losses v3 first (P94 becomes a dev probe)',
    'G4_P93': 'investigate the win-side sign / probe-specific representation; not --prove-losses',
    'G5': 'colour-conditional generalisation: analyse side bias by kind and ply before changing training',
}


def part_of(game_id: str) -> str:
    return 'A' if int(hashlib.sha256((PARTITION_SALT + str(game_id)).encode()).hexdigest(), 16) % 2 == 0 else 'B'


def partition(data: dict, game_ids: list[str], part: str) -> tuple[dict, dict]:
    """The rows of one test part (whole games); ``game_ids`` is the cache's ``test_games.json``."""
    in_part = torch.tensor([part_of(g) == part for g in game_ids], dtype=torch.bool)
    mask = in_part[data['game'].long()] if len(game_ids) else torch.zeros(0, dtype=torch.bool)
    games = sorted(g for g in game_ids if part_of(g) == part)
    info = {'part': part, 'salt': PARTITION_SALT, 'games': len(games), 'states': int(mask.sum()),
            'games_sha256': hashlib.sha256(json.dumps(games).encode()).hexdigest(),
            'other_part_games': len(game_ids) - len(games)}
    return {k: v[mask] for k, v in data.items()}, info


@torch.no_grad()
def values(value_fn, data: dict, device, batch_size: int = 4096) -> torch.Tensor:
    n = data['value'].shape[0]
    out = []
    for start in range(0, n, batch_size):
        index = torch.arange(start, min(n, start + batch_size))
        planes, _, _ = planes_from_value_cache(data, index, device)
        out.append(value_fn(planes).float().cpu())
    return torch.cat(out) if out else torch.zeros(0)


def _acc(v, t, mask):
    m = mask & (t != 0)
    return float(((v[m] > 0) == (t[m] > 0)).float().mean()) if bool(m.any()) else None


def _ba(v, t, mask):
    plus, minus = _acc(v, t, mask & (t > 0)), _acc(v, t, mask & (t < 0))
    return None if plus is None or minus is None else (plus + minus) / 2, plus, minus


def _bias(v, t, mask):
    return float(v[mask].mean() - t[mask].mean()) if bool(mask.any()) else None


def ece(v, t, bins: int = ECE_BINS) -> float | None:
    if v.numel() == 0:
        return None
    p, y = (v + 1) / 2, (t + 1) / 2
    index = torch.clamp((p * bins).long(), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = index == b
        if m.any():
            total += float(m.sum()) / v.numel() * abs(float(p[m].mean()) - float(y[m].mean()))
    return total


def split_metrics(v: torch.Tensor, data: dict) -> dict:
    t, kind, ply = data['value'], data['kind'], data['ply'].long()
    proof, outcome = kind == KIND_PROOF, kind == KIND_OUTCOME
    black, white = ply % 2 == 0, ply % 2 == 1
    proof_ba, proof_plus, proof_minus = _ba(v, t, proof)
    out_ba, out_plus, out_minus = _ba(v, t, outcome)
    black_ba, _, _ = _ba(v, t, outcome & black)
    white_ba, _, _ = _ba(v, t, outcome & white)
    out_bias = _bias(v, t, outcome)
    corrected = v - (out_bias or 0.0)
    return {
        'states': int(v.numel()),
        'proof': {'states': int(proof.sum()), 'minus_states': int((proof & (t < 0)).sum()),
                  'balanced_accuracy': proof_ba, 'plus_accuracy': proof_plus, 'minus_accuracy': proof_minus,
                  'mse': float((v[proof] - t[proof]).square().mean()) if proof.any() else None},
        'outcome': {'states': int(outcome.sum()), 'balanced_accuracy': out_ba, 'plus_accuracy': out_plus,
                    'minus_accuracy': out_minus,
                    'mse': float((v[outcome] - t[outcome]).square().mean()) if outcome.any() else None},
        'outcome_by_side': {'black': {'balanced_accuracy': black_ba, 'bias': _bias(v, t, outcome & black)},
                            'white': {'balanced_accuracy': white_ba, 'bias': _bias(v, t, outcome & white)}},
        'report': {
            'outcome_ece': ece(v[outcome], t[outcome]),
            'bias': {'all': _bias(v, t, torch.ones_like(proof)), 'outcome': out_bias, 'proof': _bias(v, t, proof),
                     'black': _bias(v, t, black), 'white': _bias(v, t, white),
                     'by_ply': {f'{lo}-{hi}': _bias(v, t, (ply >= lo) & (ply <= hi)) for lo, hi in PLY_BUCKETS},
                     'outcome_by_ply': {f'{lo}-{hi}': _bias(v, t, outcome & (ply >= lo) & (ply <= hi))
                                        for lo, hi in PLY_BUCKETS}},
            'outcome_bias_corrected': dict(zip(('balanced_accuracy', 'plus_accuracy', 'minus_accuracy'),
                                               _ba(corrected, t, outcome))),
        },
    }


@torch.no_grad()
def d4_metrics(value_fn, data: dict, device) -> dict:
    gen = torch.Generator().manual_seed(D4_SEED)
    picks = []
    for name, code in (('outcome', KIND_OUTCOME), ('proof', KIND_PROOF)):
        rows = (data['kind'] == code).nonzero().flatten()
        take = min(D4_SAMPLE[name], rows.numel())
        picks.append(rows[torch.randperm(rows.numel(), generator=gen)[:take]])
    index = torch.cat(picks).sort().values
    _, src = d4_tables(device)
    vals = []
    for start in range(0, index.numel(), 512):
        planes, _, _ = planes_from_value_cache(data, index[start:start + 512], device)
        b = planes.shape[0]
        sym = torch.arange(8, device=device).repeat(b)
        moved = augment_planes(planes.repeat_interleave(8, dim=0), sym, src)
        vals.append(value_fn(moved).float().cpu().reshape(b, 8))
    v8 = torch.cat(vals) if vals else torch.zeros(0, 8)
    std = v8.std(dim=1, unbiased=False)
    mean = v8.mean(dim=1)
    agree = ((v8 > 0).all(1) | (v8 < 0).all(1))
    confident = mean.abs() >= THRESHOLDS['G3']['confident_abs_mean_min']
    return {'states': int(v8.shape[0]), 'sample': D4_SAMPLE, 'seed': D4_SEED,
            'mean_std': float(std.mean()) if v8.numel() else None,
            'p95_std': float(torch.quantile(std, 0.95)) if v8.numel() else None,
            'max_std': float(std.max()) if v8.numel() else None,
            'confident_states': int(confident.sum()),
            'confident_sign_agreement': float(agree[confident].float().mean()) if confident.any() else None,
            'report_all_sign_agreement': float(agree.float().mean()) if v8.numel() else None}


def _position(moves) -> Game:
    game = Game()
    for move in moves:
        game.play(*move)
    return game


@torch.no_grad()
def probe_values(value_fn, planes: torch.Tensor, device) -> dict:
    _, src = d4_tables(device)
    moved = augment_planes(planes.unsqueeze(0).repeat(8, 1, 1, 1).to(device), torch.arange(8, device=device), src)
    v = value_fn(moved).float().cpu()
    return {'original': float(v[0]), 'd4_mean': float(v.mean()), 'd4_min': float(v.min()),
            'd4_max': float(v.max()), 'd4_std': float(v.std(unbiased=False))}


@torch.no_grad()
def probes(value_fn, probe_file: Path, device) -> dict:
    data = json.loads(Path(probe_file).read_text(encoding='utf-8'))
    out = {}
    for name, expected in THRESHOLDS['G4'].items():
        p = data['probes'][name]
        if p.get('value_side_to_move') != expected:
            raise ValueError(f'{probe_file}: {name} value_side_to_move {p.get("value_side_to_move")} != {expected}')
        game = _position(data['moves'][:p['plies_played']])
        r = probe_values(value_fn, encode_game(game), device)
        r['side_to_move'] = 'black' if game.to_play == 1 else 'white'
        r['expected_sign'] = expected
        r['pass'] = (r['original'] * expected > 0) and (r['d4_mean'] * expected > 0)
        out[name] = r
    # P92 (report only): black to move; rank of the move actually played among all legal moves by -v(child)
    p92 = data['probes']['P92']
    game = _position(data['moves'][:p92['plies_played']])
    played = tuple(data['moves'][p92['plies_played']])
    moves = game.legal_moves()
    children = []
    for move in moves:
        child = _position(data['moves'][:p92['plies_played']] + [list(move)])
        children.append(encode_game(child) if not child.done else None)
    scores = []
    batch = [c for c in children if c is not None]
    vals = value_fn(torch.stack(batch).to(device)).float().cpu().tolist() if batch else []
    it = iter(vals)
    for move, child in zip(moves, children):
        scores.append((1.0 if child is None else -next(it), tuple(move)))  # a move that ends the game: a win
    scores.sort(key=lambda s: -s[0])
    rank = next(i for i, (_, m) in enumerate(scores, 1) if m == played)
    out['P92'] = {'report_only': True, 'played_0idx': list(played), 'rank': rank, 'legal_moves': len(scores),
                  'played_value': scores[rank - 1][0],
                  'top5': [{'move_0idx': list(m), 'value': s} for s, m in scores[:5]]}
    return out


def judge(metrics: dict, d4: dict, probe: dict) -> dict:
    t = THRESHOLDS
    pr, oc, side = metrics['proof'], metrics['outcome'], metrics['outcome_by_side']
    ge = lambda x, lo: x is not None and x >= lo  # noqa: E731
    gates = {
        'G1': ge(pr['balanced_accuracy'], t['G1']['proof_balanced_accuracy_min'])
        and ge(pr['minus_accuracy'], t['G1']['proof_minus_accuracy_min']),
        'G2': oc['mse'] is not None and oc['mse'] < t['G2']['outcome_mse_max_exclusive']
        and ge(oc['balanced_accuracy'], t['G2']['outcome_balanced_accuracy_min']),
        'G3': d4['mean_std'] is not None and d4['mean_std'] <= t['G3']['mean_std_max']
        and d4['p95_std'] <= t['G3']['p95_std_max']
        and ge(d4['confident_sign_agreement'], t['G3']['confident_sign_agreement_min']),
        'G4': probe['P93']['pass'] and probe['P94']['pass'],
        'G5': ge(side['black']['balanced_accuracy'], t['G5']['outcome_balanced_accuracy_min'])
        and ge(side['white']['balanced_accuracy'], t['G5']['outcome_balanced_accuracy_min'])
        and abs(side['black']['balanced_accuracy'] - side['white']['balanced_accuracy']) <= t['G5']['max_difference'],
    }
    failed = [g for g, ok in gates.items() if not ok]
    paths = {g: FAILURE_PATHS[g] for g in failed if g != 'G4'}
    if 'G4' in failed:
        if not probe['P93']['pass']:
            paths['G4'] = FAILURE_PATHS['G4_P93']
        elif not {'G1', 'G2', 'G3'} & set(failed):
            paths['G4'] = FAILURE_PATHS['G4_P94_only']
        else:
            paths['G4'] = 'P94 failed together with ' + ', '.join(sorted({'G1', 'G2', 'G3'} & set(failed))) + \
                ': resolve those first; P94 alone does not yet point at --prove-losses'
    e = metrics['report']['outcome_ece']
    lam = None if e is None else (ECE_LAMBDA['lambda_low_ece'] if e <= ECE_LAMBDA['ece_max_for_lambda_1']
                                  else ECE_LAMBDA['lambda_high_ece'])
    return {'verdict': 'PASS' if not failed else 'FAIL', 'gates': gates, 'failed': failed,
            'next_steps_if_failed': paths, 'lambda': lam, 'outcome_ece': e}
