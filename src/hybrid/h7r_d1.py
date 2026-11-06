"""R2-D1-HEAD-WIDTH-1: value head width diagnosis (pure parts).

    widen_value_head   c5's value head widened to ``channels`` with its prediction preserved: channel 0 of the 1x1 conv
                       and of the BatchNorm copied, the new channels' conv weights Kaiming-initialised from their own
                       generator (never the trainer's random streams), the new channels' input columns of the first
                       Linear exactly 0, every other value-head tensor copied. The first forward equals c5's in eval
                       and in train mode; trunk and policy head are copied unchanged.
    preservation       the checks of that construction: copied slices byte-equal, new columns zero, the frozen state
                       unchanged, and the largest prediction difference on given rows
    select_subset      whole games of the c8 T-train split, in a seeded order, until a row count is first reached
                       (nested: the 1,000-row subset is a prefix of the 5,000-row one)
    floor              the weighted MSE no model can beat on a row set: rows with the same input (exact stones, or the
                       same up to D4 symmetry) predicted by their weighted mean target
    fit_probe          the trainer's own evaluation calls observed: at the parent and at every evaluation step the
                       train-subset fit of the same model is measured, and the model at the last evaluation (the last
                       step) is kept; the trainer's code and random streams are untouched
    fit_reading        the train-fit reading of the 1,000-row stage (contract section 4)
    hdev_reading       the R2 section 7 rule on paired H-dev results, unchanged
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import replace
from random import Random

FIT_CATEGORIES = ('WIDTH_EFFECT', 'BOTH_FIT', 'SEED_DISAGREEMENT', 'NEITHER_FIT', 'MIXED_FIT')


def require(ok, message) -> None:
    if not ok:
        raise ValueError(message)


# --- widening --------------------------------------------------------------------------------------------------------

def widen_value_head(parent, channels: int, init_seed: int):
    """A PolicyValueNet with ``value_channels = channels`` predicting as ``parent`` does (module docstring)."""
    import torch
    from model.network import PolicyValueNet

    old = parent.config.value_channels
    require(channels > old, f'value_channels {channels} is not wider than {old}')
    with torch.random.fork_rng(devices=[]):  # the default init of the new model does not consume the global stream
        model = PolicyValueNet(replace(parent.config, value_channels=channels))
    src, dst = parent.state_dict(), model.state_dict()
    actions = parent.config.board_size * parent.config.board_size
    out = {}
    for key, tensor in dst.items():
        if not key.startswith('value_head.'):
            out[key] = src[key].clone()
    generator = torch.Generator().manual_seed(init_seed)
    conv = torch.empty_like(dst['value_head.0.weight'])
    torch.nn.init.kaiming_normal_(conv, mode='fan_out', nonlinearity='relu', generator=generator)
    conv[:old] = src['value_head.0.weight']
    out['value_head.0.weight'] = conv
    for name, fill in (('weight', 1.0), ('bias', 0.0), ('running_mean', 0.0), ('running_var', 1.0)):
        t = torch.full_like(dst[f'value_head.1.{name}'], fill)
        t[:old] = src[f'value_head.1.{name}']
        out[f'value_head.1.{name}'] = t
    out['value_head.1.num_batches_tracked'] = src['value_head.1.num_batches_tracked'].clone()
    lin = torch.zeros_like(dst['value_head.4.weight'])
    lin[:, :old * actions] = src['value_head.4.weight']
    out['value_head.4.weight'] = lin
    for key in ('value_head.4.bias', 'value_head.6.weight', 'value_head.6.bias'):
        out[key] = src[key].clone()
    require(set(out) == set(dst), 'the widened state does not cover the new model')
    model.load_state_dict(out)
    return model.eval()


def preservation(parent, wide) -> dict:
    """Byte checks of the widened state against ``parent`` (no prediction involved)."""
    import torch
    from .h6_value_train import frozen_state, state_sha256

    old = parent.config.value_channels
    actions = parent.config.board_size * parent.config.board_size
    s, w = parent.state_dict(), wide.state_dict()
    checks = {
        'frozen_state': state_sha256(frozen_state(parent)) == state_sha256(frozen_state(wide)),
        'conv_channel_0': torch.equal(w['value_head.0.weight'][:old], s['value_head.0.weight']),
        'bn_channel_0': all(torch.equal(w[f'value_head.1.{n}'][:old], s[f'value_head.1.{n}'])
                            for n in ('weight', 'bias', 'running_mean', 'running_var')),
        'bn_batches': torch.equal(w['value_head.1.num_batches_tracked'], s['value_head.1.num_batches_tracked']),
        'linear_columns_copied': torch.equal(w['value_head.4.weight'][:, :old * actions], s['value_head.4.weight']),
        'new_columns_zero': bool((w['value_head.4.weight'][:, old * actions:] == 0).all()),
        'rest_copied': all(torch.equal(w[k], s[k]) for k in ('value_head.4.bias', 'value_head.6.weight',
                                                             'value_head.6.bias')),
    }
    return {'checks': checks, 'passed': all(checks.values()),
            'frozen_state_sha256': state_sha256(frozen_state(wide)), 'state_sha256': state_sha256(w)}


def predict_rows(model, data: dict, device, batch_size: int = 4096):
    """Eval-mode value of every row (float32, CPU)."""
    import torch
    from .h6_value_cache import planes_from_value_cache
    from .h6_value_train import value_forward

    was = model.value_head.training
    model.value_head.eval()
    out = []
    with torch.no_grad():
        for start in range(0, data['value'].shape[0], batch_size):
            index = torch.arange(start, min(data['value'].shape[0], start + batch_size))
            planes, _, _ = planes_from_value_cache(data, index, device)
            out.append(value_forward(model, planes).float().cpu())
    model.value_head.train(was)
    return torch.cat(out) if out else torch.zeros(0)


def train_mode_difference(parent, wide, data: dict, device, rows: int = 1024) -> float:
    """Largest |v| difference of one train-mode value-head forward (batch statistics) on the first ``rows`` rows,
    on copies (the models' BatchNorm statistics are not touched)."""
    import torch
    from .h6_value_cache import planes_from_value_cache
    from .h6_value_train import value_forward

    a, b = copy.deepcopy(parent).to(device), copy.deepcopy(wide).to(device)
    for m in (a, b):
        m.eval()
        m.value_head.train()
    index = torch.arange(min(rows, data['value'].shape[0]))
    planes, _, _ = planes_from_value_cache(data, index, device)
    with torch.no_grad():
        return float((value_forward(a, planes) - value_forward(b, planes)).abs().max())


# --- the subset ------------------------------------------------------------------------------------------------------

def select_subset(game_ids: list[str], rows_per_game: dict[str, int], seed: int, min_rows: list[int]) -> dict:
    """Whole games in a seeded order (sorted ids shuffled with ``seed``); for each target the first prefix whose
    rows reach it. The prefixes are nested."""
    order = sorted(game_ids)
    require(len(order) == len(set(order)), 'duplicate game ids')
    Random(seed).shuffle(order)
    out, total, taken = {}, 0, []
    targets = sorted(min_rows)
    for gid in order:
        if not targets:
            break
        taken.append(gid)
        total += rows_per_game[gid]
        while targets and total >= targets[0]:
            out[targets.pop(0)] = {'games': list(taken), 'rows': total}
    require(not targets, f'the split has fewer rows than {targets}')
    return out


# --- the floor -------------------------------------------------------------------------------------------------------

def floor(keys: list, target, weight) -> dict:
    """min over predictions of the weighted MSE when rows with the same key must share one prediction: each group
    predicted by its weighted mean target (within [-1, 1], so reachable by tanh)."""
    groups = {}
    for k, t, w in zip(keys, target.tolist(), weight.tolist()):
        g = groups.setdefault(k, [0.0, 0.0, 0.0])
        g[0] += w
        g[1] += w * t
        g[2] += w * t * t
    num = sum(g[2] - g[1] * g[1] / g[0] for g in groups.values() if g[0] > 0)
    den = sum(g[0] for g in groups.values())
    conflicting = sum(1 for g in groups.values() if g[2] - g[1] * g[1] / g[0] > 1e-12)
    return {'floor': max(0.0, num / den) if den else None, 'groups': len(groups), 'rows': len(keys),
            'conflicting_groups': conflicting}


def floors(data: dict) -> dict:
    """The floor with exact keys (stones as packed, the side to move) and with D4-canonical keys."""
    from .h7c_train import state_hashes

    exact = [(bytes(b.tolist()), bytes(w.tolist())) for b, w in zip(data['black'], data['white'])]
    target, weight = data['value'].double(), data['weight'].double()
    return {'exact': floor(exact, target, weight), 'd4': floor(state_hashes(data), target, weight)}


# --- the fit probe ---------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def fit_probe(subset: dict, steps: list[int], device):
    """During one ``train_value`` call: the trainer evaluates its cycle val set first at the parent and at each step in
    ``steps``; on each of those calls the same model's fit on ``subset`` is measured, and the state at the last
    evaluation (the last step) is copied. Yields {'fits': [...], 'final_state': ...}."""
    from . import h6_value_train as hv

    original = hv.evaluate
    seen = {'calls': 0, 'fits': [], 'final_state': None}
    points = ['parent', *steps]

    def wrapped(model, data, dev, **kw):
        out = original(model, data, dev, **kw)
        i = seen['calls']
        seen['calls'] += 1
        if i % 2 == 0:  # val_metrics(): the cycle val set first, then the replay one
            j = i // 2
            require(j < len(points), 'more trainer evaluations than expected')
            fit = original(model, subset, dev)
            seen['fits'].append({'at': points[j], 'weighted_mse': fit['weighted_mse'],
                                 'by_kind': {k: fit['by_kind'][k].get('mse') for k in ('outcome', 'proof')}})
            if j == len(points) - 1:
                seen['final_state'] = copy.deepcopy(model.state_dict())
        return out
    hv.evaluate = wrapped
    try:
        yield seen
    finally:
        hv.evaluate = original
    require(seen['calls'] == 2 * len(points), f"{seen['calls']} trainer evaluations, expected {2 * len(points)}")


# --- readings --------------------------------------------------------------------------------------------------------

def fit_reading(final: dict, floor_d4: float, margin: float, seeds: tuple) -> dict:
    """``final``: {'narrow': {seed: W}, 'wide': {seed: W}} at the same (last) step. Contract section 4; the first
    matching category in this order:

        WIDTH_EFFECT       W_wide - W_narrow < 0 for every seed and each |difference| above the narrow seed range
        BOTH_FIT           every arm and seed fits (W <= floor_d4 + margin): the direction is not read
        SEED_DISAGREEMENT  the seeds do not agree on the sign of W_wide - W_narrow (a zero counts as disagreeing)
        NEITHER_FIT        no arm and seed fits
        MIXED_FIT          some fit, some do not, the direction agreed but no width effect
    """
    success = {arm: {s: final[arm][s] <= floor_d4 + margin for s in seeds} for arm in ('narrow', 'wide')}
    diffs = {s: final['wide'][s] - final['narrow'][s] for s in seeds}
    narrow = [final['narrow'][s] for s in seeds]
    spread = max(narrow) - min(narrow)
    signs = {(d > 0) - (d < 0) for d in diffs.values()}
    if signs == {-1} and all(abs(d) > spread for d in diffs.values()):
        category = 'WIDTH_EFFECT'
    elif all(all(v.values()) for v in success.values()):
        category = 'BOTH_FIT'
    elif len(signs) > 1 or 0 in signs:
        category = 'SEED_DISAGREEMENT'
    elif not any(any(v.values()) for v in success.values()):
        category = 'NEITHER_FIT'
    else:
        category = 'MIXED_FIT'
    return {'category': category, 'order': list(FIT_CATEGORIES), 'floor_d4': floor_d4, 'margin': margin,
            'threshold': floor_d4 + margin, 'success': success, 'wide_minus_narrow': diffs,
            'narrow_seed_range': spread}


def hdev_reading(paired: dict, epsilon: float, control_primary: dict, seeds: tuple) -> dict:
    """R2 section 7: HOLD if the control seed range exceeds epsilon; DEVELOPMENT_SIGNAL if for every seed
    I_s = L(control) - L(wide) > epsilon and the paired 95% upper bound of L(wide) - L(control) is below 0."""
    values = [control_primary[s] for s in seeds]
    spread = max(values) - min(values)
    per = {s: {'improvement': paired[s]['improvement'], 'beyond_epsilon': paired[s]['improvement'] > epsilon,
               'upper95_below_0': paired[s]['ci95'] is not None and paired[s]['ci95'][1] < 0} for s in seeds}
    hold = [f'the control seed range {spread:.6g} exceeds epsilon {epsilon:.6g}'] if spread > epsilon else []
    signal = all(v['beyond_epsilon'] and v['upper95_below_0'] for v in per.values())
    return {'label': 'HOLD' if hold else ('DEVELOPMENT_SIGNAL' if signal else 'NOT_ESTABLISHED'), 'hold': hold,
            'control_seed_range': spread, 'epsilon': epsilon, 'seeds': per}
