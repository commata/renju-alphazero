"""H7-R R2 (``docs/h7r/r2/R2-DIAG-1-draft.md``): the R2 contract, the value model identity, game-balanced value
metrics with one source-game bootstrap shared by every model and group, the epsilon rule, the reproduction gate of
R2-0B and the trainer's data assignment.

Metrics (``h7r-r2-metrics-v1``), per prediction (raw / D4), group of rows and model, from per-game sums:

    pooled  sum_i w_i e_i / sum_i w_i over the group's rows (the trainer's weighted MSE)
    macro   mean over the games with group weight of L_g = sum_{i in g} w_i e_i / sum_{i in g} w_i
    sign_acc_strict_macro   mean over the games with a t != 0 row of the share of such rows with v * t > 0
                            (v = 0 fails; the trainer's ``sign_acc`` counts v = 0, t < 0 as right)
    mean_abs_value          mean over the games of the mean |v| (read on the t = 0 groups)

The bootstrap draws the source games once per replicate (with replacement, over every game of the cache) and applies
the same multiplicities to every model, prediction, group and comparison. A group without rows in a game adds nothing
to that replicate; replicates whose group denominator is 0 are counted and leave the interval unread, never dropped.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

CONTRACT_FORMAT = 'h7r-r2-contract-v1'
RESULT_FORMAT = 'h7r-r2-eval-v1'
METRIC_VERSION = 'h7r-r2-metrics-v1'
EPSILON_VERSION = 'h7r-r2-epsilon-v1'
REPRODUCTION_VERSION = 'h7r-r2-reproduction-v1'
ASSIGNMENT_VERSION = 'h7r-r2-assignment-v1'
PURPOSE = 'r2_diagnosis'
PREDICTIONS = ('raw', 'd4')
AGGREGATES = ('macro', 'pooled')
IDENTITY_KEYS = ('value_checkpoint_sha256', 'value_metadata_sha256', 'policy_checkpoint_sha256',
                 'policy_metadata_sha256')
HDEV_SHA_KEYS = ('contract_sha256', 'cache_binding_sha256', 'cache_labels_sha256', 'test_sha256', 'games_sha256',
                 'rows_sha256')
STAGES = ('R2-0A', 'R2-0B')
KIND_NAMES = {0: 'outcome', 1: 'proof'}
RESULTS = ('black', 'white', 'draw')
REPRODUCED = ('BITWISE_REPRODUCED', 'METRIC_ONLY')
_SHA = re.compile(r'[0-9a-f]{64}')


def require(ok, message: str) -> None:
    if not ok:
        raise ValueError(message)


def is_sha(value) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


# --- the contract ------------------------------------------------------------------------------------------------

def _positive_or_none(value) -> bool:
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                             and math.isfinite(value) and value > 0)


def check_contract(contract: dict) -> None:
    """The structure and the fixed choices of an R2 contract; a draft may leave the R2-0B tolerances open."""
    require(isinstance(contract, dict) and contract.get('format') == CONTRACT_FORMAT, 'not an R2 contract')
    require(isinstance(contract.get('contract_id'), str) and contract['contract_id'], 'missing contract id')
    require(contract.get('status') in ('draft', 'fixed'), 'status must be draft or fixed')
    require(contract.get('purpose') == PURPOSE, f'purpose must be {PURPOSE}')
    hdev = contract.get('hdev')
    require(isinstance(hdev, dict) and isinstance(hdev.get('contract'), str), 'missing H-dev reference')
    require(all(is_sha(hdev.get(k)) for k in HDEV_SHA_KEYS), 'H-dev hashes must be SHA-256')
    require(all(type(hdev.get(k)) is int and hdev[k] > 0 for k in ('games', 'rows')), 'H-dev games / rows')
    require(is_sha(contract.get('frozen_state_sha256')), 'missing frozen state hash')
    models = contract.get('models')
    require(isinstance(models, dict) and len(models) >= 2, 'at least two models')
    for name, spec in models.items():
        require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_-]+', name) is not None, f'model name {name!r}')
        require(isinstance(spec, dict) and all(is_sha(spec.get(k)) for k in IDENTITY_KEYS),
                f'model {name}: identity must be four SHA-256 values')
    comparisons = contract.get('comparisons')
    require(isinstance(comparisons, list) and comparisons and len(set(comparisons)) == len(comparisons),
            'comparisons must be a non-empty list without repeats')
    for spec in comparisons:
        parts = spec.split(':') if isinstance(spec, str) else []
        require(len(parts) == 2 and parts[0] != parts[1] and all(p in models for p in parts),
                f'comparison {spec!r} must be A:B of two contract models')
    metrics = contract.get('metrics')
    require(isinstance(metrics, dict) and metrics.get('version') == METRIC_VERSION, f'metrics must be {METRIC_VERSION}')
    bins = metrics.get('ply_bins')
    require(isinstance(bins, list) and len(bins) >= 2 and all(type(b) is int for b in bins)
            and all(a < b for a, b in zip(bins, bins[1:])) and bins[0] >= 0, 'ply bins must increase')
    require(type(metrics.get('min_contributing_games')) is int and metrics['min_contributing_games'] >= 1,
            'min_contributing_games must be a positive integer')
    names = group_names(bins)
    for item in [metrics.get('primary')] + list(metrics.get('components') or []):
        require(isinstance(item, dict) and set(item) == {'prediction', 'group', 'aggregate'}
                and item['prediction'] in PREDICTIONS and item['aggregate'] in AGGREGATES
                and (item['group'] in names or item['group'].startswith('network/proof:')),
                f'metric {item!r}: prediction raw / d4, a known group, aggregate macro / pooled')
    require(metrics['primary']['aggregate'] == 'macro', 'the primary metric is a game macro')
    boot = contract.get('bootstrap')
    require(isinstance(boot, dict) and type(boot.get('samples')) is int and boot['samples'] >= 100
            and type(boot.get('seed')) is int and boot.get('unit') == 'source game'
            and boot.get('method') == 'percentile' and boot.get('level') == 0.95,
            'bootstrap: samples >= 100, integer seed, source game, percentile, 0.95')
    eps = contract.get('epsilon')
    require(isinstance(eps, dict) and eps.get('version') == EPSILON_VERSION, f'epsilon must be {EPSILON_VERSION}')
    require(eps.get('reference') in comparisons, 'the epsilon reference must be a contract comparison')
    seeds = eps.get('training_seeds')
    require(isinstance(seeds, list) and len(seeds) == 3 and len(set(seeds)) == 3
            and all(type(s) is int for s in seeds), 'three distinct integer training seeds')
    require(eps.get('value') is None, 'epsilon is computed from R2-0A / R2-0B records, never written in the contract')
    rep = contract.get('reproduction')
    require(isinstance(rep, dict) and rep.get('version') == REPRODUCTION_VERSION and type(rep.get('seed')) is int
            and rep['seed'] not in seeds, 'reproduction: version and a seed outside the training seeds')
    tol = rep.get('tolerance')
    require(isinstance(tol, dict) and set(tol) == {'max_abs_prediction', 'abs_primary'}
            and all(_positive_or_none(v) for v in tol.values()), 'reproduction tolerances: positive or null')
    stages = contract.get('stages')
    require(isinstance(stages, dict) and set(stages) == set(STAGES)
            and all(isinstance(stages[s], dict) and type(stages[s].get('approved')) is bool for s in STAGES),
            f'stages {STAGES} with an approved flag')
    if contract['status'] == 'draft':
        require(not any(stages[s]['approved'] for s in STAGES), 'a draft contract approves no stage')
    if stages['R2-0B']['approved']:
        require(all(v is not None for v in tol.values()), 'R2-0B needs the reproduction tolerances first')


def load_contract(path) -> tuple[dict, str]:
    path = Path(path)
    contract = json.loads(path.read_text(encoding='utf-8'))
    check_contract(contract)
    return contract, file_sha256(path)


# --- groups ------------------------------------------------------------------------------------------------------

def ply_bin_names(bins: list[int]) -> list[str]:
    return [f'{a}-{b}' for a, b in zip(bins, bins[1:])] + [f'{bins[-1]}+']


def group_names(bins: list[int]) -> set[str]:
    """The fixed groups (the proof sub-kinds ``network/proof:<kind>`` come from the labels)."""
    names = {'all', 'network', 'exact_leaf', 'outcome', 'proof', 'target:0', 'network/target:0',
             'network/target:nonzero'}
    names |= {f'{r}/{k}' for r in ('network', 'exact_leaf') for k in KIND_NAMES.values()}
    names |= {f'network/result:{x}' for x in RESULTS} | {f'network/side:{x}' for x in ('black', 'white')}
    names |= {f'network/ply:{x}' for x in ply_bin_names(bins)}
    return names


def ply_bin(ply: int, bins: list[int]) -> str:
    require(ply >= bins[0], f'row at ply {ply} is below the first ply bin {bins[0]}')
    names = ply_bin_names(bins)
    for i, upper in enumerate(bins[1:]):
        if ply < upper:
            return names[i]
    return names[-1]


def row_groups(route: list[str], kind: list[int], ply: list[int], target: list[float], result: list[str],
               label_kind: list[str], bins: list[int]) -> dict[str, list[int]]:
    """Group name -> row indexes. ``route``: network / exact_leaf; ``result``: the source game's black / white / draw;
    the side to move is black on even plies."""
    out: dict[str, list[int]] = {name: [] for name in sorted(group_names(bins))}
    for i, (r, k, p, t, res, lk) in enumerate(zip(route, kind, ply, target, result, label_kind)):
        require(r in ('network', 'exact_leaf') and k in KIND_NAMES and res in RESULTS, f'row {i}: unknown class')
        name = KIND_NAMES[k]
        require(name == 'proof' if lk.startswith('proof_') else name == 'outcome', f'row {i}: label kind mismatch')
        keys = ['all', r, name, f'{r}/{name}']
        if t == 0:
            keys.append('target:0')
        if r == 'network':
            keys += [f'network/result:{res}', f"network/side:{'black' if p % 2 == 0 else 'white'}",
                     f'network/ply:{ply_bin(p, bins)}', 'network/target:0' if t == 0 else 'network/target:nonzero']
            if name == 'proof':
                keys.append(f'network/proof:{lk[len("proof_"):]}')
        for key in keys:
            out.setdefault(key, []).append(i)
    return out


# --- per-game sums and the metrics -------------------------------------------------------------------------------

def game_sums(pred, target, weight, game, rows: list[int], n_games: int) -> dict:
    """Per-game float64 sums of one group: weighted error and weight, strict sign hits / rows, |v| and rows."""
    import torch

    idx = torch.as_tensor(rows, dtype=torch.long)
    g = torch.as_tensor(game, dtype=torch.long)[idx]
    v = torch.as_tensor(pred, dtype=torch.float64)[idx]
    t = torch.as_tensor(target, dtype=torch.float64)[idx]
    w = torch.as_tensor(weight, dtype=torch.float64)[idx]
    zero = lambda: torch.zeros(n_games, dtype=torch.float64)  # noqa: E731
    nonzero = (t != 0).double()
    out = {'num': zero().index_add_(0, g, w * (v - t).square()), 'den': zero().index_add_(0, g, w),
           'hit': zero().index_add_(0, g, nonzero * ((v * t) > 0).double()), 'signed': zero().index_add_(0, g, nonzero),
           'absv': zero().index_add_(0, g, v.abs()), 'count': zero().index_add_(0, g, torch.ones_like(v))}
    return out


def _ratio(sums: dict, num: str, den: str):
    import torch
    present = (sums[den] > 0).double()
    value = torch.where(sums[den] > 0, sums[num] / sums[den].clamp_min(1e-300), torch.zeros_like(sums[num]))
    return value, present


def point_metrics(sums: dict) -> dict:
    loss, present = _ratio(sums, 'num', 'den')
    sign, signed = _ratio(sums, 'hit', 'signed')
    absv, counted = _ratio(sums, 'absv', 'count')
    games = int(present.sum())
    den = float(sums['den'].sum())
    mean = lambda x, m: float((x * m).sum() / m.sum()) if float(m.sum()) > 0 else None  # noqa: E731
    return {'games': games, 'rows': int(sums['count'].sum()), 'weight_sum': den,
            'macro': mean(loss, present), 'pooled': float(sums['num'].sum()) / den if den > 0 else None,
            'sign_acc_strict_macro': mean(sign, signed), 'sign_games': int(signed.sum()),
            'mean_abs_value': mean(absv, counted)}


def multiplicities(n_games: int, samples: int, seed: int):
    """[samples, n_games] float64 counts: each replicate draws n_games source games with replacement."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(0, n_games, (samples, n_games), generator=generator)
    counts = torch.zeros((samples, n_games), dtype=torch.float64)
    counts.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
    return counts


def _interval(draws, zero_rows: int, level: float) -> dict:
    """Percentile interval of the replicates (the existing ``paired`` convention: sorted, index int(p * (n - 1)))."""
    import torch

    if zero_rows:
        return {'ci95': None, 'zero_denominator_resamples': zero_rows}
    ordered = torch.sort(torch.as_tensor(draws, dtype=torch.float64)).values
    tail = (1 - level) / 2
    q = lambda p: float(ordered[int(p * (len(ordered) - 1))])  # noqa: E731
    return {'ci95': [q(tail), q(1 - tail)], 'zero_denominator_resamples': 0}


def replicate(counts, sums_a: dict, sums_b: dict | None, aggregate: str):
    """Replicates of a metric (``sums_b`` None) or of the difference A - B on the same rows; and the number of
    replicates with a zero denominator."""
    if aggregate == 'macro':
        loss_a, present = _ratio(sums_a, 'num', 'den')
        values = loss_a if sums_b is None else loss_a - _ratio(sums_b, 'num', 'den')[0]
        den = counts @ present
        num = counts @ (values * present)
    else:
        den = counts @ sums_a['den']
        num = counts @ (sums_a['num'] if sums_b is None else sums_a['num'] - sums_b['num'])
    zero = int((den <= 0).sum())
    return num / den.clamp_min(1e-300), zero


def bootstrap(counts, sums_a: dict, sums_b: dict | None, aggregate: str, level: float) -> dict:
    draws, zero = replicate(counts, sums_a, sums_b, aggregate)
    return _interval(draws, zero, level)


def difference(sums_a: dict, sums_b: dict, aggregate: str):
    a, b = point_metrics(sums_a)[aggregate], point_metrics(sums_b)[aggregate]
    return None if a is None or b is None else a - b


def reversals(comparison: dict, primary: dict) -> list[str]:
    """The pre-registered direction checks of a comparison whose sign differs from the primary's."""
    def sign(pred, group, aggregate):
        cell = comparison.get(pred, {}).get(group, {}).get(aggregate) or {}
        d = cell.get('difference')
        return None if d is None or d == 0 else (d > 0)

    p, g = primary['prediction'], primary['group']
    base = sign(p, g, 'macro')
    other = 'raw' if p == 'd4' else 'd4'
    route_all = 'all' if g == 'network' else g.replace('network/', '', 1)
    checks = {f'{other} vs {p}': sign(other, g, 'macro'), 'pooled vs macro': sign(p, g, 'pooled'),
              f'{route_all} vs {g}': sign(p, route_all, 'macro'),
              'outcome vs proof': (lambda a, b: None if a is None or b is None else a != b)(
                  sign(p, 'network/outcome', 'macro'), sign(p, 'network/proof', 'macro'))}
    out = []
    for name, s in checks.items():
        if name == 'outcome vs proof':
            if s:
                out.append(name)
        elif base is not None and s is not None and s != base:
            out.append(name)
    return out


# --- epsilon and the reproduction gate (R2-0B) -------------------------------------------------------------------

def half_width(difference: float, ci) -> float:
    """The larger distance from a difference to the ends of its interval (the sample part of epsilon)."""
    lo, hi = ci
    return max(abs(difference - lo), abs(hi - difference))


def epsilon(reference_difference, reference_ci, seed_values, reproduction: str) -> dict:
    """``h7r-r2-epsilon-v1``: max(half-width of the reference comparison's interval around its difference, the range
    of the three training seeds' primary metric). Nothing while the reproduction check has not passed."""
    out = {'version': EPSILON_VERSION, 'value': None, 'sample': None, 'seed': None}
    if reproduction not in REPRODUCED:
        return {**out, 'status': 'BLOCKED_REPRODUCTION', 'reproduction': reproduction}
    finite = lambda x: isinstance(x, (int, float)) and math.isfinite(x)  # noqa: E731
    if not (finite(reference_difference) and isinstance(reference_ci, (list, tuple)) and len(reference_ci) == 2
            and all(finite(x) for x in reference_ci) and isinstance(seed_values, (list, tuple))
            and len(seed_values) == 3 and all(finite(x) for x in seed_values)):
        return {**out, 'status': 'INCOMPLETE'}
    sample = half_width(reference_difference, reference_ci)
    seed = max(seed_values) - min(seed_values)
    value = max(sample, seed)
    if not value > 0:
        return {**out, 'sample': sample, 'seed': seed, 'status': 'ZERO'}
    return {**out, 'value': value, 'sample': sample, 'seed': seed, 'status': 'COMPUTED'}


def reproduction_verdict(checks: dict, tolerance: dict) -> dict:
    """``h7r-r2-reproduction-v1`` of one re-run against the reference run.

    ``checks``: ``state_sha256`` and ``assignment`` (pairs reference / re-run), ``best_step`` (pair),
    ``max_abs_prediction`` and ``abs_primary`` (floats). Missing items or tolerances never pass."""
    reasons = []
    pair = lambda k: checks.get(k) if isinstance(checks.get(k), (list, tuple)) and len(checks[k]) == 2 else None  # noqa: E731
    state, assignment, step = pair('state_sha256'), pair('assignment'), pair('best_step')
    if state is not None and state[0] is not None and state[0] == state[1]:
        return {'version': REPRODUCTION_VERSION, 'verdict': 'BITWISE_REPRODUCED', 'reasons': []}
    if state is None or state[0] is None:
        reasons.append('state_sha256 missing')
    for name, value in (('assignment', assignment), ('best_step', step)):
        if value is None or value[0] is None:
            reasons.append(f'{name} missing')
        elif value[0] != value[1]:
            reasons.append(f'{name} differs')
    for name, key in (('max_abs_prediction', 'max_abs_prediction'), ('abs_primary', 'abs_primary')):
        limit, value = tolerance.get(key), checks.get(name)
        if limit is None:
            reasons.append(f'{key} tolerance not fixed')
        elif not isinstance(value, (int, float)) or not math.isfinite(value):
            reasons.append(f'{name} missing')
        elif value > limit:
            reasons.append(f'{name} {value:.3g} > {limit:.3g}')
    verdict = 'NOT_REPRODUCED' if reasons else 'METRIC_ONLY'
    return {'version': REPRODUCTION_VERSION, 'verdict': verdict, 'reasons': reasons}


# --- the trainer's data assignment --------------------------------------------------------------------------------

def data_assignment(n_cycle: int, n_replay: int, seed: int, batch_size: int, epochs: float,
                    augment_d4: bool = True, detail: bool = False) -> dict:
    """What a training seed decides in ``h7c_train.train_value``: per epoch the replay rows drawn and the order of the
    merged rows (``mixed_epoch`` itself, on row ids), and the D4 symmetry of every row (the trainer's generator
    walk). Two runs with the same fingerprints saw the same rows, in the same order, under the same symmetries."""
    import torch

    from .h7c_train import mixed_epoch

    require(n_cycle > 0 and n_replay >= n_cycle and batch_size > 0, 'invalid assignment sizes')
    keys = ('black', 'white', 'legal', 'last', 'ply', 'value', 'weight', 'kind')
    cycle = {k: torch.arange(n_cycle) for k in keys}
    replay = {k: torch.arange(n_replay) for k in keys}
    n = 2 * n_cycle
    total = max(1, round(epochs * math.ceil(n / batch_size)))
    generator = torch.Generator().manual_seed(seed)
    step, epoch, out_epochs, symmetries, orders = 0, 0, [], [], []
    while step < total:
        data = mixed_epoch(cycle, replay, epoch, seed)
        epoch += 1
        source, rows = data['source'].long(), data['value'].long()
        pairs = torch.stack([source, rows], 1)
        picked = rows[source == 1].sort().values
        out_epochs.append({'replay_pick_sha256': _tensor_sha(picked), 'order_sha256': _tensor_sha(pairs)})
        orders.append(pairs)
        for start in range(0, n, batch_size):
            if step >= total:
                break
            size = min(n, start + batch_size) - start
            if augment_d4:
                symmetries.append(torch.randint(0, 8, (size,), generator=generator))
            step += 1
    sym = torch.cat(symmetries) if symmetries else torch.zeros(0, dtype=torch.long)
    out = {'version': ASSIGNMENT_VERSION, 'n_cycle': n_cycle, 'n_replay': n_replay, 'seed': seed,
           'batch_size': batch_size, 'total_steps': total, 'epochs': out_epochs,
           'augmentation_sha256': _tensor_sha(sym) if augment_d4 else None}
    out['sha256'] = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    if detail:
        out['detail'] = {'orders': orders, 'symmetries': symmetries}
    return out


def compare_assignments(a: dict, b: dict) -> dict:
    """Which parts of two assignments agree: replay picks and orders per epoch (common epochs), the augmentation
    sequence over the common prefix of steps; ``identical`` only when every part and the sizes agree."""
    common = min(len(a['epochs']), len(b['epochs']))
    picks = [a['epochs'][i]['replay_pick_sha256'] == b['epochs'][i]['replay_pick_sha256'] for i in range(common)]
    orders = [a['epochs'][i]['order_sha256'] == b['epochs'][i]['order_sha256'] for i in range(common)]
    sizes = all(a[k] == b[k] for k in ('n_cycle', 'n_replay', 'batch_size', 'total_steps'))
    out = {'sizes_equal': sizes, 'common_epochs': common, 'replay_pick_equal': picks, 'order_equal': orders,
           'augmentation_equal': a['augmentation_sha256'] == b['augmentation_sha256']}
    out['identical'] = bool(sizes and all(picks) and all(orders) and out['augmentation_equal']
                            and len(a['epochs']) == len(b['epochs']))
    return out


def _tensor_sha(t) -> str:
    from .h3_cache import tensor_bytes
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(f'{t.dtype}|{tuple(t.shape)}'.encode() + tensor_bytes(t)).hexdigest()


# --- value models -------------------------------------------------------------------------------------------------

def load_value_model(checkpoint, metadata, device: str):
    from model.checkpoint import load_checkpoint
    from model.config import ModelConfig

    meta = json.loads(Path(metadata).read_text(encoding='utf-8'))
    require(isinstance(meta.get('model_config'), dict), f'{metadata}: no model_config')
    return load_checkpoint(checkpoint, ModelConfig(**meta['model_config']), device=device), meta


def check_model(name: str, spec: dict, checkpoint, metadata, frozen_expected: str, device: str):
    """File hashes = the contract's; the frozen state recomputed from the loaded checkpoint = the contract's and the
    metadata's declaration. The loaded, frozen model and its identity record."""
    from .h6_value_train import freeze, frozen_state, state_sha256

    files = {'value_checkpoint_sha256': file_sha256(checkpoint), 'value_metadata_sha256': file_sha256(metadata)}
    for key, value in files.items():
        require(value == spec[key], f'model {name}: {key} {value[:12]} != contract {spec[key][:12]}')
    model, meta = load_value_model(checkpoint, metadata, device)
    freeze(model)
    model.value_head.eval()
    recomputed = state_sha256(frozen_state(model))
    declared = meta.get('frozen_state_sha256')
    require(declared == frozen_expected, f'model {name}: metadata frozen_state_sha256 {str(declared)[:12]} != '
                                         f'contract {frozen_expected[:12]}')
    require(recomputed == frozen_expected, f'model {name}: the loaded checkpoint\'s frozen state {recomputed[:12]} '
                                           f'!= contract {frozen_expected[:12]}')
    record = {**files, 'policy_checkpoint_sha256': spec['policy_checkpoint_sha256'],
              'policy_metadata_sha256': spec['policy_metadata_sha256'], 'policy_loaded': False,
              'frozen_state_sha256': {'recomputed': recomputed, 'declared': declared, 'contract': frozen_expected},
              'value_head_sha256': state_sha256({k: v for k, v in model.state_dict().items()
                                                 if k.startswith('value_head.')})}
    return model, record


def predict(model, data: dict, device: str, batch: int = 2048) -> dict:
    """Raw and D4-averaged values of every cache row (float64 lists); refuses non-finite values."""
    import torch

    from .h6_gate import d4_average
    from .h6_value_cache import planes_from_value_cache
    from .h6_value_train import value_forward

    average = d4_average(lambda planes: value_forward(model, planes), device)
    out = {'raw': [], 'd4': []}
    n = data['value'].shape[0]
    with torch.no_grad():
        for start in range(0, n, batch):
            index = torch.arange(start, min(n, start + batch))
            planes, _, _ = planes_from_value_cache(data, index, device)
            out['raw'] += value_forward(model, planes).double().cpu().tolist()
            out['d4'] += average(planes).double().cpu().tolist()
    for key, values in out.items():
        require(all(math.isfinite(v) for v in values), f'non-finite {key} prediction')
    return out
