"""R2-D3 preparation (outcome-contribution ablation): H-dev composition and the additive decomposition of the R2
primary metric. Read-only arithmetic on existing predictions; no training, ruling or threshold.

The R2 primary metric is a game macro (``h7r_r2.point_metrics``): per game g, ``L_g = num_g / den_g`` over the game's
network rows, then the plain mean over the games with ``den_g > 0``. Because ``num_g`` is a sum over rows, a partition
P of the network rows (label kind, kind x ply bin, ...) splits every game's loss exactly:

    L_g = sum_p num_g^p / den_g          (den_g: the game's ORIGINAL denominator over all its network rows)

and so a paired difference A - B on the same rows (same weights, so the same den_g) splits as

    macro(A) - macro(B) = sum_p mean_g [ (num_g^p(A) - num_g^p(B)) / den_g ]

Each term is the part's additive contribution. It is not the part's own macro (``network/outcome`` divides by the
part's weights only and averages over the games that have that part): the two answer different questions and are
reported separately. The same identity holds inside every bootstrap replicate (common game multiplicities and common
denominators), so the part draws add up to the total draw replicate by replicate.
"""
from __future__ import annotations

from .h7r_r2 import KIND_NAMES, bootstrap, game_sums, ply_bin, point_metrics, require


def kind_label(kind: int, label_kind: str) -> str:
    name = KIND_NAMES[kind]
    require(name == 'proof' if label_kind.startswith('proof_') else name == 'outcome', 'label kind mismatch')
    return 'outcome' if name == 'outcome' else f"proof:{label_kind[len('proof_'):]}"


def composition(route: list[str], kind: list[int], ply: list[int], label_kind: list[str], weight: list[float],
                game: list[int], bins: list[int]) -> dict:
    """Rows, games and weight per ply bin x label kind x route, with the margins; every margin adds up to the rows."""
    cells = {}
    for r, k, p, lk, w, g in zip(route, kind, ply, label_kind, weight, game):
        require(r in ('network', 'exact_leaf'), f'unknown route {r}')
        key = (ply_bin(p, bins), kind_label(k, lk), r)
        c = cells.setdefault(key, {'rows': 0, 'weight': 0.0, 'games': set()})
        c['rows'] += 1
        c['weight'] += float(w)
        c['games'].add(g)

    def margin(index):
        out = {}
        for key, c in cells.items():
            m = out.setdefault(key[index] if isinstance(index, int) else tuple(key[i] for i in index),
                               {'rows': 0, 'weight': 0.0, 'games': set()})
            m['rows'] += c['rows']
            m['weight'] += c['weight']
            m['games'] |= c['games']
        return out

    def public(d, names):
        return [{**dict(zip(names, k if isinstance(k, tuple) else (k,))), 'rows': v['rows'], 'weight': v['weight'],
                 'games': len(v['games'])} for k, v in sorted(d.items())]

    out = {'cells': public(cells, ('ply', 'kind', 'route')), 'kind_route': public(margin((1, 2)), ('kind', 'route')),
           'ply_route': public(margin((0, 2)), ('ply', 'route')), 'route': public(margin(2), ('route',))}
    out['rows'] = len(route)
    for key in ('kind_route', 'ply_route', 'route', 'cells'):
        require(sum(c['rows'] for c in out[key]) == len(route), f'composition {key} does not add up')
    return out


def check_partition(parts: dict[str, list[int]], rows: list[int]) -> None:
    seen = [i for p in parts.values() for i in p]
    require(len(seen) == len(set(seen)), 'the parts overlap')
    require(set(seen) == set(rows), 'the parts are not a partition of the rows')


def denominators(weight, game, rows: list[int], n_games: int):
    import torch
    idx = torch.as_tensor(rows, dtype=torch.long)
    den = torch.zeros(n_games, dtype=torch.float64)
    den.index_add_(0, torch.as_tensor(game, dtype=torch.long)[idx], torch.as_tensor(weight, dtype=torch.float64)[idx])
    return den


def macro_shares(weight, game, rows: list[int], parts: dict[str, list[int]], n_games: int) -> dict:
    """mean_g den_g^p / den_g over the games with rows: the part's share of the macro's per-game weight."""
    check_partition(parts, rows)
    den = denominators(weight, game, rows, n_games)
    present = den > 0
    out = {}
    for name, part in parts.items():
        d = denominators(weight, game, part, n_games)
        out[name] = float((d[present] / den[present]).mean()) if bool(present.any()) else None
    return out


def part_sums(pred_a, pred_b, target, weight, game, part: list[int], den, n_games: int) -> dict:
    """Sums in the ``h7r_r2`` layout whose macro is the part's additive contribution: the per-game error difference
    of the part over the game's original denominator."""
    a = game_sums(pred_a, target, weight, game, part, n_games)['num']
    b = game_sums(pred_b, target, weight, game, part, n_games)['num']
    return {'num': a - b, 'den': den}


def decompose(pred_a, pred_b, target, weight, game, rows: list[int], parts: dict[str, list[int]], n_games: int,
              counts=None, level: float = 0.95, tolerance: float = 1e-12) -> dict:
    """The macro difference A - B on ``rows`` and its additive contributions by part (they add up exactly)."""
    import torch
    check_partition(parts, rows)
    den = denominators(weight, game, rows, n_games)
    present = den > 0
    total = (point_metrics(game_sums(pred_a, target, weight, game, rows, n_games))['macro']
             - point_metrics(game_sums(pred_b, target, weight, game, rows, n_games))['macro'])
    out = {'total': total, 'games': int(present.sum()), 'parts': {}}
    added = 0.0
    for name, part in parts.items():
        sums = part_sums(pred_a, pred_b, target, weight, game, part, den, n_games)
        value = float(torch.where(present, sums['num'] / den.clamp_min(1e-300), torch.zeros_like(den))[present].mean())
        added += value
        cell = {'contribution': value, 'rows': len(part),
                'games_with_rows': int((denominators(weight, game, part, n_games) > 0).sum())}
        if counts is not None:
            cell['ci95'] = bootstrap(counts, sums, None, 'macro', level)
        out['parts'][name] = cell
    require(abs(added - total) <= tolerance * max(1.0, abs(total)), f'the contributions add up to {added}, not {total}')
    out['sum_of_parts'] = added
    return out
