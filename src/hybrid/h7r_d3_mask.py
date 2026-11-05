"""R2-D3-NO-OUTCOME-1: the cycle outcome mask. One ``keep`` bool per cycle cache row and split: True for a proof label
row (cache kind 1, label kind ``proof_*``), False for an outcome label row (kind 0, label kind ``outcome``). Replay
rows are never masked (the trainer gives them k = 1). The loss keeps its original batch denominator
(``h7c_train.masked_wmse``).

Every cache row is linked to its label row as ``h7c_nmask.build_mask`` links them (the label rows grouped and replayed
as the cache builder does; game, ply, value, weight, kind, stones and legal mask equal), and the kind is checked
against the label row's own ``label_kind`` string, so a mislabelled cache row stops the build instead of being kept
or removed silently. The manifest records per split the rows and weights removed and kept and by label kind, the
label and cache SHA-256 and the keep tensors' SHA-256.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MASK_FORMAT = 'h7r-d3-outcome-mask-v1'
DEFINITION = ('keep = the cycle row is a proof label (cache kind 1, label kind proof_*); outcome rows (kind 0, label '
              'kind outcome) are removed; replay rows are never masked')


def require(ok, message) -> None:
    if not ok:
        raise ValueError(message)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_kind_of(kind: int, label_kind: str) -> str:
    """The label kind string of a cache kind, refusing a mismatch (0 = outcome, 1 = proof_*)."""
    from .h6_value_cache import KIND_OUTCOME, KIND_PROOF

    require(kind in (KIND_OUTCOME, KIND_PROOF), f'unknown cache kind {kind}')
    require((kind == KIND_PROOF) == label_kind.startswith('proof_') and
            (kind == KIND_OUTCOME) == (label_kind == 'outcome'), f'cache kind {kind} is not label kind {label_kind}')
    return label_kind


def build_outcome_mask(value_rows: Path, cache_dir: Path, progress=None) -> tuple[dict, dict]:
    """(keep by split, manifest body) of the cycle cache in ``cache_dir`` and its label rows ``value_rows``."""
    import torch

    from .h3_cache import tensor_sha256
    from .h6_value_cache import KIND_PROOF, SPLITS, compact_game, game_records, iter_games, load_value_cache

    digest = hashlib.sha256()
    rows = {s: [] for s in SPLITS}
    games = {s: [] for s in SPLITS}
    for game_id, label_rows in iter_games(Path(value_rows), digest):
        _, split, moves, labelled, _ = compact_game(label_rows)
        detail = {r['ply']: r['label_kind'] for r in label_rows if not r['value_mask']}
        index = len(games[split])
        games[split].append(game_id)
        for (ply, value, weight, kind), packed in zip(labelled, game_records(moves, labelled)):
            rows[split].append({'game': index, 'ply': ply, 'value': value, 'weight': weight, 'kind': kind,
                                'packed': packed[:3], 'label_kind': label_kind_of(kind, detail[ply])})
        if progress is not None:
            progress.update(1)
    manifest = json.loads((Path(cache_dir) / 'manifest.json').read_text(encoding='utf-8'))
    require(digest.hexdigest() == manifest['labels_sha256'],
            f'{value_rows} is not the label file of {cache_dir} ({digest.hexdigest()[:12]})')
    keep, splits = {}, {}
    for split in SPLITS:
        data, _ = load_value_cache(Path(cache_dir), split, verify_hash=True)
        listed = json.loads((Path(cache_dir) / f'{split}_games.json').read_text(encoding='utf-8'))
        require(listed == games[split], f'{split}: the cache game list is not the label rows\' game order')
        mine = rows[split]
        require(data['value'].shape[0] == len(mine), f'{split}: {data["value"].shape[0]} cache rows, '
                                                     f'{len(mine)} labelled rows')
        for i, r in enumerate(mine):
            same = (int(data['game'][i]) == r['game'] and int(data['ply'][i]) == r['ply']
                    and float(data['value'][i]) == float(r['value'])
                    and float(data['weight'][i]) == float(torch.tensor(r['weight'], dtype=torch.float32))
                    and int(data['kind'][i]) == r['kind']
                    and all(bytes(data[f][i].tolist()) == r['packed'][j]
                            for j, f in enumerate(('black', 'white', 'legal'))))
            require(same, f'{split} row {i}: the cache row is not the label row {games[split][r["game"]]} '
                          f'ply {r["ply"]}')
        keep[split] = data['kind'] == KIND_PROOF
        k, w = keep[split], data['weight'].double()
        by_kind = {}
        for r, weight in zip(mine, w.tolist()):
            cell = by_kind.setdefault(r['label_kind'], {'rows': 0, 'weight': 0.0})
            cell['rows'] += 1
            cell['weight'] += weight
        removed = {lk: c for lk, c in by_kind.items() if lk == 'outcome'}
        require(int((~k).sum()) == sum(c['rows'] for c in removed.values()), f'{split}: removed rows are not outcome')
        splits[split] = {'rows': len(mine), 'removed_rows': int((~k).sum()), 'kept_rows': int(k.sum()),
                         'weight': float(w.sum()), 'removed_weight': float(w[~k].sum()),
                         'kept_weight': float(w[k].sum()),
                         'by_label_kind': {lk: {'rows': c['rows'], 'weight': c['weight'],
                                                'kept': lk != 'outcome'} for lk, c in sorted(by_kind.items())},
                         'cache_sha256': manifest['splits'][split]['sha256']}
        require(abs(splits[split]['removed_weight'] + splits[split]['kept_weight'] - splits[split]['weight'])
                <= 1e-6 * max(1.0, splits[split]['weight']), f'{split}: removed + kept weight is not the weight')
    body = {'format': MASK_FORMAT, 'definition': DEFINITION, 'labels_sha256': manifest['labels_sha256'],
            'cache_manifest_sha256': file_sha256(Path(cache_dir) / 'manifest.json'),
            'splits': splits, 'keep_sha256': tensor_sha256({s: keep[s].to(torch.uint8) for s in SPLITS}),
            'link': 'each cache row = its label row (game id order, ply, value, weight, kind, stones, legal mask); '
                    'cache kind checked against the label row\'s label_kind'}
    return keep, body


def save_mask(path: Path, keep: dict, body: dict) -> None:
    import torch

    from .h3_cache import tensor_sha256

    require(tensor_sha256({s: keep[s].to(torch.uint8) for s in keep}) == body['keep_sha256'], 'mask hash')
    require(not Path(path).exists(), f'{path} exists: never overwritten')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format': MASK_FORMAT, 'keep': keep}, path)


def load_mask(path: Path, body: dict) -> dict:
    """The keep tensors of ``path``: their SHA-256 = the manifest's, one bool per cache row, the manifest's counts."""
    import torch

    from .h3_cache import tensor_sha256

    data = torch.load(path, weights_only=True)
    require(isinstance(data, dict) and data.get('format') == MASK_FORMAT, f'{path}: not a {MASK_FORMAT} file')
    keep = data['keep']
    require(set(keep) == set(body['splits']) and all(keep[s].dtype == torch.bool for s in keep),
            f'{path}: not one bool tensor per split')
    require(tensor_sha256({s: keep[s].to(torch.uint8) for s in keep}) == body['keep_sha256'],
            f'{path} is not the mask its manifest names')
    for split, info in body['splits'].items():
        require(keep[split].shape[0] == info['rows'] and int((~keep[split]).sum()) == info['removed_rows'],
                f'{path}: {split} counts differ from the manifest')
    return keep


def check_against_cache(keep: dict, cycle: dict, body: dict) -> None:
    """The mask on the cache the trainer loaded: row counts, removed rows = the outcome rows, kept weight."""
    from .h6_value_cache import KIND_PROOF

    for split in ('train', 'val'):
        data, k = cycle[split], keep[split]
        require(k.shape == data['kind'].shape, f'{split}: the mask is not one bool per loaded cycle row')
        require(bool((k == (data['kind'] == KIND_PROOF)).all()), f'{split}: the mask is not kind == proof on the cache')
        kept = float(data['weight'].double()[k].sum())
        require(abs(kept - body['splits'][split]['kept_weight']) <= 1e-6 * max(1.0, kept),
                f'{split}: kept weight differs from the manifest')
