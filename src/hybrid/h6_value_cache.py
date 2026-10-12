"""H6-2a value cache: H6-1 label rows -> bit-packed value states (docs/h6-value.md, H6-2a).

One record per **labelled** row (``value_mask`` False); masked rows are counted and dropped here, so
the trainer cannot see them. Same bitfields as the H3 cache (``hybrid.h3_cache``: black, white and
legal as 225-bit fields, the last move, the ply), plus the value target (side to move, as written by
H6-1: no sign change anywhere), the row weight, the label kind (0 outcome, 1 proof) and a game index.
Rows keep the split H6-1 wrote; the legal mask is computed once with ``Game``.

The manifest records the SHA-256 of the decompressed label rows (and of the H6-1 report when given,
whose ``labelled`` count must match), the encoder / action-index versions and the rule sources;
``load_value_cache`` refuses a cache built with other code. RenjuNet-derived: never committed.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path

import torch

from model.config import BOARD_SIZE
from renju import Game

from .h3_cache import FIELD_BYTES, code_identity, pack, tensor_sha256, unpack
from .h6_labels import PERSPECTIVE, ROW_FORMAT
from .progress import Progress

CACHE_FORMAT = 'h6-value-cache-v1'
SPLITS = ('train', 'val', 'test')
KIND_OUTCOME, KIND_PROOF = 0, 1


def normalized_sha256(path: Path) -> str:
    """SHA-256 of a text file with CRLF read as LF (reports are committed from Windows)."""
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def iter_games(path: Path, digest=None):
    """Label rows grouped by game, in file order: ``(game_id, rows)``. A game's rows must be contiguous."""
    opener = gzip.open if path.suffix == '.gz' else open
    seen, current, rows = set(), None, []
    with opener(path, 'rb') as handle:
        for raw in handle:
            if digest is not None:
                digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get('format') != ROW_FORMAT:
                raise ValueError(f"row format {row.get('format')!r} != {ROW_FORMAT}")
            if row.get('value_perspective') != PERSPECTIVE:
                raise ValueError(f"row perspective {row.get('value_perspective')!r} != {PERSPECTIVE}")
            if row['game_id'] != current:
                if current is not None:
                    yield current, rows
                if row['game_id'] in seen:
                    raise ValueError(f"rows of game {row['game_id']} are not contiguous")
                seen.add(row['game_id'])
                current, rows = row['game_id'], []
            rows.append(row)
    if current is not None:
        yield current, rows


def compact_game(rows: list[dict]) -> tuple:
    """Check one game's rows; ``(game_id, split, moves, labelled, masked)`` with labelled =
    sorted ``(ply, value, weight, kind)`` of the unmasked rows (masked rows are only counted)."""
    split = {r['split'] for r in rows}
    if len(split) != 1:
        raise ValueError(f"game {rows[0]['game_id']} spans splits {sorted(split)}")
    kept = sorted((r for r in rows if not r['value_mask']), key=lambda r: r['ply'])
    if len({r['ply'] for r in kept}) != len(kept):
        raise ValueError(f"game {rows[0]['game_id']}: two labelled rows at one ply")
    moves = kept[-1]['moves'] if kept else []
    labelled = []
    for r in kept:
        if r['value_target'] not in (-1, 0, 1):
            raise ValueError(f"game {r['game_id']} ply {r['ply']}: value_target {r['value_target']!r}")
        if not r['weight'] > 0:
            raise ValueError(f"game {r['game_id']} ply {r['ply']}: unmasked row with weight {r['weight']}")
        if len(r['moves']) != r['ply'] or r['moves'] != moves[:r['ply']]:
            raise ValueError(f"game {r['game_id']} ply {r['ply']}: moves are not a prefix of the game")
        kind = KIND_PROOF if r['label_kind'].startswith('proof_') else KIND_OUTCOME
        labelled.append((r['ply'], r['value_target'], float(r['weight']), kind))
    return rows[0]['game_id'], split.pop(), moves, labelled, len(rows) - len(kept)


def game_records(moves: list, labelled: list[tuple]) -> list[tuple]:
    """Replay one game: (black, white, legal, last, ply, value, weight, kind) per labelled ply."""
    wanted = {ply: rest for ply, *rest in labelled}
    board = Game()
    black, white, out = [], [], []
    for ply in range(len(moves) + 1):
        if ply in wanted:
            if board.done:
                raise ValueError(f'ply {ply}: the position is terminal')
            legal = [a * BOARD_SIZE + b for a, b in board.legal_moves()]
            last = board.history[-1][0] * BOARD_SIZE + board.history[-1][1] if board.history else -1
            out.append((pack(black), pack(white), pack(legal), last, ply, *wanted[ply]))
        if ply < len(moves):
            row, col = moves[ply]
            (black if ply % 2 == 0 else white).append(row * BOARD_SIZE + col)
            board.play(row, col)
    return out


def _chunk(games):
    return [(game_id, split, game_records(moves, labelled), masked)
            for game_id, split, moves, labelled, masked in games]


def _chunks(games, size):
    chunk = []
    for game_id, rows in games:
        chunk.append(compact_game(rows))
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _bounded_map(pool, fn, items, window):
    """``pool.map`` that keeps at most ``window`` chunks in flight (the label file is large)."""
    pending = []
    for item in items:
        pending.append(pool.submit(fn, item))
        if len(pending) >= window:
            yield pending.pop(0).result()
    for future in pending:
        yield future.result()


def _tensors(records: list[tuple], game_index: list[int]) -> dict[str, torch.Tensor]:
    n = len(records)

    def field(i):
        if not n:
            return torch.zeros((0, FIELD_BYTES), dtype=torch.uint8)
        return torch.frombuffer(bytearray(b''.join(r[i] for r in records)), dtype=torch.uint8).reshape(n, FIELD_BYTES)

    return {'black': field(0), 'white': field(1), 'legal': field(2),
            'last': torch.tensor([r[3] for r in records], dtype=torch.int16),
            'ply': torch.tensor([r[4] for r in records], dtype=torch.int16),
            'value': torch.tensor([r[5] for r in records], dtype=torch.float32),
            'weight': torch.tensor([r[6] for r in records], dtype=torch.float32),
            'kind': torch.tensor([r[7] for r in records], dtype=torch.int8),
            'game': torch.tensor(game_index, dtype=torch.int32)}


def split_summary(data: dict[str, torch.Tensor]) -> dict:
    kinds = {'outcome': KIND_OUTCOME, 'proof': KIND_PROOF}
    out = {'states': int(data['value'].shape[0]), 'games': len(set(data['game'].tolist())),
           'weight_sum': float(data['weight'].double().sum()), 'by_kind': {}}
    for name, code in kinds.items():
        inside = data['kind'] == code
        w = data['weight'][inside].double()
        out['by_kind'][name] = {'states': int(inside.sum()), 'weight_sum': float(w.sum()),
                                'weight_share': float(w.sum()) / out['weight_sum'] if out['weight_sum'] else None,
                                'sign': {s: int((data['value'][inside] == v).sum()) for s, v in
                                         (('+1', 1.0), ('-1', -1.0), ('0', 0.0))}}
    return out


def build_value_cache(labels: Path, out_dir: Path, *, report: Path | None = None, workers: int = 1,
                      chunk: int = 500, git_commit: str | None = None, progress_seconds: float = 60.0) -> dict:
    digest = hashlib.sha256()
    records = {s: [] for s in SPLITS}
    games = {s: [] for s in SPLITS}
    masked = {s: 0 for s in SPLITS}
    batches = _chunks(iter_games(labels, digest), chunk)
    expected = json.loads(report.read_text(encoding='utf-8'))['summary']['labelled'] if report is not None else None
    progress = Progress('value cache', expected, 'labelled positions', every_seconds=progress_seconds)
    if workers > 1:
        pool = ProcessPoolExecutor(max_workers=workers)
        parts = _bounded_map(pool, _chunk, batches, 4 * workers)
    else:
        pool, parts = None, map(_chunk, batches)
    try:
        for part in parts:
            for game_id, split, recs, n_masked in part:
                if split not in records:
                    raise ValueError(f'game {game_id}: unknown split {split!r}')
                masked[split] += n_masked
                index = len(games[split])
                games[split].append(game_id)
                records[split].extend((r, index) for r in recs)
                progress.update(len(recs))
    finally:
        if pool is not None:
            pool.shutdown()
    manifest = {'format': CACHE_FORMAT, 'labels_sha256': digest.hexdigest(), 'labels_file': str(labels),
                'h6_report_sha256': None, **code_identity(), 'git_commit': git_commit, 'splits': {}}
    labelled = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        data = _tensors([r for r, _ in records[split]], [i for _, i in records[split]])
        torch.save(data, out_dir / f'{split}.pt')
        (out_dir / f'{split}_games.json').write_text(json.dumps(games[split]), encoding='utf-8')
        manifest['splits'][split] = {**split_summary(data), 'masked_rows_dropped': masked[split],
                                     'sha256': tensor_sha256(data)}
        labelled += int(data['value'].shape[0])
    if report is not None:
        payload = json.loads(report.read_text(encoding='utf-8'))
        if payload.get('row_format') != ROW_FORMAT:
            raise ValueError(f"{report} is not an H6-1 report for {ROW_FORMAT}")
        if payload['summary']['labelled'] != labelled:
            raise ValueError(f"{report} counts {payload['summary']['labelled']} labelled rows, the labels have {labelled}")
        manifest['h6_report_sha256'] = normalized_sha256(report)
        manifest['h6_report_restricted'] = bool(payload.get('restricted'))
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    return manifest


def load_value_cache(cache_dir: Path, split: str, *, verify_hash: bool = False) -> tuple[dict, dict]:
    """Load one split; refuse a cache built with other rules / encoder / format."""
    manifest = json.loads((cache_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != CACHE_FORMAT:
        raise ValueError(f"cache format {manifest.get('format')} != {CACHE_FORMAT}")
    stale = {k: (manifest.get(k), v) for k, v in code_identity().items() if manifest.get(k) != v}
    if stale:
        raise ValueError(f'value cache was built with different code, rebuild it: {stale}')
    data = torch.load(cache_dir / f'{split}.pt', weights_only=True)
    if verify_hash and tensor_sha256(data) != manifest['splits'][split]['sha256']:
        raise ValueError(f'{split}.pt does not match the manifest hash')
    return data, manifest


def planes_from_value_cache(data: dict[str, torch.Tensor], index: torch.Tensor, device='cpu'):
    """States [B,6,15,15] (the planes of ``model.encoding.encode_game``), value [B], weight [B]."""
    black = unpack(data['black'][index]).to(device)
    white = unpack(data['white'][index]).to(device)
    legal = unpack(data['legal'][index]).to(device)
    is_black = (data['ply'][index].remainder(2) == 0).to(device)
    last = data['last'][index].long().to(device)
    b = index.shape[0]
    planes = torch.zeros((b, 6, BOARD_SIZE * BOARD_SIZE), dtype=torch.float32, device=device)
    planes[:, 0] = torch.where(is_black[:, None], black, white)
    planes[:, 1] = torch.where(is_black[:, None], white, black)
    has_last = last >= 0
    rows = torch.arange(b, device=device)[has_last]
    planes[rows, 2, last[has_last]] = 1.0
    planes[:, 3] = is_black[:, None].float()
    planes[:, 4] = 1.0
    planes[:, 5] = legal
    return (planes.reshape(b, 6, BOARD_SIZE, BOARD_SIZE), data['value'][index].to(device),
            data['weight'][index].to(device))
