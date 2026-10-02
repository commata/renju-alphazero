"""H3 position cache: H2 games -> bit-packed policy states (docs/mcts-v8-teacher.md §12.11).

One record per policy state (ply >= ``POLICY_FROM_PLY``, not masked for val/test):
black stones, white stones and the legal mask as 225-bit fields (29 bytes each,
little-endian over row-major actions), the last move, the target action, the ply
and the game id. The side to move follows from the ply (even = black). About 95
bytes per state; the legal mask is computed once here with ``Game`` so training
never runs the rules engine.

The manifest records the H2 output SHA-256, the encoder / action-index versions
and a hash of the rule sources. ``load_cache`` refuses a cache whose recorded
identity differs from the current code, so a rules or encoder change can never
silently reuse stale masks. Like the H2 games, the cache is RenjuNet-derived and
must not be committed.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import ctypes
import gzip
import hashlib
import json
from pathlib import Path

import warnings

# The CPU desktop venv has no NumPy, which this module never needs; torch warns at import in every worker.
warnings.filterwarnings('ignore', message='Failed to initialize NumPy')
import torch  # noqa: E402

from model.config import ACTION_COUNT, ACTION_INDEX_VERSION, BOARD_SIZE, ENCODER_VERSION
from renju import Game

from .renjunet import POLICY_FROM_PLY

CACHE_FORMAT = 'h3-cache-v1'
FIELD_BYTES = (ACTION_COUNT + 7) // 8  # 29
SPLITS = ('train', 'val', 'test')
RULE_SOURCES = ('src/renju/rules.py', 'src/renju/game.py')
ROOT = Path(__file__).resolve().parents[2]


def rules_identity(root: Path = ROOT) -> str:
    """SHA-256 over the rule sources (legal masks depend on them)."""
    digest = hashlib.sha256()
    for name in RULE_SOURCES:
        digest.update(name.encode())
        digest.update((root / name).read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


def code_identity() -> dict:
    return {'encoder_version': ENCODER_VERSION, 'action_index_version': ACTION_INDEX_VERSION,
            'rules_sha256': rules_identity(), 'policy_from_ply': POLICY_FROM_PLY}


def pack(actions) -> bytes:
    bits = 0
    for action in actions:
        bits |= 1 << action
    return bits.to_bytes(FIELD_BYTES, 'little')


_SHIFTS = torch.arange(8, dtype=torch.uint8)


def unpack(fields: torch.Tensor) -> torch.Tensor:
    """[N, 29] uint8 -> [N, 225] bool."""
    bits = (fields.unsqueeze(-1) >> _SHIFTS.to(fields.device)) & 1
    return bits.reshape(fields.shape[0], -1)[:, :ACTION_COUNT].bool()


def game_records(game: dict) -> list[tuple]:
    """Policy states of one H2 game: (black, white, legal, last, target, ply, id)."""
    masked = set(game['masked_plies'])
    board = Game()
    black, white, out = [], [], []
    for ply, (row, col) in enumerate(game['moves']):
        action = row * BOARD_SIZE + col
        if ply >= POLICY_FROM_PLY and ply not in masked:
            legal = [r * BOARD_SIZE + c for r, c in board.legal_moves()]
            if action not in legal:
                raise ValueError(f"game {game['id']} ply {ply}: target is not legal")
            last = board.history[-1][0] * BOARD_SIZE + board.history[-1][1]
            out.append((pack(black), pack(white), pack(legal), last, action, ply, game['id']))
        board.play(row, col)
        (black if ply % 2 == 0 else white).append(action)
    return out


def _chunk_records(games):
    return [record for game in games for record in game_records(game)]


def read_games(path: Path) -> tuple[list[dict], str]:
    """H2 ``games.jsonl.gz`` and the SHA-256 of its decompressed content (= H2 ``output_sha256``)."""
    payload = gzip.decompress(path.read_bytes())
    games = [json.loads(line) for line in payload.decode('utf-8').splitlines() if line]
    return games, hashlib.sha256(payload).hexdigest()


def build_split(games: list[dict], workers: int = 1, chunk: int = 500) -> dict[str, torch.Tensor]:
    chunks = [games[i:i + chunk] for i in range(0, len(games), chunk)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_chunk_records, chunks))
    else:
        parts = [_chunk_records(c) for c in chunks]
    records = [r for part in parts for r in part]
    n = len(records)

    def field(index):
        return torch.frombuffer(bytearray(b''.join(r[index] for r in records)), dtype=torch.uint8).reshape(n, FIELD_BYTES) \
            if n else torch.zeros((0, FIELD_BYTES), dtype=torch.uint8)

    return {'black': field(0), 'white': field(1), 'legal': field(2),
            'last': torch.tensor([r[3] for r in records], dtype=torch.int16),
            'target': torch.tensor([r[4] for r in records], dtype=torch.int16),
            'ply': torch.tensor([r[5] for r in records], dtype=torch.int16),
            'game_id': torch.tensor([r[6] for r in records], dtype=torch.int32)}


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Raw bytes of a CPU tensor without NumPy (the desktop CPU venv has none)."""
    tensor = tensor.contiguous().cpu()
    return ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size()) if tensor.numel() else b''


def tensor_sha256(data: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(data):
        digest.update(key.encode())
        digest.update(tensor_bytes(data[key]))
    return digest.hexdigest()


def build_cache(games_path: Path, out_dir: Path, *, workers: int = 1, git_commit: str | None = None,
                h2_manifest: dict | None = None) -> dict:
    games, h2_sha = read_games(games_path)
    if h2_manifest is not None and h2_manifest.get('output_sha256') != h2_sha:
        raise ValueError('games.jsonl.gz does not match the H2 manifest output_sha256')
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'format': CACHE_FORMAT, 'h2_output_sha256': h2_sha, **code_identity(),
                'git_commit': git_commit, 'splits': {}}
    for split in SPLITS:
        data = build_split([g for g in games if g['split'] == split], workers)
        torch.save(data, out_dir / f'{split}.pt')
        manifest['splits'][split] = {'states': int(data['target'].shape[0]),
                                     'games': len({int(x) for x in data['game_id'].tolist()}),
                                     'sha256': tensor_sha256(data)}
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    return manifest


def load_cache(cache_dir: Path, split: str, *, verify_hash: bool = False) -> tuple[dict[str, torch.Tensor], dict]:
    """Load one split; refuse a cache built with other rules / encoder / format."""
    manifest = json.loads((cache_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != CACHE_FORMAT:
        raise ValueError(f"cache format {manifest.get('format')} != {CACHE_FORMAT}")
    current = code_identity()
    stale = {k: (manifest.get(k), v) for k, v in current.items() if manifest.get(k) != v}
    if stale:
        raise ValueError(f'cache was built with different code, rebuild it: {stale}')
    data = torch.load(cache_dir / f'{split}.pt', weights_only=True)
    if verify_hash and tensor_sha256(data) != manifest['splits'][split]['sha256']:
        raise ValueError(f'{split}.pt does not match the manifest hash')
    return data, manifest


def to_play_is_black(ply: torch.Tensor) -> torch.Tensor:
    return ply.remainder(2) == 0


def planes_from_cache(data: dict[str, torch.Tensor], index: torch.Tensor,
                      device: str | torch.device = 'cpu') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """States [B,6,15,15] (same planes as ``model.encoding.encode_game``), legal [B,225], target [B]."""
    black = unpack(data['black'][index]).to(device)
    white = unpack(data['white'][index]).to(device)
    legal = unpack(data['legal'][index]).to(device)
    is_black = to_play_is_black(data['ply'][index]).to(device)
    last = data['last'][index].long().to(device)
    b = index.shape[0]
    current = torch.where(is_black[:, None], black, white)
    opponent = torch.where(is_black[:, None], white, black)
    planes = torch.zeros((b, 6, ACTION_COUNT), dtype=torch.float32, device=device)
    planes[:, 0] = current
    planes[:, 1] = opponent
    planes[torch.arange(b, device=device), 2, last] = 1.0
    planes[:, 3] = is_black[:, None].float()
    planes[:, 4] = 1.0
    planes[:, 5] = legal
    return planes.reshape(b, 6, BOARD_SIZE, BOARD_SIZE), legal, data['target'][index].long().to(device)


def verify_encoding(data: dict[str, torch.Tensor], games: list[dict], indices) -> int:
    """Cached planes vs ``model.encoding.encode_game`` on the replayed game; returns mismatches."""
    from model.encoding import encode_game
    by_id = {g['id']: g for g in games}
    bad = 0
    for i in indices:
        index = torch.tensor([int(i)])
        planes, legal, target = planes_from_cache(data, index)
        game_record = by_id[int(data['game_id'][i])]
        ply = int(data['ply'][i])
        game = Game()
        for move in game_record['moves'][:ply]:
            game.play(*move)
        expected = encode_game(game)
        row, col = game_record['moves'][ply]
        if not torch.equal(planes[0], expected) or int(target[0]) != row * BOARD_SIZE + col:
            bad += 1
    return bad


def verify_d4(data: dict[str, torch.Tensor], indices) -> int:
    """For every symmetry: the transformed cached legal mask equals ``Game`` legality on the
    transformed board (forbidden-move rules must be D4-invariant). Returns mismatches."""
    from model.symmetry import transform_coordinate
    bad = 0
    for i in indices:
        index = torch.tensor([int(i)])
        black = unpack(data['black'][index])[0].nonzero().flatten().tolist()
        white = unpack(data['white'][index])[0].nonzero().flatten().tolist()
        legal = set(unpack(data['legal'][index])[0].nonzero().flatten().tolist())
        last = int(data['last'][i])
        is_black = bool(to_play_is_black(data['ply'][index])[0])
        for s in range(8):
            def t(a):
                r, c = transform_coordinate(a // BOARD_SIZE, a % BOARD_SIZE, s)
                return r * BOARD_SIZE + c
            game = Game()
            for a in black:
                r, c = divmod(t(a), BOARD_SIZE)
                game.board[r][c] = 1
            for a in white:
                r, c = divmod(t(a), BOARD_SIZE)
                game.board[r][c] = -1
            game.to_play = 1 if is_black else -1
            game.history = [divmod(t(last), BOARD_SIZE)]
            recomputed = {r * BOARD_SIZE + c for r, c in game.legal_moves()}
            if recomputed != {t(a) for a in legal}:
                bad += 1
    return bad
