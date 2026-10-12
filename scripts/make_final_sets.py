"""Sealed H8 holdout sets Final-1 / Final-2 (track-b-b1-b2.md §8.2, §13; docs/mcts-v8-teacher.md §12.36).

Run once, at the TB-Engine-v1 freeze. The sets themselves go to ``--out-dir`` OUTSIDE the repository
(with the generator secret); only ``--manifest`` (hashes and counts, no positions, no secret) is committed.
Nobody opens the sets before H8: Final-1 at the B1-Champion H8, Final-2 once after B1-Matched and B2.

Contents (counts fixed in track-b-b1-b2.md §13):
    final1_openings          100 openings  (3 fixed opponents x 100 pairs)
    final1_tactical          ~200 positions with a rules / proof answer (quota per kind below)
    final2_direct_openings   200 openings  (B2 final vs B1-Champion-v1, 200 pairs)
    final2_common_openings    50 openings  (each model x 3 opponents x 50 pairs)

Openings: the centre stone and ``OPENING_PLIES`` random plies near the stones (radius 2), one D4 class each,
none D4-equal to an opening of any benchmark result in ``docs/`` nor to another set's opening.
Tactical positions: random near-stone playouts of 8-40 plies, labelled by the rules and VCF proofs
(``analysis.tactical_labels``) or by the black forbidden points; positions D4-equal to a probe or
evaluation position (``hybrid.h6_labels.default_probe_files``) are left out.

    python scripts/make_final_sets.py --out-dir D:\\renju-sealed\\finals_v1 \\
        --manifest docs/mcts-v8-results/finals/finals_manifest.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from random import Random
import secrets
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from renju import BLACK, Game  # noqa: E402
from renju.game import OPENING_MOVE  # noqa: E402
from renju.rules import forbidden_reason  # noqa: E402

FORMAT = 'final-sets-manifest-v1'


def _transform(move, symmetry: int):
    row, col = move
    if symmetry >= 4:
        col = 14 - col
    for _ in range(symmetry % 4):
        row, col = 14 - col, row
    return row, col


def canonical_hash(moves) -> str:
    """Stones + side to move, minimized over D4 (the same rule as hybrid.h6_labels on feat/h6-value)."""
    black, white = [tuple(m) for m in moves[0::2]], [tuple(m) for m in moves[1::2]]
    key = min((len(moves) % 2, tuple(sorted(_transform(m, s) for m in black)),
               tuple(sorted(_transform(m, s) for m in white))) for s in range(8))
    return hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).hexdigest()[:24]


def default_probe_files(root: Path) -> list[str]:
    files = sorted(glob.glob(str(root / 'tests' / 'fixtures' / '*probes*.json')))
    files += sorted(glob.glob(str(root / 'docs' / 'mcts-v8-results' / 'probe_*.json')))
    files += sorted(glob.glob(str(root / 'docs' / 'mcts-v8-results' / 'e1*' / '*manifest.json')))
    return files


def probe_hashes(paths) -> set[str]:
    """Probe / evaluation positions: fixture ``probes`` lists, web-probe dicts and E1 manifests."""
    out = set()
    for path in paths:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        probes = data.get('probes')
        if isinstance(probes, list):
            out.update(canonical_hash(p['moves']) for p in probes if 'moves' in p)
        elif isinstance(probes, dict) and 'moves' in data:
            out.update(canonical_hash(data['moves'][:p['plies_played']]) for p in probes.values())
        for position in data.get('positions', []):
            if 'history' in position:
                out.add(canonical_hash(position['history']))
    return out
OPENING_PLIES = 4
RADIUS = 2
COUNTS = {'final1_openings': 100, 'final2_direct_openings': 200, 'final2_common_openings': 50}
TACTICAL_QUOTA = {'immediate_win': 35, 'must_block': 35, 'forced_loss': 30, 'unstoppable_four': 30,
                  'vcf': 35, 'black_forbidden': 35}
SOLVER_NODE_LIMIT = 20_000


def _near(game: Game, radius: int) -> list:
    stones = [(r, c) for r in range(15) for c in range(15) if game.board[r][c] != 0]
    legal = game.legal_moves()
    return [m for m in legal if any(max(abs(m[0] - r), abs(m[1] - c)) <= radius for r, c in stones)] or legal


def make_opening(rng: Random) -> list:
    game = Game()
    game.play(*OPENING_MOVE)
    for _ in range(OPENING_PLIES):
        game.play(*rng.choice(_near(game, RADIUS)))
        if game.done:
            return []
    return [list(m) for m in game.history]


def known_openings(root: Path) -> set[str]:
    """D4 classes of every opening recorded in a benchmark result under docs/."""
    out = set()
    for path in glob.glob(str(root / 'docs' / '**' / '*.json'), recursive=True):
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get('openings'), list):
            for opening in data['openings']:
                if isinstance(opening, dict) and 'moves' in opening:
                    out.add(canonical_hash(opening['moves']))
    return out


def tactical_label(game: Game, solver) -> dict | None:
    from analysis.tactical_labels import label_position

    if game.done:
        return None
    if game.to_play == BLACK:
        forbidden = sorted((r, c) for r in range(15) for c in range(15)
                           if game.board[r][c] == 0 and forbidden_reason(game.board, r, c) is not None)
        if forbidden:
            return {'kind': 'black_forbidden', 'answer': [list(m) for m in forbidden], 'value': None}
    label = label_position(game, solver)
    if label is None:
        return None
    return {'kind': label['kind'], 'answer': [list(m) for m in label['policy']], 'value': label['value']}


def make_tactical(rng: Random, quota: dict, exclude: set[str], max_attempts: int = 200_000) -> tuple[list, dict]:
    from analysis.threats import ThreatSolver

    solver = ThreatSolver(node_limit=SOLVER_NODE_LIMIT)
    need = dict(quota)
    out, seen, attempts = [], set(), 0
    while any(n > 0 for n in need.values()) and attempts < max_attempts:
        attempts += 1
        game = Game()
        game.play(*OPENING_MOVE)
        for _ in range(rng.randint(8, 40)):
            if game.done:
                break
            game.play(*rng.choice(_near(game, RADIUS)))
        if game.done:
            continue
        moves = [list(m) for m in game.history]
        key = canonical_hash(moves)
        if key in seen or key in exclude:
            continue
        label = tactical_label(game, solver)
        if label is None or need.get(label['kind'], 0) <= 0:
            continue
        seen.add(key)
        need[label['kind']] -= 1
        out.append({'moves': moves, 'side_to_move': 'black' if len(moves) % 2 == 0 else 'white', **label})
    return out, {'attempts': attempts, 'short_of_quota': {k: v for k, v in need.items() if v > 0}}


def build(secret: str, root: Path = ROOT, counts=COUNTS, quota=TACTICAL_QUOTA) -> tuple[dict, dict]:
    rng = Random(hashlib.sha256(f'final-sets-v1:{secret}'.encode()).digest())
    taken = known_openings(root)
    known = len(taken)
    sets = {}
    for name, count in counts.items():
        openings = []
        while len(openings) < count:
            moves = make_opening(rng)
            key = canonical_hash(moves) if moves else None
            if key is None or key in taken:
                continue
            taken.add(key)
            openings.append({'pair': len(openings), 'moves': moves})
        sets[name] = {'kind': 'openings', 'items': openings}
    tactical, info = make_tactical(rng, quota, probe_hashes(default_probe_files(root)))
    sets['final1_tactical'] = {'kind': 'tactical', 'items': tactical, 'generation': info}
    return sets, {'known_benchmark_openings_excluded': known, 'tactical': info}


def _sha(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out-dir', type=Path, required=True, help='OUTSIDE the repository')
    parser.add_argument('--manifest', type=Path, required=True, help='committed: hashes and counts only')
    parser.add_argument('--secret', help='generator secret (default: a fresh 128-bit one, saved in --out-dir)')
    args = parser.parse_args(argv)
    out_dir = args.out_dir.resolve()
    if out_dir.is_relative_to(ROOT):
        parser.error('the sealed sets must live outside the repository')
    if out_dir.exists() and any(out_dir.iterdir()):
        parser.error(f'{out_dir} is not empty; the sets are generated once')
    secret = args.secret or secrets.token_hex(16)
    sets, info = build(secret)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, data in sets.items():
        path = out_dir / f'{name}.json'
        path.write_text(json.dumps(data, indent=1), encoding='utf-8')
        artifacts[name] = {'items': len(data['items']), 'content_sha256': _sha(data['items'])}
    (out_dir / 'SECRET.txt').write_text(secret + '\n', encoding='utf-8')
    probe_files = [str(Path(p).resolve().relative_to(ROOT)) for p in default_probe_files(ROOT)]
    manifest = {
        'format': FORMAT, 'generator_commit': _git_commit(),
        'generator_config': {'opening_plies': OPENING_PLIES, 'radius': RADIUS, 'counts': COUNTS,
                             'tactical_quota': TACTICAL_QUOTA, 'solver_node_limit': SOLVER_NODE_LIMIT},
        'generator_config_sha256': _sha({'opening_plies': OPENING_PLIES, 'radius': RADIUS, 'counts': COUNTS,
                                         'tactical_quota': TACTICAL_QUOTA, 'solver_node_limit': SOLVER_NODE_LIMIT}),
        'd4_rule': 'stones + side to move, minimized over D4 (hybrid.h6_labels.canonical_hash)',
        'excluded': {'benchmark_openings_in_docs': info['known_benchmark_openings_excluded'],
                     'probe_files': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in probe_files}},
        # Not the RNG seed (that is sha256('final-sets-v1:' + secret)): publishing this does not let anyone regenerate the sets.
        'secret_commitment': hashlib.sha256(f'commitment:{secret}'.encode()).hexdigest(),
        'artifacts': artifacts, 'tactical_generation': info['tactical'],
        'note': 'content_sha256 = SHA-256 of the canonical JSON of each set\'s items (line endings do not matter)',
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'artifacts': artifacts, 'tactical': info['tactical']}, indent=1))
    print(f'sets and SECRET.txt in {out_dir} (keep them away from the repo and from analysis tools until H8)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
