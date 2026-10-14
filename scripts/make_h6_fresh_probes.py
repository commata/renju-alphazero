"""H6 fresh-final probes from TB-Engine-v1 games on the reserved seeds 8601-8610 (docs/h6-value.md, H6-2a').

Truth only (no model is loaded): for every game, from ply ``--from-ply`` on,
    - a **deep win** (+1): no depth-0 label (no immediate win / unstoppable four / VCF / forced loss) and a
      proven depth-1 VCT attack for the side to move (``vct1_win``, V8-B's budget);
    - a **deep loss** (-1): no depth-0 win for the side to move and not a VCF-level loss (``vcf_loss``), and
      ``defense_label`` proves ``vct_loss``: every move that survives VCF loses at depth 1 (parent level, P94's
      kind).
The earliest qualifying position of each sign is kept per game; then each game contributes **at most one**
probe, assigned in seed / game order to the sign that has fewer probes so far (ties: loss). Positions D4-equal
to an existing probe / evaluation position, or to another fresh probe, are skipped.

The output is committed and **sealed**: no model is evaluated on it before the final H6 gate (part B). It is
added to the default probe files, so a later label build drops every RenjuNet game that reaches one.

    python scripts/make_h6_fresh_probes.py --games runs/h6_fresh/*.json --workers 12 \\
        --output docs/h6/h6_fresh_final_probes_v1.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hybrid.h6_labels import (  # noqa: E402
    FRESH_PROBE_SEEDS, canonical_hash, default_probe_files, probe_hashes, proof_label,
)
from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import Game  # noqa: E402

FORMAT = 'h6-fresh-probes-v1'
SOLVER_NODE_LIMIT = 20_000
VCT1_BUDGET = {'node_limit': 20_000, 'call_limit': 10_000, 'node_budget': 200_000}  # V8-B's budget
DEFENSE_MAX_CANDIDATES = 8
DEPTH0_KINDS = ('immediate_win', 'unstoppable_four', 'vcf', 'forced_loss', 'vcf_loss')
ENGINE_ARM = 'puct_policy_vct2_50k'  # TB-Engine-v1 (E3-A)


def _game(moves) -> Game:
    game = Game()
    for move in moves:
        game.play(*move)
    return game


def classify(moves) -> str | None:
    """'win' / 'loss' / None for the position after ``moves`` (side to move's view)."""
    from analysis.tactical_labels import defense_label, label_position
    from analysis.threats import ThreatSolver

    game = _game(moves)
    if game.done:
        return None
    solver = ThreatSolver(node_limit=SOLVER_NODE_LIMIT)
    shallow = label_position(game, solver, prove_losses=True)
    if shallow is not None and shallow['kind'] in DEPTH0_KINDS:
        return None
    if shallow is None:  # a deep win only without any pending threat (must_block) of the opponent
        deep = proof_label(moves, solver, vct1_budget=VCT1_BUDGET)
        if deep is not None and deep['kind'] == 'vct1_win':
            return 'win'
    label = defense_label(_game(moves), ThreatSolver(node_limit=SOLVER_NODE_LIMIT),
                          max_candidates=DEFENSE_MAX_CANDIDATES)
    if label is not None and label['kind'] == 'vct_loss':
        return 'loss'
    return None


def scan_game(task) -> dict:
    """Earliest deep win and earliest deep loss of one game."""
    moves, from_ply = task
    found = {}
    for ply in range(from_ply, len(moves)):
        if len(found) == 2:
            break
        sign = classify(moves[:ply])
        if sign is not None and sign not in found:
            found[sign] = ply
    return found


def _scan_chunk(tasks):
    return [scan_game(t) for t in tasks]


def select(games: list[dict], found: list[dict], exclude: set[str]) -> list[dict]:
    """At most one probe per game, balancing the signs in game order."""
    probes, taken, count = [], set(exclude), {'loss': 0, 'win': 0}
    for game, hits in zip(games, found):
        options = []
        for sign in sorted(hits, key=lambda s: (count[s], s != 'loss')):
            ply = hits[sign]
            key = canonical_hash(game['moves'][:ply])
            if key not in taken:
                options.append((sign, ply, key))
        if not options:
            continue
        sign, ply, key = options[0]
        taken.add(key)
        count[sign] += 1
        probes.append({'name': f"F{len(probes) + 1:02d}", 'game_key': game['key'], 'seed': game['seed'],
                       'ply': ply, 'moves': game['moves'][:ply],
                       'side_to_move': 'black' if ply % 2 == 0 else 'white',
                       'expected_sign': 1 if sign == 'win' else -1,
                       'kind': 'vct1_win' if sign == 'win' else 'vct_loss', 'canonical_hash': key})
    return probes


def load_games(paths: list[Path]) -> tuple[list[dict], dict]:
    games, sources = [], {}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('arm') != ENGINE_ARM:
            raise ValueError(f"{path}: arm {data.get('arm')!r} is not TB-Engine-v1 ({ENGINE_ARM})")
        if int(data['seed']) not in FRESH_PROBE_SEEDS:
            raise ValueError(f"{path}: seed {data['seed']} is not a fresh-probe seed {FRESH_PROBE_SEEDS}")
        sources[str(path)] = {'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                              'seed': data['seed'], 'opponent': data['opponent'], 'git_commit': data.get('git_commit'),
                              'games': len(data['games'])}
        games.extend({'key': g['key'], 'seed': int(data['seed']), 'moves': g['moves']} for g in data['games'])
    games.sort(key=lambda g: (g['seed'], g['key']))
    return games, sources


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--games', nargs='+', required=True, help='run_mcts_v8_benchmark outputs (globs allowed)')
    parser.add_argument('--from-ply', type=int, default=10)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists: the fresh probes are made once')
    paths = [Path(p) for pattern in args.games for p in (sorted(glob.glob(pattern)) or [pattern])]
    games, sources = load_games(paths)
    tasks = [(g['moves'], args.from_ply) for g in games]
    chunks = [tasks[i:i + 1] for i in range(len(tasks))]
    progress = Progress('fresh probes', len(tasks), 'games')
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            found = [x for part in run_chunks(pool, _scan_chunk, chunks, progress, len, unit_name='games')
                     for x in part]
    else:
        found = []
        for c in chunks:
            found.extend(_scan_chunk(c))
            progress.update(1)
    exclude = probe_hashes(p for p in default_probe_files(ROOT) if Path(p).resolve() != args.output.resolve())
    probes = select(games, found, exclude)
    counts = {'win': sum(p['expected_sign'] == 1 for p in probes), 'loss': sum(p['expected_sign'] == -1 for p in probes),
              'games_scanned': len(games), 'games_with_win': sum('win' in f for f in found),
              'games_with_loss': sum('loss' in f for f in found)}
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    payload = {'format': FORMAT, 'sealed': True, 'git_commit': commit, 'sources': sources,
               'selection': {'from_ply': args.from_ply, 'one_probe_per_game': True, 'balance': 'fewer sign first',
                             'depth0_excluded': list(DEPTH0_KINDS), 'vct1_budget': VCT1_BUDGET,
                             'defense_max_candidates': DEFENSE_MAX_CANDIDATES, 'solver_node_limit': SOLVER_NODE_LIMIT},
               'counts': counts, 'probes': probes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
