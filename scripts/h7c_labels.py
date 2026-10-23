"""H7-c self-play labels: value rows (the H6 row format) and policy rows from ``h7c_selfplay`` games (docs/h7-c1.md).

The rules are H6's, reused rather than re-implemented:
    proofs      ``build_h6_labels``' proof task (``hybrid.h6_labels.proof_label``, its VCF node limit and, with
                ``--vct1`` / ``--prove-losses``, its budgets), weight ``--proof-weight`` (1.0)
    outcome     the game result from the side to move, weight ``--outcome-weight`` (0.1); a draw is 0 only for a
                full board; a proof beats the result (``outcome_conflicts_proof``)
    probes      every game that reaches a probe / evaluation position (D4, side to move; ``default_probe_files``
                plus ``--probe-files``) is dropped whole; only 3-stone opening probes may stay row-only
                (``--probe-game-min-ply 4``, H6-v2), and those rows are masked
    splits      per game (``split_for``, salt ``h7c-v1``: val 5%, the rest train); later-split duplicates of a
                position are masked and ``check_leakage`` refuses any unmasked overlap
    provenance  one engine / model per game (``check_game_provenance``); every row names the engine hash and the
                self-play model hash
Policy rows (``POLICY_FORMAT``) take the target the generator wrote (``hybrid.h7c_targets.policy_target``): kind
``visits`` (KL to H3 applies) or ``proof_one_hot`` (no KL), with the raw visits; kind ``none`` gives no row. A
policy row is dropped with its game (probe) and masked when its position is a masked probe row or a later-split
duplicate.

Inputs must be one cycle of one model: training run seeds only (never 8800 or a reserved seed), no truncated
game, the same identity in every game. Outputs go under ``data/`` or ``runs/`` (never committed).

    python -u scripts/h7c_labels.py --games runs/h7c/c1/selfplay_8801.jsonl --workers 12 --fast fast4 \\
        --value-output runs/h7c/c1/value_rows.jsonl.gz --policy-output runs/h7c/c1/policy_rows.jsonl.gz \\
        --report runs/h7c/c1/labels_report.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_h6_labels as h6b  # noqa: E402
from hybrid import h6_fast  # noqa: E402
from hybrid.h6_labels import (  # noqa: E402
    canonical_hash, check_game_provenance, check_leakage, check_reserved_seed, default_probe_files,
    first_probe, game_rows, mask_cross_split, probe_positions, report, side_name, split_for,
)
from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import BLACK  # noqa: E402

FORMAT = 'h7c-labels-v1'
POLICY_FORMAT = 'h7c-policy-row-v1'
SOURCE = 'h7c-selfplay'
SPLIT_SALT = 'h7c-v1'
SPLIT_FRACTIONS = (('val', 0.05),)
IDENTITY_KEYS = ('format', 'cycle', 'engine_id', 'engine_hash', 'runtime', 'runtime_hash', 'models', 'model_hash',
                 'frozen_models', 'opening_rule', 'sampling', 'target_format')


def game_id(game: dict) -> str:
    return f"h7c:c{game['cycle']}:{game['run_seed']}:{game['game_index']}"


def read_games(paths) -> list[dict]:
    rows = []
    for path in paths:
        with Path(path).open(encoding='utf-8') as handle:
            rows += [json.loads(line) for line in handle if line.strip()]
    return rows


def check_games(games: list[dict]) -> dict:
    """One cycle of one model from training runs, complete games, no duplicates; the shared identity."""
    import h7c_selfplay as sp

    if not games:
        raise ValueError('no games')
    first = {k: games[0].get(k) for k in IDENTITY_KEYS}
    seen = set()
    for g in games:
        if {k: g.get(k) for k in IDENTITY_KEYS} != first:
            raise ValueError(f'{game_id(g)}: another cycle, engine, runtime, model or rule than the first game')
        if g['run_seed'] not in sp.RUN_SEEDS:
            raise ValueError(f"{game_id(g)}: run seed {g['run_seed']} is not a training run seed")
        check_reserved_seed(g['run_seed'])
        if g.get('truncated') or g.get('result') == 'truncated':
            raise ValueError(f'{game_id(g)}: a truncated game is never training data')
        if game_id(g) in seen:
            raise ValueError(f'{game_id(g)}: listed twice')
        seen.add(game_id(g))
        if g['moves'][:len(g['opening'])] != g['opening'] or [p['played'] for p in g['plies']] != \
                g['moves'][len(g['opening']):]:
            raise ValueError(f'{game_id(g)}: its moves and per-move records disagree')
    return first


def policy_rows(game: dict, split: str, provenance: dict) -> list[dict]:
    out = []
    for p in game['plies']:
        target = p['target']
        if target['kind'] == 'none':
            continue
        moves = game['moves'][:p['ply']]
        out.append({'format': POLICY_FORMAT, 'game_id': game_id(game), 'source': SOURCE, 'split': split,
                    'ply': p['ply'], 'moves': moves, 'side_to_move': side_name(p['ply']),
                    'canonical_hash': canonical_hash(moves), 'kind': target['kind'], 'kl': target['kl'],
                    'target': target['target'], 'raw_visits': p['visits'], 'played': p['played'],
                    'event': p['event'], 'route': p['route'], 'mask': False, 'flags': [], **provenance})
    return out


def mask_policy(policy: list[dict], value_rows) -> dict:
    """Mask policy rows at masked probe positions and later-split duplicates (train < val), like the value rows."""
    probe_masked = {(r.game_id, r.ply) for r in value_rows if 'probe_position' in r.flags}
    counts = {'probe_position': 0, 'cross_split_duplicate': 0}
    for row in policy:
        if (row['game_id'], row['ply']) in probe_masked:
            row['mask'] = True
            row['flags'].append('probe_position')
            counts['probe_position'] += 1
    seen = {r['canonical_hash'] for r in policy if r['split'] == 'train'}
    for row in policy:
        if row['split'] != 'train' and row['canonical_hash'] in seen and not row['mask']:
            row['mask'] = True
            row['flags'].append('cross_split_duplicate')
            counts['cross_split_duplicate'] += 1
    return counts


def build(games: list[dict], *, probes: dict, from_ply: int, outcome_weight: float, proof_weight: float,
          prove_losses: bool, vct1: bool, probe_game_min_ply: int, workers: int, fast: str,
          progress_seconds: float = 60.0) -> tuple[list, list[dict], dict]:
    identity = check_games(games)
    game_probes = {h for h, p in probes.items() if p['ply'] >= probe_game_min_ply}
    row_probes = set(probes) - game_probes
    unsafe = [p for h, p in probes.items() if h in row_probes and p['ply'] > h6b.MAX_ROW_ONLY_PROBE_PLY]
    if unsafe:
        raise ValueError(f'--probe-game-min-ply {probe_game_min_ply} would keep continuations after '
                         f'{len(unsafe)} non-opening probe(s); use {h6b.MAX_ROW_ONLY_PROBE_PLY + 1} or more')
    kept, dropped = [], {'probe_game': 0}
    for g in games:
        if game_probes and first_probe(g['moves'], game_probes) is not None:
            dropped['probe_game'] += 1
            continue
        kept.append(g)
    tasks = [(g['moves'], from_ply, prove_losses, vct1, 0) for g in kept]
    positions = lambda chunk: sum(max(0, len(t[0]) - from_ply) for t in chunk)  # noqa: E731
    chunks = [tasks[i:i + h6b.PROOF_CHUNK] for i in range(0, len(tasks), h6b.PROOF_CHUNK)]
    progress = Progress('h7c-proofs', positions(tasks), 'positions', every_seconds=progress_seconds)
    progress.log(f'[h7c-proofs] {len(kept):,} games ({dropped["probe_game"]} dropped: probe), '
                 f'{progress.total:,} positions, {workers} worker(s)', flush=True)
    previous = h6_fast.installed_mode()
    expected = h6_fast.install(fast)['active']
    try:
        if workers == 1:
            parts = []
            for chunk in chunks:
                parts.append(h6b._proofs_chunk(chunk))
                progress.update(positions(chunk))
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=h6_fast.install, initargs=(fast,)) as pool:
                parts = run_chunks(pool, h6b._proofs_chunk, chunks, progress, positions)
    finally:
        h6_fast.install(previous)
    if any(active != expected for active, _ in parts):
        raise RuntimeError(f'--fast {fast}: a worker ran another solver mode')
    proofs = [proof for _, part in parts for proof, _ in part]
    solver_class = 'vcf' + ('+vct1' if vct1 else '')
    budget = {'vcf_node_limit': h6b.SOLVER_NODE_LIMIT, **({'vct1': h6b.VCT1_BUDGET} if vct1 else {}),
              **({'prove_losses': h6b.LOSS_BUDGET} if prove_losses else {})}
    provenance = {'engine_config_hash': identity['engine_hash'], 'model_hash': identity['model_hash'],
                  'git_commit': h6b._git_commit()}
    rows, policy = [], []
    for g, proof in zip(kept, proofs):
        split = split_for(game_id(g), fractions=SPLIT_FRACTIONS, salt=SPLIT_SALT)
        result = {'id': game_id(g), 'moves': g['moves'], 'winner': g['winner'], 'real_draw': g['real_draw']}
        these = game_rows(result, source=SOURCE, split=split, outcome_weight=outcome_weight,
                          proof_weight=proof_weight, proofs=proof, solver_class=solver_class, solver_budget=budget,
                          provenance=provenance, from_ply=from_ply)
        for row in these:
            if row.canonical_hash in row_probes:
                row.value_mask, row.weight = True, 0.0
                row.flags.append('probe_position')
        rows.extend(these)
        policy.extend(policy_rows(g, split, {'engine_config_hash': identity['engine_hash'],
                                             'model_hash': identity['model_hash'],
                                             'git_commit': provenance['git_commit']}))
    cross = mask_cross_split(rows)
    check_leakage(rows)
    check_game_provenance(rows)
    policy_masked = mask_policy(policy, rows)
    kinds, splits = {}, {}
    for row in policy:
        if not row['mask']:
            kinds[row['kind']] = kinds.get(row['kind'], 0) + 1
            splits[row['split']] = splits.get(row['split'], 0) + 1
    winners = {}
    for g in kept:
        key = 'draw' if g['winner'] is None else 'black' if g['winner'] == BLACK else 'white'
        winners[key] = winners.get(key, 0) + 1
    summary = {'identity': {k: identity[k] for k in ('cycle', 'engine_hash', 'runtime_hash', 'model_hash',
                                                     'frozen_models')},
               'games_in': len(games), 'games_kept': len(kept), 'dropped': dropped, 'results': winners,
               'value': {'cross_split_masked': cross, **report(rows)},
               'policy': {'rows': len(policy), 'unmasked_by_kind': kinds, 'unmasked_by_split': splits,
                          'masked': policy_masked}}
    return rows, policy, summary


def _under_data_or_runs(path: Path) -> bool:
    path = path.resolve()
    return not path.is_relative_to(ROOT) or str(path.relative_to(ROOT)).replace('\\', '/').startswith(('data/', 'runs/'))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--games', type=Path, nargs='+', required=True, help='h7c_selfplay JSONL file(s), one cycle')
    parser.add_argument('--from-ply', type=int, default=5, help='first value-labelled ply (H6)')
    parser.add_argument('--outcome-weight', type=float, default=0.1)
    parser.add_argument('--proof-weight', type=float, default=1.0)
    parser.add_argument('--prove-losses', action='store_true')
    parser.add_argument('--vct1', action='store_true')
    parser.add_argument('--probe-files', nargs='*', default=[], help='added to default_probe_files (web probes)')
    parser.add_argument('--probe-game-min-ply', type=int, default=h6b.MAX_ROW_ONLY_PROBE_PLY + 1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--fast', choices=h6_fast.MODES, default='fast4')
    parser.add_argument('--value-output', type=Path, required=True)
    parser.add_argument('--policy-output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    for path in (args.value_output, args.policy_output):
        if not _under_data_or_runs(path):
            parser.error(f'{path}: label rows go under data/ or runs/, never into the repository')
        if path.exists():
            parser.error(f'{path} exists')
    probe_files = default_probe_files(ROOT) + list(args.probe_files)
    try:
        rows, policy, summary = build(read_games(args.games), probes=probe_positions(probe_files, ROOT),
                                      from_ply=args.from_ply, outcome_weight=args.outcome_weight,
                                      proof_weight=args.proof_weight, prove_losses=args.prove_losses, vct1=args.vct1,
                                      probe_game_min_ply=args.probe_game_min_ply, workers=args.workers, fast=args.fast)
    except ValueError as exc:
        parser.error(str(exc))
    for path, items in ((args.value_output, [r.to_dict() for r in rows]), (args.policy_output, policy)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt', encoding='utf-8') as handle:
            for item in items:
                handle.write(json.dumps(item, separators=(',', ':')) + '\n')
    payload = {'format': FORMAT, 'value_row_format': rows[0].to_dict()['format'] if rows else None,
               'policy_row_format': POLICY_FORMAT, 'git_commit': h6b._git_commit(),
               'games': [str(p) for p in args.games],
               'settings': {k: getattr(args, k) for k in ('from_ply', 'outcome_weight', 'proof_weight', 'prove_losses',
                                                          'vct1', 'probe_game_min_ply', 'fast')},
               'split': {'salt': SPLIT_SALT, 'fractions': dict(SPLIT_FRACTIONS)},
               'probe_files': [str(Path(p).resolve().relative_to(ROOT)).replace('\\', '/')
                               if Path(p).resolve().is_relative_to(ROOT) else str(p) for p in probe_files],
               'summary': summary}
    print(json.dumps(summary, indent=1), flush=True)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
