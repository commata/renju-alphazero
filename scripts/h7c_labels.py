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
Policy rows (``POLICY_FORMAT``) take the target the generator wrote, once it matched ``hybrid.h7c_targets.policy_target``
(below): kind ``visits`` (KL to H3 applies) or ``proof_one_hot`` (no KL), with the raw visits; kind ``none`` (forced
moves: approved, no policy loss there) gives no row. A policy row is dropped with its game (probe) and masked when its
position is a row-only probe (by the probe hashes, also before ``--from-ply``) or a later-split duplicate.

Inputs must be one cycle of one model from one clean generator commit: training run seeds only (never 8800 or a
reserved seed), no truncated game, the same identity in every game. Every game is replayed (order, mover, legal
moves, result) and every stored target and sampling event is recomputed with ``hybrid.h7c_targets`` and must match:
the label step never trusts the generator's output. Rows carry the H6 fields plus ``h7c_provenance`` (runtime hash,
the four model SHA-256, the self-play and label commits); ``h6_value_cache`` reads the H6 fields and ignores it.
Outputs go under ``data/`` or ``runs/`` (never committed), written to ``.tmp`` files and moved into place only after
everything passed. The report records the label step's own ``git_commit`` and ``git_dirty``; a dirty tree is refused
(``--allow-dirty`` for tests only).

Retarget (cycle >= 2, ``--retarget``, rules in ``hybrid.h7c_retarget``): the records of ``scripts/h7c_retarget.py
run`` for exactly this self-play file (its SHA-256, identity and coverage: every actor ``visits`` ply once, no other)
replace the actor's search in the policy targets of those plies: ``hybrid.h7c_targets.policy_target`` over the
retarget's visits and the union of both proven-loss sets (a target with no mass gives no row, counted). Proof rows and
the value rows are unchanged. The rows are ``h7c-policy-row-v2`` (``POLICY_FORMAT_V2``) and carry the retarget's
identity; without ``--retarget`` the output is the v1 output, unchanged.

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
from hybrid import h7c_retarget as hr  # noqa: E402
from hybrid.h6_labels import (  # noqa: E402
    canonical_hash, check_game_provenance, check_leakage, check_reserved_seed, default_probe_files,
    first_probe, game_rows, mask_cross_split, probe_positions, report, side_name, split_for,
)
from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import BLACK  # noqa: E402

FORMAT = 'h7c-labels-v1'
POLICY_FORMAT = 'h7c-policy-row-v1'
POLICY_FORMAT_V2 = hr.POLICY_ROW_V2
SOURCE = 'h7c-selfplay'
SPLIT_SALT = 'h7c-v1'
SPLIT_FRACTIONS = (('val', 0.05),)
IDENTITY_KEYS = ('format', 'cycle', 'engine_id', 'engine_hash', 'runtime', 'runtime_hash', 'models', 'model_hash',
                 'frozen_models', 'opening_rule', 'sampling', 'target_format', 'generator_git')
TARGET_TOLERANCE = 1e-9


def game_id(game: dict) -> str:
    return f"h7c:c{game['cycle']}:{game['run_seed']}:{game['game_index']}"


def read_games(paths) -> list[dict]:
    rows = []
    for path in paths:
        with Path(path).open(encoding='utf-8') as handle:
            rows += [json.loads(line) for line in handle if line.strip()]
    return rows


def check_target(g: dict, p: dict, opening_plies: int) -> None:
    """The stored target and sampling event must be what ``hybrid.h7c_targets`` gives for the stored move facts."""
    from hybrid import h7c_targets as targets

    where = f"{game_id(g)} ply {p['ply']}"
    expected = targets.policy_target(p['route'], p['engine_move'], p['visits'], p['proven_loss'])
    stored = p['target']
    if stored.get('format') != expected['format'] or stored.get('kind') != expected['kind'] or \
            stored.get('kl') != expected['kl']:
        raise ValueError(f"{where}: target kind {stored.get('kind')!r} / kl {stored.get('kl')!r}, the rule gives "
                         f"{expected['kind']!r} / {expected['kl']!r}")
    if expected['target'] is None:
        if stored.get('target') is not None:
            raise ValueError(f'{where}: a target where the rule gives none')
    else:
        got = stored.get('target') or []
        if [m[:2] for m in got] != [m[:2] for m in expected['target']] or \
                any(abs(a[2] - b[2]) > TARGET_TOLERANCE for a, b in zip(got, expected['target'])):
            raise ValueError(f'{where}: the target differs from the rule\'s')
        total = sum(m[2] for m in got)
        losses = {tuple(m) for m in p['proven_loss']}
        if any(m[2] < 0 for m in got) or abs(total - 1.0) > 1e-6 or \
                any(m[2] > 0 for m in got if tuple(m[:2]) in losses):
            raise ValueError(f'{where}: the target is not a distribution without the proven losses')
    early = p['ply'] - opening_plies < targets.SAMPLE_PLIES
    if not early:
        ok = p['event'] == 'engine' and p['played'] == p['engine_move']
    elif p['event'] == 'proven_win':
        ok = p['route'] in targets.PROVEN_WIN_ROUTES and p['played'] == p['engine_move']
    elif p['event'] == 'no_distribution':
        ok = p['played'] == p['engine_move'] and (
            p['route'] in targets.FORCED_ROUTES
            or targets.visit_distribution(p['visits'], p['proven_loss']) is None)
    elif p['event'] == 'sampled':
        dist = targets.visit_distribution(p['visits'], p['proven_loss']) or []
        ok = p['route'] not in targets.PROVEN_WIN_ROUTES and any(m[:2] == p['played'] and m[2] > 0 for m in dist)
    else:
        ok = False
    if not ok:
        raise ValueError(f"{where}: sampling event {p['event']!r} does not fit the rule (played {p['played']})")


def check_game(g: dict) -> None:
    """Replay the game: the opening, every ply in order, each mover and move legal, the stored result."""
    from renju import WHITE, Game

    game = Game()
    for move in g['opening']:
        game.play(*move)
    if len(g['plies']) + len(g['opening']) != len(g['moves']) or g['length'] != len(g['moves']):
        raise ValueError(f'{game_id(g)}: its length and per-move records disagree')
    for i, p in enumerate(g['plies']):
        where = f"{game_id(g)} ply {p['ply']}"
        if p['ply'] != len(g['opening']) + i or p['ply'] != len(game.history):
            raise ValueError(f'{where}: the plies are not in order')
        if p['color'] != ('black' if game.to_play == BLACK else 'white'):
            raise ValueError(f'{where}: recorded for {p["color"]}, {game.to_play} is to move')
        if game.done or tuple(p['played']) not in set(game.legal_moves()) or g['moves'][p['ply']] != p['played']:
            raise ValueError(f'{where}: the played move {p["played"]} is not the legal recorded move')
        check_target(g, p, len(g['opening']))
        game.play(*p['played'])
    if not game.done or game.winner != g['winner']:
        raise ValueError(f'{game_id(g)}: the replayed game does not end with the recorded winner')
    expected = 'draw' if game.winner is None else 'black' if game.winner == BLACK else 'white'
    if g['result'] != expected or g['real_draw'] != (game.winner is None and len(game.history) == 15 * 15) or \
            game.winner not in (BLACK, WHITE, None):
        raise ValueError(f'{game_id(g)}: result {g["result"]!r} / real_draw do not match the replay')


def check_games(games: list[dict]) -> dict:
    """One cycle of one model from training runs, one clean generator commit, complete legal games, no duplicates;
    the shared identity."""
    import h7c_selfplay as sp

    if not games:
        raise ValueError('no games')
    first = {k: games[0].get(k) for k in IDENTITY_KEYS}
    git = first.get('generator_git') or {}
    if not git.get('commit') or git.get('dirty') is not False:
        raise ValueError(f'the self-play generator ran from {git or "an unknown tree"}: only games of a clean, '
                         f'committed generator are training data')
    seen = set()
    for g in games:
        if {k: g.get(k) for k in IDENTITY_KEYS} != first:
            raise ValueError(f'{game_id(g)}: another cycle, engine, runtime, model, rule or generator commit than '
                             f'the first game')
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
        check_game(g)
    return first


def policy_rows(game: dict, split: str, provenance: dict, retarget: dict | None = None,
                counts: dict | None = None) -> list[dict]:
    """The policy rows of one game; with ``retarget`` (``{'rows': {(game index, ply): record}, 'info': {...}}``) the
    v2 rows, the visits plies' targets from the retarget records (module docstring)."""
    out = []
    for p in game['plies']:
        target = p['target']
        if target['kind'] == 'none':
            continue
        moves = game['moves'][:p['ply']]
        row = {'format': POLICY_FORMAT, 'game_id': game_id(game), 'source': SOURCE, 'split': split,
               'ply': p['ply'], 'moves': moves, 'side_to_move': side_name(p['ply']),
               'canonical_hash': canonical_hash(moves), 'kind': target['kind'], 'kl': target['kl'],
               'target': target['target'], 'raw_visits': p['visits'], 'played': p['played'],
               'event': p['event'], 'route': p['route'], 'mask': False, 'flags': [], **provenance}
        if retarget is not None:
            row.update(format=POLICY_FORMAT_V2, retarget=retarget['info'], actor_raw_visits=p['visits'],
                       actor_proven_loss=p['proven_loss'])
            if target['kind'] == 'visits':
                record = retarget['rows'][(game['game_index'], p['ply'])]
                new, losses, loss_counts = hr.retarget_target(p, record)
                if counts is not None:
                    for k in ('same', 'actor_only', 'retarget_only'):
                        counts[k] = counts.get(k, 0) + loss_counts[k]
                    counts['move_changed'] = counts.get('move_changed', 0) + (
                        list(record['engine_move']) != list(p['engine_move']))
                if new['kind'] == 'none':  # all of the retarget's mass on proven losses: no policy loss here
                    if counts is not None:
                        counts['retarget_none'] = counts.get('retarget_none', 0) + 1
                    continue
                row.update(kind=new['kind'], kl=new['kl'], target=new['target'], raw_visits=record['visits'],
                           target_source='retarget', proven_loss=losses, retarget_proven_loss=record['proven_loss'],
                           retarget_move=record['engine_move'])
                if counts is not None:
                    counts['retargeted'] = counts.get('retargeted', 0) + 1
            else:
                row.update(target_source='actor', proven_loss=p['proven_loss'], retarget_proven_loss=None,
                           retarget_move=None)
        out.append(row)
    return out


def load_retarget(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def retarget_rows(games: list[dict], records: list[dict], sha256: str) -> dict:
    """The retarget records of exactly these games (identity, coverage, every row checked), by (game index, ply)."""
    if not records:
        raise ValueError('the retarget file is empty')
    ident = records[0].get('identity') or {}
    want = hr.selfplay_identity(games)
    if ident.get('format') != hr.FORMAT or ident.get('purpose') != 'retarget':
        raise ValueError('not a retarget run file (a G0 check file is never training data)')
    if ident.get('selfplay') != want:
        raise ValueError('the retarget file is of another self-play (cycle, seed, engine, runtime, models or commit)')
    if ident.get('budget') != hr.budget(ident.get('factor')):
        raise ValueError('the retarget budget is not 50k / 100k of its factor')
    if ident.get('coverage') != hr.coverage(games):
        raise ValueError("the retarget coverage is not this self-play's visits plies")
    rows = hr.index_rows(games, records, ident)
    info = {'sha256': sha256, 'format': hr.FORMAT, 'factor': ident['factor'], 'budget': ident['budget'],
            'mode': ident['mode'], 'selfplay_jsonl_sha256': ident['selfplay_jsonl_sha256'],
            'coverage_sha256': ident['coverage']['coverage_sha256'], 'decision_sha256': ident['decision_sha256'],
            'cost_review_sha256': ident['cost_review_sha256'], 'git_commit': ident['git']['commit']}
    return {'rows': rows, 'info': info}


def mask_policy(policy: list[dict], row_probes: set[str]) -> dict:
    """Mask policy rows at row-only probe positions (from the probe hashes themselves: policy rows start before the
    value rows' ``from_ply``) and later-split duplicates (train < val), as the value rows are."""
    counts = {'probe_position': 0, 'cross_split_duplicate': 0}
    for row in policy:
        if row['canonical_hash'] in row_probes:
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
          progress_seconds: float = 60.0, retarget: dict | None = None) -> tuple[list, list[dict], dict]:
    """``retarget``: ``{'records': [...], 'sha256': ...}`` of ``scripts/h7c_retarget.py run`` (v2 policy rows)."""
    identity = check_games(games)
    if retarget is not None:  # over every game of the file, before any is dropped (the run searched them all)
        retarget = retarget_rows(games, retarget['records'], retarget['sha256'])
    retarget_counts: dict = {}
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
    # Beside the H6 row fields (which h6_value_cache reads; it ignores the rest): what exactly made the data.
    detail = {'runtime_hash': identity['runtime_hash'], **identity['models'],
              'selfplay_git_commit': identity['generator_git']['commit'], 'label_git_commit': provenance['git_commit']}
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
                                             'git_commit': provenance['git_commit'], 'h7c_provenance': detail},
                                  retarget=retarget, counts=retarget_counts))
    cross = mask_cross_split(rows)
    check_leakage(rows)
    check_game_provenance(rows)
    policy_masked = mask_policy(policy, row_probes)
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
    if retarget is not None:
        summary['policy']['retarget'] = {'info': retarget['info'], 'kept_games': retarget_counts}
    for row in rows:
        row.h7c_provenance = detail
    return rows, policy, summary


def value_row_dict(row) -> dict:
    """The H6 row (``Row.to_dict``) plus the H7-c provenance field."""
    return {**row.to_dict(), 'h7c_provenance': row.h7c_provenance}


def write_atomic(items: dict[Path, list[dict]], report: tuple[Path, str] | None = None) -> None:
    """Every output to ``<name>.tmp`` first; only when all are written are they moved into place."""
    import os

    temps = []
    try:
        for path, rows in items.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + '.tmp')
            temps.append((temp, path))
            with gzip.open(temp, 'wt', encoding='utf-8') as handle:
                for row in rows:
                    handle.write(json.dumps(row, separators=(',', ':')) + '\n')
        if report is not None:
            path, text = report
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + '.tmp')
            temps.append((temp, path))
            temp.write_text(text, encoding='utf-8')
    except BaseException:
        for temp, _ in temps:
            temp.unlink(missing_ok=True)
        raise
    for temp, path in temps:
        os.replace(temp, path)


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
    parser.add_argument('--retarget', type=Path, help='scripts/h7c_retarget.py run records of this self-play file')
    parser.add_argument('--allow-dirty', action='store_true', help='tests only: labels from a dirty tree')
    args = parser.parse_args(argv)
    from run_mcts_v8_benchmark import _git_dirty
    git = {'commit': h6b._git_commit(), 'dirty': _git_dirty()}
    if git['dirty'] is not False and not args.allow_dirty:
        parser.error('the tree is dirty (or git is unavailable): training labels must name a clean code commit')
    for path in (args.value_output, args.policy_output):
        if not _under_data_or_runs(path):
            parser.error(f'{path}: label rows go under data/ or runs/, never into the repository')
        if path.exists() or path.with_name(path.name + '.tmp').exists():
            parser.error(f'{path} (or its .tmp) exists')
    probe_files = default_probe_files(ROOT) + list(args.probe_files)
    retarget = None
    if args.retarget is not None:
        if len(args.games) != 1:
            parser.error('--retarget: one self-play file (the one the retarget searched)')
        import h7c_retarget as rt
        retarget = {'records': load_retarget(args.retarget), 'sha256': rt.file_sha256(args.retarget)}
        if retarget['records'] and (retarget['records'][0].get('identity') or {}).get('selfplay_jsonl_sha256') != \
                rt.file_sha256(args.games[0]):
            parser.error(f'{args.retarget} was made from another self-play file than {args.games[0]}')
    try:
        rows, policy, summary = build(read_games(args.games), probes=probe_positions(probe_files, ROOT),
                                      from_ply=args.from_ply, outcome_weight=args.outcome_weight,
                                      proof_weight=args.proof_weight, prove_losses=args.prove_losses, vct1=args.vct1,
                                      probe_game_min_ply=args.probe_game_min_ply, workers=args.workers, fast=args.fast,
                                      retarget=retarget)
    except ValueError as exc:
        parser.error(str(exc))
    payload = {'format': FORMAT, 'value_row_format': rows[0].to_dict()['format'] if rows else None,
               'policy_row_format': POLICY_FORMAT if retarget is None else POLICY_FORMAT_V2,
               'git_commit': git['commit'], 'git_dirty': git['dirty'],
               'games': [str(p) for p in args.games],
               'settings': {k: getattr(args, k) for k in ('from_ply', 'outcome_weight', 'proof_weight', 'prove_losses',
                                                          'vct1', 'probe_game_min_ply', 'fast')},
               'split': {'salt': SPLIT_SALT, 'fractions': dict(SPLIT_FRACTIONS)},
               'probe_files': [str(Path(p).resolve().relative_to(ROOT)).replace('\\', '/')
                               if Path(p).resolve().is_relative_to(ROOT) else str(p) for p in probe_files],
               'summary': summary}
    print(json.dumps(summary, indent=1), flush=True)
    write_atomic({args.value_output: [value_row_dict(r) for r in rows], args.policy_output: policy},
                 None if args.report is None else (args.report, json.dumps(payload, indent=1) + '\n'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
