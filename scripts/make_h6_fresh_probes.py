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

``--dev`` (H6-2a' option B) makes the dev deep-probe set from the dev seeds 8611-8616 instead: same scan and
truth, but each game gives at most one probe **per sign** (its earliest deep win and earliest deep loss), so
each sign is still one probe per game and is judged on its own. The file is marked ``purpose: dev``, is not
sealed, and the final gate refuses it. It also skips positions D4-equal to the final probes.

    python scripts/make_h6_fresh_probes.py --dev --games "runs/h6_dev/dev_*.json" --workers 12 \\
        --jsonl runs/h6_dev/scan.jsonl --output docs/h6/h6_fresh_dev_probes_v1.json
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
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from hybrid.h6_labels import (  # noqa: E402
    DEV_PROBE_EXTENSION_SEEDS, DEV_PROBE_SEEDS, FRESH_PROBE_SEEDS, canonical_hash, default_probe_files, probe_hashes, proof_label,
)
from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import Game  # noqa: E402

FORMAT = 'h6-fresh-probes-v1'
SOLVER_NODE_LIMIT = 20_000
VCT1_BUDGET = {'node_limit': 20_000, 'call_limit': 10_000, 'node_budget': 200_000}  # V8-B's budget
DEFENSE_MAX_CANDIDATES = 8
# Per-position cap on uncached VCF calls (all checks of one position share it). A position over the cap is
# skipped (no probe), deterministically; P94 needs ~550 calls. Without it one position could run for hours.
VCF_CALL_BUDGET = 5_000
# Per-position cap on VCF search nodes (all checks of one position share it). The call cap alone did not bound
# the time: one call may use up to SOLVER_NODE_LIMIT nodes and the solver runs ~1-2k nodes/s. P94 needs
# ~20k nodes and ~550 calls. A position over either cap is skipped (no probe), deterministically.
VCF_NODE_BUDGET = 300_000
SCAN_VERSION = 3  # v1: no budget (rescanned), v2: call cap only, v3: call + node caps
# v2 rows are kept: v3 differs from v2 only on positions over 300k nodes, and the scan is truth-only (no
# model), so keeping 59 finished v2 games costs no validity. The versions used are recorded in the output.
ACCEPTED_SCAN_VERSIONS = (2, 3)
SLOW_POSITION_SECONDS = 60  # a position slower than this is logged by the worker


class _BudgetExceeded(Exception):
    pass


def _budgeted_solver():
    from analysis.threats import (VCF_NONE, VCF_UNKNOWN, VCF_WIN, ThreatSolver, _board_key,
                                  _find_vcf_with_stats, _unbounded_max_fours)

    class Budgeted(ThreatSolver):
        nodes_used = 0

        def vcf(self, game, attacker):
            """``ThreatSolver.vcf`` with the per-position call and node caps (a cap cut raises, never cached)."""
            key = (_board_key(game), attacker)
            cached = self._vcf_cache.get(key)
            if cached is not None:
                return cached
            remaining = VCF_NODE_BUDGET - self.nodes_used
            if self.vcf_calls >= VCF_CALL_BUDGET or remaining < 1:
                raise _BudgetExceeded
            limit = min(self.node_limit, remaining)
            self.vcf_calls += 1
            found, nodes, exhausted = _find_vcf_with_stats(
                game, attacker, max_fours=_unbounded_max_fours(game), node_limit=limit)
            self.nodes_used += nodes
            if found is not None:
                result = (VCF_WIN, tuple(found.attack_moves))
            elif exhausted:
                if limit < self.node_limit:
                    raise _BudgetExceeded  # cut by the position budget, not by the per-call limit
                self.vcf_exhausted += 1
                result = (VCF_UNKNOWN, ())
            else:
                result = (VCF_NONE, ())
            self._vcf_cache[key] = result
            return result

    return Budgeted(node_limit=SOLVER_NODE_LIMIT)


DEPTH0_KINDS = ('immediate_win', 'unstoppable_four', 'vcf', 'forced_loss', 'vcf_loss')
ENGINE_ARM = 'puct_policy_vct2_50k'  # TB-Engine-v1 (E3-A)


def _game(moves) -> Game:
    game = Game()
    for move in moves:
        game.play(*move)
    return game


def _vcf_survivors(game: Game, solver, limit: int):
    """Moves of the side to move that survive the opponent's VCF, stopping as soon as more than ``limit``
    survive: a list (all moves checked), ``'many'`` (> limit, not a deep loss) or ``'unknown'``. An empty list
    means every move loses to VCF (``vcf_loss``, depth 0). Same verdicts as ``defense_label``'s first step,
    without classifying every legal move."""
    from analysis.threats import SAFE, UNKNOWN, ordered_moves

    safe = []
    for move in ordered_moves(game):
        game.play(*move)
        try:
            status, _ = solver.after_move(game, 0, 0)
        finally:
            game.undo()
        if status == UNKNOWN:
            return 'unknown'
        if status == SAFE:
            safe.append(move)
            if len(safe) > limit:
                return 'many'
    return safe


def classify(moves) -> str | None:
    """'win' / 'loss' / None for the position after ``moves`` (side to move's view)."""
    game = _game(moves)
    if game.done:
        return None
    try:
        return _classify(game, moves, _budgeted_solver())
    except _BudgetExceeded:
        return 'budget'


def _classify(game, moves, solver) -> str | None:
    from analysis.tactical_labels import label_position
    from analysis.threats import SAFE, UNKNOWN

    shallow = label_position(game, solver)
    if shallow is not None and shallow['kind'] in DEPTH0_KINDS:
        return None
    survivors = _vcf_survivors(game, solver, DEFENSE_MAX_CANDIDATES)
    if survivors == []:
        return None  # vcf_loss: a depth-0 loss
    if shallow is None:  # a deep win only without any pending threat (must_block) of the opponent
        deep = proof_label(moves, solver, vct1_budget=VCT1_BUDGET)
        if deep is not None and deep['kind'] == 'vct1_win':
            return 'win'
    if isinstance(survivors, list):  # defense_label's vct_loss: every VCF survivor loses at depth 1
        deep = solver.classify(game, survivors, vct_depth=1)
        statuses = [status for status, _ in deep.values()]
        if UNKNOWN not in statuses and SAFE not in statuses:
            return 'loss'
    return None


def scan_game(task) -> dict:
    """Earliest deep win and earliest deep loss of one game, with the number of positions over budget."""
    moves, from_ply = task[:2]
    name = task[2] if len(task) > 2 else '?'
    found, cuts, started = {}, 0, perf_counter()
    for ply in range(from_ply, len(moves)):
        if len(found) == 2:
            break
        t0 = perf_counter()
        sign = classify(moves[:ply])
        took = perf_counter() - t0
        if took > SLOW_POSITION_SECONDS:
            print(f'[slow position] {name} ply {ply}: {took:.0f}s -> {sign}', flush=True)
        if sign == 'budget':
            cuts += 1
        elif sign is not None and sign not in found:
            found[sign] = ply
    return {'found': found, 'budget_cuts': cuts, 'seconds': round(perf_counter() - started, 1)}


def _scan_chunk(tasks):
    return [scan_game(t) for t in tasks]


def select(games: list[dict], found: list[dict], exclude: set[str], per_sign: bool = False) -> list[dict]:
    """At most one probe per game, balancing the signs in game order; with ``per_sign`` (dev), at most one
    probe per game and sign (no balancing)."""
    probes, taken, count = [], set(exclude), {'loss': 0, 'win': 0}
    prefix = 'D' if per_sign else 'F'
    for game, hits in zip(games, found):
        options = []
        for sign in sorted(hits, key=lambda s: (count[s], s != 'loss')):
            ply = hits[sign]
            key = canonical_hash(game['moves'][:ply])
            if key not in taken:
                options.append((sign, ply, key))
        for sign, ply, key in (options if per_sign else options[:1]):
            taken.add(key)
            count[sign] += 1
            probes.append({'name': f"{prefix}{len(probes) + 1:02d}", 'game_key': game['key'], 'seed': game['seed'],
                           'ply': ply, 'moves': game['moves'][:ply],
                           'side_to_move': 'black' if ply % 2 == 0 else 'white',
                           'expected_sign': 1 if sign == 'win' else -1,
                           'kind': 'vct1_win' if sign == 'win' else 'vct_loss', 'canonical_hash': key})
    return probes


def _is_dev_file(path) -> bool:
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get('purpose') == 'dev'


def load_games(paths: list[Path], seeds: tuple = FRESH_PROBE_SEEDS) -> tuple[list[dict], dict]:
    games, sources = [], {}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('arm') != ENGINE_ARM:
            raise ValueError(f"{path}: arm {data.get('arm')!r} is not TB-Engine-v1 ({ENGINE_ARM})")
        if int(data['seed']) not in seeds:
            raise ValueError(f"{path}: seed {data['seed']} is not one of the probe seeds {seeds}")
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
    parser.add_argument('--jsonl', type=Path, help='per-game scan results, appended as games finish (resume)')
    parser.add_argument('--dev', action='store_true',
                        help="dev deep probes from the dev seeds (H6-2a' option B), one probe per game and sign")
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists: the fresh probes are made once')
    paths = [Path(p) for pattern in args.games for p in (sorted(glob.glob(pattern)) or [pattern])]
    games, sources = load_games(paths, DEV_PROBE_SEEDS + DEV_PROBE_EXTENSION_SEEDS if args.dev else FRESH_PROBE_SEEDS)
    done = {}
    if args.jsonl is not None and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get('from_ply') == args.from_ply and row.get('scan_version') in ACCEPTED_SCAN_VERSIONS:
                    done[row['key']] = row
    todo = [i for i, g in enumerate(games) if g['key'] not in done]
    print(f'[fresh probes] {len(games)} games, {len(games) - len(todo)} already scanned', flush=True)
    chunks = [[(games[i]['moves'], args.from_ply, games[i]['key'])] for i in todo]

    def record(j, result):
        key = games[todo[j]]['key']
        row = {'key': key, 'from_ply': args.from_ply, 'scan_version': SCAN_VERSION, **result[0]}
        done[key] = row
        if args.jsonl is not None:
            args.jsonl.parent.mkdir(parents=True, exist_ok=True)
            with args.jsonl.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')

    progress = Progress('fresh probes', len(todo), 'games')
    if args.workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            run_chunks(pool, _scan_chunk, chunks, progress, len, unit_name='games', on_result=record)
    else:
        for j, c in enumerate(chunks):
            record(j, _scan_chunk(c))
            progress.update(1)
    found = [{k: int(v) for k, v in done[g['key']]['found'].items()} for g in games]
    files = [p for p in default_probe_files(ROOT) if Path(p).resolve() != args.output.resolve()]
    if args.dev:  # a rebuilt dev set (e.g. with the extension seeds) keeps the probes of an earlier dev set
        files = [p for p in files if not _is_dev_file(p)]
    exclude = probe_hashes(files)
    probes = select(games, found, exclude, per_sign=args.dev)
    counts = {'win': sum(p['expected_sign'] == 1 for p in probes), 'loss': sum(p['expected_sign'] == -1 for p in probes),
              'games_scanned': len(games), 'games_with_win': sum('win' in f for f in found),
              'games_with_loss': sum('loss' in f for f in found),
              'positions_over_budget': sum(done[g['key']]['budget_cuts'] for g in games),
              'games_with_budget_cuts': sum(done[g['key']]['budget_cuts'] > 0 for g in games),
              'games_by_scan_version': {str(v): sum(done[g['key']]['scan_version'] == v for g in games)
                                        for v in ACCEPTED_SCAN_VERSIONS}}
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    payload = {'format': FORMAT, 'purpose': 'dev' if args.dev else 'final', 'sealed': not args.dev,
               'git_commit': commit, 'sources': sources,
               'selection': {'from_ply': args.from_ply, 'one_probe_per_game': not args.dev,
                             'one_probe_per_game_and_sign': args.dev,
                             'balance': None if args.dev else 'fewer sign first',
                             'depth0_excluded': list(DEPTH0_KINDS), 'vct1_budget': VCT1_BUDGET,
                             'defense_max_candidates': DEFENSE_MAX_CANDIDATES, 'solver_node_limit': SOLVER_NODE_LIMIT,
                             'vcf_call_budget_per_position': VCF_CALL_BUDGET,
                             'vcf_node_budget_per_position': VCF_NODE_BUDGET, 'scan_version': SCAN_VERSION},
               'counts': counts, 'probes': probes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
