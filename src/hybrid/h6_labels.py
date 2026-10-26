"""H6-0 value-label contract (docs/mcts-v8-teacher.md §12.18, §12.19, §12.36; track-b-post-e1.md §4).

Infrastructure only: row schema, label rules, probe exclusion, split and leakage checks. It does not
choose a dataset; building the H6-1 label set is a separate, later step.

Row (``ROW_FORMAT``), one per labelled position (the position after ``ply`` moves of a game):

    game_id, source, split, ply, moves (the ``ply`` moves before the position), side_to_move,
    value_target (+1 / -1 / 0, side-to-move perspective), value_mask (True = no value loss),
    value_perspective = "side_to_move", label_kind, weight, canonical_hash (D4, side to move included),
    solver_class, solver_depth, solver_budget, engine_config_hash, model_hash, git_commit, flags

Label rules
- value is always from the side to move: a proven win +1, a proven loss -1, a draw 0 only for a game
  that really ended drawn (a full board); an agreed draw or an unknown end is masked.
- only proofs and real outcomes are labels: SAFE / NOT_FOUND / UNKNOWN / VCT2_CLEAR never are
  (they are masked rows only when a caller asks for them, never value targets).
- a proof beats the game result: the outcome rows of a game are masked from the first ply whose
  proof contradicts the result (``outcome_conflicts_proof``), e.g. a proven loss the opponent never
  punished.
- an outcome row whose own position has a proof is replaced by the proof row.

Leakage
- every game that reaches a probe / evaluation position (D4, side to move included) is dropped whole.
- splits are per game (``split_for``); positions of a later split whose canonical hash already occurs
  in an earlier split are masked (train < val < test), and ``check_leakage`` refuses any unmasked
  overlap that is left.
- within one game the engine config and model must not change (``check_game_provenance``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import glob
import hashlib
import json
from pathlib import Path

from renju import BLACK, Game

ROW_FORMAT = 'h6-value-row-v1'
PERSPECTIVE = 'side_to_move'
SPLIT_ORDER = ('train', 'val', 'test')
BOARD = 15
# Benchmark / evaluation seeds whose games never become training data (§12.19, §12.26, §12.31-12.35).
# 8601-8610: H6 fresh-final probe games (docs/h6-value.md, H6-2a'); reserved before any of them was played.
# 8611-8616: H6 dev deep-probe games (H6-2a' option B): 8611-8613 first, 8614-8616 only by the pre-declared
# extension rule; all six reserved before any of them was played, whether used or not.
# 8701-8725: external-engine anchor games (docs/external-engine.md §4, ``evaluation.external``), evaluation only.
# 8406: reserve H6-5 seed of the next value candidate (8404 / 8405 official), added after the 8403 REJECT.
RESERVED_SEEDS = frozenset({8401, 8402, 8403, 8404, 8405, 8406, 8411, 8412, 8413, 8414, 8415, 8416, 8417, 8418,
                            *range(8601, 8611), *range(8611, 8617), *range(8701, 8726),
                            8421, 8422, 8423,   # H7-E2 holdout (docs/h6-value.md)
                            8407, 8408, 8409,   # H7-E3 (value v3 on TB-Engine-v2)
                            8431, 8432,         # CPU-backend certification of TB-V2-Value + fast4 (pre-E4 record)
                            8441, 8442,         # H7-E4 arena (candidate vs v8:full)
                            8451, 8452,         # CPU-backend certification of the post-E4 engine
                            *range(8731, 8741),   # H7-c internal Arena: cycle c = 8730 + c (registered 2026-10-24)
                            *range(8741, 8750),  # H7-c search-budget ablation
                            *range(8751, 8761)})  # B2 drift only: cycle c = 8750+c (registered before B2)
FRESH_PROBE_SEEDS = tuple(range(8601, 8611))
DEV_PROBE_SEEDS = tuple(range(8611, 8614))
DEV_PROBE_EXTENSION_SEEDS = tuple(range(8614, 8617))
# Proof label kinds of ``analysis.tactical_labels`` that carry a value (side to move).
PROOF_VALUES = {'immediate_win': 1, 'unstoppable_four': 1, 'vcf': 1, 'forced_loss': -1, 'vcf_loss': -1}
PROOF_DEPTH = {'immediate_win': 0, 'unstoppable_four': 0, 'vcf': 0, 'forced_loss': 0, 'vcf_loss': 0,
               'vct1_win': 1}


class LeakageError(ValueError):
    pass


@dataclass
class Row:
    game_id: str
    source: str
    split: str
    ply: int
    moves: list
    side_to_move: str
    value_target: int | None
    value_mask: bool
    label_kind: str
    weight: float
    canonical_hash: str
    value_perspective: str = PERSPECTIVE
    solver_class: str | None = None
    solver_depth: int | None = None
    solver_budget: dict | None = None
    engine_config_hash: str | None = None
    model_hash: str | None = None
    git_commit: str | None = None
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {'format': ROW_FORMAT, **asdict(self)}


# -- positions -----------------------------------------------------------------

def _transform(move, symmetry: int):
    row, col = move
    if symmetry >= 4:
        col = BOARD - 1 - col
    for _ in range(symmetry % 4):
        row, col = BOARD - 1 - col, row
    return row, col


def canonical_hash(moves) -> str:
    """Stones + side to move, minimized over D4 (move order does not matter)."""
    black = [tuple(m) for m in moves[0::2]]
    white = [tuple(m) for m in moves[1::2]]
    key = min((len(moves) % 2, tuple(sorted(_transform(m, s) for m in black)),
               tuple(sorted(_transform(m, s) for m in white))) for s in range(8))
    return hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).hexdigest()[:24]


def side_name(ply: int) -> str:
    return 'black' if ply % 2 == 0 else 'white'


def outcome_value(winner: int | None, ply: int, *, real_draw: bool) -> int | None:
    """Side-to-move value at ``ply`` from the game result; None when the result is not a label."""
    if winner is None:
        return 0 if real_draw else None
    to_move = BLACK if ply % 2 == 0 else -BLACK
    return 1 if winner == to_move else -1


# -- proofs ----------------------------------------------------------------------

class BudgetExceeded(Exception):
    """A per-position solver budget ran out: no label (never a guess)."""


def budgeted_solver(*, node_limit: int, call_budget: int, node_budget: int):
    """``ThreatSolver`` whose uncached VCF calls and VCF nodes are capped for one position; a cut raises
    ``BudgetExceeded`` and is never cached. Deterministic (counts, not wall-clock time)."""
    from analysis.threats import (VCF_NONE, VCF_UNKNOWN, VCF_WIN, ThreatSolver, _board_key,
                                  _find_vcf_with_stats, _unbounded_max_fours)

    class Budgeted(ThreatSolver):
        nodes_used = 0

        def vcf(self, game, attacker):
            key = (_board_key(game), attacker)
            cached = self._vcf_cache.get(key)
            if cached is not None:
                return cached
            remaining = node_budget - self.nodes_used
            if self.vcf_calls >= call_budget or remaining < 1:
                raise BudgetExceeded
            limit = min(self.node_limit, remaining)
            self.vcf_calls += 1
            found, nodes, exhausted = _find_vcf_with_stats(
                game, attacker, max_fours=_unbounded_max_fours(game), node_limit=limit)
            self.nodes_used += nodes
            if found is not None:
                result = (VCF_WIN, tuple(found.attack_moves))
            elif exhausted:
                if limit < self.node_limit:
                    raise BudgetExceeded
                self.vcf_exhausted += 1
                result = (VCF_UNKNOWN, ())
            else:
                result = (VCF_NONE, ())
            self._vcf_cache[key] = result
            return result

    return Budgeted(node_limit=node_limit)


def vcf_loss_status(game: Game, *, node_limit: int, call_budget: int, node_budget: int) -> str | None:
    """``'loss'`` when every move of the side to move loses to the opponent's VCF (``vcf_loss``),
    ``'budget'`` when the per-position budget ran out, else None. Stops at the first surviving move, so it
    gives ``label_position(prove_losses=True)``'s verdict without classifying every legal move."""
    from analysis.threats import UNSAFE, decision_status

    solver = budgeted_solver(node_limit=node_limit, call_budget=call_budget, node_budget=node_budget)
    try:
        counts, _ = solver.decision(game, 0, stop_at_safe=True)
    except BudgetExceeded:
        return 'budget'
    return 'loss' if decision_status(counts) == UNSAFE else None


def proof_label(moves, solver, *, prove_losses: bool = False, vct1_budget: dict | None = None,
                loss_budget: dict | None = None, stats: dict | None = None) -> dict | None:
    """``{'kind', 'value', 'depth'}`` for the position after ``moves``, or None (no proof).

    VCF-class proofs come from ``analysis.tactical_labels.label_position``; with ``vct1_budget`` a
    proven depth-1 attack of the side to move (V8-B's candidates and check: every reply loses to
    our VCF) adds ``vct1_win`` (+1). Unproven or cut searches give no label.
    """
    from analysis.tactical_labels import label_position

    game = Game()
    for move in moves:
        game.play(*move)
    if game.done:
        return None
    label = label_position(game, solver, prove_losses=prove_losses and loss_budget is None)
    if label is not None and label['value'] is not None:
        return {'kind': label['kind'], 'value': label['value'], 'depth': PROOF_DEPTH[label['kind']]}
    if prove_losses and loss_budget is not None and label is None:
        # label_position checks vcf_loss only when it found nothing else (not after must_block): same here
        status = vcf_loss_status(game, **loss_budget)
        if status == 'loss':
            return {'kind': 'vcf_loss', 'value': -1, 'depth': PROOF_DEPTH['vcf_loss']}
        if status == 'budget' and stats is not None:
            stats['loss_budget_cuts'] = stats.get('loss_budget_cuts', 0) + 1
    if vct1_budget is not None:
        from analysis.mcts_v8 import WIN, SearchDiagnostics, _own_vct_attack
        from search.mcts_v5 import _RootContext

        diag = SearchDiagnostics()
        move = _own_vct_attack(game, _RootContext(game.legal_moves(), diag), diag, **vct1_budget)
        if move is not None and diag.v8_attack_status == WIN:
            return {'kind': 'vct1_win', 'value': 1, 'depth': 1}
    return None


# -- one game ----------------------------------------------------------------------

def game_rows(game: dict, *, source: str, split: str, outcome_weight: float, proof_weight: float = 1.0,
              proofs: dict[int, dict] | None = None, solver_class: str | None = None,
              solver_budget: dict | None = None, provenance: dict | None = None,
              from_ply: int = 0) -> list[Row]:
    """Rows of one game: outcome rows (masked where not a label) with proofs taking precedence.

    ``game``: ``{'id', 'moves', 'winner' (BLACK / WHITE / None), 'real_draw' (bool)}``;
    ``proofs``: ply -> ``proof_label`` result for the positions that have one.
    """
    proofs = proofs or {}
    provenance = provenance or {}
    moves = [list(m) for m in game['moves']]
    winner = game.get('winner')
    conflict_from = None
    for ply in sorted(proofs):
        implied = BLACK if (ply % 2 == 0) == (proofs[ply]['value'] > 0) else -BLACK
        if winner != implied:
            conflict_from = ply
            break
    rows = []
    for ply in range(from_ply, len(moves)):  # the final position (after the last move) has no move to label
        prefix = moves[:ply]
        common = dict(game_id=str(game['id']), source=source, split=split, ply=ply, moves=prefix,
                      side_to_move=side_name(ply), canonical_hash=canonical_hash(prefix),
                      engine_config_hash=provenance.get('engine_config_hash'),
                      model_hash=provenance.get('model_hash'), git_commit=provenance.get('git_commit'))
        if ply in proofs:
            p = proofs[ply]
            rows.append(Row(**common, value_target=p['value'], value_mask=False, label_kind=f"proof_{p['kind']}",
                            weight=proof_weight, solver_class=solver_class, solver_depth=p['depth'],
                            solver_budget=solver_budget))
            continue
        value = outcome_value(winner, ply, real_draw=bool(game.get('real_draw')))
        flags = []
        if conflict_from is not None and ply >= conflict_from:
            flags.append('outcome_conflicts_proof')
        masked = value is None or bool(flags)
        rows.append(Row(**common, value_target=value, value_mask=masked,
                        label_kind='outcome' if value is not None else 'unlabelled',
                        weight=0.0 if masked else outcome_weight, flags=flags))
    return rows


# -- probes and splits -------------------------------------------------------------

def probe_positions(paths, root: Path | None = None) -> dict[str, dict]:
    """Every probe / evaluation position in the given files: canonical hash -> ``{'ply', 'file', 'label'}``
    (the smallest ply when a position occurs twice).

    Understood layouts: ``{'probes': [{'moves'}]}`` (test fixtures), ``{'moves', 'probes': {name:
    {'plies_played'}}}`` (web-game probes) and ``{'positions': [{'history'}]}`` (E1 manifests).
    """
    out = {}

    def add(moves, path, label):
        key = canonical_hash(moves)
        name = str(Path(path).resolve().relative_to(root)) if root and Path(path).resolve().is_relative_to(root) \
            else str(path)
        if key not in out or len(moves) < out[key]['ply']:
            out[key] = {'ply': len(moves), 'file': name.replace('\\', '/'), 'label': label}

    for path in paths:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        probes = data.get('probes')
        if isinstance(probes, list):
            for i, p in enumerate(probes):
                if 'moves' in p:
                    add(p['moves'], path, str(p.get('name') or p.get('kind') or i))
        elif isinstance(probes, dict) and 'moves' in data:
            for name, p in probes.items():
                add(data['moves'][:p['plies_played']], path, name)
        for i, position in enumerate(data.get('positions', [])):
            if 'history' in position:
                add(position['history'], path, str(position.get('key', i)))
    return out


def probe_hashes(paths) -> set[str]:
    """Canonical hashes of every probe / evaluation position in the given files (``probe_positions``)."""
    return set(probe_positions(paths))


def first_probe(moves, probes) -> tuple[int, str] | None:
    """``(ply, hash)`` of the first prefix of ``moves`` that is a probe position, or None."""
    for ply in range(len(moves) + 1):
        key = canonical_hash(moves[:ply])
        if key in probes:
            return ply, key
    return None


def default_probe_files(root: Path) -> list[str]:
    files = sorted(glob.glob(str(root / 'tests' / 'fixtures' / '*probes*.json')))
    files += sorted(glob.glob(str(root / 'docs' / 'mcts-v8-results' / 'probe_*.json')))
    files += sorted(glob.glob(str(root / 'docs' / 'mcts-v8-results' / 'e1*' / '*manifest.json')))
    files += sorted(glob.glob(str(root / 'docs' / 'h6' / '*fresh*probes*.json')))
    return files


def touches_probe(moves, probes: set[str]) -> bool:
    return any(canonical_hash(moves[:ply]) in probes for ply in range(len(moves) + 1))


def split_for(game_id: str, fractions=(('test', 0.05), ('val', 0.05)), salt: str = 'h6-v1') -> str:
    """Deterministic per-game split for sources without their own (self-play, benchmark-free logs)."""
    x = int(hashlib.sha256(f'{salt}:{game_id}'.encode()).hexdigest()[:12], 16) / 16 ** 12
    edge = 0.0
    for name, fraction in fractions:
        edge += fraction
        if x < edge:
            return name
    return 'train'


def mask_cross_split(rows: list[Row]) -> dict:
    """Mask rows of a later split whose position occurs in an earlier split (train < val < test)."""
    seen: set[str] = set()
    masked = {}
    for split in SPLIT_ORDER:
        here = [r for r in rows if r.split == split]
        count = 0
        for row in here:
            if split != 'train' and row.canonical_hash in seen and not row.value_mask:
                row.value_mask, row.weight = True, 0.0
                row.flags.append('cross_split_duplicate')
                count += 1
        seen |= {r.canonical_hash for r in here}
        masked[split] = count
    return masked


def check_leakage(rows: list[Row]) -> None:
    """Refuse when an unmasked position appears in two splits."""
    keys = {s: {r.canonical_hash for r in rows if r.split == s and not r.value_mask} for s in SPLIT_ORDER}
    all_keys = {s: {r.canonical_hash for r in rows if r.split == s} for s in SPLIT_ORDER}
    bad = {}
    for i, a in enumerate(SPLIT_ORDER):
        for b in SPLIT_ORDER[i + 1:]:
            n = len(all_keys[a] & keys[b])
            if n:
                bad[f'{a}_{b}'] = n
    if bad:
        raise LeakageError(f'unmasked positions shared across splits: {bad}')


def check_game_provenance(rows: list[Row]) -> None:
    """Refuse when one game mixes engine configs or models (H7 rule, §12.18)."""
    seen: dict[str, tuple] = {}
    for row in rows:
        ident = (row.engine_config_hash, row.model_hash)
        if seen.setdefault(row.game_id, ident) != ident:
            raise ValueError(f'game {row.game_id} mixes engine/model provenance: {seen[row.game_id]} vs {ident}')


def check_reserved_seed(seed: int | None) -> None:
    if seed is not None and int(seed) in RESERVED_SEEDS:
        raise ValueError(f'seed {seed} is a benchmark / evaluation seed; its games are never training data')


def report(rows: list[Row]) -> dict:
    """Counts the H6-1 gate asks for: per source, kind, depth, sign, side and split."""
    out = {'rows': len(rows), 'labelled': 0, 'by_split': {}, 'by_kind': {}, 'by_depth': {}, 'sign': {},
           'sign_by_side': {}, 'masked_by_flag': {}}
    for r in rows:
        out['by_split'][r.split] = out['by_split'].get(r.split, 0) + 1
        for flag in r.flags:
            out['masked_by_flag'][flag] = out['masked_by_flag'].get(flag, 0) + 1
        if r.value_mask:
            continue
        out['labelled'] += 1
        out['by_kind'][r.label_kind] = out['by_kind'].get(r.label_kind, 0) + 1
        if r.solver_depth is not None:
            key = str(r.solver_depth)
            out['by_depth'][key] = out['by_depth'].get(key, 0) + 1
        sign = {1: '+1', -1: '-1', 0: '0'}[r.value_target]
        out['sign'][sign] = out['sign'].get(sign, 0) + 1
        side = out['sign_by_side'].setdefault(r.side_to_move, {})
        side[sign] = side.get(sign, 0) + 1
    return out
