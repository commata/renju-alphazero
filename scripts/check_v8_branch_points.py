"""Check which V8 module would have changed v7's losing move in the human games.

Read-only analysis for docs/mcts-v8-teacher.md §2. For each v7 branch move (the
last move that still had a VCT1-safe alternative, from tests/fixtures/vct_probes_v1.json)
it reports:

- the forced-policy stage v7 used and the Stage 4 defense set,
- whether the VCT1-safe moves were in the Stage 4 set / V5 root / V6 root candidates,
- whether a both-colour future-threat planner (``future_setups`` for the opponent)
  flags the human's winning threat or its defenses,
- whether Stage-5 multi-root injection is active at the branch point,
- the VCT1 status of every Stage 4 defense (``--root-vct`` also checks the V6 root
  candidates of the MCTS case, which takes tens of minutes).

The script preserves the same ``_RootContext`` across ``_forced_v5_move`` and
``_root_candidates_v6``, matching the actual V6/V7 call path.  ``--structure-only``
is a cheap way to re-check candidate membership/order without rerunning VCT proofs.

    python scripts/check_v8_branch_points.py [--structure-only] [--root-vct] [--node-limit 100000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from analysis.threats import ThreatSolver  # noqa: E402
from renju import BLACK, EMPTY, Game  # noqa: E402
from search.mcts_v5 import (  # noqa: E402
    _RootContext, _forced_v5_move, _root_candidates_v5, _threat_windows, _unstoppable_four_moves,
)
from search.mcts_v6 import _root_candidates_v6  # noqa: E402
from search.mcts_v7 import SearchDiagnostics  # noqa: E402
from search.threat_planning import future_setups  # noqa: E402

GAMES = ROOT / 'tests/fixtures/web_play_v7_human_games_v1.json'
PROBES = ROOT / 'tests/fixtures/vct_probes_v1.json'
# (game name prefix, 0-based index of v7's branch move)
BRANCH_POINTS = [
    ('20260929-163810', 14),
    ('20260929-163903', 12),
    ('20260929-164445', 13),
    ('20260929-164723', 19),  # MCTS 100 move: >40 VCF-safe candidates, no complete defense probe
]


def fixture_safe_moves(probes: dict, prefix: str, index: int) -> set | None:
    """Return symmetry-0 must_defend_vct correct moves for this branch, if complete."""
    matches = [
        probe for probe in probes['probes']
        if probe['kind'] == 'must_defend_vct'
        and probe.get('symmetry') == 0
        and probe['source']['game'].startswith(prefix)
        and probe['source']['ply'] == index + 1
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f'ambiguous VCT probe for {prefix} ply {index + 1}: {len(matches)}')
    return {tuple(move) for move in matches[0]['correct_moves']}


def stage4_defenses(game: Game, legal: set) -> set:
    opponent = -game.to_play
    creators = set(_unstoppable_four_moves(game, opponent))
    if not creators:
        return set()
    defenses = set(creators)
    for window in _threat_windows(game, opponent):
        if creators.intersection(window):
            defenses.update(p for p in window if game.board[p[0]][p[1]] == EMPTY)
    return defenses & legal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root-vct', action='store_true',
                        help='also VCT1-check the V6 root candidates of the MCTS case (slow)')
    parser.add_argument('--structure-only', action='store_true',
                        help='skip VCT classification; only replay stages/planner/root candidates')
    parser.add_argument('--node-limit', type=int, default=100_000)
    args = parser.parse_args()
    games = {g['name']: g for g in json.loads(GAMES.read_text(encoding='utf-8'))['games']}
    probes = json.loads(PROBES.read_text(encoding='utf-8'))
    for prefix, index in BRANCH_POINTS:
        safe = fixture_safe_moves(probes, prefix, index)
        name = next(n for n in games if n.startswith(prefix))
        moves = [tuple(m) for m in games[name]['moves']]
        game = Game()
        for move in moves[:index]:
            game.play(*move)
        opponent = -game.to_play
        context = _RootContext(game.legal_moves(), SearchDiagnostics())
        _forced_v5_move(game, context=context)
        s4 = stage4_defenses(game, set(context.legal))
        # Keep the same context: Stage 5 may have populated context.injected, and
        # the production V6/V7 path passes that context into root generation.
        base, _ = _root_candidates_v5(game, context, 20, 2)
        v6, _, _ = _root_candidates_v6(game, context, 20, 2)
        setups = future_setups(game, opponent)
        opp_defenses = set().union(*(s.defenses for s in setups.values())) if setups else set()
        chosen, human = moves[index], moves[index + 1]
        colour = 'BLACK' if game.to_play == BLACK else 'WHITE'
        print(f'== {name}  v7={colour}  ply {index + 1}  route={games[name]["ai_route"][index]}'
              f'  stage={context.diagnostics.forced_policy_stage}')
        print(f'  v7 move {chosen}  human reply {human}  (0-based)')
        print(f'  stage4 defenses {sorted(s4)}')
        print(f'  stage5 injection {context.diagnostics.stage5_multi_root_injection}: '
              f'{sorted(context.injected)}; V5 root size {len(base)}, V6 root size {len(v6)}')
        if safe is not None:
            print(f'  VCT1-safe {sorted(safe)}: in stage4 {sorted(safe & s4)}, in V5 root '
                  f'{sorted(safe & set(base))}, in V6 root {sorted(safe & set(v6))} (V6 root size {len(v6)})')
        print(f'  opponent future setups {sorted(setups)}; human reply flagged: {human in setups}; '
              f'VCT1-safe among their defenses: {sorted((safe or set()) & opp_defenses)}')
        candidates = [] if args.structure_only else (
            sorted(s4) if s4 else (list(v6) if args.root_vct else [])
        )
        if candidates:
            solver = ThreatSolver(node_limit=args.node_limit)
            started = perf_counter()
            result = solver.classify(game, candidates, vct_depth=1)
            print(f'  VCT1 over {len(candidates)} candidates: {perf_counter() - started:.1f}s '
                  f'(VCF calls {solver.vcf_calls}, exhausted {solver.vcf_exhausted})')
            print('   ', {m: status for m, (status, _) in result.items()})


if __name__ == '__main__':
    main()
