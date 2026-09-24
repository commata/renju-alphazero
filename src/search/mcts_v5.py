"""V5 root tactics: exact defensive legality and bounded threat discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from math import isfinite, sqrt
from random import Random

from renju import BLACK, EMPTY, SIZE, Game, IllegalMove
from renju.rules import DIRECTIONS, inside
from .mcts import (Move, MCTSNode, _is_legal_for_player, _wins_for_player,
                   _move_score, _backpropagate, _select_child)
from .mcts_v3 import _neighborhood_pool, _can_expand, _pop_ranked_untried
from .mcts_v4 import _validate_v4_config
from .mcts_v321 import (
    _best_immediate_win_fast, _fast_pattern_features_for_move,
    _search_candidates_v321, _rollout_v321,
    _fast_winning_extensions_in_direction, _v321_move_key,
)


WINDOWS = tuple(
    tuple((r + i * dr, c + i * dc) for i in range(5))
    for r in range(SIZE) for c in range(SIZE) for dr, dc in DIRECTIONS
    if inside(r + 4 * dr, c + 4 * dc)
)


def _threat_windows(game: Game, player: int, minimum: int = 3):
    for window in WINDOWS:
        values = [game.board[r][c] for r, c in window]
        if -player not in values and values.count(player) >= minimum:
            yield window


def _window_candidates(game: Game, player: int, minimum: int = 3) -> list[Move]:
    return sorted({
        pos for window in _threat_windows(game, player, minimum)
        for pos in window if game.board[pos[0]][pos[1]] == EMPTY
    })


def _winning_moves(game: Game, player: int) -> list[Move]:
    return [
        move for move in _window_candidates(game, player, 4)
        if _wins_for_player(game, player, move)
        and _is_legal_for_player(game, player, move)
    ]


def _placed_completions(game: Game, player: int, move: Move) -> set[Move]:
    return set().union(*(
        _fast_winning_extensions_in_direction(game.board, player, move, dr, dc)
        for dr, dc in DIRECTIONS
    ))


def _four_completions(game: Game, player: int, move: Move) -> set[Move]:
    """Winning extensions through a prospective stone, restoring the board."""
    r, c = move
    if not inside(r, c) or game.board[r][c] != EMPTY:
        return set()
    game.board[r][c] = player
    try:
        return _placed_completions(game, player, move)
    finally:
        game.board[r][c] = EMPTY


def _is_unstoppable_four(game: Game, player: int, move: Move) -> bool:
    r, c = move
    if not inside(r, c) or not _is_legal_for_player(game, player, move):
        return False
    completions = _four_completions(game, player, move)
    if not completions:
        return False
    game.board[r][c] = player
    try:
        if _winning_moves(game, -player):
            return False
        for block in sorted(completions):
            if not _is_legal_for_player(game, -player, block):
                continue
            br, bc = block
            game.board[br][bc] = -player
            try:
                if not _placed_completions(game, player, move):
                    return False
            finally:
                game.board[br][bc] = EMPTY
        return True
    finally:
        game.board[r][c] = EMPTY


def _unstoppable_four_moves(game: Game, player: int) -> list[Move]:
    return [move for move in _window_candidates(game, player)
            if _is_unstoppable_four(game, player, move)]


@dataclass
class SearchDiagnostics:
    best_root_tactical_score: int | None = None
    selected_simulations: int = 0
    forced_policy_stage: int | None = None
    stage5_multi_root_injection: bool = False
    simulation_mode: str | None = None
    root_candidates: tuple[Move, ...] = ()


@dataclass
class _RootContext:
    legal: list[Move]
    diagnostics: SearchDiagnostics
    injected: list[Move] = field(default_factory=list)
    keys: dict = field(default_factory=dict)

    def key(self, game: Game, move: Move):
        if move not in self.keys:
            self.keys[move] = _v321_move_key(game, move)
        return self.keys[move]


def _double_threat_moves(game: Game, player: int) -> list[Move]:
    result = []
    for move in _window_candidates(game, player, 2):
        features = _fast_pattern_features_for_move(game, player, move)
        if features.legal and (features.open_three_directions >= 2
                               or features.four_directions >= 2
                               or features.has_four_three):
            result.append(move)
    return result


def _forced_v5_move(game: Game, *, context: _RootContext | None = None) -> Move | None:
    context = context or _RootContext(game.legal_moves(), SearchDiagnostics())
    if not context.legal:
        raise IllegalMove("No legal moves available")
    legal = set(context.legal)
    player = game.to_play
    diag = context.diagnostics
    own_wins = _winning_moves(game, player)
    if own_wins:
        diag.forced_policy_stage = 1
        return _best_immediate_win_fast(game, player, own_wins)
    blocks = [move for move in _winning_moves(game, -player) if move in legal]
    if blocks:
        diag.forced_policy_stage = 2
        return min(blocks, key=lambda move: context.key(game, move))
    own = _unstoppable_four_moves(game, player)
    if own:
        diag.forced_policy_stage = 3
        return min(own, key=lambda move: context.key(game, move))
    creators = _unstoppable_four_moves(game, -player)
    if creators:
        defenses = set(creators)
        for window in _threat_windows(game, -player):
            if set(creators).intersection(window):
                defenses.update(pos for pos in window if game.board[pos[0]][pos[1]] == EMPTY)
        defenses &= legal
        if defenses:
            remaining = {}
            for move in sorted(defenses):
                r, c = move
                game.board[r][c] = player
                try:
                    remaining[move] = len(_unstoppable_four_moves(game, -player))
                finally:
                    game.board[r][c] = EMPTY
            diag.forced_policy_stage = 4
            return min(defenses, key=lambda move: (remaining[move], context.key(game, move)))
    danger = _double_threat_moves(game, -player)
    if len(danger) == 1 and danger[0] in legal:
        diag.forced_policy_stage = 5
        return danger[0]
    if len(danger) >= 2:
        context.injected = [move for move in danger if move in legal]
        diag.stage5_multi_root_injection = bool(context.injected)
    return None


V5_PRESETS = {
    "a": dict(simulations=50, tactical_simulations=50, tactical_score_threshold=None,
              exploration=sqrt(2), candidate_limit=20, initial_width=8,
              neighborhood_radius=2, priority_top_k=8),
    "b": dict(simulations=80, tactical_simulations=150, tactical_score_threshold=600,
              exploration=1.0, candidate_limit=20, initial_width=8,
              neighborhood_radius=2, priority_top_k=8),
    "c": dict(simulations=80, tactical_simulations=150, tactical_score_threshold=600,
              exploration=1.0, candidate_limit=14, initial_width=6,
              neighborhood_radius=2, priority_top_k=6),
}


def _validate_v5_config(*, simulations, tactical_simulations, tactical_score_threshold,
                        exploration, candidate_limit, initial_width,
                        neighborhood_radius, priority_top_k):
    if not isinstance(exploration, (int, float)) or not isfinite(exploration):
        raise ValueError("exploration must be finite and positive")
    _validate_v4_config(simulations, exploration, candidate_limit, initial_width,
                        neighborhood_radius, priority_top_k)
    if type(tactical_simulations) is not int or tactical_simulations < 1:
        raise ValueError("tactical_simulations must be a positive integer")
    if tactical_score_threshold is not None and type(tactical_score_threshold) is not int:
        raise ValueError("tactical_score_threshold must be None or int")


def _root_candidates_v5(game: Game, context: _RootContext, limit: int, radius: int):
    legal = set(context.legal)
    local = sorted((m for m in _neighborhood_pool(game, radius) if m in legal),
                   key=lambda m: _move_score(game, m))
    seen = set(local)
    pool = (local + [m for m in context.legal if m not in seen])[:limit * 2]
    for move in context.injected:
        if move not in pool:
            pool.append(move)
    ranked = sorted(pool, key=lambda m: context.key(game, m))
    injected = sorted(context.injected, key=lambda m: context.key(game, m))
    selected = injected + [m for m in ranked[:limit] if m not in injected]
    best_score = max(-context.keys[m][0] for m in selected)
    return selected, best_score


def mcts_search_v5(
    game: Game, *, simulations: int = 80, tactical_simulations: int = 150,
    tactical_score_threshold: int | None = 600, exploration: float = 1.0,
    candidate_limit: int = 14, initial_width: int = 6,
    neighborhood_radius: int = 2, priority_top_k: int = 6,
    random: Random | None = None, diagnostics: SearchDiagnostics | None = None,
) -> Move:
    """V3.2.1 search with V5 root policy and an adaptive simulation budget.

    An optional caller-owned diagnostic record avoids process-global state.
    Forced decisions use zero simulations and have no root tactical score.
    """
    _validate_v5_config(
        simulations=simulations, tactical_simulations=tactical_simulations,
        tactical_score_threshold=tactical_score_threshold, exploration=exploration,
        candidate_limit=candidate_limit, initial_width=initial_width,
        neighborhood_radius=neighborhood_radius, priority_top_k=priority_top_k,
    )
    diag = diagnostics if diagnostics is not None else SearchDiagnostics()
    diag.__dict__.update(vars(SearchDiagnostics()))
    context = _RootContext(game.legal_moves(), diag)
    forced = _forced_v5_move(game, context=context)
    if forced is not None:
        return forced
    root_moves, best_score = _root_candidates_v5(
        game, context, candidate_limit, neighborhood_radius,
    )
    tactical = tactical_score_threshold is not None and best_score >= tactical_score_threshold
    simulations = tactical_simulations if tactical else simulations
    diag.best_root_tactical_score = best_score
    diag.selected_simulations = simulations
    diag.simulation_mode = "tactical" if tactical else "normal"
    diag.root_candidates = tuple(root_moves)
    random = random or Random()
    root = MCTSNode(
        parent=None,
        move=None,
        player_just_moved=None,
        untried_moves=root_moves[:],
    )
    state = deepcopy(game)
    root_history_length = len(state.history)

    for _ in range(simulations):
        node = root

        while (
            not state.done
            and not _can_expand(node, initial_width)
            and node.children
        ):
            node = _select_child(node, exploration)
            assert node.move is not None
            state.play(*node.move)

        if not state.done and _can_expand(node, initial_width):
            move = _pop_ranked_untried(
                node,
                priority_top_k,
                random,
            )
            player = state.to_play
            state.play(*move)
            child = MCTSNode(
                parent=node,
                move=move,
                player_just_moved=player,
                untried_moves=(
                    []
                    if state.done
                    else _search_candidates_v321(
                        state,
                        candidate_limit,
                        neighborhood_radius,
                    )
                ),
            )
            node.children.append(child)
            node = child

        winner = (
            state.winner
            if state.done
            else _rollout_v321(
                state,
                random,
                candidate_limit,
                neighborhood_radius,
                priority_top_k,
            )
        )
        _backpropagate(node, winner)

        while len(state.history) > root_history_length:
            state.undo()

    max_visits = max(child.visits for child in root.children)
    candidates = [
        child
        for child in root.children
        if child.visits == max_visits
    ]
    max_value = max(child.mean_value for child in candidates)
    candidates = [
        child
        for child in candidates
        if child.mean_value == max_value
    ]
    chosen = random.choice(candidates)
    assert chosen.move is not None
    return chosen.move
