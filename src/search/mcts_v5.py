"""V5 root tactics: exact defensive legality and bounded threat discovery."""
from __future__ import annotations

from dataclasses import dataclass, field

from renju import BLACK, EMPTY, SIZE, Game, IllegalMove
from renju.rules import DIRECTIONS, inside
from .mcts import Move, _is_legal_for_player, _wins_for_player
from .mcts_v321 import (
    _best_immediate_win_fast, _fast_pattern_features_for_move,
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
