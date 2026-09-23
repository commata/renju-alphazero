"""Sequential matches with independent, reproducible per-game agent seeds."""
from collections.abc import Callable
from dataclasses import dataclass
from random import Random
from time import perf_counter

from agents import Agent
from renju import BLACK, WHITE, Game, IllegalMove


@dataclass(frozen=True)
class GameResult:
    winner: int | None
    number_of_moves: int
    elapsed_seconds: float
    black_agent: str
    white_agent: str
    history: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MatchResult:
    games: int
    black_wins: int
    white_wins: int
    draws: int
    total_moves: int
    average_moves: float
    elapsed_seconds: float
    games_per_second: float
    results: tuple[GameResult, ...]
    seed: int


def play_game(black_agent: Agent, white_agent: Agent) -> GameResult:
    game = Game()
    started = perf_counter()
    while not game.done:
        agent = black_agent if game.to_play == BLACK else white_agent
        move = agent.select_move(game)
        if (not isinstance(move, tuple) or len(move) != 2
                or any(type(value) is not int for value in move)):
            raise IllegalMove(f"{agent.name} returned an invalid move: {move!r}")
        try:
            game.play(*move)
        except IllegalMove as exc:
            raise IllegalMove(f"{agent.name} returned illegal move {move}: {exc}") from exc
    return GameResult(
        winner=game.winner,
        number_of_moves=len(game.history),
        elapsed_seconds=perf_counter() - started,
        black_agent=black_agent.name,
        white_agent=white_agent.name,
        history=tuple(game.history),
    )


def run_match(
    black_factory: Callable[[int], Agent],
    white_factory: Callable[[int], Agent],
    games: int = 100,
    seed: int = 42,
) -> MatchResult:
    """Factories receive a seed positionally; new agents and board for each game.

    Agent construction is excluded from timing. Histories and outcomes, not
    elapsed times, are the reproducibility contract.
    """
    if type(games) is not int or games <= 0:
        raise ValueError("games must be a positive integer")
    random = Random(seed)
    pairs = [(black_factory(random.getrandbits(64)), white_factory(random.getrandbits(64)))
             for _ in range(games)]
    started = perf_counter()
    results = tuple(play_game(black, white) for black, white in pairs)
    elapsed = perf_counter() - started
    total_moves = sum(result.number_of_moves for result in results)
    return MatchResult(
        games=games,
        black_wins=sum(result.winner == BLACK for result in results),
        white_wins=sum(result.winner == WHITE for result in results),
        draws=sum(result.winner is None for result in results),
        total_moves=total_moves,
        average_moves=total_moves / games,
        elapsed_seconds=elapsed,
        games_per_second=games / elapsed if elapsed else 0.0,
        results=results,
        seed=seed,
    )
