"""Measure the unmodified engine; setup and optional profiling are not timed."""
import argparse
from collections.abc import Callable
from copy import deepcopy
import cProfile
import platform
import pstats
from time import perf_counter

from agents import RandomAgent
from evaluation import play_game, run_match
from renju import Game
from renju.rules import forbidden_reason


def prepare_midgame(seed: int) -> tuple[Game, Game, int]:
    """Return legal ongoing positions at 60 (black) and 61 (white) plies."""
    for attempt in range(100):
        game = Game()
        agent = RandomAgent(seed + attempt)
        for _ in range(60):
            game.play(*agent.select_move(game))
            if game.done:
                break
        if game.done:
            continue
        black = deepcopy(game)
        game.play(*agent.select_move(game))
        if not game.done:
            return black, game, seed + attempt
    raise RuntimeError("Could not prepare a live 61-ply midgame in 100 attempts")


def measure(label: str, operation: Callable[[], object], iterations: int) -> None:
    started = perf_counter()
    for _ in range(iterations):
        operation()
    elapsed = perf_counter() - started
    rate = iterations / elapsed if elapsed else 0.0
    print(f"{label}: iterations={iterations}, total={elapsed:.6f} s, "
          f"mean={elapsed / iterations * 1000:.6f} ms, ops/sec={rate:.3f}", flush=True)


def play_undo(game: Game, move: tuple[int, int]) -> None:
    game.play(*move)
    game.undo()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true",
                        help="separately profile one additional random game (not timed above)")
    args = parser.parse_args()
    if args.iterations <= 0 or args.games <= 0:
        parser.error("--iterations and --games must be positive")

    print(f"Python: {platform.python_version()}; platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Seed: {args.seed}; preparing positions outside measurement...", flush=True)
    empty = Game()
    opening_white = Game()
    opening_white.play(7, 7)
    mid_black, mid_white, position_seed = prepare_midgame(args.seed)
    black_move = mid_black.legal_moves()[0]
    white_move = mid_white.legal_moves()[0]
    print(f"Midgame seed: {position_seed}; black plies: {len(mid_black.history)}; "
          f"white plies: {len(mid_white.history)}", flush=True)
    print("play+undo counts one pair as one operation; legal_moves counts one full list.")
    measure("play+undo / midgame black", lambda: play_undo(mid_black, black_move), args.iterations)
    measure("play+undo / midgame white", lambda: play_undo(mid_white, white_move), args.iterations)
    measure("legal_moves / empty black", empty.legal_moves, args.iterations)
    # A white-to-play empty board is not reachable from a normal game.
    measure("legal_moves / opening white (1 ply)", opening_white.legal_moves, args.iterations)
    measure("legal_moves / midgame black (60 plies)", mid_black.legal_moves, args.iterations)
    measure("legal_moves / midgame white (61 plies)", mid_white.legal_moves, args.iterations)
    measure("forbidden_reason / empty (7, 7)",
            lambda: forbidden_reason(empty.board, 7, 7), args.iterations)
    measure(f"forbidden_reason / midgame {black_move}",
            lambda: forbidden_reason(mid_black.board, *black_move), args.iterations)

    print("\n=== Random vs Random full games ===", flush=True)
    result = run_match(RandomAgent, RandomAgent, games=args.games, seed=args.seed)
    print(f"Games: {result.games}")
    print(f"Elapsed: {result.elapsed_seconds:.6f} s")
    print(f"Games/sec: {result.games_per_second:.4f}")
    print(f"Total moves: {result.total_moves}")
    rate = result.total_moves / result.elapsed_seconds if result.elapsed_seconds else 0.0
    print(f"Moves/sec: {rate:.3f}")
    print(f"Average moves: {result.average_moves:.2f}", flush=True)

    if args.profile:
        print("\n=== Separate cProfile random game (includes profiler overhead) ===", flush=True)
        black_agent = RandomAgent(args.seed)
        white_agent = RandomAgent(args.seed + 1)
        profiler = cProfile.Profile()
        profiler.runcall(play_game, black_agent, white_agent)
        pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":
    main()
