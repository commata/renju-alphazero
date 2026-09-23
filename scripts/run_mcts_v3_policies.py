"""Compare MCTS V3.1 against V3.2 in both colors."""
import argparse
from collections.abc import Callable

from agents import MCTSV3Agent, MCTSV32Agent
from evaluation import run_match


def v31_factory(
    simulations: int,
    candidate_limit: int,
    initial_width: int,
    radius: int,
    priority_top_k: int,
) -> Callable[[int], MCTSV3Agent]:
    return lambda seed: MCTSV3Agent(
        seed=seed,
        simulations=simulations,
        candidate_limit=candidate_limit,
        initial_width=initial_width,
        neighborhood_radius=radius,
        priority_top_k=priority_top_k,
    )


def v32_factory(
    simulations: int,
    candidate_limit: int,
    initial_width: int,
    radius: int,
    priority_top_k: int,
) -> Callable[[int], MCTSV32Agent]:
    return lambda seed: MCTSV32Agent(
        seed=seed,
        simulations=simulations,
        candidate_limit=candidate_limit,
        initial_width=initial_width,
        neighborhood_radius=radius,
        priority_top_k=priority_top_k,
    )


def report(label: str, result) -> None:
    print(f"\n=== {label} ===")
    print(f"Games: {result.games}")
    print(f"Black wins: {result.black_wins}")
    print(f"White wins: {result.white_wins}")
    print(f"Draws: {result.draws}")
    print(f"Average moves: {result.average_moves:.2f}")
    print(f"Elapsed: {result.elapsed_seconds:.3f} s")
    print(f"Games/sec: {result.games_per_second:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=5, help="games per color assignment")
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--candidate-limit", type=int, default=16)
    parser.add_argument("--initial-width", type=int, default=6)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--priority-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for value, name in (
        (args.games, "--games"),
        (args.simulations, "--simulations"),
        (args.candidate_limit, "--candidate-limit"),
        (args.initial_width, "--initial-width"),
        (args.radius, "--radius"),
        (args.priority_top_k, "--priority-top-k"),
    ):
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.initial_width > args.candidate_limit:
        parser.error("--initial-width must not exceed --candidate-limit")
    if args.priority_top_k > args.candidate_limit:
        parser.error("--priority-top-k must not exceed --candidate-limit")

    v31 = v31_factory(
        args.simulations,
        args.candidate_limit,
        args.initial_width,
        args.radius,
        args.priority_top_k,
    )
    v32 = v32_factory(
        args.simulations,
        args.candidate_limit,
        args.initial_width,
        args.radius,
        args.priority_top_k,
    )

    print(
        f"Seed: {args.seed}; games/color: {args.games}; "
        f"simulations/candidates: {args.simulations}/{args.candidate_limit}; "
        f"initial width: {args.initial_width}; radius: {args.radius}; "
        f"priority top-k: {args.priority_top_k}",
        flush=True,
    )

    v31_black = run_match(v31, v32, games=args.games, seed=args.seed)
    report("MCTS-v3.1 vs MCTS-v3.2 (Black vs White)", v31_black)

    v32_black = run_match(v32, v31, games=args.games, seed=args.seed)
    report("MCTS-v3.2 vs MCTS-v3.1 (Black vs White)", v32_black)

    v31_wins = v31_black.black_wins + v32_black.white_wins
    v32_wins = v31_black.white_wins + v32_black.black_wins
    draws = v31_black.draws + v32_black.draws
    print("\n=== Combined policy comparison ===")
    print(f"MCTS-v3.1 wins: {v31_wins}")
    print(f"MCTS-v3.2 wins: {v32_wins}")
    print(f"Draws: {draws}")
    print(f"Total games: {args.games * 2}")


if __name__ == "__main__":
    main()
