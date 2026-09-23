"""Run small pure-MCTS evaluation matches from the repository root."""
import argparse
from collections.abc import Callable

from agents import MCTSAgent, RandomAgent, TacticalAgent
from evaluation import run_match


def mcts_factory(simulations: int, candidate_limit: int) -> Callable[[int], MCTSAgent]:
    return lambda seed: MCTSAgent(
        seed=seed,
        simulations=simulations,
        candidate_limit=candidate_limit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1, help="games per matchup (default: 1)")
    parser.add_argument("--simulations", type=int, default=10, help="MCTS simulations per move")
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=8,
        help="maximum local MCTS candidates per node (default: 8)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-tactical",
        action="store_true",
        help="also run MCTS/Tactical in both colors",
    )
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.simulations <= 0:
        parser.error("--simulations must be positive")
    if args.candidate_limit <= 0:
        parser.error("--candidate-limit must be positive")

    mcts = mcts_factory(args.simulations, args.candidate_limit)
    matchups = [
        ("MCTS", mcts, "Random", RandomAgent),
        ("Random", RandomAgent, "MCTS", mcts),
    ]
    if args.include_tactical:
        matchups += [
            ("MCTS", mcts, "Tactical", TacticalAgent),
            ("Tactical", TacticalAgent, "MCTS", mcts),
        ]

    print(
        f"Seed: {args.seed}; games per matchup: {args.games}; "
        f"MCTS simulations/move: {args.simulations}; "
        f"candidate limit: {args.candidate_limit}",
        flush=True,
    )
    for black_name, black, white_name, white in matchups:
        print(f"\n=== {black_name} vs {white_name} (Black vs White) ===", flush=True)
        result = run_match(black, white, games=args.games, seed=args.seed)
        print(f"Games: {result.games}")
        print(f"Black wins: {result.black_wins}")
        print(f"White wins: {result.white_wins}")
        print(f"Draws: {result.draws}")
        print(f"Average moves: {result.average_moves:.2f}")
        print(f"Elapsed: {result.elapsed_seconds:.3f} s")
        print(f"Games/sec: {result.games_per_second:.4f}", flush=True)


if __name__ == "__main__":
    main()
