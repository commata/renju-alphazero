"""Compare MCTS V4.1 and V4.2 with the same 50-simulation budget."""
import argparse
from collections.abc import Callable
from pathlib import Path

from agents import MCTSV41Agent, MCTSV42Agent
from evaluation import default_log_dir, run_match, save_match_logs


def agent_factory(agent_type, args, version: str) -> Callable[[int], object]:
    if version == "v41":
        candidate_limit = args.v41_candidate_limit
        initial_width = args.v41_initial_width
        priority_top_k = args.v41_priority_top_k
    else:
        candidate_limit = args.v42_candidate_limit
        initial_width = args.v42_initial_width
        priority_top_k = args.v42_priority_top_k

    return lambda seed: agent_type(
        seed=seed,
        simulations=args.simulations,
        candidate_limit=candidate_limit,
        initial_width=initial_width,
        neighborhood_radius=args.radius,
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
    parser.add_argument("--games", type=int, default=25, help="games per color assignment")
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--radius", type=int, default=2)

    parser.add_argument("--v41-candidate-limit", type=int, default=20)
    parser.add_argument("--v41-initial-width", type=int, default=8)
    parser.add_argument("--v41-priority-top-k", type=int, default=8)

    parser.add_argument("--v42-candidate-limit", type=int, default=24)
    parser.add_argument("--v42-initial-width", type=int, default=10)
    parser.add_argument("--v42-priority-top-k", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=None)
    args = parser.parse_args()

    values = (
        (args.games, "--games"),
        (args.simulations, "--simulations"),
        (args.radius, "--radius"),
        (args.v41_candidate_limit, "--v41-candidate-limit"),
        (args.v41_initial_width, "--v41-initial-width"),
        (args.v41_priority_top_k, "--v41-priority-top-k"),
        (args.v42_candidate_limit, "--v42-candidate-limit"),
        (args.v42_initial_width, "--v42-initial-width"),
        (args.v42_priority_top_k, "--v42-priority-top-k"),
    )
    for value, name in values:
        if value <= 0:
            parser.error(f"{name} must be positive")

    if args.v41_initial_width > args.v41_candidate_limit:
        parser.error("--v41-initial-width must not exceed --v41-candidate-limit")
    if args.v41_priority_top_k > args.v41_candidate_limit:
        parser.error("--v41-priority-top-k must not exceed --v41-candidate-limit")
    if args.v42_initial_width > args.v42_candidate_limit:
        parser.error("--v42-initial-width must not exceed --v42-candidate-limit")
    if args.v42_priority_top_k > args.v42_candidate_limit:
        parser.error("--v42-priority-top-k must not exceed --v42-candidate-limit")

    log_dir = args.log_dir or default_log_dir("mcts_v41_vs_v42", args.seed)
    config = {
        "seed": args.seed,
        "games_per_color": args.games,
        "simulations": args.simulations,
        "radius": args.radius,
        "v41_candidate_limit": args.v41_candidate_limit,
        "v41_initial_width": args.v41_initial_width,
        "v41_priority_top_k": args.v41_priority_top_k,
        "v42_candidate_limit": args.v42_candidate_limit,
        "v42_initial_width": args.v42_initial_width,
        "v42_priority_top_k": args.v42_priority_top_k,
        "model_a": "MCTS-v4.1",
        "model_b": "MCTS-v4.2",
    }

    v41 = agent_factory(MCTSV41Agent, args, "v41")
    v42 = agent_factory(MCTSV42Agent, args, "v42")

    print(
        f"Seed: {args.seed}; games/color: {args.games}; simulations: {args.simulations}; "
        f"V4.1={args.v41_candidate_limit}/{args.v41_initial_width}/{args.v41_priority_top_k}; "
        f"V4.2={args.v42_candidate_limit}/{args.v42_initial_width}/{args.v42_priority_top_k}; "
        f"radius: {args.radius}",
        flush=True,
    )
    print(f"Log directory: {log_dir}", flush=True)

    first_label = "v4.1_black_vs_v4.2_white"
    second_label = "v4.2_black_vs_v4.1_white"

    v41_black = run_match(v41, v42, games=args.games, seed=args.seed)
    report("MCTS-v4.1 vs MCTS-v4.2 (Black vs White)", v41_black)
    save_match_logs(log_dir, [(first_label, v41_black)], config)

    v42_black = run_match(v42, v41, games=args.games, seed=args.seed)
    report("MCTS-v4.2 vs MCTS-v4.1 (Black vs White)", v42_black)

    paths = save_match_logs(
        log_dir,
        [
            (first_label, v41_black),
            (second_label, v42_black),
        ],
        config,
    )

    v41_wins = v41_black.black_wins + v42_black.white_wins
    v42_wins = v41_black.white_wins + v42_black.black_wins
    draws = v41_black.draws + v42_black.draws

    print("\n=== Combined V4 comparison ===")
    print(f"MCTS-v4.1 wins: {v41_wins}")
    print(f"MCTS-v4.2 wins: {v42_wins}")
    print(f"Draws: {draws}")
    print(f"Total games: {args.games * 2}")
    print(f"MCTS-v4.1 win rate: {v41_wins / (args.games * 2) * 100:.2f}%")
    print(f"MCTS-v4.2 win rate: {v42_wins / (args.games * 2) * 100:.2f}%")

    print("\n=== Saved logs ===")
    print(f"games.csv: {paths['games_csv']}")
    print(f"moves.csv: {paths['moves_csv']}")
    print(f"games.json: {paths['games_json']}")


if __name__ == "__main__":
    main()
