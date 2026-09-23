"""Compare the frozen MCTS V2 against experimental MCTS V3 in both colors."""
import argparse
from collections.abc import Callable
from pathlib import Path

from agents import MCTSV2Agent, MCTSV3Agent
from evaluation import default_log_dir, run_match, save_match_logs


def v2_factory(simulations: int, candidate_limit: int) -> Callable[[int], MCTSV2Agent]:
    return lambda seed: MCTSV2Agent(
        seed=seed,
        simulations=simulations,
        candidate_limit=candidate_limit,
    )


def v3_factory(
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
    parser.add_argument("--v2-simulations", type=int, default=10)
    parser.add_argument("--v3-simulations", type=int, default=25)
    parser.add_argument("--v2-candidate-limit", type=int, default=8)
    parser.add_argument("--v3-candidate-limit", type=int, default=16)
    parser.add_argument("--v3-initial-width", type=int, default=6)
    parser.add_argument("--v3-radius", type=int, default=2)
    parser.add_argument("--v3-priority-top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="output directory; default is logs/mcts_versions/<timestamp>_seedN",
    )
    args = parser.parse_args()

    for value, name in (
        (args.games, "--games"),
        (args.v2_simulations, "--v2-simulations"),
        (args.v3_simulations, "--v3-simulations"),
        (args.v2_candidate_limit, "--v2-candidate-limit"),
        (args.v3_candidate_limit, "--v3-candidate-limit"),
        (args.v3_initial_width, "--v3-initial-width"),
        (args.v3_radius, "--v3-radius"),
        (args.v3_priority_top_k, "--v3-priority-top-k"),
    ):
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.v3_initial_width > args.v3_candidate_limit:
        parser.error("--v3-initial-width must not exceed --v3-candidate-limit")
    if args.v3_priority_top_k > args.v3_candidate_limit:
        parser.error("--v3-priority-top-k must not exceed --v3-candidate-limit")

    log_dir = args.log_dir or default_log_dir("mcts_versions", args.seed)
    config = {
        "seed": args.seed,
        "games_per_color": args.games,
        "v2_simulations": args.v2_simulations,
        "v3_simulations": args.v3_simulations,
        "v2_candidate_limit": args.v2_candidate_limit,
        "v3_candidate_limit": args.v3_candidate_limit,
        "v3_initial_width": args.v3_initial_width,
        "v3_radius": args.v3_radius,
        "v3_priority_top_k": args.v3_priority_top_k,
    }

    v2 = v2_factory(args.v2_simulations, args.v2_candidate_limit)
    v3 = v3_factory(
        args.v3_simulations,
        args.v3_candidate_limit,
        args.v3_initial_width,
        args.v3_radius,
        args.v3_priority_top_k,
    )

    print(
        f"Seed: {args.seed}; games/color: {args.games}; "
        f"V2 simulations/candidates: {args.v2_simulations}/{args.v2_candidate_limit}; "
        f"V3 simulations/candidates: {args.v3_simulations}/{args.v3_candidate_limit}; "
        f"V3 initial width: {args.v3_initial_width}; "
        f"V3 radius: {args.v3_radius}; "
        f"V3 priority top-k: {args.v3_priority_top_k}",
        flush=True,
    )
    print(f"Log directory: {log_dir}", flush=True)

    first_label = "v2_black_vs_v3_white"
    second_label = "v3_black_vs_v2_white"

    v2_black = run_match(v2, v3, games=args.games, seed=args.seed)
    report("MCTS-v2 vs MCTS-v3 (Black vs White)", v2_black)
    save_match_logs(log_dir, [(first_label, v2_black)], config)

    v3_black = run_match(v3, v2, games=args.games, seed=args.seed)
    report("MCTS-v3 vs MCTS-v2 (Black vs White)", v3_black)

    paths = save_match_logs(
        log_dir,
        [(first_label, v2_black), (second_label, v3_black)],
        config,
    )

    v2_wins = v2_black.black_wins + v3_black.white_wins
    v3_wins = v2_black.white_wins + v3_black.black_wins
    draws = v2_black.draws + v3_black.draws
    print("\n=== Combined revision comparison ===")
    print(f"MCTS-v2 wins: {v2_wins}")
    print(f"MCTS-v3 wins: {v3_wins}")
    print(f"Draws: {draws}")
    print(f"Total games: {args.games * 2}")
    print("\n=== Saved logs ===")
    print(f"games.csv: {paths['games_csv']}")
    print(f"moves.csv: {paths['moves_csv']}")
    print(f"games.json: {paths['games_json']}")


if __name__ == "__main__":
    main()
