"""Compare MCTS V3.1 against optimized MCTS V3.2.1 in both colors."""

import argparse
from collections.abc import Callable
from pathlib import Path

from agents import MCTSV3Agent, MCTSV321Agent
from evaluation import default_log_dir, run_match, save_match_logs


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


def v321_factory(
    simulations: int,
    candidate_limit: int,
    initial_width: int,
    radius: int,
    priority_top_k: int,
) -> Callable[[int], MCTSV321Agent]:
    return lambda seed: MCTSV321Agent(
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

    parser.add_argument(
        "--games",
        type=int,
        default=25,
        help="games per color assignment",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--initial-width",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--priority-top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "output directory; "
            "default: logs/mcts_v31_vs_v321/<timestamp>_seedN"
        ),
    )

    args = parser.parse_args()

    # -----------------------------
    # argument validation
    # -----------------------------
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
        parser.error(
            "--initial-width must not exceed --candidate-limit"
        )

    if args.priority_top_k > args.candidate_limit:
        parser.error(
            "--priority-top-k must not exceed --candidate-limit"
        )

    # -----------------------------
    # logging
    # -----------------------------
    log_dir = (
        args.log_dir
        or default_log_dir(
            "mcts_v31_vs_v321",
            args.seed,
        )
    )

    config = {
        "seed": args.seed,
        "games_per_color": args.games,
        "simulations": args.simulations,
        "candidate_limit": args.candidate_limit,
        "initial_width": args.initial_width,
        "radius": args.radius,
        "priority_top_k": args.priority_top_k,
        "black_white_swap": True,
        "model_a": "MCTS-v3.1",
        "model_b": "MCTS-v3.2.1",
    }

    # -----------------------------
    # agent factories
    # -----------------------------
    v31 = v31_factory(
        simulations=args.simulations,
        candidate_limit=args.candidate_limit,
        initial_width=args.initial_width,
        radius=args.radius,
        priority_top_k=args.priority_top_k,
    )

    v321 = v321_factory(
        simulations=args.simulations,
        candidate_limit=args.candidate_limit,
        initial_width=args.initial_width,
        radius=args.radius,
        priority_top_k=args.priority_top_k,
    )

    print(
        f"Seed: {args.seed}; "
        f"games/color: {args.games}; "
        f"simulations/candidates: "
        f"{args.simulations}/{args.candidate_limit}; "
        f"initial width: {args.initial_width}; "
        f"radius: {args.radius}; "
        f"priority top-k: {args.priority_top_k}",
        flush=True,
    )

    print(
        f"Log directory: {log_dir}",
        flush=True,
    )

    first_label = "v3.1_black_vs_v3.2.1_white"
    second_label = "v3.2.1_black_vs_v3.1_white"

    # =========================================================
    # Match 1
    # V3.1 Black vs V3.2.1 White
    # =========================================================
    v31_black = run_match(
        v31,
        v321,
        games=args.games,
        seed=args.seed,
    )

    report(
        "MCTS-v3.1 vs MCTS-v3.2.1 (Black vs White)",
        v31_black,
    )

    # 첫 번째 색 배치가 끝난 즉시 checkpoint 저장
    save_match_logs(
        log_dir,
        [
            (
                first_label,
                v31_black,
            )
        ],
        config,
    )

    # =========================================================
    # Match 2
    # V3.2.1 Black vs V3.1 White
    # =========================================================
    v321_black = run_match(
        v321,
        v31,
        games=args.games,
        seed=args.seed,
    )

    report(
        "MCTS-v3.2.1 vs MCTS-v3.1 (Black vs White)",
        v321_black,
    )

    # =========================================================
    # Final logs
    # =========================================================
    paths = save_match_logs(
        log_dir,
        [
            (
                first_label,
                v31_black,
            ),
            (
                second_label,
                v321_black,
            ),
        ],
        config,
    )

    # =========================================================
    # Combined result
    # =========================================================
    v31_wins = (
        v31_black.black_wins
        + v321_black.white_wins
    )

    v321_wins = (
        v31_black.white_wins
        + v321_black.black_wins
    )

    draws = (
        v31_black.draws
        + v321_black.draws
    )

    total_games = args.games * 2

    print("\n=== Combined policy comparison ===")
    print(f"MCTS-v3.1 wins: {v31_wins}")
    print(f"MCTS-v3.2.1 wins: {v321_wins}")
    print(f"Draws: {draws}")
    print(f"Total games: {total_games}")

    if total_games > 0:
        print(
            f"MCTS-v3.1 win rate: "
            f"{v31_wins / total_games * 100:.2f}%"
        )
        print(
            f"MCTS-v3.2.1 win rate: "
            f"{v321_wins / total_games * 100:.2f}%"
        )

    print("\n=== Saved logs ===")
    print(
        f"games.csv: {paths['games_csv']}"
    )
    print(
        f"moves.csv: {paths['moves_csv']}"
    )
    print(
        f"games.json: {paths['games_json']}"
    )


if __name__ == "__main__":
    main()