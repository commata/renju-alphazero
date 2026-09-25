"""Run pairwise MCTS comparisons for V3.2.1, V4.1, and V4.2.

Each selected matchup runs both color assignments:
- A as Black vs B as White: 25 games by default
- B as Black vs A as White: 25 games by default

So each matchup is 50 games total.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agents import MCTSV321Agent, MCTSV41Agent, MCTSV42Agent
from evaluation import default_log_dir, run_match, save_match_logs


AgentFactory = Callable[[int], object]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    agent_type: type
    simulations: int
    candidate_limit: int
    initial_width: int
    radius: int
    priority_top_k: int

    def factory(self) -> AgentFactory:
        return lambda seed: self.agent_type(
            seed=seed,
            simulations=self.simulations,
            candidate_limit=self.candidate_limit,
            initial_width=self.initial_width,
            neighborhood_radius=self.radius,
            priority_top_k=self.priority_top_k,
        )


MODELS = {
    "v321": ModelSpec(
        key="v321",
        name="MCTS-v3.2.1",
        agent_type=MCTSV321Agent,
        simulations=25,
        candidate_limit=16,
        initial_width=6,
        radius=2,
        priority_top_k=5,
    ),
    "v41": ModelSpec(
        key="v41",
        name="MCTS-v4.1",
        agent_type=MCTSV41Agent,
        simulations=50,
        candidate_limit=20,
        initial_width=8,
        radius=2,
        priority_top_k=8,
    ),
    "v42": ModelSpec(
        key="v42",
        name="MCTS-v4.2",
        agent_type=MCTSV42Agent,
        simulations=50,
        candidate_limit=24,
        initial_width=10,
        radius=2,
        priority_top_k=10,
    ),
}

MATCHUPS = {
    "v321-v41": ("v321", "v41"),
    "v321-v42": ("v321", "v42"),
    "v41-v42": ("v41", "v42"),
}


def report(label: str, result) -> None:
    print(f"\n=== {label} ===")
    print(f"Games: {result.games}")
    print(f"Black wins: {result.black_wins}")
    print(f"White wins: {result.white_wins}")
    print(f"Draws: {result.draws}")
    print(f"Average moves: {result.average_moves:.2f}")
    print(f"Elapsed: {result.elapsed_seconds:.3f} s")
    print(f"Games/sec: {result.games_per_second:.4f}")


def run_pair(
    matchup_key: str,
    games_per_color: int,
    seed: int,
    base_log_dir: Path | None,
) -> None:
    left_key, right_key = MATCHUPS[matchup_key]
    left = MODELS[left_key]
    right = MODELS[right_key]

    if base_log_dir is None:
        log_dir = default_log_dir(f"mcts_{left_key}_vs_{right_key}", seed)
    else:
        log_dir = base_log_dir / matchup_key

    config = {
        "seed": seed,
        "matchup": matchup_key,
        "games_per_color": games_per_color,
        "total_games": games_per_color * 2,
        "model_a": {
            "key": left.key,
            "name": left.name,
            "simulations": left.simulations,
            "candidate_limit": left.candidate_limit,
            "initial_width": left.initial_width,
            "radius": left.radius,
            "priority_top_k": left.priority_top_k,
        },
        "model_b": {
            "key": right.key,
            "name": right.name,
            "simulations": right.simulations,
            "candidate_limit": right.candidate_limit,
            "initial_width": right.initial_width,
            "radius": right.radius,
            "priority_top_k": right.priority_top_k,
        },
    }

    print("\n" + "=" * 72)
    print(f"{left.name} vs {right.name}")
    print(f"Games/color: {games_per_color} -> total {games_per_color * 2}")
    print(
        f"{left.name}: sim={left.simulations}, candidates={left.candidate_limit}, "
        f"width={left.initial_width}, top-k={left.priority_top_k}, radius={left.radius}"
    )
    print(
        f"{right.name}: sim={right.simulations}, candidates={right.candidate_limit}, "
        f"width={right.initial_width}, top-k={right.priority_top_k}, radius={right.radius}"
    )
    print(f"Seed: {seed}")
    print(f"Log directory: {log_dir}")
    print("=" * 72, flush=True)

    first_label = f"{left.key}_black_vs_{right.key}_white"
    second_label = f"{right.key}_black_vs_{left.key}_white"

    first = run_match(
        left.factory(),
        right.factory(),
        games=games_per_color,
        seed=seed,
    )
    report(f"{left.name} Black vs {right.name} White", first)

    # Save a checkpoint after the first 25-game color assignment.
    save_match_logs(log_dir, [(first_label, first)], config)

    second = run_match(
        right.factory(),
        left.factory(),
        games=games_per_color,
        seed=seed,
    )
    report(f"{right.name} Black vs {left.name} White", second)

    paths = save_match_logs(
        log_dir,
        [
            (first_label, first),
            (second_label, second),
        ],
        config,
    )

    left_wins = first.black_wins + second.white_wins
    right_wins = first.white_wins + second.black_wins
    draws = first.draws + second.draws
    total_games = games_per_color * 2

    print("\n=== Combined result ===")
    print(f"{left.name} wins: {left_wins}")
    print(f"{right.name} wins: {right_wins}")
    print(f"Draws: {draws}")
    print(f"Total games: {total_games}")
    print(f"{left.name} win rate: {left_wins / total_games * 100:.2f}%")
    print(f"{right.name} win rate: {right_wins / total_games * 100:.2f}%")

    print("\n=== Saved logs ===")
    print(f"games.csv: {paths['games_csv']}")
    print(f"moves.csv: {paths['moves_csv']}")
    print(f"games.json: {paths['games_json']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matchup",
        choices=(*MATCHUPS.keys(), "all"),
        default="all",
        help="pair to run; 'all' runs all three pairings",
    )
    parser.add_argument(
        "--games-per-color",
        type=int,
        default=25,
        help="games for each color assignment; default 25 => 50 games per matchup",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="optional base log directory",
    )
    args = parser.parse_args()

    if args.games_per_color <= 0:
        parser.error("--games-per-color must be positive")

    selected = MATCHUPS.keys() if args.matchup == "all" else (args.matchup,)
    for matchup_key in selected:
        run_pair(
            matchup_key=matchup_key,
            games_per_color=args.games_per_color,
            seed=args.seed,
            base_log_dir=args.log_dir,
        )


if __name__ == "__main__":
    main()
