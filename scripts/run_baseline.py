"""Run the three baseline matchups from the repository root after pip install -e ."""
import argparse
import hashlib
import json

from agents import RandomAgent, TacticalAgent
from evaluation import run_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100, help="games per matchup (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")

    print(f"Seed: {args.seed}; games per matchup: {args.games}", flush=True)
    for black, white in ((RandomAgent, RandomAgent),
                         (RandomAgent, TacticalAgent), (TacticalAgent, RandomAgent)):
        print(f"\n=== {black.name} vs {white.name} (Black vs White) ===", flush=True)
        result = run_match(black, white, games=args.games, seed=args.seed)
        # Exclude timing from the fingerprint so complete sequences can be compared.
        records = [(game.winner, game.history) for game in result.results]
        fingerprint = hashlib.sha256(json.dumps(records).encode("utf-8")).hexdigest()
        print(f"Games: {result.games}")
        print(f"Black wins: {result.black_wins}")
        print(f"White wins: {result.white_wins}")
        print(f"Draws: {result.draws}")
        print(f"Total moves: {result.total_moves}")
        print(f"Average moves: {result.average_moves:.2f}")
        print(f"Elapsed: {result.elapsed_seconds:.3f} s")
        print(f"Games/sec: {result.games_per_second:.4f}")
        print(f"Outcome/history SHA256: {fingerprint}", flush=True)


if __name__ == "__main__":
    main()
