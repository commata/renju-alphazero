"""Low-cost deterministic V6-vs-V5 behavior fingerprint for CI.

Runs one game per color with the frozen V5/V6 policy parameters but reduced rollout
budgets. This is a regression fingerprint, not a playing-strength benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json

from agents import MCTSV5Agent, MCTSV6Agent
from evaluation import run_match
from search.mcts_v6 import V5_FINAL

SEED = 777
SIMULATIONS = 4
TACTICAL_SIMULATIONS = 8


def config() -> dict:
    return {
        **V5_FINAL,
        "simulations": SIMULATIONS,
        "tactical_simulations": TACTICAL_SIMULATIONS,
    }


def v5(seed: int):
    agent = MCTSV5Agent(seed=seed, **config())
    agent.name = "MCTS-v5-final-ci"
    return agent


def v6(seed: int):
    return MCTSV6Agent(seed=seed, **config())


def fingerprint(matches) -> str:
    records = [(result.winner, result.history)
               for match in matches for result in match.results]
    return hashlib.sha256(json.dumps(records).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-sha256")
    args = parser.parse_args()

    matches = (
        run_match(v6, v5, games=1, seed=SEED),
        run_match(v5, v6, games=1, seed=SEED),
    )
    digest = fingerprint(matches)
    summary = [
        {
            "black": match.results[0].black_agent,
            "white": match.results[0].white_agent,
            "winner": match.results[0].winner,
            "moves": match.results[0].number_of_moves,
        }
        for match in matches
    ]
    print(json.dumps({
        "seed": SEED,
        "simulations": SIMULATIONS,
        "tactical_simulations": TACTICAL_SIMULATIONS,
        "games": summary,
        "outcome_history_sha256": digest,
    }, indent=2))

    if args.expect_sha256 is not None and digest != args.expect_sha256:
        print(f"fingerprint mismatch: expected={args.expect_sha256} actual={digest}")
        return 1
    if args.expect_sha256 is not None:
        print(f"expected fingerprint matched: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
