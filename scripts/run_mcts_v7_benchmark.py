"""Paired-opening MCTS-v7 benchmark against frozen V6 or V5 FINAL.

This is a benchmark runner, not a CI-strength assertion. Each generated opening is
played twice with colours swapped, and all opening moves and seeds are recorded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from random import Random
from time import perf_counter

from agents import MCTSV5Agent, MCTSV6Agent, MCTSV7Agent
from renju import BLACK, WHITE, Game, IllegalMove
from renju.game import OPENING_MOVE
from search.mcts_v6 import V5_FINAL
from search.mcts_v7 import V7_FINAL, SearchDiagnostics


def derive_seed(seed: int, *parts) -> int:
    payload = json.dumps([seed, *parts], separators=(",", ":"), ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def make_opening(seed: int, random_plies: int, radius: int):
    rng = Random(seed)
    game = Game()
    game.play(*OPENING_MOVE)
    cr, cc = OPENING_MOVE
    for _ in range(random_plies):
        legal = game.legal_moves()
        near = [m for m in legal if max(abs(m[0] - cr), abs(m[1] - cc)) <= radius]
        game.play(*rng.choice(near or legal))
        if game.done:
            raise RuntimeError("benchmark opening ended the game")
    return tuple(game.history)


def v5_final(seed: int):
    agent = MCTSV5Agent(seed=seed, **V5_FINAL)
    agent.name = "MCTS-v5-final"
    return agent


def make_opponent(name: str, seed: int):
    if name == "v6":
        return MCTSV6Agent(seed=seed)
    if name == "v5":
        return v5_final(seed)
    raise ValueError(name)


def play_one(v7_color: int, opening, opponent_name: str, seed: int):
    v7 = MCTSV7Agent(seed=derive_seed(seed, "v7"))
    opponent = make_opponent(opponent_name, derive_seed(seed, opponent_name))
    game = Game()
    for move in opening:
        game.play(*move)

    v7_times, opponent_times = [], []
    diag_totals = {key: 0 for key in (
        "v7_own_vcf_found", "v7_safety_checked", "v7_safety_removed",
        "v7_safety_augmented", "v7_safety_fallback",
        "v7_self_forbidden_penalized", "v7_stage4_tiebreak_applied",
    )}
    v7_module_seconds = 0.0
    max_v7_move_seconds = 0.0

    while not game.done:
        is_v7 = game.to_play == v7_color
        agent = v7 if is_v7 else opponent
        started = perf_counter()
        move = agent.select_move(game)
        elapsed = perf_counter() - started
        if is_v7:
            v7_times.append(elapsed)
            max_v7_move_seconds = max(max_v7_move_seconds, elapsed)
            d = v7.diagnostics
            diag_totals["v7_own_vcf_found"] += int(d.v7_own_vcf_found)
            diag_totals["v7_safety_checked"] += d.v7_safety_checked
            diag_totals["v7_safety_removed"] += d.v7_safety_removed
            diag_totals["v7_safety_augmented"] += int(d.v7_safety_augmented)
            diag_totals["v7_safety_fallback"] += int(d.v7_safety_fallback)
            diag_totals["v7_self_forbidden_penalized"] += d.v7_self_forbidden_penalized
            diag_totals["v7_stage4_tiebreak_applied"] += int(d.v7_stage4_tiebreak_applied)
            v7_module_seconds += d.v7_module_seconds
        else:
            opponent_times.append(elapsed)
        try:
            game.play(*move)
        except IllegalMove as exc:
            raise RuntimeError(f"{agent.name} returned illegal move {move}: {exc}") from exc

    if game.winner is None:
        result = "draw"
    elif game.winner == v7_color:
        result = "win"
    else:
        result = "loss"
    return {
        "v7_color": "black" if v7_color == BLACK else "white",
        "opponent": opponent.name,
        "winner": game.winner,
        "result": result,
        "moves": [list(m) for m in game.history],
        "length": len(game.history),
        "v7_move_seconds": v7_times,
        "opponent_move_seconds": opponent_times,
        "v7_max_move_seconds": max_v7_move_seconds,
        "v7_module_seconds": v7_module_seconds,
        "v7_diagnostics": diag_totals,
    }


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(games):
    wins = sum(g["result"] == "win" for g in games)
    draws = sum(g["result"] == "draw" for g in games)
    losses = sum(g["result"] == "loss" for g in games)
    v7_times = [x for g in games for x in g["v7_move_seconds"]]
    opp_times = [x for g in games for x in g["opponent_move_seconds"]]
    by_color = {}
    for color in ("black", "white"):
        subset = [g for g in games if g["v7_color"] == color]
        w = sum(g["result"] == "win" for g in subset)
        d = sum(g["result"] == "draw" for g in subset)
        l = sum(g["result"] == "loss" for g in subset)
        by_color[color] = {
            "wins": w, "draws": d, "losses": l, "games": len(subset),
            "score": (w + 0.5 * d) / len(subset) if subset else None,
        }
    diag = {}
    for key in games[0]["v7_diagnostics"] if games else ():
        diag[key] = sum(g["v7_diagnostics"][key] for g in games)
    records = [(g["winner"], g["moves"]) for g in games]
    fingerprint = hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "games": len(games), "wins": wins, "draws": draws, "losses": losses,
        "score": (wins + 0.5 * draws) / len(games) if games else None,
        "by_color": by_color,
        "average_v7_move_seconds": mean(v7_times),
        "max_v7_move_seconds": max(v7_times) if v7_times else None,
        "average_opponent_move_seconds": mean(opp_times),
        "average_game_length": mean([g["length"] for g in games]),
        "v7_module_seconds": sum(g["v7_module_seconds"] for g in games),
        "v7_diagnostics": diag,
        "outcome_history_sha256": fingerprint,
        "illegal_moves": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", choices=("v6", "v5"), required=True)
    parser.add_argument("--pairs", type=int, default=50,
                        help="opening pairs; 50 means 100 games")
    parser.add_argument("--seed", type=int, default=6507)
    parser.add_argument("--opening-random-plies", type=int, default=2)
    parser.add_argument("--opening-radius", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if args.opening_random_plies < 0 or args.opening_radius < 0:
        parser.error("opening arguments must be non-negative")

    games = []
    openings = []
    for pair in range(args.pairs):
        opening_seed = derive_seed(args.seed, args.opponent, "opening", pair)
        opening = make_opening(
            opening_seed, args.opening_random_plies, args.opening_radius,
        )
        openings.append({"pair": pair, "seed": opening_seed,
                         "moves": [list(m) for m in opening]})
        for color in (BLACK, WHITE):
            game_seed = derive_seed(args.seed, args.opponent, "game", pair, color)
            record = play_one(color, opening, args.opponent, game_seed)
            record["pair"] = pair
            record["seed"] = game_seed
            games.append(record)
            print(
                f"pair={pair} v7={record['v7_color']} result={record['result']} "
                f"moves={record['length']} max_v7_s={record['v7_max_move_seconds']:.3f}",
                flush=True,
            )

    payload = {
        "format": "mcts-v7-benchmark-v1",
        "seed": args.seed,
        "opponent": args.opponent,
        "pairs": args.pairs,
        "opening_random_plies": args.opening_random_plies,
        "opening_radius": args.opening_radius,
        "v7_config": V7_FINAL,
        "opponent_config": V5_FINAL,
        "openings": openings,
        "summary": summarize(games),
        "games": games,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
