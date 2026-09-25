"""CSV/JSON exporters for reproducible match logs."""
from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from renju import BLACK, WHITE

from .match import MatchResult


NamedMatch = tuple[str, MatchResult]


def default_log_dir(prefix: str, seed: int, base_dir: str | Path = "logs") -> Path:
    """Create a Windows-safe timestamped directory name for one experiment."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(base_dir) / prefix / f"{stamp}_seed{seed}"


def _player_name(player: int | None) -> str:
    if player == BLACK:
        return "BLACK"
    if player == WHITE:
        return "WHITE"
    return "DRAW"


def save_match_logs(
    output_dir: str | Path,
    matches: Sequence[NamedMatch],
    config: Mapping[str, object],
) -> dict[str, Path]:
    """Save per-game summaries, per-move rows and complete JSON metadata."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    games_path = output / "games.csv"
    moves_path = output / "moves.csv"
    json_path = output / "games.json"

    game_rows: list[dict[str, object]] = []
    move_rows: list[dict[str, object]] = []
    json_games: list[dict[str, object]] = []
    json_matches: list[dict[str, object]] = []

    game_id = 0
    for matchup, match in matches:
        json_matches.append(
            {
                "matchup": matchup,
                "games": match.games,
                "black_wins": match.black_wins,
                "white_wins": match.white_wins,
                "draws": match.draws,
                "total_moves": match.total_moves,
                "average_moves": match.average_moves,
                "elapsed_seconds": match.elapsed_seconds,
                "games_per_second": match.games_per_second,
                "seed": match.seed,
            }
        )

        for matchup_game, result in enumerate(match.results, start=1):
            game_id += 1
            winner = _player_name(result.winner)
            if result.winner == BLACK:
                winner_agent = result.black_agent
            elif result.winner == WHITE:
                winner_agent = result.white_agent
            else:
                winner_agent = ""

            game_rows.append(
                {
                    "game_id": game_id,
                    "matchup": matchup,
                    "matchup_game": matchup_game,
                    "black_agent": result.black_agent,
                    "white_agent": result.white_agent,
                    "winner": winner,
                    "winner_agent": winner_agent,
                    "number_of_moves": result.number_of_moves,
                    "elapsed_seconds": f"{result.elapsed_seconds:.6f}",
                    "match_seed": match.seed,
                }
            )

            json_moves: list[dict[str, object]] = []
            for ply, (row0, col0) in enumerate(result.history, start=1):
                player = "BLACK" if ply % 2 == 1 else "WHITE"
                move = {
                    "ply": ply,
                    "player": player,
                    "row0": row0,
                    "col0": col0,
                    "row": row0 + 1,
                    "col": col0 + 1,
                }
                json_moves.append(move)
                move_rows.append(
                    {
                        "game_id": game_id,
                        "matchup": matchup,
                        "matchup_game": matchup_game,
                        **move,
                    }
                )

            json_games.append(
                {
                    "game_id": game_id,
                    "matchup": matchup,
                    "matchup_game": matchup_game,
                    "black_agent": result.black_agent,
                    "white_agent": result.white_agent,
                    "winner": winner,
                    "winner_agent": winner_agent,
                    "number_of_moves": result.number_of_moves,
                    "elapsed_seconds": result.elapsed_seconds,
                    "moves": json_moves,
                }
            )

    with games_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "game_id",
                "matchup",
                "matchup_game",
                "black_agent",
                "white_agent",
                "winner",
                "winner_agent",
                "number_of_moves",
                "elapsed_seconds",
                "match_seed",
            ],
        )
        writer.writeheader()
        writer.writerows(game_rows)

    with moves_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "game_id",
                "matchup",
                "matchup_game",
                "ply",
                "player",
                "row0",
                "col0",
                "row",
                "col",
            ],
        )
        writer.writeheader()
        writer.writerows(move_rows)

    payload = {
        "config": dict(config),
        "matches": json_matches,
        "games": json_games,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return {
        "games_csv": games_path,
        "moves_csv": moves_path,
        "games_json": json_path,
    }
