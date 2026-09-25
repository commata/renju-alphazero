"""Tiny local web UI for playing Renju against the existing MCTS agents.

No third-party web framework is required.

Run:
    python scripts/run_web_play.py

Then open:
    http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields, is_dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WEB = ROOT / "web"
LOG_ROOT = ROOT / "logs" / "web_play"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents import (  # noqa: E402
    MCTSV321Agent,
    MCTSV41Agent,
    MCTSV42Agent,
    MCTSV5Agent,
    MCTSV6Agent,
)
from renju import BLACK, WHITE, Game, IllegalMove  # noqa: E402
from renju.game import OPENING_MOVE  # noqa: E402
from search.mcts_v6 import V5_FINAL  # noqa: E402


VERSION_LABELS = {
    "v321": "MCTS V3.2.1",
    "v41": "MCTS V4.1",
    "v42": "MCTS V4.2",
    "v5": "MCTS V5 FINAL",
    "v6": "MCTS V6",
}


def create_agent(version: str, *, seed: int = 42):
    """Build one of the frozen/versioned agents used by the benchmark scripts."""
    if version == "v321":
        return MCTSV321Agent(seed=seed)
    if version == "v41":
        return MCTSV41Agent(seed=seed)
    if version == "v42":
        return MCTSV42Agent(seed=seed)
    if version == "v5":
        agent = MCTSV5Agent(seed=seed, stage="a", **V5_FINAL)
        agent.name = "MCTS-v5-final"
        return agent
    if version == "v6":
        return MCTSV6Agent(seed=seed)
    raise ValueError(f"unknown MCTS version: {version}")


def _winner_name(winner: int | None) -> str | None:
    if winner == BLACK:
        return "BLACK"
    if winner == WHITE:
        return "WHITE"
    return None


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _json_value(value: Any):
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _diagnostics(agent) -> dict[str, Any]:
    """Return only the small fields useful while manually testing V5/V6."""
    diagnostics = getattr(agent, "diagnostics", None)
    if diagnostics is None:
        return {}
    names = {
        "forced_policy_stage",
        "simulation_mode",
        "selected_simulations",
        "best_root_tactical_score",
        "stage5_multi_root_injection",
        "v6_selected_threat_type",
        "v6_selected_reasons",
        "v6_root_injection_count",
        "black_43_candidates",
        "white_43_candidates",
        "white_44_candidates",
        "white_33_candidates",
        "future_black_43_setups",
        "future_white_43_setups",
        "future_white_44_setups",
        "future_white_33_setups",
        "black_43_defense_candidates",
        "future_black_43_defense_candidates",
        "black_43_defense_complete_candidates",
        "planner_forced_plan_conflicts",
        "planner_forced_plan_preserved",
    }
    if is_dataclass(diagnostics):
        available = {field.name for field in fields(diagnostics)}
        names.intersection_update(available)
    result = {}
    for name in sorted(names):
        if hasattr(diagnostics, name):
            value = getattr(diagnostics, name)
            if value is not None:
                result[name] = _json_value(value)
    return result


class PlaySession:
    """Single local-browser game session.

    The server intentionally keeps one game in memory. This is a developer tool,
    not a multi-user service.
    """

    def __init__(self, *, log_root: Path | None = None):
        self.lock = Lock()
        self.log_root = Path(log_root) if log_root is not None else LOG_ROOT
        self.game = Game()
        self.agent_key = "v6"
        self.agent = create_agent(self.agent_key)
        self.human_color = BLACK
        self.seed = 42
        self.last_ai_move: tuple[int, int] | None = None
        self.last_ai_seconds: float | None = None
        self.message = "새 게임을 시작했습니다."
        self.started_at = datetime.now().astimezone()
        self.move_records: list[dict[str, Any]] = []
        self.last_log_dir: Path | None = None
        self.saved_game = False

    def reset(self, *, agent_key: str, human_color: int, seed: int = 42) -> dict[str, Any]:
        if agent_key not in VERSION_LABELS:
            raise ValueError("지원하지 않는 MCTS 버전입니다.")
        if human_color not in (BLACK, WHITE):
            raise ValueError("human_color는 BLACK 또는 WHITE여야 합니다.")
        with self.lock:
            self.game = Game()
            self.agent_key = agent_key
            self.agent = create_agent(agent_key, seed=seed)
            self.human_color = human_color
            self.seed = seed
            self.last_ai_move = None
            self.last_ai_seconds = None
            self.message = "새 게임을 시작했습니다."
            self.started_at = datetime.now().astimezone()
            self.move_records = []
            self.last_log_dir = None
            self.saved_game = False

            # Project opening rule: BLACK always starts at board center.
            opening_player = self.game.to_play
            self.game.play(*OPENING_MOVE)
            self._record_move_locked(
                opening_player,
                "OPENING_RULE",
                OPENING_MOVE,
                seconds=None,
                diagnostics={},
            )
            self.message = "흑 중앙 첫 수가 자동 배치되었습니다."

            if not self.game.done and self.game.to_play != self.human_color:
                self._play_ai_locked()
            return self._state_locked()

    def play_human(self, row: int, col: int) -> dict[str, Any]:
        with self.lock:
            if self.game.done:
                raise IllegalMove("이미 종료된 대국입니다.")
            if self.game.to_play != self.human_color:
                raise IllegalMove("현재는 AI 차례입니다.")
            player = self.game.to_play
            self.game.play(row, col)
            self._record_move_locked(player, "HUMAN", (row, col), seconds=None, diagnostics={})
            self.message = f"사람 착수: ({row + 1}, {col + 1})"
            if self.game.done:
                self._save_completed_game_locked()
            else:
                self._play_ai_locked()
            return self._state_locked()

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self._state_locked()

    def _play_ai_locked(self) -> None:
        if self.game.done or self.game.to_play == self.human_color:
            return
        player = self.game.to_play
        started = perf_counter()
        move = self.agent.select_move(self.game)
        elapsed = perf_counter() - started
        diagnostics = _diagnostics(self.agent)
        self.game.play(*move)
        self._record_move_locked(player, self.agent.name, move, seconds=elapsed, diagnostics=diagnostics)
        self.last_ai_move = move
        self.last_ai_seconds = elapsed
        self.message = f"AI 착수: ({move[0] + 1}, {move[1] + 1})"
        if self.game.done:
            self._save_completed_game_locked()

    def _record_move_locked(
        self,
        player: int,
        actor: str,
        move: tuple[int, int],
        *,
        seconds: float | None,
        diagnostics: dict[str, Any],
    ) -> None:
        row, col = move
        self.move_records.append({
            "ply": len(self.game.history),
            "player": "BLACK" if player == BLACK else "WHITE",
            "actor": actor,
            "row0": row,
            "col0": col,
            "row": row + 1,
            "col": col + 1,
            "seconds": seconds,
            "diagnostics": diagnostics,
        })

    def _save_completed_game_locked(self) -> Path | None:
        """Persist one finished human-vs-MCTS game exactly once."""
        if not self.game.done or self.saved_game:
            return self.last_log_dir

        finished_at = datetime.now().astimezone()
        human_name = "black" if self.human_color == BLACK else "white"
        stamp = finished_at.strftime("%Y%m%d-%H%M%S-%f")
        log_dir = self.log_root / f"{stamp}_{self.agent_key}_human-{human_name}_seed{self.seed}"
        log_dir.mkdir(parents=True, exist_ok=False)

        winner = _winner_name(self.game.winner)
        winner_actor = None
        if self.game.winner is not None:
            winner_actor = "HUMAN" if self.game.winner == self.human_color else self.agent.name
        result = "DRAW"
        if self.game.winner is not None:
            result = "HUMAN_WIN" if self.game.winner == self.human_color else "AI_WIN"

        ai_total_seconds = sum(
            float(record["seconds"]) for record in self.move_records
            if record["seconds"] is not None and record["actor"] != "HUMAN"
        )
        payload = {
            "started_at": self.started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "agent_key": self.agent_key,
            "agent_name": self.agent.name,
            "seed": self.seed,
            "human_color": "BLACK" if self.human_color == BLACK else "WHITE",
            "winner": winner,
            "winner_actor": winner_actor,
            "result": result,
            "number_of_moves": len(self.game.history),
            "ai_total_seconds": ai_total_seconds,
            "moves": self.move_records,
            "final_board": self.game.board,
        }
        (log_dir / "game.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with (log_dir / "moves.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "ply", "player", "actor", "row0", "col0", "row", "col", "seconds",
                "forced_policy_stage", "simulation_mode", "selected_simulations",
                "best_root_tactical_score", "v6_selected_threat_type",
                "v6_selected_reasons",
            ])
            writer.writeheader()
            for record in self.move_records:
                diagnostics = record.get("diagnostics", {})
                writer.writerow({
                    "ply": record["ply"],
                    "player": record["player"],
                    "actor": record["actor"],
                    "row0": record["row0"],
                    "col0": record["col0"],
                    "row": record["row"],
                    "col": record["col"],
                    "seconds": record["seconds"],
                    "forced_policy_stage": diagnostics.get("forced_policy_stage"),
                    "simulation_mode": diagnostics.get("simulation_mode"),
                    "selected_simulations": diagnostics.get("selected_simulations"),
                    "best_root_tactical_score": diagnostics.get("best_root_tactical_score"),
                    "v6_selected_threat_type": diagnostics.get("v6_selected_threat_type"),
                    "v6_selected_reasons": json.dumps(
                        diagnostics.get("v6_selected_reasons", []),
                        ensure_ascii=False,
                    ),
                })

        self.last_log_dir = log_dir
        self.saved_game = True
        display_path = _display_path(log_dir)
        self.message = f"대국 종료 · 로그 저장: {display_path}"
        print(f"[web] saved game log: {display_path}")
        return log_dir

    def _state_locked(self) -> dict[str, Any]:
        return {
            "board": self.game.board,
            "to_play": "BLACK" if self.game.to_play == BLACK else "WHITE",
            "winner": _winner_name(self.game.winner),
            "done": self.game.done,
            "move_count": len(self.game.history),
            "history": [list(move) for move in self.game.history],
            "last_move": list(self.game.history[-1]) if self.game.history else None,
            "human_color": "BLACK" if self.human_color == BLACK else "WHITE",
            "agent_key": self.agent_key,
            "agent_name": self.agent.name,
            "seed": self.seed,
            "last_ai_move": list(self.last_ai_move) if self.last_ai_move else None,
            "last_ai_seconds": self.last_ai_seconds,
            "diagnostics": _diagnostics(self.agent),
            "message": self.message,
            "log_saved": self.saved_game,
            "log_directory": (
                _display_path(self.last_log_dir)
                if self.last_log_dir is not None else None
            ),
            "versions": VERSION_LABELS,
        }


SESSION = PlaySession()


class Handler(BaseHTTPRequestHandler):
    server_version = "RenjuLocal/1.0"

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                content = (WEB / "index.html").read_bytes()
            except OSError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if self.path == "/api/state":
            self._send_json(SESSION.state())
            return
        if self.path == "/api/health":
            self._send_json({"ok": True})
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        try:
            payload = self._read_json()
            if self.path == "/api/new":
                color = str(payload.get("human_color", "BLACK")).upper()
                human_color = BLACK if color == "BLACK" else WHITE if color == "WHITE" else 0
                state = SESSION.reset(
                    agent_key=str(payload.get("agent", "v6")),
                    human_color=human_color,
                    seed=int(payload.get("seed", 42)),
                )
                self._send_json(state)
                return
            if self.path == "/api/move":
                state = SESSION.play_human(int(payload["row"]), int(payload["col"]))
                self._send_json(state)
                return
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, TypeError, IllegalMove) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # local developer tool: show useful error text
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON object가 필요합니다.")
        return value

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK):
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args):
        print(f"[web] {self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Renju against local MCTS versions in a browser.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Renju local web: http://{args.host}:{args.port}")
    print("종료: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
