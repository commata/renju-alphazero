"""Tiny local web UI for playing Renju against the existing MCTS agents.

No third-party web framework is required.

Run:
    python scripts/run_web_play.py

MCTS-v8 (development teacher engine, ``analysis.mcts_v8``) is the "v8" opponent. Its
VCT1 proofs can take tens of seconds to minutes on some moves; the page waits.

With an AlphaZero checkpoint (requires torch; adds the "az" opponent):
    python scripts/run_web_play.py --az-checkpoint runs/<run>/checkpoints/checkpoint_gen400.pt

The frozen S3-VCT2-v1 engine (docs/mcts-v8-teacher.md §12.25; requires torch and the H3
policy checkpoint whose bytes the S3 runs used) is the "s3" opponent. It is registered when
``--policy-checkpoint`` (default ``runs/h3_policy_64x4/best.pt``) exists and matches:
    python scripts/run_web_play.py --policy-checkpoint runs/h3_policy_64x4/best.pt
Its game logs also record the engine commit and checkpoint hashes, and every move where the
VCT2 check proved the move lost is saved at once under ``logs/web_play/veto_positions``.
"Undo" takes back the last human move (and the AI reply) for relaying games from elsewhere.

Then open:
    http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
for extra in (SRC, ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from agents import (  # noqa: E402
    MCTSV321Agent,
    MCTSV41Agent,
    MCTSV42Agent,
    MCTSV5Agent,
    MCTSV6Agent,
    MCTSV7Agent,
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
    "v7": "MCTS V7 FINAL",
    "v8": "MCTS V8 (개발)",
}


ALPHAZERO_KEY = "az"
_ALPHAZERO_FACTORY = None
S3_KEY = "s3"
_S3_FACTORY = None
_S3_INFO: dict[str, Any] = {}


def register_alphazero(factory, label: str) -> None:
    """Add the "az" opponent; ``factory()`` returns a fresh ``AlphaZeroAgent``."""
    global _ALPHAZERO_FACTORY
    _ALPHAZERO_FACTORY = factory
    VERSION_LABELS[ALPHAZERO_KEY] = label


def register_s3(factory, info: dict[str, Any]) -> None:
    """Add the "s3" opponent; ``factory(seed)`` returns a fresh S3-VCT2-v1 agent."""
    global _S3_FACTORY, _S3_INFO
    from analysis.s3_vct2_v1 import NAME

    _S3_FACTORY, _S3_INFO = factory, info
    VERSION_LABELS[S3_KEY] = f"{NAME} (동결)"


def create_agent(version: str, *, seed: int = 42):
    """Build one of the frozen/versioned agents used by the benchmark scripts."""
    if version == ALPHAZERO_KEY and _ALPHAZERO_FACTORY is not None:
        return _ALPHAZERO_FACTORY()  # deterministic search: the seed is unused
    if version == S3_KEY and _S3_FACTORY is not None:
        return _S3_FACTORY(seed)
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
    if version == "v7":
        return MCTSV7Agent(seed=seed)
    if version == "v8":
        # Imported on demand: V8 lives in ``analysis`` and is kept out of ``agents``.
        from analysis.mcts_v8_agent import MCTSV8Agent
        return MCTSV8Agent(seed=seed)
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
    """Return only the small fields useful while manually testing V5/V6/V7."""
    diagnostics = getattr(agent, "diagnostics", None)
    if diagnostics is None:
        return {}
    if isinstance(diagnostics, dict):  # AlphaZeroAgent: already JSON-shaped
        return _json_value(diagnostics)
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
        "v7_own_vcf_found",
        "v7_own_vcf_length",
        "v7_own_vcf_nodes",
        "v7_safety_checked",
        "v7_safety_removed",
        "v7_safety_inconclusive",
        "v7_safety_augmented",
        "v7_safety_fallback",
        "v7_safety_nodes",
        "v7_safety_budget_exhausted",
        "v7_self_forbidden_penalized",
        "v7_stage4_tiebreak_applied",
        "v7_stage4_vcf_nodes",
        "v7_module_seconds",
        "v8_route",
        "v8_v7_move",
        "v8_changed",
        "v8_vct_checked",
        "v8_vct_widened",
        "v8_vct_calls",
        "v8_vct_nodes",
        "v8_vct_budget_exhausted",
        "v8_vct_seconds",
        "v8_root_visits",
        "v8_attack_status",
        "v8_attack_move",
        "v8_attack_rank",
        "v8_attack_candidates",
        "v8_attack_checked",
        "v8_attack_calls",
        "v8_attack_nodes",
        "v8_attack_budget_exhausted",
        "v8_attack_seconds",
        "v8_root_checked",
        "v8_root_rank",
        "v8_root_calls",
        "v8_root_nodes",
        "v8_root_budget_exhausted",
        "v8_root_seconds",
        "v8_tree_seconds",
        "v8_puct_prior_top",
        "v8_puct_prior_top_prob",
        "v8_vct2_checked",
        "v8_vct2_switched",
        "v8_vct2_nodes",
        "v8_vct2_seconds",
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


V8_CSV_FIELDS = (
    "v8_route", "v8_changed", "v8_v7_move", "v8_attack_status", "v8_attack_move",
    "v8_attack_seconds", "v8_vct_seconds", "v8_vct_budget_exhausted",
    "v8_attack_budget_exhausted", "v8_root_rank", "v8_root_seconds", "v8_root_budget_exhausted",
)


def _v8_csv(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Compact V8 columns; moves are 1-based like the other CSV coordinates."""
    def one_based(move):
        return json.dumps([move[0] + 1, move[1] + 1]) if move else None

    return {
        "v8_route": diagnostics.get("v8_route"),
        "v8_changed": diagnostics.get("v8_changed"),
        "v8_v7_move": one_based(diagnostics.get("v8_v7_move")),
        "v8_attack_status": diagnostics.get("v8_attack_status"),
        "v8_attack_move": one_based(diagnostics.get("v8_attack_move")),
        "v8_attack_seconds": diagnostics.get("v8_attack_seconds"),
        "v8_vct_seconds": diagnostics.get("v8_vct_seconds"),
        "v8_vct_budget_exhausted": diagnostics.get("v8_vct_budget_exhausted"),
        "v8_attack_budget_exhausted": diagnostics.get("v8_attack_budget_exhausted"),
        "v8_root_rank": diagnostics.get("v8_root_rank"),
        "v8_root_seconds": diagnostics.get("v8_root_seconds"),
        "v8_root_budget_exhausted": diagnostics.get("v8_root_budget_exhausted"),
    }


S3_CSV_FIELDS = ("vct2_checked", "vct2_switched", "vct2_nodes", "vct2_seconds", "board_hash")


def _s3_csv(record: dict[str, Any]) -> dict[str, Any]:
    diagnostics = record.get("diagnostics", {})
    checked = diagnostics.get("v8_vct2_checked") or []
    return {
        "vct2_checked": json.dumps([[m[0] + 1, m[1] + 1, status] for m, status in checked]) if checked else None,
        "vct2_switched": diagnostics.get("v8_vct2_switched"),
        "vct2_nodes": diagnostics.get("v8_vct2_nodes"),
        "vct2_seconds": diagnostics.get("v8_vct2_seconds"),
        "board_hash": record.get("board_hash"),
    }


def board_hash(board, to_play: int) -> str:
    """Stones and the side to move (not D4-canonical)."""
    raw = json.dumps([[list(line) for line in board], to_play], separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


ALPHAZERO_CSV_FIELDS = (
    "az_root_value", "az_root_q", "az_tactical_allowed", "az_tactical_proven",
    "az_chosen_visits", "az_chosen_prior", "az_top_visits",
)


def _alphazero_csv(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Compact per-move summary; the full root record stays in game.json."""
    if not diagnostics:
        return {}
    chosen = diagnostics.get("chosen")
    top = diagnostics.get("top_visits", [])
    chosen_row = next((item for item in top if item["move"] == chosen), {})
    return {
        "az_root_value": diagnostics.get("root_value"),
        "az_root_q": diagnostics.get("root_q"),
        "az_tactical_allowed": diagnostics.get("tactical_allowed"),
        "az_tactical_proven": diagnostics.get("tactical_proven"),
        "az_chosen_visits": chosen_row.get("visits"),
        "az_chosen_prior": chosen_row.get("prior"),
        "az_top_visits": json.dumps(
            [[item["move"][0] + 1, item["move"][1] + 1, item["visits"]] for item in top[:3]]
        ),
    }


class PlaySession:
    """Single local-browser game session.

    The server intentionally keeps one game in memory. This is a developer tool,
    not a multi-user service.
    """

    def __init__(self, *, log_root: Path | None = None):
        self.lock = Lock()
        self.log_root = Path(log_root) if log_root is not None else LOG_ROOT
        self.game = Game()
        self.agent_key = "v7"
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
        self.undo_count = 0
        self.veto_files: list[str] = []

    def reset(self, *, agent_key: str, human_color: int, seed: int = 42) -> dict[str, Any]:
        if agent_key not in VERSION_LABELS:
            raise ValueError("지원하지 않는 MCTS 버전입니다.")
        if human_color not in (BLACK, WHITE):
            raise ValueError("human_color는 BLACK 또는 WHITE여야 합니다.")
        with self.lock:
            if (self.agent_key == S3_KEY and not self.saved_game
                    and any(r["actor"] not in ("HUMAN", "OPENING_RULE") for r in self.move_records)):
                self._save_completed_game_locked(unfinished=True)  # relayed games: keep the record
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
            self.undo_count = 0
            self.veto_files = []

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

    def undo(self) -> dict[str, Any]:
        """Take back moves until the last human move is gone (the AI reply first)."""
        with self.lock:
            if self.game.done:
                raise IllegalMove("종료된 대국은 되돌릴 수 없습니다.")
            if not any(r["actor"] == "HUMAN" for r in self.move_records):
                raise IllegalMove("되돌릴 사람 착수가 없습니다.")
            while self.move_records:
                record = self.move_records.pop()
                self.game.undo()
                if record["actor"] == "HUMAN":
                    break
            self.undo_count += 1
            ai = [r for r in self.move_records if r["actor"] not in ("HUMAN", "OPENING_RULE")]
            self.last_ai_move = (ai[-1]["row0"], ai[-1]["col0"]) if ai else None
            self.last_ai_seconds = ai[-1]["seconds"] if ai else None
            self.message = "마지막 사람 착수를 되돌렸습니다."
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
        if self.agent_key == S3_KEY:
            checked = diagnostics.get("v8_vct2_checked") or []
            if checked and checked[0][1] == "UNSAFE":
                self._save_veto_position_locked(move, elapsed, diagnostics)
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
        before = [list(line) for line in self.game.board]
        before[row][col] = 0  # the record keeps the position before the move
        self.move_records.append({
            "ply": len(self.game.history),
            "board_hash": board_hash(before, player),
            "player": "BLACK" if player == BLACK else "WHITE",
            "actor": actor,
            "row0": row,
            "col0": col,
            "row": row + 1,
            "col": col + 1,
            "seconds": seconds,
            "diagnostics": diagnostics,
        })

    def _s3_info(self) -> dict[str, Any]:
        from analysis.s3_vct2_v1 import NAME, RESULTS_COMMIT, S3_VCT2_V1
        from scripts.run_mcts_v8_benchmark import _git_commit, _git_dirty

        return {"engine": NAME, "frozen_results_commit": RESULTS_COMMIT, "engine_commit": _git_commit(),
                "git_dirty": _git_dirty(), "config": dict(S3_VCT2_V1), **_S3_INFO}

    def _save_veto_position_locked(self, move, seconds, diagnostics) -> None:
        """S3-VCT2 proved the move it checked first lost: keep the position at once (§12.25)."""
        folder = self.log_root / "veto_positions"
        folder.mkdir(parents=True, exist_ok=True)
        ply = len(self.game.history)
        stamp = self.started_at.strftime("%Y%m%d-%H%M%S")
        path = folder / f"{stamp}_ply{ply}.json"
        payload = {
            "game_started_at": self.started_at.isoformat(), "ply": ply, "board_hash": board_hash(self.game.board, self.game.to_play),
            "history": [list(m) for m in self.game.history],
            "to_play": "BLACK" if self.game.to_play == BLACK else "WHITE",
            "human_color": "BLACK" if self.human_color == BLACK else "WHITE",
            "tree_move": diagnostics.get("v8_v7_move"), "played": list(move),
            "vct2_checked": diagnostics.get("v8_vct2_checked"), "switched": diagnostics.get("v8_vct2_switched"),
            "seconds": seconds, "diagnostics": diagnostics, "engine": self._s3_info(),
            "note": "coordinates are 0-indexed (row0, col0)",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.veto_files.append(_display_path(path))
        print(f"[web] VCT2 veto position saved: {_display_path(path)}")

    def _save_completed_game_locked(self, unfinished: bool = False) -> Path | None:
        """Persist one finished human-vs-MCTS game exactly once (``unfinished``: an S3 game left early)."""
        if (not self.game.done and not unfinished) or self.saved_game:
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
        if unfinished:
            result = "UNFINISHED"

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
        alphazero = self.agent_key == ALPHAZERO_KEY
        v8 = self.agent_key == "v8"
        if v8:
            from analysis.mcts_v8 import V8_DEFAULTS
            payload["agent_info"] = {"config": _json_value(V8_DEFAULTS)}
        if alphazero:
            payload["agent_info"] = _json_value(getattr(self.agent, "info", {}))
        s3 = self.agent_key == S3_KEY
        if s3:
            payload["agent_info"] = _json_value(self._s3_info())
            payload["undo_count"] = self.undo_count
            payload["veto_positions"] = self.veto_files
        (log_dir / "game.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with (log_dir / "moves.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "ply", "player", "actor", "row0", "col0", "row", "col", "seconds",
                "forced_policy_stage", "simulation_mode", "selected_simulations",
                "best_root_tactical_score", "v6_selected_threat_type",
                "v6_selected_reasons", "v7_own_vcf_found", "v7_safety_removed",
                "v7_safety_inconclusive", "v7_self_forbidden_penalized",
                "v7_stage4_tiebreak_applied", "v7_module_seconds",
            ] + (list(ALPHAZERO_CSV_FIELDS) if alphazero else [])
              + (list(V8_CSV_FIELDS) if v8 or s3 else [])
              + (list(S3_CSV_FIELDS) if s3 else []))
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
                    "v7_own_vcf_found": diagnostics.get("v7_own_vcf_found"),
                    "v7_safety_removed": diagnostics.get("v7_safety_removed"),
                    "v7_safety_inconclusive": diagnostics.get("v7_safety_inconclusive"),
                    "v7_self_forbidden_penalized": diagnostics.get(
                        "v7_self_forbidden_penalized"
                    ),
                    "v7_stage4_tiebreak_applied": diagnostics.get(
                        "v7_stage4_tiebreak_applied"
                    ),
                    "v7_module_seconds": diagnostics.get("v7_module_seconds"),
                    **(_alphazero_csv(diagnostics) if alphazero else {}),
                    **(_v8_csv(diagnostics) if (v8 or s3) and diagnostics else {}),
                    **(_s3_csv(record) if s3 else {}),
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
            "undo_count": self.undo_count,
            "veto_positions": self.veto_files,
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
                    agent_key=str(payload.get("agent", "v7")),
                    human_color=human_color,
                    seed=int(payload.get("seed", 42)),
                )
                self._send_json(state)
                return
            if self.path == "/api/undo":
                self._send_json(SESSION.undo())
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
    parser.add_argument("--az-checkpoint", type=Path,
                        help="training checkpoint for the AlphaZero opponent (needs torch)")
    parser.add_argument("--az-simulations", type=int,
                        help="PUCT simulations per move (default: the checkpoint's evaluation setting)")
    parser.add_argument("--az-tactical-rules", choices=("auto", "on", "off"), default="auto",
                        help="PUCT v2 rules (auto = the checkpoint's evaluation setting)")
    parser.add_argument("--policy-checkpoint", type=Path, default=ROOT / "runs/h3_policy_64x4/best.pt",
                        help="H3 policy checkpoint of the frozen S3-VCT2-v1 opponent (needs torch)")
    args = parser.parse_args()
    if args.policy_checkpoint.exists():
        from analysis.s3_vct2_v1 import check_checkpoint, make_agent
        from hybrid.h4_policy import RootPolicy

        hashes = check_checkpoint(args.policy_checkpoint)  # refuses any other checkpoint
        policy = RootPolicy(args.policy_checkpoint, threads=None)
        register_s3(lambda seed: make_agent(policy, seed=seed),
                    {"policy_checkpoint": str(args.policy_checkpoint), **hashes})
        print(f"S3-VCT2-v1 opponent: {args.policy_checkpoint} (hash OK)")
    else:
        print(f"S3-VCT2-v1 opponent off: {args.policy_checkpoint} not found")
    if args.az_checkpoint is not None:
        from agents.alphazero_agent import AlphaZeroAgent

        probe = AlphaZeroAgent.from_checkpoint(
            args.az_checkpoint, simulations=args.az_simulations,
            tactical_rules=args.az_tactical_rules)
        register_alphazero(
            lambda: AlphaZeroAgent(probe.evaluator, probe.config, name=probe.name, info=probe.info),
            f"{probe.name} ({probe.config.num_simulations} sims)")
        print(f"AlphaZero opponent: {probe.name}, {probe.config.to_dict()}")
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
