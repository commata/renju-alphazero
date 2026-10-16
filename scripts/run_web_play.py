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
The frozen TB-Engine-v1 (E3-A, §12.36; same checkpoint) is the "tb1" opponent and is logged the same way.
"Undo" takes back the last human move (and the AI reply) for relaying games from elsewhere.

Every browser gets its own game (cookie session); AI replies go through a queue and the page
polls ``/api/state`` (docs/web-ser5.md §8: ``--max-sessions``, ``--max-queue``, ``--ai-slots``).

With ``--rt30`` the "rt30" opponent is TB-RT30 (docs/web-ser5.md): searched in a separate engine
worker process under an absolute deadline counted from the search start (27.5 s) with a 29 s watchdog;
it uses the H3 checkpoint of ``--policy-checkpoint``, or ``--rt30-uniform-prior`` when there is none
(a test engine, not TB1).

With ``--db data/renju_web.db`` every game is also logged move by move to SQLite
(``weblog.store``, docs/web-ser5.md §7); the files above are written as before.

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
from time import monotonic, perf_counter
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
TB1_KEY = "tb1"
_TB1_FACTORY = None
_TB1_INFO: dict[str, Any] = {}
FROZEN_KEYS = (S3_KEY, TB1_KEY)  # frozen Track B engines: full provenance and veto positions in the logs
RT30_KEY = "rt30"
_RT30: dict[str, Any] = {}  # supervisor, label, info (register_rt30)
TRACKED_KEYS = FROZEN_KEYS + (RT30_KEY,)  # logged like the frozen engines (TB-RT30 runs the TB1 flow)


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


def register_tb1(factory, info: dict[str, Any]) -> None:
    """Add the "tb1" opponent; ``factory(seed)`` returns a fresh TB-Engine-v1 agent."""
    global _TB1_FACTORY, _TB1_INFO
    from analysis.tb_engine_v1 import NAME

    _TB1_FACTORY, _TB1_INFO = factory, info
    VERSION_LABELS[TB1_KEY] = f"{NAME} (동결)"


_EXTRA_FACTORIES: dict[str, Any] = {}  # register_agent


def register_agent(key: str, label: str, factory) -> None:
    """Add an opponent ``key``; ``factory(seed)`` returns a fresh agent (tests, experiments)."""
    _EXTRA_FACTORIES[key] = factory
    VERSION_LABELS[key] = label


class SupervisedAgent:
    """An opponent searched by ``rt30.supervisor.EngineSupervisor`` in a worker process.

    ``select_move(game, received_at=..., slot=...)`` returns the move; ``last`` keeps the full result
    (times, source, watchdog), ``diagnostics`` the worker's diagnostics (a dict). ``supervisors`` has
    one supervisor per AI search slot (``AIJobQueue`` keeps a game on the slot of its first search,
    because the worker keeps the game's agent and its random stream)."""

    def __init__(self, supervisors, name: str, seed: int):
        import uuid

        self.supervisors = supervisors if isinstance(supervisors, (list, tuple)) else [supervisors]
        self.name, self.seed = name, seed
        self.game_key = uuid.uuid4().hex  # one engine agent (random stream) per game in the worker
        self.diagnostics: dict[str, Any] = {}
        self.last = None

    @property
    def supervisor(self):
        return self.supervisors[0]

    def select_move(self, game: Game, received_at: float | None = None, slot: int | None = None) -> tuple[int, int]:
        supervisor = self.supervisors[(slot or 0) % len(self.supervisors)]
        result = supervisor.select(game_key=self.game_key, seed=self.seed, history=list(game.history),
                                   received_at=received_at)
        self.diagnostics, self.last = result.diagnostics, result
        return result.move


def register_rt30(supervisors, label: str, info: dict[str, Any]) -> None:
    """Add the "rt30" opponent, searched by ``supervisors`` (one per AI search slot, or a single one)."""
    supervisors = list(supervisors) if isinstance(supervisors, (list, tuple)) else [supervisors]
    _RT30.update(supervisor=supervisors[0], supervisors=supervisors, label=label, info=info)
    VERSION_LABELS[RT30_KEY] = label


def create_agent(version: str, *, seed: int = 42):
    """Build one of the frozen/versioned agents used by the benchmark scripts."""
    if version == ALPHAZERO_KEY and _ALPHAZERO_FACTORY is not None:
        return _ALPHAZERO_FACTORY()  # deterministic search: the seed is unused
    if version == S3_KEY and _S3_FACTORY is not None:
        return _S3_FACTORY(seed)
    if version == TB1_KEY and _TB1_FACTORY is not None:
        return _TB1_FACTORY(seed)
    if version == RT30_KEY and _RT30:
        return SupervisedAgent(_RT30["supervisors"], _RT30["info"]["engine"], seed)
    if version in _EXTRA_FACTORIES:
        return _EXTRA_FACTORIES[version](seed)
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


class SessionBusy(Exception):
    """The AI is queued or searching for this session (HTTP 409)."""


class PlaySession:
    """One browser's game (``webplay.sessions.SessionManager`` keeps one per cookie, docs/web-ser5.md §8).

    Two ways to get the AI's reply:

    - synchronous (``reset`` / ``play_human``): the search runs in the caller, as the local tool and
      the tests use it;
    - queued (``new_game_async`` / ``play_human_async``): the human move is applied, an ``AIJob`` is
      queued and the call returns at once (``status`` QUEUED). A search slot runs the job with no
      session lock held, on a copy of the game, and applies the move only if the game is still the
      one it searched (``epoch``); ``state()`` (polling) never waits for a search.

    While a job is queued or running, move / undo are refused (``SessionBusy``); a new game cancels
    a job that has not started and is refused while one is running.
    """

    def __init__(self, *, log_root: Path | None = None, store=None, session_hash: str | None = None):
        self.lock = Lock()
        self.session_hash = session_hash
        self.status = "READY"            # READY / QUEUED / THINKING (FINISHED is ``done``)
        self.epoch = 0                    # bumped by every change a queued AI reply must not outlive
        self.job = None                   # the AIJob of this session, while QUEUED / THINKING
        self.jobs = None                  # the AIJobQueue of that job
        self.slot: int | None = None      # search slot of this game (engine worker affinity)
        self.last_error: str | None = None
        self.thinking_since: float | None = None
        self._last_diagnostics: dict[str, Any] = {}
        self.store = store  # weblog.store.LogStore or None
        self.db_game_id: str | None = None
        self._db_seqs: list[int] = []  # seq of each entry of move_records
        self._db_next_seq = 0
        self._db_finished = False
        self._received_at: float | None = None
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

    def reset(self, *, agent_key: str, human_color: int, seed: int = 42,
              received_at: float | None = None) -> dict[str, Any]:
        if agent_key not in VERSION_LABELS:
            raise ValueError("지원하지 않는 MCTS 버전입니다.")
        if human_color not in (BLACK, WHITE):
            raise ValueError("human_color는 BLACK 또는 WHITE여야 합니다.")
        with self.lock:
            if self.busy():
                raise SessionBusy("AI가 생각 중입니다.")
            self._new_game_locked(agent_key, human_color, seed, received_at)
            if not self.game.done and self.game.to_play != self.human_color:
                self._play_ai_locked()
            return self._state_locked()

    def _leave_game_locked(self) -> None:
        """The current game is replaced or its session closed before it ended."""
        if (self.agent_key in TRACKED_KEYS and not self.saved_game
                and any(r["actor"] not in ("HUMAN", "OPENING_RULE") for r in self.move_records)):
            self._save_completed_game_locked(unfinished=True)  # relayed games: keep the record
        self._db_leave_locked()

    def _new_game_locked(self, agent_key: str, human_color: int, seed: int, received_at: float | None) -> None:
        self._leave_game_locked()
        self.epoch += 1
        self.slot = None
        self.last_error = None
        self._last_diagnostics = {}
        self._received_at = received_at
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
        self._db_start_locked()

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

    def play_human(self, row: int, col: int, *, received_at: float | None = None) -> dict[str, Any]:
        with self.lock:
            if self.busy():
                raise SessionBusy("AI가 생각 중입니다.")
            self._received_at = received_at
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
            if self.busy():
                raise SessionBusy("AI가 생각 중입니다.")
            if self.game.done:
                raise IllegalMove("종료된 대국은 되돌릴 수 없습니다.")
            if not any(r["actor"] == "HUMAN" for r in self.move_records):
                raise IllegalMove("되돌릴 사람 착수가 없습니다.")
            undone = []
            while self.move_records:
                record = self.move_records.pop()
                if self._db_seqs:
                    undone.append(self._db_seqs.pop())
                self.game.undo()
                if record["actor"] == "HUMAN":
                    break
            self.undo_count += 1
            self.epoch += 1
            self.last_error = None
            if self.store is not None and self.db_game_id is not None:
                self.store.mark_undone(self.db_game_id, undone)
            ai = [r for r in self.move_records if r["actor"] not in ("HUMAN", "OPENING_RULE")]
            self.last_ai_move = (ai[-1]["row0"], ai[-1]["col0"]) if ai else None
            self.last_ai_seconds = ai[-1]["seconds"] if ai else None
            self.message = "마지막 사람 착수를 되돌렸습니다."
            return self._state_locked()

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self._state_locked()

    # -- queued AI replies (docs/web-ser5.md §8) ---------------------------------------
    def busy(self) -> bool:
        return self.status in ("QUEUED", "THINKING")

    def new_game_async(self, *, agent_key: str, human_color: int, seed: int, jobs,
                       received_at: float | None = None) -> dict[str, Any]:
        """``reset`` with the AI's first reply (human white) queued on ``jobs``."""
        if agent_key not in VERSION_LABELS:
            raise ValueError("지원하지 않는 MCTS 버전입니다.")
        if human_color not in (BLACK, WHITE):
            raise ValueError("human_color는 BLACK 또는 WHITE여야 합니다.")
        with self.lock:
            if self.status == "THINKING":
                raise SessionBusy("AI가 생각 중입니다. 끝난 뒤에 새 게임을 시작하세요.")
            if self.status == "QUEUED":
                if not self.jobs.cancel(self.job):  # a slot took it just now
                    raise SessionBusy("AI가 생각 중입니다. 끝난 뒤에 새 게임을 시작하세요.")
                self.job, self.status = None, "READY"
            # After the centre opening it is white's turn: the AI replies first when the human is black.
            ticket = jobs.reserve() if human_color == BLACK else None  # room first (503 changes nothing)
            try:
                self._new_game_locked(agent_key, human_color, seed, received_at)
            except Exception:
                if ticket is not None:
                    jobs.release(ticket)
                raise
            if ticket is not None:
                try:
                    if not self.game.done and self.game.to_play != self.human_color:
                        self._enqueue_locked(jobs, ticket, received_at)
                finally:
                    jobs.release(ticket)  # no-op once submitted
            return self._state_locked()

    def play_human_async(self, row: int, col: int, *, jobs, received_at: float | None = None) -> dict[str, Any]:
        """``play_human`` with the AI's reply queued on ``jobs`` (raises ``QueueFull`` before any change)."""
        with self.lock:
            if self.busy():
                raise SessionBusy("AI가 생각 중입니다.")
            if self.game.done:
                raise IllegalMove("이미 종료된 대국입니다.")
            if self.game.to_play != self.human_color:
                raise IllegalMove("현재는 AI 차례입니다.")
            ticket = jobs.reserve()
            player = self.game.to_play
            try:
                self.game.play(row, col)
            except Exception:
                jobs.release(ticket)
                raise
            try:
                self._received_at = received_at
                self.last_error = None
                self._record_move_locked(player, "HUMAN", (row, col), seconds=None, diagnostics={})
                self.message = f"사람 착수: ({row + 1}, {col + 1})"
                if self.game.done:
                    jobs.release(ticket)
                    self._save_completed_game_locked()
                else:
                    self._enqueue_locked(jobs, ticket, received_at)
            finally:
                jobs.release(ticket)  # no-op once submitted or released: never leak a queue place
            return self._state_locked()

    def close(self) -> None:
        """The session expired or the server stops: cancel a queued reply, mark the game left."""
        with self.lock:
            if self.status == "QUEUED" and self.jobs is not None and self.jobs.cancel(self.job):
                self.job, self.status = None, "READY"
            self.epoch += 1
            if not self.saved_game and not self.game.done:
                self._leave_game_locked()

    def _enqueue_locked(self, jobs, ticket, received_at: float | None) -> None:
        from webplay.jobs import AIJob

        self.epoch += 1
        snapshot = {"epoch": self.epoch, "history": list(self.game.history), "agent": self.agent,
                    "player": self.game.to_play, "received_at": monotonic() if received_at is None else received_at}
        job = AIJob(run=lambda slot: self._run_job(job, snapshot, slot), session=self, slot=self.slot)
        self.job, self.jobs, self.status = job, jobs, "QUEUED"
        self.message = "AI 차례: 대기 중입니다."
        jobs.submit(ticket, job)

    def _run_job(self, job, snapshot: dict[str, Any], slot: int) -> None:
        """In a search slot: search with no lock held, then apply the reply if the game is unchanged."""
        with self.lock:
            if self.job is not job or self.epoch != snapshot["epoch"]:
                self._drop_job_locked(job)  # cancelled or replaced while queued
                return
            self.status, self.slot = "THINKING", slot
            self.thinking_since = started = monotonic()
            self.message = "AI가 생각 중입니다."
        agent, received = snapshot["agent"], snapshot["received_at"]
        try:
            game = Game()  # a private copy: searches place probe stones on their board
            for move in snapshot["history"]:
                game.play(*move)
            supervised = isinstance(agent, SupervisedAgent)
            t = perf_counter()
            if supervised:
                move = agent.select_move(game, received_at=started, slot=slot)  # deadline from the search start
            else:
                move = agent.select_move(game)
            elapsed = perf_counter() - t
            diagnostics = _diagnostics(agent)
        except Exception as exc:  # noqa: BLE001 - reported to the browser, the session stays usable
            with self.lock:
                if self.job is job:
                    self.job, self.status, self.thinking_since = None, "READY", None
                    self.last_error = f"AI 탐색 오류: {type(exc).__name__}: {exc}"
                    self.message = self.last_error
            print(f"[web] AI search failed: {type(exc).__name__}: {exc}", flush=True)
            return
        with self.lock:
            if self.job is not job or self.epoch != snapshot["epoch"]:
                self._drop_job_locked(job)  # the game changed meanwhile: drop the reply
                return
            self.job, self.status, self.thinking_since = None, "READY", None
            now = monotonic()
            timing: dict[str, Any] = {"queue_seconds": started - received, "wall_seconds": now - received}
            if supervised:
                result = agent.last
                timing.update(self._supervised_timing(result))
                timing["queue_seconds"] = started - received + (result.queue_seconds or 0.0)
                timing["wall_seconds"] = now - received
                if result.engine_seconds is not None:
                    elapsed = result.engine_seconds
            self._apply_ai_locked(snapshot["player"], move, elapsed, diagnostics, timing)

    def _drop_job_locked(self, job) -> None:
        """A reply that will not be applied: the session must not stay QUEUED / THINKING for it."""
        if self.job is job:
            self.job, self.status, self.thinking_since = None, "READY", None

    @staticmethod
    def _supervised_timing(result) -> dict[str, Any]:
        rt = {**result.rt, "source": result.source, "watchdog": result.watchdog,
              "worker_restarted": result.worker_restarted, "best": result.best, "error": result.error}
        return {"queue_seconds": result.queue_seconds, "wall_seconds": result.wall_seconds,
                "deadline_seconds": result.deadline_seconds, "rt": rt,
                "timed_out": bool(result.rt.get("timeouts") or result.rt.get("skipped")) or result.watchdog,
                "emergency_used": result.source != "engine"}

    def _play_ai_locked(self) -> None:
        if self.game.done or self.game.to_play == self.human_color:
            return
        player = self.game.to_play
        started = perf_counter()
        received = self._received_at
        supervised = isinstance(self.agent, SupervisedAgent)
        if supervised:
            move = self.agent.select_move(self.game, received_at=received)
        else:
            move = self.agent.select_move(self.game)
        elapsed = perf_counter() - started
        if supervised:
            result = self.agent.last
            timing = self._supervised_timing(result)
            if result.engine_seconds is not None:
                elapsed = result.engine_seconds
        else:
            timing = {} if received is None else {"queue_seconds": monotonic() - received - elapsed,
                                                  "wall_seconds": monotonic() - received}
        self._apply_ai_locked(player, move, elapsed, _diagnostics(self.agent), timing)

    def _apply_ai_locked(self, player: int, move, elapsed: float, diagnostics: dict[str, Any],
                         timing: dict[str, Any]) -> None:
        self._last_diagnostics = diagnostics
        if self.agent_key in TRACKED_KEYS:
            checked = diagnostics.get("v8_vct2_checked") or []
            if checked and checked[0][1] == "UNSAFE":
                self._save_veto_position_locked(move, elapsed, diagnostics)
        self.game.play(*move)
        self._record_move_locked(player, self.agent.name, move, seconds=elapsed, diagnostics=diagnostics, **timing)
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
        queue_seconds: float | None = None,
        wall_seconds: float | None = None,
        deadline_seconds: float | None = None,
        rt: dict[str, Any] | None = None,
        timed_out: bool | None = None,
        emergency_used: bool | None = None,
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
        if self.store is not None:
            if self.db_game_id is None:  # a game played without /api/new (the initial session)
                self._db_start_locked()
            seq = self._db_next_seq
            self._db_next_seq += 1
            self._db_seqs.append(seq)
            record = self.move_records[-1]
            self.store.record_move(
                self.db_game_id, seq=seq, ply=record["ply"], player=record["player"], actor=actor, row0=row, col0=col,
                board_hash=record["board_hash"], engine_seconds=seconds, queue_seconds=queue_seconds,
                wall_seconds=wall_seconds, deadline_seconds=deadline_seconds, route=diagnostics.get("v8_route"),
                timed_out=timed_out, emergency_used=emergency_used, diagnostics=_json_value(diagnostics),
                rt=None if rt is None else _json_value(rt))

    def _db_start_locked(self) -> None:
        """A new row in ``games`` for the current game (provenance of the frozen engines included)."""
        if self.store is None:
            return
        from weblog.store import utc

        info = self._s3_info() if self.agent_key in TRACKED_KEYS else {}
        self.db_game_id = self.store.start_game(
            started_at=utc(self.started_at), agent_key=self.agent_key, agent_name=self.agent.name, seed=self.seed,
            human_color="BLACK" if self.human_color == BLACK else "WHITE", engine=info.get("engine"),
            engine_config_hash=info.get("engine_config_hash"), engine_commit=info.get("engine_commit"),
            git_dirty=info.get("git_dirty"), policy_checkpoint_sha256=info.get("policy_checkpoint_sha256"),
            policy_metadata_sha256=info.get("policy_metadata_sha256"), session_hash=self.session_hash)
        self._db_seqs, self._db_next_seq, self._db_finished = [], 0, False

    def _db_leave_locked(self) -> None:
        """The current game is replaced before it ended and was not saved: mark it abandoned."""
        if self.store is None or self.db_game_id is None or self._db_finished:
            return
        self.store.finish_game(self.db_game_id, status="abandoned", number_of_moves=len(self.game.history),
                               undo_count=self.undo_count, final_board=self.game.board)
        self._db_finished = True

    def _s3_info(self) -> dict[str, Any]:
        """Provenance of the frozen engine of this game (S3-VCT2-v1 or TB-Engine-v1)."""
        from scripts.run_mcts_v8_benchmark import _git_commit, _git_dirty

        if self.agent_key == RT30_KEY:
            from rt30.tb_rt30_v1 import TB_RT30_V1

            return {**_RT30["info"], "config": TB_RT30_V1, "engine_commit": _git_commit(), "git_dirty": _git_dirty()}
        if self.agent_key == TB1_KEY:
            from analysis.tb_engine_v1 import FREEZE_EVIDENCE, NAME, TB_ENGINE_V1, config_hash

            return {"engine": NAME, "freeze_evidence": FREEZE_EVIDENCE, "engine_config_hash": config_hash(),
                    "engine_commit": _git_commit(), "git_dirty": _git_dirty(), "config": dict(TB_ENGINE_V1),
                    **_TB1_INFO}
        from analysis.s3_vct2_v1 import NAME, RESULTS_COMMIT, S3_VCT2_V1

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
        if self.store is not None and self.db_game_id is not None:
            self.store.record_veto(self.db_game_id, ply=ply, payload=_json_value(payload), raw_path=_display_path(path))
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
        s3 = self.agent_key in TRACKED_KEYS
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
        if self.store is not None and self.db_game_id is not None and not self._db_finished:
            from weblog.store import utc

            self.store.finish_game(
                self.db_game_id, finished_at=utc(finished_at), status="unfinished" if unfinished else "finished",
                result=result, winner=winner, winner_actor=winner_actor, number_of_moves=len(self.game.history),
                ai_total_seconds=ai_total_seconds, undo_count=self.undo_count, raw_log_dir=display_path,
                final_board=self.game.board, agent_info=payload.get("agent_info"))
            self._db_finished = True
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
            "diagnostics": self._last_diagnostics,
            "message": self.message,
            "status": "FINISHED" if self.game.done and not self.busy() else self.status,
            "queue_position": (self.jobs.position(self.job)
                               if self.status == "QUEUED" and self.jobs is not None else None),
            "thinking_seconds": (round(monotonic() - self.thinking_since, 1)
                                 if self.status == "THINKING" and self.thinking_since is not None else None),
            "last_error": self.last_error,
            "epoch": self.epoch,
            "log_saved": self.saved_game,
            "log_directory": (
                _display_path(self.last_log_dir)
                if self.last_log_dir is not None else None
            ),
            "versions": VERSION_LABELS,
            "undo_count": self.undo_count,
            "veto_positions": self.veto_files,
        }


class WebApp:
    """Sessions and the AI job queue of one server (docs/web-ser5.md §8)."""

    def __init__(self, *, store=None, log_root: Path | None = None, max_sessions: int = 16,
                 session_ttl: float = 3600.0, max_queue: int = 4, ai_slots: int = 1, secure_cookie: bool = False):
        from webplay.jobs import AIJobQueue
        from webplay.sessions import SessionManager

        self.store, self.secure_cookie = store, secure_cookie
        self.sessions = SessionManager(
            lambda digest: PlaySession(store=store, log_root=log_root, session_hash=digest),
            max_sessions=max_sessions, ttl_seconds=session_ttl)
        self.jobs = AIJobQueue(max_queue=max_queue, slots=ai_slots)

    def close(self) -> None:
        self.jobs.close()
        self.sessions.close_all()


APP: WebApp | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "RenjuWeb/2.0"

    def do_GET(self):
        self._cookie = None
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
        try:
            if self.path == "/api/state":
                self._send_json(self._session().state())
                return
            if self.path == "/api/health":
                store = APP.store
                db = None if store is None else {"path": _display_path(store.path), "written": store.written,
                                                 "failures": store.failures, "last_error": store.last_error}
                self._send_json({"ok": True, "db": db, "sessions": len(APP.sessions),
                                 "sessions_expired": APP.sessions.expired, "ai": APP.jobs.stats()})
                return
        except Exception as exc:  # noqa: BLE001
            self._error(exc)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        received_at = monotonic()  # wall time of an AI move counts from here (its deadline from the search start)
        self._cookie = None
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            self._send_json({"error": "Content-Type: application/json 요청만 받습니다."},
                            HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return
        try:
            payload = self._read_json()
            session = self._session()
            if self.path == "/api/new":
                color = str(payload.get("human_color", "BLACK")).upper()
                human_color = BLACK if color == "BLACK" else WHITE if color == "WHITE" else 0
                state = session.new_game_async(agent_key=str(payload.get("agent", "v7")), human_color=human_color,
                                               seed=int(payload.get("seed", 42)), jobs=APP.jobs,
                                               received_at=received_at)
            elif self.path == "/api/undo":
                state = session.undo()
            elif self.path == "/api/move":
                state = session.play_human_async(int(payload["row"]), int(payload["col"]), jobs=APP.jobs,
                                                 received_at=received_at)
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            waiting = state.get("status") in ("QUEUED", "THINKING")
            self._send_json(state, HTTPStatus.ACCEPTED if waiting else HTTPStatus.OK)
        except Exception as exc:  # noqa: BLE001
            self._error(exc)

    def _session(self) -> PlaySession:
        from webplay.sessions import cookie_header, session_id_from_cookie

        sid = session_id_from_cookie(self.headers.get("Cookie"))
        new_id, session, created = APP.sessions.get_or_create(sid)
        if created:
            secure = APP.secure_cookie or self.headers.get("X-Forwarded-Proto", "").lower() == "https"
            self._cookie = cookie_header(new_id, secure=secure)
        return session

    def _error(self, exc: Exception) -> None:
        from webplay.jobs import QueueFull
        from webplay.sessions import SessionLimit

        if isinstance(exc, SessionBusy):
            self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
        elif isinstance(exc, QueueFull):
            self._send_json({"error": f"AI 대기열이 가득 찼습니다. 잠시 뒤에 다시 두세요. ({exc})"},
                            HTTPStatus.SERVICE_UNAVAILABLE, retry_after=10)
        elif isinstance(exc, SessionLimit):
            self._send_json({"error": f"동시 접속이 가득 찼습니다. 잠시 뒤에 다시 접속하세요. ({exc})"},
                            HTTPStatus.SERVICE_UNAVAILABLE, retry_after=60)
        elif isinstance(exc, (ValueError, KeyError, TypeError, IllegalMove)):
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        else:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON object가 필요합니다.")
        return value

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK, retry_after: int | None = None):
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        if getattr(self, "_cookie", None):
            self.send_header("Set-Cookie", self._cookie)
        if retry_after is not None:
            self.send_header("Retry-After", str(retry_after))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args):
        if "/api/state" in (args[0] if args else ""):
            return  # polling: one line a second per browser would drown the log
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
    parser.add_argument("--rt30", action="store_true",
                        help="add the TB-RT30 opponent (engine worker process, 27.5 s deadline, 29 s watchdog)")
    parser.add_argument("--rt30-uniform-prior", action="store_true",
                        help="TB-RT30 without the H3 checkpoint (uniform PUCT prior; a test engine, not TB1)")
    parser.add_argument("--rt30-deadline", type=float, help="override TB-RT30-v1's 27.5 s (experiments only)")
    parser.add_argument("--rt30-watchdog", type=float, help="override TB-RT30-v1's 29 s (experiments only)")
    parser.add_argument("--db", type=Path,
                        help="also log every game move by move to this SQLite file (e.g. data/renju_web.db)")
    parser.add_argument("--max-sessions", type=int, default=16, help="browser sessions kept at once")
    parser.add_argument("--session-ttl", type=float, default=3600.0, help="seconds an idle session is kept")
    parser.add_argument("--max-queue", type=int, default=4, help="AI moves waiting for a search slot (then 503)")
    parser.add_argument("--ai-slots", type=int, default=1,
                        help="AI searches at once (one TB-RT30 engine worker per slot; measure before raising, B3)")
    parser.add_argument("--secure-cookie", action="store_true",
                        help="send the session cookie with Secure (behind HTTPS; X-Forwarded-Proto: https also does)")
    args = parser.parse_args()
    global APP
    # Bind first: a port in use fails here, before the engine workers load the policy.
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        parser.exit(1, f"{args.host}:{args.port} 에 바인드할 수 없습니다: {exc}\n"
                       f"이미 떠 있는 서버가 있으면 먼저 끄세요 (확인: ss -ltnp 'sport = :{args.port}').\n")
    store = None
    if args.db is not None:
        from weblog.store import LogStore

        store = LogStore(args.db)
        print(f"SQLite game log: {_display_path(store.path)}")
    if args.policy_checkpoint.exists():
        from analysis.s3_vct2_v1 import check_checkpoint, make_agent
        from hybrid.h4_policy import RootPolicy

        hashes = check_checkpoint(args.policy_checkpoint)  # refuses any other checkpoint
        policy = RootPolicy(args.policy_checkpoint, threads=None)
        register_s3(lambda seed: make_agent(policy, seed=seed),
                    {"policy_checkpoint": str(args.policy_checkpoint), **hashes})
        print(f"S3-VCT2-v1 opponent: {args.policy_checkpoint} (hash OK)")
        from analysis.tb_engine_v1 import make_agent as make_tb1

        register_tb1(lambda seed: make_tb1(policy, seed=seed),
                     {"policy_checkpoint": str(args.policy_checkpoint), **hashes})
        print("TB-Engine-v1 opponent: same checkpoint")
    else:
        print(f"S3-VCT2-v1 opponent off: {args.policy_checkpoint} not found")
    supervisors = []
    if args.rt30:
        from rt30 import tb_rt30_v1
        from rt30.supervisor import AgentSpec, EngineSupervisor

        if args.policy_checkpoint.exists():
            kwargs, engine = {"checkpoint": str(args.policy_checkpoint)}, tb_rt30_v1.NAME
            extra = {"policy_checkpoint": str(args.policy_checkpoint),
                     **tb_rt30_v1.check_checkpoint(args.policy_checkpoint)}
        elif args.rt30_uniform_prior:
            kwargs, engine, extra = {"uniform_prior": True}, f"{tb_rt30_v1.NAME} (uniform prior)", {}
        else:
            parser.error("--rt30 needs --policy-checkpoint (found) or --rt30-uniform-prior")
        deadline = args.rt30_deadline or tb_rt30_v1.TB_RT30_V1["deadline_seconds"]
        watchdog = args.rt30_watchdog or tb_rt30_v1.TB_RT30_V1["watchdog_seconds"]
        if (args.rt30_deadline, args.rt30_watchdog) != (None, None):
            engine += " (deadline override)"
        print(f"{engine}: starting {args.ai_slots} engine worker(s) ...", flush=True)
        supervisors = [EngineSupervisor(AgentSpec("rt30.supervisor:make_tb_rt30_v1_agent", kwargs),
                                        engine_deadline=deadline, watchdog=watchdog,
                                        agent_cache=max(8, args.max_sessions))
                       for _ in range(args.ai_slots)]
        register_rt30(supervisors, f"{engine} (실시간 {deadline:g}초)",
                      {"engine": engine, "engine_config_hash": tb_rt30_v1.config_hash(),
                       "deadline_seconds": deadline, "watchdog_seconds": watchdog,
                       "freeze_evidence": tb_rt30_v1.FREEZE_EVIDENCE, **extra})
        print(f"{engine}: worker ready (deadline {deadline:g} s, watchdog {watchdog:g} s, "
              f"config {tb_rt30_v1.config_hash()[:12]})")
    if args.az_checkpoint is not None:
        from agents.alphazero_agent import AlphaZeroAgent

        probe = AlphaZeroAgent.from_checkpoint(
            args.az_checkpoint, simulations=args.az_simulations,
            tactical_rules=args.az_tactical_rules)
        register_alphazero(
            lambda: AlphaZeroAgent(probe.evaluator, probe.config, name=probe.name, info=probe.info),
            f"{probe.name} ({probe.config.num_simulations} sims)")
        print(f"AlphaZero opponent: {probe.name}, {probe.config.to_dict()}")
    APP = WebApp(store=store, max_sessions=args.max_sessions, session_ttl=args.session_ttl,
                 max_queue=args.max_queue, ai_slots=args.ai_slots, secure_cookie=args.secure_cookie)
    print(f"sessions: up to {args.max_sessions} (idle {args.session_ttl:g} s), AI slots {args.ai_slots}, "
          f"queue {args.max_queue}")
    print(f"Renju local web: http://{args.host}:{args.port}")
    print("종료: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    finally:
        server.server_close()
        APP.close()  # cancels queued replies; games in progress are marked abandoned
        for supervisor in supervisors:
            supervisor.close()
        if store is not None:
            store.close()  # a game in progress stays 'in_progress' in the database


if __name__ == "__main__":
    main()
