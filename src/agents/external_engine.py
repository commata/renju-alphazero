"""External Renju engines (Rapfi, Yixin, other Gomocup engines) over the Piskvork protocol.

Evaluation only (docs/external-engine.md): an external engine is a fixed outside opponent and a rule
cross-check, never a teacher. Nothing here writes training data.

Protocol (https://plastovicka.github.io/protocl2en.htm, Yixin-Board extension):

    START 15            -> OK
    INFO rule 4         (renju; always sent)
    INFO <key> <value>  (time / node / depth / thread limits from ``EngineConfig.info``)
    BOARD               -> "x,y"   (the engine's move)
    x,y,field           one line per stone in move order; field 1 = the engine's own stone
    ...                 (= the side to move), 2 = the opponent's stone. Not black/white:
    DONE                the engine takes the first stone as black.

Coordinates: Piskvork ``x,y`` is (column, row); this project uses (row, col). A consistent transpose
would only reflect the whole game (Renju is D4 invariant), but a mixed one breaks it, so both
directions go through ``to_piskvork`` / ``from_piskvork`` and are tested round trip.

The engine runs in its own process; one reader thread turns stdout into a queue so every reply has a
wall-clock timeout on every OS (Windows cannot ``select`` on pipes). ``MESSAGE``/``DEBUG`` lines are
kept per move (``last_messages``) for diagnostics; ``ERROR`` lines are kept in ``errors``.

The referee is always our ``Game``: a reply that ``Game.play`` would reject raises
``EngineIllegalMove`` (with the forbidden reason), which a match records as ``engine_illegal``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
import subprocess
from threading import Thread
from time import perf_counter

from renju import BLACK, Game, IllegalMove
from renju.rules import SIZE, forbidden_reason, inside

RENJU_RULE = 4
_MOVE = re.compile(r'^\s*(\d+)\s*,\s*(\d+)\s*$')


class EngineError(RuntimeError):
    """The engine failed: no start, crash, protocol violation or an unparsable reply."""


class EngineTimeout(EngineError):
    """No reply within the wall-clock guard."""


class EngineIllegalMove(EngineError):
    """A reply our rules reject. ``reason`` is the black forbidden reason, or 'occupied'/'outside'."""

    def __init__(self, move: tuple[int, int], reason: str):
        super().__init__(f'engine move {move} is illegal under our rules: {reason}')
        self.move, self.reason = move, reason


def to_piskvork(move: tuple[int, int]) -> str:
    row, col = move
    return f'{col},{row}'


def from_piskvork(text: str) -> tuple[int, int]:
    match = _MOVE.match(text)
    if match is None:
        raise EngineError(f'not a move reply: {text!r}')
    x, y = int(match[1]), int(match[2])
    return y, x


def board_lines(history, to_play: int) -> list[str]:
    """``BOARD`` body: stones in move order, 1 = side to move, 2 = opponent (first stone is black)."""
    lines = []
    for ply, move in enumerate(history):
        color = BLACK if ply % 2 == 0 else -BLACK
        lines.append(f'{to_piskvork(move)},{1 if color == to_play else 2}')
    return lines


def file_sha256(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EngineConfig:
    """How to run one engine at one fixed strength. ``label`` keys results, so a setting change is a new key."""
    command: tuple[str, ...]
    name: str = 'engine'
    info: tuple[tuple[str, str], ...] = ()  # extra INFO lines, e.g. (('max_node', '100000'), ('thread_num', '1'))
    cwd: str | None = None                  # default: the executable's folder (Rapfi finds config.toml there)
    start_timeout: float = 60.0
    move_timeout: float = 600.0             # wall-clock guard per reply, above any engine-side limit
    extra_files: tuple[str, ...] = field(default=())  # hashed into provenance (config.toml, weights)

    @property
    def label(self) -> str:
        settings = ''.join(f'[{k}={v}]' for k, v in self.info)
        return f'{self.name}{settings}'

    def executable(self) -> Path:
        """The executable file: the command as given, else looked up on PATH."""
        exe = Path(self.command[0])
        if not exe.is_file() and shutil.which(self.command[0]):
            exe = Path(shutil.which(self.command[0]))
        return exe

    def provenance(self) -> dict:
        exe = self.executable()
        return {
            'name': self.name, 'label': self.label, 'command': list(self.command),
            'info': [list(kv) for kv in self.info], 'rule': RENJU_RULE,
            'executable_sha256': file_sha256(exe),
            'extra_files': {str(p): file_sha256(p) for p in self.extra_files},
            'move_timeout': self.move_timeout,
        }


class PiskvorkEngine:
    """One engine process. Use as a context manager, or call ``close``."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.errors: list[str] = []
        self.last_messages: list[str] = []
        cwd = config.cwd or (str(Path(config.command[0]).resolve().parent)
                             if Path(config.command[0]).exists() else None)
        try:
            self.process = subprocess.Popen(
                list(config.command), cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace', bufsize=1)
        except OSError as exc:
            raise EngineError(f'cannot start {config.command[0]}: {exc}') from exc
        self._lines: Queue = Queue()
        Thread(target=self._read, daemon=True).start()
        try:
            self.send(f'START {SIZE}')
            self._expect_ok(config.start_timeout, 'START')
            self.send(f'INFO rule {RENJU_RULE}')
            for key, value in config.info:
                self.send(f'INFO {key} {value}')
        except BaseException:
            self.close()
            raise

    def _read(self) -> None:
        for line in self.process.stdout:
            self._lines.put(line.rstrip('\r\n'))
        self._lines.put(None)  # EOF

    def send(self, line: str) -> None:
        try:
            self.process.stdin.write(line + '\n')
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise EngineError(f'{self.config.name}: cannot write {line!r}: {exc}') from exc

    def _reply(self, timeout: float, what: str) -> str:
        """Next line that is not MESSAGE/DEBUG/ERROR (those are kept)."""
        deadline = perf_counter() + timeout
        while True:
            left = deadline - perf_counter()
            try:
                line = self._lines.get(timeout=max(left, 0.0)) if left > 0 else self._lines.get_nowait()
            except Empty:
                raise EngineTimeout(f'{self.config.name}: no reply to {what} within {timeout:.0f}s') from None
            if line is None:
                raise EngineError(f'{self.config.name}: process ended while waiting for {what} '
                                  f'(exit code {self.process.poll()})')
            word = line.split(' ', 1)[0].upper()
            if word in ('MESSAGE', 'DEBUG'):
                self.last_messages.append(line)
            elif word == 'ERROR':
                self.errors.append(line)
            elif line.strip():
                return line.strip()

    def _expect_ok(self, timeout: float, what: str) -> None:
        reply = self._reply(timeout, what)
        if reply.upper() != 'OK':
            raise EngineError(f'{self.config.name}: {what} answered {reply!r}, expected OK')

    def about(self) -> str:
        self.send('ABOUT')
        return self._reply(10.0, 'ABOUT')

    def restart(self) -> None:
        self.send('RESTART')
        self._expect_ok(self.config.start_timeout, 'RESTART')

    def best_move(self, history, to_play: int) -> tuple[int, int]:
        self.last_messages = []
        self.send('BOARD')
        for line in board_lines(history, to_play):
            self.send(line)
        self.send('DONE')
        return from_piskvork(self._reply(self.config.move_timeout, 'BOARD'))

    def forbidden_points(self, history) -> set[tuple[int, int]]:
        """Black forbidden points by the engine's rules (Yixin-Board ``YXSHOWFORBID``; black to move)."""
        self.send('YXBOARD')  # same body as BOARD, but the engine does not think
        for line in board_lines(history, BLACK):
            self.send(line)
        self.send('DONE')
        self.send('YXSHOWFORBID')
        reply = self._reply(30.0, 'YXSHOWFORBID')
        if not reply.upper().startswith('FORBID'):
            raise EngineError(f'{self.config.name}: YXSHOWFORBID answered {reply!r}')
        digits = reply[len('FORBID'):].strip().rstrip('.').strip()
        if len(digits) % 4 or (digits and not digits.isdigit()):
            raise EngineError(f'{self.config.name}: cannot parse {reply!r}')
        return {(int(digits[i + 2:i + 4]), int(digits[i:i + 2])) for i in range(0, len(digits), 4)}

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.send('END')
                self.process.wait(timeout=5)
            except (EngineError, subprocess.TimeoutExpired):
                pass
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                pipe.close()
            except OSError:
                pass

    def __enter__(self) -> 'PiskvorkEngine':
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def check_engine_move(game: Game, move: tuple[int, int]) -> None:
    """Raise ``EngineIllegalMove`` unless ``game.play(*move)`` would accept ``move``."""
    if not inside(*move):
        raise EngineIllegalMove(move, 'outside')
    if game.board[move[0]][move[1]] != 0:
        raise EngineIllegalMove(move, 'occupied')
    if game.to_play == BLACK:
        reason = forbidden_reason(game.board, *move)
        if reason:
            raise EngineIllegalMove(move, reason)
    if not game.history and move not in game.legal_moves():
        raise EngineIllegalMove(move, 'opening move must be the centre')


class ExternalEngineAgent:
    """``Agent`` around one engine process. ``select_move`` sends the whole position every time (stateless)."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.name = config.label
        self.engine = PiskvorkEngine(config)
        self.last_seconds = 0.0

    def select_move(self, game: Game) -> tuple[int, int]:
        if game.done:
            raise IllegalMove('game is over')
        started = perf_counter()
        move = self.engine.best_move(game.history, game.to_play)
        self.last_seconds = perf_counter() - started
        check_engine_move(game, move)
        return move

    def forbidden_points(self, game: Game) -> set[tuple[int, int]]:
        return self.engine.forbidden_points(game.history)

    def close(self) -> None:
        self.engine.close()

    def __enter__(self) -> 'ExternalEngineAgent':
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def our_forbidden_points(game: Game) -> set[tuple[int, int]]:
    """Black forbidden points by our rules (empty points that ``forbidden_reason`` rejects)."""
    return {(r, c) for r in range(SIZE) for c in range(SIZE)
            if game.board[r][c] == 0 and forbidden_reason(game.board, r, c)}
