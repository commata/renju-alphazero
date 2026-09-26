"""Stage 5 self-play: Sample/GameRecord generation, replay validation, canonical hashes.

Depends only on the torch-free ``Evaluator`` interface. Canonical data is integer
visit counts; the float ``pi`` is always derived and never hashed.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import hashlib
import json
from random import Random
from time import perf_counter

from model.config import (ACTION_COUNT, ACTION_INDEX_VERSION, ENCODER_VERSION,
                          action_to_coordinate, coordinate_to_action)
from renju import Game
from search.alphazero import SearchConfig, run_search, select_action
from search.evaluator import Evaluator

from .provenance import base_runtime_env, git_provenance

RECORD_FORMAT = 'stage5-record-v1'
GAME_HASH_FORMAT = 'stage5-game-v1'
RECORD_HASH_FORMAT = 'stage5-record-hash-v1'
# Ply 0 is always the forced centre move, so diversity is measured from ply 1.
DIVERSITY_START_PLY = 1


def canonical_json_bytes(payload) -> bytes:
    """The single Stage 5 canonical serialization (used by tests and the smoke runner)."""
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def canonical_sha256(payload) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def config_hash(config: SearchConfig | dict) -> str:
    data = config.to_dict() if isinstance(config, SearchConfig) else config
    return canonical_sha256(data)


def compute_z(winner: int | None, to_play: int) -> float:
    """Outcome from the sample player's perspective (never from ply parity)."""
    if winner is None:
        return 0.0
    return 1.0 if winner == to_play else -1.0


@dataclass(frozen=True)
class Sample:
    ply: int                        # moves played before this state
    to_play: int
    visit_counts: tuple[int, ...]   # len 225, canonical policy target source
    action: int
    z: float

    def to_dict(self) -> dict:
        return {'ply': self.ply, 'to_play': self.to_play,
                'visit_counts': list(self.visit_counts), 'action': self.action, 'z': self.z}

    @classmethod
    def from_dict(cls, data: dict) -> Sample:
        return cls(data['ply'], data['to_play'], tuple(data['visit_counts']), data['action'],
                   float(data['z']))


@dataclass(frozen=True)
class GameRecord:
    format_version: str
    seed: int
    model_seed: int | None
    search_config: dict
    config_hash: str
    checkpoint_hash: str | None
    git_commit: str | None
    git_dirty: bool | None
    encoder_version: str
    action_index_version: str
    runtime_env: dict               # excluded from every hash
    winner: int | None
    moves: tuple[int, ...]
    samples: tuple[Sample, ...]

    def to_dict(self) -> dict:
        return {
            'format_version': self.format_version, 'seed': self.seed,
            'model_seed': self.model_seed, 'search_config': dict(self.search_config),
            'config_hash': self.config_hash, 'checkpoint_hash': self.checkpoint_hash,
            'git_commit': self.git_commit, 'git_dirty': self.git_dirty,
            'encoder_version': self.encoder_version,
            'action_index_version': self.action_index_version,
            'runtime_env': dict(self.runtime_env), 'winner': self.winner,
            'moves': list(self.moves), 'samples': [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict) -> GameRecord:
        values = dict(data)
        values['moves'] = tuple(values['moves'])
        values['samples'] = tuple(Sample.from_dict(s) for s in values['samples'])
        return cls(**values)


def game_hash(record: GameRecord) -> str:
    return canonical_sha256({'format': GAME_HASH_FORMAT, 'winner': record.winner,
                             'moves': list(record.moves)})


def record_hash(record: GameRecord) -> str:
    """winner + moves + integer sample targets; excludes timing, pi and runtime_env."""
    return canonical_sha256({
        'format': RECORD_HASH_FORMAT, 'winner': record.winner, 'moves': list(record.moves),
        'samples': [{'ply': s.ply, 'to_play': s.to_play, 'action': s.action,
                     'visit_counts': list(s.visit_counts)} for s in record.samples],
    })


@dataclass(frozen=True)
class MoveStats:
    ply: int
    fast_path: bool
    evaluator_calls: int
    legal_moves_ms: float
    inference_ms: float
    tree_ms: float
    total_move_ms: float


@dataclass
class SelfPlayGame:
    record: GameRecord
    final_game: Game
    move_stats: list[MoveStats] = field(default_factory=list)


def play_self_play_game(evaluator: Evaluator, config: SearchConfig, seed: int, *,
                        model_seed: int | None = None, checkpoint_hash: str | None = None,
                        runtime_env: dict | None = None) -> SelfPlayGame:
    """Play one game with a single game-owned ``Random(seed)``.

    RNG order per searched move: Dirichlet gammas (legal actions ascending), then one
    ``random()`` if ``ply < temperature_moves``. The single-legal fast path uses none.
    """
    rng = Random(seed)
    game = Game()
    pending: list[tuple[int, int, tuple[int, ...], int]] = []
    stats: list[MoveStats] = []
    while not game.done:
        started = perf_counter()
        ply = len(game.history)
        result = run_search(game, evaluator, config, rng)
        action = select_action(result, ply, config, rng)
        game.play(*action_to_coordinate(action))
        total = perf_counter() - started
        timing = result.timing
        pending.append((ply, result.to_play, result.visit_counts, action))
        stats.append(MoveStats(
            ply, result.fast_path, result.evaluator_calls, timing.legal_moves_s * 1e3,
            timing.inference_s * 1e3,
            (total - timing.legal_moves_s - timing.inference_s) * 1e3, total * 1e3))
    samples = tuple(Sample(ply, to_play, counts, action, compute_z(game.winner, to_play))
                    for ply, to_play, counts, action in pending)
    commit, dirty = git_provenance()
    search_config = config.to_dict()
    record = GameRecord(
        format_version=RECORD_FORMAT, seed=seed, model_seed=model_seed,
        search_config=search_config, config_hash=config_hash(search_config),
        checkpoint_hash=checkpoint_hash, git_commit=commit, git_dirty=dirty,
        encoder_version=ENCODER_VERSION, action_index_version=ACTION_INDEX_VERSION,
        runtime_env=dict(runtime_env) if runtime_env is not None else base_runtime_env(),
        winner=game.winner, moves=tuple(s.action for s in samples), samples=samples,
    )
    return SelfPlayGame(record, game, stats)


class ReplayError(ValueError):
    pass


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def replay_record(record: GameRecord, final_game: Game | None = None) -> Game:
    """Rebuild every sample state from ``moves[:ply]`` and validate the record contract."""
    _check(record.format_version == RECORD_FORMAT, 'unsupported record format')
    _check(record.encoder_version == ENCODER_VERSION, 'encoder version mismatch')
    _check(record.action_index_version == ACTION_INDEX_VERSION, 'action index version mismatch')
    _check(record.config_hash == config_hash(record.search_config), 'config hash mismatch')
    num_simulations = SearchConfig.from_dict(record.search_config).num_simulations
    _check(len(record.samples) == len(record.moves), 'one sample per move is required')

    game = Game()
    for index, (move, sample) in enumerate(zip(record.moves, record.samples)):
        _check(sample.ply == index == len(game.history), f'sample {index}: ply mismatch')
        _check(not game.done, f'sample {index}: game already finished')
        _check(sample.to_play == game.to_play, f'sample {index}: to_play mismatch')
        _check(sample.action == move, f'sample {index}: action differs from moves')
        counts = sample.visit_counts
        _check(len(counts) == ACTION_COUNT and all(type(n) is int and n >= 0 for n in counts),
               f'sample {index}: visit counts must be 225 non-negative integers')
        legal = {coordinate_to_action(r, c) for r, c in game.legal_moves()}
        _check(sample.action in legal, f'sample {index}: illegal action')
        _check(counts[sample.action] > 0,
               f'sample {index}: played action must have a positive visit count')
        _check(all(n == 0 for a, n in enumerate(counts) if a not in legal),
               f'sample {index}: illegal visit count')
        expected_total = 1 if len(legal) == 1 else num_simulations
        _check(sum(counts) == expected_total, f'sample {index}: visit sum mismatch')
        game.play(*action_to_coordinate(move))
    _check(game.done, 'record ends before the game is finished')
    _check(game.winner == record.winner, 'winner mismatch')
    for sample in record.samples:
        _check(sample.z == compute_z(record.winner, sample.to_play), f'ply {sample.ply}: z mismatch')
    if final_game is not None:
        _check(game.board == final_game.board, 'final board mismatch')
        _check(game.history == final_game.history, 'history mismatch')
        _check((game.winner, game.done) == (final_game.winner, final_game.done),
               'final result mismatch')
    return game


def summarize_timing(stats: Iterable[MoveStats]) -> dict:
    stats = list(stats)
    searched = [s for s in stats if not s.fast_path]
    fast = [s for s in stats if s.fast_path]

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None

    return {
        'searched': {
            'count': len(searched),
            'legal_moves_ms': mean(s.legal_moves_ms for s in searched),
            'inference_ms': mean(s.inference_ms for s in searched),
            'tree_ms': mean(s.tree_ms for s in searched),
            'total_move_ms': mean(s.total_move_ms for s in searched),
            'evaluator_calls': mean(s.evaluator_calls for s in searched),
        },
        'fast_path': {
            'count': len(fast),
            'total_move_ms': mean(s.total_move_ms for s in fast),
            'total_ms': sum(s.total_move_ms for s in fast),
        },
    }


def opening_diversity(records: Iterable[GameRecord], end_ply: int) -> dict:
    """Distinct opening prefixes ``moves[DIVERSITY_START_PLY:end_ply]`` (ply 0 excluded)."""
    prefixes = [tuple(r.moves[DIVERSITY_START_PLY:end_ply]) for r in records]
    return {'start_ply': DIVERSITY_START_PLY, 'end_ply': end_ply, 'games': len(prefixes),
            'distinct_prefixes': len(set(prefixes))}
