"""Stage 6 training loop: self-play -> buffer -> training -> evaluation -> checkpoint.

Checkpoints are written only at generation boundaries, after evaluation, with
``generation`` = the next generation to run. A crash anywhere inside generation G
therefore resumes from checkpoint G and re-runs G from the start (no mid-generation
recovery). The model is always the latest one (no gating/rollback).
"""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import torch

from model.config import ACTION_INDEX_VERSION, CHECKPOINT_FORMAT_VERSION, ENCODER_VERSION
from model.evaluator import PolicyValueEvaluator
from renju import BLACK, WHITE

from .config import dump_config, self_play_search_config
from .dataset import build_batch, samples_from_record, validate_samples
from .evaluation import evaluate_generation, should_evaluate
from .metrics import (MetricsLogger, RunMetadata, timestamp, truncate_for_resume, utc_now,
                      write_json)
from .provenance import base_runtime_env, git_provenance
from .self_play import game_hash, play_self_play_game, record_hash, replay_record, summarize_timing
from .trainer import inference_mode_for, train_step
from .training_checkpoint import (INIT_NAME, LATEST_NAME, build_checkpoint, copy_atomic,
                                  generation_checkpoint_name, load_training_state,
                                  prune_checkpoints, save_atomic)
from .training_state import TrainingState, derive_seed, init_training_state

try:
    import numpy
except ModuleNotFoundError:  # pragma: no cover
    numpy = None


def _git_branch() -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                       cwd=Path(__file__).resolve().parent, text=True,
                                       stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def runtime_env(config: dict) -> dict:
    return {**base_runtime_env(), 'torch': str(torch.__version__),
            'numpy': getattr(numpy, '__version__', None), 'device': config['device'],
            'torch_threads': torch.get_num_threads()}


def _metadata_base(config: dict) -> dict:
    commit, dirty = git_provenance()
    return {'created': utc_now(), 'git_branch': _git_branch(), 'git_commit': commit,
            'git_dirty': dirty, 'seed': config['seed'], 'device': config['device'],
            'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': getattr(numpy, '__version__', None),
            'model_version': {'encoder': ENCODER_VERSION, 'action_index': ACTION_INDEX_VERSION,
                              'stage4_checkpoint_format': CHECKPOINT_FORMAT_VERSION,
                              'architecture': dict(config['model'])}}


def save_generation_checkpoint(state: TrainingState, checkpoint_dir: Path) -> Path:
    """checkpoint_gen{N}.pt (N = state.generation) then latest.pt, both atomic."""
    path = checkpoint_dir / generation_checkpoint_name(state.generation)
    state.source_checkpoint_hash = save_atomic(path, build_checkpoint(state))
    copy_atomic(path, checkpoint_dir / LATEST_NAME)
    prune_checkpoints(checkpoint_dir, state.config['training']['keep_checkpoints'])
    return path


def generate_self_play(state: TrainingState) -> tuple[list, list, float]:
    """Self-play with eval mode + no_grad; one game seed per game from self_play_rng."""
    config = state.config
    search = self_play_search_config(config)
    env = runtime_env(config)
    games = []
    started = perf_counter()
    with inference_mode_for(state.model):
        evaluator = PolicyValueEvaluator(state.model, device=config['device'])
        for _ in range(config['training']['games_per_generation']):
            seed = state.self_play_rng.getrandbits(63)
            games.append(play_self_play_game(evaluator, search, seed,
                                             checkpoint_hash=state.source_checkpoint_hash,
                                             runtime_env=env))
    elapsed = perf_counter() - started
    records = [g.record for g in games]
    for game in games:
        replay_record(game.record, game.final_game)  # IllegalMove/contract check
    return records, [g.move_stats for g in games], elapsed


def run_generation(state: TrainingState, run_dir: Path, metrics: MetricsLogger,
                   log: Callable[[str], None]) -> dict:
    config = state.config
    t = config['training']
    gen = state.generation
    final_generation = t['generations'] - 1
    previous_model = deepcopy(state.model)  # = the checkpoint this generation starts from

    records, move_stats, self_play_seconds = generate_self_play(state)
    samples = [s for game_id, record in enumerate(records)
               for s in samples_from_record(record, generation=gen, game_id=game_id)]
    validate_samples(samples)
    state.buffer.extend(samples)
    write_json(run_dir / 'self_play' / f'gen{gen:03d}.json', {
        'generation': gen, 'games': [{'game_sha256': game_hash(r), 'record_sha256': record_hash(r),
                                      'record': r.to_dict()} for r in records]})
    log(f'gen {gen}: self-play {len(records)} games, {len(samples)} samples, '
        f'{self_play_seconds:.1f}s')

    started = perf_counter()
    first = last = None
    for _ in range(t['steps_per_generation']):
        batch = build_batch(state.buffer, t['batch_size'], sample_rng=state.sample_rng,
                            augment_rng=state.augment_rng,
                            augment=config['augmentation']['enabled'])
        step = train_step(state.model, state.optimizer, batch, config['loss']['value_weight'],
                          t['grad_clip'], config['loss']['l2_coeff'])
        state.global_step += 1
        metrics.log({'type': 'train', 'generation': gen, 'global_step': state.global_step,
                     **step, 'buffer_size': len(state.buffer)})
        first = first or step
        last = step
    training_seconds = perf_counter() - started
    drawn = t['steps_per_generation'] * t['batch_size']
    lengths = [len(r.moves) for r in records]
    generation_event = {
        'type': 'generation', 'generation': gen, 'self_play_games': len(records),
        'new_samples': len(samples),
        'black_wins': sum(r.winner == BLACK for r in records),
        'white_wins': sum(r.winner == WHITE for r in records),
        'draws': sum(r.winner is None for r in records),
        'average_game_length': sum(lengths) / len(lengths),
        'game_lengths': lengths,
        'self_play_seconds': self_play_seconds, 'training_seconds': training_seconds,
        'buffer_size': len(state.buffer), 'samples_drawn': drawn,
        'sample_reuse_ratio': drawn / len(samples) if samples else None,
        'illegal_moves': 0,  # every record passed replay_record (engine legality)
        'global_step': state.global_step,
        'first_step': first, 'last_step': last,
        'self_play_timing': summarize_timing(s for stats in move_stats for s in stats),
    }
    metrics.log(generation_event)
    log(f'gen {gen}: trained {t["steps_per_generation"]} steps, total loss '
        f'{first["total_loss"]:.4f} -> {last["total_loss"]:.4f}')

    if should_evaluate(config, gen, final_generation):
        results = evaluate_generation(state.model, previous_model, gen, config, final_generation,
                                      log=log)
        write_json(run_dir / 'evaluation' / f'gen{gen:03d}.json', results)
        for name, data in results['opponents'].items():
            metrics.log({'type': 'evaluation', 'generation': gen, 'opponent': name,
                         **data['summary'], 'opponent_config': data['opponent_config'],
                         'model_search': results['model_search']})

    state.generation += 1
    path = save_generation_checkpoint(state, run_dir / 'checkpoints')
    metrics.log({'type': 'checkpoint', 'generation': gen, 'next_generation': state.generation,
                 'path': str(path.relative_to(run_dir)).replace('\\', '/'),
                 'sha256': state.source_checkpoint_hash})
    log(f'gen {gen}: checkpoint {path.name} (generation={state.generation})')
    return generation_event


def new_run_dir(config: dict) -> Path:
    out = config['output']
    return Path(out['runs_dir']) / f"{timestamp()}_{out['run_name']}"


def run_training(config: dict | None, *, run_dir: str | Path | None = None,
                 resume: str | Path | None = None, stop_after: int | None = None,
                 log: Callable[[str], None] = print) -> TrainingState:
    """Run (or resume) until ``training.generations`` or ``stop_after`` generations.

    New runs need ``config``. Resumes continue in the checkpoint's own run directory;
    ``config`` may then only change execution-control values.
    """
    if resume is not None:
        resume = Path(resume).resolve()
        state = load_training_state(resume, config)
        run_dir = resume.parent.parent
        if not (run_dir / 'metadata.json').exists():
            raise FileNotFoundError(f'{run_dir} is not a Stage 6 run directory')
        truncation = truncate_for_resume(run_dir, state.generation, timestamp())
        log(f'resume from {resume.name}: generation={state.generation}, '
            f'global_step={state.global_step}, dropped {truncation["dropped_events"]} events')
    else:
        if config is None:
            raise ValueError('a new run needs a config')
        run_dir = Path(run_dir) if run_dir is not None else new_run_dir(config)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f'run directory is not empty: {run_dir}')
        run_dir.mkdir(parents=True, exist_ok=True)
        truncation = None
        state = init_training_state(config)
    config = state.config
    torch.set_num_threads(config['torch_threads'])
    checkpoint_dir = run_dir / 'checkpoints'
    (run_dir / 'config.yaml').write_text(dump_config(config), encoding='utf-8')
    metadata = RunMetadata(run_dir / 'metadata.json', _metadata_base(config))
    if resume is None:
        state.source_checkpoint_hash = save_atomic(checkpoint_dir / INIT_NAME,
                                                   build_checkpoint(state))
        copy_atomic(checkpoint_dir / INIT_NAME, checkpoint_dir / LATEST_NAME)
    metrics = MetricsLogger(run_dir / 'metrics.jsonl')
    commit, dirty = git_provenance()
    metadata.start_segment(start_generation=state.generation, resumed=resume is not None,
                           extra={'resume_from': str(resume) if resume else None,
                                  'truncation': truncation, 'git_commit': commit,
                                  'git_dirty': dirty, 'target_generations':
                                  config['training']['generations'], 'stop_after': stop_after})
    started = perf_counter()
    status = 'completed'
    try:
        while state.generation < config['training']['generations']:
            run_generation(state, run_dir, metrics, log)
            if stop_after is not None and state.generation >= stop_after:
                status = 'stopped'
                break
    except BaseException as exc:
        metadata.finish_segment(end_generation=state.generation, status='failed',
                                duration=perf_counter() - started,
                                error=f'{type(exc).__name__}: {exc}')
        raise
    metadata.finish_segment(end_generation=state.generation, status=status,
                            duration=perf_counter() - started)
    state.run_dir = run_dir
    return state


def verify_checkpoint(path: str | Path) -> dict:
    """Reload a training checkpoint and play one validated self-play game with it.

    Uses a seed derived from (seed, 'verify', generation), not the saved RNGs.
    """
    state = load_training_state(path)
    search = self_play_search_config(state.config)
    with inference_mode_for(state.model):
        evaluator = PolicyValueEvaluator(state.model, device=state.config['device'])
        game = play_self_play_game(evaluator, search,
                                   derive_seed(state.config['seed'], 'verify', state.generation),
                                   checkpoint_hash=state.source_checkpoint_hash,
                                   runtime_env=runtime_env(state.config))
    replay_record(game.record, game.final_game)
    samples = samples_from_record(game.record, generation=state.generation, game_id=0)
    validate_samples(samples)
    return {'checkpoint': str(path), 'checkpoint_sha256': state.source_checkpoint_hash,
            'generation': state.generation, 'global_step': state.global_step,
            'buffer_size': len(state.buffer), 'winner': game.record.winner,
            'moves': len(game.record.moves), 'samples': len(samples),
            'game_sha256': game_hash(game.record), 'illegal_moves': 0}
