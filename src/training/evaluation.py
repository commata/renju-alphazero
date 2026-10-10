"""Per-generation evaluation of the training model (monitoring only; no gating).

Protocol:
- the model plays Stage 5 PUCT with noise OFF and temperature 0 (argmax visit count,
  ties by prior then smaller action index — the Stage 5 ``argmax_action`` rule);
- every game starts from a short random opening: the forced centre move plus
  ``opening_random_plies`` uniformly random legal moves within ``opening_radius``
  (Chebyshev) of the centre; each opening is played twice with colours swapped;
- all randomness is stateless: every opening/agent seed is derived from
  ``(seed, generation, opponent, index)``, so evaluation never touches the training
  RNGs and gives the same games whether or not the run was resumed.
"""
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from random import Random
from time import perf_counter

from model.config import action_to_coordinate
from model.batched_evaluator import BatchedEvaluator
from model.evaluator import PolicyValueEvaluator
from renju import BLACK, WHITE, Game, IllegalMove
from renju.game import OPENING_MOVE
from search.alphazero import run_search, search_steps, select_action
from search.evaluator import drive_requests, tag_requests
from search.scheduler import SchedulerStats, run_tasks

from .config import evaluation_search_config, parallel_setting, role_device
from .trainer import inference_mode_for, model_for_device
from .training_state import derive_seed

EVALUATION_FORMAT = 'stage6-evaluation-v1'


class EvaluationIllegalMoveError(RuntimeError):
    pass


class PUCTAgent:
    """Deterministic evaluation agent (noise OFF, temperature 0).

    ``evaluator`` replaces the default batch-1 ``PolicyValueEvaluator`` (e.g. a
    ``BatchedEvaluator`` shared by many games under ``search.scheduler``).
    """

    def __init__(self, name: str, model, search_config, device='cpu', *, evaluator=None):
        if search_config.noise_enabled or search_config.temperature_moves != 0:
            raise ValueError('evaluation PUCT requires noise OFF and temperature 0')
        self.name = name
        self.config = search_config
        self.evaluator = evaluator if evaluator is not None else PolicyValueEvaluator(model, device=device)

    def select_move(self, game: Game) -> tuple[int, int]:
        result = run_search(game, self.evaluator, self.config, None)
        return action_to_coordinate(select_action(result, len(game.history), self.config, None))

    def select_move_steps(self, game: Game):
        """``select_move`` as a generator of ``EvalRequest``s (for ``search.scheduler``)."""
        result, _ = yield from tag_requests(search_steps(game, self.config, None), self.evaluator)
        return action_to_coordinate(select_action(result, len(game.history), self.config, None))


def make_opening(rng: Random, random_plies: int, radius: int) -> tuple[tuple[int, int], ...]:
    game = Game()
    game.play(*OPENING_MOVE)
    centre_row, centre_col = OPENING_MOVE
    for _ in range(random_plies):
        legal = game.legal_moves()
        near = [(r, c) for r, c in legal
                if max(abs(r - centre_row), abs(c - centre_col)) <= radius]
        game.play(*rng.choice(near or legal))
        if game.done:
            raise RuntimeError('evaluation opening ended the game')
    return tuple(game.history)


def play_evaluation_game(model_agent, opponent, model_color: int,
                         opening: tuple[tuple[int, int], ...]) -> dict:
    return drive_requests(evaluation_game_steps(model_agent, opponent, model_color, opening))


def evaluation_game_steps(model_agent, opponent, model_color: int,
                          opening: tuple[tuple[int, int], ...]):
    """One evaluation game as a generator of ``EvalRequest``s.

    Agents with ``select_move_steps`` (PUCT) yield their leaf requests; any other agent
    moves synchronously. Move seconds include any wait for a shared batch.
    """
    game = Game()
    for move in opening:
        game.play(*move)  # the engine validates every opening move
    times = {'model': [], 'opponent': []}
    while not game.done:
        is_model = game.to_play == model_color
        agent = model_agent if is_model else opponent
        started = perf_counter()
        if hasattr(agent, 'select_move_steps'):
            move = yield from agent.select_move_steps(game)
        else:
            move = agent.select_move(game)
        times['model' if is_model else 'opponent'].append(perf_counter() - started)
        try:
            game.play(*move)
        except IllegalMove as exc:
            raise EvaluationIllegalMoveError(f'{agent.name} played illegal {move}: {exc}') from exc
    if game.winner is None:
        result = 'draw'
    else:
        result = 'win' if game.winner == model_color else 'loss'
    return {'model_color': 'black' if model_color == BLACK else 'white',
            'opponent': opponent.name, 'winner': game.winner, 'result': result,
            'moves': [list(m) for m in game.history], 'length': len(game.history),
            'opening_plies': len(opening),
            'model_move_seconds': times['model'], 'opponent_move_seconds': times['opponent']}


def _mean(values):
    return sum(values) / len(values) if values else None


def summarize_games(games: list[dict]) -> dict:
    def wld(subset):
        return {'wins': sum(g['result'] == 'win' for g in subset),
                'losses': sum(g['result'] == 'loss' for g in subset),
                'draws': sum(g['result'] == 'draw' for g in subset),
                'games': len(subset)}

    sequences = [tuple(map(tuple, g['moves'])) for g in games]
    model_times = [t for g in games for t in g['model_move_seconds']]
    opponent_times = [t for g in games for t in g['opponent_move_seconds']]
    return {
        'games': len(games), 'unique_games': len(set(sequences)),
        'duplicate_games': len(sequences) - len(set(sequences)),
        **{k: v for k, v in wld(games).items() if k != 'games'},
        'black': wld([g for g in games if g['model_color'] == 'black']),
        'white': wld([g for g in games if g['model_color'] == 'white']),
        'illegal_moves': 0,  # any illegal move aborts evaluation with an exception
        'average_model_move_seconds': _mean(model_times),
        'average_opponent_move_seconds': _mean(opponent_times),
        'average_game_length': _mean([g['length'] for g in games]),
        'opening_plies': sorted({g['opening_plies'] for g in games}),
    }


def _opponent_factories(config: dict, previous_model, *, batched: bool = False
                        ) -> dict[str, tuple[Callable, dict]]:
    from agents import MCTSV6Agent, RandomAgent, TacticalAgent
    from search.mcts_v6 import V5_FINAL

    search = evaluation_search_config(config)
    device = role_device(config, 'evaluation')
    # One evaluator shared by every 'previous' game so their requests batch together.
    previous_evaluator = BatchedEvaluator(previous_model, device=device) if batched else None
    return {
        'random': (lambda seed: RandomAgent(seed), {'agent': 'Random'}),
        'tactical': (lambda seed: TacticalAgent(seed), {'agent': 'Tactical'}),
        'previous': (lambda seed: PUCTAgent('previous', previous_model, search, device,
                                            evaluator=previous_evaluator),
                     {'agent': 'PUCT', 'search': search.to_dict()}),
        # Frozen Stage 3 preset: MCTSV6Agent is constructed without any override.
        'mcts_v6': (lambda seed: MCTSV6Agent(seed=seed),
                    {'agent': 'MCTS-v6', 'frozen_config': {
                        k: (float(v) if isinstance(v, float) else v)
                        for k, v in V5_FINAL.items()}}),
    }


def opponents_for_generation(config: dict, generation: int, final_generation: int) -> list[str]:
    e = config['evaluation']
    names = []
    for name in ('random', 'tactical', 'previous', 'mcts_v6'):
        if e[name]['black_games'] == 0:
            continue
        if name == 'mcts_v6' and e[name]['final_generation_only'] \
                and generation != final_generation:
            continue
        names.append(name)
    return names


def should_evaluate(config: dict, generation: int, final_generation: int) -> bool:
    return generation % config['evaluation']['every'] == 0 or generation == final_generation


def evaluate_generation(model, previous_model, generation: int, config: dict,
                        final_generation: int, *, log: Callable[[str], None] | None = None
                        ) -> dict:
    """Evaluate ``model`` against every configured opponent; restores model modes.

    Runs on ``devices.evaluation`` (a copy when the training model lives elsewhere).
    ``parallel.evaluation: batched`` plays each opponent's games together with shared
    ``BatchedEvaluator``s (``search.scheduler``); the games themselves are unchanged.
    """
    e = config['evaluation']
    search = evaluation_search_config(config)
    seed = config['seed']
    device = role_device(config, 'evaluation')
    batched = parallel_setting(config, 'evaluation') == 'batched'
    results = {'format_version': EVALUATION_FORMAT, 'generation': generation,
               'model_search': search.to_dict(), 'opponents': {}}
    previous_model = previous_model if previous_model is not None else deepcopy(model)
    model = model_for_device(model, device)
    previous_model = model_for_device(previous_model, device)
    with inference_mode_for(model), inference_mode_for(previous_model):
        factories = _opponent_factories(config, previous_model, batched=batched)
        model_agent = PUCTAgent('model', model, search, device,
                                evaluator=BatchedEvaluator(model, device=device) if batched else None)
        for name in opponents_for_generation(config, generation, final_generation):
            factory, opponent_config = factories[name]
            started = perf_counter()
            tasks = []
            for pair in range(e[name]['black_games']):
                opening_rng = Random(derive_seed(seed, generation, name, 'opening', pair))
                opening = make_opening(opening_rng, e['opening_random_plies'],
                                       e['opening_radius'])
                for offset, color in enumerate((BLACK, WHITE)):
                    index = 2 * pair + offset
                    opponent = factory(derive_seed(seed, generation, name, 'agent', index))
                    tasks.append(lambda o=opponent, c=color, op=opening:
                                 evaluation_game_steps(model_agent, o, c, op))
            stats = SchedulerStats() if batched else None
            if batched:
                games = run_tasks(tasks, max_batch=parallel_setting(config, 'max_batch'),
                                  eager=parallel_setting(config, 'eager'), stats=stats)
            else:
                games = [drive_requests(task()) for task in tasks]
            for index, game in enumerate(games):
                game['index'] = index
            summary = summarize_games(games)
            summary['seconds'] = perf_counter() - started
            if stats is not None:
                summary['batching'] = stats.summary()
            results['opponents'][name] = {'summary': summary, 'opponent_config': opponent_config,
                                          'games': games}
            if log:
                log(f"  eval gen {generation} vs {name}: W{summary['wins']} "
                    f"L{summary['losses']} D{summary['draws']} "
                    f"(unique {summary['unique_games']}/{summary['games']})")
    return results
