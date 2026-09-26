"""Stage 6 training config: YAML loading, defaults, validation and the critical hash.

Torch-free. Defaults for the self-play search come from the Stage 5 ``SearchConfig``
so the resolved config records the values actually used. Unknown keys are errors.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import Path

from model.config import ModelConfig
from search.alphazero import SearchConfig

from .self_play import canonical_sha256

CONFIG_FORMAT = 'stage6-config-v1'
# Stage 5 self-play has no truncation: a game ends naturally within 225 plies.
BOARD_PLY_LIMIT = 225

_STAGE5 = SearchConfig()
_MODEL = ModelConfig()

DEFAULTS: dict = {
    'format_version': CONFIG_FORMAT,
    'seed': 42,
    'device': 'cpu',
    'torch_threads': 1,
    'model': {
        'channels': _MODEL.channels,
        'blocks': _MODEL.blocks,
        'policy_channels': _MODEL.policy_channels,
        'value_channels': _MODEL.value_channels,
        'value_hidden': _MODEL.value_hidden,
    },
    'training': {
        'generations': 3,
        'games_per_generation': 4,
        'batch_size': 32,
        'steps_per_generation': 50,
        'replay_capacity': 10000,
        'init_checkpoint': None,
        'keep_checkpoints': 3,
        'grad_clip': None,
    },
    'self_play': {
        'simulations': 25,
        'c_puct': _STAGE5.c_puct,
        'tau': _STAGE5.tau,
        'temperature_moves': _STAGE5.temperature_moves,
        'dirichlet_alpha': _STAGE5.dirichlet_alpha,
        'dirichlet_epsilon': _STAGE5.dirichlet_epsilon,
        'max_moves': BOARD_PLY_LIMIT,
    },
    'augmentation': {'enabled': True},
    'optimizer': {'name': 'adam', 'lr': 0.001, 'weight_decay': 0.0001, 'momentum': 0.0},
    'loss': {'value_weight': 1.0, 'l2_coeff': 0.0},
    'evaluation': {
        'every': 1,
        'puct_simulations': 25,
        'c_puct': _STAGE5.c_puct,
        'opening_random_plies': 2,
        'opening_radius': 2,
        'random': {'black_games': 2, 'white_games': 2},
        'tactical': {'black_games': 2, 'white_games': 2},
        'previous': {'black_games': 2, 'white_games': 2},
        'mcts_v6': {'final_generation_only': True, 'black_games': 1, 'white_games': 1,
                    'use_frozen_config': True},
    },
    'output': {'runs_dir': 'runs', 'run_name': 'stage6'},
}

# Execution-control keys: may differ between a checkpoint and --config / CLI on resume.
# Everything else is training-critical and must match the checkpoint exactly.
NON_CRITICAL = (
    ('device',), ('torch_threads',), ('output',),
    ('training', 'generations'), ('training', 'keep_checkpoints'),
    ('training', 'init_checkpoint'),
)

OPPONENTS = ('random', 'tactical', 'previous', 'mcts_v6')


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict, path: str = '') -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        where = f'{path}{key}'
        if key not in base:
            raise ConfigError(f'unknown config key: {where}')
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f'{where} must be a mapping')
            merged[key] = _merge(base[key], value, where + '.')
        else:
            merged[key] = value
    return merged


def _int(value, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ConfigError(f'{name} must be an integer >= {minimum}')


def _real(value, name: str, *, minimum: float | None = None, positive: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not isfinite(value)):
        raise ConfigError(f'{name} must be a finite number')
    if positive and value <= 0:
        raise ConfigError(f'{name} must be positive')
    if minimum is not None and value < minimum:
        raise ConfigError(f'{name} must be >= {minimum}')


def validate_config(config: dict) -> dict:
    if config.get('format_version') != CONFIG_FORMAT:
        raise ConfigError(f'format_version must be {CONFIG_FORMAT}')
    _int(config['seed'], 'seed', 0)
    if not isinstance(config['device'], str):
        raise ConfigError('device must be a string')
    _int(config['torch_threads'], 'torch_threads')
    try:
        model_config(config)
    except ValueError as exc:
        raise ConfigError(f'model: {exc}') from exc

    t = config['training']
    for key in ('generations', 'games_per_generation', 'batch_size', 'steps_per_generation',
                'replay_capacity', 'keep_checkpoints'):
        _int(t[key], f'training.{key}')
    if t['init_checkpoint'] is not None and not isinstance(t['init_checkpoint'], str):
        raise ConfigError('training.init_checkpoint must be null or a path')
    if t['grad_clip'] is not None:
        _real(t['grad_clip'], 'training.grad_clip', positive=True)

    s = config['self_play']
    if s['max_moves'] != BOARD_PLY_LIMIT:
        raise ConfigError(f'self_play.max_moves must be {BOARD_PLY_LIMIT}: Stage 5 self-play '
                          'plays every game to its natural end (no truncation semantics)')
    try:
        self_play_search_config(config)
        evaluation_search_config(config)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    if type(config['augmentation']['enabled']) is not bool:
        raise ConfigError('augmentation.enabled must be a bool')

    o = config['optimizer']
    if o['name'] not in ('adam', 'sgd'):
        raise ConfigError('optimizer.name must be adam or sgd')
    _real(o['lr'], 'optimizer.lr', positive=True)
    _real(o['weight_decay'], 'optimizer.weight_decay', minimum=0)
    _real(o['momentum'], 'optimizer.momentum', minimum=0)
    if o['name'] == 'adam' and o['momentum'] != 0:
        raise ConfigError('optimizer.momentum is only used by sgd')

    loss = config['loss']
    _real(loss['value_weight'], 'loss.value_weight', minimum=0)
    _real(loss['l2_coeff'], 'loss.l2_coeff', minimum=0)
    if o['weight_decay'] != 0 and loss['l2_coeff'] != 0:
        raise ConfigError('use either optimizer.weight_decay or loss.l2_coeff, not both')

    e = config['evaluation']
    _int(e['every'], 'evaluation.every')
    _int(e['opening_random_plies'], 'evaluation.opening_random_plies', 0)
    _int(e['opening_radius'], 'evaluation.opening_radius')
    for name in OPPONENTS:
        games = e[name]
        _int(games['black_games'], f'evaluation.{name}.black_games', 0)
        _int(games['white_games'], f'evaluation.{name}.white_games', 0)
        if games['black_games'] != games['white_games']:
            # Every opening is played once per colour (opening pairs).
            raise ConfigError(f'evaluation.{name}: black_games must equal white_games')
    v6 = e['mcts_v6']
    if v6['use_frozen_config'] is not True:
        raise ConfigError('evaluation.mcts_v6.use_frozen_config must be true (frozen benchmark)')
    if type(v6['final_generation_only']) is not bool:
        raise ConfigError('evaluation.mcts_v6.final_generation_only must be a bool')

    out = config['output']
    if not isinstance(out['runs_dir'], str) or not isinstance(out['run_name'], str):
        raise ConfigError('output.runs_dir and output.run_name must be strings')
    return config


def resolve_config(user: dict | None) -> dict:
    return validate_config(_merge(DEFAULTS, user or {}))


def load_config(path: str | Path) -> dict:
    import yaml  # PyYAML is part of the neural/training extra

    text = Path(path).read_text(encoding='utf-8')
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ConfigError('config file must contain a mapping')
    data.setdefault('format_version', CONFIG_FORMAT)
    return resolve_config(data)


def dump_config(config: dict) -> str:
    import yaml

    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def critical_config(config: dict) -> dict:
    critical = deepcopy(config)
    for path in NON_CRITICAL:
        node = critical
        for key in path[:-1]:
            node = node[key]
        node.pop(path[-1], None)
    return critical


def critical_config_hash(config: dict) -> str:
    return canonical_sha256(critical_config(config))


def config_differences(a: dict, b: dict, prefix: str = '') -> list[str]:
    keys = sorted(set(a) | set(b))
    diffs = []
    for key in keys:
        where = f'{prefix}{key}'
        if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
            diffs.extend(config_differences(a[key], b[key], where + '.'))
        elif a.get(key) != b.get(key):
            diffs.append(where)
    return diffs


def model_config(config: dict) -> ModelConfig:
    return ModelConfig(**config['model'])


def self_play_search_config(config: dict) -> SearchConfig:
    s = config['self_play']
    return SearchConfig(num_simulations=s['simulations'], c_puct=s['c_puct'], tau=s['tau'],
                        temperature_moves=s['temperature_moves'],
                        dirichlet_alpha=s['dirichlet_alpha'],
                        dirichlet_epsilon=s['dirichlet_epsilon'], noise_enabled=True)


def evaluation_search_config(config: dict) -> SearchConfig:
    """Noise OFF, temperature 0 (argmax with the Stage 5 visit/prior/index tie-break)."""
    e = config['evaluation']
    return SearchConfig(num_simulations=e['puct_simulations'], c_puct=e['c_puct'],
                        temperature_moves=0, noise_enabled=False)
