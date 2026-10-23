"""External-engine anchor matches: our agent against a fixed external engine (docs/external-engine.md).

Evaluation only. The external engine (Rapfi first, any Piskvork engine) is an outside examiner:
the games are never training data (their seeds are in ``hybrid.h6_labels.RESERVED_SEEDS``), and the
score is never a promotion gate. Paired openings as in ``run_mcts_v8_benchmark.py``: every opening
(centre + ``--opening-random-plies`` random plies within ``--opening-radius``) is played twice with
colours swapped; openings and game seeds depend only on ``--seed`` and the pair index.

Referee is our ``Game``. Pre-fixed handling (§5):
    engine_illegal  the engine played a move our rules reject (black forbidden, occupied): the game
                    stops, is left out of the score and is listed for a manual rule check
    engine_error    the engine crashed, timed out (``--move-timeout``) or broke the protocol: left
                    out of the score and listed
With ``--forbid-check`` every black-to-move position is also compared with the engine's own
forbidden points (``YXSHOWFORBID``); differences are listed as ``forbid_mismatches``.

Ours (``--ours``): ``tb1`` (TB-Engine-v1 + its H3 checkpoint), ``tbv3`` (the canonical TB-V3 Resolve / fast5 with
the frozen H3 policy and H6 value, ``--value-device cuda`` or the certified ``cpu``; built only by
``analysis.canonical.factory``, refused unless the canonical, its fast5 evidence, its identity and both model files
check), ``v7``, ``v8:<arm>`` (an arm of ``run_mcts_v8_benchmark.py``; a PUCT arm needs ``--ours-puct-c``), ``az``
(a Track A training checkpoint, ``--az-checkpoint``), ``random`` (smoke tests). For ``tbv3`` the engine, runtime and
model identities go into ``ours`` and so into ``run_signature``; its workers are spawned (CUDA) and keep the models
loaded across games.

Games are appended to ``--games-jsonl`` as they finish (a rerun skips finished keys), progress goes
to stdout once a minute. Seeds must be in the external blocks (``evaluation.external``) unless
``--allow-unreserved-seed`` (smoke tests with the fake engine).

    python scripts/run_external_match.py --ours tb1 --engine C:/engines/rapfi/pbrain-rapfi.exe \\
        --engine-name rapfi --engine-info max_node=100000 --engine-info thread_num=1 \\
        --engine-info timeout_turn=600000 --engine-info timeout_match=1000000000 \\
        --seed 8701 --pairs 10 --workers 4 --forbid-check \\
        --games-jsonl runs/external/x1/n100k.jsonl --output runs/external/x1/n100k.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from multiprocessing import get_context
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agents.external_engine import (EngineConfig, EngineError, EngineIllegalMove,  # noqa: E402
                                    ExternalEngineAgent, file_sha256, our_forbidden_points)
from evaluation.external import EXTERNAL_SEEDS, summarize  # noqa: E402
from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import BLACK, WHITE, Game  # noqa: E402
from scripts.run_mcts_v8_benchmark import (_git_commit, _git_dirty, derive_seed, make_opening,  # noqa: E402
                                           parse_opponent)

FORMAT = 'external-match-v1'


def tb1_config_hash() -> str:
    from analysis.tb_engine_v1 import config_hash
    return config_hash()


TBV3 = ('TB-V3', 'fast5')  # the canonical engine and runtime --ours tbv3 stands for
_MODELS: dict = {}         # per worker process: the frozen (policy, value) of --ours tbv3


def tbv3_provenance(policy_checkpoint, value_checkpoint, device: str) -> dict:
    """Fail closed: the canonical is still TB-V3 / fast5, its evidence holds, its identity and both model files are the
    frozen ones; the identities that define our side of the games."""
    from analysis import canonical, tb_v3, tb_v3_fast5, tb_v3_runtime, track_b_guard

    if (canonical.CANONICAL_ENGINE_ID, canonical.CANONICAL_RUNTIME) != TBV3:
        raise ValueError(f'--ours tbv3 is {TBV3}, the canonical is {canonical.CANONICAL_ENGINE_ID} / '
                         f'{canonical.CANONICAL_RUNTIME}')
    canonical.check_fast5_evidence()
    tb_v3.check_identity()
    models = {**track_b_guard.check_policy_checkpoint(policy_checkpoint),
              **track_b_guard.check_value_checkpoint(value_checkpoint)}
    if models != tb_v3.MODELS:
        raise ValueError(f'--ours tbv3 needs the frozen models, got {models}')
    return {'engine_id': tb_v3.ENGINE_ID, 'engine_hash': tb_v3.ENGINE_HASH, 'runtime': TBV3[1],
            'runtime_hash': tb_v3_fast5.runtime_hash(device), 'identity_sha256': tb_v3_runtime.identity_sha256(),
            'value_device': tb_v3_fast5.device_kind(device), **models}


def ours_label(args) -> str:
    if args.ours == 'tbv3':
        return 'tbv3' + ('' if args.value_device == 'cuda' else f'[{args.value_device}]')
    if args.ours == 'az':
        return f"az:{(file_sha256(args.az_checkpoint) or 'missing')[:12]}" + (
            f'[sims={args.az_simulations}]' if args.az_simulations else '')
    label = args.ours + (f'[puct_c={args.ours_puct_c}]' if args.ours_puct_c is not None else '')
    return label + (f'[sims={args.simulations}]' if args.simulations else '')


def make_ours(spec: dict, seed: int):
    """Our agent inside a worker process (``spec`` is plain data)."""
    kind = spec['ours']
    if kind == 'random':
        from agents import RandomAgent
        return RandomAgent(seed=seed)
    if kind == 'tb1':
        from analysis.s3_vct2_v1 import check_checkpoint
        from analysis.tb_engine_v1 import make_agent
        from hybrid.h4_policy import RootPolicy
        check_checkpoint(spec['policy_checkpoint'])  # TB-Engine-v1 includes its checkpoint bytes
        return make_agent(RootPolicy(spec['policy_checkpoint']), seed=seed)
    if kind == 'tbv3':
        from analysis import canonical, tb_v2_runtime, tb_v3_fast5
        from rt30 import fast5
        if not fast5.installed():  # fast4 then fast5 in this process, once
            tb_v2_runtime.install()
            tb_v3_fast5.install()
        key = (spec['policy_checkpoint'], spec['value_checkpoint'], spec['value_device'])
        if key not in _MODELS:
            from hybrid.h4_policy import RootPolicy
            from hybrid.h6_value_leaf import ValueLeaf
            _MODELS[key] = (RootPolicy(spec['policy_checkpoint'], threads=1),
                            ValueLeaf(spec['value_checkpoint'], device=spec['value_device'], threads=1))
        policy, leaf = _MODELS[key]
        # the canonical factory checks identity, models and runtime again for every game
        return canonical.factory(*TBV3)(policy, leaf, seed, Path(spec['policy_checkpoint']),
                                        Path(spec['value_checkpoint']))
    if kind == 'az':
        from agents.alphazero_agent import AlphaZeroAgent
        return AlphaZeroAgent.from_checkpoint(spec['az_checkpoint'], simulations=spec.get('az_simulations'))
    from scripts.run_mcts_v8_benchmark import make_opponent
    overrides = dict(spec.get('overrides', {}))
    return make_opponent(kind, seed, overrides, spec.get('policy_checkpoint'))


def engine_config(spec: dict) -> EngineConfig:
    return EngineConfig(command=tuple(spec['engine_command']), name=spec['engine_name'],
                        info=tuple(tuple(kv) for kv in spec['engine_info']),
                        move_timeout=spec['move_timeout'], extra_files=tuple(spec.get('engine_files', ())))


def _our_record(agent, ply: int, seconds: float) -> dict:
    record = {'ply': ply, 'seconds': round(seconds, 4)}
    route = getattr(getattr(agent, 'diagnostics', None), 'v8_route', '')
    if route:
        record['route'] = route
    return record


def play_one(task: dict) -> dict:
    """One game; ``task`` is a plain dict so it can cross process boundaries."""
    ours = make_ours(task['spec'], derive_seed(task['seed'], 'ours'))
    color = task['ours_color']
    game = Game()
    for move in task['opening']:
        game.play(*move)
    record = {'key': task['key'], 'run_signature': task['spec']['run_signature'], 'pair': task['pair'],
              'seed': task['seed'],
              'ours_color': 'black' if color == BLACK else 'white', 'our_moves': [], 'engine_moves': [],
              'forbid_mismatch': [], 'forbid_checked': 0}
    started_game = perf_counter()
    result = None
    try:
        with ExternalEngineAgent(engine_config(task['spec'])) as engine:
            while not game.done:
                ply = len(game.history)
                if task['forbid_check'] and game.to_play == BLACK:
                    ours_set, theirs = our_forbidden_points(game), engine.forbidden_points(game)
                    record['forbid_checked'] += 1
                    if ours_set != theirs:
                        record['forbid_mismatch'].append({
                            'ply': ply, 'moves': [list(m) for m in game.history],
                            'ours_only': sorted(map(list, ours_set - theirs)),
                            'engine_only': sorted(map(list, theirs - ours_set))})
                if game.to_play == color:
                    started = perf_counter()
                    move = ours.select_move(game)
                    record['our_moves'].append(_our_record(ours, ply, perf_counter() - started))
                else:
                    try:
                        move = engine.select_move(game)
                    except EngineIllegalMove as exc:
                        record['illegal'] = {'ply': ply, 'move': list(exc.move), 'reason': exc.reason}
                        result = 'engine_illegal'
                        break
                    record['engine_moves'].append({
                        'ply': ply, 'seconds': round(engine.last_seconds, 4),
                        'info': engine.engine.last_messages[-1] if engine.engine.last_messages else None})
                game.play(*move)
            record['engine_stderr_errors'] = engine.engine.errors[-5:]
    except EngineError as exc:
        record['error'] = str(exc)
        result = 'engine_error'
    if result is None:
        result = 'draw' if game.winner is None else 'win' if game.winner == color else 'loss'
    record.update(result=result, winner=game.winner, length=len(game.history),
                  moves=[list(m) for m in game.history], game_seconds=round(perf_counter() - started_game, 2))
    return record


def build_tasks(args, spec: dict) -> tuple[list[dict], list[dict]]:
    tasks, openings = [], []
    prefix = f"{spec['ours_label']}@{spec['engine_label']}"
    for pair in range(args.pairs):
        opening_seed = derive_seed(args.seed, 'opening', pair)
        opening = make_opening(opening_seed, args.opening_random_plies, args.opening_radius)
        openings.append({'pair': pair, 'seed': opening_seed, 'moves': [list(m) for m in opening]})
        for color in (BLACK, WHITE):
            tasks.append({
                'key': f'{prefix}/{args.seed}/{pair}/{"black" if color == BLACK else "white"}',
                'pair': pair, 'seed': derive_seed(args.seed, 'game', pair, color), 'ours_color': color,
                'opening': [list(m) for m in opening], 'forbid_check': args.forbid_check, 'spec': spec,
            })
    return tasks, openings


def ours_provenance(args, overrides: dict, tbv3: dict | None = None) -> dict:
    uses_policy = args.ours in ('tb1', 'tbv3') or args.ours.startswith('v8:')
    policy = Path(args.policy_checkpoint) if uses_policy else None
    return {
        'kind': args.ours, 'label': ours_label(args), 'overrides': overrides,
        'policy_checkpoint_sha256': file_sha256(policy) if policy else None,
        'policy_metadata_sha256': file_sha256(policy.with_suffix('.json')) if policy else None,
        'az_checkpoint_sha256': file_sha256(args.az_checkpoint) if args.az_checkpoint else None,
        'az_simulations': args.az_simulations,
        'tb_engine_config_hash': tb1_config_hash() if args.ours == 'tb1' else None,
        **({'tbv3': tbv3} if tbv3 is not None else {}),
    }


def run_signature(engine: dict, ours: dict, args) -> str:
    """SHA-256 of everything a game's moves depend on besides seed and pair (§5).

    Engine bytes (executable and ``--engine-file``s), engine arguments and INFO settings, our agent and
    its checkpoint bytes, the opening rule, ``--forbid-check`` and the record format. Not ``--pairs``
    (a run can be extended with more pairs), workers, the timeout guard or the paths of the files."""
    payload = {
        'format': FORMAT,
        'engine': {'name': engine['name'], 'arguments': engine['command'][1:], 'info': engine['info'],
                   'rule': engine['rule'], 'executable_sha256': engine['executable_sha256'],
                   'extra_files': sorted(engine['extra_files'].values(), key=str)},
        'ours': ours,
        'opening_random_plies': args.opening_random_plies, 'opening_radius': args.opening_radius,
        'forbid_check': args.forbid_check,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_identity(engine: dict, *, strict: bool) -> list[str]:
    """Problems with the engine's identity: a named ``--engine-file`` that cannot be hashed always,
    an executable that cannot be hashed on an anchor seed (``strict``)."""
    problems = [f'--engine-file {path} not found' for path, digest in engine['extra_files'].items() if digest is None]
    if strict and engine['executable_sha256'] is None:
        problems.append(f"engine executable {engine['command'][0]} cannot be hashed (give its file path)")
    return problems


def load_finished(path: Path | None, keys: set[str], signature: str) -> dict[str, dict]:
    """Finished games of this run; ValueError if a line with one of ``keys`` has another signature
    (a resume after the engine, our agent or the opening rule changed would mix two opponents)."""
    finished, foreign = {}, []
    if path is None or not path.exists():
        return finished
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            record = json.loads(line)
            if record['key'] not in keys:
                continue
            if record.get('run_signature') != signature:
                foreign.append(record['key'])
            else:
                finished[record['key']] = record
    if foreign:
        raise ValueError(f'{path}: {len(foreign)} finished games were played with another setup '
                         f'(run_signature differs, e.g. {foreign[0]}); use a new --games-jsonl')
    return finished


def parse_info(values: list[str]) -> list[list[str]]:
    out = []
    for value in values:
        key, sep, setting = value.partition('=')
        if not sep or not key.strip() or not setting.strip() or ' ' in key.strip():
            raise ValueError(f'--engine-info must be KEY=VALUE, got {value!r}')
        if key.strip().lower() == 'rule':
            raise ValueError('rule is fixed to 4 (renju) by the adapter')
        out.append([key.strip().lower(), setting.strip()])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--ours', required=True, help="'tb1', 'tbv3', 'v7', 'v8:<arm>', 'az' or 'random'")
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--value-checkpoint', default=str(ROOT / 'runs/h6_value_head_v3/best.pt'),
                        help='--ours tbv3: the frozen H6 value checkpoint')
    parser.add_argument('--value-device', choices=('cuda', 'cpu'), default='cuda',
                        help='--ours tbv3: cuda (canonical) or cpu (certified)')
    parser.add_argument('--ours-puct-c', type=float, help='c_puct of a v8 PUCT arm')
    parser.add_argument('--az-checkpoint', type=Path)
    parser.add_argument('--az-simulations', type=int)
    parser.add_argument('--simulations', type=int, help='smoke tests only: v7 / v8 simulations')
    parser.add_argument('--engine', required=True, help='engine executable (a Piskvork engine)')
    parser.add_argument('--engine-arg', action='append', default=[], help='extra command-line argument')
    parser.add_argument('--engine-name', default='engine')
    parser.add_argument('--engine-info', action='append', default=[],
                        help='INFO KEY=VALUE sent after START (e.g. max_node=100000, thread_num=1)')
    parser.add_argument('--engine-file', action='append', default=[],
                        help='file hashed into provenance (config.toml, weights)')
    parser.add_argument('--move-timeout', type=float, default=600.0, help='wall-clock guard per engine reply')
    parser.add_argument('--pairs', type=int, default=10)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--allow-unreserved-seed', action='store_true', help='smoke tests only')
    parser.add_argument('--opening-random-plies', type=int, default=4)
    parser.add_argument('--opening-radius', type=int, default=2)
    parser.add_argument('--forbid-check', action='store_true', help='compare forbidden points at every black move')
    parser.add_argument('--loss-window', type=int, default=10)
    parser.add_argument('--workers', type=int, default=1, help='games in parallel processes (one engine each)')
    parser.add_argument('--games-jsonl', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.pairs < 1 or args.workers < 1:
        parser.error('--pairs and --workers must be positive')
    if args.seed not in EXTERNAL_SEEDS and not args.allow_unreserved_seed:
        parser.error(f'--seed must be an external anchor seed {sorted(EXTERNAL_SEEDS)} (docs/external-engine.md §4)')
    if args.ours == 'az' and args.az_checkpoint is None:
        parser.error('--ours az needs --az-checkpoint')
    if args.ours.startswith('v8:'):
        try:
            parse_opponent(args.ours)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.ours not in ('tb1', 'tbv3', 'v7', 'az', 'random'):
        parser.error(f'unknown --ours {args.ours!r}')
    try:
        info = parse_info(args.engine_info)
    except ValueError as exc:
        parser.error(str(exc))
    if not {key for key, _ in info} & {'max_node', 'max_depth'}:
        print('WARNING: no max_node / max_depth: a time-limited engine is not reproducible (docs/external-engine.md §3)',
              flush=True)
    overrides = {k: v for k, v in (('simulations', args.simulations), ('puct_c', args.ours_puct_c)) if v is not None}
    spec = {
        'ours': args.ours, 'policy_checkpoint': args.policy_checkpoint, 'overrides': overrides,
        'az_checkpoint': str(args.az_checkpoint) if args.az_checkpoint else None,
        'az_simulations': args.az_simulations, 'ours_label': ours_label(args),
        'value_checkpoint': args.value_checkpoint, 'value_device': args.value_device,
        'engine_command': [args.engine, *args.engine_arg], 'engine_name': args.engine_name,
        'engine_info': info, 'move_timeout': args.move_timeout, 'engine_files': args.engine_file,
    }
    config = engine_config(spec)
    spec['engine_label'] = config.label
    tbv3 = None
    try:  # fail before any game starts: our identity, the engine must start and our agent must build
        if args.ours == 'tbv3':
            tbv3 = tbv3_provenance(args.policy_checkpoint, args.value_checkpoint, args.value_device)
        with ExternalEngineAgent(config) as probe:
            about = probe.engine.about()
        make_ours(spec, 0)
    except (EngineError, OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.error(f'setup failed: {exc}')

    engine_provenance = config.provenance()
    problems = check_identity(engine_provenance, strict=args.seed in EXTERNAL_SEEDS)
    if problems:
        parser.error('; '.join(problems))
    ours_info = ours_provenance(args, overrides, tbv3)
    spec['run_signature'] = signature = run_signature(engine_provenance, ours_info, args)
    tasks, openings = build_tasks(args, spec)
    try:
        finished = load_finished(args.games_jsonl, {t['key'] for t in tasks}, signature)
    except ValueError as exc:
        parser.error(str(exc))
    pending = [t for t in tasks if t['key'] not in finished]
    print(f"ours={spec['ours_label']} engine={config.label} ({about}) games={len(tasks)} "
          f'finished={len(finished)} pending={len(pending)} workers={args.workers} '
          f'run_signature={signature[:12]}', flush=True)
    if args.games_jsonl is not None:
        args.games_jsonl.parent.mkdir(parents=True, exist_ok=True)

    def keep(_index, record):
        finished[record['key']] = record
        print(f"{record['key']} result={record['result']} moves={record['length']} "
              f"game_s={record['game_seconds']:.0f}", flush=True)
        if args.games_jsonl is not None:
            with args.games_jsonl.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, separators=(',', ':')) + '\n')

    if pending:
        progress = Progress('external', len(pending), 'games')
        context = get_context('spawn') if args.ours == 'tbv3' else None  # CUDA cannot cross a fork
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            run_chunks(pool, play_one, pending, progress, lambda _task: 1, unit_name='games', on_result=keep)

    games = [finished[t['key']] for t in tasks]
    summary = summarize(games, loss_window=args.loss_window)
    payload = {
        'format': FORMAT, 'ours': spec['ours_label'], 'seed': args.seed, 'pairs': args.pairs,
        'opening_random_plies': args.opening_random_plies, 'opening_radius': args.opening_radius,
        'forbid_check': args.forbid_check, 'git_commit': _git_commit(), 'git_dirty': _git_dirty(),
        'run_signature': signature,
        'engine': {**engine_provenance, 'about': about},
        'ours_provenance': ours_info,
        'openings': openings, 'summary': summary, 'games': games,
    }
    brief = {k: v for k, v in summary.items() if k not in ('loss_candidates',)}
    brief['loss_candidates'] = len(summary['loss_candidates'])
    print(json.dumps(brief, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
