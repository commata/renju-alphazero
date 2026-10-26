"""H7-c Arena of cycle c (docs/h7-c1.md, the contract of docs/track-b-b1-b2.md §8.1 / §9 / §13).

Every tested side is the frozen TB-V3 Resolve / fast5 with a pair of models, built by the candidate factory
``analysis.tb_v3_net.make_tb_v3_net_agent`` (one agent per game, seeded from the game seed). Arms of cycle c:

    official         candidate (c policy + c value) vs the parent (the champion of cycle c - 1), 50 pairs: promotion
    p_only           c policy + parent value vs the parent, 25 pairs (diagnosis only)
    v_only           parent policy + c value vs the parent, 25 pairs (diagnosis only)
    mon_v8_full      candidate vs ``v8:full`` (frozen V8), 25 pairs (Monitoring, record only)
    mon_puct_policy  candidate vs H5 ``v8:puct_policy`` (c_puct 1.5, the frozen H3 policy), 25 pairs (Monitoring)
    mon_e2           candidate vs the E2 VCT2 punisher ``v8:full+vct2atk`` (200k), 25 pairs (Monitoring)

Seeds: every arm of cycle c uses seed 8730 + c (8731-8740 = cycles 1-10, in ``hybrid.h6_labels.RESERVED_SEEDS``,
never training data). Openings and game seeds depend only on the seed and the pair index (the benchmark's
``derive_seed`` / ``make_opening``, 2 random plies within radius 2), so pair i of every arm has the same opening:
P-only / V-only play the first 25 openings of the official arm. Each game is played twice per opening, colours
swapped. A move the rule engine rejects (a black forbidden point, an occupied point) ends the game and is recorded
(``illegal``): by the tested side it is a loss and fails the tactical gate (forbidden errors 0); by the opponent the
game is not scored (``opponent_illegal``, its pair left out and listed), and in the official arm it makes the Arena
invalid (no promotion).

Lineage (fail closed): the candidate's policy and value metadata name this cycle and the one self-play source they were
trained on (``source_models``), which must be the parent; cycle 1's parent is the frozen TB-V3 models, a later cycle's
is the ``next_champion`` of the previous cycle's verdict (``--champion``), whose SHA-256 is in every run identity.

Modes:
    play      one arm: games appended to ``--jsonl`` as they finish (resume; a file of another setup is refused),
              summary (score, pair bootstrap of score - 0.5, Elo) to ``--output``
    tactical  candidate and parent, the same device / inference / probe files (rule ``h7c-tactical-v2``): the value
              on H6 G4's fresh dev probes v2 (D4 average, ``--device``) and P93 / P94 (diagnostic), must_block (raw
              policy, ``tests/fixtures/stage7_probes_v1.json``), E1 dev suite metrics (``unsound_witness``)
    verdict   the arm outputs and the tactical record of one cycle: promotion, convergence signal, Monitoring
              and diagnosis tables (``--history``: the verdicts of the earlier cycles, for the convergence rule)
    no-candidate  the cycle's record when the policy or value training gave ``NO_CANDIDATE``: no Arena,
              ``KEEP_CHAMPION``, no improvement bound (it counts towards the 3-cycle minimum, and breaks a run of
              stalled cycles: it is never one of them). From cycle 2
              a policy NO_CANDIDATE with a value candidate is not such a cycle: its candidate is
              ``VALUE_ONLY_FALLBACK`` (``--candidate-mode``: the parent policy byte for byte + the new value, with the
              policy training's NO_CANDIDATE ``--policy-record``), Arena without ``p_only`` (NOT_APPLICABLE) and
              ``v_only`` (SAME_AS_OFFICIAL), recorded as such in every run, the tactical record and the verdict
    VALUE_ONLY    (cycle >= 3, ``hybrid.h7c_recipe``): a recipe fixed before the self-play, not a training failure. The
              policy is frozen at the parent's (checkpoint and metadata byte for byte), only the value is trained;
              ``--recipe-decision`` (the cycle's decision: parent, champion record, Arena seed) is required and the
              value metadata must carry its SHA-256. No policy record. The same arms as a fallback (``p_only``
              NOT_APPLICABLE, ``v_only`` SAME_AS_OFFICIAL); the decision's SHA-256 is in every run, the tactical
              record and the verdict. A value NO_CANDIDATE gives a ``no-candidate`` record with the decision and the
              value record only

Promotion (all of them): a valid official arm, score > 0.50, pair-bootstrap 95% lower bound of (score - 0.50) >= -0.05,
tactical gate PASS. Tactical gate (v2): forbidden errors 0, ``unsound_witness`` 0, fresh dev probes v2 win and loss
sign accuracy >= 0.80 with >= 20 probes each (D4 average), must_block >= max(9, parent - 2); P93 / P94, the parent's
fresh accuracy and the E1 ratios are recorded beside it, never gated. (v1 gated P93 / P94, which the frozen H6 v3
parent itself fails: a protocol erratum fixed before cycle 2; the cycle-1 verdict stays as recorded.) Then the parent of cycle c + 1 is the candidate, else it stays the champion. Convergence (§9, separate):
at least 3 cycles, the lower bound of (score - 0.5) <= 0 in the last 2 cycles (a cycle without a valid official arm has
no bound and breaks the run), and the last one's tactical gate PASS. Bootstrap: 20,000 resamples of complete opening pairs, RNG seed 0.

    python -u scripts/h7c_arena.py play --cycle 1 --arm official --device cuda --workers 6 \\
        --candidate-policy runs/h7c/c1/policy/best.pt --candidate-value runs/h7c/c1/value/best.pt \\
        --parent-policy runs/h3_policy_64x4/best.pt --parent-value runs/h6_value_head_v3/best.pt \\
        --frozen-policy runs/h3_policy_64x4/best.pt [--champion docs/h7c/c<c-1>_arena_verdict.json] \\
        --jsonl runs/h7c/c1/arena/official.jsonl --output runs/h7c/c1/arena/official.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import BLACK, WHITE, Game, IllegalMove  # noqa: E402

FORMAT = 'h7c-arena-v1'
SEED_BASE, CYCLES = 8730, range(1, 11)
OPENING = {'random_plies': 2, 'radius': 2}
RUNTIME = 'fast5'
BOOTSTRAP = {'samples': 20_000, 'seed': 0}
PROMOTION = {'score_above': 0.5, 'improvement_lower_min': -0.05}
CONVERGENCE = {'min_cycles': 3, 'consecutive': 2, 'improvement_lower_max': 0.0}
PUCT_C = 1.5  # H5 c_puct (mcts-v8-teacher.md §12.16)
# side: (policy, value) of the tested side, each 'candidate' or 'parent'; opponent: 'parent' or a benchmark opponent
ARMS = {
    'official': {'pairs': 50, 'side': ('candidate', 'candidate'), 'opponent': 'parent', 'role': 'promotion'},
    'p_only': {'pairs': 25, 'side': ('candidate', 'parent'), 'opponent': 'parent', 'role': 'diagnosis'},
    'v_only': {'pairs': 25, 'side': ('parent', 'candidate'), 'opponent': 'parent', 'role': 'diagnosis'},
    'mon_v8_full': {'pairs': 25, 'side': ('candidate', 'candidate'), 'opponent': 'v8:full', 'role': 'monitoring'},
    'mon_puct_policy': {'pairs': 25, 'side': ('candidate', 'candidate'), 'opponent': 'v8:puct_policy',
                        'opponent_overrides': {'puct_c': PUCT_C}, 'role': 'monitoring'},
    'mon_e2': {'pairs': 25, 'side': ('candidate', 'candidate'), 'opponent': 'v8:full+vct2atk', 'role': 'monitoring'},
}
_MODELS: dict = {}   # per worker process: loaded policies / value leaves by path


def arena_seed(cycle: int) -> int:
    from hybrid.h6_labels import RESERVED_SEEDS

    if cycle not in CYCLES:
        raise ValueError(f'--cycle {cycle}: the Arena seeds cover cycles {CYCLES.start}-{CYCLES.stop - 1}')
    seed = SEED_BASE + cycle
    if seed not in RESERVED_SEEDS:
        raise ValueError(f'seed {seed} is not reserved (hybrid.h6_labels.RESERVED_SEEDS)')
    return seed


# --- models and agents -------------------------------------------------------------------------------------------

def side_models(arm: str, models: dict) -> tuple[dict, dict | None]:
    """(tested side, parent opponent or None) as {'policy': path, 'value': path}."""
    policy_from, value_from = ARMS[arm]['side']
    tested = {'policy': models[policy_from]['policy'], 'value': models[value_from]['value']}
    return tested, (models['parent'] if ARMS[arm]['opponent'] == 'parent' else None)


def _loaded(policy_checkpoint: str, value_checkpoint: str, device: str):
    from hybrid.h4_policy import RootPolicy
    from hybrid.h6_value_leaf import ValueLeaf

    if ('p', policy_checkpoint) not in _MODELS:
        _MODELS[('p', policy_checkpoint)] = RootPolicy(policy_checkpoint, threads=1)
    if ('v', value_checkpoint, device) not in _MODELS:
        _MODELS[('v', value_checkpoint, device)] = ValueLeaf(value_checkpoint, device=device, threads=1,
                                                             expected_sha256=None)
    return _MODELS[('p', policy_checkpoint)], _MODELS[('v', value_checkpoint, device)]


def install_runtime() -> None:
    from analysis import tb_v2_runtime, tb_v3_fast5
    from rt30 import fast5

    if not fast5.installed():  # fast4 then fast5 in this process, once
        tb_v2_runtime.install()
        tb_v3_fast5.install()


def make_side(side: dict, seed: int, device: str):
    """TB-V3 / fast5 with this model pair, after every check of the candidate factory."""
    from analysis import tb_v3_net

    install_runtime()
    policy, leaf = _loaded(side['policy'], side['value'], device)
    return tb_v3_net.make_tb_v3_net_agent(policy, leaf, seed, side['policy'], side['value'])


def make_benchmark_opponent(arm: str, seed: int, frozen_policy: str):
    from run_mcts_v8_benchmark import make_opponent

    install_runtime()  # exact solver speed-ups: the opponent's moves are those of the plain runtime
    return make_opponent(ARMS[arm]['opponent'], seed, dict(ARMS[arm].get('opponent_overrides', {})), frozen_policy)


# --- identity, tasks, games --------------------------------------------------------------------------------------

def opponent_identity(arm: str, models: dict) -> dict:
    from run_mcts_v8_benchmark import opponent_config

    spec = ARMS[arm]
    if spec['opponent'] == 'parent':
        return {'kind': 'parent', 'models': models['parent_identity']}
    out = {'kind': spec['opponent'], 'config': opponent_config(spec['opponent'], spec.get('opponent_overrides'))}
    if spec['opponent'] == 'v8:puct_policy':
        out['policy'] = models['frozen_policy_identity']
    return out


def run_identity(arm: str, cycle: int, models: dict, device: str, git: dict) -> dict:
    """What every game of one arm run shares; ``signature`` (its SHA-256) is on every game line."""
    from analysis import tb_v3, tb_v3_fast5, tb_v3_net

    tested, _ = side_models(arm, models)
    tested_identity = tb_v3_net.model_identity(tested['policy'], tested['value'])
    return {'format': FORMAT, 'arm': arm, 'role': ARMS[arm]['role'], 'cycle': cycle, 'seed': arena_seed(cycle),
            'pairs': ARMS[arm]['pairs'], 'opening_rule': OPENING, 'engine_id': tb_v3.ENGINE_ID,
            'engine_hash': tb_v3.ENGINE_HASH, 'runtime': RUNTIME, 'runtime_hash': tb_v3_fast5.runtime_hash(device),
            'value_device': device.split(':')[0],
            'candidate': models['candidate_identity'], 'parent': models['parent_identity'],
            'candidate_mode': models.get('candidate_mode', JOINT),
            'fallback_policy_record_sha256': models.get('fallback_policy_record_sha256'),
            'champion_record_sha256': models.get('champion_record_sha256'),
            'recipe_decision_sha256': models.get('recipe_decision_sha256'),
            'tested': {'models': tested_identity, 'model_hash': tb_v3_net.model_hash(tested_identity),
                       'policy_from': ARMS[arm]['side'][0], 'value_from': ARMS[arm]['side'][1]},
            'opponent': opponent_identity(arm, models), 'git': git}


def signature(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_tasks(arm: str, cycle: int, pairs: int | None = None) -> tuple[list[dict], list[dict]]:
    """Tasks of one arm; the openings and game seeds of pair i are the same in every arm of the cycle."""
    from run_mcts_v8_benchmark import derive_seed, make_opening

    seed, tasks, openings = arena_seed(cycle), [], []
    for pair in range(ARMS[arm]['pairs'] if pairs is None else pairs):
        opening_seed = derive_seed(seed, 'opening', pair)
        opening = make_opening(opening_seed, OPENING['random_plies'], OPENING['radius'])
        openings.append({'pair': pair, 'seed': opening_seed, 'moves': [list(m) for m in opening]})
        for color in (BLACK, WHITE):
            name = 'black' if color == BLACK else 'white'
            tasks.append({'key': f'{arm}/{seed}/{pair}/{name}', 'arm': arm, 'pair': pair,
                          'seed': derive_seed(seed, 'game', pair, color), 'tested_color': color,
                          'opening': [list(m) for m in opening]})
    return tasks, openings


def play_game(task: dict, make_tested=None, make_opponent=None, max_plies: int | None = None) -> dict:
    """One Arena game (``max_plies`` only for tests: such a game is ``truncated`` and never scored)."""
    from run_mcts_v8_benchmark import derive_seed

    ctx = task['context']
    tested_side, parent = side_models(task['arm'], ctx['models'])
    make_tested = make_tested or (lambda side, seed: make_side(side, seed, ctx['device']))
    if make_opponent is None:
        make_opponent = (lambda seed: make_side(parent, seed, ctx['device'])) if parent is not None else (
            lambda seed: make_benchmark_opponent(task['arm'], seed, ctx['models'].get('frozen_policy')))
    seed, tested_color = task['seed'], task['tested_color']
    tested = make_tested(tested_side, derive_seed(seed, 'tested'))
    opponent = make_opponent(derive_seed(seed, 'opponent'))
    game = Game()
    for move in task['opening']:
        game.play(*move)
    tested_moves, opponent_routes, illegal, truncated = [], {}, None, False
    started_game = perf_counter()
    while not game.done:
        if max_plies is not None and len(game.history) >= max_plies:
            truncated = True
            break
        is_tested = game.to_play == tested_color
        agent = tested if is_tested else opponent
        started = perf_counter()
        move = agent.select_move(game)
        seconds = perf_counter() - started
        route = getattr(getattr(agent, 'diagnostics', None), 'v8_route', '') or ''
        if is_tested:
            tested_moves.append({'ply': len(game.history), 'route': route, 'seconds': round(seconds, 4)})
        elif route:
            opponent_routes[route] = opponent_routes.get(route, 0) + 1
        try:
            game.play(*move)
        except IllegalMove as exc:
            illegal = {'side': 'tested' if is_tested else 'opponent', 'ply': len(game.history),
                       'color': 'black' if game.to_play == BLACK else 'white',
                       'move': [int(move[0]), int(move[1])], 'reason': str(exc)}
            break
    if illegal is not None:  # the opponent's illegal move never scores for the tested side
        result = 'loss' if illegal['side'] == 'tested' else 'opponent_illegal'
    elif truncated:
        result = 'truncated'
    else:
        result = 'draw' if game.winner is None else 'win' if game.winner == tested_color else 'loss'
    return {'key': task['key'], 'signature': ctx['signature'], 'arm': task['arm'], 'pair': task['pair'],
            'seed': seed, 'tested_color': 'black' if tested_color == BLACK else 'white', 'result': result,
            'winner': game.winner, 'length': len(game.history), 'moves': [list(m) for m in game.history],
            'illegal': illegal, 'truncated': truncated, 'tested_moves': tested_moves,
            'opponent_routes': opponent_routes, 'game_seconds': round(perf_counter() - started_game, 2)}


def _play_chunk(tasks: list[dict]) -> list[dict]:
    return [play_game(t, max_plies=t.get('max_plies')) for t in tasks]


def load_finished(path: Path, keys: set[str], sig: str, log=print) -> dict[str, dict]:
    """Finished games of this run. Refused: a line of another setup (signature), a key twice, an unreadable line
    before the last. An unreadable last line is an append cut by a crash: it is cut off and that game replayed."""
    done = {}
    if not path.exists():
        return done
    offset, content = 0, []
    for raw in path.read_bytes().split(b'\n'):
        if raw.strip():
            content.append((offset, raw))
        offset += len(raw) + 1
    for n, (start, raw) in enumerate(content):
        try:
            row = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if n != len(content) - 1:
                raise ValueError(f'{path}: unreadable line at byte {start} before the end ({exc})') from exc
            log(f'[h7c-arena] {path}: the last line is incomplete: cut, its game is played again', flush=True)
            with path.open('r+b') as handle:
                handle.truncate(start)
            break
        if row.get('signature') != sig:
            raise ValueError(f"{path}: {row.get('key')} was played with another setup (signature differs); "
                             'use a new --jsonl')
        if row['key'] in done:
            raise ValueError(f"{path}: {row['key']} is in the file twice")
        if row['key'] in keys:
            done[row['key']] = row
    return done


# --- summaries and verdicts --------------------------------------------------------------------------------------

def summarize_arm(games: list[dict]) -> dict:
    from evaluation.external import SCORED, complete_pairs, elo, pair_bootstrap

    scored = [g for g in games if g['result'] in SCORED]
    score = (lambda xs: round(sum(SCORED[g['result']] for g in xs) / len(xs), 4) if xs else None)
    pairs = complete_pairs(scored)
    paired = [g for pair in pairs.values() for g in pair]
    total = score(paired)
    ci = pair_bootstrap(paired, **BOOTSTRAP)
    lengths = [g['length'] for g in scored]
    return {
        'games': len(games), 'scored_games': len(scored), 'complete_pairs': len(pairs),
        'incomplete_pairs': sorted({g['pair'] for g in games} - set(pairs)),
        'wins': sum(g['result'] == 'win' for g in paired), 'draws': sum(g['result'] == 'draw' for g in paired),
        'losses': sum(g['result'] == 'loss' for g in paired), 'score': total,
        'score_by_color': {c: score([g for g in paired if g['tested_color'] == c]) for c in ('black', 'white')},
        'score_ci95': ci, 'improvement': None if total is None else round(total - 0.5, 4),
        'improvement_ci95': None if ci is None else [round(x - 0.5, 4) for x in ci],
        'elo': None if total is None or elo(total) is None else round(elo(total), 1),
        'pair_scores': {str(p): sum(SCORED[g['result']] for g in pair) / 2 for p, pair in pairs.items()},
        'illegal': {side: sum(bool(g['illegal']) and g['illegal']['side'] == side for g in games)
                    for side in ('tested', 'opponent')},
        'opponent_illegal_pairs': sorted({g['pair'] for g in games if g['result'] == 'opponent_illegal'}),
        'truncated': sum(g['truncated'] for g in games),
        'length_mean': round(sum(lengths) / len(lengths), 1) if lengths else None,
        'bootstrap': BOOTSTRAP,
    }


def promotion(official: dict, tactical_pass: bool, valid: bool = True) -> dict:
    """§9 / docs/h7-c1.md: a valid official arm, score > 0.50, lower bound of (score - 0.50) >= -0.05, tactical gate
    PASS. An invalid arm has no improvement bound."""
    lower = official['improvement_ci95'][0] if valid and official.get('improvement_ci95') else None
    checks = {'official_valid': bool(valid),
              'score_above_0.50': official['score'] is not None and official['score'] > PROMOTION['score_above'],
              'improvement_lower_ge_-0.05': lower is not None and lower >= PROMOTION['improvement_lower_min'],
              'tactical_gate': bool(tactical_pass)}
    promoted = all(checks.values())
    return {'rule': PROMOTION, 'checks': checks, 'improvement_lower': lower,
            'decision': 'PROMOTE' if promoted else 'KEEP_CHAMPION',
            'reason': 'PROMOTED' if promoted else 'OFFICIAL_INVALID' if not valid else 'NOT_PROMOTED',
            'next_parent': 'candidate' if promoted else 'parent'}


def convergence(cycles: list[dict]) -> dict:
    """§9 over the verdicts of cycles 1..c (in order): at least 3 cycles, the improvement lower bound <= 0 in the
    last 2 (a cycle without one, NO_CANDIDATE or an invalid official arm, breaks the run), and the last one's
    tactical gate PASS."""
    lowers = [c['promotion']['improvement_lower'] for c in cycles]
    last = lowers[-CONVERGENCE['consecutive']:]
    checks = {'min_cycles': len(cycles) >= CONVERGENCE['min_cycles'],
              'consecutive_lower_le_0': len(last) == CONVERGENCE['consecutive']
              and all(x is not None and x <= CONVERGENCE['improvement_lower_max'] for x in last),
              'tactical_gate': bool(cycles) and cycles[-1]['tactical']['verdict'] == 'PASS'}
    return {'rule': CONVERGENCE, 'cycles': [c['cycle'] for c in cycles], 'improvement_lower': lowers,
            'checks': checks, 'converged': all(checks.values())}


# --- tactical gate -----------------------------------------------------------------------------------------------

PROBE_FILE = ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json'  # P93 / P94: diagnostic only
# Tactical rule v2 (protocol erratum before c2, docs/h7-c1.md): the value-sign item is H6 G4's fresh-probe rule on the
# dev set it was fixed for (docs/h6-value.md: both signs >= 0.80, >= 20 probes each, D4 average). P93 / P94 were a
# hard item in v1, which the frozen H6 v3 parent itself fails; they stay as recorded diagnostics.
TACTICAL_RULE = 'h7c-tactical-v2'
FRESH_PROBES = ROOT / 'docs' / 'h6' / 'h6_fresh_dev_probes_v2.json'
FRESH_PROBES_SHA256 = '96407fb0cae50dcf10b3d064f23b286cda49b8adfd7b36155b7aaf60222abcc9'  # docs/h6/h6_dev_probe_eval_v2.json
FRESH_RULE = {'sign_accuracy_min': 0.80, 'min_probes_per_sign': 20, 'inference': 'd4_average'}
POLICY_PROBES = ROOT / 'tests' / 'fixtures' / 'stage7_probes_v1.json'
E1_MANIFEST = ROOT / 'docs' / 'mcts-v8-results' / 'e1' / 'e1_manifest.json'
E1_REPORT = ('rescue_rate', 'veto_recall', 'false_veto')  # compared with the parent, reported (not a gate)
MUST_BLOCK_DROP = 2  # must_block hits may fall at most 2 (of 40) below the parent's


def fresh_rule(fresh: dict) -> dict:
    """``FRESH_RULE`` on ``h6_gate.fresh_probes``' per-sign summary: correct / N, accuracy, mean, pass."""
    out = {}
    for name in ('win', 'loss'):
        rows = [r for r in fresh['probes'] if r['expected_sign'] == (1 if name == 'win' else -1)]
        correct = sum(r['correct'] for r in rows)
        accuracy = correct / len(rows) if rows else None
        out[name] = {'correct': correct, 'probes': len(rows), 'games': fresh[name]['games'], 'accuracy': accuracy,
                     'mean_value': fresh[name]['mean_value'],
                     'pass': len(rows) >= FRESH_RULE['min_probes_per_sign'] and accuracy is not None
                     and accuracy >= FRESH_RULE['sign_accuracy_min']}
    out['pass'] = out['win']['pass'] and out['loss']['pass']
    return out


def value_probes(value_checkpoint, device: str) -> dict:
    """The D4-averaged value on ``device``: the fresh-probe gate (``FRESH_RULE``) and P93 / P94 / P92 (diagnostic)."""
    from hybrid.h6_gate import d4_average, fresh_probes, probes
    from hybrid.h6_value_train import value_forward
    from model.checkpoint import load_checkpoint
    from model.config import ModelConfig

    if _file_sha(FRESH_PROBES) != FRESH_PROBES_SHA256:
        raise ValueError(f'{FRESH_PROBES} is not the H6 dev-v2 probe set the rule names ({FRESH_PROBES_SHA256[:12]})')
    meta = json.loads(Path(value_checkpoint).with_name('best.json').read_text(encoding='utf-8'))
    model = load_checkpoint(value_checkpoint, ModelConfig(**meta['model_config']), device=device)
    model.eval()
    value_fn = d4_average(lambda planes: value_forward(model, planes), device)
    diagnostic = probes(value_fn, PROBE_FILE, device)
    fresh = fresh_probes(value_fn, FRESH_PROBES, device)
    return {'device': device, 'inference': FRESH_RULE['inference'],
            'fresh': {**fresh_rule(fresh), 'file_sha256': FRESH_PROBES_SHA256, 'rows': fresh['probes']},
            'p93_p94': {name: {k: diagnostic[name][k] for k in ('original', 'd4_mean', 'expected_sign', 'pass',
                                                                 'side_to_move')} for name in ('P93', 'P94')},
            'p92': diagnostic['P92']}


def must_block(policy_checkpoint) -> dict:
    """Raw policy top-1 on the must_block probes (the H3 gate: at least the B400 anchor)."""
    from eval_h3_gate import B400_MUST_BLOCK, must_block_hits
    from h7c_train_policy import load_policy_net
    from training.probes import load_probe_set

    model, _ = load_policy_net(Path(policy_checkpoint), 'cpu')
    probes, sha = load_probe_set(POLICY_PROBES)
    hits, total, _ = must_block_hits(model, probes)
    return {'hits': hits, 'total': total, 'anchor': B400_MUST_BLOCK, 'probe_sha256': sha,
            'pass': hits >= B400_MUST_BLOCK}


def _e1_task(task):
    from e1_evaluate import evaluate_position

    position, side, device = task
    return evaluate_position(position, lambda seed: make_side(side, seed, device), 'h7c')


def e1_suite(side: dict, device: str, workers: int, label: str, log=print) -> dict:
    """The E1 dev manifest with this model pair: ``scripts/e1_evaluate.py``'s runs, outcomes and metrics."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from e1_evaluate import summarize as e1_summarize

    manifest = json.loads(E1_MANIFEST.read_text(encoding='utf-8'))
    positions = manifest['positions']
    rows = {}
    progress = Progress(f'h7c-e1-{label}', len(positions), 'positions')
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        def keep(_, result):
            rows[result['key']] = result
        run_chunks(pool, _e1_task, [(p, side, device) for p in positions], progress, lambda _: 1,
                   unit_name='positions', on_result=keep)
    ordered = [rows[p['key']] for p in positions]
    truths_of = {p['key']: {**p['truths'], **rows[p['key']]['truths_added']} for p in positions}
    summary = e1_summarize(ordered, truths_of)
    return {'split': manifest.get('split'), 'positions': len(positions),
            'runs': sum(len(r['runs']) for r in ordered), 'metrics': summary['metrics'],
            'outcomes': summary['outcomes'], 'rows': ordered}


FRESH_ITEM = 'fresh value sign (H6 G4 dev v2: win, loss >= 0.80, >= 20 each)'


def _fresh_brief(fresh: dict) -> dict:
    return {name: {k: fresh[name][k] for k in ('correct', 'probes', 'accuracy', 'mean_value')} for name in
            ('win', 'loss')} | {'pass': fresh['pass']}


def tactical_judge(candidate: dict, parent: dict) -> dict:
    """Gate items of the candidate (all must pass; rule ``TACTICAL_RULE``), the parent's values beside them
    (diagnostic) and the E1 / must_block comparison (reported). The forbidden-error item comes from the Arena
    games and is added by the verdict."""
    e1 = candidate['e1']['metrics']
    block = candidate['must_block']
    block_min = max(block['anchor'], parent['must_block']['hits'] - MUST_BLOCK_DROP)
    cv, pv = candidate['value_probes'], parent['value_probes']
    if (cv['device'], cv['inference'], cv['fresh']['file_sha256']) != (
            pv['device'], pv['inference'], pv['fresh']['file_sha256']):
        raise ValueError('the candidate and the parent were not measured on the same device / inference / probes')
    items = {'unsound_witness == 0': e1['unsound_witness'] == 0,
             FRESH_ITEM: cv['fresh']['pass'],
             'must_block >= max(B400 anchor, parent - 2)': block['hits'] >= block_min}
    evidence = {'rule_version': TACTICAL_RULE, 'fresh_rule': FRESH_RULE, 'fresh_file_sha256': cv['fresh']['file_sha256'],
                'value_device': cv['device'], 'inference': cv['inference'], 'must_block_device': 'cpu',
                'fresh': {'candidate': _fresh_brief(cv['fresh']), 'parent_diagnostic': _fresh_brief(pv['fresh'])},
                'p93_p94_diagnostic': {'candidate': cv['p93_p94'], 'parent': pv['p93_p94']},
                'must_block': {'candidate': block['hits'], 'parent': parent['must_block']['hits'],
                               'min': block_min}}
    def value(metrics, name):
        m = metrics[name]
        return m['value'] if isinstance(m, dict) else m
    compare = {name: {'candidate': value(e1, name), 'parent': value(parent['e1']['metrics'], name)}
               for name in E1_REPORT}
    compare['must_block_hits'] = {'candidate': candidate['must_block']['hits'], 'parent': parent['must_block']['hits']}
    lower_is_worse = {'rescue_rate', 'veto_recall', 'must_block_hits'}
    regressions = sorted(
        name for name, c in compare.items() if c['candidate'] is not None and c['parent'] is not None
        and (c['candidate'] < c['parent'] if name in lower_is_worse else c['candidate'] > c['parent']))
    return {'rule_version': TACTICAL_RULE, 'items': items, 'verdict': 'PASS' if all(items.values()) else 'FAIL',
            'must_block_min': block_min, 'evidence': evidence, 'compare': compare,
            'regressions_reported': regressions}


# --- commands ----------------------------------------------------------------------------------------------------

def _git() -> dict:
    from run_mcts_v8_benchmark import _git_commit, _git_dirty

    return {'commit': _git_commit(), 'dirty': _git_dirty()}


def _json(path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


JOINT, FALLBACK, VALUE_ONLY = 'JOINT', 'VALUE_ONLY_FALLBACK', 'VALUE_ONLY'
CANDIDATE_MODES = (JOINT, FALLBACK, VALUE_ONLY)
FALLBACK_FROM_CYCLE = 2  # approved 2026-10-24 before any c1 Arena result; c1 is not retroactive
FALLBACK_ARMS = {'p_only': 'NOT_APPLICABLE', 'v_only': 'SAME_AS_OFFICIAL'}  # not played when the policy is the parent's
PARENT_POLICY_MODES = (FALLBACK, VALUE_ONLY)


POLICY_RECORD_FORMAT = 'h7c-policy-train-v1'


def check_fallback_record(record: dict, cycle: int, source: dict, value_meta: dict, path) -> None:
    """The policy NO_CANDIDATE record that admits a fallback: this cycle, a clean tree, started from the parent policy
    pair and trained on the same self-play source (models, runtime, generator commit) as the value candidate."""
    inputs = record.get('inputs') or {}
    trained_on = inputs.get('source') or {}
    problems = {
        'format': record.get('format') == POLICY_RECORD_FORMAT,
        'cycle / verdict': (record.get('cycle'), record.get('verdict')) == (cycle, 'NO_CANDIDATE'),
        'clean tree': (record.get('git') or {}).get('dirty') is False,
        'parent policy pair': (inputs.get('parent_checkpoint_sha256'), inputs.get('parent_metadata_sha256')) == (
            source['policy_checkpoint_sha256'], source['policy_metadata_sha256']),
        'self-play models': trained_on.get('models') == source,
        'self-play commit': trained_on.get('selfplay_git_commit') is not None
        and trained_on.get('selfplay_git_commit') == value_meta.get('source_selfplay_git_commit'),
        'runtime': trained_on.get('runtime_hash') is not None
        and trained_on.get('runtime_hash') == value_meta.get('source_runtime_hash'),
    }
    failed = sorted(k for k, ok in problems.items() if not ok)
    if failed:
        raise ValueError(f'{path}: not the policy NO_CANDIDATE record of this fallback ({", ".join(failed)})')


def check_candidate(policy_checkpoint, value_checkpoint, cycle: int, mode: str = JOINT,
                    policy_record=None, recipe_decision=None) -> dict:
    """The candidate's metadata name this cycle and one self-play source; returns that source's four SHA-256.

    ``VALUE_ONLY_FALLBACK`` (cycle >= 2; the policy training gave NO_CANDIDATE, the value training a candidate):
    the policy is the self-play source's own policy, byte for byte (checkpoint and metadata SHA-256), and
    ``policy_record`` (that policy training's ``record.json``) passes ``check_fallback_record``.

    ``VALUE_ONLY`` (cycle >= 3): ``recipe_decision`` is the cycle's ``hybrid.h7c_recipe`` decision; the policy is its
    parent policy byte for byte, the value metadata carries the decision's SHA-256, its self-play source is the
    decision's parent and its replay anchor the decision's; no policy record."""
    from analysis import tb_v3_net
    from hybrid.h7c_train import POLICY_STAGE, SOURCE_MODEL_KEYS, VALUE_STAGE

    if mode not in CANDIDATE_MODES:
        raise ValueError(f'candidate mode {mode!r}: one of {CANDIDATE_MODES}')
    if (recipe_decision is not None) != (mode == VALUE_ONLY):
        raise ValueError(f'--recipe-decision goes with {VALUE_ONLY} and only there')
    if mode == VALUE_ONLY and policy_record is not None:
        raise ValueError(f'{VALUE_ONLY} trains no policy: no --policy-record (that is {FALLBACK})')
    value_meta = _json(Path(value_checkpoint).with_name('best.json'))
    if value_meta.get('h7c_stage') != VALUE_STAGE or value_meta.get('cycle') != cycle:
        raise ValueError(f'{value_checkpoint}: not an H7-c value candidate of cycle {cycle} '
                         f"(stage {value_meta.get('h7c_stage')}, cycle {value_meta.get('cycle')})")
    source = value_meta.get('source_models')
    if not isinstance(source, dict) or set(source) != set(SOURCE_MODEL_KEYS):
        raise ValueError(f'{value_checkpoint}: no self-play source in its metadata')
    if mode == FALLBACK:
        if cycle < FALLBACK_FROM_CYCLE:
            raise ValueError(f'{FALLBACK} applies from cycle {FALLBACK_FROM_CYCLE}')
        policy = tb_v3_net.model_identity(policy_checkpoint, value_checkpoint)
        if any(policy[k] != source[k] for k in ('policy_checkpoint_sha256', 'policy_metadata_sha256')):
            raise ValueError(f'{FALLBACK}: the policy is not the parent policy that played the self-play')
        if policy_record is None:
            raise ValueError(f"{FALLBACK} needs --policy-record (the policy training's record.json)")
        check_fallback_record(_json(policy_record), cycle, source, value_meta, policy_record)
        return source
    if mode == VALUE_ONLY:
        from hybrid import h7c_recipe

        decision, decision_sha = h7c_recipe.load(recipe_decision, cycle)
        policy = tb_v3_net.model_identity(policy_checkpoint, value_checkpoint)
        problems = {
            'policy = parent policy': all(policy[k] == decision['parent'][k]
                                          for k in ('policy_checkpoint_sha256', 'policy_metadata_sha256')),
            'self-play source = parent': source == decision['parent'],
            'value trained under this decision': value_meta.get('recipe_decision_sha256') == decision_sha,
            'replay anchor': (value_meta.get('training_sources') or {}).get('replay_anchor_labels_sha256')
            == decision['value']['replay_anchor_labels_sha256'],
        }
        failed = sorted(k for k, ok in problems.items() if not ok)
        if failed:
            raise ValueError(f'not the {VALUE_ONLY} candidate of {recipe_decision} ({", ".join(failed)})')
        return source
    policy_meta = _json(Path(policy_checkpoint).with_suffix('.json'))
    if policy_meta.get('stage') != POLICY_STAGE or policy_meta.get('cycle') != cycle:
        raise ValueError(f'{policy_checkpoint}: not an H7-c policy candidate of cycle {cycle} '
                         f"(stage {policy_meta.get('stage')}, cycle {policy_meta.get('cycle')})")
    if value_meta.get('h7c_stage') != VALUE_STAGE or value_meta.get('cycle') != cycle:
        raise ValueError(f'{value_checkpoint}: not an H7-c value candidate of cycle {cycle} '
                         f"(stage {value_meta.get('h7c_stage')}, cycle {value_meta.get('cycle')})")
    if policy_meta.get('source_models') != source:
        raise ValueError('the candidate policy and value were not trained on the same self-play source')
    return source


def check_parent(cycle: int, parent_identity: dict, champion, allow_stand_in: bool = False) -> str | None:
    """Cycle 1: the frozen TB-V3 models; cycle c > 1: the ``next_champion`` of cycle c - 1's verdict (its SHA-256)."""
    from analysis import tb_v3

    if cycle == 1:
        if champion is not None:
            raise ValueError('cycle 1 has no previous verdict (--champion)')
        if parent_identity != tb_v3.MODELS and not allow_stand_in:
            raise ValueError('cycle 1: the parent is the frozen TB-V3 models (H3 / H6 v3)')
        return None
    if champion is None:
        raise ValueError(f'cycle {cycle} needs --champion: the verdict of cycle {cycle - 1}')
    record = _json(champion)
    if record.get('format') != f'{FORMAT}-verdict' or record.get('cycle') != cycle - 1:
        raise ValueError(f'{champion} is not the verdict of cycle {cycle - 1}')
    if record.get('next_champion') != parent_identity:
        raise ValueError(f"the parent is not the champion that cycle {cycle - 1}'s verdict names")
    return _file_sha(Path(champion))


def check_models(args, need_frozen_policy: bool) -> dict:
    """Paths and identities of the cycle's models, their lineage, the frozen H3 for the puct_policy opponent."""
    from analysis import tb_v3, tb_v3_net

    models = {'candidate': {'policy': str(args.candidate_policy), 'value': str(args.candidate_value)},
              'parent': {'policy': str(args.parent_policy), 'value': str(args.parent_value)}}
    for name in ('candidate', 'parent'):
        models[f'{name}_identity'] = tb_v3_net.model_identity(models[name]['policy'], models[name]['value'])
    if models['candidate_identity'] == models['parent_identity']:
        raise ValueError('the candidate is the parent')
    mode = getattr(args, 'candidate_mode', JOINT)
    decision_path = getattr(args, 'recipe_decision', None)
    if check_candidate(args.candidate_policy, args.candidate_value, args.cycle, mode,
                       getattr(args, 'policy_record', None), decision_path) != models['parent_identity']:
        raise ValueError("the candidate was not trained on the parent's self-play")
    models['candidate_mode'] = mode
    record = getattr(args, 'policy_record', None)
    models['fallback_policy_record_sha256'] = _file_sha(Path(record)) if mode == FALLBACK else None
    models['champion_record_sha256'] = check_parent(args.cycle, models['parent_identity'],
                                                    getattr(args, 'champion', None), args.allow_stand_in)
    models['recipe_decision_sha256'] = None
    if mode == VALUE_ONLY:
        models['recipe_decision_sha256'] = check_decision_context(decision_path, args.cycle, models)
    if need_frozen_policy:
        if args.frozen_policy is None:
            raise ValueError('this arm needs --frozen-policy (the H3 policy of the H5 opponent)')
        frozen = Path(args.frozen_policy)
        identity = {'policy_checkpoint_sha256': tb_v3_net._sha(frozen),
                    'policy_metadata_sha256': tb_v3_net._sha(frozen.with_suffix('.json'))}
        if identity != {k: tb_v3.MODELS[k] for k in identity} and not args.allow_stand_in:
            raise ValueError(f'--frozen-policy {frozen} is not the frozen H3 policy')
        models['frozen_policy'], models['frozen_policy_identity'] = str(frozen), identity
    return models


def check_decision_context(path, cycle: int, models: dict) -> str:
    """The recipe decision names this cycle's parent, champion record and Arena seed; returns its SHA-256."""
    from hybrid import h7c_recipe

    decision, sha = h7c_recipe.load(path, cycle)
    if decision['parent'] != models['parent_identity']:
        raise ValueError(f'{path}: another parent than this cycle\'s')
    if decision['champion_record_sha256'] != models['champion_record_sha256']:
        raise ValueError(f'{path}: another champion record than --champion')
    if decision['seeds']['arena'] != arena_seed(cycle):
        raise ValueError(f"{path}: Arena seed {decision['seeds']['arena']}, the cycle's is {arena_seed(cycle)}")
    return sha


def play(args, log=print) -> dict:
    git = _git()
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty (or git is unavailable): Arena games must name their code commit')
    models = check_models(args, ARMS[args.arm]['opponent'] == 'v8:puct_policy')
    if models['candidate_mode'] in PARENT_POLICY_MODES and args.arm in FALLBACK_ARMS:
        raise ValueError(f"{args.arm} is {FALLBACK_ARMS[args.arm]} in a {models['candidate_mode']} cycle: not played")
    identity = run_identity(args.arm, args.cycle, models, args.device, git)
    sig = signature(identity)
    tasks, openings = build_tasks(args.arm, args.cycle, getattr(args, 'pairs', None))
    finished = load_finished(args.jsonl, {t['key'] for t in tasks}, sig, log=log)
    context = {'models': models, 'device': args.device, 'signature': sig}
    todo = [{**t, 'context': context, **({'max_plies': args.max_plies} if getattr(args, 'max_plies', None) else {})}
            for t in tasks if t['key'] not in finished]
    log(f"h7c arena: cycle {args.cycle}, arm {args.arm}, seed {identity['seed']}, games {len(tasks)} "
        f"({len(finished)} already done), workers {args.workers}, tested {identity['tested']['model_hash'][:12]}, "
        f"signature {sig[:12]}", flush=True)
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if todo:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        progress = Progress(f'h7c-arena-{args.arm}', len(todo), 'games')
        # spawn: a forked worker cannot re-initialise CUDA
        with args.jsonl.open('a', encoding='utf-8') as out, ProcessPoolExecutor(
                max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            def keep(_, rows):
                for row in rows:
                    out.write(json.dumps(row, separators=(',', ':')) + '\n')
                    finished[row['key']] = row
                out.flush()
            run_chunks(pool, _play_chunk, [[t] for t in todo], progress, len, unit_name='games', on_result=keep)
    games = [finished[t['key']] for t in tasks]
    return {'format': f'{FORMAT}-arm', 'identity': identity, 'signature': sig, 'openings': openings,
            'wall_seconds': round(perf_counter() - started, 1), 'summary': summarize_arm(games),
            'fingerprint': fingerprint(games), 'games': games}


def tactical(args, log=print) -> dict:
    git = _git()
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty (or git is unavailable): the tactical record must name its code commit')
    if args.cycle < FALLBACK_FROM_CYCLE:
        raise ValueError(f'{TACTICAL_RULE} applies from cycle {FALLBACK_FROM_CYCLE}: cycle 1 is historical and immutable')
    models = check_models(args, False)
    out = {'format': f'{FORMAT}-tactical', 'cycle': args.cycle, 'git': git, 'device': args.device,
           'candidate': models['candidate_identity'], 'parent': models['parent_identity'],
           'champion_record_sha256': models['champion_record_sha256'],
           'candidate_mode': models['candidate_mode'],
           'fallback_policy_record_sha256': models['fallback_policy_record_sha256'],
           'recipe_decision_sha256': models['recipe_decision_sha256'], 'rule_version': TACTICAL_RULE,
           'files': {'value_probes': _file_sha(PROBE_FILE), 'fresh_probes': _file_sha(FRESH_PROBES),
                     'policy_probes': _file_sha(POLICY_PROBES), 'e1_manifest': _file_sha(E1_MANIFEST)}}
    for name in ('candidate', 'parent'):
        side = models[name]
        log(f'[h7c-arena] tactical: {name} probes', flush=True)
        out[name + '_result'] = {'value_probes': value_probes(side['value'], args.device),
                                 'must_block': must_block(side['policy']),
                                 'e1': e1_suite(side, args.device, args.workers, name, log=log)}
    out['judge'] = tactical_judge(out['candidate_result'], out['parent_result'])
    return out


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_history(paths, cycle: int, parent: dict, champion_sha256: str | None) -> list[dict]:
    """The verdicts of cycles 1..c-1 in order, each cycle's parent the previous one's ``next_champion``, the last
    one's ``next_champion`` this cycle's parent, and the last file the champion record this cycle's runs named."""
    history = [_json(p) for p in paths]
    if paths and _file_sha(Path(paths[-1])) != champion_sha256:
        raise ValueError(f"{paths[-1]} is not the champion record this cycle's runs named (--champion)")
    if [h.get('cycle') for h in history] != list(range(1, cycle)) or any(
            h.get('format') != f'{FORMAT}-verdict' for h in history):
        raise ValueError(f'--history must be the verdicts of cycles 1-{cycle - 1} in order')
    for before, after in zip(history, history[1:] + [{'cycle': cycle, 'parent': parent}]):
        if before['next_champion'] != after['parent']:
            raise ValueError(f"cycle {after['cycle']}'s parent is not the champion named by cycle {before['cycle']}")
    return history


def fingerprint(games: list[dict]) -> str:
    from evaluation.external import fingerprint as games_fingerprint

    return games_fingerprint(games)


def verdict(args) -> dict:
    """One cycle's promotion record from its six arm outputs and its tactical record (all from one commit and
    the same candidate / parent / champion record)."""
    arms, arm_sha = {}, {}
    for path in args.arms:
        record = _json(path)
        arm = record['identity']['arm']
        if arm in arms:
            raise ValueError(f'{path}: arm {arm} given twice')
        arm_sha[arm] = _file_sha(Path(path))
        if signature(record['identity']) != record['signature'] or any(
                g['signature'] != record['signature'] for g in record['games']):
            raise ValueError(f'{path}: its games do not match its identity')
        if record['summary'] != summarize_arm(record['games']) or record['fingerprint'] != fingerprint(record['games']):
            raise ValueError(f'{path}: its summary or fingerprint is not the one its games give')
        arms[arm] = record
    modes = {r['identity'].get('candidate_mode', JOINT) for r in arms.values()}
    if len(modes) != 1:
        raise ValueError(f'the arms were played in different candidate modes: {sorted(modes)}')
    mode = modes.pop()
    expected = set(ARMS) - (set(FALLBACK_ARMS) if mode in PARENT_POLICY_MODES else set())
    if set(arms) != expected:
        raise ValueError(f'{mode} needs exactly the arms {sorted(expected)}, got {sorted(arms)}')
    tact = _json(args.tactical)
    if tact.get('rule_version') != TACTICAL_RULE or tact['judge'].get('rule_version') != TACTICAL_RULE:
        raise ValueError(f"{args.tactical} is a tactical record of rule {tact.get('rule_version', 'h7c-tactical-v1')}; "
                         f'verdicts from cycle 2 on need {TACTICAL_RULE}')
    if tact['cycle'] < FALLBACK_FROM_CYCLE:
        raise ValueError('cycle 1 is historical and immutable: its verdict is not made again')
    key = ('cycle', 'candidate', 'parent', 'champion_record_sha256', 'fallback_policy_record_sha256',
           'recipe_decision_sha256')
    shared = {json.dumps([r['identity'].get(k) for k in key] + [r['identity']['git']['commit'],
                         r['identity'].get('candidate_mode', JOINT)], sort_keys=True) for r in arms.values()}
    shared.add(json.dumps([tact.get(k) for k in key] + [tact['git']['commit'], tact.get('candidate_mode', JOINT)],
                          sort_keys=True))
    if len(shared) != 1:
        raise ValueError('the arm and tactical records differ in cycle, commit, candidate, parent, champion record, '
                         'recipe decision or candidate mode')
    cycle = tact['cycle']
    official = arms['official']['summary']
    invalid_pairs = official['opponent_illegal_pairs']
    if official['complete_pairs'] + len(invalid_pairs) != ARMS['official']['pairs'] or official['truncated']:
        raise ValueError(f"the official arm has {official['complete_pairs']} complete pairs of "
                         f"{ARMS['official']['pairs']}")
    forbidden = sum(r['summary']['illegal']['tested'] for r in arms.values())
    gate = dict(tact['judge']['items'])
    gate['forbidden errors == 0 (all Arena games)'] = forbidden == 0
    tactical_record = {'rule_version': TACTICAL_RULE, 'items': gate,
                       'verdict': 'PASS' if all(gate.values()) else 'FAIL', 'forbidden_errors': forbidden,
                       'must_block_min': tact['judge'].get('must_block_min'), 'evidence': tact['judge']['evidence'],
                       'compare': tact['judge']['compare'],
                       'regressions_reported': tact['judge']['regressions_reported']}
    out = {'format': f'{FORMAT}-verdict', 'cycle': cycle, 'seed': arena_seed(cycle),
           'git_commit': tact['git']['commit'], 'candidate_mode': mode,
           'fallback_policy_record_sha256': tact.get('fallback_policy_record_sha256'),
           'recipe_decision_sha256': tact.get('recipe_decision_sha256'), 'candidate': tact['candidate'],
           'parent': tact['parent'],
           'champion_record_sha256': tact['champion_record_sha256'],
           'official': {**{k: official[k] for k in ('score', 'score_ci95', 'improvement', 'improvement_ci95', 'elo',
                                                     'wins', 'draws', 'losses', 'score_by_color', 'complete_pairs')},
                        'valid': not invalid_pairs, 'opponent_illegal_pairs': invalid_pairs},
           'tactical': tactical_record,
           'monitoring': {a: _brief(arms[a]['summary']) for a in ARMS if ARMS[a]['role'] == 'monitoring'},
           'diagnosis': {a: FALLBACK_ARMS[a] if mode in PARENT_POLICY_MODES else _brief(arms[a]['summary'])
                         for a in ARMS if ARMS[a]['role'] == 'diagnosis'},
           'fingerprints': {a: arms[a]['fingerprint'] for a in sorted(arms)},
           'arm_sha256': arm_sha,
           'tactical_sha256': _file_sha(Path(args.tactical))}
    out['promotion'] = promotion(official, tactical_record['verdict'] == 'PASS', valid=not invalid_pairs)
    out['next_champion'] = out['candidate'] if out['promotion']['decision'] == 'PROMOTE' else out['parent']
    out['convergence'] = convergence(load_history(args.history, cycle, out['parent'], out['champion_record_sha256'])
                                     + [out])
    return out


def no_candidate(args) -> dict:
    """The record of a cycle whose policy or value training gave no candidate: no Arena, the champion stays."""
    from analysis import tb_v3_net

    decision_path = getattr(args, 'recipe_decision', None)
    if decision_path is not None:  # a VALUE_ONLY cycle: no policy training, the value record alone
        if args.policy_record is not None:
            raise ValueError(f'{VALUE_ONLY} trains no policy: no --policy-record')
        records = {'value': _json(args.value_record)}
    elif args.policy_record is None:
        raise ValueError('--policy-record is required (or --recipe-decision for a VALUE_ONLY cycle)')
    else:
        records = {'policy': _json(args.policy_record), 'value': _json(args.value_record)}
    parent = tb_v3_net.model_identity(args.parent_policy, args.parent_value)
    for name, record in records.items():
        if record.get('cycle') != args.cycle:
            raise ValueError(f'the {name} training record is of cycle {record.get("cycle")}, not {args.cycle}')
        if record['inputs']['parent_checkpoint_sha256'] != parent[f'{name}_checkpoint_sha256']:
            raise ValueError(f'the {name} training did not start from the parent')
    verdicts = {name: record['verdict'] for name, record in records.items()}
    if 'NO_CANDIDATE' not in verdicts.values():
        raise ValueError('both trainings gave a candidate: run the Arena')
    if args.cycle >= FALLBACK_FROM_CYCLE and verdicts == {'policy': 'NO_CANDIDATE', 'value': 'CANDIDATE'}:
        raise ValueError(f'policy NO_CANDIDATE with a value candidate: run the {FALLBACK} Arena (parent policy + '
                         'the new value), not a no-candidate record')
    champion = check_parent(args.cycle, parent, args.champion, args.allow_stand_in)
    decision_sha = None
    if decision_path is not None:
        decision_sha = check_decision_context(decision_path, args.cycle, {'parent_identity': parent,
                                                                          'champion_record_sha256': champion})
        if records['value'].get('inputs', {}).get('recipe_decision_sha256') != decision_sha:
            raise ValueError('the value training record was not made under this recipe decision')
    paths = {'policy': args.policy_record, 'value': args.value_record}
    out = {'format': f'{FORMAT}-verdict', 'cycle': args.cycle, 'seed': arena_seed(args.cycle),
           'git_commit': _git()['commit'], 'candidate': None, 'parent': parent, 'champion_record_sha256': champion,
           'candidate_mode': VALUE_ONLY if decision_path is not None else JOINT, 'recipe_decision_sha256': decision_sha,
           'training': {'verdicts': verdicts, 'record_sha256': {name: _file_sha(paths[name]) for name in records}},
           'official': None, 'tactical': {'verdict': None},
           'promotion': {'rule': PROMOTION, 'decision': 'KEEP_CHAMPION', 'reason': 'NO_CANDIDATE',
                         'improvement_lower': None, 'next_parent': 'parent'},
           'next_champion': parent}
    out['convergence'] = convergence(load_history(args.history, args.cycle, parent, champion) + [out])
    return out


def _brief(summary: dict) -> dict:
    return {k: summary[k] for k in ('score', 'score_ci95', 'improvement_ci95', 'elo', 'wins', 'draws', 'losses',
                                    'complete_pairs', 'score_by_color', 'opponent_illegal_pairs')}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('play', 'tactical'):
        p = sub.add_parser(mode)
        p.add_argument('--cycle', type=int, required=True)
        p.add_argument('--candidate-policy', type=Path, required=True)
        p.add_argument('--candidate-value', type=Path, required=True)
        p.add_argument('--parent-policy', type=Path, required=True)
        p.add_argument('--parent-value', type=Path, required=True)
        p.add_argument('--champion', type=Path, help='cycle c > 1: the verdict of cycle c - 1 (names the parent)')
        p.add_argument('--candidate-mode', choices=CANDIDATE_MODES, default=JOINT,
                       help=f'{FALLBACK}: policy NO_CANDIDATE, value CANDIDATE (cycle >= {FALLBACK_FROM_CYCLE}); '
                            f'{VALUE_ONLY}: a value-only recipe (cycle >= 3, --recipe-decision); in both the '
                            'candidate policy is the parent policy')
        p.add_argument('--policy-record', type=Path, help=f"{FALLBACK}: the policy training's record.json")
        p.add_argument('--recipe-decision', type=Path, help=f"{VALUE_ONLY}: the cycle's recipe decision")
        p.add_argument('--device', default='cuda', help='value device (cuda canonical; cpu certified)')
        p.add_argument('--workers', type=int, default=1)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--allow-dirty', action='store_true', help='tests only')
        p.add_argument('--allow-stand-in', action='store_true', help='tests only: models other than the frozen ones')
        p.add_argument('--frozen-policy', type=Path, help='the frozen H3 policy (the H5 puct_policy opponent)')
    sub.choices['play'].add_argument('--arm', choices=tuple(ARMS), required=True)
    sub.choices['play'].add_argument('--jsonl', type=Path, required=True)
    v = sub.add_parser('verdict')
    v.add_argument('--arms', nargs='+', required=True, help='the six play outputs of the cycle')
    v.add_argument('--tactical', type=Path, required=True)
    v.add_argument('--history', nargs='*', default=[], help='the verdicts of cycles 1..c-1, in order')
    v.add_argument('--output', type=Path, required=True)
    n = sub.add_parser('no-candidate')
    n.add_argument('--cycle', type=int, required=True)
    n.add_argument('--policy-record', type=Path, help="the policy training's record.json (not in a VALUE_ONLY cycle)")
    n.add_argument('--recipe-decision', type=Path, help=f"{VALUE_ONLY}: the cycle's recipe decision")
    n.add_argument('--value-record', type=Path, required=True, help="the value training's record.json")
    n.add_argument('--parent-policy', type=Path, required=True)
    n.add_argument('--parent-value', type=Path, required=True)
    n.add_argument('--champion', type=Path, help='cycle c > 1: the verdict of cycle c - 1')
    n.add_argument('--history', nargs='*', default=[], help='the verdicts of cycles 1..c-1, in order')
    n.add_argument('--allow-stand-in', action='store_true', help='tests only')
    n.add_argument('--output', type=Path, required=True)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    try:
        payload = {'play': play, 'tactical': tactical, 'verdict': verdict,
                   'no-candidate': no_candidate}[args.mode](args)
    except ValueError as exc:
        parser.error(str(exc))
    brief = (payload['summary'] if args.mode == 'play' else payload['judge'] if args.mode == 'tactical'
             else {k: payload[k] for k in ('official', 'tactical', 'promotion', 'convergence')})
    print(json.dumps(brief, indent=1), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0  # KEEP_CHAMPION is a result, not an error: the decision is in the record


if __name__ == '__main__':
    raise SystemExit(main())
