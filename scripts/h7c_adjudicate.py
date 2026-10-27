"""E2 disagreement adjudication: the moves two model sets chose differently, judged by Rapfi continuations.

Measurement only (docs/h7-c1.md, "불일치 국면 판정 (Rapfi adjudication): 사전 계약"). The result reads as the choice
quality on fixed disagreement positions under Rapfi continuations: not playing strength, not a minimax answer, not the
cause of a cycle's verdict, and never a promotion or V9 gate.

    observations  every disagreement of the sensitivity record, per comparison A:B (A the candidate, B the reference):
                  (source game, ply, A's move, B's move); counted, never deduplicated or reweighted
    branches      (position = the self-play ``moves[:ply]``, a fixed move): each unique branch is played once by a fresh
                  Rapfi process for both colours to the end, refereed by our ``Game`` (fixed-move legality,
                  ``check_engine_move``, the forbidden-point cross-check at every black move, the result); scored for
                  the side that played the fixed move: win 1, a normal (full board) draw 0.5, loss 0; any timeout,
                  engine error, illegal engine move, forbidden-point mismatch, illegal fixed move, truncation or another
                  end (no legal move with empty points left) is INVALID
    statistics    per comparison: delta = score(A's branch) - score(B's branch); ``better_share`` = share of delta > 0
                  among valid decisive (delta != 0) observations, same-result shares (both win / draw / loss), mean
                  delta, the invalid share; a source-game cluster bootstrap (``BOOTSTRAP``), 95% reported, the reading
                  on the Bonferroni 97.5% interval (BETTER / WORSE / INCONCLUSIVE, with the pre-registered reasons)
    reference     (separate subcommand) the original H3 / H6 v3 at S400 on every branch's child position: the mover's
                  value from that search; never combined with the Rapfi reading

    python -u scripts/h7c_adjudicate.py run --rapfi-dir /mnt/c/engines/rapfi --workers 4

Resume: rerun the same command (finished branches in ``--jsonl`` are kept; another identity, a key twice or an
incomplete record before the last line is refused). One executor per ``--jsonl`` (OS lock). A dirty tree is refused.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from random import Random
import shlex
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import BLACK, Game, IllegalMove  # noqa: E402

FORMAT = 'h7c-adjudicate-v1'
ORDER_SEED = 8771  # hybrid.h6_labels.RESERVED_SEEDS: the branch order only
BOOTSTRAP = {'samples': 10_000, 'seed': 20261028, 'method': 'percentile', 'cluster': 'source game'}
REPORT_LEVEL, READING_LEVEL = 0.95, 0.975  # the reading: Bonferroni over the two comparisons
MIN_DECISIVE, MIN_DECISIVE_GAMES, MAX_INVALID_SHARE = 20, 10, 0.05
COMPARES = (('c8', 'c5'), ('c5', 'orig'))
EXPECTED_DISAGREEMENTS = {'c8:c5': 210, 'c5:orig': 235}
MOVE_TIMEOUT = 600.0
DEFAULTS = {
    'manifest': ROOT / 'docs/h7c/e2/evidence_manifest.json',
    'summary': ROOT / 'docs/h7c/e2/sensitivity_c8.json',
    'sensitivity_jsonl': ROOT / 'runs/h7c/sensitivity/c8_sample.jsonl',
    'selfplay': ROOT / 'runs/h7c/c8/selfplay.jsonl',
    'verdict': ROOT / 'docs/h7c/c8/arena_verdict.json',
    'program': ROOT / 'docs/h7c/b2_program.json',
    'jsonl': ROOT / 'runs/h7c/e2_adjudicate/branches.jsonl',
    'output': ROOT / 'docs/h7c/e2_adjudicate/adjudication.json',
    'manifest_out': ROOT / 'docs/h7c/e2_adjudicate/manifest.json',
    'reference_jsonl': ROOT / 'runs/h7c/e2_adjudicate/reference_s400.jsonl',
    'reference_output': ROOT / 'docs/h7c/e2_adjudicate/reference_s400.json',
}
SMOKE_DIR = ROOT / 'runs/h7c/e2_adjudicate/smoke'  # --limit: every output stays here, never under docs/


def file_sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _rel(path) -> str:
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


# --- inputs ------------------------------------------------------------------------------------------------------

def load_inputs(manifest_path, summary_path, sensitivity_path, selfplay_path, verdict_path=None,
                expected: dict | None = EXPECTED_DISAGREEMENTS) -> dict:
    """The sensitivity record and its self-play, chained by SHA-256 (the E2 manifest, the summary, every record), the
    model sets named by the c8 verdict the manifest pins, the sample, every ply and the actor's move."""
    import h7c_labels as lab
    import h7c_sensitivity as sn
    from analysis import tb_v3

    verdict_path = verdict_path or DEFAULTS['verdict']
    manifest, summary, verdict = (json.loads(Path(p).read_text(encoding='utf-8'))
                                  for p in (manifest_path, summary_path, verdict_path))
    shas = {'manifest': file_sha(manifest_path), 'summary': file_sha(summary_path),
            'sensitivity_jsonl': file_sha(sensitivity_path), 'selfplay': file_sha(selfplay_path),
            'verdict': file_sha(verdict_path)}
    docs = manifest.get('document_files_sha256', {})
    files = manifest.get('files_sha256', {})
    identity = summary.get('identity') or {}
    models = identity.get('models') or {}
    actor = (summary.get('actor_check') or {}).get('model_set')
    checks = {
        'summary named by the manifest': shas['summary'] in docs.values(),
        'sensitivity JSONL named by the manifest': shas['sensitivity_jsonl'] in files.values(),
        'sensitivity JSONL = the summary\'s': summary.get('jsonl_sha256') == shas['sensitivity_jsonl'],
        'self-play = the summary\'s': identity.get('selfplay_jsonl_sha256') == shas['selfplay'],
        'verdict pinned by the manifest': manifest.get('champion_verdict_sha256') == shas['verdict'],
        'actor reproduced': (summary.get('actor_check') or {}).get('reproduced') is True,
        'c8 = the verdict\'s candidate': models.get('c8') == verdict.get('candidate'),
        'c5 = the verdict\'s parent': models.get('c5') == verdict.get('parent'),
        'orig = the frozen H3 / H6 v3': models.get('orig') == tb_v3.MODELS,
        'the actor is c5, the self-play models': actor == 'c5' and models.get('c5') == (
            identity.get('selfplay') or {}).get('models'),
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    if failed:
        raise ValueError(f'adjudication inputs do not chain ({", ".join(failed)})')
    records = [json.loads(line) for line in Path(sensitivity_path).read_text(encoding='utf-8').splitlines()
               if line.strip()]
    if any(r.get('identity') != identity for r in records):
        raise ValueError('a sensitivity record has another identity than the summary')
    games = {g['game_index']: g for g in lab.read_games([selfplay_path])}
    sample = sn.sample_games(list(games.values()), identity['games_sample'])
    ids = [lab.game_id(g) for g in sample]
    if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != identity['sample_sha256']:
        raise ValueError('the self-play file does not give the recorded sample')
    indices = [r['game_index'] for r in records]
    if len(set(indices)) != len(indices) or set(indices) != {g['game_index'] for g in sample}:
        raise ValueError('the sensitivity records are not the sample (missing, extra or duplicate games)')
    for r in records:
        expected_plies = games[r['game_index']]['plies']
        if len(r['plies']) != len(expected_plies):
            raise ValueError(f"game {r['game_index']}: the sensitivity plies are not the self-play's")
        for p, original in zip(r['plies'], expected_plies):
            if (p['ply'], p['color'], p['actor_route'], p['actor_move']) != (
                    original['ply'], original['color'], original['route'], original['engine_move']) \
                    or p['sets'][actor]['engine_move'] != original['engine_move'] \
                    or set(p['sets']) != set(models):
                raise ValueError(f"game {r['game_index']} ply {p['ply']}: the record is not the self-play's "
                                 f"position, actor move or model sets")
    obs = observations(records, games, summary)
    counts = {}
    for o in obs:
        counts[o['compare']] = counts.get(o['compare'], 0) + 1
    recorded = {c['pair']: c['all_plies']['different_moves'] for c in summary['comparisons']}
    for name in {o['compare'] for o in obs} | set(recorded):
        if counts.get(name, 0) != recorded.get(name):
            raise ValueError(f'{name}: {counts.get(name, 0)} disagreements in the records, the summary says '
                             f'{recorded.get(name)}')
    if expected is not None and {k: counts.get(k) for k in expected} != expected:
        raise ValueError(f'the disagreement counts {counts} are not the pre-registered {expected}')
    return {'summary': summary, 'records': records, 'games': games, 'observations': obs, 'shas': shas,
            'counts': counts}


def observations(records: list[dict], games: dict, summary: dict) -> list[dict]:
    """Every disagreement per comparison (A the candidate, B the reference), in record order."""
    pairs = [tuple(c['pair'].split(':')) for c in summary['comparisons']]
    out = []
    for r in records:
        g = games[r['game_index']]
        for p in r['plies']:
            for a, b in pairs:
                ma, mb = p['sets'][a]['engine_move'], p['sets'][b]['engine_move']
                if list(ma) != list(mb):
                    out.append({'compare': f'{a}:{b}', 'a': a, 'b': b, 'game_index': r['game_index'], 'ply': p['ply'],
                                'color': p['color'], 'moves': [list(m) for m in g['moves'][:p['ply']]],
                                'move_a': list(ma), 'move_b': list(mb)})
    return out


def branch_key(moves, move) -> str:
    return sha_json([[list(m) for m in moves], list(move)])


def branch_table(obs: list[dict]) -> dict[str, dict]:
    table = {}
    for o in obs:
        for side in ('a', 'b'):
            key = branch_key(o['moves'], o[f'move_{side}'])
            table.setdefault(key, {'key': key, 'moves': o['moves'], 'move': o[f'move_{side}']})
    return table


def order(table: dict) -> list[str]:
    keys = sorted(table)
    Random(ORDER_SEED).shuffle(keys)
    return keys


def overlap(obs: list[dict]) -> dict:
    """How the observations share positions and branches (counted, nothing is removed)."""
    out, positions = {}, {}
    for name in sorted({o['compare'] for o in obs}):
        mine = [o for o in obs if o['compare'] == name]
        keys = [branch_key(o['moves'], o[f'move_{s}']) for o in mine for s in ('a', 'b')]
        out[name] = {'observations': len(mine), 'branch_uses': len(keys), 'unique_branches': len(set(keys)),
                     'unique_positions': len({sha_json(o['moves']) for o in mine})}
        positions[name] = {sha_json(o['moves']) for o in mine}
    names = sorted(positions)
    if len(names) == 2:
        out['shared_positions'] = len(positions[names[0]] & positions[names[1]])
        a = {branch_key(o['moves'], o[f'move_{s}']) for o in obs if o['compare'] == names[0] for s in ('a', 'b')}
        b = {branch_key(o['moves'], o[f'move_{s}']) for o in obs if o['compare'] == names[1] for s in ('a', 'b')}
        out['shared_branches'] = len(a & b)
    return out


# --- the engine --------------------------------------------------------------------------------------------------

def engine_spec(program_path, rapfi_dir, command=None, allow_unregistered=False) -> dict:
    """The X2 Rapfi of the B2 program (executable, files, INFO), its hashes checked; ``command`` only for tests."""
    from agents.external_engine import EngineConfig

    external = json.loads(Path(program_path).read_text(encoding='utf-8'))['external']
    rapfi_dir = Path(rapfi_dir)
    files = [rapfi_dir / f for f in external['files']]
    info = [kv.split('=', 1) for kv in external['info']]
    exe = rapfi_dir / external['executable']
    spec = {'command': [str(exe)] if command is None else list(command), 'name': 'rapfi', 'info': info,
            'files': [str(f) for f in files], 'move_timeout': MOVE_TIMEOUT}
    config = EngineConfig(command=tuple(spec['command']), name=spec['name'], info=tuple(map(tuple, info)),
                          move_timeout=MOVE_TIMEOUT, extra_files=tuple(spec['files']))
    provenance = config.provenance()
    registered = {'executable_sha256': external['executable_sha256'], 'files': external['files'],
                  'info': external['info']}
    actual = {'executable_sha256': provenance['executable_sha256'],
              'files': {Path(p).name: s for p, s in provenance['extra_files'].items()},
              'info': [f'{k}={v}' for k, v in info]}
    if actual != registered and not allow_unregistered:
        raise ValueError('Rapfi is not the registered X2 engine (executable, files or INFO differ)')
    spec['identity'] = {'registered': registered, 'actual': actual, 'rule': provenance['rule'],
                        'arguments': spec['command'][1:], 'move_timeout': MOVE_TIMEOUT,
                        'registered_engine': actual == registered}
    return spec


def _config(spec):
    from agents.external_engine import EngineConfig

    return EngineConfig(command=tuple(spec['command']), name=spec['name'], info=tuple(map(tuple, spec['info'])),
                        move_timeout=spec['move_timeout'], extra_files=tuple(spec['files']))


def score_for(mover: int, game: Game) -> float:
    """The fixed move's side: win 1, a normal draw 0.5, loss 0 (the game must be over)."""
    if not game.done:
        raise ValueError('the game is not over')
    if game.winner is None:
        return 0.5
    return 1.0 if game.winner == mover else 0.0


def play_branch(task: dict) -> dict:
    """One branch: the fixed move on the position, then a fresh Rapfi process for both colours to the end."""
    from agents.external_engine import (EngineError, EngineIllegalMove, EngineTimeout, ExternalEngineAgent,
                                        our_forbidden_points)

    started = perf_counter()
    game = Game()
    for move in task['moves']:
        game.play(*move)
    mover = game.to_play
    out = {'key': task['key'], 'order': task['order'], 'identity_sha256': task['identity_sha256'],
           'ply': len(task['moves']), 'mover': 'black' if mover == BLACK else 'white', 'move': task['move'],
           'continuation': [], 'forbid_checked': 0}

    def invalid(reason, **extra):
        return {**out, 'status': 'INVALID', 'reason': reason, 'score': None, 'length': len(game.history),
                'seconds': round(perf_counter() - started, 2), **extra}

    try:
        game.play(*task['move'])
    except IllegalMove as exc:
        return invalid('fixed_move_illegal', detail=str(exc))
    if not game.done:
        try:
            with ExternalEngineAgent(_config(task['spec'])) as engine:  # a new process for this branch only
                while not game.done:
                    if task.get('max_plies') is not None and len(game.history) >= task['max_plies']:
                        return invalid('truncated')
                    if game.to_play == BLACK:
                        ours, theirs = our_forbidden_points(game), engine.forbidden_points(game)
                        out['forbid_checked'] += 1
                        if ours != theirs:
                            return invalid('forbid_mismatch', ply_at=len(game.history),
                                           ours_only=sorted(map(list, ours - theirs)),
                                           engine_only=sorted(map(list, theirs - ours)))
                    try:
                        move = engine.select_move(game)
                    except EngineIllegalMove as exc:
                        return invalid('engine_illegal', move_at=list(exc.move), detail=exc.reason)
                    game.play(*move)
                    out['continuation'].append([int(move[0]), int(move[1])])
        except EngineTimeout as exc:
            return invalid('timeout', detail=str(exc)[:500])
        except EngineError as exc:
            return invalid('engine_error', detail=str(exc)[:500])
    if game.winner is None and any(0 in row for row in game.board):
        return invalid('abnormal_end')  # no legal move with empty points left: not the normal (full board) draw
    winner = None if game.winner is None else 'black' if game.winner == BLACK else 'white'
    return {**out, 'status': 'VALID', 'reason': None, 'score': score_for(mover, game), 'winner': winner,
            'length': len(game.history), 'seconds': round(perf_counter() - started, 2)}


def _chunk(tasks):
    return [play_branch(t) for t in tasks]


# --- records -----------------------------------------------------------------------------------------------------

RECORD_KEYS = ('key', 'order', 'identity_sha256', 'status', 'score', 'length')


def load_finished(path: Path, identity_sha: str, keys: set[str]) -> dict[str, dict]:
    """Finished branches of this identity; refused: another identity, a key twice, an unknown key, an incomplete
    record before the last line (an incomplete last line is a crash while appending: cut and played again)."""
    done = {}
    if not path.exists():
        return done
    raw = path.read_bytes()
    offset, lines = 0, []
    for line in raw.split(b'\n'):
        lines.append((offset, line))
        offset += len(line) + 1
    content = [(o, line) for o, line in lines if line.strip()]
    for n, (start, line) in enumerate(content):
        try:
            record = json.loads(line)
            complete = all(k in record for k in RECORD_KEYS) and record['status'] in ('VALID', 'INVALID')
        except (json.JSONDecodeError, UnicodeDecodeError):
            record, complete = None, False
        if not complete:
            if n != len(content) - 1:
                raise ValueError(f'{path}: an incomplete record at byte {start} before the end')
            with path.open('r+b') as handle:
                handle.truncate(start)
            return done
        if record['identity_sha256'] != identity_sha:
            raise ValueError(f'{path}: a branch of another identity (inputs, code, engine or settings differ)')
        if record['key'] in done:
            raise ValueError(f"{path}: branch {record['key'][:12]} twice")
        if record['key'] not in keys:
            raise ValueError(f"{path}: branch {record['key'][:12]} is not one of this run's")
        done[record['key']] = record
    if raw and not raw.endswith(b'\n'):  # a complete last record cut before its newline: the next append is a line
        with path.open('ab') as handle:
            handle.write(b'\n')
    return done


# --- statistics --------------------------------------------------------------------------------------------------

def _quantiles(values: list[float], level: float) -> list[float]:
    values = sorted(values)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return [round(values[int(lo * (len(values) - 1))], 5), round(values[int(hi * (len(values) - 1))], 5)]


def bootstrap(groups: dict, statistic) -> dict:
    """Source-game cluster bootstrap: ``statistic(list of observations)`` or None when undefined."""
    ids = sorted(groups)
    rng = Random(BOOTSTRAP['seed'])
    values, undefined = [], 0
    for _ in range(BOOTSTRAP['samples']):
        draw = [o for _ in ids for o in groups[ids[rng.randrange(len(ids))]]]
        v = statistic(draw)
        if v is None:
            undefined += 1
        else:
            values.append(v)
    if undefined or not values:
        return {'undefined_resamples': undefined, 'ci95': None, 'ci975': None}
    return {'undefined_resamples': 0, 'ci95': _quantiles(values, REPORT_LEVEL),
            'ci975': _quantiles(values, READING_LEVEL)}


def _better_share(obs):
    decisive = [o for o in obs if o['delta'] != 0]
    return sum(o['delta'] > 0 for o in decisive) / len(decisive) if decisive else None


def _mean_delta(obs):
    return sum(o['delta'] for o in obs) / len(obs) if obs else None


def compare_stats(name: str, obs: list[dict], results: dict) -> dict:
    rows, invalid, missing = [], 0, 0
    for o in obs:
        ra, rb = (results.get(branch_key(o['moves'], o[f'move_{s}'])) for s in ('a', 'b'))
        if ra is None or rb is None:
            missing += 1
        elif ra['status'] != 'VALID' or rb['status'] != 'VALID':
            invalid += 1
        else:
            rows.append({**o, 'score_a': ra['score'], 'score_b': rb['score'], 'delta': ra['score'] - rb['score']})
    n = len(obs)
    decisive = [r for r in rows if r['delta'] != 0]
    same = [r for r in rows if r['delta'] == 0]
    groups = {}
    for r in rows:
        groups.setdefault(r['game_index'], []).append(r)
    out = {'comparison': name, 'candidate': obs[0]['a'] if obs else None, 'reference': obs[0]['b'] if obs else None,
           'observations': n, 'source_games': len({o['game_index'] for o in obs}), 'valid': len(rows),
           'invalid': invalid, 'missing': missing, 'invalid_share': round(invalid / n, 5) if n else None,
           'decisive': len(decisive), 'decisive_games': len({r['game_index'] for r in decisive}),
           'better': sum(r['delta'] > 0 for r in decisive), 'worse': sum(r['delta'] < 0 for r in decisive),
           'better_share': None if not decisive else round(_better_share(rows), 5),
           'same_result': {'share': round(len(same) / len(rows), 5) if rows else None,
                           'both_win': sum(r['score_a'] == 1 for r in same),
                           'both_draw': sum(r['score_a'] == 0.5 for r in same),
                           'both_loss': sum(r['score_a'] == 0 for r in same)},
           'half_point_deltas': sum(abs(r['delta']) == 0.5 for r in rows),
           'mean_delta': None if not rows else round(_mean_delta(rows), 5)}
    reasons = []
    if missing:
        reasons.append('INCOMPLETE')
    if n and invalid / n > MAX_INVALID_SHARE:
        reasons.append('INVALID_RATE')
    if len(decisive) < MIN_DECISIVE or out['decisive_games'] < MIN_DECISIVE_GAMES:
        reasons.append('TOO_FEW_DECISIVE')
    out['better_share_bootstrap'] = None
    out['mean_delta_bootstrap'] = bootstrap(groups, _mean_delta) if rows and not missing else None
    if not reasons:
        b = bootstrap(groups, _better_share)
        out['better_share_bootstrap'] = b
        if b['undefined_resamples']:
            reasons.append('DEGENERATE_BOOTSTRAP')
    if reasons:
        out['reading'], out['reasons'] = 'INCONCLUSIVE', reasons
    else:
        lo, hi = out['better_share_bootstrap']['ci975']
        out['reading'] = 'BETTER' if lo > 0.5 else 'WORSE' if hi < 0.5 else 'INCONCLUSIVE'
        out['reasons'] = [] if out['reading'] != 'INCONCLUSIVE' else ['INTERVAL_CONTAINS_HALF']
    return out


# --- the run -----------------------------------------------------------------------------------------------------

def run(args, log=print) -> dict:
    from h7c_arena import _git
    from h7c_loop import exclusive

    git = _git()
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty (or git is unavailable): the adjudication must name its code commit')
    inputs = load_inputs(args.manifest, args.summary, args.sensitivity_jsonl, args.selfplay, args.verdict,
                         None if args.allow_unregistered_engine else EXPECTED_DISAGREEMENTS)
    spec = engine_spec(args.program, args.rapfi_dir, args.engine_command, args.allow_unregistered_engine)
    obs = inputs['observations']
    table = branch_table(obs)
    keys = order(table)
    if args.limit is not None:
        keys = keys[:args.limit]
    identity = {'format': FORMAT, 'git': git, 'inputs': {k: v for k, v in inputs['shas'].items()},
                'input_paths': {k: _rel(getattr(args, k)) for k in ('manifest', 'summary', 'sensitivity_jsonl',
                                                                     'selfplay', 'verdict')},
                'engine': spec['identity'], 'order_seed': ORDER_SEED, 'branches_sha256': sha_json(sorted(table)),
                'limit': args.limit, 'max_plies': args.max_plies}
    identity_sha = sha_json(identity)
    jsonl = Path(args.jsonl)
    with exclusive(jsonl.with_name(jsonl.name + '.lock')):
        finished = load_finished(jsonl, identity_sha, set(keys))
        position = {k: i for i, k in enumerate(order(table))}
        todo = [{'key': k, 'order': position[k], 'moves': table[k]['moves'], 'move': table[k]['move'],
                 'spec': spec, 'identity_sha256': identity_sha, 'max_plies': args.max_plies}
                for k in keys if k not in finished]
        log(f'[h7c-adjudicate] {len(obs)} observations ({inputs["counts"]}), {len(table)} unique branches'
            f'{"" if args.limit is None else f", limit {args.limit}"}; {len(finished)} done, {len(todo)} to play, '
            f'workers {args.workers}', flush=True)
        if todo:
            jsonl.parent.mkdir(parents=True, exist_ok=True)
            with jsonl.open('a', encoding='utf-8') as out, ProcessPoolExecutor(max_workers=args.workers) as pool:
                def keep(_, records):
                    for record in records:
                        out.write(json.dumps(record, separators=(',', ':')) + '\n')
                        finished[record['key']] = record
                    out.flush()
                run_chunks(pool, _chunk, [[t] for t in todo], Progress('h7c-adjudicate', len(todo), 'branches'),
                           len, unit_name='branches', on_result=keep)
    names = [f'{a}:{b}' for a, b in COMPARES if any(o['compare'] == f'{a}:{b}' for o in obs)] or \
        sorted({o['compare'] for o in obs})
    stats = [compare_stats(name, [o for o in obs if o['compare'] == name], finished) for name in names]
    invalid = {}
    for r in finished.values():
        if r['status'] == 'INVALID':
            invalid[r['reason']] = invalid.get(r['reason'], 0) + 1
    result = {'format': f'{FORMAT}-summary', 'identity': identity, 'identity_sha256': identity_sha,
              'jsonl_sha256': file_sha(jsonl) if jsonl.exists() else None,
              'meaning': 'choice quality on fixed disagreement positions under Rapfi continuations; not playing '
                         'strength, not a minimax answer, not a cause, never a gate',
              'rules': {'bootstrap': BOOTSTRAP, 'report_level': REPORT_LEVEL, 'reading_level': READING_LEVEL,
                        'min_decisive': MIN_DECISIVE, 'min_decisive_games': MIN_DECISIVE_GAMES,
                        'max_invalid_share': MAX_INVALID_SHARE},
              'branches': {'unique': len(table), 'played': len(finished), 'invalid_by_reason': invalid},
              'overlap': overlap(obs), 'comparisons': stats}
    no_reading = (['LIMITED_RUN'] if args.limit is not None else []) + \
        ([] if spec['identity']['registered_engine'] else ['UNREGISTERED_ENGINE'])
    if no_reading:  # a smoke or test run: never a reading
        for s in result['comparisons']:
            s['reading'], s['reasons'] = 'SMOKE', no_reading
    return result


# --- the S400 reference ------------------------------------------------------------------------------------------

def mover_value(visits: list, done_winner: int | None, mover: int, done: bool) -> float | None:
    """The fixed move's side value after the move: the game result when over, else minus the opponent's value of
    its most visited root child (the root's mean values are for the side to move)."""
    if done:
        return 0.0 if done_winner is None else (1.0 if done_winner == mover else -1.0)
    if not visits:
        return None
    best = max(visits, key=lambda v: (v[2], v[3]))
    return -float(best[3])


def _reference_chunk(tasks):
    import h7c_budget as sb

    out = []
    for t in tasks:
        game = Game()
        for move in t['moves']:
            game.play(*move)
        mover = game.to_play
        game.play(*t['move'])
        record = {'key': t['key'], 'identity_sha256': t['identity_sha256']}
        if game.done:
            record.update(route='terminal', value=mover_value([], game.winner, mover, True))
        else:
            agent = sb.make_budget_agent(8, t['policy'], t['value'], t['seed'], t['device'])
            d = sb.decision(agent, game)
            record.update(route=d['route'], value=mover_value(d['visits'], None, mover, False),
                          seconds=d['seconds'])
        out.append(record)
    return out


def reference(args, log=print) -> dict:
    """S400 with the original H3 / H6 v3 on every branch child position (a separate reference, never combined)."""
    from analysis import tb_v3, tb_v3_net
    from h7c_arena import _git

    git = _git()
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty: the reference must name its code commit')
    if tb_v3_net.model_identity(args.policy_checkpoint, args.value_checkpoint) != tb_v3.MODELS \
            and not args.allow_stand_in:
        raise ValueError('the reference searches with the original H3 / H6 v3 only')
    inputs = load_inputs(args.manifest, args.summary, args.sensitivity_jsonl, args.selfplay, args.verdict,
                         None if args.allow_stand_in else EXPECTED_DISAGREEMENTS)
    table = branch_table(inputs['observations'])
    keys = order(table)[:args.limit] if args.limit is not None else order(table)
    identity = {'format': f'{FORMAT}-reference-s400', 'git': git, 'inputs': inputs['shas'],
                'models': tb_v3_net.model_identity(args.policy_checkpoint, args.value_checkpoint),
                'budget': {'simulations': 400, 'tactical_simulations': 800}, 'device': args.device,
                'branches_sha256': sha_json(sorted(table)), 'limit': args.limit}
    identity_sha = sha_json(identity)
    jsonl = Path(args.jsonl)
    done = {}
    if jsonl.exists():
        for line in jsonl.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                if r['identity_sha256'] != identity_sha or r['key'] in done:
                    raise ValueError(f'{jsonl}: another identity or a key twice')
                done[r['key']] = r
    todo = [{'key': k, 'moves': table[k]['moves'], 'move': table[k]['move'], 'identity_sha256': identity_sha,
             'policy': str(args.policy_checkpoint), 'value': str(args.value_checkpoint), 'device': args.device,
             'seed': ORDER_SEED} for k in keys if k not in done]
    log(f'[h7c-adjudicate] reference S400: {len(keys)} branches, {len(done)} done', flush=True)
    if todo:
        from multiprocessing import get_context
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        with jsonl.open('a', encoding='utf-8') as out, ProcessPoolExecutor(
                max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            def keep(_, records):
                for r in records:
                    out.write(json.dumps(r, separators=(',', ':')) + '\n')
                    done[r['key']] = r
                out.flush()
            run_chunks(pool, _reference_chunk, [[t] for t in todo], Progress('h7c-reference', len(todo), 'branches'),
                       len, unit_name='branches', on_result=keep)
    per = {}
    for o in inputs['observations']:
        va = done.get(branch_key(o['moves'], o['move_a']), {}).get('value')
        vb = done.get(branch_key(o['moves'], o['move_b']), {}).get('value')
        if va is not None and vb is not None:
            per.setdefault(o['compare'], []).append(va - vb)
    return {'format': f'{FORMAT}-reference-s400-summary', 'identity': identity, 'identity_sha256': identity_sha,
            'jsonl_sha256': file_sha(jsonl) if jsonl.exists() else None,
            'meaning': 'reference only: the original models at S400; never combined with the Rapfi reading',
            'comparisons': {k: {'observations': len(v), 'mean_value_difference': round(sum(v) / len(v), 5),
                                'candidate_higher': sum(x > 0 for x in v), 'reference_higher': sum(x < 0 for x in v)}
                            for k, v in sorted(per.items())}}


# --- the command line --------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'reference'):
        p = sub.add_parser(name)
        for key in ('manifest', 'summary', 'sensitivity_jsonl', 'selfplay', 'verdict'):
            p.add_argument('--' + key.replace('_', '-'), type=Path, default=DEFAULTS[key])
        p.add_argument('--workers', type=int, default=4)
        p.add_argument('--limit', type=int, help='smoke runs: the first N branches of the order, no reading')
        p.add_argument('--allow-dirty', action='store_true', help='tests only')
    run_p, ref_p = sub.choices['run'], sub.choices['reference']
    run_p.add_argument('--program', type=Path, default=DEFAULTS['program'])
    run_p.add_argument('--rapfi-dir', type=Path, default=Path('/mnt/c/engines/rapfi'))
    run_p.add_argument('--jsonl', type=Path, help=f"default {_rel(DEFAULTS['jsonl'])} ({_rel(SMOKE_DIR)} with --limit)")
    run_p.add_argument('--output', type=Path, help=f"default {_rel(DEFAULTS['output'])}")
    run_p.add_argument('--manifest-out', type=Path, help=f"default {_rel(DEFAULTS['manifest_out'])}")
    run_p.add_argument('--engine-command', type=shlex.split, help='tests only: another engine command (one quoted string)')
    run_p.add_argument('--allow-unregistered-engine', action='store_true', help='tests only')
    run_p.add_argument('--max-plies', type=int, help='tests only: truncate continuations (INVALID)')
    ref_p.add_argument('--policy-checkpoint', type=Path, default=ROOT / 'runs/h3_policy_64x4/best.pt')
    ref_p.add_argument('--value-checkpoint', type=Path, default=ROOT / 'runs/h6_value_head_v3/best.pt')
    ref_p.add_argument('--device', default='cuda')
    ref_p.add_argument('--jsonl', type=Path, help=f"default {_rel(DEFAULTS['reference_jsonl'])}")
    ref_p.add_argument('--output', type=Path, help=f"default {_rel(DEFAULTS['reference_output'])}")
    ref_p.add_argument('--allow-stand-in', action='store_true', help='tests only')
    return parser


def write_manifest(path: Path, result: dict, output: Path, jsonl: Path) -> None:
    manifest = {'format': f'{FORMAT}-manifest', 'identity_sha256': result['identity_sha256'],
                'code_git': result['identity']['git'], 'inputs_sha256': result['identity']['inputs'],
                'files_sha256': {_rel(jsonl): file_sha(jsonl)} if jsonl.exists() else {},
                'document_files_sha256': {_rel(output): file_sha(output)}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')


def resolve_paths(args) -> None:
    """Unset outputs: the registered paths, or for a ``--limit`` smoke run files under ``SMOKE_DIR`` only."""
    smoke = args.limit is not None
    names = {'jsonl': 'jsonl', 'output': 'output', 'manifest_out': 'manifest_out'} if args.command == 'run' else \
        {'jsonl': 'reference_jsonl', 'output': 'reference_output'}
    for attr, key in names.items():
        if getattr(args, attr) is None:
            setattr(args, attr, SMOKE_DIR / f'{args.command}_limit{args.limit}_{DEFAULTS[key].name}' if smoke
                    else DEFAULTS[key])
        elif smoke and Path(getattr(args, attr)).resolve() == DEFAULTS[key].resolve():
            raise ValueError(f'--limit: {attr} must not be the full run\'s {_rel(DEFAULTS[key])}')


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        resolve_paths(args)
    except ValueError as exc:
        parser.error(str(exc))
    for path in (args.output, getattr(args, 'manifest_out', None)):
        if path is not None and Path(path).exists():
            parser.error(f'{path} exists: a finished adjudication is never overwritten')
    try:
        result = run(args) if args.command == 'run' else reference(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    if args.command == 'run':
        write_manifest(args.manifest_out, result, args.output, Path(args.jsonl))
        brief = [{k: s[k] for k in ('comparison', 'observations', 'valid', 'invalid', 'decisive', 'better_share',
                                    'mean_delta', 'reading', 'reasons')} for s in result['comparisons']]
    else:
        brief = result['comparisons']
    print(json.dumps(brief, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
