"""H7-c search retarget: the cycle's self-play positions searched again at the decision's budget (docs/h7-c1.md).

The actor games stay as they are; this tool writes a separate artifact of raw search records, and
``scripts/h7c_labels.py --retarget`` turns them into policy targets with ``hybrid.h7c_targets.policy_target``
(rules in ``hybrid.h7c_retarget``). The agent is TB-V3 Resolve on fast5 with the self-play's own models (the
candidate factory's checks, once per worker) and only ``simulations`` / ``tactical_simulations`` scaled by the factor
(``{**TB_V2_VALUE, **budget(k)}``; at k = 1 exactly the frozen configuration); per game one agent per colour with the
actor's seeds ``derive_seed(game_seed, 'black' | 'white')``.

    run    cycle c >= 2: the decision of cycle c - 1 must be A (its ceiling is the cap) or A_COST_REVIEW with the
           committed cost review of that decision (its ``hard_cap_seconds`` is the cap, its factor and mode are the
           run's); the self-play must be one clean cycle (``h7c_labels.check_games``), the checkpoints its models,
           the engine / runtime / device its own. Cost gate: the run stops when the projected wall time exceeds the
           cap: elapsed + remaining rows at the steady rate (measured after the first ``--workers`` games, so
           start-up and the games that all started together are not in it), first after 20 more games and 256 more
           rows, then at 10%, 25% and 50% of the rows (resumable).
    check  G0, before any retarget is trusted: ``--factor 1`` must reproduce every actor record (route, move, visits,
           mean values within 1e-5, proven losses, E4-v2 step, target) and, with ``--policy-rows``, the v1 projection
           of the v2 rows must be the cycle's v1 rows; ``--mode both`` also requires the ``restored`` and the
           ``sequential`` replay to give the same records (G0b, at any factor). PASS or FAIL in the report.

    python -u scripts/h7c_retarget.py run --games runs/h7c/c2/selfplay_8802.jsonl \\
        --decision docs/h7c/c1/decision.json --cost-review docs/h7c/c1/cost_review.json \\
        --policy-checkpoint runs/h3_policy_64x4/best.pt --value-checkpoint runs/h6_value_head_v3/best.pt \\
        --device cuda --workers 6 --jsonl runs/h7c/c2/retarget_s200.jsonl --output docs/h7c/c2/retarget_s200.json

Resume: rerun the same command; finished games in ``--jsonl`` are skipped, another identity (self-play file, models,
factor, mode, decision, cost review, code commit) is refused. Records never go into the repository outside ``data/``
or ``runs/``; the output is a summary without moves. A dirty tree is refused (``--allow-dirty`` only for tests).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hybrid import h7c_retarget as hr  # noqa: E402
from hybrid import h7c_targets as targets  # noqa: E402
from hybrid.progress import Progress, run_chunks  # noqa: E402
from renju import BLACK, Game  # noqa: E402

_WORKER: dict = {}


class CostCapExceeded(RuntimeError):
    pass


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


# --- agents ------------------------------------------------------------------------------------------------------

def _init_worker(policy_checkpoint: str, value_checkpoint: str, device: str, factor: int) -> None:
    """The self-play worker's models and every check of the candidate factory, once per process."""
    from analysis import tb_v2_runtime, tb_v2_value, tb_v3, tb_v3_fast5, tb_v3_net, track_b_guard
    from hybrid.h4_policy import RootPolicy
    from hybrid.h6_value_leaf import ValueLeaf
    from rt30 import fast5

    if not fast5.installed():
        tb_v2_runtime.install()
        tb_v3_fast5.install()
    policy = RootPolicy(policy_checkpoint, threads=1)
    leaf = ValueLeaf(value_checkpoint, device=device, threads=1, expected_sha256=None)
    tb_v3.check_identity()
    tb_v3_net.check_models(policy, leaf, policy_checkpoint, value_checkpoint)
    track_b_guard.check_runtime_sources()
    tb_v3_fast5.check_sources()
    tb_v3_fast5.check_bound()
    if tb_v2_value.config_hash() != tb_v2_runtime.ENGINE_HASH:
        raise RuntimeError('analysis.tb_v2_value is not the frozen configuration')
    tb_v2_runtime.check_bound()
    config = {**tb_v2_value.TB_V2_VALUE, **hr.budget(factor)}
    if factor == 1 and config != tb_v2_value.TB_V2_VALUE:
        raise RuntimeError('factor 1 is not the frozen configuration')
    _WORKER.update(policy=policy, leaf=leaf, config=config, factor=factor)


def make_agent(seed: int):
    """``tb_v3_net._build`` with only the two tree budgets scaled (the worker's checks done)."""
    from analysis import tb_v2_runtime, tb_v2_value, tb_v3
    from analysis.mcts_v8_agent import MCTSV8Agent
    from analysis.tb_v2_value_e4v2 import E4v2AttackAgent

    w = _WORKER
    base = MCTSV8Agent(seed=seed, root_policy=w['policy'], leaf_value=w['leaf'], **w['config'],
                       **tb_v2_value.ENGINE_OPTIONS)
    base.name = tb_v2_runtime.NAME
    agent = E4v2AttackAgent(base, attack=True, config=tb_v3.ATTACK)
    agent.name = f"TB-V3 S{w['config']['simulations']} (retarget)"
    return agent


def search(agent, game: Game) -> dict:
    """One move of ``agent`` as the self-play records it (route, move, raw visits, proven losses, E4-v2 step)."""
    from run_mcts_v8_benchmark import _move_record

    started = perf_counter()
    move = agent.select_move(game)
    seconds = perf_counter() - started
    diag = agent.diagnostics
    record = _move_record(len(game.history), seconds, diag)
    visits = targets.raw_visits(diag)
    info = getattr(agent, 'attack_info', None) or {}
    return {'route': diag.v8_route, 'engine_move': [int(move[0]), int(move[1])], 'visits': visits,
            'proven_loss': targets.proven_losses(record), 'e4': {k: info[k] for k in hr.E4_KEYS if k in info},
            'tree_budget': diag.v8_tree_simulations if visits else None, 'seconds': round(seconds, 4)}


def _position(moves) -> Game:
    game = Game()
    for move in moves:
        game.play(*move)
    return game


def retarget_game(game: dict, mode: str, make=None) -> list[dict]:
    """The retarget records of one game's eligible plies (``hybrid.h7c_retarget``: restored or sequential)."""
    from hybrid.h6_labels import canonical_hash
    from run_mcts_v8_benchmark import derive_seed

    make = make or make_agent
    seeds = {'black': derive_seed(game['game_seed'], 'black'), 'white': derive_seed(game['game_seed'], 'white')}
    agents = {color: make(seed) for color, seed in seeds.items()}
    wanted = {p['ply'] for p in hr.eligible_plies(game)}
    rows = []

    def keep(p, record, restored):
        rows.append({'ply': p['ply'], 'color': p['color'], 'seed': seeds[p['color']],
                     'canonical_hash': canonical_hash(game['moves'][:p['ply']]), **record,
                     **({'cooldown_restored': restored} if mode == 'restored' else {})})

    if mode == 'restored':
        for p in game['plies']:
            if p['ply'] not in wanted:
                continue
            agent = agents[p['color']]
            agent.new_game()  # the only state between moves: the cooldown bit, set again from the actor's record
            restored = (p.get('e4') or {}).get('skip') == 'cooldown'
            if restored:
                agent.cooldown.exhausted(_position(game['moves'][:p['ply'] - 2]))
            keep(p, search(agent, _position(game['moves'][:p['ply']])), restored)
    elif mode == 'sequential':
        position = _position(game['opening'])
        for p in game['plies']:
            if p['color'] != ('black' if position.to_play == BLACK else 'white'):
                raise ValueError(f"game {game['game_index']} ply {p['ply']}: not the recorded mover")
            record = search(agents[p['color']], position)
            if p['ply'] in wanted:
                keep(p, record, False)
            position.play(*p['played'])
    else:
        raise ValueError(f'mode {mode!r}')
    return rows


def _chunk(tasks: list[dict]) -> list[dict]:
    """Per game ``rows`` (one mode: the run file the label step reads) or ``rows_<mode>`` (a check of both)."""
    out = []
    for t in tasks:
        result = {'game_index': t['game']['game_index']}
        for mode in t['modes']:
            result['rows' if len(t['modes']) == 1 else f'rows_{mode}'] = retarget_game(t['game'], mode)
        out.append(result)
    return out


def _rows(record: dict, mode: str) -> list[dict]:
    return record['rows'] if 'rows' in record else record[f'rows_{mode}']


# --- inputs ------------------------------------------------------------------------------------------------------

def _json(path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _under_data_or_runs(path: Path) -> bool:
    import h7c_labels as lab

    return lab._under_data_or_runs(Path(path))


def load_games(path: Path) -> tuple[list[dict], dict]:
    """The self-play games after the label step's own checks (clean generator, one cycle and model, complete games)
    and their identity (``hybrid.h7c_retarget.selfplay_identity``)."""
    import h7c_labels as lab

    games = lab.read_games([path])
    lab.check_games(games)
    return games, hr.selfplay_identity(games)


def check_engine(selfplay: dict, policy_checkpoint, value_checkpoint, device: str) -> None:
    """The retarget searches with the self-play's own models, engine, runtime and value device."""
    from analysis import tb_v3, tb_v3_fast5, tb_v3_net

    models = tb_v3_net.model_identity(policy_checkpoint, value_checkpoint)
    if models != selfplay['models']:
        raise ValueError("the checkpoints are not the models that played the self-play (the retarget uses the "
                         "cycle's parent pair)")
    if selfplay['engine_hash'] != tb_v3.ENGINE_HASH:
        raise ValueError('the self-play engine is not the frozen TB-V3')
    if device.split(':')[0] != selfplay['value_device'] or tb_v3_fast5.runtime_hash(device) != selfplay['runtime_hash']:
        raise ValueError(f"--device {device}: the self-play ran on {selfplay['value_device']} "
                         f"(runtime {selfplay['runtime_hash'][:12]})")


def approved_budget(decision: dict, decision_sha: str, cost_review: dict | None, cost_review_sha: str | None,
                    cycle: int) -> dict:
    """Factor, mode and cap of a run from the previous cycle's decision (and its cost review)."""
    if decision.get('format') != 'h7c-decision-v1' or decision.get('cycle') != cycle - 1:
        raise ValueError(f'the decision is not the decision of cycle {cycle - 1}')
    case, b = decision.get('case'), decision.get('budget') or {}
    if case not in ('A', 'A_COST_REVIEW') or not b.get('helps'):
        raise ValueError(f'decision case {case!r}: the retarget runs only for case A or A_COST_REVIEW')
    factor = b['factor']
    if b.get('budget') != hr.budget(factor):
        raise ValueError('the decision budget is not 50k / 100k')
    if case == 'A':
        if cost_review is not None:
            raise ValueError('case A needs no cost review (its own ceiling is the cap)')
        return {'factor': factor, 'mode': 'restored', 'cap_seconds': b['ceiling_seconds'], 'cost_review_sha256': None}
    if cost_review is None:
        raise ValueError('case A_COST_REVIEW: the committed cost review of this decision is required (--cost-review)')
    approved = cost_review.get('approved') or {}
    if cost_review.get('format') != 'h7c-cost-review-v1' or cost_review.get('decision_sha256') != decision_sha or \
            cost_review.get('applies_to_cycle') != cycle or cost_review.get('case') != case:
        raise ValueError('the cost review is not the review of this decision for this cycle')
    if approved.get('factor') != factor or approved.get('budget') != hr.budget(factor) or \
            approved.get('mode') not in hr.MODES:
        raise ValueError('the cost review approves another budget than the decision')
    cap = cost_review.get('hard_cap_seconds')
    if not isinstance(cap, (int, float)) or cap <= 0:
        raise ValueError('the cost review has no hard cap')
    return {'factor': factor, 'mode': approved['mode'], 'cap_seconds': cap, 'cost_review_sha256': cost_review_sha}


# --- the pool ----------------------------------------------------------------------------------------------------

def _execute(args, games: list[dict], identity: dict, modes: tuple, factor: int, log, gate=None,
             validate: bool = True) -> dict[int, dict]:
    """Every game's records (resumed from ``args.jsonl``), each checked as it arrives (``validate``: a run; a check
    counts differences instead); ``gate(elapsed, done, rows_total)`` may stop it (``done``: games, rows and the
    workers' search seconds so far)."""
    import h7c_selfplay as sp

    finished = sp.load_finished(args.jsonl, {'identity': identity}, log=log)
    todo = [g for g in games if g['game_index'] not in finished]
    rows_total = sum(len(hr.eligible_plies(g)) for g in todo)
    log(f'[h7c-retarget] factor {factor} ({hr.budget(factor)}), modes {"/".join(modes)}: {len(games)} games, '
        f'{len(finished)} already done, {rows_total:,} rows to search, workers {args.workers}', flush=True)
    progress = Progress('h7c-retarget', rows_total, 'rows')
    started, done = perf_counter(), {'games': 0, 'rows': 0, 'search_seconds': 0.0}
    if todo:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        chunks = [[{'game': g, 'modes': modes}] for g in todo]
        with args.jsonl.open('a', encoding='utf-8') as out, ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init_worker,
                initargs=(str(args.policy_checkpoint), str(args.value_checkpoint), args.device, factor)) as pool:
            by_index = {g['game_index']: g for g in todo}

            def keep(_, results):
                for result in results:
                    g = by_index[result['game_index']]
                    for mode in modes:  # fail at once, not after hours: the same checks as the label step
                        mine = _rows(result, mode)
                        if [r['ply'] for r in mine] != [p['ply'] for p in hr.eligible_plies(g)]:
                            raise ValueError(f"game {g['game_index']}: the {mode} records are not its visits plies")
                        for p, row in zip(hr.eligible_plies(g), mine):
                            if validate:
                                hr.check_row(g, p, row)
                    record = {'identity': identity, **result}
                    out.write(json.dumps(record, separators=(',', ':')) + '\n')
                    out.flush()
                    finished[result['game_index']] = record
                    done['games'] += 1
                    done['rows'] += len(_rows(result, modes[0]))
                    done['search_seconds'] += sum(r['seconds'] for r in _rows(result, modes[0]))
                if gate is not None:
                    gate(perf_counter() - started, dict(done), rows_total)

            run_chunks(pool, _chunk, chunks, progress, lambda c: len(hr.eligible_plies(c[0]['game'])),
                       unit_name='games', on_result=keep)
    return finished


def run(args, log=print) -> dict:
    from run_mcts_v8_benchmark import _git_commit, _git_dirty

    git = {'commit': _git_commit(), 'dirty': _git_dirty()}
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty (or git is unavailable): retarget data must name its code commit')
    if not _under_data_or_runs(args.jsonl):
        raise ValueError(f'{args.jsonl}: retarget records go under data/ or runs/')
    games, selfplay = load_games(args.games)
    cycle = selfplay['cycle']
    if cycle < 2:
        raise ValueError('the retarget starts at cycle 2 (the decision of the previous cycle chooses it)')
    decision_sha = file_sha256(args.decision)
    review = _json(args.cost_review) if args.cost_review else None
    plan = approved_budget(_json(args.decision), decision_sha, review,
                           file_sha256(args.cost_review) if args.cost_review else None, cycle)
    check_engine(selfplay, args.policy_checkpoint, args.value_checkpoint, args.device)
    factor, mode = plan['factor'], plan['mode']
    identity = {'format': hr.FORMAT, 'purpose': 'retarget', 'factor': factor, 'budget': hr.budget(factor),
                'mode': mode, 'selfplay_jsonl_sha256': file_sha256(args.games), 'selfplay': selfplay,
                'decision_sha256': decision_sha, 'decision_inputs_sha256': _json(args.decision)['inputs_sha256'],
                'cost_review_sha256': plan['cost_review_sha256'], 'coverage': hr.coverage(games), 'git': git}
    gates, checked, window = [], set(), {}

    def gate(elapsed, done, rows_total):
        if not window:  # the steady rate starts once the first `workers` games are in
            if done['games'] >= args.workers:
                window.update(elapsed=elapsed, games=done['games'], rows=done['rows'])
            return
        point = hr.cost_gate(elapsed=elapsed, games_done=done['games'], rows_done=done['rows'], rows_total=rows_total,
                             cap_seconds=plan['cap_seconds'], checked=checked, window=dict(window),
                             search_seconds=done['search_seconds'], workers=args.workers)
        if point is None:
            return
        gates.append(point)
        log(f"[h7c-retarget] cost gate {point['point']}: projected {point['projected_seconds']:,.0f} s "
            f"(pure search {point['search_projection_seconds']:,.0f} s; cap {point['cap_seconds']:,.1f} s) after "
            f"{done['rows']:,} of {rows_total:,} rows", flush=True)
        if point['exceeded']:
            raise CostCapExceeded(f"projected {point['projected_seconds']:,.0f} s exceeds the cap "
                                  f"{point['cap_seconds']:,.1f} s: stopped (finished games are kept for a resume "
                                  f"under a new approval)")

    started = perf_counter()
    finished = _execute(args, games, identity, (mode,), factor, log, gate)
    records = [{'identity': r['identity'], 'game_index': i, 'rows': _rows(r, mode)} for i, r in finished.items()]
    rows = hr.index_rows(games, records, identity)
    summary = summarize(games, rows)
    selfplay_wall = None
    if args.selfplay_summary:
        selfplay_wall = _json(args.selfplay_summary).get('wall_seconds')
    out = {'format': f'{hr.FORMAT}-summary', 'identity': identity, 'jsonl_sha256': file_sha256(args.jsonl),
           'cap_seconds': plan['cap_seconds'], 'gates': gates,
           'wall_seconds_this_session': round(perf_counter() - started, 1),
           'coverage': {**identity['coverage'], 'completed': len(rows), 'missing': 0, 'duplicate': 0, 'extra': 0},
           'dynamic_ceiling_diagnostic': None if selfplay_wall is None else
           {'selfplay_wall_seconds': selfplay_wall, 'ceiling_1_25x': round(1.25 * 2 * selfplay_wall, 1)},
           'summary': summary}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    return out


def summarize(games: list[dict], rows: dict) -> dict:
    """Coverage-checked rows: the targets they give, the proven-loss union, how often the move changed, time."""
    by_game = {g['game_index']: g for g in games}
    kinds, losses, changed, seconds, children = {}, {'same': 0, 'actor_only': 0, 'retarget_only': 0}, 0, [], []
    for (gi, ply), row in rows.items():
        p = next(x for x in by_game[gi]['plies'] if x['ply'] == ply)
        target, _, counts = hr.retarget_target(p, row)
        kinds[target['kind']] = kinds.get(target['kind'], 0) + 1
        for k in losses:
            losses[k] += counts[k]
        changed += list(row['engine_move']) != list(p['engine_move'])
        seconds.append(row['seconds'])
        children.append(sum(1 for v in row['visits'] if v[2] > 0))
    n = len(rows)
    return {'rows': n, 'target_kinds': kinds, 'proven_loss': losses,
            'move_changed_share': round(changed / n, 4) if n else None,
            'visited_children_mean': round(sum(children) / n, 3) if n else None,
            'seconds_mean': round(sum(seconds) / n, 4) if n else None,
            'cooldown_restored': sum(1 for r in rows.values() if r.get('cooldown_restored'))}


def _field_diff(a: dict, b: dict) -> list[str]:
    skip = {'seconds', 'cooldown_restored'}
    return sorted(k for k in set(a) | set(b) if k not in skip and a.get(k) != b.get(k))


def check(args, log=print) -> dict:
    """G0 / G0b (module docstring); PASS only with complete coverage and no difference at all."""
    from run_mcts_v8_benchmark import _git_commit, _git_dirty

    git = {'commit': _git_commit(), 'dirty': _git_dirty()}
    if git['dirty'] is not False and not args.allow_dirty:
        raise ValueError('the tree is dirty (or git is unavailable): G0 evidence must name its code commit')
    if not _under_data_or_runs(args.jsonl):
        raise ValueError(f'{args.jsonl}: retarget records go under data/ or runs/')
    games, selfplay = load_games(args.games)
    if args.games_limit:
        games = sorted(games, key=lambda g: g['game_index'])[:args.games_limit]
    check_engine(selfplay, args.policy_checkpoint, args.value_checkpoint, args.device)
    modes = hr.MODES if args.mode == 'both' else (args.mode,)
    identity = {'format': hr.FORMAT, 'purpose': 'g0', 'factor': args.factor, 'budget': hr.budget(args.factor),
                'modes': list(modes), 'selfplay_jsonl_sha256': file_sha256(args.games),
                'selfplay': selfplay, 'coverage': hr.coverage(games), 'git': git}
    finished = _execute(args, games, identity, modes, args.factor, log, validate=False)
    by_game = {g['game_index']: g for g in games}
    report = {'actor': {m: {} for m in modes} if args.factor == 1 else None,
              'modes_differ': {} if len(modes) > 1 else None, 'examples': []}
    indexed = {}
    for mode in modes:
        records = [{'identity': identity, 'game_index': i, 'rows': _rows(r, mode)} for i, r in finished.items()]
        indexed[mode] = hr.index_rows(games, records, identity, check=False)
    for mode in modes:
        for (gi, ply), row in sorted(indexed[mode].items()):
            p = next(x for x in by_game[gi]['plies'] if x['ply'] == ply)
            if args.factor == 1:
                for field in hr.compare_to_actor(p, row):
                    report['actor'][mode][field] = report['actor'][mode].get(field, 0) + 1
                    if len(report['examples']) < 20:
                        report['examples'].append({'game': gi, 'ply': ply, 'mode': mode, 'field': field})
    if len(modes) > 1:
        for key, row in sorted(indexed[modes[0]].items()):
            for field in _field_diff(row, indexed[modes[1]][key]):
                report['modes_differ'][field] = report['modes_differ'].get(field, 0) + 1
                if len(report['examples']) < 20:
                    report['examples'].append({'game': key[0], 'ply': key[1], 'mode': 'both', 'field': field})
    projection = None
    if args.policy_rows:
        projection = compare_projection(games, indexed[modes[0]], identity, args.policy_rows)
    failures = []
    if args.factor == 1 and any(report['actor'][m] for m in modes):
        failures.append('actor records not reproduced')
    if report['modes_differ']:
        failures.append('restored and sequential differ')
    if projection is not None and (projection['differ'] or projection['missing'] or not projection['compared']):
        failures.append('v1 projection differs from the cycle rows')
    out = {'format': f'{hr.FORMAT}-check', 'identity': identity, 'jsonl_sha256': file_sha256(args.jsonl),
           'rows': len(indexed[modes[0]]), 'games': len(games), **report, 'projection': projection,
           'verdict': 'FAIL' if failures else 'PASS', 'failures': failures}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    log(f"[h7c-retarget] check: {out['verdict']} {failures}", flush=True)
    return out


def compare_projection(games: list[dict], rows: dict, identity: dict, policy_rows: Path) -> dict:
    """The v1 projection of the v2 rows (``h7c_labels.policy_rows`` with these records) against the cycle's v1 rows,
    by canonical JSON without the label run's provenance, mask and flags (masks come after, from the position)."""
    import gzip

    import h7c_labels as lab
    from hybrid.h6_labels import split_for

    original = {}
    with gzip.open(policy_rows, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                original[(row['game_id'], row['ply'])] = row
    drop = set(hr.PROVENANCE_FIELDS) | {'mask', 'flags'}
    canon = lambda r: json.dumps({k: v for k, v in r.items() if k not in drop}, sort_keys=True)  # noqa: E731
    info = {'sha256': None, 'identity_sha256': hr.sha256_json(identity)}
    out = {'compared': 0, 'differ': 0, 'missing': 0, 'not_in_cycle_rows': 0}
    for g in games:
        gid = lab.game_id(g)
        split = split_for(gid, fractions=lab.SPLIT_FRACTIONS, salt=lab.SPLIT_SALT)
        provenance = {'engine_config_hash': g['engine_hash'], 'model_hash': g['model_hash']}  # as h7c_labels.build
        mine = lab.policy_rows(g, split, provenance, retarget={'rows': rows, 'info': info})
        theirs = {k: v for k, v in original.items() if k[0] == gid}
        if not theirs:
            out['not_in_cycle_rows'] += 1  # a game the label step dropped (probe)
            continue
        made = {(gid, r['ply']): hr.v1_projection(r) for r in mine}
        for key, row in theirs.items():
            if key not in made:
                out['missing'] += 1
            elif canon(made[key]) != canon(row):
                out['differ'] += 1
            out['compared'] += key in made
        out['missing'] += sum(1 for key in made if key not in theirs)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'check'):
        p = sub.add_parser(name)
        p.add_argument('--games', type=Path, required=True, help='the h7c_selfplay JSONL file of the cycle')
        p.add_argument('--policy-checkpoint', type=Path, required=True)
        p.add_argument('--value-checkpoint', type=Path, required=True)
        p.add_argument('--device', default='cuda')
        p.add_argument('--workers', type=int, default=6)
        p.add_argument('--jsonl', type=Path, required=True, help='the resumable record file (data/ or runs/)')
        p.add_argument('--output', type=Path)
    run_p, check_p = sub.choices['run'], sub.choices['check']
    run_p.add_argument('--decision', type=Path, required=True, help="the previous cycle's decision.json")
    run_p.add_argument('--cost-review', type=Path, help="that decision's cost review (case A_COST_REVIEW)")
    run_p.add_argument('--selfplay-summary', type=Path, help='diagnostic only: 1.25 x 2 x its wall time')
    for p in (run_p, check_p):
        p.add_argument('--allow-dirty', action='store_true', help='tests only')
    check_p.add_argument('--factor', type=int, default=1)
    check_p.add_argument('--mode', choices=(*hr.MODES, 'both'), default='restored')
    check_p.add_argument('--games-limit', type=int, help='the first N games by index (complete games)')
    check_p.add_argument('--policy-rows', type=Path, help="the cycle's v1 policy rows (the projection check)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out = run(args) if args.command == 'run' else check(args)
    except CostCapExceeded as exc:
        print(f'[h7c-retarget] STOP: {exc}', flush=True)
        return 2
    if args.command == 'run':
        print(json.dumps({k: out[k] for k in ('coverage', 'cap_seconds', 'gates', 'summary')}, indent=1), flush=True)
        return 0
    return 0 if out['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
