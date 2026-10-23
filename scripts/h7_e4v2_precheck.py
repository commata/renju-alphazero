"""H7-E4-v2 cost pre-check on COSTTEST: is the frozen candidate's total move time within the arena cost gate?

Fixed before any measurement (docs/h6-value.md, "E4-v2 후보 동결" and its approval):
    positions   ``docs/h7/h7_e4v2_costtest_positions.json`` (94 E2 holdout games, 2,513 positions), re-checked here:
                no D4 board of the web corpus or of COSTDEV (``h7_e4v2_costtest.build`` rule)
    order       games in the file's order (sha256 of the game key), each game's positions in ply order
    candidate   one ``TB-V2-Value+E4v2`` wrapper per game: its cooldown bit is the only state kept from one
                position to the next. At every position the base is a new canonical TB-V2-Value
                (``track_b_guard.make_canonical_agent``, seed ``derive_seed(SEED, 'h7-e4v2-precheck', id)``, the
                value cache cleared): no tree, cache or search state crosses positions
    timing      inside the candidate's own call: B = the base's ``select_move`` (its route decides the E4-v2 step,
                never ``source_recorded``), E = the whole call (base + cooldown + opponent-VCF gate + attack).
                Model loading, the native build and agent construction are outside both
    statistic   ratio = sum E / sum B; 95% interval by bootstrap over games (``h7_e4_precheck.ratio_ci``)
    rule        lower bound > 1.5 -> PRECHECK_FAIL (E4-v2 REJECT, no arena); otherwise PROCEED to the arena.
                PROCEED is not a cost pass: with a point estimate above 1.5 the cost is still open.
Resume: whole games only (the cooldown runs along a game); a game is appended to ``--jsonl`` when it ends.

    python scripts/h7_e4v2_precheck.py --positions docs/h7/h7_e4v2_costtest_positions.json \\
        --corpus docs/h7/h7_e4_web_corpus.json --costdev docs/h7/h7_timing_positions.json --device cuda \\
        --jsonl runs/h7_e4v2/precheck.jsonl --output docs/h7/h7_e4v2_precheck.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import h7_e4_precheck as v1  # noqa: E402
from scripts import h7_e4_web_corpus as web  # noqa: E402

FORMAT = 'h7-e4v2-precheck-v1'
SEED = 8440


def games_in_order(costtest: dict) -> list[tuple[str, list[dict]]]:
    by_game: dict[str, list[dict]] = {}
    for p in costtest['positions']:
        by_game.setdefault(p['id'].split('|')[1], []).append(p)
    if list(by_game) != costtest['games'] or set(by_game) != set(costtest['games']):
        raise ValueError('the positions are not grouped by the games of the file, in its order')
    for key, ps in by_game.items():
        plies = [int(p['id'].rsplit('|', 1)[1]) for p in ps]
        if plies != sorted(plies):
            raise ValueError(f'{key}: positions not in ply order')
    return [(key, by_game[key]) for key in costtest['games']]


def check_disjoint(costtest: dict, corpus: dict, costdev: list[dict]) -> None:
    from hybrid.h6_labels import canonical_hash

    banned = {canonical_hash(web.history(corpus, p)) for p in corpus['positions']}
    banned |= {canonical_hash(p['moves']) for p in costdev}
    bad = [p['id'] for p in costtest['positions'] if canonical_hash(p['moves']) in banned]
    if bad:
        raise ValueError(f'COSTTEST has web / COSTDEV boards: {bad[:5]}')


def measure_game(key: str, positions: list[dict], make_base, leaf) -> list[dict]:
    """One game in order: one candidate wrapper (one cooldown), a new canonical base at every position."""
    from analysis.tb_v2_value_e4v2 import E4v2AttackAgent
    from renju import Game
    from scripts.run_mcts_v8_benchmark import derive_seed

    agent, rows = None, []
    for p in positions:
        game = Game()
        for m in p['moves']:
            game.play(*m)
        if hasattr(leaf, '_cache'):
            leaf._cache.clear()  # cold per position, as in E4-v1's pre-check
        base = make_base(derive_seed(SEED, 'h7-e4v2-precheck', p['id']))
        if agent is None:
            agent = E4v2AttackAgent(base)
        agent.base = base  # the cooldown stays, everything else is new
        timed, inner = {}, base.select_move

        def timed_base(g, inner=inner):
            started = perf_counter()
            move = inner(g)
            timed['base'] = perf_counter() - started
            return move

        base.select_move = timed_base
        started = perf_counter()
        move = agent.select_move(game)
        total = perf_counter() - started
        info = agent.attack_info
        rows.append({'id': p['id'], 'game': v1.game_of(p['id']), 'b_seconds': round(timed['base'], 4),
                     'e_seconds': round(total, 4), 'attack_seconds': round(total - timed['base'], 4),
                     'route': info.get('base_route'), 'move': list(move), 'base_move': info.get('base_move'),
                     'attacked': bool(info.get('ran')), 'step': {k: v for k, v in info.items()
                                                                if k not in ('base_route', 'base_move')}})
    return rows


def report(rows: list[dict]) -> dict:
    out = v1.report(rows)
    skips = {}
    for r in rows:
        skip = r['step'].get('skip')
        if skip:
            skips[skip] = skips.get(skip, 0) + 1
    out['skips'] = skips
    return out


def main(argv=None, factories=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--costdev', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--policy-checkpoint', default=str(ROOT / 'runs/h3_policy_64x4/best.pt'))
    parser.add_argument('--value-checkpoint', default=str(ROOT / 'runs/h6_value_head_v3/best.pt'))
    parser.add_argument('--limit-games', type=int, help='smoke tests: first N games only')
    parser.add_argument('--jsonl', type=Path, help='one line per finished game (resume by whole games)')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from analysis import tb_v2_runtime as rt
    from analysis import tb_v2_value_e4v2 as cand
    from hybrid.progress import Progress
    from scripts.h7_timing import machine, restore_fast_mode
    from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256

    costtest = json.loads(args.positions.read_text(encoding='utf-8'))
    corpus = web.load(args.corpus)
    if costtest.get('format') != 'h7-e4v2-costtest-v1' or costtest.get('corpus_sha256') != file_sha256(args.corpus) \
            or costtest.get('costdev_sha256') != file_sha256(args.costdev):
        parser.error(f'{args.positions} is not the COSTTEST of this web corpus and COSTDEV')
    try:
        check_disjoint(costtest, corpus, json.loads(args.costdev.read_text(encoding='utf-8'))['positions'])
        games = games_in_order(costtest)[:args.limit_games]
    except ValueError as exc:
        parser.error(str(exc))
    fingerprint = {'candidate_hash': cand.candidate_hash(), 'positions_sha256': file_sha256(args.positions),
                   'device': rt.device_kind(args.device)}
    done = {}
    if args.jsonl is not None and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line: that game is measured again
            if row.get('fingerprint') != fingerprint:
                parser.error(f'{args.jsonl} has a game of another candidate / COSTTEST / device; use another file')
            done[row['game']] = row['rows']
    todo = [(k, ps) for k, ps in games if k not in done]
    total = sum(len(ps) for _, ps in todo)
    print(f'[h7-e4v2-precheck] {len(games)} games / {sum(len(ps) for _, ps in games)} positions, '
          f'{len(games) - len(todo)} games done, device {args.device}', flush=True)
    progress = Progress('h7-e4v2-precheck', total, 'positions')
    with restore_fast_mode():
        rt.install()
        cand.check_sources()
        if factories is not None:  # tests: stand-in models, the frozen factory without its model checks
            policy, leaf = factories['policy'], factories['leaf']
            make_base = (lambda seed: rt.make_agent(policy, leaf, seed=seed))
        else:
            from analysis.track_b_guard import make_canonical_agent
            from hybrid.h4_policy import RootPolicy
            from hybrid.h6_value_leaf import ValueLeaf
            policy, leaf = RootPolicy(args.policy_checkpoint), ValueLeaf(args.value_checkpoint, device=args.device)
            make_base = (lambda seed: make_canonical_agent(policy, leaf, seed, args.policy_checkpoint,
                                                           args.value_checkpoint))
        for key, ps in todo:
            rows = measure_game(key, ps, make_base, leaf)
            done[key] = rows
            if args.jsonl is not None:
                args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.jsonl.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps({'game': key, 'fingerprint': fingerprint, 'rows': rows}) + '\n')
            progress.update(len(ps), extra=key)
    rows = [r for k, _ in games for r in done[k]]
    result = v1.ratio_ci(rows)
    result['interpretation'] = ('PRECHECK_FAIL: REJECT, no arena' if result['verdict'] == 'PRECHECK_FAIL' else
                                'PROCEED (not a cost pass): ' + ('the point estimate is within 1.5'
                                                                 if result['ratio'] <= v1.RATIO_MAX else
                                                                 'the point estimate is above 1.5, the cost is open'))
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'positions_sha256': file_sha256(args.positions),
               'corpus_sha256': file_sha256(args.corpus), 'costdev_sha256': file_sha256(args.costdev),
               'candidate_hash': cand.candidate_hash(), 'fingerprint': fingerprint,
               'runtime': cand.provenance(args.device) if factories is None else None, 'machine': machine(),
               'rule': {'ratio_max': v1.RATIO_MAX, 'bootstrap': v1.BOOTSTRAP, 'seed': v1.SEED, 'cluster': 'game',
                        'order': 'file order, ply order', 'state': 'one cooldown per game'},
               'restricted': args.limit_games is not None, **result, 'report': report(rows), 'rows': rows}
    print(json.dumps({k: payload[k] for k in ('ratio', 'ci95', 'games', 'positions', 'verdict', 'interpretation',
                                              'report')}, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
