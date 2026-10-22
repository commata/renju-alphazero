"""H7-E4-v2 DEV: the eight configurations on the web DEV positions and on COSTDEV, and the fixed selection.

Two position sets, no HOLDOUT (``h7_e4_dev.dev_only`` on every web position read):
    DEV       web DEV positions (``h7_e4_web_corpus``): VCT2-only punishment, soundness, shallow regressions,
              as in ``h7_e4_dev.py`` (the E4 move is the attack's proven WIN when the step runs, else the
              recorded base move of ``h7_e4_baseline.py`` v2)
    COSTDEV   the E4-v1 pre-check positions (``h7_e4_precheck.json``): their measured base seconds, and the
              attack cost from VCF nodes and calls through a cost model fitted on that pre-check's own attacks
              (``fit_cost_model``: seconds ~ nodes + calls + 1, its coefficients, R^2 and ranges are recorded),
              times the killer and gate corrections of ``h7_e4v2_calibrate.py`` (``--calibration``)
Per position the G1 gate and, unless it stops the attack, the attack for each (budget, killer) pair are
computed once; cooldown is then replayed per configuration along each game with ``e4v2_attack.Cooldown``
(the AI moves of a web game are its consecutive main positions; COSTDEV games by position id).
Eligible positions: DEV as in E4-v1 (a proven win whose recorded base route is outside the routes is not
attacked; elsewhere the route is assumed eligible); COSTDEV where E4-v1 attacked (its base route qualified).

Selection (fixed before DEV, docs/h6-value.md "E4-v2 설계 확정"):
    1-3  no false proof, no contradiction, no shallow regression
    4    COSTDEV cost ratio <= 1.35 (model estimate: 1 + sum of attack and gate seconds / sum of base seconds)
    5-6  most DEV VCT2-only positions punished; within one of the best is a tie
    7    the tie goes to the lowest COSTDEV ratio (then the grid order)
No configuration passing 1-4: E4-v2 stops at DEV (REJECT).

    python scripts/h7_e4v2_dev.py --corpus docs/h7/h7_e4_web_corpus.json --truth docs/h7/h7_e4_truth.json \\
        --baseline docs/h7/h7_e4_baseline_v2.json --precheck docs/h7/h7_e4_precheck.json \\
        --positions docs/h7/h7_timing_positions.json --workers 12 --fast fast4 \\
        --jsonl runs/h7_e4v2/dev.jsonl --output docs/h7/h7_e4v2_dev.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import h7_e4_baseline as base  # noqa: E402
from scripts import h7_e4_truth as truth  # noqa: E402
from scripts import h7_e4_web_corpus as web  # noqa: E402
from scripts.h7_e4_dev import SHALLOW, dev_only  # noqa: E402

FORMAT = 'h7-e4v2-dev-v1'
COST_MAX = 1.35
TIE = 1


# -- cost model --------------------------------------------------------------------------------------------

def fit_cost_model(rows: list[dict]) -> dict:
    """Least squares seconds = a * nodes + b * calls + c over the attacked pre-check rows (no numpy needed)."""
    xs = [(r['attack_nodes'], r['attack_calls'], 1.0) for r in rows if r['attacked']]
    ys = [r['attack_seconds'] for r in rows if r['attacked']]
    m = [[sum(x[i] * x[j] for x in xs) for j in range(3)] for i in range(3)]
    v = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(3)]
    for col in range(3):  # Gauss-Jordan on the 3 x 3 normal equations
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        m[col], m[pivot], v[col], v[pivot] = m[pivot], m[col], v[pivot], v[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
                v[r] -= f * v[col]
    a, b, c = (v[i] / m[i][i] for i in range(3))
    pred = [a * x[0] + b * x[1] + c for x in xs]
    mean = sum(ys) / len(ys)
    r2 = 1 - sum((y - p) ** 2 for y, p in zip(ys, pred)) / sum((y - mean) ** 2 for y in ys)
    return {'seconds_per_node': a, 'seconds_per_call': b, 'seconds_per_attack': c, 'r2': round(r2, 4),
            'n': len(xs), 'sum_ratio': round(sum(pred) / sum(ys), 4),
            'nodes_range': [min(x[0] for x in xs), max(x[0] for x in xs)],
            'calls_range': [min(x[1] for x in xs), max(x[1] for x in xs)]}


def model_seconds(model: dict, nodes: int, calls: int, attack: bool) -> float:
    """Seconds for a search of ``nodes`` / ``calls``; the per-attack constant only for an attack (not the gate)."""
    return model['seconds_per_node'] * nodes + model['seconds_per_call'] * calls + (
        model['seconds_per_attack'] if attack else 0.0)


def step_seconds(model: dict, correction: dict, step: dict, killer: bool) -> float:
    """Model seconds of one step: its gate (if computed) and its attack (if run), with the calibration's
    relative corrections (``h7_e4v2_calibrate.py``: 'gate', 'killer'; 1.0 when not given)."""
    seconds = 0.0
    if 'gate' in step:
        seconds += correction.get('gate', 1.0) * model_seconds(model, step['gate']['nodes'], step['gate']['calls'], False)
    if step['ran']:
        seconds += (correction.get('killer', 1.0) if killer else 1.0) * model_seconds(
            model, step['nodes'], step['calls'], True)
    return seconds


# -- work --------------------------------------------------------------------------------------------------

def _init(fast: str) -> None:
    if fast != 'off':
        from hybrid.h6_fast import install
        install(fast)


def position_task(task: dict) -> dict:
    """G1 and, unless it stops the attack, the attack for every (budget, killer) pair; DEV WINs checked."""
    from analysis import e4v2_attack as v2
    from renju import Game

    game = Game()
    for m in task['moves']:
        game.play(*m)
    gate = v2.opponent_vcf(game)
    row = {'set': task['set'], 'id': task['id'], 'fingerprint': task['fingerprint'], 'gate': gate, 'attacks': {}}
    if gate['status'] == 'WIN':
        return row
    for budget in v2.BUDGETS:
        for killer in (False, True):
            move, info = v2.attack(game, budget, killer)
            if move is not None and task['set'] == 'dev':
                info['check'] = base.check_move(task['moves'], move, task['check_budget'])
            row['attacks'][f"{budget}/{'killer' if killer else 'plain'}"] = info
    return row


def _chunk(tasks):
    return [position_task(t) for t in tasks]


# -- replay and scores -------------------------------------------------------------------------------------

def replay(config: dict, sequence: list[dict], rows: dict) -> dict:
    """Per position of one game (in order): whether the step ran and its attack record (``Cooldown`` replayed).

    ``sequence`` items: {'id', 'moves', 'eligible'}; ``rows`` the computed position rows by id."""
    from analysis.e4v2_attack import Cooldown

    cooldown, out = Cooldown(), {}
    pair = f"{config['budget']}/{'killer' if config['killer'] else 'plain'}"
    for item in sequence:
        game = SimpleNamespace(history=[tuple(m) for m in item['moves']])
        blocked = config['cooldown'] and cooldown.blocks(game)  # consumed at every own move, as in ``decide``
        if not item['eligible']:
            out[item['id']] = {'ran': False, 'skip': 'route'}
            continue
        if blocked:
            out[item['id']] = {'ran': False, 'skip': 'cooldown'}
            continue
        row = rows[item['id']]
        if row['gate']['status'] == 'WIN':
            out[item['id']] = {'ran': False, 'skip': 'opponent_vcf', 'gate': row['gate']}
            continue
        info = row['attacks'][pair]
        if config['cooldown'] and info['move'] is None and info['exhausted']:
            cooldown.exhausted(game)
        out[item['id']] = {'ran': True, 'gate': row['gate'], **info}
    return out


def dev_score(config: dict, positions: list[dict], boards: dict, baseline: dict, steps: dict, model: dict,
              correction: dict | None = None) -> dict:
    out = {'punish': 0, 'targets': 0, 'punished': [], 'missed': [], 'unclear': [], 'shallow_regressions': [],
           'false_proofs': [], 'contradictions': [], 'unverified': [], 'ran': 0, 'skips': {}, 'switched': 0}
    seconds = 0.0
    for p in positions:
        dev_only(p)
        step = steps[p['id']]
        if not step['ran']:
            out['skips'][step['skip']] = out['skips'].get(step['skip'], 0) + 1
        seconds += step_seconds(model, correction or {}, step, config['killer'])
        win = step['ran'] and step['move'] is not None
        if step['ran']:
            out['ran'] += 1
        if win:
            outcome = step['check']['outcome']
            if outcome == 'MISS':
                out['false_proofs'].append(p['id'])
            elif outcome == 'UNCLEAR':
                out['unverified'].append(p['id'])
            if boards[p['key']]['class'] == 'NOT_FOUND':
                out['contradictions'].append(p['id'])
        b = baseline.get(p['id'])
        if b is None:
            continue
        grade = boards[p['key']]['grade']
        e_outcome = b['b']['outcome']
        if win and step['move'] != b['b']['move']:
            out['switched'] += 1
            e_outcome = step['check']['outcome']
        if grade == truth.TARGET:
            out['targets'] += 1
            out[{'PUNISH': 'punished', 'MISS': 'missed'}.get(e_outcome, 'unclear')].append(p['id'])
            out['punish'] += e_outcome == 'PUNISH'
        elif grade in SHALLOW and b['b']['outcome'] == 'PUNISH' and e_outcome != 'PUNISH':
            out['shallow_regressions'].append(p['id'])
    out['model_seconds_per_position'] = round(seconds / len(positions), 4) if positions else None
    return out


def costdev_score(config: dict, steps: dict, pre_rows: list[dict], model: dict, correction: dict | None = None) -> dict:
    base_seconds = sum(r['b_seconds'] for r in pre_rows)
    extra, ran, switched, skips = 0.0, 0, 0, {}
    for r in pre_rows:
        step = steps[r['id']]
        extra += step_seconds(model, correction or {}, step, config['killer'])
        if step['ran']:
            ran += 1
            switched += step['move'] is not None and step['move'] != r['base_move']
        else:
            skips[step['skip']] = skips.get(step['skip'], 0) + 1
    return {'ratio': round(1 + extra / base_seconds, 4), 'ran': ran, 'switched': switched, 'skips': skips,
            'positions': len(pre_rows)}


def select(scores: list[dict]) -> dict:
    """The fixed rule: sound and COSTDEV <= 1.35, the most punished (ties within 1), then the cheapest."""
    order = {s['config']['name']: i for i, s in enumerate(scores)}
    passing = [s for s in scores if s['eligible'] and s['costdev']['ratio'] <= COST_MAX]
    if not passing:
        return {'selected': None, 'verdict': 'REJECT_AT_DEV',
                'reason': f'no configuration is sound with a COSTDEV ratio <= {COST_MAX}'}
    best = max(s['dev']['punish'] for s in passing)
    tied = [s for s in passing if s['dev']['punish'] >= best - TIE]
    chosen = min(tied, key=lambda s: (s['costdev']['ratio'], order[s['config']['name']]))
    return {'selected': chosen['config'], 'verdict': 'SELECTED', 'best_punish': best,
            'tied': [s['config']['name'] for s in tied], 'passing': [s['config']['name'] for s in passing]}


def sequences_dev(corpus: dict, positions: list[dict], eligible: set) -> list[list[dict]]:
    """Web DEV games: main positions in ply order (the AI's consecutive moves); undone positions alone."""
    games: dict[str, list[dict]] = {}
    out = []
    for p in positions:
        item = {'id': p['id'], 'moves': web.history(corpus, p), 'eligible': p['id'] in eligible}
        if p['set'] == 'main':
            games.setdefault(p['game_id'], []).append((p['ply'], item))
        else:
            out.append([item])
    out += [[item for _, item in sorted(seq, key=lambda x: x[0])] for seq in games.values()]
    return out


def sequences_costdev(pre_rows: list[dict], timing: dict) -> list[list[dict]]:
    games: dict[str, list] = {}
    for r in pre_rows:
        ply = int(r['id'].rsplit('|', 1)[1]) if r['id'].count('|') >= 2 else 0
        games.setdefault(r['game'], []).append((ply, {'id': r['id'], 'moves': timing[r['id']]['moves'],
                                                      'eligible': r['attacked']}))
    return [[item for _, item in sorted(seq, key=lambda x: x[0])] for seq in games.values()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--precheck', type=Path, required=True, help='the E4-v1 pre-check (COSTDEV)')
    parser.add_argument('--positions', type=Path, required=True, help='the timing positions of that pre-check')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--fast', choices=('off', 'fast2', 'fast3', 'fast4'), default='off')
    parser.add_argument('--calibration', type=Path, help='h7_e4v2_calibrate.py output: killer / gate corrections')
    parser.add_argument('--limit', type=int, help='smoke tests: first N positions of each set')
    parser.add_argument('--jsonl', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from analysis import e4v2_attack as v2
    from hybrid.progress import Progress, run_chunks
    from scripts.h7_timing import restore_fast_mode
    from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256

    corpus = web.load(args.corpus)
    t = json.loads(args.truth.read_text(encoding='utf-8'))
    b = json.loads(args.baseline.read_text(encoding='utf-8'))
    pre = json.loads(args.precheck.read_text(encoding='utf-8'))
    if t['corpus_sha256'] != file_sha256(args.corpus) or b['truth_sha256'] != file_sha256(args.truth):
        parser.error('the truth and the baseline must be of this corpus')
    if b.get('format') != base.FORMAT or pre.get('format') != 'h7-e4-precheck-v1':
        parser.error('needs the baseline v2 and the E4-v1 pre-check')
    if pre.get('positions_sha256') != file_sha256(args.positions) or pre.get('corpus_sha256') != file_sha256(args.corpus):
        parser.error(f'{args.precheck} was not measured on these timing positions and this web corpus')
    from analysis import tb_v2_value_e4
    if pre.get('candidate_hash') != tb_v2_value_e4.candidate_hash():
        parser.error(f'{args.precheck} is not the pre-check of the frozen E4-v1 candidate')
    timing = {p['id']: p for p in json.loads(args.positions.read_text(encoding='utf-8'))['positions']}
    fingerprint = {'e4v2_attack_sha256': file_sha256(ROOT / 'src/analysis/e4v2_attack.py'),
                   'e4_attack_sha256': file_sha256(ROOT / 'src/analysis/e4_attack.py'),
                   'truth_sha256': file_sha256(args.truth), 'check_budget': t['budget'], 'budgets': v2.BUDGETS,
                   'gate_node_limit': v2.GATE_NODE_LIMIT}
    positions = [dev_only(p) for p in corpus['positions'] if p['split'] == 'DEV'][:args.limit]
    baseline = {p['id']: b['positions'][p['id']] for p in positions if p['id'] in b['positions']}
    eligible = {p['id'] for p in positions
                if p['id'] not in baseline or baseline[p['id']]['b']['route'] in v2.ROUTES}
    pre_rows = pre['rows'][:args.limit]
    tasks = [{'set': 'dev', 'id': p['id'], 'moves': web.history(corpus, p), 'check_budget': t['budget'],
              'fingerprint': fingerprint} for p in positions if p['id'] in eligible]
    tasks += [{'set': 'costdev', 'id': r['id'], 'moves': timing[r['id']]['moves'], 'fingerprint': fingerprint}
              for r in pre_rows if r['attacked']]
    done = {}
    if args.jsonl is not None and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('fingerprint') != json.loads(json.dumps(fingerprint)):
                parser.error(f'{args.jsonl} has rows of another code / truth / budget ({row["id"]}); '
                             'use another --jsonl file')
            done[(row['set'], row['id'])] = row
    todo = [x for x in tasks if (x['set'], x['id']) not in done]
    print(f'[h7-e4v2-dev] DEV {len(positions)} positions ({len(eligible)} eligible), COSTDEV {len(pre_rows)} '
          f'positions, {len(tasks)} tasks, {len(tasks) - len(todo)} done, workers {args.workers}', flush=True)

    def keep(_, rows):
        for row in rows:
            done[(row['set'], row['id'])] = row
            if args.jsonl is not None:
                args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.jsonl.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')

    progress = Progress('h7-e4v2-dev', len(todo), 'positions')
    chunks = [[x] for x in todo]
    if args.workers > 1 and todo:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'), initializer=_init,
                                 initargs=(args.fast,)) as pool:
            run_chunks(pool, _chunk, chunks, progress, len, unit_name='positions', on_result=keep)
    else:
        with restore_fast_mode():
            _init(args.fast)
            for i, c in enumerate(chunks):
                keep(i, _chunk(c))
                progress.update(1)
    model = fit_cost_model(pre['rows'])
    correction, calibration = {}, None
    if args.calibration is not None:
        calibration = json.loads(args.calibration.read_text(encoding='utf-8'))
        if calibration.get('precheck_sha256') != file_sha256(args.precheck):
            parser.error(f'{args.calibration} is not a calibration of this pre-check')
        correction = calibration['correction']
    dev_rows = {i: r for (s, i), r in done.items() if s == 'dev'}
    cost_rows = {i: r for (s, i), r in done.items() if s == 'costdev'}
    dev_seq = sequences_dev(corpus, positions, eligible)
    cost_seq = sequences_costdev(pre_rows, timing)
    scores = []
    for config in v2.GRID:
        dev_steps = {k: v for seq in dev_seq for k, v in replay(config, seq, dev_rows).items()}
        cost_steps = {k: v for seq in cost_seq for k, v in replay(config, seq, cost_rows).items()}
        dev = dev_score(config, positions, t['boards'], baseline, dev_steps, model, correction)
        costdev = costdev_score(config, cost_steps, pre_rows, model, correction)
        scores.append({'config': config, 'dev': dev, 'costdev': costdev,
                       'eligible': not (dev['false_proofs'] or dev['contradictions'] or dev['shallow_regressions'])})
    selection = select(scores)
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'corpus_sha256': file_sha256(args.corpus),
               'truth_sha256': file_sha256(args.truth), 'baseline_sha256': file_sha256(args.baseline),
               'precheck_sha256': file_sha256(args.precheck), 'positions_sha256': file_sha256(args.positions),
               'split': 'DEV', 'fast': args.fast, 'restricted': args.limit is not None,
               'rule': {'cost_max': COST_MAX, 'tie': TIE, 'gate_node_limit': v2.GATE_NODE_LIMIT},
               'fingerprint': fingerprint,
               'cost_model': model, 'correction': correction,
               'calibration_sha256': file_sha256(args.calibration) if args.calibration else None, 'selection': selection, 'scores': scores,
               'rows': {'dev': dev_rows, 'costdev': cost_rows}}
    table = [{'name': s['config']['name'], 'punish': f"{s['dev']['punish']}/{s['dev']['targets']}",
              'shallow': len(s['dev']['shallow_regressions']), 'false': len(s['dev']['false_proofs']),
              'contra': len(s['dev']['contradictions']), 'costdev': s['costdev']['ratio'],
              'costdev_switched': s['costdev']['switched'], 'dev_ran': s['dev']['ran']} for s in scores]
    print(json.dumps({'cost_model': model, 'table': table, 'selection': selection}, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
