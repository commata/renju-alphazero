"""H6-6D: is a 4M-UNKNOWN tree move a real risk or only hard to prove? (docs/h6-value.md, H6-6D). Report only.

H6-6 stays REJECT; the seeds used here (8404-8406) are never used to promote an engine. Input: the H6-6
aux runs B (TB-V1-Value vs v8:full) and C (TB-Engine-v1 vs v8:full) of every seed and their d2 files.

Every tree move gets a V8-C status: ``safe`` (standard SAFE), or from the 4M recheck ``resolved_safe``,
``unknown4m``, ``unsafe4m``. Three measures, each answering one question:

    loss12      did the recorded opponent punish it: the side that played it lost within 12 plies of it
                (the losing five at most 12 plies later). Only for tree moves that were actually played
                (V8-C did not switch); a switched move is counted apart. An observation, not a truth.
    v8_win      can TB-Engine-v1's own attack layer punish it now: after the move, the opponent has a five,
                or V8's own VCF (TB-Engine-v1 limits), or a V8-B proven depth-1 VCT attack.
    vct2_win    can the planned H7 attack layer punish it: v8_win, or the E2 selective depth-2 attack
                (``analysis.vct2_attack.own_vct2_attack``, the E2 budget). The H7 B-engine candidate of
                track-b-post-e1.md section 3.
A found win is a proof (every defender reply is played). Not found is not SAFE: both attacks are selective.

The attack checks run on every non-SAFE tree move and on one matched SAFE control each (same run and game,
nearest ply; ``h6_5d.match_controls``). Rates come with a 95% bootstrap over games; the reading compares
``unknown4m`` with its matched controls (paired by position, bootstrap over games), on the candidate arm
(whose excess failed the guard), with the pooled and baseline rows reported:

    danger        loss12 and vct2_win both higher than the controls (95% low > 0)
    proof_hard    loss12 and vct2_win both indistinguishable from the controls (95% intervals contain 0)
    mixed         otherwise (including too few pairs, < 20)

    python -u scripts/h6_6d.py --runs runs/h6_6/value_vs_full_840{4,5,6}.json runs/h6_6/tb1_vs_full_840{4,5,6}.json \\
        --d2 docs/h6/h6_6_d2_840{4,5,6}.json --fast fast3 --workers 12 --jsonl runs/h6_6/h6_6d.jsonl \\
        --output docs/h6/h6_6d.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / 'src', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.tb_engine_v1 import TB_ENGINE_V1  # noqa: E402
from scripts import h6_5d  # noqa: E402

FORMAT = 'h6-6d-v1'
CANDIDATE, BASELINE = h6_5d.CANDIDATE_ARM, h6_5d.BASELINE_ARM
PLIES = 12
GROUPS = ('safe', 'resolved_safe', 'unknown4m', 'unsafe4m')
RULES = {'loss_plies': PLIES, 'min_pairs': 20, 'bootstrap': 10_000, 'seed': 0}


# ---------- tree moves ----------

def tree_moves(runs: list[dict], d2_files: list[dict]) -> list[dict]:
    large = {r['id']: r['large']['status'] for d in d2_files for r in d['rows']}
    out = []
    for run in runs:
        for g in sorted(run['games'], key=lambda g: g['key']):
            for m in g['v8_moves']:
                if m['route'] != 'tree':
                    continue
                key = f"{run['arm']}|{g['key']}|{m['ply']}"
                std = h6_5d.tree_status(m)
                if std == 'SAFE':
                    group = 'safe'
                else:
                    if key not in large:
                        raise ValueError(f'no 4M recheck for {key}')
                    group = {'SAFE': 'resolved_safe', 'UNKNOWN': 'unknown4m', 'UNSAFE': 'unsafe4m'}[large[key]]
                played = list(m.get('played') or g['moves'][m['ply']])
                switched = played != list(m['v7_move'])
                last = g['length'] - 1  # index of the last move (the five, or a full-board draw)
                lost = g['result'] == 'loss' and last - m['ply'] <= PLIES
                out.append({'id': key, 'arm': run['arm'], 'seed': run['seed'], 'game': f"{run['arm']}|{g['key']}",
                            'ply': m['ply'], 'tree_move': list(m['v7_move']), 'switched': switched,
                            'standard': std, 'group': group, 'loss12': None if switched else lost})
    return out


# ---------- attack checks ----------

def attack_task(task: dict) -> dict:
    """After the tree move, can the opponent's attack layers prove a win?"""
    from analysis.mcts_v8 import SearchDiagnostics, _own_vct_attack
    from analysis.vct2_attack import own_vct2_attack
    from search.mcts_v5 import _RootContext
    from search.mcts_v7 import _find_vcf_with_stats
    from search.tactics import winning_points

    h6_5d._ensure_fast(task.get('fast', 'off'))
    game = h6_5d.replay(task['moves'] + [task['tree_move']])
    started = perf_counter()
    if game.done:
        return {'id': task['id'], 'done': True, 'five': False, 'vcf': False, 'v8b': False, 'vct2': False,
                'v8_win': False, 'vct2_win': False, 'seconds': 0.0}
    opponent = game.to_play
    five = bool(winning_points(game.board, opponent))
    found, _, _ = _find_vcf_with_stats(game, opponent, max_fours=TB_ENGINE_V1['own_vcf_max_fours'],
                                       node_limit=TB_ENGINE_V1['own_vcf_node_limit'])
    vcf = found is not None
    diag = SearchDiagnostics()
    v8b = _own_vct_attack(game, _RootContext(game.legal_moves(), diag), diag,
                          node_limit=TB_ENGINE_V1['attack_vcf_node_limit'],
                          call_limit=TB_ENGINE_V1['attack_call_limit'],
                          node_budget=TB_ENGINE_V1['attack_node_budget']) is not None
    vct2 = own_vct2_attack(game)[0] is not None
    v8_win = five or vcf or v8b
    return {'id': task['id'], 'done': False, 'five': five, 'vcf': vcf, 'v8b': v8b, 'vct2': vct2,
            'v8_win': v8_win, 'vct2_win': v8_win or vct2, 'seconds': round(perf_counter() - started, 2)}


def _chunk(tasks):
    return [attack_task(t) for t in tasks]


# ---------- statistics ----------

def _bootstrap(groups: dict[str, list[float]]) -> dict:
    """Mean of the values, 95% bootstrap over the groups (games)."""
    keys = sorted(k for k, v in groups.items() if v)
    if not keys:
        return {'n': 0}

    def mean_of(ks):
        vals = [x for k in ks for x in groups[k]]
        return sum(vals) / len(vals)

    rng, b = Random(RULES['seed']), RULES['bootstrap']
    draws = sorted(mean_of([rng.choice(keys) for _ in keys]) for _ in range(b))
    return {'n': sum(len(groups[k]) for k in keys), 'games': len(keys), 'mean': round(mean_of(keys), 4),
            'ci95': [round(draws[int(0.025 * b)], 4), round(draws[int(0.975 * b) - 1], 4)]}


def _rate(rows, value) -> dict:
    groups = {}
    for r in rows:
        v = value(r)
        if v is not None:
            groups.setdefault(r['game'], []).append(float(v))
    return _bootstrap(groups)


def _paired(pairs, value) -> dict:
    """``pairs``: (row, control) with the same game; value(target) - value(control), bootstrap over games."""
    groups = {}
    for a, c in pairs:
        va, vc = value(a), value(c)
        if va is not None and vc is not None:
            groups.setdefault(a['game'], []).append(float(va) - float(vc))
    return _bootstrap(groups)


def _contains_zero(stat) -> bool:
    return stat.get('n', 0) > 0 and stat['ci95'][0] <= 0 <= stat['ci95'][1]


def summarize(rows: list[dict], attacks: dict[str, dict], controls: dict[str, str]) -> dict:
    """``controls``: target id -> control id."""
    for r in rows:
        a = attacks.get(r['id'])
        r['v8_win'] = None if a is None or a['done'] else a['v8_win']
        r['vct2_win'] = None if a is None or a['done'] else a['vct2_win']
    by_id = {r['id']: r for r in rows}
    out = {'tree_moves': len(rows), 'switched': sum(r['switched'] for r in rows), 'arms': {}}
    for arm in (CANDIDATE, BASELINE, 'pooled'):
        sub = [r for r in rows if arm in ('pooled', r['arm'])]
        groups = {}
        for g in GROUPS:
            part = [r for r in sub if r['group'] == g]
            groups[g] = {'n': len(part), 'switched': sum(r['switched'] for r in part),
                         'loss12': _rate(part, lambda r: r['loss12']),
                         'v8_win': _rate(part, lambda r: r['v8_win']),
                         'vct2_win': _rate(part, lambda r: r['vct2_win'])}
        pairs = [(by_id[t], by_id[c]) for t, c in controls.items()
                 if by_id[t]['group'] == 'unknown4m' and arm in ('pooled', by_id[t]['arm'])]
        paired = {'pairs': len(pairs), **{k: _paired(pairs, lambda r, k=k: r[k]) for k in ('loss12', 'v8_win', 'vct2_win')}}
        out['arms'][arm] = {'groups': groups, 'unknown4m_vs_control': paired, 'reading': reading(paired)}
    out['reading'] = out['arms'][CANDIDATE]['reading']
    return out


def reading(paired: dict) -> str:
    if paired['pairs'] < RULES['min_pairs']:
        return 'mixed'
    loss, vct2 = paired['loss12'], paired['vct2_win']
    if loss.get('n') and vct2.get('n') and loss['ci95'][0] > 0 and vct2['ci95'][0] > 0:
        return 'danger'
    if _contains_zero(loss) and _contains_zero(vct2):
        return 'proof_hard'
    return 'mixed'


# ---------- main ----------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--runs', nargs='+', type=Path, required=True, help='B and C runs of every seed')
    parser.add_argument('--d2', nargs='+', type=Path, required=True, help='h6_5d.py d2 output of every seed')
    parser.add_argument('--fast', choices=('off', 'fast2', 'fast3'), default='off')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--jsonl', type=Path, help='attack results, appended as they finish (resume)')
    parser.add_argument('--limit', type=int, help='first N attack checks (smoke runs)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    runs = [json.loads(p.read_text(encoding='utf-8')) for p in args.runs]
    d2_files = [json.loads(p.read_text(encoding='utf-8')) for p in args.d2]
    seeds = sorted({r['seed'] for r in runs})
    for seed in seeds:
        h6_5d.load_runs([p for p, r in zip(args.runs, runs) if r['seed'] == seed])  # both arms, same opponent
    for d in d2_files:
        bad = [r['id'] for r in d['rows']
               if h6_5d.replay_kind(r['recorded_status'], r['standard']['status'], r['large']['status']) == 'mismatch']
        if bad:
            parser.error(f'a d2 file is not consistent with its recorded statuses: {bad[:3]}')
    if args.fast != 'off':
        try:
            h6_5d._ensure_fast(args.fast)
        except RuntimeError as exc:
            parser.error(str(exc))
    rows = tree_moves(runs, d2_files)
    positions = [p for r in runs for p in h6_5d.tree_positions(r)]
    controls = {c['control_for']: c['id'] for c in h6_5d.match_controls(positions)}
    targets = [p for p in positions if p['recorded_status'] != 'SAFE']
    by_id = {p['id']: p for p in positions}
    chosen = [*targets, *(by_id[c] for c in controls.values())]
    tasks = [{'id': p['id'], 'moves': p['moves'], 'tree_move': p['tree_move'], 'fast': args.fast}
             for p in chosen][:args.limit]
    done = h6_5d._run(tasks, _chunk, args.workers, args.jsonl, 'h6-6d attack')
    attacks = {a['id']: a for a in done}
    summary = summarize(rows, attacks, {t: c for t, c in controls.items() if t in attacks and c in attacks})
    from scripts.run_mcts_v8_benchmark import _fast_provenance
    result = {'format': FORMAT, 'report_only': True, 'rules': RULES, 'seeds': seeds, 'fast': _fast_provenance(args.fast),
              'sources': {str(p): {'arm': r['arm'], 'seed': r['seed']} for p, r in zip(args.runs, runs)},
              'summary': summary, 'attacks': done}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'reading': summary['reading'],
                      'candidate': summary['arms'][CANDIDATE]['unknown4m_vs_control']}, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
