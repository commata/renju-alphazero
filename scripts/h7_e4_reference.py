"""H7-E4 reference solver DEV check: must contain every DEV win of the selected candidate (DEV only).

Rule (fixed before running, docs/h6-value.md "H7-E4 다음 단계 확정"):
    containment   on every DEV position where the selected candidate (``h7_e4_dev.py`` selection) proved a
                  WIN, the reference solver (``analysis.e4_reference``) must prove a WIN too
    soundness     every reference WIN is re-checked by the truth solver (``h7_e4_baseline.check_move``):
                  MISS = a false proof; on a truth NOT_FOUND position a WIN is a contradiction
    verdict       FREEZE       containment complete, no false proof, no contradiction
                  RAISE_TO_4M  (2M run only) a candidate WIN is missing: run once more with ``--budget 4m``
                  INVESTIGATE  4M still misses one, or any false proof / contradiction: no arena until solved
Reported only: recall on the DEV WIN_VCT2_ONLY positions (main + undone), cost (nodes, calls, seconds).
Every position read here must be DEV (``h7_e4_dev.dev_only``).

    python scripts/h7_e4_reference.py --corpus docs/h7/h7_e4_web_corpus.json --truth docs/h7/h7_e4_truth.json \\
        --dev docs/h7/h7_e4_dev.json --budget 2m --workers 4 --fast fast4 \\
        --jsonl runs/h7_e4/reference_2m.jsonl --output docs/h7/h7_e4_reference_2m.json

E4-v2 (``h7_e4v2_dev.py`` result): its candidate wins are replayed with the cooldown (``--baseline`` needed):

    python scripts/h7_e4_reference.py --corpus docs/h7/h7_e4_web_corpus.json --truth docs/h7/h7_e4_truth.json \\
        --dev docs/h7/h7_e4v2_dev.json --baseline docs/h7/h7_e4_baseline_v2.json --budget 4m --workers 3 \\
        --fast fast4 --jsonl runs/h7_e4v2/reference_4m.jsonl --output docs/h7/h7_e4v2_reference_4m.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import h7_e4_baseline as base  # noqa: E402
from scripts import h7_e4_truth as truth  # noqa: E402
from scripts import h7_e4_web_corpus as web  # noqa: E402
from scripts.h7_e4_dev import dev_only  # noqa: E402

FORMAT = 'h7-e4-reference-v1'


def _init(fast: str) -> None:
    if fast != 'off':
        from hybrid.h6_fast import install
        install(fast)


def reference_task(task: dict) -> dict:
    from analysis.e4_reference import reference_attack
    from renju import Game

    game = Game()
    for m in task['moves']:
        game.play(*m)
    move, info = reference_attack(game, task['budget'])
    row = {'id': task['id'], 'key': task['key'], 'budget': task['budget'], 'check_budget': task['check_budget'],
           'reference': info, 'check': None}
    if move is not None:
        row['check'] = base.check_move(task['moves'], move, task['check_budget'])
    return row


def _chunk(tasks):
    return [reference_task(t) for t in tasks]


def judge(rows: dict, candidate_wins: list[str], targets: list[str], boards: dict, keys: dict, budget: str) -> dict:
    win = {i for i, r in rows.items() if r['reference']['move'] is not None}
    missing = [i for i in candidate_wins if i not in win]
    false = [i for i in win if rows[i]['check']['outcome'] == 'MISS']
    unverified = [i for i in win if rows[i]['check']['outcome'] == 'UNCLEAR']
    contradictions = [i for i in win if boards[keys[i]]['class'] == 'NOT_FOUND']
    if false or contradictions:
        verdict = 'INVESTIGATE'
    elif missing:
        verdict = 'RAISE_TO_4M' if budget == '2m' else 'INVESTIGATE'
    else:
        verdict = 'FREEZE'
    nodes = [r['reference']['nodes'] for r in rows.values()]
    seconds = [r['reference']['seconds'] for r in rows.values()]
    return {'verdict': verdict, 'budget': budget,
            'containment': {'candidate_wins': len(candidate_wins), 'contained': len(candidate_wins) - len(missing),
                            'missing': missing},
            'recall': {'targets': len(targets), 'found': sum(t in win for t in targets),
                       'missed': [t for t in targets if t not in win]},
            'false_proofs': false, 'contradictions': contradictions, 'unverified': unverified,
            'cost': {'positions': len(rows), 'nodes_mean': round(sum(nodes) / len(nodes), 1) if nodes else None,
                     'nodes_max': max(nodes, default=None), 'exhausted': sum(r['reference']['exhausted'] for r in rows.values()),
                     'seconds_mean': round(sum(seconds) / len(seconds), 2) if seconds else None,
                     'seconds_max': max(seconds, default=None)}}


def candidate_wins_v2(corpus: dict, dev: dict, baseline: dict, selected: dict) -> list[str]:
    """DEV positions where the selected E4-v2 step ran and proved a WIN (``h7_e4v2_dev.replay``, as scored)."""
    from analysis.e4v2_attack import ROUTES
    from scripts import h7_e4v2_dev as dev2

    positions = [dev_only(p) for p in corpus['positions'] if p['split'] == 'DEV']
    recorded = {p['id']: baseline['positions'][p['id']] for p in positions if p['id'] in baseline['positions']}
    eligible = {p['id'] for p in positions if p['id'] not in recorded or recorded[p['id']]['b']['route'] in ROUTES}
    steps = {}
    for seq in dev2.sequences_dev(corpus, positions, eligible):
        steps.update(dev2.replay(selected, seq, dev['rows']['dev']))
    return sorted(i for i, step in steps.items() if step['ran'] and step['move'] is not None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True, help='h7_e4_dev.py output (its selection)')
    parser.add_argument('--baseline', type=Path, help='E4-v2 DEV results: the baseline v2 they were scored on')
    parser.add_argument('--budget', choices=('2m', '4m'), default='2m')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--fast', choices=('off', 'fast2', 'fast3', 'fast4'), default='off')
    parser.add_argument('--jsonl', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from hybrid.progress import Progress, run_chunks
    from scripts.h7_timing import restore_fast_mode
    from scripts.run_mcts_v8_benchmark import _git_commit, _git_dirty, file_sha256

    corpus = web.load(args.corpus)
    t = json.loads(args.truth.read_text(encoding='utf-8'))
    d = json.loads(args.dev.read_text(encoding='utf-8'))
    if t['corpus_sha256'] != file_sha256(args.corpus) or d['truth_sha256'] != file_sha256(args.truth):
        parser.error('the truth and the DEV result must be of this corpus')
    selected = d['selection']['selected']
    if selected is None:
        parser.error(f'{args.dev} selected no configuration')
    positions = {p['id']: p for p in corpus['positions'] if p['split'] == 'DEV'}
    if d.get('format') == 'h7-e4v2-dev-v1':  # E4-v2: the wins of the selected step, cooldown replayed
        if args.baseline is None:
            parser.error('an E4-v2 DEV result needs --baseline (the recorded base routes)')
        b = json.loads(args.baseline.read_text(encoding='utf-8'))
        if d['baseline_sha256'] != file_sha256(args.baseline):
            parser.error(f'{args.baseline} is not the baseline of {args.dev}')
        candidate_wins = candidate_wins_v2(corpus, d, b, selected)
    else:
        candidate_wins = sorted(a['id'] for a in d['attacks']
                                if a['budget_name'] == selected['budget'] and a['candidates'] == selected['candidates']
                                and a['attack']['move'] is not None)
    targets = sorted(i for i, p in positions.items() if t['boards'][p['key']]['grade'] == truth.TARGET)
    ids = sorted(set(candidate_wins) | set(targets))
    tasks = [{'id': i, 'key': dev_only(positions[i])['key'], 'moves': web.history(corpus, positions[i]),
              'budget': args.budget, 'check_budget': t['budget']} for i in ids]
    done = {}
    if args.jsonl is not None and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row['budget'] == args.budget and row['check_budget'] == t['budget']:
                done[row['id']] = row
    todo = [x for x in tasks if x['id'] not in done]
    print(f'[h7-e4-reference] {args.budget}: {len(candidate_wins)} candidate wins, {len(targets)} VCT2-only targets, '
          f'{len(tasks)} positions, {len(tasks) - len(todo)} done, workers {args.workers}', flush=True)

    def keep(_, rows):
        for row in rows:
            done[row['id']] = row
            if args.jsonl is not None:
                args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.jsonl.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')

    progress = Progress('h7-e4-reference', len(todo), 'positions')
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
    rows = {i: done[i] for i in ids}
    keys = {i: positions[i]['key'] for i in ids}
    result = judge(rows, candidate_wins, targets, t['boards'], keys, args.budget)
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'git_dirty': _git_dirty(),
               'reference_sha256': file_sha256(ROOT / 'src/analysis/e4_reference.py'),
               'e4_attack_sha256': file_sha256(ROOT / 'src/analysis/e4_attack.py'), 'corpus_sha256': file_sha256(args.corpus),
               'truth_sha256': file_sha256(args.truth), 'dev_sha256': file_sha256(args.dev), 'split': 'DEV',
               'selected': selected, 'fast': args.fast, **result, 'positions': rows}
    print(json.dumps(result, indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0 if result['verdict'] == 'FREEZE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
