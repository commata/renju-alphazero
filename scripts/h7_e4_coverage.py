"""H7-E4 DEV coverage: is the truth witness's first move a V8-B attack candidate, and a VCF threat?

DEV positions only (HOLDOUT is never inspected here). For every DEV position graded WIN_VCT1 or WIN_VCT2_ONLY
by the truth (``h7_e4_truth.py``), with the baseline outcome of the frozen engine (``h7_e4_baseline.py``):

    in_v8b       the witness's first move is in ``mcts_v8._attack_candidates`` (a four or an open three):
                 the set V8-B and the E2 module ``own_vct2_attack`` start from
    vcf_threat   after the witness the attacker has a VCF if the defender passes (``ThreatSolver.vcf``,
                 board-based): the witness is a threat even when it is neither a four nor a three

The truth tries the four / open-three moves first (``TruthSolver``), so a witness outside V8-B means no V8-B
candidate was proven to win within the truth's budget (each was SAFE or UNKNOWN).

    python scripts/h7_e4_coverage.py --corpus docs/h7/h7_e4_web_corpus.json --truth docs/h7/h7_e4_truth.json \\
        --baseline docs/h7/h7_e4_baseline_v2.json --output docs/h7/h7_e4_dev_coverage.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'src'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import h7_e4_web_corpus as web  # noqa: E402

FORMAT = 'h7-e4-dev-coverage-v1'
GRADES = ('WIN_VCT1', 'WIN_VCT2_ONLY')


def coverage(moves, witness) -> dict:
    from analysis.mcts_v8 import SearchDiagnostics, _attack_candidates
    from analysis.threats import ThreatSolver
    from renju import Game
    from search.mcts_v321 import _fast_pattern_features_for_move
    from search.mcts_v5 import _RootContext

    game = Game()
    for m in moves:
        game.play(*m)
    me, w = game.to_play, tuple(witness)
    candidates = _attack_candidates(game, _RootContext(game.legal_moves(), SearchDiagnostics()))
    features = _fast_pattern_features_for_move(game, me, w, assume_legal=True)
    game.play(*w)
    threat = 'FIVE' if game.done else ThreatSolver(node_limit=100_000).vcf(game, me)[0]
    return {'in_v8b': w in candidates, 'rank': candidates.index(w) + 1 if w in candidates else None,
            'candidates': len(candidates), 'four': features.four_directions,
            'open_three': features.open_three_directions, 'vcf_threat': threat}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        key = f"{r['set']}/{r['grade']}/{r['outcome']}"
        s = out.setdefault(key, {'positions': 0, 'block': 0, 'in_v8b': 0, 'not_v8b_vcf_threat': 0, 'other': 0})
        s['positions'] += 1
        if r['kind'] == 'block':
            s['block'] += 1
        elif r['in_v8b']:
            s['in_v8b'] += 1
        elif r['vcf_threat'] == 'WIN':
            s['not_v8b_vcf_threat'] += 1
        else:
            s['other'] += 1
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--fast', choices=('off', 'fast4'), default='off')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f'{args.output} exists')
    from scripts.h7_timing import restore_fast_mode
    from scripts.run_mcts_v8_benchmark import _git_commit, file_sha256

    corpus = web.load(args.corpus)
    truth = json.loads(args.truth.read_text(encoding='utf-8'))
    base = json.loads(args.baseline.read_text(encoding='utf-8'))
    if truth['corpus_sha256'] != file_sha256(args.corpus) or base['truth_sha256'] != file_sha256(args.truth):
        parser.error('the truth and the baseline must be of this corpus')
    rows = []
    with restore_fast_mode():
        if args.fast != 'off':
            from hybrid.h6_fast import install
            install(args.fast)
        for p in corpus['positions']:
            r = base['positions'].get(p['id'])
            if p['split'] != 'DEV' or r is None:
                continue  # HOLDOUT is not inspected
            t = truth['boards'][p['key']]
            if t['grade'] not in GRADES:
                continue
            rows.append({'id': p['id'], 'set': p['set'], 'grade': t['grade'], 'kind': t['witness'][0],
                         'witness': t['witness'][1], 'outcome': r['b']['outcome'], 'b_route': r['b']['route'],
                         **coverage(web.history(corpus, p), t['witness'][1])})
    payload = {'format': FORMAT, 'git_commit': _git_commit(), 'corpus_sha256': file_sha256(args.corpus),
               'truth_sha256': file_sha256(args.truth), 'baseline_sha256': file_sha256(args.baseline),
               'split': 'DEV', 'coordinates': '0-indexed (row, col)', 'summary': summarize(rows), 'positions': rows}
    print(json.dumps(payload['summary'], indent=1), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
