"""B2 guard: promoted VALUE_ONLY candidate versus the original frozen H3/H6 v3, 50 new seed pairs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts import h7c_arena as ar
from hybrid import h7c_b2 as b2, h7c_recipe


def play(args):
    from analysis import tb_v3, tb_v3_net
    git = ar._git()
    b2.require(git['dirty'] is False, 'drift requires a clean code commit')
    d, decision_sha = h7c_recipe.load(args.recipe_decision, args.cycle)
    _, program_sha = b2.validate_decision(d)
    models = ar.check_models(args, False)
    frozen = {'policy': str(args.frozen_policy), 'value': str(args.frozen_value)}
    b2.require(tb_v3_net.model_identity(frozen['policy'], frozen['value']) == tb_v3.MODELS,
               'drift opponent must be the original frozen H3/H6 v3')
    verdict = b2.read(args.verdict)
    b2.require(verdict.get('cycle') == args.cycle and verdict.get('promotion', {}).get('decision') == 'PROMOTE'
               and verdict.get('candidate') == models['candidate_identity']
               and verdict.get('recipe_decision_sha256') == decision_sha, 'drift needs this candidate promotion')
    seed = d['seeds']['drift']
    b2.require(seed in b2.DRIFT_SEEDS, 'unregistered drift seed')
    identity = {'format': 'h7c-drift-v1', 'cycle': args.cycle, 'seed': seed, 'pairs': 50,
                'candidate': models['candidate_identity'], 'training_parent': models['parent_identity'],
                'frozen_opponent': tb_v3.MODELS, 'recipe_decision_sha256': decision_sha,
                'b2_program_sha256': program_sha, 'verdict_sha256': ar._file_sha(args.verdict),
                'engine_hash': tb_v3.ENGINE_HASH, 'device': args.device, 'git': git}
    from analysis import tb_v3_fast5
    identity['runtime_hash'] = tb_v3_fast5.runtime_hash(args.device)
    sig = ar.signature(identity)
    tasks, openings = ar.build_tasks('official', args.cycle, 50, seed=seed)
    for task in tasks:
        task['key'] = task['key'].replace('official/', 'drift/', 1)
    done = ar.load_finished(args.jsonl, {t['key'] for t in tasks}, sig)
    # Reuse the official candidate-vs-parent game worker with the fixed baseline as its opponent.
    context = {'models': {**models, 'parent': frozen}, 'device': args.device, 'signature': sig}
    pending = [{**t, 'context': context} for t in tasks if t['key'] not in done]
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    if pending:
        with args.jsonl.open('a', encoding='utf-8') as out, ProcessPoolExecutor(
                max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            for rows in pool.map(ar._play_chunk, [[t] for t in pending]):
                for row in rows:
                    row['arm'] = 'drift'
                    out.write(json.dumps(row, separators=(',', ':')) + '\n')
                    out.flush()
                    done[row['key']] = row
                print(f"drift {len(done)}/100 games", flush=True)
    games = [done[t['key']] for t in tasks]
    summary = ar.summarize_arm(games)
    return {'format': 'h7c-drift-v1', 'identity': identity, 'signature': sig, 'openings': openings,
            'summary': summary, 'judge': b2.drift_judge(summary), 'fingerprint': ar.fingerprint(games), 'games': games}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cycle', type=int, required=True)
    for key in ('candidate-policy', 'candidate-value', 'parent-policy', 'parent-value', 'champion',
                'recipe-decision', 'frozen-policy', 'frozen-value', 'verdict', 'jsonl', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args(argv)
    args.candidate_mode, args.policy_record, args.allow_stand_in = 'VALUE_ONLY', None, False
    try:
        b2.require(not args.output.exists(), f'{args.output} exists')
        record = play(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
        print(json.dumps(record['judge']), flush=True)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    return 0  # guard is a recorded stop decision, not an execution failure


if __name__ == '__main__':
    raise SystemExit(main())
