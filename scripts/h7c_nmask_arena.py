"""H7C-NMASK-1 official Arena, tactical gate and verdict (docs/h7-c1.md, "VALUE_ONLY_NETWORK_MASK 계약").

The candidate (frozen H3 policy + the masked value of ``scripts/h7c_nmask.py train``) against its parent c5, with the
games, the tactical record and the promotion rule of the cycle Arena (``scripts/h7c_arena.py``: ``play_game``, 50 pairs,
two random opening plies, colours swapped, the pair bootstrap, ``h7c-tactical-v2``, ``promotion``). Only the identity
differs: the experiment, the committed decision (``hybrid.h7c_nmask``), its seed 8761 (``RESERVED_SEEDS``) and the
candidate's training record. A PROMOTE here is recorded as the rule gives it; it changes no champion, B2 program,
canonical engine or web model.

    python -u scripts/h7c_nmask_arena.py play --device cuda --workers 6
    python -u scripts/h7c_nmask_arena.py tactical --device cuda --workers 6
    python -u scripts/h7c_nmask_arena.py verdict

Outputs: ``runs/h7c/nmask/arena/official.jsonl`` (resume) and ``official.json``, ``runs/h7c/nmask/arena/tactical.json``,
``docs/h7c/nmask/verdict.json``. A dirty tree or an uncommitted decision is refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h7c_arena as ar  # noqa: E402
from hybrid import h7c_nmask as nm  # noqa: E402
from hybrid.progress import Progress, run_chunks  # noqa: E402

FORMAT = 'h7c-nmask-arena-v1'
ARM = 'official'
DEFAULTS = {
    'candidate_value': ROOT / 'runs/h7c/nmask/value/best.pt', 'policy': ROOT / nm.PATHS['policy'],
    'parent_value': ROOT / nm.PATHS['parent_value'], 'jsonl': ROOT / 'runs/h7c/nmask/arena/official.jsonl',
    'output': ROOT / 'runs/h7c/nmask/arena/official.json', 'tactical': ROOT / 'runs/h7c/nmask/arena/tactical.json',
    'verdict': ROOT / 'docs/h7c/nmask/verdict.json',
}


def arena_seed(decision: dict) -> int:
    from hybrid.h6_labels import RESERVED_SEEDS

    seed = decision['evaluation']['official']['seed']
    nm.require(seed == nm.SEEDS['arena_official'] and seed in RESERVED_SEEDS, f'seed {seed} is not the reserved one')
    return seed


def check_models(args, decision: dict, decision_sha: str) -> dict:
    """The candidate of this decision (H3 policy byte for byte + the masked value) and its parent c5."""
    from analysis import tb_v3_net
    import h7c_nmask as cmd

    candidate_record = cmd.check_candidate(args.candidate_value, decision, decision_sha)
    models = {'candidate': {'policy': str(args.policy), 'value': str(args.candidate_value)},
              'parent': {'policy': str(args.policy), 'value': str(args.parent_value)}}
    for name in ('candidate', 'parent'):
        models[f'{name}_identity'] = tb_v3_net.model_identity(models[name]['policy'], models[name]['value'])
    policy_keys = ('policy_checkpoint_sha256', 'policy_metadata_sha256')
    checks = {
        'parent = c5': models['parent_identity'] == decision['parent'],
        'candidate policy = the parent policy': all(models['candidate_identity'][k] == decision['parent'][k]
                                                    for k in policy_keys),
        'candidate value = its record': models['candidate_identity']['value_checkpoint_sha256']
        == candidate_record['checkpoint_sha256'],
        'candidate is not the parent or the control': models['candidate_identity'] not in (
            decision['parent'], decision['control']),
    }
    failed = sorted(k for k, ok in checks.items() if not ok)
    nm.require(not failed, f'not the H7C-NMASK-1 models ({", ".join(failed)})')
    models['candidate_record_sha256'] = nm.file_sha256(Path(args.candidate_value).with_name('record.json'))
    return models


def identity(decision: dict, decision_sha: str, models: dict, device: str, git: dict) -> dict:
    from analysis import tb_v3, tb_v3_fast5, tb_v3_net

    return {'format': FORMAT, 'experiment': nm.EXPERIMENT, 'arm': ARM, 'role': 'promotion',
            'seed': arena_seed(decision), 'pairs': ar.ARMS[ARM]['pairs'], 'opening_rule': ar.OPENING,
            'engine_id': tb_v3.ENGINE_ID, 'engine_hash': tb_v3.ENGINE_HASH, 'runtime': ar.RUNTIME,
            'runtime_hash': tb_v3_fast5.runtime_hash(device), 'value_device': device.split(':')[0],
            'candidate': models['candidate_identity'], 'parent': models['parent_identity'],
            'decision_sha256': decision_sha, 'candidate_record_sha256': models['candidate_record_sha256'],
            'tested': {'models': models['candidate_identity'],
                       'model_hash': tb_v3_net.model_hash(models['candidate_identity'])},
            'opponent': {'kind': 'parent', 'models': models['parent_identity']}, 'git': git}


def _context(args):
    git = ar._git()
    nm.require(git['dirty'] is False or args.allow_dirty, 'the tree is dirty: Arena games must name their code commit')
    decision, sha = nm.load_decision(args.decision, require_committed=not args.allow_uncommitted)
    return git, decision, sha, check_models(args, decision, sha)


def play(args, log=print) -> dict:
    git, decision, dsha, models = _context(args)
    ident = identity(decision, dsha, models, args.device, git)
    sig = ar.signature(ident)
    tasks, openings = ar.build_tasks(ARM, 0, getattr(args, 'pairs', None), seed=ident['seed'])
    finished = ar.load_finished(Path(args.jsonl), {t['key'] for t in tasks}, sig, log=log)
    context = {'models': models, 'device': args.device, 'signature': sig}
    todo = [{**t, 'context': context, **({'max_plies': args.max_plies} if getattr(args, 'max_plies', None) else {})}
            for t in tasks if t['key'] not in finished]
    log(f'[h7c-nmask-arena] seed {ident["seed"]}, games {len(tasks)} ({len(finished)} done), workers {args.workers}, '
        f'signature {sig[:12]}', flush=True)
    Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if todo:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        with Path(args.jsonl).open('a', encoding='utf-8') as out, ProcessPoolExecutor(
                max_workers=args.workers, mp_context=get_context('spawn')) as pool:
            def keep(_, rows):
                for row in rows:
                    out.write(json.dumps(row, separators=(',', ':')) + '\n')
                    finished[row['key']] = row
                out.flush()
            run_chunks(pool, ar._play_chunk, [[t] for t in todo], Progress('h7c-nmask-arena', len(todo), 'games'),
                       len, unit_name='games', on_result=keep)
    games = [finished[t['key']] for t in tasks]
    return {'format': f'{FORMAT}-arm', 'identity': ident, 'signature': sig, 'openings': openings,
            'wall_seconds': round(perf_counter() - started, 1), 'summary': ar.summarize_arm(games),
            'fingerprint': ar.fingerprint(games), 'games': games}


def tactical(args, log=print) -> dict:
    git, decision, dsha, models = _context(args)
    out = {'format': f'{FORMAT}-tactical', 'experiment': nm.EXPERIMENT, 'git': git, 'device': args.device,
           'candidate': models['candidate_identity'], 'parent': models['parent_identity'], 'decision_sha256': dsha,
           'candidate_record_sha256': models['candidate_record_sha256'], 'rule_version': ar.TACTICAL_RULE,
           'files': {'value_probes': ar._file_sha(ar.PROBE_FILE), 'fresh_probes': ar._file_sha(ar.FRESH_PROBES),
                     'policy_probes': ar._file_sha(ar.POLICY_PROBES), 'e1_manifest': ar._file_sha(ar.E1_MANIFEST)}}
    for name in ('candidate', 'parent'):
        side = models[name]
        log(f'[h7c-nmask-arena] tactical: {name}', flush=True)
        out[name + '_result'] = {'value_probes': ar.value_probes(side['value'], args.device),
                                 'must_block': ar.must_block(side['policy']),
                                 'e1': ar.e1_suite(side, args.device, args.workers, name, log=log)}
    out['judge'] = ar.tactical_judge(out['candidate_result'], out['parent_result'])
    return out


def verdict(args) -> dict:
    arm, tact = ar._json(args.arm), ar._json(args.tactical)
    nm.require(arm.get('format') == f'{FORMAT}-arm' and tact.get('format') == f'{FORMAT}-tactical',
               'not an H7C-NMASK-1 arm and tactical record')
    nm.require(ar.signature(arm['identity']) == arm['signature'] and all(
        g['signature'] == arm['signature'] for g in arm['games']), 'the arm\'s games do not match its identity')
    nm.require(arm['summary'] == ar.summarize_arm(arm['games']) and arm['fingerprint'] == ar.fingerprint(arm['games']),
               'the arm summary or fingerprint is not the one its games give')
    decision, dsha = nm.load_decision(args.decision, require_committed=not args.allow_uncommitted)
    i = arm['identity']
    same = {'decision': i['decision_sha256'] == tact['decision_sha256'] == dsha,
            'candidate / parent': (i['candidate'], i['parent']) == (tact['candidate'], tact['parent']),
            'parent = c5': i['parent'] == decision['parent'],
            'candidate record': i['candidate_record_sha256'] == tact['candidate_record_sha256'],
            'commit': i['git']['commit'] == tact['git']['commit'],
            'clean trees': args.allow_dirty or (i['git']['dirty'] is False and tact['git']['dirty'] is False),
            'seed': i['seed'] == arena_seed(decision), 'rule': tact['rule_version'] == ar.TACTICAL_RULE}
    failed = sorted(k for k, ok in same.items() if not ok)
    nm.require(not failed, f'the arm and tactical records differ ({", ".join(failed)})')
    official = arm['summary']
    invalid = official['opponent_illegal_pairs']
    nm.require(official['complete_pairs'] + len(invalid) == ar.ARMS[ARM]['pairs'] and not official['truncated'],
               f"the official arm has {official['complete_pairs']} complete pairs")
    forbidden = official['illegal']['tested']
    gate = dict(tact['judge']['items'])
    gate['forbidden errors == 0 (all Arena games)'] = forbidden == 0
    tactical_record = {'rule_version': ar.TACTICAL_RULE, 'items': gate,
                       'verdict': 'PASS' if all(gate.values()) else 'FAIL', 'forbidden_errors': forbidden,
                       'evidence': tact['judge']['evidence'], 'compare': tact['judge']['compare'],
                       'regressions_reported': tact['judge']['regressions_reported']}
    out = {'format': f'{FORMAT}-verdict', 'experiment': nm.EXPERIMENT, 'seed': i['seed'], 'decision_sha256': dsha,
           'git_commit': tact['git']['commit'], 'candidate': i['candidate'], 'parent': i['parent'],
           'candidate_record_sha256': i['candidate_record_sha256'],
           'official': {**{k: official[k] for k in ('score', 'score_ci95', 'improvement', 'improvement_ci95', 'elo',
                                                     'wins', 'draws', 'losses', 'score_by_color', 'complete_pairs')},
                        'valid': not invalid, 'opponent_illegal_pairs': invalid},
           'tactical': tactical_record, 'fingerprint': arm['fingerprint'],
           'arm_sha256': ar._file_sha(Path(args.arm)), 'tactical_sha256': ar._file_sha(Path(args.tactical)),
           'promotion': ar.promotion(official, tactical_record['verdict'] == 'PASS', valid=not invalid),
           'meaning': 'the existing promotion rule on this experiment; it changes no champion, program, canonical '
                      'engine or web model'}
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    for name in ('play', 'tactical', 'verdict'):
        p = sub.add_parser(name)
        p.add_argument('--decision', type=Path, default=ROOT / nm.RECORDS['decision'])
        p.add_argument('--allow-uncommitted', action='store_true', help='tests only')
        p.add_argument('--allow-dirty', action='store_true', help='tests only')
    for name in ('play', 'tactical'):
        p = sub.choices[name]
        p.add_argument('--candidate-value', type=Path, default=DEFAULTS['candidate_value'])
        p.add_argument('--policy', type=Path, default=DEFAULTS['policy'])
        p.add_argument('--parent-value', type=Path, default=DEFAULTS['parent_value'])
        p.add_argument('--device', default='cuda')
        p.add_argument('--workers', type=int, default=6)
    sub.choices['play'].add_argument('--jsonl', type=Path, default=DEFAULTS['jsonl'])
    sub.choices['play'].add_argument('--output', type=Path, default=DEFAULTS['output'])
    sub.choices['tactical'].add_argument('--output', type=Path, default=DEFAULTS['tactical'])
    v = sub.choices['verdict']
    v.add_argument('--arm', type=Path, default=DEFAULTS['output'])
    v.add_argument('--tactical', type=Path, default=DEFAULTS['tactical'])
    v.add_argument('--output', type=Path, default=DEFAULTS['verdict'])
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error(f'{args.output} exists: never overwritten')
    try:
        out = {'play': play, 'tactical': tactical, 'verdict': verdict}[args.mode](args)
    except ValueError as exc:
        parser.error(str(exc))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    brief = out.get('summary') or out.get('judge') or {k: out[k] for k in ('official', 'tactical', 'promotion')}
    print(json.dumps(brief, indent=1, default=str)[:4000], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
