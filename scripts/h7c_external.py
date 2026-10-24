"""H7-c X2: the cycle's candidate against Rapfi L_mid, record only (docs/h7-c1.md, docs/external-engine.md).

The candidate (c policy + c value) on the frozen TB-V3 / fast5, built by ``analysis.tb_v3_net.make_tb_v3_net_agent``
(its policy and value metadata must name this cycle and one self-play source) in ``scripts/run_external_match.py``'s
games (``--ours h7c``, which only this tool can give). Cycle c plays seed
8710 + c (``evaluation.external.MONITORING_SEEDS``; 8710 = cycle 0, the pre-RL baseline), 25 pairs, whatever the
promotion verdict. Rapfi never decides a promotion. The candidate's engine, runtime and model identities go into
``ours`` and so into ``run_signature``; every other option (engine, INFO, engine files, opening, ``--forbid-check``,
workers, output files) is passed through to the runner unchanged.

    python -u scripts/h7c_external.py --cycle 1 --policy-checkpoint runs/h7c/c1/policy/best.pt \\
        --value-checkpoint runs/h7c/c1/value/best.pt --value-device cuda \\
        --engine $EXE --engine-name rapfi $INFO $FILES --workers 4 --forbid-check \\
        --games-jsonl runs/external/x2/c1.jsonl --output docs/external/x2_c1.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

X2_BASE, PAIRS = 8710, 25
RESERVED_OPTIONS = ('--ours', '--seed', '--pairs', '--allow-unreserved-seed', '--policy-checkpoint',
                    '--value-checkpoint', '--value-device')


def x2_seed(cycle: int) -> int:
    from evaluation.external import MONITORING_SEEDS

    seed = X2_BASE + cycle
    if cycle < 1 or seed not in MONITORING_SEEDS:
        raise ValueError(f'--cycle {cycle}: X2 seeds 8711-8720 are cycles 1-10 (8710 is the pre-RL baseline)')
    return seed


def candidate_info(cycle: int, policy_checkpoint, value_checkpoint, device: str, mode: str = 'JOINT',
                   policy_record=None) -> dict:
    """Label and provenance of the candidate: the frozen engine identity, fast5 on this device, the four model SHA,
    the self-play source it was trained on (its metadata must name this cycle) and its candidate mode
    (``VALUE_ONLY_FALLBACK``: the parent policy with the cycle's value, ``h7c_arena.check_candidate``)."""
    from analysis import tb_v3, tb_v3_net
    from h7c_arena import check_candidate

    tb_v3.check_identity()
    source = check_candidate(policy_checkpoint, value_checkpoint, cycle, mode, policy_record)
    identity = tb_v3_net.model_identity(policy_checkpoint, value_checkpoint)
    provenance = {'cycle': cycle, 'engine_id': tb_v3.ENGINE_ID, 'engine_hash': tb_v3.ENGINE_HASH,
                  'candidate_mode': mode, 'source_models': source,
                  **tb_v3_net.provenance(device, identity)}
    return {'label': f"h7c:c{cycle}:{provenance['model_hash'][:12]}" + ('' if device == 'cuda' else f'[{device}]'),
            'provenance': provenance}


def make_candidate(spec: dict, seed: int):
    """``--ours h7c`` inside a runner worker: the models stay loaded across games, the factory checks every game."""
    from h7c_arena import make_side

    return make_side({'policy': spec['policy_checkpoint'], 'value': spec['value_checkpoint']}, seed,
                     spec['value_device'])


def runner_argv(args, rest: list[str]) -> list[str]:
    for option in rest:
        if option.split('=')[0] in RESERVED_OPTIONS:
            raise ValueError(f'{option} is fixed by this tool')
    return ['--ours', 'h7c', '--policy-checkpoint', str(args.policy_checkpoint),
            '--value-checkpoint', str(args.value_checkpoint), '--value-device', args.value_device,
            '--seed', str(x2_seed(args.cycle)), '--pairs', str(PAIRS), *rest]


def main(argv=None) -> int:
    import run_external_match as runner

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cycle', type=int, required=True)
    parser.add_argument('--policy-checkpoint', type=Path, required=True)
    parser.add_argument('--value-checkpoint', type=Path, required=True)
    parser.add_argument('--value-device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--candidate-mode', choices=('JOINT', 'VALUE_ONLY_FALLBACK'), default='JOINT')
    parser.add_argument('--policy-record', type=Path, help="VALUE_ONLY_FALLBACK: the policy training's record.json")
    args, rest = parser.parse_known_args(argv)
    try:
        forwarded = runner_argv(args, rest)
        info = candidate_info(args.cycle, args.policy_checkpoint, args.value_checkpoint, args.value_device,
                              args.candidate_mode, args.policy_record)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return runner.main(forwarded, h7c=info)


if __name__ == '__main__':
    raise SystemExit(main())
