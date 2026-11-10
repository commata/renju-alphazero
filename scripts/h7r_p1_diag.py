"""R2-P1-DIAG-1: why the R2-P1-POLICY-REPLAY-1 reproduction gate failed (docs/h7r/r2/R2-P1-DIAG-1-draft.md).

Diagnosis only: nothing here changes a contract, a tolerance, a trainer or a record of R2-P1-POLICY-REPLAY-1 (its
failed gate ``a1cd152`` stays as it is), and no treatment run or Arena is made.

    run      one training run in its own process, on the c2 inputs, seed and settings (checked against the P1
             contract's committed expectations with ``h7r_p1.check_inputs``):
               --trainer old    ``h7c_train.train_policy`` (the c2 trainer)
               --trainer new    ``h7r_p1_train.train_policy``, replay off (``--diag`` adds the RenjuNet val KL
                                diagnostic the gate run used)
               --deterministic  ``torch.use_deterministic_algorithms(True)`` and cuDNN deterministic (the caller
                                sets CUBLAS_WORKSPACE_CONFIG before the process starts)
             records: the numeric settings (torch, CUDA, cuDNN, GPU, determinism, TF32, threads, environment
             variables), the input SHAs, the initial parent / teacher state hashes, the per-step fingerprints (cycle
             batch indices, D4 symmetries, learning rate, training loss), the evaluation curve, eligibility, the
             selected step, the selected state hash, and the selected policy's logits on fixed positions (the first
             FIXED_RENJUNET RenjuNet val positions and every cycle val row; kept under runs/, hashed in the record)
             The fingerprints come from passive wrappers that return what the wrapped function returns.
    compare  every recorded run against the others and against the committed c2 record: where they first differ
             (step), by how much (curves, selected state, fixed-input logits), and whether the fingerprints, inputs
             and initial states are equal. A description, not a gate: no tolerance is derived from it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FORMAT = 'h7r-p1-diag-v1'
D = 'docs/h7r/r2/R2-P1-DIAG-1'
RUNS = 'runs/h7r/R2-P1-DIAG-1'
SEED = 20261024
FIXED_RENJUNET = 4096
ENV_KEYS = ('CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE', 'CUDA_LAUNCH_BLOCKING', 'OMP_NUM_THREADS',
            'MKL_NUM_THREADS', 'PYTHONHASHSEED')
RUN_NAME = {('old', False, False): 'old', ('new', False, False): 'new', ('new', True, False): 'new-diag',
            ('old', False, True): 'old-det', ('new', False, True): 'new-det'}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
                          .encode()).hexdigest()


def state_sha256(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def settings(device: str) -> dict:
    from hybrid.h7r_d1 import numeric_settings
    import torch

    out = numeric_settings(device)
    out['cuda_available'] = torch.cuda.is_available()
    out['environment'] = {k: os.environ.get(k) for k in ENV_KEYS}
    return out


class Fingerprints:
    """Per training step: the cycle batch indices, the D4 symmetries, the learning-rate factor and the training
    loss, recorded by wrappers that return what the wrapped functions return."""

    def __init__(self, train: dict):
        self.train, self.steps = train, []

    def install(self, stack):
        from unittest import mock

        from hybrid import h3_train, h7c_train as tr

        real_planes, real_aug, real_lr, real_terms = (tr.planes_from_states, h3_train.augment, tr.lr_factor,
                                                      tr._policy_terms)
        import torch

        def planes(data, index, device='cpu'):
            if data is self.train:
                self.steps.append({'batch': index.tolist()})
            return real_planes(data, index, device)

        def aug(planes_, legal, target, symmetry, dest, src):
            if self.steps and 'sym' not in self.steps[-1]:
                self.steps[-1]['sym'] = symmetry.tolist()
            return real_aug(planes_, legal, target, symmetry, dest, src)

        def lr(step, total, warm, min_ratio):
            out = real_lr(step, total, warm, min_ratio)
            if self.steps:
                self.steps[-1]['lr_factor'] = out
            return out

        def terms(*a, **k):
            out = real_terms(*a, **k)
            if torch.is_grad_enabled() and self.steps and 'loss' not in self.steps[-1]:
                self.steps[-1]['loss'] = float(out[2].detach().mean())
            return out
        for target, name, fn in ((tr, 'planes_from_states', planes), (h3_train, 'augment', aug),
                                 (tr, 'lr_factor', lr), (tr, '_policy_terms', terms)):
            stack.enter_context(mock.patch.object(target, name, side_effect=fn))

    def summary(self) -> dict:
        return {'steps': len(self.steps),
                'batch_sha256': digest([s.get('batch') for s in self.steps]),
                'sym_sha256': digest([s.get('sym') for s in self.steps]),
                'lr_sha256': digest([s.get('lr_factor') for s in self.steps]),
                'loss': [s.get('loss') for s in self.steps]}


def fixed_logits(model, checked: dict, device: str):
    """The policy logits (masked to the legal moves, log-softmax) on fixed positions: the first FIXED_RENJUNET
    RenjuNet val positions and every cycle val row."""
    import torch

    from hybrid import h7c_train as tr
    from hybrid.h3_cache import planes_from_cache
    from hybrid.h3_train import policy_logits

    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        val = checked['cache']['val']
        n = min(FIXED_RENJUNET, int(val['target'].shape[0]))
        planes, legal, _ = planes_from_cache(val, torch.arange(n), device)
        out.append(policy_logits(model, planes).float().masked_fill(~legal, -1e9).log_softmax(1).cpu())
        cyc = tr.policy_tensors(checked['rows'].get('val', []))
        if cyc['target'].shape[0]:
            planes, legal = tr.planes_from_states(cyc, torch.arange(cyc['target'].shape[0]), device)
            out.append(policy_logits(model, planes).float().masked_fill(~legal, -1e9).log_softmax(1).cpu())
    return torch.cat(out)


def run_one(c: dict, checked: dict, trainer: str, diag: bool, deterministic: bool, device: str, out_dir,
            root=ROOT, log=print) -> dict:
    """One run (module docstring); writes record.json, state.pt and logits.pt to ``out_dir`` (never over files)."""
    from contextlib import ExitStack

    import torch

    import h7r_p1 as p1
    from hybrid import h7c_train as tr, h7r_p1_train as pt
    from hybrid.h3_train import H3Config, evaluate
    from h7c_train_policy import load_policy_net

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} holds a run: never overwritten')
    if deterministic:
        if str(device).startswith('cuda') and not os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
            raise ValueError('--deterministic on CUDA needs CUBLAS_WORKSPACE_CONFIG set before the process starts')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    cfg = p1.policy_config(c, SEED, device)
    h3 = Path(root) / p1.PATHS['h3']
    parent, _ = load_policy_net(h3, device)
    teacher, _ = load_policy_net(h3, device)
    initial = {'parent': state_sha256(parent.state_dict()), 'teacher': state_sha256(teacher.state_dict())}
    rows = checked['rows']
    train, val = tr.policy_tensors(rows['train']), tr.policy_tensors(rows.get('val', []))
    guard = checked['cache']['val']
    h3_cfg = H3Config(seed=cfg.seed)
    ge = lambda m, d: evaluate(m, d, h3_cfg, torch.device(device))  # noqa: E731
    fp = Fingerprints(train)
    with ExitStack() as stack:
        fp.install(stack)
        if trainer == 'old':
            result = tr.train_policy(cfg, parent, teacher, train, val, guard, ge, log=log)
        else:
            result = pt.train_policy(cfg, parent, teacher, train, val, guard, ge, replay=None, replay_data=None,
                                     diag=guard if diag else None, log=log)
    rec = result['record']
    record = {'format': FORMAT, 'name': RUN_NAME[(trainer, diag, deterministic)], 'trainer': trainer,
              'diag': diag, 'deterministic': deterministic, 'seed': SEED, 'settings': settings(device),
              'config': tr.config_dict(cfg),
              'inputs': {'cycle_rows_sha256': c['inputs']['cycle_rows']['sha256'], 'h3': checked['h3'],
                         'cache': {s: c['inputs']['renjunet_cache']['expected'][s]['tensor_sha256'] for s in
                                   ('train', 'val')}},
              'initial_state_sha256': initial, 'fingerprints': fp.summary(),
              'evaluations': rec['evaluations'], 'eligibility': [e['eligible'] for e in rec['evaluations']],
              'selected_step': (rec['best'] or {}).get('step'), 'verdict': rec['verdict'],
              'teacher_guard_top1': rec['teacher_guard']['top1'], 'parent_val': rec['parent_val']}
    out.mkdir(parents=True, exist_ok=True)
    if result['state'] is not None:
        parent.load_state_dict(result['state'])
        record['selected_state_sha256'] = state_sha256(result['state'])
        torch.save({k: v.detach().cpu() for k, v in result['state'].items()}, out / 'state.pt')
        logits = fixed_logits(parent, checked, device)
        torch.save(logits, out / 'logits.pt')
        record['fixed_logits'] = {'positions': int(logits.shape[0]), 'sha256': hashlib.sha256(
            logits.numpy().tobytes()).hexdigest()}
    (out / 'record.json').write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
    return record


# --- compare ----------------------------------------------------------------------------------------------------

def first_difference(a: list, b: list):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i + 1  # a 1-based step
    return None if len(a) == len(b) else min(len(a), len(b)) + 1


def eval_diffs(a: list, b: list) -> dict:
    keys = ('objective', 'ce', 'kl')
    out = {'steps_equal': [x['step'] for x in a] == [y['step'] for y in b]}
    out |= {k: max(abs(x['val'][k] - y['val'][k]) for x, y in zip(a, b)) for k in keys}
    out['guard_top1'] = max(abs(x['guard_top1'] - y['guard_top1']) for x, y in zip(a, b))
    out['first_differing_evaluation'] = next((x['step'] for x, y in zip(a, b)
                                              if (x['val'], x['guard_top1']) != (y['val'], y['guard_top1'])), None)
    return out


def pair(a: dict, b: dict, run_dir=None) -> dict:
    """How two runs differ (a description)."""
    out = {'runs': [a['name'], b['name']],
           'settings_equal': {k: a['settings'][k] == b['settings'][k] for k in a['settings']},
           'inputs_equal': a['inputs'] == b['inputs'], 'config_equal': a['config'] == b['config'],
           'initial_state_equal': a['initial_state_sha256'] == b['initial_state_sha256']}
    fa, fb = a['fingerprints'], b['fingerprints']
    out['fingerprints_equal'] = {k: fa[k] == fb[k] for k in ('steps', 'batch_sha256', 'sym_sha256', 'lr_sha256')}
    out['loss_first_differing_step'] = first_difference(fa['loss'], fb['loss'])
    out['loss_max_abs_diff'] = max((abs(x - y) for x, y in zip(fa['loss'], fb['loss'])), default=None)
    out['evaluations'] = eval_diffs(a['evaluations'], b['evaluations'])
    out['eligibility_equal'] = a['eligibility'] == b['eligibility']
    out['selected_step_equal'] = a['selected_step'] == b['selected_step']
    out['selected_state_equal'] = a.get('selected_state_sha256') == b.get('selected_state_sha256')
    out['fixed_logits_equal'] = (a.get('fixed_logits') or {}).get('sha256') == (b.get('fixed_logits') or {}).get('sha256')
    if run_dir is not None and 'selected_state_sha256' in a and 'selected_state_sha256' in b:
        import torch
        sa = torch.load(Path(run_dir) / a['dir'] / 'state.pt', weights_only=True)
        sb = torch.load(Path(run_dir) / b['dir'] / 'state.pt', weights_only=True)
        out['selected_state_max_abs_diff'] = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)
        la = torch.load(Path(run_dir) / a['dir'] / 'logits.pt', weights_only=True)
        lb = torch.load(Path(run_dir) / b['dir'] / 'logits.pt', weights_only=True)
        legal = (la > -1e8) & (lb > -1e8)
        out['fixed_logits_max_abs_diff'] = float((la - lb).abs()[legal].max())
        out['fixed_argmax_agreement'] = float((la.argmax(1) == lb.argmax(1)).float().mean())
    return out


def against_c2(run: dict, c2: dict) -> dict:
    return {'run': run['name'], 'evaluations': eval_diffs(run['evaluations'], c2['evaluations']),
            'eligibility_equal': run['eligibility'] == [e['eligible'] for e in c2['evaluations']],
            'selected_step_equal': run['selected_step'] == (c2['best'] or {}).get('step'),
            'teacher_guard_equal': run['teacher_guard_top1'] == c2['teacher_guard']['top1'],
            'parent_val_equal': run['parent_val'] == c2['parent_val']}


def compare(run_dir, c2_path, root=ROOT) -> dict:
    run_dir = Path(run_dir)
    runs = []
    for rec_path in sorted(run_dir.glob('*/record.json')):
        rec = json.loads(rec_path.read_text(encoding='utf-8'))
        rec['dir'] = rec_path.parent.name
        runs.append(rec)
    if not runs:
        raise ValueError(f'{run_dir}: no runs')
    c2 = json.loads((Path(root) / c2_path).read_text(encoding='utf-8'))
    by = {}
    for r in runs:
        by.setdefault(r['name'], []).append(r)
    pairs = []
    for name, group in sorted(by.items()):  # repeats of one setup: run-to-run variation
        for i in range(1, len(group)):
            pairs.append({'kind': f'repeat {name}', **pair(group[0], group[i], run_dir)})
    first = {name: group[0] for name, group in by.items()}
    for a, b in (('old', 'new'), ('new', 'new-diag'), ('old-det', 'new-det'), ('old', 'old-det')):
        if a in first and b in first:
            pairs.append({'kind': f'{a} vs {b}', **pair(first[a], first[b], run_dir)})
    return {'format': f'{FORMAT}-compare', 'runs': [{k: r[k] for k in ('dir', 'name', 'trainer', 'diag',
                                                                      'deterministic', 'selected_step', 'verdict')}
                                                    for r in runs],
            'settings': {r['dir']: r['settings'] for r in runs}, 'pairs': pairs,
            'against_c2': [against_c2(r, c2) for r in runs],
            'note': 'a description of the differences; no tolerance or gate is derived from it here'}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--trainer', choices=('old', 'new'), required=True)
    r.add_argument('--diag', action='store_true')
    r.add_argument('--deterministic', action='store_true')
    r.add_argument('--device', default='cuda')
    r.add_argument('--out-dir', required=True)
    m = sub.add_parser('compare')
    m.add_argument('--run-dir', default=RUNS)
    m.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            import h7r_p1 as p1
            c, _, _ = p1.load_contract(None, ROOT)  # read only: the committed P1 contract and its pinned sources
            checked = p1.check_inputs(c, ROOT)
            rec = run_one(c, checked, args.trainer, args.diag, args.deterministic, args.device, args.out_dir)
            print(json.dumps({k: rec.get(k) for k in ('name', 'selected_step', 'eligibility', 'selected_state_sha256')}),
                  flush=True)
        else:
            out = Path(args.output)
            if out.exists():
                raise ValueError(f'{out} exists: never overwritten')
            from hybrid.h7r_ledger import publish_json
            body = compare(args.run_dir, 'docs/h7c/c2/policy_record.json')
            publish_json(out, body)
            for p in body['pairs']:
                print(json.dumps({k: p[k] for k in ('kind', 'loss_first_differing_step', 'selected_state_equal',
                                                    'selected_step_equal')}), flush=True)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
