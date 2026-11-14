"""Desktop attempt records (sidecars) for R2-S400-TRAIN stages whose result keeps no log name (search, rows).

The training tool's records are pinned by the contract (``sources``), so the link between a result and the run that
wrote it lives next to the run log instead. One attempt is one run of a desktop block; its record
``<log dir>/<stage>-XXXXXXXX.attempt.json`` is written before the run and finished after it:

    start    before a run: the stage, the contract's experiment SHA-256 and file SHA-256, HEAD, the log, the result
             path and the output files to hash. Earlier open attempts of the stage (no finish: interrupted) are closed
             first, their log ended by an ``__INTERRUPTED__`` line when it has no exit line. ``--set-aside`` moves a
             file an interrupted run left behind into the interrupted folder (moved, never deleted or overwritten)
             and records its SHA-256 and the attempt that left it.
    finish   after the block wrote ``__EXIT__ <code>`` to the log: the exit code, the log SHA-256, the result's SHA-256
             (it must be of the attempt's experiment and HEAD and mean that exit code; codes 0 / 2 need a result)
             and the output files' SHA-256 (with whether each equals the file set aside for it).
    recover  a result exists but is not committed: the attempt that wrote it (finished with its SHA-256, or else the
             one open attempt of the result's experiment and HEAD, finished now from the exit code the result means).
             Prints ``<log name> <exit code>``.
    copy     a finished attempt into the records folder next to its log copy (whose SHA-256 must be the attempt's).

Nothing here judges a result: the tool's own checks (``verified_gate``, ``verified_rows``) stay the judgement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

FORMAT = 'h7r-attempt-v1'
CONTRACT_ID = 'R2-S400-TRAIN'
CONTRACT = 'docs/h7r/r2/R2-S400-TRAIN/s400-train-contract.json'
NAME = re.compile(r'[a-z]+-[A-Za-z0-9]+')


def _search_code(r: dict) -> int:
    codes = {'SEARCHED': 0, 'S200_NOT_REPRODUCED': 2}
    if r.get('status') not in codes:
        raise ValueError(f"search result status {r.get('status')!r}")
    return codes[r['status']]


# the stage's result format and the exit code the training tool returns with that result
STAGES = {'search': ('h7r-s400-train-search-v1', _search_code),
          'rows': ('h7r-s400-train-rows-v1', lambda r: 0)}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def experiment_sha256(contract: dict) -> str:
    """``h7r_p1.experiment_sha256``: the canonical JSON of the contract without ``approved``."""
    body = {k: v for k, v in contract.items() if k != 'approved'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False)
                          .encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def head(root) -> str:
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True, capture_output=True,
                          text=True).stdout.strip()


def write(path: Path, body: dict) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(body, indent=1) + '\n', encoding='utf-8')
    tmp.replace(path)


def load(path) -> dict:
    a = json.loads(Path(path).read_text(encoding='utf-8'))
    if a.get('format') != FORMAT or a.get('stage') not in STAGES or not NAME.fullmatch(a.get('name', '')):
        raise ValueError(f'{path}: not an attempt record')
    return a


def last_line(path) -> str:
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    return lines[-1] if lines else ''


def exit_code(line: str):
    m = re.fullmatch(r'__EXIT__ ([0-9]+)', line)
    return int(m.group(1)) if m else None


def attempts(log_dir, stage) -> list[tuple[Path, dict]]:
    return [(p, load(p)) for p in sorted(Path(log_dir).glob(f'{stage}-*.attempt.json'))]


def result_view(stage: str, result) -> tuple[dict, int]:
    r = json.loads(Path(result).read_text(encoding='utf-8'))
    if r.get('format') != STAGES[stage][0]:
        raise ValueError(f'{result}: not a {stage} result')
    return r, STAGES[stage][1](r)


def hashed_outputs(root, a: dict) -> dict:
    return {rel: (sha256(root / rel) if (root / rel).is_file() else None) for rel in a['outputs']}


def close(root, path: Path, a: dict, by: str) -> None:
    """An open attempt of a run that never finished its record: its log ended, its outputs hashed as left."""
    log = root / a['log']
    code = exit_code(last_line(log)) if log.is_file() else None
    if log.is_file() and code is None:
        with log.open('a', encoding='utf-8') as f:
            f.write(f'\n__INTERRUPTED__ no exit line; closed by attempt {by}\n')
    a['finish'] = {'state': 'interrupted', 'exit_code': code, 'log_sha256': sha256(log) if log.is_file() else None,
                   'result_sha256': None, 'outputs': hashed_outputs(root, a), 'closed_by': by, 'finished': now()}
    write(path, a)
    print(f'attempt {a["name"]}: closed as interrupted (exit line {code})', flush=True)


def start(args, root: Path) -> dict:
    log = Path(args.log)
    name = log.name[:-len('.log')] if log.name.endswith('.log') else ''
    if not NAME.fullmatch(name) or not name.startswith(args.stage + '-') or not (root / log).is_file():
        raise ValueError(f'{log}: not a fresh {args.stage}-XXXXXXXX.log')
    path = (root / log).with_name(name + '.attempt.json')
    if path.exists():
        raise ValueError(f'{path} exists: never overwritten')
    if (root / args.result).exists():
        raise ValueError(f'{args.result} exists: recover it, never a new run over it')
    contract = root / args.contract
    c = json.loads(contract.read_text(encoding='utf-8'))
    if c.get('contract_id') != CONTRACT_ID:
        raise ValueError(f'{contract}: not the {CONTRACT_ID} contract')
    for p, a in attempts(path.parent, args.stage):
        if a.get('finish') is None:
            close(root, p, a, name)
    a = {'format': FORMAT, 'contract_id': CONTRACT_ID, 'stage': args.stage, 'name': name,
         'experiment_sha256': experiment_sha256(c), 'contract_file_sha256': sha256(contract), 'head': head(root),
         'log': log.as_posix(), 'result': Path(args.result).as_posix(),
         'outputs': [Path(o).as_posix() for o in args.output], 'set_aside': [], 'started': now(), 'finish': None}
    moves = []
    for rel in args.set_aside:
        src = root / rel
        if not src.is_file():
            raise ValueError(f'{rel}: nothing to set aside')
        dest = Path(args.aside_dir) / f'{name}.{src.name}'
        if (root / dest).exists():
            raise ValueError(f'{dest} exists: never overwritten')
        digest = sha256(src)
        left = [b['name'] for _, b in attempts(path.parent, args.stage)
                if (b.get('finish') or {}).get('outputs', {}).get(Path(rel).as_posix()) == digest]
        a['set_aside'].append({'from': Path(rel).as_posix(), 'to': dest.as_posix(), 'sha256': digest,
                               'left_by': left[-1] if left else None})
        moves.append((src, root / dest))
    write(path, a)  # recorded before anything moves: an interruption here leaves the file in place
    for src, dest in moves:
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dest)
        print(f'set aside: {src.relative_to(root).as_posix()} -> {dest.relative_to(root).as_posix()}', flush=True)
    print(f"attempt {name}: {args.stage}, experiment {a['experiment_sha256'][:12]}, HEAD {a['head'][:12]}", flush=True)
    return a


def finish(root: Path, path: Path, code: int, recovered: bool = False) -> dict:
    a = load(path)
    if a.get('finish') is not None:
        raise ValueError(f'{path}: already finished')
    log = root / a['log']
    if exit_code(last_line(log)) != code:
        raise ValueError(f"{a['log']}: its last line is not __EXIT__ {code}")
    result, result_sha = root / a['result'], None
    if result.exists():
        r, means = result_view(a['stage'], result)
        if (r.get('experiment_sha256'), (r.get('git') or {}).get('commit')) != (a['experiment_sha256'], a['head']):
            raise ValueError(f"{a['result']}: not of this attempt's experiment and HEAD")
        if means != code:
            raise ValueError(f"{a['result']} means exit {means}, the run exited {code}")
        result_sha = sha256(result)
    elif code in (0, 2):
        raise ValueError(f"exit {code} without {a['result']}: the run did not produce its result")
    outputs = hashed_outputs(root, a)
    same = {s['from']: outputs.get(s['from']) == s['sha256'] for s in a['set_aside'] if s['from'] in outputs}
    a['finish'] = {'state': 'exited', 'exit_code': code, 'log_sha256': sha256(log), 'result_sha256': result_sha,
                   'outputs': outputs, 'same_as_set_aside': same, 'recovered': recovered, 'finished': now()}
    write(path, a)
    return a


def recover(args, root: Path) -> tuple[str, int]:
    result = root / args.result
    r, code = result_view(args.stage, result)
    digest = sha256(result)
    found = attempts(root / args.log_dir, args.stage)
    done = [a for _, a in found if (a.get('finish') or {}).get('result_sha256') == digest]
    if done:
        if done[0]['finish']['exit_code'] != code:
            raise ValueError(f"{args.result} means exit {code}, its attempt recorded {done[0]['finish']['exit_code']}")
        return done[0]['name'], code
    mine = [(p, a) for p, a in found if a.get('finish') is None and a['result'] == Path(args.result).as_posix()
            and (a['experiment_sha256'], a['head']) == (r.get('experiment_sha256'), (r.get('git') or {}).get('commit'))]
    if len(mine) != 1:
        raise ValueError(f'{args.result}: {len(mine)} open attempts of its experiment and HEAD (one wrote it): '
                         'nothing recovered')
    path, a = mine[0]
    log = root / a['log']
    last = exit_code(last_line(log))
    if last is None:
        with log.open('a', encoding='utf-8') as f:
            f.write(f"\nRecovered exit code from {a['result']}\n__EXIT__ {code}\n")
    elif last != code:
        raise ValueError('the result and its log disagree on the exit code')
    finish(root, path, code, recovered=True)
    return a['name'], code


def copy(args, root: Path) -> None:
    a = load(args.attempt)
    if a.get('finish') is None:
        raise ValueError(f'{args.attempt}: not finished (run the block again to close or recover it)')
    dest = root / args.dest_dir
    log_copy = dest / f"{a['name']}.log"
    if not log_copy.is_file() or sha256(log_copy) != a['finish']['log_sha256']:
        raise ValueError(f'{log_copy}: missing or not the log the attempt recorded')
    out = dest / f"{a['name']}.attempt.json"
    data = Path(args.attempt).read_bytes()
    if out.exists():
        if out.read_bytes() != data:
            raise ValueError(f'{out}: an existing copy differs (never overwritten)')
    else:
        out.write_bytes(data)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', default='.')
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('start')
    s.add_argument('--stage', required=True, choices=sorted(STAGES))
    s.add_argument('--log', required=True)
    s.add_argument('--result', required=True)
    s.add_argument('--contract', default=CONTRACT)
    s.add_argument('--output', action='append', default=[])
    s.add_argument('--set-aside', action='append', default=[])
    s.add_argument('--aside-dir', default='runs/h7r/R2-S400-TRAIN/interrupted')
    f = sub.add_parser('finish')
    f.add_argument('--attempt', required=True)
    f.add_argument('--rc', required=True, type=int)
    r = sub.add_parser('recover')
    r.add_argument('--stage', required=True, choices=sorted(STAGES))
    r.add_argument('--log-dir', required=True)
    r.add_argument('--result', required=True)
    c = sub.add_parser('copy')
    c.add_argument('--attempt', required=True)
    c.add_argument('--dest-dir', required=True)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    try:
        if args.command == 'start':
            start(args, root)
        elif args.command == 'finish':
            finish(root, Path(args.attempt), args.rc)
        elif args.command == 'recover':
            print(*recover(args, root))
        else:
            copy(args, root)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
