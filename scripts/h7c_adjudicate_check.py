"""Read-only check of the E2 Rapfi adjudication (v1, ``scripts/h7c_adjudicate.py``): the raw branch records against the
committed result, the shared structure of the two comparisons, the ``abnormal_end`` branches, a sensitivity table.

Nothing here changes the v1 code, records or reading (docs/h7-c1.md, "불일치 국면 판정 결과 검토" and its approval).
Every input is read, never rewritten (the v1 loader would cut an incomplete line: this one only reports it).

    files        ``branches.jsonl`` and ``adjudication.json`` against the committed v1 manifest (SHA-256), the identity
                 SHA-256 recomputed, the v1 code at the recorded commit = the code in this checkout (scripts and rules)
    inputs       the v1 inputs loaded with the v1 loader (every chain check, the pre-registered counts); their SHA-256
                 and the branch list SHA-256 = the identity's
    records      every line complete, this identity, each key once, exactly the branch list (missing / extra), the
                 recorded order = the order seed's, the position / move / mover = the branch's
    replay       every branch replayed with ``Game``: the fixed move and every continuation move legal, the game over
                 exactly at the last move, the recorded winner / score / length / forbid checks reproduced
    statistics   ``compare_stats`` / ``overlap`` of v1 on the records = the committed comparisons, overlap and counts
    shared       per comparison, the observations by the position's role in the other comparison: ``mirror`` (c8 and
                 orig chose the same move: the same two branches, the delta's sign reversed), ``three_distinct``,
                 ``only_here``; positions, observations, source games, better / worse / same / invalid per class;
                 nothing merged or removed
    abnormal     each ``abnormal_end`` branch: side to move, empty points, legal moves, black forbidden points, whether
                 every empty point is a black forbidden point, the observations that use it
    sensitivity  ``abnormal_end`` scored 0.5 (a full-board draw's score): reported apart, never a reading
    timing       worker seconds from the records (sum / mean / median / p90 / max) and, from ``run.log`` when given,
                 the pool's wall time per session (progress lines: 0.1-minute resolution, setup not included)

    python -u scripts/h7c_adjudicate_check.py --run-log runs/h7c/e2_adjudicate/run.log

Output: ``docs/h7c/e2_adjudicate_check/check.json`` and ``manifest.json`` (never overwritten; a dirty tree is refused).
``verdict`` is PASS only when every check passes; a missing or different raw record is a FAIL with its reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src', ROOT / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h7c_adjudicate as ad  # noqa: E402
from renju import BLACK, Game, IllegalMove  # noqa: E402

FORMAT = 'h7c-adjudicate-check-v1'
V1_CODE = ('scripts/h7c_adjudicate.py', 'src/renju/game.py', 'src/renju/rules.py', 'src/agents/external_engine.py')
DOCUMENTED = {'shared_positions': 85, 'mirror_positions': 67}  # docs/h7-c1.md (55333bc): 152 - 85
DEFAULTS = {
    'adjudication': ROOT / 'docs/h7c/e2_adjudicate/adjudication.json',
    'adjudication_manifest': ROOT / 'docs/h7c/e2_adjudicate/manifest.json',
    'jsonl': ROOT / 'runs/h7c/e2_adjudicate/branches.jsonl',
    'output': ROOT / 'docs/h7c/e2_adjudicate_check/check.json',
    'manifest_out': ROOT / 'docs/h7c/e2_adjudicate_check/manifest.json',
}
_PROGRESS = re.compile(r'^\[h7c-adjudicate\] ([\d,]+)/([\d,]+) branches\s+[\d.]+%\s+([\d.]+) min')
_HEADER = re.compile(r'^\[h7c-adjudicate\] \d+ observations .*; (\d+) done, (\d+) to play')


def _json_eq(a, b) -> bool:
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


class Checks:
    def __init__(self):
        self.items: dict[str, dict] = {}

    def add(self, name: str, ok: bool, **detail) -> bool:
        self.items[name] = {'pass': bool(ok), **detail}
        return bool(ok)

    @property
    def failed(self) -> list[str]:
        return [k for k, v in self.items.items() if not v['pass']]


# --- the raw records ---------------------------------------------------------------------------------------------

def read_records(path: Path) -> tuple[list[dict], list[str]]:
    """Every line of the branch file, strictly (nothing cut or repaired): records and the problems found."""
    records, problems = [], []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        problems.append('the file does not end with a newline')
    for n, line in enumerate(raw.split(b'\n')):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            problems.append(f'line {n + 1}: not JSON')
            continue
        missing = [k for k in ad.RECORD_KEYS if k not in record]
        if missing or record.get('status') not in ('VALID', 'INVALID'):
            problems.append(f'line {n + 1}: incomplete record ({", ".join(missing) or "status"})')
            continue
        records.append(record)
    return records, problems


def replay(record: dict, moves: list) -> dict:
    """One branch record replayed with ``Game``: what the record says against what the rules give."""
    from agents.external_engine import our_forbidden_points
    from renju.rules import legal_black_points

    game = Game()
    for move in moves:
        game.play(*move)
    mover = game.to_play
    out = {'key': record['key'], 'problems': []}
    sequence = [('fixed', record['move'])] + [('continuation', m) for m in record.get('continuation', [])]
    black_turns = 0
    for n, (what, move) in enumerate(sequence):
        if game.done:
            out['problems'].append(f'{what} move {n} after the game was over')
            break
        if what == 'continuation' and game.to_play == BLACK:
            black_turns += 1
        try:
            game.play(*move)
        except IllegalMove as exc:
            out['problems'].append(f'{what} move {n} {move} illegal: {exc}')
            break
    out['length'] = len(game.history)
    empties = sum(row.count(0) for row in game.board)
    out.update(done=game.done, winner=None if game.winner is None else 'black' if game.winner == BLACK else 'white',
               to_play='black' if game.to_play == BLACK else 'white', empty_points=empties)
    if record['status'] == 'VALID':
        if not game.done:
            out['problems'].append('VALID but the game is not over')
        elif game.winner is None and empties:
            out['problems'].append('VALID draw with empty points left')
        else:
            if out['winner'] != record.get('winner'):
                out['problems'].append(f"winner {out['winner']} != recorded {record.get('winner')}")
            if ad.score_for(mover, game) != record['score']:
                out['problems'].append(f"score {ad.score_for(mover, game)} != recorded {record['score']}")
        if record.get('forbid_checked') != black_turns:
            out['problems'].append(f"forbid checks {record.get('forbid_checked')} != black turns {black_turns}")
    if record['length'] != out['length'] and not out['problems']:
        out['problems'].append(f"length {out['length']} != recorded {record['length']}")
    if record.get('reason') == 'abnormal_end':
        forbidden = our_forbidden_points(game)
        legal = len(legal_black_points(game.board)) if game.to_play == BLACK else empties
        out['abnormal'] = {'to_play': out['to_play'], 'empty_points': empties, 'legal_moves': legal,
                           'black_forbidden_points': len(forbidden), 'continuation': len(sequence) - 1,
                           'every_empty_point_black_forbidden': game.to_play == BLACK and len(forbidden) == empties,
                           'winner': out['winner'], 'over': game.done}
        explained = (game.done and game.winner is None and game.to_play == BLACK and empties > 0
                     and legal == 0 and len(forbidden) == empties)
        if not explained:
            out['problems'].append('abnormal_end is not "black to move, every empty point forbidden"')
    return out


# --- the shared structure ----------------------------------------------------------------------------------------

def _outcome(o: dict, results: dict) -> str:
    ra, rb = (results.get(ad.branch_key(o['moves'], o[f'move_{s}'])) for s in ('a', 'b'))
    if ra is None or rb is None:
        return 'missing'
    if ra['status'] != 'VALID' or rb['status'] != 'VALID':
        return 'invalid'
    delta = ra['score'] - rb['score']
    return 'better' if delta > 0 else 'worse' if delta < 0 else 'same'


def shared_structure(obs: list[dict], results: dict) -> dict:
    """Observations of each comparison by the role of their position in the other one (nothing merged)."""
    names = sorted({o['compare'] for o in obs})
    if len(names) != 2:
        raise ValueError(f'the shared structure needs two comparisons, got {names}')
    moves_by = {}  # (comparison, position) -> {model: set of moves}
    for o in obs:
        at = moves_by.setdefault((o['compare'], ad.sha_json(o['moves'])), {})
        at.setdefault(o['a'], set()).add(tuple(o['move_a']))
        at.setdefault(o['b'], set()).add(tuple(o['move_b']))
    inconsistent = sorted({p for (_, p), at in moves_by.items() if any(len(v) > 1 for v in at.values())})
    out, mirror_keys = {'inconsistent_positions': len(inconsistent)}, {}
    for name in names:
        other = names[1] if name == names[0] else names[0]
        classes = {}
        for o in obs:
            if o['compare'] != name:
                continue
            position = ad.sha_json(o['moves'])
            there = moves_by.get((other, position))
            if there is None:
                cls = 'only_here'
            else:
                mine = moves_by[(name, position)]
                shared = set(mine) & set(there)  # the model both comparisons contain (c5)
                if len(shared) != 1 or mine[next(iter(shared))] != there[next(iter(shared))]:
                    cls = 'inconsistent'
                else:
                    own = {m for k, v in mine.items() if k not in shared for m in v}
                    theirs = {m for k, v in there.items() if k not in shared for m in v}
                    cls = 'mirror' if own == theirs else 'three_distinct'
            if cls == 'mirror':
                mirror_keys.setdefault(name, set()).add(position)
            c = classes.setdefault(cls, {'positions': set(), 'observations': 0, 'source_games': set(),
                                         'better': 0, 'worse': 0, 'same': 0, 'invalid': 0, 'missing': 0})
            c['positions'].add(position)
            c['observations'] += 1
            c['source_games'].add(o['game_index'])
            c[_outcome(o, results)] += 1
        total = sum(c['observations'] for c in classes.values())
        out[name] = {cls: {**{k: v for k, v in c.items() if k not in ('positions', 'source_games')},
                           'positions': len(c['positions']), 'source_games': len(c['source_games']),
                           'observation_share': round(c['observations'] / total, 5)}
                     for cls, c in sorted(classes.items())}
    shared = {p for (n, p) in moves_by if n == names[0]} & {p for (n, p) in moves_by if n == names[1]}
    out['shared_positions'] = len(shared)
    out['mirror_positions'] = len(mirror_keys.get(names[0], set()))
    out['mirror_positions_same_both_ways'] = mirror_keys.get(names[0], set()) == mirror_keys.get(names[1], set())
    # in a mirror position both comparisons use the same two branches: their deltas are exact opposites
    opposite = 0
    for position in mirror_keys.get(names[0], set()):
        deltas = {}
        for o in obs:
            if ad.sha_json(o['moves']) == position:
                ra, rb = (results.get(ad.branch_key(o['moves'], o[f'move_{s}'])) for s in ('a', 'b'))
                if ra and rb and ra['status'] == rb['status'] == 'VALID':
                    deltas.setdefault(o['compare'], set()).add(ra['score'] - rb['score'])
        if len(deltas) == 2 and all(len(v) == 1 for v in deltas.values()) and \
                next(iter(deltas[names[0]])) == -next(iter(deltas[names[1]])):
            opposite += 1
        elif len(deltas) < 2:
            opposite += 1  # an invalid branch: nothing to compare (counted with the mirrors)
    out['mirror_deltas_opposite'] = opposite
    return out


# --- timing ------------------------------------------------------------------------------------------------------

def timing(records: list[dict], run_log: Path | None) -> dict:
    seconds = sorted(r['seconds'] for r in records if isinstance(r.get('seconds'), (int, float)))
    out = {'worker_seconds': None, 'run_log': None}
    if seconds:
        p90 = seconds[min(len(seconds) - 1, int(0.9 * (len(seconds) - 1) + 0.5))]
        out['worker_seconds'] = {'branches': len(seconds), 'sum': round(sum(seconds), 1),
                                 'mean': round(statistics.fmean(seconds), 2),
                                 'median': round(statistics.median(seconds), 2), 'p90': round(p90, 2),
                                 'max': round(seconds[-1], 2),
                                 'note': 'per branch, inside the worker (engine start to the end of the game)'}
    if run_log is not None and Path(run_log).exists():
        sessions, current = [], None
        for line in Path(run_log).read_text(encoding='utf-8', errors='replace').splitlines():
            header, progress = _HEADER.match(line), _PROGRESS.match(line)
            if header:
                current = {'done_before': int(header[1]), 'to_play': int(header[2]), 'last_progress': None,
                           'elapsed_minutes': None}
                sessions.append(current)
            elif progress and current is not None:
                current['last_progress'] = f'{progress[1]}/{progress[2]}'
                current['elapsed_minutes'] = float(progress[3])
        known = [s['elapsed_minutes'] for s in sessions if s['elapsed_minutes'] is not None]
        out['run_log'] = {'file_sha256': ad.file_sha(run_log), 'sessions': sessions,
                          'pool_wall_minutes': round(sum(known), 1) if known else None,
                          'note': 'progress lines: elapsed of the process pool, 0.1-minute resolution, '
                                  'input loading and the final statistics not included'}
        if known and out['worker_seconds']:
            out['parallelism'] = round(out['worker_seconds']['sum'] / (60 * sum(known)), 2)
    return out


# --- the check ---------------------------------------------------------------------------------------------------

def _git_show_sha(commit: str, path: str) -> str | None:
    try:
        blob = subprocess.run(['git', 'show', f'{commit}:{path}'], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(blob).hexdigest()


def check(args, log=print) -> dict:
    c = Checks()
    out = {'format': FORMAT, 'meaning': 'a read-only check of the v1 adjudication; descriptive, never a reading'}
    result = json.loads(Path(args.adjudication).read_text(encoding='utf-8'))
    manifest = json.loads(Path(args.adjudication_manifest).read_text(encoding='utf-8'))
    identity = result['identity']
    c.add('adjudication.json = the v1 manifest',  # the v1 manifest names its files by repository path
          manifest.get('document_files_sha256', {}).get(ad._rel(args.adjudication)) == ad.file_sha(args.adjudication))
    c.add('identity SHA-256 recomputed', ad.sha_json(identity) == result['identity_sha256'] ==
          manifest.get('identity_sha256'))
    c.add('a full, clean run', identity.get('limit') is None and identity.get('max_plies') is None
          and identity['git'].get('dirty') is False and manifest.get('code_git') == identity['git']
          and identity['engine'].get('registered_engine') is True,
          limit=identity.get('limit'), dirty=identity['git'].get('dirty'))
    commit = identity['git']['commit']
    code = {p: {'recorded': _git_show_sha(commit, p), 'checkout': ad.file_sha(ROOT / p)} for p in V1_CODE}
    c.add('v1 code at the recorded commit = this checkout', all(v['recorded'] == v['checkout'] for v in code.values()),
          commit=commit, files={p: v['recorded'] == v['checkout'] for p, v in code.items()})
    jsonl = Path(args.jsonl)
    expected_jsonl = manifest.get('files_sha256', {}).get(ad._rel(jsonl))
    if not jsonl.exists():
        c.add('branches.jsonl present', False, reason=f'{jsonl} not found')
        return finish(out, c)
    jsonl_sha = ad.file_sha(jsonl)
    if not c.add('branches.jsonl = the v1 manifest', jsonl_sha == expected_jsonl == result['jsonl_sha256'],
                 actual=jsonl_sha, manifest=expected_jsonl):
        return finish(out, c)
    log('[h7c-adjudicate-check] files match the manifest; loading the v1 inputs', flush=True)

    inputs = ad.load_inputs(args.manifest, args.summary, args.sensitivity_jsonl, args.selfplay, args.verdict,
                            ad.EXPECTED_DISAGREEMENTS)
    c.add('v1 inputs = the identity\'s', inputs['shas'] == identity['inputs'], counts=inputs['counts'])
    obs = inputs['observations']
    table = ad.branch_table(obs)
    order = {k: i for i, k in enumerate(ad.order(table))}
    c.add('branch list = the identity\'s', ad.sha_json(sorted(table)) == identity['branches_sha256'],
          branches=len(table))

    records, problems = read_records(jsonl)
    keys = [r['key'] for r in records]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    c.add('every line complete', not problems, problems=problems[:20])
    c.add('every record of this identity', all(r['identity_sha256'] == result['identity_sha256'] for r in records))
    c.add('each branch once', not duplicates, duplicates=duplicates[:20])
    c.add('exactly the branch list', set(keys) == set(table), missing=len(set(table) - set(keys)),
          extra=len(set(keys) - set(table)))
    by_key = {r['key']: r for r in records}
    mismatch = []
    for r in records:
        t = table.get(r['key'])
        if t is None:
            continue
        game = Game()
        for m in t['moves']:
            game.play(*m)
        if (r['order'], r['move'], r['ply'], r.get('mover')) != (
                order[r['key']], t['move'], len(t['moves']), 'black' if game.to_play == BLACK else 'white'):
            mismatch.append(r['key'][:12])
    c.add('order / position / move / mover = the branch', not mismatch, mismatched=mismatch[:20])

    log(f'[h7c-adjudicate-check] replaying {len(records)} branches', flush=True)
    replays = {r['key']: replay(r, table[r['key']]['moves']) for r in records if r['key'] in table}
    bad = {k[:12]: v['problems'] for k, v in replays.items() if v['problems'] and 'abnormal' not in v}
    c.add('every branch replays with Game', not bad, problems=dict(list(bad.items())[:20]))

    names = [f'{a}:{b}' for a, b in ad.COMPARES]
    stats = [ad.compare_stats(n, [o for o in obs if o['compare'] == n], by_key) for n in names]
    no_reading = (['LIMITED_RUN'] if identity.get('limit') is not None else []) + \
        ([] if identity['engine'].get('registered_engine') else ['UNREGISTERED_ENGINE'])
    for s in stats if no_reading else []:  # what v1 ``run`` does after the statistics
        s['reading'], s['reasons'] = 'SMOKE', no_reading
    c.add('comparisons reproduced', _json_eq(stats, result['comparisons']))
    c.add('overlap reproduced', _json_eq(ad.overlap(obs), result['overlap']))
    invalid = {}
    for r in records:
        if r['status'] == 'INVALID':
            invalid[r['reason']] = invalid.get(r['reason'], 0) + 1
    c.add('branch counts reproduced', _json_eq({'unique': len(table), 'played': len(by_key),
                                                'invalid_by_reason': invalid}, result['branches']))

    shared = shared_structure(obs, by_key)
    reproduced = {k: shared[k] for k in DOCUMENTED}
    c.add('shared structure consistent', shared['inconsistent_positions'] == 0
          and shared['mirror_positions_same_both_ways']
          and shared['mirror_deltas_opposite'] == shared['mirror_positions'])
    out['shared'] = {**shared, 'documented': DOCUMENTED, 'documented_matches': reproduced == DOCUMENTED}

    abnormal = {k: v for k, v in replays.items() if 'abnormal' in v}
    uses = {}
    for o in obs:
        for s in ('a', 'b'):
            key = ad.branch_key(o['moves'], o[f'move_{s}'])
            if key in abnormal:
                uses.setdefault(key, []).append({'compare': o['compare'], 'game_index': o['game_index'],
                                                 'ply': o['ply'], 'side': o[s]})
    c.add('abnormal_end = black to move, every empty point forbidden',
          all(not v['problems'] for v in abnormal.values()),
          problems={k[:12]: v['problems'] for k, v in abnormal.items() if v['problems']})
    out['abnormal_end'] = {'branches': len(abnormal),
                           'rows': [{'key': k, 'ply': by_key[k]['ply'], 'mover': by_key[k]['mover'],
                                     'length': v['length'], **v['abnormal'], 'observations': uses.get(k, [])}
                                    for k, v in sorted(abnormal.items(), key=lambda kv: by_key[kv[0]]['order'])]}

    as_draw = {k: ({**r, 'status': 'VALID', 'score': 0.5} if r.get('reason') == 'abnormal_end' else r)
               for k, r in by_key.items()}
    sensitivity = []
    for n in names:
        s = ad.compare_stats(n, [o for o in obs if o['compare'] == n], as_draw)
        s['sensitivity_reading'], s['sensitivity_reasons'] = s.pop('reading'), s.pop('reasons')
        sensitivity.append(s)
    out['sensitivity_abnormal_end_as_draw'] = {
        'meaning': 'abnormal_end branches scored 0.5; reported only: the v1 reading stays as recorded',
        'not_a_reading': True, 'comparisons': sensitivity}
    out['v1_readings'] = {s['comparison']: s['reading'] for s in result['comparisons']}
    out['timing'] = timing(records, args.run_log)
    out['inputs'] = {'adjudication_sha256': ad.file_sha(args.adjudication),
                     'adjudication_manifest_sha256': ad.file_sha(args.adjudication_manifest),
                     'branches_jsonl_sha256': jsonl_sha, 'v1_inputs': inputs['shas'], 'v1_code': code}
    return finish(out, c)


def finish(out: dict, c: Checks) -> dict:
    out['checks'] = c.items
    out['failed'] = c.failed
    out['verdict'] = 'PASS' if not c.failed else 'FAIL'
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key in ('adjudication', 'adjudication_manifest', 'jsonl'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path, default=DEFAULTS[key])
    for key in ('manifest', 'summary', 'sensitivity_jsonl', 'selfplay', 'verdict'):
        parser.add_argument('--' + key.replace('_', '-'), type=Path, default=ad.DEFAULTS[key])
    parser.add_argument('--run-log', type=Path, help='runs/h7c/e2_adjudicate/run.log (wall time; optional)')
    parser.add_argument('--output', type=Path, default=DEFAULTS['output'])
    parser.add_argument('--manifest-out', type=Path, default=DEFAULTS['manifest_out'])
    parser.add_argument('--allow-dirty', action='store_true', help='tests only')
    return parser


def main(argv=None) -> int:
    from h7c_arena import _git

    parser = build_parser()
    args = parser.parse_args(argv)
    for path in (args.output, args.manifest_out):
        if Path(path).exists():
            parser.error(f'{path} exists: a check is never overwritten')
    git = _git()
    if git['dirty'] is not False and not args.allow_dirty:
        parser.error('the tree is dirty: the check must name its code commit')
    try:
        result = check(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(f'the check could not run: {exc}')
    result['git'] = git
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    files = {ad._rel(p): ad.file_sha(p) for p in (args.jsonl, args.run_log) if p is not None and Path(p).exists()}
    manifest = {'format': f'{FORMAT}-manifest', 'code_git': git, 'verdict': result['verdict'],
                'files_sha256': files, 'document_files_sha256': {ad._rel(args.output): ad.file_sha(args.output)},
                'inputs_sha256': result.get('inputs')}
    args.manifest_out.write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({'verdict': result['verdict'], 'failed': result['failed'],
                      'shared': {k: result.get('shared', {}).get(k) for k in DOCUMENTED},
                      'abnormal_end': result.get('abnormal_end', {}).get('branches')}, indent=1), flush=True)
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
