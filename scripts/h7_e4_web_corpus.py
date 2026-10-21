"""H7-E4 web corpus: the positions of the web games where the AI is to move after a human move.

Development and probe data only, never training data (docs/h6-value.md, H7-E4 decisions).

    python scripts/h7_e4_web_corpus.py --db data/web/renju_web_full.db --output docs/h7/h7_e4_web_corpus.json

The raw DB (session hashes, timestamps, log paths, diagnostics) stays under ``data/web/`` and is never
committed; the output carries only what the E4 tools need: per game its split and the actual moves, per
position the human move that led to it, the web AI's reply and (undone positions only; a main position is
a prefix of its game's line, ``history``) the move list.

Every recorded move is rebuilt from the board as it stood when the move was recorded (the moves with a
smaller ``seq`` not yet undone at that time), checked against the recorded ``ply`` and ``board_hash`` and
played through ``renju.Game`` (legality, forbidden points). A game with any failure is left out whole.

Sets:
    main     an active (never undone) human move followed by an active AI move: the AI's real decision
    undone   a human move later undone, the AI to move after it: DEV only (supplementary). Dropped when
             its game is in HOLDOUT (the split stays clean by session) or when the same board and side to
             move is already a main position or an earlier undone one.
Split (fixed before any result): sessions ordered by sha256(session_hash); HOLDOUT takes whole sessions in
that order until it holds at least 25% of the main positions, the rest is DEV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

FORMAT = 'h7-e4-web-corpus-v1'
HOLDOUT_SHARE = 0.25
COLOUR = {'BLACK': 1, 'WHITE': -1}


def board_hash(board, to_play: int) -> str:
    """``scripts/run_web_play.board_hash``: stones and the side to move (not D4-canonical)."""
    raw = json.dumps([[list(line) for line in board], to_play], separators=(',', ':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_db(path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    con = sqlite3.connect(f'file:{path.resolve().as_posix()}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    try:
        games = [dict(r) for r in con.execute(
            'SELECT game_id, session_hash, human_color, agent_name, status, result, winner FROM games '
            'ORDER BY started_at, game_id')]
        moves: dict[str, list[dict]] = {}
        for r in con.execute('SELECT game_id, seq, ply, player, actor, row0, col0, board_hash, recorded_at, route, '
                             'undone_at FROM moves ORDER BY game_id, seq'):
            moves.setdefault(r['game_id'], []).append(dict(r))
    finally:
        con.close()
    return games, moves


def rebuild(rows: list[dict]) -> list[dict]:
    """Per recorded move: the moves on the board just before it; raises ValueError on any mismatch."""
    from renju import Game, IllegalMove

    out = []
    for move in rows:
        on = sorted((r for r in rows if r['seq'] < move['seq']
                     and (r['undone_at'] is None or r['undone_at'] > move['recorded_at'])), key=lambda r: r['ply'])
        if [r['ply'] for r in on] != list(range(1, len(on) + 1)) or move['ply'] != len(on) + 1:
            raise ValueError(f"seq {move['seq']}: ply {move['ply']} does not follow the {len(on)} stones on the board")
        game = Game()
        try:
            for r in on:
                game.play(r['row0'], r['col0'])
            if board_hash(game.board, game.to_play) != move['board_hash']:
                raise ValueError(f"seq {move['seq']}: board_hash does not match the rebuilt board")
            if COLOUR[move['player']] != game.to_play:
                raise ValueError(f"seq {move['seq']}: {move['player']} is not the side to move")
            game.play(move['row0'], move['col0'])
        except IllegalMove as exc:
            raise ValueError(f"seq {move['seq']}: illegal ({exc})") from None
        out.append({**move, 'before': [[r['row0'], r['col0']] for r in on], 'done_after': game.done,
                    'key_after': board_hash(game.board, game.to_play)})
    return out


def is_human(move: dict) -> bool:
    return move['actor'] == 'HUMAN'


def is_ai(move: dict) -> bool:
    return move['actor'] not in ('HUMAN', 'OPENING_RULE')


def positions_of(game: dict, rows: list[dict]) -> list[dict]:
    """Main and undone candidates of one rebuilt game (the split is set later)."""
    out = []
    active = {r['ply']: r for r in rows if r['undone_at'] is None}
    by_seq = {r['seq']: r for r in rows}
    for move in rows:
        if not is_human(move) or move['done_after']:
            continue
        history = [*move['before'], [move['row0'], move['col0']]]
        if move['undone_at'] is None:
            reply = active.get(move['ply'] + 1)
            if reply is None or not is_ai(reply):
                continue
            kind = 'main'
        else:
            reply = by_seq.get(move['seq'] + 1)
            if reply is not None and not (is_ai(reply) and reply['ply'] == move['ply'] + 1):
                reply = None
            kind = 'undone'
        pid = f"{game['game_id']}:{move['ply']}" if kind == 'main' else f"{game['game_id']}:u{move['seq']}"
        out.append({'id': pid, 'game_id': game['game_id'], 'set': kind, 'ply': len(history),
                    'to_play': 'WHITE' if COLOUR[move['player']] == 1 else 'BLACK', 'key': move['key_after'],
                    'human_move': history[-1], **({} if kind == 'main' else {'moves': history}),
                    'web_ai': None if reply is None else {'actor': reply['actor'], 'move': [reply['row0'], reply['col0']],
                                                          'route': reply['route']}})
    return out


def session_order(session_hash: str) -> str:
    return hashlib.sha256(session_hash.encode('utf-8')).hexdigest()


def assign_splits(games: list[dict], main_counts: dict[str, int]) -> dict[str, str]:
    """game_id -> DEV / HOLDOUT: whole sessions in sha256 order until HOLDOUT >= 25% of the main positions."""
    sessions: dict[str, list[str]] = {}
    for g in games:
        sessions.setdefault(g['session_hash'] or f"game:{g['game_id']}", []).append(g['game_id'])
    total = sum(main_counts.values())
    split, holdout = {}, 0
    for session in sorted(sessions, key=session_order):
        side = 'HOLDOUT' if holdout < HOLDOUT_SHARE * total else 'DEV'
        for gid in sessions[session]:
            split[gid] = side
            if side == 'HOLDOUT':
                holdout += main_counts.get(gid, 0)
    return split


def build(games: list[dict], moves: dict[str, list[dict]]) -> dict:
    excluded, candidates, kept_games = {}, [], []
    for g in games:
        try:
            rows = rebuild(moves.get(g['game_id'], []))
        except ValueError as exc:
            excluded[g['game_id']] = str(exc)
            continue
        kept_games.append((g, rows))
        candidates.extend(positions_of(g, rows))
    main_counts = {}
    for p in candidates:
        if p['set'] == 'main':
            main_counts[p['game_id']] = main_counts.get(p['game_id'], 0) + 1
    split = assign_splits([g for g, _ in kept_games], main_counts)
    seen = {p['key'] for p in candidates if p['set'] == 'main'}
    positions, dropped = [], {'undone_in_holdout': 0, 'undone_duplicate': 0}
    for p in candidates:
        p = {'split': split[p['game_id']], **p}
        if p['set'] == 'undone':
            if p['split'] == 'HOLDOUT':
                dropped['undone_in_holdout'] += 1
                continue
            if p['key'] in seen:
                dropped['undone_duplicate'] += 1
                continue
            seen.add(p['key'])
        positions.append(p)
    out_games = {}
    for g, rows in kept_games:
        line = sorted((r for r in rows if r['undone_at'] is None), key=lambda r: r['ply'])
        out_games[g['game_id']] = {'split': split[g['game_id']], 'human_color': g['human_color'],
                                   'agent_name': g['agent_name'], 'status': g['status'], 'result': g['result'],
                                   'moves': [[r['row0'], r['col0']] for r in line]}
    built = {'games': out_games, 'positions': positions, 'excluded_games': excluded, 'dropped': dropped}
    for p in positions:  # a main position must be the prefix of its game's line
        if position_key(history(built, p)) != p['key']:
            raise AssertionError(f"{p['id']}: the stored moves do not give the recorded board")
    return built


def position_key(moves) -> str:
    from renju import Game

    game = Game()
    for move in moves:
        game.play(*move)
    return board_hash(game.board, game.to_play)


def report(corpus: dict) -> dict:
    out = {'games': {}, 'positions': {}, 'holdout_share_of_main': None, 'holdout_keys_also_in_dev': 0}
    for g in corpus['games'].values():
        out['games'][g['split']] = out['games'].get(g['split'], 0) + 1
    keys = {'DEV': set(), 'HOLDOUT': set()}
    for p in corpus['positions']:
        name = f"{p['split']}/{p['set']}"
        out['positions'][name] = out['positions'].get(name, 0) + 1
        keys[p['split']].add(p['key'])
    main = sum(v for k, v in out['positions'].items() if k.endswith('/main'))
    out['holdout_share_of_main'] = round(out['positions'].get('HOLDOUT/main', 0) / main, 4) if main else None
    out['holdout_keys_also_in_dev'] = len(keys['HOLDOUT'] & keys['DEV'])
    return out


def write(corpus: dict, path: Path) -> None:
    """Indented header, one position per line (small diffs, easy grep)."""
    head = {k: v for k, v in corpus.items() if k not in ('games', 'positions')}
    lines = ['{', *(f' {json.dumps(k)}: {json.dumps(v)},' for k, v in head.items()), ' "games": {']
    games = list(corpus['games'].items())
    lines += [f'  {json.dumps(k)}: {json.dumps(v)}' + (',' if i < len(games) - 1 else '') for i, (k, v) in enumerate(games)]
    lines += [' },', ' "positions": [']
    ps = corpus['positions']
    lines += [f'  {json.dumps(p)}' + (',' if i < len(ps) - 1 else '') for i, p in enumerate(ps)]
    lines += [' ]', '}']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def history(corpus: dict, position: dict) -> list[tuple[int, int]]:
    """The moves of a position: a main position is a prefix of its game's actual line."""
    moves = position['moves'] if 'moves' in position else corpus['games'][position['game_id']]['moves'][:position['ply']]
    return [tuple(m) for m in moves]


def load(path: Path) -> dict:
    corpus = json.loads(Path(path).read_text(encoding='utf-8'))
    if corpus.get('format') != FORMAT:
        raise ValueError(f'{path}: not a {FORMAT} file')
    return corpus


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--db', type=Path, required=True, help='the web DB (data/web/, never committed)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'{args.output} exists')
    games, moves = read_db(args.db)
    built = build(games, moves)
    corpus = {'format': FORMAT, 'db_sha256': file_sha256(args.db),
              'rules': {'holdout_share': HOLDOUT_SHARE, 'split': 'sessions by sha256(session_hash), HOLDOUT first',
                        'use': 'development and probes only, never training'},
              'report': None, **built}
    corpus['report'] = report(corpus)
    write(corpus, args.output)
    print(json.dumps({'report': corpus['report'], 'excluded_games': corpus['excluded_games'],
                      'dropped': corpus['dropped']}, indent=1), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
