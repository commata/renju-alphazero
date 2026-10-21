"""H7-E4 tools: web corpus (rebuild, sets, session split), truth grades, the baseline punish check."""
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from renju import Game
from scripts import h7_e4_baseline as base
from scripts import h7_e4_truth as truth
from scripts import h7_e4_web_corpus as web

# Black (the human) moved last, White (the AI) is to move.
FIVE = [(7, 7), (5, 5), (5, 4), (5, 6), (0, 0), (5, 7), (0, 14), (5, 8), (14, 14)]  # White plays (5, 9)
VCF = [(7, 7), (5, 5), (0, 0), (5, 6), (0, 14), (5, 7), (14, 0), (6, 8), (14, 14), (7, 8), (12, 3)]  # (5, 8): 4x4
SCHEMA = """
CREATE TABLE games (game_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT, status TEXT, result TEXT,
    winner TEXT, agent_name TEXT, human_color TEXT, session_hash TEXT);
CREATE TABLE moves (id INTEGER PRIMARY KEY, game_id TEXT, seq INTEGER, ply INTEGER, player TEXT, actor TEXT,
    row0 INTEGER, col0 INTEGER, board_hash TEXT, recorded_at TEXT, route TEXT, undone_at TEXT);
"""


def make_db(path: Path, games: list[dict]) -> None:
    """``games``: {id, session, human, moves: [(r, c)], undone: {seq: [(r, c) replayed after the undo]}}."""
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    clock = iter(f'2026-10-20T00:{i // 60:02d}:{i % 60:02d}' for i in range(10_000))
    for g in games:
        con.execute('INSERT INTO games VALUES (?, ?, NULL, ?, NULL, NULL, ?, ?, ?)',
                    (g['id'], next(clock), 'finished', 'TB-RT30-v3', g['human'], g['session']))
        game, seq, rows = Game(), 0, []

        def record(move, undone=False):
            nonlocal seq
            player = 'BLACK' if game.to_play == 1 else 'WHITE'
            actor = 'OPENING_RULE' if not game.history else ('HUMAN' if player == g['human'] else 'TB-RT30-v3')
            rows.append([g['id'], seq, len(game.history) + 1, player, actor, move[0], move[1],
                         web.board_hash(game.board, game.to_play), next(clock), 'tree', None])
            seq += 1
            game.play(*move)
            return rows[-1]

        for i, move in enumerate(g['moves']):
            for wrong in g.get('undone', {}).get(i, []):  # a move played and then taken back
                rows_wrong = record(wrong)
                game.undo()
                rows_wrong[-1] = next(clock)
            record(move)
        for r in rows:
            con.execute('INSERT INTO moves (game_id, seq, ply, player, actor, row0, col0, board_hash, recorded_at, '
                        'route, undone_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', r)
    con.commit()
    con.close()


def corpus_of(tmp: Path, games: list[dict]) -> Path:
    db, out = tmp / 'web.db', tmp / 'corpus.json'
    make_db(db, games)
    assert web.main(['--db', str(db), '--output', str(out)]) == 0
    return out


GAMES = [{'id': 'g1', 'session': 's1', 'human': 'BLACK', 'moves': FIVE + [(5, 9)], 'undone': {4: [(1, 1)]}},
         {'id': 'g2', 'session': 's2', 'human': 'BLACK', 'moves': VCF + [(5, 8)]},
         {'id': 'g3', 'session': 's2', 'human': 'WHITE', 'moves': [(7, 7), (7, 8), (8, 8), (6, 6)]}]


class CorpusTest(unittest.TestCase):
    def test_sets_split_and_moves(self):
        with TemporaryDirectory() as tmp:
            corpus = web.load(corpus_of(Path(tmp), GAMES))
        by_id = {p['id']: p for p in corpus['positions']}
        mains = [p for p in corpus['positions'] if p['set'] == 'main']
        self.assertEqual(len(mains), 4 + 5 + 1)  # human moves followed by an AI move
        last = by_id[f'g1:{len(FIVE)}']
        self.assertEqual(web.history(corpus, last), FIVE)
        self.assertEqual((last['to_play'], last['web_ai']['move']), ('WHITE', [5, 9]))
        self.assertEqual(web.history(corpus, by_id['g3:2']), [(7, 7), (7, 8)])  # human White's first move
        # whole sessions; s1 (4 main positions) or s2 (6) goes first, by sha256 order, until >= 25%
        self.assertEqual(corpus['games']['g2']['split'], corpus['games']['g3']['split'])
        self.assertNotEqual(corpus['games']['g1']['split'], corpus['games']['g2']['split'])
        text = json.dumps({k: corpus[k] for k in ('games', 'positions')})
        self.assertFalse('"s1"' in text or '"s2"' in text or 'session' in text or 'recorded_at' in text)
        undone = [p for p in corpus['positions'] if p['set'] == 'undone']
        if corpus['games']['g1']['split'] == 'DEV':
            self.assertEqual(len(undone), 1)
            self.assertEqual(web.history(corpus, undone[0]), [*FIVE[:4], (1, 1)])
            self.assertIsNone(undone[0]['web_ai'])
        else:
            self.assertEqual((undone, corpus['dropped']['undone_in_holdout']), ([], 1))

    def test_increment_keeps_committed_splits(self):
        g4 = {'id': 'g4', 'session': 's3', 'human': 'WHITE', 'moves': [(7, 7), (6, 6), (8, 8), (9, 9)]}
        g5 = {'id': 'g5', 'session': 's2', 'human': 'WHITE', 'moves': [(7, 7), (7, 6), (8, 8), (5, 5)]}
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            first = corpus_of(tmp, GAMES[:2])
            make_db(tmp / 'more.db', [*GAMES[:2], g4, g5])
            out = tmp / 'more.json'
            self.assertEqual(web.main(['--db', str(tmp / 'more.db'), '--previous', str(first), '--output', str(out)]), 0)
            a, b = web.load(first), web.load(out)
            make_db(tmp / 'less.db', GAMES[1:2])
            with self.assertRaises(SystemExit):  # a committed game is gone
                web.main(['--db', str(tmp / 'less.db'), '--previous', str(first), '--output', str(tmp / 'x.json')])
        for gid in ('g1', 'g2'):
            self.assertEqual(b['games'][gid]['split'], a['games'][gid]['split'])
        self.assertEqual(b['games']['g5']['split'], a['games']['g2']['split'])  # an old session keeps its split
        self.assertEqual(b['games']['g4']['split'], 'HOLDOUT')  # the only new session: first in its own order
        self.assertTrue(all(p in b['positions'] for p in a['positions']))
        self.assertEqual(b['increments'][0]['games'], 2)

    def test_a_bad_record_drops_the_game(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            make_db(tmp / 'web.db', GAMES)
            con = sqlite3.connect(tmp / 'web.db')
            con.execute("UPDATE moves SET board_hash = 'x' WHERE game_id = 'g2' AND seq = 3")
            con.commit()
            con.close()
            web.main(['--db', str(tmp / 'web.db'), '--output', str(tmp / 'c.json')])
            corpus = web.load(tmp / 'c.json')
        self.assertIn('g2', corpus['excluded_games'])
        self.assertFalse(any(p['game_id'] == 'g2' for p in corpus['positions']))


class TruthTest(unittest.TestCase):
    def test_grades(self):
        g = truth.grade_of
        self.assertEqual(g(['UNSAFE'] * 3, ('five', (1, 1))), ('PROVEN_WIN', 'WIN_5'))
        self.assertEqual(g(['UNSAFE'] * 3, ('vcf', ((1, 1),))), ('PROVEN_WIN', 'WIN_VCF'))
        self.assertEqual(g(['SAFE', 'UNSAFE', 'UNSAFE'], ('threat', (1, 1))), ('PROVEN_WIN', 'WIN_VCT1'))
        self.assertEqual(g(['UNKNOWN', 'UNSAFE', 'UNSAFE'], ('threat', (1, 1))), ('PROVEN_WIN', 'WIN_VCT1_OR_LESS'))
        self.assertEqual(g(['SAFE', 'SAFE', 'UNSAFE'], ('threat', (1, 1))), ('PROVEN_WIN', 'WIN_VCT2_ONLY'))
        self.assertEqual(g(['SAFE', 'UNKNOWN', 'UNSAFE'], ('threat', (1, 1))), ('PROVEN_WIN', 'WIN_VCT2_OR_LESS'))
        self.assertEqual(g(['SAFE', 'SAFE', 'SAFE'], None), ('NOT_FOUND', None))
        self.assertEqual(g(['SAFE', 'SAFE', 'UNKNOWN'], None), ('UNKNOWN', None))  # never a negative

    def test_solve(self):
        r = truth.solve(FIVE, truth.DEFAULT_BUDGET)
        self.assertEqual((r['class'], r['grade'], r['witness']), ('PROVEN_WIN', 'WIN_5', ['five', [5, 9]]))
        r = truth.solve(VCF, truth.DEFAULT_BUDGET)
        self.assertEqual((r['class'], r['grade'], r['statuses']), ('PROVEN_WIN', 'WIN_VCF', ['UNSAFE'] * 3))
        game = Game()
        for m in VCF:
            game.play(*m)
        self.assertIn(tuple(r['witness'][1]), game.legal_moves())
        tiny = {'node_limit': 10, 'call_limit': 1, 'node_budget': 10}
        self.assertEqual(truth.solve(VCF[:5], tiny)['class'], 'UNKNOWN')

    def test_quiet_moves_are_the_full_set(self):
        from analysis.threats import ordered_moves
        game = Game()
        for m in VCF:
            game.play(*m)
        solver = truth._truth_solver_class()(node_limit=10, call_limit=10, node_budget=10)
        quiet = solver.quiet_moves(game)
        self.assertEqual(sorted(quiet), sorted(ordered_moves(game)))
        self.assertEqual(len(quiet), len(set(quiet)))
        from analysis.selective_vct import threat_moves
        threats = threat_moves(game)
        self.assertIn((5, 8), threats)
        self.assertEqual(quiet[:len(threats)], threats)  # threat moves first, then the rest

    def test_resume_refuses_another_budget(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 't.jsonl'
            path.write_text(json.dumps({'key': 'k', 'budget': {'node_limit': 1}}) + '\n{"torn', encoding='utf-8')
            with self.assertRaises(ValueError):
                truth.load_done(path, truth.DEFAULT_BUDGET)
            done, torn = truth.load_done(path, {'node_limit': 1})
            self.assertEqual((list(done), torn), (['k'], 1))


class CheckMoveTest(unittest.TestCase):
    def test_punish_and_miss(self):
        self.assertEqual(base.check_move(FIVE, (5, 9), truth.DEFAULT_BUDGET)['outcome'], 'PUNISH')
        self.assertEqual(base.check_move(VCF, (5, 8), truth.DEFAULT_BUDGET)['outcome'], 'PUNISH')
        self.assertEqual(base.check_move(FIVE, (0, 7), truth.DEFAULT_BUDGET)['outcome'], 'MISS')
        tiny = {'node_limit': 10, 'call_limit': 1, 'node_budget': 10}
        self.assertEqual(base.check_move(FIVE, (0, 7), tiny)['outcome'], 'UNCLEAR')


class RemainingDepthTest(unittest.TestCase):
    def test_a_forced_block_keeps_depth_two(self):
        four = [(7, 7), (7, 6), (7, 8), (0, 0), (7, 9), (0, 2), (7, 10)]  # Black's four, White to move
        game = Game()
        for m in four:
            game.play(*m)
        self.assertEqual(base.remaining_depth(game, (7, 11)), 2)
        self.assertEqual(base.remaining_depth(game, (0, 4)), 1)
        game = Game()
        for m in FIVE[:-1]:  # White just made a four: any move but its block uses a depth
            game.play(*m)
        self.assertEqual(base.remaining_depth(game, (1, 1)), 1)


class LossReviewTest(unittest.TestCase):
    def test_walk_and_status(self):
        from scripts import h7_web_loss_review as review
        lost = [(7, 7), (0, 0), (7, 8), (0, 2), (7, 9), (0, 4), (7, 10), (0, 6), (7, 11)]  # the human (Black) wins
        game = {'id': 'g9', 'session': 's9', 'human': 'BLACK', 'moves': lost}
        self.assertEqual(review.ai_plies({'human_color': 'BLACK', 'moves': lost}), [2, 4, 6, 8])
        self.assertEqual(review.ai_plies({'human_color': 'WHITE', 'moves': lost}), [3, 5, 7, 9])
        self.assertEqual(review.depth1_status(lost[:5], (0, 4)), 'LOST_D0')  # the open three becomes an open four
        self.assertEqual(review.depth1_status(lost[:5], (7, 6)), 'SAFE')
        with TemporaryDirectory() as tmp:
            path = corpus_of(Path(tmp), [game])
            out = Path(tmp) / 'r.json'
            self.assertEqual(review.main(['--corpus', str(path), '--games', 'g9', '--max-steps', '2', '--fast', 'off',
                                          '--output', str(out)]), 0)
            row = json.loads(out.read_text(encoding='utf-8'))['games'][0]
        self.assertEqual([(w['ply'], w['grade']) for w in row['walk']], [(8, 'WIN_5'), (6, 'WIN_VCF')])
        self.assertIsNone(row['first_loss'])  # the walk did not reach a move that is not lost


class BaselineTest(unittest.TestCase):
    def test_end_to_end(self):
        from rt30 import fast4_forbidden
        from tests.test_h7_timing import FACTORIES
        if not fast4_forbidden.available():
            self.skipTest('python -m rt30.native.build first')
        from scripts.run_mcts_v8_benchmark import file_sha256
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = corpus_of(tmp, GAMES[:1])
            corpus = web.load(path)
            p = next(p for p in corpus['positions'] if p['id'] == f'g1:{len(FIVE)}')
            boards = {p['key']: {'key': p['key'], 'grade': 'WIN_5', 'class': 'PROVEN_WIN'}}
            (tmp / 'truth.json').write_text(json.dumps({'format': truth.FORMAT, 'corpus_sha256': file_sha256(path),
                                                        'budget': truth.DEFAULT_BUDGET, 'boards': boards}))
            out = tmp / 'base.json'
            self.assertEqual(base.main(['--corpus', str(path), '--truth', str(tmp / 'truth.json'), '--device', 'cpu',
                                        '--output', str(out)], factories=FACTORIES), 0)
            result = json.loads(out.read_text(encoding='utf-8'))
        row = result['positions'][p['id']]
        self.assertEqual((row['b']['move'], row['b']['outcome'], row['web_ai']['outcome']), ([5, 9], 'PUNISH', 'PUNISH'))
        split = corpus['games']['g1']['split']
        self.assertEqual(result['summary']['by_split'][f'{split}/main']['WIN_5']['b']['PUNISH'], 1)
        self.assertEqual((result['summary']['shallow_flags'], result['summary']['holdout_gate']), ([], 'report-only'))
        from hybrid.h6_fast import installed_mode
        self.assertEqual(installed_mode(), 'off')  # the run does not leave fast4 installed


if __name__ == '__main__':
    unittest.main()


class E4AttackTest(unittest.TestCase):
    def test_trigger_and_vcf_threat_candidates(self):
        from analysis import e4_attack as e4
        game = Game()
        for m in VCF:
            game.play(*m)
        self.assertFalse(e4.opponent_threat(game))  # Black's (12, 3) made nothing
        self.assertEqual(game.history[-1], VCF[-1])  # the game is restored
        three = Game()
        for m in [(7, 7), (0, 0), (7, 8), (0, 2), (7, 9)]:
            three.play(*m)
        self.assertTrue(e4.opponent_threat(three))  # an open three
        self.assertEqual(len(e4.GRID), 8)
        self.assertEqual({c['name'] for c in e4.GRID}, {f'{b}/{c}/{t}' for b in ('200k', '400k')
                                                        for c in e4.CANDIDATES for t in e4.TRIGGERS})

    def test_attack_proves_a_win_and_is_checked(self):
        from analysis import e4_attack as e4
        game = Game()
        for m in VCF:
            game.play(*m)
        for candidates in e4.CANDIDATES:
            move, info = e4.e4_attack(game, {'budget': '200k', 'candidates': candidates})
            self.assertIsNotNone(move)
            self.assertEqual(base.check_move(VCF, move, truth.DEFAULT_BUDGET)['outcome'], 'PUNISH')
        self.assertEqual(game.history, VCF)


class DevSelectionTest(unittest.TestCase):
    def score(self, name, punish, shallow=0, false=0, nodes=10.0):
        return {'config': {'name': name}, 'punish': punish, 'eligible': not (shallow or false),
                'cost': {'nodes_per_position': nodes}}

    def test_rule(self):
        from scripts import h7_e4_dev as dev
        scores = [self.score('a', 5, nodes=50), self.score('b', 6, false=1, nodes=1), self.score('c', 4, nodes=20),
                  self.score('d', 3, nodes=1)]
        out = dev.select(scores)
        self.assertEqual(out['selected']['name'], 'c')  # b has a false proof; c ties a (within 1) and is cheaper
        self.assertEqual(out['tied'], ['a', 'c'])
        self.assertIsNone(dev.select([self.score('x', 3, shallow=1)])['selected'])

    def test_holdout_never_reaches_the_score(self):
        from scripts import h7_e4_dev as dev
        with self.assertRaises(AssertionError):
            dev.dev_only({'id': 'g:3', 'split': 'HOLDOUT'})
        p = {'id': 'g:3', 'split': 'HOLDOUT', 'key': 'k'}
        with self.assertRaises(AssertionError):
            dev.score({'name': 'n', 'budget': '200k', 'candidates': 'four_three', 'trigger': 'all'}, [p],
                      {'k': {'grade': None, 'class': 'NOT_FOUND'}}, {}, {})

    def test_end_to_end(self):
        from scripts import h7_e4_dev as dev
        from scripts.run_mcts_v8_benchmark import file_sha256
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = corpus_of(tmp, GAMES[:2])
            corpus = web.load(path)
            boards = {p['key']: {'key': p['key'], 'class': 'NOT_FOUND', 'grade': None} for p in corpus['positions']}
            last = next(p for p in corpus['positions'] if p['id'] == f'g2:{len(VCF)}')
            boards[last['key']] = {'key': last['key'], 'class': 'PROVEN_WIN', 'grade': 'WIN_VCF'}
            (tmp / 't.json').write_text(json.dumps({'format': truth.FORMAT, 'corpus_sha256': file_sha256(path),
                                                    'budget': truth.DEFAULT_BUDGET, 'boards': boards}))
            b_row = {'id': last['id'], 'b': {'move': [5, 8], 'route': 'own_vcf', 'outcome': 'PUNISH'}, 'web_ai': None}
            (tmp / 'b.json').write_text(json.dumps({'format': base.FORMAT, 'truth_sha256': file_sha256(tmp / 't.json'),
                                                    'positions': {last['id']: b_row}}))
            dev_positions = [p for p in corpus['positions'] if p['split'] == 'DEV']
            out = tmp / 'dev.json'
            self.assertEqual(dev.main(['--corpus', str(path), '--truth', str(tmp / 't.json'), '--baseline',
                                       str(tmp / 'b.json'), '--limit', '2', '--output', str(out)]), 0)
            result = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual(result['split'], 'DEV')
        self.assertEqual(result['positions'], min(2, len(dev_positions)))
        self.assertTrue(all(a['id'] in {p['id'] for p in dev_positions} for a in result['attacks']))
        self.assertEqual(len(result['scores']), 8)
        self.assertIsNotNone(result['selection']['selected'])
