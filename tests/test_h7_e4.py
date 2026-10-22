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


class CandidateTest(unittest.TestCase):
    def test_identity(self):
        import hashlib
        from analysis import e4_attack, tb_v2_value, tb_v2_value_e4 as cand
        root = Path(__file__).resolve().parents[1]
        for name, digest in cand.ATTACK_SOURCES.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(cand.BASE_ENGINE_HASH, tb_v2_value.config_hash())
        cand.check_sources()
        self.assertEqual(e4_attack.BUDGETS[cand.ATTACK['budget']], cand.ATTACK['budget_values'])
        selected = json.loads((root / cand.SELECTION['dev_result']).read_text(encoding='utf-8'))['selection']['selected']
        self.assertEqual({k: cand.ATTACK[k] for k in selected}, selected)  # the DEV selection, literally
        self.assertNotEqual(cand.candidate_hash(), cand.BASE_ENGINE_HASH)
        self.assertNotEqual(cand.runtime_hash('cuda'), cand.runtime_hash('cpu'))

    def agents(self, attack):
        from analysis import tb_v2_value, tb_v2_value_e4 as cand
        from tests.test_tb_v2_value import Leaf
        from tests.tb_engine_v1_regression import hashed_policy
        leaf = Leaf(tb_v2_value.VALUE_CHECKPOINT_SHA256)
        return (tb_v2_value.make_agent(hashed_policy, leaf, seed=5),
                cand.make_agent(hashed_policy, leaf, seed=5, attack=attack))

    def test_off_is_the_base_engine(self):
        from tests.tb_engine_v1_regression import WEB
        moves = [tuple(m) for m in json.loads(WEB.read_text(encoding='utf-8'))['moves']]
        base_agent, off = self.agents(attack=False)
        for ply in (12, 20, 28):
            game = Game()
            for m in moves[:ply]:
                game.play(*m)
            self.assertEqual(off.select_move(game), base_agent.select_move(game), ply)
            self.assertEqual(off.attack_info, {})

    def test_attack_on_replaces_only_with_a_proof(self):
        _, on = self.agents(attack=True)
        game = Game()
        for m in VCF:
            game.play(*m)
        move = on.select_move(game)  # a forced stage (not an attack route): the attack does not run
        self.assertNotIn(on.diagnostics.v8_route, ('stage4', 'stage5', 'tree'))
        self.assertEqual(on.attack_info, {})
        self.assertEqual(base.check_move(VCF, move, truth.DEFAULT_BUDGET)['outcome'], 'PUNISH')

    def test_a_proven_win_replaces_a_tree_move(self):
        from types import SimpleNamespace
        from analysis.tb_v2_value_e4 import E4AttackAgent

        class Stub:
            name = 'stub'

            def __init__(self):
                self.diagnostics = SimpleNamespace(v8_route='tree', v8_changed=False)

            def select_move(self, game):
                return (0, 7)

        game = Game()
        for m in VCF:
            game.play(*m)
        agent = E4AttackAgent(Stub())
        move = agent.select_move(game)
        self.assertNotEqual(move, (0, 7))
        self.assertEqual((agent.diagnostics.v8_route, agent.diagnostics.v8_changed), ('own_vct2', True))
        self.assertEqual(agent.attack_info['base_move'], [0, 7])
        self.assertEqual(base.check_move(VCF, move, truth.DEFAULT_BUDGET)['outcome'], 'PUNISH')
        self.assertEqual(E4AttackAgent(Stub(), attack=False).select_move(game), (0, 7))


class ReferenceTest(unittest.TestCase):
    def test_quiet_moves_extend_the_candidate_set(self):
        from analysis.e4_reference import BUDGETS, ReferenceSolver, reference_attack
        from analysis.selective_vct import threat_moves
        game = Game()
        for m in VCF:
            game.play(*m)
        solver = ReferenceSolver(node_limit=20_000, call_limit=10_000, node_budget=1_000_000)
        solver._cap, solver._call_cap = solver.node_budget, solver.call_limit
        quiet = solver.quiet_moves(game)
        threats = threat_moves(game)
        self.assertEqual(quiet[:len(threats)], threats)
        self.assertEqual(len(quiet), len(set(quiet)))
        move, info = reference_attack(game, '2m')
        self.assertIsNotNone(move)
        self.assertEqual(base.check_move(VCF, move, truth.DEFAULT_BUDGET)['outcome'], 'PUNISH')
        self.assertEqual(set(BUDGETS), {'2m', '4m'})

    def test_judge(self):
        from scripts import h7_e4_reference as ref
        boards = {'k1': {'class': 'PROVEN_WIN'}, 'k2': {'class': 'PROVEN_WIN'}, 'k3': {'class': 'NOT_FOUND'}}
        keys = {'a': 'k1', 'b': 'k2', 'c': 'k3'}
        win = {'reference': {'move': [1, 1], 'nodes': 5, 'seconds': 0.1, 'exhausted': False},
               'check': {'outcome': 'PUNISH'}}
        none = {'reference': {'move': None, 'nodes': 5, 'seconds': 0.1, 'exhausted': True}, 'check': None}
        self.assertEqual(ref.judge({'a': win, 'b': win}, ['a'], ['a', 'b'], boards, keys, '2m')['verdict'], 'FREEZE')
        self.assertEqual(ref.judge({'a': none, 'b': win}, ['a'], ['b'], boards, keys, '2m')['verdict'], 'RAISE_TO_4M')
        self.assertEqual(ref.judge({'a': none}, ['a'], [], boards, keys, '4m')['verdict'], 'INVESTIGATE')
        self.assertEqual(ref.judge({'c': win}, [], [], boards, keys, '2m')['verdict'], 'INVESTIGATE')
        false = {**win, 'check': {'outcome': 'MISS'}}
        self.assertEqual(ref.judge({'a': false}, ['a'], [], boards, keys, '2m')['false_proofs'], ['a'])


class PrecheckTest(unittest.TestCase):
    def test_game_cluster_and_ratio(self):
        from scripts import h7_e4_precheck as pre
        self.assertEqual(pre.game_of('e3a|arm@v8:x/8407/0/black|12'), 'e3a|arm@v8:x/8407/0/black')
        self.assertEqual(pre.game_of('probe|P92'), 'probe')
        rows = [{'game': g, 'e_seconds': e, 'b_seconds': 1.0} for g, e in
                [('a', 2.0), ('a', 2.0), ('b', 1.9), ('c', 2.1), ('d', 2.0)]]
        out = pre.ratio_ci(rows)
        self.assertEqual((out['ratio'], out['games'], out['verdict']), (2.0, 4, 'PRECHECK_FAIL'))
        cheap = [{**r, 'e_seconds': 1.2} for r in rows]
        self.assertEqual(pre.ratio_ci(cheap)['verdict'], 'PROCEED')

    def test_end_to_end_drops_web_boards(self):
        from rt30 import fast4_forbidden
        from scripts import h7_e4_precheck as pre
        from tests.test_h7_timing import FACTORIES
        from tests.tb_engine_v1_regression import WEB
        if not fast4_forbidden.available():
            self.skipTest('python -m rt30.native.build first')
        moves = json.loads(WEB.read_text(encoding='utf-8'))['moves']
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = corpus_of(tmp, GAMES[:1])
            positions = [{'id': f'e3a|g/1/0/black|{n}', 'source': 'e3a', 'moves': moves[:n]} for n in (14, 22)]
            positions.append({'id': 'e3a|g/1/1/black|5', 'source': 'e3a', 'moves': [list(m) for m in FIVE[:5]]})
            (tmp / 'p.json').write_text(json.dumps({'positions': positions}))
            out = tmp / 'pre.json'
            self.assertEqual(pre.main(['--positions', str(tmp / 'p.json'), '--corpus', str(path), '--device', 'cpu',
                                       '--output', str(out)], factories=FACTORIES), 0)
            result = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual(result['dropped_web_boards'], ['e3a|g/1/1/black|5'])  # a web corpus board
        self.assertEqual(result['positions'], 2)
        for row in result['rows']:
            self.assertGreaterEqual(row['e_seconds'], row['b_seconds'])
        from hybrid.h6_fast import installed_mode
        self.assertEqual(installed_mode(), 'off')


class BenchmarkArmTest(unittest.TestCase):
    def test_e4_arm_is_the_frozen_base_plus_the_wrapper(self):
        from analysis.tb_v2_value import ENGINE_OPTIONS, TB_V2_VALUE
        from scripts.run_mcts_v8_benchmark import ARMS, parse_opponent, v8_config
        e4 = v8_config('puct_value_vct2_50k_tt_e4')
        self.assertEqual({k: v for k, v in e4.items() if k != 'e4_attack'}, v8_config('puct_value_vct2_50k_tt'))
        self.assertEqual({k: e4[k] for k in {**TB_V2_VALUE, **ENGINE_OPTIONS}}, {**TB_V2_VALUE, **ENGINE_OPTIONS})
        self.assertTrue(ARMS['puct_value_vct2_50k_tt_e4']['e4_attack'])
        with self.assertRaises(ValueError):
            parse_opponent('v8:puct_value_vct2_50k_tt_e4')  # a tested arm only


class E4v2Test(unittest.TestCase):
    def test_killer_keeps_every_status(self):
        from analysis import e4v2_attack as v2
        from analysis.mcts_v8 import SearchDiagnostics, _attack_candidates
        from analysis.selective_vct import SelectiveSolver
        from analysis.vct2_attack import _depth2_attack_check
        from search.mcts_v5 import _RootContext
        from tests.tb_engine_v1_regression import WEB
        web_moves = [tuple(m) for m in json.loads(WEB.read_text(encoding='utf-8'))['moves']]
        for moves in (VCF, FIVE, web_moves[:13], web_moves[:20]):
            game = Game()
            for m in moves:
                game.play(*m)
            for move in _attack_candidates(game, _RootContext(game.legal_moves(), SearchDiagnostics()))[:6]:
                statuses = []
                for cls in (SelectiveSolver, v2.KillerSolver):
                    s = cls(node_limit=20_000, call_limit=1_000_000, node_budget=50_000_000)
                    statuses.append(s._bounded(game, move, None, None, lambda g, s=s: _depth2_attack_check(s, g)))
                self.assertEqual(statuses[0], statuses[1], (len(moves), move))
                self.assertNotEqual(statuses[0], 'UNKNOWN')

    def test_cooldown_is_one_move_of_the_same_game(self):
        from types import SimpleNamespace
        from analysis.e4v2_attack import Cooldown
        g = (lambda moves: SimpleNamespace(history=list(moves)))
        cd = Cooldown()
        self.assertFalse(cd.blocks(g(VCF[:4])))
        cd.exhausted(g(VCF[:4]))
        self.assertTrue(cd.blocks(g(VCF[:6])))  # the next own move
        self.assertFalse(cd.blocks(g(VCF[:8])))  # cleared after one move
        cd.exhausted(g(VCF[:4]))
        self.assertFalse(cd.blocks(g(VCF[:8])))  # not the next own move: cleared, no skip
        cd.exhausted(g(VCF[:4]))
        self.assertFalse(cd.blocks(g([*VCF[:3], (1, 1), (2, 2), (3, 3)])))  # another game (other prefix)
        self.assertIsNone(cd.history)

    def test_decide_route_gate_and_attack(self):
        from analysis import e4v2_attack as v2
        config = {'budget': '200k', 'cooldown': True, 'killer': False}
        game = Game()
        for m in VCF:
            game.play(*m)
        self.assertEqual(v2.decide(game, config, 'own_vcf')[1]['skip'], 'route')
        move, info = v2.decide(game, config, 'tree', v2.Cooldown())
        self.assertTrue(info['ran'])
        self.assertEqual(base.check_move(VCF, move, truth.DEFAULT_BUDGET)['outcome'], 'PUNISH')
        three = Game()
        for m in [(7, 7), (0, 0), (7, 8), (0, 2), (7, 9)]:  # Black's open three, White to move
            three.play(*m)
        move, info = v2.decide(three, config, 'tree', v2.Cooldown())
        self.assertEqual((move, info['skip'], info['gate']['status']), (None, 'opponent_vcf', 'WIN'))
        self.assertEqual(len(v2.GRID), 8)
        self.assertEqual(len({c['name'] for c in v2.GRID}), 8)

    def test_a_forced_move_consumes_the_cooldown(self):
        from analysis import e4v2_attack as v2
        from scripts import h7_e4v2_dev as dev2
        config = {'budget': '200k', 'cooldown': True, 'killer': False}
        positions = [Game() for _ in range(3)]
        for game, n in zip(positions, (7, 9, 11)):  # three own moves of White on one line
            for m in VCF[:n]:
                game.play(*m)
        cd = v2.Cooldown()
        cd.exhausted(positions[0])
        self.assertEqual(v2.decide(positions[1], config, 'own_vcf', cd)[1]['skip'], 'route')
        self.assertIsNone(cd.history)  # consumed by the forced move
        self.assertTrue(v2.decide(positions[2], config, 'tree', cd)[1]['ran'])  # not delayed to this move
        cd.exhausted(positions[1])
        self.assertEqual(v2.decide(positions[2], config, 'tree', cd)[1]['skip'], 'cooldown')
        # the offline replay does the same
        seq = [{'id': i, 'moves': VCF[:n], 'eligible': e} for i, n, e in (('a', 7, True), ('b', 9, False), ('c', 11, True))]
        info = {'move': None, 'exhausted': True, 'nodes': 1, 'calls': 1}
        rows = {i: {'gate': {'status': 'NONE', 'nodes': 1, 'calls': 1}, 'attacks': {'200k/plain': info}} for i in 'ac'}
        out = dev2.replay(config, seq, rows)
        self.assertEqual((out['b']['skip'], out['c']['ran']), ('route', True))
        out = dev2.replay(config, [seq[0], {**seq[2], 'moves': VCF[:9]}], {'a': rows['a'], 'c': rows['c']})
        self.assertEqual(out['c']['skip'], 'cooldown')

    def test_cost_model_fit(self):
        from scripts import h7_e4v2_dev as dev2
        rows = [{'attacked': True, 'attack_nodes': n, 'attack_calls': c, 'attack_seconds': 2e-5 * n + 1e-4 * c + 0.05}
                for n, c in [(1000, 10), (5000, 200), (200000, 4000), (80000, 900), (30000, 1500)]]
        m = dev2.fit_cost_model(rows + [{'attacked': False}])
        self.assertAlmostEqual(m['seconds_per_node'], 2e-5, places=9)
        self.assertAlmostEqual(m['seconds_per_call'], 1e-4, places=8)
        self.assertAlmostEqual(m['seconds_per_attack'], 0.05, places=6)
        self.assertEqual((m['n'], m['r2']), (5, 1.0))
        self.assertAlmostEqual(dev2.model_seconds(m, 1000, 1, False), 2e-2 + 1e-4, places=6)

    def test_select_rule(self):
        from scripts import h7_e4v2_dev as dev2
        s = (lambda name, punish, ratio, ok=True: {'config': {'name': name}, 'eligible': ok,
                                                   'dev': {'punish': punish}, 'costdev': {'ratio': ratio}})
        out = dev2.select([s('a', 30, 1.40), s('b', 29, 1.30), s('c', 28, 1.20), s('d', 31, 1.10, ok=False)])
        self.assertEqual((out['selected']['name'], out['tied']), ('c', ['b', 'c']))  # a is over 1.35
        self.assertEqual(dev2.select([s('a', 30, 1.40)])['verdict'], 'REJECT_AT_DEV')

    def test_end_to_end(self):
        from scripts import h7_e4v2_dev as dev2
        from scripts.run_mcts_v8_benchmark import file_sha256
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = corpus_of(tmp, GAMES[:2])
            corpus = web.load(path)
            boards = {p['key']: {'key': p['key'], 'class': 'NOT_FOUND', 'grade': None} for p in corpus['positions']}
            (tmp / 't.json').write_text(json.dumps({'format': truth.FORMAT, 'corpus_sha256': file_sha256(path),
                                                    'budget': truth.DEFAULT_BUDGET, 'boards': boards}))
            (tmp / 'b.json').write_text(json.dumps({'format': base.FORMAT, 'truth_sha256': file_sha256(tmp / 't.json'),
                                                    'positions': {}}))
            timing = [{'id': f'e3a|g/1/0/black|{n}', 'moves': [list(m) for m in VCF[:n]]} for n in (4, 6)]
            rows = [{'id': p['id'], 'game': 'e3a|g/1/0/black', 'attacked': True, 'b_seconds': 1.0,
                     'attack_seconds': 0.1 + i, 'attack_nodes': 1000 * (i + 1), 'attack_calls': 10 * (i + 3),
                     'base_move': [0, 0]} for i, p in enumerate(timing)]
            rows.append({**rows[0], 'id': 'e3a|g/1/0/black|8', 'attacked': False, 'attack_seconds': 0.0})
            timing.append({'id': 'e3a|g/1/0/black|8', 'moves': [list(m) for m in VCF[:8]]})
            from analysis import tb_v2_value_e4
            (tmp / 'pos.json').write_text(json.dumps({'positions': timing}))
            pre = {'format': 'h7-e4-precheck-v1', 'rows': rows, 'positions_sha256': file_sha256(tmp / 'pos.json'),
                   'corpus_sha256': file_sha256(path), 'candidate_hash': tb_v2_value_e4.candidate_hash()}
            (tmp / 'pre.json').write_text(json.dumps({**pre, 'corpus_sha256': 'x'}))
            with self.assertRaises(SystemExit):  # COSTDEV must be the pre-check of these positions and corpus
                dev2.main(['--corpus', str(path), '--truth', str(tmp / 't.json'), '--baseline', str(tmp / 'b.json'),
                           '--precheck', str(tmp / 'pre.json'), '--positions', str(tmp / 'pos.json'),
                           '--output', str(tmp / 'x.json')])
            (tmp / 'pre.json').write_text(json.dumps(pre))
            out = tmp / 'v2.json'
            argv = ['--corpus', str(path), '--truth', str(tmp / 't.json'), '--baseline', str(tmp / 'b.json'),
                    '--precheck', str(tmp / 'pre.json'), '--positions', str(tmp / 'pos.json'), '--limit', '3',
                    '--jsonl', str(tmp / 'v2.jsonl')]
            self.assertEqual(dev2.main([*argv, '--output', str(out)]), 0)
            result = json.loads(out.read_text(encoding='utf-8'))
            lines = (tmp / 'v2.jsonl').read_text(encoding='utf-8').splitlines()
            stale = json.loads(lines[0])
            stale['fingerprint'] = {**stale['fingerprint'],
                                    'code': {**stale['fingerprint']['code'], 'src/analysis/selective_vct.py': 'old'}}
            (tmp / 'v2.jsonl').write_text('\n'.join([json.dumps(stale), *lines[1:]]) + '\n', encoding='utf-8')
            with self.assertRaises(SystemExit):  # a row of other code is refused, not mixed in
                dev2.main([*argv, '--output', str(tmp / 'v2b.json')])
            cal = {'precheck_sha256': file_sha256(tmp / 'pre.json'), 'fast': 'off', 'code': dev2.code_shas(),
                   'correction': {'killer': 1.0, 'gate': 1.0}}
            for bad in ({'fast': 'fast4'}, {'code': {**cal['code'], 'src/analysis/e4v2_attack.py': 'old'}}):
                (tmp / 'cal.json').write_text(json.dumps({**cal, **bad}))
                with self.assertRaises(SystemExit):  # another mode / other code than this run
                    dev2.main([*argv[:-2], '--calibration', str(tmp / 'cal.json'), '--output', str(tmp / 'v2c.json')])
            (tmp / 'cal.json').write_text(json.dumps(cal))
            self.assertEqual(dev2.main([*argv[:-2], '--calibration', str(tmp / 'cal.json'),
                                        '--output', str(tmp / 'v2d.json')]), 0)
        self.assertEqual(len(result['scores']), 8)
        self.assertEqual(result['split'], 'DEV')
        self.assertTrue(all(s['costdev']['positions'] == 3 for s in result['scores']))
        self.assertIn(result['selection']['verdict'], ('SELECTED', 'REJECT_AT_DEV'))


class CosttestTest(unittest.TestCase):
    def test_whole_games_in_hash_order_without_known_boards(self):
        from scripts import h7_e4v2_costtest as ct
        with TemporaryDirectory() as tmp:
            corpus = web.load(corpus_of(Path(tmp), GAMES[:1]))
        game = (lambda key, moves: {'key': key, 'moves': [list(m) for m in moves],
                                    'v8_moves': [{'ply': p, 'route': 'tree', 'seconds': 1.0}
                                                 for p in range(1, len(moves), 2)]})
        runs = [{'arm': ct.ARM, 'opponent': ct.OPPONENT, 'seed': s,
                 'games': [game('k/0/white', FIVE), game('k/1/white', VCF)]} for s in ct.SEEDS]
        costdev = [{'id': 'x', 'moves': [list(m) for m in VCF[:3]]}]
        clean = (lambda key: game(key, [(7, 7), (6, 6), (8, 8), (5, 5), (9, 9), (4, 4), (10, 10)]))
        runs = [{**r, 'games': [*r['games'], clean('k/2/white'), clean('k/3/white')]} for r in runs]
        out = ct.build(runs, corpus, costdev, count=6)
        dirty = {f'{s}/k/{i}/white' for s in ct.SEEDS for i in (0, 1)}  # a web board, a COSTDEV board
        self.assertTrue(dirty.isdisjoint(out['games']))
        self.assertTrue(set(out['dropped_games']) <= dirty)
        ids = [p['id'] for p in out['positions']]
        self.assertEqual(len(set(ids)), len(ids))
        for key in out['games']:  # whole games, every tested-arm position, in ply order
            plies = [int(i.rsplit('|', 1)[1]) for i in ids if i.split('|')[1] == key]
            self.assertEqual(plies, [1, 3, 5])
        self.assertTrue(all('source_recorded' in p and 'recorded' not in p for p in out['positions']))
        with self.assertRaises(ValueError):  # not enough clean positions
            ct.build(runs, corpus, costdev, count=100)
        with self.assertRaises(ValueError):
            ct.build([{**runs[0], 'opponent': 'v7'}, runs[1]], corpus, costdev)


class E4v2EquivalenceTest(unittest.TestCase):
    def test_plain_200k_attack_is_the_v1_attack(self):
        from analysis import e4_attack, e4v2_attack as v2
        from tests.tb_engine_v1_regression import WEB
        web_moves = [tuple(m) for m in json.loads(WEB.read_text(encoding='utf-8'))['moves']]
        for moves in (VCF, web_moves[:13], web_moves[:20]):
            game = Game()
            for m in moves:
                game.play(*m)
            m1, i1 = e4_attack.e4_attack(game, {'budget': '200k', 'candidates': 'vcf_threat'})
            m2, i2 = v2.attack(game, '200k', False)
            self.assertEqual((m1, i1['nodes'], i1['calls'], i1['candidates']), (m2, i2['nodes'], i2['calls'], i2['candidates']))


class CalibrationTest(unittest.TestCase):
    def test_relative_corrections(self):
        from scripts import h7_e4v2_calibrate as cal
        from scripts import h7_e4v2_dev as dev2
        model = {'seconds_per_node': 1e-5, 'seconds_per_call': 0.0, 'seconds_per_attack': 0.0}
        rows = [{'gate': {'nodes': 1000, 'calls': 1, 'seconds': 0.02},  # 2x the model
                 'plain': {'nodes': 100_000, 'calls': 10, 'seconds': 1.5},  # this machine: 1.5x
                 'killer': {'nodes': 100_000, 'calls': 10, 'seconds': 1.65}}]  # 1.1x plain
        out = cal.summarize(rows, model)
        self.assertEqual(out['factor'], {'plain': 1.5, 'killer': 1.65, 'gate': 2.0})
        self.assertEqual(out['correction'], {'killer': 1.1, 'gate': 1.3333})
        step = {'ran': True, 'gate': {'nodes': 1000, 'calls': 1}, 'nodes': 100_000, 'calls': 10}
        self.assertAlmostEqual(dev2.step_seconds(model, out['correction'], step, True), 1.3333 * 0.01 + 1.1 * 1.0)
        self.assertAlmostEqual(dev2.step_seconds(model, {}, step, False), 0.01 + 1.0)
        picked = cal.subset([{'id': str(i), 'attacked': i % 2 == 0} for i in range(10)], 3)
        self.assertEqual(len(picked), 3)
        self.assertTrue(all(r['attacked'] for r in picked))
