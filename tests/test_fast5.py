"""rt30.fast5 (X2-F, G0 unit part): the native VCF core against the frozen Python searches."""
import json
from pathlib import Path
from random import Random
import unittest

from analysis import vcf_tt_v1
from analysis.threats import _unbounded_max_fours
from renju import BLACK, EMPTY, SIZE, WHITE, Game
from search import mcts, mcts_v5, mcts_v7
from rt30 import fast5

ROOT = Path(__file__).resolve().parents[1]


def random_board(rng, stones, edge=False):
    """A board with ``stones`` random stones (no five check: the searches only read the board)."""
    game = Game()
    cells = [(r, c) for r in range(SIZE) for c in range(SIZE)
             if not edge or min(r, c, SIZE - 1 - r, SIZE - 1 - c) <= 2]
    rng.shuffle(cells)
    for i, (r, c) in enumerate(cells[:stones]):
        game.board[r][c] = BLACK if i % 2 == 0 else WHITE
    return game


def same(test, game, attacker, limit, max_fours=None):
    board = [row[:] for row in game.board]
    mf = max_fours or _unbounded_max_fours(game)
    a = mcts_v7._find_vcf_with_stats(game, attacker, max_fours=mf, node_limit=limit)
    b = fast5.find_vcf_with_stats(game, attacker, max_fours=mf, node_limit=limit)
    test.assertEqual(a, b)
    sa, sb = vcf_tt_v1.TableStats(), vcf_tt_v1.TableStats()
    a = vcf_tt_v1.make_find(sa)(game, attacker, max_fours=mf, node_limit=limit)
    b = fast5.make_find(sb)(game, attacker, max_fours=mf, node_limit=limit)
    test.assertEqual((a, sa), (b, sb))
    test.assertEqual(game.board, board)  # never written
    return a


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class NativeVcfTest(unittest.TestCase):
    def setUp(self):
        from hybrid.h6_fast import install, installed_mode
        self.previous = installed_mode()
        install('fast4')  # the Python oracle runs on the frozen-equivalent fast4 helpers, like the engine

    def tearDown(self):
        from hybrid.h6_fast import install
        fast5.uninstall()
        install(self.previous)

    def test_helpers(self):
        rng = Random(5)
        for _ in range(40):
            game = random_board(rng, rng.randrange(10, 120))
            for player in (BLACK, WHITE):
                for minimum in (3, 4):
                    got = fast5._native.probe(game.board, 0, player, minimum)
                    self.assertEqual(got, mcts_v5._window_candidates(game, player, minimum))
                self.assertEqual(fast5._native.probe(game.board, 1, player, 0), mcts_v5._winning_moves(game, player))
                for r in range(SIZE):
                    for c in range(SIZE):
                        p = r * SIZE + c
                        self.assertEqual(fast5._native.probe(game.board, 3, player, p),
                                         mcts._is_legal_for_player(game, player, (r, c)))
                        if game.board[r][c] == EMPTY:
                            self.assertEqual(fast5._native.probe(game.board, 4, player, p),
                                             mcts._wins_for_player(game, player, (r, c)))
                        elif game.board[r][c] == player:
                            self.assertEqual(fast5._native.probe(game.board, 2, player, p),
                                             list(mcts_v7._legal_completions(game, player, (r, c))))

    def test_random_boards_all_limits(self):
        rng = Random(11)
        wins = 0
        for i in range(300):
            game = random_board(rng, rng.randrange(6, 160), edge=i % 5 == 0)
            for attacker in (BLACK, WHITE):
                for limit in (1, 7, 300, 20_000):
                    wins += same(self, game, attacker, limit)[0] is not None
        self.assertGreater(wins, 20)  # the sample has real VCFs, not only empty searches

    def test_small_max_fours_and_full_board(self):
        rng = Random(3)
        for _ in range(60):
            game = random_board(rng, rng.randrange(20, 100))
            for attacker in (BLACK, WHITE):
                for max_fours in (1, 2, 3):
                    same(self, game, attacker, 5000, max_fours)
        full = random_board(Random(1), SIZE * SIZE)
        for attacker in (BLACK, WHITE):
            self.assertEqual(same(self, full, attacker, 100), (None, 0, False))

    def test_corpus_positions(self):
        positions = json.loads((ROOT / 'docs/h7/h7_e4v2_costtest_positions.json').read_text(encoding='utf-8'))
        for p in positions['positions'][::25]:
            game = Game()
            for m in p['moves']:
                game.play(*m)
            for attacker in (BLACK, WHITE):
                for limit in (200, 20_000):
                    same(self, game, attacker, limit)

    def test_errors_are_the_frozen_ones(self):
        game = Game()
        for kwargs in ({'attacker': 0, 'max_fours': 3, 'node_limit': 5},
                       {'attacker': BLACK, 'max_fours': 0, 'node_limit': 5},
                       {'attacker': BLACK, 'max_fours': 3, 'node_limit': 0}):
            with self.assertRaises(ValueError) as frozen:
                mcts_v7._find_vcf_with_stats(game, kwargs['attacker'], max_fours=kwargs['max_fours'],
                                            node_limit=kwargs['node_limit'])
            with self.assertRaises(ValueError) as native:
                fast5.find_vcf_with_stats(game, kwargs['attacker'], max_fours=kwargs['max_fours'],
                                          node_limit=kwargs['node_limit'])
            self.assertEqual(str(frozen.exception), str(native.exception))

    def test_install_rebinds_and_restores(self):
        from analysis import mcts_v8, threats
        bound = fast5.install()
        self.assertTrue(fast5.installed())
        self.assertIs(threats._find_vcf_with_stats, fast5.find_vcf_with_stats)
        self.assertIs(mcts_v8._find_vcf_with_stats, fast5.find_vcf_with_stats)
        self.assertIs(vcf_tt_v1.make_find, fast5.make_find)
        self.assertIn('search.mcts_v7._find_vcf_with_stats', bound['find'])
        fast5.uninstall()
        self.assertFalse(fast5.installed())
        self.assertIs(threats._find_vcf_with_stats, fast5.FROZEN)
        self.assertIs(vcf_tt_v1.make_find, fast5.FROZEN_MAKE_FIND)


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class Fast5RuntimeTest(unittest.TestCase):
    def tearDown(self):
        from hybrid.h6_fast import install
        fast5.uninstall()
        install('off')

    def test_identity_and_factory(self):
        from analysis import canonical, tb_v3, tb_v3_fast5, tb_v3_runtime
        tb_v3_fast5.check_sources()
        self.assertNotEqual(tb_v3_fast5.runtime_hash('cuda'), tb_v3_runtime.runtime_hash('cuda'))
        self.assertNotEqual(tb_v3_fast5.runtime_hash('cuda'), tb_v3_fast5.runtime_hash('cpu'))
        self.assertEqual(canonical.CANONICAL_RUNTIME, 'fast4')  # until fast5's G1 / G2
        self.assertIs(canonical.factory('TB-V3', 'fast5'), tb_v3_fast5.make_tb_v3_fast5_agent)
        self.assertIs(canonical.factory('TB-V3'), tb_v3_runtime.make_tb_v3_agent)
        with self.assertRaises(ValueError):
            canonical.factory('TB-V2-Value', 'fast5')
        tb_v3.check_identity()  # adding fast5 changed none of the engine's 63 files
        from tests.test_h7_timing import FACTORIES
        with self.assertRaises(ValueError):  # fails closed like the fast4 factory
            tb_v3_fast5.make_tb_v3_fast5_agent(FACTORIES['policy'], FACTORIES['leaf'], 1, None, None)
        with self.assertRaises(RuntimeError):  # fast5 not installed
            tb_v3_fast5.check_bound()
        tb_v3_fast5.install()
        tb_v3_fast5.check_bound()

    def test_g1_replay_on_fast5_matches_every_record_field(self):
        from analysis import tb_v2_runtime, tb_v3_fast5, tb_v3_runtime as v3rt
        from scripts import tb_v3_promotion as promo
        from tests.test_h7_timing import FACTORIES
        from tests.test_tb_v3 import recorded_game
        tb_v2_runtime.install()
        task = recorded_game(own=4)  # the candidate on fast4, as the arena recorded it
        tb_v3_fast5.install()
        row = promo.replay_game(task, lambda engine, seed: v3rt._build(FACTORIES['policy'], FACTORIES['leaf'], seed))
        self.assertEqual((row['moves'], row['move_mismatch'], row['decision_mismatch'], row['record_mismatch']),
                         (4, [], [], []))
        e = promo.summarize([row | {'engine': 'TB-V3'}], 'TB-V3')
        self.assertEqual(promo.verdict(e, 1, 4, None, 'fast5')['verdict'], 'G1_PASS')
        self.assertEqual(promo.verdict({**e, 'record_mismatch_moves': 1}, 1, 4, None, 'fast5')['verdict'], 'G1_FAIL')
        self.assertEqual(promo.verdict({**e, 'record_mismatch_moves': 1}, 1, 4, None)['verdict'], 'PROMOTION_PASS')


class G2Test(unittest.TestCase):
    def measure(self, runtime, seconds, e4=1.0, move=(1, 1)):
        from analysis import tb_v3
        rows = []
        for i, s in enumerate(seconds):
            r = {'id': f'p{i}', 'source': 'e3a', 'engine': 'tb-v3', 'move': list(move), 'seconds': s, 'e4_seconds': e4,
                 'decision': {'route': 'tree', 'root_nodes': 10}, 'times': {'tree': 1.0, 'root': s / 2, 'vct2': 0.1,
                                                                            'value': 0.1},
                 'e4_attack': {'ran': True, 'exhausted': i == 0, 'nodes': 5, 'seconds': e4}}
            if runtime == 'fast5':
                r['runtime'] = 'fast5'
            rows.append(r)
        return {'engine': tb_v3.ENGINE_HASH, 'runtime_mode': runtime, 'rows': rows}

    def test_tail_table_and_verdicts(self):
        from scripts import fast5_g2 as g2
        a = [self.measure('fast4', [29.0, 3.0]), self.measure('fast4', [29.4, 3.0])]
        b = [self.measure('fast5', [20.0, 1.0], e4=0.2), self.measure('fast5', [21.0, 1.0], e4=0.2)]
        out = g2.analyse(a, b, machine='ser5')
        self.assertEqual(out['verdict'], 'SER5_CERTIFIED')
        self.assertEqual([t['id'] for t in out['tracked']], ['p0'])  # over 25 s on fast4
        self.assertEqual(out['tracked'][0]['fast5'], 20.5)
        self.assertEqual(out['ratio_e4'], 0.2)
        slow = [self.measure('fast5', [26.0, 1.0]), self.measure('fast5', [20.0, 1.0])]
        self.assertEqual(g2.analyse(a, slow, machine='ser5')['verdict'], 'SER5_NOT_CERTIFIED')
        with self.assertRaises(ValueError):  # a different move: not the same engine
            g2.analyse(a, [self.measure('fast5', [20.0, 1.0], move=(2, 2))])
        with self.assertRaises(ValueError):  # the runtimes swapped
            g2.analyse(b, a)
