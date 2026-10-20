"""rt30.fast4_forbidden: the native forbidden points equal renju.rules.forbidden_reason; fast4 is exact."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'tests') not in sys.path:
    sys.path.insert(0, str(ROOT / 'tests'))

from renju import Game  # noqa: E402
from renju.rules import forbidden_reason  # noqa: E402
from rt30 import fast4_forbidden  # noqa: E402
from rule_validation import positions  # noqa: E402

NATIVE = unittest.skipUnless(fast4_forbidden.available(), 'python -m rt30.native.build first')


@NATIVE
class DifferentialTest(unittest.TestCase):
    def test_rule_corpus_every_empty_cell(self):
        reasons, checked = set(), 0
        for name, game in positions(11, 600):
            board = game.board
            for r in range(15):
                for c in range(15):
                    if board[r][c] == 0:
                        want = forbidden_reason(board, r, c)
                        self.assertEqual(fast4_forbidden.forbidden_native(board, r, c), want, (name, r, c))
                        reasons.add(want)
                        checked += 1
        self.assertEqual(reasons, {None, '장목', '사사', '삼삼'})  # every reason is exercised
        self.assertGreater(checked, 50_000)

    def test_board_is_not_written_and_errors_match(self):
        game = Game()
        for r, c in ((7, 6), (7, 8), (6, 7), (8, 7)):
            game.board[r][c] = 1
        before = [row[:] for row in game.board]
        self.assertEqual(fast4_forbidden.forbidden_native(game.board, 7, 7), '삼삼')
        self.assertEqual(game.board, before)
        for cell in ((7, 6), (-1, 0), (0, 15)):
            with self.assertRaises(ValueError) as want:
                forbidden_reason(game.board, *cell)
            with self.assertRaises(ValueError) as got:
                fast4_forbidden.forbidden_native(game.board, *cell)
            self.assertEqual(str(got.exception), str(want.exception))
        with self.assertRaises(TypeError):
            fast4_forbidden.forbidden_native([[0] * 15] * 14, 0, 0)


@NATIVE
class InstallTest(unittest.TestCase):
    def tearDown(self):
        from hybrid.h6_fast import install
        install('off')

    def test_modes(self):
        from hybrid.h6_fast import install, provenance
        from renju import game as game_module, rules
        from search import mcts, mcts_v7

        state = install('fast4')
        self.assertEqual(state['active']['forbidden'], 'forbidden')
        self.assertIs(mcts.forbidden_reason, fast4_forbidden.forbidden_native)
        self.assertIs(mcts_v7.forbidden_reason, fast4_forbidden.forbidden_native)
        self.assertIs(game_module.forbidden_reason, fast4_forbidden.forbidden_native)
        self.assertIsNot(rules.forbidden_reason, fast4_forbidden.forbidden_native)  # the oracle stays
        self.assertIn('native_build', provenance('fast4'))
        self.assertNotIn('forbidden', install('fast3')['active'])
        self.assertIs(mcts.forbidden_reason, rules.forbidden_reason)

    def test_v8c_checks_are_identical(self):
        from hybrid.h6_fast import install
        from scripts import h6_5d
        from tests.tb_engine_v1_regression import WEB

        moves = json.loads(WEB.read_text(encoding='utf-8'))['moves']
        for ply in (40, 92, 94):
            out = {}
            for mode in ('fast3', 'fast4'):
                install(mode)
                r = h6_5d.check_status(moves[:ply], moves[ply], h6_5d.STANDARD, table=True)
                out[mode] = {k: r[k] for k in ('status', 'nodes', 'calls', 'budget_cut', 'cut_kind')}
            self.assertEqual(out['fast3'], out['fast4'], ply)


@NATIVE
class PairedTest(unittest.TestCase):
    def test_paired_round_and_judge(self):
        from scripts import h7_timing as t
        from tests.test_h7_timing import FACTORIES, e3_run

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run = tmp / 'run.json'
            run.write_text(json.dumps(e3_run()), encoding='utf-8')
            pos = tmp / 'pos.json'
            t.main(['positions', '--runs', str(run), '--output', str(pos)])
            ids = [p['id'] for p in json.loads(pos.read_text(encoding='utf-8'))['positions'][:3]]
            corpus = tmp / 'corpus.json'
            corpus.write_text(json.dumps({'ids': {i: ['random'] for i in ids}}), encoding='utf-8')
            outs = []
            for label in ('1', '2'):
                out = tmp / f'r{label}.json'
                self.assertEqual(t.main(['paired', '--positions', str(pos), '--corpus', str(corpus), '--baseline-mode',
                                         'fast3', '--candidate-mode', 'fast4', '--round', label, '--output', str(out)],
                                        factories=FACTORIES), 0)
                outs.append(out)
            r = json.loads(outs[0].read_text(encoding='utf-8'))
            self.assertEqual(r['summary']['semantics'], 'PASS')
            self.assertEqual(len(r['rows']), 3)
            self.assertTrue(all(len(x['runs']['A']) == 2 and len(x['runs']['B']) == 2 for x in r['rows']))
            self.assertEqual({x['order'] for x in r['rows']} <= {'ABBA', 'BAAB'}, True)
            judged = t.judge_rounds([json.loads(p.read_text(encoding='utf-8')) for p in outs])
            self.assertIn(judged['verdict'], ('PASS', 'SPEED_FAIL'))  # 3 tiny positions: speed is not judged here
            self.assertTrue(judged['semantics'])
            self.assertEqual(t.judge_rounds([r])['verdict'], 'INCOMPLETE')
        self.assertEqual({t.order_of(f'x{i}') for i in range(20)}, {'ABBA', 'BAAB'})


if __name__ == '__main__':
    unittest.main()
