"""External-engine adapter, anchor runner and summary (docs/external-engine.md), against the fake engine."""
from argparse import Namespace
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agents.external_engine import (EngineConfig, EngineError, EngineIllegalMove, EngineTimeout,  # noqa: E402
                                    ExternalEngineAgent, board_lines, from_piskvork, our_forbidden_points,
                                    to_piskvork)
from evaluation.external import EXTERNAL_SEEDS, elo, pair_bootstrap, summarize  # noqa: E402
from hybrid.h6_labels import RESERVED_SEEDS  # noqa: E402
from renju import BLACK, WHITE, Game  # noqa: E402

FAKE = str(ROOT / 'scripts' / 'fake_piskvork_engine.py')
# Black (7,7), (7,8), (8,6), (9,6): (7,6) makes two open threes, a double-three (forbidden).
DOUBLE_THREE = [(7, 7), (0, 0), (7, 8), (0, 14), (8, 6), (14, 0), (9, 6), (14, 14)]


def fake(*args, **kwargs) -> EngineConfig:
    return EngineConfig(command=(sys.executable, FAKE, *args), name='fake', **kwargs)


def position(moves) -> Game:
    game = Game()
    for move in moves:
        game.play(*move)
    return game


class CoordinateTest(unittest.TestCase):
    def test_round_trip_is_column_row(self):
        self.assertEqual(to_piskvork((3, 11)), '11,3')
        self.assertEqual(from_piskvork(' 11 , 3 '), (3, 11))
        for move in [(0, 0), (14, 0), (0, 14), (7, 9)]:
            self.assertEqual(from_piskvork(to_piskvork(move)), move)

    def test_bad_reply(self):
        with self.assertRaises(EngineError):
            from_piskvork('SWAP')

    def test_board_fields_are_side_to_move_relative(self):
        game = position([(7, 7), (7, 8), (8, 8)])  # white to move
        self.assertEqual(game.to_play, WHITE)
        self.assertEqual(board_lines(game.history, game.to_play), ['7,7,2', '8,7,1', '8,8,2'])
        self.assertEqual(board_lines(game.history, BLACK), ['7,7,1', '8,7,2', '8,8,1'])


class AdapterTest(unittest.TestCase):
    def test_protocol_and_legal_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'log.txt'
            config = fake('--log', str(log), info=(('max_node', '1000'), ('thread_num', '1')))
            self.assertEqual(config.label, 'fake[max_node=1000][thread_num=1]')
            game = position([(7, 7), (6, 6)])
            with ExternalEngineAgent(config) as agent:
                self.assertIn('fake', agent.engine.about())
                move = agent.select_move(game)
                game.play(*move)  # legal under our rules
                self.assertTrue(agent.engine.last_messages)
            lines = log.read_text(encoding='utf-8').splitlines()
        self.assertEqual(lines[:4], ['START 15', 'INFO rule 4', 'INFO max_node 1000', 'INFO thread_num 1'])
        board = lines.index('BOARD')
        self.assertEqual(lines[board + 1:board + 4], ['7,7,1', '6,6,2', 'DONE'])

    def test_forbidden_reply_is_engine_illegal(self):
        game = position(DOUBLE_THREE)
        self.assertEqual(our_forbidden_points(game), {(7, 6)})
        with ExternalEngineAgent(fake('--mode', 'forbidden')) as agent:
            with self.assertRaises(EngineIllegalMove) as caught:
                agent.select_move(game)
        self.assertEqual(caught.exception.move, (7, 6))
        self.assertIn('삼삼', caught.exception.reason)

    def test_forbidden_points_query(self):
        game = position(DOUBLE_THREE)
        with ExternalEngineAgent(fake()) as agent:
            self.assertEqual(agent.forbidden_points(game), {(7, 6)})
        with ExternalEngineAgent(fake('--forbid-lie')) as agent:
            self.assertEqual(agent.forbidden_points(game), set())

    def test_timeout_crash_garbage(self):
        game = position([(7, 7)])
        with ExternalEngineAgent(fake('--mode', 'silent', move_timeout=1.0)) as agent:
            with self.assertRaises(EngineTimeout):
                agent.select_move(game)
        for mode in ('crash', 'garbage'):
            with self.subTest(mode=mode), ExternalEngineAgent(fake('--mode', mode)) as agent:
                with self.assertRaises(EngineError):
                    agent.select_move(game)

    def test_missing_executable(self):
        with self.assertRaises(EngineError):
            ExternalEngineAgent(EngineConfig(command=(str(ROOT / 'no-such-engine'),)))


def spec(*engine_args) -> dict:
    return {'ours': 'random', 'overrides': {}, 'ours_label': 'random', 'engine_label': 'fake',
            'engine_command': [sys.executable, FAKE, *engine_args], 'engine_name': 'fake',
            'engine_info': [], 'move_timeout': 30.0, 'run_signature': 'sig'}


class RunnerTest(unittest.TestCase):
    def setUp(self):
        from scripts import run_external_match
        self.runner = run_external_match

    def task(self, *engine_args, color=BLACK, forbid_check=False) -> dict:
        return {'key': 'k', 'pair': 0, 'seed': 5, 'ours_color': color, 'opening': [[7, 7]],
                'forbid_check': forbid_check, 'spec': spec(*engine_args)}

    def test_full_game(self):
        record = self.runner.play_one(self.task(forbid_check=True))
        self.assertIn(record['result'], ('win', 'loss', 'draw'))
        replay = position(map(tuple, record['moves']))
        self.assertTrue(replay.done)
        self.assertEqual(record['forbid_mismatch'], [])
        self.assertGreater(record['forbid_checked'], 0)
        self.assertEqual(len(record['our_moves']) + len(record['engine_moves']), len(record['moves']) - 1)

    def test_engine_illegal_and_error(self):
        task = {**self.task('--mode', 'forbidden', color=WHITE), 'opening': [list(m) for m in DOUBLE_THREE]}
        record = self.runner.play_one(task)
        self.assertEqual(record['result'], 'engine_illegal')
        self.assertEqual(record['illegal']['move'], [7, 6])
        self.assertEqual(record['moves'], [list(m) for m in DOUBLE_THREE])  # the illegal move is not played
        record = self.runner.play_one(self.task('--mode', 'crash'))
        self.assertEqual(record['result'], 'engine_error')

    def test_forbid_lie_is_reported(self):
        record = self.runner.play_one(self.task('--forbid-lie', forbid_check=True))
        for mismatch in record['forbid_mismatch']:
            self.assertEqual(mismatch['engine_only'], [])
            self.assertTrue(mismatch['ours_only'])

    def test_parse_info_and_seed_guard(self):
        self.assertEqual(self.runner.parse_info(['MAX_NODE=100']), [['max_node', '100']])
        for bad in ('rule=1', 'max_node', '=3'):
            with self.assertRaises(ValueError):
                self.runner.parse_info([bad])
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self.runner.main(['--ours', 'random', '--engine', sys.executable, '--seed', '1'])

    def test_resume_refuses_another_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'games.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in [
                {'key': 'a', 'run_signature': 'old'}, {'key': 'b', 'run_signature': 'new'},
                {'key': 'other', 'run_signature': 'old'}]) + '\n', encoding='utf-8')
            self.assertEqual(list(self.runner.load_finished(path, {'b'}, 'new')), ['b'])
            with self.assertRaises(ValueError):
                self.runner.load_finished(path, {'a', 'b'}, 'new')
            path.write_text(json.dumps({'key': 'a'}) + '\n', encoding='utf-8')  # no signature: never reused
            with self.assertRaises(ValueError):
                self.runner.load_finished(path, {'a'}, 'new')

    def test_signature_covers_engine_bytes_and_openings(self):
        args = Namespace(opening_random_plies=4, opening_radius=2, forbid_check=False)
        engine = {'name': 'rapfi', 'command': ['/x/rapfi', '--x'], 'info': [['max_node', '1000']], 'rule': 4,
                  'executable_sha256': 'aa', 'extra_files': {'/x/config.toml': 'cc'}}
        ours = {'kind': 'tb1', 'policy_checkpoint_sha256': 'pp'}
        base = self.runner.run_signature(engine, ours, args)
        moved = {**engine, 'command': ['/elsewhere/rapfi', '--x'], 'extra_files': {'/y/config.toml': 'cc'}}
        self.assertEqual(self.runner.run_signature(moved, ours, args), base)  # paths do not matter
        changes = [({**engine, 'executable_sha256': 'ab'}, ours, args),
                   ({**engine, 'extra_files': {'/x/config.toml': 'cd'}}, ours, args),
                   ({**engine, 'info': [['max_node', '3000']]}, ours, args),
                   (engine, {**ours, 'policy_checkpoint_sha256': 'pq'}, args),
                   (engine, ours, Namespace(opening_random_plies=2, opening_radius=2, forbid_check=False)),
                   (engine, ours, Namespace(opening_random_plies=4, opening_radius=3, forbid_check=False))]
        for changed in changes:
            self.assertNotEqual(self.runner.run_signature(*changed), base)

    def test_x1_signatures_still_reproduce(self):
        """The signature function is the one X1 ran: a rerun of the frozen anchor (e.g. on WSL2) can be compared
        by its run_signature with the committed X1 record."""
        record = json.loads((ROOT / 'docs' / 'external' / 'x1_provenance.json').read_text(encoding='utf-8'))
        for run in record['runs']:
            e = run['engine']
            engine = {'name': 'rapfi', 'command': ['pbrain-rapfi-windows-avx2.exe'], 'info': e['info'],
                      'rule': e['rule'], 'executable_sha256': e['executable_sha256'], 'extra_files': e['extra_files']}
            args = Namespace(opening_random_plies=run['opening_random_plies'], opening_radius=run['opening_radius'],
                             forbid_check=run['forbid_check'])
            self.assertEqual(self.runner.run_signature(engine, run['ours'], args), run['run_signature'], run['file'])

    def test_identity_fail_fast(self):
        engine = {'command': ['rapfi'], 'executable_sha256': None, 'extra_files': {'w.bin': None, 'c.toml': 'cc'}}
        self.assertEqual(len(self.runner.check_identity(engine, strict=False)), 1)  # missing --engine-file
        self.assertEqual(len(self.runner.check_identity(engine, strict=True)), 2)   # + unhashable executable
        ok = {'command': ['rapfi'], 'executable_sha256': 'aa', 'extra_files': {}}
        self.assertEqual(self.runner.check_identity(ok, strict=True), [])
        resolved = EngineConfig(command=(Path(sys.executable).name,)).provenance()  # found on PATH
        self.assertIsNotNone(resolved['executable_sha256'])

    def test_external_seeds_are_reserved(self):
        self.assertTrue(EXTERNAL_SEEDS <= RESERVED_SEEDS)
        from evaluation.external import MONITORING_SEEDS
        self.assertEqual(MONITORING_SEEDS, tuple(range(8710, 8721)))  # 8710 = X2 cycle 0, the pre-RL baseline
        self.assertIn(8710, EXTERNAL_SEEDS)
        self.assertIn(8710, RESERVED_SEEDS)  # never training data

    def test_tbv3_identity_fails_closed(self):
        from unittest import mock

        from analysis import canonical, tb_v3, tb_v3_fast5
        halves = (lambda path, policy=None: {k: v for k, v in tb_v3.MODELS.items() if k.startswith('policy')},
                  lambda path: {k: v for k, v in tb_v3.MODELS.items() if k.startswith('value')})
        with mock.patch('analysis.track_b_guard.check_policy_checkpoint', halves[0]), \
                mock.patch('analysis.track_b_guard.check_value_checkpoint', halves[1]):
            cuda = self.runner.tbv3_provenance('p.pt', 'v.pt', 'cuda')
            cpu = self.runner.tbv3_provenance('p.pt', 'v.pt', 'cpu')
            self.assertEqual((cuda['engine_hash'], cuda['runtime_hash'], cuda['value_checkpoint_sha256']),
                             (tb_v3.ENGINE_HASH, tb_v3_fast5.runtime_hash('cuda'), tb_v3.MODELS['value_checkpoint_sha256']))
            self.assertNotEqual(cuda['runtime_hash'], cpu['runtime_hash'])
            with mock.patch.object(canonical, 'CANONICAL_RUNTIME', 'fast4'), self.assertRaises(ValueError):
                self.runner.tbv3_provenance('p.pt', 'v.pt', 'cuda')  # no longer the canonical it stands for
            with mock.patch.object(canonical, '_git_blob_sha256', lambda commit, path: 'f' * 64), \
                    self.assertRaises(RuntimeError):  # the fast5 evidence does not hold for this checkout
                self.runner.tbv3_provenance('p.pt', 'v.pt', 'cuda')
        other = lambda path: {'value_checkpoint_sha256': 'f' * 64, 'value_metadata_sha256': 'f' * 64}
        with mock.patch('analysis.track_b_guard.check_policy_checkpoint', halves[0]), \
                mock.patch('analysis.track_b_guard.check_value_checkpoint', other), self.assertRaises(ValueError):
            self.runner.tbv3_provenance('p.pt', 'v.pt', 'cuda')
        # the identities are part of the run signature: another device or model is another run
        args = Namespace(opening_random_plies=4, opening_radius=2, forbid_check=True)
        engine = {'name': 'rapfi', 'command': ['/x/rapfi'], 'info': [], 'rule': 4, 'executable_sha256': 'aa',
                  'extra_files': {}}
        sig = lambda tbv3: self.runner.run_signature(engine, {'kind': 'tbv3', 'tbv3': tbv3}, args)
        self.assertNotEqual(sig(cuda), sig(cpu))
        self.assertEqual(self.runner.ours_label(Namespace(ours='tbv3', value_device='cuda')), 'tbv3')
        self.assertEqual(self.runner.ours_label(Namespace(ours='tbv3', value_device='cpu')), 'tbv3[cpu]')

    def test_tbv3_game_loop(self):
        """--ours tbv3 in a game against the fake engine: fast5 bound, models cached, the canonical factory's agent
        (stand-in models here; the real factory refuses anything but the frozen ones)."""
        from unittest import mock

        from rt30 import fast5
        if not fast5.available():
            self.skipTest('python -m rt30.vcfcore.build first')
        from analysis import canonical, tb_v3_runtime
        from hybrid.h6_fast import install
        from tests.test_h7_timing import FACTORIES
        self.addCleanup(lambda: (fast5.uninstall(), install('off')))
        tbv3 = {**spec(), 'ours': 'tbv3', 'ours_label': 'tbv3', 'policy_checkpoint': 'p.pt',
                'value_checkpoint': 'v.pt', 'value_device': 'cpu'}
        self.runner._MODELS[('p.pt', 'v.pt', 'cpu')] = (FACTORIES['policy'], FACTORIES['leaf'])
        self.addCleanup(self.runner._MODELS.clear)
        built = []

        def factory(engine, runtime):
            self.assertEqual((engine, runtime), ('TB-V3', 'fast5'))
            def make(policy, leaf, seed, policy_checkpoint, value_checkpoint):
                built.append(seed)
                return tb_v3_runtime._build(policy, leaf, seed)
            return make

        with mock.patch.object(canonical, 'factory', factory):
            record = self.runner.play_one({**self.task(), 'spec': tbv3})
        self.assertTrue(fast5.installed())
        self.assertEqual(len(built), 1)
        self.assertIn(record['result'], ('win', 'loss', 'draw'))
        self.assertTrue(all('route' in m for m in record['our_moves']))

    def test_tbv3_without_the_frozen_models_never_plays(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with self.assertRaises(SystemExit), redirect_stderr(out):
                self.runner.main(['--ours', 'tbv3', '--engine', sys.executable, '--engine-arg', FAKE,
                                  '--seed', '1', '--allow-unreserved-seed', '--pairs', '1',
                                  '--policy-checkpoint', str(Path(tmp) / 'none.pt'),
                                  '--value-checkpoint', str(Path(tmp) / 'none' / 'best.pt'),
                                  '--output', str(Path(tmp) / 'o.json')])
            self.assertIn('setup failed', out.getvalue())
            self.assertFalse((Path(tmp) / 'o.json').exists())


class SummaryTest(unittest.TestCase):
    def game(self, pair, color, result, moves=None):
        moves = moves or [[7, 7], [6, 6], [8, 8], [5, 5]]
        return {'key': f'{pair}/{color}', 'pair': pair, 'ours_color': color, 'result': result, 'moves': moves,
                'our_moves': [{'ply': 1}, {'ply': 3}] if color == 'white' else [{'ply': 2}]}

    def test_counts_score_and_exclusions(self):
        games = [self.game(0, 'black', 'win'), self.game(0, 'white', 'loss'),
                 self.game(1, 'black', 'draw'), self.game(1, 'white', 'win'),
                 {**self.game(2, 'black', 'engine_illegal'), 'illegal': {'ply': 2, 'move': [1, 1], 'reason': 'x'}},
                 {**self.game(2, 'white', 'engine_error'), 'error': 'timeout'}]
        summary = summarize(games, loss_window=2)
        self.assertEqual((summary['raw_scored_games'], summary['engine_illegal'], summary['engine_error']), (4, 1, 1))
        self.assertEqual((summary['complete_pairs'], summary['incomplete_pairs']), (2, [2]))
        self.assertEqual(summary['score'], 0.625)
        self.assertEqual(summary['by_colour']['black'], {'games': 2, 'score': 0.75})
        self.assertEqual(len(summary['illegal_games']), 1)
        self.assertEqual([c['ply'] for c in summary['loss_candidates']], [3])  # last 2 plies of the lost game
        self.assertEqual(summary['fingerprint'], summarize(list(reversed(games)))['fingerprint'])

    def test_half_broken_pair_drops_out_of_every_main_number(self):
        # Pair 1: we won as black, the engine crashed in the white game.
        games = [self.game(0, 'black', 'win'), self.game(0, 'white', 'loss'),
                 self.game(1, 'black', 'win'), {**self.game(1, 'white', 'engine_error'), 'error': 'crash'},
                 self.game(2, 'black', 'draw'), self.game(2, 'white', 'loss')]
        summary = summarize(games)
        self.assertEqual((summary['complete_pairs'], summary['incomplete_pairs']), (2, [1]))
        self.assertEqual(summary['score'], 0.375)  # (1 + 0 + 0.5 + 0) / 4: pairs 0 and 2 only
        self.assertEqual(summary['elo'], round(elo(0.375), 1))
        self.assertEqual(summary['by_colour'], {'black': {'games': 2, 'score': 0.75},
                                                'white': {'games': 2, 'score': 0.0}})
        self.assertEqual((summary['raw_scored_games'], summary['raw_score']), (5, 0.5))  # diagnostics only
        low, high = summary['score_ci95_pairs']  # resamples the same two pairs
        self.assertEqual((low, high), (0.25, 0.5))

    def test_elo_and_bootstrap(self):
        self.assertIsNone(elo(0.0))
        self.assertAlmostEqual(elo(0.75), 190.8, places=1)
        games = [self.game(p, c, r) for p, rs in enumerate([('win', 'win'), ('loss', 'loss'), ('win', 'loss')])
                 for c, r in zip(('black', 'white'), rs)]
        low, high = pair_bootstrap(games)
        self.assertLessEqual(low, 0.5)
        self.assertGreaterEqual(high, 0.5)


if __name__ == '__main__':
    unittest.main()
