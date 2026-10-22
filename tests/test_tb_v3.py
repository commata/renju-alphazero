"""TB-V3 Resolve: identity, the fail-closed factory, the canonical alias and the promotion replay tool."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from analysis import canonical, tb_v2_value_e4v2 as cand, tb_v3, tb_v3_runtime as v3rt, track_b_guard as guard
from renju import Game
from scripts import tb_v3_promotion as promo

ROOT = Path(__file__).resolve().parents[1]
LINE = [(7, 7), (7, 8)]  # the opening of the replayed test game


class IdentityTest(unittest.TestCase):
    def test_names_and_hash(self):
        self.assertEqual((tb_v3.ENGINE_ID, tb_v3.ENGINE_CODENAME, tb_v3.ENGINE_NAME),
                         ('TB-V3', 'Resolve', 'TB-V3 Resolve'))
        self.assertEqual(tb_v3.config_hash(), tb_v3.ENGINE_HASH)
        tb_v3.check_identity()

    def test_it_is_the_adopted_candidate(self):
        self.assertEqual(tb_v3.ATTACK, cand.ATTACK)
        self.assertEqual(tb_v3.ATTACK_SOURCES, cand.ATTACK_SOURCES)
        self.assertEqual(tb_v3.BASE_ENGINE_HASH, cand.BASE_ENGINE_HASH)
        self.assertEqual(tb_v3.ADOPTION['candidate_hash'], cand.candidate_hash())
        arena = json.loads((ROOT / tb_v3.ADOPTION['arena']).read_text(encoding='utf-8'))
        self.assertEqual((arena['verdict']['decision'], arena['candidate_hash']), ('ADOPT', cand.candidate_hash()))
        from scripts.run_mcts_v8_benchmark import file_sha256
        self.assertEqual(file_sha256(ROOT / tb_v3.ADOPTION['arena']), tb_v3.ADOPTION['arena_sha256'])

    def test_the_hash_covers_models_and_code(self):
        for name, change in (('MODELS', {**tb_v3.MODELS, 'value_checkpoint_sha256': 'f' * 64}),
                             ('CODE_MANIFEST_SHA256', 'f' * 64), ('ATTACK', {**tb_v3.ATTACK, 'killer': False})):
            with mock.patch.object(tb_v3, name, change):
                self.assertNotEqual(tb_v3.config_hash(), tb_v3.ENGINE_HASH, name)
                with self.assertRaises(RuntimeError):
                    tb_v3.check_identity()

    def test_manifest_is_the_closure_and_extends_tb_v2(self):
        from scripts import check_track_b_frozen as checker
        v3 = guard.read_manifest(checker.MANIFEST_V3)
        self.assertEqual(sorted(v3), checker.closure('tb-v3'))
        self.assertNotIn('src/analysis/tb_v3.py', v3)  # it holds the manifest's hash: pinned by the runtime hash
        v2 = guard.read_manifest()
        self.assertEqual({k: v3[k] for k in v2}, v2)  # TB-V2-Value's files, unchanged
        for name in ('src/analysis/tb_v3_runtime.py', 'src/analysis/tb_v2_value_e4v2.py', *tb_v3.ATTACK_SOURCES):
            self.assertIn(name, v3)

    def test_runtime_hash_pins_the_identity_file(self):
        before = v3rt.runtime_hash('cuda')
        self.assertNotEqual(before, v3rt.runtime_hash('cpu'))
        with mock.patch.object(v3rt, 'identity_sha256', lambda: 'f' * 64):
            self.assertNotEqual(v3rt.runtime_hash('cuda'), before)


class FactoryTest(unittest.TestCase):
    def test_fails_closed(self):
        from tests.test_h7_timing import FACTORIES
        policy, leaf = FACTORIES['policy'], FACTORIES['leaf']
        with self.assertRaises(ValueError):  # no checkpoint paths: nothing to verify, so refused
            v3rt.make_tb_v3_agent(policy, leaf, 1, None, None)
        with TemporaryDirectory() as tmp:  # a checkpoint that cannot be read is refused, never skipped
            missing = Path(tmp) / 'best.pt'
            for check in (guard.check_policy_checkpoint, guard.check_value_checkpoint):
                with self.assertRaises((FileNotFoundError, ValueError)):
                    check(missing)
        with self.assertRaises(TypeError):  # stand-ins are not the real RootPolicy / ValueLeaf
            with mock.patch.object(guard, 'check_policy_checkpoint', lambda *a: {}), \
                    mock.patch.object(guard, 'check_value_checkpoint', lambda *a: {}):
                v3rt.make_tb_v3_agent(policy, leaf, 1, 'p', 'v')
        with mock.patch.object(tb_v3, 'ENGINE_HASH', 'f' * 64), self.assertRaises(RuntimeError):
            v3rt.make_tb_v3_agent(policy, leaf, 1, 'p', 'v')  # the identity is checked first

    def test_same_moves_and_steps_as_the_candidate(self):
        from rt30 import fast4_forbidden
        from scripts.h7_timing import restore_fast_mode
        from tests.test_h7_timing import FACTORIES
        if not fast4_forbidden.available():
            self.skipTest('python -m rt30.native.build first')
        from analysis import tb_v2_runtime
        with restore_fast_mode():
            tb_v2_runtime.install()
            a = v3rt._build(FACTORIES['policy'], FACTORIES['leaf'], 5)
            b = cand.E4v2AttackAgent(tb_v2_runtime.make_agent(FACTORIES['policy'], FACTORIES['leaf'], seed=5))
            self.assertEqual(a.name, 'TB-V3 Resolve/fast4')
            game = Game()
            for m in LINE:
                game.play(*m)
            for _ in range(3):
                ma, mb = a.select_move(game), b.select_move(game)
                self.assertEqual((ma, promo._untimed(a.attack_info), a.cooldown.history),
                                 (mb, promo._untimed(b.attack_info), b.cooldown.history))
                game.play(*ma)
                game.play(*game.legal_moves()[0])


class CanonicalTest(unittest.TestCase):
    def test_one_alias_versioned_factories(self):
        self.assertEqual(canonical.CANONICAL_ENGINE_ID, 'TB-V3')  # since the promotion replay passed
        record = json.loads((ROOT / 'docs/h7/tb_v3_promotion.json').read_text(encoding='utf-8'))
        self.assertEqual((record['verdict'], record['engine_hash'], record['identity_sha256'], record['runtime_hash']),
                         ('PROMOTION_PASS', tb_v3.ENGINE_HASH, v3rt.identity_sha256(), v3rt.runtime_hash('cuda')))
        self.assertIs(canonical.factory(), v3rt.make_tb_v3_agent)
        self.assertIs(canonical.factory('TB-V3'), v3rt.make_tb_v3_agent)
        self.assertIs(canonical.factory('TB-V2-Value'), guard.make_tb_v2_agent)
        self.assertIs(guard.make_canonical_agent, guard.make_tb_v2_agent)  # the candidate's base, for good
        with self.assertRaises(ValueError):
            canonical.factory('TB-V4')

    def test_benchmark_arm(self):
        from analysis.tb_v2_value import ENGINE_OPTIONS, TB_V2_VALUE
        from scripts.run_mcts_v8_benchmark import is_canonical, make_opponent, v8_config
        config = v8_config('tb_v3', {'puct_c': 1.5})
        self.assertEqual(config, {**TB_V2_VALUE, **ENGINE_OPTIONS, 'tb_v3': True})
        self.assertTrue(is_canonical('tb_v3', 'fast4', config))
        self.assertFalse(is_canonical('tb_v3', 'fast3', config))
        for name in ('v8:tb_v3', 'v8:puct_value_vct2_50k_tt_e4v2'):
            with self.assertRaises(ValueError):  # never a bare V8 without its wrapper
                make_opponent(name, 1, {'puct_c': 1.5})


def recorded_game(own=3):
    """A short game of the candidate (black, stand-in models) against first-legal-move, recorded as the arena
    records it."""
    from analysis import tb_v2_runtime
    from scripts.run_mcts_v8_benchmark import _move_record
    from tests.test_h7_timing import FACTORIES

    agent = cand.E4v2AttackAgent(tb_v2_runtime.make_agent(FACTORIES['policy'], FACTORIES['leaf'], seed=
                                                          promo_seed()))
    game, v8 = Game(), []
    for m in LINE:
        game.play(*m)
    for _ in range(own):
        move = agent.select_move(game)
        record = _move_record(len(game.history), 1.0, agent.diagnostics)
        record['e4_attack'] = dict(agent.attack_info)
        record['played'] = list(move)
        v8.append(record)
        game.play(*move)
        game.play(*game.legal_moves()[0])
    return {'key': 'e@v8:full/8441/0/black', 'engine': 'TB-V3', 'seed': 77, 'moves': [list(m) for m in game.history],
            'v8_moves': v8, 'run_sha256': 'x'}


def promo_seed():
    from scripts.run_mcts_v8_benchmark import derive_seed
    return derive_seed(77, 'v8')


class PromotionTest(unittest.TestCase):
    def test_replay_matches_and_finds_differences(self):
        from rt30 import fast4_forbidden
        from scripts.h7_timing import restore_fast_mode
        from tests.test_h7_timing import FACTORIES
        if not fast4_forbidden.available():
            self.skipTest('python -m rt30.native.build first')

        def build(engine, seed):
            self.assertEqual(engine, 'TB-V3')
            return v3rt._build(FACTORIES['policy'], FACTORIES['leaf'], seed)

        from analysis import tb_v2_runtime
        with restore_fast_mode():
            tb_v2_runtime.install()
            task = recorded_game()
            row = promo.replay_game(task, build)
            self.assertEqual((row['moves'], row['move_mismatch'], row['decision_mismatch'], row['record_mismatch']),
                             (3, [], [], []))
            bad = json.loads(json.dumps(task))
            bad['v8_moves'][1]['e4_attack']['skip'] = 'cooldown'
            bad['v8_moves'][2]['route'] = 'own_vct2'
            row = promo.replay_game(bad, build)
            self.assertEqual([d['ply'] for d in row['decision_mismatch']], [4, 6])
            self.assertEqual(row['decision_mismatch'][0]['fields'], ['e4.skip'])
            bad = json.loads(json.dumps(task))
            bad['v8_moves'][0]['played'] = [0, 0]  # the record and the game disagree: refused
            with self.assertRaises(ValueError):
                promo.replay_game(bad, build)
        e = promo.summarize([row | {'engine': 'TB-V3'}], 'TB-V3')
        self.assertEqual(promo.verdict(e, 1, 3, None)['verdict'], 'PROMOTION_FAIL')

    def test_verdict(self):
        good = {'games': 200, 'moves': 3201, 'move_mismatches': [], 'decision_mismatches': []}
        self.assertEqual(promo.verdict(good, 200, 3201, None)['verdict'], 'PROMOTION_PASS')
        self.assertEqual(promo.verdict({**good, 'moves': 3200}, 200, 3201, None)['verdict'], 'PROMOTION_FAIL')
        self.assertEqual(promo.verdict({**good, 'move_mismatches': ['k@4']}, 200, 3201, None)['verdict'],
                         'PROMOTION_FAIL')
        out = promo.verdict(good, 200, 3201, {**good, 'decision_mismatches': [{'ply': 3}]})
        self.assertEqual((out['verdict'], out['baseline']), ('PROMOTION_PASS', 'DIFFERS'))  # B is not the gate

    def test_compare_ignores_times_only(self):
        rec = {'ply': 4, 'seconds': 1.0, 'route': 'tree', 'changed': False, 'v7_move': [1, 1], 'played': [1, 1],
               'tree': {'seconds': 0.5, 'simulations': 50}, 'e4_attack': {'ran': True, 'seconds': 0.1, 'move': None}}
        same = json.loads(json.dumps(rec))
        same['seconds'], same['tree']['seconds'], same['e4_attack']['seconds'] = 9.0, 9.0, 9.0
        self.assertEqual(promo.compare(rec, same), {'move': False, 'decision': [], 'record': []})
        other = json.loads(json.dumps(rec))
        other['tree']['simulations'] = 49
        self.assertEqual(promo.compare(rec, other), {'move': False, 'decision': [], 'record': ['tree']})

    def test_refuses_another_arena(self):
        with TemporaryDirectory() as tmp:
            fake = Path(tmp) / 'arena.json'
            fake.write_text(json.dumps({'verdict': {'decision': 'ADOPT'}, 'runs': {}}))
            with self.assertRaises(SystemExit):
                promo.main(['--runs', str(fake), '--arena', str(fake)])


if __name__ == '__main__':
    unittest.main()


class Ser5TailTest(unittest.TestCase):
    def test_pairs_tb_v2_and_what_ifs(self):
        from scripts import tb_v3_ser5_tail as tail

        def row(pid, seconds, e4=None, ran=True):
            r = {'id': pid, 'source': 'e3a', 'move': [1, 1], 'seconds': seconds, 'decision': {'route': 'tree'},
                 'times': {'tree': 1.0, 'root': 1.0, 'vct2': 0.5, 'value': 0.5}}
            if e4 is not None:
                r.update(e4_seconds=e4, e4_attack={'ran': ran, 'skip': None if ran else 'route', 'exhausted': ran})
            return r

        v3 = {'engine_name': 'TB-V3 Resolve', 'rows': [row('a', 28.0, 8.0), row('b', 3.0, 0.0, ran=False)]}
        v2 = {'rows': [row('a', 20.0), row('b', 3.0)]}
        out = tail.analyse([v3, v3], [v2])
        self.assertEqual(out['over_rule'], ['a'])
        self.assertEqual(out['slowest'][0]['step_seconds'], 8.0)
        self.assertEqual(out['what_if']['step_beside_base']['max'], 20.0)
        self.assertEqual(out['what_if']['attack_capped_4s']['max'], 24.0)
        self.assertEqual(out['skips']['route'], 1)
        with self.assertRaises(ValueError):
            tail.analyse([v2], [])
