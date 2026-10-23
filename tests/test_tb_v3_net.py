"""analysis.tb_v3_net: the frozen TB-V3 engine / fast5 runtime with candidate models, never a canonical path."""
from dataclasses import asdict
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from analysis import tb_v3, tb_v3_net
from rt30 import fast5

ROOT = Path(__file__).resolve().parents[1]


def write_models(tmp, *, channels=64, parent=None, seed=0):
    """A random policy-only checkpoint and an H6-2a-style value checkpoint, as the H7-c trainers will write them."""
    import torch

    from model.checkpoint import save_checkpoint
    from model.config import ModelConfig
    from model.network import PolicyValueNet

    torch.manual_seed(seed)
    config = ModelConfig(channels=channels)
    policy = Path(tmp) / 'policy' / 'best.pt'
    value = Path(tmp) / 'value' / 'best.pt'
    for path in (policy, value):
        path.parent.mkdir(parents=True)
        save_checkpoint(path, PolicyValueNet(config))
    policy.with_suffix('.json').write_text(json.dumps(
        {'stage': 'H7-c1-policy', 'policy_trained': True, 'value_trained': False,
         'config': {'channels': config.channels, 'blocks': config.blocks}}), encoding='utf-8')
    value.with_name('best.json').write_text(json.dumps(
        {'stage': 'H6-2a', 'policy_trained': True, 'value_trained': True, 'trunk_frozen': True,
         'policy_head_frozen': True, 'model_config': asdict(config),
         'parent_h3_checkpoint_sha256': parent or tb_v3.MODELS['policy_checkpoint_sha256']}), encoding='utf-8')
    return policy, value


def load(policy_path, value_path):
    from hybrid.h4_policy import RootPolicy
    from hybrid.h6_value_leaf import ValueLeaf

    return RootPolicy(policy_path), ValueLeaf(value_path, expected_sha256=None)


class ModelIdentityTest(unittest.TestCase):
    def test_architecture_is_the_frozen_default(self):
        from model.config import ModelConfig
        self.assertEqual(asdict(ModelConfig()), tb_v3_net.ARCHITECTURE)

    def test_identity_and_refusals(self):
        with TemporaryDirectory() as tmp:
            policy_path, value_path = write_models(tmp)
            policy, leaf = load(policy_path, value_path)
            identity = tb_v3_net.check_models(policy, leaf, policy_path, value_path)
            self.assertEqual(identity, tb_v3_net.model_identity(policy_path, value_path))
            self.assertEqual(sorted(identity), sorted(tb_v3.MODELS))
            self.assertNotEqual(identity, tb_v3.MODELS)
            self.assertTrue(tb_v3_net.agent_name(identity).startswith('TB-V3 Resolve/fast5 + net '))
            with self.assertRaises(ValueError):
                tb_v3_net.check_models(policy, leaf, None, value_path)
            with self.assertRaises(ValueError):  # the policy came from another file
                tb_v3_net.check_models(policy, leaf, value_path, value_path)
            with self.assertRaises(TypeError):
                tb_v3_net.check_models(object(), leaf, policy_path, value_path)
            value_path.write_bytes(value_path.read_bytes() + b'\0')  # the leaf no longer matches its file
            with self.assertRaises(ValueError):
                tb_v3_net.check_models(policy, leaf, policy_path, value_path)
        with TemporaryDirectory() as tmp:
            policy_path, value_path = write_models(tmp, parent='f' * 64)  # value head on another trunk
            with self.assertRaises(ValueError):
                tb_v3_net.check_models(*load(policy_path, value_path), policy_path, value_path)
        with TemporaryDirectory() as tmp:
            policy_path, value_path = write_models(tmp, channels=32)
            meta = json.loads(policy_path.with_suffix('.json').read_text(encoding='utf-8'))
            self.assertEqual(meta['config']['channels'], 32)
            with self.assertRaises(ValueError):  # another architecture
                tb_v3_net.check_models(*load(policy_path, value_path), policy_path, value_path)

    def test_never_a_canonical_path(self):
        from analysis import canonical
        self.assertFalse(tb_v3_net.CANONICAL)
        for engine in canonical.ENGINES:
            for runtime in canonical.RUNTIMES[engine]:
                self.assertIsNot(canonical.factory(engine, runtime), tb_v3_net.make_tb_v3_net_agent)
        users = []
        for path in sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'scripts').rglob('*.py')):
            if path.name != 'tb_v3_net.py' and re.search(r'\btb_v3_net\b', path.read_text(encoding='utf-8')):
                users.append(path.relative_to(ROOT).as_posix())
        # Only the H7-c tools may build candidate models; the canonical, web and benchmark code never does.
        self.assertTrue(all(re.match(r'scripts/h7c_', p) for p in users), users)


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class CandidateAgentTest(unittest.TestCase):
    def tearDown(self):
        from hybrid.h6_fast import install
        fast5.uninstall()
        install('off')

    def test_builds_the_frozen_engine_with_candidate_models(self):
        from analysis import tb_v2_runtime, tb_v3_fast5
        from renju import Game
        from renju.game import OPENING_MOVE

        with TemporaryDirectory() as tmp:
            policy_path, value_path = write_models(tmp)
            policy, leaf = load(policy_path, value_path)
            with self.assertRaises(RuntimeError):  # fast5 not bound
                tb_v3_net.make_tb_v3_net_agent(policy, leaf, 1, policy_path, value_path)
            tb_v2_runtime.install()
            tb_v3_fast5.install()
            agent = tb_v3_net.make_tb_v3_net_agent(policy, leaf, 1, policy_path, value_path)
            identity = tb_v3_net.model_identity(policy_path, value_path)
            self.assertEqual((agent.name, agent.models), (tb_v3_net.agent_name(identity), identity))
            game = Game()
            game.play(*OPENING_MOVE)
            move = agent.select_move(game)
            self.assertIn(tuple(move), {tuple(m) for m in game.legal_moves()})
            p = tb_v3_net.provenance('cuda', identity)
            self.assertEqual((p['canonical'], p['frozen_models'], p['runtime_hash']),
                             (False, False, tb_v3_fast5.runtime_hash('cuda')))
            self.assertEqual(p['model_hash'], tb_v3_net.model_hash(identity))

    def test_same_agent_as_the_frozen_build(self):
        """The arena record of the candidate replays through this build exactly (moves, decisions, every field)."""
        from analysis import tb_v2_runtime, tb_v3_fast5
        from scripts import tb_v3_promotion as promo
        from tests.test_h7_timing import FACTORIES
        from tests.test_tb_v3 import recorded_game

        tb_v2_runtime.install()
        task = recorded_game(own=4)
        tb_v3_fast5.install()
        row = promo.replay_game(task, lambda engine, seed: tb_v3_net._build(FACTORIES['policy'], FACTORIES['leaf'],
                                                                             seed))
        self.assertEqual((row['moves'], row['move_mismatch'], row['decision_mismatch'], row['record_mismatch']),
                         (4, [], [], []))


@unittest.skipUnless((ROOT / 'runs/h3_policy_64x4/best.pt').is_file() and (ROOT / 'runs/h6_value_head_v3/best.pt').is_file(),
                     'the frozen models are not on this machine')
class FrozenModelsTest(unittest.TestCase):
    def test_frozen_models_pass_as_candidates(self):
        """The frozen H3 / H6 pair is a valid candidate (cycle 0 of H7-c) and is recognised as the frozen one."""
        policy_path, value_path = ROOT / 'runs/h3_policy_64x4/best.pt', ROOT / 'runs/h6_value_head_v3/best.pt'
        identity = tb_v3_net.check_models(*load(policy_path, value_path), policy_path, value_path)
        self.assertEqual(identity, tb_v3.MODELS)

    @unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
    def test_frozen_models_play_as_the_canonical(self):
        """With the frozen models the candidate factory plays the canonical's moves and records."""
        from analysis import tb_v2_runtime, tb_v3_fast5
        from hybrid.h6_fast import install
        from renju import Game
        from scripts.run_mcts_v8_benchmark import _move_record
        from scripts.tb_v3_promotion import _untimed
        from tests.test_tb_v3 import LINE

        policy_path, value_path = ROOT / 'runs/h3_policy_64x4/best.pt', ROOT / 'runs/h6_value_head_v3/best.pt'
        tb_v2_runtime.install()
        tb_v3_fast5.install()
        try:
            out = []
            for make in (tb_v3_fast5.make_tb_v3_fast5_agent, tb_v3_net.make_tb_v3_net_agent):
                policy, leaf = load(policy_path, value_path)
                agent, game, rows = make(policy, leaf, 7, policy_path, value_path), Game(), []
                for m in LINE:
                    game.play(*m)
                for _ in range(3):
                    move = agent.select_move(game)
                    record = _move_record(len(game.history), 0.0, agent.diagnostics)
                    rows.append((list(move), _untimed(record)))
                    game.play(*move)
                    game.play(*game.legal_moves()[0])
                out.append(rows)
            self.assertEqual(out[0], out[1])
        finally:
            fast5.uninstall()
            install('off')


if __name__ == '__main__':
    unittest.main()
