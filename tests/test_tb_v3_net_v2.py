"""analysis.tb_v3_net_v2 (candidate factory v2): a 1-channel model pair is checked and plays exactly as under factory
v1; the D1 widened value head (value_channels 4) is accepted and plays; other architectures, another trunk, another
H3 parent, a wrong declared frozen state and a non-H6-2a value are refused; never a canonical path."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from analysis import tb_v3, tb_v3_net, tb_v3_net_v2 as v2
from rt30 import fast5

ROOT = Path(__file__).resolve().parents[1]


def write_pair(tmp, *, value_channels=1, parent=True, declared=True, perturb_trunk=False, stage='H6-2a',
               policy_value_channels=1, seed=0):
    """A random policy-only checkpoint and a value checkpoint on the same trunk and policy head (value head
    re-initialised, optionally widened with the D1 construction), as the H7-c / H7-R trainers write them."""
    import torch

    from hybrid import h7r_d1 as d1
    from hybrid.h6_value_train import frozen_state, state_sha256
    from model.checkpoint import save_checkpoint
    from model.config import ModelConfig
    from model.network import PolicyValueNet

    torch.manual_seed(seed)
    policy_net = PolicyValueNet(ModelConfig(value_channels=policy_value_channels))
    value_net = copy.deepcopy(policy_net) if policy_value_channels == 1 else PolicyValueNet(ModelConfig())
    with torch.no_grad():
        for name, t in value_net.named_parameters():
            if name.startswith('value_head.'):
                t.normal_(0, 0.05)
    if value_channels != 1:
        value_net = d1.widen_value_head(value_net, value_channels, 7)
    if perturb_trunk:
        with torch.no_grad():
            next(value_net.trunk.parameters()).add_(0.01)
    frozen = state_sha256(frozen_state(value_net))
    policy = Path(tmp) / 'policy' / 'best.pt'
    value = Path(tmp) / 'value' / 'best.pt'
    for path, net in ((policy, policy_net), (value, value_net)):
        path.parent.mkdir(parents=True)
        save_checkpoint(path, net)
    policy.with_suffix('.json').write_text(json.dumps(
        {'stage': 'H7-c1-policy', 'policy_trained': True, 'value_trained': False,
         'config': {'channels': policy_net.config.channels, 'blocks': policy_net.config.blocks}}), encoding='utf-8')
    meta = {'stage': stage, 'policy_trained': True, 'value_trained': True, 'trunk_frozen': True,
            'policy_head_frozen': True, 'model_config': asdict(value_net.config)}
    if parent:
        meta['parent_h3_checkpoint_sha256'] = parent if isinstance(parent, str) else tb_v3.MODELS[
            'policy_checkpoint_sha256']
    if declared:
        meta['frozen_state_sha256'] = declared if isinstance(declared, str) else frozen
    value.with_name('best.json').write_text(json.dumps(meta), encoding='utf-8')
    return policy, value, frozen


class CheckTest(unittest.TestCase):
    def test_one_channel_pair_has_the_v1_identity(self):
        with TemporaryDirectory() as tmp:
            policy_path, value_path, frozen = write_pair(tmp)
            policy, leaf = v2.load(policy_path, value_path)
            with mock.patch.object(v2, 'FROZEN_STATE_SHA256', frozen):
                self.assertEqual(v2.check_models(policy, leaf, policy_path, value_path),
                                 tb_v3_net.check_models(policy, leaf, policy_path, value_path))

    def test_the_widened_head_is_accepted_only_by_v2(self):
        with TemporaryDirectory() as tmp:
            policy_path, value_path, frozen = write_pair(tmp, value_channels=4, parent=False)
            policy, leaf = v2.load(policy_path, value_path)
            self.assertEqual(v2.value_architecture(leaf)['value_channels'], 4)
            with self.assertRaisesRegex(ValueError, 'not the frozen architecture'):
                tb_v3_net.check_models(policy, leaf, policy_path, value_path)
            with mock.patch.object(v2, 'FROZEN_STATE_SHA256', frozen):
                identity = v2.check_models(policy, leaf, policy_path, value_path)
            self.assertEqual(identity, tb_v3_net.model_identity(policy_path, value_path))
            self.assertEqual(v2.VALUE_ARCHITECTURES[1], {**tb_v3_net.ARCHITECTURE, 'value_channels': 4})

    def test_refusals(self):
        cases = {
            'not an architecture of tb-v3-net-v2': dict(value_channels=8),
            'are not the frozen ones': dict(perturb_trunk=True),
            'names another H3 parent': dict(parent='f' * 64),
            'metadata frozen state': dict(declared='0' * 64),
        }
        for message, kw in cases.items():
            with self.subTest(message), TemporaryDirectory() as tmp:
                policy_path, value_path, frozen = write_pair(tmp, **kw)
                policy, leaf = v2.load(policy_path, value_path)
                reference = frozen if 'perturb_trunk' not in kw else write_pair(tmp + '/ref')[2]
                with mock.patch.object(v2, 'FROZEN_STATE_SHA256', reference), \
                        self.assertRaisesRegex(ValueError, message):
                    v2.check_models(policy, leaf, policy_path, value_path)
        with TemporaryDirectory() as tmp:  # a widened policy network: the frozen policy loader refuses it already
            policy_path, value_path, _ = write_pair(tmp, policy_value_channels=4)
            with self.assertRaises((ValueError, RuntimeError)):
                v2.load(policy_path, value_path)
        with TemporaryDirectory() as tmp:
            policy_path, value_path, _ = write_pair(tmp, stage='H7-c-value')
            with self.assertRaisesRegex(ValueError, 'not an H6-2a value checkpoint'):
                v2.load(policy_path, value_path)
        with TemporaryDirectory() as tmp:
            policy_path, value_path, frozen = write_pair(tmp)
            policy, leaf = v2.load(policy_path, value_path)
            with mock.patch.object(v2, 'FROZEN_STATE_SHA256', frozen):
                with self.assertRaises(ValueError):
                    v2.check_models(policy, leaf, None, value_path)
                with self.assertRaises(ValueError):
                    v2.check_models(policy, leaf, value_path, value_path)
                with self.assertRaises(TypeError):
                    v2.check_models(object(), leaf, policy_path, value_path)
                value_path.write_bytes(value_path.read_bytes() + b'\0')
                with self.assertRaisesRegex(ValueError, 'leaf reports'):
                    v2.check_models(policy, leaf, policy_path, value_path)

    def test_the_frozen_state_is_the_contracts(self):
        for path in ('docs/h7r/r2/R2-DIAG-1.contract.json', 'docs/h7r/r2/R2-D4-BASELINE-1.contract.json'):
            self.assertEqual(json.loads((ROOT / path).read_text())['frozen_state_sha256'], v2.FROZEN_STATE_SHA256)

    def test_never_a_canonical_path_and_only_research_tools_use_it(self):
        from analysis import canonical
        self.assertFalse(v2.CANONICAL)
        for engine in canonical.ENGINES:
            for runtime in canonical.RUNTIMES[engine]:
                self.assertIsNot(canonical.factory(engine, runtime), v2.make_agent)
        users = []
        for path in sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'scripts').rglob('*.py')):
            if path.name != 'tb_v3_net_v2.py' and re.search(r'\btb_v3_net_v2\b', path.read_text(encoding='utf-8')):
                users.append(path.relative_to(ROOT).as_posix())
        self.assertTrue(all(re.match(r'scripts/h7[cr]_', p) for p in users), users)


class TimingGateTest(unittest.TestCase):
    """scripts/h7r_factory_v2_timing.py: both agents search the same positions (A's trajectory), alternating order,
    warm-up not counted, and the gate thresholds."""

    def test_positions_order_and_gate(self):
        import json as _json
        from scripts import h7r_factory_v2_timing as tg

        seen = {'a': [], 'b': []}

        class Agent:
            def __init__(self, name, cost):
                self.name, self.cost = name, cost

            def select_move(self, game):
                seen[self.name].append((len(game.history), tuple(game.history[-1]) if game.history else None))
                clock.t += self.cost
                return game.legal_moves()[len(game.history) % 3]

        class Clock:
            t = 0.0

            def __call__(self):
                return self.t
        clock = Clock()
        suite = _json.loads((ROOT / tg.SUITE).read_text())
        openings = suite['openings'][:3]
        rows = tg.measure(lambda s: Agent('a', 1.0), lambda s: Agent('b', 1.05), openings, 4, tg.SEED_BASE,
                          clock=clock, log=lambda *a, **k: None)
        self.assertEqual(len(rows), 12)
        self.assertEqual(seen['a'], seen['b'])  # the same positions, warm-up included for both
        self.assertEqual([r['first'] for r in rows[:4]], ['a', 'b', 'a', 'b'])
        self.assertTrue(all(abs(r['a_seconds'] - 1.0) < 1e-9 and abs(r['b_seconds'] - 1.05) < 1e-9 for r in rows))
        g = tg.gate(rows)
        self.assertTrue(g['passed'])
        self.assertAlmostEqual(g['mean_ratio'], 1.05)
        slow = [{**r, 'b_seconds': 1.15} for r in rows]
        self.assertFalse(tg.gate(slow)['passed'])  # +15 % mean
        spiky = [{**r, 'b_seconds': 2.0 if i == 0 else 1.0} for i, r in enumerate(rows)]
        self.assertFalse(tg.gate(spiky)['passed'])  # p95 x2 although the mean is +8 %
        self.assertEqual((tg.MEAN_RATIO, tg.P95_RATIO, tg.OPENINGS, tg.PLIES), (1.10, 1.20, 24, 6))


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class AgentTest(unittest.TestCase):
    def tearDown(self):
        from hybrid.h6_fast import install
        fast5.uninstall()
        install('off')

    def play(self, make, policy_path, value_path, plies=3):
        from renju import Game
        from scripts.run_mcts_v8_benchmark import _move_record
        from scripts.tb_v3_promotion import _untimed
        from tests.test_tb_v3 import LINE

        policy, leaf = v2.load(policy_path, value_path)
        agent, game, rows = make(policy, leaf, 7, policy_path, value_path), Game(), []
        for m in LINE:
            game.play(*m)
        for _ in range(plies):
            move = agent.select_move(game)
            rows.append((list(move), _untimed(_move_record(len(game.history), 0.0, agent.diagnostics))))
            game.play(*move)
            game.play(*game.legal_moves()[0])
        return agent, rows

    def test_a_one_channel_pair_plays_as_under_v1_and_the_widened_head_plays(self):
        from analysis import tb_v2_runtime, tb_v3_fast5
        tb_v2_runtime.install()
        tb_v3_fast5.install()
        with TemporaryDirectory() as tmp:
            policy_path, value_path, frozen = write_pair(tmp)
            with mock.patch.object(v2, 'FROZEN_STATE_SHA256', frozen):
                _, rows1 = self.play(tb_v3_net.make_tb_v3_net_agent, policy_path, value_path)
                agent, rows2 = self.play(v2.make_agent, policy_path, value_path)
            self.assertEqual(rows1, rows2)
            self.assertEqual(agent.factory, v2.FACTORY_VERSION)
        with TemporaryDirectory() as tmp:
            policy_path, value_path, frozen = write_pair(tmp, value_channels=4, parent=False)
            with mock.patch.object(v2, 'FROZEN_STATE_SHA256', frozen):
                agent, rows = self.play(v2.make_agent, policy_path, value_path)
            self.assertEqual(len(rows), 3)
            identity = tb_v3_net.model_identity(policy_path, value_path)
            p = v2.provenance('cuda', identity, {**tb_v3_net.ARCHITECTURE, 'value_channels': 4})
            self.assertEqual((p['factory'], p['canonical'], p['value_architecture']['value_channels']),
                             (v2.FACTORY_VERSION, False, 4))


if __name__ == '__main__':
    unittest.main()
