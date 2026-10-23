"""H7-c training: policy fine-tune (CE + KL, guard-aware selection) and value-head continuation (50:50, leakage)."""
import argparse
import copy
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

from hybrid import h7c_train as tr
from hybrid.h3_cache import FIELD_BYTES, pack
from hybrid.h6_labels import canonical_hash
from model.config import ModelConfig
from model.network import PolicyValueNet
from renju import Game


def random_game(seed: int, plies: int) -> list[list[int]]:
    rng, game = Random(seed), Game()
    game.play(7, 7)
    while len(game.history) < plies and not game.done:
        near = [m for m in game.legal_moves() if abs(m[0] - 7) <= 3 and abs(m[1] - 7) <= 3]
        game.play(*rng.choice(near))
    return [list(m) for m in game.history]


def policy_rows(n_games=6, split='train'):
    rows = []
    for g in range(n_games):
        moves = random_game(g + (100 if split == 'val' else 0), 12)
        for ply in range(3, len(moves)):
            game = Game()
            for m in moves[:ply]:
                game.play(*m)
            legal = game.legal_moves()
            a, b = legal[0], legal[len(legal) // 2]
            kl = ply % 2 == 0
            target = [[*a, 0.75], [*b, 0.25]] if kl else [[*a, 1.0]]
            rows.append({'game_id': f'h7c:c1:8801:{g}', 'ply': ply, 'moves': moves[:ply], 'target': target,
                         'kl': kl, 'mask': False, 'split': split, 'format': 'h7c-policy-row-v1'})
    return rows


def guard_data(rows):
    """An H3-cache-like split of the same positions (target = the first target move)."""
    t = tr.policy_tensors(rows)
    t['target'] = t['target'].argmax(1)
    return t


def value_split(seed: int, games: int, *, shared=None):
    """Packed value states (h6_value_cache layout) of random games; ``shared`` adds those positions."""
    from hybrid.h6_value_cache import _tensors, game_records

    records, index = [], []
    sources = [random_game(seed * 50 + g, 10) for g in range(games)] + list(shared or [])
    for g, moves in enumerate(sources):
        labelled = [(ply, 1 if ply % 2 else -1, 0.1, 0) for ply in range(3, len(moves))]
        for rec in game_records(moves, labelled):
            records.append(rec)
            index.append(g)
    return _tensors(records, index), sources


def small_model(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet(ModelConfig()).eval()


class ScheduleTest(unittest.TestCase):
    def test_warmup_is_five_percent_capped(self):
        self.assertEqual(tr.warmup_steps(110), 6)
        self.assertEqual(tr.warmup_steps(40), 5)
        self.assertEqual(tr.warmup_steps(100_000), 100)
        self.assertEqual(tr.evaluation_steps(110, 10)[-1], 110)
        self.assertEqual(len(tr.evaluation_steps(110, 10)), 10)
        self.assertEqual(tr.evaluation_steps(3, 10), [1, 2, 3])
        self.assertAlmostEqual(tr.lr_factor(109, 110, 6, 0.05), 0.05, places=2)


class HashTest(unittest.TestCase):
    def test_state_hash_is_the_canonical_hash(self):
        data, games = value_split(1, 3)
        want = [canonical_hash(moves[:ply]) for moves in games for ply in range(3, len(moves))]
        self.assertEqual(tr.state_hashes(data), want)


class PolicyTest(unittest.TestCase):
    def test_terms(self):
        model = small_model()
        t = tr.policy_tensors(policy_rows(2))
        index = torch.arange(t['target'].shape[0])
        planes, legal = tr.planes_from_states(t, index)
        ce, kld, loss = tr._policy_terms(model, model, planes, legal, t['target'], t['kl'], 0.5)
        self.assertTrue(torch.allclose(kld, torch.zeros_like(kld), atol=1e-6))  # student = teacher
        self.assertTrue(torch.allclose(loss, ce))
        other = small_model(1)
        _, kld2, loss2 = tr._policy_terms(other, model, planes, legal, t['target'], t['kl'], 0.5)
        self.assertTrue((kld2 > 0).all())
        self.assertTrue(torch.allclose(loss2 - tr._policy_terms(other, model, planes, legal, t['target'],
                                                                t['kl'], 0.0)[2], 0.5 * t['kl'].float() * kld2))

    def test_planes_match_the_encoder(self):
        from model.encoding import encode_game

        rows = policy_rows(1)
        t = tr.policy_tensors(rows)
        planes, _ = tr.planes_from_states(t, torch.arange(len(rows)))
        for i, row in enumerate(rows):
            game = Game()
            for m in row['moves']:
                game.play(*m)
            self.assertTrue(torch.equal(planes[i], encode_game(game)))

    def test_illegal_target_refused(self):
        rows = policy_rows(1)
        rows[0]['target'] = [[*rows[0]['moves'][0], 1.0]]  # an occupied point
        with self.assertRaises(ValueError):
            tr.policy_tensors(rows)

    def run_policy(self, **changes):
        from hybrid.h3_train import H3Config, evaluate

        parent = small_model()
        train, val = tr.policy_tensors(policy_rows(6)), tr.policy_tensors(policy_rows(2, 'val'))
        guard = guard_data(policy_rows(2, 'val'))
        cfg = tr.PolicyConfig(batch_size=16, epochs=1.0, evaluations=3, learning_rate=1e-3, **changes)
        before = copy.deepcopy(parent.state_dict())
        result = tr.train_policy(cfg, parent, parent, train, val, guard,
                                 lambda m, d: evaluate(m, d, H3Config(), torch.device('cpu')), log=lambda *a, **k: None)
        for k, v in parent.state_dict().items():  # the parent and the teacher are never trained in place
            self.assertTrue(torch.equal(v, before[k]), k)
        return result

    def test_training_and_selection(self):
        result = self.run_policy(guard_drop=1.0)  # every evaluation eligible: the selection rule decides
        record = result['record']
        self.assertEqual(record['warmup_steps'], tr.warmup_steps(record['total_steps']))
        self.assertEqual(len(record['evaluations']), 3)
        self.assertEqual(record['verdict'], 'CANDIDATE')
        eligible = [e for e in record['evaluations'] if e['eligible']]
        self.assertEqual(record['best']['val']['objective'], min(e['val']['objective'] for e in eligible))
        bn = [k for k in result['state'] if 'running_mean' in k or 'running_var' in k]
        parent = small_model()
        for k in bn:  # BatchNorm statistics stay the parent's
            self.assertTrue(torch.equal(result['state'][k], parent.state_dict()[k]), k)
        nothing = self.run_policy(guard_drop=-1.0)  # an impossible guard: no candidate
        self.assertEqual((nothing['record']['verdict'], nothing['state']), ('NO_CANDIDATE', None))


class ValueTest(unittest.TestCase):
    def caches(self):
        cycle_train, games = value_split(1, 4)
        replay_train, _ = value_split(2, 10)
        cycle_val, _ = value_split(3, 2, shared=games[:1])        # one cycle val game repeats a cycle train game
        replay_val, _ = value_split(4, 2, shared=games[1:2])      # one H6 val game repeats a cycle train game
        return {'train': cycle_train, 'val': cycle_val}, {'train': replay_train, 'val': replay_val}

    def test_union_masks(self):
        cycle, replay = self.caches()
        c, r, counts = tr.union_masks(cycle, replay)
        self.assertGreater(counts['cycle_val_dropped'], 0)
        self.assertGreater(counts['replay_val_dropped'], 0)
        train = set(tr.state_hashes(cycle['train'])) | set(tr.state_hashes(replay['train']))
        self.assertFalse(set(tr.state_hashes(c['val'])) & train)
        self.assertFalse(set(tr.state_hashes(r['val'])) & train)

    def test_mixed_epoch_is_half_and_half(self):
        cycle, replay = self.caches()
        a = tr.mixed_epoch(cycle['train'], replay['train'], 0, 7)
        n = cycle['train']['value'].shape[0]
        self.assertEqual((int((a['source'] == 0).sum()), int((a['source'] == 1).sum())), (n, n))
        b = tr.mixed_epoch(cycle['train'], replay['train'], 1, 7)
        self.assertFalse(torch.equal(a['black'], b['black']))  # another replay sample each epoch
        self.assertTrue(torch.equal(a['black'], tr.mixed_epoch(cycle['train'], replay['train'], 0, 7)['black']))
        with self.assertRaises(ValueError):
            tr.mixed_epoch(replay['train'], cycle['train'], 0, 7)  # the replay must be the larger source

    def test_value_training(self):
        from hybrid.h6_value_train import frozen_state, state_sha256

        cycle, replay = self.caches()
        parent = small_model()
        frozen = state_sha256(frozen_state(parent))
        cfg = tr.ValueConfig(batch_size=16, epochs=1.0, evaluations=3, learning_rate=1e-3)
        result = tr.train_value(tr.ValueConfig(**{**tr.config_dict(cfg), 'h6_guard_ratio': 100.0}), parent, cycle,
                                replay, log=lambda *a, **k: None)
        self.assertEqual(result['record']['frozen_state_sha256'], frozen)
        self.assertIsNotNone(result['state'])
        model = small_model()
        model.load_state_dict(result['state'])
        self.assertEqual(state_sha256(frozen_state(model)), frozen)  # only the value head moved
        none = tr.train_value(tr.ValueConfig(batch_size=16, epochs=1.0, evaluations=2, h6_guard_ratio=0.0),
                              parent, cycle, replay, log=lambda *a, **k: None)
        self.assertEqual(none['record']['verdict'], 'NO_CANDIDATE')


class CliTest(unittest.TestCase):
    """Both training commands end to end on stand-in networks and tiny data."""

    def test_value_cli(self):
        from hybrid.h6_value_cache import build_value_cache
        from scripts import h7c_train_value as tv
        from tests.test_h7c_labels import build, game
        from tests.test_tb_v3_net import write_models

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows, _, _ = build([game(i) for i in range(12)])
            from scripts.h7c_labels import value_row_dict, write_atomic
            write_atomic({tmp / 'cycle_rows.jsonl.gz': [value_row_dict(r) for r in rows]})
            build_value_cache(tmp / 'cycle_rows.jsonl.gz', tmp / 'cycle_cache', progress_seconds=1e9)
            # the replay: H6-format rows of random games, its own cache
            from hybrid.h6_labels import game_rows
            replay_rows = []
            for g in range(30):
                moves = random_game(500 + g, 12)
                replay_rows += game_rows({'id': f'renjunet:{g}', 'moves': moves, 'winner': 1, 'real_draw': False},
                                         source='renjunet', split='val' if g < 4 else 'train', outcome_weight=0.1,
                                         from_ply=3)
            write_atomic({tmp / 'replay_rows.jsonl.gz': [r.to_dict() for r in replay_rows]})
            manifest = build_value_cache(tmp / 'replay_rows.jsonl.gz', tmp / 'replay_cache', progress_seconds=1e9)
            _, value = write_models(tmp)
            meta = json.loads(value.with_name('best.json').read_text(encoding='utf-8'))
            meta['labels_sha256'] = manifest['labels_sha256']
            value.with_name('best.json').write_text(json.dumps(meta), encoding='utf-8')
            config = tmp / 'value.json'
            config.write_text(json.dumps({'batch_size': 16, 'epochs': 1.0, 'evaluations': 2, 'h6_guard_ratio': 100.0}),
                              encoding='utf-8')
            args = argparse.Namespace(cycle=1, config=config, cycle_cache=tmp / 'cycle_cache',
                                      replay_cache=tmp / 'replay_cache', parent=value, device='cpu',
                                      out_dir=tmp / 'out', allow_dirty=True)
            record = tv.run(args, log=lambda *a, **k: None)
            self.assertEqual(record['verdict'], 'CANDIDATE')
            from hybrid.h6_value_leaf import ValueLeaf
            leaf = ValueLeaf(tmp / 'out' / 'best.pt', expected_sha256=None)  # the frozen loader takes it
            self.assertEqual(leaf.meta['h7c_stage'], 'H7-c-value')
            self.assertEqual(leaf.meta['parent_h3_checkpoint_sha256'], meta['parent_h3_checkpoint_sha256'])
            with self.assertRaises(ValueError):  # the same output directory twice
                tv.run(args, log=lambda *a, **k: None)
            meta['labels_sha256'] = 'f' * 64
            value.with_name('best.json').write_text(json.dumps(meta), encoding='utf-8')
            with self.assertRaises(ValueError):  # a replay cache that is not the parent's data
                tv.run(argparse.Namespace(**{**vars(args), 'out_dir': tmp / 'out2'}), log=lambda *a, **k: None)
            with self.assertRaises(ValueError):  # a cycle cache of other games
                tv.run(argparse.Namespace(**{**vars(args), 'cycle': 2, 'out_dir': tmp / 'out3'}),
                       log=lambda *a, **k: None)

    def test_policy_cli(self):
        from scripts import h7c_train_policy as tp
        from scripts.h7c_labels import write_atomic
        from tests.test_tb_v3_net import write_models

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            policy, _ = write_models(tmp)
            write_atomic({tmp / 'policy_rows.jsonl.gz': policy_rows(6) + policy_rows(2, 'val')})
            config = tmp / 'policy.json'
            config.write_text(json.dumps({'batch_size': 16, 'epochs': 1.0, 'evaluations': 2, 'guard_drop': 1.0}),
                              encoding='utf-8')
            args = argparse.Namespace(cycle=1, config=config, policy_rows=tmp / 'policy_rows.jsonl.gz',
                                      parent=policy, teacher=policy, h3_cache=tmp, device='cpu',
                                      out_dir=tmp / 'out', allow_dirty=True, allow_stand_in=True)
            guard = guard_data(policy_rows(2, 'val'))
            with mock.patch('hybrid.h3_cache.load_cache', lambda *a, **k: (guard, {})):
                record = tp.run(args, log=lambda *a, **k: None)
                self.assertEqual(record['verdict'], 'CANDIDATE')  # a guard drop of 1.0 admits every step
                from hybrid.h4_policy import RootPolicy
                loaded = RootPolicy(tmp / 'out' / 'best.pt')  # the policy loader takes it
                self.assertEqual(loaded.meta['stage'], 'H7-c-policy')
                with self.assertRaises(ValueError):  # the frozen H3 is required as the teacher
                    tp.run(argparse.Namespace(**{**vars(args), 'out_dir': tmp / 'o2', 'allow_stand_in': False}),
                           log=lambda *a, **k: None)


if __name__ == '__main__':
    unittest.main()
