"""H3: RenjuNet policy cache and policy-only trainer (hybrid.h3_cache, hybrid.h3_train)."""
import gzip
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

from renju import Game

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None

if torch is not None:
    from hybrid.h3_cache import (FIELD_BYTES, build_cache, tensor_bytes, load_cache, pack, planes_from_cache, unpack,
                                 verify_d4, verify_encoding)
    from hybrid.h3_train import H3Config, Trainer, augment, d4_tables, evaluate
    from model.symmetry import transform_action, transform_spatial


def random_games(count=40, seed=3):
    rng = random.Random(seed)
    games = []
    for gid in range(1, count + 1):
        game = Game()
        moves = []
        length = rng.randint(12, 30)
        while len(moves) < length and not game.done:
            legal = game.legal_moves()
            near = [m for m in legal if abs(m[0] - 7) <= 4 and abs(m[1] - 7) <= 4] or legal
            move = rng.choice(near)
            game.play(*move)
            moves.append(list(move))
        split = ('train', 'train', 'train', 'val', 'test')[gid % 5]
        masked = [p for p in range(len(moves)) if split != 'train' and p % 7 == 0]
        games.append({'id': gid, 'tournament': gid, 'rule': 1, 'split': split, 'bresult': '0.5',
                      'end': 'unknown', 'winner': '', 'moves': moves, 'masked_plies': masked})
    return games


def write_games(path: Path, games):
    payload = ('\n'.join(json.dumps(g, separators=(',', ':')) for g in games) + '\n').encode()
    path.write_bytes(gzip.compress(payload))


@unittest.skipIf(torch is None, 'torch is not installed')
class CacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.games = random_games()
        write_games(root / 'games.jsonl.gz', cls.games)
        cls.cache = root / 'cache'
        cls.manifest = build_cache(root / 'games.jsonl.gz', cls.cache)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pack_roundtrip(self):
        actions = [0, 7, 8, 112, 224]
        fields = torch.frombuffer(bytearray(pack(actions)), dtype=torch.uint8).reshape(1, FIELD_BYTES)
        self.assertEqual(unpack(fields)[0].nonzero().flatten().tolist(), actions)

    def test_tensor_bytes_without_numpy(self):
        # The desktop CPU venv has no NumPy: hashing must not call Tensor.numpy().
        self.assertEqual(tensor_bytes(torch.tensor([1, 258], dtype=torch.int16)), bytes([1, 0, 2, 1]))
        self.assertEqual(tensor_bytes(torch.zeros((0, 3), dtype=torch.uint8)), b'')

    def test_counts_exclude_opening_and_masked_plies(self):
        expected = {s: 0 for s in ('train', 'val', 'test')}
        for g in self.games:
            expected[g['split']] += sum(1 for p in range(5, len(g['moves'])) if p not in g['masked_plies'])
        self.assertEqual({s: v['states'] for s, v in self.manifest['splits'].items()}, expected)

    def test_planes_match_encode_game_and_d4_legality(self):
        for split in ('train', 'val', 'test'):
            data, _ = load_cache(self.cache, split, verify_hash=True)
            everything = range(int(data['target'].shape[0]))
            self.assertEqual(verify_encoding(data, self.games, everything), 0)
            self.assertEqual(verify_d4(data, list(everything)[:40]), 0)

    def test_stale_code_identity_is_refused(self):
        manifest_path = self.cache / 'manifest.json'
        original = manifest_path.read_text(encoding='utf-8')
        try:
            data = json.loads(original)
            data['rules_sha256'] = '0' * 64
            manifest_path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_cache(self.cache, 'train')
        finally:
            manifest_path.write_text(original, encoding='utf-8')

    def test_augment_matches_model_symmetry(self):
        data, _ = load_cache(self.cache, 'train')
        index = torch.arange(8)
        planes, legal, target = planes_from_cache(data, index)
        dest, src = d4_tables('cpu')
        symmetry = torch.arange(8)
        out_planes, out_legal, out_target = augment(planes, legal, target, symmetry, dest, src)
        for i in range(8):
            s = int(symmetry[i])
            self.assertTrue(torch.equal(out_planes[i], transform_spatial(planes[i], s)))
            self.assertTrue(torch.equal(out_legal[i], out_planes[i, 5].reshape(-1).bool()))
            self.assertEqual(int(out_target[i]), transform_action(int(target[i]), s))
            self.assertTrue(bool(out_legal[i, out_target[i]]))

    def _config(self, out, **overrides):
        base = dict(cache_dir=str(self.cache), out_dir=str(out), device='cpu', torch_threads=1, seed=5,
                    channels=8, blocks=1, batch_size=16, epochs=50, max_steps=6, learning_rate=3e-3,
                    warmup_steps=2, eval_every=3, checkpoint_every=3, log_every=3)
        return H3Config(**{**base, **overrides})

    def test_policy_only_leaves_value_head_untouched_and_resume_is_exact(self):
        with TemporaryDirectory() as tmp:
            full = Trainer(self._config(Path(tmp, 'full')), log=lambda *_: None)
            value_before = {k: v.clone() for k, v in full.model.value_head.state_dict().items()}
            full.train()
            for key, value in full.model.value_head.state_dict().items():
                self.assertTrue(torch.equal(value, value_before[key]), key)  # weights and BN statistics
            self.assertTrue(Path(tmp, 'full', 'best.pt').exists())
            meta = json.loads(Path(tmp, 'full', 'best.json').read_text(encoding='utf-8'))
            self.assertEqual((meta['policy_trained'], meta['value_trained']), (True, False))

            half = Trainer(self._config(Path(tmp, 'half'), max_steps=3), log=lambda *_: None)
            half.train()
            changed = Trainer(self._config(Path(tmp, 'half')), log=lambda *_: None)
            with self.assertRaises(ValueError):  # max_steps sets the LR schedule: training-critical
                changed.resume(Path(tmp, 'half', 'snapshot.pt'))
        with TemporaryDirectory() as tmp:
            a = Trainer(self._config(Path(tmp, 'a')), log=lambda *_: None)
            a.train()
            b = Trainer(self._config(Path(tmp, 'b'), max_steps=6), log=lambda *_: None)
            b.total_steps = 3  # stop early, as an interruption would
            b.train()
            c = Trainer(self._config(Path(tmp, 'b')), log=lambda *_: None)
            c.resume(Path(tmp, 'b', 'snapshot.pt'))
            c.train()
            for key, value in a.model.state_dict().items():
                self.assertTrue(torch.allclose(value.float(), c.model.state_dict()[key].float(), atol=1e-6), key)

    def test_overfit_small_set(self):
        with TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp, 'o'), channels=16, blocks=2, max_steps=200, augment_d4=False,
                               learning_rate=5e-3, warmup_steps=5, eval_every=1000, checkpoint_every=1000,
                               log_every=1000)
            trainer = Trainer(cfg, log=lambda *_: None)
            before = evaluate(trainer.model, trainer.train_data, cfg, 'cpu')
            trainer.train()
            after = evaluate(trainer.model, trainer.train_data, cfg, 'cpu')
            self.assertLess(after['ce'], before['ce'] * 0.5)
            self.assertGreater(after['top1'], 0.5)


if __name__ == '__main__':
    unittest.main()
