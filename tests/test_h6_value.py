"""H6-2a: value cache and value-head-only trainer (hybrid.h6_value_cache, hybrid.h6_value_train)."""
import gzip
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from renju import BLACK, WHITE, Game

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None

if torch is not None:
    from hybrid import h6_value_train
    from hybrid.h6_labels import ROW_FORMAT, check_leakage, game_rows, mask_cross_split, report
    from hybrid.h6_value_cache import (KIND_PROOF, build_value_cache, load_value_cache,
                                       planes_from_value_cache)
    from hybrid.h6_value_train import (H6ValueConfig, Trainer, augment_planes, frozen_state, value_forward,
                                       weighted_mse)
    from hybrid.h3_train import d4_tables
    from model.checkpoint import load_checkpoint, save_checkpoint
    from model.config import ModelConfig
    from model.encoding import encode_game
    from model.network import PolicyValueNet
    from model.symmetry import transform_coordinate


def random_games(count=30, seed=5):
    rng = random.Random(seed)
    games = []
    for gid in range(1, count + 1):
        game, moves = Game(), []
        length = rng.randint(14, 30)
        while len(moves) < length and not game.done:
            legal = game.legal_moves()
            near = [m for m in legal if abs(m[0] - 7) <= 4 and abs(m[1] - 7) <= 4] or legal
            move = rng.choice(near)
            game.play(*move)
            moves.append(list(move))
        winner = (BLACK, WHITE, None)[gid % 3]
        split = ('train', 'train', 'train', 'val', 'test')[gid % 5]
        games.append({'id': f'g{gid}', 'moves': moves, 'winner': winner, 'real_draw': False, 'split': split})
    return games


def label_rows(games):
    """Rows as H6-1 writes them: outcome rows (0.1), a few proofs (1.0), some masked."""
    rows = []
    for g in games:
        proofs = {}
        if g['winner'] is not None:
            ply = 9  # a proof agreeing with the result: +1 when the winner is to move
            proofs[ply] = {'kind': 'vcf', 'value': 1 if (ply % 2 == 0) == (g['winner'] == BLACK) else -1, 'depth': 0}
            if g['id'] == 'g4':  # a contradicting proof: outcome rows from here are masked
                proofs[12] = {'kind': 'forced_loss', 'value': 1 if (12 % 2 == 0) == (g['winner'] == BLACK) else -1,
                              'depth': 0}
                proofs[12]['value'] = -proofs[12]['value']
        rows.extend(game_rows(g, source='test', split=g['split'], outcome_weight=0.1, proof_weight=1.0,
                              proofs=proofs, solver_class='vcf', from_ply=5))
    mask_cross_split(rows)
    check_leakage(rows)
    return rows


def write_rows(path: Path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        for r in rows:
            handle.write(json.dumps(r.to_dict(), separators=(',', ':')) + '\n')


def write_report(path: Path, rows):
    path.write_text(json.dumps({'format': 'h6-label-report-v1', 'row_format': ROW_FORMAT, 'restricted': False,
                                'summary': report(rows)}), encoding='utf-8')


def h3_model(seed=11):
    """A stand-in H3 network whose BatchNorm statistics are not the defaults (so a BN update shows)."""
    torch.manual_seed(seed)
    model = PolicyValueNet(ModelConfig())
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.uniform_(-0.5, 0.5)
            module.running_var.uniform_(0.5, 1.5)
            module.num_batches_tracked.fill_(100)
    return model.eval()


def write_h3(directory: Path, model, **meta):
    directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(directory / 'best.pt', model)
    metadata = {'stage': 'H3', 'track': 'B', 'policy_trained': True, 'value_trained': False,
                'config': {'channels': 64, 'blocks': 4}, 'step': 1, **meta}
    (directory / 'best.json').write_text(json.dumps(metadata), encoding='utf-8')
    return directory / 'best.pt'


def replay(moves):
    game = Game()
    for m in moves:
        game.play(*m)
    return game


@unittest.skipIf(torch is None, 'torch is not installed')
class ValueCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.rows = label_rows(random_games())
        write_rows(root / 'rows.jsonl.gz', cls.rows)
        write_report(root / 'report.json', cls.rows)
        cls.root, cls.cache = root, root / 'cache'
        cls.manifest = build_value_cache(root / 'rows.jsonl.gz', cls.cache, report=root / 'report.json')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_masked_rows_are_dropped(self):
        for split in ('train', 'val', 'test'):
            mine = [r for r in self.rows if r.split == split]
            kept = [r for r in mine if not r.value_mask]
            data, _ = load_value_cache(self.cache, split, verify_hash=True)
            self.assertEqual(data['value'].shape[0], len(kept))
            self.assertEqual(self.manifest['splits'][split]['masked_rows_dropped'], len(mine) - len(kept))
        self.assertTrue(any(r.value_mask for r in self.rows))
        self.assertTrue(any('outcome_conflicts_proof' in r.flags for r in self.rows))

    def test_values_weights_and_planes_match_rows(self):
        """No sign change: the cached value is the row's side-to-move value; planes = encode_game."""
        for split in ('train', 'val'):
            data, _ = load_value_cache(self.cache, split)
            kept = [r for r in self.rows if r.split == split and not r.value_mask]
            # rows are cached in file order (game by game, ply ascending)
            for i, r in enumerate(kept):
                self.assertEqual(int(data['ply'][i]), r.ply)
                self.assertEqual(float(data['value'][i]), float(r.value_target))
                self.assertAlmostEqual(float(data['weight'][i]), r.weight)
                self.assertEqual(int(data['kind'][i]) == KIND_PROOF, r.label_kind.startswith('proof_'))
            for i in range(0, len(kept), 7):
                planes, value, weight = planes_from_value_cache(data, torch.tensor([i]))
                self.assertTrue(torch.equal(planes[0], encode_game(replay(kept[i].moves))))
                self.assertEqual(float(planes[0, 3, 0, 0]), 1.0 if kept[i].side_to_move == 'black' else 0.0)
        both = {(r.side_to_move, r.value_target) for r in self.rows if not r.value_mask}
        self.assertTrue({('black', 1), ('black', -1), ('white', 1), ('white', -1)} <= both)

    def test_weights_are_ten_to_one(self):
        weights = {r.label_kind.startswith('proof_'): r.weight for r in self.rows if not r.value_mask}
        self.assertEqual(weights, {True: 1.0, False: 0.1})
        v = torch.tensor([0.0, 0.0])
        t = torch.tensor([1.0, -1.0])
        loss = weighted_mse(v, t, torch.tensor([1.0, 0.1]))
        self.assertAlmostEqual(float(loss), (1.0 * 1 + 0.1 * 1) / 1.1, places=6)
        # a proof error moves the loss ten times as much as an outcome error
        a = weighted_mse(torch.tensor([0.5, 0.0]), t, torch.tensor([1.0, 0.1]))
        b = weighted_mse(torch.tensor([0.0, -0.5]), t, torch.tensor([1.0, 0.1]))
        self.assertAlmostEqual(float(loss - a) / float(loss - b), 10.0, places=4)

    def test_report_count_must_match(self):
        bad = self.root / 'bad_report.json'
        write_report(bad, self.rows[:-40])
        with self.assertRaises(ValueError):
            build_value_cache(self.root / 'rows.jsonl.gz', self.root / 'other', report=bad)

    def test_non_contiguous_game_is_refused(self):
        rows = list(self.rows)
        rows.append(rows[0])
        write_rows(self.root / 'split_game.jsonl.gz', rows)
        with self.assertRaises(ValueError):
            build_value_cache(self.root / 'split_game.jsonl.gz', self.root / 'other2')

    def test_d4_moves_planes_not_target(self):
        _, src = d4_tables('cpu')
        data, _ = load_value_cache(self.cache, 'train')
        kept = [r for r in self.rows if r.split == 'train' and not r.value_mask]
        for i in (0, 5, 11):
            planes, value, _ = planes_from_value_cache(data, torch.tensor([i] * 8))
            moved = augment_planes(planes, torch.arange(8), src)
            for s in range(8):
                game = Game()
                for m in kept[i].moves:
                    game.play(*transform_coordinate(m[0], m[1], s))
                self.assertTrue(torch.equal(moved[s], encode_game(game)), (i, s))
            self.assertTrue(torch.equal(value, torch.full((8,), float(kept[i].value_target))))


@unittest.skipIf(torch is None, 'torch is not installed')
class TrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        root = Path(cls.tmp.name)
        rows = label_rows(random_games())
        write_rows(root / 'rows.jsonl.gz', rows)
        write_report(root / 'report.json', rows)
        build_value_cache(root / 'rows.jsonl.gz', root / 'cache', report=root / 'report.json')
        (root / 'cache' / 'test.pt').unlink()  # H6-2a must never read the test split
        cls.root = root
        cls.h3_model = h3_model()
        cls.h3 = write_h3(root / 'h3', cls.h3_model)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cfg(self, name, **kw):
        base = dict(h3_checkpoint=str(self.h3), cache_dir=str(self.root / 'cache'),
                    h6_report=str(self.root / 'report.json'), out_dir=str(self.root / name), device='cpu',
                    torch_threads=1, batch_size=16, epochs=1, max_steps=6, learning_rate=0.01, warmup_steps=1,
                    eval_every=3, checkpoint_every=3, log_every=3, require_clean_git=False)
        return H6ValueConfig(**{**base, **kw})

    def test_freeze_and_policy_identical(self):
        trainer = Trainer(self.cfg('run_freeze'), log=lambda *_: None)
        model = trainer.model
        self.assertEqual({n for n, p in model.named_parameters() if p.requires_grad},
                         {f'value_head.{n}' for n, _ in model.value_head.named_parameters()})
        self.assertEqual({id(p) for g in trainer.optimizer.param_groups for p in g['params']},
                         {id(p) for p in model.value_head.parameters()})
        self.assertFalse(model.trunk.training)
        self.assertFalse(model.policy_head.training)
        self.assertTrue(model.value_head.training)
        before_value = {k: v.clone() for k, v in model.value_head.state_dict().items()}
        trainer.train()
        h3_state = self.h3_model.state_dict()
        for name, tensor in frozen_state(model).items():  # parameters and BN buffers
            self.assertTrue(torch.equal(tensor, h3_state[name]), name)
        self.assertTrue(any(not torch.equal(before_value[k], v) for k, v in model.value_head.state_dict().items()
                            if v.dtype.is_floating_point))
        best = load_checkpoint(Path(trainer.cfg.out_dir) / 'best.pt', ModelConfig())
        x = torch.stack([encode_game(replay(m)) for m in ([], [[7, 7]], [[7, 7], [8, 8], [6, 9]])])
        with torch.no_grad():
            self.assertTrue(torch.equal(best(x)[0], self.h3_model(x)[0]))
            self.assertFalse(torch.equal(best(x)[1], self.h3_model(x)[1]))

    def test_provenance(self):
        trainer = Trainer(self.cfg('run_prov'), log=lambda *_: None)
        trainer.train()
        meta = json.loads((Path(trainer.cfg.out_dir) / 'best.json').read_text(encoding='utf-8'))
        for key, value in {'stage': 'H6-2a', 'track': 'B', 'policy_trained': True, 'value_trained': True,
                           'trunk_frozen': True, 'policy_head_frozen': True}.items():
            self.assertEqual(meta[key], value, key)
        self.assertEqual(meta['parent_h3_checkpoint_sha256'], h6_value_train.file_sha256(self.h3))
        self.assertEqual(meta['parent_h3_metadata_sha256'],
                         h6_value_train.text_sha256(self.h3.with_name('best.json')))
        manifest = json.loads((self.root / 'cache' / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['labels_sha256'], manifest['labels_sha256'])
        self.assertEqual(meta['h6_report_sha256'], h6_value_train.text_sha256(self.root / 'report.json'))
        for key in ('model_config', 'encoder_version', 'git_commit', 'git_dirty', 'config', 'best_step', 'val'):
            self.assertIn(key, meta)
        self.assertEqual(meta['val']['states'], manifest['splits']['val']['states'])
        self.assertNotIn('test', meta['cache_sha256'])

    def test_refuses_value_trained_parent_and_wrong_architecture(self):
        bad = write_h3(self.root / 'h3_bad', self.h3_model, value_trained=True)
        with self.assertRaises(ValueError):
            Trainer(self.cfg('run_bad', h3_checkpoint=str(bad)), log=lambda *_: None)
        with self.assertRaises(ValueError):
            Trainer(self.cfg('run_bad2', channels=32), log=lambda *_: None)

    def test_strict_load_refuses_other_weights(self):
        other = self.root / 'h3_other'
        other.mkdir()
        state = self.h3_model.state_dict()
        state.pop('value_head.6.bias')
        torch.save({**torch.load(self.h3, weights_only=True), 'model_state': state}, other / 'best.pt')
        (other / 'best.json').write_text(self.h3.with_name('best.json').read_text(encoding='utf-8'), encoding='utf-8')
        with self.assertRaises(ValueError):
            Trainer(self.cfg('run_strict', h3_checkpoint=str(other / 'best.pt')), log=lambda *_: None)

    def test_dirty_worktree_is_refused(self):
        with mock.patch.object(h6_value_train, '_git_provenance', return_value=('abc', True)):
            with self.assertRaises(RuntimeError):
                Trainer(self.cfg('run_dirty', require_clean_git=True), log=lambda *_: None)

    def test_report_must_match_cache(self):
        other = self.root / 'other_report.json'
        other.write_text('{"row_format": "h6-value-row-v1", "summary": {"labelled": 0}}', encoding='utf-8')
        with self.assertRaises(ValueError):
            Trainer(self.cfg('run_report', h6_report=str(other)), log=lambda *_: None)

    def test_resume_reproduces_straight_run(self):
        straight = Trainer(self.cfg('run_a'), log=lambda *_: None)
        straight.train()
        first = Trainer(self.cfg('run_b'), log=lambda *_: None)
        first.total_steps = 3  # interrupted after the step-3 snapshot (the schedule keeps 6 steps)
        first.train()
        resumed = Trainer(self.cfg('run_b'), log=lambda *_: None)
        resumed.resume(Path(resumed.cfg.out_dir) / 'snapshot.pt')
        self.assertEqual(resumed.step, 3)
        resumed.train()
        for k, v in straight.model.value_head.state_dict().items():
            self.assertTrue(torch.equal(v, resumed.model.value_head.state_dict()[k]), k)
        self.assertEqual(straight.best['step'], resumed.best['step'])

    def test_resume_refuses_changes(self):
        trainer = Trainer(self.cfg('run_c'), log=lambda *_: None)
        trainer.total_steps = 3
        trainer.train()
        snapshot = Path(trainer.cfg.out_dir) / 'snapshot.pt'
        with self.assertRaises(ValueError):
            Trainer(self.cfg('run_c', batch_size=8), log=lambda *_: None).resume(snapshot)
        with self.assertRaises(ValueError):
            Trainer(self.cfg('run_c', seed=1), log=lambda *_: None).resume(snapshot)
        other = write_h3(self.root / 'h3_other_seed', h3_model(seed=12))
        with self.assertRaises(ValueError):
            Trainer(self.cfg('run_c', h3_checkpoint=str(other)), log=lambda *_: None).resume(snapshot)

    def test_non_finite_loss_fails(self):
        trainer = Trainer(self.cfg('run_nan'), log=lambda *_: None)
        trainer.train_data['value'].fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            trainer.train()

    def test_value_forward_refuses_unfrozen_trunk(self):
        trainer = Trainer(self.cfg('run_mode'), log=lambda *_: None)
        trainer.model.train()  # what H6-2a must never do
        with self.assertRaises(RuntimeError):
            value_forward(trainer.model, torch.zeros(2, 6, 15, 15))


class ProgressTest(unittest.TestCase):
    def test_progress_lines(self):
        from hybrid.progress import Progress
        lines = []
        p = Progress('proofs', 200, 'positions', every_seconds=0, log=lambda text, flush: lines.append(text))
        p.update(50)
        p.update(150, extra='3/3 games')
        self.assertIn('[proofs] 50/200 positions  25.0%', lines[0])
        self.assertIn('100.0%', lines[1])
        self.assertIn('3/3 games', lines[1])
        q = Progress('cache', None, 'rows', every_seconds=0, log=lambda text, flush: lines.append(text))
        q.update(7)
        self.assertTrue(lines[-1].startswith('[cache] 7 rows'))


if __name__ == '__main__':
    unittest.main()
