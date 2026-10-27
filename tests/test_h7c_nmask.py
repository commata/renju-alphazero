"""H7C-NMASK-1: the masked loss and its denominator, the mask's row-by-row link, the confirmatory sample, the
decision and the training command end to end on stand-in networks and tiny data (no real c8 files)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

from hybrid import h7c_nmask as nm
from hybrid import h7c_train as tr
from tests.test_h7c_train import cycle_value_rows, random_game, small_model, value_split

QUIET = lambda *a, **k: None  # noqa: E731


class LossTest(unittest.TestCase):
    def test_masked_loss_keeps_the_original_denominator(self):
        v, t = torch.tensor([0.5, 0.0, -0.5, 1.0]), torch.tensor([1.0, 1.0, -1.0, 1.0])
        w, k = torch.tensor([1.0, 0.1, 1.0, 2.0]), torch.tensor([0.0, 1.0, 1.0, 1.0])
        expected = (0.1 * 1.0 + 1.0 * 0.25 + 0.0) / (1.0 + 0.1 + 1.0 + 2.0)  # row 0 removed, sum w unchanged
        self.assertAlmostEqual(float(tr.masked_wmse(v, t, w, k)), expected, places=6)
        from hybrid.h6_value_train import weighted_mse
        self.assertEqual(float(tr.masked_wmse(v, t, w, torch.ones(4))), float(weighted_mse(v, t, w)))
        # every row of the batch masked: 0, never NaN (the replay rows of a real batch keep it positive)
        self.assertEqual(float(tr.masked_wmse(v, t, w, torch.zeros(4))), 0.0)

    def caches(self):
        cycle_train, games = value_split(1, 4)
        replay_train, _ = value_split(2, 10)
        cycle_val, _ = value_split(3, 2)
        replay_val, _ = value_split(4, 2)
        return {'train': cycle_train, 'val': cycle_val}, {'train': replay_train, 'val': replay_val}

    def test_an_all_true_mask_is_the_unmasked_recipe(self):
        cycle, replay = self.caches()
        cfg = tr.ValueConfig(batch_size=16, epochs=1.0, evaluations=2, learning_rate=1e-3, h6_guard_ratio=100.0)
        plain = tr.train_value(cfg, small_model(), cycle, replay, log=QUIET)
        keep = {s: torch.ones(cycle[s]['value'].shape[0], dtype=torch.bool) for s in ('train', 'val')}
        masked = tr.train_value(cfg, small_model(), cycle, replay, log=QUIET, cycle_keep=keep)
        for a, b in zip(plain['record']['evaluations'], masked['record']['evaluations']):
            self.assertEqual(a['step'], b['step'])
            self.assertAlmostEqual(a['cycle_wmse'], b['cycle_wmse'], places=5)
            self.assertAlmostEqual(a['cycle_wmse'], b['cycle_wmse_original'], places=6)
            self.assertEqual(a['replay_wmse'], b['replay_wmse'])
        for name, tensor in plain['state'].items():
            self.assertTrue(torch.equal(tensor, masked['state'][name]), name)  # the same steps, bit for bit
        self.assertNotIn('mask', plain['record'])

    def test_mask_record_zero_batches_and_refusals(self):
        from hybrid.h6_value_train import frozen_state, state_sha256

        cycle, replay = self.caches()
        n = cycle['train']['value'].shape[0]
        cfg = tr.ValueConfig(batch_size=4, epochs=1.0, evaluations=2, learning_rate=1e-3, h6_guard_ratio=100.0)
        keep = {'train': torch.arange(n) % 2 == 0, 'val': torch.ones(cycle['val']['value'].shape[0], dtype=bool)}
        parent = small_model()
        out = tr.train_value(cfg, parent, cycle, replay, log=QUIET, cycle_keep=keep)
        mask = out['record']['mask']
        self.assertGreater(mask['zero_cycle_weight_batches'] + mask['batches_without_cycle_rows'], 0)
        e = mask['epochs'][0]
        self.assertAlmostEqual(e['cycle_kept_weight'], float(cycle['train']['weight'][keep['train']].double().sum()),
                               places=4)
        self.assertAlmostEqual(e['replay_weight'], e['cycle_weight'], places=4)  # 50 : 50, equal weights here
        self.assertEqual(state_sha256(frozen_state(parent)), out['record']['frozen_state_sha256'])
        model = small_model()
        model.load_state_dict(out['state'])
        self.assertEqual(state_sha256(frozen_state(model)), out['record']['frozen_state_sha256'])
        with self.assertRaisesRegex(ValueError, 'no cycle train row'):
            tr.train_value(cfg, small_model(), cycle, replay, log=QUIET,
                           cycle_keep={**keep, 'train': torch.zeros(n, dtype=torch.bool)})
        with self.assertRaisesRegex(ValueError, 'one bool'):
            tr.train_value(cfg, small_model(), cycle, replay, log=QUIET,
                           cycle_keep={**keep, 'train': torch.ones(n + 1, dtype=torch.bool)})
        with self.assertRaisesRegex(ValueError, 'one bool'):
            tr.train_value(cfg, small_model(), cycle, replay, log=QUIET,
                           cycle_keep={**keep, 'train': torch.ones(n)})
        nan = {**cycle, 'train': {**cycle['train'], 'value': torch.full_like(cycle['train']['value'], float('nan'))}}
        with self.assertRaisesRegex(ValueError, 'not finite'):
            tr.train_value(cfg, small_model(), nan, replay, log=QUIET, cycle_keep=keep)

    def test_masked_metric(self):
        cycle, _ = self.caches()
        val = {**cycle['val'], 'keep': (torch.arange(cycle['val']['value'].shape[0]) % 3 != 0).float()}
        model = small_model()
        from hybrid.h6_value_train import freeze, evaluate
        freeze(model)
        m = tr.masked_metric(model, val, 'cpu')
        full = evaluate(model, cycle['val'], 'cpu')['weighted_mse']
        self.assertLess(m['masked_wmse'], full)
        self.assertAlmostEqual(m['weight'], float(cycle['val']['weight'].sum()), places=4)
        all_kept = tr.masked_metric(model, {**cycle['val'], 'keep': torch.ones(cycle['val']['value'].shape[0])}, 'cpu')
        self.assertAlmostEqual(all_kept['masked_wmse'], full, places=5)


# --- a synthetic repository: c5 / c8 stand-ins, their records, the E2 summary and the check ---------------------

def world(tmp: Path) -> dict:
    """A repository-shaped tree with the records ``hybrid.h7c_nmask`` reads, all from stand-in models and tiny data."""
    from analysis import tb_v3_net
    from hybrid.h6_labels import game_rows
    from hybrid.h6_value_cache import build_value_cache
    from scripts import h7c_sensitivity as sn
    from scripts import h7c_train_value as tv
    from scripts.h7c_labels import game_id, write_atomic
    from tests.test_h7c_labels import game
    from tests.test_tb_v3_net import write_models

    root = tmp / 'root'
    policy, h6 = write_models(tmp / 'm')
    replay_rows = []
    for g in range(30):
        moves = random_game(500 + g, 12)
        replay_rows += game_rows({'id': f'renjunet:{g}', 'moves': moves, 'winner': 1, 'real_draw': False},
                                 source='renjunet', split='val' if g < 4 else 'train', outcome_weight=0.1, from_ply=3)
    write_atomic({tmp / 'replay_rows.jsonl.gz': [r.to_dict() for r in replay_rows]})
    replay_cache = root / nm.PATHS['replay_cache']
    anchor = build_value_cache(tmp / 'replay_rows.jsonl.gz', replay_cache, progress_seconds=1e9)
    meta = json.loads(h6.with_name('best.json').read_text())
    meta['labels_sha256'] = anchor['labels_sha256']
    h6.with_name('best.json').write_text(json.dumps(meta))
    config = root / 'configs/h7c/c1_value.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'batch_size': 16, 'epochs': 1.0, 'evaluations': 2, 'h6_guard_ratio': 100.0}))
    run = lambda **kw: tv.run(argparse.Namespace(**{'config': config, 'replay_cache': replay_cache,  # noqa: E731
                                                    'device': 'cpu', 'allow_dirty': True, 'allow_stand_in': True,
                                                    **kw}), log=QUIET)
    c1_rows, c1_cache = cycle_value_rows(tmp, 'c1', policy, h6, cycle=1)
    run(cycle=1, cycle_cache=c1_cache, cycle_rows=c1_rows, parent=h6, out_dir=root / 'runs/h7c/c5/value')
    c5 = root / nm.PATHS['parent_value']
    rows, cache = cycle_value_rows(tmp, 'c8', policy, c5, cycle=8)
    (root / nm.PATHS['value_rows']).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(rows, root / nm.PATHS['value_rows'])
    shutil.copytree(cache, root / nm.PATHS['cycle_cache'])
    record = run(cycle=8, cycle_cache=root / nm.PATHS['cycle_cache'], cycle_rows=root / nm.PATHS['value_rows'],
                 parent=c5, out_dir=root / 'runs/h7c/c8/value', allow_stand_in=False)
    c8 = root / nm.PATHS['control_value']
    models = tb_v3_net.model_identity(policy, c5)
    games = [game(i, models=models, model_hash=tb_v3_net.model_hash(models), cycle=8) for i in range(12)]
    selfplay = root / nm.PATHS['selfplay']
    selfplay.write_text(''.join(json.dumps(g) + '\n' for g in games))
    docs = lambda name: root / nm.RECORDS[name]  # noqa: E731
    for name in ('c8_verdict', 'c8_evidence', 'c8_value_record', 'c8_cache_manifest', 'b2_program', 'd4_c8',
                 'e2_summary', 'check'):
        docs(name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(root / 'runs/h7c/c8/value/record.json', docs('c8_value_record'))
    shutil.copy(root / nm.PATHS['cycle_cache'] / 'manifest.json', docs('c8_cache_manifest'))
    docs('c8_verdict').write_text(json.dumps({'format': 'h7c-arena-v1-verdict', 'cycle': 8,
                                              'promotion': {'decision': 'KEEP_CHAMPION'}, 'parent': models,
                                              'candidate': tb_v3_net.model_identity(policy, c8)}))
    sha = nm.file_sha256
    docs('c8_evidence').write_text(json.dumps({'verdict_sha256': sha(docs('c8_verdict')), 'files_sha256': {
        nm.PATHS['value_rows']: sha(root / nm.PATHS['value_rows']),
        nm.PATHS['cycle_cache'] + '/manifest.json': sha(root / nm.PATHS['cycle_cache'] / 'manifest.json'),
        nm.PATHS['selfplay']: sha(selfplay)}}))
    real = json.loads((nm.ROOT / nm.RECORDS['b2_program']).read_text())
    docs('b2_program').write_text(json.dumps({'value': {'config': 'configs/h7c/c1_value.json'},
                                              'external': real['external']}))
    e2 = sn.sample_games(games, 4)
    docs('e2_summary').write_text(json.dumps({'identity': {
        'games_sample': 4, 'selfplay_jsonl_sha256': sha(selfplay),
        'sample_sha256': hashlib.sha256(json.dumps([game_id(g) for g in e2]).encode()).hexdigest()}}))
    docs('check').write_text(json.dumps({'format': 'h7c-adjudicate-check-v1', 'verdict': 'PASS', 'failed': []}))
    docs('e2_manifest').write_text(json.dumps({'format': 'h7c-e2-evidence-v1'}))
    _, body = nm.build_mask(root / nm.PATHS['value_rows'], root / nm.PATHS['cycle_cache'])
    docs('d4_c8').write_text(json.dumps({'parts': {'d4': {'exact_leaf': {'by_kind': body['by_kind']}}}}))
    stand_in = {**tb_v3_net.model_identity(policy, h6)}
    return {'root': root, 'policy': policy, 'c5': c5, 'c8': c8, 'record': record, 'games': games, 'config': config,
            'models_patch': {k: stand_in[k] for k in ('policy_checkpoint_sha256', 'policy_metadata_sha256')}}


class WorldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        cls.w = world(Path(cls.tmp.name))
        from analysis import tb_v3
        cls.patch = mock.patch.dict(tb_v3.MODELS, cls.w['models_patch'])
        cls.patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls.tmp.cleanup()

    def cli(self, command, **kw):
        from scripts import h7c_nmask as cmd
        args = argparse.Namespace(command=command, allow_dirty=True, root=self.w['root'], **kw)
        return {'mask': cmd.mask, 'sample': cmd.sample, 'decide': cmd.decide, 'train': cmd.train}[command](
            args, log=QUIET)

    def test_flow(self):
        root = self.w['root']
        # 1. the mask: linked row by row, = the D4 classification, never overwritten
        body = self.cli('mask', value_rows=root / nm.PATHS['value_rows'], cycle_cache=root / nm.PATHS['cycle_cache'],
                        mask=root / nm.PATHS['mask'])
        self.assertGreater(sum(v['masked_rows'] for v in body['splits'].values()), 0)  # the five at the end
        keep = nm.load_mask(root / nm.PATHS['mask'], body)
        self.assertEqual({s: int((~keep[s]).sum()) for s in keep}, {s: v['masked_rows'] for s, v in body['splits'].items()})
        exact = {k: v['exact_leaf']['rows'] for k, v in body['by_kind'].items()}
        self.assertEqual(exact.get('outcome', 0), 0)
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            self.cli('mask', value_rows=root / nm.PATHS['value_rows'], cycle_cache=root / nm.PATHS['cycle_cache'],
                     mask=root / nm.PATHS['mask'])
        # 2. the sample: the E2 games excluded exactly
        sample = self.cli('sample', selfplay=root / nm.PATHS['selfplay'])
        self.assertEqual((sample['games'], sample['e2_sample']['games'], sample['selfplay_games']), (8, 4, 12))
        from scripts.h7c_labels import game_id
        from scripts import h7c_sensitivity as sn
        e2 = {game_id(g) for g in sn.sample_games(self.w['games'], 4)}
        self.assertFalse(e2 & set(sample['game_ids']))
        # 3. the decision
        decision = self.cli('decide', allow_uncommitted=True)
        self.assertEqual((decision['parent'], decision['control']['value_checkpoint_sha256']),
                         (json.loads((root / nm.RECORDS['c8_verdict']).read_text())['parent'],
                          self.w['record']['candidate']['checkpoint_sha256']))
        self.assertEqual(decision['evaluation']['confirm']['reading_level'], 1 - 0.05 / 3)
        loaded, dsha = nm.load_decision(root / nm.RECORDS['decision'], root, require_committed=False)
        with self.assertRaisesRegex(ValueError, 'committed'):
            nm.load_decision(root / nm.RECORDS['decision'], root, require_committed=True)
        # 4. the training: c8's inputs and settings, the mask, the reproduction of the c8 record
        train = lambda **kw: self.cli('train', **{  # noqa: E731
            'decision': root / nm.RECORDS['decision'], 'config': self.w['config'],
            'cycle_cache': root / nm.PATHS['cycle_cache'], 'cycle_rows': root / nm.PATHS['value_rows'],
            'replay_cache': root / nm.PATHS['replay_cache'], 'parent': self.w['c5'], 'control': self.w['c8'],
            'mask': root / nm.PATHS['mask'], 'device': 'cpu', 'smoke': False, 'allow_uncommitted': True,
            'out_dir': root / 'runs/h7c/nmask/value', **kw})
        record = train()
        self.assertTrue(all(record['reproduction'].values()), record['reproduction'])
        self.assertEqual(record['parent_val']['cycle_wmse_original'], self.w['record']['parent_val']['cycle_wmse'])
        self.assertEqual({k: record[k] for k in decision['value']['steps']}, decision['value']['steps'])
        self.assertEqual(record['frozen_state_sha256'], self.w['record']['frozen_state_sha256'])
        self.assertIn('loss', record['mask'])
        self.assertEqual(record['config'], self.w['record']['config'])  # every setting is c8's
        from scripts import h7c_nmask as cmd
        self.assertEqual(record['verdict'], 'CANDIDATE')
        best = root / 'runs/h7c/nmask/value/best.pt'
        with self.assertRaisesRegex(ValueError, 'clean training tree'):  # this test tree is dirty
            cmd.check_candidate(best, loaded, dsha)
        saved = best.with_name('record.json').read_text()
        best.with_name('record.json').write_text(json.dumps({**json.loads(saved), 'git': {'commit': 'x',
                                                                                         'dirty': False}}))
        self.assertEqual(cmd.check_candidate(best, loaded, dsha), record['candidate'])
        meta = json.loads(best.with_name('best.json').read_text())
        self.assertEqual((meta['experiment'], meta['cycle'], meta['recipe_decision_sha256']), (nm.EXPERIMENT, 8, None))
        from hybrid.h6_value_leaf import ValueLeaf
        ValueLeaf(best, expected_sha256=None)  # the frozen loader takes it
        best.with_name('record.json').write_text(saved)
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            train()
        smoke = train(smoke=True, out_dir=root / 'runs/h7c/nmask/smoke')
        self.assertTrue(smoke['smoke'])
        if smoke['verdict'] == 'CANDIDATE':
            with self.assertRaisesRegex(ValueError, 'smoke'):
                cmd.check_candidate(root / 'runs/h7c/nmask/smoke/best.pt', loaded, dsha)
        # refusals: another parent, another config, another mask
        with self.assertRaisesRegex(ValueError, 'parent value'):
            train(parent=self.w['c8'], out_dir=root / 'o1')
        other = root / 'other.json'
        other.write_text(json.dumps({'batch_size': 8}))
        with self.assertRaisesRegex(ValueError, 'config'):
            train(config=other, out_dir=root / 'o2')
        data = torch.load(root / nm.PATHS['mask'], weights_only=True)
        data['keep']['train'][0] = ~data['keep']['train'][0]
        bad = root / 'bad_mask.pt'
        torch.save(data, bad)
        with self.assertRaisesRegex(ValueError, 'mask'):
            train(mask=bad, out_dir=root / 'o3')

    def test_mask_link_refuses_another_cache(self):
        root = self.w['root']
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / 'cache'
            shutil.copytree(root / nm.PATHS['cycle_cache'], cache)
            games = json.loads((cache / 'train_games.json').read_text())
            (cache / 'train_games.json').write_text(json.dumps(list(reversed(games))))
            with self.assertRaisesRegex(ValueError, 'game order'):
                nm.build_mask(root / nm.PATHS['value_rows'], cache)
        _, body = nm.build_mask(root / nm.PATHS['value_rows'], root / nm.PATHS['cycle_cache'])
        changed = json.loads(json.dumps(body['by_kind']))
        next(iter(changed.values()))['network']['rows'] += 1
        with self.assertRaisesRegex(ValueError, 'D4'):
            nm.check_mask_against_d4(body, {'parts': {'d4': {'exact_leaf': {'by_kind': changed}}}})


class DecisionTest(unittest.TestCase):
    def test_check_must_pass_and_records_must_match(self):
        with TemporaryDirectory() as tmp:
            w = world(Path(tmp))
            from analysis import tb_v3
            with mock.patch.dict(tb_v3.MODELS, w['models_patch']):
                root = w['root']
                (root / nm.RECORDS['mask_manifest']).parent.mkdir(parents=True, exist_ok=True)
                keep, body = nm.build_mask(root / nm.PATHS['value_rows'], root / nm.PATHS['cycle_cache'])
                (root / nm.RECORDS['mask_manifest']).write_text(json.dumps(body))
                from scripts import h7c_sensitivity  # noqa: F401  (confirm_sample imports it)
                summary = json.loads((root / nm.RECORDS['e2_summary']).read_text())
                (root / nm.RECORDS['sample']).write_text(json.dumps(nm.confirm_sample(w['games'], summary)))
                decision = nm.expected_decision(root)
                (root / nm.RECORDS['decision']).write_text(json.dumps(decision))
                nm.load_decision(root / nm.RECORDS['decision'], root, require_committed=False)
                check = root / nm.RECORDS['check']
                check.write_text(json.dumps({'format': 'h7c-adjudicate-check-v1', 'verdict': 'FAIL',
                                             'failed': ['x']}))
                with self.assertRaisesRegex(ValueError, 'did not pass'):
                    nm.expected_decision(root)
                check.write_text(json.dumps({'format': 'h7c-adjudicate-check-v1', 'verdict': 'PASS', 'failed': []}))
                edited = {**decision, 'seeds': {**decision['seeds'], 'arena_official': 8739}}
                (root / nm.RECORDS['decision']).write_text(json.dumps(edited))
                with self.assertRaisesRegex(ValueError, 'not the decision'):
                    nm.load_decision(root / nm.RECORDS['decision'], root, require_committed=False)
                sample = json.loads((root / nm.RECORDS['sample']).read_text())
                sample['game_ids'] = sample['game_ids'][1:]
                (root / nm.RECORDS['sample']).write_text(json.dumps(sample))
                with self.assertRaises(ValueError):
                    nm.expected_decision(root)

    def test_seeds_are_reserved_and_unused_elsewhere(self):
        from hybrid.h6_labels import RESERVED_SEEDS
        self.assertTrue(set(nm.SEED_BLOCK) <= RESERVED_SEEDS)
        self.assertTrue(set(nm.SEEDS.values()) <= set(nm.SEED_BLOCK))
        from scripts import h7c_adjudicate as ad
        from hybrid.h7c_b2 import DRIFT_SEEDS
        self.assertFalse(set(nm.SEED_BLOCK) & (set(DRIFT_SEEDS) | {ad.ORDER_SEED} | set(range(8731, 8750))))


if __name__ == '__main__':
    unittest.main()
