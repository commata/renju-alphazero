"""H7-c diagnosis (scripts/h7c_diagnose.py): measures, input checks, and one end-to-end run on stand-in models."""
import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from renju import Game
from rt30 import fast5
from scripts import h7c_arena as ar
from scripts import h7c_diagnose as dg


class MeasureTest(unittest.TestCase):
    def test_distributions(self):
        a, b = (7, 7), (7, 8)
        self.assertEqual(dg.restricted({a: 0.0, b: 0.0}, [a, b]), {a: 0.5, b: 0.5})  # the PUCT fallback
        r = dg.restricted({a: 0.3, b: 0.1, (0, 0): 0.6}, [a, b])
        self.assertAlmostEqual(r[a], 0.75)
        self.assertAlmostEqual(r[b], 0.25)
        self.assertAlmostEqual(dg.entropy({a: 0.5, b: 0.5}), math.log(2))
        p, q = {a: 0.9, b: 0.1}, {a: 0.5, b: 0.5}
        self.assertGreater(dg.kl(p, q), 0)
        self.assertAlmostEqual(dg.js(p, q), dg.js(q, p))
        self.assertEqual(dg.kl(p, p), 0)

    def test_threat_classes(self):
        game = Game()
        for move in [(7, 7), (0, 0), (7, 8), (0, 2), (7, 9), (0, 4)]:  # black three on row 7, black to move
            game.play(*move)
        classes = dg.threat_classes(game, game.legal_moves())
        self.assertIn((7, 10), classes['four'])
        self.assertIn((7, 10), classes['open_four'])
        self.assertFalse(classes['five'])
        self.assertTrue(classes['four'] <= classes['window3'])

    def test_d5_raw_visits_and_target_support(self):
        rows = [{'mask': False, 'moves': [[7, 7]], 'kind': 'visits', 'target': [[1, 1, 0.5], [2, 2, 0.5]],
                 'raw_visits': [[1, 1, 20, 0.1], [2, 2, 20, 0.0], [3, 3, 10, -0.9], [4, 4, 0, 0.0]]},
                {'mask': False, 'moves': [], 'kind': 'proof_one_hot', 'target': [[1, 1, 1.0]], 'raw_visits': []},
                {'mask': True, 'moves': [], 'kind': 'visits', 'target': [[1, 1, 1.0]], 'raw_visits': [[1, 1, 5, 0.0]]}]
        out = dg.d5(rows)
        white = out['white']
        self.assertEqual((white['visits_rows'], white['raw_visits']['children'], white['target']['children']), (1, 3, 2))
        self.assertEqual((white['filtered_children'], white['rows_with_filtered_children']), (1, 1))  # a proven loss
        self.assertEqual((white['raw_visits']['top_share'], white['target']['top_share']), (0.4, 0.5))
        self.assertAlmostEqual(white['target']['support'], 2.0, places=4)
        self.assertEqual((out['black']['visits_rows'], out['black']['proof_one_hot_rows']), (0, 1))

    def test_d2_strata(self):
        a, b = (7, 8), (8, 8)
        rows = [{'mask': False, 'split': 'val', 'kind': 'visits', 'moves': [[7, 7]], 'target': [[7, 8, 0.75], [8, 8, 0.25]]},
                {'mask': False, 'split': 'train', 'kind': 'visits', 'moves': [[7, 7]], 'target': [[7, 8, 1.0]]}]
        parent = lambda game: {a: 0.4, b: 0.6}  # noqa: E731
        candidate = lambda game: {a: 0.9, b: 0.1}  # noqa: E731
        out = dg.d2(rows, parent, candidate)
        cell = out['visits/white']
        self.assertEqual(cell['n'], 1)  # train rows are not validation
        self.assertAlmostEqual(cell['parent']['ce'], -(0.75 * math.log(0.4) + 0.25 * math.log(0.6)), places=4)
        self.assertEqual((cell['parent']['top1'], cell['candidate']['top1']), (0, 1))  # the mode is (7, 8)
        self.assertLess(cell['delta_ce'], 0)

    def test_top1_counts_tied_maxima_as_the_trainer_does(self):
        a, b, c = (7, 8), (8, 8), (6, 6)
        target = {a: 0.5, b: 0.5}
        self.assertTrue(dg.top1_correct({a: 0.1, b: 0.8, c: 0.1}, target))   # b is a maximum too
        self.assertTrue(dg.top1_correct({a: 0.8, b: 0.1, c: 0.1}, target))
        self.assertFalse(dg.top1_correct({a: 0.1, b: 0.1, c: 0.8}, target))
        rows = [{'mask': False, 'split': 'val', 'kind': 'visits', 'moves': [[7, 7]],
                 'target': [[7, 8, 0.5], [8, 8, 0.5]]}]
        out = dg.d2(rows, lambda g: {b: 0.9, a: 0.1}, lambda g: {a: 0.9, b: 0.1})
        self.assertEqual((out['visits/white']['parent']['top1'], out['visits/white']['candidate']['top1']), (1, 1))


WHITE_ROWS, WHITE_COLS = (0, 1, 2, 12, 13, 14), range(0, 9)


def varied_game(index: int, models: dict, variant: int):
    """test_h7c_labels' game ``index`` (black's five on row 7) with white's four far stones placed by ``variant``:
    distinct positions per variant, so the union masking keeps validation rows."""
    import copy

    from analysis import tb_v3_net
    from hybrid import h7c_targets as targets
    from tests.test_h7c_labels import MOVES, game

    row, col = WHITE_ROWS[variant % 6], WHITE_COLS[(variant // 6) % 9]
    moves = copy.deepcopy(MOVES)
    for k, ply in enumerate((1, 3, 5, 7)):
        moves[ply] = [row, col + 2 * k]
    g = game(index, models=models, model_hash=tb_v3_net.model_hash(models))
    g['moves'], g['opening'] = moves, moves[:3]
    for p in g['plies']:
        move = moves[p['ply']]
        p['engine_move'] = p['played'] = move
        if p['visits']:
            p['visits'] = [[*move, 40, 0.1], [3, 3, 10, 0.0]]
        p['target'] = targets.policy_target(p['route'], move, p['visits'], [])
    return g


# game ids h7c:c1:8801:<i> that the label split puts in val (salt h7c-v1, 5%): a few of them with train games
GAME_INDICES = (22, 47, 50, 53, 72) + tuple(range(0, 22)) + tuple(range(23, 40))


def training_inputs(tmp: Path, parent_policy, parent_value):
    """The parent's self-play rows, the cycle cache, a replay-anchor cache and a trained value candidate."""
    import argparse as ap

    from analysis import tb_v3_net
    from hybrid.h6_labels import game_rows
    from hybrid.h6_value_cache import build_value_cache
    from scripts import h7c_train_value as tv
    from scripts.h7c_labels import value_row_dict, write_atomic
    from tests.test_h7c_labels import build
    from tests.test_h7c_train import random_game

    replay_rows = []
    for g in range(30):
        replay_rows += game_rows({'id': f'renjunet:{g}', 'moves': random_game(500 + g, 12), 'winner': 1,
                                  'real_draw': False}, source='renjunet', split='val' if g < 4 else 'train',
                                 outcome_weight=0.1, from_ply=3)
    write_atomic({tmp / 'replay_rows.jsonl.gz': [r.to_dict() for r in replay_rows]})
    replay = build_value_cache(tmp / 'replay_rows.jsonl.gz', tmp / 'replay_cache', progress_seconds=1e9)
    meta = json.loads(parent_value.with_name('best.json').read_text(encoding='utf-8'))
    parent_value.with_name('best.json').write_text(json.dumps({**meta, 'labels_sha256': replay['labels_sha256']}))
    models = tb_v3_net.model_identity(parent_policy, parent_value)
    rows, policy_rows, _ = build([varied_game(i, models, v) for v, i in enumerate(GAME_INDICES)])
    write_atomic({tmp / 'value_rows.jsonl.gz': [value_row_dict(r) for r in rows],
                  tmp / 'policy_rows.jsonl.gz': policy_rows})
    build_value_cache(tmp / 'value_rows.jsonl.gz', tmp / 'value_cache', progress_seconds=1e9)
    config = tmp / 'value.json'
    config.write_text(json.dumps({'batch_size': 16, 'epochs': 1.0, 'evaluations': 2, 'h6_guard_ratio': 100.0}))
    record = tv.run(ap.Namespace(cycle=1, config=config, cycle_cache=tmp / 'value_cache',
                                 cycle_rows=tmp / 'value_rows.jsonl.gz', replay_cache=tmp / 'replay_cache',
                                 parent=parent_value, device='cpu', out_dir=tmp / 'value', allow_dirty=True,
                                 allow_stand_in=True), log=lambda *a, **k: None)
    return record, tmp / 'value' / 'best.pt'

@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class EndToEndTest(unittest.TestCase):
    def tearDown(self):
        from hybrid.h6_fast import install
        fast5.uninstall()
        install('off')
        ar._MODELS.clear()

    def test_run_and_refusals(self):
        from analysis import tb_v3_net
        from tests.test_h7c_arena import play_args, write_candidate
        from tests.test_tb_v3_net import write_models

        quiet = lambda *a, **k: None  # noqa: E731
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            parent_policy, parent_value = write_models(tmp / 'p', seed=2)
            record, value = training_inputs(tmp, parent_policy, parent_value)  # sets the parent's anchor first
            self.assertEqual(record['verdict'], 'CANDIDATE')
            self.assertIsNotNone(record['best']['cycle_wmse'])  # varied games: the cycle val survives the masking
            (tmp / 'value_record.json').write_text(json.dumps(record))
            policy, _ = write_candidate(tmp / 'c', parent_policy, parent_value)
            out = ar.play(play_args(tmp, policy, value, parent_policy, parent_value, arm='p_only', max_plies=7),
                          log=quiet)
            out['games'][0]['result'] = 'loss'  # a cut game stands in for a loss: D3 replays its moves
            p_only = tmp / 'p_only.json'
            p_only.write_text(json.dumps(out), encoding='utf-8')
            parent = tb_v3_net.model_identity(parent_policy, parent_value)
            sha = {name: ar._file_sha(tmp / name) for name in ('value_rows.jsonl.gz', 'policy_rows.jsonl.gz')}
            verdict, evidence = tmp / 'verdict.json', tmp / 'evidence.json'
            verdict.write_text(json.dumps({'format': 'h7c-arena-v1-verdict', 'cycle': 1,
                                           'candidate': tb_v3_net.model_identity(policy, value), 'parent': parent,
                                           'arm_sha256': {'p_only': ar._file_sha(p_only)}}))
            evidence.write_text(json.dumps({
                'files_sha256': sha, 'arena': {'arm_sha256': {'p_only': ar._file_sha(p_only)}},
                'training': {'cycle_value_labels_sha256': record['inputs']['cycle_labels_sha256']}}))
            args = argparse.Namespace(cycle=1, verdict=verdict, evidence=evidence, p_only=p_only,
                                      candidate_policy=policy, candidate_value=value, parent_policy=parent_policy,
                                      parent_value=parent_value, policy_rows=tmp / 'policy_rows.jsonl.gz',
                                      value_rows=tmp / 'value_rows.jsonl.gz', value_record=tmp / 'value_record.json',
                                      cycle_cache=tmp / 'value_cache', replay_cache=tmp / 'replay_cache', parts=None,
                                      device='cpu', workers=1, output=tmp / 'diagnosis.json', allow_dirty=True)
            result = dg.run(args, log=quiet)
            parts = result['parts']
            # D1 / D3: one replay; D1 on the engine's own root children, checked against its prior entropy
            tree_moves = sum(m['route'] == 'tree' for g in out['games'] for m in g['tested_moves'])
            self.assertEqual(parts['d1']['positions'], tree_moves)
            self.assertEqual(parts['d1']['replay']['not_reproduced'], [])  # the record replays move for move
            if tree_moves:
                self.assertLessEqual(parts['d1']['engine_prior_entropy_max_diff'], 2e-4)
                self.assertEqual(parts['d1']['summary']['root']['all']['n'], tree_moves)
            self.assertEqual((parts['d3']['losses'], parts['d3']['status']['NOT_REPRODUCED']), (1, 0))
            # D4: the training metric reproduces the record; the engine metric is the same position set
            d4 = parts['d4']
            reproduced = d4['training_metric']['reproduction']
            self.assertAlmostEqual(reproduced['reproduced']['candidate']['cycle'], record['best']['cycle_wmse'], 6)
            self.assertEqual(reproduced['dropped'], record['union_masking'])
            n_train = sum(c['n'] for c in d4['training_metric']['strata'].values())
            self.assertEqual(n_train, d4['positions'])
            self.assertEqual({k: c['n'] for k, c in d4['engine_metric']['strata'].items()},
                             {k: c['n'] for k, c in d4['training_metric']['strata'].items()})
            self.assertTrue(0 < d4['exact_leaf']['exact_leaf_rows_share'] < 1)  # five-in-one positions: exact
            self.assertEqual(d4['probes']['parent']['device'], 'cpu')
            self.assertIn('black', parts['d5'])
            self.assertEqual(result['inputs']['p_only_signature'], out['signature'])
            # fail closed: inputs the records do not name, a device the games did not run on, another record
            other = tmp / 'other.json'
            other.write_text(json.dumps({**out, 'games': out['games'][:1]}))
            bad_record = tmp / 'bad_record.json'
            bad_record.write_text(json.dumps({**record, 'union_masking': {'cycle_val_dropped': -1}}))
            for change in ({'p_only': other}, {'candidate_value': parent_value},
                           {'policy_rows': tmp / 'value_rows.jsonl.gz'}, {'cycle': 2}, {'device': 'cuda'},
                           {'replay_cache': tmp / 'value_cache'}):
                with self.assertRaises(ValueError, msg=str(change)):
                    dg.check_inputs(argparse.Namespace(**{**vars(args), **change}))
            with self.assertRaises(ValueError):  # the union masking does not reproduce the record
                dg.training_metric(argparse.Namespace(**{**vars(args), 'value_record': bad_record}),
                                   json.loads(bad_record.read_text()), log=quiet)
            tampered = {**record, 'best': {**record['best'], 'cycle_wmse': record['best']['cycle_wmse'] + 0.01}}
            with self.assertRaises(ValueError):  # a recorded WMSE that the models do not give
                dg.training_metric(args, tampered, log=quiet)

if __name__ == '__main__':
    unittest.main()
