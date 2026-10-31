"""H7-R R2: the R2 contract, game-macro vs pooled metrics, the shared source-game bootstrap (empty groups), the frozen
state recomputed from the checkpoint, the trainer's data assignment under one seed, the reproduction gate and epsilon,
and R2-0A end to end on a fixture H-dev cache (stand-in networks)."""
import argparse
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

from hybrid import h7r_r2 as r2

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'docs/h7r/r2/R2-DIAG-1.contract.json'
QUIET = lambda *a, **k: None  # noqa: E731


def committed_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding='utf-8'))


def draft_contract() -> dict:
    """An explicit draft fixture from the committed contract: draft status, no stage approved."""
    body = committed_contract()
    body['status'] = 'draft'
    body['stages'] = {'R2-0A': {'approved': False}, 'R2-0B': {'approved': False}}
    return body


class ContractTest(unittest.TestCase):
    def test_committed_contract(self):
        contract, _ = r2.load_contract(CONTRACT)
        self.assertEqual(contract['status'], 'fixed')
        self.assertTrue(contract['stages']['R2-0A']['approved'])
        self.assertFalse(contract['stages']['R2-0B']['approved'])  # no training, no epsilon
        self.assertIsNone(contract['epsilon']['value'])
        self.assertEqual(contract['reproduction']['tolerance'], {'max_abs_prediction': None, 'abs_primary': None})
        self.assertEqual(contract['comparisons'], ['c6:c5', 'c7:c5', 'c8:c5', 'mask:c5', 'orig:c5', 'mask:c8'])
        self.assertEqual(contract['bootstrap'], {'samples': 20000, 'seed': 20261030, 'unit': 'source game',
                                                 'method': 'percentile', 'level': 0.95})
        self.assertEqual(contract['metrics']['ply_bins'], [5, 20, 40, 80])
        self.assertEqual(contract['metrics']['components'], [
            {'prediction': 'd4', 'group': 'network/outcome', 'aggregate': 'macro'},
            {'prediction': 'd4', 'group': 'network/proof', 'aggregate': 'macro'},
            {'prediction': 'd4', 'group': 'network', 'aggregate': 'pooled'}])
        self.assertEqual(contract['metrics']['primary'], {'prediction': 'd4', 'group': 'network', 'aggregate': 'macro'})
        self.assertEqual(contract['epsilon']['reference'], 'c8:c5')
        manifest = json.loads((ROOT / contract['hdev']['manifest']).read_text())
        for key in ('contract_sha256', 'cache_binding_sha256', 'cache_labels_sha256', 'games_sha256', 'rows_sha256'):
            self.assertEqual(contract['hdev'][key], manifest[key])
        self.assertEqual(contract['hdev']['test_sha256'], manifest['cache_splits']['test']['sha256'])
        self.assertEqual((contract['hdev']['games'], contract['hdev']['rows']), (396, 15985))
        self.assertEqual(r2.file_sha256(ROOT / contract['hdev']['contract']), contract['hdev']['contract_sha256'])
        for name, spec in contract['models'].items():  # every identity is in its committed source record
            self.assertIn(spec['value_checkpoint_sha256'], (ROOT / spec['source']).read_text(), name)
            self.assertIn(spec['value_metadata_sha256'], (ROOT / spec['source']).read_text(), name)

    def test_rules(self):
        base = draft_contract()
        r2.check_contract(base)

        def refused(change, message):
            body = copy.deepcopy(base)
            change(body)
            with self.assertRaisesRegex(ValueError, message):
                r2.check_contract(body)

        refused(lambda b: b['stages']['R2-0A'].update(approved=True), 'a draft contract approves no stage')
        refused(lambda b: (b.update(status='fixed'), b['stages']['R2-0B'].update(approved=True)),
                'R2-0B needs the reproduction tolerances')
        refused(lambda b: b['epsilon'].update(value=0.001), 'epsilon is computed')
        refused(lambda b: b['comparisons'].append('c9:c5'), 'two contract models')
        refused(lambda b: b['comparisons'].append('c5:c5'), 'two contract models')
        refused(lambda b: b['metrics'].update(ply_bins=[5, 5, 40]), 'ply bins')
        refused(lambda b: b['metrics']['primary'].update(aggregate='pooled'), 'primary metric is a game macro')
        refused(lambda b: b['metrics']['primary'].update(group='network/nothing'), 'a known group')
        refused(lambda b: b['reproduction'].update(seed=31001), 'outside the training seeds')
        refused(lambda b: b['reproduction']['tolerance'].update(abs_primary=0), 'positive or null')
        refused(lambda b: b['models']['c5'].update(value_checkpoint_sha256='x'), 'four SHA-256')
        refused(lambda b: b['stages']['R2-0B'].update(approved=True), 'a draft contract approves no stage')
        fixed = copy.deepcopy(base)
        fixed['status'] = 'fixed'
        fixed['stages']['R2-0A']['approved'] = True
        r2.check_contract(fixed)  # R2-0A alone needs no reproduction tolerance


def sums_of(pred, target, weight, game, rows=None, n_games=None):
    rows = list(range(len(pred))) if rows is None else rows
    return r2.game_sums(pred, target, weight, game, rows, n_games or (max(game) + 1))


class MetricTest(unittest.TestCase):
    def test_a_long_game_dominates_pooled_not_macro(self):
        # game 0: 100 rows with error 0.01; game 1: 2 rows with error 1.0 (equal weights)
        pred = [0.9] * 100 + [0.0, 0.0]
        target = [1.0] * 100 + [1.0, -1.0]
        game = [0] * 100 + [1, 1]
        s = sums_of(pred, target, [0.1] * 102, game)
        m = r2.point_metrics(s)
        self.assertAlmostEqual(m['macro'], (0.01 + 1.0) / 2, places=9)
        self.assertAlmostEqual(m['pooled'], (100 * 0.01 + 2 * 1.0) / 102, places=9)
        self.assertEqual((m['games'], m['rows']), (2, 102))

    def test_weights_inside_a_game_and_strict_sign(self):
        pred, target, weight, game = [0.5, 0.0, -0.2], [1.0, -1.0, 0.0], [1.0, 0.1, 0.1], [0, 0, 0]
        m = r2.point_metrics(sums_of(pred, target, weight, game))
        self.assertAlmostEqual(m['macro'], (1.0 * 0.25 + 0.1 * 1.0 + 0.1 * 0.04) / 1.2, places=9)
        self.assertAlmostEqual(m['sign_acc_strict_macro'], 0.5)  # v = 0 with t = -1 fails; t = 0 is not counted
        self.assertAlmostEqual(m['mean_abs_value'], 0.7 / 3, places=9)

    def test_shared_bootstrap_and_empty_groups(self):
        counts = r2.multiplicities(3, 400, seed=7)
        self.assertTrue(torch.equal(counts, r2.multiplicities(3, 400, seed=7)))
        self.assertTrue(bool((counts.sum(1) == 3).all()))
        pred, target, weight, game = [0.1, 0.2, 0.3, 0.4], [1.0, 1.0, -1.0, 0.0], [1.0] * 4, [0, 1, 2, 2]
        everywhere = sums_of(pred, target, weight, game, n_games=3)
        only_game0 = sums_of(pred, target, weight, game, rows=[0], n_games=3)
        whole = r2.bootstrap(counts, everywhere, None, 'macro', 0.95)
        self.assertIsNotNone(whole['ci95'])
        self.assertEqual(whole['zero_denominator_resamples'], 0)
        sparse = r2.bootstrap(counts, only_game0, None, 'macro', 0.95)
        expected_zero = int((counts[:, 0] == 0).sum())  # replicates that never drew game 0
        self.assertGreater(expected_zero, 0)
        self.assertEqual(sparse, {'ci95': None, 'zero_denominator_resamples': expected_zero})
        # the same model against itself: every replicate difference is 0 (same multiplicities, same rows)
        same = r2.bootstrap(counts, everywhere, everywhere, 'macro', 0.95)
        self.assertEqual(same['ci95'], [0.0, 0.0])
        # one replicate with all counts 1 is the point estimate
        ones = torch.ones((1, 3), dtype=torch.float64)
        draws, _ = r2.replicate(ones, everywhere, None, 'macro')
        self.assertAlmostEqual(float(draws[0]), r2.point_metrics(everywhere)['macro'], places=12)
        draws, _ = r2.replicate(ones, everywhere, None, 'pooled')
        self.assertAlmostEqual(float(draws[0]), r2.point_metrics(everywhere)['pooled'], places=12)

    def test_groups(self):
        bins = [5, 20, 40, 80]
        groups = r2.row_groups(['network', 'network', 'exact_leaf'], [0, 1, 1], [6, 85, 30], [0.0, 1.0, -1.0],
                               ['draw', 'black', 'white'], ['outcome', 'proof_vcf', 'proof_immediate_win'], bins)
        self.assertEqual(groups['network'], [0, 1])
        self.assertEqual(groups['network/target:0'], [0])
        self.assertEqual(groups['network/ply:5-20'], [0])
        self.assertEqual(groups['network/ply:80+'], [1])
        self.assertEqual(groups['network/side:white'], [1])
        self.assertEqual(groups['network/proof:vcf'], [1])
        self.assertEqual(groups['exact_leaf/proof'], [2])
        self.assertEqual(groups['network/result:white'], [])  # an empty group stays listed
        with self.assertRaisesRegex(ValueError, 'below the first ply bin'):
            r2.row_groups(['network'], [0], [4], [0.0], ['draw'], ['outcome'], bins)
        with self.assertRaisesRegex(ValueError, 'label kind mismatch'):
            r2.row_groups(['network'], [0], [9], [1.0], ['black'], ['proof_vcf'], bins)

    def test_reversals(self):
        cell = lambda d: {'difference': d}  # noqa: E731
        comp = {'d4': {'network': {'macro': cell(-0.01), 'pooled': cell(0.002)}, 'all': {'macro': cell(-0.02)},
                       'network/outcome': {'macro': cell(-0.01)}, 'network/proof': {'macro': cell(0.03)}},
                'raw': {'network': {'macro': cell(-0.005)}}}
        out = r2.reversals(comp, {'prediction': 'd4', 'group': 'network', 'aggregate': 'macro'})
        self.assertEqual(out, ['pooled vs macro', 'outcome vs proof'])


class GateTest(unittest.TestCase):
    TOL = {'max_abs_prediction': 1e-3, 'abs_primary': 1e-4}

    def test_reproduction(self):
        open_tol = {'max_abs_prediction': None, 'abs_primary': None}
        same = {'state_sha256': ('a' * 64, 'a' * 64), 'assignment': ('x', 'x'), 'best_step': (147, 147)}
        self.assertEqual(r2.reproduction_verdict(same, open_tol)['verdict'], 'BITWISE_REPRODUCED')
        # an equal state never skips the required checks
        for change, reason in (({'assignment': None}, 'assignment missing'),
                               ({'assignment': ('x', 'y')}, 'assignment differs'),
                               ({'best_step': None}, 'best_step missing'),
                               ({'best_step': (147, None)}, 'best_step missing'),
                               ({'best_step': (147, 210)}, 'best_step differs'),
                               ({'state_sha256': ('a', 'a')}, 'state_sha256 missing')):
            out = r2.reproduction_verdict({**same, **change}, self.TOL)
            self.assertEqual(out['verdict'], 'NOT_REPRODUCED', change)
            self.assertIn(reason, out['reasons'])
        close = {'state_sha256': ('a' * 64, 'b' * 64), 'assignment': ('x', 'x'), 'best_step': (147, 147),
                 'max_abs_prediction': 5e-4, 'abs_primary': 1e-5}
        self.assertEqual(r2.reproduction_verdict(close, self.TOL)['verdict'], 'METRIC_ONLY')
        unfixed = r2.reproduction_verdict(close, open_tol)
        self.assertEqual(unfixed['verdict'], 'NOT_REPRODUCED')  # no tolerance fixed: never METRIC_ONLY
        self.assertIn('abs_primary tolerance not fixed', unfixed['reasons'])
        for change, reason in (({'assignment': ('x', 'y')}, 'assignment differs'),
                               ({'best_step': (147, 210)}, 'best_step differs'),
                               ({'max_abs_prediction': 0.01}, 'max_abs_prediction'),
                               ({'abs_primary': None}, 'abs_primary missing')):
            out = r2.reproduction_verdict({**close, **change}, self.TOL)
            self.assertEqual(out['verdict'], 'NOT_REPRODUCED')
            self.assertTrue(any(reason in r for r in out['reasons']), out)

    def test_epsilon_stops_without_reproduction(self):
        blocked = r2.epsilon(-0.001, [-0.004, 0.003], [0.30, 0.31, 0.305], 'NOT_REPRODUCED')
        self.assertEqual((blocked['status'], blocked['value']), ('BLOCKED_REPRODUCTION', None))
        out = r2.epsilon(-0.001, [-0.004, 0.003], [0.30, 0.31, 0.305], 'METRIC_ONLY')
        self.assertAlmostEqual(out['sample'], 0.004)
        self.assertAlmostEqual(out['seed'], 0.01)
        self.assertAlmostEqual(out['value'], 0.01)
        self.assertEqual(r2.epsilon(-0.001, None, [0.3, 0.31, 0.3], 'BITWISE_REPRODUCED')['status'], 'INCOMPLETE')
        self.assertEqual(r2.epsilon(-0.001, [-0.004, 0.003], [0.3, 0.3], 'BITWISE_REPRODUCED')['status'], 'INCOMPLETE')
        zero = r2.epsilon(0.0, [0.0, 0.0], [0.3, 0.3, 0.3], 'BITWISE_REPRODUCED')
        self.assertEqual((zero['status'], zero['value']), ('ZERO', None))


class AssignmentTest(unittest.TestCase):
    def test_the_fingerprint_is_what_the_trainer_does(self):
        from hybrid import h7c_train as tr
        from hybrid import h6_value_train as hv
        from tests.test_h7c_train import small_model, value_split

        cycle_train, _ = value_split(1, 4)
        replay_train, _ = value_split(2, 10)
        cycle = {'train': cycle_train, 'val': value_split(3, 2)[0]}
        replay = {'train': replay_train, 'val': value_split(4, 2)[0]}
        cfg = tr.ValueConfig(seed=11, batch_size=16, epochs=2.0, evaluations=2, learning_rate=1e-3,
                             h6_guard_ratio=100.0)
        seen_data, seen_sym = [], []
        real_mixed, real_augment = tr.mixed_epoch, hv.augment_planes

        def mixed(*a, **k):
            out = real_mixed(*a, **k)
            seen_data.append(out)
            return out

        def augment(planes, sym, src):
            seen_sym.append(sym.cpu())
            return real_augment(planes, sym, src)

        with mock.patch.object(tr, 'mixed_epoch', mixed), mock.patch.object(hv, 'augment_planes', augment):
            tr.train_value(cfg, small_model(), cycle, replay, log=QUIET)
        n_cycle, n_replay = cycle_train['value'].shape[0], replay_train['value'].shape[0]
        fp = r2.data_assignment(n_cycle, n_replay, cfg.seed, cfg.batch_size, cfg.epochs, detail=True)
        self.assertEqual(len(seen_data), len(fp['epochs']))
        for data, order in zip(seen_data, fp['detail']['orders']):  # the trainer's rows = the fingerprint's ids
            for key in ('value', 'ply', 'black', 'weight'):
                expected = torch.stack([(cycle_train if s == 0 else replay_train)[key][i] for s, i in order.tolist()])
                self.assertTrue(torch.equal(data[key], expected), key)
        self.assertTrue(torch.equal(torch.cat(seen_sym), torch.cat(fp['detail']['symmetries'])))
        self.assertEqual(len(seen_sym), fp['total_steps'])  # one symmetry draw per training step

    def test_same_seed_same_sizes_only(self):
        a = r2.data_assignment(100, 300, 31001, 32, 3.0)
        self.assertTrue(r2.compare_assignments(a, r2.data_assignment(100, 300, 31001, 32, 3.0))['identical'])
        other_seed = r2.compare_assignments(a, r2.data_assignment(100, 300, 31002, 32, 3.0))
        self.assertFalse(other_seed['identical'])
        self.assertFalse(any(other_seed['replay_pick_equal']))
        bigger = r2.compare_assignments(a, r2.data_assignment(120, 300, 31001, 32, 3.0))  # D2-like: more cycle rows
        self.assertFalse(bigger['sizes_equal'] or bigger['identical'])
        self.assertFalse(any(bigger['order_equal']))
        replay = r2.compare_assignments(a, r2.data_assignment(100, 400, 31001, 32, 3.0))  # another replay population
        self.assertFalse(replay['identical'])


def shared_trunk_models(folder: Path, names: list[str], tamper: str | None = None):
    """Value checkpoints with one trunk and different heads; ``tamper``: that model's trunk changes but its metadata
    still declares the shared frozen state."""
    from model.checkpoint import load_checkpoint, save_checkpoint
    from model.config import ModelConfig
    from hybrid.h6_value_train import frozen_state, state_sha256
    from tests.test_tb_v3_net import write_models

    _, base = write_models(folder / 'base', seed=1)
    meta = json.loads(base.with_name('best.json').read_text())
    model = load_checkpoint(base, ModelConfig(**meta['model_config']), device='cpu')
    frozen = state_sha256(frozen_state(model))
    out = {}
    for i, name in enumerate(names):
        torch.manual_seed(100 + i)
        current = copy.deepcopy(model)
        with torch.no_grad():
            for p in current.value_head.parameters():
                p.add_(0.05 * torch.randn_like(p))
            if name == tamper:
                next(current.trunk.parameters()).add_(0.01)
        d = folder / name
        d.mkdir(parents=True)
        save_checkpoint(d / 'best.pt', current)
        (d / 'best.json').write_text(json.dumps({**meta, 'frozen_state_sha256': frozen}))
        out[name] = (d / 'best.pt', d / 'best.json')
    return out, frozen


def identity(paths):
    return {'value_checkpoint_sha256': r2.file_sha256(paths[0]), 'value_metadata_sha256': r2.file_sha256(paths[1]),
            'policy_checkpoint_sha256': 'a' * 64, 'policy_metadata_sha256': 'b' * 64, 'source': 'fixture'}


class ModelIdentityTest(unittest.TestCase):
    def test_frozen_state_is_recomputed_from_the_checkpoint(self):
        with TemporaryDirectory() as tmp:
            models, frozen = shared_trunk_models(Path(tmp), ['a', 'bad'], tamper='bad')
            _, record = r2.check_model('a', identity(models['a']), *models['a'], frozen, 'cpu')
            self.assertEqual(record['frozen_state_sha256']['recomputed'], frozen)
            self.assertFalse(record['policy_loaded'])
            with self.assertRaisesRegex(ValueError, "loaded checkpoint's frozen state"):
                r2.check_model('bad', identity(models['bad']), *models['bad'], frozen, 'cpu')
            with self.assertRaisesRegex(ValueError, 'value_checkpoint_sha256'):
                r2.check_model('a', identity(models['bad']), *models['a'], frozen, 'cpu')
            with self.assertRaisesRegex(ValueError, 'metadata frozen_state_sha256'):
                r2.check_model('a', identity(models['a']), *models['a'], '0' * 64, 'cpu')
            # a metadata file without the declaration is refused (no compatibility rule in R2-DIAG-1)
            meta = json.loads(models['a'][1].read_text())
            del meta['frozen_state_sha256']
            bare = Path(tmp) / 'bare.json'
            bare.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'metadata frozen_state_sha256 None'):
                r2.check_model('a', {**identity(models['a']), 'value_metadata_sha256': r2.file_sha256(bare)},
                               models['a'][0], bare, frozen, 'cpu')


class EvaluateTest(unittest.TestCase):
    """R2-0A on a fixture H-dev cache built through the R1 holdout path."""

    def fixture(self, tmp: Path):
        from scripts import h7c_labels as lab
        from scripts import h7r_holdout as ho
        from hybrid.h6_value_cache import build_value_cache
        from hybrid import h7r_eval as ev
        from tests.test_h7c_labels import build
        from tests.test_h7r_holdout import holdout_contract, holdout_games

        path = holdout_contract(tmp / 'docs/h7r/contracts')
        contract, sha = ev.load_contract(path, tmp, check_files=False, contracts_dir=path.parent)
        games = holdout_games(path)
        rows, _, _ = build(games, evaluation={'check': lambda g: ho.check_holdout_games(g, contract, sha),
                                              'purpose': ho.PURPOSE, 'split': 'test'})
        lab.write_atomic({tmp / 'runs/rows.jsonl.gz': [lab.value_row_dict(r) for r in rows]})
        manifest = build_value_cache(tmp / 'runs/rows.jsonl.gz', tmp / 'runs/cache', progress_seconds=1e9)
        (tmp / 'runs/games.jsonl').write_text(''.join(json.dumps(g) + '\n' for g in games))
        ho.seal_cache(tmp / 'runs/cache', path, tmp / 'runs/games.jsonl', tmp / 'runs/rows.jsonl.gz')
        models, frozen = shared_trunk_models(tmp / 'models', ['c5', 'c8', 'twin'])
        models['twin'] = models['c5']  # the same files under another name: every difference is exactly 0
        body = committed_contract()
        body['hdev'] = {'contract': str(path.relative_to(tmp)), 'contract_sha256': sha,
                        'cache_binding_sha256': r2.file_sha256(tmp / 'runs/cache/holdout.json'),
                        'cache_labels_sha256': manifest['labels_sha256'],
                        'test_sha256': manifest['splits']['test']['sha256'],
                        'games_sha256': r2.file_sha256(tmp / 'runs/games.jsonl'),
                        'rows_sha256': r2.file_sha256(tmp / 'runs/rows.jsonl.gz'),
                        'games': manifest['splits']['test']['games'], 'rows': manifest['splits']['test']['states']}
        body['frozen_state_sha256'] = frozen
        body['models'] = {n: identity(p) for n, p in models.items()}
        body['comparisons'] = ['c8:c5', 'twin:c5']
        body['epsilon']['reference'] = 'c8:c5'
        body['metrics']['min_contributing_games'] = 2
        body['bootstrap']['samples'] = 300
        cpath = tmp / 'contract.json'
        cpath.write_text(json.dumps(body))
        return cpath, models

    def args(self, tmp, cpath, models, **change):
        a = argparse.Namespace(root=tmp, contract=cpath, cache=tmp / 'runs/cache', device='cpu',
                               model=[f'{n}={c},{m}' for n, (c, m) in models.items()],
                               predictions=tmp / 'runs/r2/predictions.pt', output=tmp / 'out/r2_0a.json',
                               allow_dirty=True, allow_draft=True)
        for k, v in change.items():
            setattr(a, k, v)
        return a

    def test_contract_gate_and_model_preflight(self):
        from scripts import h7r_holdout as ho
        from scripts import h7r_r2 as cli

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cpath, models = self.fixture(tmp)
            fixed = json.loads(cpath.read_text())

            def gate(body, message):
                cpath.write_text(json.dumps(body))
                with mock.patch('hybrid.h7c_nmask.committed', return_value=True), \
                        self.assertRaisesRegex(ValueError, message):
                    cli.evaluate(self.args(tmp, cpath, models, allow_draft=False), log=QUIET)

            draft = {**copy.deepcopy(fixed), 'status': 'draft',
                     'stages': {'R2-0A': {'approved': False}, 'R2-0B': {'approved': False}}}
            gate(draft, 'not a fixed, committed R2 contract')
            unapproved = copy.deepcopy(fixed)
            unapproved['stages']['R2-0A']['approved'] = False
            gate(unapproved, 'R2-0A is not approved')
            cpath.write_text(json.dumps(fixed))
            with mock.patch('hybrid.h7c_nmask.committed', return_value=True):
                out = cli.preflight(self.args(tmp, cpath, models, allow_draft=False), log=QUIET)
            self.assertEqual(set(out['models']), set(models))
            self.assertFalse((tmp / 'out/r2_0a.json').exists())  # preflight writes nothing
            # every model is checked before any route or prediction: a bad last model stops the run first
            bad, frozen = shared_trunk_models(tmp / 'bad', ['x'], tamper='x')
            body = copy.deepcopy(fixed)
            body['models']['twin'] = identity(bad['x'])
            cpath.write_text(json.dumps(body))
            with mock.patch.object(ho, 'routes') as routes, mock.patch.object(r2, 'predict') as predict, \
                    self.assertRaisesRegex(ValueError, "model twin: the loaded checkpoint's frozen state"):
                cli.evaluate(self.args(tmp, cpath, {**models, 'twin': bad['x']}), log=QUIET)
            routes.assert_not_called()
            predict.assert_not_called()
            self.assertFalse((tmp / 'out/r2_0a.json').exists())

    def test_eval(self):
        from scripts import h7r_r2 as cli

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cpath, models = self.fixture(tmp)
            with self.assertRaisesRegex(ValueError, 'not a fixed, committed R2 contract'):
                cli.evaluate(self.args(tmp, cpath, models, allow_draft=False), log=QUIET)
            with self.assertRaisesRegex(ValueError, 'exactly the contract models'):
                cli.evaluate(self.args(tmp, cpath, models, model=[f'c5={models["c5"][0]},{models["c5"][1]}']),
                             log=QUIET)
            out = cli.evaluate(self.args(tmp, cpath, models), log=QUIET)
            self.assertEqual(json.loads((tmp / 'out/r2_0a.json').read_text())['format'], r2.RESULT_FORMAT)
            self.assertEqual(sum(out['route_counts'].values()), out['hdev']['rows'])
            self.assertGreater(out['route_counts']['exact_leaf'], 0)
            self.assertEqual(out['groups']['all']['rows'], out['hdev']['rows'])
            self.assertEqual(out['groups']['network']['rows'] + out['groups']['exact_leaf']['rows'],
                             out['hdev']['rows'])
            self.assertEqual(out['groups']['network/result:black']['rows'], out['groups']['network']['rows'])
            self.assertEqual(out['groups']['network/result:draw']['rows'], 0)  # the fixture games end in a five
            self.assertNotIn('macro_ci', out['models']['c5']['metrics']['d4']['network/result:draw'])
            twin = out['summary']['twin:c5']
            self.assertEqual(twin['primary']['difference'], 0.0)
            self.assertEqual(twin['primary']['ci95'], [0.0, 0.0])
            self.assertEqual(len(twin['components']), 3)
            c8 = out['summary']['c8:c5']['primary']
            self.assertLessEqual(c8['ci95'][0], c8['difference'] + 1e-12)
            self.assertLessEqual(c8['difference'], c8['ci95'][1] + 1e-12)
            m = out['models']['c5']['metrics']['d4']['network']
            self.assertEqual(m['games'], out['hdev']['games'])
            self.assertIsNone(out['epsilon']['value'])
            self.assertEqual(out['epsilon']['status'], 'AWAITING_R2_0B')
            self.assertAlmostEqual(out['epsilon']['sample'], r2.half_width(c8['difference'], c8['ci95']))
            saved = torch.load(tmp / 'runs/r2/predictions.pt', weights_only=False)
            self.assertEqual(saved['models']['c8']['d4'].shape[0], out['hdev']['rows'])
            with self.assertRaisesRegex(ValueError, 'exists'):  # never overwrite a record
                cli.evaluate(self.args(tmp, cpath, models), log=QUIET)
            body = json.loads(cpath.read_text())
            body['hdev']['rows_sha256'] = '0' * 64
            cpath.write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, 'H-dev rows_sha256'):
                cli.evaluate(self.args(tmp, cpath, models, output=tmp / 'out/x.json', predictions=None),
                             log=QUIET)


def committed_amendment() -> dict:
    return json.loads((ROOT / 'docs/h7r/r2/R2-DIAG-1-A1.json').read_text(encoding='utf-8'))


class AmendmentTest(unittest.TestCase):
    def test_committed_amendment(self):
        contract, sha = r2.load_contract(CONTRACT)
        a, _ = r2.load_amendment(ROOT / 'docs/h7r/r2/R2-DIAG-1-A1.json', contract, sha)
        self.assertTrue(a['x1']['approved'] and a['x1']['exploratory'])
        self.assertTrue(a['r2_0b']['approved'])  # R2-0B approved (recipe, seeds, tolerances unchanged)
        self.assertEqual(a['r2_0b']['reproduction']['tolerance'], {'max_abs_prediction': 0.001, 'abs_primary': 0.0002})
        self.assertEqual(a['x1']['inputs']['result']['sha256'],
                         r2.file_sha256(ROOT / 'docs/h7r/r2/R2-DIAG-1/r2_0a.json'))
        result = json.loads((ROOT / 'docs/h7r/r2/R2-DIAG-1/r2_0a.json').read_text())
        self.assertEqual(a['x1']['inputs']['predictions']['sha256'], result['predictions']['sha256'])
        self.assertEqual(result['contract_sha256'], sha)
        for key in ('reference_record', 'config'):
            self.assertEqual(r2.file_sha256(ROOT / a['r2_0b'][key]['path']), a['r2_0b'][key]['sha256'])
        self.assertIsNone(contract['epsilon']['value'])  # the contract itself is unchanged

    def test_rules(self):
        contract, sha = r2.load_contract(CONTRACT)

        def refused(change, message, contract_sha=sha):
            body = committed_amendment()
            change(body)
            with self.assertRaisesRegex(ValueError, message):
                r2.check_amendment(body, contract, contract_sha)

        refused(lambda b: None, 'names another contract', contract_sha='0' * 64)
        refused(lambda b: b.update(status='draft'), 'only when fixed')
        refused(lambda b: b['r2_0b']['reproduction']['tolerance'].update(abs_primary=None), 'positive numbers')
        refused(lambda b: b['r2_0b'].update(training_seeds=[1, 2, 3]), 'the contract seeds')
        refused(lambda b: b['x1'].update(exploratory=False), 'exploratory')
        refused(lambda b: b['x1']['folds'].update(k=1), 'k >= 2')
        refused(lambda b: b['x1'].update(groups=['all']), 'x1 groups')


class SideOffsetTest(unittest.TestCase):
    def data(self, shift=(0.0, 0.0), seed=3):
        g = torch.Generator().manual_seed(seed)
        n_games, rows = 40, []
        game, ply, target, weight = [], [], [], []
        for gi in range(n_games):
            for p in range(5, 5 + int(torch.randint(4, 20, (1,), generator=g))):
                game.append(gi), ply.append(p)
                target.append(float(torch.randint(-1, 2, (1,), generator=g)))
                weight.append(0.1 if p % 3 else 1.0)
        noise = (torch.rand(len(game), generator=g) - 0.5) * 0.4
        pred = [t + shift[p % 2] + float(e) for t, p, e in zip(target, ply, noise)]
        return pred, target, weight, game, ply, list(range(len(game))), n_games

    def test_exact_decomposition_with_the_contract_weights(self):
        pred, target, weight, game, ply, rows, n = self.data(shift=(0.1, -0.05))
        s = r2.side_sums(pred, target, weight, game, ply, rows, n)
        d = r2.decompose(s)
        macro = r2.point_metrics(r2.game_sums(pred, target, weight, game, rows, n))['macro']
        self.assertAlmostEqual(d['loss'], macro, places=12)  # the same weights as the primary metric
        self.assertAlmostEqual(d['bias'] + d['variance'], d['loss'], places=12)
        self.assertAlmostEqual(sum(d['sides'][x]['share'] for x in r2.SIDES), 1.0, places=12)
        self.assertGreater(d['sides']['black']['mean_residual'], 0.05)
        self.assertLess(d['sides']['white']['mean_residual'], 0.0)
        ones = torch.ones((1, n), dtype=torch.float64)
        self.assertAlmostEqual(float(r2.decompose(s, ones)['loss'][0]), d['loss'], places=12)

    def test_crossfit_removes_a_side_shift_out_of_fold(self):
        pred, target, weight, game, ply, rows, n = self.data(shift=(0.3, -0.2))
        s = r2.side_sums(pred, target, weight, game, ply, rows, n)
        ids = [f'h7c:c1:8787:{i}' for i in range(n)]
        folds = r2.game_folds(ids, 5)
        self.assertEqual(folds, r2.game_folds(ids, 5))
        self.assertEqual(set(folds), set(range(5)))
        d, cf = r2.decompose(s), r2.crossfit(s, folds, 5)
        self.assertFalse(cf['undefined'])
        self.assertLess(cf['loss'], d['loss'] * 0.5)  # the shift dominates and is learnt out of fold
        self.assertGreaterEqual(cf['loss'], d['variance'] - 1e-12)  # out of fold never beats the in-sample optimum
        for f in range(5):
            self.assertAlmostEqual(cf['offsets'][f'{f}/black'], 0.3, delta=0.05)
        counts = r2.multiplicities(n, 200, seed=1)
        rep = r2.crossfit(s, folds, 5, counts)
        self.assertEqual(rep['undefined'], 0)
        self.assertEqual(rep['loss'].shape[0], 200)
        ones = torch.ones((1, n), dtype=torch.float64)
        self.assertAlmostEqual(float(r2.crossfit(s, folds, 5, ones)['loss'][0]), cf['loss'], places=12)
        # a fold whose games are its side's only weight cannot learn the offset: undefined, never a silent 0
        lone = {k: v.clone() for k, v in s.items()}
        only = [i for i, f in enumerate(folds) if f == 0]
        keep = torch.zeros(n, dtype=torch.float64)
        keep[only] = 1.0
        for key in ('u', 'ur', 'urr', 'uv', 'ut'):
            lone[key][:, 1] *= keep  # white rows only in fold 0
        self.assertTrue(r2.crossfit(lone, folds, 5)['undefined'])
        no_shift = self.data()
        s0 = r2.side_sums(*no_shift[:6], no_shift[6])
        self.assertGreaterEqual(r2.crossfit(s0, folds, 5)['loss'], r2.decompose(s0)['loss'] - 1e-3)


class X1Test(unittest.TestCase):
    def test_x1(self):
        from scripts import h7r_r2 as cli
        from scripts import h7r_r2_x1 as x1

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ev = EvaluateTest()
            cpath, models = ev.fixture(tmp)
            result = cli.evaluate(ev.args(tmp, cpath, models), log=QUIET)
            contract, csha = r2.load_contract(cpath)
            body = committed_amendment()
            body['parent'] = {'path': str(cpath), 'sha256': csha}
            body['x1']['inputs'] = {
                'result': {'path': 'out/r2_0a.json', 'sha256': r2.file_sha256(tmp / 'out/r2_0a.json')},
                'predictions': {'path': 'runs/r2/predictions.pt',
                                'sha256': r2.file_sha256(tmp / 'runs/r2/predictions.pt')}}
            apath = tmp / 'A1.json'
            apath.write_text(json.dumps(body))
            args = argparse.Namespace(root=tmp, contract=cpath, amendment=apath, cache=tmp / 'runs/cache',
                                      result=tmp / 'out/r2_0a.json', predictions=tmp / 'runs/r2/predictions.pt',
                                      output=tmp / 'out/x1.json', allow_dirty=True, allow_draft=True)
            out = x1.run(args, log=QUIET)
            self.assertTrue(out['exploratory'])
            for name in models:
                for p in ('d4', 'raw'):
                    for g in ('network', 'network/outcome', 'network/proof'):
                        cell = out['models'][name][p][g]
                        self.assertAlmostEqual(cell['descriptive']['loss'], result['models'][name]['metrics'][p][g]['macro'],
                                               places=9)
                        d = cell['descriptive']
                        self.assertAlmostEqual(d['bias'] + d['variance'], d['loss'], places=12)
            twin = out['comparisons']['twin:c5']['d4']['network']
            for key in ('loss', 'bias', 'variance', 'crossfit'):
                self.assertEqual(twin[key]['difference'], 0.0)
            for key in ('loss', 'bias', 'variance'):
                self.assertEqual(twin[key]['ci95'], [0.0, 0.0])
            # 4 fixture games: some replicates draw no game outside a fold, so the cross-fitted interval is unread
            self.assertIsNone(twin['crossfit']['ci95'])
            self.assertGreater(twin['crossfit']['zero_denominator_resamples'], 0)
            self.assertEqual(sum(out['folds']['games']), result['hdev']['games'])
            with self.assertRaisesRegex(ValueError, 'exists'):
                x1.run(args, log=QUIET)
            body['x1']['inputs']['predictions']['sha256'] = '0' * 64
            apath.write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, 'not the predictions the amendment pins'):
                x1.run(argparse.Namespace(**{**vars(args), 'output': tmp / 'out/x1b.json'}), log=QUIET)
            body['x1']['inputs']['predictions']['sha256'] = r2.file_sha256(tmp / 'runs/r2/predictions.pt')
            body['x1']['approved'] = False
            apath.write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, 'X1 is not approved'):
                x1.run(argparse.Namespace(**{**vars(args), 'output': tmp / 'out/x1c.json'}), log=QUIET)


class SeedRunTest(unittest.TestCase):
    """R2-0B on stand-ins: a reference 'c8' run, the seed run refused until approved, the reproduction run bitwise
    equal on CPU, another seed with another assignment."""

    def test_train_and_reproduce(self):
        from hybrid.h6_labels import game_rows
        from hybrid.h6_value_cache import build_value_cache
        from scripts import h7c_train_value as tv
        from scripts import h7r_r2_seed as seed_tool
        from scripts.h7c_labels import write_atomic
        from tests.test_h7c_train import cycle_value_rows, random_game

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ev = EvaluateTest()
            cpath, hmodels = ev.fixture(tmp)  # the H-dev fixture; its models are replaced below
            parent = hmodels['c5'][0]
            replay_rows = []
            for g in range(30):
                replay_rows += game_rows({'id': f'renjunet:{g}', 'moves': random_game(500 + g, 12), 'winner': 1,
                                          'real_draw': False}, source='renjunet', split='val' if g < 4 else 'train',
                                         outcome_weight=0.1, from_ply=3)
            write_atomic({tmp / 'replay_rows.jsonl.gz': [r.to_dict() for r in replay_rows]})
            manifest = build_value_cache(tmp / 'replay_rows.jsonl.gz', tmp / 'replay_cache', progress_seconds=1e9)
            meta = json.loads(parent.with_name('best.json').read_text())
            meta['labels_sha256'] = manifest['labels_sha256']
            parent.with_name('best.json').write_text(json.dumps(meta))
            from tests.test_tb_v3_net import write_models
            policy, _ = write_models(tmp / 'pol', seed=9)
            cycle_rows, cycle_cache = cycle_value_rows(tmp, 'c8', policy, parent, cycle=8)
            config = tmp / 'value.json'
            config.write_text(json.dumps({'seed': 20261025, 'batch_size': 16, 'epochs': 1.0, 'evaluations': 2,
                                          'h6_guard_ratio': 100.0}))
            ref = tv.run(argparse.Namespace(cycle=8, config=config, cycle_cache=cycle_cache, cycle_rows=cycle_rows,
                                            replay_cache=tmp / 'replay_cache', parent=parent, device='cpu',
                                            out_dir=tmp / 'ref', recipe_decision=None, allow_dirty=True,
                                            allow_stand_in=True), log=QUIET)
            self.assertEqual(ref['verdict'], 'CANDIDATE')
            body = json.loads(cpath.read_text())
            body['models'] = {'c5': identity((parent, parent.with_name('best.json'))),
                              'c8': identity((tmp / 'ref/best.pt', tmp / 'ref/best.json'))}
            body['comparisons'] = ['c8:c5']
            body['frozen_state_sha256'] = ref['frozen_state_sha256']
            cpath.write_text(json.dumps(body))
            contract, csha = r2.load_contract(cpath)
            # the committed R2-0A record of these models, pinned by the amendment before any R2-0B run
            from scripts import h7r_r2 as cli
            pair = {'c5': (parent, parent.with_name('best.json')), 'c8': (tmp / 'ref/best.pt', tmp / 'ref/best.json')}
            cli.evaluate(ev.args(tmp, cpath, pair), log=QUIET)
            amendment = committed_amendment()
            amendment['parent'] = {'path': str(cpath), 'sha256': csha}
            amendment['x1']['inputs'] = {
                'result': {'path': 'out/r2_0a.json', 'sha256': r2.file_sha256(tmp / 'out/r2_0a.json')},
                'predictions': {'path': 'runs/r2/predictions.pt',
                                'sha256': r2.file_sha256(tmp / 'runs/r2/predictions.pt')}}
            amendment['r2_0b']['reference_record'] = {'path': 'ref/record.json',
                                                      'sha256': r2.file_sha256(tmp / 'ref/record.json')}
            amendment['r2_0b']['config'] = {'path': 'value.json', 'sha256': r2.file_sha256(config)}
            amendment['r2_0b']['approved'] = False  # the approval gate is tested; the other calls pass it explicitly
            apath = tmp / 'A1.json'
            apath.write_text(json.dumps(amendment))

            def train_args(seed, out, **change):
                return argparse.Namespace(**{
                    'root': tmp, 'contract': cpath, 'amendment': apath, 'seed': seed, 'cycle_cache': cycle_cache,
                    'cycle_rows': cycle_rows, 'replay_cache': tmp / 'replay_cache', 'parent': parent,
                    'device': 'cpu', 'out_dir': tmp / out, 'allow_dirty': True, 'allow_draft': True,
                    'allow_stand_in': True, 'allow_unapproved': True, **change})

            with self.assertRaisesRegex(ValueError, 'R2-0B is not approved'):
                seed_tool.train(train_args(20261025, 'runs_x', allow_unapproved=False), log=QUIET)
            self.assertFalse((tmp / 'runs_x').exists())
            with self.assertRaisesRegex(ValueError, 'not a training seed'):
                seed_tool.train(train_args(999, 'runs_y'), log=QUIET)
            with self.assertRaisesRegex(ValueError, 'reproduction record is required first'):  # the order
                seed_tool.train(train_args(31001, 'runs_w'), log=QUIET)
            self.assertFalse((tmp / 'runs_w').exists())
            with self.assertRaisesRegex(ValueError, "not the reference run's replay"):
                seed_tool.train(train_args(20261025, 'runs_z', replay_cache=cycle_cache), log=QUIET)
            with self.assertRaises(ValueError):  # H-dev never enters training: its cache is not the c8 cycle cache
                seed_tool.train(train_args(20261025, 'runs_h', cycle_cache=tmp / 'runs/cache'), log=QUIET)
            rep = seed_tool.train(train_args(20261025, 'rep'), log=QUIET)
            self.assertEqual((rep['role'], rep['verdict'], rep['hdev_used']), ('reproduction', 'CANDIDATE', False))
            self.assertEqual(rep['best_step'], ref['best']['step'])
            self.assertIn('torch', rep['environment'])
            rargs = argparse.Namespace(root=tmp, contract=cpath, amendment=apath, cache=tmp / 'runs/cache',
                                       reference_model=f"{tmp / 'ref/best.pt'},{tmp / 'ref/best.json'}",
                                       run_dir=tmp / 'rep', device='cpu', output=tmp / 'out/repro.json',
                                       allow_dirty=True, allow_draft=True, allow_unapproved=True)
            out = seed_tool.reproduce(rargs, log=QUIET)
            self.assertEqual(out['verdict']['verdict'], 'BITWISE_REPRODUCED', out['verdict'])
            self.assertTrue(out['assignment']['compare']['identical'])
            self.assertTrue(out['assignment']['reference_source']['kind'].startswith('reconstructed'))
            self.assertEqual(out['predictions']['d4']['max_abs'], 0.0)
            self.assertEqual(out['primary']['abs_difference'], 0.0)
            self.assertIn('amendment', out['tolerance_source'])
            with self.assertRaisesRegex(ValueError, 'exists'):  # never overwritten
                seed_tool.reproduce(rargs, log=QUIET)

            # a failed reproduction record, or one whose verdict was edited, stops the seeds and epsilon
            stored = json.loads((tmp / 'out/repro.json').read_text())
            failed = {**stored, 'checks': {**stored['checks'], 'best_step': [1, 2]}}
            failed['verdict'] = r2.reproduction_verdict(failed['checks'], amendment['r2_0b']['reproduction']['tolerance'])
            (tmp / 'out/failed.json').write_text(json.dumps(failed))
            with self.assertRaisesRegex(ValueError, 'NOT_REPRODUCED'):
                seed_tool.train(train_args(31001, 'runs_f', reproduction=tmp / 'out/failed.json'), log=QUIET)
            (tmp / 'out/edited.json').write_text(json.dumps({**failed, 'verdict': stored['verdict']}))
            with self.assertRaisesRegex(ValueError, 'does not follow from its checks'):
                seed_tool.train(train_args(31001, 'runs_e', reproduction=tmp / 'out/edited.json'), log=QUIET)

            # Self-consistent checks and verdict are still not evidence: compare them to the source artifacts.
            forged = copy.deepcopy(stored)
            forged['checks']['state_sha256'] = ['a' * 64, 'b' * 64]
            forged['verdict'] = r2.reproduction_verdict(forged['checks'],
                                                       amendment['r2_0b']['reproduction']['tolerance'])
            self.assertEqual(forged['verdict']['verdict'], 'METRIC_ONLY')
            (tmp / 'out/forged.json').write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError, 'differs from source artifacts'):
                seed_tool.train(train_args(31001, 'runs_forged', reproduction=tmp / 'out/forged.json'), log=QUIET)
            self.assertFalse((tmp / 'runs_forged').exists())
            bad_detail = copy.deepcopy(stored)
            bad_detail['predictions']['raw']['max_abs'] = 0.1
            bad_detail['primary']['abs_difference'] = 0.01
            (tmp / 'out/bad_detail.json').write_text(json.dumps(bad_detail))
            with self.assertRaisesRegex(ValueError, 'differs from source artifacts'):
                seed_tool.train(train_args(31001, 'runs_bad_detail', reproduction=tmp / 'out/bad_detail.json'), log=QUIET)

            seeds = {}
            for seed in (31001, 31002, 31003):
                seeds[seed] = seed_tool.train(train_args(seed, f'seed-{seed}', reproduction=tmp / 'out/repro.json'),
                                              log=QUIET)
                self.assertEqual(seeds[seed]['role'], 'seed_spread')
                self.assertEqual(json.loads((tmp / f'seed-{seed}/config.json').read_text())['seed'], seed)
            self.assertFalse(r2.compare_assignments(rep['assignment'], seeds[31001]['assignment'])['identical'])
            with self.assertRaisesRegex(ValueError, 'not the reproduction run'):
                seed_tool.reproduce(argparse.Namespace(**{**vars(rargs), 'run_dir': tmp / 'seed-31001',
                                                          'output': tmp / 'out/r2.json'}), log=QUIET)

            def eps_args(**change):
                return argparse.Namespace(**{
                    'root': tmp, 'contract': cpath, 'amendment': apath, 'cache': tmp / 'runs/cache',
                    'result': tmp / 'out/r2_0a.json', 'reproduction': tmp / 'out/repro.json',
                    'run_dir': [tmp / f'seed-{s}' for s in (31001, 31002, 31003)],
                    'model': [f'{n}={c},{m}' for n, (c, m) in pair.items()], 'device': 'cpu',
                    'predictions': tmp / 'runs/r2/seeds.pt', 'output': tmp / 'out/eps.json',
                    'allow_dirty': True, 'allow_draft': True, 'allow_unapproved': True, **change})

            # Every seed's source files are revalidated by epsilon, not only its model and self-reported seed.
            for filename, change, message in (
                    ('config.json', lambda x: x.update(seed=31003), 'config file changed'),
                    ('record.json', lambda x: x['inputs'].update(replay_labels_sha256='0' * 64),
                     'trainer record changed'),
                    ('r2_train.json', lambda x: x.update(best_step=999), 'selected step / eligibility differs')):
                path = tmp / 'seed-31001' / filename
                original = path.read_bytes()
                changed = json.loads(original)
                change(changed)
                path.write_text(json.dumps(changed))
                try:
                    with self.assertRaisesRegex(ValueError, message):
                        seed_tool.epsilon_command(eps_args(predictions=None), log=QUIET)
                finally:
                    path.write_bytes(original)

            # Even rewriting the outer hash cannot disguise a seed/config change inside the trainer record.
            trainer_path = tmp / 'seed-31001/record.json'
            outer_path = tmp / 'seed-31001/r2_train.json'
            trainer_bytes, outer_bytes = trainer_path.read_bytes(), outer_path.read_bytes()
            changed = json.loads(trainer_bytes)
            changed['config']['seed'] = 31003
            trainer_path.write_text(json.dumps(changed))
            outer = json.loads(outer_bytes)
            outer['trainer_record_sha256'] = r2.file_sha256(trainer_path)
            outer_path.write_text(json.dumps(outer))
            try:
                with self.assertRaisesRegex(ValueError, 'trainer config / seed differs'):
                    seed_tool.epsilon_command(eps_args(predictions=None), log=QUIET)
            finally:
                trainer_path.write_bytes(trainer_bytes)
                outer_path.write_bytes(outer_bytes)

            with self.assertRaisesRegex(ValueError, 'exactly the seeds'):  # a missing seed
                seed_tool.epsilon_command(eps_args(run_dir=[tmp / 'seed-31001', tmp / 'seed-31002']), log=QUIET)
            with self.assertRaisesRegex(ValueError, 'given twice'):  # a repeated seed
                seed_tool.epsilon_command(eps_args(run_dir=[tmp / 'seed-31001', tmp / 'seed-31001',
                                                            tmp / 'seed-31003']), log=QUIET)
            with self.assertRaisesRegex(ValueError, 'not a seed-spread run'):  # the reproduction run is not a seed
                seed_tool.epsilon_command(eps_args(run_dir=[tmp / 'rep', tmp / 'seed-31002', tmp / 'seed-31003']),
                                          log=QUIET)
            with self.assertRaisesRegex(ValueError, 'NOT_REPRODUCED'):
                seed_tool.epsilon_command(eps_args(reproduction=tmp / 'out/failed.json'), log=QUIET)
            eps = seed_tool.epsilon_command(eps_args(), log=QUIET)
            e = eps['epsilon']
            ref_primary = json.loads((tmp / 'out/r2_0a.json').read_text())['summary']['c8:c5']['primary']
            values = [eps['seeds'][s]['primary'] for s in (31001, 31002, 31003)]
            self.assertEqual(e['status'], 'COMPUTED')
            self.assertAlmostEqual(e['seed'], max(values) - min(values), places=15)
            self.assertAlmostEqual(e['sample'], r2.half_width(ref_primary['difference'], ref_primary['ci95']), places=15)
            self.assertAlmostEqual(e['value'], max(e['sample'], e['seed']), places=15)
            self.assertEqual(set(eps['paired']), {f'seed{s}:{o}' for s in (31001, 31002, 31003) for o in ('c5', 'c8')})
            self.assertIn(eps['r2_0a_reread']['c8:c5']['reading'], ('BEYOND_EPSILON', 'WITHIN_EPSILON'))
            self.assertIn('amendment', eps['tolerance_source'])
            with self.assertRaisesRegex(ValueError, 'exists'):
                seed_tool.epsilon_command(eps_args(), log=QUIET)
            # a seed checkpoint changed after its run: refused (its file SHA is in its record)
            with open(tmp / 'seed-31002/best.pt', 'ab') as handle:
                handle.write(b'0')
            with self.assertRaisesRegex(ValueError, 'value_checkpoint_sha256'):
                seed_tool.epsilon_command(eps_args(output=tmp / 'out/eps2.json', predictions=None), log=QUIET)

    def test_beyond(self):
        from scripts import h7r_r2_seed as seed_tool
        self.assertTrue(seed_tool.beyond(-0.005, [-0.007, -0.003], 0.002))
        self.assertFalse(seed_tool.beyond(-0.001, [-0.002, -0.0005], 0.002))  # inside epsilon
        self.assertFalse(seed_tool.beyond(-0.005, [-0.009, 0.001], 0.002))  # interval crosses 0
        self.assertTrue(seed_tool.beyond(0.004, [0.001, 0.006], 0.002))
        self.assertFalse(seed_tool.beyond(-0.005, [-0.007, -0.003], None))

    def test_source_recheck_roundoff_does_not_relax_identities(self):
        from scripts import h7r_r2_seed as seed_tool
        self.assertTrue(seed_tool.same_source_values({'loss': 0.4, 'step': 147},
                                                     {'loss': 0.4 + 1e-9, 'step': 147}))
        self.assertFalse(seed_tool.same_source_values({'step': 147}, {'step': 148}))
        self.assertFalse(seed_tool.same_source_values({'step': 147}, {'step': 147.0}))
        self.assertFalse(seed_tool.same_source_values({'sha': 'a' * 64}, {'sha': 'b' * 64}))
        self.assertFalse(seed_tool.same_source_values({'loss': 0.4}, {'loss': 0.401}))
        self.assertFalse(seed_tool.same_source_values(float('nan'), float('nan')))

if __name__ == '__main__':
    unittest.main()
