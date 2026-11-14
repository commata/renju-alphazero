"""R2-D1-D4-1 (``scripts/h7r_d1_d4.py``): the contract keeps the baseline D4 method and pins (H-dev, runtime, Rapfi,
statistics, INVALID policy, sample, budget) with the D1 pairs and factory v2; models are linked from the committed
full-scale run records; the entry gates (comparable fit, DEVELOPMENT_SIGNAL, time gate on the linked pair); the joint
reading on the three primary pairs; the replay builds every agent through factory v2."""
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import h7r_d1_d4 as dd

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'docs/h7r/r2/R2-D1-HEAD-WIDTH-1/d4/d1-d4-contract.json'


def contract():
    return json.loads((ROOT / CONTRACT).read_text())


def as_fixed(c):
    """The contract as fixed, before any approval: approval commits flip only these flags (the experiment id
    leaves them out), so tests that walk the stages from the start do not depend on the committed approvals."""
    return {**c, 'approved': {m: False for m in dd.MODES}}


def fixed_file_sha256():
    """SHA-256 of the committed contract file with its approval flags reset in the text, as it was fixed."""
    text, n = re.subn(r'("approved": \{\s*"smoke": )(?:true|false)(,\s*"full": )(?:true|false)', r'\1false\2false',
                      (ROOT / CONTRACT).read_text(encoding='utf-8'), count=1)
    assert n == 1, 'no approved block in the contract file'
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class ContractTest(unittest.TestCase):
    def load(self, c, mode=None):
        with TemporaryDirectory(dir=ROOT / 'docs') as tmp:
            path = Path(tmp) / 'd4.json'
            path.write_text(json.dumps(c))
            with mock.patch.object(dd, 'committed', return_value=True):
                return dd.load_contract(path, ROOT, mode)

    def test_the_contract_is_the_baseline_method_on_the_d1_pairs(self):
        c = contract()
        base = json.loads((ROOT / 'docs/h7r/r2/R2-D4-BASELINE-1.contract.json').read_text())
        for key in ('hdev', 'frozen_state_sha256', 'runtime_identity', 'rapfi_program', 'rapfi_identity', 'selection',
                    'budget', 'device'):
            self.assertEqual(c[key], base[key], key)
        self.assertEqual(c['c5'], base['models']['c5'])
        self.assertEqual(c['comparisons']['primary'], ['wide31001:narrow31001', 'wide31002:narrow31002',
                                                       'wide31003:narrow31003'])
        self.assertEqual(set(c['approved']), set(dd.MODES))  # approvals change over time, never the id
        self.assertEqual(dd.experiment_sha256(c), 'ee06faf80d9b0eda0c11eced10aaf352dda7d869373e5a473e502ab077d2ec74')
        _, _, eid = self.load(c)
        self.assertEqual(eid, dd.experiment_sha256(c))

    def test_the_eligible_binding_survives_approvals(self):
        """Amendment A2: the eligible manifest binds the contract as fixed (approvals false), which equals the
        committed file's SHA while nothing is approved and does not change when an approval is committed."""
        c = contract()
        self.assertEqual(dd.contract_base_sha256(c), fixed_file_sha256())
        manifest = json.loads((ROOT / c['output']['sample']).read_text())
        self.assertEqual(manifest['contract_sha256'], dd.contract_base_sha256(c))  # committed before any approval
        for approved in ({'smoke': False, 'full': False}, {'smoke': True, 'full': False}, {'smoke': True, 'full': True}):
            self.assertEqual(dd.contract_base_sha256({**c, 'approved': approved}), dd.contract_base_sha256(c))
        self.assertNotEqual(dd.contract_base_sha256({**c, 'smoke': {'games': 3, 'branches': 8}}),
                            dd.contract_base_sha256(c))
        seen = []
        with mock.patch.object(dd.base, 'inputs', side_effect=lambda cc, sha, cache: seen.append(sha) or ({}, {}, {})):
            dd.inputs({**c, 'approved': {'smoke': True, 'full': False}}, dd.contract_base_sha256(
                {**c, 'approved': {'smoke': True, 'full': False}}), 'cache')
        self.assertEqual(seen, [fixed_file_sha256()])
        self.assertTrue(c['output']['sample'].endswith('/d4/eligible-a2.json'))

    def test_prepare_then_approve_then_run_accepts_the_manifest(self):
        """The order the desktop runs: prepare (nothing approved), commit approved.smoke, run smoke. The run's
        preflight must accept the committed manifest (the first attempt refused it: amendment A2)."""
        import argparse

        class Passed(Exception):
            pass
        c = contract()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / CONTRACT
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(c, indent=2, ensure_ascii=False) + '\n')
            fake_inputs = lambda cc, sha, cache: ({'contract_sha256': sha, 'games': [1], 'rows': 1, 'warmup_plies': 0},  # noqa: E731
                                                  {}, {})
            common = [mock.patch.object(dd, 'ROOT', root), mock.patch.object(dd, 'committed', return_value=True),
                      mock.patch.object(dd.base, 'git_state', return_value={'commit': 'h', 'dirty': False}),
                      mock.patch.object(dd.base, 'inputs', side_effect=fake_inputs),
                      mock.patch.object(dd, 'load_contract', side_effect=lambda p, root=None, mode=None: (
                          json.loads(Path(p).read_text()), dd.d4.file_sha256(p),
                          dd.experiment_sha256(json.loads(Path(p).read_text()))))]
            for m in common:
                m.start()
            try:
                dd.prepare(argparse.Namespace(contract=path, cache='cache'))
                text = path.read_text()
                path.write_text(text.replace('"smoke": false', '"smoke": true', 1))
                self.assertNotEqual(dd.d4.file_sha256(path), json.loads((root / c['output']['sample']).read_text())[
                    'contract_sha256'])  # the file SHA moved with the approval
                with mock.patch.object(dd, 'entry_checks', return_value={}), \
                        mock.patch.object(dd, 'checked_models', side_effect=Passed):
                    with self.assertRaises(Passed):  # past the manifest comparison
                        dd.run(argparse.Namespace(contract=path, mode='smoke', cache='cache', device='cuda',
                                                  workers=6, model=[], rapfi_dir='r', out_dir='o'))
            finally:
                for m in common:
                    m.stop()

    def test_wrong_pins_are_refused(self):
        cases = {
            'comparisons': lambda c: c['comparisons']['primary'].reverse(),
            'statistics': lambda c: c['stats'].update(samples=10),
            'factory v2': lambda c: c['factory'].update(module='analysis.tb_v3_net'),
            'smoke size': lambda c: c['smoke'].update(games=5),
            'hdev: not the baseline': lambda c: c['hdev'].update(rows=1),
            'smoke before full': lambda c: c['approved'].update(smoke=False, full=True),
            'pinned experiment': lambda c: c['d1_full'].update(experiment_sha256='0' * 64),
            'checkout differs': lambda c: c['sources'][0].update(sha256='0' * 64),
            'keys differ': lambda c: c.update(extra=1),
        }
        for message, change in cases.items():
            with self.subTest(message):
                c = as_fixed(contract())
                change(c)
                with self.assertRaisesRegex(ValueError, message):
                    self.load(c)
        with self.assertRaisesRegex(ValueError, 'smoke stage is not approved'):
            self.load(as_fixed(contract()), 'smoke')
        c = as_fixed(contract())
        c['approved']['smoke'] = True
        self.load(c, 'smoke')
        with self.assertRaisesRegex(ValueError, 'full stage is not approved'):
            self.load(c, 'full')


def write_records(root, c, verdict='CANDIDATE', experiment=None):
    for arm in ('narrow', 'wide'):
        for s in dd.SEEDS:
            path = Path(root) / c['model_link']['run_record'].format(arm=arm, seed=s)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'experiment_sha256': experiment or c['model_link']['experiment_sha256'],
                                        'arm': arm, 'seed': s, 'verdict': verdict,
                                        'selected': {'state': {'checkpoint_sha256': f'{arm}{s}'.ljust(64, 'c'),
                                                               'metadata_sha256': f'{arm}{s}'.ljust(64, 'm')}}}))


class LinkEntryTest(unittest.TestCase):
    def test_models_are_linked_from_the_committed_run_records(self):
        c = contract()
        with TemporaryDirectory() as tmp, mock.patch.object(dd, 'committed', return_value=True):
            write_records(tmp, c)
            models = dd.linked_models(c, tmp)
            self.assertEqual(sorted(models), sorted(dd.MODELS))
            self.assertEqual(models['c5'], c['c5'])
            self.assertEqual(models['wide31002']['value_checkpoint_sha256'], 'wide31002'.ljust(64, 'c'))
            self.assertEqual(models['wide31002']['policy_checkpoint_sha256'], c['c5']['policy_checkpoint_sha256'])
        for kw in ({'verdict': 'NO_CANDIDATE'}, {'experiment': '0' * 64}):
            with self.subTest(kw), TemporaryDirectory() as tmp, mock.patch.object(dd, 'committed', return_value=True):
                write_records(tmp, c, **kw)
                with self.assertRaisesRegex(ValueError, 'not a selected run'):
                    dd.linked_models(c, tmp)
        with TemporaryDirectory() as tmp, mock.patch.object(dd, 'committed', return_value=False):
            write_records(tmp, c)
            with self.assertRaisesRegex(ValueError, 'not committed'):
                dd.linked_models(c, tmp)

    def entry(self, tmp, c, fit=None, label='DEVELOPMENT_SIGNAL', passed=True, pair=None):
        write_records(tmp, c)
        files = {c['entry']['fit']: fit or {'comparable': True, 'reading': {'category': 'WIDTH_EFFECT'}},
                 c['entry']['hdev']: {'hdev_reading': {'label': label}},
                 c['entry']['timing']: {'result': {'passed': passed},
                                        'reference': {'models': {'value_checkpoint_sha256': (pair or (
                                            'narrow31001', 'wide31001'))[0].ljust(64, 'c')}},
                                        'candidate': {'models': {'value_checkpoint_sha256': (pair or (
                                            'narrow31001', 'wide31001'))[1].ljust(64, 'c')}}}}
        for rel, body in files.items():
            (Path(tmp) / rel).parent.mkdir(parents=True, exist_ok=True)
            (Path(tmp) / rel).write_text(json.dumps(body))
        return dd.entry_checks(c, tmp)

    def test_the_entry_gates(self):
        c = contract()
        with mock.patch.object(dd, 'committed', return_value=True):
            with TemporaryDirectory() as tmp:
                out = self.entry(tmp, c)
                self.assertEqual((out['fit_category'], out['hdev_label']), ('WIDTH_EFFECT', 'DEVELOPMENT_SIGNAL'))
            cases = {
                'not comparable': dict(fit={'comparable': False, 'reading': None}),
                'is NOT_ESTABLISHED: no D4': dict(label='NOT_ESTABLISHED'),
                'is HOLD: no D4': dict(label='HOLD'),
                'time gate did not pass': dict(passed=False),
                'not measured on the linked pair': dict(pair=('narrow31002', 'wide31001')),
            }
            for message, kw in cases.items():
                with self.subTest(message), TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, message):
                    self.entry(tmp, c, **kw)


class ReadingReplayTest(unittest.TestCase):
    def test_joint_reading(self):
        def rows(l95, lind):
            return {p: {'label95': a, 'label_individual': b} for p, a, b in zip(dd.PRIMARY, l95, lind)}
        self.assertEqual(dd.joint_reading(rows(['BETTER'] * 3, ['BETTER'] * 3)), 'ALL_THREE_BETTER')
        self.assertEqual(dd.joint_reading(rows(['WORSE'] * 3, ['WORSE'] * 3)), 'ALL_THREE_WORSE')
        self.assertEqual(dd.joint_reading(rows(['BETTER', 'INCONCLUSIVE', 'BETTER'], ['BETTER', 'WORSE', 'BETTER'])),
                         'MIXED_DIRECTION')
        self.assertEqual(dd.joint_reading(rows(['BETTER', 'BETTER', 'INCONCLUSIVE'], ['BETTER'] * 2 + ['INCONCLUSIVE'])),
                         'PARTIAL_SIGNAL')
        self.assertEqual(dd.joint_reading(rows(['INCONCLUSIVE'] * 3, ['INCONCLUSIVE'] * 3)), 'NOT_ESTABLISHED')
        refs = {p: {'label95': 'BETTER', 'label_individual': 'BETTER'} for p in dd.REFERENCE}
        self.assertEqual(dd.joint_reading({**rows(['INCONCLUSIVE'] * 3, ['INCONCLUSIVE'] * 3), **refs}),
                         'NOT_ESTABLISHED')  # references never enter the reading

    def test_the_replay_builds_every_agent_through_factory_v2(self):
        import h7c_sensitivity as sn
        from analysis import tb_v3_net_v2 as v2
        built = []
        with mock.patch.object(sn, 'replay_game', side_effect=lambda game, models, device, make: (
                make({'policy': 'p', 'value': 'v'}, 11), {'game_index': game['game_index'], 'plies': []})[1]), \
                mock.patch.object(v2, 'make_side', side_effect=lambda side, seed, device: built.append(
                    (side, seed, device))):
            out = dd.replay_chunk([{'game': {'game_index': 3}, 'models': {}, 'device': 'cuda',
                                    'identity_sha256': 'i'}])
        self.assertEqual(built, [({'policy': 'p', 'value': 'v'}, 11, 'cuda')])
        self.assertEqual(out, [{'game_index': 3, 'plies': [], 'identity_sha256': 'i'}])

    def test_sensitivity_covers_every_comparison(self):
        import torch
        from hybrid import h7r_d4 as d4

        def ply(i, wide_move):
            sets = {m: {'route': 'tree', 'engine_move': [7, 7], 'visits': [[7, 7, 10, 0.1], [7, 8, 5, 0.0]]}
                    for m in dd.MODELS}
            for s in dd.SEEDS:
                sets[f'wide{s}'] = {**sets[f'wide{s}'], 'engine_move': wide_move}
            return {'ply': i, 'color': 'black' if i % 2 == 0 else 'white', 'sets': sets}
        records = [{'game_index': 0, 'plies': [ply(5, [7, 7]), ply(6, [7, 8])]},
                   {'game_index': 1, 'plies': [ply(5, [7, 7])]}]
        counts = d4.multiplicities(2, 50, 1)
        out = dd.sensitivity(records, counts)
        self.assertEqual(set(out), set(dd.COMPARISONS))
        self.assertEqual((out['wide31001:narrow31001']['different'], out['wide31001:narrow31001']['plies']), (1, 3))
        self.assertEqual(out['wide31002:c5']['first_difference_ply_median'], 6)
        self.assertIsInstance(counts, torch.Tensor)


    def test_a_widened_checkpoint_passes_the_d4_model_check(self):
        from analysis import tb_v3_net
        from tests.test_tb_v3_net_v2 import write_pair
        with TemporaryDirectory() as tmp:
            policy, value, frozen = write_pair(tmp, value_channels=4, parent=False)
            identity = tb_v3_net.model_identity(policy, value)
            c = {**contract(), 'frozen_state_sha256': frozen}
            specs = [f'{m}={policy},{value}' for m in dd.MODELS]
            with mock.patch.object(dd, 'linked_models', return_value={m: identity for m in dd.MODELS}):
                models, identities = dd.checked_models(c, specs, 'cpu')
            self.assertEqual(sorted(models), sorted(dd.MODELS))
            self.assertEqual(identities['wide31001']['frozen_state_sha256']['recomputed'], frozen)
            with mock.patch.object(dd, 'linked_models', return_value={
                    **{m: identity for m in dd.MODELS}, 'wide31001': {**identity, 'value_checkpoint_sha256': '0' * 64}}):
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    dd.checked_models(c, specs, 'cpu')


if __name__ == '__main__':
    unittest.main()
