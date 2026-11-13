"""Seed reservation v2 (hybrid/h7r_seeds.py), v2 evaluation contracts (hybrid/h7r_eval_v2.py), the optional class
exclusion of hybrid/h7r_openings.build_suite and the R2-S400-TRAIN development suites (scripts/h7r_dev_suites.py):
the registry collides with nothing; v1 contracts, checkers and pinned sources are untouched; v2 contracts are checked
on the v2 path only; v2 Arena games never enter the training label step; suites are class-distinct inside and outside."""
import copy
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from hybrid import h7r_eval as ev
from hybrid import h7r_eval_v2 as ev2
from hybrid import h7r_openings as op
from hybrid import h7r_seeds as seeds
from hybrid.h6_labels import RESERVED_SEEDS

ROOT = Path(__file__).resolve().parents[1]
V1_BLOCK = range(8781, 8800)


class RegistryTest(unittest.TestCase):
    def test_no_collision(self):
        self.assertEqual(seeds.conflicts(), {k: [] for k in seeds.conflicts()})
        import h7c_selfplay as sp
        self.assertEqual((sp.RUN_SEEDS, sp.MEASURE_SEED), (seeds.TRAINING_RUN_SEEDS, seeds.MEASURE_SEED))
        from tests import h7r_helpers
        fixtures = {v for k, v in vars(h7r_helpers).items() if k.startswith('FIXTURE_') and k.endswith('_SEED')}
        self.assertFalse(fixtures & set(seeds.ARENA_SEEDS) or fixtures & set(seeds.SUITE_SEEDS))

    def test_no_committed_record_uses_a_v2_seed(self):
        used = set()
        for path in ROOT.joinpath('docs').rglob('*.json'):
            if path.name.startswith('O1-dev-'):  # the suites generated with the v2 suite seeds
                continue
            used |= {int(x) for x in re.findall(r'"(?:seed|run_seed|base_seed)": (\d+)', path.read_text(errors='ignore'))}
        self.assertFalse(used & set(seeds.ARENA_SEEDS), sorted(used & set(seeds.ARENA_SEEDS)))
        self.assertFalse(used & set(seeds.SUITE_SEEDS))

    def test_v2_seeds_are_never_training_run_seeds(self):
        import h7c_selfplay as sp
        for seed in (seeds.ARENA_SEEDS.start, seeds.ARENA_SEEDS.stop - 1, *seeds.SUITE_SEEDS):
            with self.subTest(seed):
                with self.assertRaisesRegex(ValueError, 'training runs use'):
                    sp.check_seeds(seed, 2, [0])
                with self.assertRaisesRegex(ValueError, 'holdout run uses a reserved H7-R seed'):
                    sp.check_seeds(seed, 2, [0], holdout=True)


class V1UntouchedTest(unittest.TestCase):
    def test_v1_checker_block_and_contracts(self):
        self.assertEqual(ev.SEED_BLOCK, V1_BLOCK)
        self.assertTrue(set(V1_BLOCK) <= RESERVED_SEEDS)
        for path in sorted(ROOT.joinpath('docs/h7r/contracts').glob('*.json')):
            with self.subTest(path.name):
                ev.load_contract(path, ROOT, check_files=False)

    def test_contracts_pinning_the_seed_sources_still_verify(self):
        import h7r_p1 as p1
        import h7r_p1a1 as a1
        import h7r_s400_pre as pre
        p1.load_contract(None, ROOT)
        a1.load_contract(None, ROOT)
        pre.load_contract(None, ROOT)


def v2_contract(seed=9101, **changes):
    suite = 'docs/h7r/suites/O1.json'
    model = {'policy': 'runs/h3_policy_64x4/best.pt', 'value': 'runs/h7c/c5/value/best.pt',
             'identity': {k: 'a' * 64 for k in ev.MODEL_KEYS}}
    arm = lambda name, tested: {'name': name, 'tested': {'model': tested, 'budget': 1},  # noqa: E731
                                'opponent': {'model': 'champion', 'budget': 1}, 'pairs': 300, 'seed': seed,
                                'pairing': 'P', 'suite': {'path': suite, 'sha256': ev.file_sha256(ROOT / suite)}}
    c = {'format': ev2.CONTRACT_FORMAT, 'seed_policy': seeds.POLICY, 'contract_id': 'V2-TEST', 'purpose': 'pilot',
         'status': 'fixed', 'models': {'champion': model, 'X': model, 'T': model},
         'arms': [arm('X_vs_champion', 'X'), arm('T_vs_champion', 'T')],
         'stats': {'alpha_one_sided': 0.025, 'bootstrap': {'samples': 20000, 'seed': 1, 'method': 'percentile',
                                                           'unit': 'opening pair'}}}
    c.update(changes)
    return c


class V2ContractTest(unittest.TestCase):
    def test_the_v2_path_only(self):
        ev2.check_contract(v2_contract(), 'c', ROOT, check_files=False)
        cases = {
            'not a v2 Arena seed': v2_contract(seed=8788),
            'not a v2 Arena seed ': v2_contract(seed=9201),
            'seed policy': v2_contract(seed_policy='h7r-seeds-v1'),
            'not an h7r-eval-contract-v2 contract': {**v2_contract(), 'format': ev.CONTRACT_FORMAT},
            'purpose / status': v2_contract(purpose='confirm'),
        }
        for message, bad in cases.items():
            with self.subTest(message):
                with self.assertRaisesRegex(ValueError, message.strip()):
                    ev2.check_contract(bad, 'c', ROOT, check_files=False)
        unpaired = v2_contract()
        for a in unpaired['arms']:
            a.pop('pairing')
        with self.assertRaisesRegex(ValueError, 'a seed used twice outside a pairing'):
            ev2.check_contract(unpaired, 'c', ROOT, check_files=False)
        with self.assertRaisesRegex(ValueError, 'not an h7r-eval-contract-v1 contract'):
            ev.check_contract(v2_contract(), 'c', ROOT, check_files=False)  # the v1 checker refuses v2
        v1 = json.loads((ROOT / 'docs/h7r/contracts/R2-P1-A1-ARENA-1.json').read_text())
        with self.assertRaisesRegex(ValueError, 'not an h7r-eval-contract-v2 contract'):
            ev2.check_contract(v1, 'c', ROOT, check_files=False)
        self.assertEqual(ev.SEED_BLOCK, V1_BLOCK)  # nothing of the v1 checker was rebound

    def test_seeds_and_ids_are_unique_across_v1_and_v2(self):
        import shutil
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in (*(p.relative_to(ROOT).as_posix() for p in ROOT.joinpath('docs/h7r/suites').glob('O[012]*.json')),
                        *(p.relative_to(ROOT).as_posix() for p in ROOT.joinpath('docs/h7r/contracts').glob('*.json'))):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / rel, root / rel)
            d = root / ev2.CONTRACTS_DIR
            d.mkdir(parents=True)
            (d / 'one.json').write_text(json.dumps(v2_contract(seed=9101, contract_id='ONE')))
            ev2.load_contract(d / 'one.json', root, check_files=False)
            (d / 'two.json').write_text(json.dumps(v2_contract(seed=9101, contract_id='TWO')))
            with self.assertRaisesRegex(ValueError, r'seeds \[9101\] are already used by one.json'):
                ev2.load_contract(d / 'two.json', root, check_files=False)
            (d / 'two.json').write_text(json.dumps(v2_contract(seed=9102, contract_id='ONE')))
            with self.assertRaisesRegex(ValueError, 'the same contract id'):
                ev2.load_contract(d / 'two.json', root, check_files=False)
            (d / 'two.json').write_text(json.dumps(v2_contract(seed=9102, contract_id='TWO')))
            ev2.load_contract(d / 'two.json', root, check_files=False)
            for path in sorted((root / 'docs/h7r/contracts').glob('*.json')):  # v1 still loads beside v2 files
                ev.load_contract(path, root, check_files=False)


class TrainingIntakeTest(unittest.TestCase):
    """A v2 Arena's games (journal lines as the arena writes them) and its arm record never pass the label step."""

    def arena_games(self):
        arm = {'name': 'X_vs_champion', 'seed': seeds.ARENA_SEEDS.start}
        o = json.loads((ROOT / 'docs/h7r/suites/O1.json').read_text())['openings'][:2]
        import h7r_eval as cli
        return [{**t, 'signature': 's' * 64, 'arm': arm['name'], 'moves': t['opening'], 'result': 'draw',
                 'game_seconds': 1.0} for t in cli.build_tasks(arm, o)]

    def test_refused_by_the_label_step(self):
        import h7c_labels as lab
        games = self.arena_games()
        with self.assertRaisesRegex(ValueError, 'clean, committed generator'):
            lab.check_games(games)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'arm.json'
            path.write_text(json.dumps({'format': 'h7r-arena-v1-arm', 'identity': {'seed': 9101}, 'games': games,
                                        'openings': []}))
            with self.assertRaises((ValueError, KeyError)):
                lab.check_games(lab.read_games([path]))


class SuiteBuildTest(unittest.TestCase):
    def test_defaults_rebuild_the_committed_suites(self):
        for name in ('O0', 'O1', 'O2'):
            with self.subTest(name):
                b = json.loads((ROOT / f'docs/h7r/suites/{name}.json').read_text())
                again = op.build_suite(name, b['count'] if name != 'O2' else None, b['seed'])
                self.assertEqual(again, {k: v for k, v in b.items() if k not in ('git', 'overlap_with_earlier')})
                self.assertNotIn('class_exclusion', again)
                self.assertNotIn('distinct_classes', again)

    def test_class_exclusion_and_distinct_classes(self):
        plain = op.build_suite('O0', 120, 7)
        classes = {o['class_id'] for o in plain['openings']}
        self.assertLess(len(classes), 120)  # O0 repeats classes
        distinct = op.build_suite('O0', 40, 7, distinct_classes=True)
        self.assertEqual(len({o['class_id'] for o in distinct['openings']}), 40)
        self.assertTrue(distinct['distinct_classes'])
        banned = {o['class_id'] for o in distinct['openings'][:10]}
        other = op.build_suite('O0', 30, 7, exclude_classes=banned, distinct_classes=True)
        self.assertFalse(banned & {o['class_id'] for o in other['openings']})
        self.assertEqual(other['class_exclusion']['class_ids'], 10)
        self.assertGreater(other['class_exclusion']['excluded_draws'], 0)
        op.check_suite(other)
        bad = copy.deepcopy(distinct)
        bad['openings'][1] = {**bad['openings'][0], 'index': 1}
        with self.assertRaises(ValueError):
            op.check_suite(bad)


class DevSuitesTest(unittest.TestCase):
    def test_the_committed_suites_check(self):
        import h7r_dev_suites as ds
        out = ds.check(None, ROOT, log=lambda *a, **k: None)
        self.assertTrue(out['passed'])
        record = json.loads((ROOT / ds.RECORD).read_text())
        for label, seed in ds.SUITES:
            body = json.loads((ROOT / ds.suite_path(label)).read_text())
            self.assertEqual((body['seed'], body['count'], body['distinct']['class_id']), (seed, 300, 300))
            self.assertEqual(record['suites'][label]['seed'], seed)
            self.assertIn(seed, seeds.SUITE_SEEDS)
        self.assertEqual(record['overlaps']['between_a_and_b'], {'class_id': 0, 'history_id': 0})
        self.assertTrue(all(c['class_id'] == 0 for label in ('a', 'b')
                            for c in record['overlaps'][label]['by_source'].values()))

    def test_checks_catch_shared_classes(self):
        import h7r_dev_suites as ds
        suites = {label: json.loads((ROOT / ds.suite_path(label)).read_text()) for label in ('a', 'b')}
        bad = copy.deepcopy(suites)
        bad['b']['openings'][0] = suites['a']['openings'][0]
        srcs = ds.sources(ROOT)
        self.assertIn('suites a and b share classes or openings', ds.checks(ds.overlaps(srcs, bad), bad))
        o1 = json.loads((ROOT / 'docs/h7r/suites/O1.json').read_text())
        bad = copy.deepcopy(suites)
        bad['a']['openings'][0] = o1['openings'][0]
        self.assertTrue(any('classes shared with docs/h7r/suites/O1.json' in m
                            for m in ds.checks(ds.overlaps(srcs, bad), bad)))


if __name__ == '__main__':
    unittest.main()
