"""H7-R holdout protection: the generator's holdout mode, every training entry refusing holdout games / rows, the
separate evaluation labels and the holdout value evaluation (stand-in networks)."""
import argparse
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hybrid import h7r_eval as ev
from tests.h7r_helpers import FIXTURE_HOLDOUT_SEED as SEED, FIXTURE_OTHER_SEED as OTHER_SEED
from tests.test_h7c_labels import build, game

ROOT = Path(__file__).resolve().parents[1]

ACTOR = {'policy_checkpoint_sha256': 'a' * 64, 'policy_metadata_sha256': 'b' * 64,
         'value_checkpoint_sha256': 'c' * 64, 'value_metadata_sha256': 'd' * 64}  # test_h7c_labels.game's models


def holdout_contract(folder: Path, **change) -> Path:
    body = {'format': ev.CONTRACT_FORMAT, 'contract_id': 'T-HOLDOUT', 'purpose': 'holdout', 'status': 'fixed',
            'models': {'c5': {'policy': 'p.pt', 'value': 'v.pt', 'identity': ACTOR}}, 'arms': [],
            'holdout': {'seed': SEED, 'split': 'dev', 'games': 4, 'actor': 'c5', 'cycle': 1},
            'stats': {'alpha_one_sided': 0.025, 'bootstrap': {'samples': 1000, 'seed': 1, 'method': 'percentile',
                                                              'unit': 'opening pair'}}}
    body.update(change)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{body['contract_id']}.json"
    path.write_text(json.dumps(body))
    return path


def holdout_games(path: Path, n=4):
    sha = ev.file_sha256(path)
    return [game(i, run_seed=SEED, purpose='holdout', holdout={'contract_id': 'T-HOLDOUT', 'contract_sha256': sha,
                                                               'split': 'dev'}) for i in range(n)]


class FixtureSeedTest(unittest.TestCase):
    def test_fixture_seeds_are_reserved_and_free_of_committed_contracts(self):
        from hybrid.h6_labels import RESERVED_SEEDS
        used = set()
        for path in (ROOT / 'docs/h7r/contracts').glob('*.json'):
            used |= ev.contract_seeds(json.loads(path.read_text()))
        for seed in (SEED, OTHER_SEED):
            self.assertIn(seed, ev.SEED_BLOCK)
            self.assertIn(seed, RESERVED_SEEDS)
            self.assertNotIn(seed, used, 'a committed contract took a test fixture seed: move tests/h7r_helpers.py '
                                         'FIXTURE_*_SEED to a free H7-R seed')


class GeneratorTest(unittest.TestCase):
    def test_holdout_seed_rules(self):
        from scripts import h7c_selfplay as sp
        sp.check_seeds(SEED, 1, range(3), holdout=True)
        for seed in (8801, 8771):  # a training seed, a reserved seed outside the H7-R block
            with self.assertRaises(ValueError):
                sp.check_seeds(seed, 1, range(3), holdout=True)
        with self.assertRaises(ValueError):
            sp.check_seeds(SEED, 1, range(3))  # the training path refuses the reserved seed

    def test_holdout_identity_is_the_contract_run_only(self):
        from scripts import h7c_selfplay as sp
        with TemporaryDirectory() as tmp:
            path = holdout_contract(Path(tmp) / 'contracts')
            ok = argparse.Namespace(run_seed=SEED, cycle=1, start=0, games=4, allow_dirty=True)
            out = sp.holdout_identity(path, ok, ACTOR)
            self.assertEqual((out['purpose'], out['holdout']['contract_id']), ('holdout', 'T-HOLDOUT'))
            for change, models in (({'run_seed': OTHER_SEED}, ACTOR), ({'games': 5}, ACTOR), ({'cycle': 2}, ACTOR),
                                   ({}, {**ACTOR, 'value_checkpoint_sha256': 'e' * 64})):
                with self.assertRaises(ValueError):
                    sp.holdout_identity(path, argparse.Namespace(**{**vars(ok), **change}), models)
            draft = holdout_contract(Path(tmp) / 'drafts', status='draft')
            with self.assertRaisesRegex(ValueError, 'fixed'):
                sp.holdout_identity(draft, ok, ACTOR)
            with self.assertRaisesRegex(ValueError, 'committed'):  # a real run needs the committed contract
                sp.holdout_identity(path, argparse.Namespace(**{**vars(ok), 'allow_dirty': False}), ACTOR)


@unittest.skipUnless(__import__('rt30.fast5', fromlist=['available']).available(), 'python -m rt30.vcfcore.build first')
class GeneratorRunTest(unittest.TestCase):
    """Real TB-V3 / fast5 holdout games (stand-in networks), marked and refused by the training labels."""

    def tearDown(self):
        from hybrid.h6_fast import install
        from rt30 import fast5
        fast5.uninstall()
        install('off')

    def test_holdout_run(self):
        from analysis import tb_v3_net
        from scripts import h7c_labels as lab
        from scripts import h7c_selfplay as sp
        from tests.test_h7c_selfplay import args_for
        from tests.test_tb_v3_net import write_models

        with TemporaryDirectory() as tmp:
            policy, value = write_models(Path(tmp) / 'm', seed=4)
            actor = tb_v3_net.model_identity(policy, value)
            path = holdout_contract(Path(tmp) / 'contracts', models={'c5': {'policy': 'p', 'value': 'v',
                                                                            'identity': actor}})
            args = args_for(tmp, policy, value, run_seed=SEED, games=1, holdout_contract=path)
            out = sp.run(args, log=lambda *a, **k: None)
            self.assertEqual(out['identity']['purpose'], 'holdout')
            rows = [json.loads(x) for x in args.jsonl.read_text().splitlines()]
            self.assertEqual((rows[0]['purpose'], rows[0]['holdout']['contract_sha256']),
                             ('holdout', ev.file_sha256(path)))
            with self.assertRaisesRegex(ValueError, 'never training data'):
                lab.check_games(rows)
            with self.assertRaises(ValueError):  # without the contract the reserved seed is refused
                sp.run(args_for(tmp, policy, value, run_seed=SEED, games=1, jsonl=Path(tmp) / 'x.jsonl'),
                       log=lambda *a, **k: None)
            plain = sp.run_identity(1, 8801, actor, 'cpu')
            self.assertNotIn('purpose', plain)  # the training identity is unchanged


class TrainingEntryTest(unittest.TestCase):
    def test_training_labels_refuse_holdout_games(self):
        from scripts import h7c_labels as lab
        with TemporaryDirectory() as tmp:
            games = holdout_games(holdout_contract(Path(tmp)))
            with self.assertRaisesRegex(ValueError, 'never training data'):
                lab.check_games(games)
            # the holdout mark cannot be dropped by giving the games a training seed
            with self.assertRaisesRegex(ValueError, 'never training data'):
                lab.check_games([{**g, 'run_seed': 8801} for g in games])
            # nor can the mark be dropped while keeping the reserved seed
            plain = [{k: v for k, v in g.items() if k not in ('purpose', 'holdout')} for g in games]
            with self.assertRaises(ValueError):
                lab.check_games(plain)

    def test_training_row_sources_and_cycle_cache_refuse(self):
        from hybrid import h7c_train as tr
        from scripts import h7c_train_value as tv
        detail = {**ACTOR, 'runtime_hash': 'r' * 64, 'selfplay_git_commit': 'x', 'purpose': 'holdout_eval'}
        with self.assertRaisesRegex(ValueError, 'never training data'):
            tr.rows_source([detail], ['h7c:c1:8801:0'], 1)
        with TemporaryDirectory() as tmp:
            rows = Path(tmp) / 'rows.jsonl.gz'
            with gzip.open(rows, 'wt') as handle:
                handle.write(json.dumps({'game_id': f'h7c:c1:{SEED}:0', 'split': 'train'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'reserved'):
                tv.check_cycle_cache(Path(tmp), 1, rows=rows)
            cache = Path(tmp) / 'cache'
            cache.mkdir()
            for split in ('train', 'val', 'test'):
                (cache / f'{split}_games.json').write_text(json.dumps([f'h7c:c1:{SEED}:0'] if split == 'test' else []))
            with self.assertRaisesRegex(ValueError, 'reserved'):
                tv.check_cycle_cache(cache, 1)


class EvaluationLabelsTest(unittest.TestCase):
    def test_labels_and_eval(self):
        from scripts import h7c_labels as lab
        from scripts import h7r_holdout as ho
        from hybrid import h7c_train as tr
        from hybrid.h6_value_cache import build_value_cache
        from tests.test_tb_v3_net import write_models

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = holdout_contract(tmp / 'contracts')
            contract, sha = ev.load_contract(path, tmp, check_files=False, contracts_dir=path.parent)
            games = holdout_games(path)
            with self.assertRaisesRegex(ValueError, 'not a game of this holdout'):
                ho.check_holdout_games([{**games[0], 'run_seed': OTHER_SEED}] + games[1:], contract, sha)
            with self.assertRaisesRegex(ValueError, 'not a game of this holdout'):
                ho.check_holdout_games([{**games[0], 'holdout': {**games[0]['holdout'],
                                                                 'contract_sha256': '0' * 64}}], contract, sha)
            rows, _, _ = build(games, evaluation={'check': lambda g: ho.check_holdout_games(g, contract, sha),
                                                  'purpose': ho.PURPOSE, 'split': 'test'})
            self.assertTrue(rows and all(r.split == 'test' for r in rows))
            dicts = [lab.value_row_dict(r) for r in rows]
            self.assertTrue(all(d['h7c_provenance']['purpose'] == 'holdout_eval' for d in dicts))
            lab.write_atomic({tmp / 'rows.jsonl.gz': dicts})
            build_value_cache(tmp / 'rows.jsonl.gz', tmp / 'cache', progress_seconds=1e9)
            rawpath = tmp / 'games.jsonl'
            rawpath.write_text(''.join(json.dumps(g) + '\n' for g in games))
            ho.seal_cache(tmp / 'cache', path, rawpath, tmp / 'rows.jsonl.gz')
            with self.assertRaisesRegex(ValueError, 'never training data'):
                tr.value_rows_source(tmp / 'rows.jsonl.gz', 1, json.loads(
                    (tmp / 'cache/manifest.json').read_text())['labels_sha256'])
            # the holdout value evaluation on that cache
            _, v1 = write_models(tmp / 'm1', seed=1)
            _, v2 = write_models(tmp / 'm2', seed=2)
            _, v3 = write_models(tmp / 'm3', seed=3)
            args = argparse.Namespace(root=tmp, cache=tmp / 'cache', model=[f'a={v1}', f'b={v2}', f'c={v3}'], pair=['a:b'],
                                      group=['r=b,c'], device='cpu', d4=True, bootstrap_samples=200,
                                      bootstrap_seed=5)
            out = ho.evaluate(args, log=lambda *a, **k: None)
            self.assertEqual(out['rows'], len([r for r in rows if not r.value_mask]))
            self.assertEqual(sum(out['route_counts'].values()), out['rows'])
            self.assertGreater(out['route_counts']['exact_leaf'], 0)  # the fixture game ends in a five
            cell = out['paired']['a:b']['raw']['all']
            self.assertEqual(cell['games'], 4)
            self.assertLessEqual(cell['ci95'][0], cell['difference'] + 1e-9)
            self.assertIn('d4', out['models']['a'])
            self.assertEqual(out['seed_spread']['r']['raw']['all']['seeds'], 2)
            self.assertIsNone(out['epsilon'])


if __name__ == '__main__':
    unittest.main()
