"""H7-c search-budget ablation (scripts/h7c_budget.py): budgets, seeds, the reading rule, real games and positions."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rt30 import fast5
from scripts import h7c_arena as ar
from scripts import h7c_budget as sb


def fake_game(k, pair, color, result, tree=True):
    move = {'ply': 3, 'route': 'tree' if tree else 'stage2', 'seconds': 0.2 * k, 'tree_budget': 50 * k,
            'chose_prior_top': True, 'top_share': 0.5, 'visited_children': 4}
    return {'key': f's{50 * k}/{sb.SEED}/{pair}/{color}', 'pair': pair, 'tested_color': color, 'result': result,
            'length': 30, 'illegal': None, 'truncated': False, 'moves': [[7, 7], [pair, 0]], 'tested_moves': [move],
            'seconds': {'tested': 0.2 * k, 'opponent': 0.2}}


def write_arm(tmp, k, pair_results):
    ident = {'format': sb.FORMAT, 'factor': k, 'budget': sb.budget(k), 'opponent_budget': sb.budget(1),
             'seed': sb.SEED, 'engine_hash': 'e', 'runtime_hash': 'r', 'models': {'m': 1}, 'git': {'commit': 'abc'}}
    sig = ar.signature(ident)
    games = [{**fake_game(k, pair, color, result), 'signature': sig}
             for pair, (black, white) in enumerate(pair_results) for color, result in (('black', black), ('white', white))]
    path = Path(tmp) / f'k{k}.json'
    path.write_text(json.dumps({'identity': ident, 'signature': sig, 'games': games,
                                'summary': sb.summarize_arm(games), 'fingerprint': ar.fingerprint(games)}))
    return str(path)


class RuleTest(unittest.TestCase):
    def test_budget_scales_both_numbers(self):
        self.assertEqual(sb.budget(1), {'simulations': 50, 'tactical_simulations': 100})  # the frozen TB-V3
        self.assertEqual(sb.budget(4), {'simulations': 200, 'tactical_simulations': 400})
        from analysis.tb_v2_value import TB_V2_VALUE
        self.assertEqual({k: TB_V2_VALUE[k] for k in sb.BASE}, sb.BASE)
        with self.assertRaises(ValueError):
            sb.budget(0)

    def test_seeds_and_pairing(self):
        from hybrid.h6_labels import RESERVED_SEEDS, check_reserved_seed

        self.assertTrue({8741, 8742} <= RESERVED_SEEDS)
        with self.assertRaises(ValueError):  # never training data
            check_reserved_seed(8741)
        by_k = {k: sb.build_tasks(k) for k in sb.FACTORS}
        self.assertEqual({len(t) for t in by_k.values()}, {100})
        for a, b in zip(by_k[2], by_k[8]):  # pair i: the same opening and game seeds for every budget
            self.assertEqual((a['opening'], a['seed'], a['tested_color']), (b['opening'], b['seed'], b['tested_color']))
            self.assertNotEqual(a['key'], b['key'])

    def test_summary_reading(self):
        with TemporaryDirectory() as tmp:
            helps = [write_arm(tmp, 2, [('win', 'loss')] * 50), write_arm(tmp, 4, [('win', 'win')] * 35
                                                                          + [('win', 'loss')] * 15),
                     write_arm(tmp, 8, [('win', 'win')] * 40 + [('loss', 'loss')] * 10)]
            out = sb.summary(argparse.Namespace(arms=helps, positions=None))
            self.assertEqual(out['reading'], 'BUDGET_HELPS')
            self.assertEqual((out['arms']['S100']['stronger'], out['arms']['S200']['stronger']), (False, True))
            self.assertEqual(out['smallest_stronger_budget'], {'simulations': 200, 'tactical_simulations': 400})
            self.assertEqual(out['arms']['S200']['cost']['time_ratio'], 4.0)
        with TemporaryDirectory() as tmp:
            flat = [write_arm(tmp, k, [('win', 'loss')] * 50) for k in sb.FACTORS]
            self.assertEqual(sb.summary(argparse.Namespace(arms=flat, positions=None))['reading'], 'NOT_DETECTED')
            with self.assertRaises(ValueError):  # every factor exactly once
                sb.summary(argparse.Namespace(arms=flat[:2], positions=None))
            record = json.loads(Path(flat[0]).read_text())
            record['summary']['score'] = 0.9
            Path(flat[0]).write_text(json.dumps(record))
            with self.assertRaises(ValueError):  # a summary its games do not give
                sb.summary(argparse.Namespace(arms=flat, positions=None))


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class EngineTest(unittest.TestCase):
    """Real TB-V3 / fast5 agents on random stand-in networks of the frozen architecture."""

    def setUp(self):
        from tests.test_tb_v3_net import write_models

        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.policy, self.value = write_models(Path(self.tmp.name) / 'm', seed=3)

    def tearDown(self):
        from hybrid.h6_fast import install
        fast5.uninstall()
        install('off')
        ar._MODELS.clear()

    def test_only_the_budget_changes(self):
        from renju import Game

        game = Game()
        for move in [(7, 7), (6, 8), (8, 8), (6, 6)]:
            game.play(*move)
        decisions = {}
        for k in (1, 2, 4):
            agent = sb.make_budget_agent(k, str(self.policy), str(self.value), 11, 'cpu')
            decisions[k] = sb.decision(agent, game)
        routes = {d['route'] for d in decisions.values()}
        self.assertEqual(len(routes), 1)  # the stage and solver routes do not depend on the tree budget
        if routes == {'tree'}:
            budgets = [decisions[k]['tree_budget'] for k in (1, 2, 4)]
            self.assertIn(budgets, ([50, 100, 200], [100, 200, 400]))  # normal or tactical, scaled alike
            self.assertEqual(sum(v[2] for v in decisions[4]['visits']), budgets[2])

    def test_games_and_positions(self):
        from analysis import tb_v3_net
        from scripts.h7c_labels import write_atomic
        from tests.test_h7c_labels import build, game

        quiet = lambda *a, **k: None  # noqa: E731
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            args = argparse.Namespace(factor=2, policy_checkpoint=self.policy, value_checkpoint=self.value,
                                      device='cpu', workers=1, jsonl=tmp / 'k2.jsonl', output=tmp / 'k2.json',
                                      allow_dirty=True, allow_stand_in=True, pairs=1, max_plies=6)
            out = sb.play(args, log=quiet)
            self.assertEqual(len(out['games']), 2)
            self.assertEqual(out['identity']['budget'], {'simulations': 100, 'tactical_simulations': 200})
            self.assertTrue(all(m['tree_budget'] in (None, 100, 200) for g in out['games'] for m in g['tested_moves']))
            again = sb.play(args, log=quiet)  # resumed: the same games
            self.assertEqual([g['moves'] for g in again['games']], [g['moves'] for g in out['games']])
            with self.assertRaises(ValueError):  # the ablation runs the frozen models only
                sb.play(argparse.Namespace(**{**vars(args), 'allow_stand_in': False, 'jsonl': tmp / 'x.jsonl'}),
                        log=quiet)
            models = tb_v3_net.model_identity(self.policy, self.value)
            _, policy_rows, _ = build([game(i, models=models, model_hash=tb_v3_net.model_hash(models))
                                       for i in (22, 47, 50)])  # val games of the label split
            write_atomic({tmp / 'policy_rows.jsonl.gz': policy_rows})
            evidence = tmp / 'evidence.json'
            evidence.write_text(json.dumps({'files_sha256': {'policy_rows.jsonl.gz':
                                                             ar._file_sha(tmp / 'policy_rows.jsonl.gz')}}))
            pargs = argparse.Namespace(policy_checkpoint=self.policy, value_checkpoint=self.value, device='cpu',
                                       workers=1, output=tmp / 'pos.json', allow_dirty=True, allow_stand_in=True,
                                       policy_rows=tmp / 'policy_rows.jsonl.gz', evidence=evidence, limit=2)
            result = sb.positions(pargs, log=quiet)
            self.assertEqual(result['positions'], 2)
            self.assertEqual(set(result['summary']['by_factor']), {'1', '2', '4', '8'})
            self.assertIn('reproduced_at_k1', result['summary'])  # the fixture's visits are made up: a count only
            evidence.write_text(json.dumps({'files_sha256': {'policy_rows.jsonl.gz': 'f' * 64}}))
            with self.assertRaises(ValueError):  # rows the evidence manifest does not name
                sb.positions(argparse.Namespace(**{**vars(pargs), 'output': tmp / 'p2.json'}), log=quiet)


if __name__ == '__main__':
    unittest.main()
