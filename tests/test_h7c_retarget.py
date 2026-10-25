"""H7-c search retarget (hybrid/h7c_retarget.py, scripts/h7c_retarget.py, h7c_labels --retarget, the trainer's rows)."""
import argparse
import copy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from hybrid import h7c_retarget as hr
from hybrid.h6_labels import canonical_hash
from rt30 import fast5
from scripts import h7c_labels as lab
from scripts import h7c_retarget as rt
from scripts import h7c_selfplay as sp
from tests.test_h7c_labels import MOVES, build, game

ROOT = Path(__file__).resolve().parents[1]


def identity_for(games, **changes):
    ident = {'format': hr.FORMAT, 'purpose': 'retarget', 'factor': 4, 'budget': hr.budget(4), 'mode': 'restored',
             'selfplay_jsonl_sha256': 'e' * 64, 'selfplay': hr.selfplay_identity(games), 'decision_sha256': 'd' * 64,
             'decision_inputs_sha256': {}, 'cost_review_sha256': 'c' * 64, 'coverage': hr.coverage(games),
             'git': {'commit': 'r1', 'dirty': False}}
    ident.update(changes)
    return ident


def records_for(games, identity, change=None):
    """Retarget records of every visits ply: by default another visit split (the 'deeper search')."""
    out = []
    for g in games:
        rows = []
        for p in hr.eligible_plies(g):
            row = {'ply': p['ply'], 'color': p['color'], 'seed': 1,
                   'canonical_hash': canonical_hash(g['moves'][:p['ply']]),
                   'route': 'tree', 'engine_move': p['engine_move'],
                   'visits': [[*p['engine_move'], 150, 0.2], [3, 3, 50, 0.0]], 'proven_loss': [], 'e4': {},
                   'tree_budget': 200, 'seconds': 0.5}
            if change:
                change(g, p, row)
            rows.append(row)
        out.append({'identity': identity, 'game_index': g['game_index'], 'rows': rows})
    return out


def retarget(games, records=None, identity=None):
    identity = identity or identity_for(games)
    return {'records': records if records is not None else records_for(games, identity), 'sha256': 'f' * 64}


class RuleTest(unittest.TestCase):
    def test_budget_and_frozen_base(self):
        from analysis.tb_v2_value import TB_V2_VALUE

        self.assertEqual(hr.budget(4), {'simulations': 200, 'tactical_simulations': 400})
        self.assertEqual({**TB_V2_VALUE, **hr.budget(1)}, TB_V2_VALUE)  # factor 1 is the frozen configuration
        for bad in (0, 1.5):
            with self.assertRaises(ValueError):
                hr.budget(bad)

    def test_every_visits_ply_is_eligible(self):
        """Policy rows start at ply 3 (``--from-ply`` is the value rows' start): ply 3 and 4 are retargeted too."""
        g = game(0)
        self.assertEqual([p['ply'] for p in hr.eligible_plies(g)], [3, 4, 7])
        cov = hr.coverage([g, game(1)])
        self.assertEqual((cov['eligible'], cov['games']), (6, 2))
        self.assertNotEqual(cov['coverage_sha256'], hr.coverage([g])['coverage_sha256'])

    def test_loss_union(self):
        losses, counts = hr.effective_losses([[1, 1], [2, 2]], [[2, 2], [3, 3]])
        self.assertEqual(losses, [[1, 1], [2, 2], [3, 3]])
        self.assertEqual(counts, {'same': 1, 'actor_only': 1, 'retarget_only': 1, 'union': 3})

    def test_cost_gate(self):
        checked = set()
        gate = lambda **k: hr.cost_gate(**{'elapsed': 100.0, 'rows_total': 10_000, 'cap_seconds': 10_728.5,  # noqa
                                           'checked': checked, **k})
        self.assertIsNone(gate(games_done=19, rows_done=900))      # not 20 games yet
        self.assertIsNone(gate(games_done=40, rows_done=255))      # not 256 rows yet
        first = gate(games_done=20, rows_done=256)
        self.assertEqual((first['point'], first['projected_seconds'], first['exceeded']), ('first', 3906.2, False))
        self.assertIsNone(gate(games_done=30, rows_done=300))      # the next point is 10%
        late = gate(games_done=50, rows_done=1_000, elapsed=1_100.0)
        self.assertEqual((late['point'], late['projected_seconds'], late['exceeded']), ('0.10', 11_000.0, True))

    def test_compare_to_actor(self):
        p = game(0)['plies'][0]
        row = {'route': p['route'], 'engine_move': p['engine_move'], 'visits': copy.deepcopy(p['visits']),
               'proven_loss': [], 'e4': {}}
        self.assertEqual(hr.compare_to_actor(p, row), [])
        row['visits'][0][3] += 9e-6
        self.assertEqual(hr.compare_to_actor(p, row), [])
        row['visits'][0][3] += 2e-5
        self.assertEqual(hr.compare_to_actor(p, row), ['mean_value'])
        row['visits'][0][2] += 1
        self.assertEqual(hr.compare_to_actor(p, row), ['visits', 'target'])
        row = {**row, 'visits': copy.deepcopy(p['visits']), 'proven_loss': [[3, 3]], 'e4': {'skip': 'cooldown'}}
        self.assertEqual(hr.compare_to_actor(p, row), ['proven_loss', 'e4', 'target'])


class PlanTest(unittest.TestCase):
    """The decision of cycle c - 1 (and its committed cost review) chooses factor, mode and cap."""

    def setUp(self):
        self.decision = json.loads((ROOT / 'docs/h7c/c1/decision.json').read_text(encoding='utf-8'))
        self.review = json.loads((ROOT / 'docs/h7c/c1/cost_review.json').read_text(encoding='utf-8'))
        self.sha = rt.file_sha256(ROOT / 'docs/h7c/c1/decision.json')

    def test_committed_c1_review(self):
        plan = rt.approved_budget(self.decision, self.sha, self.review, 'r' * 64, 2)
        self.assertEqual((plan['factor'], plan['mode'], plan['cap_seconds']), (4, 'restored', 10728.5))
        self.assertEqual(self.review['hard_cap_seconds'], round(1.25 * self.decision['budget']['ceiling_seconds'], 1))
        self.assertEqual(self.review['decision_inputs_sha256'], self.decision['inputs_sha256'])

    def test_refusals(self):
        def refused(decision=None, sha=None, review='same', cycle=2):
            with self.assertRaises(ValueError):
                rt.approved_budget(decision or self.decision, sha or self.sha,
                                   self.review if review == 'same' else review, 'r' * 64, cycle)
        refused(sha='0' * 64)                                          # another decision file
        refused(cycle=3)                                               # the decision of another cycle
        refused(review=None)                                           # A_COST_REVIEW needs its review
        refused(review={**self.review, 'approved': {**self.review['approved'], 'factor': 8}})
        refused(review={**self.review, 'applies_to_cycle': 3})
        refused(review={**self.review, 'hard_cap_seconds': None})
        refused(decision={**self.decision, 'case': 'B_POLICY'})
        refused(decision={**self.decision, 'case': 'V9_ENTRY_E1'})
        refused(decision={**self.decision, 'case': 'A'})               # case A: no review (its ceiling is the cap)
        plan = rt.approved_budget({**self.decision, 'case': 'A'}, self.sha, None, None, 2)
        self.assertEqual(plan['cap_seconds'], self.decision['budget']['ceiling_seconds'])


class FakeAgent:
    def __init__(self, seed):
        from analysis.e4v2_attack import Cooldown

        self.seed, self.cooldown, self.resets = seed, Cooldown(), 0

    def new_game(self):
        from analysis.e4v2_attack import Cooldown

        self.cooldown, self.resets = Cooldown(), self.resets + 1


class ReplayTest(unittest.TestCase):
    """Which plies each mode searches, with which agent and state (the search itself faked)."""

    def calls(self, g, mode):
        log = []

        def fake_search(agent, position):
            blocked = agent.cooldown.blocks(position)
            log.append((agent.seed, len(position.history), blocked))
            move = g['moves'][len(position.history)]
            return {'route': 'tree', 'engine_move': move, 'visits': [[*move, 1, 0.0]], 'proven_loss': [],
                    'e4': {'skip': 'cooldown'} if blocked else {}, 'tree_budget': 200, 'seconds': 0.0}

        with mock.patch.object(rt, 'search', fake_search):
            rows = rt.retarget_game(g, mode, make=FakeAgent)
        return rows, log

    def test_restored_and_sequential(self):
        from run_mcts_v8_benchmark import derive_seed

        g = game(0, game_seed=77)
        g['plies'][4]['e4'] = {'skip': 'cooldown'}  # ply 7 (tree, white) was blocked in the actor's game
        black, white = derive_seed(77, 'black'), derive_seed(77, 'white')
        rows, log = self.calls(g, 'restored')
        self.assertEqual(log, [(white, 3, False), (black, 4, False), (white, 7, True)])  # visits plies only
        self.assertEqual([(r['ply'], r['seed'], r['cooldown_restored']) for r in rows],
                         [(3, white, False), (4, black, False), (7, white, True)])
        rows, log = self.calls(g, 'sequential')
        self.assertEqual([n for _, n, _ in log], [3, 4, 5, 6, 7, 8])  # every ply, the actor's moves played
        self.assertEqual([r['ply'] for r in rows], [3, 4, 7])
        self.assertTrue(all('cooldown_restored' not in r for r in rows))


class LabelsTest(unittest.TestCase):
    def test_visits_rows_only_are_replaced(self):
        games = [game(0), game(1)]
        rows_v1, policy_v1, _ = build(games)
        rows_v2, policy_v2, summary = build(games, retarget=retarget(games))
        self.assertEqual([lab.value_row_dict(r) for r in rows_v1], [lab.value_row_dict(r) for r in rows_v2])
        self.assertEqual([(r['game_id'], r['ply'], r['kind']) for r in policy_v1],
                         [(r['game_id'], r['ply'], r['kind']) for r in policy_v2])
        for a, b in zip(policy_v1, policy_v2):
            self.assertEqual(b['format'], 'h7c-policy-row-v2')
            self.assertEqual(b['retarget']['factor'], 4)
            if a['kind'] == 'visits':
                self.assertEqual(b['target_source'], 'retarget')
                self.assertEqual(sorted(b['target']), sorted([[*a['played'], 0.75], [3, 3, 0.25]]))  # 150 : 50
                self.assertEqual(sorted(a['target']), sorted([[*a['played'], 0.8], [3, 3, 0.2]]))    # 40 : 10
                self.assertEqual(b['actor_raw_visits'], a['raw_visits'])
            else:  # proof rows keep the actor's target
                self.assertEqual((b['target_source'], b['target'], b['kl']), ('actor', a['target'], a['kl']))
        self.assertEqual(summary['policy']['retarget']['kept_games']['retargeted'], 6)
        self.assertFalse(any(set(hr.V2_FIELDS) & set(r) for r in policy_v1))  # the v1 output is unchanged

    def test_union_of_proven_losses_and_no_mass(self):
        games = [game(0)]

        def change(g, p, row):
            if p['ply'] == 3:
                row['proven_loss'] = [[3, 3]]             # the retarget proved the alternative losing
            if p['ply'] == 4:
                row['proven_loss'] = [list(p['engine_move']), [3, 3]]  # every visited move proven losing

        ident = identity_for(games)
        _, policy, summary = build(games, retarget=retarget(games, records_for(games, ident, change), ident))
        by_ply = {r['ply']: r for r in policy}
        self.assertEqual(by_ply[3]['target'], [[*MOVES[3], 1.0]])
        self.assertEqual(by_ply[3]['proven_loss'], [[3, 3]])
        self.assertNotIn(4, by_ply)                               # no mass: no policy loss there
        counts = summary['policy']['retarget']['kept_games']
        self.assertEqual((counts['retarget_none'], counts['retarget_only'], counts['retargeted']), (1, 3, 2))

    def test_v1_projection_reproduces_the_actor_rows(self):
        """G0 at the label level: records equal to the actor's give the v1 rows back."""
        games = [game(0), game(1)]

        def same(g, p, row):
            row['visits'], row['proven_loss'] = copy.deepcopy(p['visits']), list(p['proven_loss'])

        ident = identity_for(games, factor=1, budget=hr.budget(1))
        _, policy_v1, _ = build(games)
        _, policy_v2, _ = build(games, retarget=retarget(games, records_for(games, ident, same), ident))
        drop = lambda r: {k: v for k, v in r.items() if k not in hr.PROVENANCE_FIELDS}  # noqa: E731
        self.assertEqual([json.dumps(drop(hr.v1_projection(r)), sort_keys=True) for r in policy_v2],
                         [json.dumps(drop(r), sort_keys=True) for r in policy_v1])

    def test_refusals(self):
        games = [game(0), game(1)]
        ident = identity_for(games)
        good = records_for(games, ident)

        def refused(records=None, identity=None, these=games):
            with self.assertRaises(ValueError):
                build(these, retarget=retarget(these, records if records is not None else
                                               records_for(these, identity or ident), identity or ident))

        missing = copy.deepcopy(good)
        missing[0]['rows'].pop()
        refused(missing)
        twice = copy.deepcopy(good)
        twice[0]['rows'].append(twice[0]['rows'][0])
        refused(twice)
        extra_ply = copy.deepcopy(good)
        extra_ply[0]['rows'].append({**extra_ply[0]['rows'][0], 'ply': 5})  # a forced ply
        refused(extra_ply)
        refused(good + [{**copy.deepcopy(good[0]), 'game_index': 9}])  # a game of another file
        for name, edit in (('route', {'route': 'stage2'}), ('e4', {'e4': {'skip': 'cooldown'}}),
                           ('position', {'canonical_hash': 'x'}), ('illegal', {'visits': [[0, 0, 10, 0.0]]})):
            bad = copy.deepcopy(good)
            bad[1]['rows'][0].update(edit)
            with self.subTest(name):
                refused(bad)
        refused(identity=identity_for(games, purpose='g0'))                     # a G0 check file
        refused(identity=identity_for(games, budget=hr.budget(8)))              # budget not of its factor
        other = identity_for(games)
        other['selfplay'] = {**other['selfplay'], 'models': {**other['selfplay']['models'],
                                                             'policy_checkpoint_sha256': '9' * 64}}
        refused(identity=other)                                                  # another model pair
        refused(identity=identity_for([games[0]]))                              # coverage of another game set
        mixed = copy.deepcopy(good)
        mixed[1]['identity'] = identity_for(games, mode='sequential')
        refused(mixed)                                                           # two identities in one file


class TrainerRowsTest(unittest.TestCase):
    def write(self, tmp, rows):
        path = Path(tmp) / 'p.jsonl.gz'
        with gzip.open(path, 'wt', encoding='utf-8') as handle:
            for r in rows:
                handle.write(json.dumps(r) + '\n')
        return path

    def test_v2_rows_and_their_retarget(self):
        from hybrid import h7c_train as tr

        games = [game(0)]
        _, policy_v2, _ = build(games, retarget=retarget(games))
        _, policy_v1, _ = build(games)
        with TemporaryDirectory() as tmp:
            source = tr.policy_rows_source(self.write(tmp, policy_v2), 1)
            self.assertEqual((source['retarget']['factor'], source['retarget']['sha256']), (4, 'f' * 64))
            self.assertEqual(len(tr.load_policy_rows(self.write(tmp, policy_v2))['train'] +
                                 tr.load_policy_rows(self.write(tmp, policy_v2))['val']), len(policy_v2))
            self.assertIsNone(tr.policy_rows_source(self.write(tmp, policy_v1), 1)['retarget'])
            with self.assertRaises(ValueError):  # one row format per file
                tr.load_policy_rows(self.write(tmp, policy_v1 + policy_v2))
            with self.assertRaises(ValueError):  # one retarget per file
                tr.policy_rows_source(self.write(tmp, policy_v1 + policy_v2), 1)


@unittest.skipUnless(fast5.available(), 'python -m rt30.vcfcore.build first')
class EngineTest(unittest.TestCase):
    """Real TB-V3 / fast5 agents on random stand-in networks: G0 (factor 1 reproduces the actor) and G0b."""

    @classmethod
    def setUpClass(cls):
        from analysis import tb_v3_net
        from tests.test_tb_v3_net import write_models

        cls.tmp = TemporaryDirectory()
        cls.policy, cls.value = write_models(Path(cls.tmp.name) / 'm', seed=5)
        sp._init_worker(str(cls.policy), str(cls.value), 'cpu')
        models = tb_v3_net.model_identity(cls.policy, cls.value)
        cls.games = []
        for i in range(2):
            task = {'identity': {**sp.run_identity(2, 8802, models, 'cpu'),
                                 'generator_git': {'commit': 'x', 'dirty': False}},
                    'game_index': i, 'game_seed': sp.game_seed(8802, 2, i), 'opening_seed': sp.opening_seed(8802, 2, i)}
            cls.games.append(sp.play_game(task, max_plies=12))

    @classmethod
    def tearDownClass(cls):
        from hybrid.h6_fast import install
        fast5.uninstall()
        install('off')
        cls.tmp.cleanup()

    def test_g0_factor_one_reproduces_the_actor(self):
        rt._init_worker(str(self.policy), str(self.value), 'cpu', 1)
        rows = 0
        for g in self.games:
            for mode in hr.MODES:
                for p, row in zip(hr.eligible_plies(g), rt.retarget_game(g, mode)):
                    self.assertEqual(hr.compare_to_actor(p, row), [], (g['game_index'], p['ply'], mode))
                    hr.check_row(g, p, row)
                    rows += 1
        self.assertGreater(rows, 0)

    def test_g0b_modes_agree_at_a_larger_budget(self):
        rt._init_worker(str(self.policy), str(self.value), 'cpu', 2)
        for g in self.games:
            restored, sequential = rt.retarget_game(g, 'restored'), rt.retarget_game(g, 'sequential')
            self.assertEqual([rt._field_diff(a, b) for a, b in zip(restored, sequential)], [[]] * len(restored))
            self.assertTrue(all(r['tree_budget'] in (100, 200) for r in restored))

    def test_check_g0_report(self):
        """``check --factor 1 --mode both --policy-rows``: PASS on the cycle's rows, FAIL once a row is not theirs."""
        from hybrid.h6_labels import split_for

        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            label_provenance = lambda g: {'engine_config_hash': g['engine_hash'], 'model_hash': g['model_hash'],  # noqa
                                          'git_commit': 'label', 'h7c_provenance': {'label_git_commit': 'label'}}
            rows = [r for g in self.games for r in lab.policy_rows(
                g, split_for(lab.game_id(g), fractions=lab.SPLIT_FRACTIONS, salt=lab.SPLIT_SALT), label_provenance(g))]
            rows[0]['mask'], rows[0]['flags'] = True, ['probe_position']  # masks come after: not compared
            self.assertTrue(any(r['kind'] == 'visits' for r in rows))
            paths = {'good': tmp / 'good.jsonl.gz', 'bad': tmp / 'bad.jsonl.gz'}
            bad = copy.deepcopy(rows)
            next(r for r in bad if r['kind'] == 'visits')['target'][0][2] += 0.01
            for name, these in (('good', rows), ('bad', bad)):
                with gzip.open(paths[name], 'wt', encoding='utf-8') as handle:
                    handle.write(''.join(json.dumps(r) + '\n' for r in these))
            jsonl = ROOT / 'runs' / '_test_retarget_check' / 'g0.jsonl'
            self.addCleanup(lambda: (jsonl.unlink(missing_ok=True), jsonl.parent.rmdir()))
            args = argparse.Namespace(games=tmp / 'g.jsonl', policy_checkpoint=self.policy,
                                      value_checkpoint=self.value, device='cpu', workers=1, jsonl=jsonl,
                                      output=tmp / 'g0.json', factor=1, mode='both', games_limit=None,
                                      policy_rows=paths['good'], allow_dirty=True)
            args.games.write_text(''.join(json.dumps(g) + '\n' for g in self.games), encoding='utf-8')
            quiet = lambda *a, **k: None  # noqa: E731
            with mock.patch.object(rt, 'load_games', lambda path: (self.games, hr.selfplay_identity(self.games))):
                out = rt.check(args, log=quiet)
                self.assertEqual((out['verdict'], out['modes_differ'], out['actor']),
                                 ('PASS', {}, {'restored': {}, 'sequential': {}}))
                self.assertEqual((out['projection']['compared'], out['projection']['differ']), (len(rows), 0))
                out = rt.check(argparse.Namespace(**{**vars(args), 'policy_rows': paths['bad']}), log=quiet)
                self.assertEqual((out['verdict'], out['projection']['differ']), ('FAIL', 1))

    def test_run_resume_and_identity(self):
        """``run`` end to end at factor 2 (the games are cut, so the label step's game check is stubbed)."""
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            games_path = tmp / 'games.jsonl'
            games_path.write_text(''.join(json.dumps(g) + '\n' for g in self.games), encoding='utf-8')
            decision = {'format': 'h7c-decision-v1', 'cycle': 1, 'case': 'A_COST_REVIEW', 'inputs_sha256': {},
                        'budget': {'helps': True, 'factor': 2, 'budget': hr.budget(2), 'ceiling_seconds': 1.0}}
            (tmp / 'decision.json').write_text(json.dumps(decision), encoding='utf-8')
            review = {'format': 'h7c-cost-review-v1', 'decision_sha256': rt.file_sha256(tmp / 'decision.json'),
                      'applies_to_cycle': 2, 'case': 'A_COST_REVIEW', 'hard_cap_seconds': 1e6,
                      'approved': {'factor': 2, 'budget': hr.budget(2), 'mode': 'restored'}}
            (tmp / 'review.json').write_text(json.dumps(review), encoding='utf-8')
            selfplay = hr.selfplay_identity(self.games)
            args = argparse.Namespace(games=games_path, decision=tmp / 'decision.json', cost_review=tmp / 'review.json',
                                      policy_checkpoint=self.policy, value_checkpoint=self.value, device='cpu',
                                      workers=1, jsonl=ROOT / 'runs' / '_test_retarget' / 'r.jsonl',
                                      output=tmp / 'summary.json', selfplay_summary=None, allow_dirty=True)
            self.addCleanup(lambda: (args.jsonl.unlink(missing_ok=True), args.jsonl.parent.rmdir()))
            quiet = lambda *a, **k: None  # noqa: E731
            with mock.patch.object(rt, 'load_games', lambda path: (self.games, selfplay)):
                out = rt.run(args, log=quiet)
                self.assertEqual(out['identity']['factor'], 2)
                self.assertEqual(out['coverage']['completed'], out['coverage']['eligible'])
                again = rt.run(args, log=quiet)  # resumed: nothing new
                self.assertEqual(again['jsonl_sha256'], out['jsonl_sha256'])
                records = lab.load_retarget(args.jsonl)
                self.assertEqual(lab.retarget_rows(self.games, records, 'x')['info']['factor'], 2)
                review['hard_cap_seconds'] = 2e6  # another approval: another identity, never mixed into this file
                (tmp / 'review.json').write_text(json.dumps(review), encoding='utf-8')
                with self.assertRaises(ValueError):
                    rt.run(args, log=quiet)


if __name__ == '__main__':
    unittest.main()
