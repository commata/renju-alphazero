"""H7-E2: TB-Engine-v1-T (``root_vcf_table``), the offline soundness recheck and the §2.3 gate summary."""
import json
from pathlib import Path
from random import Random
import unittest

from analysis import mcts_v8
from analysis.mcts_v8 import _BudgetedSolver
from analysis.tb_engine_v1 import TB_ENGINE_V1
from analysis.tb_engine_v1 import config_hash as tb1_hash
from analysis.tb_engine_v1_t import ENGINE_OPTIONS, NAME, TB_ENGINE_V1_T, config_hash, make_agent
from renju import Game
from scripts import h6_5d
from scripts import h7_e2_soundness as snd
from scripts import summarize_h7_e2 as e2
from scripts.run_mcts_v8_benchmark import parse_opponent, v8_config
from tests.tb_engine_v1_regression import FIXTURE, SEED, WEB, hashed_policy

MOVES = json.loads(WEB.read_text(encoding='utf-8'))['moves']


def game_at(ply):
    game = Game()
    for move in MOVES[:ply]:
        game.play(*move)
    return game


class EngineTest(unittest.TestCase):
    def test_configuration(self):
        self.assertEqual(TB_ENGINE_V1_T, TB_ENGINE_V1)
        self.assertEqual(ENGINE_OPTIONS, {'root_vcf_table': True})
        self.assertEqual(v8_config('puct_policy_vct2_50k_tt', {'puct_c': 1.5}), {**TB_ENGINE_V1, 'root_vcf_table': True})
        self.assertNotEqual(config_hash(), tb1_hash())
        agent = make_agent(hashed_policy, seed=3)
        self.assertEqual((agent.name, agent.root_vcf_table), (NAME, True))
        with self.assertRaises(ValueError):
            parse_opponent('v8:puct_policy_vct2_50k_tt')
        with self.assertRaises(ValueError):
            mcts_v8.mcts_search_v8(game_at(8), root_vcf_table=1)

    def test_same_decisions_as_the_frozen_fixture(self):
        """Every root check of the fixture positions finishes inside the frozen budget, so the table changes
        nothing there but the node count."""
        for want in json.loads(FIXTURE.read_text(encoding='utf-8')):
            agent = make_agent(hashed_policy, seed=SEED)
            move = agent.select_move(game_at(want['ply']))
            d = agent.diagnostics
            self.assertEqual(list(move), want['move'])
            self.assertEqual([[list(m), s] for m, s in d.v8_root_checked], want['root_checked'])
            self.assertTrue(d.v8_root_vcf_table)
            self.assertGreater(d.v8_root_table_stores, 0)

    def test_off_never_touches_the_table(self):
        from analysis.tb_engine_v1 import make_agent as tb1_agent
        agent = tb1_agent(hashed_policy, seed=SEED)
        agent.select_move(game_at(20))
        self.assertFalse(agent.diagnostics.v8_root_vcf_table)
        self.assertEqual(agent.diagnostics.v8_root_table_stores, 0)

    def test_solver_uses_its_vcf_search(self):
        calls = []

        def find(game, attacker, *, max_fours, node_limit):
            calls.append(attacker)
            return mcts_v8._find_vcf_with_stats(game, attacker, max_fours=max_fours, node_limit=node_limit)

        solver = _BudgetedSolver(node_limit=20_000, call_limit=10_000, node_budget=400_000)
        solver.vcf_search = find
        solver.status_after(game_at(94), tuple(MOVES[94]), 200_000, 5_000)
        self.assertEqual(len(calls), solver.vcf_calls)


class SoundnessTest(unittest.TestCase):
    def task(self, ply, recorded):
        return {'id': f'x|{ply}', 'seed': 8421, 'moves': MOVES[:ply], 'tree_move': MOVES[ply], 'recorded_status': recorded}

    def test_unsafe_is_replayed(self):
        row = snd.sound_task(self.task(94, 'UNSAFE'))  # frozen and table both prove the web loss
        self.assertEqual((row['frozen_standard'], row['replay'], row['contradiction']),
                         ('UNSAFE', {'status': 'UNSAFE', 'line_fail': 0}, False))

    def test_contradiction_and_summary(self):
        bad = snd.sound_task(self.task(94, 'SAFE'))  # a SAFE the frozen solver refutes
        self.assertTrue(bad['contradiction'])
        good = snd.sound_task(self.task(40, 'SAFE'))
        s = snd.summarize([bad, good])
        self.assertFalse(s['sound'])
        self.assertEqual(s['transitions'], {'SAFE->SAFE': 1, 'UNSAFE->SAFE': 1})
        self.assertTrue(snd.summarize([good])['sound'])
        self.assertFalse(snd.opposite('UNKNOWN', 'SAFE'))


def tree_move(ply, status, seconds=1.0):
    move = [ply % 15, ply // 15]
    return {'ply': ply, 'seconds': seconds, 'route': 'tree', 'v7_move': move, 'played': move, 'changed': False,
            'root': {'checked': [[move, status]], 'seconds': seconds / 2}}


def run(arm, seed, games, table):
    config = {**TB_ENGINE_V1, 'root_vcf_table': True} if arm == e2.CANDIDATE else dict(TB_ENGINE_V1)
    out = []
    for (pair, colour), (result, statuses, seconds) in games.items():
        moves = [tree_move(10 + i, s, seconds) for i, s in enumerate(statuses)]
        if table:
            for m in moves:
                m['root']['table'] = {'hits': 1, 'stores': 1}
        out.append({'key': f'{arm}@v8:full/{seed}/{pair}/{colour}', 'pair': pair, 'v8_color': colour, 'result': result,
                    'moves': [[0, 0]] * len(statuses), 'v8_moves': moves, 'opponent_move_seconds': [1.0],
                    'game_seconds': 1.0})
    return {'arm': arm, 'seed': seed, 'opponent': 'v8:full', 'v8_config': config, 'fast': {'requested': 'fast3'},
            'games': out}


def sound_file(seeds, runs, contradictions=()):
    rows = [{} for r in runs if r['arm'] == e2.CANDIDATE for g in r['games'] for m in g['v8_moves']]
    return {'fast': {'requested': 'fast3'}, 'rows': rows,
            'summary': {'seeds': seeds, 'contradictions': list(contradictions), 'unsafe_not_replayed': [],
                        'line_fail': 0, 'unsafe': 0, 'transitions': {}, 'legacy_4m': {}}}


class SummaryTest(unittest.TestCase):
    def build(self, seeds, cand_unknown=1, base_unknown=3, cand_loss_every=0, seconds=1.0, pairs=40, contra=()):
        rng = Random(1)
        runs = []
        for seed in seeds:
            cand, base = {}, {}
            for p in range(pairs):
                for colour in ('black', 'white'):
                    base_statuses = ['UNKNOWN'] * base_unknown + ['SAFE'] * (20 - base_unknown)
                    k = cand_unknown if rng.random() < 0.8 else base_unknown
                    cand_statuses = ['UNKNOWN'] * k + ['SAFE'] * (20 - k)
                    lost = cand_loss_every and p % cand_loss_every == 0 and colour == 'black'
                    cand[(p, colour)] = ('loss' if lost else 'win', cand_statuses, seconds)
                    base[(p, colour)] = ('win', base_statuses, 1.0)
            runs += [run(e2.CANDIDATE, seed, cand, True), run(e2.BASELINE, seed, base, False)]
        return e2.evaluate(runs, [sound_file(list(seeds), runs, contra)])

    def test_stages(self):
        self.assertEqual(self.build([8421])['verdict']['decision'], 'RUN_8422')
        self.assertEqual(self.build([8421], contra=['x'])['verdict']['decision'], 'REJECT')
        self.assertEqual(self.build([8421, 8422])['verdict']['decision'], 'ADOPT')
        self.assertEqual(self.build([8421, 8422], cand_unknown=3)['verdict']['decision'], 'REJECT')  # no decrease
        self.assertEqual(self.build([8421, 8422], seconds=2.0)['verdict']['decision'], 'REJECT')  # cost 2.0
        self.assertEqual(self.build([8421, 8422], cand_loss_every=4)['verdict']['decision'], 'REJECT')  # -0.125
        self.assertEqual(self.build([8421, 8422], cand_loss_every=15)['verdict']['decision'], 'RUN_8423')
        self.assertEqual(self.build([8421, 8422, 8423], cand_loss_every=15)['verdict']['decision'], 'REJECT')
        self.assertEqual(self.build([8422])['verdict']['decision'], 'INCOMPLETE')

    def test_gates_and_report(self):
        out = self.build([8421, 8422])
        f = out['gates']['frequency']
        self.assertLess(f['ci95'][1], 0)
        self.assertEqual(set(f['per_seed']), {'8421', '8422'})
        self.assertEqual(out['gates']['strength']['diff'], 0.0)
        self.assertEqual(out['gates']['cost']['ratio'], 1.0)
        self.assertEqual(set(out['report'][e2.CANDIDATE]['move_seconds']), {'moves', 'p50', 'p90', 'p95', 'p99', 'max'})

    def test_configuration_is_checked(self):
        runs = [run(e2.CANDIDATE, 8421, {(0, 'black'): ('win', ['SAFE'], 1.0)}, False),
                run(e2.BASELINE, 8421, {(0, 'black'): ('win', ['SAFE'], 1.0)}, False)]
        with self.assertRaises(ValueError):  # the candidate ran without the table
            e2.evaluate(runs, [sound_file([8421], runs)])


if __name__ == '__main__':
    unittest.main()
