"""The H7-E4-v2 candidate: identity, and the cooldown of the real wrapper against the DEV replay."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from analysis import e4v2_attack as v2
from analysis import tb_v2_value_e4v2 as cand
from renju import Game
from scripts import h7_e4v2_dev as dev2
from tests.test_h7_e4 import VCF

ROOT = Path(__file__).resolve().parents[1]
PLIES = (5, 7, 9, 11)  # four own moves of White along one line (VCF prefixes, White to move)


class IdentityTest(unittest.TestCase):
    def test_frozen_files_and_dev_selection(self):
        from analysis.tb_v2_value import config_hash
        for name, digest in {**cand.ATTACK_SOURCES, cand.MANIFEST: cand.MANIFEST_SHA256}.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(cand.BASE_ENGINE_HASH, config_hash())
        cand.check_sources()
        dev = json.loads((ROOT / cand.SELECTION['dev_result']).read_text(encoding='utf-8'))
        selected = dev['selection']['selected']
        self.assertEqual({k: cand.ATTACK[k] for k in selected}, selected)  # 150k/cd/killer, literally
        measured = dev['fingerprint']['code']  # the files the DEV run measured
        for name, digest in cand.ATTACK_SOURCES.items():
            if name in measured:
                self.assertEqual(measured[name], digest, name)
        self.assertEqual(measured[cand.MANIFEST], cand.MANIFEST_SHA256)
        self.assertNotEqual(cand.candidate_hash(), cand.BASE_ENGINE_HASH)
        self.assertNotEqual(cand.runtime_hash('cuda'), cand.runtime_hash('cpu'))

    def test_benchmark_arm_is_canonical_only(self):
        from scripts.run_mcts_v8_benchmark import is_canonical, parse_opponent, v8_config
        self.assertTrue(is_canonical('puct_value_vct2_50k_tt_e4v2', 'fast4',
                                     v8_config('puct_value_vct2_50k_tt_e4v2', {'puct_c': 1.5})))
        self.assertFalse(is_canonical('puct_value_vct2_50k_tt_e4v2', 'off',
                                      v8_config('puct_value_vct2_50k_tt_e4v2', {'puct_c': 1.5})))
        with self.assertRaises(ValueError):
            parse_opponent('v8:puct_value_vct2_50k_tt_e4v2')


class ScriptedBase:
    """A base agent whose route per ply is given; it always plays its first legal move."""
    name = 'scripted'

    def __init__(self, routes):
        self.routes = routes
        self.diagnostics = SimpleNamespace(v8_route='', v8_changed=False)

    def select_move(self, game):
        self.diagnostics.v8_route = self.routes[len(game.history)]
        return game.legal_moves()[0]


def scripted(outcomes):
    """Patch the gate and the attack: ``outcomes[ply]`` is 'exhausted', 'refuted', 'win' or 'gate'."""
    def gate(game):
        return {'status': 'WIN' if outcomes.get(len(game.history)) == 'gate' else 'NONE', 'nodes': 1, 'calls': 1}

    def attack(game, budget, killer):
        kind = outcomes.get(len(game.history), 'refuted')
        move = list(game.legal_moves()[-1]) if kind == 'win' else None
        return (tuple(move) if move else None), {'move': move, 'exhausted': kind == 'exhausted', 'nodes': 1,
                                                 'calls': 1, 'candidates': 1, 'vcf_threat_candidates': 0,
                                                 'seconds': 0.0}
    return mock.patch.multiple(v2, opponent_vcf=gate, attack=attack)


def positions(plies=PLIES, moves=VCF):
    out = []
    for n in plies:
        game = Game()
        for m in moves[:n]:
            game.play(*m)
        out.append(game)
    return out


def run(agent, games):
    out = []
    for game in games:
        agent.select_move(game)
        info = agent.attack_info
        out.append('ran' if info.get('ran') else info.get('skip', 'off'))
    return out


class CooldownIntegrationTest(unittest.TestCase):
    def agent(self, routes, **config):
        return cand.E4v2AttackAgent(ScriptedBase(routes), config={**cand.ATTACK, **config})

    def test_one_skip_after_an_exhausted_attack_then_attacks_again(self):
        routes = {p: 'tree' for p in PLIES}
        with scripted({5: 'exhausted'}):
            self.assertEqual(run(self.agent(routes), positions()), ['ran', 'cooldown', 'ran', 'ran'])

    def test_a_forced_move_consumes_it(self):
        routes = {5: 'tree', 7: 'own_vcf', 9: 'tree', 11: 'tree'}
        with scripted({5: 'exhausted'}):
            self.assertEqual(run(self.agent(routes), positions()), ['ran', 'route', 'ran', 'ran'])

    def test_a_gated_or_finished_attack_sets_nothing(self):
        routes = {p: 'tree' for p in PLIES}
        with scripted({5: 'gate', 7: 'refuted'}):
            self.assertEqual(run(self.agent(routes), positions()), ['opponent_vcf', 'ran', 'ran', 'ran'])

    def test_new_game_and_no_leak_between_games(self):
        routes = {p: 'tree' for p in PLIES}
        with scripted({5: 'exhausted'}):
            a = self.agent(routes)
            run(a, positions(plies=(5,)))
            a.new_game()
            self.assertEqual(run(a, positions(plies=(7,))), ['ran'])  # reset by new_game
            b, c = self.agent(routes), self.agent(routes)
            run(b, positions(plies=(5,)))  # b exhausted
            self.assertEqual(run(c, positions(plies=(7,))), ['ran'])  # c has its own bit
            other = [(7, 7), (0, 0), (7, 8), (0, 2), (6, 6), (0, 4), (6, 7)]  # another game, also 7 plies
            self.assertEqual(run(b, positions(plies=(7,), moves=other)), ['ran'])  # not the next move of b's game

    def test_wrapper_equals_the_dev_replay(self):
        routes = {5: 'tree', 7: 'tree', 9: 'stage2', 11: 'tree'}
        outcomes = {5: 'exhausted', 9: 'exhausted', 11: 'win'}
        seq = [{'id': str(p), 'moves': VCF[:p], 'eligible': routes[p] in v2.ROUTES} for p in PLIES]
        with scripted(outcomes):
            rows = {}
            for item, game in zip(seq, positions()):
                gate = v2.opponent_vcf(game)
                attacks = {}
                for budget in v2.BUDGETS:
                    for killer in (False, True):
                        _, info = v2.attack(game, budget, killer)
                        attacks[f"{budget}/{'killer' if killer else 'plain'}"] = info
                rows[item['id']] = {'gate': gate, 'attacks': attacks}
            replay = dev2.replay(cand.ATTACK, seq, rows)
            live = run(self.agent(routes), positions())
        self.assertEqual(live, [r['skip'] if not r['ran'] else 'ran' for r in replay.values()])
        self.assertEqual(live, ['ran', 'cooldown', 'route', 'ran'])

    def test_cooldown_off_is_stateless(self):
        routes = {p: 'tree' for p in PLIES}
        with scripted({5: 'exhausted', 7: 'exhausted', 9: 'exhausted'}):
            self.assertEqual(run(self.agent(routes, cooldown=False), positions()), ['ran'] * 4)

    def test_attack_off_plays_the_base_move(self):
        routes = {p: 'tree' for p in PLIES}
        with scripted({p: 'win' for p in PLIES}):
            base = ScriptedBase(routes)
            agent = cand.E4v2AttackAgent(base, attack=False)
            for game in positions():
                self.assertEqual(agent.select_move(game), game.legal_moves()[0])
                self.assertEqual(agent.attack_info, {})

    def test_a_win_replaces_the_move_and_marks_the_route(self):
        routes = {p: 'tree' for p in PLIES}
        with scripted({5: 'win'}):
            agent = self.agent(routes)
            game = positions(plies=(5,))[0]
            move = agent.select_move(game)
        self.assertEqual(move, game.legal_moves()[-1])
        self.assertEqual((agent.diagnostics.v8_route, agent.diagnostics.v8_changed), ('own_vct2', True))

    def test_only_the_cooldown_is_kept(self):
        agent = self.agent({p: 'tree' for p in PLIES})
        self.assertEqual(set(vars(agent)), {'base', 'attack', 'config', 'name', 'cooldown', 'attack_info'})


if __name__ == '__main__':
    unittest.main()


class ReferenceInputTest(unittest.TestCase):
    def test_the_candidate_dev_wins(self):
        from scripts import h7_e4_reference as ref
        from scripts import h7_e4_web_corpus as web
        load = (lambda p: json.loads((ROOT / p).read_text(encoding='utf-8')))
        dev = load('docs/h7/h7_e4v2_dev.json')
        wins = ref.candidate_wins_v2(web.load(ROOT / 'docs/h7/h7_e4_web_corpus.json'), dev,
                                     load('docs/h7/h7_e4_baseline_v2.json'), dev['selection']['selected'])
        self.assertEqual(len(wins), 21)
        self.assertIn('c94bc588390b49e3a778e543b82e3421:16', wins)  # the one the frozen 2M reference misses
