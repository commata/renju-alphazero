"""H6 fresh-final probes (scripts/make_h6_fresh_probes.py) and their gate (hybrid.h6_gate.fresh_probes)."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hybrid.h6_labels import (DEV_PROBE_EXTENSION_SEEDS, DEV_PROBE_SEEDS, FRESH_PROBE_SEEDS, RESERVED_SEEDS,
                               canonical_hash, check_reserved_seed, default_probe_files)
from scripts.make_h6_fresh_probes import ENGINE_ARM, load_games, scan_game, select

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None

ROOT = Path(__file__).resolve().parents[1]
WEB = json.loads((ROOT / 'docs' / 'mcts-v8-results' / 'probe_web_v8_loss_20261007.json').read_text(encoding='utf-8'))


class FreshProbeTest(unittest.TestCase):
    def test_seeds_are_reserved(self):
        self.assertEqual(FRESH_PROBE_SEEDS, tuple(range(8601, 8611)))
        self.assertTrue(set(FRESH_PROBE_SEEDS) <= RESERVED_SEEDS)
        with self.assertRaises(ValueError):
            check_reserved_seed(8605)

    def test_dev_seeds_are_reserved_and_disjoint(self):
        self.assertEqual(DEV_PROBE_SEEDS, (8611, 8612, 8613))
        self.assertEqual(DEV_PROBE_EXTENSION_SEEDS, (8614, 8615, 8616))
        self.assertTrue(set(DEV_PROBE_SEEDS + DEV_PROBE_EXTENSION_SEEDS) <= RESERVED_SEEDS)
        self.assertFalse(set(DEV_PROBE_SEEDS + DEV_PROBE_EXTENSION_SEEDS) & set(FRESH_PROBE_SEEDS))
        with self.assertRaises(ValueError):
            check_reserved_seed(8616)

    def test_dev_select_one_per_game_and_sign(self):
        moves = WEB['moves']
        games = [{'key': f'g{i}', 'seed': 8611, 'moves': moves} for i in range(3)]
        found = [{'loss': 90, 'win': 91}, {'loss': 92, 'win': 93}, {'win': 91}]
        probes = select(games, found, exclude=set(), per_sign=True)
        self.assertEqual([(p['game_key'], p['ply']) for p in probes],
                         [('g0', 90), ('g0', 91), ('g1', 92), ('g1', 93)])  # g2's win is D4-equal to g0's
        for sign in (1, -1):
            keys = [p['game_key'] for p in probes if p['expected_sign'] == sign]
            self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(p['name'].startswith('D') for p in probes))

    def test_dev_mode_takes_dev_seeds_only(self):
        from scripts import make_h6_fresh_probes as maker
        with TemporaryDirectory() as tmp:
            games = {'arm': ENGINE_ARM, 'seed': 8601, 'opponent': 'v8:full', 'games': [{'key': 'g1', 'moves': []}]}
            Path(tmp, 'f.json').write_text(json.dumps(games), encoding='utf-8')
            with self.assertRaises(ValueError):
                maker.main(['--dev', '--games', str(Path(tmp, 'f.json')), '--output', str(Path(tmp, 'x.json'))])
            games = {**games, 'seed': 8611, 'games': [{'key': 'g1', 'moves': WEB['moves'][:96]}]}
            Path(tmp, 'd.json').write_text(json.dumps(games), encoding='utf-8')
            self.assertEqual(maker.main(['--dev', '--games', str(Path(tmp, 'd.json')), '--from-ply', '93',
                                         '--output', str(Path(tmp, 'dev.json'))]), 0)
            out = json.loads(Path(tmp, 'dev.json').read_text(encoding='utf-8'))
            self.assertEqual((out['purpose'], out['sealed']), ('dev', False))
            self.assertTrue(out['selection']['one_probe_per_game_and_sign'])

    def test_dev_rebuild_ignores_earlier_dev_sets(self):
        from scripts import make_h6_fresh_probes as maker
        with TemporaryDirectory() as tmp:
            dev = Path(tmp, 'h6_fresh_dev_probes_v1.json')
            dev.write_text(json.dumps({'purpose': 'dev', 'probes': []}), encoding='utf-8')
            final = Path(tmp, 'h6_fresh_final_probes_v1.json')
            final.write_text(json.dumps({'purpose': 'final', 'probes': []}), encoding='utf-8')
            self.assertTrue(maker._is_dev_file(dev))
            self.assertFalse(maker._is_dev_file(final))
            self.assertFalse(maker._is_dev_file(Path(tmp, 'missing.json')))

    def test_dev_verdict_rule(self):
        from scripts.h6_dev_probe_eval import RULE, verdict

        def fresh(win, loss):
            rows = [{'expected_sign': 1, 'correct': i < win[0]} for i in range(win[1])]
            rows += [{'expected_sign': -1, 'correct': i < loss[0]} for i in range(loss[1])]
            out = {'probes': rows}
            for name, (c, n) in (('win', win), ('loss', loss)):
                out[name] = {'probes': n, 'games': n, 'accuracy': c / n if n else None, 'mean_value': 0.0}
            return out

        self.assertEqual(RULE['sign_accuracy_min'], 0.80)
        ok = verdict(fresh((48, 57), (40, 50)))
        self.assertEqual(ok['verdict'], 'PASS')
        self.assertEqual(ok['win']['text'], '48 / 57 = 84.2%')
        self.assertEqual(verdict(fresh((48, 57), (39, 50)))['verdict'], 'FAIL')  # 78% loss
        self.assertEqual(verdict(fresh((19, 19), (40, 50)))['verdict'], 'FAIL')  # too few wins
        self.assertEqual(verdict(fresh((48, 57), (0, 0)))['verdict'], 'FAIL')

    def test_dev_eval_refuses_the_final_probes(self):
        from scripts.h6_dev_probe_eval import check_probe_file
        with self.assertRaises(ValueError):
            check_probe_file(ROOT / 'docs' / 'h6' / 'h6_fresh_final_probes_v1.json')

    def test_fresh_files_are_probe_files(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / 'docs' / 'h6').mkdir(parents=True)
            (Path(tmp) / 'docs' / 'h6' / 'h6_fresh_final_probes_v1.json').write_text('{}', encoding='utf-8')
            self.assertTrue(any(p.endswith('h6_fresh_final_probes_v1.json') for p in default_probe_files(Path(tmp))))

    def test_scan_finds_the_p94_kind(self):
        """P94 (every move loses at depth 1) is a parent-level deep loss."""
        self.assertEqual(scan_game((WEB['moves'][:95], 94))['found'], {'loss': 94})

    def test_select_one_per_game_balanced_and_deduplicated(self):
        moves = WEB['moves']
        games = [{'key': f'g{i}', 'seed': 8601, 'moves': moves} for i in range(4)]
        found = [{'loss': 90, 'win': 91}, {'loss': 90, 'win': 92}, {'loss': 93}, {}]
        probes = select(games, found, exclude=set())
        self.assertEqual(len({p['game_key'] for p in probes}), len(probes))  # at most one per game
        self.assertEqual([p['expected_sign'] for p in probes], [-1, 1, -1])  # balanced, ties to loss
        self.assertEqual(len({p['canonical_hash'] for p in probes}), 3)
        excluded = select(games, found, exclude={canonical_hash(moves[:90])})
        self.assertNotIn(90, [p['ply'] for p in excluded if p['expected_sign'] == -1])

    def test_load_games_checks_arm_and_seed(self):
        with TemporaryDirectory() as tmp:
            good = {'arm': ENGINE_ARM, 'seed': 8601, 'opponent': 'v8:full', 'games': [{'key': 'k', 'moves': []}]}
            Path(tmp, 'a.json').write_text(json.dumps(good), encoding='utf-8')
            games, sources = load_games([Path(tmp, 'a.json')])
            self.assertEqual(games[0]['seed'], 8601)
            Path(tmp, 'b.json').write_text(json.dumps({**good, 'seed': 8417}), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_games([Path(tmp, 'b.json')])
            Path(tmp, 'c.json').write_text(json.dumps({**good, 'arm': 'puct_policy_vct2'}), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_games([Path(tmp, 'c.json')])

    def test_budget_skips_a_position_deterministically(self):
        from scripts import make_h6_fresh_probes as maker
        saved = maker.VCF_CALL_BUDGET
        maker.VCF_CALL_BUDGET = 50  # P94 needs ~550 calls
        try:
            result = scan_game((WEB['moves'][:95], 94))
        finally:
            maker.VCF_CALL_BUDGET = saved
        self.assertEqual(result['found'], {})
        self.assertEqual(result['budget_cuts'], 1)

    def test_node_budget_skips_a_position(self):
        from scripts import make_h6_fresh_probes as maker
        saved = maker.VCF_NODE_BUDGET
        maker.VCF_NODE_BUDGET = 2_000  # P94 needs ~20k nodes
        try:
            result = scan_game((WEB['moves'][:95], 94))
        finally:
            maker.VCF_NODE_BUDGET = saved
        self.assertEqual((result['found'], result['budget_cuts']), ({}, 1))

    def test_resume_skips_scanned_games(self):
        from scripts import make_h6_fresh_probes as maker
        with TemporaryDirectory() as tmp:
            games = {'arm': ENGINE_ARM, 'seed': 8601, 'opponent': 'v8:full',
                     'games': [{'key': 'g1', 'moves': WEB['moves'][:96]}]}
            Path(tmp, 'g.json').write_text(json.dumps(games), encoding='utf-8')
            args = ['--games', str(Path(tmp, 'g.json')), '--from-ply', '93', '--jsonl', str(Path(tmp, 'scan.jsonl'))]
            self.assertEqual(maker.main(args + ['--output', str(Path(tmp, 'a.json'))]), 0)
            calls = []
            original = maker._scan_chunk
            maker._scan_chunk = lambda tasks: calls.append(tasks) or original(tasks)
            try:
                self.assertEqual(maker.main(args + ['--output', str(Path(tmp, 'b.json'))]), 0)
            finally:
                maker._scan_chunk = original
            self.assertEqual(calls, [])  # nothing rescanned
            a = json.loads(Path(tmp, 'a.json').read_text(encoding='utf-8'))
            b = json.loads(Path(tmp, 'b.json').read_text(encoding='utf-8'))
            self.assertEqual(a['counts'], b['counts'])
            self.assertEqual(a['counts']['games_with_loss'], 1)


@unittest.skipIf(torch is None, 'requires torch')
class FreshGateTest(unittest.TestCase):
    def test_fresh_gate_counts_games_and_signs(self):
        from hybrid.h6_gate import FRESH, d4_average, fresh_probes, judge

        def side_value(planes):
            return 0.5 - planes[:, 3, 0, 0]  # + for white to move, - for black to move

        probes = []
        for i in range(FRESH['min_games_per_sign']):
            probes.append({'name': f'W{i}', 'game_key': f'w{i}', 'expected_sign': 1, 'kind': 'vct1_win', 'ply': 93,
                           'moves': WEB['moves'][:93]})  # white to move
            probes.append({'name': f'L{i}', 'game_key': f'l{i}', 'expected_sign': -1, 'kind': 'vct_loss', 'ply': 94,
                           'moves': WEB['moves'][:94]})  # black to move
        with TemporaryDirectory() as tmp:
            path = Path(tmp, 'fresh.json')
            path.write_text(json.dumps({'format': 'h6-fresh-probes-v1', 'probes': probes}), encoding='utf-8')
            ok = fresh_probes(d4_average(side_value, 'cpu'), path, 'cpu')
            self.assertTrue(ok['pass'])
            self.assertEqual(ok['win']['games'], FRESH['min_games_per_sign'])
            bad = fresh_probes(lambda p: -side_value(p), path, 'cpu')
            self.assertFalse(bad['pass'])
            path.write_text(json.dumps({'format': 'h6-fresh-probes-v1', 'probes': probes[:4]}), encoding='utf-8')
            self.assertFalse(fresh_probes(side_value, path, 'cpu')['pass'])  # too few games
        metrics_ok = {'proof': {'balanced_accuracy': 0.95, 'minus_accuracy': 0.9},
                      'outcome': {'mse': 0.9, 'balanced_accuracy': 0.65},
                      'outcome_by_side': {'black': {'balanced_accuracy': 0.64}, 'white': {'balanced_accuracy': 0.65}},
                      'report': {'outcome_ece': 0.07}}
        d4 = {'mean_std': 0.0, 'p95_std': 0.0, 'confident_sign_agreement': 1.0}
        failing_p94 = {'P93': {'pass': True}, 'P94': {'pass': False}}
        self.assertEqual(judge(metrics_ok, d4, failing_p94, ok)['verdict'], 'PASS')  # fresh replaces P93/P94
        self.assertIn('fresh', judge(metrics_ok, d4, failing_p94, bad)['next_steps_if_failed']['G4'])

    def test_part_b_needs_fresh_and_d4_average(self):
        from scripts.h6_offline_gate import main
        with self.assertRaises(SystemExit):
            main(['--checkpoint', 'x.pt', '--cache-dir', 'c', '--part', 'B', '--output', 'o.json'])

    def test_final_gate_refuses_dev_probes(self):
        from scripts.h6_offline_gate import main
        with TemporaryDirectory() as tmp:
            dev = Path(tmp, 'dev.json')
            dev.write_text(json.dumps({'format': 'h6-fresh-probes-v1', 'purpose': 'dev', 'sealed': False,
                                       'probes': []}), encoding='utf-8')
            with self.assertRaises(SystemExit):
                main(['--checkpoint', 'x.pt', '--cache-dir', 'c', '--part', 'B', '--d4-average',
                      '--fresh-probes', str(dev), '--output', str(Path(tmp, 'o.json'))])


@unittest.skipIf(torch is None, 'requires torch')
class DevEvalTest(unittest.TestCase):
    def test_dev_eval_end_to_end(self):
        from model.checkpoint import save_checkpoint
        from scripts.h6_dev_probe_eval import main
        from tests.test_h6_cross_eval import MODEL_CONFIG
        from tests.test_h6_value import h3_model
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'v3').mkdir()
            save_checkpoint(root / 'v3' / 'best.pt', h3_model(seed=12))
            (root / 'v3' / 'best.json').write_text(json.dumps({'stage': 'H6-2a', 'labels_sha256': 'x',
                                                               'model_config': MODEL_CONFIG}), encoding='utf-8')
            probes = [{'name': 'D01', 'game_key': 'a', 'seed': 8611, 'expected_sign': 1, 'kind': 'vct1_win',
                       'ply': 93, 'moves': WEB['moves'][:93]},
                      {'name': 'D02', 'game_key': 'a', 'seed': 8611, 'expected_sign': -1, 'kind': 'vct_loss',
                       'ply': 94, 'moves': WEB['moves'][:94]}]
            dev = root / 'dev.json'
            dev.write_text(json.dumps({'format': 'h6-fresh-probes-v1', 'purpose': 'dev', 'sealed': False,
                                       'probes': probes}), encoding='utf-8')
            args = ['--model', f"v3={root / 'v3' / 'best.pt'}", '--probes', str(dev), '--allow-dirty']
            with self.assertRaises(SystemExit):  # not the frozen v3 checkpoint
                main(args + ['--output', str(root / 'a.json')])
            self.assertEqual(main(args + ['--allow-unfrozen', '--output', str(root / 'b.json')]), 0)
            out = json.loads((root / 'b.json').read_text(encoding='utf-8'))
            v3 = out['models']['v3']
            self.assertEqual((v3['win']['probes'], v3['loss']['probes']), (1, 1))
            self.assertEqual(out['decision'], 'FAIL')  # 1 probe per sign is below the minimum
            final = ROOT / 'docs' / 'h6' / 'h6_fresh_final_probes_v1.json'
            with self.assertRaises(SystemExit):
                main(['--model', f"v3={root / 'v3' / 'best.pt'}", '--probes', str(final), '--allow-dirty',
                      '--allow-unfrozen', '--output', str(root / 'c.json')])


if __name__ == '__main__':
    unittest.main()
