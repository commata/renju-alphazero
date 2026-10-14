"""H6 fresh-final probes (scripts/make_h6_fresh_probes.py) and their gate (hybrid.h6_gate.fresh_probes)."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hybrid.h6_labels import FRESH_PROBE_SEEDS, RESERVED_SEEDS, canonical_hash, check_reserved_seed, default_probe_files
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

    def test_fresh_files_are_probe_files(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / 'docs' / 'h6').mkdir(parents=True)
            (Path(tmp) / 'docs' / 'h6' / 'h6_fresh_final_probes_v1.json').write_text('{}', encoding='utf-8')
            self.assertTrue(any(p.endswith('h6_fresh_final_probes_v1.json') for p in default_probe_files(Path(tmp))))

    def test_scan_finds_the_p94_kind(self):
        """P94 (every move loses at depth 1) is a parent-level deep loss."""
        self.assertEqual(scan_game((WEB['moves'][:95], 94)), {'loss': 94})

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


@unittest.skipIf(torch is None, 'torch is not installed')
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


if __name__ == '__main__':
    unittest.main()
