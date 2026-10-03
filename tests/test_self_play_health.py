"""Stability gate of scripts/analyze_self_play_health.py on synthetic run directories."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

from analyze_self_play_health import analyze  # noqa: E402

CENTER = 112


def game(length: int, winner: int, opening: int) -> dict:
    moves = [CENTER] + [opening + i for i in range(length - 1)]
    return {'moves': moves, 'winner': winner}


def write_run(root: Path, plan) -> Path:
    """plan: list of (generation, [(length, winner, opening), ...], reuse)."""
    (root / 'self_play').mkdir(parents=True)
    lines = []
    step = 0
    for generation, games, reuse in plan:
        (root / 'self_play' / f'gen{generation:03d}.json').write_text(json.dumps(
            {'generation': generation, 'games': [{'record': game(*g)} for g in games]}))
        fresh = sum(g[0] for g in games)
        step += 50
        lines.append(json.dumps({'type': 'generation', 'generation': generation,
                                 'new_samples': fresh, 'sample_reuse_ratio': reuse,
                                 'samples_drawn': 1600, 'global_step': step,
                                 'train_steps': 50}))
    (root / 'metrics.jsonl').write_text('\n'.join(lines) + '\n')
    return root


class SelfPlayHealthTest(unittest.TestCase):
    def test_healthy_long_games_pass(self):
        plan = [(g, [(20, 1, 10 * (g % 7)), (24, -1, 3 + g % 5)], 6.0) for g in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze(write_run(Path(tmp), plan), 0, 100, 4, None, 6.0)
        self.assertEqual(result['gate']['level'], 'OK')
        self.assertEqual(result['windows'][0]['short_share'], 0.0)

    def test_two_windows_of_short_games_stop(self):
        plan = ([(g, [(20, 1, g), (22, -1, g + 1)], 6.0) for g in range(4)]
                + [(g, [(9, 1, 0), (10, -1, 0)], 8.5) for g in range(4, 12)])
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze(write_run(Path(tmp), plan), 0, 100, 4, 0, None)
        last = result['windows'][-1]
        self.assertEqual(last['short_share'], 1.0)
        self.assertEqual(last['ply9_share'], 0.5)
        self.assertEqual(last['one_sided_share'], 0.0)     # one win each per generation
        self.assertEqual(result['gate']['level'], 'STOP')
        self.assertTrue(any('short games' in f for f in result['gate']['flags']))
        self.assertTrue(any('reuse' in f for f in result['gate']['flags']))

    def test_one_sided_generations(self):
        plan = [(g, [(20, 1 if g % 2 else -1, g), (22, 1 if g % 2 else -1, g + 1)], 6.0)
                for g in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze(write_run(Path(tmp), plan), 0, 100, 4, None, 6.0)
        self.assertEqual(result['windows'][0]['one_sided_share'], 1.0)
        self.assertEqual(result['windows'][0]['black_share'], 0.5)   # balanced on average

    def test_collapsed_openings_warn(self):
        plan = ([(g, [(20, 1, 10 * g), (20, -1, 7 * g + 1)], 6.0) for g in range(4)]
                + [(g, [(20, 1, 0), (20, -1, 0)], 6.0) for g in range(4, 8)])
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze(write_run(Path(tmp), plan), 0, 100, 4, 0, 6.0)
        self.assertEqual(result['windows'][-1]['prefix6_distinct'], 1)
        self.assertEqual(result['gate']['level'], 'WARN')
        self.assertTrue(any('entropy' in f for f in result['gate']['flags']))


if __name__ == '__main__':
    unittest.main()


class SegmentGateTest(unittest.TestCase):
    def test_decisions(self):
        from segment_gate import decide

        h2h = {'summary': {'a_score': 0.62, 'p_two_sided': 0.02, 'p_pairs_two_sided': 0.03,
                           'a_pair_wins': 20, 'a_pair_losses': 8}, 'anchor': {'label': 'C'}}
        self.assertEqual(decide('OK', h2h)[0], 'PROMOTE')
        self.assertEqual(decide('WARN', h2h)[0], 'PROMOTE')
        self.assertEqual(decide('STOP', h2h)[0], 'STOP')
        weak = {'summary': {'a_score': 0.62, 'p_two_sided': 0.02, 'p_pairs_two_sided': 0.2}}
        self.assertEqual(decide('OK', weak)[0], 'HOLD')     # pair-level p decides
        self.assertEqual(decide('OK', None)[0], 'HOLD')

    def test_cli_exit_code_on_a_collapsed_segment(self):
        import subprocess

        plan = ([(g, [(20, 1, g), (22, -1, g + 1)], 6.0) for g in range(4)]
                + [(g, [(9, 1, 0), (10, -1, 0)], 8.5) for g in range(4, 12)])
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(Path(tmp), plan)
            done = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'segment_gate.py'),
                                   str(run), '--from', '8', '--to', '12'],
                                  capture_output=True, text=True)
        self.assertEqual(done.returncode, 12, done.stdout + done.stderr)
        self.assertIn('decision: STOP', done.stdout)
