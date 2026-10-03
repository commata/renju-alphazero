"""scripts/run_champion_loop.py: resumable state, promotion and stop rules (no training)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

from run_champion_loop import build_parser, gate_path, loop, replay_state  # noqa: E402


class FakeRunner:
    """Stands in for the training / gate children; decisions by segment end."""

    def __init__(self, args, decisions, fail_training_at=None):
        self.args, self.decisions, self.fail_at = args, decisions, fail_training_at
        self.trained, self.anchors = [], []

    def __call__(self, command, log):
        script = Path(command[1]).name
        if script == 'run_stage8_training.py':
            end = int(command[command.index('--target-generation') + 1])
            self.anchors.append(command[command.index('--h2h-anchor') + 1].split('=')[0])
            if end == self.fail_at:
                return 1
            self.trained.append(end)
            ckpt = self.args.run_dir / 'checkpoints' / f'checkpoint_gen{end:03d}.pt'
            ckpt.write_text(f'gen {end}')
            return 0
        end = int(command[command.index('--to') + 1])
        decision = self.decisions[end]
        out = Path(command[command.index('--output') + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'decision': decision}))
        return {'PROMOTE': 10, 'HOLD': 11, 'STOP': 12}[decision]


def make_args(tmp: Path, *extra):
    run = tmp / 'stage8_ada_c1120'
    (run / 'checkpoints').mkdir(parents=True)
    start = run / 'checkpoints' / 'checkpoint_gen100.pt'
    start.write_text('gen 100')
    return build_parser().parse_args([
        '--run-dir', str(run), '--config', 'c.yaml', '--start', '100', '--end', '300',
        '--champion', f'ADA100={start}', '--anchors-dir', str(tmp / 'anchors'),
        '--gates-dir', str(tmp / 'gates'), *extra])


class ChampionLoopTest(unittest.TestCase):
    def test_promotion_changes_the_next_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            runner = FakeRunner(args, {140: 'HOLD', 180: 'PROMOTE', 220: 'HOLD',
                                       260: 'PROMOTE', 300: 'HOLD'})
            self.assertEqual(loop(args, runner), 0)
            self.assertEqual(runner.anchors, ['ADA100', 'ADA100', 'ADA180', 'ADA180', 'ADA260'])
            self.assertTrue((Path(tmp) / 'anchors' / 'ADA260.pt').is_file())

    def test_resume_after_a_crash_rebuilds_the_champion(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            decisions = {140: 'PROMOTE', 180: 'HOLD', 220: 'HOLD', 260: 'PROMOTE', 300: 'HOLD'}
            first = FakeRunner(args, decisions, fail_training_at=220)
            self.assertEqual(loop(args, first), 1)
            (Path(tmp) / 'anchors' / 'ADA140.pt').unlink()   # crash before the copy
            state = replay_state(args)
            self.assertEqual((state['champion'][0], state['holds'], state['next']),
                             ('ADA140', 1, 220))
            self.assertTrue((Path(tmp) / 'anchors' / 'ADA140.pt').is_file())  # redone
            second = FakeRunner(args, decisions)
            self.assertEqual(loop(args, second), 0)
            self.assertEqual(second.trained, [220, 260, 300])
            self.assertEqual(second.anchors, ['ADA140', 'ADA140', 'ADA260'])

    def test_plateau_and_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            runner = FakeRunner(args, {140: 'HOLD', 180: 'HOLD', 220: 'HOLD'})
            self.assertEqual(loop(args, runner), 13)
            self.assertEqual(loop(args, runner), 13)         # resumed: still a plateau
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            runner = FakeRunner(args, {140: 'PROMOTE', 180: 'STOP'})
            self.assertEqual(loop(args, runner), 12)
            self.assertEqual(loop(args, runner), 12)
            self.assertEqual(runner.trained, [140, 180])

    def test_fixed_anchor_records_promotions_but_keeps_the_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp), '--fixed-anchor', '--prefix', 'LR')
            runner = FakeRunner(args, {140: 'HOLD', 180: 'HOLD', 220: 'HOLD',
                                       260: 'PROMOTE', 300: 'HOLD'})
            self.assertEqual(loop(args, runner), 0)          # no plateau stop
            self.assertEqual(set(runner.anchors), {'ADA100'})
            self.assertTrue((Path(tmp) / 'anchors' / 'LR260.pt').is_file())
            self.assertEqual(replay_state(args)['champion'][0], 'ADA100')

    def test_refuses_a_champion_match_against_another_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_args(Path(tmp))
            h2h = args.run_dir / 'external_eval' / 'gen140_h2h.json'
            h2h.parent.mkdir()
            h2h.write_text(json.dumps({'anchor': {'label': 'C1120'}}))
            with self.assertRaises(SystemExit):
                loop(args, FakeRunner(args, {}))
            self.assertEqual(gate_path(args.gates_dir, args.run_dir, 140).name,
                             'stage8_ada_c1120_140.json')


if __name__ == '__main__':
    unittest.main()
