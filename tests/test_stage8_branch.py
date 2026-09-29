"""Stage 8: branching a run at a generation (no stale results) and anchor head-to-head."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from training.config import load_config

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    torch = None


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob('gen*.json')) if directory.is_dir() else []


@unittest.skipIf(torch is None, 'requires torch')
class BranchTest(unittest.TestCase):
    def test_branch_copies_only_the_past_and_resumes(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        from branch_stage8_run import branch
        from run_stage8_training import build_parser, orchestrate

        threads = torch.get_num_threads()
        config = load_config(ROOT / 'configs' / 'stage6_test.yaml')
        config['training'].update(generations=3, keep_every=1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / 'source'
                argv = ['--run-dir', str(source), '--new-run', '--anchor', '0',
                        '--light-every', '1', '--heavy-every', '2', '--light-opponents', 'random',
                        '--light-pairs', '1', '--heavy-opponents', 'random', '--heavy-pairs', '1',
                        '--skip-probes', '--h2h-anchor',
                        f"init={source / 'checkpoints' / 'checkpoint_init.pt'}", '--h2h-pairs', '1']
                args = build_parser().parse_args(argv)
                self.assertEqual(orchestrate(source, config, args, log=lambda m: None), 3)
                self.assertIn('gen002_h2h.json', _names(source / 'external_eval'))
                h2h = json.loads((source / 'external_eval' / 'gen002_h2h.json')
                                 .read_text(encoding='utf-8'))
                self.assertEqual(h2h['summary']['games'], 2)
                self.assertEqual(h2h['anchor']['label'], 'init')

                dest = Path(tmp) / 'branch'
                record = branch(source, 2, dest)
                self.assertEqual(_names(dest / 'external_eval'),
                                 [n for n in _names(source / 'external_eval')
                                  if int(n[3:6]) <= 2])
                self.assertNotIn('gen003_light.json', _names(dest / 'external_eval'))
                self.assertEqual(_names(dest / 'self_play'), ['gen000.json', 'gen001.json'])
                events = [json.loads(line) for line in
                          (dest / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertTrue(events and max(e['generation'] for e in events) < 2)
                self.assertEqual(record['copied_files']['external_eval'],
                                 len(_names(dest / 'external_eval')))
                with self.assertRaises(FileExistsError):
                    branch(source, 2, dest)

                # the branch resumes from its own latest.pt and evaluates gen 3 afresh
                args = build_parser().parse_args(
                    [a if a != str(source) else str(dest) for a in argv if a != '--new-run'])
                self.assertEqual(orchestrate(dest, config, args, log=lambda m: None), 3)
                self.assertIn('gen003_light.json', _names(dest / 'external_eval'))
                generations = [e['generation'] for e in
                               (json.loads(line) for line in (dest / 'metrics.jsonl')
                                .read_text(encoding='utf-8').splitlines())
                               if e['type'] == 'generation']
                self.assertEqual(generations, [0, 1, 2])
                self.assertEqual(
                    torch.load(dest / 'checkpoints' / 'latest.pt', weights_only=True)['generation'], 3)
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
