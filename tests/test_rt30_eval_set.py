"""RT30 evaluation manifests (scripts/build_rt30_eval_set.py, docs/rt30-eval/)."""
from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from build_rt30_eval_set import DEFAULT_JSONL, build, load_keys  # noqa: E402

EVAL = ROOT / 'docs' / 'rt30-eval'


def recorded_keys():
    keys = {}
    for path in DEFAULT_JSONL:
        with path.open(encoding='utf-8') as handle:
            for line_no, line in enumerate(handle):
                for move in json.loads(line)['v8_moves']:
                    keys[f'{path.stem}/{line_no}/{move["ply"]}'] = move
    return keys


class StratifiedSetTest(unittest.TestCase):
    def test_committed_manifest_is_what_the_builder_draws(self):
        committed = json.loads((EVAL / 'slow-stratified-v1.json').read_text(encoding='utf-8'))
        rebuilt = build(DEFAULT_JSONL, threshold=committed['source_threshold_seconds'],
                        per_file=committed['per_file'], per_game=committed['max_positions_per_game'],
                        seed=committed['sampling_seed'], version=committed['version'])
        self.assertEqual(rebuilt['keys'], committed['keys'])

    def test_constraints(self):
        manifest = json.loads((EVAL / 'slow-stratified-v1.json').read_text(encoding='utf-8'))
        recorded = recorded_keys()
        keys = manifest['keys']
        self.assertEqual(len(keys), len(set(keys)))
        per_file = Counter(k.split('/')[0] for k in keys)
        self.assertEqual(sorted(per_file.values()), [manifest['per_file']] * len(DEFAULT_JSONL))
        per_game = Counter(tuple(k.split('/')[:2]) for k in keys)
        self.assertLessEqual(max(per_game.values()), manifest['max_positions_per_game'])
        for key in keys:
            self.assertGreater(recorded[key]['seconds'], manifest['source_threshold_seconds'], key)

    def test_same_seed_same_keys_other_seed_other_keys(self):
        a, b = build(DEFAULT_JSONL, seed=42), build(DEFAULT_JSONL, seed=42)
        self.assertEqual(a['keys'], b['keys'])
        self.assertNotEqual(a['keys'], build(DEFAULT_JSONL, seed=43)['keys'])


class CriticalSetTest(unittest.TestCase):
    def test_keys_exist_and_cases_match(self):
        manifest = json.loads((EVAL / 'critical-regressions-v1.json').read_text(encoding='utf-8'))
        recorded = recorded_keys()
        self.assertEqual(manifest['keys'], [c['key'] for c in manifest['cases']])
        for key in manifest['keys']:
            self.assertIn(key, recorded)


class LoadKeysTest(unittest.TestCase):
    def test_manifest_and_plain_list(self):
        with TemporaryDirectory() as tmp:
            plain = Path(tmp) / 'keys.json'
            plain.write_text(json.dumps(['a/1/2', 'b/3/4']), encoding='utf-8')
            self.assertEqual(load_keys(plain), (['a/1/2', 'b/3/4'], None))
        keys, version = load_keys(EVAL / 'critical-regressions-v1.json')
        self.assertEqual((len(keys), version), (6, 'critical-regressions-v1'))


if __name__ == '__main__':
    unittest.main()
