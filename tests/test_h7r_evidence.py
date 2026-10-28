"""Negative regressions for the seven R1 review findings; no real model evaluation."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from hybrid import h7r_eval as ev, h7r_evidence as evidence
from hybrid.h7r_ledger import bind, bind_contract
from scripts import h7r_eval as cli, h7r_holdout as ho
from tests.h7r_helpers import selection_source, timing_record, tactical_record
from tests.test_h7r_cli import world, fake_play_game, QUIET
from tests.test_h7r_holdout import holdout_contract, holdout_games


@contextmanager
def rewritten(path, value):
    before = path.read_bytes()
    path.write_text(json.dumps(value))
    try:
        yield
    finally:
        path.write_bytes(before)


class EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        cls.w = world(Path(cls.tmp.name))
        root, body = cls.w['root'], cls.w['body']
        selection_source(root, body, 'pending')
        cls.w['contract'].write_text(json.dumps(body))
        _, cls.sha = ev.load_contract(cls.w['contract'], root, check_files=False)
        cls.source = selection_source(root, body, cls.sha)
        cls.tactical = tactical_record(body)
        cls.timing = timing_record(body, cls.sha, root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tactical_binds_models_rule_probes_and_raw_checks(self):
        self.assertEqual(evidence.tactical(self.tactical, self.w['body'])['verdict'], 'PASS')
        for mutate in (
            lambda r: r.update(candidate=self.w['body']['models']['orig']['identity']),
            lambda r: r.update(rule_version='old'),
            lambda r: r['files'].update(fresh_probes='0'*64),
            lambda r: r['candidate_result']['value_probes']['fresh']['rows'][0].update(value=-r['candidate_result']['value_probes']['fresh']['rows'][0]['value']),
            lambda r: r['candidate_result']['e1']['rows'][0].update(runs=[]),
            lambda r: r['candidate_result']['e1']['rows'][0]['runs'][0].update(outcome='FAKE'),
            lambda r: r['candidate_result']['must_block'].update(total=999),
        ):
            record = deepcopy(self.tactical)
            mutate(record)
            with self.assertRaises(ValueError):
                evidence.tactical(record, self.w['body'])
        with self.assertRaises(ValueError):
            evidence.tactical({'verdict': 'PASS'}, self.w['body'])

    def test_committed_cycle_and_nmask_tactical_records_remain_readable(self):
        paths = sorted((cli.ROOT / 'docs/h7c').glob('*/arena/tactical.json'))
        self.assertGreaterEqual(len(paths), 6)
        for path in paths:
            with self.subTest(path=path):
                record = json.loads(path.read_text())
                contract = {'roles': {'candidate': 'candidate', 'champion': 'champion', 'anchor': 'champion'},
                            'models': {'candidate': {'identity': record['candidate']},
                                       'champion': {'identity': record['parent']}},
                            'gates': {'tactical_rule': record['rule_version']}}
                self.assertEqual(evidence.tactical(record, contract)['verdict'], record['judge']['verdict'])

    def test_timing_binds_direction_models_positions_and_recomputes_gate(self):
        self.assertTrue(evidence.timing(self.timing, self.w['body'], self.sha, self.w['root'])['gate']['pass'])
        for mutate in (
            lambda r: r.update(comparison=r['comparison'][::-1]),
            lambda r: r['sides']['cand'].update(identity=self.w['body']['models']['orig']['identity']),
            lambda r: r['sides']['cand'].update(samples=[4., 4.]),
            lambda r: r.update(position_ids=['unrelated-a', 'unrelated-b']),
            lambda r: r.update(warmup=0),
            lambda r: r.update(git={'commit': 'dirty', 'dirty': True}),
            lambda r: r['sides']['cand'].update(samples=[]),
        ):
            record = deepcopy(self.timing)
            mutate(record)
            with self.assertRaises(ValueError):
                evidence.timing(record, self.w['body'], self.sha, self.w['root'])

    def test_selection_uses_conf_source_and_replayed_branches(self):
        result = evidence.selection(self.source, self.w['body'], self.sha, self.w['root'])
        self.assertEqual(result['status'], 'NONINFERIOR')
        args = argparse.Namespace(contract=self.w['contract'], root=self.w['root'],
                                  rows=self.source, evidence=None, allow_dirty=True)
        with self.assertRaisesRegex(ValueError, '--evidence'):
            cli.selection(args)
        body = deepcopy(self.w['body'])
        body['selection']['holdout_contract_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'H-conf'):
            evidence.selection(self.source, body, self.sha, self.w['root'])
        record = json.loads(self.source.read_text())
        record['engine'] = {'registered_engine': False}
        with rewritten(self.source, record), self.assertRaisesRegex(ValueError, 'engine'):
            evidence.selection(self.source, self.w['body'], self.sha, self.w['root'])
        # Even a matching file checksum cannot turn a wrong branch outcome into evidence.
        original = json.loads(self.source.read_text())
        branches = ho.read_ref(original['branches'], self.source.parent)
        lines = [json.loads(x) for x in branches.read_text().splitlines()]
        for mutation in ('score', 'continuation', 'identity_sha256'):
            changed = deepcopy(lines)
            changed[0][mutation] = {'score': 0., 'continuation': [[1, 1]], 'identity_sha256': '0'*64}[mutation]
            old = branches.read_bytes()
            try:
                branches.write_text(''.join(json.dumps(row)+'\n' for row in changed))
                manifest = deepcopy(original)
                manifest['branches'] = ho.file_ref(branches, self.source.parent)
                with rewritten(self.source, manifest), self.assertRaises(ValueError):
                    evidence.selection(self.source, self.w['body'], self.sha, self.w['root'])
            finally:
                branches.write_bytes(old)

    def test_selection_source_cli_reuses_runners_and_resumes_without_replaying(self):
        manifest = json.loads(self.source.read_text())
        original_sources = [json.loads(x) for x in (self.source.parent/'sensitivity.jsonl').read_text().splitlines()]
        by_index = {r['game_index']: r for r in original_sources}
        with TemporaryDirectory(dir=self.w['root']) as temp:
            folder = Path(temp)
            args = argparse.Namespace(contract=self.w['contract'], root=self.w['root'], cache=self.source.parent/'cache',
                                      source_dir=folder, output=folder/'source.json', program=folder/'program.json',
                                      rapfi_dir=folder, workers=1, device='cuda', allow_dirty=True, threads_for_tests=True)
            def sensitivity(tasks):
                return [{**deepcopy(by_index[t['game']['game_index']]), 'identity': t['identity']} for t in tasks]
            def branch(tasks):
                return [{'key': t['key'], 'identity_sha256': t['identity_sha256'], 'order': t['order'],
                         'status': 'VALID', 'score': 1., 'length': 9, 'winner': 'black',
                         'ply': 8, 'move': t['move'], 'continuation': []} for t in tasks]
            with patch.object(cli, '_git', return_value={'commit': 'test', 'dirty': False}), \
                    patch('h7c_labels._under_data_or_runs', return_value=True), \
                    patch('h7c_adjudicate.engine_spec', return_value={'identity': manifest['engine']}), \
                    patch.object(cli, '_sensitivity_chunk', side_effect=sensitivity) as sr, \
                    patch.object(cli, '_branch_chunk', side_effect=branch) as br:
                result = cli.selection_source(args, root=self.w['root'])
                self.assertTrue(sr.called and br.called)
                args.output.write_text(json.dumps(result))
                self.assertEqual(evidence.selection(args.output, self.w['body'], self.sha, self.w['root'])['status'],
                                 'NONINFERIOR')
                sr.reset_mock(); br.reset_mock()
                self.assertEqual(cli.selection_source(args, root=self.w['root']), result)
                self.assertFalse(sr.called or br.called)
                # A torn last append is discarded under the directory lock, then resumed.
                with (folder/'sensitivity.jsonl').open('ab') as stream:
                    stream.write(b'{"identity":')
                self.assertEqual(cli.selection_source(args, root=self.w['root']), result)
                self.assertFalse(sr.called or br.called)

    def test_holdout_requires_full_index_set_and_cache_binding(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            cpath = holdout_contract(root / 'contracts')
            contract, sha = ev.load_contract(cpath, root, check_files=False)
            games = holdout_games(cpath)
            ho.check_holdout_games(games, contract, sha)
            for subset in (games[:-1], [*games[:-1], {**games[-1], 'game_index': 99}]):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    ho.check_holdout_games(subset, contract, sha)
            cache = root/'training-cache'; cache.mkdir()
            (cache/'manifest.json').write_text('{}')
            with self.assertRaises((OSError, ValueError)):
                ho.checked_cache(cache, root)

    def test_model_identity_null_missing_and_malformed_are_rejected(self):
        for identity in (None, {}, {'policy_checkpoint_sha256': 'x'}, {k: 'not-a-hash' for k in ev.MODEL_KEYS}):
            bad = deepcopy(self.w['body'])
            bad['models']['cand']['identity'] = identity
            with self.assertRaisesRegex(ValueError, 'identity'):
                ev.check_contract(bad, 'test', self.w['root'], check_files=False)
        bad = deepcopy(self.w['body'])
        bad['models']['cand']['value'] = 'absent.pt'
        with self.assertRaisesRegex(ValueError, 'missing'):
            ev.check_contract(bad, 'test', self.w['root'], check_files=True)

    def test_arm_replay_and_resume_refuse_changed_seed_opening_and_result(self):
        arm = self.w['body']['arms'][0]['name']
        with TemporaryDirectory(dir=self.w['root']) as temp:
            jsonl = Path(temp)/'games.jsonl'
            args = argparse.Namespace(contract=self.w['contract'], arm=arm, device='cpu', workers=1,
                                      jsonl=jsonl, allow_dirty=True, threads_for_tests=True)
            with patch('h7c_arena.play_game', fake_play_game):
                rec = cli.play(args, log=QUIET, root=self.w['root'])
            for field, value in (('seed', -1), ('moves', []), ('winner', None), ('result', 'draw')):
                corrupted = deepcopy(rec)
                corrupted['games'][0][field] = value
                path = Path(temp)/'bad.json'; path.write_text(json.dumps(corrupted))
                with self.assertRaises(ValueError):
                    cli.load_arm(path, self.w['body'], self.sha, self.w['root'])
            before = jsonl.read_bytes()
            corrupted = deepcopy(rec['games']); corrupted[0]['moves'] = []
            jsonl.write_text(''.join(json.dumps(g)+'\n' for g in corrupted))
            with self.assertRaises(ValueError):
                cli.play(args, log=QUIET, root=self.w['root'])
            self.assertNotEqual(before, jsonl.read_bytes())  # refusal did not repair or overwrite the input

    def test_record_publication_never_overwrites_or_leaves_failed_json(self):
        from hybrid.h7r_ledger import publish_json
        with TemporaryDirectory() as temp:
            path = Path(temp)/'record.json'
            publish_json(path, {'decision': 'KEEP_CHAMPION'})
            with self.assertRaises(FileExistsError):
                publish_json(path, {'decision': 'PROMOTE'})
            self.assertEqual(json.loads(path.read_text()), {'decision': 'KEEP_CHAMPION'})
            with self.assertRaises(ValueError):
                publish_json(Path(temp)/'bad.json', {'score': float('nan')})
            self.assertFalse((Path(temp)/'bad.json').exists())

    def test_ledger_is_independent_of_output_path_and_rejects_new_inputs(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            body = self.w['body']
            bind_contract(root, body, self.sha)
            bind_contract(root, body, self.sha)
            bind(root, 'judgment', self.sha, {'arms': 'same', 'selection': 'same'})
            bind(root, 'judgment', self.sha, {'selection': 'same', 'arms': 'same'})  # audit recomputation
            with self.assertRaisesRegex(ValueError, 'different confirmation inputs'):
                bind(root, 'judgment', self.sha, {'arms': 'changed', 'selection': 'same'})
            with self.assertRaisesRegex(ValueError, 'different confirmation inputs'):
                bind_contract(root, body, 'new-contract-sha')
            with self.assertRaisesRegex(ValueError, 'different confirmation inputs'):
                bind(root, 'seed', str(body['arms'][0]['seed']), {'sha256': 'new'})


if __name__ == '__main__':
    unittest.main()
