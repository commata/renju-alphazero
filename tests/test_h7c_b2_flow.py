"""Synthetic stage-process integration: real command routing, receipts, lineage, summaries and exports.

The subprocess boundary substitutes tiny files for expensive games/training. These tests are infrastructure
evidence only; they do not claim learned-model, CUDA, Rapfi or full-cycle strength validation.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from analysis import tb_v3, tb_v3_net
from hybrid import h7c_b2 as b2
from scripts import h7c_arena as ar, h7c_loop as loop, h7c_train_value as tv
from tests.test_h7c_arena import judge_record

ROOT = Path(__file__).resolve().parents[1]


class FlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.code = {'commit': 'fixed-code', 'dirty': False}
        self.mode, self.alarm, self.commands = 'CANDIDATE', False, []
        self.policy = self.root / 'runs/h3_policy_64x4/best.pt'
        self.frozen = self.root / 'runs/h6_value_head_v3/best.pt'
        for p, content in ((self.policy, b'policy'), (self.frozen, b'frozen-value')):
            p.parent.mkdir(parents=True)
            p.write_bytes(content)
            loop.atomic(p.with_suffix('.json'), {'fixture': True})
        self.reference = tb_v3_net.model_identity(self.policy, self.frozen)
        start = self.root / 'runs/h7c/c3/value/best.pt'
        start.parent.mkdir(parents=True)
        start.write_bytes(b'c3-value')
        loop.atomic(start.with_suffix('.json'), {'fixture': 'c3'})
        self.initial = tb_v3_net.model_identity(self.policy, start)
        for c in (1, 2, 3):
            loop.atomic(self.root / f'docs/h7c/c{c}/arena_verdict.json', {
                'format': 'h7c-arena-v1-verdict', 'cycle': c, 'parent': self.reference,
                'next_champion': self.initial if c == 3 else self.reference,
                'promotion': {'improvement_lower': -.1}, 'tactical': {'verdict': 'PASS'}})
        config = self.root / 'configs/h7c/c1_value.json'
        config.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / 'configs/h7c/c1_value.json', config)
        self.p = b2.read(ROOT / 'docs/h7c/b2_program.json')
        self.p.update(frozen_models=self.reference)
        self.p['start'].update(models=self.initial, verdict_sha256=loop.digest(self.root / self.p['start']['verdict']))
        self.args = argparse.Namespace(program=self.root / 'docs/h7c/b2_program.json', device='cuda',
                                       rapfi_dir=self.root / 'rapfi')
        self.args.rapfi_dir.mkdir()
        self.p['external'] = {'executable': 'engine', 'files': {'config': 'unused-in-process-fixture'},
                              'info': ['max_node=300'], 'executable_sha256': 'unused-in-process-fixture'}
        for f in ('engine', 'config'):
            (self.args.rapfi_dir / f).write_text(f)
        loop.atomic(self.args.program, self.p)
        self.psha = loop.digest(self.args.program)
        replay = self.root / 'data/tactical/h6/value_cache_v3'
        replay.mkdir(parents=True)
        for name in ('manifest.json', 'train.pt', 'val.pt', 'test.pt'):
            (replay / name).write_text('replay')
        for patcher in (mock.patch.object(b2, 'ROOT', self.root), mock.patch.object(tb_v3, 'MODELS', self.reference),
                        mock.patch.object(loop, 'git_identity', return_value=self.code),
                        mock.patch.object(tv, 'load_cache_splits', side_effect=lambda cache, *a:
                                          ({}, b2.read(cache / 'manifest.json')))):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.store = loop.Stages(self.root, self.root / 'runs/h7c/b2', {'git': self.code}, self.execute)

    def execute(self, argv, log, root, running):
        self.commands.append(argv)
        script = Path(argv[2]).name
        def arg(key):
            return argv[argv.index('--' + key) + 1]
        def path(key):
            return Path(arg(key))
        cycle = int(arg('cycle')) if '--cycle' in argv else None
        dp = path('recipe-decision') if '--recipe-decision' in argv else None
        d = b2.read(dp) if dp else None
        if script == 'h7c_selfplay.py':
            path('jsonl').parent.mkdir(parents=True, exist_ok=True)
            path('jsonl').write_text('synthetic-selfplay')
            loop.atomic(path('output'), {'identity': {'models': self.initial if cycle == 4 else
                b2.read(self.root / f'docs/h7c/c{cycle-1}/arena_verdict.json')['next_champion'], 'cycle': cycle,
                'run_seed': 8800+cycle}, 'games': 1000, 'summary': {'games': 1000}, 'git': self.code})
        elif script == 'h7c_labels.py':
            cycle = int(path('games').parent.name[1:])
            parent = b2.read(self.root / f'docs/h7c/c{cycle}/decision.json')['parent']
            row = {'game_id': f'h7c:c{cycle}:{8800+cycle}:0', 'split': 'train', 'value_mask': False,
                'format': 'h6-value-row-v1', 'value_perspective': 'side_to_move', 'h7c_provenance': {
                    **parent, 'runtime_hash': 'f'*64, 'selfplay_git_commit': self.code['commit']}}
            with gzip.open(path('value-output'), 'wt', encoding='utf-8') as out:
                out.write(json.dumps(row)+'\n')
            path('policy-output').write_bytes(b'policy-rows')
            loop.atomic(path('report'), {'summary': {'identity': {'cycle': cycle}, 'games_in': 1000},
                                        'git_commit': self.code['commit'], 'git_dirty': False})
        elif script == 'build_h6_value_cache.py':
            cache = path('out-dir')
            cache.mkdir()
            row = json.loads(gzip.decompress(path('labels').read_bytes()))
            loop.atomic(cache / 'manifest.json', {'labels_sha256': hashlib.sha256(
                gzip.decompress(path('labels').read_bytes())).hexdigest()})
            for split in ('train', 'val', 'test'):
                (cache / f'{split}.pt').write_bytes(b'tiny-cache')
                loop.atomic(cache / f'{split}_games.json', [row['game_id']] if split == 'train' else [])
        elif script == 'h7c_train_value.py':
            out = path('out-dir')
            out.mkdir(parents=True)
            source = {'models': d['parent'], 'runtime_hash': 'f'*64,
                      'selfplay_git_commit': self.code['commit'], 'rows': 1}
            labels = b2.read(path('cycle-cache') / 'manifest.json')['labels_sha256']
            r = {'format': 'h7c-value-train-v1', 'cycle': cycle, 'git': self.code, 'verdict': self.mode,
                 'inputs': {'recipe_decision_sha256': loop.digest(dp), 'source': source,
                    'cycle_labels_sha256': labels, 'selfplay_run_seed': 8800+cycle,
                    'config_sha256': loop.digest(path('config')), 'replay_labels_sha256': self.p['value']['replay_anchor_labels_sha256'],
                    'parent_checkpoint_sha256': loop.digest(path('parent')),
                    'parent_metadata_sha256': loop.digest(path('parent').with_name('best.json'))}}
            if self.mode == 'CANDIDATE':
                (out / 'best.pt').write_bytes(f'candidate-c{cycle}'.encode())
                loop.atomic(out / 'best.json', {'h7c_stage': 'H7-c-value', 'cycle': cycle,
                    'source_models': d['parent'], 'recipe_decision_sha256': loop.digest(dp),
                    'training_sources': {'replay_anchor_labels_sha256': self.p['value']['replay_anchor_labels_sha256']}})
                r['candidate'] = {'checkpoint_sha256': loop.digest(out/'best.pt'),
                                  'metadata_sha256': loop.digest(out/'best.json')}
            loop.atomic(out/'record.json', r)
            return 0 if self.mode == 'CANDIDATE' else 1
        elif script == 'h7c_arena.py' and argv[3] in ('verdict', 'no-candidate'):
            parsed = ar.build_parser().parse_args(argv[3:])
            with mock.patch.object(ar, '_git', return_value=self.code):
                r = ar.verdict(parsed) if argv[3] == 'verdict' else ar.no_candidate(parsed)
            loop.atomic(path('output'), r)
        elif script in ('h7c_arena.py', 'h7c_drift.py'):
            candidate = tb_v3_net.model_identity(path('candidate-policy'), path('candidate-value'))
            common = {'cycle': cycle, 'candidate': candidate, 'parent': d['parent'],
                'champion_record_sha256': loop.digest(path('champion')), 'recipe_decision_sha256': loop.digest(dp),
                'candidate_mode': 'VALUE_ONLY', 'git': self.code}
            if script == 'h7c_arena.py' and argv[3] == 'tactical':
                loop.atomic(path('output'), {**common, 'rule_version': ar.TACTICAL_RULE, 'judge': judge_record()})
                return 0
            drift = script == 'h7c_drift.py'
            arm = 'drift' if drift else arg('arm')
            i = {**common, 'arm': arm, 'seed': (8750 if drift else 8730)+cycle}
            if drift:
                i['training_parent'] = d['parent']
            sig = ar.signature(i)
            games = [{'key': f'{arm}/{p}/{color}', 'pair': p, 'tested_color': color,
                      'result': 'loss' if drift and self.alarm else 'win', 'illegal': None,
                      'truncated': False, 'length': 2, 'moves': [[7,7],[7,8]], 'signature': sig}
                     for p in range(50 if drift or arm == 'official' else 25) for color in ('black','white')]
            s = ar.summarize_arm(games)
            r = {'identity': i, 'signature': sig, 'games': games, 'summary': s, 'fingerprint': ar.fingerprint(games)}
            if drift:
                r['judge'] = b2.drift_judge(s)
                from scripts import h7c_drift
                with mock.patch.object(h7c_drift, 'play', return_value=r):
                    self.assertEqual(h7c_drift.main(argv[3:]), 0)  # real CLI parser accepts executor argv
            else:
                loop.atomic(path('output'), r)
            path('jsonl').write_text('synthetic-games')
        elif script == 'h7c_external.py':
            from evaluation.external import summarize
            games = [{'key':f'{p}/{color}', 'pair':p, 'ours_color':color, 'result':'win', 'moves':[[7,7]],
                      'our_moves':[], 'run_signature':'fixture'} for p in range(25) for color in ('black','white')]
            loop.atomic(path('output'), {'seed':8710+cycle,'pairs':25,'git_dirty':False,'run_signature':'fixture',
                'ours_provenance':{'h7c':{'candidate_mode':'VALUE_ONLY','recipe_decision_sha256':loop.digest(dp),
                    'models':tb_v3_net.model_identity(path('policy-checkpoint'),path('value-checkpoint'))}},
                'games':games,'summary':summarize(games)})
            path('games-jsonl').write_text('synthetic-external')
        else:
            self.fail(f'unexpected subprocess: {argv}')
        return 0

    def cycle(self, number):
        return loop.run_cycle(self.root, self.p, self.psha, number, self.args, self.store)

    def test_promoted_champion_flows_to_next_cycle_and_resume_does_not_rerun(self):
        v4, guard = self.cycle(4)
        self.assertEqual(v4['promotion']['decision'], 'PROMOTE')
        self.assertFalse(guard['stop'])
        count = len(self.commands)
        self.cycle(4)
        self.assertEqual(len(self.commands), count)
        self.cycle(5)
        d5 = b2.read(self.root / 'docs/h7c/c5/decision.json')
        self.assertEqual(d5['parent'], v4['next_champion'])
        m = b2.read(self.root / 'docs/h7c/c4/evidence_manifest.json')
        self.assertIn('runs/h7c/c4/arena/drift.json', m['files_sha256'])
        self.assertTrue((self.root / 'docs/external/x2_c5.json').exists())

    def test_no_candidate_skips_arena_drift_x2_and_keeps_parent(self):
        self.mode = 'NO_CANDIDATE'
        v, guard = self.cycle(4)
        self.assertIsNone(guard)
        self.assertEqual(v['next_champion'], self.initial)
        self.assertEqual(v['b2_skipped']['x2'], 'NO_CANDIDATE')
        self.assertFalse(any(Path(a[2]).name in ('h7c_drift.py','h7c_external.py') for a in self.commands))

    def test_drift_alarm_preserves_promote_but_blocks_next_cycle(self):
        self.alarm = True
        v, guard = self.cycle(4)
        self.assertEqual(v['promotion']['decision'], 'PROMOTE')
        self.assertEqual(b2.stop_reason(self.p, [v], guard), 'DRIFT_DETECTED')
        self.assertEqual(v['b2_skipped']['x2'], 'DRIFT_DETECTED')

    def test_unexpected_runtime_failure_is_recorded_and_stops(self):
        for f in self.p['external']['files']:
            self.p['external']['files'][f] = loop.digest(self.args.rapfi_dir / f)
        self.p['external']['executable_sha256'] = loop.digest(self.args.rapfi_dir / 'engine')
        loop.atomic(self.args.program, self.p)
        self.store.path.unlink()  # main creates its own environment-bound state in this temporary repository
        with mock.patch.object(loop, 'ROOT', self.root), mock.patch.object(loop.subprocess, 'run'), \
                mock.patch.object(loop, 'run_cycle', side_effect=RuntimeError('simulated CUDA failure')):
            code = loop.main(['--program', str(self.args.program), '--rapfi-dir', str(self.args.rapfi_dir)])
        self.assertEqual(code, 1)
        state = b2.read(self.store.path)
        self.assertEqual(state['status'], 'ERROR')
        self.assertIn('simulated CUDA failure', state['error'])

    def test_code_change_during_stage_cannot_get_a_completion_receipt(self):
        with mock.patch.object(loop, 'git_identity', side_effect=[self.code, {'commit': 'other-code', 'dirty': False}]), \
                self.assertRaisesRegex(ValueError, 'while a B2 stage'):
            self.cycle(4)
        self.assertEqual(len(self.commands), 1)
        self.assertNotEqual(self.store.state['stages']['c4-selfplay']['status'], 'COMPLETE')

    def test_x2_duplicate_colour_cannot_masquerade_as_complete_pairs(self):
        v, _ = self.cycle(4)
        path = self.root / 'runs/h7c/c4/x2.json'
        r = b2.read(path)
        r['games'][1]['ours_color'] = r['games'][0]['ours_color']
        from evaluation.external import summarize
        r['summary'] = summarize(r['games'])
        loop.atomic(path, r)
        with self.assertRaisesRegex(ValueError, 'games/signature'):
            loop.validate_x2(path, 4, v['candidate'], loop.digest(self.root / 'docs/h7c/c4/decision.json'))


if __name__ == '__main__':
    unittest.main()
