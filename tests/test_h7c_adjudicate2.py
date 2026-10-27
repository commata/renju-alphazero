"""H7C-NMASK-1 confirmatory adjudication (scripts/h7c_adjudicate2.py) and the fixed sensitivity sample: input chain,
the decision's levels, v1 compatibility of the statistics, and a fake-engine run (no real Rapfi)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from hybrid import h7c_nmask as nm
from scripts import h7c_adjudicate as ad
from scripts import h7c_adjudicate2 as a2
from tests.test_h7c_adjudicate import FAKE, obs_rows
from tests.test_h7c_labels import game

ROOT = Path(__file__).resolve().parents[1]
MODEL = lambda c: {k: c * 64 for k in ('policy_checkpoint_sha256', 'policy_metadata_sha256',  # noqa: E731
                                       'value_checkpoint_sha256', 'value_metadata_sha256')}
# per ply: (mask's move, c8's move); c5 is the actor
CHOICES = {3: ([7, 9], [7, 9]), 4: ([6, 6], None), 6: ([5, 5], [4, 4]), 8: ([7, 6], None)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_confirm(tmp: Path, candidate: dict) -> dict:
    """Self-play of 10 games, an E2-like sample of 4, the confirmatory 6, their sensitivity record and a decision."""
    import h7c_labels as lab

    games = [game(i) for i in range(10)]
    selfplay = tmp / 'selfplay.jsonl'
    selfplay.write_text(''.join(json.dumps(g) + '\n' for g in games))
    from scripts import h7c_sensitivity as sn
    e2 = sn.sample_games(games, 4)
    e2_summary = {'identity': {'games_sample': 4, 'selfplay_jsonl_sha256': sha(selfplay), 'sample_sha256':
                               hashlib.sha256(json.dumps([lab.game_id(g) for g in e2]).encode()).hexdigest()}}
    sample = nm.confirm_sample(games, e2_summary)
    sample_path = tmp / 'confirm_sample.json'
    sample_path.write_text(json.dumps(sample))
    parent, control = MODEL('5'), MODEL('8')
    models = {'mask': candidate, 'c8': control, 'c5': parent}
    chosen = [g for g in games if lab.game_id(g) in set(sample['game_ids'])]
    identity = {'format': 'h7c-sensitivity-v1', 'selfplay_jsonl_sha256': sha(selfplay),
                'selfplay': {'models': parent}, 'games_sample': len(chosen),
                'sample_sha256': sample['ids_sha256'], 'models': models,
                'sample_file': {'sha256': sha(sample_path), 'rule': 'the listed games, in the listed order'}}
    records = []
    for g in reversed(chosen):
        plies = []
        for p in g['plies']:
            m, c8 = CHOICES.get(p['ply'], (None, None))
            sets = {'c5': p['engine_move'], 'mask': m or p['engine_move'], 'c8': c8 or p['engine_move']}
            plies.append({'ply': p['ply'], 'color': p['color'], 'actor_route': p['route'],
                          'actor_move': p['engine_move'],
                          'sets': {k: {'route': p['route'], 'engine_move': v, 'visits': []} for k, v in sets.items()}})
        records.append({'identity': identity, 'game_index': g['game_index'], 'plies': plies})
    jsonl = tmp / 'confirm.jsonl'
    jsonl.write_text(''.join(json.dumps(r) + '\n' for r in records))
    pairs = [('mask', 'c8'), ('mask', 'c5'), ('c8', 'c5')]
    counts = {f'{a}:{b}': sum(p['sets'][a]['engine_move'] != p['sets'][b]['engine_move'] for r in records
                              for p in r['plies']) for a, b in pairs}
    summary = tmp / 'summary.json'
    summary.write_text(json.dumps({'identity': identity, 'jsonl_sha256': sha(jsonl),
                                   'actor_check': {'model_set': 'c5', 'reproduced': True},
                                   'comparisons': [{'pair': k, 'all_plies': {'different_moves': v}}
                                                   for k, v in counts.items()]}))
    program = json.loads((ROOT / nm.RECORDS['b2_program']).read_text())
    decision = {'parent': parent, 'control': control, 'source': {'selfplay_jsonl_sha256': sha(selfplay)},
                'evaluation': {'confirm': {**nm.CONFIRM, 'sample_sha256': sha(sample_path),
                                           'sample_ids_sha256': sample['ids_sha256'], 'games': sample['games'],
                                           'engine': {'registered': program['external'], 'rule': 4,
                                                      'move_timeout': 600.0}}}}
    return {'decision': decision, 'summary': summary, 'sensitivity_jsonl': jsonl, 'selfplay': selfplay,
            'sample': sample_path, 'counts': counts, 'games': games, 'sample_body': sample}


def load(p, candidate):
    return a2.load_inputs(p['decision'], candidate, p['summary'], p['sensitivity_jsonl'], p['selfplay'], p['sample'])


class InputTest(unittest.TestCase):
    def test_chain(self):
        with TemporaryDirectory() as tmp:
            p = write_confirm(Path(tmp), MODEL('9'))
            inputs = load(p, MODEL('9'))
            self.assertEqual(inputs['counts'], p['counts'])
            self.assertEqual(p['counts'], {'mask:c8': 18, 'mask:c5': 24, 'c8:c5': 12})  # 6 games x 3 / 4 / 2 plies
            o = next(o for o in inputs['observations'] if o['compare'] == 'mask:c8')
            self.assertEqual((o['a'], o['b']), ('mask', 'c8'))
            with self.assertRaisesRegex(ValueError, 'model sets'):
                load(p, MODEL('7'))                                   # another candidate than the record's

    def test_refusals(self):
        def edited(p, name, fn):
            data = json.loads(Path(p[name]).read_text())
            fn(data)
            Path(p[name]).write_text(json.dumps(data))

        cases = {
            'actor': lambda p: edited(p, 'summary', lambda s: s['actor_check'].update(reproduced=False)),
            'comparisons': lambda p: edited(p, 'summary', lambda s: s['comparisons'].pop()),
            'sample file': lambda p: p['decision']['evaluation']['confirm'].update(sample_sha256='0' * 64),
            'games': lambda p: p['decision']['evaluation']['confirm'].update(games=5),
            'self-play': lambda p: p['decision']['source'].update(selfplay_jsonl_sha256='0' * 64),
        }
        for name, change in cases.items():
            with self.subTest(name), TemporaryDirectory() as tmp:
                p = write_confirm(Path(tmp), MODEL('9'))
                change(p)
                with self.assertRaises(ValueError):
                    load(p, MODEL('9'))
        with TemporaryDirectory() as tmp:  # a record of an E2 game (not in the confirmatory sample)
            p = write_confirm(Path(tmp), MODEL('9'))
            lines = [json.loads(x) for x in Path(p['sensitivity_jsonl']).read_text().splitlines()]
            e2_index = next(g['game_index'] for g in p['games'] if g['game_index'] not in
                            {r['game_index'] for r in lines})
            lines[0]['game_index'] = e2_index
            Path(p['sensitivity_jsonl']).write_text(''.join(json.dumps(x) + '\n' for x in lines))
            edited(p, 'summary', lambda s: s.update(jsonl_sha256=sha(p['sensitivity_jsonl'])))
            with self.assertRaisesRegex(ValueError, 'confirmatory sample'):
                load(p, MODEL('9'))

    def test_fixed_sensitivity_sample(self):
        from scripts import h7c_sensitivity as sn
        with TemporaryDirectory() as tmp:
            p = write_confirm(Path(tmp), MODEL('9'))
            games = p['games']
            chosen = sn.fixed_sample(games, p['sample'], sha(p['selfplay']))
            self.assertEqual([g['game_index'] for g in chosen], p['sample_body']['game_indices'])
            with self.assertRaisesRegex(ValueError, 'another self-play'):
                sn.fixed_sample(games, p['sample'], '0' * 64)
            with self.assertRaisesRegex(ValueError, 'not in the self-play'):
                sn.fixed_sample(games[:3], p['sample'], sha(p['selfplay']))
            body = dict(p['sample_body'], game_ids=p['sample_body']['game_ids'][:-1])
            Path(p['sample']).write_text(json.dumps(body))
            with self.assertRaises(ValueError):
                sn.fixed_sample(games, p['sample'], sha(p['selfplay']))
            args = argparse.Namespace(sample_file=p['sample'], games_sample=3)
            with self.assertRaisesRegex(ValueError, 'no --games-sample'):
                sn.run(argparse.Namespace(**vars(args), workers=1, allow_dirty=True), log=lambda *a, **k: None)


class StatisticsTest(unittest.TestCase):
    V1_RULES = {'bootstrap': ad.BOOTSTRAP, 'report_level': ad.REPORT_LEVEL, 'reading_level': ad.READING_LEVEL,
                'min_decisive': ad.MIN_DECISIVE, 'min_decisive_games': ad.MIN_DECISIVE_GAMES,
                'max_invalid_share': ad.MAX_INVALID_SHARE, 'primary': 'c8:c5'}

    def test_v1_rules_give_v1_numbers(self):
        rows = [(g, 1.0, 0.0) for g in range(15)] + [(g, 0.0, 1.0) for g in range(15, 25)] + [(g, 0.5, 0.5)
                                                                                            for g in range(5)]
        obs, results = obs_rows(rows)
        v1 = ad.compare_stats('c8:c5', obs, results)
        v2 = a2.compare_stats('c8:c5', obs, results, self.V1_RULES)
        self.assertEqual(v2['better_share_bootstrap']['ci_reading'], v1['better_share_bootstrap']['ci975'])
        self.assertEqual(v2['better_share_bootstrap']['ci95'], v1['better_share_bootstrap']['ci95'])
        for k in ('valid', 'decisive', 'better', 'worse', 'better_share', 'same_result', 'mean_delta', 'reading',
                  'reasons'):
            self.assertEqual(v2[k], v1[k], k)
        self.assertTrue(v2['primary'])

    def test_three_comparison_level(self):
        rows = [(g, 1.0, 0.0) for g in range(30)] + [(g, 0.0, 1.0) for g in range(30, 42)]
        obs, results = obs_rows(rows)
        out = a2.compare_stats('mask:c8', obs, results, nm.CONFIRM)
        b = out['better_share_bootstrap']
        self.assertAlmostEqual(b['reading_level'], 1 - 0.05 / 3)
        self.assertTrue(b['ci_reading'][0] <= b['ci95'][0] and b['ci95'][1] <= b['ci_reading'][1])
        self.assertEqual(nm.CONFIRM['bootstrap']['seed'], 8764)
        few = a2.compare_stats('mask:c8', *obs_rows([(g, 1.0, 0.0) for g in range(5)]), nm.CONFIRM)
        self.assertEqual(few['reasons'], ['TOO_FEW_DECISIVE'])
        empty = a2.compare_stats('mask:c8', [], {}, nm.CONFIRM)
        self.assertEqual((empty['reading'], empty['reasons']), ('INCONCLUSIVE', ['TOO_FEW_DECISIVE']))


class RunTest(unittest.TestCase):
    def test_fake_engine_run(self):
        from analysis import tb_v3_net
        from tests.test_tb_v3_net import write_models

        quiet = lambda *a, **k: None  # noqa: E731
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            policy, value = write_models(tmp / 'm')
            candidate = tb_v3_net.model_identity(policy, value)
            p = write_confirm(tmp, candidate)
            args = argparse.Namespace(
                decision=None, candidate_value=value, policy=policy, summary=p['summary'],
                sensitivity_jsonl=p['sensitivity_jsonl'], selfplay=p['selfplay'], sample=p['sample'],
                program=ROOT / nm.RECORDS['b2_program'], rapfi_dir=tmp / 'rapfi', workers=2, limit=None,
                jsonl=tmp / 'out' / 'b.jsonl', output=tmp / 'out' / 'a.json', manifest_out=tmp / 'out' / 'm.json',
                engine_command=[sys.executable, FAKE, '--seed', '3'], allow_unregistered_engine=True, max_plies=None,
                allow_dirty=True, allow_uncommitted=True)
            with mock.patch.object(nm, 'load_decision', lambda *a, **k: (p['decision'], 'd' * 64)), \
                    mock.patch('h7c_nmask.check_candidate', lambda *a, **k: {
                        'checkpoint_sha256': candidate['value_checkpoint_sha256']}):
                out = a2.run(args, log=quiet)
                self.assertEqual([s['comparison'] for s in out['comparisons']], ['mask:c8', 'mask:c5', 'c8:c5'])
                self.assertEqual([s['primary'] for s in out['comparisons']], [True, False, False])
                self.assertEqual({s['reading'] for s in out['comparisons']}, {'SMOKE'})
                self.assertTrue(out['sensitivity_abnormal_end_as_draw']['not_a_reading'])
                self.assertEqual(out['branches']['played'], out['branches']['unique'])
                self.assertEqual(out['identity']['order_seed'], 8763)
                lines = []
                again = a2.run(args, log=lambda *a, **k: lines.append(a[0]))
                self.assertIn('0 to play', lines[0])
                self.assertEqual(again['jsonl_sha256'], out['jsonl_sha256'])
                with self.assertRaisesRegex(ValueError, 'another identity'):
                    a2.run(argparse.Namespace(**{**vars(args), 'max_plies': 300}), log=quiet)
                bad = {**p['decision'], 'evaluation': {'confirm': {**p['decision']['evaluation']['confirm'],
                                                                   'engine': {'registered': {
                                                                       'executable_sha256': 'x', 'files': {},
                                                                       'info': []}}}}}
                with mock.patch.object(nm, 'load_decision', lambda *a, **k: (bad, 'd' * 64)):
                    with self.assertRaisesRegex(ValueError, 'engine'):
                        a2.run(argparse.Namespace(**{**vars(args), 'jsonl': tmp / 'o2' / 'b.jsonl'}), log=quiet)

    def test_smoke_paths(self):
        args = a2.build_parser().parse_args(['--limit', '5'])
        a2.resolve_paths(args)
        self.assertTrue(all(Path(getattr(args, k)).is_relative_to(a2.SMOKE_DIR)
                            for k in ('jsonl', 'output', 'manifest_out')))
        with self.assertRaises(ValueError):
            a2.resolve_paths(a2.build_parser().parse_args(['--limit', '5', '--output', str(a2.DEFAULTS['output'])]))


if __name__ == '__main__':
    unittest.main()
