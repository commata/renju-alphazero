"""E2 disagreement adjudication (scripts/h7c_adjudicate.py): inputs, branches, scoring, INVALID paths, resume, the
cluster bootstrap and the reading. The engine is the fake Piskvork engine or a scripted stand-in (no real Rapfi)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from renju import BLACK, WHITE, Game
from scripts import h7c_adjudicate as ad
from tests.test_h7c_labels import MOVES, game

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / 'scripts' / 'fake_piskvork_engine.py')
MODEL = lambda c: {k: c * 64 for k in ('policy_checkpoint_sha256', 'policy_metadata_sha256',  # noqa: E731
                                       'value_checkpoint_sha256', 'value_metadata_sha256')}
# a black double three at (7, 9) with black to move after these eight stones
DOUBLE_THREE = [[7, 7], [0, 0], [7, 8], [0, 14], [8, 9], [14, 0], [9, 9]]
# per ply of the MOVES game: (c8's move, orig's move); c5 is the actor (the self-play's own move)
CHOICES = {3: ([7, 9], None), 4: (None, [6, 6]), 6: ([5, 5], [5, 5]), 8: ([7, 6], None)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec(*extra, timeout=None):
    out = ad.engine_spec(ROOT / 'docs/h7c/b2_program.json', '/nonexistent-rapfi',
                         [sys.executable, FAKE, *extra], allow_unregistered=True)
    if timeout is not None:
        out['move_timeout'] = timeout
    return out


def task(moves, move, engine=None, max_plies=None):
    return {'key': ad.branch_key(moves, move), 'order': 0, 'identity_sha256': 'i' * 64, 'moves': moves,
            'move': move, 'spec': engine or spec('--seed', '1'), 'max_plies': max_plies}


def record(g, identity, choices=CHOICES):
    plies = []
    for p in g['plies']:
        c8, orig = choices.get(p['ply'], (None, None))
        sets = {'c5': p['engine_move'], 'c8': c8 or p['engine_move'], 'orig': orig or p['engine_move']}
        plies.append({'ply': p['ply'], 'color': p['color'], 'actor_route': p['route'], 'actor_move': p['engine_move'],
                      'sets': {k: {'route': p['route'], 'engine_move': m, 'visits': []} for k, m in sets.items()}})
    return {'identity': identity, 'game_index': g['game_index'], 'plies': plies}


def write_inputs(tmp, n_games=6, sample=4, choices=CHOICES):
    """Synthetic E2 inputs chained like the real ones (manifest, summary, sensitivity JSONL, self-play, verdict)."""
    from analysis import tb_v3
    import h7c_labels as lab
    from scripts import h7c_sensitivity as sn

    tmp = Path(tmp)
    games = [game(i) for i in range(n_games)]
    selfplay = tmp / 'selfplay.jsonl'
    selfplay.write_text(''.join(json.dumps(g) + '\n' for g in games), encoding='utf-8')
    chosen = sn.sample_games(games, sample)
    models = {'orig': tb_v3.MODELS, 'c5': MODEL('5'), 'c8': MODEL('8')}
    identity = {'format': 'h7c-sensitivity-v1', 'selfplay_jsonl_sha256': sha(selfplay),
                'selfplay': {'cycle': 8, 'models': models['c5']}, 'games_sample': sample,
                'sample_sha256': hashlib.sha256(json.dumps([lab.game_id(g) for g in chosen]).encode()).hexdigest(),
                'models': models}
    records = [record(g, identity, choices) for g in reversed(chosen)]  # completion order, not sample order
    sensitivity = tmp / 'c8_sample.jsonl'
    sensitivity.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    counts = {name: sum(p['sets'][a]['engine_move'] != p['sets'][b]['engine_move'] for r in records
                        for p in r['plies']) for name, (a, b) in (('c5:orig', ('c5', 'orig')), ('c8:c5', ('c8', 'c5')))}
    summary = tmp / 'sensitivity_c8.json'
    summary.write_text(json.dumps({
        'format': 'h7c-sensitivity-v1-summary', 'identity': identity, 'jsonl_sha256': sha(sensitivity),
        'actor_check': {'model_set': 'c5', 'reproduced': True},
        'comparisons': [{'pair': k, 'all_plies': {'different_moves': v}} for k, v in counts.items()]}))
    verdict = tmp / 'arena_verdict.json'
    verdict.write_text(json.dumps({'candidate': models['c8'], 'parent': models['c5']}))
    manifest = tmp / 'evidence_manifest.json'
    manifest.write_text(json.dumps({'champion_verdict_sha256': sha(verdict),
                                    'document_files_sha256': {'docs/h7c/e2/sensitivity_c8.json': sha(summary)},
                                    'files_sha256': {'runs/h7c/sensitivity/c8_sample.jsonl': sha(sensitivity)}}))
    return {'manifest': manifest, 'summary': summary, 'sensitivity_jsonl': sensitivity, 'selfplay': selfplay,
            'verdict': verdict, 'counts': counts}


def load(paths, expected=None):
    return ad.load_inputs(paths['manifest'], paths['summary'], paths['sensitivity_jsonl'], paths['selfplay'],
                          paths['verdict'], expected)


class InputTest(unittest.TestCase):
    def test_chain_and_observations(self):
        with TemporaryDirectory() as tmp:
            paths = write_inputs(tmp)
            inputs = load(paths, expected=paths['counts'])
            self.assertEqual(inputs['counts'], {'c8:c5': 12, 'c5:orig': 8})  # 4 games x (3 and 2 plies)
            o = next(o for o in inputs['observations'] if o['compare'] == 'c8:c5' and o['ply'] == 3)
            self.assertEqual((o['a'], o['b'], o['move_a'], o['move_b'], o['color']), ('c8', 'c5', [7, 9], [0, 2],
                                                                                       'white'))
            self.assertEqual(o['moves'], MOVES[:3])
            # a ply where both pick the same move is not an observation of that comparison
            self.assertFalse(any(o['compare'] == 'c8:c5' and o['ply'] == 4 for o in inputs['observations']))
            with self.assertRaisesRegex(ValueError, 'pre-registered'):
                load(paths, expected=ad.EXPECTED_DISAGREEMENTS)

    def test_overlap_counts_without_removing(self):
        with TemporaryDirectory() as tmp:
            obs = load(write_inputs(tmp))['observations']
            out = ad.overlap(obs)
            self.assertEqual(out['c8:c5']['observations'], 12)
            self.assertEqual(out['c8:c5']['branch_uses'], 24)
            self.assertEqual(out['c8:c5']['unique_branches'], 6)  # every game has the same moves
            self.assertEqual(out['c8:c5']['unique_positions'], 3)
            self.assertEqual(out['shared_positions'], 1)          # ply 6: both comparisons disagree there
            self.assertEqual(out['shared_branches'], 2)           # c5's and the shared [5, 5] at ply 6
            table = ad.branch_table(obs)
            self.assertEqual(len(table), 8)  # 4 positions x 2 moves
            self.assertEqual(ad.order(table), ad.order(dict(reversed(list(table.items())))))

    def test_broken_chains_are_refused(self):
        def edit(path, fn):
            data = json.loads(Path(path).read_text())
            fn(data)
            Path(path).write_text(json.dumps(data))

        cases = {
            'summary not in the manifest': lambda p: edit(p['manifest'], lambda m: m.update(document_files_sha256={})),
            'verdict not pinned': lambda p: edit(p['verdict'], lambda v: v.update(extra=1)),
            'actor not reproduced': lambda p: edit(p['summary'], lambda s: s['actor_check'].update(reproduced=False)),
            'c8 not the candidate': lambda p: edit(p['summary'], lambda s: s['identity']['models'].update(
                c8=MODEL('9'))),
            'jsonl changed': lambda p: Path(p['sensitivity_jsonl']).write_text(
                Path(p['sensitivity_jsonl']).read_text() + '\n'),
            'selfplay changed': lambda p: Path(p['selfplay']).write_text(Path(p['selfplay']).read_text() + '\n'),
            'count differs': lambda p: edit(p['summary'], lambda s: s['comparisons'][0]['all_plies'].update(
                different_moves=1)),
        }
        for name, change in cases.items():
            with self.subTest(name), TemporaryDirectory() as tmp:
                paths = write_inputs(tmp)
                change(paths)
                with self.assertRaises(ValueError):
                    load(paths)

    def test_records_must_be_the_sample_and_its_plies(self):
        def rewrite(paths, fn):
            lines = [json.loads(x) for x in Path(paths['sensitivity_jsonl']).read_text().splitlines()]
            lines = fn(lines)
            Path(paths['sensitivity_jsonl']).write_text(''.join(json.dumps(x) + '\n' for x in lines))
            for name, key in (('summary', 'jsonl_sha256'),):
                data = json.loads(Path(paths[name]).read_text())
                data[key] = sha(paths['sensitivity_jsonl'])
                Path(paths[name]).write_text(json.dumps(data))
            manifest = json.loads(Path(paths['manifest']).read_text())
            manifest['files_sha256'] = {'x': sha(paths['sensitivity_jsonl'])}
            manifest['document_files_sha256'] = {'y': sha(paths['summary'])}
            Path(paths['manifest']).write_text(json.dumps(manifest))

        def actor_moved(lines):
            lines[0]['plies'][1]['sets']['c5']['engine_move'] = [1, 1]
            return lines

        def ply_dropped(lines):
            lines[0]['plies'].pop()
            return lines

        def other_game(lines):
            lines[0]['game_index'] = 5 if lines[0]['game_index'] != 5 else 4
            return lines

        for name, fn in {'duplicate game': lambda x: x[:-1] + x[:1], 'missing game': lambda x: x[:-1],
                         'actor move': actor_moved, 'ply dropped': ply_dropped, 'not sampled': other_game}.items():
            with self.subTest(name), TemporaryDirectory() as tmp:
                paths = write_inputs(tmp)
                rewrite(paths, fn)
                with self.assertRaises(ValueError):
                    load(paths)


class ScoreTest(unittest.TestCase):
    def test_score_is_for_the_side_of_the_fixed_move(self):
        over = lambda w: SimpleNamespace(done=True, winner=w)  # noqa: E731
        self.assertEqual(ad.score_for(BLACK, over(BLACK)), 1.0)
        self.assertEqual(ad.score_for(WHITE, over(BLACK)), 0.0)
        self.assertEqual(ad.score_for(WHITE, over(WHITE)), 1.0)
        self.assertEqual(ad.score_for(BLACK, over(None)), 0.5)
        with self.assertRaises(ValueError):
            ad.score_for(BLACK, Game())

    def test_fixed_five_is_a_win_without_an_engine(self):
        with mock.patch('agents.external_engine.ExternalEngineAgent', side_effect=AssertionError('no engine')):
            r = ad.play_branch(task(MOVES[:8], [7, 11]))
        self.assertEqual((r['status'], r['score'], r['winner'], r['mover'], r['continuation']),
                         ('VALID', 1.0, 'black', 'black', []))

    def test_scripted_loss_and_isolation(self):
        """A scripted engine: one new agent per branch, closed after it; the mover's loss scores 0."""
        from agents.external_engine import our_forbidden_points
        made = []

        class Scripted:
            def __init__(self, config):
                self.script, self.closed = [[7, 11]], False
                made.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.closed = True

            def forbidden_points(self, g):
                return our_forbidden_points(g)

            def select_move(self, g):
                return tuple(self.script.pop(0))

        with mock.patch('agents.external_engine.ExternalEngineAgent', Scripted):
            lost = ad.play_branch(task(MOVES[:7], [14, 14]))      # white ignores the four: black's five
            again = ad.play_branch(task(MOVES[:7], [13, 13]))
        self.assertEqual((lost['status'], lost['mover'], lost['score'], lost['winner']), ('VALID', 'white', 0.0,
                                                                                         'black'))
        self.assertEqual((again['score'], again['continuation'], again['forbid_checked']), (0.0, [[7, 11]], 1))
        self.assertEqual(len(made), 2)
        self.assertTrue(all(m.closed for m in made))

    def test_a_new_engine_process_per_branch(self):
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / 'engine.log'
            engine = spec('--seed', '3', '--log', str(log))
            first = ad.play_branch(task(MOVES[:3], [0, 2], engine))
            second = ad.play_branch(task(MOVES[:3], [7, 9], engine))
            text = log.read_text().splitlines()
        self.assertEqual((first['status'], second['status']), ('VALID', 'VALID'))
        self.assertEqual(sum(x.startswith('START') for x in text), 2)
        self.assertEqual(sum(x == 'END' for x in text), 2)
        self.assertLess(text.index('END'), [i for i, x in enumerate(text) if x.startswith('START')][1])
        # nothing carries over: the same branch in a fresh process plays the same continuation
        replay = ad.play_branch(task(MOVES[:3], [0, 2], spec('--seed', '3')))
        self.assertEqual((replay['continuation'], replay['score']), (first['continuation'], first['score']))


class InvalidTest(unittest.TestCase):
    def reason(self, t):
        r = ad.play_branch(t)
        self.assertEqual((r['status'], r['score']), ('INVALID', None), r)
        return r['reason']

    def test_invalid_paths(self):
        self.assertEqual(self.reason(task(MOVES[:3], [7, 7])), 'fixed_move_illegal')
        self.assertEqual(self.reason(task(DOUBLE_THREE + [[14, 14]], [7, 9])), 'fixed_move_illegal')  # forbidden
        self.assertEqual(self.reason(task(MOVES[:3], [0, 2], spec('--mode', 'silent', timeout=1.0))), 'timeout')
        self.assertEqual(self.reason(task(MOVES[:3], [0, 2], spec('--mode', 'crash'))), 'engine_error')
        self.assertEqual(self.reason(task(MOVES[:3], [0, 2], spec('--mode', 'garbage'))), 'engine_error')
        self.assertEqual(self.reason(task(DOUBLE_THREE, [14, 14], spec('--mode', 'forbidden'))), 'engine_illegal')
        self.assertEqual(self.reason(task(DOUBLE_THREE, [14, 14], spec('--forbid-lie'))), 'forbid_mismatch')
        self.assertEqual(self.reason(task(MOVES[:3], [0, 2], max_plies=6)), 'truncated')
        self.assertEqual(self.reason(task(MOVES[:3], [0, 2], spec('--seed', '1') | {
            'command': ['/nonexistent-engine']})), 'engine_error')

    def test_no_legal_move_with_empty_points_is_not_a_draw(self):
        from agents.external_engine import our_forbidden_points

        class Stuck:  # the engine's move leaves black only forbidden points (the rules say done, no winner)
            def __init__(self, config):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def forbidden_points(self, g):
                return our_forbidden_points(g)

            def select_move(self, g):
                return (14, 13)

        def end_after(game_self, *a, **k):
            original(game_self, *a, **k)
            if len(game_self.history) == 10:
                game_self.done = True

        original = Game.play
        with mock.patch('agents.external_engine.ExternalEngineAgent', Stuck), \
                mock.patch.object(Game, 'play', end_after):
            r = ad.play_branch(task(DOUBLE_THREE + [[14, 14]], [6, 6]))
        self.assertEqual((r['status'], r['reason']), ('INVALID', 'abnormal_end'))

    def test_black_double_three_fixture(self):
        from agents.external_engine import our_forbidden_points
        g = Game()
        for m in DOUBLE_THREE + [[14, 14]]:
            g.play(*m)
        self.assertEqual(g.to_play, BLACK)
        self.assertEqual(our_forbidden_points(g), {(7, 9)})


class RecordTest(unittest.TestCase):
    def test_resume_refusals(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'b.jsonl'
            row = lambda k, i='x': {'key': k, 'order': 0, 'identity_sha256': i, 'status': 'VALID', 'score': 1.0,  # noqa
                                    'length': 9}
            write = lambda rows, tail='': path.write_text(''.join(json.dumps(r) + '\n' for r in rows) + tail)  # noqa
            write([row('a'), row('b')])
            self.assertEqual(set(ad.load_finished(path, 'x', {'a', 'b', 'c'})), {'a', 'b'})
            write([row('a')], '{"key": "b", "ord')  # a crash while appending: cut and played again
            self.assertEqual(set(ad.load_finished(path, 'x', {'a', 'b'})), {'a'})
            self.assertEqual(path.read_text().count('\n'), 1)
            write([row('a')])
            path.write_text(path.read_text().rstrip('\n'))  # complete, but cut before its newline
            self.assertEqual(set(ad.load_finished(path, 'x', {'a', 'b'})), {'a'})
            self.assertTrue(path.read_text().endswith('}\n'))
            for rows, tail in (([row('a'), row('a')], ''), ([row('a', 'y')], ''), ([row('z')], ''),
                               ([{**row('a'), 'status': None}, row('b')], ''),
                               ([{k: v for k, v in row('a').items() if k != 'score'}, row('b')], '')):
                write(rows, tail)
                with self.assertRaises(ValueError):
                    ad.load_finished(path, 'x', {'a', 'b'})


def obs_rows(spec_rows):
    """``(game, score_a, score_b)`` -> observations and the branch results compare_stats reads."""
    obs, results = [], {}
    for n, (g, sa, sb) in enumerate(spec_rows):
        moves = [[7, 7], [n // 15, n % 15]]
        o = {'compare': 'c8:c5', 'a': 'c8', 'b': 'c5', 'game_index': g, 'ply': 2, 'color': 'black', 'moves': moves,
             'move_a': [10, 10], 'move_b': [11, 11]}
        obs.append(o)
        for move, score in (([10, 10], sa), ([11, 11], sb)):
            key = ad.branch_key(moves, move)
            results[key] = {'status': 'INVALID' if score is None else 'VALID', 'score': score}
    return obs, results


class StatisticsTest(unittest.TestCase):
    def test_better_when_the_candidate_wins_most_decisive(self):
        rows = [(g, 1.0, 0.0) for g in range(20) for _ in range(2)] + [(g, 0.5, 0.5) for g in range(5)]
        out = ad.compare_stats('c8:c5', *obs_rows(rows))
        self.assertEqual((out['reading'], out['better'], out['decisive'], out['decisive_games']), ('BETTER', 40, 40,
                                                                                                    20))
        self.assertEqual(out['same_result'], {'share': round(5 / 45, 5), 'both_win': 0, 'both_draw': 5,
                                              'both_loss': 0})
        self.assertEqual(out['better_share'], 1.0)
        self.assertEqual(out['better_share_bootstrap']['undefined_resamples'], 0)

    def test_worse_and_inconclusive(self):
        worse = [(g, 0.0, 1.0) for g in range(25)] + [(0, 0.5, 0.0)]
        self.assertEqual(ad.compare_stats('x', *obs_rows(worse))['reading'], 'WORSE')
        even = [(g, 1.0, 0.0) if g % 2 else (g, 0.0, 1.0) for g in range(40)]
        out = ad.compare_stats('x', *obs_rows(even))
        self.assertEqual((out['reading'], out['reasons']), ('INCONCLUSIVE', ['INTERVAL_CONTAINS_HALF']))
        lo, hi = out['better_share_bootstrap']['ci975']
        lo95, hi95 = out['better_share_bootstrap']['ci95']
        self.assertTrue(lo <= lo95 <= 0.5 <= hi95 <= hi)

    def test_inconclusive_reasons(self):
        few = [(g, 1.0, 0.0) for g in range(15)]                        # 15 decisive
        self.assertEqual(ad.compare_stats('x', *obs_rows(few))['reasons'], ['TOO_FEW_DECISIVE'])
        few_games = [(g % 5, 1.0, 0.0) for g in range(40)]              # 40 decisive in 5 games
        self.assertEqual(ad.compare_stats('x', *obs_rows(few_games))['reasons'], ['TOO_FEW_DECISIVE'])
        # a resample with no decisive observation: undefined, never dropped (with 10 decisive games this is rare,
        # about e^-10 per resample, so the thresholds are lowered to reach it)
        sparse = [(g, 1.0, 0.0) for g in range(2) for _ in range(10)] + [(g, 1.0, 1.0) for g in range(2, 20)]
        with mock.patch.object(ad, 'MIN_DECISIVE_GAMES', 2):
            out = ad.compare_stats('x', *obs_rows(sparse))
        self.assertEqual(out['reasons'], ['DEGENERATE_BOOTSTRAP'])
        self.assertGreater(out['better_share_bootstrap']['undefined_resamples'], 0)
        self.assertIsNone(out['better_share_bootstrap']['ci975'])
        invalid = [(g, 1.0, 0.0) for g in range(30)] + [(g, None, 0.0) for g in range(2)]
        out = ad.compare_stats('x', *obs_rows(invalid))
        self.assertEqual((out['invalid'], out['valid'], out['reasons']), (2, 30, ['INVALID_RATE']))
        obs, results = obs_rows([(g, 1.0, 0.0) for g in range(30)])
        results.pop(ad.branch_key(obs[0]['moves'], obs[0]['move_b']))
        out = ad.compare_stats('x', obs, results)
        self.assertEqual((out['missing'], out['reasons'][0], out['mean_delta_bootstrap']), (1, 'INCOMPLETE', None))

    def test_bootstrap_is_fixed_and_counts_undefined(self):
        groups = {0: [{'delta': 1}], 1: [{'delta': 0}]}
        out = ad.bootstrap(groups, ad._better_share)
        self.assertEqual(out['ci95'], None)
        self.assertAlmostEqual(out['undefined_resamples'] / ad.BOOTSTRAP['samples'], 0.25, delta=0.02)
        self.assertEqual(out, ad.bootstrap(groups, ad._better_share))
        self.assertEqual(ad.bootstrap({0: [{'delta': 0.5}]}, ad._mean_delta)['ci95'], [0.5, 0.5])


class RunTest(unittest.TestCase):
    """The whole command with the fake engine: the smoke of the tool, not of Rapfi."""

    def args(self, paths, tmp, **change):
        base = dict(command='run', manifest=paths['manifest'], summary=paths['summary'],
                    sensitivity_jsonl=paths['sensitivity_jsonl'], selfplay=paths['selfplay'],
                    verdict=paths['verdict'], workers=2, limit=None, allow_dirty=True,
                    program=ROOT / 'docs/h7c/b2_program.json', rapfi_dir=Path(tmp) / 'rapfi',
                    jsonl=Path(tmp) / 'out' / 'branches.jsonl', output=Path(tmp) / 'out' / 'adjudication.json',
                    manifest_out=Path(tmp) / 'out' / 'manifest.json',
                    engine_command=[sys.executable, FAKE, '--seed', '2'], allow_unregistered_engine=True,
                    max_plies=None)
        return argparse.Namespace(**{**base, **change})

    def test_run_resume_and_refusals(self):
        from h7c_loop import exclusive
        quiet = lambda *a, **k: None  # noqa: E731
        with TemporaryDirectory() as tmp:
            paths = write_inputs(tmp)
            args = self.args(paths, tmp)
            result = ad.run(args, log=quiet)
            self.assertEqual(result['branches']['unique'], 8)
            self.assertEqual(result['branches']['played'], 8)
            stats = {s['comparison']: s for s in result['comparisons']}
            self.assertEqual(list(stats), ['c8:c5', 'c5:orig'])
            self.assertEqual((stats['c8:c5']['observations'], stats['c5:orig']['observations']), (12, 8))
            self.assertEqual(stats['c8:c5']['missing'], 0)
            self.assertEqual((stats['c8:c5']['reading'], stats['c8:c5']['reasons']), ('SMOKE', ['UNREGISTERED_ENGINE']))
            rows = [json.loads(x) for x in args.jsonl.read_text().splitlines()]
            self.assertEqual(sorted(r['order'] for r in rows), list(range(8)))
            five = ad.branch_key(MOVES[:8], [7, 6])
            self.assertEqual(next(r for r in rows if r['key'] == five)['score'], 1.0)

            lines = []
            again = ad.run(args, log=lambda *a, **k: lines.append(a[0]))
            self.assertIn('8 done, 0 to play', lines[0])
            self.assertEqual(again['jsonl_sha256'], result['jsonl_sha256'])

            for change in ({'max_plies': 300}, {'engine_command': [sys.executable, FAKE, '--seed', '4']}):
                with self.assertRaisesRegex(ValueError, 'another identity'):
                    ad.run(self.args(paths, tmp, **change), log=quiet)
            with exclusive(args.jsonl.with_name(args.jsonl.name + '.lock')):
                with self.assertRaisesRegex(ValueError, 'lock'):
                    ad.run(args, log=quiet)

            # an interrupted run resumes: the first rows kept, the rest played; a limited run never reads
            limited = self.args(paths, tmp, limit=3, jsonl=Path(tmp) / 'smoke' / 'b.jsonl')
            out = ad.run(limited, log=quiet)
            self.assertEqual(out['branches']['played'], 3)
            self.assertEqual(out['comparisons'][0]['reasons'], ['LIMITED_RUN', 'UNREGISTERED_ENGINE'])
            self.assertGreater(out['comparisons'][0]['missing'], 0)

    def test_registered_engine_is_required(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'registered X2'):
                ad.engine_spec(ROOT / 'docs/h7c/b2_program.json', tmp)
            program = json.loads((ROOT / 'docs/h7c/b2_program.json').read_text())['external']
            for name in program['files']:
                (Path(tmp) / name).write_text('x')
            (Path(tmp) / program['executable']).write_text('x')
            with self.assertRaisesRegex(ValueError, 'registered X2'):
                ad.engine_spec(ROOT / 'docs/h7c/b2_program.json', tmp)

    def test_smoke_paths_never_hit_the_full_run(self):
        parser = ad.build_parser()
        args = parser.parse_args(['run', '--limit', '4'])
        ad.resolve_paths(args)
        for attr in ('jsonl', 'output', 'manifest_out'):
            self.assertTrue(Path(getattr(args, attr)).is_relative_to(ad.SMOKE_DIR), attr)
        args = parser.parse_args(['run'])
        ad.resolve_paths(args)
        self.assertEqual(args.output, ad.DEFAULTS['output'])
        with self.assertRaises(ValueError):
            ad.resolve_paths(parser.parse_args(['run', '--limit', '4', '--jsonl', str(ad.DEFAULTS['jsonl'])]))


class ReferenceTest(unittest.TestCase):
    @unittest.skipUnless(__import__('rt30.fast5', fromlist=['available']).available(),
                         'python -m rt30.vcfcore.build first')
    def test_reference_with_stand_in_models(self):
        """S400 on two branches with random stand-in networks (CPU): only the stand-in flag lets them in."""
        from tests.test_tb_v3_net import write_models

        with TemporaryDirectory() as tmp:
            paths = write_inputs(tmp)
            policy, value = write_models(Path(tmp) / 'm', seed=3)
            args = argparse.Namespace(command='reference', **{k: paths[k] for k in (
                'manifest', 'summary', 'sensitivity_jsonl', 'selfplay', 'verdict')}, workers=1, limit=2,
                allow_dirty=True, policy_checkpoint=policy, value_checkpoint=value, device='cpu',
                jsonl=Path(tmp) / 'ref.jsonl', output=Path(tmp) / 'ref.json', allow_stand_in=False)
            with self.assertRaisesRegex(ValueError, 'original'):
                ad.reference(args, log=lambda *a, **k: None)
            out = ad.reference(argparse.Namespace(**{**vars(args), 'allow_stand_in': True}), log=lambda *a, **k: None)
            rows = [json.loads(x) for x in args.jsonl.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['value'] is None or -1 <= r['value'] <= 1 for r in rows))
            self.assertIn('never combined', out['meaning'])

    def test_mover_value(self):
        self.assertEqual(ad.mover_value([], BLACK, BLACK, True), 1.0)
        self.assertEqual(ad.mover_value([], BLACK, WHITE, True), -1.0)
        self.assertEqual(ad.mover_value([], None, WHITE, True), 0.0)
        self.assertEqual(ad.mover_value([[1, 1, 30, 0.4], [2, 2, 10, -0.9]], None, BLACK, False), -0.4)
        self.assertIsNone(ad.mover_value([], None, BLACK, False))


if __name__ == '__main__':
    unittest.main()
