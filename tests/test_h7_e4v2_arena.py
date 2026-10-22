"""scripts/summarize_h7_e4.py: the E4-v2 arena summary (synthetic runs, stand-in solver results)."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from analysis import tb_v2_runtime as rt
from analysis import tb_v2_value_e4v2 as cand
from analysis.tb_v2_value import ENGINE_OPTIONS, TB_V2_VALUE
from scripts import summarize_h7_e4 as arena

OPENING = [[7, 7], [7, 8]]
SWITCH = [14, 14]  # E's attack move at ply 6 where it switches
LENGTH = 12


def line(pair):
    """The game both arms play unless E switches: distinct points, the opening first."""
    rest = [[r, c] for r in range(15) for c in range(15) if [r, c] not in OPENING + [SWITCH]]
    return OPENING + rest[pair:pair + LENGTH - 2]


def game(arm, seed, pair, colour, switch=False, result='win', seconds=1.0):
    moves = line(pair)
    if switch:
        moves = moves[:6] + [SWITCH] + moves[7:]
    first = 2 if colour == 'black' else 3
    v8 = []
    for ply in range(first, LENGTH, 2):
        attacked = switch and ply == 6
        m = {'ply': ply, 'seconds': seconds, 'route': 'own_vct2' if attacked else 'tree', 'changed': attacked,
             'v7_move': moves[ply], 'played': moves[ply], 'root': {'checked': [[moves[ply], 'SAFE']], 'seconds': 0.1}}
        if arm == arena.CANDIDATE:
            m['e4_attack'] = ({'base_route': 'tree', 'base_move': line(pair)[ply], 'ran': True, 'move': SWITCH,
                               'exhausted': False, 'seconds': 0.5} if attacked else
                              {'base_route': 'tree', 'base_move': moves[ply], 'ran': True, 'move': None,
                               'exhausted': ply == 4, 'seconds': 0.5})
        v8.append(m)
    return {'key': f'{arm}@v8:full/{seed}/{pair}/{colour}', 'arm': arm, 'opponent': 'v8:full', 'pair': pair,
            'v8_color': colour, 'result': result, 'moves': moves, 'v8_moves': v8,
            'opponent_move_seconds': [1.0], 'game_seconds': 10.0}


def run(arm, seed, switch_pairs=(), loss_pairs=(), seconds=1.0, pairs=50):
    is_e = arm == arena.CANDIDATE
    games = [game(arm, seed, p, c, switch=is_e and p in switch_pairs and c == 'black',
                  result='loss' if is_e and p in loss_pairs else 'win', seconds=seconds)
             for p in range(pairs) for c in ('black', 'white')]
    runtime = (cand.provenance(arena.DEVICE) if is_e else
               {'engine_hash': rt.ENGINE_HASH, 'runtime_hash': rt.runtime_hash(arena.DEVICE)})
    return {'arm': arm, 'opponent': 'v8:full', 'seed': seed, 'pairs': pairs, 'fast': {'requested': 'fast4'},
            'value_device': 'cuda', 'canonical_factory': True, 'runtime': runtime,
            'v8_config': {**TB_V2_VALUE, **ENGINE_OPTIONS, **({'e4v2_attack': True} if is_e else {})},
            'openings': [{'pair': p, 'moves': OPENING} for p in range(pairs)], 'games': games}


def runs(**kw):
    return [run(arena.BASELINE, s) for s in arena.SEEDS] + [run(arena.CANDIDATE, s, **kw) for s in arena.SEEDS]


def fake_solve(task, b_punish=False):
    """WIN on every position at ply 6; the switch move punishes, B's move punishes if ``b_punish``."""
    win = len(task['moves']) == 6
    checks = {}
    for m in (task['played'] if win else []) + task['attack']:
        checks[json.dumps(m)] = {'outcome': 'PUNISH' if m == SWITCH or b_punish else 'MISS'}
    return {'key': task['key'], 'played': task['played'], 'attack': task['attack'],
            'reference': {'move': [0, 0] if win else None}, 'checks': checks}


def solved(rs, **kw):
    return {t['key']: fake_solve(t, **kw) for t in arena.tasks_of(arena.positions(rs))}


class ArenaTest(unittest.TestCase):
    def test_adopt_when_every_gate_passes(self):
        rs = runs(switch_pairs=range(0, 50, 2))  # E punishes on 25 of 50 openings per seed, B never
        out = arena.evaluate(rs, solved(rs))
        g = out['gates']
        self.assertEqual(out['verdict']['decision'], 'ADOPT', out['verdict'])
        # the events: ply 6 of every black game (white moves on odd plies), 50 pairs x 2 seeds
        self.assertEqual(g['frequency']['baseline'], {'events': 100, 'punished': 0, 'unclear': 0})
        self.assertEqual(g['frequency']['candidate'], {'events': 100, 'punished': 50, 'unclear': 0})
        self.assertEqual(g['frequency']['rate'], {'baseline': 0.0, 'candidate': 0.5})
        self.assertGreater(g['frequency']['ci95'][0], 0)
        self.assertEqual((g['soundness']['attack_wins'], g['soundness']['switched'], g['soundness']['miss']),
                         (50, 50, []))
        self.assertEqual(g['cost']['ratio'], 1.0)
        self.assertEqual(out['report']['e4v2_steps']['skips'], {})
        self.assertEqual(out['report']['isolation']['left_before_first_switch'], [])
        self.assertEqual(out['report']['isolation']['identical_games'], 150)

    def test_event_positions_exclude_the_one_after_the_opening(self):
        rows = arena.positions(runs())
        black = [r for r in rows if r['game'].endswith('/0/black')]
        self.assertEqual([(r['ply'], r['event_position']) for r in black][:2], [(2, False), (4, True)])

    def test_rates_are_pooled_not_averaged(self):
        a, b = arena.BASELINE, arena.CANDIDATE
        def c(events, punished):
            return {'events': events, 'punished': punished, 'unclear': 0}

        counts = {(1, 0): {a: c(1, 1), b: c(9, 0)}, (1, 1): {a: c(9, 0), b: c(1, 1)}}
        out = arena.frequency(counts)
        self.assertEqual(out['rate'], {'baseline': 0.1, 'candidate': 0.1})  # per-opening means would be 0.5 / 0.5
        self.assertEqual(out['verdict'], 'FAIL')
        empty = {(1, 0): {a: c(0, 0), b: c(0, 0)}}
        self.assertEqual(arena.frequency(empty)['verdict'], 'INCOMPLETE')

    def test_rejects(self):
        rs = runs(switch_pairs=range(0, 50, 2))
        out = arena.evaluate(rs, solved(rs, b_punish=True))  # B punishes every event as well
        self.assertEqual(out['gates']['frequency']['verdict'], 'FAIL')
        self.assertEqual(out['verdict']['decision'], 'REJECT')
        rs = runs(switch_pairs=range(0, 50, 2), loss_pairs=range(10))
        self.assertEqual(arena.evaluate(rs, solved(rs))['gates']['strength']['verdict'], 'FAIL')
        rs = runs(switch_pairs=range(0, 50, 2), seconds=1.6)
        self.assertEqual(arena.evaluate(rs, solved(rs))['gates']['cost']['verdict'], 'FAIL')
        rs = runs(switch_pairs=range(0, 50, 2))
        bad = solved(rs)
        for row in bad.values():
            for k in row['checks']:
                if k == json.dumps(SWITCH):
                    row['checks'][k] = {'outcome': 'MISS'}
        out = arena.evaluate(rs, bad)
        self.assertEqual((out['gates']['soundness']['verdict'], len(out['gates']['soundness']['miss'])), ('FAIL', 50))
        self.assertEqual(out['verdict']['decision'], 'REJECT')

    def test_refusals(self):
        good = runs()
        with self.assertRaises(ValueError):  # a seed missing
            arena.evaluate(good[:1] + good[2:], solved(good))
        with self.assertRaises(ValueError):  # unsolved positions
            arena.evaluate(good, {})
        for change in ({'fast': {'requested': 'fast3'}}, {'pairs': 25}, {'value_device': 'cpu'},
                       {'canonical_factory': None}, {'runtime': {'engine_hash': rt.ENGINE_HASH, 'runtime_hash': 'x'}},
                       {'v8_config': {**TB_V2_VALUE, **ENGINE_OPTIONS}}):
            bad = run(arena.CANDIDATE, 8441)
            bad.update(change)
            with self.assertRaises(ValueError, msg=str(change)):
                arena.check_run(bad)
        bad = run(arena.BASELINE, 8441)
        bad['games'][0]['v8_moves'][0]['e4_attack'] = {'ran': False, 'skip': 'route'}
        with self.assertRaises(ValueError):  # B ran the attack step
            arena.check_run(bad)
        bad = run(arena.CANDIDATE, 8441)
        del bad['games'][0]['v8_moves'][0]['e4_attack']
        with self.assertRaises(ValueError):  # a move of E without its step record
            arena.check_run(bad)
        bad = run(arena.CANDIDATE, 8441, switch_pairs=[0])
        bad['games'][0]['v8_moves'][2]['route'] = 'tree'
        with self.assertRaises(ValueError):  # an attack WIN that was not played
            arena.positions([bad])
        with self.assertRaises(ValueError):
            arena.check_reference({'verdict': 'FREEZE', 'budget': '2m'})
        ref = json.loads((arena.ROOT / arena.REFERENCE_RESULT).read_text(encoding='utf-8'))
        arena.check_reference(ref)  # the committed 4M check is of the current detector file

    def test_one_task_per_history(self):
        rs = runs(switch_pairs=[0])
        rows = arena.positions(rs)
        tasks = arena.tasks_of(rows)
        self.assertLess(len(tasks), len(rows))
        at6 = [t for t in tasks if t['moves'] == line(0)[:6]]
        self.assertEqual(len(at6), 1)
        self.assertEqual((at6[0]['played'], at6[0]['attack']), (sorted([line(0)[6], SWITCH]), [SWITCH]))

    def test_main_resumes_from_the_jsonl(self):
        rs = runs(switch_pairs=range(0, 50, 2))
        calls = []

        def stand_in(task):
            calls.append(task['key'])
            return fake_solve(task)

        with TemporaryDirectory() as tmp:
            paths = []
            for r in rs:
                p = Path(tmp) / f"{r['arm']}_{r['seed']}.json"
                p.write_text(json.dumps(r), encoding='utf-8')
                paths.append(str(p))
            jsonl = Path(tmp) / 'pos.jsonl'
            argv = ['--runs', *paths, '--fast', 'off', '--jsonl', str(jsonl)]
            with mock.patch.object(arena, 'position_task', stand_in):
                self.assertEqual(arena.main(argv + ['--output', str(Path(tmp) / 'a.json')]), 0)
                first = len(calls)
                lines = jsonl.read_text(encoding='utf-8').splitlines()
                jsonl.write_text('\n'.join(lines[:-3]) + '\n{"torn', encoding='utf-8')
                self.assertEqual(arena.main(argv + ['--output', str(Path(tmp) / 'b.json')]), 0)
            self.assertEqual(len(calls) - first, 3)
            a, b = (json.loads((Path(tmp) / n).read_text(encoding='utf-8')) for n in ('a.json', 'b.json'))
            self.assertEqual((a['verdict'], a['gates']), (b['verdict'], b['gates']))
            self.assertEqual(a['verdict']['decision'], 'ADOPT')
            rows = [json.loads(x) for x in jsonl.read_text(encoding='utf-8').splitlines()[:1]]
            rows[0]['fingerprint']['budget'] = '2m'
            jsonl.write_text(json.dumps(rows[0]) + '\n', encoding='utf-8')
            with mock.patch.object(arena, 'position_task', stand_in), self.assertRaises(SystemExit):
                arena.main(argv)  # another reference budget in the file

    def test_shards_then_merge(self):
        rs = runs(switch_pairs=[0, 1])
        calls = []

        def stand_in(task):
            calls.append(task['key'])
            return fake_solve(task)

        with TemporaryDirectory() as tmp:
            paths = []
            for r in rs:
                p = Path(tmp) / f"{r['arm']}_{r['seed']}.json"
                p.write_text(json.dumps(r), encoding='utf-8')
                paths.append(str(p))
            base = ['--runs', *paths, '--fast', 'off']
            parts = [str(Path(tmp) / f'part_{k}.jsonl') for k in range(2)]
            with mock.patch.object(arena, 'position_task', stand_in):
                for k in range(2):
                    self.assertEqual(arena.main(base + ['--shard', f'{k}/2', '--jsonl', parts[k]]), 0)
                solved_once = len(calls)
                self.assertEqual(solved_once, len(set(calls)))
                out = Path(tmp) / 'a.json'
                self.assertEqual(arena.main(base + ['--merge', *parts, '--jsonl', str(Path(tmp) / 'all.jsonl'),
                                                    '--output', str(out)]), 0)
                self.assertEqual(len(calls), solved_once)  # nothing solved again
                with self.assertRaises(SystemExit):
                    arena.main(base + ['--shard', '2/2', '--jsonl', parts[0]])
            self.assertEqual(len(json.loads(out.read_text(encoding='utf-8'))['positions']), solved_once)

    def test_real_position_task(self):
        """The reference and the checker on a real position: white's 4x4 at (5, 8) after the VCF line."""
        from tests.test_h7_e4 import VCF

        moves = [list(m) for m in VCF]
        task = {'key': arena.history_key(moves), 'moves': moves, 'played': [[5, 8]], 'attack': []}
        row = arena.position_task(task)
        self.assertIsNotNone(row['reference']['move'])
        self.assertEqual(row['checks'][json.dumps([5, 8])]['outcome'], 'PUNISH')


if __name__ == '__main__':
    unittest.main()
