"""scripts/summarize_h6_6.py: guard v2 (4M proof non-inferiority by the upper bound) and the stages."""
import unittest

from scripts import summarize_h6_6 as s


def move(ply, status):
    return {'route': 'tree', 'ply': ply, 'v7_move': [ply % 15, 0],
            'root': {'checked': [[[ply % 15, 0], status]]}}


def run(arm, seed, per_game):
    games = [{'key': f'{arm}[x]@v8:full/{seed}/{i}/black', 'v8_moves': [move(10 + j, st) for j, st in enumerate(sts)]}
             for i, sts in enumerate(per_game)]
    return {'arm': arm, 'seed': seed, 'games': games}


def d2_for(runs, large):
    """d2 rows for every not-SAFE tree move; ``large(arm, status)`` gives the 4M status."""
    rows = []
    for r in runs:
        for g in r['games']:
            for m in g['v8_moves']:
                st = m['root']['checked'][0][1]
                if st != 'SAFE':
                    rows.append({'id': f"{r['arm']}|{g['key']}|{m['ply']}", 'large': {'status': large(r['arm'], st)}})
    return {'summary': {'replay_mismatches': []}, 'fast': {'requested': 'fast3'}, 'rows': rows}


class ProofTest(unittest.TestCase):
    def test_status_mapping_and_missing_recheck(self):
        runs = [run(s.CANDIDATE, 8404, [['SAFE', 'UNKNOWN']]), run(s.BASELINE, 8404, [['UNSAFE', 'SAFE']])]
        statuses = s.proof_status(runs, [d2_for(runs, lambda arm, st: 'SAFE' if st == 'UNKNOWN' else st)])
        self.assertEqual(statuses[s.CANDIDATE]['8404/0/black'], ['SAFE', 'SAFE'])
        self.assertEqual(statuses[s.BASELINE]['8404/0/black'], ['UNSAFE', 'SAFE'])
        empty = {'summary': {'replay_mismatches': []}, 'fast': {'requested': 'fast3'}, 'rows': []}
        with self.assertRaises(ValueError):
            s.proof_status(runs, [empty])
        with self.assertRaises(ValueError):
            s.proof_status(runs, [{**d2_for(runs, lambda a, st: st), 'fast': {'requested': 'off'}}])

    def test_upper_bound_decides(self):
        same = {'a': {f'k{i}': ['SAFE'] * 20 for i in range(30)}}
        statuses = {s.CANDIDATE: same['a'], s.BASELINE: same['a']}
        self.assertEqual(s.proof_guard(statuses, 'UNSAFE')['verdict'], 'PASS')
        worse = {f'k{i}': ['UNSAFE'] * 4 + ['SAFE'] * 16 for i in range(30)}  # +20pp everywhere
        self.assertEqual(s.proof_guard({s.CANDIDATE: worse, s.BASELINE: same['a']}, 'UNSAFE')['verdict'], 'REJECT')
        noisy = {f'k{i}': (['UNSAFE'] * 5 + ['SAFE'] * 15 if i < 3 else ['SAFE'] * 20) for i in range(30)}
        out = s.proof_guard({s.CANDIDATE: noisy, s.BASELINE: same['a']}, 'UNSAFE')
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')  # +2.5pp, but a few openings carry it all
        self.assertGreater(out['ci95'][1], 0.03)


def row(seeds, score=0.6, low=0.52, black=0.6, white=0.6, time_ratio=1.0, violations=()):
    return {'seeds': seeds, 'score': score, 'ci95_pairs': [low, 0.7], 'by_colour': {'black': black, 'white': white},
            'time_ratio': time_ratio, 'safety_violations': list(violations)}


def proofs(unsafe='PASS', unknown='PASS'):
    return {'UNSAFE': {'target': 'UNSAFE', 'verdict': unsafe, 'ci95': [0.0, 0.02]},
            'UNKNOWN': {'target': 'UNKNOWN', 'verdict': unknown, 'ci95': [0.0, 0.02]}}


class DecideTest(unittest.TestCase):
    def go(self, seeds, p=None, p92='PASS', aux_violation=False, **main):
        aux = {s.CANDIDATE: row(seeds, violations=[{'k': 1}] if aux_violation else ()), s.BASELINE: row(seeds)}
        return s.decide(row(seeds, **main), aux, p or proofs(), p92)['decision']

    def test_first_seed_uses_basic_guards_only(self):
        self.assertEqual(self.go([8404], score=0.5), 'RUN_8405')
        self.assertEqual(self.go([8404], p=proofs('REJECT', 'REJECT')), 'RUN_8405')  # reported, not applied
        self.assertEqual(self.go([8404], score=0.49), 'REJECT')
        self.assertEqual(self.go([8404], aux_violation=True), 'REJECT')
        self.assertEqual(self.go([8404], time_ratio=1.51), 'REJECT')
        self.assertEqual(self.go([8404], white=0.44), 'REJECT')
        self.assertEqual(self.go([8404], p92='FAIL'), 'REJECT')
        self.assertEqual(self.go([8404], p92='NOT_DISCRIMINATING'), 'RUN_8405')

    def test_two_and_three_seeds(self):
        two, three = [8404, 8405], [8404, 8405, 8406]
        self.assertEqual(self.go(two), 'ADOPT')
        self.assertEqual(self.go(two, score=0.54), 'REJECT')
        self.assertEqual(self.go(two, p=proofs('REJECT')), 'REJECT')
        self.assertEqual(self.go(two, p=proofs(unknown='INCONCLUSIVE')), 'INCONCLUSIVE')
        self.assertEqual(self.go(two, low=0.5), 'INCONCLUSIVE')
        self.assertEqual(self.go(three, low=0.5), 'REJECT')
        self.assertEqual(self.go(three), 'ADOPT')
        self.assertEqual(self.go([8405]), 'INCOMPLETE')
        self.assertEqual(self.go(two, p92=None), 'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
