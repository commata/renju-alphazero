"""scripts/run_rt30_match.py: the A3 games are E3-A's games, and the paired comparison (docs/web-ser5.md §9.3)."""
from argparse import Namespace
import json
from pathlib import Path
import unittest

from scripts import run_rt30_match as match

ROOT = Path(__file__).resolve().parents[1]
E3A = [ROOT / 'docs' / 'mcts-v8-results' / 'e3' / f'puct_policy_vct2_50k_{seed}.jsonl' for seed in (8415, 8416)]


def args(**kw):
    base = dict(seeds=[8415, 8416], pairs=25, opening_random_plies=2, opening_radius=2, opponent='e3',
                deadline=27.5, max_plies=None)
    return Namespace(**{**base, **kw})


def game(key, run_seed, pair, color, result, moves, rt30_moves=()):
    return {'key': key, 'run_seed': run_seed, 'pair': pair, 'rt30_color': color, 'result': result,
            'moves': moves, 'rt30_moves': list(rt30_moves), 'length': len(moves), 'game_seconds': 1.0}


def rt(ply, timeouts=(), skipped=()):
    return {'ply': ply, 'seconds': 1.0, 'route': 'tree', 'timeouts': list(timeouts), 'skipped': list(skipped),
            'rescued': False, 'move': [0, 0]}


class E3TasksTest(unittest.TestCase):
    def test_openings_and_seeds_are_e3a_s(self):
        reference = match.load_reference(E3A)
        tasks = match.build_tasks(args())
        self.assertEqual(len(tasks), 100)
        self.assertEqual(len(reference), 100)
        for task in tasks:
            record = reference[(task['run_seed'], task['pair'], task['rt30_color'])]
            self.assertEqual(task['seed'], record['seed'])
            self.assertEqual(record['moves'][:len(task['opening'])], task['opening'])
            self.assertEqual(record['opponent'], match.E3_OPPONENT)

    def test_reference_scores_e3a(self):
        refs = list(match.load_reference(E3A).values())
        self.assertEqual(match.score(refs), 0.81)  # e3a_compare.json


class DivergenceTest(unittest.TestCase):
    ref = {'moves': [[7, 7], [7, 8], [8, 8], [6, 6], [9, 9]]}

    def test_kinds(self):
        same = game('k', 1, 0, 'black', 'win', self.ref['moves'])
        self.assertEqual(match.divergence(same, self.ref), {'kind': 'same', 'ply': None})
        # ply 2 is a black (RT30) move: changed under a deadline, or without one.
        moves = [[7, 7], [7, 8], [9, 9]]
        self.assertEqual(match.divergence(game('k', 1, 0, 'black', 'win', moves, [rt(2, timeouts=['root'])]),
                                          self.ref)['kind'], 'deadline')
        self.assertEqual(match.divergence(game('k', 1, 0, 'black', 'win', moves, [rt(2, skipped=['puct'])]),
                                          self.ref)['kind'], 'deadline')
        self.assertEqual(match.divergence(game('k', 1, 0, 'black', 'win', moves, [rt(2)]), self.ref),
                         {'kind': 'rt30_no_deadline', 'ply': 2})
        # ply 3 is a white (opponent) move.
        moves = [[7, 7], [7, 8], [8, 8], [1, 1]]
        self.assertEqual(match.divergence(game('k', 1, 0, 'black', 'win', moves, [rt(2)]), self.ref)['kind'],
                         'opponent')


class PairedSummaryTest(unittest.TestCase):
    def test_paired_difference(self):
        ref_moves = [[7, 7], [7, 8]]
        reference = {}
        games = []
        # pair 0: E3-A win/win, RT30 win/loss -> -0.5; pair 1: E3-A loss/draw, RT30 the same -> 0.
        for pair, ref_results, results in ((0, ('win', 'win'), ('win', 'loss')), (1, ('loss', 'draw'), ('loss', 'draw'))):
            for color, r, mine in zip(('black', 'white'), ref_results, results):
                reference[(5, pair, color)] = {'key': f'x/5/{pair}/{color}', 'pair': pair, 'v8_color': color,
                                               'result': r, 'moves': ref_moves}
                games.append(game(f'rt30@e3/5/{pair}/{color}', 5, pair, color, mine, ref_moves))
        summary = match.summarize(games, reference)
        self.assertEqual(summary['score'], 0.375)
        vs = summary['vs_reference']
        self.assertEqual(vs['games'], 4)
        self.assertEqual(vs['reference_score'], 0.625)
        self.assertEqual(vs['diff'], -0.25)
        self.assertEqual(vs['divergence']['same'], 4)
        self.assertEqual([c['key'] for c in vs['result_changed']], ['rt30@e3/5/0/white'])
        lo, hi = vs['diff_ci95_pairs']
        self.assertTrue(-0.5 <= lo <= -0.25 <= hi <= 0.0)

    def test_unfinished_games_are_left_out(self):
        games = [game('a', 1, 0, 'black', 'win', []), game('b', 1, 0, 'white', 'unfinished', [])]
        summary = match.summarize(games)
        self.assertEqual((summary['finished'], summary['unfinished'], summary['score']), (1, 1, 1.0))

    def test_wilson(self):
        self.assertEqual(match.wilson(81, 100), [0.722, 0.875])  # e3a_compare.json ci95_wilson


if __name__ == '__main__':
    unittest.main()
