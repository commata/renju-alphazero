"""H7-R evaluation library (hybrid.h7r_eval): contracts, pair tables, the pair bootstrap, the anchor difference,
selection non-inferiority, sizing and the promotion rule v3."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from hybrid import h7r_eval as ev
from hybrid import h7r_openings as op

STATS = {'alpha_one_sided': 0.025, 'bootstrap': {'samples': 4000, 'seed': 7, 'method': 'percentile',
                                                 'unit': 'opening pair'},
         'error_budget': {'max_cycles': 3, 'per_cycle_alpha': 0.025}, 'delta': 0.03, 'power': 0.8}
MODEL = lambda c: {'policy': f'm/{c}/p.pt', 'value': f'm/{c}/v.pt',  # noqa: E731
                   'identity': {k: c * 64 for k in ('policy_checkpoint_sha256', 'policy_metadata_sha256',
                                                    'value_checkpoint_sha256', 'value_metadata_sha256')}}


def openings(n=20):
    return op.build_suite('O0', n, 8781)['openings']


def games(ops, results, split=None, replayable=False):
    """``results``: per opening (black game, white game) of the tested side; ``split``: openings whose two games
    have different move sequences (default: the score-split ones)."""
    out = []
    for i, (o, (b, w)) in enumerate(zip(ops, results)):
        differ = (b, w) not in (('win', 'loss'), ('loss', 'win'), ('draw', 'draw')) if split is None else i in split
        for color, result in (('black', b), ('white', w)):
            moves = o['moves'] + ([[0, 0]] if differ and color == 'white' else [[0, 1]])
            from tests.h7r_helpers import winning_moves
            from renju import BLACK, WHITE
            tested = BLACK if color == 'black' else WHITE
            moves = o['moves'] + ([[0, 0]] if differ and color == 'white' else [[0, 1]])
            if replayable and result in ('win', 'loss'):
                moves = winning_moves(o['moves'], tested if result == 'win' else -tested)
            out.append({'key': f'a/{i}/{color}', 'opening_id': o['history_id'], 'tested_color': color,
                        'result': result, 'moves': moves, 'length': len(moves),
                        'winner': (tested if result == 'win' else -tested) if result != 'draw' else None, 'illegal': None, 'truncated': False, 'game_seconds': 1.0})
    return out


def contract(root: Path, **change) -> dict:
    suite = op.build_suite('O0', 30, 8781)
    (root / 'suites').mkdir(parents=True, exist_ok=True)
    (root / 'suites/o0.json').write_text(json.dumps(suite))
    ref = {'path': 'suites/o0.json', 'sha256': ev.file_sha256(root / 'suites/o0.json')}
    arm = lambda name, t, o: {'name': name, 'tested': {'model': t, 'budget': 1},  # noqa: E731
                              'opponent': {'model': o, 'budget': 1}, 'pairs': 10, 'seed': 8790, 'pairing': 'P',
                              'suite': ref}
    body = {'format': ev.CONTRACT_FORMAT, 'contract_id': 'T-CONFIRM', 'purpose': 'confirm', 'status': 'fixed',
            'models': {'cand': MODEL('a'), 'champ': MODEL('b'), 'orig': MODEL('c')},
            'roles': {'candidate': 'cand', 'champion': 'champ', 'anchor': 'orig'},
            'arms': [arm('candidate_vs_champion', 'cand', 'champ'), arm('candidate_vs_anchor', 'cand', 'orig'),
                     arm('champion_vs_anchor', 'champ', 'orig')],
            'stats': STATS, 'informative': {'kind': 'score_split', 'm': 3},
            'anchor': {'model': 'orig', 'margin': -0.02}, 'selection': {'margin': -0.02, 'alpha': 0.025, 'min_observations': 20,
                'holdout_contract_sha256': 'f'*64, 'engine': {'registered_engine': True}},
            'gates': {'time': {'mean_ratio_max': 1.1, 'p95_ratio_max': 1.2}, 'tactical_rule': 'h7c-tactical-v2',
                      'forbidden_errors': 0}}
    body.update(change)
    return body


class ContractTest(unittest.TestCase):
    def write(self, root, body, name='c.json'):
        (root / 'contracts').mkdir(exist_ok=True)
        path = root / 'contracts' / name
        path.write_text(json.dumps(body))
        return path

    def test_valid_and_seed_rules(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = contract(root)
            path = self.write(root, body)
            loaded, sha = ev.load_contract(path, root, check_files=False, contracts_dir=root / 'contracts')
            self.assertEqual(ev.unset(loaded), [])
            # another contract may not reuse the seed
            self.write(root, {**body, 'contract_id': 'OTHER'}, 'd.json')
            with self.assertRaisesRegex(ValueError, 'already used'):
                ev.load_contract(path, root, check_files=False, contracts_dir=root / 'contracts')
            (root / 'contracts/d.json').unlink()
            bad = {**body, 'arms': [{**a, 'seed': 8771} for a in body['arms']]}  # reserved but not H7-R
            with self.assertRaisesRegex(ValueError, 'H7-R reserved'):
                ev.load_contract(self.write(root, bad), root, check_files=False, contracts_dir=root / 'x')
            unpaired = {**body, 'arms': [{**a, 'pairing': None} for a in body['arms']]}
            with self.assertRaisesRegex(ValueError, 'pairing'):
                ev.load_contract(self.write(root, unpaired), root, check_files=False, contracts_dir=root / 'x')
            moved = {**body, 'arms': [{**a, 'suite': {**a['suite'], 'sha256': '0' * 64}} for a in body['arms']]}
            with self.assertRaisesRegex(ValueError, 'suite'):
                ev.load_contract(self.write(root, moved), root, check_files=False, contracts_dir=root / 'x')
            with self.assertRaisesRegex(ValueError, 'identity'):  # the model files do not exist
                ev.load_contract(self.write(root, body), root, check_files=True, contracts_dir=root / 'x')

    def test_unset_values(self):
        with TemporaryDirectory() as tmp:
            body = contract(Path(tmp), status='draft', informative={'kind': 'score_split', 'm': None})
            self.assertEqual(set(ev.unset(body)), {'status', 'informative.m'})


def table(results, split=None, n=None):
    ops = openings(n or len(results))
    with patch.object(ev, 'validate_game'):
        return ev.pair_table(games(ops, results, split), ops), ops


class PairTest(unittest.TestCase):
    def setUp(self):
        # Pure statistics include synthetic draw samples; replay itself is tested separately.
        patcher = patch.object(ev, 'validate_game')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_every_valid_pair_counts_and_two_information_counts(self):
        results = [('win', 'loss')] * 12 + [('win', 'win')] * 3 + [('loss', 'loss')] + [('draw', 'win')] * 2 + \
            [('win', 'loss')] * 2
        t, _ = table(results, split={0, 1, 15, 16, 17, 18, 19})  # moves differ in 7, score splits in 6
        s = ev.arm_summary(t, STATS)
        self.assertEqual((s['pairs'], s['move_split'], s['score_split']), (20, 7, 6))
        self.assertEqual((s['score_split_positive'], s['score_split_negative']), (5, 1))
        self.assertAlmostEqual(s['score'], (14 * 0.5 + 3 * 1.0 + 0.0 + 2 * 0.75) / 20, places=5)  # 0.5 pairs in
        self.assertLessEqual(s['ci95'][0], s['lower_one_sided'] + 1e-9)
        self.assertEqual(s, ev.arm_summary(t, STATS))  # seeded

    def test_refusals_and_invalid(self):
        ops = openings(4)
        g = games(ops, [('win', 'loss')] * 4)
        with self.assertRaisesRegex(ValueError, 'missing'):
            ev.pair_table(g[:-1], ops)
        with self.assertRaisesRegex(ValueError, 'twice'):
            ev.pair_table(g + g[:1], ops)
        with self.assertRaisesRegex(ValueError, 'truncated'):
            ev.pair_table([{**g[0], 'truncated': True, 'result': 'truncated'}] + g[1:], ops)
        with self.assertRaisesRegex(ValueError, 'outside'):
            ev.pair_table(g, ops[:3])
        illegal = [{**g[0], 'result': 'opponent_illegal'}] + g[1:]
        t = ev.pair_table(illegal, ops)
        self.assertFalse(t['valid'])
        self.assertEqual(len(t['pairs']), 3)
        lost = [{**g[0], 'result': 'loss', 'illegal': {'side': 'tested'}}] + g[1:]
        self.assertEqual(ev.pair_table(lost, ops)['tested_illegal'], 1)

    def test_anchor_difference_is_paired_on_the_same_openings(self):
        ops = openings(40)
        a = ev.pair_table(games(ops, [('win', 'loss')] * 35 + [('win', 'win')] * 5), ops)
        b = ev.pair_table(games(ops, [('win', 'loss')] * 38 + [('win', 'win')] * 2), ops)
        d = ev.anchor_difference(a, b, STATS, -0.02)
        self.assertAlmostEqual(d['mean_difference'], 3 * 0.5 / 40, places=5)
        self.assertEqual(d['status'], 'NONINFERIOR')
        worse = ev.anchor_difference(b, a, STATS, -0.02)
        self.assertIn(worse['status'], ('INCONCLUSIVE', 'INFERIOR'))
        self.assertEqual(ev.anchor_difference(a, b, STATS, None)['status'], 'UNSET')
        other = ev.pair_table(games(openings(41)[1:], [('win', 'loss')] * 40), openings(41)[1:])
        with self.assertRaisesRegex(ValueError, 'same openings'):
            ev.anchor_difference(a, other, STATS, -0.02)


class SelectionTest(unittest.TestCase):
    def test_inconclusive_is_not_a_pass(self):
        rows = [{'game': g, 'delta': d} for g in range(30) for d in (0.0, 1.0 if g % 2 else -1.0)]
        out = ev.cluster_noninferiority(rows, -0.02, 0.025, STATS['bootstrap'], 20)
        self.assertEqual(out['status'], 'INCONCLUSIVE')
        good = [{'game': g, 'delta': 0.0} for g in range(60)] + [{'game': g, 'delta': 0.5} for g in range(10)]
        self.assertEqual(ev.cluster_noninferiority(good, -0.02, 0.025, STATS['bootstrap'], 20)['status'],
                         'NONINFERIOR')
        bad = [{'game': g, 'delta': -0.5} for g in range(60)]
        self.assertEqual(ev.cluster_noninferiority(bad, -0.02, 0.025, STATS['bootstrap'], 20)['status'],
                         'INFERIOR')
        self.assertEqual(ev.cluster_noninferiority(good[:5], -0.02, 0.025, STATS['bootstrap'], 20)['reason'],
                         'TOO_FEW')
        self.assertEqual(ev.cluster_noninferiority(good, None, 0.025, STATS['bootstrap'], 20)['status'], 'UNSET')

    def test_rows_of_one_game_move_together(self):
        # all the evidence in one game: the game-level bootstrap cannot be sure of it
        one = [{'game': 0, 'delta': 1.0}] * 50 + [{'game': g, 'delta': 0.0} for g in range(1, 6)]
        out = ev.cluster_noninferiority(one, 0.0, 0.025, STATS['bootstrap'], 20)
        self.assertEqual(out['status'], 'NONINFERIOR')  # >= margin 0 even without game 0
        self.assertLessEqual(out['lower_one_sided'], 0.0)


class SizingTest(unittest.TestCase):
    def test_proposal(self):
        t, _ = table([('win', 'loss')] * 40 + [('win', 'win')] * 6 + [('loss', 'loss')] * 4)
        out = ev.sizing(t['pairs'], STATS, 0.03, 0.8, [20, 30], 20.0)
        self.assertGreater(out['pair_sd_upper'], out['pair_sd'])
        self.assertGreater(out['pairs_needed'], 0)
        self.assertEqual(out['estimated_seconds'], out['pairs_needed'] * 20)
        self.assertEqual(out['informative_rates']['score_split']['count'], 10)
        self.assertTrue(0 <= out['chance_of_m_at_rate_lower_bound']['score_split']['30'] <= 1)
        self.assertIn('starts nothing', out['note'])
        self.assertEqual(ev.binomial_at_least(0, 10, 0.3), 1.0)


class JudgeTest(unittest.TestCase):
    def arms(self, cand_results, cand_anchor, champ_anchor):
        ops = openings(40)
        out = {}
        for name, results in (('candidate_vs_champion', cand_results), ('candidate_vs_anchor', cand_anchor),
                              ('champion_vs_anchor', champ_anchor)):
            t = ev.pair_table(games(ops, results, replayable=True), ops)
            out[name] = {'table': t, 'summary': ev.arm_summary(t, STATS)}
        return out

    GOOD = dict(cand_results=[('win', 'win')] * 12 + [('win', 'loss')] * 28,
                cand_anchor=[('win', 'loss')] * 40, champ_anchor=[('win', 'loss')] * 40)
    PASS = {'status': 'NONINFERIOR'}
    TIME = {'gate': {'pass': True, 'limits': {'mean_ratio_max': 1.1, 'p95_ratio_max': 1.2}}}

    def judge(self, body=None, arms=None, selection=PASS, tactical=None, timing=TIME):
        with TemporaryDirectory() as tmp:
            body = body or contract(Path(tmp))
        models = {r: body['models'][n]['identity'] for r, n in body['roles'].items()}
        def bound(record): return {'evidence_verified': True, 'models': models, **record}
        selection = bound(selection) if selection is not None else None
        timing = bound(timing) if timing is not None else None
        tactical = {'verdict': 'PASS'} if tactical is None else tactical
        tactical = bound({'source_format': ev.TACTICAL_FORMAT, **tactical}) if tactical != 'missing' else None
        return ev.judge(body, arms or self.arms(**self.GOOD), selection, tactical, timing)

    def test_promote_only_when_every_item_passes(self):
        out = self.judge()
        self.assertEqual((out['decision'], out['reasons']), ('PROMOTE', []), out)
        self.assertEqual(out['anchor']['status'], 'NONINFERIOR')
        self.assertIsNone(out['keep_kind'])

    def test_every_failure_keeps_the_champion(self):
        with TemporaryDirectory() as tmp:
            cases = {
                'CONFIG_UNSET': dict(body=contract(Path(tmp), anchor={'model': 'orig', 'margin': None})),
                'TOO_FEW_INFORMATIVE': dict(body=contract(Path(tmp), informative={'kind': 'score_split', 'm': 20})),
                'NOT_SUPERIOR': dict(arms=self.arms([('win', 'loss')] * 40, *list(self.GOOD.values())[1:])),
                'ANCHOR_INCONCLUSIVE': dict(arms=self.arms(self.GOOD['cand_results'],
                                                           [('win', 'loss')] * 36 + [('loss', 'loss')] * 4,
                                                           self.GOOD['champ_anchor'])),
                'SELECTION_INCONCLUSIVE': dict(selection={'status': 'INCONCLUSIVE'}),
                'SELECTION_MISSING': dict(selection=None),
                'GATE_TACTICAL': dict(tactical={'verdict': 'FAIL'}),
                'TACTICAL_MISSING': dict(tactical='missing'),
                'TIME_MISSING': dict(timing=None),
                'GATE_TIME': dict(timing={'gate': {'pass': True, 'limits': {'mean_ratio_max': 1.5,
                                                                            'p95_ratio_max': 2.0}}}),
            }
            for reason, kw in cases.items():
                with self.subTest(reason):
                    out = self.judge(**kw)
                    self.assertEqual(out['decision'], 'KEEP_CHAMPION')
                    self.assertTrue(any(r.startswith(reason.split('_')[0]) and reason in r or r == reason
                                        for r in out['reasons']), out['reasons'])
            arms = self.arms(**self.GOOD)
            g = games(openings(40), self.GOOD['cand_results'], replayable=True)
            g[0] = {**g[0], 'result': 'loss', 'winner': None, 'moves': g[0]['moves'][:4], 'length': 4,
                    'illegal': {'side': 'tested', 'ply': 4, 'move': [7, 7]}}
            arms['candidate_vs_champion']['table'] = ev.pair_table(g, openings(40))
            arms['candidate_vs_champion']['summary'] = ev.arm_summary(arms['candidate_vs_champion']['table'], STATS)
            self.assertIn('GATE_FORBIDDEN', self.judge(arms=arms)['reasons'])


if __name__ == '__main__':
    unittest.main()
