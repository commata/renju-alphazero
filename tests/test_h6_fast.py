"""H6 builder --fast (hybrid.h6_fast): the exact speed-ups change no VCF result, node count, budget cut or row."""
import json
from pathlib import Path
import unittest
from unittest import mock

from hybrid import h6_fast
from hybrid.h6_labels import vcf_loss_status
from renju import BLACK, WHITE, Game
from rt30 import fast3_windows

ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / 'docs' / 'mcts-v8-results' / 'e3'
MODES = h6_fast.MODES if fast3_windows.available() else ('off', 'fast2')  # fast3: python -m rt30.native.build


def engine_games(count: int) -> list[dict]:
    """E3-A engine games (tactical, VCF-rich), in the H2 games format."""
    games = []
    for seed in (8415, 8416):
        with (E3 / f'puct_policy_vct2_50k_{seed}.jsonl').open(encoding='utf-8') as handle:
            for i, line in enumerate(handle):
                record = json.loads(line)
                games.append({'id': f'e3-{seed}-{i}', 'split': 'train', 'end': 'five', 'masked_plies': [],
                              'bresult': '1' if record['winner'] == BLACK else '0', 'moves': record['moves']})
    return games[:count]


def positions(games: list[dict], last: int):
    for game in games:
        moves = game['moves']
        for ply in range(max(0, len(moves) - last), len(moves)):
            position = Game()
            for move in moves[:ply]:
                position.play(*move)
            yield position


def per_mode(fn):
    """``{mode: fn()}`` with each mode installed in this process; the frozen functions come back after."""
    out = {}
    try:
        for mode in MODES:
            h6_fast.install(mode)
            out[mode] = fn()
    finally:
        h6_fast.install('off')
    return out


class InstallTest(unittest.TestCase):
    def tearDown(self):
        h6_fast.install('off')

    def test_modes_bind_what_they_name(self):
        for mode in MODES:
            self.assertEqual(h6_fast.install(mode), {'mode': mode, 'active': h6_fast.expected_active(mode)})
            self.assertEqual(h6_fast.provenance(mode)['active'], h6_fast.expected_active(mode))
        self.assertEqual(h6_fast.install('off')['active'], {'threat_windows': '_threat_windows',
                                                            'winning_extensions': '_fast_winning_extensions_in_direction'})
        with self.assertRaises(ValueError):
            h6_fast.install('fast4')

    def test_fast3_is_strict(self):
        with mock.patch.object(fast3_windows, '_native', None):  # not built: an error, never fast2
            with self.assertRaises(RuntimeError):
                h6_fast.install('fast3')
        self.assertEqual(h6_fast.state()['active'], h6_fast.expected_active('off'))
        if fast3_windows.available():
            with mock.patch('rt30.native.build.stale', return_value='sources changed'):
                with self.assertRaises(RuntimeError):
                    h6_fast.install('fast3')
            self.assertIn('native_build', h6_fast.provenance('fast3'))


class SameSearchTest(unittest.TestCase):
    def test_vcf_result_nodes_and_exhausted(self):
        from analysis.threats import _find_vcf_with_stats, _unbounded_max_fours

        games = list(positions(engine_games(4), 14))

        def run():
            return [_find_vcf_with_stats(game, attacker, max_fours=_unbounded_max_fours(game), node_limit=limit)
                    for game in games for attacker in (BLACK, WHITE) for limit in (50, 3_000)]

        out = per_mode(run)
        self.assertTrue(any(found is not None for found, _, _ in out['off']))
        self.assertTrue(any(exhausted for _, _, exhausted in out['off']))  # node-limit cuts are compared too
        for mode in MODES:
            self.assertEqual(out[mode], out['off'], mode)

    def test_loss_budget_cuts(self):
        games = list(positions(engine_games(6), 8))

        def run():
            return [vcf_loss_status(game, node_limit=20_000, call_budget=5_000, node_budget=budget)
                    for game in games for budget in (300, 100_000)]

        out = per_mode(run)
        self.assertEqual({'loss', 'budget', None}, set(out['off']))
        for mode in MODES:
            self.assertEqual(out[mode], out['off'], mode)


class BuilderTest(unittest.TestCase):
    def build(self, fast: str, workers: int = 1):
        from scripts.build_h6_labels import build

        rows, summary = build(engine_games(4), splits={'train'}, from_ply=5, outcome_weight=0.1, proof_weight=1.0,
                              prove_losses=True, vct1=False, probes=set(), prove_losses_last_n=6, workers=workers,
                              fast=fast, progress_seconds=600)
        return [row.to_dict() for row in rows], summary

    def test_same_rows_and_summary(self):
        from scripts import build_h6_labels

        with mock.patch.dict(build_h6_labels.LOSS_BUDGET, node_budget=2_000):  # also some budget cuts
            out = {mode: self.build(mode) for mode in MODES}
        self.assertGreater(out['off'][1]['prove_losses_budget_cuts'], 0)
        self.assertIn('vcf_loss', out['off'][1]['proof_by_distance_to_end'])
        for mode in MODES:
            self.assertEqual(out[mode], out['off'], mode)
        self.assertEqual(h6_fast.state()['active'], h6_fast.expected_active('off'))  # restored

    def test_pool_workers_install_the_mode(self):
        fastest = MODES[-1]
        self.assertEqual(self.build(fastest, workers=2), self.build('off'))
        self.assertEqual(h6_fast.state()['active'], h6_fast.expected_active('off'))

    def test_spawned_worker_installs_on_start(self):
        # Windows (spawn) and Python 3.14 Linux (forkserver) workers do not inherit the parent's rebinding
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        fastest = MODES[-1]
        h6_fast.install(fastest)  # the parent's state must not be what the worker reports
        h6_fast.install('off')
        with ProcessPoolExecutor(1, mp_context=get_context('spawn'), initializer=h6_fast.install,
                                 initargs=(fastest,)) as pool:
            self.assertEqual(pool.submit(h6_fast.state).result()['active'], h6_fast.expected_active(fastest))

    def test_a_worker_on_another_mode_fails(self):
        from scripts import build_h6_labels

        wrong = lambda tasks: ({'threat_windows': 'x'}, [build_h6_labels._proofs_task(t) for t in tasks])  # noqa: E731
        with mock.patch.object(build_h6_labels, '_proofs_chunk', wrong):
            with self.assertRaisesRegex(RuntimeError, 'a worker ran'):
                self.build('fast2')


if __name__ == '__main__':
    unittest.main()
