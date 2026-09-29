"""Stage 9 entry rule of scripts/compare_teacher_arms.py on synthetic run directories."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

from compare_teacher_arms import arm_table, direct_result, verdict  # noqa: E402


def write_run(root: Path, scores: dict, vct_top3: dict) -> Path:
    for generation, (score, p) in scores.items():
        path = root / 'external_eval' / f'gen{generation:03d}_h2h.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'summary': {'a_score': score, 'p_two_sided': p}}))
    for generation, top3 in vct_top3.items():
        path = root / 'probes' / f'gen{generation:03d}_vct.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'summary': {'must_defend_vct': {'top3': top3}},
                                    'value_overall': {}}))
    return root


FLAT = {480: (0.52, 0.7), 560: (0.50, 1.0), 640: (0.54, 0.5)}


class CompareTeacherArmsTest(unittest.TestCase):
    def run_verdict(self, control_scores, teacher_scores, control_vct, teacher_vct, direct=None):
        with tempfile.TemporaryDirectory() as tmp:
            c = write_run(Path(tmp) / 'c', control_scores, control_vct)
            t = write_run(Path(tmp) / 't', teacher_scores, teacher_vct)
            return verdict(arm_table(c), arm_table(t), c, t, direct)

    def test_both_flat_is_capacity(self):
        flat = {560: 0.3, 640: 0.31}
        self.assertEqual(self.run_verdict(FLAT, FLAT, flat, flat)['decision'], 'capacity')

    def test_rising_probes_keep_it_undecided(self):
        result = self.run_verdict(FLAT, FLAT, {560: 0.3, 640: 0.31}, {560: 0.3, 640: 0.4})
        self.assertEqual(result['decision'], 'undecided')

    def test_improving_point_breaks_plateau(self):
        improving = {**FLAT, 640: (0.62, 0.01)}
        result = self.run_verdict(FLAT, improving, {640: 0.3}, {640: 0.3})
        self.assertFalse(result['plateau']['teacher'])
        self.assertEqual(result['decision'], 'teacher_better')  # anchor gap 0.08 (weak)

    def test_direct_match_decides(self):
        direct = {'match': 'teacher640 vs control640', 'teacher_score': 0.64, 'p': 0.01,
                  'games': 100, 'generation': 640}
        result = self.run_verdict(FLAT, FLAT, {640: 0.3}, {640: 0.3}, direct)
        self.assertEqual(result['decision'], 'teacher_better')

    def test_direct_result_reads_either_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.json'
            path.write_text(json.dumps({'matches': [{'summary': {
                'a': 'control640', 'b': 'teacher640', 'a_score': 0.3, 'p_two_sided': 0.001,
                'games': 100}}]}))
            result = direct_result([path], 'teacher', 'control')
            self.assertAlmostEqual(result['teacher_score'], 0.7)
            self.assertEqual(result['generation'], 640)

    def test_round_robin_uses_only_teacher_vs_control(self):
        def summary(a, b, score):
            return {'summary': {'a': a, 'b': b, 'a_score': score, 'p_two_sided': 0.01,
                                'games': 100}}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rr.json'
            path.write_text(json.dumps({'matches': [
                summary('control480', 'T1_480', 0.40), summary('control480', 'T2_480', 0.45),
                summary('T1_480', 'T2_480', 0.90)]}))
            t1 = direct_result([path], 'T1', 'control')
            t2 = direct_result([path], 'T2', 'control')
            self.assertEqual(t1['match'], 'control480 vs T1_480')
            self.assertAlmostEqual(t1['teacher_score'], 0.60)
            self.assertEqual(t2['match'], 'control480 vs T2_480')
            self.assertAlmostEqual(t2['teacher_score'], 0.55)


if __name__ == '__main__':
    unittest.main()
