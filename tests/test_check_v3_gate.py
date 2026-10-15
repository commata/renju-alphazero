"""scripts/check_v3_gate.py gate: PASS / REVIEW / FAIL on synthetic v3 outputs (docs/web-ser5.md §9.8)."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('check_v3_gate', ROOT / 'scripts' / 'check_v3_gate.py')
gate_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate_script)


def case(key, recovered=True, timeouts=(), skipped=(), helper_failed=''):
    return {'key': key, 'move': [7, 7] if recovered else [7, 8], 'tb1_move': [7, 7], 'recovered': recovered,
            'timeouts': list(timeouts), 'skipped': list(skipped), 'attack_helper_failed': helper_failed}


def divergence_summary(rows, net=6.0):
    return {'cases': len(rows), 'recovered': sum(r['recovered'] for r in rows),
            'loss_making': {'points_if_recovered': net}, 'gain_making': {'points_if_recovered': 0.0},
            'seconds': {'over_27.5s': 0}}


class GateTest(unittest.TestCase):
    def run_gate(self, control_rows, divergence_rows=None, a1_inline=0):
        divergence_rows = divergence_rows or [case(f'd/{i}') for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            def write(name, data):
                (out / name).write_text(json.dumps(data), encoding='utf-8')

            write('v3_equivalence_h3_8.json', {'summary': {'identical': 8, 'positions': 8}})
            write('a31_v3_w4.json', {'rows': divergence_rows, 'summary': divergence_summary(divergence_rows)})
            write('ctrl_v3_w4.json', {'rows': control_rows, 'summary': {
                'cases': len(control_rows), 'recovered': sum(r['recovered'] for r in control_rows)}})
            for name in ('s60', 'crit'):
                write(f'v3_h3_{name}.json', {'summary': {'rt3': {'seconds': {'over_27.5s': 0}},
                                                         'rt_deadline': {'attack_helper_failed': a1_inline}}})
                write(f'recheck_v3_h3_{name}.json', {'summary': {'changed_move_proven_unsafe': 0}})
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                code = gate_script.main(['gate', '--out', str(out)])
        return code, printed.getvalue()

    def test_all_kept_passes(self):
        code, text = self.run_gate([case(f'c/{i}') for i in range(26)])
        self.assertEqual(code, 0)
        self.assertIn('A3-v3 gate: PASS', text)

    def test_changed_control_move_cut_by_the_deadline_needs_review(self):
        rows = [case(f'c/{i}') for i in range(25)] + [case('c/25', recovered=False, timeouts=['root'])]
        code, text = self.run_gate(rows)
        self.assertEqual(code, 2)
        self.assertIn('`c/25`', text)

    def test_changed_control_move_without_a_cut_fails(self):
        rows = [case(f'c/{i}') for i in range(25)] + [case('c/25', recovered=False)]
        code, text = self.run_gate(rows)
        self.assertEqual(code, 1)
        self.assertIn('exactness failure', text)

    def test_the_old_23_of_26_tolerance_is_gone(self):
        rows = [case(f'c/{i}') for i in range(23)] + [case(f'c/{i}', recovered=False, skipped=['puct'])
                                                      for i in range(23, 26)]
        self.assertEqual(self.run_gate(rows)[0], 2)

    def test_v8b_inline_fails(self):
        control = [case(f'c/{i}') for i in range(26)]
        self.assertEqual(self.run_gate(control, [case('d/0', helper_failed='AssertionError: daemonic')])[0], 1)
        self.assertEqual(self.run_gate(control, a1_inline=1)[0], 1)

    def test_missing_output_fails(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gate_script.main(['gate', '--out', tmp]), 1)


if __name__ == '__main__':
    unittest.main()
