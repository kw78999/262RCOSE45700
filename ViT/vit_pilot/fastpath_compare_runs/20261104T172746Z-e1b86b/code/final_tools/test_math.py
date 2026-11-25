import json
from pathlib import Path
import sys
import unittest
from fractions import Fraction
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastpath_math import solve, costs, T, G


class BreakEvenTest(unittest.TestCase):
    def test_requested_cases(self):
        for f in json.loads(Path(__file__).with_name('fixtures.json').read_text()):
            with self.subTest(f['id']):
                r = solve(f['Ae'], f['Ac'], f['te'], f['tc'], batch=4)
                self.assertEqual(r['status'], f['status'])
                self.assertEqual(r['N_win'], f['N'])
                if 'last' in f:
                    self.assertEqual(r['last_advantage_N'], f['last'])
                if f['N']:
                    self.assertEqual(r['effective_images'], 4*f['N'])
                    self.assertEqual(r['K_win'], f['N']+6)

    def test_exact_tie_and_decimal_boundary(self):
        r = solve('0.1', '0.3', '0.03', '0.01')
        self.assertEqual(r['N_win'], 11)
        self.assertEqual(r['difference_at_predecessor_ms'], 0)
        self.assertEqual(solve(10,110,10,5)['difference_at_N_ms'], -5)

    def test_nonfinite_and_negative_times(self):
        for v in [float('nan'),float('inf'),float('-inf')]:
            self.assertEqual(solve(0,v,10,5)['status'], 'invalid')
        self.assertEqual(solve(0,1,-1,5)['status'], 'invalid')

    def test_no_double_count_and_prefix(self):
        r = dict(status='ok', output_valid=True, runner_setup_ms=2, timings=[])
        for phase, values in [('first',[10]),('warmup',[2,3,4,5,6]),('repeat',[7,9])]:
            r['timings'] += [dict(phase=phase,iteration=i,elapsed_ms=v,status='ok',timing_valid=True) for i,v in enumerate(values,1)]
        c = costs(r,5,2)
        self.assertEqual(c['A'],32)
        self.assertEqual(c['t'],8)
        self.assertEqual(G(c,1),12)
        self.assertEqual(G(c,6),32)
        for n in [1,2,21]:
            self.assertEqual(G(c,n+6), T(c,n))
        r['timings'].pop()
        self.assertEqual(costs(r,5,2)['status'], 'insufficient_data')


if __name__ == '__main__':
    unittest.main()
