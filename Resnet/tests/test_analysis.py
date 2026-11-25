import math
import unittest

from Resnet.analysis import break_even, compare_tensors, statistics_ms


class TimingAnalysisTests(unittest.TestCase):
    def test_integer_boundary_distinguishes_tie_from_first_compile_win(self):
        result = break_even(10, 30, 8, 4)
        self.assertEqual(result["boundary_calls"], 5)
        self.assertEqual(result["boundary_ceil_calls"], 5)
        self.assertEqual(result["first_strict_compile_win_calls"], 6)
        self.assertEqual(10 + 5 * 8, 30 + 5 * 4)
        self.assertGreater(10 + 6 * 8, 30 + 6 * 4)

    def test_no_repeated_execution_gain_is_not_divided_by_zero(self):
        for compiled in (8, 10):
            result = break_even(0, 20, 8, compiled)
            self.assertEqual(result["status"], "no_steady_state_gain")
            self.assertIsNone(result["first_strict_compile_win_calls"])

    def test_fractional_and_negative_boundaries(self):
        self.assertEqual(break_even(0, 21, 8, 4)["first_strict_compile_win_calls"], 6)
        self.assertEqual(break_even(30, 0, 8, 4)["first_strict_compile_win_calls"], 1)

    def test_invalid_or_empty_measurements_cannot_be_summarized(self):
        for values in ([], [0], [-1], [math.inf], [math.nan]):
            with self.assertRaises(ValueError):
                statistics_ms(values)

    def test_median_and_range_preserve_outlier(self):
        result = statistics_ms([1, 2, 100])
        self.assertEqual(result["median_ms"], 2)
        self.assertEqual((result["min_ms"], result["max_ms"], result["n"]), (1, 100, 3))

    def test_prediction_disagreement_and_no_arbitrary_pass_threshold(self):
        import torch
        result = compare_tensors(torch.tensor([[1., 2.], [4., 3.]]),
                                 torch.tensor([[2., 1.], [4., 3.]]))
        self.assertEqual(result["prediction_agreement"], 0.5)
        self.assertEqual(result["max_abs_error"], 1)
        self.assertEqual(result["status"], "measured")
        self.assertNotIn("passed", result)

    def test_zero_vector_cosine_is_explicitly_undefined(self):
        import torch
        result = compare_tensors(torch.zeros(1, 3), torch.zeros(1, 3))
        self.assertIsNone(result["cosine_min"])
        self.assertEqual(result["cosine_undefined_samples"], 1)

    def test_identical_large_conv_maps_have_unit_cosine(self):
        import torch
        generator = torch.Generator().manual_seed(123)
        feature_map = torch.randn(1, 64, 112, 112, generator=generator).to(torch.bfloat16)
        result = compare_tensors(feature_map, feature_map.clone(), classifier=False)
        self.assertEqual(result["max_abs_error"], 0)
        self.assertAlmostEqual(result["cosine_mean"], 1, places=12)
        self.assertIsNone(result["prediction_agreement"])

    def test_shape_and_nonfinite_outputs_are_not_compared(self):
        import torch
        self.assertEqual(compare_tensors(torch.zeros(1, 2), torch.zeros(1, 3))["status"], "shape_mismatch")
        self.assertEqual(compare_tensors(torch.tensor([[math.nan]]), torch.zeros(1, 1))["status"], "nonfinite")


if __name__ == "__main__":
    unittest.main()
