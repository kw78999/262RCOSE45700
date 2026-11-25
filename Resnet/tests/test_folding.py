import unittest

import torch

from Resnet.folding import fold_batchnorm, state_fingerprint


class FoldingTests(unittest.TestCase):
    def test_folding_preserves_nontrivial_eval_batchnorm_and_original(self):
        torch.manual_seed(5)
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1, bias=False),
                                    torch.nn.BatchNorm2d(4), torch.nn.ReLU()).eval()
        with torch.no_grad():
            model[1].running_mean.copy_(torch.randn(4))
            model[1].running_var.copy_(torch.rand(4) + 0.2)
            model[1].weight.copy_(torch.randn(4))
            model[1].bias.copy_(torch.randn(4))
        original_hash = state_fingerprint(model)
        folded, metadata = fold_batchnorm(model)
        inputs = torch.randn(2, 3, 8, 8)
        with torch.inference_mode():
            torch.testing.assert_close(folded(inputs), model(inputs), rtol=1e-5, atol=1e-5)
        self.assertEqual(metadata["pair_count"], 1)
        self.assertIsInstance(folded[1], torch.nn.Identity)
        self.assertIsInstance(model[1], torch.nn.BatchNorm2d)
        self.assertEqual(state_fingerprint(model), original_hash)
        self.assertIsNotNone(folded[0].bias)

    def test_training_low_precision_and_unmatched_batchnorm_are_rejected(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4))
        with self.assertRaises(ValueError):
            fold_batchnorm(model)
        with self.assertRaises(ValueError):
            fold_batchnorm(model.eval().bfloat16())
        with self.assertRaises(ValueError):
            fold_batchnorm(torch.nn.Sequential(torch.nn.BatchNorm2d(3)).eval())


if __name__ == "__main__":
    unittest.main()
