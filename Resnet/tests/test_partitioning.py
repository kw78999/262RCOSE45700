import unittest

import torch

from Resnet.folding import state_fingerprint
from Resnet.partitioning import OutputPartitionConv, partition_stage4


class PartitioningTests(unittest.TestCase):
    def test_all_input_channels_and_bias_are_preserved_at_both_strides(self):
        torch.manual_seed(17)
        for stride in (1, 2):
            with self.subTest(stride=stride):
                conv = torch.nn.Conv2d(5, 7, 3, stride=stride, padding=1).eval()
                fingerprint = state_fingerprint(conv)
                split = OutputPartitionConv(conv, chunk=3)
                inputs = torch.randn(2, 5, 13, 13)
                with torch.inference_mode():
                    torch.testing.assert_close(split(inputs), conv(inputs), rtol=1e-5, atol=1e-5)
                self.assertEqual([part.in_channels for part in split.parts], [5, 5, 5])
                self.assertEqual([part.out_channels for part in split.parts], [3, 3, 1])
                self.assertEqual(state_fingerprint(conv), fingerprint)

    def test_invalid_chunk_grouped_conv_and_wrong_resnet_structure_are_rejected(self):
        for chunk in (0, -1, True):
            with self.assertRaises(ValueError):
                OutputPartitionConv(torch.nn.Conv2d(2, 2, 3), chunk)
        with self.assertRaises(ValueError):
            OutputPartitionConv(torch.nn.Conv2d(2, 2, 3, groups=2))
        with self.assertRaises(AttributeError):
            partition_stage4(torch.nn.Identity())


class BatchPartitionTests(unittest.TestCase):
    def test_conv_pool_and_residual_block_match_and_keep_state(self):
        from torchvision.models.resnet import Bottleneck
        from Resnet.partitioning import partition_batch
        torch.manual_seed(5)
        block = Bottleneck(8, 2, downsample=torch.nn.Sequential(torch.nn.Conv2d(8, 8, 1)),
                           norm_layer=lambda c: torch.nn.Identity())
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                                    torch.nn.MaxPool2d(3, 2, 1), block,
                                    torch.nn.AdaptiveAvgPool2d(1)).eval()
        fingerprint = state_fingerprint(model)
        split, info = partition_batch(model, ["0", "1", "2", "3", "4"], chunk=3)
        self.assertEqual(state_fingerprint(split), fingerprint)
        self.assertEqual([m["type"] for m in info["modules"]],
                         ["Conv2d", "ReLU", "MaxPool2d", "Bottleneck", "AdaptiveAvgPool2d"])
        for batch in (1, 3, 7):
            with self.subTest(batch=batch), torch.inference_mode():
                inputs = torch.randn(batch, 3, 12, 12)
                torch.testing.assert_close(split(inputs), model(inputs), rtol=1e-5, atol=1e-5)

    def test_rejects_bad_chunk_nested_paths_training_and_batchnorm(self):
        from Resnet.partitioning import partition_batch
        model = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Conv2d(2, 2, 1))).eval()
        for chunk in (0, -2, True):
            with self.assertRaises(ValueError):
                partition_batch(model, ["0"], chunk)
        with self.assertRaises(ValueError):
            partition_batch(model, ["0", "0.0"], 2)
        with self.assertRaises(ValueError):
            partition_batch(torch.nn.Sequential(torch.nn.Conv2d(2, 2, 1)).train(), ["0"], 2)
        with self.assertRaises(ValueError):
            partition_batch(torch.nn.Sequential(torch.nn.BatchNorm2d(2)).eval(), ["0"], 2)
        split, _ = partition_batch(model, ["0"], 2, inplace=True)
        with self.assertRaises(ValueError):
            partition_batch(split, ["0"], 2, inplace=True)
