import json
from pathlib import Path
import tempfile
import unittest

import torch

from Resnet.artifacts import write_csv
from Resnet.module_profiling import capture_operations, inventory, map_tensors, ReplayNode
from Resnet.Phase_1.module_run import aggregate_modules


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.identity = torch.nn.Identity()

    def forward(self, inputs):
        a = self.relu(inputs)
        b = self.relu(a + 1)
        return torch.cat([self.identity(a), b], dim=1).flatten(1)


class ModuleProfilingTests(unittest.TestCase):
    def test_capture_preserves_pre_inplace_inputs_and_distinct_call_sites(self):
        operations, output = capture_operations(Toy().eval(), -torch.ones(1, 2, 3, 3))
        relus = [item for item in operations if item["module"] == "relu"]
        self.assertEqual(len(relus), 2)
        self.assertNotEqual(relus[0]["node"], relus[1]["node"])
        self.assertTrue((relus[0]["args"][0] < 0).all())
        self.assertEqual(list(output.shape), [1, 36])
        self.assertTrue(any(item["operator"] == "add" for item in operations))
        self.assertTrue(any(item["operator"] == "cat" for item in operations))
        self.assertTrue(any(item["kind"] == "call_method" for item in operations))
        self.assertEqual(sum(item["identity"] for item in operations), 1)
        json.dumps(inventory(operations))

    def test_independent_replay_matches_each_captured_reference(self):
        operations, _ = capture_operations(Toy().eval(), torch.randn(1, 2, 3, 3))
        for item in operations:
            with self.subTest(node=item["node"]):
                function = ReplayNode(item["kind"], item["operation"])
                args, kwargs = map_tensors((item["args"], item["kwargs"]), lambda tensor: tensor.clone())
                torch.testing.assert_close(function(*args, **kwargs), item["reference"])

    def test_partial_summary_exposes_failure_without_fabricating_speedup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fields = ("case", "mode", "cache_intent", "node", "module", "status", "first_ms", "median_ms")
            for mode in ("eager", "compile"):
                rows = [{"case": mode, "mode": mode, "cache_intent": "empty_project_cache", "node": "conv",
                         "module": "conv", "status": "ok" if mode == "eager" else "failed", "first_ms": 3, "median_ms": 1}]
                write_csv(root / "jobs" / mode / "modules.csv", rows, fields)
            aggregate_modules(root)
            import csv
            with (root / "module_comparison.csv").open() as stream:
                comparison = list(csv.DictReader(stream))[0]
            self.assertEqual(comparison["isolated_speedup"], "")
            self.assertEqual(comparison["compile_status"], "failed")


if __name__ == "__main__":
    unittest.main()
