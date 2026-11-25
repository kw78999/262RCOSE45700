import csv
import json
import tempfile
import unittest
from pathlib import Path

from Resnet.artifacts import write_csv, write_json
from Resnet.Phase_1.aggregate import aggregate
from Resnet.Phase_1.run import validate_config, cold_job_succeeded


class ArtifactIntegrityTests(unittest.TestCase):
    def test_failed_csv_update_preserves_last_complete_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            write_csv(path, [{"latency": 1}], ("latency",))
            original = path.read_bytes()
            with self.assertRaises(AttributeError):
                write_csv(path, [None], ("latency",))
            self.assertEqual(path.read_bytes(), original)

    def test_failed_partial_runs_do_not_produce_performance_or_break_even(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "jobs" / "eager-b1-cold"
            write_json(job / "result.json", {"mode": "eager", "target": "resnet50", "status": "failed",
                       "stage": "first", "error_type": "RuntimeError", "error": "unsupported BN"})
            write_csv(job / "samples.csv", [{"phase": "measure", "elapsed_ms": 1}], ("phase", "elapsed_ms"))
            result = aggregate(root)
            self.assertEqual(result, {"summary_rows": 0, "failures": 1, "comparisons": 0})
            with (root / "summary.csv").open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            with (root / "amortization_estimates.csv").open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            self.assertEqual(json.loads((root / "correctness.json").read_text()), [])

    def test_config_rejects_silent_precision_model_or_shape_substitution(self):
        config = json.loads(Path(__file__).parents[1].joinpath("Phase_1/config.json").read_text())
        validate_config(config)
        for changes in ({"precision": "float32"}, {"weights": "DEFAULT"}, {"iterations": 0},
                        {"batches": [1, 1]}, {"device": "cpu"}, {"mixed_batches": [999]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({**config, **changes})

    def test_json_rejects_nan_instead_of_publishing_invalid_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_json(Path(directory) / "bad.json", {"value": float("nan")})

    def test_different_folded_weights_cannot_be_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("eager", "compile"):
                job = root / "jobs" / mode
                write_json(job / "result.json", {
                    "mode": mode, "target": "resnet50", "status": "ok", "workload": "fixed",
                    "cache_intent": "empty_project_cache", "batch": 1,
                    "weights": {"sha256": "same-original-checkpoint"}, "config": {"bn_folded": True},
                    "model_transform": {"name": "conv_bn_eval_folded", "state_sha256": mode},
                })
                write_csv(job / "samples.csv", [], ("phase", "elapsed_ms"))
            with self.assertRaisesRegex(ValueError, "different model transformations"):
                aggregate(root)

    def test_cached_failure_file_does_not_qualify_for_warm_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "caches/eager-b1/key/output.bin"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"cached compiler failure")
            result = root / "jobs/eager-b1-cold/result.json"
            write_json(result, {"status": "failed"})
            self.assertFalse(cold_job_succeeded(root, "eager", "b1"))
            write_json(result, {"status": "ok"})
            self.assertTrue(cold_job_succeeded(root, "eager", "b1"))

    def test_split_configuration_requires_folded_model_and_positive_timeouts(self):
        config = json.loads(Path(__file__).parents[1].joinpath("Phase_1/config-bn-folded-split128.json").read_text())
        validate_config(config)
        for changes in ({"bn_folded": False}, {"output_split_channels": 256},
                        {"output_split_channels": True}, {"mixed_timeout_s": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({**config, **changes})


if __name__ == "__main__":
    unittest.main()
