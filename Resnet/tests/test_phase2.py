import csv
import json
import tempfile
import unittest
from pathlib import Path

from Resnet.Phase_2.run import build_plan, foreign_npu_processes, skip_reason, validate_config
from Resnet.Phase_2.stats import percentile, scaling_efficiency, speedup, summarize, throughput

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "Phase_2" / "config.json").read_text())


class StatisticsTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(percentile(list(range(1, 101)), 95), 95.05)
        self.assertEqual(percentile([7], 95), 7)

    def test_summary_reports_sample_standard_deviation(self):
        result = summarize([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        self.assertEqual(result["n"], 8)
        self.assertAlmostEqual(result["mean_ms"], 5.0)
        self.assertAlmostEqual(result["stdev_ms"], 2.138089935)
        self.assertEqual(result["median_ms"], 4.5)

    def test_invalid_samples_are_rejected(self):
        for values in ([], [0.0], [-1.0], [float("nan")]):
            with self.assertRaises(ValueError):
                summarize(values)

    def test_throughput_speedup_and_scaling_efficiency(self):
        self.assertEqual(throughput(4, 8.0), 500.0)
        self.assertEqual(speedup(16.0, 8.0), 2.0)
        self.assertEqual(scaling_efficiency(8, 10.0, 1, 10.0), 1.0)      # flat latency: linear scaling
        self.assertEqual(scaling_efficiency(8, 80.0, 1, 10.0), 0.125)    # proportional latency: no gain


class PlanTests(unittest.TestCase):
    def test_plan_covers_every_condition_once_and_alternates_mode_order(self):
        validate_config(CONFIG)
        plan = build_plan(CONFIG)
        names = [job["name"] for job in plan]
        self.assertEqual(len(names), len(set(names)))
        core = [job for job in plan if job["phase"] == "core"]
        self.assertEqual(len(core), 2 * len(CONFIG["core_batches"]) * CONFIG["core_trials"])
        self.assertEqual([job["trial"] for job in core], sorted(job["trial"] for job in core))  # trial-major
        first_modes = {(job["batch"], job["trial"]): job["mode"] for job in reversed(plan)}
        self.assertNotEqual(first_modes[(1, 1)], first_modes[(2, 1)])
        self.assertNotEqual(first_modes[(1, 1)], first_modes[(1, 2)])
        extra = [job for job in plan if job["phase"] == "revalidation_extra"]
        self.assertEqual({job["trial"] for job in extra}, {4, 5})
        self.assertEqual({job["batch"] for job in extra}, {CONFIG["revalidation_batch"]})

    def test_failed_condition_is_not_repeated_but_other_batches_continue(self):
        history = {("compile", 16): ["timeout"]}
        self.assertIsNotNone(skip_reason({"phase": "core", "mode": "compile", "batch": 16, "trial": 2}, history, CONFIG))
        self.assertIsNone(skip_reason({"phase": "core", "mode": "compile", "batch": 32, "trial": 1}, history, CONFIG))
        self.assertIsNone(skip_reason({"phase": "core", "mode": "eager", "batch": 16, "trial": 2}, history, CONFIG))

    def test_extension_requires_success_at_the_previous_batch(self):
        job = {"phase": "extension", "mode": "eager", "batch": 64, "trial": 1}
        self.assertIsNone(skip_reason(job, {("eager", 32): ["ok", "ok", "ok"]}, CONFIG))
        self.assertIsNotNone(skip_reason(job, {("eager", 32): ["failed"]}, CONFIG))
        self.assertIsNotNone(skip_reason(job, {}, CONFIG))

    def test_npu_process_table_parsing(self):
        empty = "+-----+----------+-----+\n| PID | NPU core | CMD |\n+-----+----------+-----+\n"
        self.assertEqual(foreign_npu_processes(empty), [])
        self.assertEqual(foreign_npu_processes(empty + "| 123 | npu0:0   | python |\n"), [123])


class AggregateTests(unittest.TestCase):
    def write_job(self, root, mode, batch, trial, latency, status="ok"):
        import torch
        name = f"{mode}-b{batch}-t{trial}"
        directory = root / "jobs" / name
        directory.mkdir(parents=True)
        rows = [dict(job=name, mode=mode, batch=batch, trial=trial, scope="forward_to_cpu", phase="first",
                     iteration=1, elapsed_ms=100000.0)]
        if status == "ok":
            rows += [dict(job=name, mode=mode, batch=batch, trial=trial, scope="forward_to_cpu", phase=phase,
                          iteration=i, elapsed_ms=latency + 0.01 * (i % 3))
                     for phase, count in (("warmup", 10), ("repeat", 50)) for i in range(1, count + 1)]
            rows += [dict(job=name, mode=mode, batch=batch, trial=trial, scope=scope, phase="aux",
                          iteration=i, elapsed_ms=0.2) for scope in ("h2d", "forward_sync", "d2h", "e2e", "idle_synchronize")
                     for i in range(1, 51)]
            torch.save(torch.ones(batch, 1000, dtype=torch.bfloat16), directory / "output.pt")
        with (directory / "samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = dict(status=status, job=name, mode=mode, batch=batch, trial=trial, config=CONFIG, smoke=False,
                      first_call_ms=100000.0, model_load_ms=1.0, model_to_device_ms=1.0, process_elapsed_s=101.0,
                      backend_calls=int(mode == "compile"), stream_idle_after_every_cpu_copy=True,
                      first_vs_last_output_max_abs_diff=0.0, cache_bytes=1)
        if status != "ok":
            result.update(error_type="UnsupportedOpError", error="unsupported batch", traceback="Traceback ...", stage="first_call")
        (directory / "result.json").write_text(json.dumps(result))
        (root / "jobs" / f"{name}.log").write_text("log")

    def test_failed_first_call_never_enters_statistics_and_its_log_is_kept(self):
        from Resnet.Phase_2.analyze import aggregate, read_csv
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(CONFIG))
            (root / "manifest.json").write_text(json.dumps({"plan": build_plan(CONFIG), "skipped": []}))
            for trial in (1, 2, 3):
                self.write_job(root, "eager", 1, trial, 16.0)
                self.write_job(root, "compile", 1, trial, 8.0)
                self.write_job(root, "eager", 2, trial, 16.0)
            self.write_job(root, "compile", 2, 1, 8.0, status="failed")
            tables = aggregate(root)
            self.assertEqual((tables["ok_jobs"], tables["failed_jobs"]), (9, 1))
            scaling = {row["batch"]: row for row in read_csv(root / "batch_scaling.csv")}
            self.assertAlmostEqual(float(scaling["1"]["speedup_mean"]), 16.01 / 8.01, places=3)
            self.assertEqual(scaling["1"]["eager_n"], "150")
            self.assertEqual(scaling["2"]["speedup_mean"], "")           # no compile result: no speedup
            self.assertAlmostEqual(float(scaling["2"]["eager_scaling_efficiency"]), 1.0, places=6)
            self.assertTrue((root / "errors" / "compile-b2-t1.log").exists())
            self.assertTrue((root / "errors" / "compile-b2-t1.traceback.txt").exists())
            coverage = {row["job"]: row["state"] for row in read_csv(root / "coverage.csv")}
            self.assertEqual(coverage["compile-b2-t1"], "failed")
            self.assertEqual(coverage["compile-b4-t1"], "not_run")


if __name__ == "__main__":
    unittest.main()


class FusionConfigTests(unittest.TestCase):
    def test_fusion4_config_is_valid_and_same_plan(self):
        fused = json.loads((Path(__file__).resolve().parents[1] / "Phase_2" / "config-fusion4.json").read_text())
        validate_config(fused)
        self.assertEqual(fused["num_fusion"], 4)
        self.assertEqual(build_plan(fused), build_plan(CONFIG))

    def test_unsupported_fusion_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_config({**CONFIG, "num_fusion": 8})


class PeIsolationTests(unittest.TestCase):
    TABLE = ("| PID   | NPU core    | CMD |\n"
             "| 99178 | npu0:[0, 1] | bert_worker.py |\n"
             "| 99200 | npu0:[2, 5] | other |\n"
             "| 99300 | npu0:[0, 3] | bert f4 |\n")

    def test_fusion4_uses_pes_4_to_7(self):
        from Resnet.Phase_2.run import config_pes
        fused = json.loads((Path(__file__).resolve().parents[1] / "Phase_2" / "config-fusion4.json").read_text())
        self.assertEqual(config_pes(fused), {4, 5, 6, 7})
        self.assertEqual(config_pes(CONFIG), {0})

    def test_only_overlapping_processes_block(self):
        self.assertEqual(foreign_npu_processes(self.TABLE, {6, 7}), [])
        self.assertEqual(foreign_npu_processes(self.TABLE, {4, 5, 6, 7}), [99200])
        self.assertEqual(foreign_npu_processes(self.TABLE, {1}), [99178, 99300])
        self.assertEqual(foreign_npu_processes(self.TABLE, {2}), [99200, 99300])
        self.assertEqual(foreign_npu_processes(self.TABLE), [99178, 99200, 99300])

    def test_unparsed_core_column_counts_as_overlap(self):
        self.assertEqual(foreign_npu_processes("| 1 | npu0:? | x |", {4}), [1])
