import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from Resnet.Phase_1.budget import supervise


class BudgetTests(unittest.TestCase):
    def test_runner_crash_does_not_leave_independent_worker_running(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "run"
            root.mkdir()
            child = [sys.executable, "-c", "import time; time.sleep(120)"]
            code = ("import subprocess,time;from pathlib import Path;"
                    f"p=subprocess.Popen({child!r},start_new_session=True);"
                    f"Path({str(root / 'pid')!r}).write_text(str(p.pid));"
                    "time.sleep(1);raise SystemExit(1)")
            record = supervise([sys.executable, "-c", code], root, 15, os.environ.copy())
            self.assertEqual(record["exit_code"], 1)
            pid = (root / "pid").read_text()
            state = subprocess.run(["ps", "-p", pid, "-o", "stat="], capture_output=True, text=True).stdout.strip()
            self.assertTrue(not state or state.startswith("Z"), state)

    def test_short_successful_run_returns_without_waiting_for_deadline(self):
        with tempfile.TemporaryDirectory() as folder:
            record = supervise([sys.executable, "-c", "print('done')"],
                               Path(folder) / "run", 15, os.environ.copy())
            self.assertEqual(record["status"], "finished")
            self.assertEqual(record["exit_code"], 0)
            self.assertLess(record["elapsed_s"], 5)

    def test_budget_stops_worker_in_separate_session_and_preserves_completed_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "run"
            root.mkdir()
            command = [sys.executable, "-c", "import time; time.sleep(120)"]
            code = ("import subprocess,time,json;from pathlib import Path;"
                    f"r=Path({str(root)!r});"
                    f"p=subprocess.Popen({command!r},start_new_session=True);"
                    "(r/'worker.json').write_text(json.dumps({'pid':p.pid}));"
                    "(r/'manifest.json').write_text(json.dumps({'target':'resnet50',"
                    "'started_kst':'test','status':'running','jobs':["
                    "{'name':'done','status':'ok','elapsed_s':1},"
                    "{'name':'active','status':'running','worker_pid':p.pid}], 'skipped':[]}));"
                    "time.sleep(120)")
            record = supervise([sys.executable, "-c", code], root, 15, os.environ.copy())
            self.assertEqual(record["status"], "time_budget_exceeded")
            self.assertLess(record["elapsed_s"], 15)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["jobs"][0]["status"], "ok")
            self.assertEqual(manifest["jobs"][1]["status"], "time_budget_exceeded")
            pid = json.loads((root / "worker.json").read_text())["pid"]
            state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                                   capture_output=True, text=True).stdout.strip()
            self.assertTrue(not state or state.startswith("Z"), state)


if __name__ == "__main__":
    unittest.main()
