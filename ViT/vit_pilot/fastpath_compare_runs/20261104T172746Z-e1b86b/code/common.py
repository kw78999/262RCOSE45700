"""Model-independent persistence, timing and process helpers (no device guesses)."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_hash(t):
    return hashlib.sha256(t.contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def write_csv(path, rows, fields):
    path = Path(path)
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()})
    tmp.replace(path)


class Events:
    def __init__(self, path):
        self.path = Path(path)

    def emit(self, event, **kw):
        row = dict(utc=now(), monotonic_ns=time.perf_counter_ns(), pid=os.getpid(), event=event, **kw)
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        with self.path.open('a') as f:
            f.write(line + '\n')
        print(line, flush=True)
        return row


def timed_call(runner, inputs, synchronize):
    """CPU wall time: dispatch + execution + completion wait; excludes prewait."""
    synchronize()
    start = time.perf_counter_ns()
    output = runner(inputs)
    synchronize()
    stop = time.perf_counter_ns()
    elapsed = (stop - start) / 1e6
    return output, elapsed, math.isfinite(elapsed) and elapsed > 0
