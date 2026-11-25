"""Read Chrome traces without assuming every host span is a device kernel."""
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from Resnet.artifacts import write_csv, write_json


def summarize_traces(root):
    rows = []
    notes = []
    for path in sorted(Path(root).rglob("*trace.json")):
        payload = json.loads(path.read_text())
        events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
        durations = defaultdict(list)
        for event in events:
            if event.get("ph") == "X" and isinstance(event.get("dur"), (float, int)):
                durations[(event.get("cat", ""), event.get("name", ""))].append(event["dur"])
        for (category, name), values in sorted(durations.items()):
            rows.append({"trace": str(path.relative_to(root)), "category": category, "name": name,
                         "count": len(values), "sum_us": sum(values), "max_us": max(values),
                         "is_named_npu_task": "::Task::" in name})
        notes.append({"trace": str(path.relative_to(root)), "event_count": len(events),
                      "phases": dict(Counter(event.get("ph", "") for event in events)),
                      "named_npu_tasks": sum("::Task::" in event.get("name", "") for event in events)})
    write_csv(Path(root) / "trace_spans.csv", rows,
              ("trace", "category", "name", "count", "sum_us", "max_us", "is_named_npu_task"))
    write_json(Path(root) / "trace_notes.json", {
        "traces": notes,
        "units": "Chrome trace duration in microseconds",
        "limit": "Spans may nest or overlap; sums are not end-to-end latency. Named NPU tasks are identified from trace labels, not assumed for all events.",
        "fusion": "A separately compiled Conv or block changes fusion boundaries; module times cannot be added to reconstruct compiled ResNet50",
    })
    return notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    print(json.dumps(summarize_traces(parser.parse_args().run_dir), ensure_ascii=False, indent=2))
