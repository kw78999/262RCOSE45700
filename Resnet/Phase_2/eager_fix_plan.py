"""Turn compile-probe failures into a per-batch Eager batch-partition plan.

For every op that failed at batch B, the smallest module that contains it is run on batch slices:
the Conv / pool / stem activation itself, or the residual block for `add_` and in-block ReLUs
(a block's ReLU module is reused at several shapes). Slices are 8 for B > 8 and 1 otherwise:
Eager batch 8 and batch 1 passed the full model on the NPU, so every sliced op compiles at a
shape already validated on hardware. The plan is only a proposal until a probe of the
partitioned model shows no failure.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch


CHUNK = None   # --chunk: the largest batch whose whole model compiles at this PE fusion


def chunk_for(batch):
    if CHUNK is not None:
        return min(CHUNK, batch)
    return 8 if batch > 8 else 1


def plan_for_batch(batch, failures, model, recorded):
    owners = {}
    for (name, arg_text), entry in recorded.items():
        owners[(name, arg_text)] = set(entry["modules"])
    chosen, sources = set(), []
    for row in failures:
        modules = owners.get((row["op"], row["args"]))
        if not modules:
            raise ValueError(f"b{batch}: failing {row['op']} {row['args'][:80]} is not in the recorded forward")
        for path in modules:
            if not path:
                raise ValueError(f"b{batch}: {row['op']} ran outside every named module")
            target = path
            parent = path.rpartition(".")[0]
            if isinstance(model.get_submodule(path), torch.nn.ReLU) and parent:
                target = parent
            chosen.add(target)
        sources.append({"op": row["op"], "args": row["args"], "reason": row["reason"][:200],
                        "modules": sorted(modules)})
    chosen = sorted(p for p in chosen if not any(p.startswith(q + ".") for q in chosen if q != p))
    return {"batch": batch, "chunk": chunk_for(batch), "modules": chosen, "failures": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, nargs="+", required=True, help="probe.csv files of the base model")
    parser.add_argument("--batches", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=None,
                        help="slice size (default: 8 for batch > 8, else 1, validated at PE fusion 4)")
    args = parser.parse_args()
    global CHUNK
    CHUNK = args.chunk

    from Resnet.models import load_model
    from Resnet.Phase_2.eager_op_probe import NON_COMPUTE, record_ops
    # A later file wins for the same (batch, op, args): rechecks override earlier probe rows.
    latest = {}
    for path in args.probe:
        for r in csv.DictReader(path.open()):
            latest[(int(r["batch"]), r["op"], r["args"])] = r
    rows = list(latest.values())
    plan = {"rule": "smallest module containing each failing op; ReLU inside a block -> the block; "
                    + (f"chunk {args.chunk}" if args.chunk else "chunk 8 for batch > 8, else 1"),
            "num_fusion": sorted({int(r["num_fusion"]) for r in rows}), "batches": {}}
    for batch in args.batches:
        batch_rows = [r for r in rows if int(r["batch"]) == batch]
        compute = {(name, text) for (name, text) in
                   record_ops(load_model(2026, bn_folded=True, output_split_channels=128).to("meta"), batch, 224)
                   if name not in NON_COMPUTE}
        probed = {(r["op"], r["args"]) for r in batch_rows}
        if compute - probed:
            raise ValueError(f"b{batch}: {len(compute - probed)} ops have no probe result yet")
        model = load_model(2026, bn_folded=True, output_split_channels=128).to("meta")
        recorded = record_ops(model, batch, 224)
        failures = [r for r in batch_rows if r["status"] != "OK"]
        plan["batches"][str(batch)] = plan_for_batch(batch, failures, model, recorded)
        entry = plan["batches"][str(batch)]
        print(f"b{batch}: {len(failures)} failing ops -> chunk {entry['chunk']} on {entry['modules']}")
    args.output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
