"""Compile-only probe of every ATen op the Eager backend would compile, per batch.

Eager compiles each dispatched ATen op as its own program, so one rejected op stops the whole
forward and hides every later op. This probe records the ops of one forward with shape-only
(meta) tensors, then compiles each unique (op, shapes, arguments) for the configured PE fusion.
No NPU is opened and nothing is measured: it answers only "which op fails to compile, and why".
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

# Ops the Eager backend does not compile into a tensor program (metadata or aliasing only).
NON_COMPUTE = {"aten::view", "aten::_unsafe_view", "aten::t", "aten::detach", "aten::alias",
               "aten::expand", "aten::permute", "aten::transpose", "aten::reshape"}


def op_name(func):
    return str(func).replace("aten.", "aten::").removesuffix(".default")


def describe(value):
    if isinstance(value, torch.Tensor):
        return f"{tuple(value.shape)}:{str(value.dtype).removeprefix('torch.')}:{tuple(value.stride())}"
    return repr(value)


class Recorder(TorchDispatchMode):
    def __init__(self, module_stack):
        super().__init__()
        self.module_stack = module_stack
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.calls.append((func, args, kwargs, self.module_stack[-1] if self.module_stack else ""))
        return func(*args, **kwargs)


def record_ops(model, batch, image_size):
    stack = []

    def enter(name):
        def hook(module, inputs):
            stack.append(name)
        return hook

    def leave(module, inputs, output):
        stack.pop()   # returning None keeps the module output unchanged

    for name, module in model.named_modules():
        if name:   # innermost active module: a leaf, or the residual block for add_
            module.register_forward_pre_hook(enter(name))
            module.register_forward_hook(leave)
    x = torch.empty(batch, 3, image_size, image_size, dtype=torch.bfloat16, device="meta")
    recorder = Recorder(stack)
    with torch.inference_mode(), recorder:
        model(x)
    unique = {}
    for func, args, kwargs, module in recorder.calls:
        key = (op_name(func), json.dumps(pytree.tree_map(describe, (args, kwargs)), default=str))
        entry = unique.setdefault(key, {"func": func, "args": args, "kwargs": kwargs, "modules": []})
        entry["modules"].append(module)
    return unique


def root_reason(error):
    root = error
    while root.__cause__ is not None:
        root = root.__cause__
    lines = [line.strip() for line in str(root).splitlines() if line.strip()]
    reason = next((l for l in lines if "Caused by" not in l and not l.startswith("(cache at")), "")
    if "Caused by:" in lines:
        reason = " | ".join(lines[lines.index("Caused by:") + 1:lines.index("Location:")]
                            if "Location:" in lines else lines[lines.index("Caused by:") + 1:])
    location = lines[lines.index("Location:") + 1] if "Location:" in lines else ""
    return type(error).__name__, reason, location


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", required=True)
    parser.add_argument("--num-fusion", type=int, default=4)
    parser.add_argument("--only", default=None, help="probe only this ATen op name (e.g. aten::convolution)")
    parser.add_argument("--split-channels", type=int, default=None,
                        help="also apply output-channel split to the convs listed in --split-modules")
    parser.add_argument("--split-modules", nargs="*", default=())
    parser.add_argument("--batch-chunk", type=int, default=None,
                        help="run the convs in --batch-chunk-modules on batch slices of this size")
    parser.add_argument("--batch-chunk-modules", nargs="*", default=())
    parser.add_argument("--plan", type=Path, default=None,
                        help="eager_batch_partition.json: apply its per-batch module slicing")
    parser.add_argument("--device-index", type=int, default=0,
                        help="Furiosa device the fake tensors refer to (only one probe per device)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import furiosa.torch as ft
    from furiosa.torch._backend.aten import AtenOp
    from Resnet.models import load_model
    if args.num_fusion != 1:
        ft.set_fusion(args.num_fusion)
    ft.config.set_cpu_fallback(False)
    plan = json.loads(args.plan.read_text())["batches"] if args.plan else {}
    with torch.furiosa.device(torch.device(f"furiosa:{args.device_index}")):   # SDK compares on exit
        return probe_batches(args, ft, AtenOp, load_model, plan)


def probe_batches(args, ft, AtenOp, load_model, plan):
    rows_path = args.output / "probe.csv"
    fields = ("batch", "num_fusion", "op", "modules", "args", "status", "error_type", "reason",
              "location", "compile_s")
    done = set()
    if rows_path.exists() and rows_path.stat().st_size > 0:
        with rows_path.open() as f:
            done = {(int(r["batch"]), r["op"], r["args"]) for r in csv.DictReader(f)}
    new_file = not rows_path.exists() or rows_path.stat().st_size == 0
    with rows_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for batch in args.batches:
            model = load_model(2026, bn_folded=True, output_split_channels=128)
            if args.split_channels:
                from Resnet.partitioning import OutputPartitionConv
                for path in args.split_modules:
                    parent, _, leaf = path.rpartition(".")
                    owner = model.get_submodule(parent)
                    setattr(owner, leaf, OutputPartitionConv(getattr(owner, leaf), args.split_channels))
            if args.batch_chunk:
                from Resnet.partitioning import partition_batch
                model, _ = partition_batch(model, args.batch_chunk_modules, args.batch_chunk, inplace=True)
            if str(batch) in plan:
                from Resnet.partitioning import partition_batch
                entry = plan[str(batch)]
                model, _ = partition_batch(model, entry["modules"], entry["chunk"], inplace=True)
            model = model.to("meta")
            for (name, arg_text), entry in record_ops(model, batch, 224).items():
                if name in NON_COMPUTE or (args.only and name != args.only) or (batch, name, arg_text) in done:
                    continue
                cache = args.output / "cache" / f"b{batch}"
                started = time.perf_counter()
                # Fresh plain meta tensors (shape/stride/dtype only), as the SDK dryrun probe does:
                # recorded Parameters and views otherwise reach export in a form Eager never sends.
                fresh = pytree.tree_map(
                    lambda v: torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device="meta")
                    if isinstance(v, torch.Tensor) else v, (entry["args"], entry["kwargs"]))
                with ft.config.cache_context(str(cache.resolve())):
                    result = AtenOp(name, entry["func"], *fresh).compile_raw()
                elapsed = time.perf_counter() - started
                failed = isinstance(result.cached, Exception)
                error_type, reason, location = root_reason(result.cached) if failed else ("", "", "")
                if failed and not reason:
                    import traceback
                    reason = " ".join("".join(traceback.format_exception(result.cached)).split())[-400:]
                modules = sorted(set(entry["modules"]))
                writer.writerow(dict(batch=batch, num_fusion=args.num_fusion, op=name,
                                     modules=";".join(modules), args=arg_text,
                                     status="FAIL" if failed else "OK", error_type=error_type,
                                     reason=reason, location=location, compile_s=f"{elapsed:.1f}"))
                f.flush()
                print(f"b{batch} {name} {modules[0] if modules else ''} (+{len(modules) - 1}) "
                      f"{'FAIL ' + reason if failed else 'OK'} {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
