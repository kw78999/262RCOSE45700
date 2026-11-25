"""Compile-only check of model-side rewrites for one Conv the Eager backend rejects.

Each variant replaces one aten conv2d by mathematically identical smaller convs (output-channel
split, or batch split followed by concat). Every resulting conv is compiled alone for the
configured PE fusion, exactly as Eager would. No NPU is opened.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--in-ch", type=int, required=True)
    parser.add_argument("--out-ch", type=int, required=True)
    parser.add_argument("--hw", type=int, required=True, help="input spatial size")
    parser.add_argument("--kernel", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--out-splits", type=int, nargs="*", default=(), help="output-channel chunk sizes")
    parser.add_argument("--batch-splits", type=int, nargs="*", default=(), help="batch chunk sizes")
    parser.add_argument("--num-fusion", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import furiosa.torch as ft
    from furiosa.torch._backend.aten import AtenOp
    if args.num_fusion != 1:
        ft.set_fusion(args.num_fusion)
    ft.config.set_cpu_fallback(False)

    def probe(label, batch, out_ch):
        x = torch.empty(batch, args.in_ch, args.hw, args.hw, dtype=torch.bfloat16, device="meta")
        w = torch.empty(out_ch, args.in_ch, args.kernel, args.kernel, dtype=torch.bfloat16, device="meta")
        b = torch.empty(out_ch, dtype=torch.bfloat16, device="meta")
        conv_args = (x, w, b, [args.stride] * 2, [args.padding] * 2, [1, 1], 1)
        started = time.perf_counter()
        with ft.config.cache_context(str((args.output / "cache" / label).resolve())):
            result = AtenOp("aten::conv2d", torch.ops.aten.conv2d.default, conv_args, {}).compile_raw()
        failed = isinstance(result.cached, Exception)
        reason = ""
        if failed:
            root = result.cached
            while root.__cause__ is not None:
                root = root.__cause__
            lines = [l.strip() for l in str(root).splitlines() if l.strip()]
            reason = lines[lines.index("Caused by:") + 1] if "Caused by:" in lines else lines[-1]
        return dict(variant=label, batch=batch, in_ch=args.in_ch, out_ch=out_ch, hw=args.hw,
                    kernel=args.kernel, stride=args.stride, status="FAIL" if failed else "OK",
                    reason=reason, compile_s=f"{time.perf_counter() - started:.1f}")

    plan = [("original", args.batch, args.out_ch)]
    plan += [(f"out_split{c}", args.batch, c) for c in args.out_splits]
    plan += [(f"batch_split{c}", c, args.out_ch) for c in args.batch_splits]
    path = args.output / "variants.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = None
        for label, batch, out_ch in plan:
            row = probe(f"b{args.batch}-{label}", batch, out_ch)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()
            print(row, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
