"""Diagnostic A/B: does the Eager load-cache size (SDK LRU, 32 entries) set the ~13 ms Eager latency?

Runs one model in one process on a copied compiler cache and alternates the load-cache size
(32 = the size the SDK module actually uses, 2048 = the documented default of
FURIOSA_TORCH_EAGER_NUM_LOAD) in blocks, so drift affects both arms alike. Not a measurement under
the Phase 2 protocol (cache is pre-populated); it only tests the cause.
"""
import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch
import furiosa.torch as ft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--device", default="furiosa:0")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 2048, 32, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    from Resnet.models import load_model, make_input
    from Resnet.runtime import eager_cache_stats, make_callable
    from Resnet.Phase_2.measure import forward_to_cpu

    cache = args.work / f"cache-b{args.batch}"
    if not cache.exists():
        shutil.copytree(args.source_cache, cache)
    torch.set_num_threads(1)
    ft.set_fusion(4)
    ft.config.set_cpu_fallback(False)
    device = torch.device(args.device)
    model = load_model(2026, bn_folded=True, output_split_channels=128).to(device)
    function, _ = make_callable(model, "eager")
    x = make_input(args.batch, 224, 2026).to(device)
    blocks = []
    with ft.config.cache_context(str(cache.resolve())), torch.inference_mode():
        started = time.perf_counter()
        forward_to_cpu(function, x, device)
        first_s = time.perf_counter() - started
        for size in args.sizes:
            ft.config.set_eager_cache_size(num_load=size)   # clears the load cache
            for _ in range(args.warmup):
                forward_to_cpu(function, x, device)
            before = eager_cache_stats()["_compile_and_load_cached"]
            times = [forward_to_cpu(function, x, device)[0] for _ in range(args.repeats)]
            after = eager_cache_stats()["_compile_and_load_cached"]
            blocks.append({"num_load": size, "mean_ms": statistics.fmean(times),
                           "median_ms": statistics.median(times), "stdev_ms": statistics.stdev(times),
                           "load_misses_per_forward": (after["misses"] - before["misses"]) / args.repeats,
                           "load_hits_per_forward": (after["hits"] - before["hits"]) / args.repeats})
            print(json.dumps(blocks[-1]), flush=True)
    out = {"batch": args.batch, "device": args.device, "first_call_s": first_s, "blocks": blocks,
           "note": "diagnostic, copied cache; not a Phase 2 measurement"}
    (args.work / f"lru-ab-b{args.batch}.json").write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
