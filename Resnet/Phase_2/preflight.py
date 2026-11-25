"""Environment readiness check for Phase 2. Not a measurement; produces no latency result.

Checks the cross-compiler headers, the device, the pretrained checkpoint and model variant,
and that one small BF16 Conv2d compiles from an empty cache and runs in both modes with
CPU fallback disabled. The full ResNet50 is NOT compiled here.
"""
import argparse
import json
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from Resnet.artifacts import write_json
from Resnet.Phase_2.run import device_snapshot, foreign_npu_processes, toolchain_preflight

PROJECT = Path(__file__).resolve().parents[1]


def small_compile(mode, cache):
    import torch
    import furiosa.torch as ft
    from Resnet.runtime import make_callable, synchronize
    ft.config.set_cpu_fallback(False)
    torch._dynamo.config.suppress_errors = False
    device = torch.device("furiosa:0")
    torch.manual_seed(2026)
    module = torch.nn.Conv2d(3, 8, 3, padding=1).eval().to(torch.bfloat16)
    cpu_input = torch.randn(1, 3, 16, 16).to(torch.bfloat16)
    with torch.inference_mode():
        reference = module(cpu_input)
        function, counter = make_callable(module.to(device), mode)
        with ft.config.cache_context(str(cache)):
            started = time.perf_counter()
            output = function(cpu_input.to(device))
            synchronize(device)
            elapsed = time.perf_counter() - started
            cpu_output = output.to("cpu")
    return {"status": "ok", "first_call_s": elapsed, "backend_calls": counter.calls,
            "output_shape": list(cpu_output.shape), "output_dtype": str(cpu_output.dtype),
            "max_abs_error_vs_cpu_bf16": (cpu_output.float() - reference.float()).abs().max().item(),
            "cache_files": sum(1 for p in Path(cache).rglob("*") if p.is_file())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("evidence") / "preflight.json")
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    report = {"time_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
              "scope": "environment readiness only; no ResNet50 NPU compile and no latency measurement",
              "checks": {}}
    checks = report["checks"]
    checks["cross_compiler_libc_headers"] = toolchain_preflight()

    import importlib.metadata as metadata
    import torch
    import furiosa.torch as ft
    checks["packages"] = {"ok": True, **{name: metadata.version(name) for name in
                                         ("torch", "torchvision", "furiosa-torch", "furiosa-kernels", "numpy")}}
    snapshot = device_snapshot()
    checks["device"] = {"ok": bool(ft.is_available()) and config["device"] in map(str, ft.get_available_devices())
                        and not foreign_npu_processes(snapshot["ps"]),
                        "configuration": str(ft.get_device_configuration()),
                        "other_npu_processes": foreign_npu_processes(snapshot["ps"]), "info": snapshot["info"]}

    try:
        from Resnet.models import load_model, weight_metadata
        model = load_model(config["seed"], bn_folded=config["bn_folded"],
                           output_split_channels=config["output_split_channels"])
        state = model._benchmark_transform["state_sha256"]
        checks["model_variant"] = {"ok": state == config["expected_state_sha256"], "state_sha256": state,
                                   "variant": model._benchmark_transform["name"], "weights": weight_metadata()}
    except Exception:
        checks["model_variant"] = {"ok": False, "error": traceback.format_exc()}

    for mode in ("eager", "compile"):
        with tempfile.TemporaryDirectory(prefix=f"phase2-preflight-{mode}-") as cache:
            try:
                checks[f"small_fresh_cache_compile_{mode}"] = {"ok": True, **small_compile(mode, cache)}
            except Exception:
                checks[f"small_fresh_cache_compile_{mode}"] = {"ok": False, "error": traceback.format_exc()}

    plot_python = Path("/root/ViT/vit_pilot/.plot-venv/bin/python")
    checks["plot_interpreter"] = {"ok": plot_python.exists(), "path": str(plot_python),
                                  "note": "only needed for figures; reads CSV tables"}
    report["ready"] = all(check["ok"] for check in checks.values())
    write_json(args.output, report)
    print(json.dumps({"ready": report["ready"], **{name: check["ok"] for name, check in checks.items()}}, indent=1))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
