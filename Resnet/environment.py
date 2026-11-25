"""Version/device evidence kept alongside each experiment, without secrets."""
import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from Resnet.artifacts import write_json


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {"command": command, "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def collect(destination, include_installed_sources=True):
    import torch
    import furiosa.torch as ft
    destination = Path(destination)
    packages = {}
    for name in ("torch", "torchvision", "furiosa-torch", "furiosa-kernels", "numpy", "pillow"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "time_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "python_executable": sys.executable, "python": sys.version,
        "platform": platform.platform(), "packages": packages,
        "furiosa_build": str(ft.version()), "available": ft.is_available(),
        "devices": list(map(str, ft.get_available_devices())),
        "device_configuration": str(ft.get_device_configuration()),
        "dynamo_automatic_dynamic_shapes_after_sdk_import": torch._dynamo.config.automatic_dynamic_shapes,
        "cpu_fallback_default": ft.config.get_cpu_fallback(),
        "cache_default": str(ft.config.get_cache_config()),
        "compiler_default": str(ft.config.get_compiler_config()),
        "num_threads_before_worker_override": torch.get_num_threads(),
        "environment": {key: os.environ[key] for key in (
            "FURIOSA_TORCH_LOG", "FURIOSA_TORCH_PROFILE", "TORCH_LOGS",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RUST_LOG") if key in os.environ},
        "device_info": command_output(["furiosa-smi", "info"]),
        "device_status": command_output(["furiosa-smi", "status"]),
        "device_processes": command_output(["furiosa-smi", "ps"]),
        "device_versions": command_output(["furiosa-smi", "version"]),
    }
    write_json(destination / "environment.json", result)
    if not include_installed_sources:
        return result
    root = Path(ft.__file__).parent
    paths = ["_register.py", "_backend/eager.py", "_backend/aten.py",
             "_backend/torch_compile.py", "config/_cache.py", "config/_cpu_fallback.py",
             "_C/stream/__init__.pyi", "_C/profiler/__init__.pyi", "_C/compiler/__init__.pyi"]
    with (destination / "installed-sdk-sources.txt").open("w") as stream:
        for relative in paths:
            path = root / relative
            stream.write(f"\n### {path}\n")
            for index, line in enumerate(path.read_text().splitlines(), 1):
                stream.write(f"{index:4}: {line}\n")
    import importlib
    source_path = Path(importlib.import_module("torchvision.models.resnet").__file__)
    (destination / "installed-torchvision-resnet.py").write_text(source_path.read_text())
    return result
