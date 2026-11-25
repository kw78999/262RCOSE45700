"""Official stream completion, mode selection and separately measured scopes."""
import time

import torch
import furiosa.torch as ft


class BackendCounter:
    """Counts TorchDynamo backend entries, not compiler code-generation runs."""

    def __init__(self):
        self.calls = 0

    def __call__(self, graph, example_inputs):
        self.calls += 1
        return ft.backend(graph, example_inputs)


def make_callable(model, mode):
    counter = BackendCounter()
    if mode == "eager":
        return model, counter
    if mode != "compile":
        raise ValueError(mode)
    # SDK defaults: no dynamic/fullgraph/mode/options or decomposition overrides.
    return torch.compile(model, backend=counter), counter


def synchronize(device):
    ft.current_stream(device).synchronize()


def measure_e2e(function, cpu_input, device):
    """H2D + forward + completion + D2H; generation/preprocessing excluded."""
    synchronize(device)
    started = time.perf_counter_ns()
    device_input = cpu_input.to(device)
    output = function(device_input)
    synchronize(device)
    cpu_output = output.to("cpu", non_blocking=False)
    synchronize(device)
    elapsed = (time.perf_counter_ns() - started) / 1e6
    if output.device.type != "furiosa":
        raise RuntimeError(f"Unexpected output device: {output.device}")
    return elapsed, cpu_output


def measure_forward(function, device_input, device):
    """Host wall-clock forward + stream wait, with input already on NPU."""
    synchronize(device)
    started = time.perf_counter_ns()
    output = function(device_input)
    synchronize(device)
    elapsed = (time.perf_counter_ns() - started) / 1e6
    return elapsed, output


def eager_cache_stats():
    # Read-only diagnostics, private API for the pinned SDK version.
    from furiosa.torch._backend import aten
    return {
        name: getattr(aten, name).cache_info()._asdict()
        for name in ("_build_exported_program_cached", "_compile_cached", "_compile_and_load_cached")
    }
