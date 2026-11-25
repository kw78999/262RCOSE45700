"""Timed scopes for Phase 2. Host wall-clock only; none of these is pure NPU kernel time.

Primary scope (the Phase 2 comparison basis):
    input already resident on the NPU -> forward -> output ``.cpu()`` returned.
Auxiliary scope (separate loop, never mixed into the primary statistics):
    H2D, forward + stream completion, and D2H timed as three consecutive segments.
"""
import time

import torch

from Resnet.runtime import synchronize


def forward_to_cpu(function, device_input, device):
    """Forward on a pre-staged NPU input until the output copy to CPU has returned."""
    synchronize(device)
    started = time.perf_counter_ns()
    output = function(device_input)
    cpu_output = output.to("cpu", non_blocking=False)
    elapsed = (time.perf_counter_ns() - started) / 1e6
    return elapsed, output, cpu_output


def e2e_split(function, cpu_input, device):
    """H2D | forward + completion | D2H, each closed by a stream synchronize."""
    synchronize(device)
    t0 = time.perf_counter_ns()
    device_input = cpu_input.to(device)
    synchronize(device)
    t1 = time.perf_counter_ns()
    output = function(device_input)
    synchronize(device)
    t2 = time.perf_counter_ns()
    cpu_output = output.to("cpu", non_blocking=False)
    synchronize(device)
    t3 = time.perf_counter_ns()
    return {"h2d": (t1 - t0) / 1e6, "forward_sync": (t2 - t1) / 1e6,
            "d2h": (t3 - t2) / 1e6, "e2e": (t3 - t0) / 1e6}, cpu_output


def idle_synchronize_cost(device, calls=50):
    """Cost of a synchronize on an already idle stream, to bound timer overhead."""
    synchronize(device)
    values = []
    for _ in range(calls):
        started = time.perf_counter_ns()
        synchronize(device)
        values.append((time.perf_counter_ns() - started) / 1e6)
    return values


def check_output(cpu_output, batch):
    if list(cpu_output.shape) != [batch, 1000]:
        raise RuntimeError(f"Unexpected ResNet50 output shape: {list(cpu_output.shape)}")
    if cpu_output.dtype != torch.bfloat16 or not torch.isfinite(cpu_output.float()).all():
        raise RuntimeError("Output must be finite BF16")
