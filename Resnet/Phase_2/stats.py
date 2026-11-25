"""Pure-Python statistics for steady-state latency samples (no torch, no NPU)."""
import math
import statistics


def percentile(values, q):
    """Linear-interpolation percentile (same definition as numpy's default)."""
    if not values:
        raise ValueError("No samples")
    if not 0 <= q <= 100:
        raise ValueError("q must be within [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values):
    """Mean / sample standard deviation / median / P95 and range, in the input unit."""
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Timing samples must be finite and positive")
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"n": len(values), "mean_ms": mean, "stdev_ms": stdev,
            "median_ms": statistics.median(values), "p95_ms": percentile(values, 95),
            "min_ms": min(values), "max_ms": max(values), "cv": stdev / mean}


def throughput(batch, latency_ms):
    """Samples per second for one batch completing in latency_ms."""
    if batch < 1 or latency_ms <= 0:
        raise ValueError("batch and latency must be positive")
    return batch * 1000.0 / latency_ms


def scaling_efficiency(batch, latency_ms, base_batch, base_latency_ms):
    """Throughput gain over the base batch divided by the ideal (linear) gain.

    1.0 means throughput grew in proportion to the batch (latency stayed flat);
    base_batch / batch means no batching benefit (latency grew in proportion).
    """
    gain = throughput(batch, latency_ms) / throughput(base_batch, base_latency_ms)
    return gain / (batch / base_batch)


def speedup(eager_ms, compile_ms):
    if eager_ms <= 0 or compile_ms <= 0:
        raise ValueError("latencies must be positive")
    return eager_ms / compile_ms
