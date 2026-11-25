"""Statistics, numerical comparisons and explicit startup-cost estimates."""
import math
import statistics


def statistics_ms(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Timing samples must be finite and positive")
    mean = statistics.mean(values)
    return {"n": len(values), "median_ms": statistics.median(values),
            "min_ms": min(values), "max_ms": max(values), "mean_ms": mean,
            "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
            "cv": statistics.stdev(values) / mean if len(values) > 1 else 0.0}


def break_even(eager_initial_ms, compile_initial_ms, eager_ms, compile_ms):
    """Boundary and first STRICT Compile win; ties use Eager in the original plan."""
    if any(not math.isfinite(x) or x < 0 for x in
           (eager_initial_ms, compile_initial_ms, eager_ms, compile_ms)):
        raise ValueError("Costs must be finite and nonnegative")
    saving = eager_ms - compile_ms
    if saving <= 0:
        return {"status": "no_steady_state_gain", "boundary_calls": None,
                "boundary_ceil_calls": None, "first_strict_compile_win_calls": None}
    boundary = (compile_initial_ms - eager_initial_ms) / saving
    return {"status": "estimated", "boundary_calls": boundary,
            "boundary_ceil_calls": max(1, math.ceil(boundary)),
            "first_strict_compile_win_calls": max(1, math.floor(boundary) + 1)}


def compare_tensors(eager, compiled, classifier=True):
    import torch
    if eager.shape != compiled.shape:
        return {"status": "shape_mismatch", "eager_shape": list(eager.shape),
                "compile_shape": list(compiled.shape)}
    a, b = eager.float(), compiled.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return {"status": "nonfinite"}
    difference = (a - b).abs()
    # Large Conv feature maps accumulate noticeable reduction error in FP32;
    # compute cosine reductions in FP64, outside every timed interval.
    flat_a, flat_b = a.flatten(1).double(), b.flatten(1).double()
    norm_a, norm_b = flat_a.norm(dim=1), flat_b.norm(dim=1)
    valid = (norm_a > 0) & (norm_b > 0)
    cosines = ((flat_a[valid] * flat_b[valid]).sum(dim=1) /
               (norm_a[valid] * norm_b[valid])).clamp(-1, 1)
    return {
        "status": "measured", "shape": list(a.shape),
        "max_abs_error": difference.max().item(), "mean_abs_error": difference.mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "cosine_mean": cosines.mean().item() if cosines.numel() else None,
        "cosine_min": cosines.min().item() if cosines.numel() else None,
        "cosine_undefined_samples": int((~valid).sum()),
        "prediction_agreement": (a.argmax(1) == b.argmax(1)).float().mean().item() if classifier else None,
        "acceptance": "not_defined; numerical metrics only, no ImageNet accuracy claim",
    }
