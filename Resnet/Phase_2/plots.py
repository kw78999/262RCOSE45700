"""Static figures from the Phase 2 CSV tables. Needs matplotlib only (no torch, no NPU).

Run with an interpreter that has matplotlib, for example the existing plotting venv:
    /root/ViT/vit_pilot/.plot-venv/bin/python Resnet/Phase_2/plots.py <run-dir>
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"eager": ("#2a78d6", "o", "Eager"), "compile": ("#eb6834", "s", "Compile")}


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def axes(title, ylabel, batches, width=7.2):
    figure, axis = plt.subplots(figsize=(width, 4.4), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    style(axis, title, ylabel, batches)
    return figure, axis


def style(axis, title, ylabel, batches):
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    axis.set_xlabel("Batch size (log scale)", color=MUTED, fontsize=9)
    axis.set_ylabel(ylabel, color=MUTED, fontsize=9)
    axis.set_xscale("log", base=2)
    axis.set_xticks(batches)
    axis.set_xticklabels([str(b) for b in batches])
    axis.minorticks_off()
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)


def line(axis, mode, points, errors=None, label_format="{:.1f}", label_last=True):
    """points: [(batch, value)], errors: [(low, high)] absolute bounds or None."""
    if not points:
        return
    color, marker, name = SERIES[mode]
    xs, ys = zip(*points)
    if errors:
        lows = [y - low for y, (low, _) in zip(ys, errors)]
        highs = [high - y for y, (_, high) in zip(ys, errors)]
        axis.errorbar(xs, ys, yerr=[lows, highs], fmt="none", ecolor=color, elinewidth=1.2, capsize=3, alpha=0.7)
    axis.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=6.5,
              markeredgecolor=SURFACE, markeredgewidth=1.2, label=name)
    if label_last:
        axis.annotate(label_format.format(ys[-1]), (xs[-1], ys[-1]), xytext=(8, 0),
                      textcoords="offset points", va="center", color=INK, fontsize=8.5)


def finish(figure, axis, path, legend=True):
    if legend:
        axis.legend(frameon=False, fontsize=9, labelcolor=INK, loc="best")
    axis.margins(x=0.08)
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def main():
    root = Path(sys.argv[1])
    out = root / "figures"
    out.mkdir(exist_ok=True)
    scaling = read(root / "batch_scaling.csv")
    batches = [int(row["batch"]) for row in scaling]
    written = []

    def series(key):
        return {mode: [(int(row["batch"]), number(row, f"{mode}_{key}")) for row in scaling
                       if number(row, f"{mode}_{key}") is not None] for mode in SERIES}

    # 1. Batch latency: pooled mean, bars = +/- 1 sample standard deviation.
    figure, axis = axes("Steady-state batch latency (mean ± 1 SD of pooled repeats)", "Latency per batch (ms)", batches)
    means, stdevs = series("mean_ms"), series("stdev_ms")
    for mode in SERIES:
        errors = [(m - s, m + s) for (_, m), (_, s) in zip(means[mode], stdevs[mode])]
        line(axis, mode, means[mode], errors, "{:.2f} ms")
    axis.set_ylim(bottom=0)
    finish(figure, axis, out / "latency_vs_batch.png")
    written.append("latency_vs_batch.png")

    # 2. Throughput.
    figure, axis = axes("Throughput (batch × 1000 / mean latency)", "Images per second", batches)
    for mode, points in series("throughput_samples_per_s").items():
        line(axis, mode, points, None, "{:,.0f} img/s")
    axis.set_ylim(bottom=0)
    finish(figure, axis, out / "throughput_vs_batch.png")
    written.append("throughput_vs_batch.png")

    # 3. Speedup: one series, so the title names it and there is no legend box.
    rows = [row for row in scaling if number(row, "speedup_mean") is not None]
    if rows:
        figure, axis = axes("Compile speedup over Eager (Eager mean / Compile mean; bars = per-trial range)",
                            "Speedup (×)", [int(row["batch"]) for row in rows])
        xs = [int(row["batch"]) for row in rows]
        ys = [number(row, "speedup_mean") for row in rows]
        lows = [y - (number(row, "speedup_trial_min") or y) for y, row in zip(ys, rows)]
        highs = [(number(row, "speedup_trial_max") or y) - y for y, row in zip(ys, rows)]
        axis.errorbar(xs, ys, yerr=[lows, highs], fmt="none", ecolor=MUTED, elinewidth=1.2, capsize=3)
        axis.plot(xs, ys, color=SERIES["compile"][0], linewidth=2, marker="s", markersize=6.5,
                  markeredgecolor=SURFACE, markeredgewidth=1.2)
        for x, y in zip(xs, ys):
            axis.annotate(f"{y:.2f}×", (x, y), xytext=(0, 11), textcoords="offset points",
                          ha="center", color=INK, fontsize=8.5)
        axis.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        axis.annotate("parity (1.0×)", (xs[0], 1.0), xytext=(0, 4), textcoords="offset points",
                      color=MUTED, fontsize=8)
        axis.set_ylim(bottom=0, top=max(ys) * 1.18)
        finish(figure, axis, out / "speedup_vs_batch.png", legend=False)
        written.append("speedup_vs_batch.png")

    # 4. Scaling efficiency against batch 1.
    figure, axis = axes("Scaling efficiency ((throughput(B) / throughput(1)) / B)", "Efficiency (1.0 = linear)", batches)
    for mode, points in series("scaling_efficiency").items():
        line(axis, mode, points, None, "{:.2f}")
    axis.set_ylim(0, 1.08)
    finish(figure, axis, out / "scaling_efficiency.png")
    written.append("scaling_efficiency.png")

    # 5. Latency per image.
    figure, axis = axes("Latency per image (mean batch latency / batch)", "ms per image", batches)
    for mode, points in series("ms_per_sample").items():
        line(axis, mode, points, None, "{:.2f} ms")
    axis.set_ylim(bottom=0)
    finish(figure, axis, out / "latency_per_image.png")
    written.append("latency_per_image.png")

    # 6. First call (compile/load inclusive); recorded only, excluded from the comparison.
    initial = read(root / "initial_costs.csv")
    if initial:
        figure, axis = axes("First call per fresh cache (median; bars = min–max over trials)",
                            "First-call time (s)", sorted({int(row["batch"]) for row in initial}))
        for mode in SERIES:
            rows = sorted((row for row in initial if row["mode"] == mode), key=lambda r: int(r["batch"]))
            points = [(int(row["batch"]), float(row["first_call_median_s"])) for row in rows]
            errors = [(float(row["first_call_min_s"]), float(row["first_call_max_s"])) for row in rows]
            line(axis, mode, points, errors, "{:,.0f} s")
        axis.set_ylim(bottom=0)
        finish(figure, axis, out / "first_call_vs_batch.png")
        written.append("first_call_vs_batch.png")

    # 7. Transfer segments from the separate auxiliary loop: two panels, one measure each.
    condition = read(root / "condition_summary.csv")
    figure, panels = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    for axis, scope, title in zip(panels, ("h2d", "d2h"), ("H2D input transfer (mean)", "D2H output transfer (mean)")):
        style(axis, title, "Time (ms)", batches)
        for mode in SERIES:
            points = sorted((int(row["batch"]), float(row["mean_ms"])) for row in condition
                            if row["scope"] == scope and row["mode"] == mode)
            line(axis, mode, points, None, "{:.2f} ms")
        axis.set_ylim(bottom=0)
        axis.margins(x=0.1)
        axis.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    figure.tight_layout()
    figure.savefig(out / "transfer_vs_batch.png", facecolor=SURFACE)
    plt.close(figure)
    written.append("transfer_vs_batch.png")
    print("\n".join(str(out / name) for name in written))


if __name__ == "__main__":
    main()
