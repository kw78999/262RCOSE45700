"""Render saved CSV only; run in the existing isolated matplotlib environment."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
p.add_argument('--attention-run', type=Path, help='Optional later run with a user-code Attention boundary')
a = p.parse_args()
cfg = json.loads((a.run/'config.json').read_text())
with (a.run/'summary.csv').open() as f:
    rows = list(csv.DictReader(f))
modes = ['eager_default', 'attention_compile', 'eager_fastpath_off', 'full_compile']
labels = ['Eager\nCPU fallback', 'Attention\ncompile', 'Eager\nfastpath off', 'Whole ViT\ncompile']
destination = a.run
if a.attention_run:
    extra_cfg = json.loads((a.attention_run/'config.json').read_text())
    assert all(cfg[k] == extra_cfg[k] for k in ['assets_dir', 'device', 'warmup', 'repeats'])
    with (a.attention_run/'summary.csv').open() as f:
        additions = list(csv.DictReader(f))
    overrides = {(r['mode'], r['batch']): r for r in additions}
    rows = [overrides.get((r['mode'], r['batch']), r) for r in rows]
    labels[1] = 'Attention boundary\ncompile'
    destination = a.attention_run
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
for col, batch in enumerate([1, 4]):
    for i, mode in enumerate(modes):
        r = next(s for s in rows if s['mode'] == mode and int(s['batch']) == batch)
        for ax, key in [(axes[0, col], 'first_call_ms'), (axes[1, col], 'mean_ms')]:
            if r['status'] == 'ok' and r[key]:
                val = float(r[key])
                err = float(r['stdev_ms']) if key == 'mean_ms' and r['stdev_ms'] else None
                ax.bar(i, val, yerr=err, capsize=4, color=['#999999', '#3b82b6', '#45a580', '#b78042'][i])
                ax.annotate(f'{val:.2f}', (i, val), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8)
            else:
                ax.text(i, .1, r['status'], transform=ax.get_xaxis_transform(), ha='center', fontsize=8)
        axes[0, col].set_title(f'Batch {batch}: first call (1 observation)')
        axes[1, col].set_title(f'Batch {batch}: repeat mean ± sample SD (not CI)')
    for ax in axes[:, col]:
        ax.set_xticks(range(4), labels)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.17)
    axes[0, col].set_ylim(bottom=0)
    axes[0, col].set_ylabel('Whole-model batch call ms (linear scale)')
    axes[1, col].set_yscale('log')
    axes[1, col].set_ylabel('Whole-model batch call ms (log scale)')
fig.suptitle('Preliminary ViT BF16 / RNGD 1 PE / 4 unique photos / independent trial = 1\n'
    f"{cfg['repeats']} repeats after {cfg['warmup']} warmups; stream-completion wall time; output tolerance pending\n"
    'Fresh process/private SDK cache initially absent; system cold unverified; Eager baseline mixes CPU/NPU', fontsize=10)
target = destination/'figures/01_attention_paths.png'
target.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(target, dpi=180)
plt.close(fig)
print(target)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
colors = ['#777777', '#3b82b6', '#45a580', '#b78042']
with (a.run/'timings.csv').open() as f:
    raw = list(csv.DictReader(f))
if a.attention_run:
    with (a.attention_run/'timings.csv').open() as f:
        extra = list(csv.DictReader(f))
    raw = [r for r in raw if r['mode'] != 'attention_compile'] + extra
for ax, batch in zip(axes, [1, 4]):
    for mode, label, color in zip(modes, labels, colors):
        calls = [r for r in raw if r['mode'] == mode and int(r['batch']) == batch and r['phase'] == 'repeat'
                 and r['status'] == 'ok' and r['timing_valid'] == 'True']
        if calls:
            ax.plot([int(r['iteration']) for r in calls], [float(r['elapsed_ms']) for r in calls],
                    marker='.', label=label.replace('\n', ' '), color=color)
    ax.set_title(f'Batch {batch}: repeat call order')
    ax.set_yscale('log')
    ax.set_ylabel('Whole-model batch call ms (log scale)')
    ax.set_xlabel('Repeat iteration (warmups excluded)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
fig.suptitle(f"Preliminary BF16 / RNGD 1 PE / 4 unique photos / 1 trial / {cfg['repeats']} repeats after {cfg['warmup']} warmups\n"
    'Private SDK cache initially absent; system cold unverified; baseline CPU fallback included\n'
    'Detailed logs off; timing spikes alone do not establish compilation or cache events', fontsize=10)
target = destination/'figures/02_repeat_order.png'
fig.savefig(target, dpi=180)
plt.close(fig)
print(target)
