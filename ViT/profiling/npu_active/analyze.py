"""Analyze recorded device Task intervals; never execute NPU workloads."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

rows, captures = [], {}
for base in ('vit_pilot', 'vit_scaling'):
    for path in sorted((ROOT/base).rglob('furiosa_trace.json')):
        if any(part in ('interim', 'code') for part in path.parts):
            continue
        events = json.loads(path.read_text())['traceEvents']
        tasks = [e for e in events if e.get('ph') == 'X'
                 and '::Task::' in e.get('name', '')
                 and {'begin_cycle', 'end_cycle', 'cycle_actual'} <= e.get('args', {}).keys()
                 and e.get('dur', 0) > 0]
        intervals = merge([(e['ts'], e['ts']+e['dur']) for e in tasks])
        active = sum(b-a for a,b in intervals)/1000 if intervals else None
        result_path = path.parent/'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        # Final manifest classification takes priority over worker-local status.
        manifest_path = path.parents[2]/'manifest.json'
        classification = result.get('status', 'unknown')
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for worker in manifest.get('workers', []):
                if worker.get('case_id') == path.parent.name:
                    classification = worker.get('status', classification)
        row = dict(source=str(path.relative_to(ROOT)), case=path.parent.name,
                   classification=classification, task_events=len(tasks),
                   observed_active_union_ms=active,
                   summed_task_ms=sum(e['dur'] for e in tasks)/1000 if tasks else None,
                   first_to_last_task_ms=(intervals[-1][1]-intervals[0][0])/1000 if intervals else None,
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(row)
        captures[path.parent.name] = (row, intervals)
with (OUT/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(OUT/'summary.json').write_text(json.dumps(rows, indent=2))
plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
colors={'eager':'#3182bd','compile':'#ed8936'}
fig, axes=plt.subplots(1,2,figsize=(13,5.2))
for ax, panel in zip(axes, ['fastpath','scaling']):
    groups=[1,4] if panel=='fastpath' else [1,2,4,8]
    for j, mode in enumerate(['eager','compile']):
        for i, group in enumerate(groups):
            if panel=='fastpath':
                case=f'diagnostic_{mode}_b{group}_t1'+('_attempt1' if group==1 else '')
            else:
                attempt=2 if group==2 and mode=='compile' else 1
                case=f'diagnostic_p{group}_b1_{mode}_t1_a{attempt}'
            row,_=captures[case];v=row['observed_active_union_ms'];x=i+(j-.5)*.36
            if v is None:
                ax.text(x,.15,'N/A',ha='center',color=colors[mode],fontsize=10,rotation=90)
            else:
                ax.bar(x,v,width=.33,color=colors[mode],label=mode if i==0 else None)
                ax.text(x,v+.14,f'{v:.3f}',ha='center',fontsize=10)
    ax.set_xticks(range(len(groups)),[f'Batch {g}' if panel=='fastpath' else f'{g} PE' for g in groups])
    ax.set_ylabel('Recorded NPU Task active time (ms)')
    ax.set_title('Fastpath-off comparison' if panel=='fastpath' else 'PE scaling / Batch 1')
    ax.set_ylim(0,17 if panel=='fastpath' else 6.4);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend()
fig.suptitle('ViT / BF16 — recorded NPU execution intervals only',fontsize=15)
fig.text(.02,.025,'One diagnostic capture per condition. Overlapping intervals counted once. Missing Task timing = N/A, not zero.\nTask spans may include device-side DMA / internal waits; these are not isolated compute-kernel times.',fontsize=10,color='#555555')
fig.tight_layout(rect=[0,.12,1,.94])
fig.savefig(OUT/'npu_active.png',dpi=180);fig.savefig(OUT/'npu_active.svg');plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(13,5.7))
for ax,batch in zip(axes,[1,4]):
    maxend=0
    for j,mode in enumerate(['eager','compile']):
        case=f'diagnostic_{mode}_b{batch}_t1'+('_attempt1' if batch==1 else '')
        row,intervals=captures[case];origin=intervals[0][0]
        bars=[((a-origin)/1000,(b-a)/1000) for a,b in intervals]
        ax.broken_barh(bars,(j-.22,.44),facecolors=colors[mode])
        maxend=max(maxend,(intervals[-1][1]-origin)/1000)
    ax.set_yticks([0,1],['eager','compile']);ax.set_ylim(-.6,1.6);ax.set_xlim(0,maxend*1.03)
    ax.set_title(f'Batch {batch} / Fastpath-off');ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('Time since first recorded NPU Task in each capture (ms)')
fig.suptitle('NPU execution timeline — colored intervals only',fontsize=15)
fig.text(.02,.02,'Blank gaps are outside recorded NPU Task intervals; no CPU submission or synchronization spans are drawn.\nEach row starts at its own first NPU Task. Diagnostic captures are not formal latency measurements.',fontsize=10,color='#555555')
fig.tight_layout(rect=[0,.13,1,.94]);fig.savefig(OUT/'npu_timeline.png',dpi=180);fig.savefig(OUT/'npu_timeline.svg');plt.close(fig)
for row in rows:
    if 'fastpath_compare' in row['source'] or 'vit_scaling' in row['source']:
        print(row['case'],row['classification'],row['observed_active_union_ms'])
