"""Plot mutually exclusive measured warm-inference components, without new NPU runs."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COMPONENTS=[('npu_task','NPU Task execution','#3776c9'),
            ('h2d','CPU → NPU copy call','#e9a13e'),
            ('d2h','NPU → CPU copy call','#8d78c4'),
            ('compile','Compilation (warmed: 0%)','#59ad99'),
            ('other','Other + profiling effects¹','#9ca3af')]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();run=args.run.resolve()
    out=run/'figures';out.mkdir(exist_ok=True)
    samples=json.loads((run/'samples.json').read_text());summary=json.loads((run/'summary.json').read_text())
    groups=collections.defaultdict(list)
    for s in samples:groups[s['case']].append(s)
    rows=[];lookup={};errors=[]
    for r in summary:
        if r['status']!='ok':continue
        case=r['case'];observed=groups[case]
        if len(observed)!=5 or any(s['npu_task_ms'] is None for s in observed):raise ValueError('Incomplete case '+case)
        if r['backend_calls_during_samples']!=0 or r['recompile_log_lines']!=0:raise ValueError('Compilation not excluded '+case)
        for s in observed:
            ev=json.loads((run/'cases'/case/f"forward_{s['iteration']:02d}.json").read_text())['traceEvents']
            if any(e.get('name')=='compile' and e.get('ph')=='X' for e in ev):raise ValueError('Compile event in warm capture '+case)
            parts=[s['npu_task_ms'],s['h2d_wall_ms'],s['d2h_wall_ms'],0.,s['forward_wall_ms']-s['npu_task_ms']]
            if min(parts)<0 or not math.isclose(sum(parts),s['serial_request_wall_sum_ms'],abs_tol=1e-8):raise ValueError('Nonexclusive decomposition '+case)
        means={'npu_task':st.mean(s['npu_task_ms'] for s in observed),
               'h2d':st.mean(s['h2d_wall_ms'] for s in observed),
               'd2h':st.mean(s['d2h_wall_ms'] for s in observed),
               'compile':0.,'other':st.mean(s['forward_wall_ms']-s['npu_task_ms'] for s in observed)}
        total=st.mean(s['serial_request_wall_sum_ms'] for s in observed)
        row={'case':case,'pe':r['pe'],'batch':r['batch'],'mode':r['mode'],'captures':5,'total_mean_ms':total}
        for key,_,_ in COMPONENTS:
            row[key+'_mean_ms']=means[key];row[key+'_pct']=100*means[key]/total
        if not math.isclose(sum(row[k+'_pct'] for k,_,_ in COMPONENTS),100.,abs_tol=1e-8):errors.append(case)
        rows.append(row);lookup[(r['pe'],r['batch'],r['mode'])]=row
    with (run/'breakdown.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (run/'breakdown.json').write_text(json.dumps(rows,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    handles=[Patch(color=color,label=name) for _,name,color in COMPONENTS]
    def bars(ax,pe,batch,large=False):
        for y,mode in zip([1,0],['eager','compile']):
            row=lookup.get((pe,batch,mode))
            if row is None:
                ax.barh(y,100,height=.38,color='#f5f5f5',edgecolor='#b9b9b9',hatch='//')
                ax.text(50,y,'Not measured — compilation memory limit',ha='center',va='center',fontsize=10,color='#555')
                continue
            left=0
            for key,_,color in COMPONENTS:
                pct=row[key+'_pct'];ax.barh(y,pct,left=left,height=.38,color=color,linewidth=0)
                if pct>=(3 if large else 6):
                    ax.text(left+pct/2,y,f'{pct:.1f}%',ha='center',va='center',color='white',weight='bold',fontsize=13 if large else 10)
                left+=pct
            ax.text(102,y,f"{row['total_mean_ms']:.2f} ms",ha='left',va='center',fontsize=12 if large else 10,color='#333',clip_on=False)
        ax.set_xlim(0,100);ax.set_ylim(-.45,1.45);ax.set_yticks([1,0],['Eager','Compile'])
        ax.set_xticks([0,25,50,75,100],['0%','25%','50%','75%','100%'])
        ax.spines['bottom'].set_color('#dddddd');ax.tick_params(axis='y',length=0,pad=10)
        ax.tick_params(axis='x',length=0,pad=8,colors='#666666');ax.set_axisbelow(True)
    fig=plt.figure(figsize=(12.8,7.5),facecolor='white')
    ax=fig.add_axes([.09,.48,.77,.32]);bars(ax,1,1,True)
    fig.text(.045,.935,'Profiled Inference Time Breakdown',fontsize=23,weight='bold',color='#151515')
    fig.text(.045,.875,'Profiler ON · Batch 1 · 1 PE · ViT-B/16 BF16 · 5 captures per mode',fontsize=12,color='#555555')
    fig.text(.875,.815,'Mean total²',fontsize=10,color='#666666')
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.045,.43),ncol=2,frameon=False,fontsize=11,columnspacing=2,labelspacing=.8)
    # Exact figures keep tiny transfer components readable without widening them.
    for y,mode in zip([.245,.205],['eager','compile']):
        r=lookup[(1,1,mode)]
        fig.text(.055,y,f"{mode.title():8}  NPU {r['npu_task_pct']:.2f}%    H2D {r['h2d_pct']:.2f}%    D2H {r['d2h_pct']:.2f}%    Compile 0%    Other {r['other_pct']:.2f}%",fontsize=10,color='#444444',family='DejaVu Sans Mono')
    fig.text(.045,.075,'Shares include profiler overhead; they do not describe ordinary unprofiled inference. Copy calls include allocation and waiting.\n¹ Other = forward wall − NPU Task; includes dispatch, gaps, synchronization overhead and profiling effects.\n² Total = H2D call + resident forward + D2H call. Profiler setup / finalization and compilation warmup are excluded.',fontsize=10,color='#666666',linespacing=1.6)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'breakdown_p1_b1.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    pairs=[(1,1),(2,1),(4,1),(8,1),(1,4),(4,4),(8,4),(1,16)]
    fig,axes=plt.subplots(4,2,figsize=(15,12.8),facecolor='white')
    for ax,(pe,batch) in zip(axes.flat,pairs):
        bars(ax,pe,batch);ax.set_title(f'Batch {batch} · {pe} PE',loc='left',fontsize=13,weight='bold',pad=14)
    fig.suptitle('Profiled Warm-Inference Breakdown by Batch × PE',x=.065,y=.975,ha='left',fontsize=21,weight='bold')
    fig.text(.065,.945,'Profiler ON · 100% per mode · 5 captures · mean ms at right · not an unprofiled latency breakdown',fontsize=12,color='#555555')
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.055,.075),ncol=3,frameon=False,fontsize=11)
    fig.text(.065,.025,'NPU Task ≠ isolated arithmetic time. H2D / D2H = copy-call wall time, including allocation and waiting.\nOther = forward wall − NPU Task (includes profiling effects). Compilation occurred before these warm captures; 0% in all bars.\nHatched rows have no completed inference. A 100% bar compares composition, not absolute speed.',fontsize=10,color='#666666',linespacing=1.5)
    fig.subplots_adjust(left=.09,right=.9,top=.9,bottom=.18,hspace=.8,wspace=.55)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'breakdown_all.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    verification={'passed':not errors,'conditions':len(rows),'captures':sum(r['captures'] for r in rows),'sum_tolerance_pct':1e-8,
        'aggregation':'sum(component durations across 5 captures) / sum(serial request durations across same captures)',
        'compilation_checked':'no forward compile span, no backend re-entry, no recompile log',
        'errors':errors,'source_sha256':hashlib.sha256((run/'samples.json').read_bytes()).hexdigest()}
    (run/'breakdown_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))
    for mode in ['eager','compile']:
        r=lookup[(1,1,mode)];print(mode,{k:round(v,3) for k,v in r.items() if isinstance(v,float)})

if __name__=='__main__':main()
