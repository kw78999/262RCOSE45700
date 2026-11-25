"""Standalone publication/export PNGs and the exact CSV data used for each figure."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(run,name):
    with (run/name).open() as f:return list(csv.DictReader(f))


def generate(run):
    figdir=run/'figures';figdir.mkdir(exist_ok=True)
    cond=read(run,'condition_summary.csv');trial=read(run,'trial_summary.csv')
    paired=read(run,'paired_speedups.csv');scale=read(run,'scaling_summary.csv')
    warm=read(run,'warmup_curves.csv');stable=read(run,'stability_bins.csv');coverage=read(run,'coverage.csv')
    cfg=json.loads((run/'experiment_config.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    colors={'eager':'#a34d16','compile':'#1264a3'}
    def export(name,fig,rows,caption):
        fig.text(.01,.005,caption,fontsize=8,va='bottom',wrap=True)
        fig.tight_layout(rect=(0,.065,1,1));fig.savefig(figdir/(name+'.png'),dpi=160);plt.close(fig)
        fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else ['status','reason']
        with (figdir/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    def empty(ax,msg='No completed eligible formal trials yet'):
        ax.text(.5,.5,msg,ha='center',va='center',transform=ax.transAxes,color='#666666')
    def panels():
        fig,axs=plt.subplots(2,2,figsize=(12,8));return fig,axs.ravel()
    for name,metric,ylabel in [('01_throughput','median_trial_images_s','Images / s'),('02_batch_latency','median_trial_mean_ms','Batch-call latency (ms)')]:
        fig,axs=panels()
        for ax,p in zip(axs,cfg['pe_candidates']):
            subset=[r for r in cond if int(r['pe'])==p]
            if not subset:empty(ax,'No measured eligible conditions')
            for mode in ('eager','compile'):
                for stage,ls,mark in [('formal','-','o'),('exploration',':','x')]:
                    rows=sorted([r for r in subset if r['mode']==mode and r['stage']==stage],key=lambda r:int(r['batch']))
                    if rows:ax.plot([int(r['batch']) for r in rows],[float(r[metric]) for r in rows],ls,marker=mark,color=colors[mode],label=f'{mode} {stage}')
            ax.set(xscale='log',xlabel='Batch size (log scale)',ylabel=ylabel,title=f'Requested/SDK Fusion {p} PE');ax.grid(alpha=.25)
            if subset:ax.legend(fontsize=7)
        export(name,fig,cond,'Formal: median of all completed trial means/rates. Dotted: separate exploration. PE denotes SDK allocation, not verified utilization; task evidence varies by mode (pe_observations.csv). Provisional.')
    fig,ax=plt.subplots(figsize=(14,5));batches=cfg['batches']+cfg['extension_batches'];pes=cfg['pe_candidates']
    data=np.full((len(pes),len(batches)),np.nan)
    used=[]
    for i,p in enumerate(pes):
        for j,b in enumerate(batches):
            rows=[r for r in paired if r['stage']=='formal' and int(r['pe'])==p and int(r['batch'])==b]
            if rows:data[i,j]=st.median(float(r['eager_over_compile']) for r in rows);used.extend(rows)
    finite=data[np.isfinite(data)]
    if len(finite):
        im=ax.imshow(np.ma.masked_invalid(data),aspect='auto',cmap='coolwarm',vmin=min(1,finite.min()),vmax=max(1,finite.max()))
        fig.colorbar(im,ax=ax,label='Eager mean / Compile mean; 1 = tie')
    else:ax.set_facecolor('#f1f1f1')
    for i,p in enumerate(pes):
        for j,b in enumerate(batches):
            value=data[i,j]
            if np.isfinite(value):label=f'{value:.2f}×'
            else:
                cs=[r for r in coverage if int(r['pe'])==p and int(r['batch'])==b]
                reason=next((r['reason'] for r in cs if r['reason']),'unmeasured')
                label='—\n'+reason.replace('not_started_or_extension_prerequisite_unmet','not started').replace('formal_not_measured','no formal').replace('held_budget','budget').replace('held_device_busy_or_unknown','device busy')[:22]
            ax.text(j,i,label,ha='center',va='center',fontsize=7)
    ax.set(xticks=range(len(batches)),xticklabels=batches,yticks=range(len(pes)),yticklabels=pes,xlabel='Batch size',ylabel='SDK Fusion PE',title='03 Paired Eager / Compile ratio (formal); missing cells retain reasons')
    export('03_eager_compile_heatmap',fig,used,'Median of paired trial ratios, not ratio of pooled calls. Blank/grey = missing, never zero. Ratio 1 = tie. Provisional.')
    for name,metric,ylabel in [('04_pe_scaling','pe_scaling_speedup','1 PE mean / P PE mean'),('05_relative_efficiency','relative_scaling_efficiency','Scaling speedup / allocated PE')]:
        fig,axs=plt.subplots(1,2,figsize=(13,5))
        used=[r for r in scale if r['stage']=='formal']
        for ax,mode in zip(axs,('eager','compile')):
            subset=[r for r in used if r['mode']==mode]
            if not subset:empty(ax)
            for b in batches:
                ps=sorted({int(r['pe']) for r in subset if int(r['batch'])==b});med=[];low=[];high=[]
                for p in ps:
                    v=[float(r[metric]) for r in subset if int(r['batch'])==b and int(r['pe'])==p]
                    med.append(st.median(v));low.append(min(v));high.append(max(v))
                if ps:
                    ax.plot(ps,med,marker='o',label=f'B{b}');ax.fill_between(ps,low,high,alpha=.12)
            ax.axhline(1,color='black',ls='--',lw=1);ax.set(xlabel='SDK Fusion PE',ylabel=ylabel,title=mode,xticks=pes);ax.grid(alpha=.25)
            if subset:ax.legend(fontsize=7,ncol=2)
        export(name,fig,used,'Matched trial ratios: median and min–max, not confidence intervals. Valid 1 PE reference required. PE is SDK allocation; task observations vary by mode. Efficiency is not utilization. Provisional.')
    fig,axs=panels();used=[]
    reps=manifest.get('selection',{}).get('representatives',[])
    for ax,p in zip(axs,pes):
        selected=[int(pair[1]) for pair in reps if int(pair[0])==p]
        if not selected:selected=sorted({int(r['batch']) for r in warm if int(r['pe'])==p and r['stage']=='stabilization'})[:3]
        subset=[r for r in warm if int(r['pe'])==p and int(r['batch']) in selected and r['stage']=='stabilization'];used.extend(subset)
        if not subset:empty(ax,'No completed stabilization probe')
        for b in selected:
            for mode in ('eager','compile'):
                rows=[r for r in subset if int(r['batch'])==b and r['mode']==mode]
                if rows:ax.plot([int(r['iteration']) for r in rows],[float(r['elapsed_ms']) for r in rows],label=f'B{b} {mode}',lw=.8)
        ax.axvline(5,color='grey',ls=':',label='old 5 calls');ax.axvline(100,color='black',ls='--',lw=.8)
        ax.set(xlabel='Warmup call (first call separate)',ylabel='Latency (ms; log scale)',yscale='log',title=f'Fusion {p} PE');ax.grid(alpha=.2)
        if subset:ax.legend(fontsize=7)
    export('06_warmup_curves',fig,used,'All probe calls retained. 100–500 rule is a heuristic, not proof of steady state. First call excluded from curve and stored in initial_costs.csv. Y axis log scale.')
    fig,axs=panels();used=[r for r in trial if r['stage']=='formal' and r['eligible_for_descriptive_performance']=='True']
    for ax,p in zip(axs,pes):
        subset=[r for r in used if int(r['pe'])==p]
        if not subset:empty(ax)
        for mode in ('eager','compile'):
            rows=[r for r in subset if r['mode']==mode]
            if rows:
                xs=[int(r['batch'])*(1+(int(r['trial'])-3)*.018) for r in rows]
                ax.scatter(xs,[float(r['mean_ms']) for r in rows],label=mode,c=colors[mode],alpha=.7)
        ax.set(xscale='log',yscale='log',xlabel='Batch size (log; trial jitter)',ylabel='Trial mean latency (ms; log)',title=f'Fusion {p} PE');ax.grid(alpha=.25)
        if subset:ax.legend(fontsize=8)
    export('07_trial_means',fig,used,'Each point is one independent fresh-process trial mean. Every completed eligible trial shown; no fastest-trial selection. Log axes. Provisional.')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    if not stable:
        for ax in axs:empty(ax,'No completed 300-second stability run')
    cases=sorted(set(r['case_id'] for r in stable))
    for case in cases:
        rows=[r for r in stable if r['case_id']==case];r=rows[0];label=f'P{r["pe"]} B{r["batch"]} {r["mode"]}'
        xs=[float(x['time_start_s']) for x in rows]
        axs[0].plot(xs,[float(x['mean_ms']) for x in rows],label=label)
        axs[1].plot(xs,[float(x['images_s']) for x in rows],label=label)
    for ax,label in zip(axs,('Mean batch latency (ms)','Images / s')):
        ax.set(xlabel='Seconds after measured stability phase starts',ylabel=label);ax.grid(alpha=.25)
        if stable:ax.legend(fontsize=8)
    export('08_stability_trend',fig,stable,'Separate stability workers; 10-second bins, last may be partial. No formal statistics pooled. No thermal-throttling causal claim. Provisional.')
    costs=read(run,'initial_costs.csv');used=[r for r in costs if r['stage']=='formal' and r.get('logging_protocol_current','True')=='True']
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,mode in zip(axs,('eager','compile')):
        rows=[r for r in used if r['mode']==mode]
        if not rows:empty(ax)
        for p in pes:
            rs=[r for r in rows if int(r['pe'])==p]
            if rs:ax.scatter([int(r['batch']) for r in rs],[float(r['A_ms'])/1000 for r in rs],label=f'P{p}')
        ax.set(xscale='log',yscale='log',xlabel='Batch size (log)',ylabel='S + F + W (seconds; log)',title=mode)
        if rows:ax.legend()
    export('09_initial_costs',fig,used,'Preparation-after scenario includes real common-warmup forwards. S wrapper + F first + W warmup, not pure compiler time. Both axes log; model/input preparation separately recorded.')
    device=[]
    import re
    path=run/'device_states.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            info=r['commands'][0].get('stdout','');status=r['commands'][1].get('stdout','')
            temp=re.search(r'([\d.]+)°C',info);power=re.search(r'\|\s*(\d+) W',info)
            device.append(dict(utc=r['utc'],label=r['label'],worker=r.get('worker'),temperature_c=float(temp[1]) if temp else None,power_w=int(power[1]) if power else None,utilization='unavailable: status reports -'))
    fields=list(device[0]) if device else ['utc','temperature_c','power_w','utilization']
    with (run/'device_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(device)
    (figdir/'README.md').write_text('Figures 01–09 include matching CSV exports. Formal and exploration distinguished. All performance provisional; numerical tolerance pending. Blank panels/cells mean missing data, with coverage reasons in coverage.csv. Min–max ranges are descriptive, not confidence intervals. PNGs are static export artifacts.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();generate(a.run)
