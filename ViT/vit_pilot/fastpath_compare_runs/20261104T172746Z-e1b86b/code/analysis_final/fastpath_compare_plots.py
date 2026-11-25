"""Standard matplotlib figures from saved current-run CSV/JSON only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
    cfg=json.loads((run/'config.json').read_text());be=json.loads((run/'break_even_summary.json').read_text())
    cum=json.loads((run/'cumulative_validation_summary.json').read_text())
    rows=[r for r in load_csv(run/'trial_summary.csv') if r['status']=='ok' and r['cost_status']=='valid']
    target=run/'figures';target.mkdir(exist_ok=True)
    if not rows:
        (target/'unavailable.txt').write_text('No completed valid performance measurements; no hypothetical charts generated.\n');return
    caption=f"ViT BF16 / 4 unique real photos / RNGD 1 PE / MHA fastpath=False / CPU fallback=False\n{cfg['trials']} independent trials requested, {cfg['repeats']} repeats after {cfg['warmup']} warmups; output tolerance pending\nPrivate SDK cache initially absent; system cold and all-op placement unverified"
    colors={'eager':'#387bb4','compile':'#dd8740'};made=[]
    def save(fig,name):
        path=target/name;fig.savefig(path,dpi=180);plt.close(fig);made.append(path);print(path)
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),constrained_layout=True)
    for ax,batch in zip(axes,[1,4]):
        for mode,offset in [('eager',-.18),('compile',.18)]:
            selected=[r for r in rows if int(r['batch'])==batch and r['mode']==mode]
            if selected:
                ax.bar([int(r['trial'])+offset for r in selected],[float(r['mean_ms']) for r in selected],
                    width=.34,yerr=[float(r['stdev_ms']) for r in selected],capsize=4,label=mode,color=colors[mode])
        ax.set_title(f'Batch {batch}: repeat mean ± sample SD (not CI)');ax.set_xticks(range(1,cfg['trials']+1));ax.set_xlabel('Independent trial')
        ax.set_yscale('log');ax.set_ylabel('Batch call ms (log scale)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend()
    fig.suptitle('Preliminary matched-condition repeat comparison\n'+caption,fontsize=9);save(fig,'repeat_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),constrained_layout=True)
    for ax,batch in zip(axes,[1,4]):
        selected=sorted([r for r in rows if int(r['batch'])==batch],key=lambda r:(int(r['trial']),r['mode']=='compile'))
        positions=list(range(len(selected)));base=[0.0]*len(selected)
        for field,label,color in [('runner_setup_ms','S: runner setup','#478f80'),('first_call_ms','F: first call','#df9b44'),('warmup_sum_ms','W: warmup sum','#777bb5')]:
            vals=[float(r[field])/1000 for r in selected]
            ax.bar(positions,vals,bottom=base,label=label,color=color);base=[v+w for v,w in zip(base,vals)]
        ax.set_xticks(positions,[f"{'E' if r['mode']=='eager' else 'C'}{r['trial']}" for r in selected]);ax.set_title(f'Batch {batch}: A=S+F+W')
        ax.set_ylabel('Measured startup interval sum seconds');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    fig.suptitle('Preliminary startup components; E=Eager, C=whole-model Compile\n'+caption+'\nModel/input prepare and outside-timer gaps excluded; first call is not pure compile time',fontsize=9)
    save(fig,'startup_components.png')
    raw=load_csv(run/'timings.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for row_index,batch in enumerate([1,4]):
        for col,mode in enumerate(['eager','compile']):
            ax=axes[row_index,col]
            for s in [r for r in rows if int(r['batch'])==batch and r['mode']==mode]:
                calls=[r for r in raw if r['case_id']==s['case_id'] and r['phase']=='repeat' and r['status']=='ok' and r['timing_valid']=='True']
                ax.plot([int(r['iteration']) for r in calls],[float(r['elapsed_ms']) for r in calls],marker='.',markersize=3,label='trial '+s['trial'])
            ax.set_title(f'{mode}, batch {batch}');ax.set_xlabel('Repeat iteration (warmups excluded)');ax.set_ylabel('Batch call ms (linear scale)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Preliminary repeat call order; all valid slow calls retained\n'+caption+'\nDetailed performance logs off; time spikes alone do not identify compilation/cache events',fontsize=9)
    save(fig,'repeat_order.png')
    observed=load_csv(run/'cumulative_validation.csv')
    for batch in [1,4]:
        info=next(b for b in be['batches'] if b['batch']==batch);trial=info['representative_trial']
        pair=next((v for v in be['trials'] if v['batch']==batch and v['trial']==trial and v.get('A_e_ms') is not None),None)
        if pair is None:
            (target/f'cumulative_time_b{batch}_unavailable.txt').write_text('No valid finite current paired prediction.\n');continue
        ob=[r for r in observed if int(r['batch'])==batch and r['output_validation_status']=='eligible']
        stats=next(v for v in cum['batches'] if v['batch']==batch)
        nwin=pair['N_win'];maximum=max([math.ceil(1.2*nwin),20]+[int(r['N']) for r in ob])
        step=max(1,maximum//800);xs=list(range(0,maximum+1,step));xs.append(maximum)
        fig,ax=plt.subplots(figsize=(10,5.7),constrained_layout=True)
        for mode,aa,tt in [('eager','A_e_ms','t_e_ms'),('compile','A_c_ms','t_c_ms')]:
            ax.plot(xs,[(pair[aa]+x*pair[tt])/1000 for x in xs],linestyle='--',color=colors[mode],label=f'{mode}: prediction from paired trial {trial}')
            if ob:
                ax.plot([int(r['N']) for r in ob],[float(r['observed_'+mode+'_ms'])/1000 for r in ob],color=colors[mode],label=mode+': separate observed prefix')
        ax.axvline(nwin,color='#777777',linestyle=':',label=f'Primary predicted N_win={nwin}')
        if stats['first_observed_compile_advantage_N'] is not None:
            n=stats['first_observed_compile_advantage_N'];ax.axvline(n,color='#9c3f64',linestyle='-.',label=f'First observed Compile advantage N={n}')
        ax.set_title(f'Batch {batch}: post-preparation work N; startup A included\nValidation: {stats["status"]}',fontsize=10)
        ax.set_xlabel(f'N: work batch calls after first+warmup (valid images={batch}×N)')
        ax.set_ylabel('Cumulative measured/predicted interval sum seconds');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Preliminary cumulative cost; predictions not refitted to validation\n'+caption+
            f"\nPrediction: one actual paired main trial; validation: 1 fresh E/C pair, target N={stats.get('planned_N')}"+
            '\nSums exclude process wall gaps and model/input preparation; observed checks occur outside timers',fontsize=9)
        save(fig,f'cumulative_time_b{batch}.png')
    manifest=dict(figures={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in made},
        inputs=['trial_summary.csv','timings.csv','break_even_summary.json','cumulative_validation.csv','cumulative_validation_summary.json'],
        no_old_performance_inputs=True,units='ms per batch call; cumulative seconds',SD='sample SD ddof=1; not CI')
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
