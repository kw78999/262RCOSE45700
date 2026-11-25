"""Plot only recorded follow-up trial values in the isolated matplotlib environment."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root):
    d=json.loads((root/'figures/plot_data.json').read_text());c=d['config'];rows=d['rows']
    valid=[x for x in rows if x['status']=='ok' and x['mean_ms'] is not None]
    if not valid:return
    colors={'eager':'#276b9d','compile':'#d88425'}
    markers=['o','s','^','D','v']
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,b in zip(axes,[1,4]):
        for pos,mode in enumerate(['eager','compile']):
            items=[x for x in valid if x['batch']==b and x['mode']==mode]
            if not items:
                ax.text(pos,.6,'Held\nCPU fallback' if mode=='eager' else 'Not measured',transform=ax.get_xaxis_transform(),ha='center',color='gray')
            for s in items:
                offset=(s['trial']-(c['trials']+1)/2)*.13
                ax.errorbar(pos+offset,s['mean_ms'],yerr=s['stdev_ms'],fmt=markers[(s['trial']-1)%len(markers)],
                            color=colors[mode],capsize=4,label=f'T{s["trial"]}' if pos==1 else None)
                ax.annotate(f'T{s["trial"]}: {s["mean_ms"]:.3f}',(pos+offset,s['mean_ms']),xytext=(0,10+14*(s['trial']%2)),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks([0,1],['Eager','Compile']);ax.set_xlim(-.5,1.5);ax.set_yscale('log')
        ax.set_ylabel('repeat mean ms / batch call (log scale)');ax.set_title(f'B{b}: independent trial observations');ax.grid(axis='y',alpha=.2)
        ax.margins(y=.6)
    fig.suptitle(f'PRELIMINARY ViT BF16 / default 1 PE; real photos=4; {c["trials"]} requested independent trials, {c["repeats"]} repeats each')
    fig.text(.5,.015,'Error bars: within-trial sample SD, not CI. New SDK cache path/worker; system cold unverified. Tolerance pending; all-op placement unverified.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.91));fig.savefig(root/'figures/01_repeat_trials.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,b in zip(axes,[1,4]):
        items=[s for s in valid if s['batch']==b and s['first_call_ms'] is not None]
        labels=[f'{s["mode"]} T{s["trial"]}' for s in items]
        for i,s in enumerate(items):
            ax.bar(i,s['first_call_ms'],color=colors[s['mode']]);ax.annotate(f'{s["first_call_ms"]:.1f}',(i,s['first_call_ms']),xytext=(0,7),textcoords='offset points',ha='center')
        ax.set_xticks(range(len(items)),labels);ax.set_ylabel('first call ms / batch call');ax.set_title(f'B{b}: one observation per independent trial');ax.margins(y=.2);ax.grid(axis='y',alpha=.2)
    fig.suptitle(f'PRELIMINARY ViT BF16 / default 1 PE / 4 real photos; {c["trials"]} independent trials\nFirst calls separate from repeats; Eager conditions with CPU fallback held',fontsize=12)
    fig.text(.5,.012,'Absent private SDK cache path per worker; system cold / actual disk miss unverified. Not pure compiler time.\nTolerance pending; all-op placement unverified. One first-call observation per trial; no confidence interval.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.88));fig.savefig(root/'figures/02_first_calls.png',dpi=160);plt.close(fig)


if __name__=='__main__':main(Path(sys.argv[1]))
