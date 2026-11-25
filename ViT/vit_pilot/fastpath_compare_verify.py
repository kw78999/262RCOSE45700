"""Independent CSV/ns recomputation; never launches a device worker."""
import argparse
import csv
from decimal import Decimal,ROUND_HALF_EVEN
from fractions import Fraction
import math
from pathlib import Path
import statistics
from common import read_json,write_json,sha256,now
from fastpath_support import find_case,result_for


def csv_rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def ns(v):
    return int((Decimal(str(v))*Decimal(1000000)).quantize(Decimal(1),rounding=ROUND_HALF_EVEN))


def close(a,b):
    assert a is not None and b is not None
    assert abs(float(a)-float(b))<=max(1e-8,abs(float(b))*1e-12),(a,b)


def verify(run):
    run=Path(run);cfg,m,assets=[read_json(run/n) for n in ['config.json','manifest.json','assets.json']]
    timings=csv_rows(run/'timings.csv');summaries=csv_rows(run/'trial_summary.csv')
    be=read_json(run/'break_even_summary.json');checks=[];initial={};costs={};count=0
    for w in m['workers']:
        if w is not find_case(m,w['stage'],w['mode'],w['batch'],w['trial']):continue
        r=result_for(run,w)
        if w['status']!='ok':continue
        assert r['status']=='ok' and r['output_valid'] and r['sync_checked'] and r['original_model_no_child_compile']
        assert r['input_hash']==assets['batch_hashes'][str(w['batch'])] and r['weights_sha256']==assets['weights_sha256']
        assert r['state_dict_bf16_cpu_sha256']==read_json(run/'asset_verification.json')['state_dict_bf16_cpu_sha256']
        assert not r['effective']['cpu_fallback'] and not r['effective']['mha_fastpath'] and not r['effective']['suppress_errors'] and not r['effective']['backend_eager_fallback']
        assert all(not s['mha_fastpath'] and not s['cpu_fallback'] for s in r['control_snapshots'].values())
        assert r['cache_existed_at_entry'] is False
        for output in r['output_checks'].values():
            assert output['shape']==[w['batch'],1000] and output['dtype']=='torch.bfloat16' and output['nan_count']==output['inf_count']==0
        if w['stage']=='performance':
            rows=[t for t in timings if t['case_id']==w['case_id']]
            assert len(rows)==1+cfg['warmup']+cfg['repeats']
            assert all(t['status']=='ok' and t['timing_valid']=='True' and float(t['elapsed_ms'])>0 for t in rows)
            assert all(int(t['elapsed_ns'])==ns(t['elapsed_ms']) for t in rows)
            assert all(t['fastpath']=='False' and t['cpu_fallback']=='False' for t in rows)
            count+=len(rows)
            parts={p:[t for t in rows if t['phase']==p] for p in ['first','warmup','repeat']}
            for p,n in [('first',1),('warmup',cfg['warmup']),('repeat',cfg['repeats'])]:
                assert [int(t['iteration']) for t in parts[p]]==list(range(1,n+1))
            repeat=[float(t['elapsed_ms']) for t in parts['repeat']]
            s=next(s for s in summaries if s['case_id']==w['case_id'])
            close(s['mean_ms'],statistics.mean(repeat));close(s['median_ms'],statistics.median(repeat));close(s['stdev_ms'],statistics.stdev(repeat))
            close(s['throughput_images_s'],w['batch']*len(repeat)*1000/sum(repeat))
            sn=ns(r['runner_setup_ms']);fn=int(parts['first'][0]['elapsed_ns']);wn=sum(int(t['elapsed_ns']) for t in parts['warmup'])
            an=sn+fn+wn;tn=Fraction(sum(int(t['elapsed_ns']) for t in parts['repeat']),len(repeat))
            close(s['A_ms'],Fraction(an,1000000));close(s['warmup_sum_ms'],Fraction(wn,1000000))
            key=(w['batch'],w['trial'],w['mode']);costs[key]=(Fraction(an),tn)
            initial[key]=(sn,[int(t['elapsed_ns']) for t in parts['first']+parts['warmup']])
            checks.append(dict(case_id=w['case_id'],status='pass',rows=len(rows),metrics_and_A_recomputed=True))
    for v in be['trials']:
        if v['status'] in ['insufficient_data','invalid']:continue
        ae,te=costs[v['batch'],v['trial'],'eager'];ac,tc=costs[v['batch'],v['trial'],'compile']
        da,dt=ac-ae,te-tc
        close(v['A_e_ms'],ae/1000000);close(v['A_c_ms'],ac/1000000)
        close(v['delta_A_ms'],da/1000000);close(v['delta_t_ms'],dt/1000000)
        if dt>0:
            n=max(1,math.floor(da/dt)+1)
            assert n==v['N_win'] and da-n*dt<0
            assert n==1 or da-(n-1)*dt>=0
            assert v['K_win']==n+cfg['warmup']+1 and v['effective_images']==v['batch']*n
            assert v['including_preparation_images']==v['batch']*(n+cfg['warmup']+1)
        close(v['compile_speedup'],te/tc)
    for p in read_json(run/'initial_prefix.json')['rows']:
        for mode,field in [('eager','G_eager_ms'),('compile','G_compile_ms')]:
            sn,calls=initial[p['batch'],p['trial'],mode]
            close(p[field],Fraction(sn+sum(calls[:p['K']]),1000000))
    cumulative=csv_rows(run/'cumulative_validation.csv');csummary=read_json(run/'cumulative_validation_summary.json')
    for batch in [1,4]:
        saved=[r for r in cumulative if int(r['batch'])==batch]
        if not saved:continue
        rs={mode:result_for(run,find_case(m,'cumulative',mode,batch)) for mode in ['eager','compile']}
        initial_ns={mode:ns(r['runner_setup_ms'])+sum(ns(t['elapsed_ms']) for t in r['timings'] if t['phase'] in ['first','warmup']) for mode,r in rs.items()}
        work={mode:[ns(t['elapsed_ms']) for t in r['timings'] if t['phase']=='work'] for mode,r in rs.items()}
        sums=dict(initial_ns);first_win=None;last_entry=None;prev=None
        for row in saved:
            n=int(row['N']);assert n>=0
            for mode in ['eager','compile']:
                if n:sums[mode]+=work[mode][n-1]
                close(row['observed_'+mode+'_ms'],Fraction(sums[mode],1000000))
                aa,tt=costs[batch,int(row['prediction_source_trial']),mode]
                close(row['predicted_'+mode+'_ms'],(aa+n*tt)/1000000)
            assert int(row['effective_images'])==batch*n and int(row['K'])==n+cfg['warmup']+1
            if n:
                win=sums['compile']<sums['eager']
                if win and first_win is None:first_win=n
                if win and prev is not True:last_entry=n
                prev=win
        stat=next(b for b in csummary['batches'] if b['batch']==batch)
        if stat['validation_output_eligible']:
            assert stat['first_observed_compile_advantage_N']==first_win
            assert stat['final_winning_suffix_start_N']==(last_entry if prev else None)
            for mode,r in rs.items():assert r['cumulative_work_outputs_checked']==len(work[mode])
    for path,digest in m['protected_files_sha256'].items():assert sha256(Path(path))==digest
    assert m['npu_budget_used_s']<=cfg['total_budget_s']
    completed=sum(w['stage']=='performance' and w['status']=='ok' and w is find_case(m,'performance',w['mode'],w['batch'],w['trial']) for w in m['workers'])
    if completed==4*cfg['trials']:assert count==m['expected_performance_rows']
    result=dict(status='pass',utc=now(),checks=checks,successful_performance_rows=count,expected_if_all_completed=m['expected_performance_rows'],
        direct_CSV_recalculation=True,exact_ns_boundary_checks=True,initial_prefix_no_double_count=True,
        cumulative_prefix_recomputed=True,old_measurements_in_new_prediction=False,protected_files_unchanged=True,
        all_workers_complete=m['all_workers_complete'],all_unheld_workers_complete=m['all_unheld_workers_complete'],
        all_attempts_successful=m['all_attempts_successful'])
    write_json(run/'independent_verification.json',result);print('Independent verification PASS:',count,'successful performance rows')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
