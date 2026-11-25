"""Only current-run measurements feed paired costs and cumulative predictions."""
from fractions import Fraction
import math
from pathlib import Path
import re
import statistics as st
from common import read_json, write_json, write_csv, sha256
from fastpath_math import costs, solve, G, T, duration_ms, NUMERIC_RULE
from fastpath_support import find_case, result_for, compare_pair
from analyze import warm_state


def valid(row):
    v=row.get('elapsed_ms')
    return row.get('status')=='ok' and row.get('timing_valid') and isinstance(v,(int,float)) and math.isfinite(v) and v>0


def mean_range(values):
    return dict(values=values,median=st.median(values) if values else None,
                minimum=min(values) if values else None,maximum=max(values) if values else None,count=len(values))


def output_pair(run,stage,batch,trial,cfg,manifest):
    """Primary pair plus first/last warmup/last work comparisons, all current run."""
    p=compare_pair(run,stage,batch,trial)
    if not p.get('structural_finite_ok'):return p
    from attention_analysis import compare_files
    import torch
    assets=read_json(run/'assets.json');rs=[result_for(run,find_case(manifest,stage,mode,batch,trial)) for mode in ['eager','compile']]
    phases=sorted(set(rs[0]['outputs']) & set(rs[1]['outputs']))
    p['phase_comparisons']=[]
    for phase in phases:
        e,c=[Path(r['outputs'][phase]) for r in rs]
        v=compare_files(c,e,assets['input_ids'][:batch])
        if cfg['atol'] is not None:
            u,z=[torch.load(path,map_location='cpu',weights_only=True).float() for path in [e,c]]
            v.update(allclose=bool(torch.allclose(u,z,atol=cfg['atol'],rtol=cfg['rtol'])),tolerance_status='specified')
        p['phase_comparisons'].append(dict(phase=phase,**v))
        if not v['structural_finite_ok'] or v['top1_matches']!=batch or v.get('allclose') is False:
            p.update(eligible=False,reason='One of first/last warmup/last repeat phase outputs failed consistency')
    return p


def worker_summary(run,w,cfg,assets):
    r=result_for(run,w);rows=r.get('timings',[])
    completed=[x for x in rows if x['phase']=='repeat' and x['status']=='ok']
    values=[x['elapsed_ms'] for x in completed if valid(x)]
    state,why=warm_state(values)
    c=costs(r,cfg['warmup'],cfg['repeats'])
    first=[x['elapsed_ms'] for x in rows if x['phase']=='first' and valid(x)]
    warm=[x['elapsed_ms'] for x in rows if x['phase']=='warmup' and valid(x)]
    s=dict(run_id=cfg['run_id'],case_id=w['case_id'],mode=w['mode'],batch=w['batch'],trial=w['trial'],status=w['status'],
        model_id=assets['model_id'],weights_id=assets['weights_id'],shape=assets['input_shapes'][str(w['batch'])],dtype='torch.bfloat16',
        fastpath=False,cpu_fallback=False,requested_repeats=cfg['repeats'],completed_repeats=len(completed),valid_repeats=len(values),
        model_input_prepare_ms=r.get('model_input_prepare_ms'),runner_setup_ms=r.get('runner_setup_ms'),
        first_call_ms=first[0] if len(first)==1 else None,warmup_ms=warm,
        warmup_sum_ms=sum(warm) if len(warm)==cfg['warmup'] else None,A_ms=float(c['A']) if c['status']=='valid' else None,
        mean_ms=st.mean(values) if values else None,median_ms=st.median(values) if values else None,
        stdev_ms=st.stdev(values) if len(values)>1 else None,CV=st.stdev(values)/st.mean(values) if len(values)>1 else None,
        min_ms=min(values) if values else None,max_ms=max(values) if values else None,
        throughput_images_s=w['batch']*len(values)*1000/sum(values) if values and c['status']=='valid' else None,
        cost_status=c['status'],cost_reason=c.get('reason'),warm_state_assessment=state,warm_state_reason=why,
        validation_status='structure_finite_tolerance_pending' if r.get('output_valid') and cfg['atol'] is None else 'specified_or_unavailable',
        timing_scope=r.get('timing_scope'),cache_policy=r.get('cache_state'),input_hash=r.get('input_hash'),
        weights_hash=r.get('weights_sha256'),state_dict_bf16_hash=r.get('state_dict_bf16_cpu_sha256'),
        path_status='strict diagnostic NPU evidence; all-op placement unverified',compiler_time_ms=None)
    return s,c


def diagnostic_observations(run,m):
    out=[]
    for w in m['workers']:
        if w['stage']!='diagnostic':continue
        r=result_for(run,w);p=run/'logs'/f"{w['case_id']}.stderr.log"
        text=p.read_text() if p.exists() else ''
        out.append(dict(case_id=w['case_id'],mode=w['mode'],batch=w['batch'],status=w['status'],
            latest_attempt=w is find_case(m,'diagnostic',w['mode'],w['batch']),
            **r.get('path_observations',{}),
            graph_segments=[dict(submodules=int(a),to_compile=int(b)) for a,b in re.findall(r'split into (\d+) submodules, (\d+) to compile',text)],
            observed_recompile_log_entries=sum('Recompiling function' in line for line in text.splitlines()),
            graph_break_log_lines=[line for line in text.splitlines() if '[__graph_breaks]' in line],
            guard_failure_log_lines=[line for line in text.splitlines() if '[__recompiles]' in line],
            exact_codegen_count=None,
            error=r.get('error'),raw_log=str(p),cpu_trace=str(run/'cases'/w['case_id']/'torch_cpu_trace.json'),
            npu_trace=str(run/'cases'/w['case_id']/'furiosa_trace.json')))
    return out


def initial_and_work(result,warmups):
    """Recover only a contiguous observed prefix; never skip a failed work call."""
    try:
        s=duration_ms(result.get('runner_setup_ms'))
        rows=result.get('timings',[])
        first=[x for x in rows if x['phase']=='first']
        warm=sorted([x for x in rows if x['phase']=='warmup'],key=lambda x:x['iteration'])
        if len(first)!=1 or len(warm)!=warmups or any(not valid(x) for x in first+warm):raise LookupError('Initial calls incomplete/invalid')
        if [x['iteration'] for x in warm]!=list(range(1,warmups+1)):raise LookupError('Warmup order invalid')
        f=duration_ms(first[0]['elapsed_ms'],True);w=sum((duration_ms(x['elapsed_ms'],True) for x in warm),Fraction())
        work=[]
        for x in sorted([x for x in rows if x['phase']=='work'],key=lambda x:x['iteration']):
            if x['iteration']!=len(work)+1 or not valid(x):break
            work.append(duration_ms(x['elapsed_ms'],True))
        return dict(status='valid_prefix',S=s,F=f,W=w,A=s+f+w,work=work)
    except (LookupError,ValueError,KeyError) as e:return dict(status='insufficient_data',reason=str(e))


def analyze_cumulative(run,cfg,m,be,batches,outputs):
    rows=[];summaries=[]
    for batch in [1,4]:
        b=next(x for x in batches if x['batch']==batch)
        trial=b['representative_trial']
        pred=next((x for x in be if x['batch']==batch and x['trial']==trial),None)
        ws=[find_case(m,'cumulative',mode,batch) for mode in ['eager','compile']]
        rs=[result_for(run,w) for w in ws]
        cs=[initial_and_work(r,cfg['warmup']) for r in rs]
        output=next(x for x in outputs if (x['stage'],x['batch'])==('cumulative',batch))
        summary=dict(batch=batch,status='predicted_only',prediction_source_trial=trial,planned_N=ws[0].get('target_calls'),
            eager_case_id=ws[0]['case_id'],compile_case_id=ws[1]['case_id'],first_observed_compile_advantage_N=None,
            final_winning_suffix_start_N=None,transitions=[],checkpoints=[],completed_common_calls=0,
            scope='sum of S/F/W/work measured intervals; outside-timer checks/logs/prewait and process wall excluded',
            validation_output_eligible=output['eligible'],reason='Cumulative workers not yet completed')
        if pred and all(c['status']=='valid_prefix' for c in cs):
            e,c=cs;count=min(len(e['work']),len(c['work']))
            observed=[e['A'],c['A']];state_before=None;transitions=[];first_win=None;last_win_entry=None
            ps=[result_for(run,find_case(m,'performance',mode,batch,trial)) for mode in ['eager','compile']]
            pc=[costs(r,cfg['warmup'],cfg['repeats']) for r in ps]
            for n in range(0,count+1):
                if n:
                    observed[0]+=e['work'][n-1];observed[1]+=c['work'][n-1]
                # Exact primary prediction reconstructed from this run's representative trial.
                prediction=[q['A']+n*q['t'] for q in pc]
                diff=observed[1]-observed[0]
                row=dict(run_id=cfg['run_id'],batch=batch,N=n,K=n+cfg['warmup']+1,effective_images=batch*n,
                    prediction_source_trial=trial,observed_eager_ms=float(observed[0]),observed_compile_ms=float(observed[1]),
                    observed_difference_ms=float(diff),predicted_eager_ms=float(prediction[0]),predicted_compile_ms=float(prediction[1]),
                    predicted_difference_ms=float(prediction[1]-prediction[0]),output_validation_status='eligible' if output['eligible'] else 'incomplete_or_failed')
                rows.append(row)
                if n:
                    state='compile_faster' if diff<0 else 'tie' if diff==0 else 'eager_faster'
                    if state!=state_before:transitions.append(dict(N=n,state=state,difference_ms=float(diff)))
                    if state=='compile_faster' and first_win is None:first_win=n
                    if state=='compile_faster' and state_before!='compile_faster':last_win_entry=n
                    state_before=state
            requested=ws[0].get('target_calls')
            normal=all(w['status']=='ok' and r.get('status')=='ok' for w,r in zip(ws,rs)) and requested==count and output['eligible']
            if normal and first_win is not None:status='observed_crossing_reached'
            elif normal:status='prediction_not_confirmed_within_observed_range'
            else:status='not_reached_within_budget_or_incomplete'
            summary.update(status=status,completed_common_calls=count,first_observed_compile_advantage_N=first_win if output['eligible'] else None,
                final_winning_suffix_start_N=last_win_entry if normal and state_before=='compile_faster' else None,
                observed_end_difference_ms=float(observed[1]-observed[0]),transitions=transitions,
                reason=None if normal else 'Incomplete/invalid paired execution; raw prefixes preserved',
                first_observed_K=first_win+1+cfg['warmup'] if first_win and output['eligible'] else None,
                first_observed_effective_images=batch*first_win if first_win and output['eligible'] else None,
                observed_startup={mode:{k+'_ms':float(cost[k]) for k in ['S','F','W','A']} for mode,cost in zip(['eager','compile'],cs)},
                observed_work_mean_ms={mode:float(sum(cost['work'],Fraction())/len(cost['work'])) if cost['work'] else None for mode,cost in zip(['eager','compile'],cs)},
                primary_median_N=b['N_win']['median'],primary_trial_N_range=[b['N_win']['minimum'],b['N_win']['maximum']])
            if first_win and b['N_win']['median']:
                summary['observed_minus_predicted_median_N']=first_win-b['N_win']['median']
                summary['relative_crossing_difference']=first_win/b['N_win']['median']-1
            ns=[max(1,math.floor(Fraction(4,5)*Fraction(str(b['N_win']['median'])))),int(b['N_win']['median']),math.ceil(Fraction(6,5)*Fraction(str(b['N_win']['median'])))]
            for label,n in zip(['0.8x','1.0x','1.2x'],ns):
                row=next((r for r in rows if r['batch']==batch and r['N']==n),None)
                summary['checkpoints'].append(dict(label=label,N=n,observed=row,reason=None if row else 'Not reached in saved common prefix'))
            # Explicitly post-hoc explanation only: never replace primary A/t curves.
            summary['posthoc_observed_startup_with_primary_rates']=solve(e['A'],c['A'],pc[0]['t'],pc[1]['t'],batch,cfg['warmup'])
            summary['posthoc_note']='Uses new cumulative startup with unchanged main-trial rates; not the primary prediction or an extra independent trial'
        summaries.append(summary)
    fields=['run_id','batch','N','K','effective_images','prediction_source_trial','observed_eager_ms','observed_compile_ms',
        'observed_difference_ms','predicted_eager_ms','predicted_compile_ms','predicted_difference_ms','output_validation_status']
    write_csv(run/'cumulative_validation.csv',rows,fields)
    result=dict(batches=summaries,folded_into_independent_trials=False,primary_predictions_refitted=False,
        cumulative_output_checks='every work output copied to CPU and checked outside model timer; no per-output files',
        outside_timer_gap_limit='Output checks and scheduling between calls can affect execution; sums are not process wall time')
    write_json(run/'cumulative_validation_summary.json',result)
    return result


def plan_cumulative(run):
    run=Path(run);cfg,m,be=[read_json(run/n) for n in ['config.json','manifest.json','break_even_summary.json']]
    if any(w['status']=='running' for w in m['workers']):raise RuntimeError('Plan cumulative workers only between active workers')
    plans=[];remaining=max(0,cfg['total_budget_s']-m['npu_budget_used_s'])
    for batch in [1,4]:
        ws=[find_case(m,'cumulative',mode,batch) for mode in ['eager','compile']]
        if any(w.get('started_utc') for w in ws):
            plans.append(dict(batch=batch,status='existing_plan_preserved',target_calls=ws[0].get('target_calls')))
            continue
        paired=[v for v in be['trials'] if v['batch']==batch]
        good=[v for v in paired if v['status'] in ['finite_break_even','already_beneficial','always_beneficial'] and v['N_win'] is not None]
        reason=None
        if len(good)!=len(paired) or not good:reason='No uniformly valid finite paired prediction; preserve mixed/no-crossing/insufficient statuses'
        if reason:
            for w in ws:w.update(status='held',reason=reason)
            plans.append(dict(batch=batch,status='predicted_only',reason=reason));continue
        median=st.median([v['N_win'] for v in good]);desired=math.ceil(Fraction(6,5)*Fraction(str(median)))
        if median==1:desired=max(20,desired)
        requested=min(desired,cfg['max_cumulative_calls'])
        def bound(mode,n):
            values=[]
            for trial in range(1,cfg['trials']+1):
                w=find_case(m,'performance',mode,batch,trial);r=result_for(run,w);c=costs(r,cfg['warmup'],cfg['repeats'])
                if c['status']=='valid':values.append((r['model_input_prepare_ms']+float(c['A']+n*c['t']))/1000)
            return 1.5*max(values)+60+n*.001
        def fits(n):
            bounds=[bound(mode,n) for mode in ['eager','compile']]
            return max(bounds)<=cfg['worker_timeout_s'] and sum(bounds)<=remaining
        low,high=0,requested
        while low<high:
            mid=(low+high+1)//2
            if fits(mid):low=mid
            else:high=mid-1
        n=low
        if n==0:
            reason='Conservative new-trial estimate cannot fit both fresh workers in remaining budget'
            for w in ws:w.update(status='held',reason=reason)
            plans.append(dict(batch=batch,status='predicted_only',reason=reason,desired_calls=desired));continue
        bounds={mode:bound(mode,n) for mode in ['eager','compile']};remaining-=sum(bounds.values())
        for w in ws:w.update(status='not_run',target_calls=n)
        plans.append(dict(batch=batch,status='planned_full_target' if n>=desired else 'not_reached_within_budget_or_call_cap',
            median_N_win=median,desired_calls=desired,target_calls=n,max_calls=cfg['max_cumulative_calls'],
            conservative_worker_wall_bounds_s=bounds,remaining_reserved_budget_s=remaining,
            prediction_inputs='current performance trials only; previous experiments excluded'))
    write_json(run/'cumulative_plan.json',dict(plans=plans,budget_used_at_planning_s=m['npu_budget_used_s'],
        total_budget_s=cfg['total_budget_s'],rule=cfg['cumulative_wall_bound_rule']))
    write_json(run/'manifest.json',m)


def analyze(run):
    run=Path(run);cfg,m,assets=[read_json(run/n) for n in ['config.json','manifest.json','assets.json']]
    timings=[];summary=[];components={};output=[];be=[];prefix=[]
    for w in m['workers']:
        r=result_for(run,w)
        if w['stage']=='performance':
            for row in r.get('timings',[]):
                timings.append(dict(**row,dtype='torch.bfloat16',fastpath=False,cpu_fallback=False,
                    input_hash=r.get('input_hash'),weights_hash=r.get('weights_sha256'),cache_policy=r.get('cache_state')))
            if w is find_case(m,'performance',w['mode'],w['batch'],w['trial']):
                s,c=worker_summary(run,w,cfg,assets);summary.append(s);components[w['case_id']]=c
    for stage in ['diagnostic','performance','cumulative']:
        for b in [1,4]:
            for trial in range(1,cfg['trials']+1) if stage=='performance' else [1]:
                output.append(output_pair(run,stage,b,trial,cfg,m))
    for b in [1,4]:
        for trial in range(1,cfg['trials']+1):
            ws=[find_case(m,'performance',mode,b,trial) for mode in ['eager','compile']]
            cs=[components[w['case_id']] for w in ws]
            eligible=next(p for p in output if (p['stage'],p['batch'],p['trial'])==('performance',b,trial))
            if eligible['eligible'] and all(c['status']=='valid' for c in cs):
                e,c=cs;v=solve(e['A'],c['A'],e['t'],c['t'],b,cfg['warmup'])
                for mode,cost in zip(['eager','compile'],cs):
                    v.update({f'S_{mode}_ms':float(cost['S']),f'F_{mode}_ms':float(cost['F']),f'W_{mode}_ms':float(cost['W'])})
                v['compile_speedup']=float(e['t']/c['t'])
                ss=[next(s for s in summary if s['case_id']==w['case_id']) for w in ws]
                pooled=math.sqrt(sum(s['stdev_ms']**2 for s in ss))
                v['variation_assessment']='uncertain' if abs(v['delta_t_ms'])<=pooled else 'mean_gap_exceeds_descriptive_combined_SD'
                v['combined_repeat_SD_ms']=pooled
                v['warm_state_assessments']={s['mode']:s['warm_state_assessment'] for s in ss}
                v['limitations']=['Output tolerance pending' if cfg['atol'] is None else 'Specified tolerance only',
                    'All-op placement/codegen/disk cache unverified','Only three independent paired trials']
                if any(s['warm_state_assessment']!='preliminary_stable' for s in ss):v['limitations'].append('One or both workers not preliminary_stable')
                for k in range(1,2+cfg['warmup']):
                    prefix.append(dict(batch=b,trial=trial,K=k,G_eager_ms=float(G(e,k)),G_compile_ms=float(G(c,k)),
                        difference_ms=float(G(c,k)-G(e,k)),scope='actual initial measured interval sum, excludes prepare and outside-timer gaps'))
                v['initial_K_with_compile_advantage']=[p['K'] for p in prefix if p['batch']==b and p['trial']==trial and p['difference_ms']<0]
                if v['N_win'] is not None:
                    n=v['N_win'];v['G_Nplus_initial_equals_T']=G(e,n+1+cfg['warmup'])==T(e,n) and G(c,n+1+cfg['warmup'])==T(c,n)
                    assert v['G_Nplus_initial_equals_T']
            else:
                v=solve(None,None,None,None,b,cfg['warmup']);v['reason']=eligible.get('reason') or str([c.get('reason') for c in cs])
            v.update(run_id=cfg['run_id'],batch=b,trial=trial,eager_case_id=ws[0]['case_id'],compile_case_id=ws[1]['case_id'],
                     output_validation_status='eligible_tolerance_pending' if eligible['eligible'] and cfg['atol'] is None else 'specified_or_held')
            be.append(v)
    by_batch=[]
    for b in [1,4]:
        paired=[v for v in be if v['batch']==b];good=[v for v in paired if v['N_win'] is not None and v['status'] in ['finite_break_even','already_beneficial','always_beneficial']]
        representative=sorted(good,key=lambda x:(x['N_win'],x['trial']))[len(good)//2] if good else None
        by_batch.append(dict(batch=b,statuses=[dict(trial=v['trial'],status=v['status'],N_win=v['N_win']) for v in paired],
            valid_finite_paired_trials=len(good),requested_paired_trials=cfg['trials'],N_win=mean_range([v['N_win'] for v in good]),
            representative_trial=representative['trial'] if representative else None,
            mode_mean_ms={mode:mean_range([s['mean_ms'] for s in summary if s['batch']==b and s['mode']==mode and s['cost_status']=='valid']) for mode in ['eager','compile']}))
    write_csv(run/'timings.csv',timings,['run_id','case_id','stage','mode','batch','trial','phase','iteration','elapsed_ns','elapsed_ms','timing_valid','status',
        'dtype','fastpath','cpu_fallback','input_hash','weights_hash','cache_policy'])
    write_csv(run/'trial_summary.csv',summary,list(summary[0]) if summary else ['case_id','status'])
    be_fields=['run_id','batch','trial','eager_case_id','compile_case_id','status','A_e_ms','A_c_ms','t_e_ms','t_c_ms',
        'delta_A_ms','delta_t_ms','real_crossing_N','N_win','K_win','effective_images','forward_count','including_preparation_images',
        'compile_speedup','last_advantage_N','tie_N','first_eager_strict_N','strict_at_N','predecessor_N','difference_at_predecessor_ms',
        'difference_at_N_ms','variation_assessment','output_validation_status','reason']
    write_csv(run/'break_even_trials.csv',be,be_fields)
    write_json(run/'break_even_summary.json',dict(numeric_rule=NUMERIC_RULE,scenario='T(N)=S+F+sum(W)+N*mean_repeat; N>=1 additional work batch calls after first+warmups',
        prepare_excluded=True,process_wall_excluded=True,pure_compiler_time_ms=None,old_measurements_used=False,trials=be,batches=by_batch))
    write_json(run/'output_comparison.json',dict(unique_real_images=4,synthetic_input=False,accuracy_computed=False,loss_computed=False,
        tolerance_status='pending' if cfg['atol'] is None else 'specified',pairs=output))
    write_json(run/'initial_prefix.json',dict(rows=prefix,K_lte_initial_count='actual measured prefix sum',K_gt_initial_count='mean-repeat prediction'))
    observations=diagnostic_observations(run,m);write_json(run/'diagnostic_observations.json',observations)
    cumulative=analyze_cumulative(run,cfg,m,be,by_batch,output)
    logical=[w for w in m['workers'] if w is find_case(m,w['stage'],w['mode'],w['batch'],w['trial'])]
    m.update(all_workers_complete=all(w['status']=='ok' for w in logical),
        all_unheld_workers_complete=all(w['status']=='ok' for w in logical if w['status']!='held'),
        all_attempts_successful=all(w['status']=='ok' for w in m['workers']),logical_worker_count=len(logical),attempt_count=len(m['workers']))
    # Never replace an active orchestrator's heartbeat/budget with an older snapshot.
    if not any(w['status']=='running' for w in m['workers']):
        write_json(run/'manifest.json',m)
    protected={p:sha256(Path(p))==value for p,value in m['protected_files_sha256'].items()}
    write_json(run/'artifact_audit.json',dict(status='pass' if all(protected.values()) else 'fail',
        performance_rows=len(timings),expected_rows=m['expected_performance_rows'],
        completed_performance_workers=sum(s['status']=='ok' for s in summary),
        all_workers_complete=m['all_workers_complete'],all_unheld_workers_complete=m['all_unheld_workers_complete'],
        all_attempts_successful=m['all_attempts_successful'],protected_files_unchanged=protected,
        numeric_rule=NUMERIC_RULE,npu_budget_used_s=m['npu_budget_used_s'],budget_s=cfg['total_budget_s']))
    try:
        from fastpath_compare_report import report
        report(run,cfg,m,summary,be,by_batch,output,observations,cumulative)
    except ModuleNotFoundError:
        pass  # Analysis tables remain available while the report-only module is prepared.
    return dict(summary=summary,break_even=be,batches=by_batch,cumulative=cumulative)
