"""Recompute descriptive statistics and cautious scaling/report from persisted raw calls."""
import argparse
from collections import defaultdict, Counter
import csv
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vit_pilot'))
from common import read_json,write_json,write_csv,now,sha256


def quantile(v,q):
    v=sorted(v);pos=(len(v)-1)*q;a=int(pos);b=min(a+1,len(v)-1)
    return v[a]+(v[b]-v[a])*(pos-a)


def stats(v):
    if not v:return {}
    mean=st.mean(v);sd=st.stdev(v) if len(v)>1 else None
    return dict(n=len(v),mean_ms=mean,median_ms=st.median(v),sample_sd_ms=sd,cv=sd/mean if sd is not None else None,
        p95_ms=quantile(v,.95),p99_ms=quantile(v,.99),min_ms=min(v),max_ms=max(v),sum_ms=sum(v))


def observation_end(m):
    """Reanalysis time is distinct from the experiment's recorded stop time."""
    if m.get('status')!='running' and m.get('updated_utc'):
        return datetime.fromisoformat(m['updated_utc'])
    return datetime.now(timezone.utc)


def failure_category(w,r):
    error=r.get('error','')
    if 'fatal error: assert.h: No such file or directory' in error:
        return 'host_c_header_missing'
    return w['status']


def load(run):
    m=read_json(run/'manifest.json');rows=[]
    for w in m['workers']:
        p=run/'cases'/w['case_id']/'result.json'
        r=read_json(p) if p.exists() else {}
        rows.append((w,r))
    return m,rows


def compare(x,y):
    import torch
    if x.shape!=y.shape:return dict(status='structural_error',shape_x=list(x.shape),shape_y=list(y.shape))
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():return dict(status='nonfinite')
    diff=(x-y).abs();cos=torch.nn.functional.cosine_similarity(x,y,dim=-1)
    matches=x.argmax(-1)==y.argmax(-1)
    return dict(status='finite_tolerance_pending',shape=list(x.shape),max_abs_error=diff.max().item(),
        mean_abs_error=diff.mean().item(),cosine_mean=cos.mean().item(),cosine_min=cos.min().item(),
        top1_matches=int(matches.sum()),top1_denominator=len(matches),top1_agreement=matches.float().mean().item(),
        numeric_tolerance=None)


def diagnostic_evidence(workers):
    """Allocation, task observation, and utilization are distinct claims."""
    evidence={}
    for w,r in workers:
        if w['stage']!='diagnostic' or w['status']!='ok':continue
        observed=r.get('path_observations',{})
        clusters=sorted({str(x['cluster_index']) for x in observed.get('task_arg_configurations',[]) if x.get('cluster_index') is not None})
        count=observed.get('npu_task_events')
        evidence[w['pe'],w['mode']]=dict(
            pe_evidence_case=w['case_id'],pe_evidence_batch=w['batch'],
            diagnostic_npu_task_events=count,diagnostic_cluster_indices=clusters,
            npu_task_evidence='observed_in_separate_diagnostic' if count else 'not_observed_in_accepted_diagnostic',
            multi_cluster_task_evidence='both_clusters_observed' if w['pe']==8 and {'0','1'}<=set(clusters) else 'unverified' if w['pe']==8 else 'single_cluster_configuration',
            per_pe_utilization_verified=False,
            all_pe_completion_internals='SDK stream contract; binary implementation unverified')
    return evidence


def validation(run,workers,assets):
    import torch
    outputs={};checks=[];relations=[]
    for w,r in workers:
        if w['status']=='ok' and w['stage'] in ('diagnostic','validation') and r.get('output_file'):
            y=torch.load(r['output_file'],map_location='cpu',weights_only=True).float()
            key=w['pe'],w['batch'],w['mode']; outputs[key]=(y,w,r)
            checks.append(dict(case_id=w['case_id'],pe=w['pe'],batch=w['batch'],mode=w['mode'],
                **r['output_check'],input_hash=r['input_hash'],hash_matches_expected=r['input_hash']==assets['batches'][str(w['batch'])]['tensor_sha256'],
                fusion_requested=w['pe'],fusion_sdk=r.get('fusion'),numeric_status='pending',
                unique_image_count=min(w['batch'],4),repeated_samples_are_not_independent=True))
    for (p,b,mode),(y,w,r) in outputs.items():
        if mode=='compile' and (p,b,'eager') in outputs:
            x,ref,_=outputs[p,b,'eager'];unique=min(b,4)
            relations.append(dict(kind='eager_vs_compile',pe=p,batch=b,mode=mode,case=w['case_id'],reference=ref['case_id'],
                all_batch_rows=compare(x,y),unique_images=compare(x[:unique],y[:unique]),unique_image_count=unique))
        if p!=1 and (1,b,mode) in outputs:
            x,ref,_=outputs[1,b,mode];unique=min(b,4)
            relations.append(dict(kind='one_pe_vs_multi_pe',pe=p,batch=b,mode=mode,case=w['case_id'],reference=ref['case_id'],
                all_batch_rows=compare(x,y),unique_images=compare(x[:unique],y[:unique]),unique_image_count=unique))
        if b>1 and (p,1,mode) in outputs:
            x,ref,_=outputs[p,1,mode]
            relations.append(dict(kind='batch1_vs_larger_batch',pe=p,batch=b,mode=mode,case=w['case_id'],reference=ref['case_id'],
                compared_image_ids=[assets['input_ids'][0]],unique_image_count=1,comparison=compare(x,y[:1])))
        if b>4:
            reference=y[[i%4 for i in range(b)]]
            relations.append(dict(kind='within_batch_repeated_images',pe=p,batch=b,mode=mode,
                unique_image_count=4,repeated_samples_are_not_independent=True,comparison=compare(reference,y)))
    result=dict(status='provisional_numeric_tolerance_pending',checks=checks,comparisons=relations,
        unique_source_images=4,labelled_accuracy_evaluation=False,
        batch1_comparison_scope='first image only; no separate batch1 runs for images 2–4',
        additional_images='No additional input set introduced; original four images reused',
        all_op_placement='unverified',actual_codegen_count=None,compiler_time_ms=None,disk_cache_hit=None,disk_cache_miss=None)
    write_json(run/'output_validation.json',result);return result


def analyze(run,plots=True):
    cfg=read_json(run/'experiment_config.json');assets=read_json(run/'assets.json');m,workers=load(run)
    stop=observation_end(m)
    pe_evidence=diagnostic_evidence(workers)
    latest={}
    for w,r in workers:latest[(w['stage'],w['pe'],w['batch'],w['mode'],w['trial'])]=(w,r)
    timings=[];trials=[];costs=[];stability=[];warmrows=[];diag=[]
    for w,r in workers:
        base={k:w[k] for k in ('case_id','stage','pe','batch','mode','trial','attempt')}
        for row in r.get('timings',[]):
            timings.append(dict(**base,**row,worker_protocol=w.get('worker_protocol',1),worker_status=w['status'],pid=w.get('pid'),started_utc=w.get('started_utc')))
            if row['phase']=='warmup':warmrows.append(dict(**base,**row,worker_protocol=w.get('worker_protocol',1)))
        if w['stage']=='diagnostic':diag.append(dict(**base,status=w['status'],fusion=r.get('fusion'),device_configuration=r.get('device_configuration'),path_observations=r.get('path_observations'),backend_entries=r.get('backend_entries')))
        repeat=[t['elapsed_ms'] for t in r.get('timings',[]) if t['phase']=='repeat']
        first=next((t['elapsed_ms'] for t in r.get('timings',[]) if t['phase']=='first'),None)
        warm=[t['elapsed_ms'] for t in r.get('timings',[]) if t['phase']=='warmup']
        setup=r.get('runner_setup_ms');A=setup+first+sum(warm) if setup is not None and first is not None else None
        if w['stage'] in ('formal','exploration'):
            settings_ok=bool(r.get('controls')) and all(not any(c.values()) for c in r['controls'])
            identity_ok=r.get('input_hash')==assets['batches'][str(w['batch'])]['tensor_sha256'] and r.get('weights_sha256')==assets['weights_sha256'] and r.get('state_dict_bf16_cpu_sha256')==assets['state_dict_bf16_cpu_sha256']
            allocation_ok=r.get('fusion',{}).get('num_fusion')==w['pe']
            logging_protocol_ok=w.get('worker_protocol',1)>=cfg.get('worker_protocol',1)
            data=stats(repeat)
            eligible=w['status']=='ok' and r.get('status')=='ok' and bool(repeat) and settings_ok and identity_ok and allocation_ok and logging_protocol_ok and (w['stage']=='exploration' or r.get('repeat_minimum_met') is True)
            row=dict(**base,status=w['status'],eligible_for_descriptive_performance=eligible,provisional=True,
                worker_protocol=w.get('worker_protocol',1),logging_protocol_current=logging_protocol_ok,
                logging_policy='explicit WARNING outside diagnostics' if w.get('worker_protocol',1)>=2 else 'SDK default Python DEBUG; legacy exception, excluded from current performance aggregates',
                **pe_evidence.get((w['pe'],w['mode']),dict(npu_task_evidence='no_accepted_diagnostic',per_pe_utilization_verified=False)),
                completed_independent_trial=int(w['status']=='ok'),warmup_count=len(warm),repeat_count=len(repeat),repeat_sum_ms=sum(repeat),
                repeat_minimum_met=r.get('repeat_minimum_met'),**{k:v for k,v in data.items() if k not in ('n','sum_ms')},
                raw_observed_images_s=w['batch']*len(repeat)*1000/sum(repeat) if repeat else None,
                images_s=w['batch']*len(repeat)*1000/sum(repeat) if eligible else None,
                amortized_ms_image=data.get('mean_ms',0)/w['batch'] if eligible else None,
                images_s_per_allocated_pe=w['batch']*len(repeat)*1000/sum(repeat)/w['pe'] if eligible else None,
                model_input_prepare_ms=r.get('model_input_prepare_ms'),runner_setup_ms=setup,first_call_ms=first,warmup_sum_ms=sum(warm),A_ms=A,
                compiler_time_ms=None,stabilization_status=r.get('stabilization_status'),output_status='separate validation; tolerance pending',
                controls_verified=settings_ok,input_weight_state_verified=identity_ok,sdk_allocation_verified=allocation_ok,
                input_hash=r.get('input_hash'),cache_path=r.get('cache_path'),pid=r.get('pid'),started_utc=r.get('started_utc'),
                timing_scope=r.get('timing_scope'),quantile_method='linear interpolation (n-1)*q; within-trial calls only')
            trials.append(row)
        incomplete_first_wall=None
        event_file=run/'cases'/w['case_id']/'events.jsonl'
        if first is None and w.get('ended_utc') and event_file.exists():
            events=[json.loads(line) for line in event_file.read_text().splitlines()]
            start=next((e['utc'] for e in events if e['event']=='first_start'),None)
            if start:incomplete_first_wall=(datetime.fromisoformat(w['ended_utc'])-datetime.fromisoformat(start)).total_seconds()
        costs.append(dict(**base,status=w['status'],worker_protocol=w.get('worker_protocol',1),logging_protocol_current=w.get('worker_protocol',1)>=cfg.get('worker_protocol',1),model_input_prepare_ms=r.get('model_input_prepare_ms'),S_ms=setup,F_ms=first,W_ms=sum(warm),A_ms=A,compiler_time_ms=None,
            incomplete_first_attempt_wall_s=incomplete_first_wall,incomplete_first_wall_scope='start event to worker end including termination; not a completed timed_call' if incomplete_first_wall is not None else None))
        if w['stage']=='stability' and w['status']=='ok':
            ts=[t for t in r['timings'] if t['phase']=='stability'];bins=defaultdict(list)
            if ts:
                start=ts[0]['offset_s']
                for t in ts:bins[int((t['offset_s']-start)//10)].append(t['elapsed_ms'])
                for bin_,values in sorted(bins.items()):
                    stability.append(dict(**base,time_start_s=bin_*10,time_end_s=(bin_+1)*10,bin_width_s=10,
                        calls=len(values),**stats(values),images_s=w['batch']*len(values)*1000/sum(values)))
    def csvout(name,rows,default):
        fields=list(dict.fromkeys(k for r in rows for k in r)) if rows else default
        write_csv(run/name,rows,fields)
    csvout('timings.csv',timings,['case_id','stage','pe','batch','mode','trial','phase','iteration','elapsed_ms'])
    csvout('trial_summary.csv',trials,['case_id','stage','pe','batch','mode','trial','status','mean_ms'])
    csvout('initial_costs.csv',costs,['case_id','S_ms','F_ms','W_ms','A_ms'])
    csvout('warmup_curves.csv',warmrows,['case_id','pe','batch','mode','iteration','elapsed_ms'])
    csvout('stability_bins.csv',stability,['pe','batch','mode','time_start_s','mean_ms','images_s'])
    failure_rows=[dict(case_id=w['case_id'],stage=w['stage'],pe=w['pe'],batch=w['batch'],mode=w['mode'],
        trial=w['trial'],status=w['status'],failure_category=failure_category(w,r),
        first_call_completed=any(t['phase']=='first' for t in r.get('timings',[])),
        repeat_calls=sum(t['phase']=='repeat' for t in r.get('timings',[])),
        error_excerpt='fatal error: assert.h: No such file or directory' if failure_category(w,r)=='host_c_header_missing' else '',
        source_result=f'cases/{w["case_id"]}/result.json',wall_s=w.get('wall_s'))
        for w,r in workers if w['status'] not in ('ok','running')]
    csvout('failure_summary.csv',failure_rows,['case_id','stage','status','failure_category','error_excerpt'])
    for row in diag:
        p=run/'cases'/row['case_id']/'stderr.log'
        lines=p.read_text(errors='replace').splitlines() if p.exists() else []
        row['torch_graph_break_log_lines']=[line for line in lines if 'Graph break' in line or 'Graph Break' in line]
        row['torch_recompile_log_lines']=[line for line in lines if 'Recompiling function' in line or '__recompiles' in line]
        row['diagnostic_log_settings']={'TORCH_LOGS':'recompiles,graph_breaks','FURIOSA_TORCH_LOG':'debug'}
        row['absence_of_lines_is_not_proof_of_no_codegen_or_recompile']=True
    write_json(run/'diagnostic_summary.json',diag)
    pe_observations=[]
    device_rows=[json.loads(line) for line in (run/'device_states.jsonl').read_text().splitlines()] if (run/'device_states.jsonl').exists() else []
    for w,r in workers:
        if w['stage']!='diagnostic':continue
        cores=set()
        for sample in device_rows:
            if sample.get('worker')!=w['case_id']:continue
            for line in sample['commands'][-1].get('stdout','').splitlines():
                fields=[x.strip() for x in line.split('|')]
                if len(fields)>3 and fields[1]==str(w.get('pid')):cores.add(fields[2])
        observed=r.get('path_observations',{})
        pe_observations.append(dict(case_id=w['case_id'],mode=w['mode'],status=w['status'],requested_pe=w['pe'],sdk_fusion=r.get('fusion'),
            sdk_configuration=r.get('device_configuration'),smi_observed_core_names=sorted(cores),npu_task_event_count=observed.get('npu_task_events'),
            npu_task_names=observed.get('task_names'),cluster_indices=sorted({x.get('cluster_index') for x in observed.get('task_arg_configurations',[]) if x.get('cluster_index') is not None}),
            observed_stream_ids=sorted({str(x['stream_id']) for x in observed.get('task_arg_configurations',[]) if x.get('stream_id') is not None}),
            observed_device_indices=sorted({str(x['device_index']) for x in observed.get('task_arg_configurations',[]) if x.get('device_index') is not None}),
            per_pe_utilization_verified=False,unknown='Task event/core allocation does not establish per-PE work balance, occupancy, or full operator placement'))
    csvout('pe_observations.csv',pe_observations,['requested_pe','sdk_fusion','smi_observed_core_names','npu_task_names','per_pe_utilization_verified'])
    out=validation(run,workers,assets)
    # Never pool exploration with formal or count repeat calls as independent trials.
    groups=defaultdict(list)
    latest_ids={w['case_id'] for w,r in latest.values()}
    for r in trials:
        if r['eligible_for_descriptive_performance'] and r['case_id'] in latest_ids:groups[r['stage'],r['pe'],r['batch'],r['mode']].append(r)
    aggregate=[]
    for (stage,p,b,mode),rs in sorted(groups.items()):
        means=[r['mean_ms'] for r in rs]
        aggregate.append(dict(stage=stage,pe=p,batch=b,mode=mode,trials_completed=len(rs),
            **pe_evidence.get((p,mode),dict(npu_task_evidence='no_accepted_diagnostic',per_pe_utilization_verified=False)),
            median_trial_mean_ms=st.median(means),min_trial_mean_ms=min(means),max_trial_mean_ms=max(means),
            trial_mean_sample_sd_ms=st.stdev(means) if len(means)>1 else None,
            trial_mean_cv=st.stdev(means)/st.mean(means) if len(means)>1 else None,
            median_trial_images_s=st.median(r['images_s'] for r in rs),
            median_images_s_per_allocated_pe=st.median(r['images_s_per_allocated_pe'] for r in rs),
            warmup_counts=sorted(set(r['warmup_count'] for r in rs)),
            repeat_counts=[r['repeat_count'] for r in rs],repeat_measured_s=[r['repeat_sum_ms']/1000 for r in rs],
            stabilization_status=','.join(sorted(set(str(r['stabilization_status']) for r in rs))),
            range_is_not_confidence_interval=True,provisional=True))
    csvout('condition_summary.csv',aggregate,['stage','pe','batch','mode','trials_completed','median_trial_mean_ms'])
    index={(r['stage'],r['pe'],r['batch'],r['mode'],r['trial']):r for r in trials if r['eligible_for_descriptive_performance'] and r['case_id'] in latest_ids}
    scaling=[];paired=[];breakeven=[]
    for (stage,p,b,mode,t),r in index.items():
        e=index.get((stage,p,b,'eager',t))
        if mode=='compile' and e and (stage!='formal' or e['warmup_count']==r['warmup_count']):
            ratio=e['mean_ms']/r['mean_ms']
            paired.append(dict(stage=stage,pe=p,batch=b,trial=t,eager_case=e['case_id'],compile_case=r['case_id'],
                eager_mean_ms=e['mean_ms'],compile_mean_ms=r['mean_ms'],eager_over_compile=ratio,
                same_input_hash=e['input_hash']==r['input_hash'],same_common_warmup=e['warmup_count']==r['warmup_count'],provisional=True))
            if stage=='formal' and e['warmup_count']==r['warmup_count']:
                deltaA=r['A_ms']-e['A_ms'];delta=e['mean_ms']-r['mean_ms']
                if delta>0:
                    N=max(1,math.floor(deltaA/delta)+1);state='finite' if deltaA>=0 else 'already_advantaged'
                elif delta==0:N=1 if deltaA<0 else None;state='equal_rate'
                else:N=None;state='initial_advantage_only' if deltaA<0 else 'no_break_even'
                breakeven.append(dict(pe=p,batch=b,trial=t,eager_case=e['case_id'],compile_case=r['case_id'],
                    eager_A_ms=e['A_ms'],compile_A_ms=r['A_ms'],eager_mean_ms=e['mean_ms'],compile_mean_ms=r['mean_ms'],
                    common_warmup=e['warmup_count'],status=state,N_prepared_useful_batch_calls=N,
                    useful_images=b*N if N is not None else None,model='conditional linear preparation-after scenario; warmup forwards already executed',
                    numerical_crossing_precision='float descriptive estimate; no cumulative validation performed'))
        ref=index.get((stage,1,b,mode,t))
        if ref and r['sdk_allocation_verified'] and ref['sdk_allocation_verified']:
            speedup=ref['mean_ms']/r['mean_ms']
            scaling.append(dict(stage=stage,pe=p,batch=b,mode=mode,trial=t,baseline_case=ref['case_id'],case_id=r['case_id'],
                npu_task_evidence=r['npu_task_evidence'],pe_evidence_case=r.get('pe_evidence_case'),
                multi_cluster_task_evidence=r.get('multi_cluster_task_evidence'),per_pe_utilization_verified=False,
                pe_scaling_speedup=speedup,relative_scaling_efficiency=speedup/p,
                images_s=r['images_s'],images_s_per_allocated_pe=r['images_s_per_allocated_pe'],
                efficiency_is_not_hardware_utilization=True,provisional=True))
    csvout('paired_speedups.csv',paired,['stage','pe','batch','trial','eager_over_compile'])
    csvout('scaling_summary.csv',scaling,['stage','pe','batch','mode','trial','pe_scaling_speedup','relative_scaling_efficiency'])
    csvout('break_even.csv',breakeven,['pe','batch','trial','status','N_prepared_useful_batch_calls','useful_images'])
    coverage=[]
    for p in cfg['pe_candidates']:
        for b in cfg['batches']+cfg['extension_batches']:
            for mode in ('eager','compile'):
                ws=[w for w,r in workers if (w['pe'],w['batch'],w['mode'])==(p,b,mode)]
                formal=groups.get(('formal',p,b,mode),[]);explore=groups.get(('exploration',p,b,mode),[])
                holds=[e for e in m['events'] if e.get('pe') in (None,p) and e.get('batch',e.get('next_batch'))==b and e.get('mode') in (None,mode) and e.get('status','').startswith(('held_','exploration_held','extension_held','larger_batches_held'))]
                target=cfg['representative_trials'] if [p,b] in m.get('selection',{}).get('representatives',[]) else cfg['formal_trials']
                failed=next((w for w in reversed(ws) if w['status'] not in ('ok','running')),None)
                blocked=next((w for w,r in reversed(workers) if w['pe']==p and w['batch']<b and w['status'] in ('oom','timeout')),None)
                reason=''
                if len(formal)<target:
                    if failed:
                        result=next(r for w,r in workers if w['case_id']==failed['case_id'])
                        reason=failure_category(failed,result)
                    elif blocked:reason='blocked_by_lower_batch_'+blocked['status']
                    elif holds:reason=holds[-1]['status']
                    elif ws:reason=ws[-1]['status'] if ws[-1]['status']!='ok' else 'formal_trial_target_incomplete' if formal else 'formal_not_measured'
                    else:reason='not_started_or_extension_prerequisite_unmet'
                coverage.append(dict(pe=p,batch=b,mode=mode,formal_trials=len(formal),exploration_trials=len(explore),
                    formal_target_trials=target,formal_trial_shortfall=max(0,target-len(formal)),
                    status='formal_measured_provisional' if formal else 'exploration_only' if explore else 'unmeasured',reason=reason,
                    budget_hold_reason=holds[-1]['status'] if holds else None,
                    blocking_case=blocked['case_id'] if blocked and len(formal)<target else None,
                    last_attempt=ws[-1]['case_id'] if ws else None,last_status=ws[-1]['status'] if ws else None))
    csvout('coverage.csv',coverage,['pe','batch','mode','formal_trials','status','reason'])
    m['requested_matrix']=coverage
    m['analysis_utc']=now();m['analysis_summary']=dict(formal_workers=sum(r['stage']=='formal' and r['eligible_for_descriptive_performance'] for r in trials),
        exploration_workers=sum(r['stage']=='exploration' and r['eligible_for_descriptive_performance'] for r in trials),
        worker_status_counts=dict(Counter(w['status'] for w,r in workers)),numeric_status='pending')
    write_json(run/'analysis_manifest.json',m)
    if m['status']!='running': write_json(run/'manifest.json',m)
    report(run,cfg,m,workers,aggregate,paired,scaling,breakeven,stability,out)
    write_json(run/'budget_accounting.json',dict(utc=now(),
        observation_end_utc=stop.isoformat(),scope='Experiment observation end; report regeneration time does not consume measurement budget.',
        elapsed_wall_s=(stop-datetime.fromisoformat(cfg['created_utc'])).total_seconds(),
        budget_s=cfg['total_elapsed_budget_s'],remaining_s=max(0,cfg['total_elapsed_budget_s']-(stop-datetime.fromisoformat(cfg['created_utc'])).total_seconds()),
        completed_and_partial_worker_wall_s=sum(w.get('wall_s',0) for w,r in workers),
        incomplete_first_attempts=[dict(case_id=c['case_id'],status=c['status'],wall_s=c['incomplete_first_attempt_wall_s'],scope=c['incomplete_first_wall_scope']) for c in costs if c['incomplete_first_attempt_wall_s'] is not None],
        active_worker_wall_s=sum((datetime.now(timezone.utc)-datetime.fromisoformat(w['started_utc'])).total_seconds() for w,r in workers if w['status']=='running'),
        model_input_prepare_ms=sum(r.get('model_input_prepare_ms',0) for w,r in workers),
        runner_wrapper_setup_ms=sum(r.get('runner_setup_ms',0) for w,r in workers),
        first_call_total_ms=sum(t['elapsed_ms'] for t in timings if t['phase']=='first'),
        warmup_total_ms=sum(t['elapsed_ms'] for t in timings if t['phase']=='warmup'),
        exploration_and_formal_repeat_total_ms=sum(t['elapsed_ms'] for t in timings if t['phase']=='repeat'),
        stability_total_ms=sum(t['elapsed_ms'] for t in timings if t['phase']=='stability'),
        pure_compiler_time_ms=None,first_call_is_not_pure_compiler_time=True))
    # Traceable independent artifact audit, no additional NPU calls.
    audit=dict(utc=now(),worker_count=len(workers),raw_timing_rows=len(timings),
        performance_loops_no_cpu_copy=all(r.get('performance_loop_output_copies',0)==0 for w,r in workers),
        positive_finite_timings=all(math.isfinite(t['elapsed_ms']) and t['elapsed_ms']>0 for t in timings),
        completed_formal_minima=all(r['repeat_count']>=cfg['repeat_min_calls'] and r['repeat_sum_ms']>=cfg['repeat_min_seconds']*1000 for r in trials if r['stage']=='formal' and r['status']=='ok'),
        private_cache_paths_unique=len({w['cache_path'] for w,r in workers})==len(workers),
        code_hashes={str(p.relative_to(run)):sha256(p) for p in (run/'code').glob('*.py')},
        quantile_scope='within one trial; no independent-trial inference from repeated calls',
        no_ci_claim=True)
    audit['completed_formal_worker_count']=sum(r['stage']=='formal' and r['status']=='ok' for r in trials)
    audit['current_protocol_eligible_formal_workers']=sum(r['stage']=='formal' and r['eligible_for_descriptive_performance'] for r in trials)
    audit['legacy_logging_performance_workers_preserved']=sum(r['worker_protocol']<cfg.get('worker_protocol',1) for r in trials)
    write_json(run/'artifact_audit.json',audit)
    if plots:subprocess.run([str(ROOT/'vit_pilot/.plot-venv/bin/python'),str(ROOT/'vit_scaling/plots.py'),'--run',str(run)],check=True)
    return audit


def report(run,cfg,m,workers,agg,paired,scaling,be,stability,out):
    fmt=lambda x:'미측정' if x is None else f'{x:.4g}'
    formal=[r for r in agg if r['stage']=='formal'];ex=[r for r in agg if r['stage']=='exploration']
    best=max(formal,key=lambda r:r['median_trial_images_s']) if formal else None
    eff=max(formal,key=lambda r:r['median_images_s_per_allocated_pe']) if formal else None
    exploration_best=max(ex,key=lambda r:r['median_trial_images_s']) if ex else None
    exploration_eff=max(ex,key=lambda r:r['median_images_s_per_allocated_pe']) if ex else None
    stop=observation_end(m)
    elapsed=(stop-datetime.fromisoformat(cfg['created_utc'])).total_seconds()
    lines=[f'# ViT RNGD PE·batch 성능 확장 보고서',f'\nrun `{run.name}`, 상태 `{m["status"]}`. 분석 {datetime.now(timezone(timedelta(hours=9))).isoformat()} (KST). elapsed {elapsed/3600:.3f}/{cfg["total_elapsed_budget_s"]/3600:g}시간. 수치 허용 기준 pending으로 모든 성능은 **provisional**이다.',
        '\n## 증거 구분',
        '\n① 실행: 각 worker의 새 프로세스/PID/모델/runner, BF16 state·입력·weights 해시, strict control, 요청 Fusion/SDK 구성, timed_call 원본을 저장했다. 완료한 범위만 아래에 제공한다.',
        '\n② 설치 소스: set_fusion(P)는 장치 생성 전에 한 번 지정한다. 1/2/4PE는 하드웨어 fusion, 8PE는 4PE×2 cluster 소프트웨어 fusion. 동일 fused logical device의 current stream은 그 stream에 제출된 작업 완료를 기다린다는 SDK 계약을 적용했다. evidence/ 및 pe_configuration.md 참조.',
        '\n③ 조건부 추정: 손익분기/확장 효율은 이 실험의 성능 비율이다. 상대 효율은 실제 하드웨어 사용률이 아니다. 시간 변화만으로 codegen/cache/thermal throttling을 확정하지 않는다.',
        '\n④ 미확인: 실제 compiler 시간/codegen/disk hit·miss, 모든 연산 placement, binary runtime 내부의 PE별 상세 동기화, 수치 허용 기준. 프로세스 새 시작은 시스템 전체 cold가 아니다.',
        '\n## 완료 범위와 연구 질문',
        f'\n정식 mode worker {sum(r["trials_completed"] for r in formal)}개, 탐색 mode worker {sum(r["trials_completed"] for r in ex)}개. 반복 호출 수를 독립 trial 수로 세지 않았다. coverage.csv에 미지원/OOM/timeout/미측정/보류를 0 성능 대신 빈칸·사유로 보존한다.',
        '\n|단계|PE|Batch|Mode|독립 trial|warmup|반복 수/trial|반복 측정 s/trial|trial 평균 중앙값 ms|trial 평균 최소–최대 ms|중앙 images/s|안정화|',
        '|---|---:|---:|---|---:|---|---|---|---:|---|---:|---|']
    for r in agg:
        lines.append(f'|{r["stage"]}|{r["pe"]}|{r["batch"]}|{r["mode"]}|{r["trials_completed"]}|{r["warmup_counts"]}|{r["repeat_counts"]}|{[round(s,3) for s in r["repeat_measured_s"]]}|{fmt(r["median_trial_mean_ms"])}|{fmt(r["min_trial_mean_ms"])}–{fmt(r["max_trial_mean_ms"])}|{fmt(r["median_trial_images_s"])}|{r["stabilization_status"]}|')
    active=m.get('selection',{}).get('active_pe') or cfg['pe_candidates']
    common=m.get('selection',{}).get('common_large_batch')
    counts={(r['pe'],r['batch'],r['mode']):r['trials_completed'] for r in formal}
    core_missing=[(p,b,mode,max(0,cfg['formal_trials']-counts.get((p,b,mode),0))) for p in active for b in dict.fromkeys([1,4,16]+([common] if common else [])) for mode in ('eager','compile') if counts.get((p,b,mode),0)<cfg['formal_trials']]
    representatives=m.get('selection',{}).get('representatives',[])
    rep_missing=[(p,b,mode,cfg['representative_trials']-counts.get((p,b,mode),0))
        for p,b in representatives for mode in ('eager','compile')
        if counts.get((p,b,mode),0)<cfg['representative_trials']]
    stable_attempts=[w for w,r in workers if w['stage']=='stability']
    stable_ok=sum(w['status']=='ok' for w in stable_attempts)
    header_failures=[w for w,r in workers if w['status']=='error' and failure_category(w,r)=='host_c_header_missing']
    remaining=max(0,cfg['total_elapsed_budget_s']-elapsed)
    lines[2:2]=[
        '\n## 검토된 완료 상태',
        f'\n**{"부분 완료" if core_missing or rep_missing or stable_ok<2 else "선정 범위 완료"}**: 현재 규약의 정식 mode trial {sum(r["trials_completed"] for r in formal)}개, Eager/Compile paired trial {sum(r["stage"]=="formal" for r in paired)}쌍이다. '
        f'측정 제어 종료 시각은 {stop.astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")}, 당시 예산은 약 {remaining/60:.1f}분 남아 있었다. '
        '<code>finished_with_coverage_recorded</code>는 제어 루프가 종료됐다는 상태이며 전체 실험 목표를 달성했다는 뜻은 아니다.',
        f'\n대표 5 trial 미완료 (PE, batch, mode, 부족 수): {rep_missing or "없음"}. '
        f'별도 stability 성공/시도: {stable_ok}/{len(stable_attempts)}. '
        'B1/B4 결과에서 큰 배치 처리량 포화점이나 B16 이상 다중 PE 이점을 추론하지 않는다. 원본 12시간 기한과 보고서 재집계 시각은 구분했다.']
    if header_failures:
        lines[5:5]=[
            f'\n**재개 후 환경 오류:** {len(header_failures)}개 worker가 SDK 내부 C 컴파일 단계에서 '
            '<code>fatal error: assert.h: No such file or directory</code>로 실패했다. '
            '대표 조건 trial 4/5와 stability는 해당 첫 호출을 완료하지 못했으며, 이 실패에서 나온 반복 성능 자료는 없다. '
            '실행 프로세스와 NPU 할당이 관측됐다는 사실은 성공한 forward를 뜻하지 않는다. '
            '제어기가 이 공통 환경 오류를 전체 중단 사유로 분류하지 못해 후속 시도도 실패했다. '
            'SDK/ViT/PE 미지원으로 판정하지 않는다. 정확히 어떤 환경 패키지가 달라졌는지는 전후 식별 자료가 부족해 미확인이다. '
            '[failure_summary.csv](failure_summary.csv)와 각 case의 원본 error를 근거로 한다.']
    lines.append(f'\n최소 핵심 범위의 미완료 (PE, batch, mode, 부족 trial 수): {core_missing or "없음"}. 공통 최대 탐색 배치는 {common if common is not None else "선정 전"}이다. 이 값은 관측 범위의 최댓값이며 B1/B4라면 큰 배치의 검증을 대신하지 않는다. 미완료 핵심 범위를 완료한 확장 연구로 표현하지 않는다.')
    lines.append('\ntrial별 평균·중앙값·표본 SD·CV·p95/p99, 실제 호출 수·측정 합계·PID·cache 경로·안정화 상태는 [trial_summary.csv](trial_summary.csv)에 있다. [timings.csv](timings.csv)는 느린 호출을 제거하지 않은 원본 호출 시간이다.')
    lines += ['\n### RQ1·RQ3: PE 및 Eager/Compile 비율',
        '\n아래 각 범위는 paired trial 비율의 최소–최대이며 신뢰구간이 아니다. exploration과 formal을 합치지 않았다.',
        '\n|단계|PE|Batch|paired n|Eager/Compile 중앙값|최소–최대|','|---|---:|---:|---:|---:|---|']
    groups=defaultdict(list)
    for r in paired:groups[r['stage'],r['pe'],r['batch']].append(r['eager_over_compile'])
    for (stage,p,b),vs in sorted(groups.items()):lines.append(f'|{stage}|{p}|{b}|{len(vs)}|{fmt(st.median(vs))}|{fmt(min(vs))}–{fmt(max(vs))}|')
    scale_groups=defaultdict(list)
    for row in scaling:scale_groups[row['stage'],row['pe'],row['batch'],row['mode']].append(row)
    lines += ['\n|단계|PE|Batch|Mode|짝 n|PE speedup 중앙값|상대 효율 중앙값|', '|---|---:|---:|---|---:|---:|---:|']
    for (stage,p,b,mode), rows in sorted(scale_groups.items()):
        lines.append(f'|{stage}|{p}|{b}|{mode}|{len(rows)}|{fmt(st.median(x["pe_scaling_speedup"] for x in rows))}|{fmt(st.median(x["relative_scaling_efficiency"] for x in rows))}|')
    lines += ['\nPE scaling은 같은 mode/batch/trial의 유효 1PE를 기준으로만 계산했다. 비교 범위·relative efficiency·PE당 처리량은 scaling_summary.csv에 있다. SDK allocation 확인은 PE 활용률 증명이 아니며 진단 task 관측과 구분한다. 요청 PE/SDK 구성/SMI core/trace 이름/미확인은 pe_observations.csv에 따로 보존했다. PE 간 trial index를 맞췄지만 측정 시각이 인접한 것은 아니므로 시간 경과의 영향이 남을 수 있다.',
        '\n### RQ2: Batch 증가',
        '\nbatch-call latency는 한 번의 배치 호출 전체 시간이다. images/s=B×호출 수/호출 시간 합계, amortized ms/image=평균 batch 시간/B. 후자를 개별 요청 응답시간으로 부르지 않는다. 고정 네 이미지 반복이며 다양한 이미지 정확도 평가가 아니다. 각 배치의 입력 hash는 조건 간 확인한다. 처리량 증감은 condition_summary.csv와 figures/01·02에서 그대로 표시한다.',
        '\n|PE|Mode|B1→B4 batch 지연시간 ms|B1→B4 images/s|처리량 증가 비율|',
        '|---:|---|---|---|---:|']
    lookup={(r['pe'],r['batch'],r['mode']):r for r in formal}
    for p in active:
        for mode in ('eager','compile'):
            low=lookup.get((p,1,mode));high=lookup.get((p,4,mode))
            if low and high:
                lines.append(f'|{p}|{mode}|{fmt(low["median_trial_mean_ms"])}→{fmt(high["median_trial_mean_ms"])}|'
                    f'{fmt(low["median_trial_images_s"])}→{fmt(high["median_trial_images_s"])}|'
                    f'{fmt(high["median_trial_images_s"]/low["median_trial_images_s"])}|')
    lines += ['\n여기서 B4는 B1 대비 배치 확대의 관측점이며 큰 배치 포화점을 뜻하지 않는다.',
        '\n### RQ4: 워밍업·독립 trial·장시간 변화']
    probes=[(w,r) for w,r in workers if w['stage']=='stabilization' and w['status']=='ok' and w.get('worker_protocol',1)>=cfg.get('worker_protocol',1)]
    longer=[w['case_id']+':'+str(r.get('warmup_count')) for w,r in probes if r.get('warmup_count',0)>100]
    unconfirmed=[w['case_id'] for w,r in probes if r.get('stabilization_status')=='stabilization_unconfirmed']
    lines.append(f'\n100회 초과를 필요로 한 안정화 탐색: {longer or "관측된 완료 probe에서 없음(미완료 조건에 일반화 불가)"}. 최대500회에도 불명확: {unconfirmed or "완료 probe에서 없음"}. 최근20회×3 구간 평균 변화≤2%, 각 CV≤0.05는 보조 휴리스틱이며 steady state 증명이 아니다. Eager/Compile은 두 probe가 요구한 횟수의 최댓값을 같은 PE/batch의 모든 정식 trial에 공통 적용했다.')
    lines.append('\n안정화 상태는 별도 probe의 최근 구간 판정이며 정식 trial 전체의 steady state를 증명하지 않는다. 정식 trial에서는 공통 고정 warmup 수를 지키고, 반복 구간의 CV·느린 호출을 그대로 보고한다. 기존 warmup5 충분성 검토: warmup_curves.csv의 1–5/6–20/마지막60회 구간을 warmup_assessment.csv에 함께 제공한다. 첫 다섯 호출 이후 변화가 작더라도 과거 측정 시점·장치·자원 상태가 달라 워밍업 효과 하나로 이전 차이를 설명하지 않는다. 느린 호출은 제거하지 않았다.')
    wa=[]
    for w,r in probes:
        v=[x['elapsed_ms'] for x in r.get('timings',[]) if x['phase']=='warmup']
        if v:wa.append(dict(case_id=w['case_id'],pe=w['pe'],batch=w['batch'],mode=w['mode'],warmup_count=len(v),first5_mean_ms=st.mean(v[:5]),next15_mean_ms=st.mean(v[5:20]),last60_mean_ms=st.mean(v[-60:]),first5_to_last60_ratio=st.mean(v[:5])/st.mean(v[-60:]),stabilization_status=r['stabilization_status']))
    write_csv(run/'warmup_assessment.csv',wa,list(wa[0]) if wa else ['case_id','warmup_count','first5_mean_ms','last60_mean_ms'])
    for r in formal:
        cv=f'{r["trial_mean_cv"]*100:.3f}%' if r["trial_mean_cv"] is not None else '산출 불가(n<2)'
        lines.append(f'\nPE{r["pe"]} B{r["batch"]} {r["mode"]}: trial 평균 CV {cv}, 최소–최대 {fmt(r["min_trial_mean_ms"])}–{fmt(r["max_trial_mean_ms"])}ms (n={r["trials_completed"]}).')
    sg=defaultdict(list)
    for r in stability:sg[r['pe'],r['batch'],r['mode']].append(r)
    if not sg:lines.append(f'\n장시간 stability: 완료 자료 없음. 시도한 worker {len(stable_attempts)}개, 성공 {stable_ok}개다. 첫 호출 실패로 300초 연속 측정은 시작하지 못했다. 정식 trial 통계와 합치지 않으며 지속 성능을 검증했다고 표현하지 않는다.')
    for (p,b,mode),rs in sg.items():
        means=[r['mean_ms'] for r in rs]
        lines.append(f'\n장시간 PE{p} B{b} {mode}: {len(rs)}개 10초 bin, bin 평균 범위 {fmt(min(means))}–{fmt(max(means))}ms, 마지막/첫 bin {fmt(means[-1]/means[0])}. 마지막 bin은 부분 구간일 수 있다. 온도/전력은 30초 furiosa-smi snapshot과 함께 읽되 throttling으로 단정하지 않는다.')
    lines += ['\n### RQ5: 초기 비용',
        '\ninitial_costs.csv와 break_even.csv는 반복 성능과 별도다. A=S+F+sum(W)는 이번 공통 warmup 정책을 포함한 준비 후 시나리오이며 모델·입력 준비 시간은 별도다. warmup도 실제 forward다. N은 준비 후 유효 batch 호출 수, 유효 이미지는 B×N이다. 같은 trial의 초기 비용과 반복 평균을 결합한 선형 추정이며 이번에는 수천 회 누적 검증을 하지 않았다. compiler_time_ms는 null. 기존 5회 warmup의 경계와 동일시하지 않는다.']
    with (run/'trial_summary.csv').open() as f:current_trials=[r for r in csv.DictReader(f) if r['stage']=='formal' and r['eligible_for_descriptive_performance']=='True']
    cost_groups=defaultdict(list)
    for r in current_trials:cost_groups[int(r['pe']),int(r['batch']),r['mode']].append(r)
    cost_summary=[]
    lines+=['\n|PE|Batch|Mode|trial n|준비 중앙값 s|S 중앙값 s|F 중앙값 s|W 중앙값 s|A 중앙값 s|A 최소–최대 s|','|---:|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for (p,b,mode),rs in sorted(cost_groups.items()):
        row=dict(pe=p,batch=b,mode=mode,trials=len(rs))
        for field in ('model_input_prepare_ms','runner_setup_ms','first_call_ms','warmup_sum_ms','A_ms'):
            values=[float(r[field]) for r in rs if r.get(field) not in (None,'')]
            row['median_'+field]=st.median(values) if values else None
        row.update(min_A_ms=min(float(r['A_ms']) for r in rs),max_A_ms=max(float(r['A_ms']) for r in rs));cost_summary.append(row)
        values=[row['median_'+field]/1000 if row['median_'+field] is not None else None for field in ('model_input_prepare_ms','runner_setup_ms','first_call_ms','warmup_sum_ms','A_ms')]
        lines.append(f'|{p}|{b}|{mode}|{len(rs)}|'+ '|'.join(fmt(x) for x in values)+f'|{fmt(row["min_A_ms"]/1000)}–{fmt(row["max_A_ms"]/1000)}|')
    write_csv(run/'initial_cost_summary.csv',cost_summary,list(cost_summary[0]) if cost_summary else ['pe','batch','mode','trials'])
    be_groups=defaultdict(list)
    for r in be:be_groups[r['pe'],r['batch']].append(r)
    lines+=['\n|PE|Batch|paired n|각 trial 추정 N의 중앙값|최소–최대 N|유효 이미지 중앙값|상태|','|---:|---:|---:|---:|---|---:|---|']
    for (p,b),rs in sorted(be_groups.items()):
        ns=[r['N_prepared_useful_batch_calls'] for r in rs if r['N_prepared_useful_batch_calls'] is not None]
        lines.append(f'|{p}|{b}|{len(rs)}|{fmt(st.median(ns)) if ns else "없음"}|{str(min(ns))+"–"+str(max(ns)) if ns else "없음"}|{fmt(b*st.median(ns)) if ns else "없음"}|{dict(Counter(r["status"] for r in rs))}|')
    lines+=['\nN 요약은 같은 trial에서 계산한 추정값들의 기술 통계이며, 서로 다른 trial의 초기 비용·반복 평균을 새로 결합한 추정이 아니다. 범위는 신뢰구간이 아니다.',
        '\n## 실행 경로·출력·장치 관측',
        f'\n별도 출력 구조/finite 체크 {len(out["checks"])}건, 비교 관계 {len(out["comparisons"])}건. max/mean abs error, cosine, Top1은 output_validation.json. 반복 이미지는 독립 검증 표본으로 세지 않는다. Batch1 vs 큰 batch는 첫 이미지 한 장에 대한 비교다. 허용 오차 pending이므로 출력 검증 완료나 배포 추천이 아니다. 성능 반복의 모든 출력에 finite 검사를 했다는 뜻은 아니며 별도 검증에서 관측한 출력만 확인했다. BF16은 모델·입출력 dtype이며 SDK 기본 implicit casting을 유지해 모든 내부 연산의 dtype을 보장하지 않는다. 진단의 backend 진입은 codegen 횟수가 아니며 Eager backend 진입0도 실제 컴파일0을 뜻하지 않는다.',
        '\nfuriosa-smi info/status/ps를 controller에서30초 간격 및 worker 경계에 수집했다. 온도/전력/메모리/core process는 지원되며 이 환경에서 utilization은 `-`로 미지원/미확인이다. 장치 reset·governor·전력 설정 변경·타인 프로세스 종료는 하지 않았다. 공용 장치 예약은 원자적이지 않으며 관측 시 타인 PID가 있으면 추가 실행을 보류한다.',
        '\nPE 관측을 mode별로 구분한 아래 표는 별도 진단 배치의 근거다. 다른 배치에서 동일 task 구성이 재관측됐다는 뜻은 아니다. task 미관측은 NPU 미실행 증명이 아니며, SDK 할당과 stream launch가 확인돼도 실제 PE 활용 전체를 확인한 것으로 표시하지 않는다.',
        '\n|요청 PE|Mode|진단 Batch|NPU task 수|관측 cluster|관측 상태|',
        '|---:|---|---:|---:|---|---|']
    for (p,mode),ev in sorted(diagnostic_evidence(workers).items()):
        lines.append(f'|{p}|{mode}|{ev["pe_evidence_batch"]}|{ev["diagnostic_npu_task_events"]}|{ev["diagnostic_cluster_indices"]}|{ev["npu_task_evidence"]}; {ev["multi_cluster_task_evidence"]}|')
    lines += ['\nPE별 utilization 및 작업 균형은 모두 미확인이다. 설치된 Profiler 문서는 NPU task 기록을 설명하지만 이번 task 누락 원인이나 다중 cluster 제한을 명시하지 않는다. NPU task 수를 커널 수 또는 fusion 횟수로 해석하지 않는다. 최초 2PE Compile 진단 a1은 장치 감시가 worker의 gcc/cc1 자식 프로세스를 외부 점유로 오인해 contaminated로 보존했다. process group/session/부모 계보 확인을 추가한 뒤 새 a2로 재실행했다. a1의 task 관측은 원본 보존 자료이며 승인된 a2의 task 누락을 대체하는 근거로 쓰지 않는다. compiler 자식 프로세스 관측으로 SDK 전체 codegen 횟수·순수 compiler 시간·disk miss를 확정하지 않는다.',
        '\n## 이전 기준점과 차이',
        '\nSDK/torch 버전은 environment.json을 기준으로 비교한다. 이전 npu1 PCI04:00.0과 이번 npu0 PCI03:00.0은 장치 식별이 다르다. CPU threads22/inter-op64 고정, 양쪽 strict path, 기존 입력·가중치 해시 재사용. warmup5/repeat50에서 공통>=100/repeat>=500 및>=10초로 측정 규약이 달라졌다. 이전과 같게 만들기 위한 설정 조정은 하지 않았다. baseline_comparison.csv는 현재 유효1PE formal trial 중앙값과 이전 trial평균 중앙값의 기술적 비교다.']
    with (Path(cfg['prior_run'])/'trial_summary.csv').open() as source: old=list(csv.DictReader(source))
    oldg=defaultdict(list)
    for r in old:oldg[int(r['batch']),r['mode']].append(float(r['mean_ms']))
    baseline=[]
    for r in formal:
        key=r['batch'],r['mode']
        if r['pe']==1 and key in oldg:baseline.append(dict(batch=r['batch'],mode=r['mode'],old_trials=len(oldg[key]),old_median_trial_mean_ms=st.median(oldg[key]),new_trials=r['trials_completed'],new_median_trial_mean_ms=r['median_trial_mean_ms'],new_over_old=r['median_trial_mean_ms']/st.median(oldg[key]),causal_attribution='not established; changed physical device/date/warmup and repeat policy'))
    write_csv(run/'baseline_comparison.csv',baseline,list(baseline[0]) if baseline else ['batch','mode','old_median_trial_mean_ms','new_median_trial_mean_ms'])
    failures=[w for w,r in workers if w['status'] not in ('ok','running')]
    observed_effects=[]
    largest=max([k[2] for k in scale_groups if k[0]=='formal'],default=1)
    for (stage,p,b,mode), rows in sorted(scale_groups.items()):
        if stage=='formal' and p>1 and b in (1,largest):
            observed_effects.append(f'P{p} B{b} {mode}: {st.median(x["pe_scaling_speedup"] for x in rows):.3f}배 (1PE 대비, n={len(rows)})')
    held_events=sum(e.get('status','').startswith(('held_','exploration_held','extension_held','larger_batches_held')) for e in m['events'])
    lines += ['\n## 실패·보류와 한계',f'\n실패/timeout/중단 상태 수: {dict(Counter(w["status"] for w in failures))}. 보류 이벤트 {held_events}건(전체 이벤트 {len(m["events"])}건). '
        '개별 실패의 원인·단계·측정 완료 여부는 [failure_summary.csv](failure_summary.csv), 전체 명령·시각·PID·예산은 [manifest.json](manifest.json)에 있다. '
        'timeout은 하드웨어 미지원 판정이 아니다. 탐색/정식/안정화/stability는 합산하지 않는다.',
        '\n8PE B4 Compile 검증은 3600초 실행 기한으로 종료됐다. manifest wall_s에는 감지·종료 정리 시간이 추가되며 완료된 first_call F는 측정되지 않았다(null). initial_costs.csv 및 budget_accounting.json의 incomplete_first_attempt_wall_s는 first_start 이벤트부터 worker 종료까지의 wall 관측이며 정확한 모델 호출 시간이나 compiler 시간이 아니다.',
        '\n추가 보류: exploration_p4_b4_compile_t1_a1에서 PID267156가 SMI 조회 뒤 사라져 신원을 확인하지 못했다. 이 PID를 타인 또는 자체 compiler라고 확정하지 않고 contaminated 원본을 유지했다. 재개 전 장치가 비어 있음을 확인했다. 이후 장치 조회는30초 간격을 유지하면서 SMI ps 실행 중에만 /proc identity를5ms 간격으로 보강했다. 여전히 미확인 PID는 보류하며 supervisor는 두 번 연속 빈 장치를 확인한 뒤 새 attempt로 재개한다. 관측 방법 변경의 영향은 별도 통제 실험으로 분해하지 않았다.',
        '\n시간 배분 변경: 핵심3 trial을 먼저 완료하고, 가능하면 대표5/stability 예산을 남긴 상태에서 B16 이상 frontier 탐색을 추가한다. 대표5는 완료 핵심 중 baseline B1 및 Compile 처리량 중앙값이 가장 높은 조합 등 최대2개다. 첫 호출 비용과 보류·규약 수정 재측정·중단 시간 모두 최초 생성 시각부터의12시간 wall 예산에 포함된다. 이 때문에 누락된 조건을 SDK 미지원으로 해석하지 않는다.',
        '\n로그 규약 수정: SDK import가 Python furiosa_torch logger를 DEBUG로 지정하므로 root WARNING만으로는 첫 호출 상세 로그가 차단되지 않았다. worker v1 원본·결과를 모두 보존하고, v2에서 해당 logger를 명시적으로 WARNING(진단만 DEBUG)으로 설정했다. 모델·정밀도·backend·동기화·측정 경계는 바꾸지 않았다. v1 성능은 trial_summary.csv에 남기되 현재 집계·비율·최고 처리량에서 제외한다. 정식 trial과 안정화는 새 attempt로 재측정한다. 이전 탐색은 실행 가능성·출력·대략 비용 및 조건부 진단 조건 선정의 참고로 재사용하며 현재 규약의 성능 비교 근거로 합치지 않는다. 별도 진단 출력·trace는 계속 근거로 사용한다. initial_costs.csv도 버전을 표시하며 초기 비용 그림은 현재 버전만 사용한다. logs/logging_protocol_correction.json 참조.',
        '\n## 마지막 질문에 대한 답변',
        f'\n1. 실제 비교 범위: 정식 {sorted(set((r["pe"],r["batch"]) for r in paired if r["stage"]=="formal")) or "미완료"}; 탐색 {sorted(set((r["pe"],r["batch"]) for r in paired if r["stage"]=="exploration")) or "미완료"}. 탐색과 정식 통계는 별도다.',
        f'\n2. 작은/큰 batch의 다중PE 효과: {"; ".join(observed_effects) or "아직 유효한 formal 다중PE pair 없음"}. 1보다 작으면 성능 저하다. 측정되지 않은 batch/PE로 일반화하지 않는다.',
        f'\n3. 최고 관측 정식 처리량: {str(dict(pe=best["pe"],batch=best["batch"],mode=best["mode"],images_s=best["median_trial_images_s"],trials=best["trials_completed"])) if best else "아직 정식 자료 없음"}. 완료 trial 전체의 중앙값 기준이며 가장 빠른 trial을 고르지 않았다.',
        f'\n탐색 최고 처리량(별도 first1/warmup20/repeat50 규약): {str(dict(pe=exploration_best["pe"],batch=exploration_best["batch"],mode=exploration_best["mode"],images_s=exploration_best["median_trial_images_s"])) if exploration_best else "자료 없음"}. 정식 최고와 합산·혼합 순위화하지 않았다.',
        f'\n4. PE당 최고 정식 처리량: {str(dict(pe=eff["pe"],batch=eff["batch"],mode=eff["mode"],images_s_pe=eff["median_images_s_per_allocated_pe"],trials=eff["trials_completed"])) if eff else "아직 정식 자료 없음"}. 하드웨어 utilization 측정이 아니다.',
        f'\n탐색 PE당 최고 처리량(별도 규약): {str(dict(pe=exploration_eff["pe"],batch=exploration_eff["batch"],mode=exploration_eff["mode"],images_s_pe=exploration_eff["median_images_s_per_allocated_pe"])) if exploration_eff else "자료 없음"}.',
        '\n5. Eager/Compile 차이: paired trial 표와 paired_speedups.csv의 같은PE/batch 비율. 1은 동률,1보다 크면 Compile이 빠르다. 모든PE가 유리하다고 가정하지 않는다.',
        f'\n6. warmup100 초과: {longer or "완료 probe에서는 없음"}; 미확인 안정화 {unconfirmed}.',
        '\n7. 독립 trial/장시간 유지 정도: 위 trial CV·범위와 10초 stability bin을 참조한다. 미완료 stability를 안정적이라고 판정하지 않는다.',
        '\n8. 출력/실행 경로 미확인: 수치 허용 기준·전체 placement·codegen/cache·PE별 활용률·binary runtime 내부 구현. 관측 trace와 SDK 계약의 범위를 구분했다.',
        '\n9. 다음 가치: 우선 공통 C 컴파일 환경 오류를 복구하고 전체 오류 발생 시 중단 판단을 보완해야 한다. 이후 별도 예산으로 미완료 대표5/stability, B16 핵심 paired3, 출력 허용 기준 합의, 다양한 실제 이미지 검증 및 반복일·장치 비교를 진행할 가치가 있다. 이번 검토에서는 환경·SDK·드라이버를 변경하거나 NPU 측정을 추가 실행하지 않았다.',
        '\n## 산출물',
        '\n원본timings.csv / trial_summary.csv / condition_summary.csv / paired_speedups.csv / scaling_summary.csv / initial_costs.csv / break_even.csv / output_validation.json / warmup_curves.csv / stability_bins.csv / device_states.jsonl / diagnostic_summary.json / coverage.csv / figures/*.png 및 대응CSV. p95/p99는 각trial 내 n개 호출과 repeat_sum_ms 범위의 표본 분위수이며 독립 trial 불확실성 추정이 아니다.']
    for recovery_path in sorted((run/'logs').glob('runtime_recovery_*.json')):
        recovery=read_json(recovery_path)
        lines += ['\n## 실행 환경 재개 기록',
            f'\n추가 재개 {recovery.get("utc")}: 이전 worker {recovery.get("previous_worker")}는 {recovery.get("previous_worker_status")}. '
            f'PID1 시작 추정 시각 {recovery.get("pid1_estimated_start_kst")}가 이전 manifest 중단 시각과 근접하고 이전 supervisor/controller/worker PID가 모두 없어 실행 컨테이너 재시작 정황이 있다. 재시작을 유발한 원인은 미확인이다. '
            '새 supervisor/controller/단일 NPU worker 실행 및 SIGHUP 무시를 확인했다. SIGHUP 무시는 실행 환경 자체 종료를 막지 못한다. 완료 결과·중단 attempt·기존 12시간 wall 예산을 보존했다. '
            f'근거: {recovery_path.relative_to(run)}.']
    lines += ['\n## 그림과 대응 CSV',
        '\n누락 조건은 0이 아닌 빈칸과 사유로 표시하며, 누락 지점을 연결해 보간하지 않는다. 08번은 성공한 stability 측정이 없어 빈 패널이다.',
        '\n|번호|그림|CSV|축·해석|','|---:|---|---|---|']
    figure_info=[
        ('01_throughput','PE별 batch 처리량','batch x축 log; formal과 exploration 별도'),
        ('02_batch_latency','PE별 batch-call 지연시간','batch x축 log'),
        ('03_eager_compile_heatmap','Eager/Compile 배속','1=동률; paired n 및 누락 사유'),
        ('04_pe_scaling','PE scaling speedup','1 기준선; 같은 batch/mode/trial의 1PE 기준'),
        ('05_relative_efficiency','상대 scaling 효율','1 기준선; 하드웨어 utilization 아님'),
        ('06_warmup_curves','warmup 원본 지연시간','y축 log; 첫 호출 별도; 5/100회 기준'),
        ('07_trial_means','독립 trial 평균 분포','x/y log; 모든 유효 trial 표시'),
        ('08_stability_trend','장시간 추세','측정 실패로 유효 데이터 없음'),
        ('09_initial_costs','초기 비용','x/y log; A=S+F+W, 순수 compiler 시간 아님')]
    for name,title,caption in figure_info:
        lines.append(f'|{name[:2]}|[{title}](figures/{name}.png)|[CSV](figures/{name}.csv)|{caption}|')
    lines += ['\n![Eager/Compile 배속과 paired trial 수](figures/03_eager_compile_heatmap.png)',
        '\n![워밍업 지연시간; y축 log](figures/06_warmup_curves.png)']
    if (run/'report_review.md').exists():
        lines += ['\n## 보고서 검토 기록',
            '\n[원본 대조 검토](report_review.md)와 [기계 판독 결과](report_review.json)에 통계·출력·파일 불변 검증을 남겼다. 검토 통과는 전체 실험 완료나 수치 허용 기준의 합격을 뜻하지 않는다.']
    (run/'report.md').write_text('\n'.join(lines)+'\n')
    best_text=f'PE{best["pe"]} B{best["batch"]} {best["mode"]}: {best["median_trial_images_s"]:.2f} images/s (n={best["trials_completed"]})' if best else '자료 없음'
    eff_text=f'PE{eff["pe"]} B{eff["batch"]} {eff["mode"]}: {eff["median_images_s_per_allocated_pe"]:.2f} images/s/할당 PE' if eff else '자료 없음'
    (run/'feedback_summary.md').write_text(f'''# 실험 피드백

run {run.name}: 제어 루프는 종료됐으나 목표 일부 미완료. 모든 성능은 수치 허용 기준 pending인 provisional 자료다.

- 현재 규약 정식 mode trial {sum(r['trials_completed'] for r in formal)}개, paired trial {sum(r['stage']=='formal' for r in paired)}쌍. B1은 PE1/2/4/8, B4는 PE1/4에서 각 mode 3 trial이다.
- 최고 관측 처리량: {best_text}. 가장 빠른 trial을 고르지 않은 중앙값이다.
- PE당 최고 관측 처리량: {eff_text}. utilization 측정이 아니다.
- 대표5 미완료: {rep_missing}. Stability 성공/시도 {stable_ok}/{len(stable_attempts)}.
- assert.h 미발견 C 컴파일 환경 오류 {len(header_failures)}건. 실패 조건에서 유효 반복 시간은 얻지 못했다. SDK/ViT/PE 미지원 판정이 아니다.
- 측정 종료 {stop.astimezone(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')}, 예산 약 {remaining/60:.1f}분 잔여. 예산 소진만으로 모든 미완료를 설명하지 않는다.
- 큰 배치 포화점·장시간 안정성은 미확인. 수치 허용 기준·전체 op placement·PE별 활용·codegen/cache도 미확인이다.

세부 통계와 그림은 [report.md](report.md), 원인별 실패는 [failure_summary.csv](failure_summary.csv), 누락·부족 trial은 [coverage.csv](coverage.csv). 다음 우선순위는 공통 컴파일 환경 복구 후 별도 예산으로 대표5·stability와 B16 비교를 보완하는 것이다.
''')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--no-plots',action='store_true');a=ap.parse_args()
    print(json.dumps(analyze(a.run,not a.no_plots)))
