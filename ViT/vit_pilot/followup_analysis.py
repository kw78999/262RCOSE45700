"""Follow-up evidence, saved-logit comparison and raw timing aggregation only."""
import collections
import csv
import json
import math
from pathlib import Path
import re
import statistics as stats
import subprocess
import sys
from common import read_json,write_json,write_csv,sha256,now,tensor_hash
from analyze import compare_outputs,valid,warm_state,table,fmt,TIMING_FIELDS


def result(root,spec):
    p=root/'cases'/spec['case_id']/'result.json'
    return read_json(p) if p.exists() else {}


def trace_summary(path):
    if not path.exists():return dict(status='unavailable',reason='No saved trace')
    data=read_json(path)
    events=data['traceEvents'] if isinstance(data,dict) else data
    names=collections.Counter(x.get('name','') for x in events)
    tasks=[x for x in events if re.search(r'npu\d+pe\d+::Task::',x.get('name',''))]
    spans=[]
    for key in ['followup_representative_call','run_by_cpu::executed_by_cpu','Stream::synchronize',
                'TpModule::run','Runnable::forward','torch_compile::run_sequence','flatten_inputs','memory::copy']:
        selected=[x for x in events if x.get('name')==key and x.get('ph')=='X']
        spans.append(dict(name=key,count=names[key],observed_durations_ms=[x['dur']/1000 for x in selected if 'dur' in x]))
    return dict(status='available',trace_path=str(path),events=len(events),name_counts=dict(names),
        npu_task_count=len(tasks),npu_task_names=dict(collections.Counter(x['name'] for x in tasks)),
        cpu_fallback_execution_spans=names['run_by_cpu::executed_by_cpu'],selected_host_spans=spans,
        npu_task_durations_ms=[x['dur']/1000 for x in tasks if 'dur' in x],
        duration_note='Chrome trace durations in microseconds converted to ms; spans/tasks may overlap or nest. Do not sum as a latency decomposition. Profile call is separate from performance.')


def default_path_evidence(root,mode,b):
    m=read_json(root/'manifest.json')
    spec=[x for x in m['workers'] if x['stage']=='diagnostic' and x['mode']==mode and x['batch']==b][-1]
    cid=spec['case_id']
    r=result(root,spec)
    err=root/'logs'/f'{cid}.stderr.log'
    text=err.read_text(errors='replace') if err.exists() else ''
    tr=trace_summary(root/'cases'/cid/'furiosa_trace.json')
    snapshot=r.get('device_use_snapshot',{}).get('stdout','')
    pid=spec.get('worker_id')
    own_npu_worker=bool(pid and re.search(r'\|\s*'+str(pid)+r'\s*\|\s*npu',snapshot))
    cpu=bool(tr.get('cpu_fallback_execution_spans',0) or ('fallback to CPU' in text and r.get('output_valid')))
    # SDK native PrivateUse1 CPU fallback bypasses the Python run_by_cpu logger.
    # Correlate its explicit strict-mode rejection with default-mode CPU compute trace.
    cpu_summary=root/'cases'/cid/'torch_cpu_summary.json'
    native_cpu={}
    if cpu_summary.exists():
        native_cpu=next((x for x in read_json(cpu_summary) if x['name']=='aten::_native_multi_head_attention'),{})
    strict_errors=[]
    for s in m['workers']:
        if s['stage']=='strict' and s['mode']==mode:
            error=result(root,s).get('error','')
            if 'CPU fallback is disabled for aten::_native_multi_head_attention' in error:strict_errors.append(s['case_id'])
    native_fallback=bool(native_cpu.get('count',0) and strict_errors and r.get('output_valid'))
    cpu=cpu or native_fallback
    return dict(mode=mode,batch=b,diagnostic_status=r.get('status',spec['status']),
        npu_use_confirmed=bool(tr.get('npu_task_count',0) or own_npu_worker),
        npu_task_count=tr.get('npu_task_count'),own_npu_worker_observed=own_npu_worker,
        actual_cpu_fallback_observed=cpu,full_operator_placement='unverified',
        native_attention_cpu_fallback_evidence=dict(observed=native_fallback,default_cpu_profile=native_cpu,strict_rejection_cases=strict_errors),
        path_label='일부 CPU tensor 연산 확인' if cpu else ('NPU 사용 확인 / 전체 연산 배치 미확인' if tr.get('npu_task_count',0) or own_npu_worker else 'NPU 사용 미확인'),
        stderr=str(err.relative_to(root)),trace=str((root/'cases'/cid/'furiosa_trace.json').relative_to(root)),
        actual_codegen_count=None,disk_cache_hit=None,disk_cache_miss=None,
        note='No fallback log is not proof of all-NPU placement; host dispatch is not tensor fallback')


def compare_real(root,stage='validation',trial=1):
    import torch
    from torchvision.models import ViT_B_16_Weights
    cfg=read_json(root/'config.json');m=read_json(root/'manifest.json');assets=read_json(root/'assets.json')
    cats=ViT_B_16_Weights.IMAGENET1K_V1.meta['categories']
    batches=[]
    for b in [1,4]:
        specs=[next(s for s in m['workers'] if s['stage']==stage and s['mode']==mode and s['batch']==b and s['trial']==trial) for mode in ['eager','compile']]
        rs=[result(root,s) for s in specs]
        pair=[]
        for s,r in zip(specs,rs):
            pair.append(dict(case_id=s['case_id'],batch=b,trial=trial,weights_sha256=r.get('weights_sha256'),
                input_hash=r.get('input_hash'),dtype=r.get('dtype'),timing_scope=r.get('timing_scope'),
                cache_state='same_stage_cache_policy; individual initial states recorded separately'))
        raw=compare_outputs(root,*pair,assets,cfg)
        raw.pop('accuracy_vs_ground_truth',None);raw.pop('accuracy_reason',None)
        raw['tolerance_status']='pending' if cfg['atol'] is None else 'explicit_cli_thresholds'
        raw['synthetic_input']=False;raw['ground_truth_available']=False
        raw['accuracy_computed']=False;raw['loss_computed']=False;raw['prediction_score_type']='logit'
        raw['images']=[];raw['identical_logits_for_distinct_inputs']=[]
        if raw['status'] not in ['not_compared','failed']:
            tensors=[torch.load(root/'outputs'/f'{s["case_id"]}.pt',weights_only=True,map_location='cpu').float() for s in specs]
            for i,info in enumerate(read_json(root/'image_manifest.json')['images'][:b]):
                predictions=[]
                for mode,t in zip(['eager','compile'],tensors):
                    cls=int(t[i].argmax());predictions.append(dict(mode=mode,class_index=cls,class_name=cats[cls],logit=float(t[i,cls])))
                raw['images'].append(dict(input_id=info['id'],actual_path=info['actual_absolute_path'],
                    file_sha256=info['file_sha256'],preprocessed_bf16_tensor_sha256=info['preprocessed_bf16_tensor_sha256'],
                    eager=predictions[0],compile=predictions[1],top1_match=predictions[0]['class_index']==predictions[1]['class_index'],
                    cosine=raw['cosine_per_image'][i],max_abs_error=float((tensors[0][i]-tensors[1][i]).abs().max())))
            for mode,t in zip(['eager','compile'],tensors):
                for i in range(b):
                    for j in range(i+1,b):
                        if torch.equal(t[i],t[j]):raw['identical_logits_for_distinct_inputs'].append(dict(mode=mode,indices=[i,j]))
        reasons=[]
        if not all(r.get('output_valid') and r.get('sync_checked') for r in rs):reasons.append('Missing/invalid output or completion evidence')
        if raw['status'] in ['not_compared','failed']:reasons.append(raw.get('reason','Output comparison failed'))
        if raw['top1_matches'] is not None and raw['top1_matches']!=b:reasons.append('Top-1 mismatch; preserve inputs/logits and investigate before expanding')
        if raw['identical_logits_for_distinct_inputs']:reasons.append('Identical logits for distinct inputs; pipeline investigation required')
        per_mode={}
        common_reasons=list(reasons)
        for mode in ['eager','compile']:
            path=default_path_evidence(root,mode,b)
            own_reasons=list(common_reasons)
            if path['actual_cpu_fallback_observed']:own_reasons.append(mode+': actual CPU model tensor fallback observed')
            if not path['npu_use_confirmed']:own_reasons.append(mode+': NPU use not confirmed')
            per_mode[mode]=dict(eligible=not own_reasons,hold_reason='; '.join(own_reasons) or None)
        # Successful earlier performance pairs can veto later expansion if they expose a new mismatch.
        if stage=='validation':
            for t in range(1,cfg['trials']+1):
                previous=[s for s in m['workers'] if s['stage']=='performance' and s['batch']==b and s['trial']==t]
                for done in previous:
                    if done['status'] in ['error','timeout','interrupted','budget_exhausted']:
                        reasons.append(f'Prior {done["mode"]} trial {t} failed/interrupted; later expansion held for investigation')
                    if done['status']!='ok':continue
                    measured=[x for x in result(root,done).get('timings',[]) if x['phase']=='repeat' and valid(x)]
                    if len(measured)!=cfg.get('repeats',50):
                        reasons.append(f'Prior {done["mode"]} trial {t} has invalid/incomplete timing; later expansion held')
                    base=next(s for s in specs if s['mode']==done['mode'])
                    current=torch.load(root/'outputs'/f'{done["case_id"]}.pt',weights_only=True,map_location='cpu')
                    reference=torch.load(root/'outputs'/f'{base["case_id"]}.pt',weights_only=True,map_location='cpu')
                    if current.shape!=reference.shape or not torch.isfinite(current).all() or not torch.equal(current.argmax(1),reference.argmax(1)):
                        reasons.append(f'Prior {done["mode"]} performance trial {t} differs from validation predictions; later expansion held')
                    elif cfg['atol'] is not None and not torch.allclose(current,reference,atol=cfg['atol'],rtol=cfg['rtol']):
                        reasons.append(f'Prior performance trial {t} explicit tolerance violation against validation output')
                if len(previous)==2 and all(s['status']=='ok' for s in previous):
                    u,v=[torch.load(root/'outputs'/f'{s["case_id"]}.pt',weights_only=True,map_location='cpu') for s in previous]
                    if not torch.isfinite(u).all() or not torch.isfinite(v).all() or not torch.equal(u.argmax(1),v.argmax(1)):
                        reasons.append(f'Prior performance trial {t} output issue; later expansion held')
        raw['eligible_for_timing']=not reasons;raw['hold_reason']='; '.join(dict.fromkeys(reasons)) or None
        raw['mode_eligibility']=per_mode
        batches.append(raw)
    return dict(run_id=cfg['run_id'],stage=stage,trial=trial,synthetic_input=False,global_unique_images=4,
                tolerance_status='pending' if cfg['atol'] is None else 'explicit_cli_thresholds',
                accuracy_computed=False,loss_computed=False,batches=batches)


def old_audit(root):
    import torch
    old=Path(read_json(root/'config.json')['previous_run']);assets=read_json(old/'assets.json')
    records=list(csv.DictReader((old/'timings.csv').open())); summaries=[]
    for cid in sorted(set(x['case_id'] for x in records)):
        calls=[x for x in records if x['case_id']==cid]
        vals=[float(x['elapsed_ms']) for x in calls if x['phase']=='repeat' and x['status']=='ok' and x['timing_valid']=='True']
        summaries.append(dict(case_id=cid,mean_ms=stats.mean(vals),valid_repeats=len(vals),
            stdev_ms=stats.stdev(vals),batch=int(calls[0]['batch']),mode=calls[0]['mode']))
    data=torch.load(old/'inputs.pt',weights_only=True,map_location='cpu').to(torch.bfloat16)
    identities={str(b):tensor_hash(data[:b])==assets['batch_hashes'][str(b)] for b in [1,4]}
    audit=dict(previous_run=str(old),timing_records=len(records),recomputed=summaries,input_tensor_hashes_match=identities,
        input_file_matches=sha256(old/'inputs.pt')==assets['input_file_sha256'],
        source_review=dict(timer='common.timed_call: prewait outside timer; runner + stream completion inside; no print/file I/O/cpu output copy',
            model_load='vit.prepare_model_inputs before runner/timer; no pre-forward',
            runner_reuse='one runner per worker; compile/model loading outside loop',
            input='fixed stored CPU synthetic tensor; no regeneration',
            runtime_internal='SDK dispatch/program loading/internal copies can still occur inside runner',
            precision='BF16 model/input/output; SDK implicit casting default retained',
            torch_cpu_build='Not evidence of CPU tensor execution; PrivateUse1 Furiosa extension observed'),
        original_common_sha256=sha256(HERE/'common.py') if (HERE:=Path(__file__).parent) else None,
        previous_config=read_json(old/'config.json'))
    audit['original_synthetic_paired_speedup_recalculated']={str(b):
        next(x['mean_ms'] for x in summaries if x['batch']==b and x['mode']=='eager')/
        next(x['mean_ms'] for x in summaries if x['batch']==b and x['mode']=='compile') for b in [1,4]}
    audit['original_speedup_interpretation']='Historical mixed-path ratio; not a demonstrated NPU-only Eager/Compile speedup'
    write_json(root/'previous_audit.json',audit);return audit


def diagnostic_analysis(root):
    m=read_json(root/'manifest.json');rows=[]
    for s in m['workers']:
        if s['stage'] not in ['diagnostic','strict']:continue
        r=result(root,s)
        text=(root/'logs'/f'{s["case_id"]}.stderr.log').read_text(errors='replace') if (root/'logs'/f'{s["case_id"]}.stderr.log').exists() else ''
        tr=trace_summary(root/'cases'/s['case_id']/'furiosa_trace.json')
        if tr['status']=='available':write_json(root/'cases'/s['case_id']/'trace_summary.json',tr)
        per_call=[]
        stdout=(root/'logs'/f'{s["case_id"]}.stdout.log')
        segments={};current=None
        if stdout.exists():
            for line in stdout.read_text(errors='replace').splitlines():
                try:e=json.loads(line)
                except ValueError:e={}
                if isinstance(e,dict) and e.get('event')=='call_start':current=e['call_index'];segments[current]=[]
                if current is not None:segments[current].append(line)
                if isinstance(e,dict) and e.get('event') in ['call_end','call_error']:current=None
        for o in r.get('observations',[]):
            fragment='\n'.join(segments.get(o['call_index'],[]))
            per_call.append(dict(o,sdk_compile_entry_stdout_log_count=fragment.count('compile without given compiler config'),
                program_load_stdout_log_count=len(re.findall(r'Loading to furiosa:',fragment)),
                actual_codegen_count=None,disk_cache_hit=None,disk_cache_miss=None))
        rows.append(dict(case_id=s['case_id'],stage=s['stage'],mode=s['mode'],batch=s['batch'],status=r.get('status',s['status']),
            cpu_fallback_setting=r.get('effective',{}).get('cpu_fallback'),output_valid=r.get('output_valid'),
            timings=r.get('timings',[]),per_call=per_call,backend_entries=r.get('backend_entries',[]),
            pytorch_recompile_log_count=sum('Recompiling function' in x for x in text.splitlines()),
            cpu_fallback_log_count=text.count('fallback to CPU'),trace=tr,error=r.get('error'),
            stderr=f'logs/{s["case_id"]}.stderr.log',profile=f'cases/{s["case_id"]}/python_profile.txt'))
    write_json(root/'diagnostic_observations.json',rows);return rows


def summarize(root):
    cfg=read_json(root/'config.json');m=read_json(root/'manifest.json');rows=[];timings=[];pairs=[]
    outputs={t:compare_real(root,'performance',t) for t in range(1,cfg['trials']+1)}
    write_json(root/'performance_output_comparison.json',outputs)
    for s in m['workers']:
        if s['stage']!='performance':continue
        r=result(root,s);raw=r.get('timings',[]);timings+=raw
        repeats=[x for x in raw if x['phase']=='repeat'];values=[x['elapsed_ms'] for x in repeats if valid(x)]
        mean=stats.mean(values) if values else None
        normal=r.get('status')=='ok' and len(values)==cfg['repeats'] and r.get('output_valid')
        output=next(x for x in outputs[s['trial']]['batches'] if x['batch']==s['batch'])
        first=next((x['elapsed_ms'] for x in raw if x['phase']=='first' and valid(x)),None)
        warm,why=warm_state(values)
        rows.append(dict(run_id=cfg['run_id'],case_id=s['case_id'],mode=s['mode'],batch=s['batch'],trial=s['trial'],
            status=r.get('status',s['status']),reason=s.get('reason') or r.get('error'),model_id=cfg['model_id'],weights_id=cfg['weights_id'],
            input_hash=r.get('input_hash'),input_file_sha256=r.get('input_file_sha256'),weights_sha256=r.get('weights_sha256'),dtype=r.get('dtype'),
            timing_scope=r.get('timing_scope'),cache_state=r.get('cache_state'),cache_path=s['cache_path'],cache_existed_at_entry=r.get('cache_existed_at_entry'),
            requested_repeats=cfg['repeats'],completed_repeats=sum(x['status']=='ok' for x in repeats),valid_repeats=len(values),
            runner_setup_ms=r.get('runner_setup_ms'),first_call_ms=first,mean_ms=mean,median_ms=stats.median(values) if values else None,
            stdev_ms=stats.stdev(values) if len(values)>=2 else None,throughput_images_s=s['batch']*len(values)*1000/sum(values) if normal else None,
            first_call_extra_ms=first-mean if first is not None and mean is not None else None,compiler_time_ms=None,
            warm_state_assessment=warm,warm_state_reason=why,output_validation_status=output['status'],
            individual_output_structure_finite_ok=r.get('output_valid'),
            tolerance_status=output['tolerance_status'],execution_path_status=default_path_evidence(root,s['mode'],s['batch'])['path_label'],
            full_operator_placement='unverified',synthetic_input=False))
    for b in [1,4]:
        for trial in range(1,cfg['trials']+1):
            e,k=[next(x for x in rows if x['mode']==mode and x['batch']==b and x['trial']==trial) for mode in ['eager','compile']]
            out=next(x for x in outputs[trial]['batches'] if x['batch']==b)
            reasons=[]
            for key in ['input_hash','input_file_sha256','weights_sha256','dtype','timing_scope','cache_state']:
                if e[key]!=k[key] or e[key] is None:reasons.append('Pair identity mismatch/missing '+key)
            if not all(x['status']=='ok' and x['valid_repeats']==cfg['repeats'] and x['throughput_images_s'] is not None for x in [e,k]):reasons.append('Paired worker incomplete/invalid')
            if not out['eligible_for_timing']:reasons.append(out['hold_reason'])
            if any(not x['eligible'] for x in out['mode_eligibility'].values()):reasons.append('One or more modes ineligible for NPU performance expansion')
            pairs.append(dict(run_id=cfg['run_id'],batch=b,trial=trial,eager_case_id=e['case_id'],compile_case_id=k['case_id'],
                compile_speedup=e['mean_ms']/k['mean_ms'] if not reasons else None,comparable=not reasons,
                provisional=True,hold_reason='; '.join(reasons) or None,
                provisional_reason='All-op placement unverified; tolerance pending unless explicit CLI thresholds',output_status=out['status']))
    across=[]
    for b in [1,4]:
        for mode in ['eager','compile']:
            good=[x for x in rows if x['batch']==b and x['mode']==mode and x['status']=='ok' and x['throughput_images_s'] is not None]
            means=[x['mean_ms'] for x in good]
            across.append(dict(batch=b,mode=mode,completed_independent_trials=len(means),trial_mean_ms=means,
                trial_throughput_images_s=[x['throughput_images_s'] for x in good],
                median_of_trial_means_ms=stats.median(means) if means else None,min_trial_mean_ms=min(means) if means else None,
                max_trial_mean_ms=max(means) if means else None,between_trial_sample_stdev_ms=stats.stdev(means) if len(means)>=2 else None))
    write_csv(root/'timings.csv',timings,TIMING_FIELDS)
    write_csv(root/'trial_summary.csv',rows,list(rows[0]))
    write_csv(root/'comparisons.csv',pairs,list(pairs[0]))
    write_csv(root/'across_trial_summary.csv',across,list(across[0]))
    return rows,timings,pairs,across


def verify(root,rows,timings,pairs):
    cfg=read_json(root/'config.json');errors=[]
    def close(x,y,label):
        if x is None and y is None:return
        if x is None or y is None or not math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-10):errors.append(label)
    with (root/'timings.csv').open() as f:raw=list(csv.DictReader(f))
    recalculated={}
    for s in rows:
        own=[x for x in raw if x['case_id']==s['case_id']]
        v=[float(x['elapsed_ms']) for x in own if x['phase']=='repeat' and x['status']=='ok' and x['timing_valid']=='True' and float(x['elapsed_ms'])>0]
        n=len(v);avg=sum(v)/n if n else None
        sd=math.sqrt(sum((x-avg)**2 for x in v)/(n-1)) if n>=2 else None
        close(s['mean_ms'],avg,s['case_id']+' mean');close(s['stdev_ms'],sd,s['case_id']+' sample SD')
        if s['throughput_images_s'] is not None:close(s['throughput_images_s'],s['batch']*n*1000/sum(v),s['case_id']+' images/s')
        if s['status']=='ok' and len(own)!=1+cfg['warmup']+cfg['repeats']:errors.append(s['case_id']+' timing count')
        recalculated[s['case_id']]=avg
        if s['status']=='ok' and s['cache_existed_at_entry'] is not False:errors.append(s['case_id']+' SDK cache was not new')
    for pair in pairs:
        if pair['compile_speedup'] is not None:
            close(pair['compile_speedup'],recalculated[pair['eager_case_id']]/recalculated[pair['compile_case_id']],str(pair['batch'])+' speedup')
    all_ok=all(s['status']=='ok' for s in rows)
    if all_ok and len(raw)!=cfg['expected_performance_records']:errors.append('Total performance timing count')
    assets=read_json(root/'assets.json')
    complete=[s for s in rows if s['status']=='ok']
    for key in ['input_file_sha256','weights_sha256','dtype','timing_scope']:
        if len(set(s[key] for s in complete))>1:errors.append('Across-trial mismatch '+key)
    for s in complete:
        if s['input_hash']!=assets['batch_hashes'][str(s['batch'])]:errors.append(s['case_id']+' input tensor hash')
    if len(set(s['cache_path'] for s in rows))!=len(rows):errors.append('Performance cache paths reused')
    held=[s for s in rows if s['status']=='held']
    eligible=[s for s in rows if s['status']!='held']
    eligible_complete=bool(eligible) and all(s['status']=='ok' for s in eligible)
    expected_eligible_records=len(eligible)*(1+cfg['warmup']+cfg['repeats'])
    if eligible_complete and len(raw)!=expected_eligible_records:errors.append('Eligible performance count mismatch')
    audit=dict(utc=now(),status='pass' if not errors else 'fail',errors=errors,
        expected_performance_records=cfg['expected_performance_records'],actual_performance_records=len(raw),
        completed_valid_performance_records=sum(valid(t) for t in timings),completed_workers=len(complete),
        expected_workers=4*cfg['trials'],all_workers_complete=all_ok,
        held_workers=len(held),expected_records_for_unheld_workers=expected_eligible_records,
        all_unheld_workers_complete=eligible_complete,
        note='Only performance rows counted; diagnostics and output verification excluded; repeats are not independent trials')
    write_json(root/'artifact_audit.json',audit);return audit


def validation_reference_checks(root,rows):
    """Check executed performance logits against the independent mixed Eager reference; no latency pairing."""
    import torch
    from torchvision.models import ViT_B_16_Weights
    cfg=read_json(root/'config.json');m=read_json(root/'manifest.json');assets=read_json(root/'assets.json')
    categories=ViT_B_16_Weights.IMAGENET1K_V1.meta['categories'];checks=[]
    for s in rows:
        current_spec=next(x for x in m['workers'] if x['case_id']==s['case_id'])
        current=result(root,current_spec)
        if current.get('status')!='ok' or not current.get('output_valid'):continue
        reference_spec=next(x for x in m['workers'] if x['stage']=='validation' and x['mode']=='eager' and x['batch']==s['batch'])
        reference=result(root,reference_spec)
        if not reference.get('output_valid'):continue
        pair=[]
        for spec,r in [(reference_spec,reference),(current_spec,current)]:
            pair.append(dict(case_id=spec['case_id'],batch=s['batch'],trial=s['trial'],weights_sha256=r.get('weights_sha256'),
                input_hash=r.get('input_hash'),dtype=r.get('dtype'),timing_scope=r.get('timing_scope'),
                cache_state='Output-only reference check; different cache stages explicitly retained'))
        check=compare_outputs(root,*pair,assets,cfg)
        check['tolerance_status']='pending' if cfg['atol'] is None else 'explicit_cli_thresholds'
        check.update(comparison_type='performance logits vs independent validation Eager reference',
            reference_execution_path='CPU/NPU mixed Eager; output reference only, never a performance denominator',
            reference_cache_path=reference_spec['cache_path'],performance_cache_path=current_spec['cache_path'],
            speedup=None,speedup_reason='Reference is from output verification, not a performance trial')
        y=torch.load(root/'outputs'/f'{s["case_id"]}.pt',weights_only=True,map_location='cpu')
        check['current_predictions']=[dict(input_id=assets['input_ids'][i],class_index=int(t.argmax()),
            class_name=categories[int(t.argmax())],logit=float(t.max())) for i,t in enumerate(y)]
        checks.append(check)
    write_json(root/'trial_validation_reference_checks.json',dict(checks=checks,accuracy_computed=False,loss_computed=False))
    return checks


def analyze(root):
    root=Path(root).resolve()
    old=old_audit(root);diagnostics=diagnostic_analysis(root)
    comparison=compare_real(root);write_json(root/'real_output_comparison.json',comparison)
    rows,timings,pairs,across=summarize(root)
    validation_reference_checks(root,rows)
    audit=verify(root,rows,timings,pairs)
    plot_data=dict(config=read_json(root/'config.json'),rows=rows,across=across,pairs=pairs)
    write_json(root/'figures/plot_data.json',plot_data)
    plot_python=Path(__file__).parent/'.plot-venv/bin/python'
    if plot_python.exists() and any(s['mean_ms'] is not None for s in rows):
        p=subprocess.run([str(plot_python),str(Path(__file__).parent/'followup_plots.py'),str(root)],capture_output=True,text=True,timeout=120)
        (root/'logs/plots.log').write_text(p.stdout+'\n'+p.stderr)
        write_json(root/'figures/status.json',dict(status='created' if p.returncode==0 else 'error',files=[x.name for x in (root/'figures').glob('*.png')]))
    else:write_json(root/'figures/status.json',dict(status='not_created',reason='No measured performance or isolated plot environment unavailable'))
    write_reports(root,old,diagnostics,comparison,rows,pairs,across,audit)
    print(json.dumps(dict(run_dir=str(root),audit=audit),ensure_ascii=False))


def write_reports(root,old,diagnostics,comparison,rows,pairs,across,audit):
    # Kept as one small reporting function so re-analysis never launches an NPU model.
    cfg=read_json(root/'config.json');m=read_json(root/'manifest.json');images=read_json(root/'image_manifest.json')
    paths=[default_path_evidence(root,mode,b) for b in [1,4] for mode in ['eager','compile']]
    path_table=table(['케이스','상태','NPU 사용','CPU fallback 관측','전체 연산 배치'],[
        [f'{p["mode"]} B{p["batch"]}',p['diagnostic_status'],p['npu_use_confirmed'],p['actual_cpu_fallback_observed'],p['full_operator_placement']] for p in paths])
    strict_table=table(['strict case','상태','CPU fallback 설정','출력 유효'],[[x['case_id'],x['status'],x['cpu_fallback_setting'],x['output_valid']] for x in diagnostics if x['stage']=='strict'])
    diag_table=table(['case','first ms','진단 repeat ms (3회)','대표 trace NPU task','CPU fallback span'],[
        [d['case_id'],next((t['elapsed_ms'] for t in d['timings'] if t['phase']=='first'),None),
         [round(t['elapsed_ms'],4) for t in d['timings'] if t['phase']=='repeat' and valid(t)],d['trace'].get('npu_task_count'),
         d['trace'].get('cpu_fallback_execution_spans')] for d in diagnostics if d['stage']=='diagnostic'])
    hypotheses='''| 가설 | 관측·판단 | 근거 |
| --- | --- | --- |
| 기존 타이머에 매회 모델 로드/전처리/compile 래퍼/CPU 출력 저장 포함 | [문서·소스 확인] 코드에서 해당 작업은 타이머 밖이다. SDK 내부 준비 비용은 별개다. | common.py timed_call, vit.py prepare_model_inputs, 이전 worker 소스 사본 |
| 모델 tensor 연산의 CPU fallback이 Eager 지연의 원인 | [실행 확인/미확인] 아래 fallback 로그·executed_by_cpu trace·strict 결과 범위로 판단한다. 호스트 Python 실행 자체는 tensor fallback이 아니다. | diagnostics JSON, 원본 로그·trace |
| 반복 중 실제 codegen/프로그램 재로드 | [실행 확인/미확인] 호출별 SDK compile-entry/load 로그와 LRU 전후를 분리 기록했다. 실제 codegen·디스크 hit/miss는 unknown이다. | diagnostic_observations.json per_call |
| NPU 호출 수·CPU dispatch/런타임·내부 전송·대기 | [실행 확인/추정] 대표 호출의 NPU task와 호스트 span/cProfile을 관측한다. 큰 지연 기여 후보지만 중첩 span을 더해 원인별 비율을 만들지 않는다. | cases/diagnostic_*/furiosa_trace.json, torch_cpu_trace.json, python_profile.txt |
'''
    native_table=table(['기본 Eager','대표 CPU native attention 호출 수','해당 호스트 구간 inclusive ms','공식 strict 오류'],[
        [f'B{p["batch"]}',p['native_attention_cpu_fallback_evidence']['default_cpu_profile'].get('count'),
         p['native_attention_cpu_fallback_evidence']['default_cpu_profile'].get('total_cpu_ms'),
         ('B1 strict: CPU fallback is disabled for aten::_native_multi_head_attention' if p['batch']==1 else 'B1 strict 오류를 참고; B4 strict는 보류') if p['native_attention_cpu_fallback_evidence']['observed'] else '미관측'] for p in paths if p['mode']=='eager'])
    path_doc=f'''# ViT 실행 경로·Eager 지연 근거

이전 run: `{cfg['previous_run']}`. 합성 원본 입력을 그대로 사용했다. 파일/텐서 해시 재확인은 [previous_audit.json](previous_audit.json), 원본 시간 재계산도 같은 파일에 있다. 기존 common.py timer와 vit.py 모델 준비 함수는 수정하지 않았다. +cpu 버전 표기는 tensor CPU fallback 증거가 아니다.

{path_table}

{strict_table}

{native_table}

[실행 확인] 기본 Eager의 native attention 12회와 CPU 계산(mm/bmm/softmax 등), native attention의 PrivateUse1 전용 kernel 부재, strict B1의 명시적 CPU fallback 금지 오류를 함께 확인했다. B4 strict는 B1 실패 규약에 따라 미실시했으며 B4 기본 trace에서 같은 native attention CPU 경로가 관측됐다. Python `run_by_cpu::executed_by_cpu` span 또는 `fallback to CPU` 로그가 0이어도 이 native dispatcher fallback은 존재한다. 따라서 이전 Python fallback 로그 부재를 CPU fallback 부재로 확장할 수 없다. NPU와 CPU tensor 연산이 섞인 기본 Eager 결과로 판단하며, 해당 성능 확대를 보류한다. 위 inclusive 호스트 시간은 복사·대기를 포함할 수 있고 순수 CPU kernel 시간/전체 지연 비율로 해석하지 않는다.

추가 원본 근거: logs/native_attention_dispatch.json, logs/followup_installed_sources.txt, cases/diagnostic_eager_b*/torch_cpu_summary.json 및 python_profile.txt. 반복별 compile-entry/load 로그와 프로세스 LRU는 diagnostic_observations.json에 남겼다. 해당 캐시에서 반복 compile-entry/load가 관측되지 않아도 실제 compiler 내부 이벤트의 완전한 부재를 증명하지는 않는다.

공식 제어 [문서·소스 확인]: 설치 `furiosa/torch/config/_cpu_fallback.py:cpu_fallback(False)`는 해당 worker context에서 SDK CPU fallback을 비활성화하고 원래 값으로 복원한다. Python UnsupportedOpError 분기 외에 관측한 native backend fallback에도 적용되어 strict B1에서 RuntimeError가 발생했다. Compile backend eager_fallback=False와 별개다. strict 성공은 이 계약 범위의 점검이며 전체 연산 배치 증명은 아니다. 기본 성능 worker에는 strict 설정을 넣지 않았다.

{diag_table}

{hypotheses}

SDK 관측 근거: 설치 `_C/profiler/__init__.pyi:Profiler/RecordFunction`, config/_env.py의 FURIOSA_TORCH_PROFILE=trace, config/_stream.py current_stream 및 `_C/stream/__init__.pyi:Stream.synchronize`. CPU fallback 실제 실행 span은 `_backend/_exported_program.py:run_by_cpu::executed_by_cpu`, Eager 분기 근거는 `_backend/eager.py`다. 설치 사본/해시: logs/installed_sources.txt 및 environment.json. 상세 로그와 profiler는 진단 worker 및 마지막 추가 대표 호출에서만 사용했다. profiler 호출은 672개 성능 시간에 포함하지 않는다. CPU profiler 시간은 NPU kernel 시간이 아니며 비동기·중첩 span을 단순 합산하지 않는다.

정확한 최초 오류가 있으면 cases/<case>/result.json 및 logs/<case>.stderr.log의 traceback과 input_metadata를 참고한다. 기본·strict 오류는 별개 상태다. 실제 codegen 시간이나 CPU/NPU 시간 비율은 독립적으로 관측하지 않아 만들지 않았다.
'''
    (root/'path_evidence.md').write_text(path_doc)
    perf_table=table(['trial','mode','B','상태','setup ms','first ms','repeat mean±SD ms','images/s'],[
        [s['trial'],s['mode'],s['batch'],s['status'],s['runner_setup_ms'],s['first_call_ms'],f'{fmt(s["mean_ms"])}±{fmt(s["stdev_ms"])}',s['throughput_images_s']] for s in rows])
    across_table=table(['B','mode','독립 trial 수','trial mean ms','중앙값 [최소, 최대] ms','trial images/s'],[
        [s['batch'],s['mode'],s['completed_independent_trials'],[round(x,4) for x in s['trial_mean_ms']] or '보류/미실시',
         f'{fmt(s["median_of_trial_means_ms"])} [{fmt(s["min_trial_mean_ms"])}, {fmt(s["max_trial_mean_ms"])}]',
         [round(x,2) for x in s['trial_throughput_images_s']] or '보류/미실시'] for s in across])
    pair_table=table(['B','trial','예비 paired speedup','상태/보류 이유'],[[x['batch'],x['trial'],x['compile_speedup'],
        'Eager CPU fallback 조건의 성능 확대 보류' if x['compile_speedup'] is None and any(p['mode']=='eager' and p['batch']==x['batch'] and p['actual_cpu_fallback_observed'] for p in paths) else x['hold_reason'] or '전체 연산 배치/출력 허용 기준 미확인'] for x in pairs])
    output_table=table(['B','max/mean abs','cosine 평균/최소','Top-1 일치','허용 기준','출력 구조·예측 일치'],[
        [x['batch'],f'{fmt(x["max_abs_error"])}/{fmt(x["mean_abs_error"])}',
         f'{x["cosine_mean"]:.8f}/{x["cosine_min"]:.8f}' if x['cosine_mean'] is not None else None,
         f'{x["top1_matches"]}/{x["compared_images"]}',x['tolerance_status'],x['eligible_for_timing']] for x in comparison['batches']])
    image_rows=[]
    for b in comparison['batches']:
        for x in b['images']:
            image_rows.append([b['batch'],x['input_id'],x['eager']['class_name'],x['eager']['class_index'],x['eager']['logit'],x['compile']['class_name'],x['compile']['class_index'],x['compile']['logit'],x['top1_match']])
    predictions=table(['B','이미지','Eager 예측','index','logit','Compile 예측','index','logit','일치'],image_rows)
    unique_images=table(['순서/ID','실제 절대 경로','원본 W×H','제공된 출처/조건'],[
        [x['id'],x['actual_absolute_path'],x['actual_original_size_wh'],x['author_or_credit']+' / '+x['license']] for x in images['images']])
    old_table=table(['이전 합성 / mode','B','repeat 평균 ms','반복수'],[[x['mode'],x['batch'],x['mean_ms'],x['valid_repeats']] for x in old['recomputed']])
    completed=audit['completed_workers'];full=len(rows)
    decision='조건부 진행' if completed==full else '문제 해결 또는 보류 조건 확인 우선'
    if any(x['actual_cpu_fallback_observed'] for x in paths):decision='Compile은 유효 조건에서 조건부 진행 / Eager 기본 경로는 문제 해결 우선'
    intro=f'''# ViT 후속 실험: 실행 경로·실제 이미지·독립 trial

Run `{root.name}`, 이전 합성 run은 `{cfg['previous_run']}`. 모든 시간 원본은 UTC, 사용자 일정 표기는 KST(UTC+9). **예비 결과**, BF16·기본 1 PE, B1/B4 유지.

1. **기존 약 1초 Eager 지연 [문서·소스/실행 확인]**: 명시적 모델 로드·입력 준비·출력 저장은 타이머 밖이었다. 기본 Eager에서 native attention CPU fallback을 확인했고, 대표 호출에서 native attention 12회의 호스트 inclusive 합계는 B1 약 1017ms, B4 약 1006ms였다. 복사·대기를 포함할 수 있어 원인별 비율이나 순수 CPU 계산 시간으로 표현하지 않는다. 기존 합성 배속을 NPU 전용 두 모드 비교로 해석할 수 없다. [실행 경로 근거](path_evidence.md).
2. **CPU fallback 및 NPU 사용 [실행 확인/미확인]**: {sum(x['npu_use_confirmed'] for x in paths)}/4 경로에서 NPU 사용 근거, {sum(x['actual_cpu_fallback_observed'] for x in paths)}/4 경로에서 실제 CPU tensor fallback을 관측했다. 전체 연산 배치는 미확인이다. strict 진단과 기본 설정 상태를 분리했다.
3. **실제 이미지 [실행 확인]**: 제공된 cat → coffee → astronaut → rocket 순서, 파일 해시 검증 후 weights.transforms()를 한 번 적용해 저장했다. synthetic_input=false, 고유 이미지 4개이며 B1은 B4 첫 사진을 재사용한다. 정답 라벨이 없어 accuracy/loss를 계산하지 않았다.
4. **독립 실행 재현성 [실행 확인/미확인]**: 성능 worker {completed}/{full} 완료, {audit['held_workers']}개 보류. 유효 시간 {audit['completed_valid_performance_records']}개; 전체 설계는 {cfg['expected_performance_records']}개이며 보류 제외 예정 수는 {audit['expected_records_for_unheld_workers']}개다. 각 worker 첫 1 + warmup {cfg['warmup']} + repeat {cfg['repeats']}. Eager 확대가 보류되면 paired speedup도 계산하지 않는다. 독립 trial 평균의 중앙값/최소/최대를 아래에 제시하며 반복 150회를 독립 실험 150개로 세지 않는다.
5. **본실험 판단: {decision} [실행 확인/미확인]**. 출력 허용 기준은 {'pending' if cfg['atol'] is None else 'CLI 명시 기준'}, 전체 연산 배치 및 실제 codegen/disk cache hit는 미확인이다. batch 8·16/모듈 분석으로 확대하지 않았다.

{across_table}

{pair_table}

{output_table}

위 출력 점검과 모드별 확대 조건은 별개다. 출력이 일치해도 CPU fallback이 확인된 Eager 성능 확대는 보류했다.
'''
    feedback=intro+f'''\n실행 경로: {', '.join(p['mode']+' B'+str(p['batch'])+': '+p['path_label'] for p in paths)}.

캐시: 진단/실제 출력 검증/성능을 분리했다. 성능 worker별 새 SDK cache_context 경로를 사용한다. 해당 경로 초기 부재가 시스템 전체 cold 또는 실제 disk miss를 뜻하지 않는다. CPU dispatch·내부 전송·실행·완료 대기를 포함한 ms/batch call이며 kernel-only가 아니다. 상세 로그/프로파일은 성능 worker에 없다.

장애물·질문(최대 3개): SDK 2026.4.0의 원본 native attention에 대한 공식 NPU 지원·해결 경로; BF16 logits atol/rtol 합의; Compile 전체 연산 배치·실제 codegen·disk cache의 공식 관측 방법. 실제 이미지 4장과 trial 3개는 전체 입력 분포나 일반적 우열을 대표하지 않는다.
'''
    (root/'feedback_summary.md').write_text(feedback)
    details=f'''
## 실제 입력 및 전처리

{unique_images}

원본 제공 manifest 사본: inputs/provided_image_manifest.json. 출처·설명·순서는 보존했고 설명을 정답으로 사용하지 않았다. [image_manifest.json](image_manifest.json)은 파일 SHA256과 FP32/BF16 tensor SHA256을 구분한다. 전처리 정보: `{images['preprocessing']}`. 모든 worker가 같은 `{root/'inputs/preprocessed_bf16.pt'}`를 로드하고 batch별 해시를 확인한다. 전처리를 반복 수행하지 않는다. 원본 FP32 텐서는 `{root/'inputs/preprocessed_fp32.pt'}`.

{predictions}

클래스 이름은 weights.meta.categories의 예측 이름이다. 점수는 raw logit이며 정답/accuracy/loss가 아니다. 허용 기준을 결과에 맞춰 정하지 않았다. 영벡터 cosine은 null 및 이유, 이미지별 오차/예측과 의심스러운 동일 출력 검사는 real_output_comparison.json에 보존했다.

## 실제 이미지 성능 trial별 원본 요약

{perf_table}

표준편차는 worker 내 반복의 표본 SD(ddof=1), 독립 trial 사이 변동은 across_trial_summary.csv에 별도 기록한다. 처리량=B×유효 반복 수/유효 반복 초 합계이며 서버 최대 처리량이 아니다. 첫 호출은 새 SDK 캐시 경로 조건에서 별도 관측했고 first_call_extra는 순수 compile 시간이 아니다. compiler_time_ms=null.

## 이전 합성 결과: 별도 표

{old_table}

이전 run은 반복 20회·trial 1회·smoke와 SDK cache 공유, 새 run은 실제 이미지·반복 {cfg['repeats']}회·trial {cfg['trials']}회·worker별 새 SDK cache다. 입력 내용 하나의 효과로 차이를 해석하지 않는다.

## 실행·캐시·코드·보호 규칙

환경은 environment.json, 실제 worker 순서·PID·시각·명령·cache 경로·초기 상태는 manifest.json 및 cases/*/result.json에 기록했다. 성능 worker별 torch thread/inter-op thread, SDK compiler/device config, CPU fallback 기본값, 입력·가중치 해시와 측정 경계가 저장된다. 원래 common.py/vit.py는 그대로 재사용하고 새 followup.py/followup_worker.py만 진입점을 추가했다. 변경 전후 해시와 code/ 사본을 보존한다. 기존 합성 run과 기존 SDK/torch/드라이버/공용 캐시를 수정하거나 삭제하지 않는다.

진단 코드 수정 이력: 최초 diagnostic_compile_b1은 관측 wrapper의 중복 utc 인자로 실패했다. SDK 미지원 오류가 아니다. 원본 실패를 보존하고 새 retry case ID에서 수정된 관측 wrapper로 재실행했다. 수정·재개를 위해 이 실행의 orchestrator에만 SIGINT를 보냈고 당시 strict_compile_b4의 자기 worker만 중단했다. 이 중단은 timeout/미지원으로 해석하지 않는다. strict_compile_b4도 새 attempt에서 재개하며 기존 성공 결과·사용 예산을 유지한다. 성능 worker의 모델·입력·timer·backend 설정은 이 수정으로 바뀌지 않았다. 최초/재개 명령과 해시는 manifest 및 resume_history에 남겼다.

공식 current_stream.synchronize 준비/호출 전/후 완료 대기를 사용한다. 준비에만 query=True를 확인하고 timer 안에는 runner + 완료 대기만 있다. 모델/입력 전송·로그·파일 저장·CPU 출력 복사는 timer 밖이다. 기본 torch.compile(backend=ft.backend)에서 dynamic/fullgraph/mode/options를 지정하지 않는다. 성능 runner에 진단 backend wrapper/profiler/hook/strict 옵션을 넣지 않는다. eval/inference_mode와 BF16을 유지한다.

trace의 `npu...::Task::compile`은 이름이 compile인 NPU 실행 프로그램 task이며 실제 compiler codegen 이벤트가 아니다. 진단 Profiler는 자체 profile 결과 D2H 전송도 추가할 수 있으므로 trace의 모든 copy 이벤트를 모델 내부 전송으로 합산하지 않는다. native attention CPU 구간과 copy 함수 호출의 포함 관계도 단순 더하지 않는다.

NPU worker는 직렬, 매 worker 전 device usage를 재확인한다. 기존 사용자 PID가 보이면 실행을 보류한다. worker 기본 1800초, 전체 NPU 예산 7200초(설정값은 config.json). 운영상 상한이며 timeout은 미지원 판정이 아니다. 이 실행에서 생성한 process group만 정리한다. 실제 worker 중에는 외부 작업이 새로 시작할 수 있어 원자적 장치 예약을 보장하지 않는다.

## 산출물 일관성

`{audit}`

미측정/실패/보류는 0ms로 채우지 않는다. JSON nonfinite 값은 허용하지 않는다. 느린 정상 호출은 삭제하지 않는다. input/weight/dtype/timer identity, worker별 cache path 고유성 및 초기 부재, CSV를 다시 읽은 평균/표본 SD/throughput/paired speedup을 검사했다. [원본 시간](timings.csv), [trial별 요약](trial_summary.csv), [배속](comparisons.csv), [trial 사이 요약](across_trial_summary.csv), [원본 audit](artifact_audit.json). 그림은 figures/ 아래 PNG이며 로그 축이면 축과 caption에 표시한다.

## 재현·단계별 재개

정확한 실행 명령: `{m['command']}`. `followup.py run --images FILE FILE FILE FILE`는 새 고유 run을 만든다. `--stop-after diagnostic` 또는 `--stop-after validation`으로 단계 경계에서 멈춘 뒤 `followup.py resume --run-dir RUN`으로 미실시 단계만 진행한다. 이미 실행한 worker는 덮어쓰지 않는다. 재집계: `/root/venv/bin/python vit_pilot/followup.py analyze --run-dir {root}`. 기존 CLI 형식도 `pilot.py --images`가 nargs=4임을 확인했다. 모델당 첫 호출을 검증 forward로 소비하지 않는다.

최종 판단: **{decision}**. 실제 장애물은 raw 오류/보류 이유를 참고하고, 추가 본실험은 출력 기준·실행 경로 확인 범위를 합의한 뒤 검토한다. 이번 작업은 B1·B4·ViT 세 범위만 수행한다.
'''
    (root/'report.md').write_text(intro+details)


if __name__=='__main__':analyze(Path(sys.argv[1]))
