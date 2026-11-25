"""Aggregate only saved raw records; output comparison and optional matplotlib plots."""
import csv
import json
import math
from pathlib import Path
import re
import statistics as stats
import sys
from common import read_json, write_json, write_csv, now, sha256

TIMING_FIELDS=['run_id','case_id','mode','batch','trial','phase','iteration','elapsed_ms','timing_valid','status']
SUMMARY_FIELDS=['run_id','case_id','mode','batch','trial','status','reason','model_id','weights_id','weights_sha256',
 'input_hash','input_ids','dtype','shape','requested_repeats','completed_repeats','valid_repeats','first_call_ms',
 'runner_setup_ms','mean_ms','median_ms','stdev_ms','throughput_images_s','throughput_reason',
 'first_call_extra_ms','compiler_time_ms','warm_state_assessment','warm_state_reason','validation_status',
 'execution_path_status','timing_scope','cache_state']
SHAPE_FIELDS=['run_id','case_id','mode','call_index','batch','status','start_utc','end_utc','elapsed_ms','timing_valid',
 'pytorch_recompile_events','guard_failure_reason','backend_calls','backend_event_kind','sdk_compile_entry_log_count',
 'furiosa_actual_compile_count','disk_cache_hit','disk_cache_miss','process_lru_delta','cache_state','execution_path_status',
 'graph_break_log_count','sdk_segment_log_records','cpu_fallback_log_count','observation_reason','log_ref']
COMPARE_FIELDS=['run_id','batch','trial','eager_case_id','compile_case_id','compile_speedup','comparable',
 'provisional','hold_reason','output_validation_status']


def valid(row):
    x=row.get('elapsed_ms')
    return row.get('status')=='ok' and row.get('timing_valid') is True and isinstance(x,(int,float)) and math.isfinite(x) and x>0


def warm_state(values):
    if len(values)<10:
        return 'unknown','Fewer than 10 valid repeat calls; no steady-state claim'
    cv=stats.stdev(values)/stats.mean(values)
    n=len(values)//2
    drift=abs(stats.mean(values[:n])-stats.mean(values[n:]))/stats.mean(values)
    state='preliminary_stable' if cv<=0.10 and drift<=0.10 else 'not_stable'
    return state,f'Heuristic only: repeat CV={cv:.4f}, half-mean relative drift={drift:.4f}; thresholds 0.10; compiler events unobserved in performance logs'


def load_result(run,case):
    p=run/'cases'/case['case_id']/'result.json'
    return read_json(p) if p.exists() else {}


def path_status(text):
    if 'fallback to CPU' in text:
        return 'cpu_fallback_observed'
    segments=re.findall(r'split into (\d+) submodules',text)
    if segments:
        return ('sdk_graph_segmentation_observed_all_op_placement_unverified' if any(int(x)>1 for x in segments)
                else 'single_sdk_segment_observed_all_op_placement_unverified')
    return 'unknown_all_op_placement_unverified'


def compare_outputs(run,eager,compiled,assets,c):
    base=dict(batch=eager['batch'],trial=eager['trial'],eager_case_id=eager['case_id'],
              compile_case_id=compiled['case_id'],status='not_compared',tolerance_status='unset',
              accuracy_vs_ground_truth=None,accuracy_reason='No labeled accuracy evaluation',
              input_kind='synthetic' if assets.get('synthetic_input') else 'real',
              global_unique_samples=assets.get('unique_samples'),
              max_abs_error=None,mean_abs_error=None,cosine_per_image=None,cosine_mean=None,
              cosine_min=None,top1_matches=None,compared_images=None,top1_ratio=None,mismatches=[])
    for k in ['weights_sha256','dtype','input_hash','timing_scope','cache_state']:
        if eager.get(k)!=compiled.get(k) or eager.get(k) is None:
            base['reason']=f'Pair condition mismatch/missing: {k}'; return base
    paths=[run/'outputs'/f'{x["case_id"]}.pt' for x in [eager,compiled]]
    if not all(p.exists() for p in paths):
        base['reason']='Paired saved logits unavailable'; return base
    import torch
    u,v=[torch.load(p,map_location='cpu',weights_only=True).float() for p in paths]
    base.update(eager_shape=list(u.shape),compile_shape=list(v.shape),
        eager_nan=int(torch.isnan(u).sum()),eager_inf=int(torch.isinf(u).sum()),
        compile_nan=int(torch.isnan(v).sum()),compile_inf=int(torch.isinf(v).sum()))
    if list(u.shape)!=[eager['batch'],1000] or u.shape!=v.shape:
        base.update(status='failed',reason='Output structure mismatch'); return base
    if not (torch.isfinite(u).all() and torch.isfinite(v).all()):
        base.update(status='failed',reason='NaN/Inf in logits'); return base
    d=(u-v).abs()
    cosine=[]
    reasons=[]
    for i,(x,y) in enumerate(zip(u.double(),v.double())):
        denom=x.norm()*y.norm()
        value=float(x.dot(y)/denom) if denom>0 else None
        if value is None: reasons.append(f'image {i}: zero norm; cosine undefined')
        cosine.append(value)
    finite_cos=[x for x in cosine if x is not None]
    eu,co=u.argmax(1),v.argmax(1)
    n=int((eu==co).sum()); b=len(u)
    mismatch=[]
    for i in range(b):
        if eu[i]!=co[i]:
            e,k=int(eu[i]),int(co[i])
            mismatch.append(dict(input_id=assets['input_ids'][i],eager_class=e,compile_class=k,
                eager_top1_logit=float(u[i,e]),compile_top1_logit=float(v[i,k]),
                eager_logit_at_compile_class=float(u[i,k]),compile_logit_at_eager_class=float(v[i,e])))
    base.update(status='structure_finite_ok_tolerance_unset',max_abs_error=float(d.max()),mean_abs_error=float(d.mean()),
        cosine_per_image=cosine,cosine_mean=stats.mean(finite_cos) if len(finite_cos)==b else None,
        cosine_min=min(finite_cos) if len(finite_cos)==b else None,cosine_undefined_reasons=reasons,
        top1_matches=n,compared_images=b,top1_ratio=n/b,mismatches=mismatch,
        input_ids=assets['input_ids'][:b],reason='Output tolerance not agreed; small-input consistency only')
    if c['atol'] is not None and c['rtol'] is not None:
        ok=bool(torch.allclose(u,v,atol=c['atol'],rtol=c['rtol']))
        base.update(tolerance_status='explicit_cli_thresholds',atol=c['atol'],rtol=c['rtol'],allclose=ok,
                    status='passed_explicit_tolerance' if ok else 'failed',reason='Explicit CLI atol/rtol check')
    return base


def diagnostic_rows(run,manifest,c):
    rows=[]
    for case in manifest['workers']:
        if case['kind']!='switch': continue
        r=load_result(run,case)
        raw={x['call_index']:x for x in r.get('calls',[])}
        stderr=run/'logs'/f'{case["case_id"]}.stderr.log'
        text=stderr.read_text(errors='replace') if stderr.exists() else ''
        stdout=run/'logs'/f'{case["case_id"]}.stdout.log'
        stdout_parts={}; current=None
        if stdout.exists():
            for line in stdout.read_text(errors='replace').splitlines():
                try: event=json.loads(line)
                except (ValueError,TypeError): event={}
                if isinstance(event,dict) and event.get('event')=='call_start':
                    current=event['call_index']; stdout_parts[current]=[]
                if current is not None: stdout_parts[current].append(line)
                if isinstance(event,dict) and event.get('event') in ['call_end','call_error']:
                    current=None
        events_path=run/'cases'/case['case_id']/'events.jsonl'
        events=[json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
        for i,b in enumerate(c['shape_sequence'],1):
            x=raw.get(i,{})
            active=r.get('active_call') or {}
            match=re.search(r'VIT_CALL_BEGIN [^\n]*"call_index": '+str(i)+r'[,}][^\n]*\n(.*?)(?:VIT_CALL_END '+str(i)+r' |\Z)',text,re.S)
            fragment=match.group(1) if match else ''
            sdk_fragment='\n'.join(stdout_parts.get(i,[]))
            backend=sum(e['event']=='backend_entry' and e.get('call_index')==i for e in events)
            recompiles=len(re.findall(r'\[__recompiles\].*Recompiling function',fragment)) if match else None
            guards=[line.strip() for line in fragment.splitlines() if any(s in line for s in ['guard failure','tensor \'','size mismatch','GLOBAL_STATE'])]
            delta={}
            if x.get('lru_before') and x.get('lru_after'):
                for name,before in x['lru_before'].items():
                    after=x['lru_after'][name]
                    delta[name]={k:after[k]-before[k] for k in ['hits','misses']}
            started=any(e['event']=='call_start' and e.get('call_index')==i for e in events)
            status=x.get('status') or (r.get('status',case['status']) if started else 'not_run')
            rows.append(dict(run_id=c['run_id'],case_id=case['case_id'],mode=case['mode'],call_index=i,batch=b,
                status=status,start_utc=x.get('start_utc'),end_utc=x.get('end_utc'),elapsed_ms=x.get('elapsed_ms'),
                timing_valid=x.get('timing_valid',False),pytorch_recompile_events=recompiles,
                guard_failure_reason=guards or None,backend_calls=backend if events else None,
                backend_event_kind=('initial_backend_entry' if sum(e['event']=='backend_entry' and e.get('call_index',i)<i for e in events)==0
                                    else 'subsequent_backend_entry') if backend else None,
                sdk_compile_entry_log_count=len(re.findall('compile without given compiler config',sdk_fragment)) if i in stdout_parts else None,
                furiosa_actual_compile_count=None,disk_cache_hit=None,disk_cache_miss=None,
                process_lru_delta=delta or None,cache_state=x.get('cache_state','unknown'),
                execution_path_status=path_status(fragment+'\n'+sdk_fragment),
                graph_break_log_count=len(re.findall(r'\[__graph_breaks\].*Graph break',fragment)) if match else None,
                sdk_segment_log_records=re.findall(r'split into \d+ submodules, \d+ to compile',fragment) if match else None,
                cpu_fallback_log_count=fragment.count('fallback to CPU') if match else None,
                observation_reason='PyTorch event counts are matched stderr entries; SDK entry count is stdout log lines, not actual codegen; disk hit/miss unknown',
                log_ref=str(stderr.relative_to(run))+f' :: VIT_CALL_BEGIN call_index={i}; '+str(stdout.relative_to(run))+' :: call_start/end'))
    return rows


def summarize(run,c,manifest,assets):
    rows=[]; timings=[]
    for case in manifest['workers']:
        if case['kind']!='fixed': continue
        r=load_result(run,case)
        raw=r.get('timings',[])
        timings+=raw
        repeats=[t for t in raw if t['phase']=='repeat']
        values=[t['elapsed_ms'] for t in repeats if valid(t)]
        completed=sum(t['status']=='ok' for t in repeats)
        mean=stats.mean(values) if values else None
        first=next((t['elapsed_ms'] for t in raw if t['phase']=='first' and valid(t)),None)
        all_done=r.get('status')=='ok' and len(values)==c['repeats'] and completed==c['repeats']
        structure=r.get('output_structure_valid') and r.get('dtype_valid')
        assessment,reason=warm_state(values)
        row=dict(run_id=c['run_id'],case_id=case['case_id'],mode=case['mode'],batch=case['batch'],trial=case['trial'],
            status=r.get('status',case['status']),reason=case.get('reason') or r.get('error'),
            model_id=c['model_id'],weights_id=c['weights_id'],weights_sha256=assets.get('weights_sha256'),
            input_hash=assets.get('batch_hashes',{}).get(str(case['batch'])),input_ids=assets.get('input_ids',[])[:case['batch']],
            dtype=c['dtype'],shape=[case['batch'],3,224,224],requested_repeats=c['repeats'],
            completed_repeats=completed,valid_repeats=len(values),first_call_ms=first,runner_setup_ms=r.get('runner_setup_ms'),
            mean_ms=mean,median_ms=stats.median(values) if values else None,
            stdev_ms=stats.stdev(values) if len(values)>=2 else None,
            throughput_images_s=case['batch']*len(values)/(sum(values)/1000) if all_done and structure else None,
            throughput_reason=None if all_done and structure else 'Not normally completed with requested valid repeats and valid output structure/dtype',
            first_call_extra_ms=first-mean if first is not None and mean is not None else None,
            compiler_time_ms=None,warm_state_assessment=assessment,warm_state_reason=reason,
            validation_status='structure_finite_ok_tolerance_unset' if structure else ('failed' if r.get('output_metadata') else 'not_run'),
            execution_path_status='unknown_all_op_placement_unverified',timing_scope=manifest['timing_scope'],
            cache_state=r.get('cache_state',c['cache_state']))
        # Diagnostics inform route risks but cannot prove every performance operation's placement.
        diag=run/'logs'/f'switch_{case["mode"]}.stderr.log'
        smoke=run/'logs'/f'smoke_{case["mode"]}_b{case["batch"]}.stderr.log'
        text='\n'.join(p.read_text(errors='replace') for p in [diag,smoke] if p.exists())
        row['execution_path_status']=path_status(text)
        rows.append(row)
    return rows,timings


def fmt(x):
    if x is None: return '미측정/미확인'
    if isinstance(x,float): return f'{x:.4f}'
    return str(x).replace('|','/').replace('\n',' ')


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join(
        '| '+' | '.join(fmt(x) for x in row)+' |\n' for row in rows)


def figures(run,summary,shape,c):
    good=[s for s in summary if s['mean_ms'] is not None or s['first_call_ms'] is not None]
    if not good:
        return dict(status='not_created',reason='No completed fixed performance measurements; no hypothetical charts')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,field,title in zip(axes,['first_call_ms','mean_ms'],['First call (one observation / trial)','Repeat mean ± sample SD (not CI)']):
            for i,s in enumerate(summary):
                if s[field] is None: continue
                ax.bar(i,s[field],yerr=s['stdev_ms'] if field=='mean_ms' else None,
                       color='#276b9d' if s['mode']=='eager' else '#db8c2b',capsize=4)
                ax.annotate(f'{s[field]:.2f}',(i,s[field]),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
            ax.set_xticks(range(len(summary)),[f'{s["mode"]} B{s["batch"]} T{s["trial"]}' for s in summary],rotation=30)
            if field=='mean_ms':
                ax.set_yscale('log'); title += ' — log y-axis'
            ax.set_ylabel('ms / batch call' + (' (log scale)' if field=='mean_ms' else '')); ax.set_title(title); ax.grid(axis='y',alpha=.2)
            ax.margins(y=.16)
        fig.suptitle(f'PRELIMINARY ViT BF16 RNGD; {c["trials"]} independent trial(s)/case; {c["repeats"]} repeats, {c["warmup"]} warmups')
        fig.text(.5,.01,'CPU dispatch + current-stream completion; run-local persistent cache shared; smoke may populate cache',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.92)); fig.savefig(run/'figures/01_fixed_batch.png',dpi=160); plt.close(fig)
        if any(valid(s) for s in shape):
            fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
            for ax,mode in zip(axes,['eager','compile']):
                calls=[s for s in shape if s['mode']==mode]
                ax.plot([s['call_index'] for s in calls],[s['elapsed_ms'] if valid(s) else float('nan') for s in calls],'o-')
                for s in calls:
                    if valid(s):
                        ax.annotate(f'B{s["batch"]}\n{s["elapsed_ms"]:.2f}',(s['call_index'],s['elapsed_ms']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
                        if s['backend_calls']:
                            ax.scatter([s['call_index']],[s['elapsed_ms']],marker='x',color='red',s=80)
                ax.set_yscale('log'); ax.margins(y=.30)
                ax.set_title(f'{mode}: detailed diagnostic logging'); ax.set_xlabel('call index'); ax.set_ylabel('ms / batch call (log scale)'); ax.set_xticks(range(1,7)); ax.grid(alpha=.2)
            fig.suptitle('PRELIMINARY shape switch [1,1,4,4,1,4]; one worker/mode, six requested calls/mode')
            fig.text(.5,.01,'Shared run-local persistent cache; x = observed backend entry (NOT actual codegen); no inferred cache hits',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.04,1,.92)); fig.savefig(run/'figures/02_shape_switch.png',dpi=160); plt.close(fig)
        return dict(status='created',files=[p.name for p in (run/'figures').glob('*.png')])
    except ImportError as e:
        plot_python=Path(__file__).parent/'.plot-venv/bin/python'
        if plot_python.exists() and sys.executable!=str(plot_python):
            import subprocess
            write_json(run/'figures/plot_data.json',dict(summary=summary,shape=shape,config=c))
            p=subprocess.run([str(plot_python),str(Path(__file__).parent/'plots.py'),str(run)],
                             capture_output=True,text=True,timeout=120)
            (run/'logs/plots.log').write_text(p.stdout+'\n'+p.stderr)
            if p.returncode==0:
                return read_json(run/'figures/plot_status.json')
            return dict(status='blocked',reason='Isolated plot process failed',log='logs/plots.log')
        return dict(status='blocked',reason=repr(e),note='Existing torch/SDK environment not modified; CSV/Markdown complete')


def report(run,c,m,assets,summaries,comparisons,outputs,shape,audit,figure_status):
    ok=sum(s['status']=='ok' for s in summaries)
    failures=[x for x in m['workers'] if x['status'] not in ['ok','not_run']]
    ready=ok==len(summaries) and all(s['warm_state_assessment']=='preliminary_stable' for s in summaries)
    usable=ok==len(summaries) and all(s['mean_ms'] is not None and s['throughput_images_s'] is not None for s in summaries)
    decision='조건부 진행' if usable else '문제 해결 우선'
    if ready and not assets.get('synthetic_input') and all(s['execution_path_status']=='npu_verified' for s in summaries):
        decision='본 실험 진행 가능'
    status_table=table(['mode','batch','실행 상태','runner setup ms','first ms','반복 mean ± SD ms','median ms','images/s','출력 상태'],[
        [s['mode'],s['batch'],s['status'],s['runner_setup_ms'],s['first_call_ms'],
         f'{fmt(s["mean_ms"])} ± {fmt(s["stdev_ms"])}',s['median_ms'],s['throughput_images_s'],
         '형태·유한값 정상 / 오차 기준 미정' if s['validation_status']=='structure_finite_ok_tolerance_unset' else s['validation_status']] for s in summaries])
    shape_table=table(['mode/호출','batch','ms','상태','graph 재컴파일/backend 진입','실제 compile / disk hit','로그'],[
        [f'{x["mode"]}/{x["call_index"]}',x['batch'],x['elapsed_ms'],x['status'],
         f'{fmt(x["pytorch_recompile_events"])}/{fmt(x["backend_calls"])}','unknown / unknown',
         f'[원본]({x["log_ref"].split(" :: ")[0]})'] for x in shape])
    output_table=table(['batch','입력/고유 표본','max / mean abs','cosine mean / min','Top-1 분자/분모','판정'],[
        [x['batch'],f'{x["input_kind"]}; 비교 {fmt(x["compared_images"])} / 전체 고유 {fmt(x["global_unique_samples"])}개',
         f'{x["max_abs_error"]:.6f}/{x["mean_abs_error"]:.6f}' if x['max_abs_error'] is not None else '미실시',
         f'{x["cosine_mean"]:.8f}/{x["cosine_min"]:.8f}' if x['cosine_mean'] is not None else '미실시/정의 불가',
         f'{fmt(x["top1_matches"])}/{fmt(x["compared_images"])} ({x["top1_ratio"]:.0%})' if x['top1_ratio'] is not None else '미실시',
         '허용 오차 미정' if x['status']=='structure_finite_ok_tolerance_unset' else x['status']] for x in outputs])
    pair_text='\n'.join(f'- B{x["batch"]}, trial {x["trial"]}: speedup={fmt(x["compile_speedup"])}; {x["hold_reason"] or "예비 비교"}' for x in comparisons)
    error_text='\n'.join(f'- {x["case_id"]}: {x["status"]}; [stderr]({x.get("stderr","logs/orchestrator.jsonl")})' for x in failures)
    measurements='; '.join(f'{s["mode"]} B{s["batch"]}: 첫 {fmt(s["first_call_ms"])} ms, 반복 {fmt(s["mean_ms"])}±{fmt(s["stdev_ms"])} ms, {fmt(s["throughput_images_s"])} images/s' for s in summaries)
    compile_shape=[x for x in shape if x['mode']=='compile']
    graph_events=sum(x['pytorch_recompile_events'] or 0 for x in compile_shape)
    backend_events=sum(x['backend_calls'] or 0 for x in compile_shape)
    observations=any(x['status']=='ok' for x in compile_shape)
    output_note='; '.join(f'B{x["batch"]}: max abs={fmt(x["max_abs_error"])}, Top-1 {fmt(x["top1_matches"])}/{fmt(x["compared_images"])}' for x in outputs)
    smoke_count=sum(x['status']=='ok' for x in m['workers'] if x['kind']=='smoke')
    stable_count=sum(x['warm_state_assessment']=='preliminary_stable' for x in summaries)
    tolerance_note='미정' if c['atol'] is None else f'CLI 지정 atol={c["atol"]}, rtol={c["rtol"]}; JSON 판정 참고'
    guard_note='B1→B4에서 tensor x의 index 0 크기 expected 1 / actual 4 guard 실패가 남았다.' if any(x['guard_failure_reason'] for x in compile_shape) else '추가 guard 실패 관측은 shape_switch.csv를 참고한다.'
    scope_note=f'고유 합성 입력 {assets.get("unique_samples")}개·허용 오차 {tolerance_note}·전체 연산 실행 장치 미확인·실제 codegen/disk hit 미관측' if assets.get('synthetic_input') else '출력 기준과 실행/캐시 관측 범위를 확인한다'
    intro=f'''# ViT RNGD Eager vs torch.compile 초기 점검

Run `{c['run_id']}`. 시간 원본은 UTC, 사용자 기준일은 KST(UTC+9). **예비 결과, 독립 trial={c['trials']}/케이스.**

1. **실행 가능성 [실행 확인]**: Eager/Compile B1·B4 smoke {smoke_count}/4, 고정 성능 케이스 {ok}/{len(summaries)} 완료. 정상 완료 케이스는 BF16 logits 형태·유한값과 공식 stream 완료 대기를 확인했다. 기본 장치 구성을 유지했다.
2. **첫 호출·반복 성능 [실행 확인]**: {measurements}. 잠정 안정 기준 충족 {stable_count}/{len(summaries)}; steady state 증명은 아니다. 아래 배속은 같은 batch의 반복 평균만 사용한 예비 수치다.
3. **재컴파일·캐시 [실행 확인/미확인]**: {'Compile 진단에서 PyTorch 재컴파일 '+str(graph_events)+'회, backend 진입 '+str(backend_events)+'회(초기 포함)를 관측했다. '+guard_note if observations else '진단 실행 결과 미확보.'} 실제 codegen·disk hit/miss는 **미확인**이다. SDK compile 진입 로그와 동일시하지 않는다. smoke와 전용 디스크 캐시를 공유한 후속 첫 호출은 완전 cold가 아니다.
4. **출력 [실행 확인/미확인]**: {output_note}. CPU FP32 비교이며 NaN/Inf·형태 상태는 원본 JSON에 기록했다. 허용 오차: {tolerance_note}. 전체 고유 입력 {fmt(assets.get('unique_samples'))}개, B1 첫 입력이 B4에도 포함된다. Top-1 일치율은 정답 accuracy가 아니다.

{status_table}

평균±표준편차는 유효 repeat만 사용하고 표본 표준편차(ddof=1)를 표시한다. 첫 호출은 1회 관측이다. 워밍업 {c['warmup']}회, 반복 {c['repeats']}회/worker. 실패·미실시 값은 0으로 채우지 않았다.

{pair_text}

{shape_table}

진단 시간에는 상세 로그 영향이 있다. 기본 성능 표에 합치지 않았다. 로그 행의 컴파일 진입은 실제 codegen 횟수와 다르며 프로세스 LRU hit/miss는 디스크 hit/miss가 아니다.

{output_table}

**진행 판단: {decision}.** {scope_note}. 순차 images/s는 서버 최대 처리량이 아니다. 출력 허용 오차 확정 여부와 후속 실험 확대 판단은 별개다.
'''
    issues=[]
    errors='\n'.join(load_result(run,x).get('error','') for x in failures)
    if failures:
        issues.append('현재 SDK에서 아래 실제 실패 연산/경로를 원본 torchvision BF16 실행 그대로 지원할 수 있는지 확인한다. dtype·Attention 구현·fallback 설정을 바꿔 우회하지 않는다.')
    if 'Unsupported' in errors or 'aten::' in errors:
        issues.append('피드백 질문 1: 기록된 미지원 연산/변환 경로에 대한 SDK 2026.4.0 지원 여부와 공식 해결 방법은 무엇인가?')
    if usable:
        issues.append('피드백 질문 1: 이 SDK에서 실제 codegen과 disk cache hit/miss를 구분하는 공식 관측 방법은 무엇인가?')
    issues.append('피드백 질문 2: BF16 Eager/Compile logits 비교의 atol/rtol 기준을 어떻게 합의할 것인가?')
    if usable:
        issues.append('피드백 질문 3: 기본 Eager CPU fallback=True에서 전체 연산의 NPU 실행 범위를 확인할 최소 공식 방법은 무엇인가?')
    follow='\n'.join('- '+x for x in issues[:4])
    feedback=intro+f'\n최소 조치: 실제 이미지 4장으로 일관성 점검, 출력 기준·연산 실행 경로·컴파일 관측 방법 확인 후 독립 trial 확대를 검토한다. B8·B16 및 대표 모듈 분석은 후속 단계이며 이번에 수행하지 않았다.\n\n첫 피드백:\n\n{follow}\n\n'+(f'실패 근거:\n\n{error_text}\n' if error_text else '')
    (run/'feedback_summary.md').write_text(feedback)
    environment=read_json(run/'environment.json')
    env_table=table(['항목','실제 설치/확인'],[[k,v['version'] if v else '미설치/metadata 없음'] for k,v in environment['packages'].items()])
    details=f'''
## 환경·실행·측정 경계

Python `{environment['executable']}`, `{environment['python']}`. OS `{environment['os']}`. 환경 및 장치 원문: [environment.json](environment.json), [종료 장치 상태](logs/device_after.json).

{env_table}

기존 공용 여부는 미확인으로 취급하고 기존 Python/torch/SDK/드라이버를 변경하지 않았다. torchvision은 import 및 패키지 metadata에서 torch==2.10.0 호환을 확인했다. 공식 [호환 표](https://github.com/pytorch/vision)와 [CPU 배포 안내](https://pytorch.org/get-started/previous-versions/)도 확인했다. 추가 설치가 없어 dry-run은 해당하지 않는다.

지침: /AGENTS.md, /root/AGENTS.md, /root/ViT/AGENTS.md 및 저장소 하위 AGENTS.md 없음. 기존 preflight/를 읽고 공식 sync/cache API와 의미를 재사용했다. 기존 preflight의 덧셈·batch2·DEBUG 포함 시간과 본 실험을 합치지 않았다. 이번 공통 timer는 query 계측을 timer에서 제외했다. 기존 파일은 변경하지 않았다.

모델·입력: [assets.json](assets.json), [config.json](config.json). 공식 가중치는 표준 torch hub 경로에 다운로드하고 URL SHA256 prefix 및 전체 파일 SHA256을 보존했다. weights=None 대체는 없다. 실제 이미지가 저장소에서 발견되지 않아 CPU FP32 seed=42 정규분포 텐서 4개를 한 번 저장하고 BF16으로 변환했다. batch별·sample별 BF16 바이트 해시를 검증한다. B1은 첫 샘플, B4는 전체 4개다. 224×224 / patch16의 내부 토큰은 이미지당 197개이며 B4를 토큰 길이 788로 표현하지 않는다.

준비에는 가중치 로드, BF16 변환, NPU 전송 및 준비 stream 완료 대기가 포함된다. 준비 중 사전 forward는 없다. runner_setup_ms는 Eager runner 참조 준비 또는 torch.compile 래퍼 생성만 별도 기록한다. 시간은 prewait → perf_counter_ns → runner(inputs) → current_stream.synchronize → perf_counter_ns 순서다. CPU 디스패치와 완료 대기를 포함하고 kernel-only 시간이 아니다. 출력 CPU 복사·파일 쓰기·print·오차 계산은 timer 밖이다. 동일 eval/inference_mode를 모든 forward에 적용한다. 기본 dynamic/fullgraph/mode/재컴파일 제한을 변경하지 않는다. compiler_time_ms는 독립 SDK 구간 미확인으로 null이다. first_call_extra_ms는 순수 compile 시간으로 부르지 않는다.

SDK 기본 Eager CPU fallback={environment.get('furiosa',{}).get('cpu_fallback_default')}; 이 값을 새로 켜거나 오류 억제를 켜지 않았다. Compile backend의 eager_fallback=False와 별개다. 기본을 보존했으므로 성공한 output.device만으로 NPU 전체 실행을 주장하지 않는다. 진단의 fallback/graph 분할 로그와 실행 경로 확인 상태를 별도 남긴다.

동기화 근거: 설치 furiosa/torch/config/_stream.py의 current_stream, _C/stream/__init__.pyi의 Stream.synchronize/query; backend 근거: _backend/torch_compile.py의 backend=_Backend(), 기본 eager_fallback=False. 설치 사본과 줄 번호·해시: [installed_sources.txt](logs/installed_sources.txt), environment.json.sources. 준비 동기화 후 query=True를 실행으로 점검하며 완료 API 실패 시 시간 측정으로 진행하지 않는다. 기본 단일 current stream에 대한 계약이며 다른 스트림을 임의 생성하지 않는다.

캐시: 확인된 SDK cache_context(root_dir, summary=False)로 run/compiler-cache를 사용했다. 실행 전 경로는 없었고 worker들이 순차 공유한다. smoke가 후속 성능 캐시를 채웠을 수 있다. fresh_process_persistent_cache_possible은 worker 내부 LRU가 새로 시작함을 뜻할 뿐 disk cache hit를 증명하지 않는다. 공유 캐시 삭제는 없다. 시스템의 다른 캐시는 cold라고 주장하지 않는다.

실행 한도: worker 기본 1800초, 전체 최대 7200초(본 CLI는 자산 준비부터 계산해 더 보수적이다). 운영상 상한이며 예상 compile 시간이 아니다. timeout은 미지원 확정이 아니다. worker는 새 세션으로 생성하고 이 실행의 process group만 TERM/KILL한다. worker 전 device ps/status를 다시 기록하며 다른 PID가 보이면 실행하지 않는다. 장치 ps의 순간 스냅샷 사이에 외부 작업이 생기는 경쟁 가능성은 완전히 제거하지 못한다. utilization '-'는 0%로 추정하지 않았다.

## 진단 이벤트 해석

TORCH_LOGS 항목은 설치 torch registry에서 확인한 뒤 worker 시작 전 설정한다. FURIOSA_TORCH_LOG=debug는 설치 config/_env.py 근거를 사용한다. Python furiosa_torch DEBUG는 smoke/shape 진단에만 적용한다. 성능 worker는 warning 로그를 사용하므로 추가 컴파일 이벤트가 없었다고 단정하지 않는다. shape 진단 backend wrapper는 전달받은 gm/inputs를 SDK 기본 ft.backend로 그대로 전달한다. backend 호출의 집계 단위는 Dynamo graph이고 SDK 실제 compiler count가 아니다. stderr의 VIT_CALL_BEGIN/END와 stdout events.jsonl로 요청 귀속을 남긴다. 재컴파일 로그 행·guard 이유, backend entry, SDK compile-entry 로그, 실제 codegen, disk hit/miss를 각각 분리한다. 초기 backend entry와 후속 entry도 구분한다. 시간 증가/감소만으로 컴파일/cache hit를 판정하지 않는다.

## 실패 및 미실시

{error_text or '실패 worker 없음; 모든 상태는 manifest.json을 참고한다.'}

{table(['case','상태','이유'],[[x['case_id'],x['status'],x.get('reason')] for x in m['workers'] if x['status']!='ok'])}

정확한 실행 명령과 worker PID·시작/종료 시각: [manifest.json](manifest.json). 오류 traceback은 cases/*/result.json과 logs/*stderr.log에 원형 보존했다.

## 집계·그림 검증

{json.dumps(audit,ensure_ascii=False,indent=2)}

그림 상태: {json.dumps(figure_status,ensure_ascii=False)}. 값이 없으면 0으로 그리지 않는다. 정상 성능 데이터가 없으면 가상의 차트를 만들지 않는다. 원본 [timings.csv](timings.csv), [summary.csv](summary.csv), [comparisons.csv](comparisons.csv), [shape_switch.csv](shape_switch.csv), [output_comparison.json](output_comparison.json).

warm_state는 repeat CV≤0.10 및 앞/뒤 절반 평균 상대차≤0.10을 잠정 안정 기준으로 사용한다. 10회 미만이면 unknown이다. 이는 steady state 증명이나 상세 compiler 이벤트 부재 증명이 아니다. 느린 정상 호출을 이상치로 제거하지 않는다. 평균/처리량은 원시 유효 반복에서 재계산한다. 반복 미완료나 출력 구조 실패의 throughput 및 배속은 null이다. 반복간 변동과 trial간 변동은 별개다. 단일 trial 20회로 P95/P99, 일반적 우열, 원인, 손익분기점, 최적 batch를 주장하지 않는다.

## 재현과 다음 단계

저장소 루트에서 `{sys.executable} -u vit_pilot/pilot.py run`을 실행하면 새 고유 run_id를 만든다. 재집계: `{sys.executable} vit_pilot/pilot.py analyze --run-dir {run}`. 설치 환경을 바꾸지 않으며 동일 가중치/입력·해시를 확인한다. CLI --worker-timeout, --total-budget, --warmup, --repeats, --trials, --device, --images, --atol/--rtol 지원. 독립 trial 확대와 B8·B16은 장애 해결 및 피드백 후, 그 뒤 대표 Encoder/Attention/MLP 분석을 수행한다. 이번 실행에서는 범위 확대하지 않았다.
'''
    details += '\n장치 범위 [실행 확인/문서·소스 확인]: 기본 Fusion(num_pe_per_cluster=1), MeshKind.Single을 유지했다. 실행 중 smi의 worker는 npu1:0으로 표시됐다. 전체 물리 칩의 최대 처리량을 측정한 결과가 아니다. 타이머에서 제외한 전송은 명시적인 사전 입력/모델 전송과 사후 출력 CPU 복사다. 모델 내부의 DMA/프로그램 로드/장치 전환 비용은 호출 구간에 발생하면 포함된다.\n'
    if observations:
        details += f'\n진단 관측 [실행 확인]: Compile backend 진입 {backend_events}회(초기 포함), PyTorch graph 재컴파일 로그 {graph_events}회. SDK split 로그는 각 backend 진입 시 1개 segment를 기록했다. 진단에서 CPU fallback 및 graph break 로그는 관측되지 않았으나, 전체 연산 배치의 NPU 전용 실행 증명은 아니다. Eager process LRU와 SDK compile-entry 로그 집계는 shape_switch.csv의 별도 필드이며 실제 codegen/disk cache hit와 구분한다.\n'
    if (run/'figures/01_fixed_batch.png').exists():
        details += '\n그림 축: 01의 first-call 패널은 선형, repeat 패널은 로그 y축이다. 02는 두 모드 모두 같은 로그 y축을 쓴다. 각 그림에 축 종류·단위·관측값을 명시했다. 오차막대는 repeat 표본 표준편차이며 신뢰구간이 아니다.\n'
    details += '\n코드 근거: manifest.json의 초기 해시 및 worker별 해시, consistency_checks.json의 분석 시점 해시를 참고한다. code/ 사본과 provenance가 있으면 실행 중 완성한 진단 로그 표식/재현 CLI의 버전 차이를 함께 확인할 수 있다. 모델·입력·측정 timer를 바꾸어 성능을 유도하지 않았다.\n'
    (run/'report.md').write_text(intro+details+'\n'+follow+f'\n\n최종 진행 판단: **{decision}** [실행 확인/미확인]. {scope_note}. 최소 조치는 실제 이미지 4장 확보, BF16 출력 기준 합의 및 전체 연산 실행/컴파일 관측 방법 확인이다. 이후 독립 trial 확대 → B8·B16 → 대표 Encoder·Attention·MLP 분석 순서를 검토하며 이번에는 추가 실행하지 않았다.\n')


def analyze(run):
    run=Path(run).resolve()
    c=read_json(run/'config.json'); m=read_json(run/'manifest.json')
    assets=read_json(run/'assets.json') if (run/'assets.json').exists() else {}
    summaries,timings=summarize(run,c,m,assets)
    shape=diagnostic_rows(run,m,c)
    pairs=[]; outputs=[]
    for b in [1,4]:
        for trial in range(1,c['trials']+1):
            e=next(s for s in summaries if s['batch']==b and s['trial']==trial and s['mode']=='eager')
            k=next(s for s in summaries if s['batch']==b and s['trial']==trial and s['mode']=='compile')
            out=compare_outputs(run,e,k,assets,c); outputs.append(out)
            e['validation_status']=out['status']; k['validation_status']=out['status']
            normal=all(x['status']=='ok' and x['valid_repeats']==c['repeats'] and x['mean_ms'] is not None for x in [e,k])
            paired=normal and out['status'] in ['structure_finite_ok_tolerance_unset','passed_explicit_tolerance']
            reasons=[]
            if not normal: reasons.append('Paired cases not both normally completed with valid timing')
            if out['status'] in ['not_compared','failed']: reasons.append(out.get('reason','Output comparison unavailable/failed'))
            provisional=out['tolerance_status']=='unset' or any(x['execution_path_status']!='npu_verified' for x in [e,k])
            if provisional and paired: reasons.append('Preliminary only: output tolerance or all-op execution path unverified')
            pairs.append(dict(run_id=c['run_id'],batch=b,trial=trial,eager_case_id=e['case_id'],compile_case_id=k['case_id'],
                compile_speedup=e['mean_ms']/k['mean_ms'] if paired else None,comparable=paired,provisional=provisional,
                hold_reason='; '.join(reasons) or None,output_validation_status=out['status']))
    write_csv(run/'timings.csv',timings,TIMING_FIELDS)
    write_csv(run/'summary.csv',summaries,SUMMARY_FIELDS)
    write_csv(run/'comparisons.csv',pairs,COMPARE_FIELDS)
    write_csv(run/'shape_switch.csv',shape,SHAPE_FIELDS)
    write_json(run/'output_comparison.json',dict(run_id=c['run_id'],global_unique_samples=assets.get('unique_samples'),
        note='B1 reuses the first B4 input; do not count as five independent samples',comparisons=outputs))
    count_ok=sum(valid(t) for t in timings)
    all_fixed=all(s['status']=='ok' for s in summaries)
    all_shape=all(x['status']=='ok' for x in shape)
    checks=[]
    for s in summaries:
        vals=[t['elapsed_ms'] for t in timings if t['case_id']==s['case_id'] and t['phase']=='repeat' and valid(t)]
        checks.append(dict(case_id=s['case_id'],valid_repeats_match=len(vals)==s['valid_repeats'],
            mean_matches=(s['mean_ms']==stats.mean(vals) if vals else s['mean_ms'] is None),
            throughput_matches=(math.isclose(s['throughput_images_s'],s['batch']*len(vals)/(sum(vals)/1000),rel_tol=1e-12)
                                if s['throughput_images_s'] is not None else None)))
    # Read exported CSV back so unit/count audits also cover serialization.
    with (run/'timings.csv').open() as f: csv_count=sum(1 for _ in csv.DictReader(f))
    audit=dict(utc=now(),expected_fixed_records=c['expected_fixed_records'],actual_fixed_records=len(timings),
        completed_valid_fixed_records=count_ok,fixed_count_check='pass' if all_fixed and count_ok==c['expected_fixed_records'] else 'incomplete',
        expected_shape_calls=c['expected_shape_records'],shape_status_rows=len(shape),completed_shape_calls=sum(valid(x) for x in shape),
        shape_count_check='pass' if all_shape and sum(valid(x) for x in shape)==12 else 'incomplete',
        csv_row_count_matches=csv_count==len(timings),recalculation=checks,
        script_sha256_at_analysis={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
    write_json(run/'consistency_checks.json',audit)
    fs=figures(run,summaries,shape,c); write_json(run/'figures/status.json',fs)
    report(run,c,m,assets,summaries,pairs,outputs,shape,audit,fs)
    print(json.dumps(dict(run_dir=str(run),fixed_valid_records=count_ok,shape_valid_records=sum(valid(x) for x in shape),figures=fs),ensure_ascii=False))


if __name__=='__main__':
    analyze(Path(sys.argv[1]))
