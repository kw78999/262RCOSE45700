"""Strict fastpath-off ViT comparison, resumable serial workers, bounded budget."""
import argparse
from datetime import datetime, timezone
import importlib
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from common import Events, now, read_json, write_json, sha256, tensor_hash
from fastpath_support import CACHE_POLICY, find_case, result_for, compare_pair, state_hash

HERE=Path(__file__).resolve().parent
MEASUREMENT_SOURCES=['fastpath_compare.py','fastpath_compare_worker.py','fastpath_support.py','fastpath_math.py',
                     'common.py','vit.py','environment.py','attention_analysis.py']
STAGES=['diagnostic','performance','cumulative']


def fresh(a):
    from environment import collect
    run=HERE/'fastpath_compare_runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+os.urandom(3).hex())
    run.mkdir(parents=True,exist_ok=False)
    for name in ['logs','cases','outputs','figures','code']: (run/name).mkdir()
    limits=dict(worker_timeout_s=a.worker_timeout or 1800,total_budget_s=a.total_budget or 7200,
                warmup=5 if a.warmup is None else a.warmup,repeats=a.repeats or 50,trials=a.trials or 3,
                max_cumulative_calls=a.max_cumulative_calls or 10000)
    cfg=dict(run_id=run.name,assets_dir=str(a.assets_dir.resolve()),device=a.device,
        dtype='torch.bfloat16',mha_fastpath=False,cpu_fallback=False,cache_policy=CACHE_POLICY,
        atol=a.atol,rtol=a.rtol,tolerance_source=a.tolerance_source if a.atol is not None else None,
        numeric_rule=__import__('fastpath_math').NUMERIC_RULE,
        stability_rule='reuse analyze.warm_state: >=10 repeats, CV<=0.10 and half-mean drift<=0.10; heuristic',
        uncertainty_rule='abs(mean E-C) <= sqrt(SD_E^2+SD_C^2), descriptive only; never changes exact sign',
        cumulative_target_rule='ceil(1.2 * median paired N_win); N_win=1 -> at least20; cap/budget may shorten',
        cumulative_wall_bound_rule='1.5*max(new-trial prepare+A+N*t)/1000 +60 seconds +N*0.001 seconds per mode',
        **limits)
    write_json(run/'config.json',cfg)
    workers=[]
    def add(stage,mode,b,trial=1):
        cid=f'{stage}_{mode}_b{b}_t{trial}'
        workers.append(dict(case_id=cid,root_case_id=cid,stage=stage,mode=mode,batch=b,trial=trial,
            status='not_planned' if stage=='cumulative' else 'not_run',target_calls=None,
            cache_path=str(run/'caches'/stage/cid)))
    for b in [1,4]:
        for mode in ['eager','compile']:add('diagnostic',mode,b)
    orders=[[('eager',1),('compile',1),('compile',4),('eager',4)],
            [('compile',1),('eager',1),('eager',4),('compile',4)],
            [('eager',4),('compile',4),('compile',1),('eager',1)]]
    for trial in range(1,cfg['trials']+1):
        for mode,b in orders[(trial-1)%3]:add('performance',mode,b,trial)
    for b in [1,4]:
        for mode in ['eager','compile']:add('cumulative',mode,b)
    sources={name:sha256(HERE/name) for name in MEASUREMENT_SOURCES}
    m=dict(run_id=run.name,created_utc=now(),command=[sys.executable,*sys.argv],status='preparing',workers=workers,
        npu_budget_used_s=0,execution_order=[],resume_history=[],code_sha256=sources,
        sdk_actual_codegen_count=None,disk_cache_hit=None,disk_cache_miss=None,
        expected_performance_rows=4*cfg['trials']*(1+cfg['warmup']+cfg['repeats']))
    for name in MEASUREMENT_SOURCES:shutil.copyfile(HERE/name,run/'code'/name)
    # Protect original project sources and previous reports/inputs without touching caches.
    protected=[p for p in HERE.glob('*.py') if p.name not in MEASUREMENT_SOURCES and not p.name.startswith('fastpath_')]
    protected += [HERE/n for n in ['common.py','vit.py','README.md','FOLLOWUP_README.md','ATTENTION_README.md']]
    previous=[HERE/'attention_runs/20261104T153417Z-33d046',HERE/'attention_runs/20261104T160225Z-35e28e',Path(cfg['assets_dir'])]
    for root in previous:
        protected += [root/n for n in ['assets.json','manifest.json','timings.csv','summary.csv','report.md','feedback_summary.md'] if (root/n).exists()]
    m['protected_files_sha256']={str(p):sha256(p) for p in protected if p.exists()}
    write_json(run/'manifest.json',m)
    collect(run)
    # Mandatory math fixtures execute before any NPU worker.
    tests=[sys.executable,'-m','unittest','discover','-s',str(HERE/'tests/fastpath_compare'),'-p','test_math.py','-v']
    tested=subprocess.run(tests,capture_output=True,text=True)
    (run/'logs/math_tests.log').write_text(tested.stdout+tested.stderr)
    if tested.returncode:raise RuntimeError('Math fixture tests failed; no NPU worker launched')
    import torch
    import furiosa.torch as ft
    assets=read_json(Path(cfg['assets_dir'])/'assets.json')
    if assets['synthetic_input'] or assets['unique_samples']!=4:raise RuntimeError('Four existing real-image tensors required')
    if sha256(assets['weights_file'])!=assets['weights_sha256']:raise RuntimeError('Checkpoint changed')
    if sha256(Path(cfg['assets_dir'])/'inputs.pt')!=assets['input_file_sha256']:raise RuntimeError('Input file changed')
    x=torch.load(Path(cfg['assets_dir'])/'inputs.pt',map_location='cpu',weights_only=True)
    if list(x.shape)!=[4,3,224,224] or x.dtype!=torch.bfloat16:raise RuntimeError('Stored BF16 tensor shape/dtype mismatch')
    if any(tensor_hash(x[:b])!=assets['batch_hashes'][str(b)] for b in [1,4]):raise RuntimeError('Input tensor hash mismatch')
    image_manifest=read_json(Path(cfg['assets_dir'])/'image_manifest.json')
    for image in image_manifest['images']:
        if sha256(image['actual_absolute_path'])!=image['file_sha256']:raise RuntimeError('Source image changed')
    state=torch.load(assets['weights_file'],map_location='cpu',weights_only=True)
    raw_checkpoint_hash=state_hash(state)
    from torchvision.models import vit_b_16,ViT_B_16_Weights
    reference=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval()
    original_hash=state_hash(reference.state_dict())
    bf16_hash=state_hash(reference.to(torch.bfloat16).state_dict())
    write_json(run/'asset_verification.json',dict(status='pass',input_file_sha256=assets['input_file_sha256'],
        weights_file_sha256=assets['weights_sha256'],state_dict_raw_checkpoint_fp32_sha256=raw_checkpoint_hash,
        state_dict_official_fp32_cpu_sha256=original_hash,
        state_dict_bf16_cpu_sha256=bf16_hash,state_entry_count=len(state),
        state_hash_scope='sorted keys; length-prefixed JSON[name,dtype,shape] and contiguous CPU tensor bytes; official torchvision-loaded FP32/BF16 CPU model before NPU transfer; loader remaps legacy MLP keys',
        state_dict_after_device_hash=None,after_device_reason='Not copied back before first call; device/dtype/sync metadata checked instead',
        batch_hashes=assets['batch_hashes'],math_tests_command=tests,math_tests_passed=True))
    del state,x,reference
    for name in ['assets.json','image_manifest.json']:shutil.copyfile(Path(cfg['assets_dir'])/name,run/name)
    sdk=Path(ft.__file__).parent; tr=Path(torch.__file__).parent
    extra=[tr/'backends/mha/__init__.py',tr/'nn/modules/activation.py',sdk/'_C/profiler/__init__.pyi']
    with (run/'logs/control_installed_sources.txt').open('w') as f:
        for path in extra:
            f.write(f'\n### {path}\n');f.writelines(f'{i}: {line}\n' for i,line in enumerate(path.read_text().splitlines(),1))
    m['control_source_sha256']={str(p):sha256(p) for p in extra}
    write_json(run/'logs/native_mha_dispatch.json',dict(privateuse1_kernel=torch._C._dispatch_has_kernel_for_dispatch_key('aten::_native_multi_head_attention','PrivateUse1'),
        table=torch._C._dispatch_dump_table('aten::_native_multi_head_attention')))
    m['status']='ready';write_json(run/'manifest.json',m)
    return run


def stop_own(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def launch(run,cfg,m,w,ev):
    from environment import command
    check=command(['furiosa-smi','ps']);write_json(run/'logs'/f"{w['case_id']}.device_before.json",check)
    if check.get('returncode')!=0 or any(line.strip().startswith('|') and line.strip().split('|')[1].strip().isdigit() for line in check.get('stdout','').splitlines()):
        w.update(status='held',reason='NPU use inspection failed or another NPU process is active');return
    if {name:sha256(HERE/name) for name in MEASUREMENT_SOURCES}!=m['code_sha256']:raise RuntimeError('Measurement code changed; preserve run and use an explicit new attempt/version')
    env=os.environ.copy()
    for key in ['TORCH_LOGS','FURIOSA_TORCH_LOG','FURIOSA_TORCH_PROFILE']:env.pop(key,None)
    if w['stage']=='diagnostic':
        if not all(read_json(run/'environment.json').get('torch_logs_supported',{}).values()):raise RuntimeError('Requested logging keys not verified')
        env.update(TORCH_LOGS='recompiles,graph_breaks',FURIOSA_TORCH_LOG='debug',FURIOSA_TORCH_PROFILE='trace')
    limit=min(cfg['worker_timeout_s'],cfg['total_budget_s']-m['npu_budget_used_s'])
    if limit<=0:w.update(status='held',reason='Overall NPU budget exhausted');return
    cmd=[sys.executable,'-u',str(HERE/'fastpath_compare_worker.py'),'--run',str(run),'--case',w['case_id']]
    w.update(status='running',started_utc=now(),command=cmd,limit_s=limit,wall_elapsed_s=0)
    m['execution_order'].append(w['case_id']);write_json(run/'manifest.json',m)
    ev.emit('worker_start',case_id=w['case_id'],limit_s=limit,budget_remaining_s=cfg['total_budget_s']-m['npu_budget_used_s'])
    proc=None;started=time.perf_counter();accounted=started
    try:
        with (run/'logs'/f"{w['case_id']}.stdout.log").open('w') as out,(run/'logs'/f"{w['case_id']}.stderr.log").open('w') as err:
            proc=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,start_new_session=True)
            w['pid']=proc.pid;write_json(run/'manifest.json',m)
            while True:
                remaining=limit-(time.perf_counter()-started)
                if remaining<=0:
                    stop_own(proc);w.update(status='timeout',reason='Operational time limit; not an unsupported-operation verdict');break
                try:
                    code=proc.wait(timeout=min(30,remaining))
                    rr=result_for(run,w);w.update(returncode=code,status=rr.get('status','error') if code==0 else 'error')
                    break
                except subprocess.TimeoutExpired:
                    tick=time.perf_counter();m['npu_budget_used_s']+=tick-accounted;accounted=tick
                    w['wall_elapsed_s']=tick-started;write_json(run/'manifest.json',m)
                    ev.emit('worker_progress',case_id=w['case_id'],wall_elapsed_s=w['wall_elapsed_s'],npu_budget_used_s=m['npu_budget_used_s'])
    except BaseException:
        if proc is not None:stop_own(proc)
        w.update(status='interrupted',reason='Own orchestrator interrupted; completed case files retained');raise
    finally:
        tick=time.perf_counter();m['npu_budget_used_s']+=tick-accounted
        w.update(ended_utc=now(),wall_elapsed_s=tick-started)
        rr=result_for(run,w)
        for key in ['input_hash','weights_sha256','state_dict_bf16_cpu_sha256','timing_scope','cache_state','control_snapshots']:
            if key in rr:w[key]=rr[key]
        write_json(run/'manifest.json',m);ev.emit('worker_end',case_id=w['case_id'],status=w['status'],wall_elapsed_s=w['wall_elapsed_s'],npu_budget_used_s=m['npu_budget_used_s'])


def advance(run,stop_after):
    cfg,m=read_json(run/'config.json'),read_json(run/'manifest.json');ev=Events(run/'logs/orchestrator.jsonl')
    m['status']='running';write_json(run/'manifest.json',m)
    try:
        for stage in STAGES[:STAGES.index(stop_after)+1]:
            if stage=='cumulative':
                from fastpath_compare_analysis import analyze,plan_cumulative
                analyze(run);plan_cumulative(run)
                m=read_json(run/'manifest.json')
            candidates=[w for w in m['workers'] if w['stage']==stage and w is find_case(m,stage,w['mode'],w['batch'],w['trial'])]
            if stage=='diagnostic':candidates.sort(key=lambda w:(w['batch'],w['mode']=='compile'))
            for w in candidates:
                if w['status'] not in ['not_run','not_planned','held'] or (w['status']=='held' and w.get('started_utc')):continue
                reason=None
                if stage=='diagnostic' and w['batch']==4:
                    b1=find_case(m,'diagnostic',w['mode'],1)
                    if b1['status']!='ok':reason='Corresponding B1 diagnostic did not succeed'
                    b1pair=compare_pair(run,'diagnostic',1)
                    if all(find_case(m,'diagnostic',mode,1)['status']=='ok' for mode in ['eager','compile']) and not b1pair['eligible']:
                        reason='B1 paired output/controls failed validation'
                if stage in ['performance','cumulative']:
                    pair=compare_pair(run,'diagnostic',w['batch'])
                    if not pair['eligible']:reason='Diagnostic pair held: '+str(pair.get('reason'))
                    for mode in ['eager','compile']:
                        d=result_for(run,find_case(m,'diagnostic',mode,w['batch']));obs=d.get('path_observations',{})
                        if not obs.get('npu_task_events') or obs.get('native_mha_host_count'):
                            reason='Required diagnostic NPU/native-MHA evidence unavailable'
                if stage=='cumulative' and not w.get('target_calls'):reason=w.get('reason','No validated finite target within budget')
                if m['npu_budget_used_s']>=cfg['total_budget_s']:reason='Overall NPU execution budget exhausted'
                if reason:
                    w.setdefault('hold_history',[]).append(dict(utc=now(),reason=reason));w.update(status='held',reason=reason)
                    write_json(run/'manifest.json',m);continue
                launch(run,cfg,m,w,ev);write_json(run/'manifest.json',m)
            # Save paired outputs as each stage completes; performance/cumulative stay separate.
            pairs=[compare_pair(run,stage,b,t) for b in [1,4] for t in (range(1,cfg['trials']+1) if stage=='performance' else [1])]
            write_json(run/f'{stage}_output_comparison.json',dict(unique_images=4,synthetic_input=False,accuracy_computed=False,pairs=pairs))
        m['status']='finished' if stop_after=='cumulative' else 'awaiting_resume'
    finally:
        m['last_update_utc']=now();m['code_sha256_after']={name:sha256(HERE/name) for name in MEASUREMENT_SOURCES}
        m['all_workers_complete']=all(w['status']=='ok' for w in m['workers'])
        m['all_unheld_workers_complete']=all(w['status']=='ok' for w in m['workers'] if w['status']!='held')
        write_json(run/'manifest.json',m)
        try:
            from fastpath_compare_analysis import analyze
            analyze(run)
        except ImportError as e:
            ev.emit('analysis_pending',reason=str(e))
        from environment import command
        write_json(run/'logs/device_after.json',command(['furiosa-smi','ps']))
        ev.emit('run_stage_end',status=m['status'],stop_after=stop_after,npu_budget_used_s=m['npu_budget_used_s'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','resume','analyze'])
    p.add_argument('--run-dir',type=Path);p.add_argument('--assets-dir',type=Path,default=HERE/'followup_runs/20261104T132538Z-c031a6')
    p.add_argument('--device',default='furiosa:0');p.add_argument('--stop-after',choices=STAGES,default='cumulative')
    p.add_argument('--worker-timeout',type=float);p.add_argument('--total-budget',type=float)
    p.add_argument('--warmup',type=int);p.add_argument('--repeats',type=int);p.add_argument('--trials',type=int)
    p.add_argument('--max-cumulative-calls',type=int);p.add_argument('--atol',type=float);p.add_argument('--rtol',type=float)
    p.add_argument('--tolerance-source');p.add_argument('--retry-case',action='append',default=[]);p.add_argument('--retry-reason')
    a=p.parse_args()
    for name in ['worker_timeout','total_budget','repeats','trials','max_cumulative_calls']:
        value=getattr(a,name)
        if value is not None and (not math.isfinite(value) or value<=0):p.error(name+' must be positive finite')
    if a.warmup is not None and a.warmup<0:p.error('warmup must be nonnegative')
    if (a.atol is None)!=(a.rtol is None) or any(v is not None and (not math.isfinite(v) or v<0) for v in [a.atol,a.rtol]):p.error('Supply both nonnegative finite atol/rtol or neither')
    if a.atol is not None and not a.tolerance_source:p.error('--tolerance-source required for explicit agreed tolerances')
    if a.action=='run':run=fresh(a)
    else:
        if not a.run_dir:p.error('--run-dir required')
        run=a.run_dir.resolve()
    if a.action=='analyze':
        from fastpath_compare_analysis import analyze
        analyze(run);return
    if a.action=='resume':
        m,cfg=read_json(run/'manifest.json'),read_json(run/'config.json')
        for w in m['workers']:
            if w['status']=='running':
                proc=Path(f"/proc/{w.get('pid')}/cmdline")
                if proc.exists() and str(run) in proc.read_text():raise RuntimeError('Own worker still active; do not launch another')
                w.update(status='interrupted',reason='Previously running worker no longer alive; preserved for explicit retry')
        for opt,key in [('total_budget','total_budget_s'),('worker_timeout','worker_timeout_s')]:
            if getattr(a,opt) is not None:cfg[key]=getattr(a,opt)
        if a.retry_case and not a.retry_reason:p.error('--retry-reason required')
        current_sources={name:sha256(HERE/name) for name in MEASUREMENT_SOURCES}
        if current_sources!=m['code_sha256']:
            if not a.retry_case:raise RuntimeError('Code revision requires explicit preserved retry cases/reason')
            revision=len(m.get('code_revision_history',[]))+1
            m.setdefault('code_revision_history',[]).append(dict(utc=now(),reason=a.retry_reason,
                before=m['code_sha256'],after=current_sources,revision=revision))
            archive=run/'code'/f'revision_{revision}';archive.mkdir(exist_ok=False)
            for name in MEASUREMENT_SOURCES:shutil.copyfile(HERE/name,archive/name)
            m['code_sha256']=current_sources
        for cid in a.retry_case:
            old=next(w for w in m['workers'] if w['case_id']==cid)
            if old['status']=='ok':raise RuntimeError('Successful cases are preserved, never retried in place')
            suffix=1
            while any(w['case_id']==old['root_case_id']+f'_attempt{suffix}' for w in m['workers']):suffix+=1
            fresh_id=old['root_case_id']+f'_attempt{suffix}'
            w={k:old[k] for k in ['root_case_id','stage','mode','batch','trial','target_calls']}
            w.update(case_id=fresh_id,status='not_run',retry_reason=a.retry_reason,cache_path=str(run/'caches'/w['stage']/fresh_id))
            # Preserve requested order within each stage; dependency lookups use latest attempts.
            m['workers'].append(w)
        m['resume_history'].append(dict(utc=now(),command=[sys.executable,*sys.argv],budget_used_before=m['npu_budget_used_s']))
        write_json(run/'config.json',cfg);write_json(run/'manifest.json',m)
    print('FASTPATH_COMPARE_RUN '+str(run),flush=True)
    advance(run,a.stop_after)


if __name__=='__main__':main()
