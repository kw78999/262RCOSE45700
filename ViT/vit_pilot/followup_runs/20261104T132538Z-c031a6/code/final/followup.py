#!/usr/bin/env python3
"""Diagnostic -> real output validation -> gated independent trials; no shared-cache deletion."""
import argparse
from datetime import datetime,timezone
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from common import Events,now,read_json,write_json,sha256,tensor_hash

HERE=Path(__file__).resolve().parent
ORDER=['01_cat.png','02_coffee.png','03_astronaut.png','04_rocket.jpg']


def prepare_real(root,images,dataset):
    import torch
    import torchvision
    from PIL import Image
    import PIL
    from vit import prepare_assets
    supplied=read_json(dataset/'image_manifest.json')
    if [Path(x['path']).name for x in supplied['images']]!=ORDER:
        raise ValueError('Provided manifest order mismatch')
    if [Path(x).resolve() for x in images]!=[(dataset/'images'/n).resolve() for n in ORDER]:
        raise ValueError('--images must be the four supplied files in fixed manifest order')
    for info,p in zip(supplied['images'],images):
        if sha256(p)!=info['sha256']: raise ValueError('Image file hash mismatch: '+p)
    shutil.copyfile(dataset/'image_manifest.json',root/'inputs/provided_image_manifest.json')
    for name in ['README.md','SOURCES.md','verification.log','deployment.json']:
        if (dataset/name).exists(): shutil.copyfile(dataset/name,root/'inputs'/name)
    assets=prepare_assets(root,images)  # Exact enum transforms, exactly once, CPU only.
    fp32=torch.load(root/'inputs.pt',map_location='cpu',weights_only=True)
    (root/'inputs.pt').rename(root/'inputs/preprocessed_fp32.pt')
    bf16=fp32.to(torch.bfloat16)
    torch.save(bf16,root/'inputs/preprocessed_bf16.pt')
    (root/'inputs.pt').symlink_to('inputs/preprocessed_bf16.pt')
    assets.update(input_ids=[x['id'] for x in supplied['images']],input_file_sha256=sha256(root/'inputs.pt'),
        stored_dtype='torch.bfloat16',seed=None,preprocessed_fp32_file=str(root/'inputs/preprocessed_fp32.pt'),
        preprocessed_bf16_file=str(root/'inputs/preprocessed_bf16.pt'),
        fp32_serialized_file_sha256=sha256(root/'inputs/preprocessed_fp32.pt'),
        bf16_serialized_file_sha256=sha256(root/'inputs/preprocessed_bf16.pt'),
        sample_fp32_tensor_sha256=[tensor_hash(x) for x in fp32],
        batch_fp32_tensor_sha256={str(b):tensor_hash(fp32[:b]) for b in [1,4]})
    write_json(root/'assets.json',assets)
    manifest=dict(supplied)  # Preserve original sources, descriptions, labels and order without reinterpretation.
    manifest['provided_manifest_sha256']=sha256(dataset/'image_manifest.json')
    manifest['actual_dataset_root']=str(dataset)
    manifest['actual_input_directory']=str(dataset/'images')
    manifest['preprocessing']=dict(weights_id=assets['weights_id'],transform=assets['preprocessing'],
        rgb_conversion=True,random_augmentation=False,torchvision=torchvision.__version__,pillow=PIL.__version__,
        fp32_tensor_file=assets['preprocessed_fp32_file'],bf16_tensor_file=assets['preprocessed_bf16_file'],
        batch_fp32_tensor_sha256=assets['batch_fp32_tensor_sha256'],batch_bf16_tensor_sha256=assets['batch_hashes'],
        file_hashes_are_not_tensor_hashes=True,prepared_once=True)
    manifest['images']=[]
    for i,(original,path) in enumerate(zip(supplied['images'],images)):
        with Image.open(path) as im: size=list(im.size);mode=im.mode
        assert size==[original['width'],original['height']]
        manifest['images'].append(dict(original,actual_absolute_path=str(Path(path).resolve()),
            actual_original_size_wh=size,actual_mode=mode,file_sha256=sha256(path),
            preprocessed_fp32_tensor_sha256=assets['sample_fp32_tensor_sha256'][i],
            preprocessed_bf16_tensor_sha256=assets['sample_bf16_sha256'][i]))
    write_json(root/'image_manifest.json',manifest)
    return assets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','analyze','resume'])
    p.add_argument('--images',nargs=4,metavar='FILE')
    p.add_argument('--dataset-root',type=Path,default=Path('/root/ViT/data/vit_real_images_4'))
    p.add_argument('--previous-run',type=Path,default=HERE/'runs/20261104T113745Z-8fccff')
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--retry-case',action='append',default=[],help='Resume a failed/held diagnostic or strict case as a new preserved attempt')
    p.add_argument('--stop-after',choices=['diagnostic','validation','performance'],default='performance')
    p.add_argument('--device',default='furiosa:0')
    p.add_argument('--worker-timeout',type=float,default=1800)
    p.add_argument('--total-budget',type=float,default=7200)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--repeats',type=int,default=50)
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--atol',type=float)
    p.add_argument('--rtol',type=float)
    a=p.parse_args()
    if a.action=='analyze':
        if not a.run_dir: p.error('--run-dir required')
        from followup_analysis import analyze
        analyze(a.run_dir.resolve());return 0
    if any(not math.isfinite(x) or x<=0 for x in [a.worker_timeout,a.total_budget,a.repeats,a.trials]) or a.warmup<0:
        p.error('Positive finite limits/repeats/trials required')
    if (a.atol is None)!=(a.rtol is None) or any(x is not None and (not math.isfinite(x) or x<0) for x in [a.atol,a.rtol]):
        p.error('Supply both nonnegative finite atol/rtol or neither')
    if a.action=='run':
        if not a.images: p.error('--images requires four ordered FILE arguments')
        root=HERE/'followup_runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+os.urandom(3).hex())
        root.mkdir(parents=True,exist_ok=False)
        for name in ['inputs','outputs','logs','figures','cases','code']: (root/name).mkdir()
        cfg=dict(run_id=root.name,device=a.device,worker_timeout_seconds=a.worker_timeout,total_budget_seconds=a.total_budget,
            warmup=a.warmup,repeats=a.repeats,trials=a.trials,atol=a.atol,rtol=a.rtol,
            previous_run=str(a.previous_run.resolve()),images=[str(Path(x).resolve()) for x in a.images],
            dataset_root=str(a.dataset_root.resolve()),dtype='torch.bfloat16',synthetic_input=False,
            model_id='torchvision.models.vit_b_16',weights_id='ViT_B_16_Weights.IMAGENET1K_V1',
            expected_performance_records=4*a.trials*(1+a.warmup+a.repeats),unique_images=4)
        write_json(root/'config.json',cfg)
        specs=[]
        for stage in ['diagnostic','strict','validation']:
            for b in [1,4]:
                for mode in ['eager','compile']:
                    specs.append(dict(case_id=f'{stage}_{mode}_b{b}',stage=stage,mode=mode,batch=b,trial=1,status='not_run',
                        assets_dir=cfg['previous_run'] if stage in ['diagnostic','strict'] else str(root),
                        cache_path=str(root/'caches'/('diagnostic' if stage=='strict' else stage)/'compiler-cache')))
        order=[[('eager',1),('compile',1),('compile',4),('eager',4)],
               [('compile',1),('eager',1),('eager',4),('compile',4)],
               [('eager',4),('compile',4),('compile',1),('eager',1)]]
        for trial in range(1,a.trials+1):
            for mode,b in order[(trial-1)%3]:
                cid=f'performance_{mode}_b{b}_t{trial}'
                specs.append(dict(case_id=cid,stage='performance',mode=mode,batch=b,trial=trial,status='not_run',
                                  assets_dir=str(root),cache_path=str(root/'caches/performance'/cid)))
        m=dict(run_id=root.name,started_utc=now(),status='preparing',command=[sys.executable,*sys.argv],workers=specs,
               previous_run=cfg['previous_run'],npu_budget_used_seconds=0,
               source_hashes_before={f.name:sha256(f) for f in HERE.glob('*.py')},
               original_timer_sha256=sha256(HERE/'common.py'),original_vit_sha256=sha256(HERE/'vit.py'))
        write_json(root/'manifest.json',m)
        for f in HERE.glob('*.py'):shutil.copyfile(f,root/'code'/f.name)
        from environment import collect
        collect(root)
        prepare_real(root,cfg['images'],a.dataset_root.resolve())
        # Audit original input identity before any diagnostic forward.
        old=read_json(a.previous_run/'assets.json')
        if sha256(a.previous_run/'inputs.pt')!=old['input_file_sha256']: raise RuntimeError('Previous synthetic inputs changed')
        write_json(root/'inputs/previous_input_identity.json',dict(path=str(a.previous_run/'inputs.pt'),sha256=old['input_file_sha256'],
                   batch_hashes=old['batch_hashes'],regenerated=False))
    else:
        if not a.run_dir:p.error('--run-dir required for resume')
        root=a.run_dir.resolve();cfg=read_json(root/'config.json');m=read_json(root/'manifest.json')
        if any(x['status']=='running' for x in m['workers']):
            raise RuntimeError('Manifest has running worker; inspect own worker before resuming')
        if a.retry_case:
            write_json(root/'logs'/('manifest_before_resume_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),m)
            for cid in a.retry_case:
                old=next(x for x in m['workers'] if x['case_id']==cid)
                if old['stage'] not in ['diagnostic','strict'] or old['status'] in ['ok','running']:
                    raise RuntimeError('Only failed/held diagnostic or strict attempts may be retried here')
                n=1+sum(x.get('supersedes')==cid for x in m['workers'])
                new={k:old[k] for k in ['stage','mode','batch','trial','assets_dir','cache_path']}
                new.update(case_id=f'{cid}_retry{n}',status='not_run',supersedes=cid)
                old['superseded_by']=new['case_id'];m['workers'].append(new)
            rank={'diagnostic':0,'strict':1,'validation':2,'performance':3}
            m['workers'].sort(key=lambda x:rank[x['stage']])
            m.setdefault('resume_history',[]).append(dict(utc=now(),command=[sys.executable,*sys.argv],
                reason='Preserved diagnostic instrumentation error and own interrupted worker; retries use new case IDs',
                prior_status=m['status'],prior_error=m.get('error')))
            m.pop('error',None);write_json(root/'manifest.json',m)
    ev=Events(root/'logs/orchestrator.jsonl')
    ev.emit('followup_start',run_dir=str(root),action=a.action)
    budget_left=cfg['total_budget_seconds']-m.get('npu_budget_used_seconds',0)
    start=time.monotonic();deadline=start+budget_left
    def persist():
        m['npu_budget_used_seconds']=cfg['total_budget_seconds']-max(0,deadline-time.monotonic())
        write_json(root/'manifest.json',m)
    def result(spec):
        path=root/'cases'/spec['case_id']/'result.json'
        return read_json(path) if path.exists() else {}
    def find(stage,mode,b):return [x for x in m['workers'] if x['stage']==stage and x['mode']==mode and x['batch']==b][-1]
    def run_worker(spec):
        from environment import command
        snap=[command(['furiosa-smi',x]) for x in ['status','ps']]
        write_json(root/'logs'/f'{spec["case_id"]}.device_before.json',snap)
        if snap[1].get('returncode')!=0 or re.search(r'^\|\s*\d+\s*\|',snap[1].get('stdout',''),re.M):
            spec.update(status='held',reason='NPU in use or usage status unavailable');persist();return
        left=deadline-time.monotonic()
        if left<=0:spec.update(status='budget_exhausted');persist();return
        limit=min(cfg['worker_timeout_seconds'],left)
        cmd=[sys.executable,'-u',str(HERE/'followup_worker.py'),'--run',str(root),'--case',spec['case_id']]
        env=os.environ.copy();env.pop('TORCH_LOGS',None)
        env['FURIOSA_TORCH_LOG']='warn' if spec['stage']=='performance' else 'debug'
        if spec['stage']=='diagnostic':
            env['FURIOSA_TORCH_PROFILE']='trace'
            supported=read_json(root/'environment.json').get('torch_logs_supported',{})
            if supported.get('recompiles') and supported.get('graph_breaks'):env['TORCH_LOGS']='recompiles,graph_breaks'
        info=dict(command=cmd,status='running',started_utc=now(),timeout_seconds=limit,
            stdout=f'logs/{spec["case_id"]}.stdout.log',stderr=f'logs/{spec["case_id"]}.stderr.log',
            code_sha256={f.name:sha256(f) for f in HERE.glob('*.py')},
            environment={k:env.get(k) for k in ['FURIOSA_TORCH_LOG','FURIOSA_TORCH_PROFILE','TORCH_LOGS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
        t=time.monotonic();ev.emit('stage_start',case_id=spec['case_id'],**info)
        with (root/info['stdout']).open('x') as out,(root/info['stderr']).open('x') as err:
            proc=subprocess.Popen(cmd,stdout=out,stderr=err,env=env,start_new_session=True)
            spec.update(info,worker_id=proc.pid);persist();heartbeat=t+30
            try:
                while proc.poll() is None:
                    if time.monotonic()-t>=limit:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
                        spec['status']='timeout' if limit==cfg['worker_timeout_seconds'] else 'budget_exhausted';break
                    if time.monotonic()>=heartbeat:
                        active=result(spec).get('active_call')
                        ev.emit('stage_progress',case_id=spec['case_id'],elapsed_seconds=time.monotonic()-t,
                                remaining_seconds=max(0,deadline-time.monotonic()),active_call=active)
                        heartbeat=time.monotonic()+30
                    time.sleep(1)
                else:spec['status']='ok' if proc.returncode==0 else 'error'
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
                spec['status']='interrupted';raise
            finally:
                spec.update(returncode=proc.returncode,ended_utc=now(),elapsed_seconds=time.monotonic()-t)
                r=result(spec)
                if r and spec['status'] in ['timeout','budget_exhausted','interrupted']:
                    r['status']=spec['status']
                    if r.get('active_call'):
                        row=r['active_call'];row.update(status=spec['status'],timing_valid=False)
                        r['timings'].append(row);r['active_call']=None
                    write_json(root/'cases'/spec['case_id']/'result.json',r)
                persist();ev.emit('stage_end',case_id=spec['case_id'],**{k:spec[k] for k in ['status','worker_id','elapsed_seconds','ended_utc']})
    try:
        m['status']='running';persist()
        for spec in m['workers']:
            if spec['status']!='not_run':continue
            stage=spec['stage'];mode=spec['mode'];b=spec['batch']
            if stage in ['diagnostic','strict','validation'] and b==4:
                prior=find(stage,mode,1)
                if prior['status']!='ok':spec.update(status='held',reason='Same-mode B1 prerequisite failed');persist();continue
            if stage=='strict':
                if not (Path(__import__('furiosa.torch',fromlist=['x']).__file__).parent/'config/_cpu_fallback.py').exists():
                    spec.update(status='held',reason='Official strict fallback control source unavailable');persist();continue
            if stage=='performance':
                from followup_analysis import compare_real,default_path_evidence
                comparison=compare_real(root,stage='validation')
                write_json(root/'real_output_comparison.json',comparison)
                pair=next(x for x in comparison['batches'] if x['batch']==b)
                path=default_path_evidence(root,mode,b)
                if not pair['eligible_for_timing'] or path['actual_cpu_fallback_observed']:
                    spec.update(status='held',reason=pair.get('hold_reason') or 'Actual model CPU fallback observed');persist();continue
            run_worker(spec)
            if spec['status']=='budget_exhausted':break
            # Stop/resume boundaries preserve already completed cases and never overwrite them.
            if stage=='strict' and spec==[x for x in m['workers'] if x['stage']=='strict'][-1] and a.stop_after=='diagnostic':break
            if stage=='validation' and spec==[x for x in m['workers'] if x['stage']=='validation'][-1] and a.stop_after=='validation':break
        m['status']='finished' if all(x['status']=='ok' for x in m['workers']) else ('finished_with_holds' if all(x['status'] in ['ok','held'] for x in m['workers'] if x['stage']=='performance') else 'partial_or_held')
    except BaseException:
        m.update(status='error',error=traceback.format_exc());ev.emit('orchestrator_error',error=m['error'])
    finally:
        m['ended_utc']=now();persist()
        from environment import command
        write_json(root/'logs/device_after.json',[command(['furiosa-smi',x]) for x in ['status','ps']])
        ev.emit('followup_end',status=m['status'],run_dir=str(root))
    if (HERE/'followup_analysis.py').exists():
        from followup_analysis import analyze
        analyze(root)
    print('RUN_DIR='+str(root),flush=True)
    return 0 if m['status'] in ['finished','finished_with_holds'] else 2


if __name__=='__main__':raise SystemExit(main())
