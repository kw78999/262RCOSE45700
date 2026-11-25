#!/usr/bin/env python3
"""Sequential RNGD ViT pilot; preserves partial results and only kills own workers."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
from common import Events, now, read_json, write_json, sha256

HERE = Path(__file__).resolve().parent


def run(a):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + os.urandom(3).hex()
    root = HERE/'runs'/run_id
    root.mkdir(parents=True,exist_ok=False)
    for name in ['logs','outputs','figures','cases']:
        (root/name).mkdir()
    c = dict(run_id=run_id, device=a.device, warmup=a.warmup, repeats=a.repeats, trials=a.trials,
             worker_timeout_seconds=a.worker_timeout, total_budget_seconds=a.total_budget,
             dtype='torch.bfloat16', batches=[1,4], shape_sequence=[1,1,4,4,1,4],
             model_id='torchvision.models.vit_b_16',weights_id='ViT_B_16_Weights.IMAGENET1K_V1',
             atol=a.atol,rtol=a.rtol,images=a.images,
             cache_state='fresh_process_persistent_cache_possible',
             cache_note='new run-local cache via verified SDK cache_context; shared by all sequential workers; smoke may populate it',
             synthetic_input=not bool(a.images),expected_fixed_records=4*a.trials*(1+a.warmup+a.repeats),
             expected_shape_records=12,independent_trials_per_case=a.trials)
    write_json(root/'config.json',c)
    planned = []
    for b in [1,4]:
        for mode in ['eager','compile']:
            planned.append(dict(case_id=f'smoke_{mode}_b{b}',kind='smoke',mode=mode,batch=b,trial=1,status='not_run'))
    for b in [1,4]:
        for mode in ['eager','compile']:
            for trial in range(1,a.trials+1):
                planned.append(dict(case_id=f'fixed_{mode}_b{b}_t{trial}',kind='fixed',mode=mode,batch=b,trial=trial,status='not_run'))
    for mode in ['eager','compile']:
        planned.append(dict(case_id=f'switch_{mode}',kind='switch',mode=mode,batch=None,trial=1,status='not_run'))
    manifest = dict(run_id=run_id,started_utc=now(),status='running',
        command=[sys.executable,*sys.argv],code_sha256={p.name:sha256(p) for p in HERE.glob('*.py')},
        git_commit=None,git_reason='/root/ViT is not a git repository',workers=planned,
        sync_method='furiosa.torch.current_stream(device).synchronize()',
        timing_scope='cpu_dispatch_model_call_and_current_stream_completion_excludes_transfers',
        cache_conditions=c['cache_note'],interpreter=sys.executable)
    write_json(root/'manifest.json',manifest)
    ev = Events(root/'logs/orchestrator.jsonl')
    ev.emit('run_start',run_id=run_id,run_dir=str(root))
    deadline = time.monotonic() + a.total_budget
    def persist():
        write_json(root/'manifest.json',manifest)
    def execute(cmd,label,diagnostic=False):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            return dict(status='budget_exhausted',reason='total operational budget reached')
        env = os.environ.copy()
        # Diagnostic logging is process-local and separate from performance.
        env.pop('TORCH_LOGS',None)
        env['FURIOSA_TORCH_LOG'] = 'debug' if diagnostic else 'warn'
        if diagnostic and all(read_json(root/'environment.json').get('torch_logs_supported',{}).values()):
            env['TORCH_LOGS']='recompiles,graph_breaks'
        limit = min(a.worker_timeout,remaining)
        start = time.monotonic()
        info = dict(command=cmd,started_utc=now(),timeout_seconds=limit,
                    stdout=f'logs/{label}.stdout.log',stderr=f'logs/{label}.stderr.log',
                    diagnostic_logging=diagnostic,logging_env={k:env.get(k) for k in ['TORCH_LOGS','FURIOSA_TORCH_LOG']})
        ev.emit('stage_start',label=label,**info)
        with (root/info['stdout']).open('w') as out, (root/info['stderr']).open('w') as err:
            proc = subprocess.Popen(cmd,stdout=out,stderr=err,env=env,start_new_session=True)
            info['worker_id']=proc.pid
            next_heartbeat=start+30
            try:
                while proc.poll() is None:
                    t=time.monotonic()
                    if t-start >= limit:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try: proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid,signal.SIGKILL)
                            proc.wait(timeout=10)
                        info['status']='timeout' if limit == a.worker_timeout else 'budget_exhausted'
                        break
                    if t>=next_heartbeat:
                        ev.emit('stage_progress',label=label,worker_id=proc.pid,elapsed_seconds=t-start,
                                total_budget_remaining_seconds=max(0,deadline-t))
                        next_heartbeat=t+30
                    time.sleep(1)
                else:
                    info['status']='ok' if proc.returncode==0 else 'error'
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL)
                        proc.wait(timeout=10)
                info['status']='interrupted'
                raise
            finally:
                info.update(returncode=proc.returncode,ended_utc=now(),elapsed_seconds=time.monotonic()-start)
                write_json(root/'logs'/f'{label}.process.json',info)
                ev.emit('stage_end',label=label,**info)
        return info
    def device_free(label):
        from environment import command
        snapshots=[command(['furiosa-smi',x]) for x in ['status','ps']]
        write_json(root/'logs'/f'{label}.device.json',snapshots)
        ps=snapshots[1]
        if ps.get('returncode')!=0:
            return False,'device use status could not be read'
        if re.search(r'^\|\s*\d+\s*\|',ps.get('stdout',''),re.M):
            return False,'NPU shared/in use; no worker launched; see device snapshot'
        return True,None
    try:
        from environment import collect
        ev.emit('environment_start')
        environment=collect(root)
        ev.emit('environment_end')
        if not environment.get('torch_import',{}).get('ok') or not environment.get('torchvision_import',{}).get('ok'):
            raise RuntimeError('Required torch/torchvision import unavailable; no packages modified')
        if not environment.get('furiosa',{}).get('available'):
            raise RuntimeError('Furiosa NPU unavailable; CPU benchmark not substituted')
        cmd=[sys.executable,'-u',str(HERE/'prepare.py'),'--run',str(root)]
        if a.images: cmd += ['--images',*a.images]
        manifest['asset_preparation']=execute(cmd,'assets')
        persist()
        if manifest['asset_preparation']['status']!='ok':
            raise RuntimeError('Model/input asset preparation failed; see logs/assets.stderr.log')
        manifest['assets']=read_json(root/'assets.json')
        persist()
        smoke_ok={1:False,4:False}
        for case in planned:
            if case['kind']=='smoke' and case['batch']==4 and not smoke_ok[1]:
                case['reason']='Both B1 smoke paths must succeed before larger batches'; persist(); continue
            if case['kind']!='smoke' and not all(smoke_ok.values()):
                case['reason']='Smoke prerequisites failed; larger-batch diagnostic sequence and performance held'; persist(); continue
            free,reason=device_free(case['case_id'])
            if not free:
                case['reason']=reason
                persist()
                continue
            cmd=[sys.executable,'-u',str(HERE/'worker.py'),'--run',str(root),'--case',case['case_id'],
                 '--kind',case['kind'],'--mode',case['mode'],'--trial',str(case['trial'])]
            if case['batch']: cmd += ['--batch',str(case['batch'])]
            case.update(execute(cmd,case['case_id'],case['kind'] in ['smoke','switch']))
            result_path=root/'cases'/case['case_id']/'result.json'
            if result_path.exists():
                r=read_json(result_path)
                if case['status']=='ok' and r['status']!='ok': case['status']=r['status']
                if case['status'] in ['timeout','budget_exhausted']:
                    r['status']=case['status']
                    if r.get('active_call'):
                        row=r['active_call']; row['status']=case['status']; row['timing_valid']=False
                        r['timings'].append(row); r['active_call']=None
                    write_json(result_path,r)
            if case['kind']=='smoke':
                b=case['batch']
                smoke_ok[b]=all(x['status']=='ok' for x in planned if x['kind']=='smoke' and x['batch']==b)
            persist()
            if case['status']=='budget_exhausted':
                break
        manifest['status']='finished' if all(x['status']=='ok' for x in planned) else 'partial_or_blocked'
    except BaseException:
        manifest['status']='blocked'
        manifest['error']=traceback.format_exc()
        ev.emit('run_error',error=manifest['error'])
    finally:
        manifest['ended_utc']=now()
        manifest['budget_remaining_seconds']=max(0,deadline-time.monotonic())
        for case in planned:
            if case['status']=='not_run' and not case.get('reason'):
                case['reason']=manifest.get('error','Previous prerequisite or total budget prevented execution')
        persist()
        from environment import command
        write_json(root/'logs/device_after.json',[command(['furiosa-smi',x]) for x in ['status','ps']])
        ev.emit('run_end',status=manifest['status'],run_dir=str(root))
    if (HERE/'analyze.py').exists():
        subprocess.run([sys.executable,str(HERE/'analyze.py'),str(root)],check=True)
    print(f'RUN_DIR={root}',flush=True)
    return 0 if manifest['status']=='finished' else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','analyze'])
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--device',default='furiosa:0')
    p.add_argument('--worker-timeout',type=float,default=1800)
    p.add_argument('--total-budget',type=float,default=7200)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--trials',type=int,default=1)
    p.add_argument('--atol',type=float)
    p.add_argument('--rtol',type=float)
    p.add_argument('--images',nargs=4)
    a=p.parse_args()
    if a.action=='analyze':
        if not a.run_dir: p.error('--run-dir required for analyze')
        from analyze import analyze
        analyze(a.run_dir.resolve()); return 0
    if min(a.worker_timeout,a.total_budget,a.repeats,a.trials)<=0 or a.warmup<0:
        p.error('Positive budgets/repeats/trials and nonnegative warmup required')
    if (a.atol is None)!=(a.rtol is None) or any(x is not None and x<0 for x in [a.atol,a.rtol]):
        p.error('Supply both nonnegative --atol and --rtol, or neither')
    return run(a)


if __name__=='__main__':
    raise SystemExit(main())
