"""Inventory old attempts, copy caches, run isolated serial profiling, generate report."""
import argparse
import collections
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vit_pilot'))
from common import read_json,write_json,now
SOURCE=ROOT/'vit_scaling/runs/20261118T081725Z-a6c274'

def smi():
    p=subprocess.run(['furiosa-smi','ps'],capture_output=True,text=True,timeout=20)
    if p.returncode:raise RuntimeError('Cannot inspect NPU owners: '+p.stderr)
    pids=[int(x) for x in re.findall(r'^\|\s*(\d+)\s*\|',p.stdout,re.M)]
    return {'utc':now(),'pids':pids,'stdout':p.stdout}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path);ap.add_argument('--only',default='');a=ap.parse_args()
    run=a.run or ROOT/'profiling/batch_pe/runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=run.resolve()
    run.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'profiling/batch_pe/controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read_json(SOURCE/'manifest.json');groups=collections.defaultdict(list)
    for w in old['workers']:groups[(w['pe'],w['batch'],w['mode'])].append(w)
    if not (run/'inventory.json').exists():
        inv=[]
        for (pe,b,mode),ws in sorted(groups.items()):
            ok=[w for w in ws if w['status']=='ok'];chosen=(ok or ws)[-1]
            # Prefer a successful formal cache; failures are listed but not reused if success exists.
            successful_formal=[w for w in ok if w['stage']=='formal']
            chosen=(successful_formal or ok or ws)[-1]
            inv.append({'pe':pe,'batch':b,'mode':mode,'attempts':len(ws),
                'old_status_counts':dict(collections.Counter(w['status'] for w in ws)),
                'successful_stages':sorted(set(w['stage'] for w in ok)),
                'cache_source':chosen['cache_path'],'cache_source_case':chosen['case_id'],
                'source_run':str(SOURCE),'warmup':100 if any(w['stage']=='formal' for w in ok) else 20,
                'samples':5,'timeout_s':3600})
        write_json(run/'inventory.json',inv)
        write_json(run/'planned_matrix.json',old['requested_matrix'])
        write_json(run/'metadata.json',{'created_utc':now(),'source_run':str(SOURCE),'command':sys.argv,
            'cache_policy':'copy source cache; first-call times are cache-reuse startup, not cold compilation',
            'original_artifacts_unchanged':True})
        for name in ['worker.py','run.py']:
            shutil.copy2(Path(__file__).parent/name,run/name)
        for cmd,name in [(['furiosa-smi','info'],'device_info.txt'),(['furiosa-smi','status'],'device_status.txt')]:
            p=subprocess.run(cmd,capture_output=True,text=True);(run/name).write_text(p.stdout+p.stderr)
    inv=read_json(run/'inventory.json')
    # Small, previously successful conditions first. All historical attempts stay in inventory.
    for s in sorted(inv,key=lambda s:(s['batch'],s['pe'],s['mode'])):
        case=f"p{s['pe']}_b{s['batch']}_{s['mode']}"
        if a.only and case!=a.only:continue
        folder=run/'cases'/case
        if (folder/'result.json').exists():continue
        before=smi();write_json(run/'device_before.json',before)
        if before['pids']:
            print('DEVICE_BUSY',before['pids'],flush=True);break
        folder.mkdir(parents=True,exist_ok=True);write_json(folder/'spec.json',s)
        if not (folder/'cache').exists():shutil.copytree(s['cache_source'],folder/'cache',dirs_exist_ok=True)
        env=os.environ.copy();env.update(FURIOSA_TORCH_PROFILE='trace',FURIOSA_TORCH_LOG='warn',TORCH_LOGS='graph_breaks,recompiles')
        print(json.dumps({'event':'start','run':str(run),'case':case,'utc':now()}),flush=True)
        t=time.monotonic();status='running';contamination=[]
        with (folder/'stdout.log').open('w') as out,(folder/'stderr.log').open('w') as err:
            p=subprocess.Popen(['/root/venv/bin/python','-u',str(Path(__file__).parent/'worker.py'),'--case',str(folder)],env=env,stdout=out,stderr=err,start_new_session=True)
            write_json(folder/'process.json',{'pid':p.pid,'started_utc':now()})
            last=time.monotonic()
            while p.poll() is None:
                time.sleep(1)
                if time.monotonic()-last>=30:
                    snap=smi()
                    # Existing ownership helper also covers compiler children and SMI helper processes.
                    sys.path.insert(0,str(ROOT/'vit_scaling'));from process_guard import inventory,relation
                    procs=inventory()
                    foreign=[pid for pid in snap['pids'] if not relation(pid,p.pid,procs,procs).get('owned')]
                    with (folder/'device_samples.jsonl').open('a') as f:f.write(json.dumps(dict(snap,foreign=foreign))+'\n')
                    if foreign:contamination+=foreign
                    last=time.monotonic()
                limit_path=Path('/sys/fs/cgroup/memory.max');used_path=Path('/sys/fs/cgroup/memory.current')
                low_memory=False
                if limit_path.exists() and limit_path.read_text().strip()!='max':
                    maximum=int(limit_path.read_text());used=int(used_path.read_text())
                    low_memory=maximum-used<8*1024**3
                    if low_memory:write_json(folder/'resource_guard.json',dict(pid=p.pid,memory_limit_bytes=maximum,memory_used_bytes=used,reason='cgroup memory headroom below 8 GiB'))
                if time.monotonic()-t>s['timeout_s'] or contamination or low_memory:
                    status='resource_limit' if low_memory else 'timeout' if not contamination else 'contaminated'
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                    break
        r=read_json(folder/'result.json') if (folder/'result.json').exists() else dict(s,status='error',samples=[])
        r['controller_status']=status if status!='running' else 'ok' if p.returncode==0 else 'error'
        r['returncode']=p.returncode;r['wall_s']=time.monotonic()-t;r['foreign_pids']=contamination
        write_json(folder/'result.json',r)
        print(json.dumps({'event':'end','case':case,'status':r['controller_status'],'wall_s':round(r['wall_s'],1)}),flush=True)
        subprocess.run([sys.executable,str(Path(__file__).parent/'report.py'),'--run',str(run)],check=True)
        if contamination:break
    write_json(run/'controller_finished.json',dict(ended_utc=now()))
    print('RUN',run,flush=True)

if __name__=='__main__':main()
