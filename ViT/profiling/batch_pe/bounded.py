"""One short attempt per historically unstarted case, with strict resource budgets."""
import argparse
import collections
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from run import SOURCE, smi
from report import events, task_events, union_ms


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget-s', type=int, default=600)
    ap.add_argument('--worker-s', type=int, default=120)
    a = ap.parse_args()
    if min(a.budget_s, a.worker_s) <= 0:
        ap.error('budgets must be positive')
    run = ROOT/'profiling/batch_pe/runs'/('bounded_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run.mkdir(parents=True)
    locks = []
    for path in [ROOT/'profiling/batch_pe/controller.lock', ROOT/'vit_scaling/device.lock']:
        handle = path.open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    old = json.loads((SOURCE/'manifest.json').read_text())
    historical = {(w['pe'], w['batch']) for w in old['workers']}
    plan = [dict(pe=p, batch=b, mode=m, status='pending')
            for b in [4,8,16,32,64,128,256,512] for p in [1,2,4,8]
            if (p,b) not in historical for m in ['eager','compile']]
    write(run/'metadata.json', dict(source_run=str(SOURCE), budget_s=a.budget_s,
          worker_timeout_s=a.worker_s, warmup=3, samples=5, independent_trials=1,
          retries=0, memory_headroom_gib=8, performance_status='exploratory; no stabilization or statistical confidence claim',
          selection='only batch/PE combinations without any historical worker', command=sys.argv))
    shutil.copy2(__file__, run/'bounded_snapshot.py')
    shutil.copy2(Path(__file__).with_name('worker.py'), run/'worker_snapshot.py')
    start = time.monotonic()
    stop = None
    frontiers = {}
    try:
        for item in plan:
            p,b,m = item['pe'],item['batch'],item['mode']
            if stop:
                item.update(status='not_attempted', reason=stop)
                continue
            if (p,m) in frontiers and b > frontiers[p,m]:
                item.update(status='not_attempted', reason='lower_batch_resource_or_timeout; support unverified')
                continue
            remaining = a.budget_s-(time.monotonic()-start)
            if remaining < 30:
                stop = 'total_time_budget'; item.update(status='not_attempted', reason=stop); continue
            before = smi()
            if before['pids']:
                stop='device_busy'; item.update(status='not_attempted',reason=stop); continue
            folder=run/'cases'/f'p{p}_b{b}_{m}'; folder.mkdir(parents=True)
            write(folder/'device_before.json',before)
            # Reuse only this PE/mode's successful smaller-batch operator cache.
            candidates=[w for w in old['workers'] if w['pe']==p and w['mode']==m and w['batch']<=b and w['status']=='ok']
            candidates.sort(key=lambda w:(w['batch'],w['stage']=='formal',w.get('order',0)))
            origin=Path(candidates[-1]['cache_path']) if candidates else None
            if origin and origin.exists(): shutil.copytree(origin,folder/'cache')
            else: (folder/'cache').mkdir()
            spec=dict(pe=p,batch=b,mode=m,source_run=str(SOURCE),warmup=3,samples=5,
                      cache_source=str(origin) if origin else None, timeout_s=min(a.worker_s,remaining-5))
            write(folder/'spec.json',spec)
            env=os.environ.copy()
            env.update(FURIOSA_TORCH_PROFILE='trace',FURIOSA_TORCH_LOG='warn',TORCH_LOGS='graph_breaks,recompiles')
            print(json.dumps(dict(event='start',case=folder.name,run=str(run))),flush=True)
            began=time.monotonic(); status=None
            with (folder/'stdout.log').open('w') as out,(folder/'stderr.log').open('w') as err:
                proc=subprocess.Popen(['/root/venv/bin/python','-u',str(Path(__file__).with_name('worker.py')),'--case',str(folder)],env=env,stdout=out,stderr=err,start_new_session=True)
                write(folder/'process.json',dict(pid=proc.pid))
                try:
                    last=began
                    while proc.poll() is None:
                        time.sleep(0.5)
                        limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
                        used=int(Path('/sys/fs/cgroup/memory.current').read_text())
                        if limit!='max' and int(limit)-used<8*1024**3:
                            status='resource_limit';write(folder/'resource_guard.json',dict(used_bytes=used,limit_bytes=int(limit)));break
                        if time.monotonic()-began>spec['timeout_s']: status='timeout';break
                        if time.monotonic()-last>30:
                            from process_guard import inventory, relation
                            snap=smi(); procs=inventory()
                            if any(not relation(pid,proc.pid,procs,procs).get('owned') for pid in snap['pids']):
                                status='contaminated';stop='device_busy';break
                            last=time.monotonic()
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            result_path=folder/'result.json'
            r=json.loads(result_path.read_text()) if result_path.exists() else {}
            status=status or ('ok' if proc.returncode==0 and r.get('status')=='ok' else 'error')
            logs=(folder/'stderr.log').read_text(errors='replace')
            reason=''
            if status=='error':
                reason=str(r.get('error',logs)).strip().split('\n')[-1][:400]
                if any(token in logs for token in ['assert.h: No such file', 'failed to run cc', 'No module named', 'cannot open shared object file']):
                    stop='common_environment_error';reason=stop+': '+reason
            if status in ['resource_limit','timeout']: frontiers[p,m]=b
            r.update(controller_status=status, wall_s=time.monotonic()-began, returncode=proc.returncode)
            write(result_path,r)
            rows=r.get('samples',[])
            npu=[union_ms(task_events(events(folder/f"forward_{row['iteration']:02d}.json"))) for row in rows]
            npu=[v for v in npu if v is not None]
            item.update(status=status,reason=reason,case=str(folder.relative_to(run)),wall_s=r['wall_s'],samples=len(rows),
                        forward_median_ms=statistics.median([x['forward_wall_ms'] for x in rows]) if rows else None,
                        npu_task_median_ms=statistics.median(npu) if npu else None)
            print(json.dumps(item),flush=True)
            write(run/'coverage.json',plan)
    finally:
        for item in plan:
            if item['status']=='pending': item.update(status='not_attempted',reason='controller_interrupted')
        write(run/'coverage.json',plan)
        fields=sorted(set().union(*(x.keys() for x in plan)))
        with (run/'coverage.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(plan)
        write(run/'finished.json',dict(elapsed_s=time.monotonic()-start,stop_reason=stop,
              counts=dict(collections.Counter(x['status'] for x in plan))))
        print('RUN',run,flush=True)

if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'vit_scaling'))
    main()
