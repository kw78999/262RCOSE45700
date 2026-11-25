"""Paired profiler off/on control; same resident input, same process and warm model."""
import json
import os
from pathlib import Path
import re
import shutil
import statistics as st
import subprocess

ROOT=Path(__file__).resolve().parents[2]
run=ROOT/'profiling/batch_pe/runs/20261124T181821Z'
base=run/'profiler_control';base.mkdir(exist_ok=True)
results=[]
for mode in ['eager','compile']:
    ps=subprocess.run(['furiosa-smi','ps'],capture_output=True,text=True,check=True)
    if re.search(r'^\|\s*\d+\s*\|',ps.stdout,re.M):raise RuntimeError('NPU busy')
    case=base/mode
    if not (case/'result.json').exists():
        case.mkdir(exist_ok=True)
        source=run/'cases'/f'p1_b1_{mode}'
        spec=json.loads((source/'spec.json').read_text());spec['control']='OFF/ON/ON/OFF; 10 resident forward calls per block'
        (case/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
        shutil.copytree(source/'cache',case/'cache',dirs_exist_ok=True)
        env=os.environ.copy();env.update(FURIOSA_TORCH_PROFILE='trace',FURIOSA_TORCH_LOG='warn',TORCH_LOGS='graph_breaks,recompiles')
        with (case/'stdout.log').open('w') as out,(case/'stderr.log').open('w') as err:
            p=subprocess.run(['/root/venv/bin/python','-u',str(ROOT/'profiling/batch_pe/control_worker.py'),'--case',str(case)],env=env,stdout=out,stderr=err,timeout=180,check=True)
    r=json.loads((case/'result.json').read_text())
    off=[x['forward_wall_ms'] for x in r['control_samples'] if not x['profiled']]
    on=[x['forward_wall_ms'] for x in r['control_samples'] if x['profiled']]
    row={'mode':mode,'off_calls':len(off),'on_calls':len(on),'off_mean_ms':st.mean(off),'off_median_ms':st.median(off),'on_mean_ms':st.mean(on),'on_median_ms':st.median(on),'on_over_off_median':st.median(on)/st.median(off),'backend_reentries':r['backend_calls_during_samples']}
    results.append(row);print(json.dumps(row),flush=True)
(base/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
