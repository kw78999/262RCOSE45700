"""One-off safe controller checkpoint; never terminates the NPU worker."""
import json,os,signal,time,sys
from pathlib import Path
run=Path(sys.argv[1]).resolve()
while True:
    results=[]
    for mode in ('eager','compile'):
        p=run/'cases'/f'formal_p1_b4_{mode}_t1_a1'/'result.json'
        try:r=json.loads(p.read_text())
        except (FileNotFoundError,json.JSONDecodeError):continue
        if r.get('status')=='ok' and r.get('ended_utc'):results.append(r)
    if len(results)==2:
        m=json.loads((run/'manifest.json').read_text());pid=m['controller_pid']
        try:args=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        except FileNotFoundError:break
        if str(run) in args and 'experiment.py' in args:
            os.kill(pid,signal.SIGTERM)
            (run/'logs/controller_checkpoint_2.json').write_text(json.dumps(dict(controller_pid=pid,reason='Apply balanced PE ordering and stability budget reservation after completed baseline',worker_terminated=False),indent=2)+'\n')
        break
    time.sleep(.2)
