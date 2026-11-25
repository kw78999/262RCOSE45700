"""Stop only this profiling worker if cgroup memory headroom falls below 8 GiB."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
run=a.run.resolve();limit_path=Path('/sys/fs/cgroup/memory.max');used_path=Path('/sys/fs/cgroup/memory.current')
while True:
    active=False
    for p in (run/'cases').glob('*/process.json'):
        result=p.parent/'result.json'
        if not result.exists():continue
        d=json.loads(result.read_text())
        if d.get('controller_status') or d.get('status')!='running':continue
        pid=json.loads(p.read_text())['pid']
        try:
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00',b' ').decode()
            own=(os.getpgid(pid)==pid and 'profiling/batch_pe/worker.py' in cmd and str(p.parent) in cmd)
        except (ProcessLookupError,FileNotFoundError):continue
        if not own:continue
        active=True
        if limit_path.exists() and used_path.exists() and limit_path.read_text().strip()!='max':
            maximum=int(limit_path.read_text());used=int(used_path.read_text())
            if maximum-used<8*1024**3:
                row={'pid':pid,'memory_limit_bytes':maximum,'memory_used_bytes':used,'reason':'cgroup memory headroom below 8 GiB','action':'SIGTERM own worker process group'}
                (p.parent/'resource_guard.json').write_text(json.dumps(row,indent=2))
                os.killpg(pid,signal.SIGTERM);print(json.dumps(row),flush=True)
    if (run/'controller_finished.json').exists():break
    time.sleep(1)
