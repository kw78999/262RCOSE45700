"""Keep the authorized run alive independently of a terminal/tool session."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]

def alive(pid,run):
    try:
        if Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]=='Z':return False
        text=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        return str(run) in text and 'experiment.py' in text
    except FileNotFoundError:return False

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--detach',action='store_true');a=ap.parse_args();run=a.run.resolve()
    if a.detach:
        with (run/'logs/supervisor.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--run',str(run)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        print(json.dumps(dict(supervisor_pid=p.pid,session_detached=True,run=str(run))));return
    import fcntl
    lock=(run/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    restarts=0;initial=True
    while True:
        m=json.loads((run/'manifest.json').read_text())
        if m['status'] not in ('running','ready','preparing') and not initial:
            print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),event='study_stopped',status=m['status'])),flush=True);return
        cfg=json.loads((run/'experiment_config.json').read_text())
        if alive(m.get('controller_pid',0),run):
            initial=False;time.sleep(15);continue
        if restarts>=3:
            print('Repeated controller failures; supervisor stops without modifying results',flush=True);return
        initial=False
        with (run/'logs'/f'controller_supervised_{os.getpid()}_{restarts+1}.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(ROOT/'vit_scaling/experiment.py'),'resume','--run',str(run)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),event='controller_started',pid=p.pid,attempt=restarts+1)),flush=True)
            p.wait()
        restarts+=1
        print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),event='controller_exit',returncode=p.returncode)),flush=True)
        time.sleep(3)

if __name__=='__main__':main()
