"""Keep the authorized run alive independently of a terminal/tool session."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import re
import signal
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]

def alive(pid,run):
    try:
        if Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]=='Z':return False
        text=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        return str(run) in text and 'experiment.py' in text
    except FileNotFoundError:return False

def main():
    # Preserve this disposition through exec in controllers/workers. setsid()
    # removes the terminal association; explicitly ignore any delivered HUP too.
    # This cannot protect against SIGKILL, container teardown, or host shutdown.
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
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
            if m['status']!='held_external_device' or any(e.get('status')=='held_global_configuration_error' for e in m.get('events',[])):
                print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),event='study_stopped',status=m['status'])),flush=True);return
            # A held measurement stays excluded. Resume a fresh attempt only
            # after two empty device observations; never terminate foreign PIDs.
            idle=0
            while idle<2:
                cfg=json.loads((run/'experiment_config.json').read_text())
                elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(cfg['created_utc'])).total_seconds()
                if elapsed>=cfg['total_elapsed_budget_s']:return
                try:
                    observed=subprocess.run(['furiosa-smi','ps'],capture_output=True,text=True,timeout=20)
                    known=observed.returncode==0 and '| PID' in observed.stdout
                    pids=[int(x) for x in re.findall(r'^\|\s*(\d+)\s*\|',observed.stdout,re.M)] if known else None
                except Exception:pids=None
                idle=idle+1 if pids==[] else 0
                print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),event='held_device_observation',pids=pids,consecutive_idle=idle)),flush=True)
                if idle<2:time.sleep(min(30,cfg['monitor_period_s']))
            initial=True
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
        latest=json.loads((run/'manifest.json').read_text())
        if latest['status']=='running':restarts+=1
        print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),event='controller_exit',returncode=p.returncode)),flush=True)
        time.sleep(3)

if __name__=='__main__':main()
