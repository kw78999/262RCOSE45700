import json,os,signal,sys,time
from pathlib import Path
run=Path(sys.argv[1]);start_order=int(sys.argv[2])
while True:
 m=json.loads((run/'manifest.json').read_text())
 if m['status']!='running':break
 w=m['workers'][-1]
 if w['order']>start_order and w['batch']!=1:
  pid=m['controller_pid']
  try:args=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
  except FileNotFoundError:break
  if str(run) in args and 'experiment.py' in args:
   os.kill(pid,signal.SIGTERM)
   (run/'logs/controller_checkpoint_3.json').write_text(json.dumps(dict(reason='Reserve paired stability budget before core trials; only start complete trial groups',controller_pid=pid,worker_case=w['case_id'],worker_terminated=False),indent=2)+'\n')
  break
 time.sleep(.25)
