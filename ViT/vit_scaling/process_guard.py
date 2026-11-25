"""Read-only Linux process identity checks; never trust UID or command names alone."""
from pathlib import Path

def inventory():
    rows={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            s=(p/'stat').read_text().rsplit(') ',1)[1].split()
            rows[int(p.name)]=dict(pid=int(p.name),state=s[0],ppid=int(s[1]),pgrp=int(s[2]),session=int(s[3]),start_ticks=int(s[19]))
        except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError,ValueError):continue
    return rows

def relation(pid,worker_pid,before,after,during=None):
    if pid==worker_pid:return dict(pid=pid,owned=True,basis='worker_pid')
    observations=[('after_smi',after),('during_smi',during or {}),('before_smi',before)]
    identities={rows[pid]['start_ticks'] for _,rows in observations if pid in rows}
    if len(identities)>1:return dict(pid=pid,owned=False,basis='pid_identity_changed_during_observation')
    for name,rows in observations:
        row=rows.get(pid)
        if not row:continue
        if row['pgrp']==worker_pid and row['session']==worker_pid:
            return dict(**row,owned=True,basis='worker_process_group_and_session',observation=name)
        ancestor=pid;chain=[]
        for _ in range(32):
            item=rows.get(ancestor)
            if not item:break
            chain.append(ancestor)
            if item['ppid']==worker_pid:return dict(**row,owned=True,basis='worker_ancestor',ancestor_chain=chain+[worker_pid],observation=name)
            if item['ppid']<=1 or item['ppid']==ancestor:break
            ancestor=item['ppid']
    row=after.get(pid) or before.get(pid)
    return dict(pid=pid,owned=False,basis='foreign_process' if row else 'exited_before_identity_verification',process=row)
