"""Read-only manifest watch; never imports the SDK or launches workers."""
import argparse
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--period',type=float,default=55);a=p.parse_args()
    cfg=json.loads((a.run/'experiment_config.json').read_text())
    while True:
        m=json.loads((a.run/'manifest.json').read_text());w=m['workers'][-1] if m['workers'] else {}
        events=a.run/'cases'/w.get('case_id','none')/'events.jsonl'
        lines=events.read_text().splitlines() if events.exists() else []
        phase=json.loads(lines[-1]).get('event') if lines else w.get('status')
        remaining=max(0,cfg['total_elapsed_budget_s']-(datetime.now(timezone.utc)-datetime.fromisoformat(cfg['created_utc'])).total_seconds())
        print(json.dumps(dict(kst=datetime.now(timezone(timedelta(hours=9))).strftime('%m-%d %H:%M:%S'),status=m['status'],completed=sum(x['status']=='ok' for x in m['workers']),workers=len(m['workers']),case=w.get('case_id'),phase=phase,remaining_h=round(remaining/3600,3),controller_pid=m.get('controller_pid')),ensure_ascii=False),flush=True)
        if m['status'].startswith('finished') or remaining==0:return
        time.sleep(a.period)

if __name__=='__main__':main()
