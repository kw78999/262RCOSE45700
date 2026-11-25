"""Independent sweep-line checks on captured device intervals and report completeness."""
import argparse
import csv
import json
import math
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();run=args.run
rows=json.loads((run/'samples.json').read_text());summary=json.loads((run/'summary.json').read_text())
errors=[]
for row in rows:
    path=run/'cases'/row['case']/f"forward_{row['iteration']:02d}.json"
    events=json.loads(path.read_text())['traceEvents']
    tasks=[e for e in events if '::Task::' in e.get('name','') and {'begin_cycle','end_cycle','cycle_actual'}<=e.get('args',{}).keys() and e.get('ph')=='X' and e.get('dur',0)>0]
    ticks=[]
    for e in tasks:
        if int(e['args']['end_cycle'])-int(e['args']['begin_cycle'])!=int(e['args']['cycle_actual']):errors.append(f'{path}: cycle metadata inconsistent')
        ticks.extend([(e['ts'],1),(e['ts']+e['dur'],-1)])
    active=0;last=0.;total=0.
    for t,delta in sorted(ticks):
        if active:total+=t-last
        active+=delta;last=t
    expected=total/1000 if ticks else None
    if expected is None:
        if row['npu_task_ms'] is not None:errors.append(f'{path}: missing should be null')
    elif not math.isclose(expected,row['npu_task_ms'],abs_tol=1e-8):errors.append(f'{path}: interval union inconsistent')
    if expected is not None and expected>row['forward_wall_ms']+.05:errors.append(f'{path}: device time exceeds observed forward wall')
    for phase in ['h2d','forward','d2h']:
        p=path.parent/f"{phase}_{row['iteration']:02d}.json"
        if not p.exists():errors.append(f'{p}: missing trace')
for s in summary:
    if s['status']=='ok':
        if s['samples']!=5 or s['valid_npu_samples']!=5:errors.append(f"{s['case']}: incomplete capture")
        r=json.loads((run/'cases'/s['case']/'result.json').read_text())
        if not r.get('output_finite') or not r.get('stream_complete'):errors.append(f"{s['case']}: output/completion invalid")
        if r.get('backend_calls_during_samples')!=0:errors.append(f"{s['case']}: new backend calls during measured samples")
        if r.get('foreign_pids'):errors.append(f"{s['case']}: foreign NPU process")
report=(run/'REPORT.ko.md').read_text()
if len(summary)!=16:errors.append('Expected 16 historically attempted mode conditions')
if len({(x['pe'],x['batch']) for x in summary})!=8:errors.append('Expected 8 historical batch x PE pairs')
table_columns=None
for line in report.splitlines():
    if not line.startswith('|'):
        table_columns=None;continue
    if not line.endswith('|'):errors.append('Malformed Markdown table row')
    count=line.count('|')
    if table_columns is None:table_columns=count
    elif count!=table_columns:errors.append('Markdown table column mismatch')
result={'samples_checked':len(rows),'mode_conditions':len(summary),'success_conditions':sum(x['status']=='ok' for x in summary),'errors':errors,'passed':not errors}
(run/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
raise SystemExit(1 if errors else 0)
