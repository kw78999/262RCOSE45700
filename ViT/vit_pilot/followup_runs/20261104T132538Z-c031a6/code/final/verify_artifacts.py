"""Independent CSV arithmetic/count/pair audit. Does not execute a model."""
import argparse
import csv
import json
import math
from pathlib import Path
from common import read_json, write_json, now


def verify(run):
    def rows(name):
        with (run/name).open() as f: return list(csv.DictReader(f))
    def number(row,key):
        return float(row[key]) if row[key] else None
    def check(got,wanted,label):
        ok=(got is None and wanted is None) or (got is not None and wanted is not None and math.isclose(got,wanted,rel_tol=1e-10,abs_tol=1e-10))
        if not ok: errors.append(f'{label}: got={got}, expected={wanted}')
    errors=[]
    c=read_json(run/'config.json'); manifest=read_json(run/'manifest.json')
    timings=rows('timings.csv'); summary=rows('summary.csv'); comparisons=rows('comparisons.csv'); shape=rows('shape_switch.csv')
    recomputed={}
    for s in summary:
        raw=[r for r in timings if r['case_id']==s['case_id']]
        good=[r for r in raw if r['status']=='ok' and r['timing_valid']=='True' and r['elapsed_ms'] and float(r['elapsed_ms'])>0]
        values=[float(r['elapsed_ms']) for r in good if r['phase']=='repeat']
        mean=sum(values)/len(values) if values else None
        sorted_values=sorted(values); n=len(values)
        median=(sorted_values[n//2] if n%2 else (sorted_values[n//2-1]+sorted_values[n//2])/2) if n else None
        sd=math.sqrt(sum((v-mean)**2 for v in values)/(n-1)) if n>=2 else None
        check(number(s,'mean_ms'),mean,s['case_id']+' mean')
        check(number(s,'median_ms'),median,s['case_id']+' median')
        check(number(s,'stdev_ms'),sd,s['case_id']+' sample SD')
        check(number(s,'valid_repeats'),n,s['case_id']+' valid count')
        first=next((float(r['elapsed_ms']) for r in good if r['phase']=='first'),None)
        check(number(s,'first_call_ms'),first,s['case_id']+' first')
        if s['throughput_images_s']:
            check(number(s,'throughput_images_s'),int(s['batch'])*n*1000/sum(values),s['case_id']+' images/s')
        if s['status']=='ok' and len(good)!=1+c['warmup']+c['repeats']:
            errors.append(s['case_id']+' incorrect successful record count')
        recomputed[s['case_id']]=mean
    by_id={s['case_id']:s for s in summary}
    for pair in comparisons:
        e=by_id[pair['eager_case_id']]; k=by_id[pair['compile_case_id']]
        for field in ['batch','weights_sha256','dtype','input_hash','input_ids','timing_scope','cache_state']:
            if e[field]!=k[field]: errors.append(f'Pair mismatch B{pair["batch"]}: {field}')
        if pair['compile_speedup']:
            check(float(pair['compile_speedup']),recomputed[e['case_id']]/recomputed[k['case_id']],f'B{pair["batch"]} speedup')
        elif pair['comparable']=='True': errors.append('Comparable pair missing speedup')
    if all(s['status']=='ok' for s in summary) and len(timings)!=c['expected_fixed_records']:
        errors.append('Total fixed record count mismatch')
    if len(shape)!=c['expected_shape_records']: errors.append('Missing shape status rows')
    for mode in ['eager','compile']:
        group=[r for r in shape if r['mode']==mode]
        if [int(r['batch']) for r in group]!=c['shape_sequence']: errors.append(mode+' shape order mismatch')
    for path in run.glob('*.json'):
        json.loads(path.read_text(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))
    data=run/'figures/plot_data.json'
    if data.exists():
        d=read_json(data)
        for item in d['summary']:
            s=by_id[item['case_id']]
            for field in ['mean_ms','stdev_ms','first_call_ms']:
                check(item[field],number(s,field),item['case_id']+' plot '+field)
    audit=dict(utc=now(),status='pass' if not errors else 'fail',errors=errors,
        fixed_records=len(timings),fixed_valid_records=sum(r['status']=='ok' and r['timing_valid']=='True' for r in timings),
        shape_status_records=len(shape),shape_completed_valid_records=sum(r['status']=='ok' and r['timing_valid']=='True' for r in shape),
        note='Independent read-back CSV arithmetic, units, counts, pair metadata, JSON finite serialization and plot-source checks; not a model correctness pass')
    write_json(run/'artifact_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False))
    return 0 if not errors else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path)
    raise SystemExit(verify(p.parse_args().run))
