"""Independent final audit of follow-up CSVs, actual assets and execution metadata; no forward."""
import csv
import json
import math
from pathlib import Path
import sys
from common import read_json,write_json,sha256,tensor_hash,now


def verify(root):
    import torch
    cfg=read_json(root/'config.json');m=read_json(root/'manifest.json');assets=read_json(root/'assets.json')
    def csv_rows(name):
        with (root/name).open() as f:return list(csv.DictReader(f))
    raw=csv_rows('timings.csv');summary=csv_rows('trial_summary.csv');pairs=csv_rows('comparisons.csv')
    errors=[]
    def require(ok,label):
        if not ok:errors.append(label)
    def equal(x,y,label):require((x is None and y is None) or (x is not None and y is not None and math.isclose(float(x),float(y),rel_tol=1e-10,abs_tol=1e-10)),label)
    means={}
    for row in summary:
        records=[r for r in raw if r['case_id']==row['case_id']]
        values=[float(r['elapsed_ms']) for r in records if r['phase']=='repeat' and r['status']=='ok' and r['timing_valid']=='True' and float(r['elapsed_ms'])>0]
        n=len(values);avg=sum(values)/n if n else None
        sd=math.sqrt(sum((v-avg)**2 for v in values)/(n-1)) if n>=2 else None
        vals=sorted(values);median=(vals[n//2] if n%2 else (vals[n//2-1]+vals[n//2])/2) if n else None
        equal(row['mean_ms'] or None,avg,row['case_id']+' mean')
        equal(row['median_ms'] or None,median,row['case_id']+' median')
        equal(row['stdev_ms'] or None,sd,row['case_id']+' sample SD')
        equal(row['valid_repeats'],n,row['case_id']+' valid repeat count')
        if row['throughput_images_s']:equal(row['throughput_images_s'],int(row['batch'])*n*1000/sum(values),row['case_id']+' images/s')
        first=next((float(x['elapsed_ms']) for x in records if x['phase']=='first' and x['status']=='ok' and x['timing_valid']=='True'),None)
        equal(row['first_call_ms'] or None,first,row['case_id']+' first call')
        if row['status']=='held':require(not records and not row['mean_ms'] and not row['throughput_images_s'],row['case_id']+' held case has measurements')
        if row['status']=='ok':require(len(records)==1+cfg['warmup']+cfg['repeats'],row['case_id']+' call count')
        means[row['case_id']]=avg
    for pair in pairs:
        if pair['compile_speedup']:
            equal(pair['compile_speedup'],means[pair['eager_case_id']]/means[pair['compile_case_id']],'paired speedup')
        else:require(pair['comparable']=='False','Missing comparable speedup')
    bf16=torch.load(root/'inputs/preprocessed_bf16.pt',weights_only=True,map_location='cpu')
    fp32=torch.load(root/'inputs/preprocessed_fp32.pt',weights_only=True,map_location='cpu')
    require(torch.equal(fp32.to(torch.bfloat16),bf16),'Saved FP32->BF16 identity')
    require(tuple(bf16.shape)==(4,3,224,224),'Preprocessed shape')
    require(sha256(root/'inputs.pt')==assets['input_file_sha256'],'Preprocessed serialized file hash')
    require(len(set(tensor_hash(t) for t in bf16))==4,'Unique preprocessed images')
    for b in [1,4]:require(tensor_hash(bf16[:b])==assets['batch_hashes'][str(b)],f'B{b} tensor bytes')
    im=read_json(root/'image_manifest.json')
    original=read_json(root/'inputs/provided_image_manifest.json')
    require(original['synthetic_input'] is False and im['synthetic_input'] is False,'Real input flags')
    for supplied,used in zip(original['images'],im['images']):
        require(all(used[k]==v for k,v in supplied.items()),supplied['id']+' provided provenance preservation')
        require(sha256(used['actual_absolute_path'])==supplied['sha256'],supplied['id']+' original image hash')
        require(supplied['ground_truth_label'] is None,supplied['id']+' labels must remain absent')
    successful=[s for s in m['workers'] if s['stage']=='performance' and s['status']=='ok']
    metadata=[read_json(root/'cases'/s['case_id']/'result.json') for s in successful]
    for field in ['input_file_sha256','weights_sha256','dtype','timing_scope','stream_id']:
        require(len(set(json.dumps(r[field],sort_keys=True) for r in metadata))<=1,'Across-case '+field)
    for field in ['torch_threads','torch_interop_threads','compiler_config','device_configuration','cpu_fallback']:
        require(len(set(json.dumps(r['effective'][field],sort_keys=True) for r in metadata))<=1,'Across-case effective '+field)
    require(len({s['cache_path'] for s in successful})==len(successful),'Unique successful worker SDK cache paths')
    for s,r in zip(successful,metadata):
        require(r['cache_existed_at_entry'] is False,s['case_id']+' cache initially absent')
        require(r['input_hash']==assets['batch_hashes'][str(s['batch'])],s['case_id']+' tensor identity')
        require(r['synthetic_input'] is False,s['case_id']+' real assets')
        require(r['output_valid'] and r['sync_checked'],s['case_id']+' completion/output validity')
        require(not r['backend_entries'] and not (root/'cases'/s['case_id']/'furiosa_trace.json').exists(),s['case_id']+' performance instrumentation must be absent')
    stable_files=['common.py','vit.py','pilot.py','worker.py','environment.py','analyze.py','prepare.py','plots.py','verify_artifacts.py']
    for name in stable_files:require(sha256(Path(__file__).parent/name)==m['source_hashes_before'][name],'Existing source unchanged: '+name)
    old=Path(cfg['previous_run']);old_assets=read_json(old/'assets.json')
    require(sha256(old/'inputs.pt')==old_assets['input_file_sha256'],'Previous synthetic input preserved')
    plots=root/'figures/plot_data.json'
    if plots.exists():
        plotted=read_json(plots)['rows']
        for r in plotted:equal(r['mean_ms'],means[r['case_id']],r['case_id']+' plotted mean')
    for file in root.glob('*.json'):
        json.loads(file.read_text(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))
    result=dict(utc=now(),status='pass' if not errors else 'fail',errors=errors,performance_records=len(raw),
                successful_workers=len(successful),held_workers=sum(s['stage']=='performance' and s['status']=='held' for s in m['workers']),
                expected_full_design_records=cfg['expected_performance_records'],
                expected_records_for_successful_workers=len(successful)*(1+cfg['warmup']+cfg['repeats']),
                existing_sources_unchanged=True if not any('Existing source' in x for x in errors) else False,
                notes='Independent CSV arithmetic and metadata/file/tensor/plot-source audit; no NPU or model forward, no output tolerance acceptance claim')
    write_json(root/'final_verification.json',result);print(json.dumps(result,ensure_ascii=False))
    return 0 if not errors else 1


if __name__=='__main__':raise SystemExit(verify(Path(sys.argv[1]).resolve()))
