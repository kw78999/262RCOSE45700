"""Small helpers shared by this comparison; original model/timer stay unchanged."""
import hashlib
import json
from pathlib import Path
from common import read_json

CACHE_POLICY = 'fresh_process_private_sdk_cache_initially_absent_system_cold_unverified'
TIMING_SCOPE = 'cpu_dispatch_runner_and_current_stream_completion_excludes_external_prepost_transfers'


def state_hash(state):
    """CPU canonical state_dict SHA256, before device transfer; not a file hash."""
    import torch
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        header = json.dumps([name, str(t.dtype), list(t.shape)], separators=(',', ':')).encode()
        data = t.view(torch.uint8).numpy().tobytes()
        h.update(len(header).to_bytes(8,'little')); h.update(header)
        h.update(len(data).to_bytes(8,'little')); h.update(data)
    return h.hexdigest()


def find_case(manifest, stage, mode, batch, trial=1):
    return next((w for w in reversed(manifest['workers']) if
        (w['stage'], w['mode'], w['batch'], w['trial']) == (stage, mode, batch, trial)), None)


def result_for(run, case):
    if not case:
        return {}
    p = Path(run)/'cases'/case['case_id']/'result.json'
    return read_json(p) if p.exists() else {}


def compare_pair(run, stage, batch, trial=1):
    import torch
    from attention_analysis import compare_files
    run = Path(run)
    cfg, m, assets = [read_json(run/name) for name in ['config.json','manifest.json','assets.json']]
    cases = [find_case(m,stage,mode,batch,trial) for mode in ['eager','compile']]
    results = [result_for(run,c) for c in cases]
    out = dict(stage=stage, batch=batch, trial=trial, eligible=False, tolerance_status='pending',
               eager_case_id=cases[0]['case_id'] if cases[0] else None,
               compile_case_id=cases[1]['case_id'] if cases[1] else None)
    if any(not c or c['status'] != 'ok' or r.get('status') != 'ok' or not r.get('output_valid') for c,r in zip(cases,results)):
        out['reason'] = 'Both current-condition workers and outputs must complete normally'
        return out
    keys = ['input_hash','weights_sha256','state_dict_bf16_cpu_sha256','dtype','timing_scope','cache_state']
    if any(results[0].get(k) is None or results[0].get(k) != results[1].get(k) for k in keys):
        out['reason'] = 'Paired input/weights/state/dtype/timing/cache policy mismatch'; return out
    effective_keys = ['device_configuration','compiler_config','torch_threads','torch_interop_threads',
                      'backend_eager_fallback','suppress_errors','cpu_fallback','mha_fastpath',
                      'automatic_dynamic_shapes','recompile_limit']
    if any(results[0]['effective'].get(k) != results[1]['effective'].get(k) for k in effective_keys):
        out['reason'] = 'Effective settings differ'; return out
    for r in results:
        if not r.get('sync_checked') or r['effective']['cpu_fallback'] or r['effective']['mha_fastpath']:
            out['reason'] = 'Strict controls or completion verification failed'; return out
        if any(s['mha_fastpath'] or s['cpu_fallback'] for s in r['control_snapshots'].values()):
            out['reason'] = 'Controls did not remain false'; return out
        if r.get('path_observations',{}).get('native_mha_host_count',0):
            out['reason'] = 'Unexpected native MHA observed'; return out
    comp = compare_files(Path(results[1]['output_file']), Path(results[0]['output_file']), assets['input_ids'][:batch])
    out.update(comp)
    out.update(eager_case_id=cases[0]['case_id'],compile_case_id=cases[1]['case_id'],
               input_hash=results[0]['input_hash'],weights_sha256=results[0]['weights_sha256'],
               state_dict_bf16_cpu_sha256=results[0]['state_dict_bf16_cpu_sha256'],
               reference_is_current_fastpath_off_eager=True)
    if cfg['atol'] is not None:
        u,v = [torch.load(r['output_file'],map_location='cpu',weights_only=True).float() for r in results]
        out.update(tolerance_status='specified',atol=cfg['atol'],rtol=cfg['rtol'],
                   tolerance_source=cfg['tolerance_source'], allclose=bool(torch.allclose(u,v,atol=cfg['atol'],rtol=cfg['rtol'])))
    out['eligible'] = bool(out['structural_finite_ok'] and out['top1_matches'] == batch and out.get('allclose') is not False)
    out['reason'] = 'Output tolerance pending; small-image consistency only' if out['eligible'] and cfg['atol'] is None else None if out['eligible'] else 'Structure/finite/Top-1/explicit tolerance check failed'
    return out
