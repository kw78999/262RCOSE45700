"""One isolated profiling condition; never edits source experiment artifacts."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vit_pilot'))
from common import now, read_json, write_json, sha256, tensor_hash
from fastpath_support import state_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=Path,required=True);args=ap.parse_args()
    folder=args.case;spec=read_json(folder/'spec.json');source=Path(spec['source_run'])
    cfg=read_json(source/'experiment_config.json');assets=read_json(source/'assets.json')
    r=dict(spec,status='running',started_utc=now(),pid=os.getpid(),samples=[],backend_calls=[],warmup_ms=[])
    def save():write_json(folder/'result.json',r)
    def event(name,**kwargs):
        print(json.dumps(dict(utc=now(),event=name,**kwargs)),flush=True);save()
    save()
    try:
        import torch
        import furiosa.torch as ft
        from torchvision.models import vit_b_16
        from furiosa.torch.profiler import Profiler,RecordFunction
        r['versions']={'torch':torch.__version__,'furiosa':ft.version()}
        fusion=ft.set_fusion(spec['pe'])
        if fusion.num_fusion!=spec['pe']:raise RuntimeError('Fusion mismatch')
        torch.set_num_threads(cfg['torch_threads']);torch.set_num_interop_threads(cfg['torch_interop_threads'])
        torch.backends.mha.set_fastpath_enabled(False)
        if ft.backend.eager_fallback or torch._dynamo.config.suppress_errors:raise RuntimeError('Fallback configured')
        r['fusion']=str(fusion);r['device_configuration']=str(ft.get_device_configuration())
        r['effective']={'device':cfg['device'],'dtype':'BF16','mha_fastpath':False,'cpu_fallback':False,
                        'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
                        'compiler_config':str(ft.config.get_compiler_config()),'profile_level':os.environ['FURIOSA_TORCH_PROFILE']}
        if sha256(assets['weights_file'])!=assets['weights_sha256']:raise RuntimeError('Checkpoint mismatch')
        model=vit_b_16(weights=None).eval()
        model.load_state_dict(torch.load(assets['weights_file'],map_location='cpu',weights_only=True))
        model=model.to(dtype=torch.bfloat16)
        if state_hash(model.state_dict())!=assets['state_dict_bf16_cpu_sha256']:raise RuntimeError('Model state mismatch')
        cpu=torch.load(assets['batches'][str(spec['batch'])]['path'],map_location='cpu',weights_only=True)
        if tensor_hash(cpu)!=assets['batches'][str(spec['batch'])]['tensor_sha256']:raise RuntimeError('Input mismatch')
        r['weights_sha256']=assets['weights_sha256'];r['input_hash']=tensor_hash(cpu)
        with ft.config.cpu_fallback(False),ft.config.cache_context(str(folder/'cache'),summary=False),torch.inference_mode():
            stream=ft.current_stream(cfg['device'])
            t=time.perf_counter_ns();model=model.to(cfg['device']);x=cpu.to(cfg['device']);stream.synchronize()
            r['initial_model_and_input_transfer_wall_ms']=(time.perf_counter_ns()-t)/1e6
            def backend(gm,inputs):
                t=time.perf_counter_ns()
                try:return ft.backend(gm,inputs)
                finally:
                    r['backend_calls'].append({'wall_ms':(time.perf_counter_ns()-t)/1e6})
            runner=model if spec['mode']=='eager' else torch.compile(model,backend=backend)
            event('first_start')
            with Profiler(str(folder/'first.json')):
                with RecordFunction('first_forward_includes_export_compile_cache_load'):
                    t=time.perf_counter_ns();y=runner(x);stream.synchronize()
                    r['first_forward_wall_ms']=(time.perf_counter_ns()-t)/1e6
            out=y.cpu().float()
            if list(out.shape)!=[spec['batch'],1000] or not torch.isfinite(out).all():raise RuntimeError('Invalid output')
            torch.save(out,folder/'output.pt');r['output_finite']=True
            event('first_end',ms=r['first_forward_wall_ms'])
            for i in range(spec['warmup']):
                t=time.perf_counter_ns();y=runner(x);stream.synchronize();r['warmup_ms'].append((time.perf_counter_ns()-t)/1e6)
            event('warmup_end',calls=spec['warmup'])
            r['backend_calls_before_samples']=len(r['backend_calls'])
            for i in range(spec['samples']):
                row={'iteration':i+1}
                # Separate captures prevent device-clock/host-clock alignment assumptions.
                with Profiler(str(folder/f'h2d_{i+1:02d}.json')):
                    with RecordFunction('input_h2d_with_allocation_and_wait'):
                        t=time.perf_counter_ns();new_x=cpu.to(cfg['device']);stream.synchronize()
                        row['h2d_wall_ms']=(time.perf_counter_ns()-t)/1e6
                with Profiler(str(folder/f'forward_{i+1:02d}.json')):
                    with RecordFunction('resident_forward_no_external_copies'):
                        t=time.perf_counter_ns();y=runner(new_x);stream.synchronize()
                        row['forward_wall_ms']=(time.perf_counter_ns()-t)/1e6
                with Profiler(str(folder/f'd2h_{i+1:02d}.json')):
                    with RecordFunction('output_d2h_with_allocation_and_wait'):
                        t=time.perf_counter_ns();out=y.cpu();stream.synchronize()
                        row['d2h_wall_ms']=(time.perf_counter_ns()-t)/1e6
                if not torch.isfinite(out.float()).all():raise RuntimeError('Nonfinite output after capture')
                row['serial_request_wall_sum_ms']=sum(row[k] for k in ['h2d_wall_ms','forward_wall_ms','d2h_wall_ms'])
                r['samples'].append(row);del new_x;event('sample_end',**row)
            r['backend_calls_during_samples']=len(r['backend_calls'])-r['backend_calls_before_samples']
            r['stream_complete']=stream.query()
            if not r['stream_complete'] or ft.config.get_cpu_fallback() or torch.backends.mha.get_fastpath_enabled():raise RuntimeError('Completion/control failure')
            r['status']='ok'
    except BaseException:
        r['status']='error';r['error']=traceback.format_exc();traceback.print_exc()
    finally:
        r['ended_utc']=now();save()
    return 0 if r['status']=='ok' else 1

if __name__=='__main__':sys.exit(main())
