"""Serial single-model Fusion scaling worker. Performance loops never copy logits."""
import argparse
import json
import logging
import os
from pathlib import Path
import signal
import statistics as st
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vit_pilot'))
from common import now, read_json, write_json, sha256, tensor_hash, timed_call
from fastpath_support import state_hash


def stabilization(times, cfg):
    n = cfg['window']
    if len(times) < max(cfg['min_warmup'], 3*n):
        return dict(stable=False, windows=[])
    windows = [times[-3*n:-2*n], times[-2*n:-n], times[-n:]]
    means = [st.mean(w) for w in windows]
    cvs = [st.stdev(w)/st.mean(w) for w in windows]
    drift = max(means)/min(means)-1
    return dict(stable=drift <= cfg['mean_change'] and max(cvs) <= cfg['cv'],
                window_means_ms=means, window_cvs=cvs, max_relative_mean_change=drift,
                heuristic_only=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--spec', type=Path, required=True); a=ap.parse_args()
    cfg=read_json(a.run/'experiment_config.json'); spec=read_json(a.spec)
    d=a.spec.parent
    r=dict(**spec, status='running', started_utc=now(), pid=os.getpid(), timings=[],
           compiler_time_ms=None, actual_codegen_count=None, disk_cache_hit=None,
           disk_cache_miss=None, backend_entries=[], controls=[], provisional=True,
           timing_scope='CPU dispatch + runner + current fused-device stream completion; external transfers excluded',
           system_cold=False, cache_miss_verified=False, performance_loop_output_copies=0)
    def save(): write_json(d/'result.json',r)
    def event(name, **kw):
        row=dict(utc=now(), pid=os.getpid(), event=name, **kw)
        with (d/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    def stop(sig, frame): raise TimeoutError('Own worker termination requested')
    signal.signal(signal.SIGTERM,stop); save()
    try:
        import torch
        import furiosa.torch as ft
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        from torchvision.models.vision_transformer import VisionTransformer
        logging.basicConfig(level=logging.WARNING)
        diagnostic=spec['stage']=='diagnostic'
        logging.getLogger('furiosa_torch').setLevel(logging.DEBUG if diagnostic else logging.WARNING)
        r['python_sdk_logger_level']=logging.getLogger('furiosa_torch').getEffectiveLevel()
        # Must precede any device creation or device-configuration read.
        fusion=ft.set_fusion(spec['pe'])
        r['fusion']={k:getattr(fusion,k) for k in ('num_fusion','num_pe_per_cluster','num_chips','num_clusters_per_chip')}
        if fusion.num_fusion != spec['pe']: raise RuntimeError('Fusion request mismatch')
        r['device_configuration']=str(ft.get_device_configuration())
        r['available_logical_devices']=ft.get_available_devices()
        if cfg['device'] not in r['available_logical_devices']: raise RuntimeError('Requested device unavailable')
        torch.set_num_threads(cfg['torch_threads']); torch.set_num_interop_threads(cfg['torch_interop_threads'])
        torch.backends.mha.set_fastpath_enabled(False)
        cache=Path(spec['cache_path'])
        r['cache_existed_at_entry']=cache.exists()
        if cache.exists(): raise RuntimeError('Refuse to reuse existing private attempt cache')
        if ft.backend.eager_fallback or torch._dynamo.config.suppress_errors: raise RuntimeError('Fallback/error suppression enabled')
        with ft.config.cpu_fallback(False), ft.config.cache_context(str(cache), summary=False), torch.inference_mode():
            r['effective']=dict(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                compiler_config=str(ft.config.get_compiler_config()), automatic_dynamic_shapes=torch._dynamo.config.automatic_dynamic_shapes,
                recompile_limit=getattr(torch._dynamo.config,'recompile_limit',None), backend_eager_fallback=ft.backend.eager_fallback,
                suppress_errors=torch._dynamo.config.suppress_errors, compile_kwargs='backend only; all other arguments omitted')
            def controls():
                v=dict(mha_fastpath=torch.backends.mha.get_fastpath_enabled(), cpu_fallback=ft.config.get_cpu_fallback())
                r['controls'].append(v)
                if any(v.values()): raise RuntimeError('Strict control changed')
            controls(); event('prepare_start'); t=time.perf_counter_ns()
            assets=read_json(a.run/'assets.json'); inp=assets['batches'][str(spec['batch'])]
            if sha256(assets['weights_file']) != assets['weights_sha256']: raise RuntimeError('Checkpoint hash mismatch')
            cpu=torch.load(inp['path'], map_location='cpu', weights_only=True)
            if tensor_hash(cpu)!=inp['tensor_sha256']: raise RuntimeError('Input hash mismatch')
            r.update(input_hash=inp['tensor_sha256'], unique_image_count=inp['unique_image_count'],
                     weights_sha256=assets['weights_sha256'])
            model=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval().to(dtype=torch.bfloat16)
            r['state_dict_bf16_cpu_sha256']=state_hash(model.state_dict())
            if r['state_dict_bf16_cpu_sha256']!=assets['state_dict_bf16_cpu_sha256']: raise RuntimeError('Model hash mismatch')
            if type(model) is not VisionTransformer: raise RuntimeError('Original model required')
            if any(type(b.self_attention) is not torch.nn.MultiheadAttention for b in model.encoder.layers): raise RuntimeError('Modified attention')
            if any(getattr(m,'_compiled_call_impl',None) is not None or m.training for m in model.modules()): raise RuntimeError('Model path changed')
            if any(p.dtype!=torch.bfloat16 for p in model.parameters()): raise RuntimeError('Model dtype changed')
            model=model.to(cfg['device']); x=cpu.to(cfg['device']); del cpu
            stream=ft.current_stream(cfg['device']); stream.synchronize()
            if not stream.query(): raise RuntimeError('Stream completion failed')
            r['model_input_prepare_ms']=(time.perf_counter_ns()-t)/1e6
            r['stream_id']=stream.stream_id; r['sync_contract']='installed logical-device current_stream contract; per-PE internal implementation opaque'
            controls(); active=None
            def observed_backend(gm, args):
                row=dict(index=len(r['backend_entries'])+1, call=active, utc=now())
                r['backend_entries'].append(row)
                (d/f"backend_{row['index']:02d}.txt").write_text(str(gm.graph))
                return ft.backend(gm,args)
            t=time.perf_counter_ns()
            runner=model if spec['mode']=='eager' else torch.compile(model,backend=observed_backend if diagnostic else ft.backend)
            r['runner_setup_ms']=(time.perf_counter_ns()-t)/1e6
            controls(); event('first_start'); save()
            def call(phase, i):
                nonlocal active
                active=dict(phase=phase, iteration=i)
                offset=(time.perf_counter_ns()-origin)/1e9
                y,ms,valid=timed_call(runner,x,stream.synchronize)
                if not valid: raise RuntimeError('Invalid timing')
                r['timings'].append(dict(phase=phase,iteration=i,elapsed_ms=ms,offset_s=offset))
                return y
            origin=time.perf_counter_ns()
            y=call('first',1); event('first_end', elapsed_ms=r['timings'][-1]['elapsed_ms']); save()
            if list(y.shape)!=[spec['batch'],1000] or y.dtype!=torch.bfloat16: raise RuntimeError('Invalid output metadata')
            if spec['stage'] in ('validation','diagnostic'):
                # A distinct validation/diagnostic worker, never a formal timing sample.
                out=y.detach().to('cpu', non_blocking=False).float(); stream.synchronize()
                r['output_check']=dict(shape=list(out.shape), nan_count=int(torch.isnan(out).sum()), inf_count=int(torch.isinf(out).sum()))
                if r['output_check']['nan_count'] or r['output_check']['inf_count']: raise RuntimeError('Nonfinite output')
                torch.save(out,d/'output.pt'); r['output_file']=str(d/'output.pt')
                if diagnostic:
                    from furiosa.torch.profiler import Profiler, RecordFunction
                    with Profiler(str(d/'furiosa_trace.json')):
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as tp:
                            with RecordFunction('vit_scaling_single_model'):
                                z=runner(x);stream.synchronize()
                    tp.export_chrome_trace(str(d/'torch_cpu_trace.json'))
                    cpu_rows=[dict(name=e.key,count=e.count,total_cpu_us=e.cpu_time_total) for e in tp.key_averages()]
                    write_json(d/'torch_cpu_summary.json',cpu_rows)
                    events=read_json(d/'furiosa_trace.json').get('traceEvents',[])
                    tasks=[e for e in events if '::Task::' in e.get('name','')]
                    r['path_observations']=dict(npu_task_events=len(tasks),
                        task_names=sorted(set(e['name'] for e in tasks)),
                        task_arg_configurations=[dict(t) for t in {tuple(sorted(e.get('args',{}).items())) for e in tasks}],
                        native_mha_host_count=sum(e['count'] for e in cpu_rows if e['name']=='aten::_native_multi_head_attention'),
                        backend_entry_count=len(r['backend_entries']),all_op_placement='unverified',
                        task_count_is_not_kernel_or_fusion_count=True)
            else:
                n=0; wt=[]; adaptive=spec['stage']=='stabilization'
                target=cfg['stabilization']['min_warmup'] if adaptive else spec['warmup']
                event('warmup_start',target=target)
                while n<target:
                    n+=1; y=call('warmup',n); wt.append(r['timings'][-1]['elapsed_ms'])
                    if adaptive and n==target:
                        assessment=stabilization(wt,cfg['stabilization']); r.setdefault('stabilization_checks',[]).append(dict(calls=n,**assessment))
                        if not assessment['stable'] and target<cfg['stabilization']['max_warmup']: target+=cfg['stabilization']['window']
                r['warmup_count']=n
                r['stabilization_status']=('heuristic_satisfied' if r['stabilization_checks'][-1]['stable'] else 'stabilization_unconfirmed') if adaptive else spec.get('stabilization_status','exploration')
                event('warmup_end',calls=n,sum_ms=sum(wt));save()
                if not adaptive:
                    event('repeat_start'); n=0; total_ms=0.; start=time.perf_counter(); block_start=start
                    while True:
                        n+=1; y=call('stability' if spec['stage']=='stability' else 'repeat',n); total_ms+=r['timings'][-1]['elapsed_ms']
                        if spec['stage']=='stability':
                            done=time.perf_counter()-start>=cfg['stability_seconds']
                        elif spec['stage']=='exploration': done=n>=cfg['exploration_repeats']
                        else: done=n>=cfg['repeat_min_calls'] and total_ms>=cfg['repeat_min_seconds']*1000
                        limit=cfg.get('stability_max_calls',10000000) if spec['stage']=='stability' else cfg['repeat_max_calls']
                        if done or n>=limit: break
                    r.update(repeat_count=n, repeat_sum_ms=total_ms,
                        repeat_minimum_met=(n>=cfg['repeat_min_calls'] and total_ms>=cfg['repeat_min_seconds']*1000) if spec['stage'] in ('formal','baseline') else None)
                    event('repeat_end',calls=n,sum_ms=total_ms)
            stream.synchronize()
            if not stream.query() or ft.current_stream(cfg['device']).stream_id!=r['stream_id']: raise RuntimeError('Stream completion/path changed')
            controls()
            r['python_sdk_logger_level_at_end']=logging.getLogger('furiosa_torch').getEffectiveLevel()
            if not diagnostic and r['python_sdk_logger_level_at_end']<logging.WARNING:raise RuntimeError('Detailed SDK Python logging unexpectedly enabled')
            r['status']='ok'
    except BaseException:
        r.update(status='error',error=traceback.format_exc()); traceback.print_exc()
    finally:
        r['ended_utc']=now(); save()
    return 0 if r['status']=='ok' else 1

if __name__=='__main__': sys.exit(main())
