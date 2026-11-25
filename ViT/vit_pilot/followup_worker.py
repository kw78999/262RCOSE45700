"""Follow-up worker; reuses original ViT preparation and the unchanged call timer."""
import argparse
import contextlib
import cProfile
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from common import Events, now, read_json, write_json, sha256, timed_call


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--case',required=True)
    a=p.parse_args()
    cfg=read_json(a.run/'config.json')
    spec=next(x for x in read_json(a.run/'manifest.json')['workers'] if x['case_id']==a.case)
    folder=a.run/'cases'/a.case; folder.mkdir(parents=True,exist_ok=False)
    ev=Events(folder/'events.jsonl')
    diagnostic=spec['stage']=='diagnostic'
    r=dict(case_id=a.case,stage=spec['stage'],mode=spec['mode'],batch=spec['batch'],trial=spec['trial'],
        status='running',started_utc=now(),timings=[],observations=[],backend_entries=[],active_call=None,
        input_hash=None,weights_sha256=None,dtype='torch.bfloat16',sync_checked=False,
        timing_scope='cpu_dispatch_runner_and_current_stream_completion_excludes_external_prepost_transfers',
        code_sha256={name:sha256(Path(__file__).parent/name) for name in ['followup_worker.py','common.py','vit.py']},
        actual_codegen_count=None,disk_cache_hit=None,disk_cache_miss=None,
        compiler_time_ms=None,unknown_reason='No independent actual codegen/disk hit-miss instrumentation')
    def save(): write_json(folder/'result.json',r)
    save()
    try:
        logging.basicConfig(level=logging.WARNING,format='%(asctime)s %(levelname)s %(name)s %(message)s')
        if spec['stage']!='performance': logging.getLogger('furiosa_torch').setLevel(logging.DEBUG)
        import torch
        import furiosa.torch as ft
        from vit import prepare_model_inputs
        if not ft.is_available() or cfg['device'] not in ft.get_available_devices():
            raise RuntimeError('Requested Furiosa device unavailable')
        if torch._dynamo.config.suppress_errors or ft.backend.eager_fallback:
            raise RuntimeError('Error suppression or backend eager fallback must not be enabled')
        assets_dir=Path(spec['assets_dir'])
        assets=read_json(assets_dir/'assets.json')
        r.update(input_hash=assets['batch_hashes'][str(spec['batch'])],weights_sha256=assets['weights_sha256'],
                 assets_dir=str(assets_dir),input_file_sha256=assets['input_file_sha256'],
                 synthetic_input=assets['synthetic_input'])
        cache=Path(spec['cache_path'])
        r['cache_path']=str(cache); r['cache_existed_at_entry']=cache.exists()
        if spec['stage']=='performance' and cache.exists():
            raise RuntimeError('Performance requires a new worker-specific SDK cache path')
        r['cache_state']='isolated_sdk_cache_path_initially_absent' if not cache.exists() else 'stage_local_sdk_cache_reuse'
        r['effective']=dict(cpu_fallback_original=ft.config.get_cpu_fallback(),
            compiler_config=str(ft.config.get_compiler_config()),backend_eager_fallback=ft.backend.eager_fallback,
            device_configuration=str(ft.get_device_configuration()),torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
            torch_compile_kwargs='backend only; dynamic/fullgraph/mode/options omitted',
            suppress_errors=torch._dynamo.config.suppress_errors,
            dynamic_shapes=torch._dynamo.config.dynamic_shapes,
            automatic_dynamic_shapes=torch._dynamo.config.automatic_dynamic_shapes,
            recompile_limit=getattr(torch._dynamo.config,'recompile_limit',None))
        context=ft.config.cpu_fallback(False) if spec['stage']=='strict' else contextlib.nullcontext()
        with context, ft.config.cache_context(str(cache),summary=False), torch.inference_mode():
            r['effective']['cpu_fallback']=ft.config.get_cpu_fallback()
            r['effective']['cache_config']=str(ft.config.get_cache_config())
            ev.emit('prepare_start',stage=spec['stage'],mode=spec['mode'],batch=spec['batch'])
            t=time.perf_counter()
            model,xs=prepare_model_inputs(assets_dir,[spec['batch']],torch.device(cfg['device']))
            x=xs[spec['batch']]
            stream=ft.current_stream(cfg['device']); stream.synchronize()
            if not stream.query(): raise RuntimeError('Completion query false after official synchronize')
            r.update(sync_checked=True,stream_id=stream.stream_id,preparation_ms=(time.perf_counter()-t)*1000,
                input_metadata=dict(shape=list(x.shape),dtype=str(x.dtype),device=str(x.device)),
                model_metadata=dict(training=model.training,parameter_dtype=str(next(model.parameters()).dtype),
                                    parameter_device=str(next(model.parameters()).device)))
            ev.emit('prepare_end',**{k:r[k] for k in ['input_metadata','model_metadata','preparation_ms','stream_id']})
            from environment import command
            r['device_use_snapshot']=command(['furiosa-smi','ps'])
            active=None
            def backend(gm,example_inputs):
                entry=dict(call_index=active,backend_entry_utc=now(),nodes=len(list(gm.graph.nodes)))
                r['backend_entries'].append(entry);ev.emit('backend_entry',**entry)
                return ft.backend(gm,example_inputs)
            t=time.perf_counter_ns()
            runner=model if spec['mode']=='eager' else torch.compile(model,backend=backend if diagnostic else ft.backend)
            r['runner_setup_ms']=(time.perf_counter_ns()-t)/1e6
            ev.emit('runner_ready',runner_setup_ms=r['runner_setup_ms']);save()
            schedule=[('first',1)]
            if diagnostic:
                schedule += [('warmup',i) for i in range(1,6)] + [('repeat',i) for i in range(1,4)]
            elif spec['stage']=='performance':
                schedule += [('warmup',i) for i in range(1,cfg['warmup']+1)]
                schedule += [('repeat',i) for i in range(1,cfg['repeats']+1)]
            def lru():
                if not diagnostic: return None
                from furiosa.torch._backend import aten
                return {name:getattr(aten,name).cache_info()._asdict() for name in
                    ['_build_exported_program_cached','_compile_cached','_compile_and_load_cached']}
            last=None
            for index,(phase,iteration) in enumerate(schedule,1):
                active=index
                row=dict(run_id=cfg['run_id'],case_id=a.case,mode=spec['mode'],batch=spec['batch'],trial=spec['trial'],
                         phase=phase,iteration=iteration,status='running',elapsed_ms=None,timing_valid=False)
                r['active_call']=row; save();before=lru()
                ev.emit('call_start',call_index=index,phase=phase,iteration=iteration,batch=spec['batch'])
                if diagnostic:
                    print('FOLLOWUP_CALL_BEGIN '+json.dumps(dict(call_index=index,phase=phase)),file=sys.stderr,flush=True)
                try:
                    last,elapsed,valid=timed_call(runner,x,stream.synchronize)
                except Exception as error:
                    row.update(status='error',error=repr(error));r['timings'].append(row)
                    ev.emit('call_error',call_index=index,error=repr(error));raise
                row.update(status='ok',elapsed_ms=elapsed,timing_valid=valid)
                r['timings'].append(row);r['active_call']=None
                ev.emit('call_end',call_index=index,phase=phase,elapsed_ms=elapsed)
                if diagnostic:
                    print(f'FOLLOWUP_CALL_END {index}',file=sys.stderr,flush=True)
                    r['observations'].append(dict(call_index=index,phase=phase,lru_before=before,lru_after=lru(),
                        backend_entries_cumulative=len(r['backend_entries'])))
                save()
            # Save only the last non-profiled output. All host copies are outside timers.
            cpu=last.detach().to('cpu',non_blocking=False).float();stream.synchronize()
            metadata=dict(shape=list(last.shape),dtype=str(last.dtype),device=str(last.device),
                          nan_count=int(torch.isnan(cpu).sum()),inf_count=int(torch.isinf(cpu).sum()))
            r['output_metadata']=metadata
            r['output_valid']=metadata['shape']==[spec['batch'],1000] and metadata['dtype']=='torch.bfloat16' and not (metadata['nan_count'] or metadata['inf_count'])
            target=a.run/'outputs'/f'{a.case}.pt';torch.save(cpu,target);r['output_file']=str(target)
            if not r['output_valid']: raise RuntimeError('Invalid output shape/dtype/NaN/Inf')
            if diagnostic:
                # One extra representative call AFTER the nine diagnostic calls.
                # No profile, monkey patch, wrapper or hook is installed in performance workers.
                from furiosa.torch.profiler import Profiler,RecordFunction
                ev.emit('profile_start',note='Extra call; excluded from all diagnostic/performance timing rows')
                cp=cProfile.Profile()
                with Profiler(str(folder/'furiosa_trace.json')):
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as tp:
                        cp.enable()
                        with RecordFunction('followup_representative_call'):
                            z=runner(x);stream.synchronize()
                        cp.disable()
                cp.dump_stats(str(folder/'python_profile.pstats'))
                tp.export_chrome_trace(str(folder/'torch_cpu_trace.json'))
                import pstats
                with (folder/'python_profile.txt').open('w') as out:
                    pstats.Stats(cp,stream=out).sort_stats('cumulative').print_stats(50)
                profile_summary=[dict(name=e.key,count=e.count,self_cpu_ms=e.self_cpu_time_total/1000,
                                      total_cpu_ms=e.cpu_time_total/1000) for e in tp.key_averages()]
                write_json(folder/'torch_cpu_summary.json',profile_summary)
                ev.emit('profile_end')
            r['status']='ok'
        r['cpu_fallback_after_context']=ft.config.get_cpu_fallback()
    except Exception:
        r['status']='error';r['error']=traceback.format_exc();traceback.print_exc();ev.emit('worker_error',error=r['error'])
    finally:
        r['ended_utc']=now();save();ev.emit('worker_end',status=r['status'])
    return 0 if r['status']=='ok' else 1


if __name__=='__main__': raise SystemExit(main())
