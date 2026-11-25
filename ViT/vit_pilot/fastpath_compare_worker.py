"""Original ViT, strict CPU fallback off and MHA fastpath off in BOTH modes."""
import argparse
import logging
from pathlib import Path
import signal
import time
import traceback
from common import Events, now, read_json, write_json, sha256, timed_call
from fastpath_support import CACHE_POLICY, TIMING_SCOPE, state_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True); p.add_argument('--case',required=True)
    a=p.parse_args(); cfg=read_json(a.run/'config.json')
    spec=next(w for w in read_json(a.run/'manifest.json')['workers'] if w['case_id']==a.case)
    folder=a.run/'cases'/a.case; folder.mkdir(parents=True,exist_ok=False)
    ev=Events(folder/'events.jsonl'); diagnostic=spec['stage']=='diagnostic'; cumulative=spec['stage']=='cumulative'
    r=dict(**{k:spec[k] for k in ['case_id','stage','mode','batch','trial']},status='running',started_utc=now(),
        timings=[],control_snapshots={},backend_entries=[],outputs={},output_checks={},output_valid=False,
        dtype='torch.bfloat16',cache_state=CACHE_POLICY,timing_scope=TIMING_SCOPE,compiler_time_ms=None,
        actual_codegen_count=None,disk_cache_hit=None,disk_cache_miss=None,
        script_sha256=sha256(Path(__file__)),inference_mode=True,active_call=None)
    def save(): write_json(folder/'result.json',r)
    def terminate(signum,frame): raise TimeoutError('Own worker received termination signal')
    signal.signal(signal.SIGTERM,terminate)
    save(); original_fastpath=None
    try:
        logging.basicConfig(level=logging.WARNING)
        if diagnostic: logging.getLogger('furiosa_torch').setLevel(logging.DEBUG)
        import torch
        import furiosa.torch as ft
        from vit import prepare_model_inputs
        original_fastpath=torch.backends.mha.get_fastpath_enabled()
        r['original_controls']=dict(mha_fastpath=original_fastpath,cpu_fallback=ft.config.get_cpu_fallback())
        torch.backends.mha.set_fastpath_enabled(False)
        if not ft.is_available() or cfg['device'] not in ft.get_available_devices(): raise RuntimeError('Allowed device unavailable')
        if torch._dynamo.config.suppress_errors or ft.backend.eager_fallback: raise RuntimeError('Error suppression/backend eager fallback must be disabled')
        assets=read_json(a.run/'assets.json'); verified=read_json(a.run/'asset_verification.json')
        r.update(input_hash=assets['batch_hashes'][str(spec['batch'])],weights_sha256=assets['weights_sha256'],
                 input_file_sha256=assets['input_file_sha256'],synthetic_input=False)
        cache=Path(spec['cache_path']); r.update(cache_path=str(cache),cache_existed_at_entry=cache.exists())
        if cache.exists(): raise RuntimeError('Never reuse or delete an existing worker cache')
        with ft.config.cpu_fallback(False),ft.config.cache_context(str(cache),summary=False),torch.inference_mode():
            r['effective']=dict(cpu_fallback=ft.config.get_cpu_fallback(),mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
                device_configuration=str(ft.get_device_configuration()),compiler_config=str(ft.config.get_compiler_config()),
                cache_config=str(ft.config.get_cache_config()),torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(),backend_eager_fallback=ft.backend.eager_fallback,
                suppress_errors=torch._dynamo.config.suppress_errors,automatic_dynamic_shapes=torch._dynamo.config.automatic_dynamic_shapes,
                recompile_limit=getattr(torch._dynamo.config,'recompile_limit',None),compile_kwargs='backend only; all other compile options omitted')
            def controls(label):
                value=dict(mha_fastpath=torch.backends.mha.get_fastpath_enabled(),cpu_fallback=ft.config.get_cpu_fallback())
                r['control_snapshots'][label]=value
                if any(value.values()): raise RuntimeError('Controls must stay false')
            ev.emit('model_input_prepare_start')
            start=time.perf_counter_ns()
            # Existing adapter prepares the unchanged model and tensor on CPU first.
            # Hash BF16 state on CPU BEFORE explicit device transfer; never forward here.
            model,xs=prepare_model_inputs(Path(cfg['assets_dir']),[spec['batch']],torch.device('cpu'))
            r['state_dict_bf16_cpu_sha256']=state_hash(model.state_dict())
            if r['state_dict_bf16_cpu_sha256']!=verified['state_dict_bf16_cpu_sha256']: raise RuntimeError('BF16 state_dict mismatch')
            from torchvision.models.vision_transformer import VisionTransformer
            if type(model) is not VisionTransformer: raise RuntimeError('Original VisionTransformer required')
            if any(type(b.self_attention) is not torch.nn.MultiheadAttention for b in model.encoder.layers): raise RuntimeError('Attention wrapper detected')
            if any(getattr(m,'_compiled_call_impl',None) is not None for m in model.modules()): raise RuntimeError('Previously compiled child detected')
            if any(m.training for m in model.modules()): raise RuntimeError('All modules must be eval')
            if any(p.dtype!=torch.bfloat16 for p in model.parameters()): raise RuntimeError('All model parameters must remain BF16')
            model=model.to(torch.device(cfg['device'])); x=xs[spec['batch']].to(torch.device(cfg['device']))
            stream=ft.current_stream(cfg['device']); stream.synchronize()
            if not stream.query(): raise RuntimeError('Official completion query failed')
            r.update(model_input_prepare_ms=(time.perf_counter_ns()-start)/1e6,sync_checked=True,stream_id=stream.stream_id,
                input_metadata=dict(shape=list(x.shape),dtype=str(x.dtype),device=str(x.device)),original_model_no_child_compile=True)
            controls('after_prepare');ev.emit('model_input_prepare_end',elapsed_ms=r['model_input_prepare_ms'])
            active=None
            def backend(gm,example_inputs):
                index=len(r['backend_entries'])+1
                entry=dict(index=index,call=active,entry_utc=now(),targets=[str(n.target) for n in gm.graph.nodes if n.op.startswith('call_')])
                r['backend_entries'].append(entry); (folder/f'backend_{index:02d}.txt').write_text(str(gm.graph))
                ev.emit('backend_entry',index=index,call=active)
                return ft.backend(gm,example_inputs)
            ev.emit('runner_setup_start');start=time.perf_counter_ns()
            runner=model if spec['mode']=='eager' else torch.compile(model,backend=backend if diagnostic else ft.backend)
            r['runner_setup_ms']=(time.perf_counter_ns()-start)/1e6
            ev.emit('runner_setup_end',elapsed_ms=r['runner_setup_ms'])
            controls('before_measure');save()
            schedule=[('first',1)]
            if not diagnostic:
                schedule += [('warmup',i) for i in range(1,cfg['warmup']+1)]
                phase='work' if cumulative else 'repeat';n=spec['target_calls'] if cumulative else cfg['repeats']
                schedule += [(phase,i) for i in range(1,n+1)]
            def check_output(y,key,store=True):
                cpu=y.detach().to('cpu',non_blocking=False).float();stream.synchronize()
                m=dict(shape=list(y.shape),dtype=str(y.dtype),device=str(y.device),
                    nan_count=int(torch.isnan(cpu).sum()),inf_count=int(torch.isinf(cpu).sum()))
                ok=m['shape']==[spec['batch'],1000] and m['dtype']=='torch.bfloat16' and not(m['nan_count'] or m['inf_count'])
                if not ok: raise RuntimeError('Invalid output '+key+': '+str(m))
                if store:
                    target=a.run/'outputs'/f'{a.case}_{key}.pt';torch.save(cpu,target)
                    r['outputs'][key]=str(target);r['output_checks'][key]=dict(**m,valid=True)
                    r['output_file']=str(target)
                return ok
            previous_phase=None;phase_started=None;phase_times=[]
            for call_index,(phase,iteration) in enumerate(schedule,1):
                active=dict(call_index=call_index,phase=phase,iteration=iteration)
                if phase!=previous_phase:
                    if previous_phase is not None:
                        ev.emit('phase_end',phase=previous_phase,wall_ms=(time.perf_counter_ns()-phase_started)/1e6,measured_sum_ms=sum(phase_times))
                    phase_started=time.perf_counter_ns();phase_times=[];previous_phase=phase
                    r['active_call']=active;save();ev.emit('phase_start',phase=phase,requested_calls=sum(p==phase for p,_ in schedule))
                row=dict(run_id=cfg['run_id'],case_id=a.case,stage=spec['stage'],mode=spec['mode'],batch=spec['batch'],trial=spec['trial'],
                    phase=phase,iteration=iteration,elapsed_ms=None,timing_valid=False,status='running')
                try:
                    y,elapsed,valid=timed_call(runner,x,stream.synchronize)
                except Exception as e:
                    row.update(status='error',error=repr(e));r['timings'].append(row);raise
                row.update(elapsed_ms=elapsed,elapsed_ns=round(elapsed*1000000),timing_valid=valid,status='ok')
                r['timings'].append(row);phase_times.append(elapsed)
                last_warm=phase=='warmup' and iteration==cfg['warmup']
                last_work=phase in ['repeat','work'] and iteration==n
                if phase=='first' or last_warm or last_work:
                    check_output(y,phase);save()
                elif cumulative and phase=='work':
                    # All cumulative work outputs checked, with no per-output files.
                    # CPU copy/check is outside the model timer and documented separately.
                    check_output(y,'work_check',store=False)
                if cumulative and phase=='work' and iteration%100==0:
                    r['active_call']=active;save();ev.emit('work_progress',completed=iteration,target=n,measured_sum_ms=sum(phase_times))
            ev.emit('phase_end',phase=previous_phase,wall_ms=(time.perf_counter_ns()-phase_started)/1e6,measured_sum_ms=sum(phase_times))
            r.update(active_call=None,output_valid=True,cumulative_work_outputs_checked=n if cumulative else None)
            controls('after_measure');save()
            if diagnostic:
                from furiosa.torch.profiler import Profiler,RecordFunction
                ev.emit('profile_start',note='Extra call excluded from all main and cumulative timing rows')
                with Profiler(str(folder/'furiosa_trace.json')):
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as tp:
                        with RecordFunction('fastpath_off_representative_call'):
                            z=runner(x);stream.synchronize()
                tp.export_chrome_trace(str(folder/'torch_cpu_trace.json'))
                cpu=[dict(name=e.key,count=e.count,self_cpu_ms=e.self_cpu_time_total/1000,total_cpu_ms=e.cpu_time_total/1000) for e in tp.key_averages()]
                write_json(folder/'torch_cpu_summary.json',cpu)
                events=read_json(folder/'furiosa_trace.json').get('traceEvents',[])
                r['path_observations']=dict(native_mha_host_count=sum(e['count'] for e in cpu if e['name']=='aten::_native_multi_head_attention'),
                    npu_task_events=sum('::Task::' in e.get('name','') for e in events),backend_entry_count=len(r['backend_entries']),
                    all_op_placement='unverified',actual_codegen_count=None,disk_cache_hit=None,disk_cache_miss=None)
                ev.emit('profile_end',**r['path_observations'])
            r['status']='ok'
        r['cpu_fallback_after_context']=ft.config.get_cpu_fallback()
    except Exception:
        r.update(status='error',error=traceback.format_exc());traceback.print_exc();ev.emit('worker_error',error=r['error'])
    finally:
        if original_fastpath is not None:
            import torch
            torch.backends.mha.set_fastpath_enabled(original_fastpath)
            r['mha_fastpath_restored']=torch.backends.mha.get_fastpath_enabled()
        r['ended_utc']=now();save();ev.emit('worker_end',status=r['status'])
    return 0 if r['status']=='ok' else 1


if __name__=='__main__': raise SystemExit(main())
