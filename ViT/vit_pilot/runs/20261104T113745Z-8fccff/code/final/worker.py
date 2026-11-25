"""One process, one mode; smoke/fixed/switch are separate invocations."""
import argparse
import logging
from pathlib import Path
import time
import traceback
import sys
import json
from common import Events, now, read_json, write_json, timed_call


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--mode', choices=['eager','compile'], required=True)
    p.add_argument('--kind', choices=['smoke','fixed','switch'], required=True)
    p.add_argument('--batch', type=int, choices=[1,4], default=1)
    p.add_argument('--trial', type=int, default=1)
    a = p.parse_args()
    c = read_json(a.run/'config.json')
    case_dir = a.run/'cases'/a.case
    case_dir.mkdir(parents=True, exist_ok=False)
    ev = Events(case_dir/'events.jsonl')
    result = dict(case_id=a.case, mode=a.mode, kind=a.kind, batch=a.batch, trial=a.trial,
                  status='running', started_utc=now(), timings=[], calls=[],
                  compiler_time_ms=None, compiler_time_reason='SDK codegen duration not independently observed',
                  execution_path_status='unknown', cache_hit=None, cache_miss=None,
                  sync_method='furiosa.torch.current_stream(device).synchronize()',
                  timing_scope='cpu_dispatch_model_call_and_current_stream_completion_excludes_transfers',
                  cache_state='fresh_process_persistent_cache_possible')
    def save():
        write_json(case_dir/'result.json', result)
    save()
    try:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(name)s %(message)s')
        if a.kind in ['smoke','switch']:
            logging.getLogger('furiosa_torch').setLevel(logging.DEBUG)
        import torch
        import furiosa.torch as ft
        from vit import prepare_model_inputs
        device = torch.device(c['device'])
        if not ft.is_available() or str(device) not in ft.get_available_devices():
            raise RuntimeError('Requested Furiosa device not available')
        if torch._dynamo.config.suppress_errors:
            raise RuntimeError('Existing suppress_errors=True is incompatible with this pilot')
        result['effective'] = dict(cpu_fallback=ft.config.get_cpu_fallback(),
            compiler_config=str(ft.config.get_compiler_config()), backend_eager_fallback=ft.backend.eager_fallback,
            torch_compile=dict(dynamic='default (omitted)', fullgraph='default (omitted)', mode='default (omitted)'),
            dynamo=dict(suppress_errors=torch._dynamo.config.suppress_errors,
                dynamic_shapes=torch._dynamo.config.dynamic_shapes,
                automatic_dynamic_shapes=torch._dynamo.config.automatic_dynamic_shapes,
                recompile_limit=getattr(torch._dynamo.config,'recompile_limit',None)),
            original_cache_config=str(ft.config.get_cache_config()))
        # Documented SDK cache API; a single new run-local cache is shared sequentially.
        cache = a.run/'compiler-cache'
        with ft.config.cache_context(str(cache), summary=False):
            result['effective']['cache_config'] = str(ft.config.get_cache_config())
            result['cache_path'] = str(cache)
            ev.emit('prepare_start', kind=a.kind, mode=a.mode, batch=a.batch)
            t = time.perf_counter()
            with torch.inference_mode():
                batches = [1,4] if a.kind == 'switch' else [a.batch]
                model, inputs = prepare_model_inputs(a.run, batches, device)
                stream = ft.current_stream(device)
                stream.synchronize()
                if not stream.query():
                    raise RuntimeError('Official stream synchronize/query completion check failed')
                result['preparation_ms'] = (time.perf_counter()-t)*1000
                result['sync_checked'] = True
                result['stream_id'] = stream.stream_id
                ev.emit('prepare_end', elapsed_ms=result['preparation_ms'], stream_id=stream.stream_id,
                        input_metadata={str(b):dict(shape=list(x.shape),dtype=str(x.dtype),device=str(x.device))
                                        for b,x in inputs.items()}, training=model.training)
                backend_count = 0
                active = None
                def observed_backend(gm, example_inputs):
                    nonlocal backend_count
                    backend_count += 1
                    ev.emit('backend_entry', call_index=active, ordinal=backend_count,
                            nodes=len(list(gm.graph.nodes)), unit='Dynamo graph/backend invocation; not codegen')
                    return ft.backend(gm, example_inputs)
                # Performance uses SDK backend directly; diagnostic wrapper only forwards unchanged graph.
                t = time.perf_counter_ns()
                runner = model if a.mode == 'eager' else torch.compile(
                    model, backend=observed_backend if a.kind == 'switch' else ft.backend)
                result['runner_setup_ms'] = (time.perf_counter_ns()-t)/1e6
                ev.emit('runner_ready', runner_setup_ms=result['runner_setup_ms'])
                save()
                if a.kind == 'switch':
                    schedule = [('switch', i, b) for i,b in enumerate([1,1,4,4,1,4],1)]
                elif a.kind == 'smoke':
                    schedule = [('smoke',1,a.batch)]
                else:
                    schedule = [('first',1,a.batch)] + [('warmup',i,a.batch) for i in range(1,c['warmup']+1)]
                    schedule += [('repeat',i,a.batch) for i in range(1,c['repeats']+1)]
                last = None
                def lru():
                    if a.kind != 'switch':
                        return None
                    from furiosa.torch._backend import aten
                    return {n:getattr(aten,n).cache_info()._asdict() for n in
                        ['_build_exported_program_cached','_compile_cached','_compile_and_load_cached']}
                for idx,(phase,iteration,b) in enumerate(schedule,1):
                    active = idx
                    before = lru()
                    start = ev.emit('call_start', call_index=idx, phase=phase, iteration=iteration, batch=b)
                    if a.kind == 'switch':
                        print('VIT_CALL_BEGIN '+json.dumps(dict(call_index=idx,batch=b,utc=start['utc'])),file=sys.stderr,flush=True)
                    row = dict(run_id=c['run_id'],case_id=a.case,mode=a.mode,batch=b,trial=a.trial,
                               phase=phase,iteration=iteration,elapsed_ms=None,timing_valid=False,status='running')
                    result['active_call'] = row
                    save()
                    try:
                        y, elapsed, valid = timed_call(runner,inputs[b],stream.synchronize)
                    except Exception as e:
                        row.update(status='error',error=repr(e))
                        result['timings'].append(row)
                        ev.emit('call_error',call_index=idx,error=repr(e),elapsed_since_start_ms=
                                (time.perf_counter_ns()-start['monotonic_ns'])/1e6)
                        if a.kind == 'switch':
                            print(f'VIT_CALL_END {idx} error',file=sys.stderr,flush=True)
                        raise
                    # Everything below is outside the model timer.
                    row.update(elapsed_ms=elapsed,timing_valid=valid,status='ok')
                    end = ev.emit('call_end',call_index=idx,batch=b,phase=phase,elapsed_ms=elapsed)
                    if a.kind == 'switch':
                        print(f'VIT_CALL_END {idx} ok',file=sys.stderr,flush=True)
                    result['timings'].append(row)
                    if a.kind == 'switch':
                        result['calls'].append(dict(call_index=idx,batch=b,elapsed_ms=elapsed,timing_valid=valid,
                            status='ok',start_utc=start['utc'],end_utc=end['utc'],
                            start_monotonic_ns=start['monotonic_ns'],end_monotonic_ns=end['monotonic_ns'],
                            backend_calls_cumulative=backend_count,lru_before=before,lru_after=lru(),
                            pytorch_recompile_events=None,guard_failure_reason=None,
                            furiosa_actual_compile_count=None,disk_cache_hit=None,disk_cache_miss=None,
                            cache_state='same_process_reuse' if idx > 1 else result['cache_state']))
                    last = y  # Retain just one output; never an iteration-sized tensor list.
                    result['active_call'] = None
                    save()
                # CPU copies, finite tests, persistence are outside all call timers.
                out = last.detach().to('cpu',non_blocking=False).float()
                stream.synchronize()
                metadata = dict(shape=list(last.shape),dtype=str(last.dtype),device=str(last.device),
                    nan_count=int(torch.isnan(out).sum()),inf_count=int(torch.isinf(out).sum()))
                correct_shape = metadata['shape'] == [schedule[-1][2],1000]
                result['output_metadata'] = metadata
                result['output_structure_valid'] = correct_shape and metadata['nan_count']==0 and metadata['inf_count']==0
                result['dtype_valid'] = last.dtype == torch.bfloat16
                if a.kind != 'switch':
                    target = a.run/'outputs'/f'{a.case}.pt'
                    torch.save(out,target)
                    result['output_file'] = str(target)
                if not result['output_structure_valid'] or not result['dtype_valid']:
                    raise RuntimeError('Invalid output shape/finite values/BF16 dtype')
                result['status'] = 'ok'
        ev.emit('worker_end',status='ok')
    except Exception:
        result['status'] = 'error'
        result['error'] = traceback.format_exc()
        traceback.print_exc()
        ev.emit('worker_error',error=result['error'])
    finally:
        result['ended_utc'] = now()
        save()
    return 0 if result['status']=='ok' else 1


if __name__=='__main__':
    raise SystemExit(main())
