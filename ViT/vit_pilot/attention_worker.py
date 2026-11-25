"""Isolated whole-ViT attention-path experiment; no SDK or model math patches."""
import argparse
import contextlib
import logging
from pathlib import Path
import time
import traceback
from common import Events, now, read_json, write_json, timed_call


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--case', required=True)
    a = p.parse_args()
    cfg = read_json(a.run/'config.json')
    spec = next(x for x in read_json(a.run/'manifest.json')['workers'] if x['case_id'] == a.case)
    folder = a.run/'cases'/a.case
    folder.mkdir(parents=True, exist_ok=False)
    ev = Events(folder/'events.jsonl')
    diagnostic = spec['stage'] == 'diagnostic'
    r = dict(**{k: spec[k] for k in ['case_id', 'mode', 'stage', 'batch', 'strict']},
             status='running', started_utc=now(), timings=[], backend_entries=[],
             actual_codegen_count=None, disk_cache_hit=None, disk_cache_miss=None,
             compiler_time_ms=None, output_valid=False, trial=1,
             timing_scope='cpu_dispatch_runner_and_current_stream_completion_excludes_external_prepost_transfers')
    def save():
        write_json(folder/'result.json', r)
    save()
    try:
        logging.basicConfig(level=logging.WARNING)
        if diagnostic:
            logging.getLogger('furiosa_torch').setLevel(logging.DEBUG)
        import torch
        import furiosa.torch as ft
        from vit import prepare_model_inputs
        assets_dir = Path(cfg['assets_dir'])
        assets = read_json(assets_dir/'assets.json')
        r.update(input_hash=assets['batch_hashes'][str(spec['batch'])],
                 weights_sha256=assets['weights_sha256'], input_file_sha256=assets['input_file_sha256'],
                 dtype='torch.bfloat16', synthetic_input=assets['synthetic_input'])
        if not ft.is_available() or cfg['device'] not in ft.get_available_devices():
            raise RuntimeError('Requested Furiosa device unavailable')
        if torch._dynamo.config.suppress_errors or ft.backend.eager_fallback:
            raise RuntimeError('Suppression/backend eager fallback must remain disabled')
        cache = Path(spec['cache_path'])
        if cache.exists():
            raise RuntimeError('Worker SDK cache must initially be absent; never remove old caches')
        r.update(cache_path=str(cache), cache_existed_at_entry=False,
                 cache_state='fresh_process_private_sdk_cache_initially_absent_system_cold_unverified')
        r['effective'] = dict(device_configuration=str(ft.get_device_configuration()),
            torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
            compiler_config=str(ft.config.get_compiler_config()),
            backend_eager_fallback=ft.backend.eager_fallback, suppress_errors=torch._dynamo.config.suppress_errors,
            automatic_dynamic_shapes=torch._dynamo.config.automatic_dynamic_shapes,
            recompile_limit=getattr(torch._dynamo.config, 'recompile_limit', None),
            cpu_fallback_original=ft.config.get_cpu_fallback(),
            mha_fastpath_original=torch.backends.mha.get_fastpath_enabled(),
            compile_kwargs='backend only; dynamic/fullgraph/mode/options/recompile_limit unchanged')
        original_fastpath = torch.backends.mha.get_fastpath_enabled()
        try:
            if spec['mode'] == 'eager_fastpath_off':
                torch.backends.mha.set_fastpath_enabled(False)
            strict = ft.config.cpu_fallback(False) if spec['strict'] else contextlib.nullcontext()
            with strict, ft.config.cache_context(str(cache), summary=False), torch.inference_mode():
                r['effective'].update(cpu_fallback=ft.config.get_cpu_fallback(),
                    mha_fastpath=torch.backends.mha.get_fastpath_enabled(), cache_config=str(ft.config.get_cache_config()))
                ev.emit('prepare_start', mode=spec['mode'], batch=spec['batch'])
                t = time.perf_counter_ns()
                model, xs = prepare_model_inputs(assets_dir, [spec['batch']], torch.device(cfg['device']))
                x = xs[spec['batch']]
                stream = ft.current_stream(cfg['device'])
                stream.synchronize()
                if not stream.query():
                    raise RuntimeError('Completion query false after synchronize')
                r.update(sync_checked=True, stream_id=stream.stream_id,
                    preparation_ms=(time.perf_counter_ns()-t)/1e6,
                    input_metadata=dict(shape=list(x.shape), dtype=str(x.dtype), device=str(x.device)),
                    model_training=model.training)
                # No forward before the first timed call. Capture only graphs/metadata in diagnostics.
                active = None
                def observed_backend(gm, example_inputs):
                    i = len(r['backend_entries']) + 1
                    entry = dict(index=i, call_index=active, backend_entry_utc=now(),
                                 targets=[str(n.target) for n in gm.graph.nodes if n.op.startswith('call_')])
                    r['backend_entries'].append(entry)
                    (folder/f'backend_{i:02d}.txt').write_text(str(gm.graph))
                    ev.emit('backend_entry', index=i, call_index=active)
                    return ft.backend(gm, example_inputs)
                backend = observed_backend if diagnostic else ft.backend
                t = time.perf_counter_ns()
                r['compiled_modules'] = []
                if spec['mode'] == 'attention_compile':
                    for name, block in model.encoder.layers.named_children():
                        # Public Module.compile keeps the same module, parameters, kwargs and state_dict.
                        block.self_attention.compile(backend=backend)
                        r['compiled_modules'].append(f'encoder.layers.{name}.self_attention')
                    if len(r['compiled_modules']) != 12:
                        raise RuntimeError('Expected all 12 existing attention modules')
                    runner = model
                elif spec['mode'] == 'full_compile':
                    runner = torch.compile(model, backend=backend)
                else:
                    runner = model
                r['runner_setup_ms'] = (time.perf_counter_ns()-t)/1e6
                ev.emit('prepare_end', preparation_ms=r['preparation_ms'])
                ev.emit('runner_ready', runner_setup_ms=r['runner_setup_ms'], compiled_modules=r['compiled_modules'])
                schedule = [('first', 1)]
                if not diagnostic:
                    schedule += [('warmup', i) for i in range(1, cfg['warmup']+1)]
                    schedule += [('repeat', i) for i in range(1, cfg['repeats']+1)]
                save()
                for index, (phase, iteration) in enumerate(schedule, 1):
                    active = index
                    row = dict(run_id=cfg['run_id'], case_id=a.case, mode=spec['mode'], batch=spec['batch'],
                               trial=1, phase=phase, iteration=iteration, elapsed_ms=None,
                               timing_valid=False, status='running')
                    r['active_call'] = row
                    save()
                    ev.emit('call_start', call_index=index, phase=phase, iteration=iteration)
                    try:
                        y, elapsed, valid = timed_call(runner, x, stream.synchronize)
                    except Exception as e:
                        row.update(status='error', error=repr(e))
                        r['timings'].append(row)
                        raise
                    row.update(status='ok', elapsed_ms=elapsed, timing_valid=valid)
                    r['timings'].append(row)
                    r['active_call'] = None
                    ev.emit('call_end', call_index=index, phase=phase, elapsed_ms=elapsed)
                    save()
                cpu = y.detach().to('cpu', non_blocking=False).float()
                stream.synchronize()
                r['output_metadata'] = dict(shape=list(y.shape), dtype=str(y.dtype), device=str(y.device),
                    nan_count=int(torch.isnan(cpu).sum()), inf_count=int(torch.isinf(cpu).sum()))
                m = r['output_metadata']
                r['output_valid'] = m['shape'] == [spec['batch'], 1000] and m['dtype'] == 'torch.bfloat16' and not (m['nan_count'] or m['inf_count'])
                if not r['output_valid']:
                    raise RuntimeError('Invalid output shape/dtype/NaN/Inf')
                target = a.run/'outputs'/f'{a.case}.pt'
                torch.save(cpu, target)
                r['output_file'] = str(target)
                if diagnostic:
                    # Extra profiled call; never part of the performance timing CSV.
                    from furiosa.torch.profiler import Profiler, RecordFunction
                    ev.emit('profile_start')
                    with Profiler(str(folder/'furiosa_trace.json')):
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as tp:
                            with RecordFunction('attention_path_representative_call'):
                                z = runner(x)
                                stream.synchronize()
                    tp.export_chrome_trace(str(folder/'torch_cpu_trace.json'))
                    write_json(folder/'torch_cpu_summary.json', [dict(name=e.key, count=e.count,
                        self_cpu_ms=e.self_cpu_time_total/1000, total_cpu_ms=e.cpu_time_total/1000) for e in tp.key_averages()])
                    ev.emit('profile_end')
                r['status'] = 'ok'
        finally:
            torch.backends.mha.set_fastpath_enabled(original_fastpath)
        r['settings_restored'] = dict(cpu_fallback=ft.config.get_cpu_fallback(),
                                      mha_fastpath=torch.backends.mha.get_fastpath_enabled())
    except Exception:
        r['status'] = 'error'
        r['error'] = traceback.format_exc()
        traceback.print_exc()
        ev.emit('worker_error', error=r['error'])
    finally:
        r['ended_utc'] = now()
        save()
        ev.emit('worker_end', status=r['status'])
    return 0 if r['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
