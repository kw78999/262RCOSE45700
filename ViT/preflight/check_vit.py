#!/usr/bin/env python3
"""Small RNGD preflight. Never installs packages, rewrites ops, or removes caches.
Default: ViT; --substrate: BF16 addition API check, never a ViT benchmark.
Use an external `timeout` to bound compiler/runtime waits.
"""
import argparse, contextlib, json, logging, pathlib, sys, time, traceback

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--substrate',action='store_true')
 p.add_argument('--device',default='furiosa:0')
 p.add_argument('--repeats',type=int,default=2,choices=range(1,4))
 p.add_argument('--run-dir',required=True,type=pathlib.Path)
 p.add_argument('--cache-dir',type=pathlib.Path,help='Explicit opt-in reuse of a private compiler cache')
 p.add_argument('--profile',action='store_true',help='Extra profiled calls AFTER unprofiled measurements')
 p.add_argument('--allow-weight-download',action='store_true')
 a=p.parse_args()
 a.run_dir.mkdir(parents=True,exist_ok=False)
 logging.basicConfig(level=logging.WARNING,format='%(levelname)s %(name)s %(message)s')
 logging.getLogger('furiosa_torch').setLevel(logging.DEBUG)
 import torch, furiosa.torch as ft
 from furiosa.torch._backend import aten
 records=[]
 def log(kind,**kw):
  row=dict(kind=kind,**kw); records.append(row); print(json.dumps(row,default=str),flush=True)
  (a.run_dir/'results.json').write_text(json.dumps(records,indent=2,default=str))
 log('scope',substrate=a.substrate,python=sys.executable,torch=torch.__version__,runtime=ft.version(),
     device=a.device,cpu_fallback=ft.config.get_cpu_fallback(),compiler=str(ft.config.get_compiler_config()),
     note='Preflight only; no accuracy pass criterion; no cold-cache claim')
 cache=(a.cache_dir or a.run_dir/'compiler-cache').resolve()
 log('cache',original=str(ft.config.get_cache_config()),selected=str(cache),already_exists=cache.exists())
 device=torch.device(a.device)
 if not ft.is_available() or a.device not in ft.get_available_devices():
  raise RuntimeError('Selected Furiosa device unavailable')
 if not a.substrate:
  # Fail on the real dependency error; never substitute weights=None or a custom ViT.
  from torchvision.models import vit_b_16, ViT_B_16_Weights
  weights=ViT_B_16_Weights.IMAGENET1K_V1
  from urllib.parse import urlparse
  checkpoint=pathlib.Path(torch.hub.get_dir())/'checkpoints'/pathlib.Path(urlparse(weights.url).path).name
  log('weights',enum=str(weights),url=weights.url,checkpoint=str(checkpoint),exists=checkpoint.is_file())
  if not checkpoint.is_file() and not a.allow_weight_download:
   raise RuntimeError('Exact weights absent; use --allow-weight-download explicitly to download')
  model=vit_b_16(weights=weights).eval().to(dtype=torch.bfloat16)
  log('model',name='torchvision.models.vit_b_16',training=model.training,dtype='bfloat16')
 else:
  model=None
 def caches():
  return {n:getattr(aten,n).cache_info()._asdict() for n in
   ['_build_exported_program_cached','_compile_cached','_compile_and_load_cached']}
 backend_calls=0
 def observed_backend(gm,inputs):
  nonlocal backend_calls
  backend_calls+=1
  log('dynamo_backend_call',count=backend_calls,nodes=len(list(gm.graph.nodes)))
  # Delegate untouched graph to the SDK default backend, with no compile options.
  return ft.backend(gm,inputs)
 def measure(fn,x,label):
  s=ft.current_stream(device)
  s.synchronize()  # Excludes input transfer and all previously queued work.
  before=caches(); start=time.perf_counter_ns()
  y=fn(x)
  returned=time.perf_counter_ns()
  complete_at_return=s.query()  # Observational; does NOT imply guaranteed synchronous forward.
  s.synchronize()
  done=time.perf_counter_ns()
  # CPU output transfer explicitly outside the timed model+completion interval.
  yc=y.to('cpu',non_blocking=False)
  s.synchronize()
  if not torch.isfinite(yc.float()).all(): raise RuntimeError('Nonfinite output')
  log('call',label=label,shape=list(y.shape),dtype=str(y.dtype),device=str(y.device),
      host_return_ms=(returned-start)/1e6,completion_ms=(done-start)/1e6,
      query_at_return=complete_at_return,backend_calls=backend_calls,lru_before=before,lru_after=caches())
  return yc
 with ft.config.cache_context(str(cache),summary=True):
  with torch.inference_mode():
   if model is not None: model=model.to(device)
   eager=(lambda x:x+x) if a.substrate else model
   compiled=torch.compile(eager,backend=observed_backend)
   # No dynamic/fullgraph/mode/options overrides; SDK and PyTorch defaults.
   torch.manual_seed(2026)
   cpu_inputs={b:torch.randn((b,16) if a.substrate else (b,3,224,224),dtype=torch.bfloat16) for b in [1,2]}
   saved={}
   for stage,b in [('batch1',1),('batch2',2),('return_batch1',1)]:
    x=cpu_inputs[b].to(device)
    for name,fn in [('eager',eager),('compile',compiled)]:
     for i in range(a.repeats+1 if stage=='batch1' else 1):
      saved[name]=measure(fn,x,f'{stage}/{name}/{i}')
    u=saved['eager'].float(); v=saved['compile'].float(); d=(u-v).abs()
    log('difference',stage=stage,max_abs=d.max().item(),mean_abs=d.mean().item(),
        max_relative_to_eager=d.div(u.abs().clamp_min(1e-8)).max().item(),
        rmse=(u-v).square().mean().sqrt().item(),pass_fail='UNASSESSED: tolerance not specified')
   if a.profile:
    x=cpu_inputs[1].to(device); s=ft.current_stream(device); s.synchronize()
    from furiosa.torch.profiler import Profiler,RecordFunction
    with Profiler(str(a.run_dir/'furiosa-trace.json')):
     for name,fn in [('eager',eager),('compile',compiled)]:
      with RecordFunction(name): y=fn(x); s.synchronize()
    activities=torch.profiler.supported_activities()
    with torch.profiler.profile(activities=list(activities)) as prof:
     with torch.profiler.record_function('rngd-preflight'):
      y=eager(x); s.synchronize()
    prof.export_chrome_trace(str(a.run_dir/'torch-trace.json'))
    log('torch_profiler',activities=str(activities),event_devices=sorted(set(str(e.device_type) for e in prof.events())))
 log('finished',status='success',scope='substrate only' if a.substrate else 'ViT')

if __name__=='__main__':
 try: main()
 except Exception:
  traceback.print_exc(); sys.exit(1)
