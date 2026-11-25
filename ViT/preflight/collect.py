"""Read-only inventory and installed source evidence; no NPU computation."""
import importlib.metadata as md
import os, pathlib, platform, shutil, subprocess, sys
from datetime import datetime, timezone
root = pathlib.Path(__file__).resolve().parent
log = (root/'logs/environment.log').open('w')
def emit(*a):
 print(*a,flush=True); print(*a,file=log,flush=True)
emit('UTC',datetime.now(timezone.utc).isoformat())
emit('executable',sys.executable,'prefix',sys.prefix,'base_prefix',sys.base_prefix)
emit('python',sys.version,'platform',platform.platform())
for k in ['VIRTUAL_ENV','CONDA_PREFIX','PATH','TORCH_HOME','XDG_CACHE_HOME']:
 emit(k,os.environ.get(k))
for k,v in sorted(os.environ.items()):
 if k.startswith(('FURIOSA','TORCH_LOGS','TORCH_COMPILE','RUST_LOG')): emit(k,v)
for n in ['torch','torchvision','furiosa-torch','furiosa-kernels','furiosa-runtime','furiosa-sdk']:
 try: emit('package',n,md.version(n))
 except md.PackageNotFoundError: emit('package',n,'NOT INSTALLED')
for cmd in [['furiosa-smi','version'],['furiosa-smi','info'],['furiosa-smi','status'],['furiosa-smi','ps'],['python','-m','pip','list'],['dpkg-query','-W','*furiosa*']]:
 emit('$', ' '.join(cmd))
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
 emit(p.stdout,'exit',p.returncode)
import torch, furiosa.torch as ft
emit('torch',torch.__version__,'cuda available',torch.cuda.is_available())
emit('furiosa runtime/compiler build',ft.version())
for n in ['is_available','get_available_devices','get_device_configuration','get_amp_supported_dtype']:
 try: emit(n,getattr(ft,n)())
 except Exception as e: emit(n,repr(e))
emit('cpu_fallback',ft.config.get_cpu_fallback())
emit('cache',ft.config.get_cache_config())
emit('compiler',ft.config.get_compiler_config())
emit('torch.profiler.supported_activities',torch.profiler.supported_activities())
from furiosa.torch._backend import aten
for n in ['_build_exported_program_cached','_compile_cached','_compile_and_load_cached']:
 emit('eager lru',n,getattr(aten,n).cache_info())
base=pathlib.Path(ft.__file__).parent
files=['__init__.py','_backend/eager.py','_backend/aten.py','_backend/torch_compile.py','_backend/_exported_program.py','_backend/runnable_sequence.py','_module/dfg.py','_module/runnable.py','config/__init__.py','config/_env.py','config/_cache.py','config/_stream.py','_C/stream/__init__.pyi','_C/profiler/__init__.pyi','_C/config/cache/__init__.pyi','_C/config/compiler/__init__.pyi','_C/device/__init__.pyi','_custom_ops/copy_from.py','_register.py']
with (root/'evidence/installed_sources.txt').open('w') as out:
 for f in files:
  p=base/f; out.write(f'\n### {p}\n')
  for i,line in enumerate(p.read_text().splitlines(),1): out.write(f'{i:4}: {line}\n')
 emit('source evidence',str(base),files)
 for p in [pathlib.Path(torch.__file__).parent/'_logging/_registrations.py',pathlib.Path('/root/torchTest.py')]:
  out.write(f'\n### {p}\n')
  for i,line in enumerate(p.read_text().splitlines(),1): out.write(f'{i:4}: {line}\n')
for p in [pathlib.Path(torch.hub.get_dir())/'checkpoints',pathlib.Path(ft.config.get_cache_config().root_dir)]:
 emit('existing cache',str(p),'exists',p.exists())
 if p.exists(): emit('top-level entries',sorted(x.name for x in p.iterdir()))
