"""Read-only inventory and installed source evidence. No package installation."""
import importlib.metadata as md
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from common import now, write_json, sha256


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except Exception as e:
        return dict(command=args, error=repr(e))


def collect(run):
    packages = {}
    for name in ['torch', 'torchvision', 'furiosa-torch', 'furiosa-kernels', 'furiosa-runtime',
                 'furiosa-sdk', 'numpy', 'Pillow', 'matplotlib']:
        try:
            packages[name] = dict(version=md.version(name), requires=md.requires(name))
        except md.PackageNotFoundError:
            packages[name] = None
    env = dict(utc=now(), executable=sys.executable, python=sys.version, os=platform.platform(),
               prefix=sys.prefix, base_prefix=sys.base_prefix, packages=packages,
               environment={k:v for k,v in os.environ.items() if k.startswith(('FURIOSA', 'TORCH_LOGS'))
                            or k in ['VIRTUAL_ENV', 'CONDA_PREFIX', 'TORCH_HOME', 'XDG_CACHE_HOME', 'RUST_LOG']},
               environment_ownership='unknown; existing environment left unchanged',
               commands=[command(c) for c in [['furiosa-smi','version'], ['furiosa-smi','info'],
                   ['furiosa-smi','status'], ['furiosa-smi','ps'], ['dpkg-query','-W','*furiosa*']]])
    sources = []
    try:
        import torch
        env['torch_import'] = dict(ok=True, version=torch.__version__, path=torch.__file__)
        try:
            import torchvision
            env['torchvision_import'] = dict(ok=True, version=torchvision.__version__, path=torchvision.__file__)
        except Exception:
            env['torchvision_import'] = dict(ok=False, error=traceback.format_exc())
        import furiosa.torch as ft
        env['furiosa'] = dict(version=ft.version(), available=ft.is_available(),
            devices=ft.get_available_devices(), device_configuration=str(ft.get_device_configuration()),
            cpu_fallback_default=ft.config.get_cpu_fallback(), compiler_config=str(ft.config.get_compiler_config()),
            cache_config=str(ft.config.get_cache_config()))
        registry = torch._logging._internal.log_registry
        env['torch_logs_supported'] = {k: k in registry.artifact_names for k in ['recompiles', 'graph_breaks']}
        root = Path(ft.__file__).parent
        paths = [root/p for p in ['__init__.py','_backend/torch_compile.py','_backend/eager.py',
                 '_backend/aten.py','_backend/_exported_program.py','config/_cache.py',
                 'config/_stream.py','config/_env.py','_C/stream/__init__.pyi', 'config/_cpu_fallback.py']]
        paths += [Path(torch.__file__).parent/'_logging/_registrations.py']
        if env.get('torchvision_import',{}).get('ok'):
            paths += [Path(torchvision.__file__).parent/'models/vision_transformer.py']
        with (run/'logs/installed_sources.txt').open('w') as out:
            for p in paths:
                if not p.exists():
                    continue
                sources.append(dict(path=str(p), sha256=sha256(p)))
                out.write(f'\n### {p}\n')
                out.writelines(f'{i}: {line}\n' for i,line in enumerate(p.read_text().splitlines(),1))
        env['sources'] = sources
    except Exception:
        env['import_error'] = traceback.format_exc()
    write_json(run/'environment.json', env)
    return env
