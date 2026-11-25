"""Serial, bounded attention-path research using existing ViT assets and timer."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid
from common import Events, now, read_json, write_json, sha256

ROOT = Path(__file__).resolve().parent
DEFAULT_ASSETS = ROOT/'followup_runs/20261104T132538Z-c031a6'
MODES = ['eager_default', 'attention_compile', 'eager_fastpath_off', 'full_compile']


def stop_own_worker(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def run(a):
    from environment import collect, command
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    folder = ROOT/'attention_runs'/run_id
    folder.mkdir(parents=True, exist_ok=False)
    for name in ['logs', 'cases', 'outputs', 'code', 'figures']:
        (folder/name).mkdir()
    cfg = dict(run_id=run_id, assets_dir=str(a.assets_dir.resolve()), device='furiosa:0',
               warmup=a.warmup, repeats=a.repeats, independent_trials=1,
               worker_timeout_s=a.worker_timeout, total_budget_s=a.total_budget,
               purpose='attention fallback mechanism and partial compile feasibility; preliminary single trial',
               atol=None, rtol=None, synthetic_input=False)
    write_json(folder/'config.json', cfg)
    assets = read_json(a.assets_dir/'assets.json')
    if assets['synthetic_input']:
        raise RuntimeError('This experiment requires the existing four real-image tensors')
    for name in ['assets.json', 'image_manifest.json']:
        shutil.copyfile(a.assets_dir/name, folder/name)
    sources = ['attention.py', 'attention_worker.py', 'attention_analysis.py', 'common.py', 'vit.py', 'environment.py']
    code_hashes = {name: sha256(ROOT/name) for name in sources}
    for name in sources:
        shutil.copyfile(ROOT/name, folder/'code'/name)
    m = dict(run_id=run_id, created_utc=now(), command=sys.argv, code_sha256=code_hashes,
             assets_dir=str(a.assets_dir.resolve()), weights_sha256=assets['weights_sha256'],
             input_hashes=assets['batch_hashes'], workers=[], npu_used_s=0, status='running',
             sync='furiosa.torch.current_stream(device).synchronize(); query checked after preparation',
             timing_scope='CPU dispatch through current-stream completion; internal transfers included; external pre/post transfers excluded',
             cache_condition='new private SDK cache per worker; system-wide cold and actual cache hit/miss unverified')
    # Strict default failure reproduction is B1-only; larger strict baseline is intentionally held.
    plan = [('diagnostic', 'eager_default', 1, True), ('diagnostic', 'eager_default', 1, False)]
    plan += [('diagnostic', mode, 1, True) for mode in ['attention_compile', 'eager_fastpath_off']]
    plan += [('diagnostic', 'eager_default', 4, False)]
    plan += [('diagnostic', mode, 4, True) for mode in ['attention_compile', 'eager_fastpath_off']]
    # New uninstrumented performance workers, including a fresh full-compile control.
    plan += [('performance', mode, b, False) for b in [1, 4] for mode in MODES]
    for stage, mode, b, strict in plan:
        case_id = f'{stage}_{mode}_b{b}'+('_strict' if strict else '')
        m['workers'].append(dict(case_id=case_id, stage=stage, mode=mode, batch=b, strict=strict,
            status='pending', cache_path=str(folder/'cache'/case_id)))
    write_json(folder/'manifest.json', m)
    ev = Events(folder/'logs/orchestrator.jsonl')
    ev.emit('run_start', run_dir=str(folder))
    collect(folder)
    # Additional installed-source snapshots with exact line numbers and hashes.
    import torch
    import furiosa.torch as ft
    installed = [Path(torch.__file__).parent/p for p in ['nn/modules/activation.py', 'nn/modules/module.py',
        '_dynamo/variables/torch.py', 'backends/mha/__init__.py']]
    installed += [Path(ft.__file__).parent/p for p in ['_db/local_decomp.py', 'export/passes.py']]
    m['additional_sources'] = [{'path': str(p), 'sha256': sha256(p)} for p in installed]
    with (folder/'logs/attention_installed_sources.txt').open('w') as f:
        for p in installed:
            f.write(f'\n### {p}\n')
            f.writelines(f'{i}: {line}\n' for i, line in enumerate(p.read_text().splitlines(), 1))
    write_json(folder/'logs/native_attention_dispatch.json', dict(torch=torch.__version__,
        privateuse1_specific_kernel=torch._C._dispatch_has_kernel_for_dispatch_key('aten::_native_multi_head_attention', 'PrivateUse1'),
        dispatch_table=torch._C._dispatch_dump_table('aten::_native_multi_head_attention')))
    def lookup(stage, mode, b, strict):
        return next(x for x in m['workers'] if (x['stage'], x['mode'], x['batch'], x['strict']) == (stage, mode, b, strict))
    for w in m['workers']:
        reason = None
        if w['batch'] == 4 and w['stage'] == 'diagnostic':
            base = lookup('diagnostic', w['mode'], 1, w['strict'])
            if base['status'] != 'ok' or base.get('validation_for_expansion') is False:
                reason = 'Corresponding batch-1 diagnostic did not succeed'
        if w['stage'] == 'performance' and w['mode'] in ['attention_compile', 'eager_fastpath_off']:
            base = lookup('diagnostic', w['mode'], w['batch'], True)
            if base['status'] != 'ok' or base.get('validation_for_expansion') is False:
                reason = 'Strict diagnostic failed; retain failure and do not expand this variant'
        if w['stage'] == 'performance' and w['batch'] == 4:
            b1 = lookup('performance', w['mode'], 1, False)
            if b1['status'] != 'ok' or b1.get('validation_for_expansion') is False:
                reason = 'Corresponding batch-1 performance did not succeed'
        if m['npu_used_s'] >= a.total_budget:
            reason = 'Overall NPU execution budget reached'
        if reason:
            w.update(status='held', reason=reason)
            write_json(folder/'manifest.json', m)
            continue
        snapshot = command(['furiosa-smi', 'ps'])
        write_json(folder/'logs'/f"{w['case_id']}.device_before.json", snapshot)
        if snapshot.get('returncode') != 0:
            w.update(status='held', reason='Could not inspect device ownership')
            continue
        # Conservatively hold when any NPU user is present. Never terminate other users.
        if any(line.strip().startswith('|') and line.strip().split('|')[1].strip().isdigit()
               for line in snapshot['stdout'].splitlines()):
            w.update(status='held', reason='NPU is in use; serial execution on idle device only')
            continue
        cmd = [sys.executable, '-u', str(ROOT/'attention_worker.py'), '--run', str(folder), '--case', w['case_id']]
        env = os.environ.copy()
        for k in ['TORCH_LOGS', 'FURIOSA_TORCH_LOG', 'FURIOSA_TORCH_PROFILE']:
            env.pop(k, None)
        if w['stage'] == 'diagnostic':
            if not all(read_json(folder/'environment.json')['torch_logs_supported'].values()):
                raise RuntimeError('Requested TORCH_LOGS keys are not supported')
            env.update(TORCH_LOGS='recompiles,graph_breaks', FURIOSA_TORCH_LOG='debug', FURIOSA_TORCH_PROFILE='trace')
        timeout = min(a.worker_timeout, a.total_budget-m['npu_used_s'])
        w.update(command=cmd, started_utc=now(), status='running', timeout_s=timeout)
        write_json(folder/'manifest.json', m)
        ev.emit('worker_start', case_id=w['case_id'], limit_s=timeout)
        t = time.perf_counter()
        proc = None
        try:
            with (folder/'logs'/f"{w['case_id']}.stdout.log").open('w') as out, (folder/'logs'/f"{w['case_id']}.stderr.log").open('w') as err:
                proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=env, start_new_session=True)
                w['pid'] = proc.pid
                write_json(folder/'manifest.json', m)
                try:
                    w['returncode'] = proc.wait(timeout=timeout)
                    result_file = folder/'cases'/w['case_id']/'result.json'
                    result = read_json(result_file) if result_file.exists() else {}
                    w['status'] = result.get('status', 'error') if w['returncode'] == 0 else 'error'
                    if result.get('output_file'):
                        from attention_analysis import compare_files
                        ref_id = f"validation_eager_b{w['batch']}"
                        ref_result = read_json(a.assets_dir/'cases'/ref_id/'result.json')
                        compatible = all(result.get(k) == ref_result.get(k) for k in ['input_hash', 'weights_sha256', 'dtype'])
                        comparison = compare_files(Path(result['output_file']), a.assets_dir/'outputs'/f'{ref_id}.pt', assets['input_ids'][:w['batch']]) if compatible else {'structural_finite_ok': False}
                        write_json(folder/'cases'/w['case_id']/'reference_check.json', comparison)
                        w['validation_for_expansion'] = bool(comparison['structural_finite_ok'] and comparison.get('top1_matches') == comparison.get('compared_images'))
                except subprocess.TimeoutExpired:
                    stop_own_worker(proc)
                    w.update(status='timeout', reason='Operational time limit; not evidence of unsupported operation')
        except BaseException:
            if proc is not None:
                stop_own_worker(proc)
            w['status'] = 'interrupted'
            raise
        finally:
            elapsed = time.perf_counter()-t
            m['npu_used_s'] += elapsed
            w.update(ended_utc=now(), worker_elapsed_s=elapsed)
            write_json(folder/'manifest.json', m)
            ev.emit('worker_end', case_id=w['case_id'], status=w['status'], elapsed_s=elapsed, npu_used_s=m['npu_used_s'])
    m.update(status='finished', ended_utc=now(), code_sha256_after={name: sha256(ROOT/name) for name in sources})
    write_json(folder/'manifest.json', m)
    write_json(folder/'logs/device_after.json', command(['furiosa-smi', 'ps']))
    from attention_analysis import analyze
    analyze(folder)
    ev.emit('run_end', run_dir=str(folder), npu_used_s=m['npu_used_s'])


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='action', required=True)
    r = sp.add_parser('run')
    r.add_argument('--assets-dir', type=Path, default=DEFAULT_ASSETS)
    r.add_argument('--warmup', type=int, default=5)
    r.add_argument('--repeats', type=int, default=20)
    r.add_argument('--worker-timeout', type=float, default=1800)
    r.add_argument('--total-budget', type=float, default=7200)
    s = sp.add_parser('analyze')
    s.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'run':
        if min(a.worker_timeout, a.total_budget) <= 0 or a.warmup < 0 or a.repeats < 1:
            p.error('Limits/repeats must be positive; warmup must be nonnegative')
        run(a)
    else:
        from attention_analysis import analyze
        analyze(a.run)


if __name__ == '__main__':
    main()
