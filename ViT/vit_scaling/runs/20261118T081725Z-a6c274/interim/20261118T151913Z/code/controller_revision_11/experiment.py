"""Resumable serial RNGD ViT scaling study; never changes SDK/global cache."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import statistics as st
import subprocess
import sys
import time
import threading
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vit_pilot'))
from common import now, read_json, write_json, sha256, tensor_hash
from process_guard import inventory, relation
PYTHON='/root/venv/bin/python'
PRIOR=ROOT/'vit_pilot/fastpath_compare_runs/20261104T172746Z-e1b86b'
DEFAULT=dict(total_elapsed_budget_s=43200, worker_timeout_s=3600, device='furiosa:0',
    worker_protocol=2,worker_filename='worker.py',
    pe_candidates=[1,2,4,8], batches=[1,4,8,16,32,64,128], extension_batches=[256,512],
    seed=20261118, torch_threads=22, torch_interop_threads=64,
    exploration_warmup=20, exploration_repeats=50,
    stabilization=dict(min_warmup=100,max_warmup=500,window=20,mean_change=0.02,cv=0.05),
    repeat_min_calls=500, repeat_min_seconds=10, repeat_max_calls=10000,
    formal_trials=3, representative_trials=5, stability_seconds=300,
    monitor_period_s=30, broad_exploration_budget_fraction=0.45, safety_reserve_s=45, unknown_worker_estimate_s=1200,
    numeric_tolerance=None, precision='BF16', mha_fastpath=False, sdk_cpu_fallback=False,
    max_representative_conditions=2,post_core_frontier=True,
    compile_options='backend=furiosa.torch.backend only', single_worker=True,
    prior_run=str(PRIOR), budget_basis='wall elapsed since run creation, includes pauses and preparation',
    output_status='provisional; tolerance pending', extension_memory_headroom_gib=8)


def cmd(args, timeout=20):
    try:
        p=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
        return dict(command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except Exception as e: return dict(command=args,error=repr(e))


def pids(snapshot):
    if snapshot.get('returncode')!=0: return None
    text=snapshot['stdout']
    if '| PID' not in text: return None
    return [int(x) for x in re.findall(r'^\|\s*(\d+)\s*\|',text,re.M)]


def create(config_path=None):
    cfg=dict(DEFAULT)
    if config_path: cfg.update(read_json(config_path))
    run=ROOT/'vit_scaling/runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6])
    run.mkdir(parents=True)
    for name in ('logs','cases','cache','inputs','code','figures','evidence'): (run/name).mkdir()
    cfg.update(run_id=run.name,created_utc=now())
    write_json(run/'experiment_config.json',cfg)
    write_json(run/'manifest.json',dict(run_id=run.name,created_utc=cfg['created_utc'],status='preparing',workers=[],events=[],
         requested_matrix=[dict(pe=p,batch=b,mode=m,status='not_measured') for p in cfg['pe_candidates'] for b in cfg['batches']+cfg['extension_batches'] for m in ('eager','compile')]))
    from environment import collect
    collect(run)
    old=read_json(PRIOR/'assets.json'); verified=read_json(PRIOR/'asset_verification.json')
    import torch
    torch.set_num_threads(cfg['torch_threads'])
    x=torch.load(old['preprocessed_bf16_file'],map_location='cpu',weights_only=True)
    if sha256(old['preprocessed_bf16_file'])!=verified['input_file_sha256']: raise RuntimeError('Old input changed')
    if sha256(old['weights_file'])!=verified['weights_file_sha256']: raise RuntimeError('Old weights changed')
    assets=dict(old, source_run=str(PRIOR), original_input_path=old['preprocessed_bf16_file'],
        state_dict_bf16_cpu_sha256=verified['state_dict_bf16_cpu_sha256'],
        fixed_input_performance_only=True, accuracy_evaluation=False, batches={})
    for b in cfg['batches']+cfg['extension_batches']:
        indices=[i%4 for i in range(b)]; xb=x[indices].contiguous()
        target=run/'inputs'/f'b{b}_bf16.pt'; torch.save(xb,target)
        assets['batches'][str(b)]=dict(path=str(target),indices=indices, image_ids=[old['input_ids'][i] for i in indices],
            unique_image_count=len(set(indices)),batch_size=b,tensor_sha256=tensor_hash(xb),file_sha256=sha256(target))
        if str(b) in old['batch_hashes'] and tensor_hash(xb)!=old['batch_hashes'][str(b)]: raise RuntimeError('Baseline input differs')
    write_json(run/'assets.json',assets)
    shutil.copy2(PRIOR/'image_manifest.json',run/'image_manifest.json')
    sources=['__init__.py','_C/device/__init__.pyi','_C/stream/__init__.pyi','config/_stream.py',
             '_backend/eager.py','_backend/aten.py','_backend/torch_compile.py','_backend/runnable_sequence.py',
             '_module/runnable.py','_module/visa.py','compiler/__init__.py','config/_cache.py']
    import furiosa.torch as ft
    sdk=Path(ft.__file__).parent; records=[]
    for name in sources:
        src=sdk/name
        if src.exists():
            dest=run/'evidence'/name.replace('/','__'); shutil.copy2(src,dest)
            records.append(dict(path=str(src),snapshot=str(dest),sha256=sha256(src)))
    write_json(run/'evidence/installed_source_index.json',records)
    for f in (ROOT/'vit_scaling').glob('*.py'): shutil.copy2(f,run/'code'/f.name)
    for name in ['common.py','fastpath_support.py']: shutil.copy2(ROOT/'vit_pilot'/name,run/'code'/name)
    write_json(run/'code/identifiers.json',{p.name:sha256(p) for p in (run/'code').glob('*.py')})
    (run/'plan.md').write_text('''# 실행 계획과 예상 소요시간

12시간 wall elapsed 예산, worker당 3600초. 기존 첫 호출 B1 Eager 약98초/Compile 약199초, B4 약166초/435초였으므로 새 경로·새 장치에서 같은 비용이라고 가정하지 않는다. 기준점 검증/안정화/첫 formal pair 약 30–60분 예상, 넓은 탐색·핵심 3 trial·대표 5 trial은 수시간 이상이며 전체 행렬은 12시간 초과 가능. 다음 worker 비용은 이미 관측한 준비·첫 호출·warmup·반복 시간으로 보수적으로 추정한다. 남은 예산 안에 완료가 어려우면 보류하고 누락 사유를 남긴다.

A: 1PE B1/B4 별도 출력/경로 검증, 안정화 탐색, 공통 warmup을 적용한 첫 formal trial. B: B1/4/16/64/128을 우선하되 작은 batch부터 PE 순환, 이후 B8/32, 조건 성공시 B256/512 검토. PE별 작은 batch 검증에서 구조/finite 오류면 그 PE 확대 중단, 자원 부족이면 그 batch 이상 보류. timeout은 unsupported가 아니다. C: 공통 큰 batch를 결정한 뒤 모든 유효 PE의 B1/B4/B16/공통 큰 batch를 3 trial씩, 남은 조건 후순위. D: 각 PE의 B1/공통 큰 batch/관측 최대 처리량 batch를 5 trial까지. E: 공통 큰 batch에서 최대 관측 Compile 처리량을 보인 PE를 골라 양쪽 300초 stability. 모두 직렬.

정식 trial과 별도 validation/diagnostic/stabilization/exploration/stability worker를 분리한다. 각 worker는 새 프로세스/모델/runner/전용 cache_context. 새로운 cache 경로는 실제 codegen/disk miss의 증거가 아니다. 반복 원본/실패/느린 호출 보존. 출력 허용 기준 pending이므로 성능은 provisional.
''')
    m=read_json(run/'manifest.json');m['status']='ready';write_json(run/'manifest.json',m)
    docs(run)
    print(run,flush=True); return run


def docs(run):
    (run/'README.md').write_text(f'''# RNGD ViT PE·batch 확장 실험

run `{run.name}`. 모델/가중치/BF16/224×224/eval/inference_mode 고정. 양쪽 MHA fastpath=False, SDK CPU fallback=False. Compile backend만 지정. 별도 단일 worker의 단일 모델을 fused logical device 하나에서 호출한다. 결과는 허용 오차 pending인 provisional 자료다.

```bash
cd /root/ViT
/root/venv/bin/python -u vit_scaling/experiment.py resume --run {run}
/root/venv/bin/python vit_scaling/experiment.py analyze --run {run}
```

새 실험: `/root/venv/bin/python -u vit_scaling/experiment.py run [--config /path/config.json]`. 위 resume 한 명령이 완료하지 않은 logical case만 새 attempt/cache로 이어간다. 완료 결과를 덮어쓰지 않는다. `--retry-errors`를 명시하면 오류 logical case도 새 attempt로 재시도한다. 실행 전 config에서 total_elapsed_budget_s, worker_timeout_s, 안정성 시간 등을 변경할 수 있으나 버전/정밀도/모델/경로를 바꾸지 않는다. wall elapsed 예산은 최초 생성 시각부터다. 중단 시간이 예산에 포함된다. `manifest.json`이 실제 시각·순서·PID·attempt·보류의 기준이다. 다른 사용자의 NPU 프로세스가 있으면 새 worker를 시작하지 않는다. 장치 원자 예약은 제공하지 않는다.

원본 `cases/*/result.json`, 통합 `timings.csv`, `trial_summary.csv`, `scaling_summary.csv`, `output_validation.json`, `device_states.jsonl`, `figures/*.png`와 대응 CSV, `report.md` 참조. 모든 호출은 CPU dispatch 포함 batch-call 완료 지연시간이며 순수 kernel 시간이 아니다. batch latency/B는 amortized ms/image이며 개별 요청 응답시간이 아니다.
''')
    (run/'pe_configuration.md').write_text('''# PE 설정과 증거

② 설치 SDK 2026.4.0 `_C/device/__init__.pyi` Fusion 문서 및 `set_fusion(int)` API를 확인했다. `ft.set_fusion(P)`를 import 직후, device/stream 조회·모델 이동·compile 전에 새 프로세스에서 딱 한 번 호출한다. `_C/device`는 num_fusion=1/2/4/8/16/32를 문서화한다. 이번 1-chip의 후보는1/2/4/8뿐이다. 1/2/4는 한 cluster 하드웨어 fusion, 8은 4PE×2 cluster 소프트웨어 fusion이다. MeshKind.Single 기본을 유지하며 mesh·tensor_parallel_size·모델 복제·다중 worker는 사용하지 않는다.

② Eager와 Compile은 같은 프로세스 전역 Fusion을 사용한다. Eager의 ATen graph 실행과 Compile의 Runnable 생성은 이 설정 후 실행한다. `_module/runnable.py:173`와 `_module/visa.py`는 runnable.expected_num_fusion이 현재 Fusion과 같아야 실행함을 검사한다. PE간 산출물 호환을 가정하지 않고 조건·mode·trial·attempt마다 별도 cache_context를 사용한다. 컴파일러 설정/모델/정밀도는 그대로다.

② `config/_stream.py`는 대상 device의 NPU 연산이 current stream에 제출됨을 설명하고 `_C/stream/__init__.pyi`는 해당 stream에 제출된 모든 작업 완료까지 synchronize가 대기함을 설명한다. 하나의 fused logical device/동일 current stream만 사용하는 호출에 이 계약을 적용한다. 관련 없는 다른 stream은 기다리지 않는다. 8PE 내부 runtime의 세부 동기화 구현은 binary로 배포돼 소스 검증 불가하므로 per-PE 하위 구현에 대한 독립 증명은 미확인이다. diagnostic의 fused-device NPU task trace와 완료 query를 추가 근거로 저장하며 하드웨어 사용률 보장으로 해석하지 않는다.

① requested PE는 spec.pe, SDK 설정은 result.fusion 및 device_configuration, 실제 관측은 diagnostic path_observations와 furiosa_trace.json, 별도 furiosa-smi ps의 NPU core 구성이다. 각 값은 분리 기록한다. PE 설정 일치만으로 모든 PE의 유효 활용을 확정하지 않는다. trace에 보이는 chip_index/cluster_index와 core 이름은 그대로 남기며 Task 수를 커널 수/fusion 수로 바꾸지 않는다.

④ 실제 compiler codegen 시간/횟수 및 disk hit/miss와 전체 op placement는 미확인. get_fusion/get_device_configuration은 할당 요청·SDK 구성 근거이며 연산별 PE 활용률 측정값이 아니다. installed source 원문·SHA256은 evidence/ 참조.

공개 공식 자료도 확인했으나 set_fusion의 버전별 상세 계약은 검색 결과로 확인하지 못해 설치 stub을 우선 근거로 사용했다: https://developer.furiosa.ai/v2026.4.0/en/ 및 https://furiosa.ai/developers (2026-11-18 확인). SDK/드라이버를 업데이트하지 않았다.
''')


class Study:
    def __init__(self,run,retry=False):
        self.run=run;self.cfg=read_json(run/'experiment_config.json');self.m=read_json(run/'manifest.json');self.retry=retry
        self.rng=random.Random(self.cfg['seed']);self.halt=False
        self.lock=(run/'controller.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.device_lock=(ROOT/'vit_scaling/device.lock').open('a');fcntl.flock(self.device_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.m['controller_pid']=os.getpid();self.m['status']='running';self.save()
        for w in self.m['workers']:
            if w['status']=='running':
                pid=w.get('pid')
                def alive():
                    try: return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]!='Z'
                    except FileNotFoundError:return False
                if pid and alive():
                    args=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
                    recorded_worker=w.get('command',[None,None,str(self.run/'code/worker.py')])[2]
                    if str(self.run) not in args or w['case_id'] not in args or recorded_worker not in args:
                        raise RuntimeError('Recorded live PID does not match this run worker; refuse to touch it')
                    self.m['events'].append(dict(utc=now(),status='recovering_existing_worker',case=w['case_id'],pid=pid));self.save()
                    last_sample=0;timed_out=False
                    while alive():
                        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(w['started_utc'])).total_seconds()
                        if elapsed>self.cfg['worker_timeout_s'] or self.elapsed()>self.cfg['total_elapsed_budget_s']:
                            os.killpg(pid,signal.SIGTERM);timed_out=True
                            for _ in range(10):
                                if not alive():break
                                time.sleep(1)
                            if alive():os.killpg(pid,signal.SIGKILL)
                            break
                        if time.monotonic()-last_sample>=self.cfg['monitor_period_s']:
                            observed=self.snapshot('recovering_worker',w['case_id']);last_sample=time.monotonic();self.save()
                            if observed is None or any(not entry['owned'] for entry in self.last_process_relations):w['resource_contamination']=dict(utc=now(),pids=observed,process_relations=self.last_process_relations);self.halt=True
                        time.sleep(1)
                    completed=self.result(w)
                    w.update(status='timeout' if timed_out else completed.get('status','interrupted') if completed.get('status')!='running' else 'interrupted',
                        ended_utc=completed.get('ended_utc',now()),recovered_existing_worker=True,
                        wall_s=(datetime.now(timezone.utc)-datetime.fromisoformat(w['started_utc'])).total_seconds())
                    if w.get('resource_contamination'):w['status']='contaminated'
                    self.save();continue
                completed=self.result(w)
                if completed.get('status') in ('ok','error') and completed.get('ended_utc'):
                    try:
                        state=Path(f'/proc/{w["pid"]}/stat').read_text().split(') ')[1][0]
                    except FileNotFoundError: state=None
                    if state is None or state=='Z':
                        w.update(status=completed['status'],ended_utc=completed['ended_utc'],recovered_from_completed_result=True,
                            wall_s=(datetime.fromisoformat(completed['ended_utc'])-datetime.fromisoformat(w['started_utc'])).total_seconds())
                        continue
                try: os.kill(w['pid'],0)
                except ProcessLookupError: w['status']='interrupted'
                else: raise RuntimeError('Previous worker may still be running; refuse overlap')
        self.save()
    def save(self):
        self.m['updated_utc']=now();self.m['elapsed_s']=self.elapsed();self.m['remaining_s']=max(0,self.cfg['total_elapsed_budget_s']-self.elapsed())
        write_json(self.run/'manifest.json',self.m)
    def elapsed(self): return (datetime.now(timezone.utc)-datetime.fromisoformat(self.cfg['created_utc'])).total_seconds()
    def latest(self,stage,p,b,mode,t=1):
        return next((w for w in reversed(self.m['workers']) if (w['stage'],w['pe'],w['batch'],w['mode'],w['trial'])==(stage,p,b,mode,t)),None)
    def result(self,w):
        path=self.run/'cases'/w['case_id']/'result.json' if w else None
        return read_json(path) if path and path.exists() else {}
    def good(self,stage,p,b,mode,t=1):
        w=self.latest(stage,p,b,mode,t)
        # Earlier exploration can establish execution feasibility. Its timings
        # remain excluded by analyze.py from the current performance aggregates.
        protocol_ok=bool(w and (stage not in ('formal','stability','stabilization') or w.get('worker_protocol',1)>=self.cfg.get('worker_protocol',1)))
        return bool(w and protocol_ok and w['status']=='ok' and self.result(w).get('status')=='ok')
    def snapshot(self,label,worker=None):
        row=dict(utc=now(),label=label,worker=worker,commands=[cmd(['furiosa-smi',c]) for c in ('info','status')])
        before=inventory();during={};captured=[]
        query=threading.Thread(target=lambda:captured.append(cmd(['furiosa-smi','ps'])),daemon=True);query.start()
        while query.is_alive():
            during.update(inventory());query.join(.005)
        row['commands'].append(captured[0]);observed=pids(captured[0]);after=inventory()
        own=next((w.get('pid') for w in reversed(self.m['workers']) if w['case_id']==worker),None) if worker else None
        self.last_process_relations=[relation(pid,own,before,after,during) for pid in observed] if observed is not None and own else []
        row['pid_identity_sampling']='5ms while furiosa-smi ps is executing; device snapshots still at configured 30-second interval'
        row['process_relations']=self.last_process_relations
        with (self.run/'device_states.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        return observed
    def estimate(self,stage,p,b,mode,warmup=100):
        candidates=[self.result(w) for w in self.m['workers'] if w['status']=='ok' and (w['pe'],w['batch'],w['mode'])==(p,b,mode)]
        if candidates:
            r=candidates[-1];ts=r.get('timings',[]);first=next((x['elapsed_ms'] for x in ts if x['phase']=='first'),0)/1000
            repeats=[x['elapsed_ms'] for x in ts if x['phase'] in ('repeat','warmup')]
            mean=sum(repeats)/len(repeats)/1000 if repeats else b*0.05
            calls=0 if stage in ('validation','diagnostic') else (warmup if stage in ('stabilization','stability') else warmup+(self.cfg['exploration_repeats'] if stage=='exploration' else self.cfg['repeat_min_calls']))
            extra=self.cfg['stability_seconds'] if stage=='stability' else 0
            return max(30,first*1.35+mean*calls*1.3+extra+30)
        near=[self.result(w) for w in self.m['workers'] if w['status']=='ok' and w['pe']==p and w['mode']==mode and w['batch']<b]
        if near:
            r=max(near,key=lambda r:r['batch']);f=next((x['elapsed_ms']/1000 for x in r['timings'] if x['phase']=='first'),0)
            return max(120,f*(b/r['batch'])**0.75*1.5+60)
        return self.cfg['unknown_worker_estimate_s']
    def execute(self,stage,p,b,mode,t=1,warmup=0,stabilization_status=None):
        if self.halt:return False
        last=self.latest(stage,p,b,mode,t)
        if self.good(stage,p,b,mode,t): return True
        if last and last['status'] in ('error','timeout','oom','unsupported') and not self.retry:return False
        estimate=self.estimate(stage,p,b,mode,warmup)
        remaining=self.cfg['total_elapsed_budget_s']-self.elapsed()
        if estimate>self.cfg['worker_timeout_s']:
            self.m['events'].append(dict(utc=now(),stage=stage,pe=p,batch=b,mode=mode,trial=t,
                status='held_predicted_worker_timeout',estimated_s=estimate,worker_timeout_s=self.cfg['worker_timeout_s'],
                hardware_support='unverified, not declared unsupported'));self.save();return False
        reserve=self.m.get('stability_reserve_s',0) if stage=='formal' else 0
        if remaining<estimate+self.cfg['safety_reserve_s']+reserve:
            self.m['events'].append(dict(utc=now(),stage=stage,pe=p,batch=b,mode=mode,trial=t,status='held_budget',estimated_s=estimate,remaining_s=remaining,stability_reserve_s=reserve));self.save();return False
        used=self.snapshot('before_worker')
        if used is None or used:
            self.m['events'].append(dict(utc=now(),status='held_device_busy_or_unknown',pids=used));self.halt=True;self.save();return False
        attempt=1+sum((w['stage'],w['pe'],w['batch'],w['mode'],w['trial'])==(stage,p,b,mode,t) for w in self.m['workers'])
        case=f'{stage}_p{p}_b{b}_{mode}_t{t}_a{attempt}';folder=self.run/'cases'/case
        while folder.exists():
            attempt+=1;case=f'{stage}_p{p}_b{b}_{mode}_t{t}_a{attempt}';folder=self.run/'cases'/case
        folder.mkdir()
        spec=dict(case_id=case,stage=stage,pe=p,batch=b,mode=mode,trial=t,attempt=attempt,warmup=warmup,
            worker_protocol=self.cfg.get('worker_protocol',1),
            stabilization_status=stabilization_status,cache_path=str(self.run/'cache'/case),estimated_wall_s=estimate)
        write_json(folder/'spec.json',spec)
        env=os.environ.copy()
        for key in ('TORCH_LOGS','FURIOSA_TORCH_LOG'):env.pop(key,None)
        if stage=='diagnostic':env.update(TORCH_LOGS='recompiles,graph_breaks',FURIOSA_TORCH_LOG='debug')
        args=[PYTHON,'-u',str(self.run/'code'/self.cfg.get('worker_filename','worker.py')),'--run',str(self.run),'--spec',str(folder/'spec.json')]
        w=dict(spec,status='running',started_utc=now(),command=args,order=len(self.m['workers'])+1)
        print(json.dumps(dict(event='worker_start',case=case,estimate_s=round(estimate),remaining_s=round(remaining))),flush=True)
        start=time.monotonic();deadline=min(self.cfg['worker_timeout_s'],remaining-self.cfg['safety_reserve_s'])
        with (folder/'stdout.log').open('w') as out,(folder/'stderr.log').open('w') as err:
            proc=subprocess.Popen(args,stdout=out,stderr=err,env=env,start_new_session=True)
            w['pid']=proc.pid; self.m['workers'].append(w);self.save();timeout=False;last_sample=start
            try:
                while proc.poll() is None:
                    time.sleep(1)
                    if time.monotonic()-last_sample>=self.cfg['monitor_period_s']:
                        observed=self.snapshot('during_worker',case);last_sample=time.monotonic();self.save()
                        if observed is None or any(not entry['owned'] for entry in self.last_process_relations):
                            w['resource_contamination']=dict(utc=now(),pids=observed,process_relations=self.last_process_relations);self.halt=True
                    if time.monotonic()-start>deadline:
                        timeout=True;os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        break
            except BaseException:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                w.update(status='interrupted',wall_s=time.monotonic()-start,ended_utc=now());self.save();raise
        result=self.result(w)
        w.update(returncode=proc.returncode,wall_s=time.monotonic()-start,ended_utc=now(),
            status='timeout' if timeout else 'ok' if proc.returncode==0 and result.get('status')=='ok' else 'error')
        text=result.get('error','')+(folder/'stderr.log').read_text()[-5000:]
        if w['status']=='error' and re.search(r'out of memory|OutOfMemory|Cannot allocate|ResourceExhausted',text,re.I):w['status']='oom'
        if w.get('resource_contamination'):w['status']='contaminated'
        if w['status']=='error' and any(token in text for token in ('Fusion request mismatch','Strict control changed','Stream completion failed','Stream completion/path changed','Input hash mismatch','Model hash mismatch','Checkpoint hash mismatch')):
            self.halt=True;self.m['events'].append(dict(utc=now(),status='held_global_configuration_error',case=case))
        self.save();self.snapshot('after_worker',case)
        print(json.dumps(dict(event='worker_end',case=case,status=w['status'],wall_s=round(w['wall_s'],2))),flush=True)
        return w['status']=='ok'
    def modes(self,key):
        modes=['eager','compile']
        if random.Random(self.cfg['seed']+key).randrange(2):modes.reverse()
        return modes
    def valid(self,p,b):return all(self.good('diagnostic' if b==1 else 'validation',p,b,m) for m in ('eager','compile'))
    def validate(self,p,b):
        stage='diagnostic' if b==1 else 'validation'
        return all([self.execute(stage,p,b,m) for m in self.modes(p*1000+b)])
    def extension_headroom(self,p,b):
        rows=[]
        path=self.run/'device_states.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                row=json.loads(line)
                if row.get('worker') and f'_p{p}_b{b}_' in row['worker']:
                    text=row['commands'][1].get('stdout','')
                    found=re.search(r'([\d.]+)/([\d.]+) GiB',text)
                    if found: rows.append(float(found[2])-float(found[1]))
        headroom=min(rows) if rows else None
        self.m['events'].append(dict(utc=now(),status='extension_headroom_observation',pe=p,previous_batch=b,minimum_sampled_free_gib=headroom,
            caveat='30-second sampling, not exact peak or proof next batch fits'));self.save()
        return headroom is not None and headroom>=self.cfg['extension_memory_headroom_gib']
    def explore(self,p,b):
        self.validate(p,b)
        stage='diagnostic' if b==1 else 'validation'
        results=[]
        for mode in self.modes(p*1000+b+1):
            results.append(self.execute('exploration',p,b,mode,warmup=self.cfg['exploration_warmup']) if self.good(stage,p,b,mode) else False)
        return all(results)
    def warm(self,p,b):
        for mode in self.modes(p*1000+b+2):self.execute('stabilization',p,b,mode,warmup=self.cfg['stabilization']['max_warmup'])
        if not all(self.good('stabilization',p,b,m) for m in ('eager','compile')):return None
        rs=[self.result(self.latest('stabilization',p,b,m)) for m in ('eager','compile')]
        n=max(r['warmup_count'] for r in rs)
        status='heuristic_satisfied' if all(r['stabilization_status']=='heuristic_satisfied' for r in rs) else 'stabilization_unconfirmed'
        self.m.setdefault('warmup_policy',{})[f'p{p}_b{b}']=dict(common_warmup=n,status=status,source_cases=[r['case_id'] for r in rs]);self.save()
        return n,status
    def formal(self,p,b,trials):
        if not self.valid(p,b):return
        trials=list(trials)
        missing=[(t,mode) for t in trials for mode in ('eager','compile') if not self.good('formal',p,b,mode,t)]
        if not missing:return
        reserve=self.m.get('stability_reserve_s',0)
        known=self.m.get('warmup_policy',{}).get(f'p{p}_b{b}',{})
        expected_warmup=known.get('common_warmup',500)
        estimated=sum(self.estimate('formal',p,b,mode,expected_warmup) for t,mode in missing)
        estimated+=sum(self.estimate('stabilization',p,b,mode,500) for mode in ('eager','compile') if not self.good('stabilization',p,b,mode))
        remaining=self.cfg['total_elapsed_budget_s']-self.elapsed()
        if remaining<estimated+reserve+self.cfg['safety_reserve_s']:
            self.m['events'].append(dict(utc=now(),status='held_budget_for_complete_trial_group',pe=p,batch=b,
                requested_trials=trials,missing_mode_trials=missing,estimated_group_s=estimated,remaining_s=remaining,stability_reserve_s=reserve,
                reason='Complete comparable trial groups instead of starting one trial everywhere'));self.save();return
        policy=self.warm(p,b)
        if not policy:return
        n,status=policy
        for t in trials:
            modes=['eager','compile']
            if (random.Random(self.cfg['seed']+p*1000+b).randrange(2)+(t-1))%2:modes.reverse()
            for m in modes:self.execute('formal',p,b,m,t,warmup=n,stabilization_status=status)
    def representatives(self):
        pairs={}
        for w in self.m['workers']:
            p,b=w['pe'],w['batch']
            if w['stage']!='formal' or not all(self.good('formal',p,b,mode,t) for mode in ('eager','compile') for t in range(1,self.cfg['formal_trials']+1)):continue
            rs=[self.result(self.latest('formal',p,b,'compile',t)) for t in range(1,self.cfg['formal_trials']+1)]
            rates=sorted(b*r['repeat_count']*1000/r['repeat_sum_ms'] for r in rs)
            pairs[p,b]=st.median(rates)
        if not pairs:return []
        baseline=(1,1) if (1,1) in pairs else min(pairs,key=lambda x:(x[1],x[0]))
        peak=max(pairs,key=lambda x:(pairs[x],-x[0],-x[1]))
        return list(dict.fromkeys([baseline,peak]))[:self.cfg.get('max_representative_conditions',2)]
    def frontier(self):
        if not self.cfg.get('post_core_frontier',False) or self.halt:return
        reps=self.representatives()
        rep_reserve=sum(self.estimate('formal',p,b,mode,self.m.get('warmup_policy',{}).get(f'p{p}_b{b}',{}).get('common_warmup',500)) for p,b in reps for mode in ('eager','compile') for t in range(self.cfg['formal_trials']+1,self.cfg['representative_trials']+1) if not self.good('formal',p,b,mode,t))
        for b in (16,64,128,8,32,256,512):
            order=list(self.cfg['pe_candidates']);random.Random(self.cfg['seed']+b).shuffle(order)
            for p in order:
                if self.halt:return
                if all(self.good('exploration',p,b,mode) for mode in ('eager','compile')):continue
                if not self.valid(p,4):continue
                if b in (256,512):
                    previous=128 if b==256 else 256
                    if not all(self.good('exploration',p,previous,mode) for mode in ('eager','compile')) or not self.extension_headroom(p,previous):continue
                blocked=any(w['pe']==p and w['batch']<=b and w['status'] in ('oom','timeout') for w in self.m['workers'])
                if blocked:continue
                if b>16 and not all(self.good('exploration',p,16,mode) for mode in ('eager','compile')):continue
                costs=[self.estimate(stage,p,b,mode,self.cfg['exploration_warmup'] if stage=='exploration' else 0) for stage in ('validation','exploration') for mode in ('eager','compile') if not self.good(stage,p,b,mode)]
                remaining=self.cfg['total_elapsed_budget_s']-self.elapsed()
                reserved=self.m.get('stability_reserve_s',0)+rep_reserve+self.cfg['safety_reserve_s']
                if not costs:continue
                if max(costs)>self.cfg['worker_timeout_s'] or sum(costs)+reserved>remaining:
                    self.m['events'].append(dict(utc=now(),status='held_frontier_budget_or_predicted_worker_cap',pe=p,batch=b,estimated_group_s=sum(costs),maximum_worker_estimate_s=max(costs),reserved_s=reserved,remaining_s=remaining,hardware_support='unverified'));self.save();continue
                self.m['events'].append(dict(utc=now(),status='post_core_frontier_started',pe=p,batch=b,estimated_group_s=sum(costs),representative_reserve_s=rep_reserve));self.save()
                self.explore(p,b)
    def run_all(self):
        ps=self.cfg['pe_candidates'];blocked={}
        # A. Baseline first trial under the final policy. Remaining paired trials in C.
        for b in (1,4):
            if self.validate(1,b):self.formal(1,b,[1])
        # B. Round-robin PE, ascending preferred batches. B8/32 after primary sweep.
        stop_broad=False
        for b in (1,4,16,64,128,8,32):
            if b!=1 and self.elapsed()>self.cfg['total_elapsed_budget_s']*self.cfg.get('broad_exploration_budget_fraction',.45):
                self.m['events'].append(dict(utc=now(),status='exploration_held_to_reserve_core_trials',next_batch=b,reason='Prioritize comparable 3-trial core over single exploratory trials everywhere'));self.save();break
            order=list(ps);random.Random(self.cfg['seed']+b).shuffle(order)
            for p in order:
                if b!=1 and self.elapsed()>self.cfg['total_elapsed_budget_s']*self.cfg.get('broad_exploration_budget_fraction',.45):
                    self.m['events'].append(dict(utc=now(),status='exploration_held_to_reserve_core_trials',pe=p,batch=b));self.save();stop_broad=True;break
                if p in blocked and b>=blocked[p]:continue
                self.explore(p,b)
                failed=[w for w in self.m['workers'] if w['pe']==p and w['batch']==b and w['status']=='oom']
                invalid=[w for w in self.m['workers'] if w['pe']==p and w['batch']==b and any(token in self.result(w).get('error','') for token in ('Nonfinite output','Invalid output metadata'))]
                timeouts=[w for w in self.m['workers'] if w['pe']==p and w['batch']==b and w['status']=='timeout']
                if failed or invalid or timeouts:
                    blocked[p]=min(b,blocked.get(p,b))
                    self.m['events'].append(dict(utc=now(),status='larger_batches_held_after_resource_structural_or_timeout_failure',pe=p,batch=b,not_a_hardware_unsupported_conclusion=True));self.save()
            if stop_broad:break
        for b in self.cfg['extension_batches']:
            if self.elapsed()>self.cfg['total_elapsed_budget_s']*self.cfg.get('broad_exploration_budget_fraction',.45):
                self.m['events'].append(dict(utc=now(),status='extension_held_to_reserve_core_trials',batch=b));self.save();break
            prev=128 if b==256 else 256
            for p in ps:
                if p in blocked and b>=blocked[p]:continue
                if all(self.good('exploration',p,prev,m) for m in ('eager','compile')) and self.extension_headroom(p,prev):
                    # Physical memory observed after previous exploration is retained in snapshots;
                    # this bound reserves input + model activations conservatively, not a proof.
                    self.explore(p,b)
                    if any(w['pe']==p and w['batch']==b and w['status']=='oom' for w in self.m['workers']):blocked[p]=b
        active=[p for p in ps if any(all(self.good('exploration',p,b,m) for m in ('eager','compile')) for b in self.cfg['batches'])]
        common=[b for b in self.cfg['batches']+self.cfg['extension_batches'] if active and all(all(self.good('exploration',p,b,m) for m in ('eager','compile')) for p in active)]
        large=max(common) if common else None
        self.m['selection']=dict(active_pe=active,common_large_batch=large,
            core_rule='each supported PE: B1, B4, B16 and largest commonly explored batch; complete 3 paired trials before extras',
            representative_rule='each active PE: B1, common large batch, and maximum explored compile throughput; ties smaller batch')
        self.save()
        # Reserve an actual paired stability run before committing the core budget.
        if large and active:
            for sb in sorted(common,reverse=True):
                candidates=[(p,self.result(self.latest('exploration',p,sb,'compile'))) for p in active]
                sp=max(candidates,key=lambda item:sb*item[1].get('repeat_count',0)/item[1].get('repeat_sum_ms',float('inf')))[0]
                estimates=[self.estimate('stability',sp,sb,mode,500) for mode in ('eager','compile')]
                if max(estimates)>self.cfg['worker_timeout_s']:continue
                probe_estimates=[self.estimate('stabilization',sp,sb,mode,500) for mode in ('eager','compile') if not self.good('stabilization',sp,sb,mode)]
                self.m['selection'].update(stability_pe=sp,stability_batch=sb,
                    stability_rule='Largest common batch whose predicted per-mode stability worker fits the cap; PE with highest explored Compile throughput; both modes run')
                self.m['stability_reserve_s']=sum(estimates)+sum(probe_estimates)+90
                self.save();break
        core=[]
        for b in dict.fromkeys([1,4,16,large]):
            if b is not None:
                order=list(active);random.Random(self.cfg['seed']+b*7).shuffle(order)
                if order:
                    shift=list(dict.fromkeys([1,4,16,large])).index(b)%len(order);order=order[shift:]+order[:shift]
                for p in order:
                    if self.valid(p,b):core.append((p,b))
        # Complete three trials for a comparable core pair before moving to next pair.
        for p,b in core:self.formal(p,b,range(1,self.cfg['formal_trials']+1))
        self.frontier()
        reps=self.representatives()
        self.m['selection'].update(representatives=reps,representative_rule='At most two completed core pairs (each >=3 current-protocol trials): baseline B1 and highest median Compile throughput across complete core pairs; ties lower PE/batch. No fastest-trial selection.');self.save()
        # D. Representative trials, before the separate stability stage.
        for p,b in reps:self.formal(p,b,range(1,self.cfg['representative_trials']+1))
        # E. Separate stability follows D representative trials; all workers remain serial.
        if self.m['selection'].get('stability_batch') is not None:
            sp=self.m['selection']['stability_pe'];sb=self.m['selection']['stability_batch']
            policy=self.warm(sp,sb)
            if policy:
                for mode in self.modes(sb):self.execute('stability',sp,sb,mode,warmup=policy[0],stabilization_status=policy[1])
        self.m['stability_reserve_s']=0;self.save()
        for p,b in core:self.formal(p,b,range(1,self.cfg['formal_trials']+1))
        for b in self.cfg['batches']+self.cfg['extension_batches']:
            for p in active:
                if (p,b) not in core and self.valid(p,b):self.formal(p,b,range(1,self.cfg['formal_trials']+1))
        self.m['status']='held_external_device' if self.halt else 'finished_with_coverage_recorded';self.save()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','run','resume','analyze','status'])
    p.add_argument('--run',type=Path);p.add_argument('--config',type=Path);p.add_argument('--retry-errors',action='store_true')
    a=p.parse_args()
    if a.action in ('init','run'):run=create(a.config)
    else:run=a.run.resolve()
    if a.action=='status':
        m=read_json(run/'manifest.json');print(json.dumps({k:v for k,v in m.items() if k not in ('workers','requested_matrix')},indent=2));return
    if a.action in ('run','resume'):
        study=Study(run,a.retry_errors)
        try:study.run_all()
        finally:
            study.save()
            if (ROOT/'vit_scaling/analyze.py').exists():subprocess.run([PYTHON,str(ROOT/'vit_scaling/analyze.py'),'--run',str(run)],check=False)
    elif a.action=='analyze':subprocess.run([PYTHON,str(ROOT/'vit_scaling/analyze.py'),'--run',str(run)],check=True)

if __name__=='__main__':main()
