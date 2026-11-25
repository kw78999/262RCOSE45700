"""Generate tables from raw traces without inferring missing device timings."""
import argparse
import collections
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics as st
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]

def union_ms(events):
    intervals=sorted((e['ts'],e['ts']+e['dur']) for e in events if e.get('ph')=='X' and e.get('dur',0)>0)
    if not intervals:return None
    merged=[]
    for a,b in intervals:
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    return sum(b-a for a,b in merged)/1000

def events(path):
    if not path.exists():return []
    try:return json.loads(path.read_text()).get('traceEvents',[])
    except json.JSONDecodeError:
        result=path.parent/'result.json'
        if result.exists() and json.loads(result.read_text()).get('status')!='ok':return []
        raise

def task_events(ev):
    return [e for e in ev if '::Task::' in e.get('name','') and {'begin_cycle','end_cycle','cycle_actual'}<=e.get('args',{}).keys()]

def med(values):
    vals=[v for v in values if v is not None];return st.median(vals) if vals else None

def fmt(v):return '—' if v is None else f'{v:.3f}'

def table(headers,rows):
    clean=lambda x:str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(clean(x) for x in row)+' |' for row in rows])+'\n\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run.resolve()
    inventory=json.loads((run/'inventory.json').read_text());summary=[];samples=[]
    for s in sorted(inventory,key=lambda s:(s['pe'],s['batch'],s['mode'])):
        case=f"p{s['pe']}_b{s['batch']}_{s['mode']}";folder=run/'cases'/case
        r=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
        rows=[]
        for row in r.get('samples',[]):
            n=row['iteration'];ev=events(folder/f'forward_{n:02d}.json');tasks=task_events(ev)
            h=events(folder/f'h2d_{n:02d}.json');d=events(folder/f'd2h_{n:02d}.json')
            item=dict(case=case,pe=s['pe'],batch=s['batch'],mode=s['mode'],**row,
                npu_task_ms=union_ms(tasks),npu_task_events=len(tasks),
                clusters=','.join(sorted(set(e['args'].get('cluster_index','') for e in tasks))),
                h2d_device_ms=union_ms([e for e in h if '::h2d::' in e.get('name','')]),
                d2h_device_ms=union_ms([e for e in d if '::d2h::' in e.get('name','')]))
            item['forward_residual_ms'] = row['forward_wall_ms']-item['npu_task_ms'] if item['npu_task_ms'] is not None else None
            rows.append(item);samples.append(item)
        valid=[v for v in rows if v['npu_task_ms'] is not None]
        logs=(folder/'stderr.log').read_text(errors='replace') if (folder/'stderr.log').exists() else ''
        first=events(folder/'first.json')
        compiled=union_ms([e for e in first if e.get('name')=='compile' and e.get('pid')!=0])
        status=r.get('controller_status',r.get('status','pending'))
        guard=folder/'resource_guard.json'
        if guard.exists():status='resource_limit'
        reason=''
        if status not in ['ok','pending','running']:
            if guard.exists():reason='cgroup memory headroom <8 GiB; stopped before OOM'
            elif 'assert.h' in logs or 'failed to run cc' in logs:reason='C compiler / C headers unavailable'
            elif status=='timeout':reason='worker time limit (3600 s)'
            else:reason=str(r.get('error','')).strip().split('\n')[-1][:220] or 'see stderr.log'
        if status=='ok' and len(valid)!=s['samples']:reason=f'NPU Task timing available in {len(valid)}/{s["samples"]} captures'
        row=dict(pe=s['pe'],batch=s['batch'],mode=s['mode'],status=status,samples=len(rows),valid_npu_samples=len(valid),
            warmup=s['warmup'],npu_task_median_ms=med([v['npu_task_ms'] for v in valid]),
            npu_task_min_ms=min((v['npu_task_ms'] for v in valid),default=None),
            npu_task_max_ms=max((v['npu_task_ms'] for v in valid),default=None),
            forward_wall_median_ms=med([v['forward_wall_ms'] for v in rows]),
            forward_residual_median_ms=med([v['forward_residual_ms'] for v in rows]),
            npu_task_count_range=f"{min((v['npu_task_events'] for v in valid),default=0)}–{max((v['npu_task_events'] for v in valid),default=0)}",
            observed_clusters=','.join(sorted(set(v['clusters'] for v in valid))),
            initial_model_and_input_transfer_wall_ms=r.get('initial_model_and_input_transfer_wall_ms'),
            h2d_device_samples=sum(v['h2d_device_ms'] is not None for v in rows),
            d2h_device_samples=sum(v['d2h_device_ms'] is not None for v in rows),
            h2d_wall_median_ms=med([v['h2d_wall_ms'] for v in rows]),
            d2h_wall_median_ms=med([v['d2h_wall_ms'] for v in rows]),
            h2d_device_median_ms=med([v['h2d_device_ms'] for v in rows]),
            d2h_device_median_ms=med([v['d2h_device_ms'] for v in rows]),
            serial_request_median_ms=med([v['serial_request_wall_sum_ms'] for v in rows]),
            first_forward_wall_ms=r.get('first_forward_wall_ms'),compile_api_first_ms=compiled,
            backend_calls_during_samples=r.get('backend_calls_during_samples'),
            graph_break_log_lines=len(re.findall(r'Graph break|Graph Break|\[__graph_breaks\]',logs)),
            recompile_log_lines=len(re.findall(r'Recompiling function',logs)),
            reason=reason,case=case)
        summary.append(row)
    for name,data in [('summary',summary),('samples',samples)]:
        (run/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
        if data:
            with (run/f'{name}.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    planned=json.loads((run/'planned_matrix.json').read_text());attempted={(s['pe'],s['batch']) for s in inventory}
    date=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
    text=f'# ViT batch × PE 추가 프로파일링\n\n작성·갱신: {date}. 원본 실험: `20261118T081725Z-a6c274`. 추가 실행: `{run.name}`.\n\n'
    done=sum(s['status']=='ok' for s in summary)
    text+=f'실제 시도 이력이 있는 **8개 batch×PE 조합 / 16개 Eager·Compile 조건**을 대상으로 한다. 추가 실행 성공 {done}/16개. 순수 연산 유닛 시간은 현재 SDK trace에서 확인되지 않아 추정하지 않는다. 주 결과는 **장치 NPU Task 실행 시간**이다.\n\n'
    counts=collections.Counter(x['status'] for x in summary)
    text+='상태 집계: '+', '.join(f'`{k}` {v}개' for k,v in sorted(counts.items()))+'.\n\n'
    text+='## 1. 기존 실험 조건 전체 목록\n\n'
    text+='계획된 조합은 36개(PE 1·2·4·8 × batch 1·4·8·16·32·64·128·256·512), Eager/Compile 72개다. 이 중 실제 worker를 시작한 8개 조합을 이번에 프로파일링한다. 계획만 있었던 조합은 실행했던 실험으로 계산하지 않는다.\n\n'
    rows=[]
    for b in sorted(set(x['batch'] for x in planned)):
        cells=[]
        for pe in [1,2,4,8]:
            ss=[s for s in inventory if (s['pe'],s['batch'])==(pe,b)]
            if not ss:cells.append('미시작')
            elif all('formal' in s['successful_stages'] for s in ss):cells.append('정식 E/C 완료')
            else:cells.append('실제 시도 — 아래 표 참조')
        rows.append([b,*cells])
    text+=table(['Batch','1 PE','2 PE','4 PE','8 PE'],rows)
    text+=table(['PE','Batch','모드','기존 시도 수','기존 상태별 건수','성공 단계'],[
        [s['pe'],s['batch'],s['mode'],s['attempts'],', '.join(f'{k}:{v}' for k,v in s['old_status_counts'].items()),', '.join(s['successful_stages']) or '없음'] for s in inventory])
    text+='이전 pilot·실제 이미지 후속·attention·fastpath 연구의 batch는 1/4, 기본 1 PE로 위 조건에 포함된다. 초기 합성 입력, native-MHA CPU fallback, Attention-only compile은 실행 경로가 다르므로 이번 수치와 합산하지 않는다. 이번 추가 실험은 최종 PE 확장 실험의 원본 ViT·fastpath-off 경로를 재현한다.\n\n'
    text+='## 2. 측정 방법과 지표\n\n'
    text+=table(['항목','방법 / 해석'],[
        ['모델·입력','원본 ViT-B/16, IMAGENET1K_V1, BF16, 224×224. 가중치·state_dict·입력 hash 확인. B>4는 기존 4장 반복 입력.'],
        ['설정','원본 thread 설정(22/64), MHA fastpath=False, CPU fallback=False, torch.compile backend=furiosa.torch.backend만 지정. PE마다 별도 프로세스.'],
        ['캐시','과거 성공 실행 캐시를 새 case/cache로 복사. 성공 이력이 없는 조건은 과거 중단·timeout 캐시를 복사. 원본 수정 없음. 첫 호출은 캐시 재사용 시작 비용이며 cold compile 시간이 아님.'],
        ['자원 한도','단일 NPU worker, 조건당 3600초. cgroup 메모리 여유 8 GiB 미만이면 자기 worker만 종료하고 resource_limit로 보존.'],
        ['워밍업·반복','정식 이력이 있는 조건 100회, 나머지 20회 warmup 후 5회 프로파일링. 기존 안정화 판정 재검증은 하지 않음. 중앙값·최솟값·최댓값 보고.'],
        ['NPU Task ms','forward trace에서 begin_cycle/end_cycle/cycle_actual이 있는 ::Task:: 구간만 선택. 여러 cluster의 겹친 구간은 합집합으로 계산. CPU export/compile/submit/synchronize span, 외부 H2D/D2H와 profile_d2h는 제외.'],
        ['순수 compute ms','NPU Task 내부 연산·내부 DMA·대기의 독립 분리 근거가 없어 미확인. Task 시간을 순수 compute로 표시하지 않음.'],
        ['Forward 잔여 시간','같은 capture의 forward wall − NPU Task 합집합. CPU 처리·제출·장치 idle·프로파일링 영향 등의 혼합 잔여값이며 순수 CPU 시간이 아님.'],
        ['Forward wall ms','입력·가중치가 NPU에 있는 상태의 runner 호출+동일 current stream 완료. profiler 활성 오버헤드 포함.'],
        ['H2D/D2H wall ms','매 반복 실제 입력 .to(device), 출력 .cpu()를 별도 capture. 메모리 할당·복사 제출·완료 대기를 포함하는 CPU 관측 시간. 순수 DMA와 다름.'],
        ['전송 device ms','장치 ::h2d:: 또는 ::d2h:: span이 존재할 때만 집계. SDK 누락은 —로 표기; 0으로 대체하지 않음.'],
        ['Compile API ms','첫 호출 trace의 compile span 합집합. 캐시 탐색·native compile 경로 등을 포함할 수 있으며 순수 codegen 시간이 아님.'],
        ['첫 호출 wall ms','graph capture/export/compile/cache/load/첫 NPU 실행을 포함. 반복 추론 시간과 별도.'],
        ['단위·정확성','모든 ms는 배치 1회당 시간. 프로파일링 진단이며 기존 정식 latency와 합산하지 않음. 프로파일 종료·JSON 저장 비용은 구간 wall 타이머 밖.']])
    text+='## 3. Batch·PE별 장치 실행 시간\n\n단위: ms / 배치 1회. —는 미측정·이벤트 누락을 뜻하며 0이 아니다. 정상 종료와 NPU 타이밍 확보 여부를 별도로 확인한다. PE 열은 SDK Fusion 설정값이며 개별 PE의 가동률을 뜻하지 않는다.\n\n'
    paired=[]
    for pe,b in sorted(attempted):
        e=next(x for x in summary if (x['pe'],x['batch'],x['mode'])==(pe,b,'eager'))
        c=next(x for x in summary if (x['pe'],x['batch'],x['mode'])==(pe,b,'compile'))
        paired.append([pe,b,fmt(e['npu_task_median_ms']),fmt(c['npu_task_median_ms']),fmt(e['forward_wall_median_ms']),fmt(c['forward_wall_median_ms']),f"{e['valid_npu_samples']}/5 · {c['valid_npu_samples']}/5",f"{e['status']} / {c['status']}"])
    text+=table(['PE','Batch','Eager NPU Task','Compile NPU Task','Eager forward wall','Compile forward wall','유효 NPU E/C','상태 E/C'],paired)
    text+='### 반복 변동과 잔여 시간\n\nForward 잔여 시간은 같은 반복의 forward wall에서 NPU Task 시간을 뺀 뒤 중앙값을 구했다. 순수 CPU 비용으로 해석하지 않는다.\n\n'
    text+=table(['PE','Batch','모드','NPU 최소–최대','Forward 잔여','Task 수','Cluster','비고 / 원본'],[
        [s['pe'],s['batch'],s['mode'],
         '—' if s['npu_task_min_ms'] is None else f"{fmt(s['npu_task_min_ms'])}–{fmt(s['npu_task_max_ms'])}",
         fmt(s['forward_residual_median_ms']),s['npu_task_count_range'],s['observed_clusters'] or '—',f"{s['reason']} [result](cases/{s['case']}/result.json)"] for s in summary])
    text+='## 4. 전송·컴파일 관련 시간\n\nH2D/D2H는 반복별 데이터 전송이며, 전체 모델 가중치 전송은 최초 준비 구간에만 있다. wall 열은 할당·CPU 처리·대기를 포함한다.\n\n'
    text+=table(['PE','Batch','모드','H2D wall','H2D device (유효/5)','D2H wall','D2H device (유효/5)','모델+입력 최초 전송 wall','첫 forward wall','첫 Compile API'],[
        [s['pe'],s['batch'],s['mode'],fmt(s['h2d_wall_median_ms']),f"{fmt(s['h2d_device_median_ms'])} ({s['h2d_device_samples']}/5)",fmt(s['d2h_wall_median_ms']),f"{fmt(s['d2h_device_median_ms'])} ({s['d2h_device_samples']}/5)",*[fmt(s[k]) for k in ['initial_model_and_input_transfer_wall_ms','first_forward_wall_ms','compile_api_first_ms']]] for s in summary])
    failed=[]
    for item in summary:
        if item['status'] in ['ok','pending','running']:continue
        folder=run/'cases'/item['case']
        result=json.loads((folder/'result.json').read_text())
        guard=folder/'resource_guard.json'
        mem='—'
        if guard.exists():
            g=json.loads(guard.read_text());mem=f"{g['memory_used_bytes']/2**30:.2f} / {g['memory_limit_bytes']/2**30:.0f} GiB"
        failed.append([item['pe'],item['batch'],item['mode'],item['status'],fmt(result.get('wall_s')),mem,item['reason']])
    if failed:
        text+='### 완료하지 못한 조건\n\n다음 실행은 첫 forward를 완료하기 전에 종료됐다. 아래 경과시간은 준비·컴파일을 포함한 worker 전체 실행 시간(초)이며 NPU 실행 시간이 아니다. 자원 한도는 이 실행 환경의 한계이며 하드웨어·SDK 미지원 판정이 아니다.\n\n'
        text+=table(['PE','Batch','모드','결과','Worker 경과(s)','cgroup 사용/한도','사유'],failed)
    text+='## 5. Graph break·재컴파일 확인\n\n각 프로세스 시작 전 `TORCH_LOGS=graph_breaks,recompiles`, `FURIOSA_TORCH_PROFILE=trace` 설정. 로그 행 수는 사건 수와 다를 수 있다. 실패한 실행의 로그 부재는 미발생 증거가 아니다.\n\n'
    text+=table(['PE','Batch','모드','상태','Graph-break 로그 행','Recompile 로그 사건','측정 중 backend 추가 호출'],[
        [s['pe'],s['batch'],s['mode'],s['status'],s['graph_break_log_lines'],s['recompile_log_lines'],s['backend_calls_during_samples'] if s['backend_calls_during_samples'] is not None else '—'] for s in summary])
    text+='## 6. 100% 시간 분해 그래프에 사용할 수 있는가?\n\n'
    text+='이번 자료는 입력 전송·resident forward·출력 전송을 순차적으로 측정한다. 그러나 장치 Task 내부 compute/DMA 분리 및 일부 device 이벤트가 미확인이고, compile API는 캐시 재사용 경로다. 따라서 **순수 compute / 순수 H2D DMA / 순수 D2H DMA / 순수 codegen / CPU overhead의 5항목 실측 100% 그래프는 아직 만들 수 없다.** 측정 범위를 명시한 전송 호출 wall / forward wall 비교와 NPU Task 중앙값 표는 사용할 수 있다. 각 항목의 중앙값 합은 요청 총시간의 중앙값과 같지 않을 수 있다.\n\n'
    if (run/'figures/breakdown_all.png').exists():
        text+='### 반복 추론 기준 100% 그래프\n\n아래는 순수 compute/DMA/codegen 분해가 아니라, **실제로 측정한 구간의 시간 분해**다. 추가 NPU 실행 없이 성공한 14개 조건·70회 캡처를 사용했다.\n\n'
        text+='- 100%의 분모: 입력 복사 호출 wall + resident forward wall + 출력 복사 호출 wall. 별도 Profiler context의 시작·종료 사이 공백, JSON 저장·출력 검사 등은 제외한다.\n- 구성: NPU Task, H2D 호출 wall, D2H 호출 wall, 컴파일 0, forward wall − NPU Task. 같은 반복의 값을 사용하므로 중복 없이 분모에 합산된다.\n- 집계: 조건별 5회 구성요소 시간 합 ÷ 같은 5회 전체 시간 합. 각 막대의 총 ms는 평균이다. 앞 표의 중앙값과 구분한다.\n- 반복 측정 중 compile span·backend 추가 호출·재컴파일 로그가 없어 컴파일을 0%로 표시했다. 최초 컴파일 비중을 뜻하지 않는다.\n- 회색 잔여값에는 CPU 제출, 장치 idle, 동기화의 호스트 비용, profiler 영향 등이 섞여 있다. 모두 CPU 연산 시간으로 해석하지 않는다.\n- NPU Task는 내부 DMA/대기를 포함할 수 있고, 전송 호출 wall은 할당·CPU 처리·대기를 포함한다. 순수 연산 또는 순수 DMA 시간으로 바꿔 읽지 않는다.\n\n'
        text+='![대표 조건: Batch 1, 1 PE](figures/breakdown_p1_b1.png)\n\n![전체 batch×PE 조건](figures/breakdown_all.png)\n\n'
        breakdown=json.loads((run/'breakdown.json').read_text())
        text+=table(['PE','Batch','모드','평균 총 ms','NPU Task %','H2D 호출 %','D2H 호출 %','컴파일 %','Forward 잔여 %'],[
            [x['pe'],x['batch'],x['mode'],fmt(x['total_mean_ms']),*[f"{x[k+'_pct']:.2f}" for k in ['npu_task','h2d','d2h','compile','other']]] for x in breakdown])
        text+='[그래프 수치 CSV](breakdown.csv) · [100% 합산 검증](breakdown_verification.json) · [전체 SVG](figures/breakdown_all.svg) · [전체 PDF](figures/breakdown_all.pdf). 반올림한 표의 합은 100%와 0.01%p 차이가 날 수 있다.\n\n'
    if (run/'profiler_control/summary.json').exists():
        controls=json.loads((run/'profiler_control/summary.json').read_text())
        text+='### 프로파일링 오버헤드 대조 검증\n\n2026-11-25 KST 추가 검증. 1 PE·batch 1의 같은 프로세스·모델·resident 입력에서 OFF / ON / ON / OFF 순서로 블록당 10회씩 측정했다. 각 모드 OFF 20회, ON 20회이며 외부 입출력 전송은 제외했다. 모델·가중치·thread·캐시·warmup 설정은 원래 추가 프로파일링과 같다.\n\n'
        text+=table(['모드','OFF forward 중앙값 ms','ON forward 중앙값 ms','ON/OFF','증가율'],[
            [x['mode'],fmt(x['off_median_ms']),fmt(x['on_median_ms']),f"{x['on_over_off_median']:.3f}×",f"{(x['on_over_off_median']-1)*100:.1f}%"] for x in controls])
        text+='프로파일러 활성화의 영향은 Eager에서 특히 컸다. 따라서 위 100% 그림의 회색 비중을 **프로파일러가 꺼진 실제 추론의 CPU 비용 비중으로 해석하면 안 된다.** NPU Task 시간도 순수 arithmetic 시간으로 분리된 것은 아니다. 정상 여부·최적 성능 여부는 이 그래프만으로 확정할 수 없다.\n\n'
        text+='[대조 실험 원본](profiler_control/summary.json). 각 모드 폴더에 40회 forward wall과 ON 구간 20개 trace, 첫 호출 trace, 모델·입력 hash, 로그를 보존했다. 측정 중 backend 재진입은 두 모드 모두 0회다.\n\n'
    text+='## 7. 재현 및 원본\n\n'
    text+=f'```bash\ncd /root/ViT\n/root/venv/bin/python profiling/batch_pe/run.py --run {run.relative_to(ROOT)}\n/root/venv/bin/python profiling/batch_pe/report.py --run {run.relative_to(ROOT)}\n```\n\n'
    text+='그래프 재생성: `vit_pilot/.plot-venv/bin/python profiling/batch_pe/plots.py --run '+str(run.relative_to(ROOT))+'`\n\n'
    text+='[검증 결과](verification.json) · [조건별 CSV](summary.csv) · [반복별 CSV](samples.csv) · [기존 조건 목록](inventory.json) · [장치 정보](device_info.txt). device 전송 타이밍 누락 시 exporter의 `invalid trace reference time` 경고 여부를 원본 로그에서 확인할 수 있다. 이는 실행 0ms를 의미하지 않는다. 각 case에 첫 호출 및 H2D/forward/D2H trace, stdout/stderr, 결과 JSON을 보존한다.\n\n'
    text+='공식 API: [Furiosa Profiler](https://developer.furiosa.ai/furiosa-torch-docs/v2026.4.0/documentation/profile/). 설치 SDK에서 trace 수준은 operator/DMA 상세를 요청하지만 실제 출력에 없는 이벤트는 추정하지 않는다.\n'
    (run/'REPORT.ko.md').write_text(text)
    print(f'Report: {run}/REPORT.ko.md ({done}/16 successful)',flush=True)

if __name__=='__main__':main()
