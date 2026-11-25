"""Create a portable Notion Markdown summary and ZIP matching the supplied prose/table style."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'profiling/batch_pe/runs/20261124T181821Z'
DEST=ROOT/'vit_scaling/runs/20261118T081725Z-a6c274'
summary=json.loads((RUN/'summary.json').read_text())
breakdown=json.loads((RUN/'breakdown.json').read_text())
controls=json.loads((RUN/'profiler_control/summary.json').read_text())
history=json.loads((RUN/'graph_break_history_audit.json').read_text())
assets=DEST/'notion_assets';assets.mkdir(exist_ok=True)
data=DEST/'notion_data';data.mkdir(exist_ok=True)
for name in ['breakdown_p1_b1.png','breakdown_all.png']:
    shutil.copy2(RUN/'figures'/name,assets/name)
for source,target in [(RUN/'summary.csv',data/'profiling_summary.csv'),(RUN/'samples.csv',data/'profiling_samples.csv'),(RUN/'breakdown.csv',data/'breakdown.csv'),(RUN/'graph_break_history_audit.json',data/'graph_break_history_audit.json'),(RUN/'verification.json',data/'profiling_verification.json'),(RUN/'breakdown_verification.json',data/'breakdown_verification.json'),(RUN/'profiler_control/summary.json',data/'profiler_control.json')]:
    shutil.copy2(source,target)
with (data/'profiler_control.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(controls[0]));w.writeheader();w.writerows(controls)

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v).replace('|','/').replace('\n',' ') for v in row)+' |' for row in rows])+'\n\n'

def fmt(x):return '—' if x is None else f'{x:.3f}'

text='''# ViT

## 실험 환경

| 항목 | 내용 |
| --- | --- |
| 모델 | torchvision `vit_b_16`, `ViT_B_16_Weights.IMAGENET1K_V1` |
| 하드웨어 | FuriosaAI RNGD NPU 1장, PE 1·2·4·8개를 Fusion으로 묶어 모델 1개 실행 |
| 소프트웨어 | furiosa-torch 2026.4.0, torch 2.10.0+cpu |
| 공통 조건 | BF16, 입력 224×224, `eval()`·`inference_mode()`, MHA fastpath 해제, `cpu_fallback=False` |
| 입력 | 기존 실험과 같은 실제 이미지 4장. batch 1은 첫 이미지, batch 4는 4장, batch 16은 4장을 반복 |
| 재현 확인 | 기존 가중치·BF16 state_dict·입력의 SHA-256 일치 확인. CPU thread 22개, interop thread 64개 유지 |
| 실험 대상 | 기존에 실제 시도한 batch×PE 8개 조합, eager·compile을 합쳐 16개 조건 |
| 측정 방식 | 조건마다 새 프로세스. 첫 호출 기록 → 웜업 100회(정식 이력 조건) 또는 20회(나머지) → 프로파일링 5회 |
| 시간 측정 | 입력 전송·resident forward·출력 전송을 각각 구분. forward는 호출 직전부터 동일 current stream의 완료 동기화까지 측정 |
| 캐시 | 기존 조건의 캐시를 새 디렉터리에 복사해 사용. 성공 이력이 없는 조건은 중단·timeout 캐시를 복사. 시스템 전체 cold 실험이 아님 |
| 대표값 | 시간 표는 조건당 5회 중앙값. 100% 그래프는 같은 5회 시간 합의 비율이며, 막대 옆 총시간은 평균 |
| 측정일 | 2026-11-25 (KST). 원본 실험 `20261118T081725Z-a6c274`, 추가 프로파일링 `20261124T181821Z` |

## 기존 측정 대비 달라진 점

- 기존 정식 실험은 프로파일러 없이 forward 호출부터 완료 동기화까지 측정함. 모델·입력의 사전 NPU 전송과 출력 CPU 전송은 타이머 밖에 있었음
- 이번에는 `FURIOSA_TORCH_PROFILE=trace`와 `Profiler`를 사용해 장치 실행 구간을 별도로 기록함
- 매 반복 입력 전송과 출력 전송도 별도 측정해, forward 시간과 구분함
- 컴파일·로드가 포함되는 첫 호출은 반복 추론 성능에서 제외하고 기록만 함
- 기존 정식 실험의 여러 독립 trial과 이번 한 프로세스 내 5회 캡처는 다른 측정임. 두 결과를 합산하거나 같은 표본으로 취급하지 않음
- 프로파일러가 실행 시간을 늘릴 수 있어, 후속 OFF/ON 대조 실험으로 영향을 확인함

## 1단계: 기존 batch×PE 조건 정리와 추가 프로파일링

- 계획에는 PE 1·2·4·8 × batch 1·4·8·16·32·64·128·256·512가 있었으나, 실제 worker를 시작한 조합은 아래 8개임
- 계획만 있었던 조합을 측정 완료로 세지 않음. batch 8·32 이상과 PE 2·batch 4 등은 이번 추가 측정 대상이 아님
- 기존에 실패·중단한 두 compile 조건도 재시도함

'''
pairs=sorted({(x['pe'],x['batch']) for x in summary})
lookup={(x['pe'],x['batch'],x['mode']):x for x in summary}
rows=[]
for pe,b in pairs:
    e,c=[lookup[(pe,b,m)] for m in ['eager','compile']]
    rows.append([b,pe,'완료' if e['status']=='ok' else e['status'],'완료' if c['status']=='ok' else '컴파일 중 메모리 기준 종료'])
text+=table(['batch','PE','eager','compile'],rows)
text+='''- 16개 조건 모두 실행을 시도했고, 14개 조건에서 5회씩 총 70회 프로파일링을 완료함
- 웜업과 반복은 독립 trial이 아님. 조건별 여러 프로세스에서 얻은 신뢰구간으로 해석하지 않음

### 완료하지 못한 조건

'''
failed=[]
for s in summary:
    if s['status']=='ok':continue
    folder=RUN/'cases'/s['case'];r=json.loads((folder/'result.json').read_text());g=json.loads((folder/'resource_guard.json').read_text())
    failed.append([s['batch'],s['pe'],s['mode'],f"{r['wall_s']:.1f}초",f"{g['memory_used_bytes']/2**30:.2f} / {g['memory_limit_bytes']/2**30:.0f} GiB"])
text+=table(['batch','PE','모드','worker 경과시간','환경 메모리 사용 / 한도'],failed)
text+='''- 두 조건 모두 첫 forward가 완료되기 전, 컴파일 중 메모리 여유가 8 GiB 아래로 내려가 자기 worker만 종료함
- 위 경과시간은 준비·컴파일을 포함한 시간이며 NPU 실행 시간이 아님
- OOM 오류나 하드웨어·SDK 미지원으로 확정한 것이 아님. 유효 추론 결과가 없어 아래 시간 표에서는 `—`로 표시함

## 2단계: batch·PE별 NPU 실행 시간

- CPU에서 측정한 전체 호출 시간과, 장치가 기록한 NPU Task 실행 시간을 구분함
- NPU Task는 하드웨어 cycle 정보가 있는 이벤트만 선택함. 여러 cluster의 실행이 겹치면 한 번만 세도록 구간의 합집합을 구함
- CPU graph capture·컴파일·제출·동기화 span, 별도 입력·출력 전송, 프로파일 결과 전송(`profile_d2h`)은 NPU Task 집계에서 제외함

'''
rows=[]
for pe,b in pairs:
    e,c=[lookup[(pe,b,m)] for m in ['eager','compile']]
    rows.append([b,pe,fmt(e['npu_task_median_ms']),fmt(c['npu_task_median_ms']),fmt(e['forward_wall_median_ms']),fmt(c['forward_wall_median_ms'])])
text+=table(['batch','PE','eager NPU Task (ms)','compile NPU Task (ms)','eager forward 전체 (ms)','compile forward 전체 (ms)'],rows)
text+='''- 시간은 **배치 1회당 중앙값**임. batch 4의 시간을 이미지 1장 지연시간으로 읽지 않음
- `forward 전체`는 프로파일러가 켜진 값이며 외부 입력·출력 전송은 제외함
- **NPU Task 시간도 순수 산술 연산 시간은 아님.** 장치 내부 DMA나 대기가 포함될 수 있어 연산 유닛이 계산한 시간만으로 확정하지 않음
- PE 열은 Fusion 설정값이며, 모든 PE가 같은 비율로 활용됐다는 뜻은 아님

### 해석

- batch 1의 compile NPU Task 시간은 PE 1 → 2 → 4에서 약 3.385 → 2.234 → 1.605 ms로 줄어듦
- PE 8은 약 1.660 ms로, 이 조건에서는 PE 4보다 더 짧아지지 않음. 원인은 이번 결과만으로 확정하지 않음
- eager도 PE 증가에 따라 NPU Task 시간은 줄지만, 프로파일링 중 전체 forward 시간은 수십 ms로 남음
- CPU 제출, 장치 idle, 내부 전송, 프로파일링 비용을 완전히 분리하지 못했으므로 전체 차이를 특정 원인 하나로 설명하지 않음

## 3단계: 전송 시간과 컴파일 시간 구분

| 항목 | 이번에 확인한 범위 | 해석할 때 주의할 점 |
| --- | --- | --- |
| CPU → NPU | 매 반복 입력 `.to(device)` 호출과 완료 대기의 wall time | 메모리 할당·CPU 처리·대기 포함. 순수 DMA 시간과 다름 |
| NPU → CPU | 매 반복 출력 `.cpu()` 호출과 완료 대기의 wall time | 메모리 할당·CPU 처리·대기 포함. 순수 DMA 시간과 다름 |
| 최초 모델·입력 전송 | 초기 준비 구간에서 별도로 기록 | 매 요청마다 전체 가중치를 다시 전송한 실험이 아님 |
| 첫 forward | graph capture·컴파일 경로·캐시·로드·첫 장치 실행을 포함해 기록 | 순수 컴파일 시간으로 해석하지 않음 |
| Compile API | 첫 호출 trace의 SDK `compile` span | 캐시 탐색 등을 포함할 수 있으며 순수 codegen과 분리하지 못함 |
| 반복 중 컴파일 | 측정 구간에서 compile span·추가 backend 호출·재컴파일 로그 미관측 | 반복 추론 그래프에서는 컴파일 비중 0%로 표시 |

- 전송 device span은 일부 캡처에서 누락됨. exporter의 `invalid trace reference time` 경고도 확인됨
- 장치 전송 이벤트가 없다고 0 ms로 대체하지 않음. 그래프에는 모든 반복에서 측정한 **전송 호출 wall time**을 사용함
- 첫 호출에 사용한 캐시는 복사한 기존 캐시이므로, 이 자료를 cold compile 시간 비교로 사용하지 않음

## 4단계: 100% 시간 분해 그래프

- 반복 추론의 시간 구성을 보기 위해 아래 5개 항목으로 표시함

| 그래프 항목 | 정의 |
| --- | --- |
| NPU Task execution | forward trace에서 기록된 NPU Task 구간의 합집합 |
| CPU → NPU copy call | 입력 전송 호출의 wall time |
| NPU → CPU copy call | 출력 전송 호출의 wall time |
| Compilation | 반복 측정 전에 완료했으므로 0% |
| Other + profiling effects | forward 전체 wall time − NPU Task 시간. CPU 제출·장치 idle·동기화의 호스트 비용·프로파일러 영향 등이 섞인 잔여값 |

- 100%의 분모 = **입력 전송 호출 + resident forward + 출력 전송 호출**
- 별도 캡처한 세 구간의 wall time 합이며, 그 사이 프로파일러 시작·종료·JSON 저장·출력 검사 등의 공백까지 포함한 요청 전체 타이머는 아님
- 각 비중 = 조건별 같은 5회 구성요소 시간 합 ÷ 같은 5회 분모 합. 서로 다른 항목의 중앙값을 더해 100%로 만든 것이 아님
- 그림 옆의 총 ms는 평균이며, 2단계 표의 중앙값과 구분함

### 대표 결과: batch 1, PE 1

![batch 1, PE 1의 100% 시간 분해](notion_assets/breakdown_p1_b1.png)

'''
rows=[]
for mode in ['eager','compile']:
    x=next(x for x in breakdown if (x['pe'],x['batch'],x['mode'])==(1,1,mode))
    rows.append([mode,fmt(x['total_mean_ms']),*[f"{x[k+'_pct']:.2f}%" for k in ['npu_task','h2d','d2h','compile','other']]])
text+=table(['모드','평균 총시간 (ms)','NPU Task','입력 전송 호출','출력 전송 호출','컴파일','나머지 시간'],rows)
text+='''- compile의 파란색 비중이 큰 것은 NPU가 더 오래 실행돼서가 아님. NPU Task 시간은 eager 약 5.29 ms → compile 약 3.39 ms로 줄었고, 전체 시간은 더 크게 줄어 비중이 커짐
- 100% 막대 길이가 같다고 실행 시간이 같은 것은 아님. 절대시간은 막대 옆 ms로 비교함

### 전체 batch×PE 결과

![전체 batch×PE 조건의 100% 시간 분해](notion_assets/breakdown_all.png)

- 빗금 행은 완료한 추론이 없는 조건이며, NPU 실행 0%를 의미하지 않음
- 이 그림은 **프로파일러가 켜진 실행의 시간 구성**임. 프로파일러가 꺼진 평상시 실행의 비중으로 읽으면 안 됨

## 5단계: 프로파일링 오버헤드 대조 확인

- 그래프의 eager 시간이 기존 정식 측정값보다 커, 프로파일러 자체가 시간을 늘리는지 추가 확인함
- batch 1·PE 1, 같은 프로세스·모델·resident 입력을 사용함
- OFF → ON → ON → OFF 순서로 블록당 10회씩, 모드마다 OFF 20회·ON 20회를 측정함
- eager·compile 합계 forward 80회. 위 70회 그래프 캡처와는 별도 대조 실험임
- 여기서는 외부 입력·출력 전송을 제외한 forward만 측정함

'''
text+=table(['모드','프로파일러 OFF 중앙값 (ms)','ON 중앙값 (ms)','증가율'],[[x['mode'],fmt(x['off_median_ms']),fmt(x['on_median_ms']),f"{(x['on_over_off_median']-1)*100:.1f}%"] for x in controls])
text+='''- 프로파일러 ON에서 eager는 약 43.5%, compile은 약 2.9% 길어짐
- **그래프 회색 91.4%를 평소 추론에서의 CPU 비용 91.4%로 해석하면 안 됨**
- CPU 처리·장치 idle·프로파일러 비용을 각각 분리한 결과가 아니므로 회색 전체를 CPU 연산 시간이라고 부르지 않음
- compile 모드의 추론 시간에는 최초 컴파일 시간이 들어 있지 않음. 이번 대조에서도 측정 중 backend 재진입은 두 모드 모두 0회였음
- 이번 수치는 같은 프로세스 내 대조 결과이며, 여러 독립 trial에서 얻은 신뢰구간이나 최적 성능 보장은 아님

## 6단계: Graph break 확인

### 확인 방법

- `TORCH_LOGS=graph_breaks,recompiles`로 기록한 로그와 backend 호출 기록을 함께 확인함
- `graph_breaks`: Dynamo가 그래프 캡처를 끊은 위치와 이유
- `recompiles`: 입력 shape 등 guard 조건이 달라져 다시 컴파일한 이유
- SDK 내부 그래프 분할, NPU Task 개수, CPU fallback은 Dynamo graph break와 구분함

### 기존 실험 로그 확인

- 중복 보고서·코드 사본·컴파일러 캐시를 제외한 원본 실행 로그 **353개**를 검색함
- 검색한 로그에서 graph-break 관련 행은 발견되지 않음
- 다만 모든 실행에서 로깅을 켠 것은 아니므로, 아래 결론은 실제 진단 실행 범위에 한정함

'''
old_rows=[
 ['Preflight',7,'덧셈 진단에서 미관측. ViT 시도는 import 실패로 실행 판단 불가','batch 1→2 변경 재컴파일 2건'],
 ['초기 ViT pilot',23,'B1/B4 smoke 및 batch 전환 진단에서 미관측','batch 1→4 변경 재컴파일 1건'],
 ['실제 이미지 후속',40,'재시도 후 성공한 B1/B4 compile 진단에서 미관측','미관측'],
 ['Attention 초기',24,'부분 compile은 backend 진입 없이 실패. 완전한 compile 실행의 미발생 근거로 사용하지 않음','미관측'],
 ['Attention 경계 후속',8,'B1/B4의 Attention compile 구간에서 미관측','미관측'],
 ['Fastpath-off 비교',46,'성공한 B1/B4 compile 진단에서 미관측','미관측'],
 ['PE 확장',205,'유효한 B1·PE 1/2/4/8 compile 진단에서 미관측','미관측']]
text+=table(['실험','검색 로그 수','Graph break 판단 범위','재컴파일'],old_rows)
text+='''- 초기 pilot의 batch 전환 로그에는 `tensor 'x' size mismatch at index 0. expected 1, actual 4`가 남아 있음. **재컴파일 확인이며 graph break 확인은 아님**
- 정식 성능 worker는 graph-break 로깅을 끈 경우가 있어, 전체 반복 구간에서 graph break가 없었다고 확정하지 않음
- Attention 초기 실패에는 Dynamo top-frame skip과 native MHA의 CPU fallback 금지가 관련됨. graph break 발생으로 단정하지 않음
- 과거 PE 2 compile의 첫 진단은 최종 manifest에서 `contaminated`로 분류돼 유효 성공 진단에서 제외함

### 추가 프로파일링에서 확인한 결과

'''
text+=table(['batch','PE','compile 결과','Graph-break 로그 행','재컴파일 로그 사건','측정 중 backend 추가 호출'],[
 [x['batch'],x['pe'],'완료' if x['status']=='ok' else '컴파일 중 자원 한도 종료',x['graph_break_log_lines'],x['recompile_log_lines'],x['backend_calls_during_samples'] if x['backend_calls_during_samples'] is not None else '—'] for x in summary if x['mode']=='compile'])
text+='''- 성공한 compile 6개 조건은 처음 호출·웜업·측정 과정의 로그에서 graph break·재컴파일이 관측되지 않았고, 5회 측정 중 backend 추가 호출도 없었음
- 실패한 compile 2개 조건은 첫 forward를 완료하지 못했으므로, 로그가 없다는 이유로 graph break 미발생을 확정하지 않음
- eager는 전체 모델을 Dynamo로 캡처하는 경로가 아니므로, eager의 로그 부재를 “단일 그래프 compile 성공”의 증거로 쓰지 않음
- 이번 확인에 `fullgraph=True`나 `torch._dynamo.explain`을 새로 사용한 것은 아님. 기존 compile 옵션을 유지한 실행 로그와 backend 기록을 근거로 함

## 해석과 현재 한계

- 성공한 조건에서 NPU Task 실행 시간이 기록됐고, 70회 캡처의 구간 합산·하드웨어 cycle 정보·완료 동기화를 검증함
- 100% 그래프는 14개 조건 모두 합이 100%가 되는지 검증함. 표시값 반올림으로 표의 합은 0.01%p 차이가 날 수 있음
- 현재 알 수 있는 것은 **NPU Task 실행 시간과 프로파일링 중 시간 구성**임. 내부 DMA를 완전히 제거한 순수 연산 시간, 순수 codegen 시간은 아직 분리하지 못함
- graph break는 성공한 진단 범위에서 미관측임. 전체 과거 정식 실행의 미발생을 증명한 것은 아님
- PE 8·batch 4와 PE 1·batch 16의 compile은 환경 메모리 기준으로 중단됨. 큰 배치의 compile 성능이나 지원 여부는 판단하지 않음
- 출력은 형상·유한값과 기존 입력·가중치 일치를 확인했지만, 이번 추가 프로파일링에서 새로운 accuracy 평가나 수치 허용 기준 합의를 한 것은 아님
- 기존 수치 허용 기준은 pending이므로, 정식 성능 보고서의 provisional 상태를 이번 진단으로 해제하지 않음

## 다음 계획

- 큰 compile 조건이 끝날 수 있도록 컴파일 메모리 요구와 실행 환경을 확인한 뒤, PE 8·batch 4와 PE 1·batch 16을 재측정
- SDK에서 Task 내부의 operator·DMA 타이밍을 독립적으로 얻을 수 있는지 확인해 순수 연산·내부 이동을 더 세분화
- 실제 latency는 프로파일러 OFF 측정을 기준으로 보고하고, ON 자료는 병목 분석 자료로 분리
- 필요하면 별도 진단에서 `fullgraph=True`로 단일 FX graph 캡처 여부를 추가 검증. 기존 결과와는 옵션이 달라 별도 기록
- 큰 batch나 미시작 조건의 성능·PE 활용률은 현재 B1/B4 자료에서 외삽하지 않음

## 원본 자료

- [조건별 프로파일링 수치](notion_data/profiling_summary.csv)
- [70회 반복별 측정값](notion_data/profiling_samples.csv)
- [100% 그래프 수치](notion_data/breakdown.csv)
- [프로파일러 OFF/ON 대조 결과](notion_data/profiler_control.csv)
- [과거 graph break 로그 감사 결과](notion_data/graph_break_history_audit.json)
- [프로파일링 검증 결과](notion_data/profiling_verification.json)
- [100% 합산 검증 결과](notion_data/breakdown_verification.json)

서버의 상세 보고서와 trace 원본 위치:

```text
/root/ViT/profiling/batch_pe/runs/20261124T181821Z/REPORT.ko.md
/root/ViT/profiling/batch_pe/runs/20261124T181821Z/cases/
/root/ViT/profiling/batch_pe/runs/20261124T181821Z/profiler_control/
```
'''
path=DEST/'notion_summary.md'
# This file did not exist when the supplied Notion format was requested.
path.write_text(text)
paths=[path,*sorted(assets.glob('*')),*sorted(data.glob('*'))]
archive=DEST/'notion_profiling_summary.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in paths:z.write(p,p.relative_to(DEST))
# Validate portable references and Markdown table shape before delivery.
errors=[]
for target in re.findall(r'\]\(([^)]+)\)',text):
    if '://' not in target and not (DEST/target).exists():errors.append('Missing link '+target)
columns=None
for line in text.splitlines():
    if not line.startswith('|'):columns=None;continue
    n=line.count('|')
    if columns is None:columns=n
    elif columns!=n:errors.append('Table column mismatch')
if len(summary)!=16 or sum(x['status']=='ok' for x in summary)!=14:errors.append('Condition count changed')
if sum(x['logs_checked'] for x in history)!=353 or sum(x['graph_break_log_lines'] for x in history)!=0:errors.append('Historical audit changed')
with zipfile.ZipFile(archive) as z:
    if z.testzip():errors.append('ZIP integrity error')
validation={'passed':not errors,'errors':errors,'markdown':str(path),'zip':str(archive),'bundle_files':len(paths),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(RUN/'notion_export_verification.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(validation,ensure_ascii=False,indent=2))
if errors:raise SystemExit(1)
