"""Korean reports and reproduction notes from current-run analysis artifacts."""
from pathlib import Path
import shutil
from common import read_json,write_json,sha256
from fastpath_support import result_for


def f(v,d=3):
    return '—' if v is None else f'{v:.{d}f}' if isinstance(v,float) else str(v)


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(f(v) for v in row)+' |\n' for row in rows)


def report(run,cfg,m,summary,be,batches,outputs,observations,cumulative):
    run=Path(run);assets=read_json(run/'assets.json')
    valid=[v for v in be if v['status'] not in ['insufficient_data','invalid']]
    actual=[s for s in summary if s['status']=='ok' and s['cost_status']=='valid']
    diagnostics=[o for o in observations if o['latest_attempt']]
    out_ok=[p for p in outputs if p['stage']=='diagnostic' and p.get('structural_finite_ok')]
    header='# ViT: 양쪽 MHA fastpath 해제·CPU fallback 차단 비교\n\n'
    header+=f"run_id: `{cfg['run_id']}`. 현재 상태 `{m['status']}`. 주 성능 완료 {len(actual)}/{4*cfg['trials']} worker. "
    header+=f"NPU worker 누적 {m['npu_budget_used_s']:.1f}/{cfg['total_budget_s']:.1f}초, 개별 상한 {cfg['worker_timeout_s']}초.\n\n"
    overview='## 일곱 질문의 현재 결론\n\n'
    overview+='1. **[실행 확인] 제어·동일성:** worker는 새 원래 ViT를 생성하고 기존 AttentionBoundary/하위 compile이 없음을 확인한다. 양쪽 fastpath=False·cpu_fallback=False를 준비 후/측정 전/종료 후 검사하고 상태를 복원한다. '
    overview+='체크포인트 파일, 기존 BF16 입력 파일/배치 바이트, CPU BF16 state_dict 해시를 확인했다. NPU 이동 후 전체 state_dict를 다시 CPU로 복사해 해시했다고 주장하지 않는다.\n'
    overview+=f"2. **[실행 확인/미확인] 경로·출력:** 최신 진단 {sum(o['status']=='ok' for o in diagnostics)}/4개 완료. "
    overview+='native MHA host event, NPU task, backend 진입 및 graph break 로그는 아래 원본과 연결했다. strict와 NPU 증거는 SDK 계약 범위이며 전체 연산 배치, 실제 codegen·disk hit/miss는 미확인이다. 출력은 이번 Eager/Compile끼리 직접 비교하며 허용 오차 미합의이면 pending이다.\n'
    overview+=f"3. **[실행 확인] 준비·호출 시간:** 완료한 {len(actual)}개 worker의 prepare/S/F/W/반복 원본을 저장했다. "
    overview+='같은 배치·trial끼리 묶고 50회 반복을 독립 trial 50개로 세지 않는다. 느린 정상 호출도 모두 포함한다.\n'
    overview+=f"4. **[조건부 추정] 손익분기:** 현재 계산 가능한 paired trial은 {len(valid)}/{2*cfg['trials']}개다. "
    overview+='새 데이터의 A=S+F+sum(W), T(N)=A+N*t만 사용한다. N은 준비 뒤 유효 배치 호출 수, K=N+6은 first+warmup 6회를 포함한 forward 수다. 부품별 좋은 값을 서로 다른 trial에서 골라 합치지 않는다.\n'
    overview+='5. **[조건부 추정/보류] 계산 상태:** finite/already/equal-rate/no-break-even/initial-advantage-only/insufficient 상태를 trial별로 구분한다. 상태가 다르면 하나의 N으로 합치지 않는다. 정밀한 경계는 정수 ns와 유리수로 확인하며 동률은 승리가 아니다.\n'
    counts={state:sum(v['status']==state for v in be) for state in sorted({v['status'] for v in be})}
    overview+=f"   이번 결과: {', '.join(f'{state} {count}쌍' for state,count in counts.items())}. 누락값을 0으로 대체하지 않았다.\n"
    overview+='6. **[별도 실행 확인/보류] 누적 검증:** 새로운 Eager/Compile process 쌍의 S/F/W와 연속 work prefix를 사용한다. 아래 실제 완료 범위를 확인하며, 실행 전 또는 미도달 구간은 predicted_only/미도달로 남긴다. 검증 시작 비용으로 주 예측을 사후 보정하지 않는다.\n'
    for b in cumulative['batches']:
        if b.get('first_observed_compile_advantage_N') is not None:
            overview+=f"   B{b['batch']}: 관측 첫 우위 N={b['first_observed_compile_advantage_N']}, 예측 중앙값 {b['primary_median_N']} 대비 {b['observed_minus_predicted_median_N']:+}회. 종료 N={b['completed_common_calls']}까지 우위를 유지했다.\n"
    overview+='7. **[소스·범위 확인] 이전 설명과의 차이:** 이전 기본 Eager의 CPU fallback 및 약 200/400회 가정 계산을 이번 수식에 넣지 않았다. 새 strict fastpath-off Eager의 실제 초기 비용도 0이 아니며, 모델 준비 비용은 별도 기록하고 주 수식에서는 제외한다.\n\n'
    overview+='   이전 값은 경로·캐시·초기 비용 가정이 다른 설명용 계산이었다. 이번에는 두 모드의 새 전용 캐시에서 측정한 S/F/W와 50회 평균을 같은 trial끼리 결합했다. A의 차이를 순수 컴파일 시간으로 해석하거나, 이전 대비 차이의 원인을 실제 codegen으로 확정할 근거는 없다.\n\n'
    protocol='## 측정 규약과 자산\n\n'
    protocol+=f"`{assets['model_id']}`, `{assets['weights_id']}`, BF16, eval/inference_mode, B1/B4×3 trial. 실제 이미지 폴더는 `/root/ViT/data/vit_real_images_4/images/`다. "
    protocol+=f"순서는 cat/coffee/astronaut/rocket이며 `{cfg['assets_dir']}/inputs/preprocessed_bf16.pt`를 재사용했다. B1의 cat은 B4에도 포함돼 고유 사진은 4개다. 이미지 출처·순서·transforms 정보와 파일/텐서 해시는 image_manifest.json과 assets.json에 보존했다. 정답 라벨이 없어 accuracy/loss는 계산하지 않는다.\n\n"
    protocol+='기존 common.timed_call을 그대로 사용한다: current stream 사전 synchronize → perf_counter_ns → runner(inputs) → synchronize → 종료 시각. '
    protocol+='시간은 CPU dispatch·SDK 내부 전송·실행·완료 대기를 포함한 ms/batch-call이다. 외부 모델/입력 준비와 전송, 출력 복사/검증, 파일 쓰기, 로그는 모델 타이머 밖이다. '
    protocol+='S는 runner callable/torch.compile 래퍼 생성, F는 첫 호출, W는 후속 warmup 5회다. model_input_prepare_ms는 CPU 모델·파일 로드·state hash 검사·NPU 이동·준비 완료 대기를 포함한 참고값으로 A에서 제외한다. '
    protocol+='순수 compiler 시간은 null이다. SDK Eager도 내부적으로 연산을 컴파일할 수 있어 Eager의 torch.compile backend 진입 0회를 실제 codegen 0회로 해석하지 않는다.\n\n'
    protocol+='각 worker는 처음 없던 고유 SDK cache_context 경로를 사용하며 진단/성능/누적 실행은 분리된다. 시스템 전체 cold나 실제 disk miss를 보장하지 않는다. '
    protocol+='Compile에는 backend만 지정했고 dynamic/fullgraph/mode/options/recompile_limit·CPU 스레드는 조정하지 않았다. 전체 모델 compile 요청과 단일 완전 캡처는 구분한다. '
    protocol+='안정성은 기존 CV≤0.10 및 전·후반 평균 차이≤10% 휴리스틱을 재사용했다. preliminary_stable도 steady state 증명은 아니다.\n\n'
    protocol+='최초 두 B1 진단은 forward 전에 참조 state hash의 키 표현 차이로 실패했다. 공식 체크포인트의 구형 MLP linear_1/linear_2 이름과 torchvision 로더의 0/3 이름 차이를 확인했고, 공식 변환 뒤 152개 tensor 값이 정확히 같음을 검증했다. '
    protocol+='최초 오류/키 기준 파일과 수정 코드 버전은 보존했으며 새 attempt로 재실행했다. 모델·타이머·SDK는 수정하지 않았다. logs/state_key_conversion.json과 manifest.code_revision_history를 참고한다.\n\n'
    path='## 새 실행 경로와 출력\n\n'
    path+=table(['케이스','상태','native MHA host','NPU task','backend 진입','SDK segment 목록','recompile 로그'],[
        [o['case_id'],o['status'],o.get('native_mha_host_count'),o.get('npu_task_events'),o.get('backend_entry_count'),o['graph_segments'],o['observed_recompile_log_entries']] for o in observations])
    path+='\n상세 로그와 profiler는 진단 worker에만 켰다. 성능에서 컴파일 이벤트 부재를 단정하지 않는다. CPU host scope 시간은 NPU kernel 시간이 아니며 Task 이름을 codegen 비용으로 사용하지 않는다. '
    path+='원본 trace/guard/graph break 위치는 diagnostic_observations.json, 설치 API 근거는 logs/installed_sources.txt 및 control_installed_sources.txt다.\n\n'
    path+=table(['배치','최대 절대오차','평균 절대오차','cosine 평균/최소','Top-1 일치','허용 오차'],[
        [p['batch'],f(p.get('max_abs_error'),6),f(p.get('mean_abs_error'),6),f"{p.get('cosine_mean',0):.8f} / {p.get('cosine_min',0):.8f}" if p.get('cosine_min') is not None else '정의 안 됨',
         f"{p.get('top1_matches')}/{p.get('compared_images')}",p.get('tolerance_status')] for p in out_ok])
    path+='\n이 표는 이번 진단의 fastpath-off Eager를 이번 fastpath-off Compile과 직접 비교한 값이다. first/마지막 warmup/마지막 repeat 비교와 이미지별 클래스·logit·cosine은 output_comparison.json에 있다. '
    path+='Top-1 일치는 정답 accuracy가 아니다. NaN/Inf/구조 오류와 합의된 오차 위반은 비교를 보류하고, 합의되지 않은 허용 오차는 사후에 만들지 않는다.\n\n'
    perf='## 독립 성능 trial\n\n'
    perf+=table(['mode','B','trial','상태','prepare ms','S ms','F ms','W 합 ms','A ms','repeat 평균 ± SD ms','중앙값 ms','CV','images/s','warm state'],[
        [s['mode'],s['batch'],s['trial'],s['status'],s['model_input_prepare_ms'],s['runner_setup_ms'],s['first_call_ms'],s['warmup_sum_ms'],s['A_ms'],
         f"{f(s['mean_ms'])} ± {f(s['stdev_ms'])}",s['median_ms'],s['CV'],s['throughput_images_s'],s['warm_state_assessment']] for s in summary])
    perf+='\nSD는 표본 SD(ddof=1)이며 신뢰구간이 아니다. 처리량은 B×유효 반복 수/유효 반복 시간 합계로 계산한 순차 호출 처리량이다. '
    perf+='50회 반복 전체 평균을 비용 예측에 사용한다. 평균 대신 빠른 중앙값을 골라 총시간을 예측하지 않는다. 호출 순서 원본은 timings.csv에 보존했다.\n\n'
    for b in batches:
        perf+=f"B{b['batch']} 독립 trial 평균(ms): "+'; '.join(f"{mode} {v['values']} (중앙값 {f(v['median'])}, 관측 범위 {f(v['minimum'])}–{f(v['maximum'])})" for mode,v in b['mode_mean_ms'].items())+'.\n\n'
    cross='## 주 손익분기와 첫 호출 포함 표현\n\n'
    cross+='```text\nA_m = S_m + F_m + sum(W_m)\nT_m(N) = A_m + N*t_m, N >= 1\ndelta_A = A_C-A_E; delta_t = t_E-t_C\nT_C(N)-T_E(N) = delta_A-N*delta_t\n```\n\n'
    cross+='delta_t>0이면 N_win=max(1,floor(delta_A/delta_t)+1)이다. 정수 ns(ROUND_HALF_EVEN)로 정규화한 뒤 유리수로 계산하고 N과 N-1에 대입한다. '
    cross+='반복 50회 합계는 A에 들어가지 않는다. F에 포함된 실행/컴파일과 W를 다시 더하지 않는다. N은 유효 배치 호출 수, 유효 이미지는 B×N, 전체 forward 수는 N+6, 준비까지 센 이미지는 B×(N+6)이다. '
    cross+='전처리·프로세스 기동·model_input_prepare·외부 전송·서비스 큐/배치 조립은 이 시나리오에서 제외한다. 운영 서비스 전체 지연이 아니다.\n\n'
    cross+=table(['B','trial','상태','A_E ms','A_C ms','t_E ms','t_C ms','E/C 반복 배속','실수 교차 N','N_win','K=N+6','유효 이미지','변동 보조 상태'],[
        [v['batch'],v['trial'],v['status'],v.get('A_e_ms'),v.get('A_c_ms'),v.get('t_e_ms'),v.get('t_c_ms'),v.get('compile_speedup'),v.get('real_crossing_N'),
         v['N_win'],v['K_win'],v['effective_images'],v.get('variation_assessment')] for v in be])
    cross+='\n각 trial의 strict 경계 차이와 N-1 값은 break_even_trials.csv/summary.json에 저장했다. 작은 반복 평균 차이는 두 SD의 제곱합 제곱근과 비교한 descriptive uncertain 표기로 보조하고 exact sign은 바꾸지 않는다. '
    cross+='수치가 나와도 출력 허용 오차 pending·미안정·관측 한계를 해소한 것은 아니다.\n\n'
    for b in batches:
        n=b['N_win'];cross+=f"B{b['batch']}: 유효 finite paired trial {b['valid_finite_paired_trials']}/{b['requested_paired_trials']}, N_win 목록 {n['values']}, 중앙값 {f(n['median'])}, 관측 범위 {f(n['minimum'])}–{f(n['maximum'])}. 상태 {b['statuses']}. 표본 3개의 범위는 CI가 아니다.\n\n"
    cross+='K≤6에서는 G(K)=S+실제 first/warmup prefix 합, K>6에서는 G(K)=A+(K-6)t를 사용한다. G(N_win+6)=T(N_win)를 exact arithmetic으로 확인했다. '
    cross+='N_win+6은 준비 뒤 시나리오의 교차 표현이며 전체 K 구간 최초 승리라고 자동 해석하지 않는다. 초기 실제 prefix에서 Compile이 유리했던 K 목록: '
    cross+=str([dict(batch=v['batch'],trial=v['trial'],K=v.get('initial_K_with_compile_advantage')) for v in valid])+'. 세부는 initial_prefix.json이다.\n\n'
    cum=render_cumulative(cumulative)
    finish='## 판단과 남은 범위\n\n'
    finish+='**조건부 추정:** 관측한 동일 조건에서만 반복 성능과 초기 비용을 바탕으로 손익분기를 해석한다. 별도 누적 실행의 검증 상태는 위 표대로이며, 초기 비용·호출 변동과 실행 시점 차이로 예측/관측 교차가 다를 수 있다. '
    finish+='출력 허용 기준은 미합의라면 계속 pending이다. 독립 3 trial과 누적 확인 1쌍을 합쳐 표본 수를 늘리지 않는다. '
    finish+='B8/B16, 다른 모델, 수식/커널 변경, 동적 compile 튜닝은 실행하지 않았다.\n\n'
    finish+=f"원본 성능 행 {sum(len(result_for(run,w).get('timings',[])) for w in m['workers'] if w['stage']=='performance')}/{m['expected_performance_rows']}. "
    finish+=f"all_workers_complete={m['all_workers_complete']}, all_unheld_workers_complete={m['all_unheld_workers_complete']}, all_attempts_successful={m['all_attempts_successful']}. "
    finish+='앞의 두 값은 최신 logical case 기준, 마지막은 보존한 최초 실패 attempt까지 포함한다. 오류·보류는 미측정 0ms로 채우지 않았다. SDK/torch/드라이버·기존 결과·공용 캐시는 보존했다.\n\n'
    finish+='재현 명령과 실제 버전은 [README.md](README.md), 입력/가중치 식별자는 assets.json/asset_verification.json, 예산·명령·PID·실제 순서는 manifest.json에 있다. '
    finish+='[PyTorch 2.10 MHA API](https://docs.pytorch.org/docs/2.10/backends.html#torch.backends.mha). Furiosa 전용 API는 저장한 설치 소스가 직접 근거다.\n'
    (run/'report.md').write_text(header+overview+protocol+path+perf+cross+cum+finish)
    (run/'path_evidence.md').write_text(header+protocol+path)
    write_feedback(run,cfg,m,summary,be,batches,outputs,cumulative)
    write_readme(run,cfg)
    archive=run/'code/analysis_final';archive.mkdir(exist_ok=True)
    names=['fastpath_compare_analysis.py','fastpath_compare_report.py','fastpath_compare_plots.py']
    for name in names:
        source=Path(__file__).parent/name
        if source.exists():shutil.copyfile(source,archive/name)
    write_json(archive/'manifest.json',{p.name:sha256(p) for p in archive.glob('*.py')})


def render_cumulative(cumulative):
    text='## 별도 누적 실행 검증\n\n'
    text+='주 성능의 paired N_win 중앙값×1.2를 올림해 목표를 잡고 N_win=1이면 최소 20회다. mode당 10000회, worker 1800초, 총 7200초 상한 안에서 양쪽 완료 비용을 보수적으로 평가했다. '
    text+='NPU worker는 직렬이며 새 모델·새 runner·처음 없던 전용 SDK cache를 사용한다. first+warmup 뒤 추가 숨은 warmup 없이 N회를 모두 유효 작업으로 센다. '
    text+='모든 work 출력의 CPU 복사/유한값 검사는 타이머 밖에서 수행했다. 따라서 검증의 호출 사이 간격/호스트 활동이 주 성능과 다를 수 있다.\n\n'
    text+=table(['B','상태','관측 공통 N','주 예측 median N','관측 첫 우위 N','관측 K','유효 이미지','관측 끝까지 유지된 마지막 전환 N'],[
        [v['batch'],v['status'],v['completed_common_calls'],v.get('primary_median_N'),v['first_observed_compile_advantage_N'],
         v.get('first_observed_K'),v.get('first_observed_effective_images'),v['final_winning_suffix_start_N']] for v in cumulative['batches']])
    for b in cumulative['batches']:
        if b.get('first_observed_compile_advantage_N') is not None:
            text+=f"\nB{b['batch']}는 주 예측 중앙값보다 {b['observed_minus_predicted_median_N']:+}회 ({b['relative_crossing_difference']*100:+.2f}%) 차이로 실제 첫 우위를 관측했다. "
            lo,hi=b['primary_trial_N_range'];inside=lo<=b['first_observed_compile_advantage_N']<=hi
            text+=f"주 성능 trial의 관측 N 범위는 {b['primary_trial_N_range']}이며 실제 교차는 이 범위 {'안' if inside else '밖'}이다. 이 범위는 신뢰구간이 아니다. "
            text+=f"관측 우위는 N={b['completed_common_calls']}까지 유지됐고, 목표 {b['planned_N']}회까지 완료했다. 그 뒤 구간은 검증하지 않았다.\n"
        text+=f"\nB{b['batch']} 체크포인트 (고정한 실제 paired trial {b.get('prediction_source_trial')} 예측, ms 원본은 CSV):\n\n"
        text+=table(['위치','N','예측 E 초','예측 C 초','관측 E 초','관측 C 초','관측 C-E 초'],[
            [p['label'],p['N'],p['observed']['predicted_eager_ms']/1000 if p['observed'] else None,
             p['observed']['predicted_compile_ms']/1000 if p['observed'] else None,
             p['observed']['observed_eager_ms']/1000 if p['observed'] else None,
             p['observed']['observed_compile_ms']/1000 if p['observed'] else None,
             p['observed']['observed_difference_ms']/1000 if p['observed'] else None] for p in b['checkpoints']])
        if b.get('observed_startup'):
            text+='\n새 검증 시작 비용과 work 평균(사후 설명용):\n\n'+table(['mode','S ms','F ms','W 합 ms','A ms','work 평균 ms'],[
                [mode,v['S_ms'],v['F_ms'],v['W_ms'],v['A_ms'],b['observed_work_mean_ms'][mode]] for mode,v in b['observed_startup'].items()])
            text+=f"\n새 검증 A와 기존 대표 trial의 t만 조합한 **사후 설명용** N_win={b['posthoc_observed_startup_with_primary_rates']['N_win']}. 주 예측 곡선은 수정하지 않았다. 전환 목록: {b['transitions']}.\n"
    text+='\nobserved_T는 새 S+F+sum(W)+work prefix 시간의 합이다. 사전 대기·출력 검증·로그·파일 쓰기·process 기동 등 타이머 밖 시간까지 포함한 wall-clock 전체가 아니다. '
    text+='관측 마지막까지의 우위 유지가 관측 이후 영구 우위를 보장하지 않는다. 교차에 미도달하면 관측 범위와 상한을 보고하며 손익분기 없음으로 결론 내리지 않는다.\n\n'
    return text


def write_feedback(run,cfg,m,summary,be,batches,outputs,cumulative):
    text='# 첫 피드백: ViT fastpath-off Eager 대 전체 Compile\n\n'
    text+='양쪽 BF16·MHA fastpath=False·CPU fallback=False이며 원래 모델/가중치/실제 사진 텐서를 사용했다. '
    text+='진단/성능/누적 실행은 다른 worker와 cache다. 결과는 예비 측정이며 출력 허용 오차가 미합의이면 계속 pending이다.\n\n'
    text+=table(['B','trial','A_E / A_C 초','반복 E / C ms','E/C 배속','상태','N_win','K','유효 이미지'],[
        [v['batch'],v['trial'],f"{f(v.get('A_e_ms')/1000 if v.get('A_e_ms') is not None else None)} / {f(v.get('A_c_ms')/1000 if v.get('A_c_ms') is not None else None)}",
         f"{f(v.get('t_e_ms'))} / {f(v.get('t_c_ms'))}",v.get('compile_speedup'),v['status'],v['N_win'],v['K_win'],v['effective_images']] for v in be])
    text+='\nA=S+F+실제 warmup 5회 합, T(N)=A+N×반복 50회 평균이다. N은 준비 뒤 유효 배치 호출 수, K=N+6은 준비 forward까지 포함한 수다. '
    text+='유효 이미지는 B×N이다. 모델/입력 준비·외부 전송·process 기동·서비스 큐는 제외한다. Eager 초기 비용도 실제 측정값을 사용했다. '
    text+='동률은 승리가 아니며 N/N-1 경계를 exact ns/rational 연산으로 검사했다. 순수 compile 비용 계산이 아니다.\n\n'
    for b in batches:
        n=b['N_win'];text+=f"B{b['batch']} paired N_win: {n['values']}, 중앙값 {f(n['median'])}, 관측 범위 {f(n['minimum'])}–{f(n['maximum'])} ({n['count']}/{cfg['trials']} 유효 쌍). "
        text+='3 trial의 범위는 CI가 아니다.\n'
    text+='\n'
    text+=table(['B','mode','독립 trial 반복 평균 ms','평균들의 중앙값 ms'],[
        [b['batch'],mode,v['values'],v['median']] for b in batches for mode,v in b['mode_mean_ms'].items()])
    unstable=[s for s in summary if s['status']=='ok' and s['warm_state_assessment']!='preliminary_stable']
    text+='\n미안정/unknown worker: '+(str([dict(case=s['case_id'],state=s['warm_state_assessment'],CV=s['CV']) for s in unstable]) if unstable else '완료한 worker에서 없음 (휴리스틱이며 steady state 증명 아님)')+'. 느린 정상 호출을 삭제하지 않았다. SD/CV/최소·최대·호출 순서는 상세 보고서/CSV에 있다.\n\n'
    text+=table(['B','별도 누적 상태','관측 공통 N','예측 median N','관측 첫 우위 N','관측 K','끝까지 유지된 전환 N'],[
        [v['batch'],v['status'],v['completed_common_calls'],v.get('primary_median_N'),v['first_observed_compile_advantage_N'],v.get('first_observed_K'),v['final_winning_suffix_start_N']] for v in cumulative['batches']])
    text+='\n별도 실행은 새 S/F/W부터 시작했고, 실제 구간 합을 고정된 주 성능 예측과 비교한다. 검증 시작 비용을 주 예측에 사후 대입하지 않았다. '
    text+='누적 확인 1쌍을 독립 3 trial에 합치지 않는다. 미도달은 손익분기 없음이 아니다.\n\n'
    for b in cumulative['batches']:
        if b.get('first_observed_compile_advantage_N') is not None:
            lo,hi=b['primary_trial_N_range'];inside=lo<=b['first_observed_compile_advantage_N']<=hi
            text+=f"B{b['batch']} 관측 첫 우위는 예측 중앙값 대비 {b['observed_minus_predicted_median_N']:+}회 ({b['relative_crossing_difference']*100:+.2f}%)였고, 주 3 trial 범위 {'안' if inside else '밖'}이다. 관측 종료 N={b['completed_common_calls']}까지 우위를 유지했다.\n"
    text+='\n'
    points=[p for p in outputs if p['stage']=='diagnostic' and p.get('structural_finite_ok')]
    text+=table(['입력','B','최대/평균 절대오차','최소 cosine','Top-1','판정'],[
        ['실제 사진 (고유 4장)',p['batch'],f"{p['max_abs_error']:.6f} / {p['mean_abs_error']:.6f}",f"{p['cosine_min']:.8f}" if p['cosine_min'] is not None else 'null',
         f"{p['top1_matches']}/{p['compared_images']}",p['tolerance_status']] for p in points])
    text+='\n이번 fastpath-off 두 모드의 직접 logits 비교이며 정답 accuracy/loss가 아니다. B1은 B4의 첫 사진이므로 고유 입력 5장으로 세지 않는다. '
    text+='strict·NPU task는 SDK 계약 범위의 확인이며 전체 연산 배치·실제 codegen·disk cache hit/miss는 미확인이다. 전용 cache가 처음 없어도 시스템 전체 cold를 보장하지 않는다.\n\n'
    text+='**판단: 조건부 추정.** 유효한 paired 데이터와 실제 누적 관측 범위 안에서만 해석한다. 이전 기본 Eager/이전 약 200·400회 가정은 새 계산에 사용하지 않았다. '
    text+='다음 최소 조치는 BF16 출력 허용 기준 합의와 관측 변동/예측 오차 확인이다. SDK/torch·기존 결과를 보존했고 범위를 확대하지 않았다.\n\n'
    text+=f"worker 누적 {m['npu_budget_used_s']:.1f}/{cfg['total_budget_s']:.1f}초. [상세 보고서](report.md), [원본](timings.csv), [손익분기](break_even_trials.csv), [누적 검증](cumulative_validation_summary.json).\n"
    (run/'feedback_summary.md').write_text(text)


def write_readme(run,cfg):
    text='# ViT fastpath-off 동일 조건 비교 재현\n\n'
    env=read_json(run/'environment.json')
    text+=f"실제 실행 Python `{env['executable']}`, `{env['python'].splitlines()[0]}`. torch `{env['torch_import']['version']}`, torchvision `{env['torchvision_import']['version']}`, Furiosa `{env.get('furiosa',{}).get('version')}`. 드라이버/펌웨어/장치 상세는 environment.json의 furiosa-smi 기록을 참고한다.\n\n"
    text+='```bash\ncd /root/ViT\n/root/venv/bin/python -u vit_pilot/fastpath_compare.py run --stop-after diagnostic\n'
    text+=f'/root/venv/bin/python -u vit_pilot/fastpath_compare.py resume --run-dir {run} --stop-after performance\n'
    text+=f'/root/venv/bin/python -u vit_pilot/fastpath_compare.py resume --run-dir {run} --stop-after cumulative\n'
    text+=f'/root/venv/bin/python vit_pilot/fastpath_compare.py analyze --run-dir {run}\n'
    text+=f'vit_pilot/.plot-venv/bin/python vit_pilot/fastpath_compare_plots.py --run {run}\n'
    text+=f'/root/venv/bin/python vit_pilot/fastpath_compare_verify.py --run {run}\n'
    text+='python -m unittest discover -s vit_pilot/tests/fastpath_compare -p test_math.py -v\n```\n\n'
    text+='기본 run은 세 단계 모두 실행한다. 단계별 stop/resume는 완료된 최신 case를 재실행하지 않고 미실시만 진행한다. 누적 NPU worker wall budget은 7200초이며 resume에서도 이미 사용한 시간을 유지한다. '
    text+='worker 1800초, repeat 50, warmup 5, trials 3, cumulative cap 10000이 기본이다. CLI `--worker-timeout`, `--total-budget`, `--warmup`, `--repeats`, `--trials`, `--max-cumulative-calls`를 지원하며 예산을 자동 연장하지 않는다. '
    text+='resume에서 명시적인 budget/timeout 변경만 적용한다. 오류 재시도는 `--retry-case CASE_ID --retry-reason TEXT`로 새 attempt/cache 경로를 만들고 최초 결과와 코드 revision을 보존한다. '
    text+='성공 case를 덮어쓰지 않는다. 자신의 process group만 정리하고 다른 사용자의 NPU 사용이 있으면 보류한다.\n\n'
    text+=f"입력은 `{cfg['assets_dir']}/inputs/preprocessed_bf16.pt`의 byte-identical 텐서다. 이미지 재생성/재전처리 없이 cat/coffee/astronaut/rocket을 사용한다. 원본 파일/텐서/가중치 파일/state_dict 해시는 서로 다른 의미로 저장한다. "
    text+='checkpoint의 구형 MLP 키는 torchvision 공식 loader가 변환하므로 state_dict 모델 해시는 loader 뒤 CPU FP32 및 CPU BF16을 기준으로 한다. NPU 이동 뒤 전체 모델을 CPU로 복사해 검증한 것은 아니다.\n\n'
    text+='양쪽 모두 `torch.backends.mha.set_fastpath_enabled(False)`와 `ft.config.cpu_fallback(False)`를 forward/추적 전에 적용한다. Eager는 원래 모델 직접 호출, Compile은 전체 모델의 기본 `torch.compile(model,backend=ft.backend)`다. '
    text+='하위 compile/AttentionBoundary/수식 변경은 없다. current_stream synchronize는 설치 SDK 계약을 사용한다. 준비/출력 검사와 계산/저장은 모델 타이머 밖이다. '
    text+='first/마지막 warmup/마지막 repeat의 출력을 저장하며 별도 cumulative 실행에서는 매 work 출력의 유한값을 타이머 밖에서 확인한다.\n\n'
    text+='프로파일·상세 로그·backend observer는 진단 worker만 사용한다. 모든 worker는 고유 SDK cache_context 경로를 사용하며 system cold/실제 disk miss는 미확인이다. '
    text+='순수 컴파일 시간은 null이다. 표준편차는 ddof=1, images/s=B×유효 반복 수/시간 합계다. 호출 시간은 CPU dispatch와 SDK 내부 비용을 포함한다. '
    text+='atol/rtol 기본은 미정이고, 합의된 값이 있을 때만 `--atol --rtol --tolerance-source`로 명시한다. 라벨이 없으므로 accuracy/loss를 계산하지 않는다.\n\n'
    text+='손익분기 fixture는 tests/fastpath_compare에 분리됐고 NPU 시간/보고서에 섞지 않는다. main 성능 12개 완료 시 672행, 별도 진단/누적 기록은 분리한다. '
    text+='manifest의 all_workers_complete는 최신 logical case 전체, all_unheld_workers_complete는 그중 보류 제외, all_attempts_successful은 보존한 과거 실패까지 포함한 값이다.\n'
    (run/'README.md').write_text(text)
