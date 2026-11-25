"""Join two saved research runs, preserving every original attempt and its logs."""
import argparse
import csv
from pathlib import Path
import shutil
import statistics
from common import read_json, write_json, write_csv, sha256
from attention_analysis import compare_files, fmt, table


def load_csv(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--attention', type=Path, required=True)
    a = p.parse_args()
    c, h = a.controls, a.attention
    cm, hm = read_json(c/'manifest.json'), read_json(h/'manifest.json')
    cc, hc = read_json(c/'config.json'), read_json(h/'config.json')
    assert cm['status'] == hm['status'] == 'finished'
    assert all(cc[k] == hc[k] for k in ['assets_dir', 'device', 'warmup', 'repeats'])
    assert read_json(c/'assets.json') == read_json(h/'assets.json')
    assert cm['npu_used_s'] + hm['npu_used_s'] <= 7200
    assets = read_json(h/'assets.json')
    labels = dict(eager_default='기본 Eager (CPU fallback 포함)', attention_compile='Attention 경계만 Compile',
                  eager_fastpath_off='Eager, MHA fastpath 해제', full_compile='전체 Compile')
    combined, output, audit = [], [], []
    for b in [1, 4]:
        for mode in labels:
            source = h if mode == 'attention_compile' else c
            case_id = f'performance_{mode}_b{b}'
            row = next(x for x in load_csv(source/'summary.csv') if x['case_id'] == case_id)
            r = read_json(source/'cases'/case_id/'result.json')
            raw = [t['elapsed_ms'] for t in r['timings'] if t['phase'] == 'repeat' and t['status'] == 'ok' and t['timing_valid']]
            assert row['status'] == r['status'] == 'ok' and len(raw) == cc['repeats']
            # Independent recomputation from per-case raw JSON, not rounded display values.
            expected = dict(mean_ms=statistics.mean(raw), median_ms=statistics.median(raw),
                stdev_ms=statistics.stdev(raw), throughput_images_s=b*len(raw)*1000/sum(raw))
            assert all(abs(float(row[k])-v) <= 1e-10*max(1, abs(v)) for k, v in expected.items())
            assert len(r['timings']) == 1+cc['warmup']+cc['repeats']
            assert r['input_hash'] == assets['batch_hashes'][str(b)] and r['weights_sha256'] == assets['weights_sha256']
            assert r['dtype'] == 'torch.bfloat16' and not r['effective']['suppress_errors']
            row.update(source_run=source.name, condition=labels[mode], input_hash=r['input_hash'])
            combined.append(row)
            baseline = read_json(c/'cases'/f'performance_eager_default_b{b}'/'result.json')
            assert all(r[k] == baseline[k] for k in ['input_hash','weights_sha256','dtype','timing_scope','cache_state'])
            assert all(r['effective'][k] == baseline['effective'][k] for k in ['device_configuration',
                'torch_threads','torch_interop_threads','compiler_config','backend_eager_fallback',
                'automatic_dynamic_shapes','recompile_limit','cpu_fallback','suppress_errors'])
            assert r['sync_checked'] and r['stream_id'] == baseline['stream_id']
            comp = compare_files(Path(r['output_file']), Path(baseline['output_file']), assets['input_ids'][:b])
            assert comp['structural_finite_ok'] and comp['top1_matches'] == b
            output.append(dict(case_id=case_id, source_run=source.name, batch=b, mode=mode, **comp))
            audit.append(dict(case_id=case_id, source_run=source.name, arithmetic_ok=True,
                              valid_repeats=len(raw), timing_rows=len(r['timings']), same_condition_metadata=True))
    write_csv(h/'study_summary.csv', combined, list(combined[0]))
    write_json(h/'study_output_comparison.json', dict(unique_real_images=4, tolerance_status='pending',
        accuracy_computed=False, loss_computed=False, comparisons=output))
    for folder, manifest in [(c, cm), (h, hm)]:
        csv_rows = load_csv(folder/'timings.csv')
        json_rows = []
        for w in manifest['workers']:
            result = folder/'cases'/w['case_id']/'result.json'
            if w['stage'] == 'performance' and result.exists():
                json_rows.extend(read_json(result)['timings'])
        assert len(csv_rows) == len(json_rows)
        for cs, js in zip(csv_rows, json_rows):
            assert cs['case_id'] == js['case_id'] and cs['phase'] == js['phase'] and int(cs['iteration']) == js['iteration']
            assert float(cs['elapsed_ms']) == js['elapsed_ms'] and cs['status'] == js['status']
    protected = read_json(c/'logs/protected_files_before.json')
    unchanged = {path: sha256(Path(path)) == value for path, value in protected.items()}
    assert all(unchanged.values())
    write_json(h/'study_verification.json', dict(status='pass', cases=audit,
        combined_successful_performance_rows=sum(x['timing_rows'] for x in audit),
        expected_selected_successful_rows=8*(1+cc['warmup']+cc['repeats']),
        controls_original_attempts_preserved=True, protected_files_unchanged=unchanged,
        npu_worker_total_s=cm['npu_used_s']+hm['npu_used_s'], budget_s=7200,
        run_refs=[str(c), str(h)]))
    n = cm['npu_used_s']+hm['npu_used_s']
    header = '# Attention CPU fallback: 원인·해결 실험\n\n'
    header += '**[실행 확인] native MHA fallback을 재현했고, fastpath 해제와 사용자 코드 경계의 Attention-only Compile 모두 B1/B4 strict 진단·유한 출력·Top-1 일치성을 확보했다.** '
    header += '부분 Compile은 실행 가능하지만, 이번 전체 ViT 반복 호출은 전체 Compile이 더 짧았다. 단일 trial의 예비 결과다.\n\n'
    mechanism = '## 왜 CPU fallback이 발생했는가\n\n'
    mechanism += '[실행 확인/소스 확인] 기본 Eager는 ViT의 eval/inference_mode self-attention에서 `aten::_native_multi_head_attention` fastpath를 선택한다. '
    mechanism += '설치 torch 2.10.0 + SDK 2026.4.0의 해당 연산에는 PrivateUse1 전용 kernel이 없고 SDK backend fallback이 등록돼 있다. '
    mechanism += 'CPU fallback을 금지한 실제 B1 호출은 이 연산에서 RuntimeError를 냈고, 기본 경로 CPU profiler는 native MHA 12회를 기록했다. '
    mechanism += '이는 Attention 전체가 NPU에서 불가능하다는 뜻이 아니다. `need_weights=False`도 native fastpath를 항상 막지는 않는다.\n\n'
    mechanism += '[소스 확인] PyTorch 2.10 Dynamo는 tracing 중 `_is_make_fx_tracing=True`로 취급해 native 경로를 건너뛴다. '
    mechanism += '실제 FX graph는 기존 functional MHA를 호출하며 Furiosa SDK에는 SDPA math 분해가 있다. '
    mechanism += '[추정] CPU 계산·내부 복사·동기 대기가 큰 Eager 지연에 기여한다. 이번 ablation은 해당 실행 경로의 영향을 지지하지만 비용별 비중은 측정하지 않았다. '
    mechanism += 'CPU host scope를 순수 CPU/NPU kernel 시간으로 읽거나 중첩 시간을 더해 원인별 비율을 만들지 않는다.\n\n'
    mechanism += '[소스/실행 확인] 기본 `self_attention.compile()`만 적용한 첫 시도는 backend 진입 0회로 동일하게 실패했다. '
    mechanism += '설치 Dynamo는 기본 MHA 메서드를 top frame에서 생략하고 다른 사용자 코드 안에서는 포함한다. '
    mechanism += '따라서 기존 MHA를 그대로 호출하는 `AttentionBoundary`를 준비 단계에 넣고 그 경계를 compile했다. '
    mechanism += '12개 모두 적용했고 원래 Parameter 객체 집합·q/k/v·kwargs를 유지했다. wrapper는 수식을 바꾸지 않으며 나머지 ViT는 Eager다. '
    mechanism += '이 방식은 strict B1/B4에서 성공했다.\n\n'
    perf = '## 전체 ViT 배치 호출 시간\n\n'
    perf += f"실제 사진 4장, BF16, 224×224, 기본 1 PE, 각 조건 새 worker, 독립 trial 1회, warmup {cc['warmup']}회 뒤 repeat {cc['repeats']}회다. "
    perf += 'CPU dispatch부터 공식 current-stream 완료까지 ms/batch-call로 측정했다. 외부 준비·전처리·입출력 전송은 제외하고 SDK 내부 복사·dispatch·대기는 포함한다. '
    perf += '첫 호출과 반복은 별도이며 순수 compiler/NPU kernel 시간이 아니다. 표준편차는 ddof=1이고 CI가 아니다.\n\n'
    perf += table(['조건','B','setup ms','first ms','repeat 평균 ± SD ms','중앙값 ms','images/s'], [
        [r['condition'],r['batch'],float(r['runner_setup_ms']),float(r['first_call_ms']),
         f"{float(r['mean_ms']):.4f} ± {float(r['stdev_ms']):.4f}",float(r['median_ms']),float(r['throughput_images_s'])] for r in combined])
    perf += '\n각 worker는 처음 없던 전용 SDK cache를 사용했다. 시스템 전체 cold와 실제 hit/miss는 미확인이다. '
    perf += '조건별 실행은 순차였으며 Attention 경계 조건은 실패 조사 뒤 별도 run에서 실행했다. 순서 효과를 통제한 혼합 workload나 통계적 우열 검정은 아니다. '
    perf += '기본 Eager는 CPU/NPU 혼합 실행이므로 이 표를 NPU-only Eager 대 Compile 배속으로 해석하지 않는다. throughput은 순차 반복 처리량이다. '
    perf += '원래 직접 MHA compile 실패·B4/성능 보류는 첫 run의 manifest와 summary에 보존했다.\n\n'
    unstable = [r for r in combined if r['warm_state_assessment'] != 'preliminary_stable']
    for row in unstable:
        source = h if row['mode'] == 'attention_compile' else c
        raw = read_json(source/'cases'/row['case_id']/'result.json')['timings']
        repeats = [t for t in raw if t['phase'] == 'repeat' and t['status'] == 'ok']
        slow = max(repeats, key=lambda t: t['elapsed_ms'])
        perf += f"[실행 확인] {row['condition']} B{row['batch']}는 `{row['warm_state_assessment']}`이다. repeat {slow['iteration']}이 {slow['elapsed_ms']:.6f}ms로 가장 길었다. "
        perf += '느린 정상 호출도 모두 평균/SD에 포함했다. 시간 증가만으로 재컴파일 또는 cache miss라고 판정하지 않는다. `figures/02_repeat_order.png`에 호출 순서를 제시한다.\n\n'
    path = '## 경로와 출력 확인\n\n'
    observations = read_json(c/'path_observations.json')+read_json(h/'path_observations.json')
    path += table(['진단','strict','상태','CPU native MHA 횟수','NPU task events','backend 진입'], [
        [o['case_id']+(' (경계)' if o in observations[-2:] else ''),o['strict'],o['status'],o['native_mha_host_count'],o['npu_task_events'],o['backend_entries']] for o in observations])
    path += '\n대표 호출의 native CPU MHA 부재, strict 성공, NPU task를 함께 확인한다. strict의 SDK 계약을 넘어 전체 연산 배치를 완전히 증명했다고 표현하지 않는다. '
    path += 'backend 진입, NPU task, 실제 codegen, disk hit/miss는 서로 다르며 뒤의 두 항목은 null/unknown이다. '
    path += '상세 로그·profiler는 별도 진단 worker에서만 사용했다. 전체 Compile의 이전 strict/trace 근거는 기존 followup run에 있고 이번 control 성능 호출에는 profiler를 넣지 않았다.\n\n'
    path += table(['조건','B','최대 절대오차','평균 절대오차','최소 cosine','Top-1 일치'], [
        [labels[o['mode']],o['batch'],o['max_abs_error'],o['mean_abs_error'],f"{o['cosine_min']:.8f}",f"{o['top1_matches']}/{o['compared_images']}"] for o in output])
    path += '\n참조는 이번 control run에서 저장한 같은 배치 기본 Eager logits다. CPU FP32로 변환한 뒤 비교했으며 모든 출력은 [B,1000] BF16 유한값이다. '
    path += 'atol/rtol 미합의로 tolerance_status=pending이다. 고유 입력은 4장이고 B1의 cat은 B4에도 포함된다. '
    path += 'Top-1 일치는 정답 accuracy가 아니며 정답 라벨이 없어 accuracy/loss를 계산하지 않았다.\n\n'
    decision = '## 해결 방법과 다음 최소 조치\n\n'
    decision += '1. **전체 Compile:** 이번 조건에서는 가장 짧은 전체 호출 시간을 관측했다. 기존 코드에서 사용하는 해결 경로다.\n'
    decision += '2. **Attention 경계만 Compile:** 실행 가능하다. 사용자 코드 경계를 두고 실제 backend 진입과 출력을 확인해야 한다. 나머지 Eager 실행과 여러 프로그램 호출은 남는다.\n'
    decision += '3. **Eager MHA fastpath 해제:** worker 내부의 공개 `torch.backends.mha.set_fastpath_enabled(False)`로 native 연산을 피했다. 기본 Eager와 구분한 조건이며 원래 설정을 복원한다. 수치 차이는 별도로 검증해야 한다.\n'
    decision += '4. **SDK 지원 개선:** 공급자에게 native MHA PrivateUse1 kernel 지원 또는 지원 여부를 고려한 dispatch 경로를 문의할 직접 재현 근거가 마련됐다. SDK·torch·드라이버·커널은 수정하지 않았다.\n\n'
    decision += '**판단: 조건부 진행.** fallback 회피 방법과 소수 입력 출력 일치성은 확보했으나, 출력 허용 오차·전체 연산 배치·실제 codegen/cache 관측과 단일 trial 한계가 남는다. '
    decision += '먼저 BF16 출력 허용 기준을 합의하고, 선택한 실행 경로의 독립 trial을 확대한다. B8/B16과 세부 모듈 벤치마크는 이번에 실행하지 않았다. '
    decision += '첫 피드백 질문은 ① SDK native MHA 지원/dispatch 개선 가능 여부 ② BF16 출력 atol/rtol 기준의 두 가지다.\n\n'
    refs = '## 원본과 재현\n\n'
    refs += f'- Control/직접 compile 실패: `{c}`\n- 사용자 경계 compile: `{h}`\n'
    refs += f'- 실제 입력: `/root/ViT/data/vit_real_images_4/images/`, 재사용 텐서: `{assets["preprocessed_bf16_file"]}`\n'
    refs += f'- 누적 worker 시간 {n:.1f}초 / 7200초. 개별 상한 1800초. 모든 원본·성공 결과·실패 로그와 기존 프로젝트 파일을 보존했다.\n\n'
    refs += '[원인 상세](../'+c.name+'/mechanism.md), [control 보고서](../'+c.name+'/report.md), [새 경계 코드](../../attention_boundary_worker.py), [재현 문서](../../ATTENTION_README.md). '
    refs += '[PyTorch MHA 문서](https://docs.pytorch.org/docs/2.10/generated/torch.nn.MultiheadAttention.html), [MHA backend API](https://docs.pytorch.org/docs/2.10/backends.html#torch-backends-mha).\n'
    (h/'study_report.md').write_text(header+mechanism+perf+path+decision+refs)
    short = header
    short += '[소스/실행 확인] Eager의 native MHA fastpath에는 Furiosa 전용 kernel이 없어 CPU fallback이 발생했다. '
    short += 'fastpath 해제 또는 Compile의 functional MHA/SDPA 경로에서는 strict B1/B4가 성공했다. '
    short += '기본 MHA에 직접 Module.compile을 적용하면 Dynamo top-frame skip으로 backend 진입이 0회였다. '
    short += '기존 MHA를 그대로 호출하는 사용자 코드 경계를 Compile하자 실제 backend 진입과 정상 출력을 확인했다.\n\n'
    short += table(['조건','B','first ms','repeat 평균 ± SD ms','images/s'], [
        [r['condition'],r['batch'],float(r['first_call_ms']),f"{float(r['mean_ms']):.3f} ± {float(r['stdev_ms']):.3f}",float(r['throughput_images_s'])] for r in combined])
    short += f"\n독립 trial 1회, warmup {cc['warmup']}회 뒤 repeat {cc['repeats']}회. SD는 ddof=1이며 CI가 아니다. "
    short += '시간은 전체 ViT 배치 호출부터 current-stream 완료까지로 내부 dispatch/전송을 포함한다. '
    short += '기본 Eager는 CPU/NPU 혼합 기준선이다. first-call은 순수 compile 시간이 아니다. '
    short += '전용 SDK cache는 처음 없던 경로이며 시스템 cold·실제 hit/miss는 미확인이다.\n\n'
    short += '**변동 주의:** 전체 Compile B1은 repeat 17의 7.094ms를 포함해 `not_stable`로 분류했다. 나머지 호출은 약 3.63–3.66ms였고 느린 정상 호출을 삭제하지 않았다. 원인은 미확인이다. 다른 조건은 휴리스틱상 preliminary_stable이며 steady state 증명은 아니다.\n\n'
    short += table(['strict/경로 진단','B','native MHA host 횟수','NPU task','backend 진입'], [
        [('Attention 경계' if i >= len(observations)-2 else o['mode']),o['batch'],o['native_mha_host_count'],o['npu_task_events'],o['backend_entries']]
        for i,o in enumerate(observations) if o['status'] == 'ok'])
    short += '\n대표 호출의 NPU task는 codegen 횟수가 아니다. 실제 codegen/cache 계수는 unknown이다. '
    short += 'strict 성공은 SDK 제어 계약의 범위이며 전체 연산 배치 증명은 아니다.\n\n'
    short += table(['조건','B','최대/평균 절대오차','최소 cosine','Top-1 일치'], [
        [labels[o['mode']],o['batch'],f"{o['max_abs_error']:.5f} / {o['mean_abs_error']:.5f}",f"{o['cosine_min']:.8f}",f"{o['top1_matches']}/{o['compared_images']}"]
        for o in output if o['mode'] != 'eager_default'])
    short += '\n실제 사진 고유 4장, 참조는 같은 입력의 이번 기본 Eager logits다. 모든 출력은 [B,1000] BF16 유한값이다. '
    short += 'B1은 B4의 첫 사진이므로 고유 표본 5장이 아니다. 정답 라벨이 없어 accuracy/loss는 계산하지 않았다. 허용 오차 판정은 pending이다.\n\n'
    short += '**조건부 진행:** Attention만 Compile은 가능하며, 이번 전체 호출 시간은 전체 Compile이 가장 짧았다. '
    short += 'fallback 제거 뒤에도 Eager dispatch/프로그램 호출 비용이 남을 수 있으나 비용 비중은 미확인이다. '
    short += 'BF16 출력 허용 기준 합의와 선택 경로의 독립 trial 확대가 다음 최소 조치다. '
    short += '피드백 질문은 SDK native MHA 지원/dispatch 개선 가능 여부와 BF16 atol/rtol 기준이다.\n\n'
    short += f'원본/상세: [study_report.md](study_report.md), [검증](study_verification.json), [그림](figures/01_attention_paths.png). 누적 worker {n:.1f}초/7200초. 기존 결과·환경은 보존했다.\n'
    (h/'feedback_summary.md').write_text(short)
    archive = h/'code/supplemental'
    archive.mkdir(exist_ok=True)
    supplemental = [Path(__file__), Path(__file__).parent/'attention_plots.py', Path(__file__).parent/'ATTENTION_README.md']
    for source in supplemental:
        shutil.copyfile(source, archive/source.name)
    write_json(archive/'manifest.json', {p.name: sha256(p) for p in supplemental})
    print(h/'study_report.md')


if __name__ == '__main__':
    main()
