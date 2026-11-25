"""Recompute raw research measurements and compare saved real-image logits."""
import math
from pathlib import Path
import statistics as st
from common import read_json, write_json, write_csv, sha256

MODES = ['eager_default', 'attention_compile', 'eager_fastpath_off', 'full_compile']


def compare_files(candidate, reference, input_ids):
    import torch
    from torchvision.models import ViT_B_16_Weights
    a = torch.load(candidate, map_location='cpu', weights_only=True).float()
    b = torch.load(reference, map_location='cpu', weights_only=True).float()
    r = dict(candidate=str(candidate), reference=str(reference),
             tolerance_status='pending', allclose=None, accuracy_computed=False,
             structural_finite_ok=False, top1_matches=None, compared_images=None)
    if a.shape != b.shape or list(a.shape) != [len(input_ids), 1000]:
        r['reason'] = 'Shape mismatch'
        return r
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        r['reason'] = 'NaN/Inf present'
        return r
    delta = (a-b).abs()
    ad, bd = a.double(), b.double()
    norms = ad.norm(dim=1)*bd.norm(dim=1)
    cos = [float((ad[i]*bd[i]).sum()/norms[i]) if float(norms[i]) > 0 else None for i in range(len(a))]
    ai, bi = a.argmax(1), b.argmax(1)
    names = ViT_B_16_Weights.IMAGENET1K_V1.meta['categories']
    r.update(structural_finite_ok=True, max_abs_error=float(delta.max()), mean_abs_error=float(delta.mean()),
        cosine_per_image=cos, cosine_mean=st.mean(cos) if all(c is not None for c in cos) else None,
        cosine_min=min(cos) if all(c is not None for c in cos) else None,
        cosine_undefined_reason='zero norm' if any(c is None for c in cos) else None,
        top1_matches=int((ai == bi).sum()), compared_images=len(a), top1_ratio=float((ai == bi).float().mean()),
        images=[dict(input_id=input_ids[i], candidate_class=int(ai[i]), reference_class=int(bi[i]),
            candidate_class_name=names[int(ai[i])], reference_class_name=names[int(bi[i])],
            candidate_logit=float(a[i, ai[i]]), reference_logit=float(b[i, bi[i]]),
            top1_match=bool(ai[i] == bi[i]), cosine=cos[i]) for i in range(len(a))])
    r['mismatches'] = [x for x in r['images'] if not x['top1_match']]
    return r


def fmt(x):
    return '—' if x is None else f'{x:.4f}' if isinstance(x, float) else str(x)


def table(headers, rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(fmt(v) for v in row)+' |\n' for row in rows)


def analyze(run):
    run = Path(run)
    cfg, m, assets = [read_json(run/name) for name in ['config.json', 'manifest.json', 'assets.json']]
    results, timings, summaries, observations, output_checks, checks = {}, [], [], [], [], []
    for w in m['workers']:
        path = run/'cases'/w['case_id']/'result.json'
        r = read_json(path) if path.exists() else {}
        results[w['case_id']] = r
        if r:
            checks.append(dict(check='input_weights_dtype', case_id=w['case_id'], passed=(
                r.get('input_hash') == assets['batch_hashes'][str(w['batch'])] and
                r.get('weights_sha256') == assets['weights_sha256'] and r.get('dtype') == 'torch.bfloat16')))
        if r.get('output_file'):
            for ref_mode in ['eager', 'compile']:
                ref = Path(cfg['assets_dir'])/'outputs'/f"validation_{ref_mode}_b{w['batch']}.pt"
                ref_result = read_json(Path(cfg['assets_dir'])/'cases'/f"validation_{ref_mode}_b{w['batch']}"/'result.json')
                compatible = all(r.get(k) == ref_result.get(k) for k in ['input_hash', 'weights_sha256', 'dtype'])
                checks.append(dict(check='reference_compatibility', case_id=w['case_id'], reference=ref_mode, passed=compatible))
                if compatible:
                    comp = compare_files(Path(r['output_file']), ref, assets['input_ids'][:w['batch']])
                    output_checks.append(dict(case_id=w['case_id'], reference_mode=ref_mode, **comp))
        rows = r.get('timings', [])
        if w['stage'] == 'performance':
            timings.extend(rows)
            valid = [x['elapsed_ms'] for x in rows if x['phase'] == 'repeat' and x['status'] == 'ok'
                     and x['timing_valid'] and math.isfinite(x['elapsed_ms']) and x['elapsed_ms'] > 0]
            completed = [x for x in rows if x['phase'] == 'repeat' and x['status'] == 'ok']
            mean = st.mean(valid) if valid else None
            sd = st.stdev(valid) if len(valid) > 1 else None
            drift = abs(st.mean(valid[:len(valid)//2])-st.mean(valid[len(valid)//2:]))/mean if len(valid) >= 10 else None
            stable = 'unknown' if drift is None else 'preliminary_stable' if sd/mean <= .1 and drift <= .1 else 'not_stable'
            output_comps = [c for c in output_checks if c['case_id'] == w['case_id']]
            pred_ok = bool(output_comps) and all(c['structural_finite_ok'] and c['top1_matches'] == c['compared_images'] for c in output_comps)
            summary = dict(case_id=w['case_id'], mode=w['mode'], batch=w['batch'], trial=1,
                status=w['status'], model_id=assets['model_id'], weights_id=assets['weights_id'], dtype='torch.bfloat16',
                shape=assets['input_shapes'][str(w['batch'])], requested_repeats=cfg['repeats'],
                completed_repeats=len(completed), valid_repeats=len(valid),
                first_call_ms=next((x['elapsed_ms'] for x in rows if x['phase'] == 'first' and x['status'] == 'ok' and x['timing_valid']), None),
                runner_setup_ms=r.get('runner_setup_ms'), mean_ms=mean, median_ms=st.median(valid) if valid else None,
                stdev_ms=sd, throughput_images_s=w['batch']*len(valid)*1000/sum(valid) if valid and w['status'] == 'ok' and r.get('output_valid') else None,
                compiler_time_ms=None, warm_state_assessment=stable,
                warm_state_reason=f'CV={sd/mean if sd is not None and mean else None}; half-mean drift={drift}; heuristic, compile-event absence unproven',
                validation_status='structure_finite_top1_match_tolerance_pending' if pred_ok else 'failed_or_unavailable',
                execution_path_status='CPU native attention fallback confirmed' if w['mode'] == 'eager_default' else 'strict diagnostic evidence; full placement unverified',
                timing_scope=r.get('timing_scope'), cache_state=r.get('cache_state'), input_hash=r.get('input_hash'),
                weights_sha256=r.get('weights_sha256'))
            summaries.append(summary)
            if w['status'] == 'ok':
                checks.append(dict(check='timing_row_count', case_id=w['case_id'], passed=len(rows) == 1+cfg['warmup']+cfg['repeats']))
        else:
            prof = run/'cases'/w['case_id']/'torch_cpu_summary.json'
            cpu = read_json(prof) if prof.exists() else []
            native = [e for e in cpu if e['name'] == 'aten::_native_multi_head_attention']
            trace = run/'cases'/w['case_id']/'furiosa_trace.json'
            events = read_json(trace).get('traceEvents', []) if trace.exists() else []
            stderr = run/'logs'/f"{w['case_id']}.stderr.log"
            log = stderr.read_text() if stderr.exists() else ''
            observations.append(dict(case_id=w['case_id'], mode=w['mode'], batch=w['batch'], strict=w['strict'],
                status=w['status'], native_mha_host_count=sum(e['count'] for e in native) if cpu else None,
                native_mha_host_inclusive_ms=sum(e['total_cpu_ms'] for e in native) if cpu else None,
                npu_task_events=sum('::Task::' in e.get('name', '') for e in events) if events else None,
                backend_entries=len(r.get('backend_entries', [])) if r else None,
                backend_targets=[x['targets'] for x in r.get('backend_entries', [])],
                recompile_limit_warning=('hit config.recompile_limit' in log or 'hit config.accumulated_recompile_limit' in log),
                recompile_logs_present='[__recompiles]' in log, graph_break_logs_present='[__graph_breaks]' in log,
                actual_codegen_count=None, disk_cache_hit=None, disk_cache_miss=None,
                log_reference=str(stderr), error=r.get('error')))
    write_csv(run/'timings.csv', timings, ['run_id','case_id','mode','batch','trial','phase','iteration','elapsed_ms','timing_valid','status'])
    fields = list(summaries[0]) if summaries else ['case_id', 'mode', 'batch', 'status']
    write_csv(run/'summary.csv', summaries, fields)
    write_csv(run/'path_observations.csv', observations, ['case_id','mode','batch','strict','status','native_mha_host_count',
        'native_mha_host_inclusive_ms','npu_task_events','backend_entries','recompile_limit_warning','recompile_logs_present',
        'graph_break_logs_present','actual_codegen_count','disk_cache_hit','disk_cache_miss','log_reference'])
    write_json(run/'path_observations.json', observations)
    write_json(run/'output_comparison.json', dict(synthetic_input=False, unique_images=4,
        tolerance_status='pending', accuracy_computed=False, loss_computed=False, comparisons=output_checks))
    # Same-batch, same-run diagnostic wall-time ratios; mixed baseline is explicitly identified.
    ratios = []
    for b in [1, 4]:
        base = next((s for s in summaries if s['batch'] == b and s['mode'] == 'eager_default'), None)
        for mode in MODES[1:]:
            candidate = next((s for s in summaries if s['batch'] == b and s['mode'] == mode), None)
            ok = base and candidate and all(s['status'] == 'ok' and s['valid_repeats'] == cfg['repeats']
                and s['validation_status'] == 'structure_finite_top1_match_tolerance_pending' for s in [base, candidate])
            compatible = bool(ok) and all(base[k] == candidate[k] for k in ['input_hash','weights_sha256','dtype','timing_scope','cache_state'])
            ratios.append(dict(batch=b, baseline_case=base['case_id'] if base else None,
                candidate_case=candidate['case_id'] if candidate else None, candidate_mode=mode,
                baseline_mean_over_candidate_mean=base['mean_ms']/candidate['mean_ms'] if compatible else None,
                comparison_possible=compatible, scope='preliminary whole-model wall time; baseline includes CPU tensor fallback; not NPU-only speedup',
                tolerance_status='pending', reason=None if compatible else 'Incomplete/invalid/incompatible pair'))
    write_csv(run/'comparisons.csv', ratios, list(ratios[0]))
    checks.append(dict(check='original_code_unchanged_during_run', passed=m['code_sha256'] == m.get('code_sha256_after', m['code_sha256'])))
    checks.append(dict(check='asset_file_hash', passed=sha256(Path(cfg['assets_dir'])/'inputs.pt') == assets['input_file_sha256']))
    write_json(run/'verification.json', dict(status='pass' if all(c['passed'] for c in checks) else 'fail',
        checks=checks, performance_records=len(timings), expected_if_all_completed=8*(1+cfg['warmup']+cfg['repeats']),
        completed_performance_workers=sum(s['status'] == 'ok' for s in summaries),
        diagnostic_records=sum(len(results[w['case_id']].get('timings', [])) for w in m['workers'] if w['stage'] == 'diagnostic')))
    text = '# Attention CPU fallback 연구 — 예비 실험\n\n'
    text += '## 확인한 원인\n\n[실행 확인/소스 확인] PyTorch native MHA 연산은 PrivateUse1 전용 kernel이 없고 SDK backend fallback이 등록되어 있다. '
    text += 'eval/inference_mode의 ViT self-attention이 이 fastpath를 선택한다. strict 오류와 CPU profiler를 함께 근거로 사용한다. '
    text += 'torch의 +cpu 버전 표기나 output.device만으로 판정하지 않는다.\n\n'
    text += '[소스 확인] torch/nn/modules/activation.py:1338–1445 및 _check_arg_device는 PrivateUse1 장치를 허용하지만 native kernel 존재를 검사하지 않는다. '
    text += 'torch/_dynamo/variables/torch.py:213은 tracing 중 _is_make_fx_tracing을 True로 취급하므로 Compile은 native 경로를 건너뛴다. '
    text += '기존 전체 Compile FX graph에서도 functional.multi_head_attention_forward를 관측했다. '
    text += 'furiosa/torch/_db/local_decomp.py:1392,1452,2478–2479에는 SDPA math 분해가 있다. 설치 소스 사본·해시는 logs/attention_installed_sources.txt와 manifest에 보존한다.\n\n'
    text += '## 이번 실행\n\n'
    text += f"실제 입력은 `{cfg['assets_dir']}/inputs/preprocessed_bf16.pt`를 다시 사용했다. 고유 사진 4개, B1은 첫 cat, B4는 전체 4개이며 이미지당 토큰은 197개다. "
    text += '입력 파일/전처리 텐서/가중치 해시는 assets.json, 원본 출처·순서·전처리는 image_manifest.json에 보존했다. 정답 라벨 없이 logits와 예측 일치성만 비교한다.\n\n'
    text += 'Attention만 Compile은 기존 12개 self_attention 모듈에 Module.compile(backend=ft.backend)을 한 번씩 적용한다. 나머지 ViT는 Eager다. '
    text += 'Attention 수식·가중치·need_weights=False·dtype·해상도를 바꾸지 않는다. fastpath-off Eager는 공개 API로 worker 내부 실행 경로만 변경하고 원래 값으로 복원한다. '
    text += '이는 기본 Eager와 구분한 ablation이다. Compile dynamic/fullgraph/mode/recompile_limit은 조정하지 않았다.\n\n'
    text += 'CPU fallback 금지 진단과 profiler는 별도 worker에서 수행하고, 성능 worker에는 상세 로그·프로파일·backend observer를 넣지 않았다. '
    text += f"warmup {cfg['warmup']}회, repeat {cfg['repeats']}회, 독립 trial 1회인 초기 진단이다. "
    text += '시간은 전체 ViT 배치 호출부터 공식 current-stream 완료까지의 CPU 관측 ms다. 내부 dispatch·복사·대기는 포함하고 외부 전처리·준비·출력 복사는 제외한다. 순수 Attention/NPU kernel 시간은 아니다.\n\n'
    text += table(['조건','B','상태','setup ms','first ms','repeat mean ± sample SD ms','median ms','images/s'], [
        [s['mode'],s['batch'],s['status'],s['runner_setup_ms'],s['first_call_ms'],f"{fmt(s['mean_ms'])} ± {fmt(s['stdev_ms'])}",s['median_ms'],s['throughput_images_s']] for s in summaries])
    text += '\n기본 Eager는 CPU fallback이 포함된 혼합 실행 기준선이다. comparisons.csv의 비율은 이 혼합 기준선 대비 전체 호출 시간의 예비 비율이며 NPU-only 가속률이 아니다. '
    text += '서로 다른 batch의 비율은 평균내지 않는다. 느린 정상 호출을 삭제하지 않았으며 throughput은 순차 반복 처리량이다. first-call은 순수 compile 시간이 아니다.\n\n'
    text += table(['진단','strict','상태','CPU native MHA 횟수','NPU task events','backend 진입','limit 경고'], [
        [o['case_id'],o['strict'],o['status'],o['native_mha_host_count'],o['npu_task_events'],o['backend_entries'],o['recompile_limit_warning']] for o in observations])
    text += '\nstrict 성공은 해당 SDK fallback 제어 계약 범위의 결과이며 전체 연산 배치의 완전한 증명은 아니다. CPU host span은 NPU kernel 시간이 아니고 중첩 구간을 더해 원인별 비중을 만들지 않는다. '
    text += 'NPU task 수, backend 진입, 실제 codegen 수, disk cache hit/miss는 별개다. 마지막 두 항목은 미확인(null)이다. '
    text += '각 worker의 SDK cache는 처음 없던 전용 경로이며, 시스템 전체 cold 또는 실제 miss를 보장하지 않는다.\n\n'
    text += table(['조건','참조','max abs','mean abs','cos mean','cos min','Top-1 분자/분모'], [
        [c['case_id'],c['reference_mode'],c.get('max_abs_error'),c.get('mean_abs_error'),c.get('cosine_mean'),c.get('cosine_min'),f"{c['top1_matches']}/{c['compared_images']}"] for c in output_checks if c['case_id'].startswith('performance')])
    text += '\n출력 허용 오차는 pending이다. Top-1 일치율은 정답 accuracy가 아니며 B1과 B4를 고유 이미지 5개로 합산하지 않는다.\n\n'
    text += '## 해결 방향과 해석 범위\n\n1. 전체 Compile: 기존 strict B1/B4 성공 근거가 있고 새 control 시간을 위 표에 남겼다.\n'
    text += '2. Attention만 Compile: 모든 12개 모듈을 감싸되 strict·FX graph·CPU profiler·출력·전체 호출 시간으로 실제 효과를 판단한다. 나머지 Eager dispatch와 fusion 경계 비용이 남을 수 있다.\n'
    text += '3. MHA fastpath 해제: native 연산 대신 기존 functional MHA/SDPA 경로를 선택하는 별도 조건이다. 다른 미지원 연산이 나타나면 설정을 숨기거나 수정하지 않는다.\n'
    text += '4. 근본 지원 개선: SDK 공급자에게 native MHA PrivateUse1 kernel 또는 kernel 지원을 고려한 dispatch 처리를 문의할 근거가 된다. 현재 환경 업그레이드·커널 개발은 수행하지 않는다.\n\n'
    text += f"worker 기본 상한 {cfg['worker_timeout_s']}초, 전체 NPU worker 예산 {cfg['total_budget_s']}초, 실제 누적 {m['npu_used_s']:.1f}초. 타임아웃은 미지원 확정이 아니다. "
    text += '원본 합성 및 실제 이미지 후속 run은 변경하지 않았다. 새 실험은 이전 범위에 추가된 사용자 요청의 Attention 부분 compile 연구로 분리했다.\n\n'
    text += '## 재현\n\n```bash\ncd /root/ViT\n/root/venv/bin/python -u vit_pilot/attention.py run\n'
    text += f'/root/venv/bin/python vit_pilot/attention.py analyze --run {run}\n```\n\n'
    text += '공식 참고: [PyTorch 2.10 MHA](https://docs.pytorch.org/docs/2.10/generated/torch.nn.MultiheadAttention.html), '
    text += '[mha backend 제어](https://docs.pytorch.org/docs/2.10/backends.html#torch-backends-mha). 실제 설치 소스와 실행 로그가 이 환경의 직접 근거다.\n'
    (run/'report.md').write_text(text)
    (run/'feedback_summary.md').write_text(text)
    return summaries
