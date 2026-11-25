"""Independent CPU verification of persisted raw results against report artifacts."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--review-dir',type=Path,required=True)
    args=ap.parse_args();run=args.run;backup=args.review_dir
    read=lambda name:json.loads((run/name).read_text())
    def rows(name):
        with (run/name).open() as f:return list(csv.DictReader(f))
    failures=[];checks=Counter()
    def check(value,label):
        checks['assertions']+=1
        if not value:failures.append(label)
    def close(a,b,label):
        check(math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-9),label)
    def percentile(v,q):
        ordered=sorted(v);position=(len(ordered)-1)*q
        lower=math.floor(position);upper=math.ceil(position)
        return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)
    manifest=read('manifest.json');cfg=read('experiment_config.json');assets=read('assets.json')
    workers={w['case_id']:w for w in manifest['workers']}
    results={case:json.loads((run/'cases'/case/'result.json').read_text())
             for case in workers if (run/'cases'/case/'result.json').exists()}
    trial_rows=rows('trial_summary.csv')
    formal=[r for r in trial_rows if r['stage']=='formal' and r['eligible_for_descriptive_performance']=='True']
    index={};groups=defaultdict(list)
    for row in formal:
        case=row['case_id'];w=workers[case];r=results[case]
        v=[x['elapsed_ms'] for x in r['timings'] if x['phase']=='repeat']
        warm=[x['elapsed_ms'] for x in r['timings'] if x['phase']=='warmup']
        check(w['status']=='ok' and r['status']=='ok',case+': completed')
        check(w.get('worker_protocol',1)>=cfg['worker_protocol'],case+': logging protocol')
        check(len(v)>=cfg['repeat_min_calls'] and sum(v)>=cfg['repeat_min_seconds']*1000,case+': repeat minima')
        check(len(v)<=cfg['repeat_max_calls'],case+': repeat cap')
        check(len(warm)>=cfg['stabilization']['min_warmup'],case+': warmup minimum')
        check(all(not any(control.values()) for control in r['controls']),case+': strict controls')
        check(r['input_hash']==assets['batches'][str(w['batch'])]['tensor_sha256'],case+': input hash')
        check(r['weights_sha256']==assets['weights_sha256'],case+': weights hash')
        check(r['state_dict_bf16_cpu_sha256']==assets['state_dict_bf16_cpu_sha256'],case+': state hash')
        check(r['fusion']['num_fusion']==w['pe'],case+': requested/SDK allocation')
        expected={'mean_ms':statistics.mean(v),'median_ms':statistics.median(v),
                  'sample_sd_ms':statistics.stdev(v),'cv':statistics.stdev(v)/statistics.mean(v),
                  'p95_ms':percentile(v,.95),'p99_ms':percentile(v,.99),
                  'repeat_sum_ms':sum(v),'images_s':w['batch']*len(v)*1000/sum(v)}
        for key,value in expected.items():close(row[key],value,case+': '+key)
        check(int(row['repeat_count'])==len(v) and int(row['warmup_count'])==len(warm),case+': actual counts')
        key=(w['pe'],w['batch'],w['mode'],w['trial'])
        index[key]=(row,expected)
        groups[key[:3]].append((row,expected))
        checks['formal_trials']+=1
    for row in rows('condition_summary.csv'):
        if row['stage']!='formal':continue
        key=(int(row['pe']),int(row['batch']),row['mode']);group=groups[key]
        means=[r[1]['mean_ms'] for r in group];rates=[r[1]['images_s'] for r in group]
        check(int(row['trials_completed'])==len(group),'condition count '+str(key))
        for name,value in [('median_trial_mean_ms',statistics.median(means)),
                           ('min_trial_mean_ms',min(means)),('max_trial_mean_ms',max(means)),
                           ('trial_mean_sample_sd_ms',statistics.stdev(means)),
                           ('trial_mean_cv',statistics.stdev(means)/statistics.mean(means)),
                           ('median_trial_images_s',statistics.median(rates))]:
            close(row[name],value,'condition '+str(key)+': '+name)
        checks['formal_conditions']+=1
    for row in rows('paired_speedups.csv'):
        if row['stage']!='formal':continue
        p,b,t=int(row['pe']),int(row['batch']),int(row['trial'])
        eager=index[p,b,'eager',t];compiled=index[p,b,'compile',t]
        close(row['eager_over_compile'],eager[1]['mean_ms']/compiled[1]['mean_ms'],'paired ratio '+str((p,b,t)))
        check(eager[0]['warmup_count']==compiled[0]['warmup_count'],'paired common warmup '+str((p,b,t)))
        checks['paired_trials']+=1
    for row in rows('scaling_summary.csv'):
        if row['stage']!='formal':continue
        p,b,t,mode=int(row['pe']),int(row['batch']),int(row['trial']),row['mode']
        ref=index[1,b,mode,t];target=index[p,b,mode,t];speedup=ref[1]['mean_ms']/target[1]['mean_ms']
        close(row['pe_scaling_speedup'],speedup,'scaling speedup '+str((p,b,mode,t)))
        close(row['relative_scaling_efficiency'],speedup/p,'scaling efficiency '+str((p,b,mode,t)))
        check(row['per_pe_utilization_verified']=='False','no utilization claim')
        checks['scaling_rows']+=1
    raw_count=sum(len(r.get('timings',[])) for r in results.values())
    timings=rows('timings.csv');check(len(timings)==raw_count,'raw timings row count')
    by_case=defaultdict(list)
    for row in timings:by_case[row['case_id']].append(row)
    for case,r in results.items():
        raw=r.get('timings',[]);export=by_case[case]
        check(len(raw)==len(export),case+': raw export count')
        check(all(a['phase']==b['phase'] and int(a['iteration'])==int(b['iteration'])
                  and float(a['elapsed_ms'])==float(b['elapsed_ms']) for a,b in zip(raw,export)),case+': raw export values')
    checks['raw_timing_rows']=raw_count
    header_errors=[]
    for case,w in workers.items():
        error=results.get(case,{}).get('error','')
        if w['status']=='error' and 'fatal error: assert.h: No such file or directory' in error:
            header_errors.append(case)
            check(not results[case].get('timings'),case+': failed first call excluded')
            check(case not in {r['case_id'] for r in formal},case+': not formal evidence')
    check(len(header_errors)==sum(w['status']=='error' for w in workers.values()),'all error root causes identified')
    checks['host_header_error_workers']=len(header_errors)
    checks['successful_stability_workers']=sum(w['stage']=='stability' and w['status']=='ok' for w in workers.values())
    check(not rows('stability_bins.csv') if checks['successful_stability_workers']==0 else True,'no invented stability bins')
    old=json.loads((backup/'raw_case_hashes_before.json').read_text())
    for name,digest in old.items():check(hashlib.sha256((run/name).read_bytes()).hexdigest()==digest,'raw case unchanged '+name)
    check(json.loads((backup/'manifest.json').read_text())['workers']==manifest['workers'],'worker manifest records unchanged')
    primary=['trial_summary.csv','condition_summary.csv','paired_speedups.csv','scaling_summary.csv','output_validation.json']
    for name in primary:check((backup/name).read_bytes()==(run/name).read_bytes(),'primary numeric artifact unchanged '+name)
    checks['raw_case_files_unchanged']=len(old)
    report=(run/'report.md').read_text()
    for target in re.findall(r'\]\(([^)]+)\)',report):
        if not target.startswith(('https://','http://','#')):
            check((run/target).exists(),'report link '+target)
    check('부분 완료' in report and 'assert.h' in report,'partial outcome and root cause prominent')
    check('pending' in report and 'provisional' in report,'output tolerance pending retained')
    check('미확인' in report and '신뢰구간이 아니다' in report,'limits retained')
    pngs=list((run/'figures').glob('*.png'));check(len(pngs)==9,'nine PNGs')
    for png in pngs:
        check(png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'),'valid PNG '+png.name)
        check(png.with_suffix('.csv').exists(),'matching CSV '+png.name)
    check({r['case_id'] for r in rows('figures/07_trial_means.csv')}=={r['case_id'] for r in formal},'trial figure includes every eligible formal trial')
    check({r['case_id'] for r in rows('figures/09_initial_costs.csv')}=={r['case_id'] for r in formal},'initial cost figure includes only current completed formal trials')
    import torch
    output=read('output_validation.json');tensors={}
    for item in output['checks']:
        case=item['case_id'];r=results[case]
        y=torch.load(r['output_file'],map_location='cpu',weights_only=True).float()
        tensors[case]=y
        check(list(y.shape)==[item['batch'],1000] and bool(torch.isfinite(y).all()),case+': finite shape')
        check(item['hash_matches_expected'] is True,case+': validation input hash')
    for comparison in output['comparisons']:
        if 'all_batch_rows' not in comparison:continue
        x=tensors[comparison['reference']];y=tensors[comparison['case']]
        expected=comparison['all_batch_rows'];diff=(x-y).abs()
        close(expected['max_abs_error'],diff.max().item(),'output max abs')
        close(expected['mean_abs_error'],diff.mean().item(),'output mean abs')
        check(expected['top1_matches']==int((x.argmax(-1)==y.argmax(-1)).sum()),'output top1 counts')
    checks['finite_output_checks']=len(output['checks'])
    audit={'utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if not failures else 'failed',
           'scope':'CPU artifact consistency and report review; not numerical tolerance approval or completion of missing experiments',
           'checks':dict(checks),'failures':failures,'primary_numeric_artifacts_unchanged':primary,
           'experiment_status':manifest['status'],'environment_error':'assert.h not found by SDK host C compiler',
           'npu_calls_added':0}
    (run/'report_review.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    (backup/'review_result.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    text=f"""# 보고서 검토 결과

검토 {audit['utc']}. 판정: **{audit['status']} — 원본과 통계·문서의 일관성 검토**. 전체 실험 목표 완료나 출력 수치 허용 기준의 합격 판정은 아니다.

- 현재 규약 formal {checks['formal_trials']}개, PE×batch×mode {checks['formal_conditions']}개, paired {checks['paired_trials']}쌍을 원본 repeat에서 독립 재계산했다.
- 평균·중앙값·표본 SD·CV·p95/p99·측정 합계·images/s, trial 그룹 중앙값·범위·SD·CV, paired 배속·PE scaling·상대 효율을 대조했다.
- raw timings {raw_count:,}행과 case 원본을 대조했다. 원본 case 파일 {len(old)}개 및 worker manifest 기록의 불변을 확인했다.
- trial_summary·condition_summary·paired_speedups·scaling_summary·output_validation은 보완 전과 바이트 단위로 동일하다.
- 출력 {checks['finite_output_checks']}개를 CPU에서 다시 읽어 shape·finite·입력 식별을 확인하고 저장된 오차·Top1 수치를 대조했다. 허용 오차 pending은 유지했다.
- PNG 9개·대응 CSV·보고서 링크를 확인했다. 성공 stability {checks['successful_stability_workers']}개로 추세 그림에 측정값을 만들지 않았다.

## 발견해 보완한 내용

1. 재개 후 {len(header_errors)}개 error의 원인은 공통으로 SDK 내부 C 컴파일의 assert.h 미발견이다. 모델/PE 미지원으로 분류하지 않았으며 [failure_summary.csv](failure_summary.csv)에 기록했다.
2. 대표5와 stability를 완료하지 못한 이유를 보고서 앞에 명시했다. 제어 루프 종료를 전체 연구 성공으로 표현하지 않았다.
3. 실제 측정 종료는 2026-11-19 04:22:05 KST이며 약55.3분 예산이 남아 있었다. 나중에 보고서를 재집계한 시간을 측정 elapsed에 더하지 않도록 고쳤다.
4. coverage에서는 실제 환경 오류·timeout·중단을 과거 budget hold 문구보다 먼저 표시하며, 하위 배치 timeout에 따른 확대 보류를 명시했다.
5. 전체 이벤트 14개 중 실제 보류는9개다. 기존의 모든 이벤트를 보류로 세던 서술을 수정했다.
6. trial 평균 CV 단위를 %로 명시하고 B1→B4 처리량·지연시간 표, 그림9개와 CSV 링크, 읽을 수 있는 feedback 요약을 추가했다.

## 남은 한계

대표5·300초 stability·B16 이상 정식 비교, 수치 허용 기준, Compile 다중PE 실제 활용·모든 op placement·binary 내부 동기화·codegen/cache는 미확인이다. 공통 환경 오류를 제어기가 전체 중단 사유로 판단하지 못한 실행 제어의 한계도 남아 있다. 이번 작업은 보고서 검토이며 환경 복구·SDK/드라이버 변경·새 NPU 실험은 수행하지 않았다.

원본 자동 보고서와 CSV는 reviews/{backup.name}/에 보존했다. 현재 [report.md](report.md), [기계 판독 검토 결과](report_review.json), [feedback_summary.md](feedback_summary.md)를 함께 제공한다.
"""
    (run/'report_review.md').write_text(text)
    print(json.dumps({'status':audit['status'],'checks':dict(checks),'failures':failures[:10]},ensure_ascii=False))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
