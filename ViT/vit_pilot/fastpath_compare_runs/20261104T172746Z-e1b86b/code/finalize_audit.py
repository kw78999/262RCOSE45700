"""Final file/count/provenance audit for this run; no device execution."""
import csv
import json
from pathlib import Path
import shutil
import sys

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
sys.path.insert(0, str(ROOT))
from common import now, read_json, sha256, write_json
from fastpath_support import find_case, result_for
from fastpath_compare_verify import verify

verify(RUN)
m = read_json(RUN / 'manifest.json')
cfg = read_json(RUN / 'config.json')
assert m['status'] == 'finished' and m['all_workers_complete'] and m['all_unheld_workers_complete']
latest = [w for w in m['workers'] if w is find_case(m, w['stage'], w['mode'], w['batch'], w['trial'])]
assert len(latest) == 20 and all(w['status'] == 'ok' for w in latest)
assert len(m['workers']) == 22 and not m['all_attempts_successful']
assert m['code_sha256'] == m['code_sha256_after']
for name, digest in m['code_sha256'].items():
    assert sha256(ROOT / name) == digest

with (RUN / 'timings.csv').open() as f:
    perf = list(csv.DictReader(f))
with (RUN / 'cumulative_validation.csv').open() as f:
    prefixes = list(csv.DictReader(f))
assert len(perf) == 672
work_calls = 0
cum_forwards = 0
targets = {}
for w in latest:
    r = result_for(RUN, w)
    if w['stage'] != 'cumulative':
        continue
    target = w['target_calls']
    assert target == {1: 2962, 4: 2896}[w['batch']]
    work = [t for t in r['timings'] if t['phase'] == 'work']
    assert len(work) == target == r['cumulative_work_outputs_checked']
    assert len(r['timings']) == target + 6
    work_calls += target
    cum_forwards += len(r['timings'])
    targets[str(w['batch'])] = target
assert work_calls == 11716 and cum_forwards == 11740
assert len(prefixes) == 5860
for batch, target in targets.items():
    rows = [r for r in prefixes if r['batch'] == batch]
    assert [int(r['N']) for r in rows] == list(range(target + 1))

parsed = []
def reject_constant(value):
    raise ValueError('Non-JSON constant: ' + value)
for folder in [RUN, RUN/'cases', RUN/'code', RUN/'figures']:
    files = folder.glob('*.json') if folder == RUN else folder.rglob('*.json')
    for path in files:
        json.loads(path.read_text(), parse_constant=reject_constant)
        parsed.append(str(path.relative_to(RUN)))
out = read_json(RUN/'output_comparison.json')
assert out['synthetic_input'] is False and out['unique_real_images'] == 4
assert not out['accuracy_computed'] and not out['loss_computed'] and out['tolerance_status'] == 'pending'
for name, digest in read_json(RUN/'figures/manifest.json')['figures'].items():
    assert sha256(RUN/'figures'/name) == digest

archive = RUN/'code/final_tools'
archive.mkdir(exist_ok=True)
for name in ['fastpath_compare_verify.py', 'fastpath_compare_analysis.py', 'fastpath_compare_report.py', 'fastpath_compare_plots.py']:
    shutil.copyfile(ROOT/name, archive/name)
for name in ['fixtures.json', 'test_math.py', 'test_cumulative.py']:
    shutil.copyfile(ROOT/'tests/fastpath_compare'/name, archive/name)
write_json(archive/'manifest.json', {p.name: sha256(p) for p in archive.iterdir() if p.is_file() and p.name != 'manifest.json'})
artifacts = [p for p in RUN.iterdir() if p.is_file() and p.suffix in ['.csv', '.json', '.md']]
m['artifacts'] = {p.name: str(p) for p in artifacts}
m['artifacts']['final_artifact_audit.json'] = str(RUN/'final_artifact_audit.json')
m['artifacts']['figures'] = str(RUN/'figures')
write_json(RUN/'manifest.json', m)
audit = dict(status='pass', utc=now(), logical_workers=20, preserved_attempts=22,
    successful_latest_workers=20, initial_pre_forward_hash_reference_errors_preserved=2,
    main_performance_rows=672, cumulative_work_calls=work_calls, cumulative_forwards=cum_forwards,
    cumulative_wide_prefix_rows=len(prefixes), prefix_N_zero_is_startup_not_missing=True,
    cumulative_targets_per_mode=targets, measurement_code_unchanged=True,
    source_and_output_JSON_nonfinite_constants_absent=True, JSON_files_checked=parsed,
    unique_real_images=4, tolerance_status='pending', pure_compiler_time_ms=None,
    sdk_actual_codegen_count=None, disk_cache_hit=None, disk_cache_miss=None,
    all_workers_complete=True, all_unheld_workers_complete=True, all_attempts_successful=False,
    protected_files_unchanged=True, npu_worker_wall_s=m['npu_budget_used_s'],
    npu_worker_wall_budget_s=cfg['total_budget_s'],
    final_files_sha256={str(p.relative_to(RUN)):sha256(p) for p in artifacts if p.name!='final_artifact_audit.json'},
    plots_visually_inspected=['repeat_comparison.png','startup_components.png','cumulative_time_b1.png','cumulative_time_b4.png'])
write_json(RUN/'final_artifact_audit.json', audit)
print('Final artifact audit PASS:', len(perf), 'main rows,', work_calls, 'separate work calls')
