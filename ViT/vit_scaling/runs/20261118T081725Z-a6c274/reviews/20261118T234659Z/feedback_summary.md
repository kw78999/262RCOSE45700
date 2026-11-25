# 실험 피드백

run 20261118T081725Z-a6c274, 상태 finished_with_coverage_recorded. 정식 mode worker 36, 탐색 mode worker 1. 모든 성능 provisional: 수치 허용 기준 pending.

최고 정식 처리량: {'stage': 'formal', 'pe': 4, 'batch': 4, 'mode': 'compile', 'trials_completed': 3, 'pe_evidence_case': 'diagnostic_p4_b1_compile_t1_a1', 'pe_evidence_batch': 1, 'diagnostic_npu_task_events': 0, 'diagnostic_cluster_indices': [], 'npu_task_evidence': 'not_observed_in_accepted_diagnostic', 'multi_cluster_task_evidence': 'single_cluster_configuration', 'per_pe_utilization_verified': False, 'all_pe_completion_internals': 'SDK stream contract; binary implementation unverified', 'median_trial_mean_ms': 5.5474634409317805, 'min_trial_mean_ms': 5.5407266066481995, 'max_trial_mean_ms': 5.563249592880979, 'trial_mean_sample_sd_ms': 0.011560510645941654, 'trial_mean_cv': 0.0020827948025318533, 'median_trial_images_s': 721.0502678550577, 'median_images_s_per_allocated_pe': 180.2625669637644, 'warmup_counts': [100], 'repeat_counts': [1805, 1798, 1803], 'repeat_measured_s': [10.001011525, 10.002722768, 10.002076584000001], 'stabilization_status': 'heuristic_satisfied', 'range_is_not_confidence_interval': True, 'provisional': True}

핵심 미확인: 실제 codegen/disk cache·전체 op placement·PE별 이용률. 이전과 물리 장치가 다르다. 실패/보류/미측정은 coverage.csv, 재개는 README.md의 단일 명령. 완료 조건은 덮어쓰지 않는다. 상세 RQ 및 마지막 아홉 질문 답변은 report.md.
