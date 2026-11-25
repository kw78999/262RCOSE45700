# PE 설정과 증거

② 설치 SDK 2026.4.0 `_C/device/__init__.pyi` Fusion 문서 및 `set_fusion(int)` API를 확인했다. `ft.set_fusion(P)`를 import 직후, device/stream 조회·모델 이동·compile 전에 새 프로세스에서 딱 한 번 호출한다. `_C/device`는 num_fusion=1/2/4/8/16/32를 문서화한다. 이번 1-chip의 후보는1/2/4/8뿐이다. 1/2/4는 한 cluster 하드웨어 fusion, 8은 4PE×2 cluster 소프트웨어 fusion이다. MeshKind.Single 기본을 유지하며 mesh·tensor_parallel_size·모델 복제·다중 worker는 사용하지 않는다.

② Eager와 Compile은 같은 프로세스 전역 Fusion을 사용한다. Eager의 ATen graph 실행과 Compile의 Runnable 생성은 이 설정 후 실행한다. `_module/runnable.py:173`와 `_module/visa.py`는 runnable.expected_num_fusion이 현재 Fusion과 같아야 실행함을 검사한다. PE간 산출물 호환을 가정하지 않고 조건·mode·trial·attempt마다 별도 cache_context를 사용한다. 컴파일러 설정/모델/정밀도는 그대로다.

② `config/_stream.py`는 대상 device의 NPU 연산이 current stream에 제출됨을 설명하고 `_C/stream/__init__.pyi`는 해당 stream에 제출된 모든 작업 완료까지 synchronize가 대기함을 설명한다. 하나의 fused logical device/동일 current stream만 사용하는 호출에 이 계약을 적용한다. 관련 없는 다른 stream은 기다리지 않는다. 8PE 내부 runtime의 세부 동기화 구현은 binary로 배포돼 소스 검증 불가하므로 per-PE 하위 구현에 대한 독립 증명은 미확인이다. diagnostic의 fused-device NPU task trace와 완료 query를 추가 근거로 저장하며 하드웨어 사용률 보장으로 해석하지 않는다.

① requested PE는 spec.pe, SDK 설정은 result.fusion 및 device_configuration, 실제 관측은 diagnostic path_observations와 furiosa_trace.json, 별도 furiosa-smi ps의 NPU core 구성이다. 각 값은 분리 기록한다. PE 설정 일치만으로 모든 PE의 유효 활용을 확정하지 않는다. trace에 보이는 chip_index/cluster_index와 core 이름은 그대로 남기며 Task 수를 커널 수/fusion 수로 바꾸지 않는다.

④ 실제 compiler codegen 시간/횟수 및 disk hit/miss와 전체 op placement는 미확인. get_fusion/get_device_configuration은 할당 요청·SDK 구성 근거이며 연산별 PE 활용률 측정값이 아니다. installed source 원문·SHA256은 evidence/ 참조.

공개 공식 자료도 확인했으나 set_fusion의 버전별 상세 계약은 검색 결과로 확인하지 못해 설치 stub을 우선 근거로 사용했다: https://developer.furiosa.ai/v2026.4.0/en/ 및 https://furiosa.ai/developers (2026-11-18 확인). SDK/드라이버를 업데이트하지 않았다.


Python 실행 경로에서 VisionTransformer 인스턴스는 하나이고 하나의 입력 batch를 fused logical device 하나에 전달한다. 별도 모델 replica/복수 worker/data-parallel 실행은 구성하지 않았다. 8PE software fusion 내부의 tensor 분할·가중치 복제 전략은 설치 공개 소스로 확인하지 못했으며 내부 구현에 모델 복제가 전혀 없다고 보장하지 않는다. Eager B1 진단에서는 cluster0/1의 stream_id0 task를 함께 관측했다. Compile2/4/8의 최신 승인 진단에서는 task 기록이 없으므로 해당 mode의 PE 활용은 할당·host launch 근거와 구분해 미확인으로 표시한다.
