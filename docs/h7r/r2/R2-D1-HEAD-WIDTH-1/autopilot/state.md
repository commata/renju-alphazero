# R2-D1-AUTOPILOT-1 상태 기록

- 위임 기록: `delegation-1.json` (범위·결정값·진행 조건·상한·중단 조건)
- 이 파일은 단계가 끝날 때마다 갱신·커밋한다. 세션이 끊기면 새 세션은 이 파일과 위임 기록만 읽고 이어간다.
- 실행 위치
  - 클라우드 세션: 코드·계약·테스트·승인 커밋·기록 분석
  - 데스크톱 `~/renju-h6`: GPU·Rapfi 실행. 긴 작업은 tmux로 띄우고, 실행 중에는 HEAD를 고정한다.
- 벽시계 상한 24시간의 시작: P1 학습 블록을 처음 실행한 시각 **2026-11-09 15:24:27 KST** → 상한 2026-11-10 15:24:27 KST

| 단계 | 상태 | 입력 SHA / 커밋 | 완료 기록 | 다음 명령 |
|---|---|---|---|---|
| P0 위임 기록 | 완료 | 시작 `2142f86` | 이 커밋 | — |
| P1 전체 규모 학습 6회 + fit | 실행 중(15:24:27 시작) | 계약 `d1-full-contract.json`(실험 `25c49e408aff`), prepare `2142f86` | — | D1 초안 §14.8 full-2 (중단 시 full-2-F) |
| P2 factory v2 | 구현 완료, 시간 gate 실행 대기 | `analysis.tb_v3_net_v2`, `scripts/h7r_factory_v2_timing.py` | — | D1 초안 §16.3 v2-1 (학습 기록 커밋 뒤) |
| P3 D4 계약·도구 | 고정(`approved` 모두 false) | `d4/d1-d4-contract.json`, `scripts/h7r_d1_d4.py` | — | §16.3 d4-1 (eligible manifest) |
| P4 전체 규모 H-dev | P2·P3 계약 고정 뒤 승인 | — | — | §14.8 full-3 (full-3-R / full-3-F) |
| P3' D4 smoke·본 평가 | P4 `DEVELOPMENT_SIGNAL`·시간 gate PASS일 때 | — | — | §16.3 d4-run smoke → full (실패 시 재개 1회 또는 d4-F) |
| P5 확인 평가 1세트 | D4 `ALL_THREE_BETTER`일 때 | — | — | — |
| P6 R4 프로그램 준비 | 세 고리 통과 시 | — | — | 최종 승인 |

## 기록

- 2026-11-09: 위임 기록 커밋, P1 train 승인(근거: prepare PASS `2142f86`, `checked_prepare` 통과)
- 2026-11-09: P1 학습 블록 실행 시작 15:24:27 KST(24시간 상한의 시작). 첫 시도는 블록 파일을 만들지 못해 실행되지 않았다(`BLOCK EXIT 127`, 저장소 변화 없음).
- 2026-11-09: P2 factory v2와 시간 gate 도구, P3 D4 도구·계약을 고정했다(D1 초안 §16). 전체 규모 H-dev evaluate 승인 전이다.
