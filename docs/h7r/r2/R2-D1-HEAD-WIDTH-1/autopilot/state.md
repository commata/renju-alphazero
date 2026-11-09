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
| P1 전체 규모 학습 6회 + fit | 완료: 6/6 `CANDIDATE`, 비교가능, `WIDTH_EFFECT` | 계약 `d1-full-contract.json`(실험 `25c49e408aff`), prepare `2142f86` | `d8c0d10` | — |
| P2 factory v2 | 구현 완료, 시간 gate 실행 대기 | `analysis.tb_v3_net_v2`, `scripts/h7r_factory_v2_timing.py` | — | D1 초안 §16.3 v2-1 (학습 기록 커밋 뒤) |
| P3 D4 계약·도구 | 고정(`approved` 모두 false) | `d4/d1-d4-contract.json`, `scripts/h7r_d1_d4.py` | — | §16.3 d4-1 (eligible manifest) |
| P4 전체 규모 H-dev | 승인됨, 실행 대기 | 학습 기록 `d8c0d10`, P2·P3 고정 커밋 뒤 | — | §14.8 full-3 (full-3-R / full-3-F) → §16.3 v2-1 → d4-1 |
| P3' D4 smoke·본 평가 | P4 `DEVELOPMENT_SIGNAL`·시간 gate PASS일 때 | — | — | §16.3 d4-run smoke → full (실패 시 재개 1회 또는 d4-F) |
| P5 확인 평가 1세트 | D4 `ALL_THREE_BETTER`일 때 | — | — | — |
| P6 R4 프로그램 준비 | 세 고리 통과 시 | — | — | 최종 승인 |

## 기록

- 2026-11-09: 위임 기록 커밋, P1 train 승인(근거: prepare PASS `2142f86`, `checked_prepare` 통과)
- 2026-11-09: P1 학습 블록 실행 시작 15:24:27 KST(24시간 상한의 시작). 첫 시도는 블록 파일을 만들지 못해 실행되지 않았다(`BLOCK EXIT 127`, 저장소 변화 없음).
- 2026-11-09: P2 factory v2와 시간 gate 도구, P3 D4 도구·계약을 고정했다(D1 초안 §16). 전체 규모 H-dev evaluate 승인 전이다.
- 2026-11-09: P1 완료(`d8c0d10`, 15:47). 6회 모두 `CANDIDATE`, 환경 필드 12개 동일(비교가능), train-fit `WIDTH_EFFECT`(wide − narrow: 31001 −0.0231, 31002 −0.0278, 31003 −0.0210). `checked_fit` 재계산이 기록과 같다. D4 모델 연결도 확인했다: 선택 checkpoint의 metadata가 `best.json`과 같고, frozen state가 `c2fbaac3…`, wide는 `value_channels` 4.
- 2026-11-09: P4 evaluate 승인(근거: 위 확인, P2·P3 고정 커밋이 이 승인보다 앞선다).
- 2026-11-09: 데스크톱 자동 진행 도구와 지시문 추가(D1 초안 §16.5). P4부터 D4 full까지 데스크톱 에이전트가 이 도구로 진행한다.
- 2026-11-09 22:56: P4 H-dev `DEVELOPMENT_SIGNAL`(`9a419af`; 개선폭 31001 0.00884, 31002 0.00777, 31003 0.01073, ε 0.00195, HOLD 없음).
- 2026-11-09 22:57: 시간 gate 첫 시도(launch 2)가 측정 전에 멈춤: factory v1이 D1 checkpoint를 거부(도구 결함), 실패 기록 경로가 `.gitignore`에 걸림. 개정 A1(D1 초안 §16.6)로 고치고, `retries.json`으로 1회 재시도를 지정.
- 2026-11-09 23:51 KST: D4 `approved.smoke` (autopilot driver; 근거: D4 entry checks pass on the committed records (fit comparable, H-dev DEVELOPMENT_SIGNAL, factory v2 time gate passed on the linked seed-31001 pair); eligible manifest committed).
- 2026-11-09 23:50: 시간 gate 재시도 PASS(`b939f2c`, 144위치), eligible PASS(`0513df3`), smoke 승인(`2897a44`).
- 2026-11-09 23:51: smoke(launch 5)가 preflight에서 멈춤: 승인 커밋이 eligible manifest가 묶은 계약 파일 SHA를 바꿈(도구 설계 결함). 실패 기록 보존(`e517117`). 개정 A2(D1 초안 §16.7): manifest를 승인 전 계약 SHA에 묶고, 계약 재고정(승인 초기화, 표본 `d4/eligible-a2.json`), `retries.json`으로 eligible·smoke 1회 재시도 지정.
- 2026-11-10: 검토 지적 4건(실행 중 pull, staged 기록과 복구, 상한, tmux 시작 실패)을 자동 진행 도구에서 고침(D1 초안 §16.8). 계약·기록 변경 없음.
- 2026-11-10 00:20 KST: D4 `approved.smoke` (autopilot driver; 근거: D4 entry checks pass on the committed records (fit comparable, H-dev DEVELOPMENT_SIGNAL, factory v2 time gate passed on the linked seed-31001 pair); eligible manifest committed).
- 2026-11-10 00:22 KST: D4 `approved.full` (autopilot driver; 근거: the committed D4 smoke is a reproduced SMOKE of this experiment; 15.0 h left of the delegation).
