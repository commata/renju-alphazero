# R2-D4-BASELINE-1 X1 — D4 기록의 사후 탐색 부록

- 작성일 2026-11-02, 같은 날 확정
- 상태: **확정(fixed)**. 사후 탐색이며, 기존 자료를 쓰는 CPU 분석만 승인 범위에 든다.
- 고정 계획 파일: `docs/h7r/r2/R2-D4-BASELINE-1/x1-plan.json`. 모든 입력을 전체 SHA-256으로 고정한다.
- 도구: `scripts/h7r_d4_x1.py`(읽기 전용), 테스트 `tests/test_h7r_d4_x1.py`
- 데스크톱 실행 명령: `docs/h7-rl-plan.md` §9.24

## 0. 성격과 바꾸지 않는 것

- 이 계획은 D4 full 결과(`NOT_ESTABLISHED`)를 **본 뒤에** 썼다.
  - 기존 기록을 기술적으로 살펴보는 **사후(post-hoc) 탐색**이다.
  - 확인 실험의 사전 등록이 **아니다**.
- 결과는 다음 하위 계약을 고르는 참고 자료일 뿐이다. 독립 증거나 확인 근거로 쓰지 않는다. 유망해 보이는 패턴이 나와도, 확인하려면 새 표본으로 별도 계약을 세워야 한다.
- 하지 않는 것:
  - 새 MCTS 탐색, GPU 추론, Rapfi 대국
  - 학습, D4 재평가
- 바꾸지 않는 것:
  - D4의 계약, 목록, 승인, runner, 기록, 판독
  - `NOT_ESTABLISHED`, c5 유지, V9 보류
- 계획(이 문서와 `x1-plan.json`), 도구, 테스트를 먼저 커밋·push한다. 분석은 그다음, 고정된 코드로 실행한다.

## 1. 질문

D4 full에서 세 seed 모델은 c5와 eligible 국면의 약 3%에서 다른 수를 골랐다. 그 수의 continuation 점수가 낫다는 신호는 없었다. 이 부록은 그 불일치가 어떻게 생기는지를 기술 통계로만 살펴본다.

1. 불일치와 결과가 어느 ply와 차례에 몰리는가
2. 결정 관측(우세 또는 열세)이 소수 대국이나 반복된 국면에 집중되는가
3. 세 seed의 관측과 결과가 얼마나 겹치는가(공유 분기를 구분해서)
4. 모델 안 탐색 값 차이와 국면 예측 값 차이가 continuation 결과와 어떤 분포 관계를 보이는가

## 2. 입력과 출처 확인

### 2.1 입력 (전체 SHA-256)

| 키 | 경로 | SHA-256 |
|---|---|---|
| contract | `docs/h7r/r2/R2-D4-BASELINE-1.contract.json` | `74ab77e28bceb392c72380c9e03b2de349442df8eb9cebf1de53d8850936b180` |
| approval | `docs/h7r/r2/R2-D4-BASELINE-1/execution-full.json` | `6c71c1d83f65ab830bccd97ca38c59040622d949bac6677234ad4c69bcf56846` |
| eligible | `docs/h7r/r2/R2-D4-BASELINE-1/eligible.json` | `93aaa113f86723bd79c2d082563d3cbac9d0137dc147138a7e057e2288aa0760` |
| identity | `runs/h7r/R2-D4-BASELINE-1/full/identity.json` | `bd2fd76786f3a5a0feebd3750509c9e09d159dadfee6abc9144a1b93779c9fe3` |
| summary | `runs/h7r/R2-D4-BASELINE-1/full/summary.json` | `3d461b4f03492131d9d8f34d2e649c7db5b753ef7135dfb36c86873aedee2606` |
| replay | `runs/h7r/R2-D4-BASELINE-1/full/replay.jsonl` | `890ac30e2640afe46fc8bc0be047fa05f62f0b288de876c9365a226780c3915d` |
| branches | `runs/h7r/R2-D4-BASELINE-1/full/branches.jsonl` | `5c022e87879440ba066344ce0e700457fed82968c40d579a7aae618db3397b4f` |
| records_identity | `…/records/full/identity.json` | `bd2fd76786f3a5a0feebd3750509c9e09d159dadfee6abc9144a1b93779c9fe3` |
| records_summary | `…/records/full/summary.json` | `3d461b4f03492131d9d8f34d2e649c7db5b753ef7135dfb36c86873aedee2606` |
| check | `…/records/full/check.json` | `dbd21aa3ef56cea67d6d84bb690391ed53e6870b773edf63069f1f160d5b0e3b` |
| check_records | `…/records/full/check-records.json` | `dbd21aa3ef56cea67d6d84bb690391ed53e6870b773edf63069f1f160d5b0e3b` |
| sha256sums | `…/records/full/SHA256SUMS` | `40d7b9bfb9af389f34c59eeed9bb15fb2081aa12c3545423da5b10224aab83bd` |
| records_manifest | `…/records/full/records_manifest.json` | `5a7c0cc4065a170182f1a109365322aecd60eb1931f250df717d5db5e5c11698` |
| log_manifest | `…/records/full/log_manifest.json` | `1819d81b9084cadaf3dfc9da0c2a0c768437212ec1c684650313878cca919344` |
| r2_contract | `docs/h7r/r2/R2-DIAG-1.contract.json` | `b8045043dc9fe081ac11a1a4dc16ca323b0eac55e393d9f8a17390869b2bce2d` |
| epsilon | `docs/h7r/r2/R2-DIAG-1/r2_0b/epsilon.json` | `d46896e99200e3ff6f3d152f1c2028cde7c5a7712e481ad1e6e486ce0cebd4a2` |
| predictions | `runs/h7r/R2-DIAG-1/r2_0b/seed_predictions.pt` | `02ea051d76e38683110957b8029bbe9918f8d1f6c4ab374699df320ea9c5862b` |

(`…` = `docs/h7r/r2/R2-D4-BASELINE-1`)

- H-dev cache `runs/h7r/R1-HDEV-1/cache`는 D4 계약의 H-dev 검사로 확인한다. 검사 항목은 binding, 게임, 행, label, test tensor다.
- R2-0B의 epsilon 기록과 `seed_predictions.pt`는 **필수 입력**이다.
  - 입력이 하나라도 없거나 SHA가 다르면 분석을 멈춘다.
  - 있는 입력만으로 범위를 조용히 좁히지 않는다.

### 2.2 실행 commit과 X1 commit

- 실행 commit은 `265afe3484d01bf5c30e396a7ad05284b28c4ab8`이다.
- 기존 full 검증기는 HEAD가 실행 commit과 같아야 통과하므로, X1에서는 쓰지 않는다. 우회하지도, identity를 고치지도 않는다. 그 대신 다음을 따로 확인한다.
  1. identity의 `git`이 계획의 실행 commit이고 clean이다. 검증기의 나머지 identity 검사는 그대로 다시 한다: 계약, 승인, 목록, 모드, 장치, runtime, Rapfi, 다섯 모델 identity.
  2. 실행 commit이 현재 HEAD의 조상이다. 계약·승인·목록 파일은 실행 commit의 바이트(`git show`)와 같다.
  3. 재사용하는 소스 21개(재구성·통계 코드)는 실행 commit과 X1 checkout에서 바이트가 같다. SHA는 `x1-plan.json`의 `run_sources`에 있다.
- X1 도구의 commit(`x1_git`, clean 필수)과 도구 SHA(`x1_sources`)는 결과에 따로 기록한다. 현재 HEAD를 실행 HEAD로 내세우지 않는다.

### 2.3 기록 교차 확인

- `check.json`
  - 무결성 PASS이고 실패가 없다.
  - `check-records.json`과 바이트가 같다.
  - `artifacts`가 위의 raw SHA 네 개와 같고, `shas`가 계약·승인·목록 SHA와 같다.
- `SHA256SUMS`는 raw SHA 네 개와 같다.
- `records_manifest.json`은 `records/full`의 파일을 정확히 나열하고, 각 SHA가 맞다.
- 커밋된 identity·summary 사본은 raw 파일과 SHA가 같다.
- summary는 identity에 묶여 있고(`identity_sha256`), 두 journal의 SHA를 담는다.
- 예측의 출처
  - D4 계약의 `sources`가 epsilon 기록과 R2 계약을 SHA로 고정한다.
  - epsilon 기록은 predictions의 경로와 SHA를 담는다.
  - epsilon 기록의 seed 모델 identity와 R2 계약의 c5·c8은 D4 계약의 모델과 같다. H-dev 검사값도 같다.

### 2.4 재구성

- journal은 full 검증기의 엄격한 reader로 읽는다. runner의 복구형 reader(찢어진 끝줄을 잘라 고치는 resume reader)는 쓰지 않는다.
- 기존 함수를 그대로 쓴다.
  - `h7c_sensitivity.check_replay`: 커버리지, 합법성, c5 actor 재현
  - `h7r_d4.eligible_replays`, `observations_for`, `h7c_adjudicate.branch_table`, 고정 순서
  - `h7r_d4.validate_branch`: Game 재생과 INVALID 정책
  - `h7r_d4_full_check.recompute`: 같은 bootstrap으로 판독 재계산
- 관측, 분기, actor 재현, 분기 결과, 판독이 summary와 같아야 다음 단계로 간다(`differences`, 정수는 정확히, 실수는 1e-9).
- 관측 행에서 다시 센 우세·열세·동일·결정 수, 결정 게임 수도 summary와 같아야 한다.

## 3. 예측 행

- `seed_predictions.pt`의 확인 항목
  - 구조: `contract_sha256`, `game`, `ply`, `route`, `models`
  - 부모 R2 계약 SHA
  - 모델 집합: c5, c8, seed31001–31003
  - 텐서: 길이 15,985의 float64, 모두 유한값
  - game·ply 배열은 정수
- 행 순서
  - game·ply·route 배열은 검증된 H-dev cache 행과 순서까지 같아야 한다.
  - `game`은 cache의 **dense index**(행이 있는 게임만의 순번)다. 원래 game_index가 아니다. 제외된 게임 4판(39, 87, 110, 300) 때문에 둘이 다르다.
- 결합
  - dense index → cache의 원래 game ID → D4 eligible 행으로, (game ID, ply)에서 1:1 결합한다.
  - 다음은 모두 오류다: 빠진 행, 중복 행, 남는 행, 순서가 다른 행, 다른 계약·cache·모델 집합
- 쓰는 값은 `d4` 예측(D4 평균)이다.
- **한계**: 이 값은 그 root 국면 하나의 네트워크 값이다. 탐색이 쓴 모든 leaf 값의 기록이 아니다.

## 4. 분석 (기술 통계만)

공통
- 관측 단위는 D4의 불일치 관측(비교마다 (게임, ply) 하나)이다.
- 결과는 다음 넷으로 나눈다.
  - 우세 / 열세 / 동일: 미해결이 아닌 관측. 구조적 무승부는 정책대로 0.5다.
  - 미해결: 따로 센다. 이번 기록에는 없다.
- 세 seed 비교와 c8 참고는 따로 집계한다. 비교끼리 독립 표본처럼 합치지 않는다.
- 분모가 0이면 비율이나 평균은 null이다.

**A. ply와 차례**
- ply 구간 [5, 20), [20, 40), [40, 80), [80, ∞) × 흑 / 백
- 칸마다 보고하는 것
  - eligible 분모, 불일치 수와 비율
  - 우세 / 열세 / 동일 / 미해결, 결정 수
  - resolved 관측의 delta 합과 mean_delta
- 칸의 합은 비교 전체와 같아야 한다. 도구가 확인한다.

**B. 집중도**
- 결정 관측이 있는 게임 수
- 한 게임의 최댓값: 결정, 우세, 열세, |우세 − 열세|
- 상위 10판
  - 결정 수 내림차순으로 고르고, 같으면 원래 game_index 오름차순이다. 결정 관측이 없는 게임은 넣지 않는다.
  - 보고: 결정 수와 그 몫(전체 결정 수 대비, 분모 0이면 null), 우세, 열세, net(우세 − 열세), delta 합
  - 전체 net에 대한 비율은 만들지 않는다. 전체의 결정·우세·열세·net·delta 합은 따로 보고한다.
- 반복: 관측 ply 직전까지의 수순 전체가 같은 관측
  - 서로 다른 history 수, 반복된 history 수, 그 관측 수와 결정 수, 최대 반복
  - 후보 수까지 같아 분기 쌍 전체를 공유하는 관측 수

**C. seed 간 겹침** ((원래 game ID, ply) 기준)
- seed 쌍마다 보고하는 것
  - 각자의 불일치 수, 공통 불일치 국면 수
  - 그중 후보 수가 같은 수와 다른 수
- 공유 분기는 둘로 나눈다.
  - c5 분기: 공통 국면에서는 항상 공유된다.
  - 후보 분기: 후보 수가 같을 때만 공유된다.
- 결과 일치는 둘로 나눠 보고한다.
  - **후보 수가 같은 경우**: 두 분기를 모두 공유하므로 일치가 자동이다. 독립 재현이 아니다. 도구는 결과가 다르면 오류로 멈춘다.
  - **후보 수가 다른 경우**: c5 분기는 공유하고 후보 분기는 다르다. 일치 수와 결과 교차표를 보고한다.
- 세 seed 합집합
  - 국면 수, 1 / 2 / 3개 seed가 불일치한 국면 수
  - 세 seed가 같은 후보 수를 고른 국면 수
  - 서로 다른 후보 분기 수와 c5 분기 수

**D. 모델 안 탐색 값 차이**
- 정의: Δq = Q(후보 수) − Q(c5 수). 같은 모델의 root 목록 안에서 계산하며, 후보 모델과 c5 각각에 대해 따로 낸다.
- 유효 조건
  - 그 모델의 route가 tree이다.
  - 두 수가 모두 root 목록에 있다.
  - 두 수 모두 visits > 0이다.
- 제외는 따로 센다. 0으로 대신하지 않는다.
  - non_tree
  - missing: 후보 수만, c5 수만, 둘 다
  - visits_zero: 후보 수만, c5 수만, 둘 다
  - 방문이 0인 자식의 Q는 `PUCTNode.mean_value`의 기본값 0.0이므로 값이 아니다.
  - 유효 + 제외 = 관측 수를 도구가 확인한다.
- 관점과 정밀도
  - root 자식의 Q는 root에서 둘 차례인 쪽의 관점이다. 색에 따라 부호를 바꾸지 않는다.
  - Q는 소수 6자리로 기록돼 있으므로 Δq도 6자리로 반올림한다. 그보다 작은 차이는 해석하지 않는다.
- 보고
  - Δq 분포: 개수, 최소, 사분위, 최대, 평균, 양 / 0 / 음의 수
  - 방문 몫 차이: (visits_후보 − visits_c5) / 그 모델 root의 양수 visits 합. 분포만 보고한다.
  - 모델 간 Q 차이 Q_후보모델(후보 수) − Q_c5(c5 수): 보정이 섞인 값이므로 분포만 보고하고, 결과와 묶지 않는다.

## 5. 사분위

| 대상 | 경계를 계산하는 집합 | 칸에 넣는 것 |
|---|---|---|
| Δq | 비교 × 모델마다, 유효한 불일치 관측의 Δq | 그 관측의 결과 분포 |
| \|Δv\| = \|v_후보 − v_c5\| | 비교마다, eligible 행 전체(15,985) | 일치 / 불일치 수와 비율, 불일치 관측의 결과 분포 |

- 경계는 정렬값의 (n − 1)p 위치에서 선형 보간한다(p = 0.25, 0.5, 0.75).
- 칸: x ≤ q25, q25 < x ≤ q50, q50 < x ≤ q75, x > q75
- 같은 값을 억지로 나누거나 jitter를 넣지 않는다. 겹친 경계(`q25=q50` 등)와 빈 칸을 그대로 보고한다.
- 칸마다 분모, 값의 수, 결과 분포를 보인다.
- |Δv|는 일치 국면과 불일치 국면의 분포도 따로 보고한다.

## 6. 출력과 해석 한계

- 출력은 새 파일로만 쓴다(있으면 멈춤). 입력의 SHA는 분석 전후로 같아야 한다.
  - `docs/h7r/r2/R2-D4-BASELINE-1/records/x1/x1_summary.json`: 집계, 계획·도구·입력 SHA, 재구성 확인
  - `runs/h7r/R2-D4-BASELINE-1/x1/x1_rows.jsonl`: 관측 행. 커밋하지 않는다.
- raw journal, 수순, tensor는 `runs/`에 남는다.
- 해석 한계
  - 기술 통계만 낸다: 상관계수, p값, 구간별 신뢰구간, 새 label은 없다.
  - Q는 탐색의 추정치다. 정답이 아니고 원인의 증명도 아니다.
  - 결과를 보고 구간, 표본, 지표를 바꾸지 않는다. 특정 칸만 골라 개선을 주장하지 않는다.
  - 공유 분기 때문에 seed 결과는 서로 의존한다.
- 실패 기록
  - 실패해도 로그를 지우거나 고치지 않고 보존한다. 원인을 보고한다.
  - 표본이나 설정을 바꿔 다시 돌리지 않는다. 다시 돌릴지는 따로 정한다.
