# 안정적인 기력 향상을 위한 강화학습 자동화 구축 계획안 (H7-R)

작성: 2026-10-28. 기준 커밋: `892155d`(`feat/h7-c1`). 상태: **계획안. R1 도구 구현까지 승인됐고(2026-10-28), pilot · holdout 본 생성 · 학습 · 확인 평가 · champion 교체는 아직 승인 전이다.**

각 단계는 별도 계약으로 승인한 뒤 실행한다. 이 문서는 순서, 목적, 판단 기준, 필요한 도구, 비용을 정한다.

목표는 "학습을 계속 돌리는 자동화"가 아니다. 다음을 할 수 있는 자동화다.
- **반복할수록 고정 기준 대비 기력이 오르는 후보를 만든다.**
- **좋아졌다고 믿을 수 있는 후보만 채택한다.**
- 불확실하면 채택하지 않고, 정체하면 멈춘다.

## 0. 범위와 전제

- 바꾸지 않는 것
  - frozen 엔진 · runtime(TB-V3 Resolve / fast5), canonical, 웹(SER5)
  - 기존 cycle · NMASK 기록과 그 판정
- champion은 c5(H3 policy + c5 value `4e28a62a…`)다.
- 모든 후보는 `analysis.tb_v3_net`(후보 factory)로만 대국한다.
- S400(탐색 예산)은 운영에 넣지 않는다. 쓰는 곳은 둘뿐이다.
  - 평가의 양성 대조군(1.3절)
  - 나중의 학습 자료 보강 실험(R3의 선택지)
- V9 gate(E1-2 · E1-3)의 해석은 바꾸지 않는다.
- 순서: **R1 평가 보강 → R2 학습 병목 진단 → R3 개선 학습법 확보 → R4 제한된 자동 루프 → R5 장기 자동화.** 앞 단계의 종료 기준을 통과해야 다음 단계를 연다.

## 1. 방향 검토: 근거가 된 사실 (커밋된 기록에서 계산)

제안된 방향(C: 학습 정체 원인 실험, 평가 먼저)에 동의한다. 아래 사실은 그 순서가 맞다는 근거이고, 일부는 계획을 고친다.

### 1.1 Arena가 구별하는 정보가 적다

| 공식 Arena | 두 판 수순이 같은 쌍 | 0.5가 아닌 쌍 | pair score 표준편차 | 쓰인 opening(대칭 동치류) |
|---|---:|---:|---:|---:|
| c4 대 parent | 34 / 50 | 7 | 0.166 | 37 |
| c5 | 34 / 50 | 9 | 0.161 | 38 |
| c6 | 34 / 50 | 3 | 0.103 | 36 |
| c7 | 35 / 50 | 7 | 0.153 | 36 |
| c8 | 25 / 50 | 16 | 0.262 | 37 |
| mask 대 c5 | 31 / 50 | 4 | 0.090 | 35 |

- TB-V3의 tree에는 난수가 없다. 두 모델이 모든 국면에서 같은 수를 두면 색을 바꾼 두 판이 똑같아진다. 그 쌍은 opening과 색이 결과를 정한다.
- **정보는 두 모델이 갈라지는 대국에서만 나온다.** VALUE_ONLY 후보끼리는 50쌍 중 3–16쌍이다.
- opening 규칙(중앙 + 반경 2 안의 무작위 2수)이 만드는 국면은 552가지, 대칭을 묶으면 **75가지**뿐이다. 매 Arena가 그중 35–38가지를 다시 쓴다.
  - 대칭인 opening끼리도 같은 판이 되지는 않는다(158쌍 중 2쌍만 회전 복사). 버려지는 것은 아니다.
  - 그래도 시작 국면의 공간 자체가 좁다.
- 승급 규칙(점수 > 0.50, 향상 하한 ≥ −0.05)은 관대하다.
  - 정보 있는 쌍이 적으면 사실상 몇 쌍의 방향이 결정을 정한다.
  - mask의 PROMOTE는 4쌍 대 0쌍, c5의 승급은 6쌍 대 3쌍이었다.
  - 음의 쌍이 없으면 pair percentile bootstrap의 하한이 양수가 된다.

- 이 표의 "두 판 수순이 같은 쌍"과 "0.5가 아닌 쌍"은 서로 다른 개념이다. 2.4절에서 (a) 행동이 갈라진 쌍, (b) 점수가 갈린 쌍으로 따로 정의한다.

→ **평가 보강의 핵심은 쌍 수만이 아니다.** 정보 있는 쌍의 비율(갈라지는 대국)을 늘리고, 그 비율을 먼저 재서 규모를 정해야 한다. 제안의 "쌍 수만 늘리는 것으로 충분하지 않다"를 수치로 뒷받침한다.

### 1.2 "c5 이후 정체"는 지표의 해상도부터 의심해야 한다

- cycle val 집합은 c6 / c7 / c8에서 **1,786 / 1,386 / 1,853행, 44 / 39 / 51게임**이다(train의 약 5%). 선택 지표는 이 40–50게임에서 나온다.
- c6–c8과 mask의 parent 대비 변화(+0.004%, −0.09%, +0.05%, +0.04%)는 이런 표본의 잡음보다 훨씬 작을 가능성이 크다. c1–c5의 개선(−0.2% ~ −2.4%)도 유의성은 확인된 적이 없다.
- 그래서 앞 검토의 "c5 이후 정체"는 **"선택 지표의 해상도 안에서 개선이 보이지 않음"**으로 고쳐 읽는다(`docs/h7-c1.md` 정오표).
- **R2의 첫 일은 "얼마의 변화부터 실제로 구별되는가"를 재는 것이다.** 게임 단위 구간과 학습 seed 사이의 변동을 잰다.

### 1.3 value head는 아주 좁고, 넓히는 것은 작은 변경이 아니다

- 지금 head: trunk 64채널을 1×1 conv로 **1채널**로 줄인 뒤 225 → 64 → 1이다(약 14.6k 매개변수). 정보 병목 후보로 충분히 의심할 만하다.
- `model/network.py` · `config.py`는 frozen이다. 다만 head 폭(`value_channels`, `value_hidden`)은 이미 설정값이라, 넓은 head는 오프라인으로 학습하고 평가할 수 있다.
- 그러나 대국에 쓰려면 후보 factory `analysis.tb_v3_net`이 `ARCHITECTURE`(1채널 · 64)를 강제하는 부분을 버전을 올려 바꿔야 한다. 추론 시간도 검사해야 한다.
- trunk 학습은 factory의 "frozen H3 trunk" 조건과 policy(같은 trunk)를 함께 건드린다.
- → 제안의 우선순위 "용량 문제면 head 확대"는 유지한다. 다만 **이 저장소에서 head 확대는 factory 계약 단계를 동반하는 중간 크기의 변경**이다. 진단(R2)은 오프라인으로 싸게 하고, 대국 투입은 R3의 별도 단계로 둔다.

### 1.4 싸게 쓸 수 있는 자원이 있다

- **c6 · c7 · c8의 self-play 3,000판은 모두 같은 champion c5가 뒀다**(value `4e28a62a…`). 지금까지 cycle마다 1,000판씩 따로 썼다.
  - 합쳐 쓰면 새 self-play 없이 자료량 효과를 잴 수 있다(R2-D2).
  - c8의 1,000판 중 800판은 NMASK 확인 평가에서 결과를 본 자료다. 학습에는 써도 되지만 평가에는 다시 쓰지 않는다.
- 비용 실측
  - self-play 1,000판: 3,862 / 4,213 / 6,826초(6 worker)
  - 공식 Arena 50쌍: 920.9초
  - Rapfi 판정: 분기당 평균 1.46초(708분기 4.3분, 4 worker)
  - value 학습: 210 step(수 분)
- **양성 대조군이 있다.** 같은 c5 모델에서 S400은 S50보다 +7.7%p(하한 +2.5%p)였다.
  - 실제 차이를 알고 있는 쌍이다. 새 평가 체계가 이 차이를 안정적으로 잡는지로 평가의 해상도를 검증할 수 있다.
  - 이 용도에서는 S400을 운영에 넣지 않는다.

### 1.5 확인 표본은 학습 자료와 분리돼야 한다

- NMASK 확인 평가의 800판은 c8 · mask 학습 자료가 나온 self-play였다(train 약 95%).
- 앞으로의 확인 국면은 **reserved seed로 둔 새 self-play**(holdout)에서 뽑는다.
  - label 도구(`h7c_labels` → `h6_labels.check_reserved_seed`)는 `RESERVED_SEEDS`의 대국을 학습 자료로 쓰지 못하게 이미 막는다. holdout seed를 예약하면 학습 혼입이 코드로 차단된다.
  - 다만 self-play 생성기(`h7c_selfplay.check_seeds`)도 지금은 reserved seed를 거부한다(학습 seed 8801–8999와 측정 seed 8800만 허용). 그래서 "예약된 holdout seed만 허용하고 출력에 용도를 표시하는 holdout 모드"를 새로 둔다(2.6절).

## 2. R1 — 좋아진 모델을 구별하는 평가 체계 (계약 E3)

### 2.1 평가 자원과 분리

| 자원 | 내용 | 쓰는 곳 | 쓰면 안 되는 곳 |
|---|---|---|---|
| T 학습 자료 | champion self-play(학습 seed) | gradient | 평가 |
| S 선택용 검증 | T의 val split + H-dev | checkpoint 선택, 진단 | 최종 판정 |
| H-dev | reserved seed self-play(진단 · 개발용) | R2 / R3 진단, 반복 사용 가능 | 승급 판정 |
| H-conf | reserved seed self-play(확인용). 계약마다 새로 만들고 한 번만 연다 | 선택 품질의 확인 판정 | 후보 선택 · 수정 |
| A-sel | 선택용 Arena(작은 규모, 반복 가능) | 후보 여럿 중 고르기 | 승급 판정 |
| A-conf | 확인 Arena(규모는 2.4절, seed는 매번 새로) | champion 대비 승급 판정 | 후보 선택 |
| Anchor | 고정 기준(원본 H3 / H6 v3). 필요하면 X2 Rapfi L_mid 기록 | 누적 향상 · 퇴보 추적 | — |
| Gate | tactical v2, 금수 오류 0, **수당 시간** | 기능 퇴보 차단 | — |

- 수당 시간 gate는 새로 둔다. Arena 기록에는 이미 수마다 `seconds`가 있으므로, champion 대비 평균 · p95 증가 상한을 계약에서 정한다. head를 넓히면 특히 필요하다.

### 2.2 시작 국면(opening suite) — 2026-10-28 보완

- 생성기는 `hybrid.h7r_openings`다. 기존 `make_opening`(cycle Arena · self-play 규칙)은 그대로 둔다.
- suite 정의. 모든 수는 `renju.Game`으로 합법성을 검사한다. 흑 금수는 규칙이 거부한다.

  | suite | 수순 길이(중앙 포함) | 수의 색 | 좌표 범위(중앙 기준 Chebyshev 반경) | 다음 착수 |
  |---|---:|---|---|---|
  | O0 | 3 | 흑 중앙, 백, 흑 | 2, 2 | 백 |
  | O1 | 5 | 흑 중앙, 백, 흑, 백, 흑 | 2, 2, 3, 3 | 백 |
  | O2 | 3 | 흑 중앙, 백(바로 옆 / 대각선 옆), 흑 | 표준 26 주형(셋째 수는 반경 2), 8방향 | 백 |

- O2는 고정된 시작 국면의 목록이다. 운영 대국의 opening 규칙은 바꾸지 않는다.
  - 26 주형 = 백이 바로 옆(13) / 대각선 옆(13)이고, 처음 두 돌을 지키는 대칭으로 묶은 대표다.
  - 8방향으로 돌린 뒤 정확히 같은 수순을 빼면 184개(대칭 동치류 26)가 된다.
- 거부 규칙
  - 첫 수가 중앙이 아님, 불법 수, opening 안에서 대국이 끝남
  - opening 직후 `exact_leaf`가 정하는 국면(둘 차례가 바로 이기거나, 상대가 두 곳 이상에서 오목을 만든다)
- ID 세 가지를 기록한다.
  - `history_id`: 수순 순서를 포함한다. 엔진은 마지막 수를 보므로 순서가 다르면 다른 opening이다. **중복의 단위다.**
  - `position_id`: 돌의 배치와 둘 차례
  - `class_id`: 8대칭 동치류. 기록용이며, 대칭이라는 이유로 같은 표본으로 처리하지 않는다(대칭 opening의 대국이 회전 복사인 경우는 158쌍 중 2쌍뿐이었다).
- 중복과 겹침
  - suite 파일 안, 한 arm 안에서는 같은 `history_id`가 두 번 나오지 않는다(생성 · `pick` · 기록 검사).
  - 이전 평가(suite 파일, Arena 기록의 opening)와의 겹침을 history · position · class별로 센다(`suite --compare-with`).
  - 확인 평가의 겹침 처리(제외할지, 독립 표본으로 세지 않을지)는 해당 확인 계약에서 미리 정한다.
- suite 선택 기준(결과 전에 고정, 후보의 승패로 고르지 않는다)
  - 1순위: 정보 쌍 비율(2.4절 정의 a / b)의 하한
  - 제외 조건: 흑백 점수 차가 지나치게 큰 suite
  - 같은 수준이면 단순한 쪽(O0 → O2 → O1)
  - 수치 문턱은 pilot 계약(`R1-PILOT-1`)에 적는다.

### 2.3 보정 pilot (한 번, 기존 모델로)

- 목적: 측정이다. 승급 판정이 아니다.
  - suite별 두 정보 수(2.4절), pair score 표준편차
  - 양성 대조군의 효과와 그 구간
- arm(작은 차이 3비교는 suite마다 같은 opening으로 묶음; 9.2절 수정 초안의 S400 대조는 별도 seed·pairing으로 더 크게 표집)
  - 작은 차이의 쌍: c8 : c5, mask : c5, c5 : 원본
  - 양성 대조군: c5 S400 : c5 S50
- 양성 대조군의 효과가 기존 +7.7%p(기존 opening 분포, 150쌍)와 같다고 가정하지 않는다.
  - pilot에서 한 번 잰 값과 구간을 그대로 기록한다.
  - 검출하지 못하면 원인을 나눠 보고한다: 구간이 넓음 / suite에서 효과가 달라짐 / 도구 문제
  - **나올 때까지 다시 돌리지 않는다.** 다시 하려면 새 계약으로 한다.
- 계약에 미리 쓰는 것: pilot의 승패로 champion · 기존 판정 · mask 채택을 정하지 않는다.
- 비용
  - 일반 arm 100쌍은 c8 공식 Arena 실측(50쌍 920.9초, 6 worker)으로 약 30분이다.
  - S400 arm은 한쪽이 수당 약 4.9배라 더 비싸다. 같은 기준을 쓰지 않는다.
  - 데스크톱 smoke(9절)에서 arm 종류별로 실측해 다시 추정한다.

### 2.4 정보 쌍의 정의와 규모

- 두 가지를 따로 센다.
  - (a) `move_split`: 색을 바꾼 두 판의 수순이 다른 쌍(행동이 갈라짐)
  - (b) `score_split`: pair score가 0.5가 아닌 쌍(점수가 갈림)
  - 수순이 달라도 점수는 0.5일 수 있으므로 (b) ⊂ (a)는 보장되지 않는다. 둘 다 보고한다.
- **점수와 모든 효과 계산에는 유효한 쌍 전부를 넣는다.** 0.5인 쌍을 빼고 계산하지 않는다(편향).
- 최소 정보 조건 m: 대상((a) 또는 (b))과 값은 pilot 뒤 확인 계약에서 정한다. 모자라면 승급하지 않는다.
- 규모 산정: `n = ((z_{1−α} + z_{1−β}) · σ_upper / δ)²`
  - δ = +3%p, 검출력 0.8은 **규모 산정의 목표**다. "실제 효과가 3%p 이상"이라는 보증이 아니다.
  - σ_upper는 pilot pair score 표준편차의 bootstrap 상한이다.
  - 정보 쌍 비율의 Wilson 하한으로 "n쌍에서 m개 이상"의 확률을 같이 낸다(`h7r_eval.py size`).
  - 계산은 제안일 뿐 본 평가를 시작하지 않는다.
  - 예시(σ = 0.167, 단측 α = 0.025): δ = 3%p → 약 240쌍. pilot 값으로 다시 계산한다.

### 2.5 승급 규칙 v3 (앞으로의 확인 계약에만, 기존 판정은 그대로)

- 통계 설정(계약에 명시)
  - 단측 α = 0.025
  - pair percentile bootstrap: 재표집 단위는 opening 쌍, 20,000회, 고정 seed
  - 오류 예산: 최대 cycle K = 3, cycle당 α = 0.025, 거짓 승급 기대 횟수 상한 K · α = 0.075
  - 후보를 여럿 보내면 α를 나눈다.
- 승급(모두 만족, 하나라도 아니면 champion 유지)
  1. 후보 대 champion(같은 opening의 색 교환 쌍): 향상의 단측 하한 > 0, 그리고 정보 쌍 ≥ m
  2. **anchor 비열등은 같은 조건의 차이로 본다.**
     - 같은 opening에서 후보와 champion이 각각 원본을 상대한다.
     - 차이 d_i = (후보 대 원본 pair score) − (champion 대 원본 pair score)
     - d_i를 opening 단위로 bootstrap해 단측 하한 ≥ −2%p(제안값)면 비열등이다.
     - 과거 champion 점추정과 직접 비교하지 않는다.
  3. 선택 품질 비열등
     - 허용 열세 폭과 필요한 관측 수는 계약에서 미리 정한다.
     - Δ(후보 − 기준)의 평균을 원본 게임 단위로 bootstrap한다(같은 게임의 관측은 함께 움직인다).
     - 단측 하한 ≥ −margin이면 비열등이다.
     - **INCONCLUSIVE · 표본 부족은 통과가 아니다** → champion 유지
     - INVALID 분기는 계약의 `selection.invalid_policy`(9.7)를 따른다. 정책이 없는 계약은 INVALID가 하나라도 있으면 INCONCLUSIVE다.
  4. gate: tactical v2 PASS, 금수 오류 0, 수당 시간(평균 +10%, p95 +20%, 잠정)
- 설정 미확정(`unset`), 기록 누락, 표본 부족, gate 미통과는 모두 승급하지 않는 상태다.
- 확인 평가 결과를 보고 후보를 고친 뒤 같은 확인 자료로 다시 판정하지 않는다.
  - 새 확인 계약 · 새 seed · 새 H-conf가 필요하다.
  - seed는 계약끼리 공유할 수 없다(도구가 거부한다). 판정 파일은 덮어쓰지 않는다.

### 2.6 R1 도구 (구현, `R1` 커밋)

| 도구 | 하는 일 |
|---|---|
| `src/hybrid/h7r_openings.py` | O0 / O1 / O2 생성 · 검사, ID, 대칭, 중복 없는 추출, 이전 평가와의 겹침 |
| `src/hybrid/h7r_eval.py` | 계약 검사(seed 예약 · 계약 간 재사용 금지 · suite SHA · 모델 SHA), 쌍 표(잘림 · 누락 · 중복 거부), pair bootstrap, anchor 차이, 선택 품질 비열등, 규모 제안, 승급 규칙 v3 |
| `scripts/h7r_eval.py` | `suite`(제외 · disjoint 출처, 9.11), `play`(계약의 한 arm, 재개, 예산 arm), `summarize`, `size`, `anchor-size`(9.11), `timing`, `selection-source`, `selection`, `tactical`(9.7), `judge` |
| `scripts/h7r_holdout.py` | `labels`(holdout 대국 → 평가 전용 라벨 · cache), `eval`(경로 × 종류 WMSE, 짝 차이, 게임 단위 bootstrap, seed 간 분산) |
| `h7c_selfplay.py --holdout-contract` | 계약에 등록된 예약 seed만 허용, identity에 `purpose: holdout` · 계약 ID · SHA |
| 학습 진입 거부 | `h7c_labels.check_games`(holdout 표시 → 거부, 표시를 지워도 예약 seed로 거부), `h7c_train.rows_source`(`purpose` 있는 행 거부), `h7c_train_value.check_cycle_cache`(예약 seed 게임 거부) |

- 대국은 cycle Arena의 `h7c_arena.play_game`을 그대로 쓴다. 각 쪽은 `h7c_budget.make_budget_agent`(1 = S50, 8 = S400)다. frozen 엔진 · runtime · factory는 바꾸지 않았다.
- 시간 gate(`timing`)
  - 한 프로세스에서 같은 국면에 두 쪽을 번갈아 두고, 국면마다 새 agent를 만들며, warm-up을 버린다.
  - 기록하는 것: 플랫폼 · CPU 수 · torch 스레드 · 장치, 평균 · p95 비율, 비율의 bootstrap 구간

### 2.7 R1 종료 기준

- pilot 보고서가 있다: suite별 (a) · (b) 비율과 구간, σ, 양성 대조군의 효과와 구간(검출 여부와 원인 구분).
- 확인 계약(E3)이 커밋됐다: suite, 규모, m의 대상과 값, anchor · 선택 품질 margin, gate.
- 도구 구현만으로 R1 완료나 기력 향상을 선언하지 않는다.
- (선택) E3의 첫 적용: mask : c5와 c5 : 원본을 새 확인 계약으로 판정한다. pilot과 별도의 seed · 계약이다.

## 3. R2 — 학습 병목 진단 (c5에서 출발, 오프라인, 작게)

모든 진단은 value만 다룬다. 대국은 하지 않는다(마지막 선택 품질 확인만 Rapfi). champion도 바꾸지 않는다. 각 조건은 학습 seed 3개로 돌린다.

### 3.1 R2-0 지표 해상도 (가장 먼저)

- H-dev: c5 self-play 300–500판(reserved seed, 약 0.5–1시간)
- 측정
  - 두 모델의 국면별 쌍 차이 WMSE에 대한 게임 단위 bootstrap 구간
  - 같은 recipe를 seed만 바꿔 3번 학습했을 때의 변동
  - 이 둘에서 **최소 구별 가능 변화 ε**를 정한다. 이후 모든 "개선"은 ε를 넘어야 인정한다.
- 기존 후보 재측정: c5 / c6 / c7 / c8 / mask의 H-dev WMSE를 같은 국면에서 잰다. "정체"가 실제인지 해상도 문제인지 여기서 갈린다.

### 3.2 진단 실험 (관측 → 의심할 원인)

| 실험 | 내용 | 관측이 이렇다면 | 의심 |
|---|---|---|---|
| D1 맞추기 | 작은 train 부분집합(1k / 5k / 전체 cycle 행)을 여러 epoch 학습. train WMSE를 기록한다(지금 trainer는 val만 기록하므로 추가 필요). 지금 head와 넓은 head(`value_channels` 4 / 8 / 16, `value_hidden` 64 / 128 / 256)를 비교 | 지금 head는 작은 자료도 못 맞추고 넓은 head는 맞춘다 | 용량 · 표현 |
| | | 둘 다 못 맞춘다 | 최적화 설정(lr, 정규화, BN 고정) 또는 target 모순 |
| D2 자료량 · 다양성 | c8만 / c6+c7+c8(3,000판, 모두 c5 actor) / 그 25 · 50%로 학습. H-dev 곡선을 본다 | 자료가 늘수록 H-dev가 ε 이상 계속 좋아진다 | 자료 부족 → 합치기 · self-play 증량이 싼 해법 |
| | | train은 좋아지는데 H-dev는 그대로다 | 과적합 · 다양성 부족 |
| D3 정답 잡음 | H-dev 오차를 label 종류(proof / outcome) × ply × 경로로 나눈다. outcome weight 0.1 / 0 / proof만의 변형 | 오차가 outcome 행에 몰리고, proof 행(정확한 정답)은 이미 작다 | 정답 잡음 → target 품질(깊은 탐색 값 등) |
| D4 예측 → 선택 | D1–D3에서 H-dev가 ε 이상 좋아진 변형만. H-dev 국면에서 수 선택 변화율(sensitivity)과 그 불일치의 Rapfi 판정(개발용) | 예측은 좋아지는데 선택이 그대로이거나 나쁘다 | 학습 목표와 탐색의 사용 방식 불일치(망을 묻는 국면 · 값의 범위) |
| D5 replay 비율 | 50 : 50의 H6 replay 비율 변형(나중) | replay를 줄이면 cycle 지표는 좋아지고 guard는 유지된다 | replay가 학습을 묶음 |

- 각 실험의 판단 수치(ε 배수, seed 일치 수 등)는 R2 계약에 실험 전에 적는다.
- 넓은 head는 이 단계에서는 오프라인 평가만 한다. 엔진 대국은 R3의 factory 단계 뒤다.
- 비용 추정
  - 학습 1회 수 분 × (조건 15–30 × seed 3) ≈ GPU 수 시간
  - H-dev self-play 0.5–1시간
  - D4 sensitivity 200판 × 2–3 세트 ≈ 1–2시간, Rapfi 수십 분

### 3.3 R2 종료 기준

- ε가 정해지고, 기존 후보의 H-dev 재측정이 있다.
- D1–D4 결과로 병목이 하나 이상의 범주로 정리된다. "어느 것도 아님"도 결과로 기록한다.
- R3에서 바꿀 변수 하나가 근거와 함께 제안된다.

## 4. R3 — 개선되는 학습법 확보 (변수 하나씩)

- 진단 → 변수의 기본 대응(제안의 우선순위를 따른다)
  1. 용량 → value head 확대. 아래 factory 단계를 동반한다.
  2. 자료 부족 · 일반화 → 자료 합치기, self-play 증량, 다양성(opening 샘플링 범위)
  3. 정답 잡음 → target 품질. 예: 망을 묻는 국면에 더 깊은 탐색 값을 target으로 쓴다. 이때 S400을 teacher로 쓴다.
     - c2의 S200 policy teacher 실패를 고려해, value target으로 작은 범위에서 먼저 시험한다.
  4. 예측 → 선택 불일치 → 탐색이 실제로 망을 묻는 국면 · 값에 맞춘 목표(NMASK가 그 한 형태였고 효과는 미입증)
  5. 그 뒤에도 표현 한계 → trunk 일부 학습
     - policy가 같은 trunk를 쓰므로 policy 품질 gate(must_block, RenjuNet top-1)를 같이 둔다.
     - BN 통계도 다뤄야 한다. 가장 큰 계약이다.
- **후보 factory v2 단계(head를 넓힐 때만)**
  - `tb_v3_net`은 그대로 둔다. 허용 architecture 목록을 버전으로 가진 새 factory를 만든다.
  - frozen 엔진 · runtime · loader(`h6_value_leaf`)는 바꾸지 않는다. loader는 metadata의 `model_config`로 읽는다.
  - 두 build의 대국 재생 동일성 테스트, 추론 시간 gate를 둔다.
  - canonical이 아니므로 CPU 인증은 canonical 승격 때 따로 한다.
- 성공 기준: 손실 하락이 아니라 **세 고리가 이어질 것**
  1. H-dev · H-conf 예측 개선 > ε(seed 3개 중 3개 같은 방향)
  2. H-conf 불일치 국면의 선택 품질이 WORSE가 아니고, 사전에 정한 주 판독을 통과
  3. 승급 규칙 v3의 확인 Arena · anchor · gate 통과
- 세 고리 중 하나라도 끊기면 그 recipe는 루프에 넣지 않는다. 다음 변수로 간다(R2 결과 순서).

## 5. R4 — 제한된 자동 루프

- 진입 조건: R3의 recipe 하나가 세 고리를 통과했다.
- 프로그램 파일(B2 program처럼 미리 등록하고 SHA로 고정)
  - recipe, 최대 cycle K(제안 3), seed 규칙, 규모, 중단 규칙

```
champion으로 self-play(학습 seed)
  → label · 자료 검사 → 학습(고정 recipe) → S로 checkpoint 선택
  → H-conf(이 cycle의 새 holdout) 예측 · 선택 품질
  → A-conf(새 seed, 정한 규모) 대 champion, anchor 대 원본, gate
      → 승급 규칙 v3 통과: champion 갱신
      → 불확실 · 실패: champion 유지
  → 중단 규칙 검사
```

- 운영 원칙
  - 승급은 의무가 아니다. INCONCLUSIVE는 유지다.
  - 선택에 쓴 평가(S, A-sel)와 승급 판정 평가(H-conf, A-conf)를 섞지 않는다. 확인 seed는 cycle마다 새로 하고, 결과를 본 표본은 확인용으로 다시 쓰지 않는다.
  - anchor 추적은 cycle마다 한다. K cycle이 끝나면 champion 대 원본을 큰 규모로 한 번 더 재서 누적 향상의 구간을 보고한다.
- 자동 중단
  - 연속 2 cycle 유지(정체)
  - anchor 퇴보: champion 대 원본 상한 < 0, 또는 비열등 실패
  - gate 연속 실패
  - 무결성 · 출처 오류
  - 예산 초과
  - **중단은 실패가 아니다.** 멈춘 이유를 기록하고 사용자가 다음을 정한다.
- seed: holdout self-play, A-conf, anchor용 블록을 cycle 수만큼 새로 예약한다. 학습 seed는 예약하지 않는다(학습 자료가 되어야 하므로).
- 비용(1 cycle)
  - self-play 1–2시간, label · 학습 수십 분
  - holdout 300판 0.5시간, sensitivity · Rapfi 1–2시간
  - A-conf 240쌍 약 75분, anchor 100–240쌍 30–75분, tactical 수십 분
  - 합계 약 5–7시간(6 worker, 추정)이다. K = 3이면 하루 안팎이다.

## 6. R5 — 장기 자동화

- 진입 조건(모두)
  - R4에서 확인된 승급이 2회 이상
  - K cycle 뒤 champion 대 원본 누적 향상의 하한 > 0
  - 중단 규칙이 실제로 작동한 기록(정체 · 실패 시 유지)
- 더할 것
  - 최대 cycle 확대와 주기적인 대규모 anchor 재평가(예: 5 cycle마다)
  - 외부 사다리(X2 Rapfi 수준 2개 이상)
  - 자원 예산과 일정, 실패 복구(재개 기록은 지금 도구들의 방식)
  - 결과 요약의 자동 커밋은 사람 확인 뒤에만 한다.
- 배포(canonical · 웹 SER5 교체)는 여기서도 자동이 아니다. 별도 승격 단계(동일성 · CPU 인증 · 웹 smoke)로 한다.
- S400의 운영 투입은 탐색 예산 엔진 버전 단계로 따로 한다. 학습 루프와 섞지 않는다.

## 7. 결정 사항 (2026-10-28 사용자 피드백 반영)

| 항목 | 결정 |
|---|---|
| δ = +3%p, 검출력 0.8 | 규모 산정 목표로 채택. 효과 크기의 보증과 구분 |
| O0 · O1 · O2 pilot | 채택. 생성 규칙 · 선택 기준을 결과 전에 고정 |
| H-dev 400판 | 개발용 초기 규모 |
| H-conf 300판 | 예산상 초기 제안. 필요한 결정 관측 수가 나오는지 계약 전에 확인 |
| 평균 +10%, p95 +20% | 잠정 gate. 같은 CPU · 스레드 · 국면 · warm-up 조건으로 비교 |
| 최대 3 cycle | 초기 자동화 범위 |
| 다중 검정 | cycle당 α와 전체 오류 예산을 계약에 따로 명시 |
| 확인 후 수정 | 같은 확인 자료로 다시 판정하지 않는다 |

## 8. 하지 않을 것

- 기존 판정의 재해석, B2 c9 재개, NMASK 변형 반복
- 결과를 본 표본(E2 200판, NMASK 800판)으로 확인 판정하기
- S400의 운영 투입, V9 본개발
- 평가 체계(R1) 없이 학습 변형을 Arena로 고르기

## 9. R1 구현 검증 · 비용 · 본 pilot 전 확정 항목 (2026-10-28)

### 9.1 실제로 한 검증

- 단위 · 통합 테스트(mock: 합성 기록 · stand-in 망 · 가짜 대국)
  - `test_h7r_openings`: suite 규칙 · 길이 · 색 · 반경 · 합법성, 26 주형(직 13 / 사 13) · 8방향 184개, 거부 규칙, ID · 대칭, 정확한 중복 거부, `pick` · 겹침
  - `test_h7r_eval`
    - 계약: 예약 seed, 계약 간 seed 재사용 거부, pairing, suite · 모델 SHA, unset 항목
    - 쌍 표: 잘림 · 누락 · 중복 · arm 밖 opening 거부, 상대 불법 수 → arm 무효
    - 모든 유효 쌍을 넣은 점수, (a) · (b)를 따로 센 수
    - anchor 차이(같은 opening 강제), 선택 품질 비열등(INCONCLUSIVE · 표본 부족 ≠ 통과, 게임 단위 bootstrap)
    - 규모 제안, 승급 규칙 v3의 PROMOTE 조건과 실패 사유 전부
  - `test_h7r_holdout`
    - holdout seed 규칙, 계약 대조(seed · cycle · 판 수 · actor · fixed · committed)
    - 학습 label 거부: 표시 있음 / 학습 seed로 바꿈 / 표시를 지우고 예약 seed 유지 — 모두 거부
    - 학습 행 출처 · cycle cache의 예약 seed 거부
    - 평가 라벨(`test` split, `purpose`) → 학습 행 출처가 거부
    - holdout 평가기(경로 · 종류 · D4 · 짝 차이 · seed 분산)
  - `test_h7r_cli`
    - 계약 3 arm 전체 흐름(재개, 같은 opening, 기록 변조 거부, 규모, 선택 품질, 판정, 기록 누락 → 유지)
    - 초안 · unset · 미커밋 계약 실행 거부, 다른 설정의 기록 혼합 거부, 출력 덮어쓰기 거부
- 실제 엔진(이 컨테이너, TB-V3 / fast5, stand-in 망, CPU)
  - holdout self-play 1판이 `purpose: holdout`으로 기록되고 학습 label이 거부한다.
  - 예산 연결(S400 = factor 8, S50 = factor 1)
  - 시간 gate 기록 형식
  - smoke: `play`로 S50 arm 2쌍(4판), S400 arm 1쌍(2판), 2 worker
- 기존 경로 회귀: `test_h7c_*` 전체, `test_h6_fresh`, `test_external_engine`, `test_h6_value`, `test_tb_v3_net`
- frozen checker: 이 컨테이너에서 늘 나오던 checkpoint 부재 2건만 FAIL, 나머지 PASS

**하지 않은 것**
- 실제 모델(c5 / c8 / mask / 원본)과 GPU로의 실행
- 전체 pilot, H-dev · H-conf 본 생성, 학습, 확인 Arena 본 실행, champion 교체
- 이 컨테이너에는 `runs/`의 checkpoint가 없다. 아래 데스크톱 smoke가 실제 모델의 첫 실행이다.

### 9.2 비용 갱신 초안 — 데스크톱 근거와 계획용 가정 (2026-10-29)

**권장 검토안:** 작은 차이 arm 60쌍, S400 양성 대조 arm 160쌍, workers=6. suite 3종 × 작은 차이 3비교 ×60쌍 + suite 3종 × S400 160쌍 = **1,020쌍 / 2,040판**이다. 계약은 draft이며, 여기의 비용·검출력은 실행 승인이나 실제 소요시간의 보장이 아니다.

작은 차이 3비교는 suite별 seed8781–8783과 `O*_small`로 묶고, S400은 seed8784–8786과 `O*_positive_control`로 분리하는 제안이다. `Random.sample`은 쌍 수를 바꾸면 같은 seed라도 포함 관계를 보장하지 않는다. 서로 다른 군의 opening 겹침은 허용하되 기록하며 독립 비교로 간주하지 않는다. 실제 겹침은 O0 30 · O1 30 · O2 55개(작은 차이 60쌍 중)이고 draft JSON의 `planning.opening_overlap_small_vs_control`에 적었다. O2의 160쌍은 184개 중 중복 없이 뽑을 수 있다(suite의 87%).

#### 비용 근거와 측정 구간

| 근거 | 관측·계산 | 이번 초안에서의 사용 |
|---|---|---|
| `docs/h7c/c8/arena/official.json` | 50쌍 wall time 920.9초 → **18.418초/쌍** | 과거 데스크톱 6 worker 실행을 기준으로 하는 S50 비용 대용값. JSON 자체에는 workers·하드웨어 환경 필드가 없으므로 기존 계획의 실행 조건을 따른다. |
| 데스크톱 smoke, 코드 `112fd97` | workers=2, S50 2쌍 wall time 78.4초 → **39.2초/쌍** | 2 worker 시나리오의 S50 기준. 과거 6 worker 자료와 같은 opening·환경으로 맞춘 비교는 아니다. |
| `docs/h7c/sb_summary.json`, S400 50쌍 | 착수시간 합: tested 7708.7초 / opponent 1521.6초 → 5.0662배 | 양쪽이 S50인 대국의 비용을 opponent 시간의 두 배로 근사하면 `(1+5.0662)/2 = 3.0331배` |
| `docs/h7c/e2/sb_c5_summary.json`, c5 S400 50쌍 | tested 9236.5초 / opponent 1869.3초 → 4.9410배 | 같은 근사로 **2.9705배**. 이번 c5 대조와 모델이 일치하는 보조 근거 |

따라서 S400 arm 비용은 **계획용으로 S50 arm의 3.0배**를 쓴다. 이는 착수시간을 합성한 가정이며, 동일 opening을 S50/S50과 S400/S50으로 각각 끝까지 두어 얻은 쌍당 wall time의 직접 비율이 아니다. 전술 경로·수의 수·대국 길이가 달라지면 이 가정도 달라질 수 있다.

이번 S400 smoke는 1쌍·29수 대국 두 판뿐이다. 이 wall time 31.8초를 전체 pilot의 S400 비용에 직접 적용했던 기존 분석 초안의 외삽은 예산 기준에서 제외한다. 과거 컨테이너의 stand-in 망에서 얻은 쌍당 약2.5배도 참고 기록으로만 남기고 데스크톱 비용 계수는 3.0으로 갱신한다.

#### 규모별 계획용 비용

`n_small`은 작은 차이 arm 하나의 쌍 수, `n_control`은 S400 arm 하나의 쌍 수다.

- 6 worker 시나리오: `(9×n_small + 3×n_control×3.0) ×18.418 /3600` 시간
- 2 worker 시나리오: 같은 식에서 18.418 대신 39.2초를 사용한다.
- 6 worker 값을 2 worker 값의 1/3로 만들지 않는다. 별도의 관측 기준을 사용한다.

| 작은 차이 / S400 (arm당) | 전체 쌍 / 판 | 6 worker 예측 | 2 worker 예측 | S400의 suite당 계획 검출력 |
|---|---:|---:|---:|---:|
| 60 / 60 | 720 / 1,440 | 약5.5시간 | 약11.8시간 | 0.399 |
| 100 / 100 | 1,200 / 2,400 | 약9.2시간 | 약19.6시간 | 0.595 |
| 60 / 120 | 900 / 1,800 | 약8.3시간 | 약17.6시간 | 0.674 |
| **60 / 160 (권장 검토안)** | **1,020 / 2,040** | **약10.1시간** | **약21.6시간** | **0.795** |

기존 검토의 약10.2/21.7시간과 소수점 차이는 S400 계수를 3.03 대신 3.0으로 반올림해 사용했기 때문이다. 이 표는 **예측이며 상·하한이나 최신 환경의 실측값이 아니다.** O1/O2·모델·대국 길이의 차이, 자원 경합, GPU 메모리·초기화 비용을 교정하지 않았다. 과거 6 worker와 최근 2 worker의 기준 비율은 약2.13이지만, 서로 다른 대국 집합이므로 병렬 확장 효율의 실측값으로 부르지 않는다. holdout 생성·라벨·Rapfi·학습은 포함하지 않는다.

draft JSON에 기록한 계획 수치(쌍·판 수, 비용 시나리오, 검출력, 163쌍, SD 민감도, opening 겹침)는 `tests/test_h7r_pilot_plan.py`가 원천 기록과 위 식으로 다시 계산해 대조한다.

#### 양성 대조 규모와 검출력 가정

- 효과 근거: `docs/h7c/e2/sb_c5_saturation.json`의 150쌍, score=0.5767(기존 opening에서 +7.67%p). 계획값은 반올림한 **δ=0.077**이다. `sb_summary.json`의 원본 모델 50쌍(score=0.59)과 구분한다.
- SD 근거: 50쌍의 구간 반폭을 `SD ≈ 반폭×sqrt(50)/1.96`로 거꾸로 근사하면 원본 SB 약0.361, c5 SB 약0.334다. 효과와 같은 출처인 `sb_c5_saturation.json`의 150쌍 구간 [0.025, 0.1267]로 같은 근사를 하면 **약0.318**이고, 이때 160쌍의 계획 검출력은 약0.866이다. **σ=0.35는 보수적인 계획 가정**으로 유지한다. 어느 값도 원시 pair score에서 계산한 정확한 S400 SD가 아니다.
- 단측 α=0.025에서 `power ≈ Φ(δ×sqrt(n)/σ − z_0.975)`를 쓴다. 위 표는 이 정규 근사다. 실제 opening 쌍 bootstrap 판정의 검출력을 보장하지 않는다.
- 160쌍은 **약79.5%(약80%)**다. 같은 근사식에서 80% 이상을 엄격히 요구하면 **163쌍 이상**이 필요하다. 160쌍을 ≥80% 보장이라고 쓰지 않는다.
- 수치는 suite 하나에 대한 것이며, 세 suite 모두에서 검출될 확률이 아니다. 새 suite에서 효과·SD가 같다는 보장도 없다. 미검출을 자동으로 suite 불량이나 도구 실패라고 판정하지 않고 구간·효과·무결성을 함께 보고한다. 원하는 양성 결과가 나올 때까지 반복하지 않는다.

#### 실행·중단 및 추가 시간 측정

- workers=6과 arm 순서는 **제안**이다. draft JSON에 결과와 무관한 순서를 기록했다. **suite 단위**(O0 → O1 → O2)로 두고, suite 안에서는 S400 대조를 먼저, 이어 c8 · mask · orig 비교를 둔다. arm 경계에서 멈추면 완료된 suite는 네 비교를 모두 갖는다(비교 종류별 순서는 멈추면 모든 suite에서 같은 비교가 빠진다). fixed 전에 합의한다.
- 예산 상한과 초과 뒤 일시중단/종료의 세부 처리는 아직 미정이다. 예산은 arm 경계에서 확인하고 진행 중 arm을 끝낸 뒤 다음 arm 전에 멈추는 안이다. 명목 시간을 한 arm만큼 넘길 수 있어 hard timeout이 아니다. 무결성·실행 오류는 즉시 멈춘다.
- 현재 CLI는 한 arm만 실행한다. JSON의 `execution_plan`·`planning`은 운영 초안이며 자동 스케줄러나 강제 예산 기능으로 구현된 것이 아니다. 필요한 비교가 빠진 pilot에서 최종 suite를 고르지 않는다.
- 같은 c5·같은 국면의 S50/S400 `timing`은 **선택 사항**으로 둔다. 기존 고정 smoke 계약과 `--side c5 --side c5:8`로 가능하다. 이 순서의 ratio는 **S50/S400**이며, S400/S50가 필요하면 역수 또는 side 순서를 바꾼다. 이는 시작 국면 한 수의 비율로, 쌍 전체 비용 3.0배를 직접 검증하는 측정은 아니다.
- 실행 전에 c5/c8뿐 아니라 mask/orig를 포함한 checkpoint·metadata 네 SHA를 모두 확인한다. 확인을 마친 후에만 fixed 전환과 커밋을 검토한다.

#### 선택 기준·예산 제안 검토 — 아직 확정하지 않음

- **score_split(b)를 중심으로 검토하는 방향은 타당하지만, 180개를 독립 Bernoulli 표본으로 간주하는 pooled Wilson 하한은 쓰지 않는다.** 작은 차이 세 arm이 같은 60개 opening을 공유하므로, 같은 opening에서 나온 세 값은 연관될 수 있다. 기록 수 180과 재표집 단위 60을 구분한다.
- 대안 초안은 opening별 `q_i=(세 arm의 score_split 표시 합)/3`를 만든 뒤 60개 `history_id`를 단위로 세 결과를 함께 bootstrap하는 것이다. 점추정은 기존 pooled 비율과 같고, 불확실성 계산만 opening 연관성을 유지한다. 하위 2.5% 분위, 20,000회, seed20261029를 제안 메타데이터에 기록했다. S400은 표본·목적이 다르므로 순위 통계에 섞지 않는다.
- 이 통계는 아직 `rank_by`로 확정하지 않았고 도구에 자동 selector를 구현한 것도 아니다. 같은 suite의 세 arm 결과를 묶는 분석이 필요하다. 분산이 0인 bootstrap 구간을 확실성으로 해석하지 않고, 이전 평가와의 겹침·대칭의 연관성도 계속 보고한다.
- 색 편향은 기존 c4–c8·NMASK 공식 기록의 `abs(tested_black_score-tested_white_score)`가 **0.07–0.33**임을 대조했다. 각 원본 경로·SHA·값을 draft에 기록했다. 이 분포만으로 O1/O2의 배제 문턱을 정하지 않는다. `colour_gap_max`와 여러 arm의 색 차이를 합치는 방식(최댓값/평균 등)은 fixed 전에 별도로 결정한다.
- **14시간은 검토용 soft 예산 제안**으로만 기록했다(6 worker 비용 대용값10.1299시간의 약1.38배). 실제 예산 필드는 null이다. 누적 시간인지 세션별 시간인지, 일시중단 시간을 어떻게 셀지, 예산 도달 후 재개 조건을 먼저 정해야 한다. arm 도중 강제로 끊지 않으므로 hard 14시간 상한도 아니며 현재 CLI가 자동으로 강제하지 않는다.
- 원격 `e49a53e`의 suite별 실행 순서·opening 겹침 기록·SD 민감도·fixture seed 충돌 수정은 유지한다. **0.35가 0.318보다 크다는 것은 그 민감도 가정보다 보수적이라는 뜻이며, 새 suite 모두에 대한 SD 상한을 보장하지 않는다.**

### 9.3 본 pilot 전에 확정할 것

1. arm당 쌍 수(9.2절 권장 검토안: 작은 차이60 / S400 160), 포함할 suite · 비교, worker 수
2. suite 선택 문턱
   - 정보 비율은 (a)와 (b) 중 무엇의 하한으로 순위를 매길지
   - 흑백 점수 차 상한
3. 양성 대조군 보고 형식: 효과와 구간을 그대로 기록하고, 미검출 원인 구분을 보고서에 둔다. 재실행 없음.
4. m의 대상((a) / (b))과 값은 pilot 뒤 확인 계약에서 정한다.
5. holdout 계약(seed, seed용 cycle 번호, H-dev 400판, actor c5)과 그 생성 시점
6. 결과와 무관한 arm 순서·예산 상한·초과 후 처리, mask/orig를 포함한 모델 SHA 확인
7. 위 미정 항목을 모두 결정한 뒤 pilot 계약 `R1-PILOT-1`을 fixed로 바꾸고 커밋한다(현재는 draft이며 실행하지 않는다).

**2026-10-29 결정(9.9):** 1 · 2 · 3 · 6은 사용자 결정에 따라 계약에 확정했고 `R1-PILOT-1`은 fixed다. 4 · 5는 pilot 뒤 단계로 남는다.

### 9.4 커밋한 suite · 계약 파일

- suite(`docs/h7r/suites/`, 코드 `0ced742` clean에서 생성)

  | suite | opening | 대칭 동치류 | 이전 Arena 6개와 겹친 수순(합) | 겹친 동치류(합) | SHA |
  |---|---:|---:|---:|---:|---|
  | O0 | 300 | 75 | 168 | 219 | `f6da0ed5ccff` |
  | O1 | 300 | 298 | 0 | 0 | `50b0e75ffef8` |
  | O2 | 184 | 26 | 102 | 74 | `28b4424b19fe` |

  - O1은 이전 평가와 겹치는 수순이 없다. O0 · O2는 이전 cycle Arena가 쓴 opening과 겹친다.
  - 겹침 처리는 확인 계약에서 정한다(2.2절).
- 계약(`docs/h7r/contracts/`)
  - `R1-PILOT-1`(`15293ca9f010`): **fixed**(9.9, 2026-10-29). 이전 draft SHA는 `c977490e2ebd`였다. 작은 차이9 arm ×60쌍 + S400 3 arm ×160쌍 =1,020쌍/2,040판. 작은 차이는 seed8781–8783(`O*_small`), S400은 seed8784–8786(`O*_positive_control`)이다. 초기 100쌍 균등 초안 SHA는 `ab48eb996271`이었다. 실행은 `scripts/h7r_pilot.py run`으로만 한다(9.9).
  - `R1-SMOKE-1`(`42e9ac6b7b27`): fixed. 데스크톱 smoke용 c8 : c5 S50 2쌍(seed 8798), c5 S400 : c5 S50 1쌍(seed 8799). 판정에 쓰지 않는다.
  - `R1-SMOKE-HOLDOUT`(`a56b0ac6ccdf`): fixed. holdout 경로 확인용 c5 2판(seed 8797, cycle 1). 학습에 쓰이지 않는다.
- 수정 draft에서 제안한 seed까지 제외하면 남는 H7-R seed는 **8787–8796**(holdout·확인 계약용)이다. 이 중 8795 · 8796은 테스트 fixture가 쓰므로(`tests/h7r_helpers.py`의 `FIXTURE_*_SEED`; 커밋된 계약이 가져가면 `test_h7r_holdout`이 알린다) 실제 계약은 8787–8794부터 쓴다. draft 계약도 seed를 차지한다: holdout 생성기는 커밋된 모든 계약과 seed 중복을 검사한다. 이번 문서 수정으로 실행 ledger를 생성하거나 seed를 실제 사용하지는 않았다.

### 9.5 데스크톱 smoke 명령 (`~/renju-h6`, `.venv`, `PYTHONPATH="$PWD/src"`)

```bash
cd ~/renju-h6 && source .venv/bin/activate && export PYTHONPATH="$PWD/src"
git pull origin feat/h7-c1 && python scripts/check_track_b_frozen.py
mkdir -p runs/h7r/smoke
# 1) Arena arm: S50, then S400 (cost per arm type)
python -u scripts/h7r_eval.py play --contract docs/h7r/contracts/R1-SMOKE-1.json --arm smoke_s50 \
  --device cuda --workers 2 --jsonl runs/h7r/smoke/s50.jsonl --output runs/h7r/smoke/s50.json
python -u scripts/h7r_eval.py play --contract docs/h7r/contracts/R1-SMOKE-1.json --arm smoke_s400 \
  --device cuda --workers 2 --jsonl runs/h7r/smoke/s400.jsonl --output runs/h7r/smoke/s400.json
python -u scripts/h7r_eval.py summarize --contract docs/h7r/contracts/R1-SMOKE-1.json \
  --arm-record runs/h7r/smoke/s50.json --output runs/h7r/smoke/s50_summary.json
# 2) time gate format (same process, alternating, warm-up discarded)
python -u scripts/h7r_eval.py timing --contract docs/h7r/contracts/R1-SMOKE-1.json --side c8 --side c5 \
  --suite docs/h7r/suites/O0.json --positions 20 --warmup 3 --device cuda --output runs/h7r/smoke/timing.json
# 3) holdout path: 2 games, evaluation labels, value evaluation
python -u scripts/h7c_selfplay.py --cycle 1 --run-seed 8797 --games 2 --workers 2 \
  --policy-checkpoint runs/h3_policy_64x4/best.pt --value-checkpoint runs/h7c/c5/value/best.pt --device cuda \
  --jsonl runs/h7r/smoke/holdout.jsonl --holdout-contract docs/h7r/contracts/R1-SMOKE-HOLDOUT.json
python -u scripts/h7r_holdout.py labels --contract docs/h7r/contracts/R1-SMOKE-HOLDOUT.json \
  --games runs/h7r/smoke/holdout.jsonl --value-output runs/h7r/smoke/holdout_rows.jsonl.gz \
  --cache-dir runs/h7r/smoke/holdout_cache --workers 2 --output runs/h7r/smoke/holdout_labels.json
python -u scripts/h7r_holdout.py eval --cache runs/h7r/smoke/holdout_cache --model c5=runs/h7c/c5/value/best.pt \
  --model c8=runs/h7c/c8/value/best.pt --pair c8:c5 --d4 --device cuda --output runs/h7r/smoke/holdout_eval.json
# 4) the training path must refuse the holdout games (expected: an error saying "never training data")
python -u scripts/h7c_labels.py --games runs/h7r/smoke/holdout.jsonl --value-output runs/h7r/smoke/x_rows.jsonl.gz \
  --policy-output runs/h7r/smoke/x_policy.jsonl.gz || echo "refused as expected"
```

- 확인할 것
  - `s50.json` · `s400.json`의 `wall_seconds` · `games[].game_seconds`: 9.2절 비용 추정을 실측으로 바꾼다.
  - `summary`의 `move_split` · `score_split`
  - `timing.json`의 `environment`, `ratio`
  - holdout 기록의 `purpose: holdout`, 4)의 거부
- smoke 결과는 `runs/`에 두고 커밋하지 않는다. 비용 수치만 다음 계약 검토 때 옮긴다.
- 전체 pilot, H-dev · H-conf 생성, 학습, 확인 Arena는 이 명령에 없다. 각각 별도 승인 뒤에 한다.

### 9.6 R1 코드 검토 후 보완: 원본 증거 검증과 확인 계약 잠금

초기 구현의 7개 오류 경로를 수정했다. 이 보완은 평가 도구에 한정되며, pilot 규모·승급 문턱·학습법을 새로 확정하거나 champion을 바꾸지 않는다. 9.4절의 기존 pilot·smoke 계약과 suite 파일도 그대로다.

| 검토 항목 | 수정한 동작 |
|---|---|
| 다른 모델의 tactical PASS 재사용 | candidate·champion의 네 SHA, rule, probe·E1 manifest SHA, clean 코드 출처를 대조한다. fresh sign과 E1 원본 run에서 판정을 재계산하고 must_block 건수를 검사한다. **9.7에서 변경:** 확인 판정은 계약 SHA를 담은 `h7r-tactical-v1`만 받는다. 기존 cycle·NMASK 형식은 읽기만 호환한다. |
| 시간 비교 방향·모델 혼동 | `h7r-timing-v2`는 candidate / champion 순서, 양쪽 모델 SHA·S50 예산, suite·국면 ID, warm-up·환경·원시 시간 표본을 기록한다. 확인 판정은 원시 표본에서 평균·p95 비율과 통과 여부를 다시 계산한다. |
| 임의 delta를 확인용 선택 품질로 사용 | 확인 계약에서는 `selection --rows`를 거부한다. 완전한 H-conf, 두 모델의 전체 sensitivity, 등록된 Rapfi의 원본 분기를 SHA로 연결하고 모든 불일치에서 delta를 만든다. 분기를 기보로 재생해 결과를 대조한다. INVALID 처리: 정책이 없는 계약은 INVALID가 있으면 INCONCLUSIVE다. 새 계약은 9.7의 버전 정책을 고를 수 있다. |
| 결과 flag만으로 Arena 집계 | 저장·재개·요약·판정 때 task seed·pair·색·opening·모델·엔진·runtime을 대조하고, 전체 착수를 `renju.Game`으로 재생한다. 종료 상태·승자·결과와 불법 수 사건을 확인한다. |
| 일부 holdout 또는 일반 test cache 사용 | 계약의 판 수와 `game_index=0..N-1`을 모두 요구한다. `holdout.json`이 계약 → 원본 대국 → 평가 행 → cache 파일을 SHA로 연결한다. `test` split이라는 이유만으로는 평가에 쓰지 않는다. |
| 출력 경로를 바꿔 재판정 | 로컬 영구 기록이 계약 ID·seed·H-conf 사용·판정 입력 SHA를 묶는다. 동일 입력의 감사용 재계산은 허용하지만, 같은 계약의 입력 변경이나 같은 H-conf의 다른 확인 계약 사용은 거부한다. |
| 누락 모델 SHA가 null로 일치 | 모델 identity에는 네 개의 유효한 SHA-256이 필수다. 실행 시 네 파일이 실제로 존재하며 각각 계약 SHA와 같은지도 확인한다. |

**확인 계약에 추가로 고정할 값**

- `selection.min_observations`: 필요한 최소 불일치 관측 수
- `selection.holdout_contract_sha256`: 해당 확인에만 사용할 fixed H-conf 계약의 SHA
- `selection.engine`: 기존 `h7c_adjudicate.engine_spec(program, rapfi_dir)['identity']`가 반환하는 등록 Rapfi identity 전체(실행 파일·부속 파일·INFO·rule·인자·timeout)

이 값이 없으면 confirm 실행·판정을 시작하지 않는다. 확인용 최소 관측 수는 이제 암묵적 기본값을 쓰지 않는다. 값과 계약을 확정·커밋하는 일은 본 확인 실행 전 단계로 남아 있다.

**선택 품질 원본 생성 절차**

확인 계약과 H-conf가 확정되어 있고 `h7r_holdout.py labels`가 완료된 뒤에 사용하는 명령 형식이다. 아래 경로의 `CONFIRM_ID`, `HCONF_ID`는 실제 계약명으로 바꾼다. 이번 수정에서는 이 실험을 실행하지 않았다.

```bash
python -u scripts/h7r_eval.py selection-source \
  --contract docs/h7r/contracts/CONFIRM_ID.json \
  --cache runs/h7r/HCONF_ID/value_cache \
  --source-dir runs/h7r/CONFIRM_ID/selection_source \
  --program docs/h7c/b2_program.json --rapfi-dir /path/to/registered/rapfi \
  --device cuda --workers 4 --output runs/h7r/CONFIRM_ID/selection_source.json
python -u scripts/h7r_eval.py selection \
  --contract docs/h7r/contracts/CONFIRM_ID.json \
  --evidence runs/h7r/CONFIRM_ID/selection_source.json \
  --output runs/h7r/CONFIRM_ID/selection.json
```

- 원본 생성은 기존 `h7c_sensitivity.replay_game`과 `h7c_adjudicate.play_branch`를 재사용한다. 후보 factory를 직접 사용하는 새 경로를 만들지 않는다.
- H-conf 전체 게임·전체 ply를 처리한다. champion의 move·route·visits가 원본 self-play와 재현되는지 확인한다. 불일치 관측을 임의로 골라 넣는 입력은 없다.
- `sensitivity.jsonl`과 `branches.jsonl`을 재개하며, 완료한 분기를 다시 실행하지 않는다. 중단으로 마지막 JSONL 행이 잘렸으면 그 행만 제거해 재개한다. 다른 identity·중간 손상·중복은 거부한다. source 디렉터리는 동시 작성 잠금을 사용한다.
- H-conf 사용은 원본 평가 시작 전에 확인 계약에 묶는다. 원본 manifest와 하위 파일을 함께 보존해야 `selection`과 `judge`가 다시 검증할 수 있다.
- pilot·개발용 `selection --rows`는 남겨 두되 `development_only`로 표시하고 확인 판정에는 쓰지 않는다.

**호환성과 보관**

- 기존 `h7r-timing-v1` 요약은 원시 표본이 없어 새 확인 판정에서 받지 않는다. `timing`으로 v2를 생성해야 한다.
- 기존 평가 cache에 `holdout.json`이 없으면 일반 cache와 구분할 수 없으므로 거부한다. 전체 원본 holdout을 `labels`에 넣고 새 행 출력·새 cache 경로에서 다시 만든다. 기존 파일에 PASS나 봉인 표시만 덧붙이지 않는다.
- ~~증거 없이 KEEP_CHAMPION을 기록한 경우도 그 입력 조합에 묶여 재판정할 수 없다.~~ **9.7에서 변경:** 확인 `judge`는 세 Arena arm과 selection·tactical·timing을 모두 요구한다. 하나라도 없거나, 파일이 없거나, 검증에 실패하면 영구 기록 전에 거부한다. 증거를 갖춰 다시 실행하면 정상 판정된다. 한 번 판정된 계약의 입력 변경은 계속 거부한다.
- `runs/h7r/confirmations/`는 커밋하지 않는 **현재 checkout의 로컬 기록**이다(보관 절차는 9.7). 다른 PC·worktree에서 실행을 이어가면 원본 증거와 이 디렉터리도 함께 옮겨 사용한다. 지우거나 누락하면 과거 사용을 추적할 수 없다. 여러 PC를 아우르는 중앙 등록 서비스는 구현하지 않았다.
- 최종 JSON은 임시 파일을 완성한 후 배타적으로 공개한다. 이미 있는 출력은 덮어쓰지 않고, 직렬화 실패가 불완전한 결과 파일을 남기지 않는다.

**검증**

`tests/test_h7r_evidence.py`에 위 7개 경로의 거부 테스트, 원본 생성·재개, 출력 원자성 검사를 추가했다. 실제 모델이나 외부 Rapfi를 실행하지 않는 fixture 테스트이며, R1 CLI 통합 테스트는 합법적인 완주 기보와 연결된 원본 증거를 사용하도록 바꿨다.

```bash
PYTHONPATH=src:scripts python -m unittest \
  tests.test_h7r_eval tests.test_h7r_cli tests.test_h7r_holdout \
  tests.test_h7r_openings tests.test_h7r_evidence -v
python scripts/check_track_b_frozen.py --no-models
```

실제 GPU 모델·등록 Rapfi의 end-to-end 실행과 전체 pilot은 이번 회귀 테스트에 포함되지 않는다. 그 결과 없이 R1 완료나 기력 향상을 주장하지 않는다.

검증 결과: 최종 R1 묶음 38개는 모두 통과했다. 기존 경로를 포함한 회귀 묶음은 187개 중 185개 통과, 실제 frozen 모델이 필요한 2개 skip이었다. `check_track_b_frozen.py --no-models`는 PASS다. 전체 checker는 이 컨테이너에 H3 policy·H6 value checkpoint가 없어 두 모델 검사만 FAIL이며, code·engine·runtime·TB-V3 identity·promotion·canonical 검사는 모두 PASS다.

### 9.7 확인 평가를 끝까지 수행하기 위한 보완: INVALID 정책 · 일반 후보 tactical · 판정 순서

`e97df11` 검토의 네 가지 문제를 고쳤다.
- INVALID 하나로 확인이 사실상 불가능했다.
- 일반 R3 후보의 tactical 기록을 만들 수 없었다.
- 증거가 빠진 판정이 ledger를 잠갔다.
- champion의 금수가 후보 실패로 기록됐다.

평가 도구에 한정한 변경이다. 기존 계약 · 원본 기록 · 판정은 고치거나 다시 판정하지 않는다. 확인 계약을 fixed로 바꾸지도 않았다. pilot · H-conf 생성 · 확인 Arena · champion 교체 · SER5 배포는 하지 않았다.

**INVALID 처리 정책 `h7r-invalid-v1` (새 확인 계약에서 선택)**

- 위치: 확인 계약의 `selection.invalid_policy`. 계약 SHA에 포함된다.
- 이 필드가 없는 계약은 기존의 엄격한 규칙 그대로다(INVALID 분기가 하나라도 있으면 `INCONCLUSIVE`, `INVALID_BRANCHES`).
- draft 예시: `docs/h7r/invalid_policy_v1.draft.json`
  - `contracts/` 밖에 두어 계약으로 읽히지 않는다.
  - 상한 0.05는 제안값이다. 실제 확인 계약에 넣고 확정하는 일은 본 확인 전 단계로 남긴다.

```json
"invalid_policy": {
  "version": "h7r-invalid-v1",
  "structural_abnormal_end": "evaluation_draw",
  "unresolved": "worst_case",
  "unresolved_reasons": ["timeout", "engine_error", "engine_illegal", "forbid_mismatch"],
  "max_invalid_share": 0.05,
  "denominator": "disagreement_observations"
}
```

모든 키가 필수다. 다른 버전 · 값 · 키가 있으면 계약 검사가 거부한다.

1. **구조적 `abnormal_end`**(`structural_abnormal_end`: `evaluation_draw` 또는 `unresolved`)
   - 원본 prefix + 고정 수 + continuation을 `renju.Game`으로 모두 재생한다.
   - 기록의 identity · 고정 수 · ply · length가 재생과 같아야 한다.
   - 다음 조건을 모두 만족해야 한다.
     - 승자가 없다.
     - 흑 차례다.
     - 빈 점이 정확히 하나다.
     - 흑의 합법 수가 없다.
     - 그 빈 점이 실제 흑 금수다.
   - `evaluation_draw`일 때만 평가 점수 0.5로 정규화한다.
     - 이를 "평가상 무승부"로 표시한다.
     - 출력에 `normalized.abnormal_end_evaluation_draw`(분기 수)와 `structural_observations`(관측 수)를 낸다.
   - 원본 INVALID 분기 기록은 그대로 둔다. `renju.Game`과 엔진의 종료 규칙도 바꾸지 않았다.
   - 조건을 하나라도 만족하지 않는 `abnormal_end`는 데이터 오류로 거부한다. 정의되지 않은 사건이므로 새 정책 버전이 필요하다.
2. **미해결 분기**(`unresolved: worst_case`)
   - 관측을 삭제하지 않는다.
   - 정상 분기는 재생으로 확인한 점수를 쓴다.
   - 미해결 후보 분기는 0점, 미해결 champion 분기는 1점으로 둔다. delta = 후보 − champion이므로, 양쪽 모두 미해결이면 delta = −1이다.
   - 같은 분기가 여러 관측에 쓰이면 각 관측에 반영한다.
3. **비율과 판정**
   - `invalid_share` = 미해결 분기가 하나 이상인 불일치 관측 수 / 전체 불일치 관측 수(`denominator`)
   - 구조적으로 정규화한 분기는 미해결에서 빼고, 따로 센다.
   - 다음 경우는 모두 `INCONCLUSIVE`다(사유를 기록한다).
     - 관측 0개: `NO_OBSERVATIONS`
     - 정책에 없는 실행 실패: `UNHANDLED_INVALID`
     - 비율 초과: `INVALID_SHARE_EXCEEDED`. 경계값은 정확한 분수로 비교하며, 상한과 같으면 통과다.
     - 실측(미해결 없는) 관측이 `min_observations` 미만: `TOO_FEW_RESOLVED`. 대체값이 최소 실측 관측 수를 채우지 못한다.
   - 이 조건을 넘으면 보수적 delta를 포함한 게임 단위 bootstrap이 비열등을 보여야 통과한다.
4. **실행 실패와 데이터 오류**
   - 처리 가능한 실행 실패는 `timeout`, `engine_error`, `engine_illegal`, `forbid_mismatch`다. 정책의 `unresolved_reasons`에 있는 것만 미해결로 처리한다.
   - 실행 실패에도 재생 검사를 한다.
     - 실패가 대국 종료 뒤에 기록되면 거부한다.
     - `engine_illegal`의 수가 실제로 합법이면 거부한다.
     - `forbid_mismatch`가 우리 금수점과 맞지 않으면 거부한다.
   - 다음은 미해결로 낮추지 않고 데이터 오류로 거부한다.
     - 누락 · 중복 · identity 불일치
     - 불법 기보 · 변조된 결과 · 길이 불일치
     - `truncated` · `fixed_move_illegal` 같은 다른 사유
   - 정책이 없는 엄격한 계약에도 같은 재생 검사를 적용한다. 엄격한 판정이 통과로 바뀌는 경우는 없다.

**일반 후보 tactical: `scripts/h7r_eval.py tactical --contract`**

- 계약의 candidate / champion 경로를 사용하고, 네 SHA를 실제 파일과 대조한다.
- 다음을 확인한 뒤에만 실행한다.
  - fixed · committed 확인 계약, 규칙 값 모두 설정
  - clean 코드 상태
- cycle 계보와 `h7c_stage`는 요구하지 않는다.
- 측정은 기존 `h7c_arena.value_probes`, `must_block`, `e1_suite`, `tactical_judge`를 그대로 쓴다. 후보 factory는 `scripts/h7c_*` 경로로만 쓰인다.
- 기록 형식은 `h7r-tactical-v1`이다.
  - 계약 ID와 SHA, 역할, 두 모델 identity
  - git, rule, probe · E1 manifest SHA, 원본 결과
- 진행 상황은 측정 단계와 E1 `Progress`로 출력한다. 출력 파일이 이미 있으면 덮어쓰지 않는다.
- 판정 때 검증하는 것
  - 계약 SHA · 역할 · 모델 SHA가 일치한다.
  - tactical 기록의 코드 commit이 세 Arena arm의 clean commit과 같다. 세 arm의 commit이 서로 다르면 그것도 거부한다.
  - H-conf 생성 commit은 별도 출처이므로 같을 것을 요구하지 않는다.
- 기존 cycle · NMASK tactical 기록은 계속 읽을 수 있다. 확인 판정에는 받지 않는다. 계약 SHA를 덧붙여도 거부한다.

**판정 순서와 ledger**

- `judge` 순서: 입력 확인 → 원본 증거 검증 → 판정 계산 → ledger 등록 → 결과 공개
  - 입력 확인: 확인 계약, fixed · committed, 세 arm과 selection · tactical · timing 파일
  - 원본 증거 검증: arm 재생 · commit 일치, tactical · timing · selection 재계산
  - ledger 등록: contract · seed · H-conf 사용 · 판정 입력 (**9.8에서 변경:** contract · seed · H-conf 사용은 `play` · `selection-source`의 사전 등록을 확인만 하고, judge는 판정 입력과 `judge-code`만 등록한다)
- 검증 함수(`h7r_evidence.selection`)는 이제 ledger를 쓰지 않는다. 다른 확인 계약에 묶인 H-conf만 읽기 전용으로 거부한다.
- 여러 binding은 `h7r_ledger.bind_all`이 한 잠금 안에서 처리한다. 충돌을 모두 먼저 검사하므로, 거부되면 아무 항목도 남지 않는다.
- 동일 입력의 감사용 재계산은 허용한다. 판정된 계약의 입력 변경은 거부한다. 입력 누락으로 실패한 뒤 증거를 갖춰 다시 실행하면 정상 판정된다.
- `selection-source`는 모든 입력을 확인한 뒤, 평가를 시작하기 전에 계약 · seed · H-conf 사용을 한 번에 예약한다. 중단 후 같은 입력으로 재개할 수 있다. 같은 H-conf를 다른 확인 계약이 쓰면 거부한다.
- `--allow-dirty`(테스트 전용)는 저장소 자체의 `runs/h7r/confirmations/`에 쓰지 못한다. 테스트는 임시 root를 쓴다.
- 최종 verdict의 `ledger.entries`에 사용한 ledger 항목의 경로와 저장 바이트의 SHA-256을 기록한다.

**ledger 보관 절차**

- ledger는 **로컬 규율 장치**다. 새 clone이나 다른 PC는 빈 ledger로 시작하므로, 그곳까지 보장하지 않는다.
- 확인 계약 하나를 끝내면 다음을 함께 보관한다.
  - `runs/h7r/confirmations/` 전체
  - 그 확인의 원본 증거: arm JSONL · 기록, selection-source 디렉터리, tactical, timing
  - verdict
- 다른 PC에서 이어서 하면 이 묶음을 먼저 옮긴다.
- verdict의 `ledger.entries` SHA와 보관본을 대조할 수 있다.
- ledger를 지우거나 강제로 초기화해 거부를 우회하지 않는다.

**금수 오류의 책임**

- 후보의 `GATE_FORBIDDEN` 실패는 후보가 tested인 두 arm(`candidate_vs_champion`, `candidate_vs_anchor`)의 불법 수만 센다(`candidate_illegal_games`).
- `champion_vs_anchor`에서 champion이 둔 불법 수는 기준 arm 오염이다.
  - `ARM_INVALID`로 처리하고, `arm_invalid`에 원인 `CHAMPION_ILLEGAL_IN_REFERENCE_ARM`을 기록한다.
  - 승급은 막지만 후보 금수 오류로 보고하지 않는다.
- 상대의 불법 수(`OPPONENT_ILLEGAL`)는 기존대로 arm을 invalid로 한다. 그 쌍이 한 anchor arm에서만 빠지면 anchor 차이를 `NOT_COMPUTED`로 둔다(예외로 중단하지 않는다).
- `decision`은 그대로 `PROMOTE` / `KEEP_CHAMPION`이다. KEEP일 때 `keep_kind`와 `reason_classes`가 판정 불가와 후보 실패를 구별한다.
  - `NOT_JUDGEABLE`: 누락 · 미설정 · invalid · INCONCLUSIVE
  - `CANDIDATE_FAILED`: `NOT_SUPERIOR`, `ANCHOR_INFERIOR`, `SELECTION_INFERIOR`, `GATE_*`
  - arm이 invalid이면 Arena에서 나온 `NOT_SUPERIOR` · `ANCHOR_INFERIOR`는 후보 실패로 세지 않는다.
  - tactical · timing이 없으면 `TACTICAL_MISSING` · `TIME_MISSING`으로 쓴다(판정 불가).

**오류 처리와 신뢰 경계**

- `main`은 입력 오류(`ValueError`, `FileNotFoundError`)만 사용법 오류로 보여 준다. 예상하지 못한 코드 오류는 traceback을 남긴다. arm 기록의 형식은 검증 단계에서 `ValueError`로 설명한다.
- tactical의 추론값(fresh probe 값) · must_block hit와 timing의 원시 표본은 **기록된 측정값을 신뢰**한다.
  - 오프라인 검증은 identity · 완전성 · 형식과, 그 값이 함축하는 판정을 다시 계산한다.
  - 모델 실행 자체를 재증명하지는 않는다.

**명령 예시** (본 확인 단계에서만, `CONFIRM_ID`는 실제 계약명)

```bash
python -u scripts/h7r_eval.py tactical --contract docs/h7r/contracts/CONFIRM_ID.json \
  --device cuda --workers 6 --output runs/h7r/CONFIRM_ID/tactical.json
python -u scripts/h7r_eval.py judge --contract docs/h7r/contracts/CONFIRM_ID.json \
  --arm-record runs/h7r/CONFIRM_ID/candidate_vs_champion.json \
               runs/h7r/CONFIRM_ID/candidate_vs_anchor.json \
               runs/h7r/CONFIRM_ID/champion_vs_anchor.json \
  --selection runs/h7r/CONFIRM_ID/selection.json --tactical runs/h7r/CONFIRM_ID/tactical.json \
  --timing runs/h7r/CONFIRM_ID/timing.json --output runs/h7r/CONFIRM_ID/verdict.json
```

**검증**

`tests/test_h7r_confirm.py`(20개)를 추가하고, 기존 R1 테스트를 새 동작에 맞췄다. 모두 fixture · stand-in 모델 · mock 실행기를 쓴다. 실제 GPU 모델, 등록 Rapfi, 전체 pilot, H-conf 생성, 확인 Arena는 실행하지 않았다.

- 구조적 `abnormal_end` 정규화
  - 고정 fixture `tests/fixtures/h7r_abnormal_end_branch.json`을 쓴다. 합성 기보이고 RenjuNet 자료가 아니다. 224수 뒤 흑 차례이고, 빈 점 (5, 12)가 장목 금수다.
  - 세 정책(`evaluation_draw` · `unresolved` · 정책 없음)에서 각각 다르게 처리되는지 본다. 원본 INVALID 기록이 보존되는지도 본다.
- 거부: 조건 불일치, 변조(불법 수 · 길이 · 고정 수 · ply · 점수), 실행 실패 사유 검사
- 정책 계산
  - 한쪽 · 양쪽 미해결의 최악값
  - 관측 단위 비율(공유 분기 포함)
  - 상한 경계(2/40, 1/20)와 초과, 관측 0개, 최소 실측 관측 수, 정책에 없는 실패
  - INVALID를 버리면 통과하지만 보수적으로 대체하면 통과하지 못하는 사례
- tactical과 판정 통합
  - cycle 메타데이터가 없는 후보로 `tactical` 생성 → 검증 → `judge` PROMOTE까지 간다.
  - 다른 계약 · 모델 · probe · 코드 commit의 tactical, 과거 형식 tactical, 서로 다른 commit의 arm을 거부한다.
- ledger
  - 증거 누락 · 파일 부재 · 검증 실패 뒤 ledger가 바뀌지 않는다. 증거를 갖춰 다시 실행하면 정상 판정된다.
  - 동일 입력의 재계산은 허용하고, 확정 후 입력 변경은 거부한다.
  - `bind_all`은 충돌을 먼저 검사한다(일부만 기록되지 않는다).
  - `selection-source`는 평가 전에 H-conf를 예약하고, 중단 후 재개하며, 다른 계약의 사용을 거부한다.
  - 실제 ROOT에 `--allow-dirty`로 쓰는 경로를 차단한다.
- 후보 금수와 champion 기준 arm 오류를 다르게 분류한다(opponent 불법 수 포함).

결과
- R1 묶음(`test_h7r_*` 6개 파일): 58개 모두 통과
- 기존 경로를 포함한 회귀(`test_h7*`, `test_tb_v3*`, `test_h6_labels`, `test_track_b*`, `test_canonical*`): 414개 실행, 412개 통과, 2개 skip
  - skip 2개는 실제 frozen H3 / H6 checkpoint가 필요한 `test_tb_v3_net` 테스트다. 코드 실패가 아니다.
  - 이 실행에는 `test_h7r_confirm`이 `JudgeTest`를 이름으로 가져와 2개가 중복 실행된 것이 포함돼 있다. 그 import를 고친 뒤 R1 묶음만 다시 실행했다(58개 통과).
- frozen checker
  - `check_track_b_frozen.py --no-models`: PASS
  - 전체 checker: policy / value checkpoint 두 항목만 FAIL이다. 이 컨테이너에 `runs/h3_policy_64x4/best.pt` · `runs/h6_value_head_v3/best.pt`가 없기 때문이며, 코드 실패가 아니다. code · engine · runtime · TB-V3 identity · promotion · canonical 검사는 모두 PASS다.

이 보완만으로 R1 완료나 기력 향상을 주장하지 않는다. 실제 모델로 만든 tactical 기록과 등록 Rapfi의 분기 기록은 본 확인 단계에서 처음 생긴다.

### 9.8 추가 검토 보완: 판정 코드 출처와 빈 anchor 표본

- **현재 판정 코드 검사**: `judge`도 실행 시 Git 상태를 확인한다. 현재 commit이 있어야 하고 작업 트리는 clean이어야 한다. 현재 commit은 세 Arena arm의 공통 commit과 정확히 같아야 한다. 별도의 버전 호환 예외는 두지 않는다.
  - verdict에 실제 판정 코드의 `judge_git`을 남긴다. 기존 `code_commit`은 arm의 공통 commit이며, 이제 둘의 commit이 같음을 검사한다.
  - ledger에 확인 계약 SHA를 key로 한 `judge-code` 항목을 `judgment`와 함께 기록한다. 저장 바이트의 SHA는 `ledger.entries`에 포함된다.
  - **후속 변경(아래 "측정 commit · 사전 등록"):** `judge-code` 값은 `{'commit': ...}`만이다. 전체 Git 정보는 verdict의 `judge_git`에 남는다.
  - 작업 트리가 dirty이거나 commit이 다르면 ledger 등록 전에 거부한다. 테스트 전용 `--allow-dirty`도 판정 코드의 clean·동일 commit 검사를 건너뛰지 않는다. 테스트는 임시 root와 명시적으로 주입한 Git 상태를 사용한다.
  - 향후 확인에서는 평가 도구·계약을 커밋한 뒤 동일 commit에서 arm·tactical·judge를 실행한다. 코드 변경 뒤 예전 arm을 현재 코드로 재판정하는 우회는 허용하지 않는다. 과거 계약·원본·판정은 수정하거나 다시 실행하지 않았다.
- **빈 anchor 표본**: 한쪽 또는 양쪽 anchor arm의 모든 쌍이 상대 불법 수로 제외되면, 빈 집합끼리 같더라도 paired 차이를 계산하지 않는다. `anchor=NOT_COMPUTED`, `reason=ARM_INVALID`, `decision=KEEP_CHAMPION`, `keep_kind=NOT_JUDGEABLE`로 기록한다. 정상적으로 남은 표본의 계산 방식은 그대로다.
- **구조적 관측 수**: `structural_observations`는 평가상 무승부로 정규화된 분기가 하나 이상 있는 관측 수다. 같은 관측의 다른 분기가 timeout 등 미해결이면 `unresolved_observations`에도 센다. 두 수는 겹칠 수 있으며 합계로 전체 관측 수를 계산하면 안 된다. 미해결 비율·최악값 delta·최소 실측 표본·승급 문턱은 바꾸지 않았다.

#### 측정 commit · 사전 등록 · anchor 짝맞춤 (6cbf802 후속)

**anchor 짝맞춤은 계약을 읽을 때 검사한다.**
- confirm 계약의 `candidate_vs_anchor`와 `champion_vs_anchor`는 같은 표본이어야 한다. 다음 네 값이 모두 같아야 한다.
  - 비어 있지 않은 문자열 `pairing`
  - `seed`
  - `pairs`
  - suite SHA
- 같은 `pairing`을 선언한 arm들은 seed · pairs · suite SHA가 모두 같아야 한다.
- 위반하면 `check_contract`가 거부한다. 따라서 대국을 두기 전에 오류가 드러난다.
- `candidate_vs_champion`은 다른 pairing과 seed를 써서 따로 뽑아도 된다. 같은 pairing을 선언하면 같은 표본이어야 한다.
- 이 검사는 confirm 계약에만 적용한다. 이미 fixed인 pilot 계약 `R1-SMOKE-1`은 크기가 다른 arm 둘에 같은 suite 이름을 붙였으므로 그대로 둔다.
- 판정 단계의 실제 opening 일치 검사(`anchor_difference`)도 방어적으로 유지한다.

**확인 하나는 코드 commit 하나로 측정한다.**
- 다음은 모두 세 Arena arm의 공통 clean commit과 정확히 같아야 한다.
  - `tactical.git.commit`
  - `timing.git.commit`
  - `selection-source`의 `source_git.commit`
  - judge를 실행하는 현재 commit
- 검증기(`h7r_evidence.timing` · `selection` · `tactical`)는 기대 commit을 인자로 받는다. 기본값이 없어서 확인 경로에서 이 검사를 건너뛸 수 없다.
- 독립 `selection` 명령도 현재 clean commit을 기대값으로 쓴다. 그 commit은 결과의 `source_commit`에 남고, judge가 arm commit과 다시 대조한다.
- H-conf 생성 commit(self-play의 `generator_git`)은 별도 출처라 대조하지 않는다. SHA · 모델 · 기보 · 재계산 검증은 그대로다.

**judge는 사전 등록을 확인만 하고 만들지 않는다.**
- `play` · `selection-source`가 남긴 다음 항목이 이 checkout의 ledger에 이미 기대값으로 있어야 판정한다.
  - 계약 ID → SHA
  - 각 seed → 계약 ID와 SHA
  - H-conf 계약 SHA → 확인 계약 SHA
- 누락 · 불일치 · 손상(JSON으로 읽을 수 없음)은 원인을 밝히고 거부한다.
- ledger가 비어 있으면 다른 checkout · clone일 가능성을 안내한다. 자동 등록 · 복구는 하지 않는다.
- 이 확인은 읽기 전용이며, 증거 검증 전에 한다.
- 최종 등록 잠금 안에서 같은 항목을 다시 확인한다. 그 사이에 항목이 없어졌으면 다시 만들지 않고 거부한다.
- judge가 새로 등록하는 것은 `judgment`와 `judge-code`뿐이다. 동일 입력의 감사용 재계산과, 같은 입력으로 하는 `selection-source` 재개는 그대로 된다.
- **이 검사는 사전 예약을 확인할 뿐이다.** 대국이 끝났다는 것이나 여러 머신의 전역 이력을 증명하지 않는다. 대국 완료는 기존 원본 증거 검증(arm 재생, 완전한 H-conf · 분기)으로 확인한다.

**`judge-code` identity**
- 새 항목은 `{'commit': ...}`만 저장한다. 같은 commit에서 branch 같은 부가 Git 필드가 달라도 충돌하지 않는다.
- 6cbf802 형식(Git 정보 전체를 담은 항목)은 제한된 규칙으로만 읽는다. `commit`과 `dirty`가 있고, commit이 같으며, `dirty`가 false여야 한다. 원본 바이트와 SHA는 그대로 둔다.
- 다른 category의 엄격한 값 비교는 그대로다.

**운영 절차** (예시이며, 이 작업에서는 실행하지 않았다)
1. 확인 실행을 시작하기 전에 평가 도구와 확인 계약을 커밋하고, 그 commit을 기록해 둔다(`ARM_COMMIT`). arm · tactical · timing · selection-source · selection · judge는 모두 이 commit에서 실행한다.
2. 확인 도중에는 문서만 커밋해도 HEAD가 바뀌어 tactical · judge가 거부된다. 분석 초안은 확인이 끝난 뒤 커밋하거나, 다음 절차로 돌아간다.
3. 실행 중인 작업이 모두 끝났고 `git status`가 clean인지 확인한다. 그다음 **같은 checkout**에서 arm commit으로 돌아가 남은 단계를 실행한다.

   ```bash
   git status --short          # 출력이 없어야 한다
   git switch --detach ARM_COMMIT
   python -u scripts/h7r_eval.py judge ...   # 9.7의 명령
   git switch feat/h7-c1       # 끝나면 원래 브랜치로
   ```
4. 새 `git worktree`나 clone은 자기 `runs/`를 쓴다. 따라서 기본적으로 기존 ledger를 공유하지 않는다. 그곳의 judge는 빈 ledger로 거부된다.
5. 다른 checkout · PC로 옮겨야 하면 `runs/h7r/confirmations/`와 원본 증거를 함께 보관 · 복원한다. 복원한 뒤 verdict의 `ledger.entries` SHA, 또는 이전 보관본과 대조해 검증한다. 빈 ledger를 새로 만들어 우회하지 않는다.
6. ledger는 로컬 규율 장치다. 여러 PC를 아우르는 중앙 등록 서비스는 구현하지 않았다.

회귀 테스트는 다음을 확인한다.
- dirty·Git 상태 미확인·현재 commit 불일치 시 판정 거부와 ledger 불변, 동일한 clean commit에서 PROMOTE 및 감사용 재계산
- 실제 기보 재생을 통과한 상대 불법 수 기록으로 한쪽·양쪽 anchor의 유효 쌍이 모두 없어지는 경우
- 후보 쪽 또는 champion 쪽 구조적 무승부와 미해결 분기가 섞인 관측의 중복 집계 및 기존 최악값 점수 유지

검증 결과: R1 6개 모듈과 `test_h7c_recipe_guards`, `test_track_b_guard`, `test_tb_v3_net`을 함께 실행해 총 87개 중 85개 통과, 실제 frozen H3/H6 모델이 필요한 2개 skip을 확인했다. `check_track_b_frozen.py --no-models`는 PASS이며, 전체 checker는 기존과 같이 checkpoint 부재 2항목만 FAIL이다. 실제 GPU·Rapfi 평가, 학습, 전체 pilot, champion·SER5 변경은 수행하지 않았다.

**검증 (6cbf802 후속)**

fixture · stand-in 모델 · mock 실행기만 사용했다. 실제 GPU 학습, 전체 pilot, H-conf 생성, 확인 Arena, checkout 전환 명령은 실행하지 않았다.

추가한 테스트(`tests/test_h7r_confirm.py`)
- anchor 짝맞춤
  - pairing 누락 · 빈 문자열 · 비문자열 · 다른 pairing을 사전 거부한다.
  - seed · pairs · suite 불일치와, 같은 pairing을 선언한 arm 사이의 모순을 사전 거부한다.
  - 정상 anchor는 통과하고, 따로 뽑은 `candidate_vs_champion`도 허용한다.
  - 커밋된 계약 세 개는 그대로 읽힌다.
  - 판정 단계의 opening 일치 검사가 남아 있다.
- 다른 commit의 측정 거부
  - 다른 commit의 timing · selection-source를 judge와 독립 `selection`이 거부하고, ledger는 바뀌지 않는다.
  - 기대 commit을 주지 않은 검증 호출을 거부한다.
  - H-conf 생성 commit(`abc123`)이 arm commit과 달라도 판정된다.
- 사전 등록
  - 빈 ledger(다른 checkout 안내), 항목 누락 · 불일치 · 손상을 거부한다. 거부해도 다시 만들지 않는다.
  - 최종 잠금 직전에 사라진 항목을 다시 만들지 않는다. 이때 `judgment` · `judge-code`도 기록되지 않는다.
- `selection-source`의 실제 예약 → 중단 · 재개 → `selection` → PROMOTE → 동일 입력 재계산이 이어진다. 다른 확인 계약의 H-conf 사용은 거부한다.
- `judge-code` identity
  - 부가 Git 필드만 다르면 충돌하지 않는다.
  - 기존 전체 Git 형식 항목은 commit이 같고 clean일 때만 받는다. 원본 바이트와 SHA는 보존한다.
  - 다른 category는 계속 엄격하게 비교한다.

결과
- R1 6개 모듈: 71개 모두 통과
- 회귀(`test_h7*`, `test_tb_v3*`, `test_h6_labels`, `test_track_b*`, `test_canonical*`): 425개 실행, 423개 통과, 2개 skip
  - skip 2개는 실제 frozen H3 / H6 checkpoint가 필요한 `test_tb_v3_net` 테스트다. 코드 실패가 아니다.
- frozen checker
  - `check_track_b_frozen.py --no-models`: PASS
  - 전체 checker: 이 컨테이너에 없는 policy / value checkpoint 2항목만 FAIL(코드 실패 아님), 나머지는 PASS다.

### 9.9 R1-PILOT-1 확정과 실행 도구 (2026-10-29)

사용자 결정에 따라 `R1-PILOT-1`을 **fixed**로 바꿨다. 제안 상태였던 값은 실제 필드로 옮겼고, 미정(null) 값을 지워 검사를 피하지 않았다. 실행 규칙이 있는 fixed pilot 계약은 `check_contract`가 모든 규칙 값을 요구한다(`hybrid.h7r_pilot.check_rules`). draft는 비워 둘 수 있다. 기존 fixed smoke 계약은 실행 계획이 없어 영향받지 않는다.

**확정한 규칙**

| 항목 | 규칙(계약 필드) |
|---|---|
| 규모 | suite마다 작은 차이 3 arm × 60쌍, S400 대조 160쌍. 총 1,020쌍 / 2,040판, `execution_plan.workers` = 6 |
| 순위 | `suite_selection.rank_by`(`h7r-suite-rank-v1`): opening마다 q_i = 세 작은 차이 arm에서 pair_score ≠ 0.5인 비율, 점추정은 opening 60개의 평균 |
| bootstrap | `history_id` 단위로 세 arm 결과를 함께 재표집. 20,000회, 하위 2.5% 분위, seed 20261029 |
| 동률 | 하한 → 점추정 순(소수 6자리). 둘 다 같으면 공동 순위 |
| 함께 보고 | move_split(같은 bootstrap), arm별 move_split · score_split 수 |
| 제외 | S400 대조와 후보 점수는 순위 통계에 넣지 않는다 |
| 분산 0 | 모든 opening의 q_i가 같으면 그렇게 표시하고 높은 확실성으로 읽지 않는다 |
| 누락 · 무효 | 보충 표본을 만들지 않는다. 계획한 유효 쌍이 부족한 arm(상대 불법 수, 오류)이 있으면 원인과 함께 미완료로 보고하고, 모든 suite의 필수 arm이 끝나기 전에는 최종 suite를 고르지 않는다 |
| 색 편향 | `colour_bias.gate` = none. arm별 tested 흑 점수 · 백 점수 · 절대 차이를 보고하고, 여러 arm을 합친 값으로 대신하지 않는다 |
| 양성 대조 | `positive_control.detection` = arm summary의 `lower_one_sided > 0`. 효과 · 구간 · 검출 여부를 그대로 보고하고 다시 실행하지 않는다 |
| 실행 순서 | O0 → O1 → O2. 각 suite에서 S400 → c8 → mask → orig |
| 시간 예산 | `execution_plan.budget`(`h7r-pilot-budget-v1`): runner 세션별 monotonic 경과시간의 합, soft 14시간. 세션 사이(중단) 시간은 빼고, 완료 arm은 포함한다. arm의 `wall_seconds`로 계산하지 않는다 |
| 예산 확인 | 각 arm 시작 전과 끝난 뒤. 도달하면 진행 중인 arm만 마치고 다음 arm은 시작하지 않는다(그래서 14시간을 넘을 수 있다) |
| 예산 도달 뒤 | 별도 승인 없이 재개하지 않는다. 재개하려면 `--budget-extension-hours H --approval TEXT`를 주고, 승인 내용이 세션 로그에 남는다. 예산을 다시 열지 못하는 연장은 기록하지 않고 거부한다 |
| 오류 | 오류나 예기치 않은 종료는 세션을 멈추고 기록한다. 자동 재시도는 없다. 운영자가 다시 실행하면 같은 commit · 계약 · JSONL로 그 arm을 재개한다 |

이 규칙은 평가 suite의 보정이다. 모델 승급이나 기력 향상 판정이 아니다.

**도구: `scripts/h7r_pilot.py`**

- `preflight`: Arena 대국과 계약 seed를 쓰지 않는다. 다음을 기록한다.
  - Git 상태, NumPy 유무와 torch 변환, CUDA 장치
  - fast5 binding과 사용 중인 threat-window 구현
  - 계약 파일(모델 네 개 × SHA 4개, suite)
  - 각 모델을 장치에 올려 한 수를 추론한 결과(국면은 중앙 한 수 뒤. 어느 suite의 opening도 아니다)
  - 모델 로딩은 기존 `h7c_budget.make_budget_agent`(→ `h7c_arena`)를 그대로 쓴다. 후보 factory는 `scripts/h7c_*` 경로로만 쓰인다.
- `run`: 계약의 arm 순서대로 `h7r_eval.play`를 부른다.
  - 끝난 arm은 기록을 다시 검증만 하고 다시 두지 않는다. 중단된 arm은 JSONL에서 재개한다.
  - `runs/` 또는 `data/` 아래 pilot 폴더에 `sessions.jsonl`을 남긴다(세션 시작 · 종료, arm 시작 · 종료, 60초 heartbeat, 예산 중단, 연장 승인).
  - 같은 폴더에는 한 runner만 돈다(잠금).
  - 모든 세션은 같은 commit과 계약 SHA여야 한다. 실행 도중 HEAD가 바뀌면 거부한다.
  - 진행 줄을 1분마다 stdout에 쓴다: 완료 pair-unit / 전체, %, 세션 · 누적 시간, ETA, arm 수.
- `report`: 모든 arm 기록을 다시 검증한 뒤 다음을 계산한다.
  - suite별 score_split 순위(완료일 때만), move_split, arm별 색 점수, 양성 대조
  - 세션 로그의 예산 합계
  - 미완료이면 `ranking: null`
- 실행 규칙이 있는 fixed pilot 계약은 `h7r_eval.py play`로 직접 둘 수 없다. runner를 거쳐야 한다.
- 세션이 강제 종료되면 그 세션의 시간은 마지막 기록까지만 센다. 최대 heartbeat 한 번(60초)만큼 적게 셀 수 있다.

**NumPy 경고에 대한 확인**

- pilot · report · holdout 평가 경로(`h7r_*`, `h7c_arena`, `h7c_budget`, E1 · tactical 검사)는 NumPy를 쓰지 않는다. `numpy`, `.numpy()`, `from_numpy`를 검색해 확인했다.
- NumPy를 쓰는 곳은 `rt30/fast_windows.py`(선택적: 없으면 순수 Python 구현)와 학습 상태 저장(`training/`)이다.
  - frozen runtime은 native 빌드가 있으면 threat-window를 native 구현(`fast3_windows`)으로 바꾼다. 따라서 NumPy 유무는 엔진 경로에 영향이 없다.
  - runtime hash에도 NumPy는 들어가지 않는다.
- NumPy가 없을 때 torch 경고는 Tensor ↔ NumPy 변환을 부를 때만 문제가 된다.
- 권장: smoke와 frozen 검사가 NumPy 없이 통과한 환경 그대로 pilot을 돌린다. `preflight`가 NumPy 상태와 사용 중인 threat-window 구현을 기록한다.
- 그래도 설치한다면 pilot을 시작하기 **전에** 한다. 설치 뒤에는 frozen 검사와 `preflight`를 다시 통과시킨다. 의존성 선언(`pyproject.toml`)에는 NumPy가 없다.

**데스크톱 명령** (`~/renju-h6`. 이 작업에서는 실행하지 않았다)

```bash
cd ~/renju-h6 && source .venv/bin/activate && export PYTHONPATH="$PWD/src:$PWD/scripts"
git status --short                      # 출력이 없어야 한다(파일이 보이면 지우지 말고 저장소 밖으로 옮긴다)
git switch feat/h7-c1 && git pull --ff-only origin feat/h7-c1
git rev-parse HEAD origin/feat/h7-c1    # 같은 SHA가 두 번
python -m rt30.native.build --if-stale && python -m rt30.vcfcore.build --if-stale
python scripts/check_track_b_frozen.py  # 전체 PASS
mkdir -p runs/h7r/R1-PILOT-1
python -u scripts/h7r_pilot.py preflight --contract docs/h7r/contracts/R1-PILOT-1.json --device cuda \
  --output runs/h7r/R1-PILOT-1/preflight.json   # "pass": true
# 본 실행: 약 10시간 예측(상한 아님). 터미널을 닫아도 계속되도록 nohup(또는 tmux)
LOG=runs/h7r/R1-PILOT-1/run-$(date +%Y%m%d-%H%M%S).log
nohup python -u scripts/h7r_pilot.py run --contract docs/h7r/contracts/R1-PILOT-1.json \
  --runs-dir runs/h7r/R1-PILOT-1 --device cuda > "$LOG" 2>&1 &
sleep 5; tail -f "$LOG"                  # 1분마다 [h7r-pilot] 진행 줄 (Ctrl-C는 tail만 끝낸다)
# runner가 끝난 뒤(또는 예산 중단 뒤)에만. 출력 파일은 덮어쓰지 않으므로 새 이름을 쓴다
python -u scripts/h7r_pilot.py report --contract docs/h7r/contracts/R1-PILOT-1.json \
  --runs-dir runs/h7r/R1-PILOT-1 --output runs/h7r/R1-PILOT-1/report-final.json
```

- 중단(오류 · 재부팅 · WSL 종료)되면 같은 `run` 명령을 다시 실행해 재개한다. 끝난 arm은 다시 두지 않는다.
- 명령을 한 번에 붙여 넣으면 `report`가 runner보다 먼저 돌 수 있다(빈 INCOMPLETE 보고). `report`는 runner가 끝난 뒤에 따로 실행한다. 540b16a의 `report`는 세션 로그를 복구 모드로 읽었다. runner가 쓰는 도중 실행하면 쓰고 있던 줄을 잘라낼 수 있었다. 이후 commit에서 읽기 전용으로 고쳤다. 고친 `report`는 pilot이 끝난 뒤 `git pull`하고 써도 된다(report는 실행 commit이 아니라 세션 · arm 기록의 commit을 대조한다).
- 예산에 도달해 멈췄다면, 재개는 별도 승인을 받은 뒤 `--budget-extension-hours`와 `--approval`을 함께 줘서 한다.
- 실행 중에는 `git pull`이나 문서 커밋으로 HEAD를 바꾸지 않는다. 바뀌면 runner가 재개를 거부한다.
- 결과와 로그는 `runs/h7r/R1-PILOT-1/`(gitignore 대상)에 남는다. 커밋하지 않는다.

**검증**

- `tests/test_h7r_pilot.py`(13개). fixture · stand-in 모델 · in-process 가짜 대국 · 가짜 시계만 쓴다.
  - 규칙: fixed는 모든 규칙 값을 요구하고 draft는 비워 둘 수 있다. 커밋된 계약 두 개를 읽는다.
  - 순위: opening 정렬 · 누락 · 중복 · 짧은 arm · 무효 arm 거부, bootstrap 재현성과 분산 0 표시, 공동 순위
  - 예산: 세션 합산, 강제 종료 세션, 연장, 로그 복구 · 손상 · commit 불일치
  - runner:
    - 순서대로 완주한 뒤 report COMPLETE, 다음 세션에서 다시 두지 않음
    - 예산 도달 시 진행 중 arm까지만 진행, 승인 없는 재개 거부, 부족한 연장은 기록 없이 거부
    - 오류 시 중단하고 재실행하면 재개, 직접 `play` 거부, `runs/` 밖 거부
    - heartbeat 기록과 진행 줄
  - `preflight`: stand-in 모델을 CPU에서 추론하고, 아무 대국도 기록하지 않는다.
- 결과
  - R1 묶음 전체 통과
  - 회귀(`test_h7*`, `test_tb_v3*`, `test_h6_labels`, `test_track_b*`, `test_canonical*`): 443개 실행, 441개 통과, 2개 skip(실제 frozen H3 / H6 checkpoint가 필요한 `test_tb_v3_net`, 코드 실패 아님)
  - `check_track_b_frozen.py --no-models`: PASS
- 실제 GPU pilot은 이 작업에서 실행하지 않았다. 이 세션은 데스크톱에 접근할 수 없다.

### 9.10 R1-PILOT-1 결과와 종료 기록 (2026-10-30)

pilot은 사전 등록한 대로 한 번 실행했고 끝났다. 결과는 데스크톱 `runs/h7r/R1-PILOT-1/`(커밋하지 않음)에 있다. 이 절의 수치는 `report-final.json`, arm summary, `size.json`의 출력에서 옮겼다. 평가 suite의 보정 기록이며 모델 선택 · 승급 · 배포가 아니다.

**실행**

- 계약 `15293ca9…`, 대국 commit `540b16a`(clean)
- 세션 1개, 누적 4.955시간(14시간 예산 미도달, 예산 중단 · 연장 0)
- 12/12 arm complete, 1,020쌍 / 2,040판
  - `report`가 12개 arm 기록 전부를 `load_arm`으로 다시 검증해 COMPLETE를 냈다. 이 검증은 코드상 모든 판의 재생, 잘린 판 거부를 포함하고, 완료 판정은 상대 불법 수가 없고 계획 쌍이 채워졌다는 조건을 포함한다. 원본 파일을 이 저장소에서 다시 계산하지는 않았다(아래 "필요한 원본").
  - `report`는 `bd893a8`에서 만들었다. `540b16a`와의 차이는 `report`가 세션 로그를 읽는 방식(읽기 전용), 문서, 테스트뿐이라 계산에는 영향이 없다.
- 비용(6 worker, 쌍당)
  - S50 작은 차이: O1 5.7–7.3초, O0 · O2 11.0–12.2초
  - S400 대조: 21.8–29.0초
  - 계획 대용값(S50 18.4초, S400 3.0배)의 약 49%였다.

**suite 순위(사전 규칙): O0 → O2 → O1**

| suite | score_split 점추정 | history 단위 하한(규칙) | 동치류 단위 하한(민감도) | move_split 점추정 · 하한 | 60개 opening의 동치류 |
|---|---:|---:|---:|---|---:|
| O0 | 0.1833 (33/180) | 0.1111 | 0.1149 | 0.4444 · 0.3389 | 43 |
| O2 | 0.1722 (31/180) | 0.1056 | 0.1111 | 0.4667 · 0.3611 | 24 |
| O1 | 0.1500 (27/180) | 0.0833 | 0.0833 | 0.4056 · 0.3056 | 60 |

- O0와 O2의 하한 차이는 1/180(arm-pair 하나)이다. 순위를 유의한 우열로 읽지 않는다.
- 대칭 민감도: 동치류 단위 bootstrap에서도 하한이 줄지 않았고 순위도 같다(2.2절의 `history_id` 단위를 지지한다). 동치류 안에서 세 arm의 쌍 점수가 모두 같은 경우는 O0 5/13, O2 6/20이었다. 쌍의 약 80%가 0.5인 자료에서는 우연으로도 흔한 수준이다.

**양성 대조(S400 대 S50, 160쌍): 세 suite 모두 검출**

| suite | 점수 | 0.5 대비 차이와 95% 구간 | pair SD |
|---|---:|---|---:|
| O0 | 0.6219 | +12.19%p [+7.50, +16.88] | 0.309 |
| O1 | 0.5844 | +8.44%p [+4.06, +12.81] | 0.284 |
| O2 | 0.6297 | +12.97%p [+8.44, +17.50] | 0.296 |

- 계획 가정(효과 7.7%p, SD 0.35)보다 효과는 크고 SD는 작았다. suite 간 효과 차이는 검정하지 않았다.

**작은 차이 비교(60쌍): 우열 미확정**

- c8 대 c5, mask 대 c5: 모든 suite에서 구간이 0을 포함한다(점추정 −0.42 ~ +1.67%p). 동등성이 입증된 것도 아니다.
- c5 대 원본: O0 −2.08, O1 −1.25, O2 −4.17%p. O2만 구간 [−7.92, −0.83]이 0을 벗어난다.
  - 참고 예시: 비교 9개가 서로 독립이고 각각 5% 오류라면, 하나 이상이 0을 벗어날 확률은 약 37%다. 실제 비교는 모델과 opening을 공유하므로 정확한 확률이 아니고, O2의 음수 결과를 무시할 근거도 아니다.
  - 이전 drift arm(기존 opening, 50쌍): c4 대 원본 −1.5%p [−6.5, +3.5], c5 대 원본 −1.0%p [−7.5, +5.5]
  - 방향은 일관되게 약간 음수이지만 확정이 아니다. 운영(canonical · web · SER5)은 원본 H3 / H6를 쓰므로 영향이 없다. 확인 계약의 champion과 R2 / R3의 출발점(계획상 c5)을 정할 때 고려한다.
- 색 편향(배제 기준 아님, arm별): O0 작은 차이 22.5–25.8%p, O1 15.0–15.8%p, O2 1.7–11.7%p, S400 6.9–17.2%p

**규모(`size`, 최소 향상량 3%p, 검출력 0.8, pair SD의 bootstrap 상한)**

| 비교 유형 | O0 필요 쌍 | O1 필요 쌍 |
|---|---:|---:|
| c8 대 c5 (SD 상한 0.246 / 0.252) | 527 | 553 |
| mask 대 c5 (0.206 / 0.216) | 369 | 409 |
| c5 대 원본 (0.212 / 0.213) | 391 | 398 |

**확인 suite에 대한 결론**(9.11에서 결정)

- O0 생성 규칙의 수순은 모두 552개다(이 저장소에서 전수 계산).
  - 이전 cycle Arena 기록(공식 · 모니터링 · drift · NMASK Arena)과 겹치는 수순: 278개
  - pilot이 쓴 수순: 190개
  - pilot만 제외하면 362개, 둘 다 제외하면 179개가 남는다.
- 결론은 다음 설계 조건에서만 성립한다.
  - 조건: pilot · 이전 평가와 정확한 수순 중복이 없고, 중복 없이 추출하며, 현재 계획 검출력을 확보한다.
  - 이 조건에서는 O0 공간이 필요한 규모를 담지 못한다. 다른 평가 설계(예: 이전 평가와의 겹침 허용, 더 작은 규모)까지 불가능하다는 뜻은 아니다.
  - 이 조건을 유지하고 O1을 쓴다. 공간 크기는 생성 규칙만으로 정해지는 설계 정보이고, 결과로 만든 기준이 아니다.
- 수순이 서로 다르다는 것은 통계적 독립성을 보장하지 않는다. 다른 국면이어도 구조가 비슷할 수 있다.
  - pilot O1의 60개가 동치류 60개였다는 것은 그 표본의 관측일 뿐이다.
  - 새 suite의 속성은 생성한 뒤 따로 기록한다(9.11의 시험 생성: 560개 → 동치류 558개, 1,040개 → 1,033개).
- 쌍 수와 정보 쌍 m은 기존 후보(c8 · mask · c5 대 원본)로 추정한 계획값이다. 새 후보가 champion과 거의 같으면 정보 쌍이 부족할 수 있다. 그때는 예정대로 KEEP / INCONCLUSIVE이며 표본을 보충하지 않는다.

**필요한 원본**(이 절의 수치를 독립적으로 다시 계산하려면, 데스크톱 `runs/h7r/R1-PILOT-1/`)

- `report-final.json`, `size.json`, `sessions.jsonl`, `preflight.json`, `run-*.log`
- `arms/*.json`(12개 arm 기록), `arms/*.jsonl`(재개 원본)

### 9.11 pilot 뒤 평가 준비: 연구 기준 · 확인 설계 초안 · opening 제외 · H-dev 초안 (2026-10-30)

확인 Arena, H-dev / H-conf 생성, R2 / R3 학습, champion 변경, SER5 배포는 하지 않았다. 확인 계약 · 확인 seed · ledger · holdout-use 예약도 만들지 않았다.

**연구 기준 모델(사용자 결정)**

- 연구 champion과 R2 출발점은 **c5**를 유지한다. 원본 H3 / H6는 **anchor**로 둔다.
- c5 대 원본의 음수 신호(9.10)는 기록해 두고 anchor 판정으로 계속 감시한다. 이번 pilot만으로 기존 승급을 취소하지 않는다.
- 운영 canonical · web · SER5는 바꾸지 않는다(이미 원본을 쓴다).

**확인 평가 설계 초안**(실제 계약은 R3 후보와 모델 identity가 나온 뒤 작성 · fixed)

| 항목 | 계획값 |
|---|---|
| suite | 새 O1(아래 제외 기능으로 생성). 9.10의 설계 조건을 유지한 선택 |
| candidate_vs_champion | 560쌍. 정보 쌍 `score_split` 최소 20쌍. 미달이면 KEEP / INCONCLUSIVE이고 표본을 보충하지 않는다 |
| anchor margin | −0.02 |
| candidate_vs_anchor · champion_vs_anchor | 각 1,040쌍(잠정). 같은 opening · pairing · seed · pairs · suite SHA(계약 검사가 강제) |
| 표본 구성 | 주 비교와 anchor 묶음은 서로 다른 opening. suite 파일 2개(주 560개, anchor 1,040개, 서로 겹치지 않음)를 만들고 각 arm이 자기 suite 전체를 쓴다. 고유 opening은 1,600개 |

- suite 파일의 크기(opening 수)와 arm의 표본 수(쌍 수)는 다른 값이다. 이 설계에서는 둘이 같도록 만든다(`pairs` = suite의 `count`).
- 가짜 candidate identity로 실행 가능한 확인 계약을 만들지 않는다.

**anchor 규모: paired difference 기준** (`h7r_eval.py anchor-size`, `ev.anchor_sizing`)

- 판정량은 같은 opening에서의 D_i = (candidate_vs_anchor pair score) − (champion_vs_anchor pair score)다. 필요한 SD는 **D_i의 SD**다.
- 개별 arm의 SD를 그대로 쓰거나 공분산을 무시한 값은 paired SD 실측이 아니다. pilot에는 candidate_vs_anchor가 없어서 D_i의 SD는 아직 가정값이다.
- 두 anchor arm 기록이 생기면 `anchor-size --contract … --arm-record A B`가 공유 opening에서 D_i의 SD를 계산한다(`ev.paired_difference_sd`).
- 조건: margin −2%p, 실제 평균 차이 0, 단측 α 0.025, 검출력 0.8, 정규 근사

| D_i의 SD 가정 | 필요 쌍 | 400쌍 검출력 | 800쌍 검출력 | 1,040쌍 검출력 |
|---|---:|---:|---:|---:|
| 0.18 | 636 | 0.603 | 0.882 | 0.948 |
| 0.205 | 825 | 0.497 | 0.788 | 0.882 |
| 0.23 | 1,039 | 0.413 | 0.691 | 0.801 |

- 1,040쌍도 SD 가정에 의존하는 계획값이고 보장된 검출력이 아니다. 실제 SD가 0.23보다 크면 0.8에 못 미친다.
- 향상 검출 규모(`size`)와 anchor 비열등 규모(`anchor-size`)는 입력과 출력이 다른 별도 명령이다. 기존 판정 규칙은 바꾸지 않았다.
- 예상 비용(6 worker, pilot O1의 60쌍 arm에서 잰 쌍당 시간. 예측일 뿐 상한이 아니다)
  - candidate_vs_champion 560쌍 × 6.1–7.3초: 약 57–68분
  - candidate_vs_anchor 1,040쌍 × 5.7–7.3초: 약 99–127분
  - champion_vs_anchor 1,040쌍 × 5.7초: 약 99분
  - 합계: 약 4.2–4.9시간

**opening 제외 기능** (`h7r_eval.py suite --exclude-from … --disjoint-from …`)

- 제외 출처
  - 실제로 둔 opening 기록(`PLAYED_FORMATS`: H7-R arm · cycle Arena · drift · NMASK · 외부 대국 · 벤치마크 기록)
  - H7-R 계약: 각 arm이 뽑은 표본(`pick`). suite 파일의 SHA와 `check_suite`를 먼저 확인한다. 이것은 검증된 표본 계획이고, 실제로 두었다는 증거는 아니다(실제 arm과의 직접 비교는 아래 검증 명령 2번).
- 거부하는 출처
  - suite 파일: 후보 목록이지 사용한 opening이 아니다. 다른 표본용으로 비워 둘 suite는 `--disjoint-from`으로 따로 준다.
  - 형식을 모르는 파일, 잘못된 수순, SHA가 맞지 않는 계약 suite
- 판단 기준: 정확한 중복은 기록에 적힌 ID가 아니라 수순에서 다시 계산한 `history_id`로 판단한다.
- 기록 항목: suite 파일의 `exclusion`에 다음을 남긴다.
  - 제외 ID 목록과 그 SHA, 제외로 건너뛴 추출 수
  - 출처별 경로 · SHA · 종류 · opening 수 · 고유 수순 수 · 수 길이
  - 출처별 history / position / class 겹침, 짧은 opening과 시작이 같은 수(shared beginnings)
  - `check_suite`가 제외 ID가 suite에 없는지와 목록의 SHA를 다시 확인한다.
- 재현성: 같은 seed와 같은 제외 집합이면 같은 suite가 나온다. 추출 순서는 같고, 제외된 것만 건너뛴다.
- 공간 부족: 부족하면 중복으로 채우지 않는다. 정해진 시도 한도에서 원인(찾은 수, 제외된 추출, 반복)을 밝히고 실패한다.
- 결과에 따라 추출 기준을 바꾸거나 특정 모델에 유리한 opening을 고르는 기능은 없다.
- position_id · class_id 겹침은 history 중복이 아니다. 기록만 하고 제외하지 않는다.
- 3수 opening과 5수 opening이 앞 세 수를 공유하는 것도 정확한 중복이 아니다. O1의 앞 세 수는 O0형 opening이고 그런 수순은 552개뿐이라, 공유가 흔하다.

**시험 생성**(이 컨테이너, 커밋하지 않음)

- 제외 출처는 세 가지다.
  - `R1-PILOT-1` 계약(12개 arm 표본 1,020개, 고유 486개)
  - `R1-SMOKE-1` 계약
  - 저장소의 이전 평가 기록 70개
- 생성 seed: 202610301(주), 202610302(anchor). 계약의 예약 seed(8781–8799)와 용도가 다르다.

| suite | opening | 고유 history / position / class | 제외 ID | `openings_sha256` |
|---|---:|---|---:|---|
| 주(560) | 560 | 560 / 560 / 558 | 898 | `3f6eb598cc31…` |
| anchor(1,040, 주와 disjoint) | 1,040 | 1,040 / 1,040 / 1,033 | 1,458 | `47a4f03255cf…` |

- 주와 anchor의 history 겹침은 0이고, class 겹침은 10이다(대칭 변형이며 중복 아님).
- 주 suite 560개 중 478개는 앞 세 수가 이전 평가의 3수 opening과 같다. 정확한 수순 중복은 모든 출처에서 0이다.
- 수순 SHA가 같다는 것은 생성 결과가 같다는 뜻일 뿐이다. 실제로 둔 표본이 계약의 추출과 같다는 증명이 아니다. 생성 중 만나지 않은 제외 ID는 달라도 같은 suite가 나올 수 있다(3수 history 차이는 5수 추출에 영향이 없다). 두 검사는 아래처럼 따로 한다.

**H-dev / H-conf**

- `docs/h7r/contracts/R1-HDEV-1.json`(**draft**)
  - actor c5, 400판, split dev
  - 예약 seed 8787. 기존 계약의 seed와 겹치지 않고, 테스트 fixture의 8795 · 8796과도 다르다.
  - cycle 1: 파생 seed 충돌 없음을 확인했다. H7-c 학습 cycle을 가리키지 않는다.
  - draft도 seed 검사에 들어가므로 seed는 하나만 차지했다.
- H-dev는 R2 / R3 개발 · 선택용이다. 선택에 한 번 쓰이면 독립 확인 근거가 아니다.
- fixed 전에 정할 것: 생성 시점, R3 후보 선택 전 라벨 접근 범위
- H-conf는 지금 만들지 않는다. 후보 선택과 확인 시점 · 접근 규칙을 정한 뒤 별도 계약 · 자료로 둔다.
- 남은 예약 seed: 8788–8794

**확인 suite 커밋(`f0b6600`)과 검증**

- 데스크톱에서 `022922e`(clean)로, pilot의 실제 arm 기록 12개를 제외 출처로 써서 만들었다.
  - `O1-confirm-main.json`: 560개, `openings_sha256` `3f6eb598cc31…`
  - `O1-confirm-anchor.json`: 1,040개, `47a4f03255cf…`
- 이 저장소에서 확인한 것
  - 두 파일 모두 `check_suite`를 통과한다. 두 파일 사이의 정확한 수순 겹침은 0이다.
  - 기록된 생성 seed로 다시 생성하면 opening 배열이 같다. 제외 출처는 계약 기반 pilot 표본, smoke 계약, 이전 기록 70개이며, 강화한 출처 검사를 모두 통과한다.
  - 파일에 기록된 제외 집합(898개 / 1,458개)은 계약 기반 시험 생성의 집합과 같다. 이는 출처 전체의 합집합이 같다는 뜻이다. 다른 출처와 겹치는 부분이 있으므로, 실제 arm 집합과 계약 추출 집합이 같다는 직접 증명은 아니다. 직접 비교는 아래 2번 명령으로 한다.
  - 파일에 기록된 데스크톱 arm 기록 12개는 이 환경에 없어 원본을 검증하지 못했다.

**데스크톱 검증 명령**(대국을 두지 않고, 추적 파일을 바꾸지 않는다)

```bash
cd ~/renju-h6 && source .venv/bin/activate && export PYTHONPATH="$PWD/src:$PWD/scripts"
if [ -n "$(git status --porcelain)" ]; then echo "STOP: 작업 트리가 clean하지 않습니다"; else git pull --ff-only origin feat/h7-c1; fi
# 1) 커밋된 suite 검사
python - <<'PY'
import json
from hybrid import h7r_openings as op
m, a = (json.load(open(f'docs/h7r/suites/O1-confirm-{n}.json')) for n in ('main', 'anchor'))
for name, b in (('main', m), ('anchor', a)):
    op.check_suite(b)
    print(name, b['count'], b['openings_sha256'][:12], b['distinct'])
print('history overlap', len({o['history_id'] for o in m['openings']} & {o['history_id'] for o in a['openings']}))
PY
# 2) 실제 arm 기록의 opening과 계약의 추출(pick)을 arm마다 직접 비교
python - <<'PY'
import json
from hybrid import h7r_openings as op
c = json.load(open('docs/h7r/contracts/R1-PILOT-1.json'))
for arm in c['arms']:
    played = [op.ids(o['moves'])['history_id'] for o in json.load(open(f"runs/h7r/R1-PILOT-1/arms/{arm['name']}.json"))['openings']]
    planned = [op.ids(o['moves'])['history_id'] for o in op.pick(json.load(open(arm['suite']['path'])), arm['pairs'], arm['seed'])]
    print(arm['name'], 'same order', played == planned, 'same set', set(played) == set(planned))
PY
# 3) (선택) 무시 경로에서 다시 생성해 수순 SHA 대조. anchor는 커밋된 main을 disjoint 출처로 쓴다
mkdir -p runs/h7r/confirm-suite-check
PRIOR=$(python - <<'PY'
import json, glob
from hybrid.h7r_openings import PLAYED_FORMATS
print(' '.join(f for f in sorted(glob.glob('docs/**/*.json', recursive=True))
               if (lambda b: isinstance(b, dict) and b.get('format') in PLAYED_FORMATS and b.get('openings'))(json.load(open(f)))))
PY
)
USED="$(ls runs/h7r/R1-PILOT-1/arms/*.json) docs/h7r/contracts/R1-SMOKE-1.json"
python -u scripts/h7r_eval.py suite --name O1 --count 560 --seed 202610301 --exclude-from $USED $PRIOR \
  --output runs/h7r/confirm-suite-check/main.json
python -u scripts/h7r_eval.py suite --name O1 --count 1040 --seed 202610302 --exclude-from $USED $PRIOR \
  --disjoint-from docs/h7r/suites/O1-confirm-main.json --output runs/h7r/confirm-suite-check/anchor.json
python -c "
import json
for n in ('main', 'anchor'):
    x, y = json.load(open(f'runs/h7r/confirm-suite-check/{n}.json')), json.load(open(f'docs/h7r/suites/O1-confirm-{n}.json'))
    print(n, 'openings', x['openings_sha256'] == y['openings_sha256'], 'exclusion', x['exclusion']['history_ids_sha256'] == y['exclusion']['history_ids_sha256'])"
```

- 비교는 수순 SHA(`openings_sha256`)와 제외 집합 SHA로 한다. 파일 전체 SHA는 비교하지 않는다. 출처 경로 · 출처 SHA · 생성 Git 정보가 달라 원래 다르다.
- 커밋된 suite를 지우고 다시 만들지 않는다. 출력 파일은 덮어쓰지 않으므로, 확인용 생성은 `runs/` 아래 새 경로에 한다.
- 이 suite 파일은 R3 후보가 나오기 전에 고정해 두는 확인용 opening 목록이다. 이것만으로는 실행 가능한 계약이 아니다.

**검토 보완(코드 검토 초안 반영, `022922e` 이후)**

- `anchor-size` 실측 모드
  - 확인 계약의 `candidate_vs_anchor`와 `champion_vs_anchor` 기록을 정확히 하나씩 요구한다. 서로 다른 두 파일이어야 하고, 순서는 상관없다.
  - 다음은 거부한다: 같은 파일 두 번, 같은 arm 두 번, anchor 이외의 arm, 상대 불법 수가 있는 표, 한 쌍 표본이 아닌 경우(seed · pairs · pairing · suite · 예산 불일치), 서로 다르거나 dirty인 코드 commit
  - 계산 방향은 항상 candidate − champion이다.
  - 실측 SD가 0이면 `ZERO_EMPIRICAL_VARIANCE`로 표시하고 실측 규모는 내지 않는다(`null`). 작은 양수로 바꿔 "1쌍" 같은 값을 만들지 않는다. 사전 SD 가정 표는 그대로 계획 근거로 남는다.
  - 기존 검토에서 재현된 "같은 일반 arm 두 개 → 필요 1쌍"은 이제 거부된다.
- 규모 수치 검사(`ev.check_planning`)
  - margin은 [−1, 0), 실제 차이는 유한하고 margin보다 크며, alpha는 (0, 0.5), 검출력은 [0.5, 1)이어야 한다.
  - SD는 유한한 양수, 쌍 수는 양의 정수여야 한다.
  - CLI뿐 아니라 하위 함수(`noninferiority_power` · `noninferiority_pairs`)에서도 검사한다.
- 제외 출처 검사
  - 실제로 둔 기록은 `openings` 목록이 있어야 하고 비어 있으면 안 된다. 없거나 null이거나 비어 있으면 원인을 밝히고 거부한다.
  - 각 opening을 `renju.Game`으로 재생한다. 중앙 시작, 반복 착수, 흑 금수, 끝난 국면을 거부한다.
  - 계약 출처는 arm이 있어야 하고(holdout 계약 거부), 각 suite가 SHA와 `check_suite`를 통과해야 한다.
  - 강화한 검사 아래에서도 저장소의 모든 실제 출처(이전 기록 70개, smoke · pilot 계약)가 통과한다. 커밋된 두 suite도 그대로 재현된다.

**데스크톱 검증 결과**(사용자가 제공한 데스크톱 출력에 근거. 이 환경에서 raw arm을 직접 검사하지 않았다)

- 실행 전 `f0686d0`으로 fast-forward했다.
- `O1-confirm-main.json` 560개, `O1-confirm-anchor.json` 1,040개. 두 suite 모두 `check_suite`를 통과했다.
- 세 ID는 의미가 다르다.
  - `history_id`(수순, 중복 판단의 단위): main 560 / anchor 1,040, 두 suite의 교집합 0
  - `position_id`(돌 배치와 둘 차례): main 560 / anchor 1,040
  - `class_id`(8대칭 동치류, 기록용): main 558 / anchor 1,033
  - 같은 동치류가 main 안에 2쌍, anchor 안에 7쌍 있다. 대칭 opening은 다른 opening이라(2.2절) 오류가 아니다. 이 수를 독립 표본 수로 해석하지도 않는다.
- pilot 12개 arm 모두, 실제 기록의 opening이 계약의 `pick` 결과와 순서 · 집합 모두 같았다.
- main과 anchor를 `runs/` 아래에서 다시 생성하자 opening SHA와 제외 집합 SHA가 커밋된 값과 같았다.

**R1 상태**

pilot 보고서와 평가 체계 준비는 완료됐으며, 확인 계약 E3는 후보 · 규모 · margin · gate 확정을 기다린다. R1 전체 완료나 기력 향상은 선언하지 않는다. E3가 아직 없는 것과 별개로 H-dev 준비를 진행한다(9.12).

### 9.12 H-dev 계약 확정(R1-HDEV-1)과 R2 경계 (2026-10-30)

이번 단계는 문서 · 계약 · 검증 보완과 명령 준비까지다. H-dev 생성, 라벨 생성, R2 / R3 학습 · 평가는 실행하지 않았다.

**확정 값**(`docs/h7r/contracts/R1-HDEV-1.json`, **fixed**)

| 항목 | 값 |
|---|---|
| 생성 시점 | 지금. R2 진단 결과를 보기 전 |
| actor | c5(기존 policy · value checkpoint와 metadata identity 그대로) |
| 판 수 · split | 400판, dev |
| run seed · cycle | 8787 · 1 |

- cycle 1은 파생 seed와 game ID(`h7c:c1:8787:i`)를 위한 값이다. H7-c 학습 cycle을 뜻하지 않는다.
- 재확인한 것
  - 커밋된 계약 4개 사이에 seed 충돌이 없다. fixed로 바꾼 자신을 중복 계약으로 보지 않는다.
  - 400판의 파생 seed가 서로 겹치지 않는다.
  - 학습용 생성기는 8787을 거부한다.
  - 생성기(`holdout_identity`)는 이 계약의 run(seed 8787, cycle 1, 400판 이하, actor c5)만 받는다.
- 이 저장소에는 생성 자료가 없다. 데스크톱에 같은 계약의 자료가 있으면 덮어쓰지 않는다. 생성기의 재개 기능이 identity(계약 SHA, 생성 commit 포함)를 대조해 이어 쓰거나 거부한다.

**라벨 접근 규칙**(계약의 `access`와 같다)

- R2 / R3 담당자는 H-dev의 기보 · 라벨 · metric을 진단, 개발, checkpoint / recipe 선택에 반복해서 쓸 수 있다.
- H-dev 행은 gradient 학습, replay, distillation target, 다른 학습 cache에 넣지 않는다(`never training data`).
- H-dev 결과는 승급 · 최종 확인 · 독립 일반화 근거로 쓰지 않는다. 한 번 선택을 이끈 H-dev는 그 선택의 독립 근거가 될 수 없다.
- H-conf · A-conf는 별도 자료 · 계약이며, 이번에 만들거나 쓰지 않는다.

**평가 전용 출처의 연결**

- 생성: `h7c_selfplay.py --holdout-contract`
  - 각 대국 identity에 `purpose: holdout`, 계약 ID · SHA, split이 들어간다.
- 라벨: `h7r_holdout.py labels`
  - fixed · committed holdout 계약만 받는다.
  - `check_holdout_games`가 0..399 전부, 중복 · 누락, 계약 SHA, seed, cycle, actor, 완료 여부를 확인한다.
- cache: `holdout.json`이 계약 → 대국 → 평가 행 → cache 파일을 SHA로 묶는다. 일반 학습용 label · cache 경로로 우회하지 않는다.
- 학습 진입은 거부된다(`h7c_labels.check_games`의 holdout 표시 확인, 표시를 지워도 예약 seed로 확인).
- 계약의 `stats` 블록은 계약 형식의 Arena 설정(opening-pair bootstrap)이다. H-dev WMSE의 판독 기준이 아니다. R2는 R2 계약에서 정할 원본 게임 단위 bootstrap으로 읽는다.

**R2와의 경계**(이번에 실행하지 않음, 다음 R2 계약에서 정한다)

- 실행하지 않은 것
  - 기존 모델의 H-dev WMSE 비교
  - 같은 recipe의 seed 3개 재학습
  - D1–D5, H-conf, 확인 suite를 쓰는 진단 대국, E3 판정, champion 교체
- R2-0 안에서도 둘을 구분한다.
  - 기존 모델 재측정(c5 / c6 / c7 / c8 / mask, 원본 포함): 평가만 한다.
  - seed 3개로 변동을 재는 단계: 새 학습이 필요하다. 이번 승인에서 제외한다.
- R2 계약에 실험 전에 적을 것
  - 비교 대상(기존 후보와 원본)
  - metric 정의와 분모(raw / D4, 전체 / network)
  - 게임 단위 쌍 차이 bootstrap
  - 학습 seed 3개와 변동 측정 방법
  - ε의 산출식 · 단위 · 적용 방법
  - D1–D5의 판단 기준과 seed 일치 조건
- ε 순서: 산출 방법은 보정 결과를 보기 전에 고정하고, 수치는 보정 실험 뒤에 정한다.
- 학습 seed 3개로는 변동 추정이 거칠다. 안정적인 분산이나 검출력을 보장하지 않는다.

### 9.13 H-dev 생성 명령 (Ubuntu, `~/renju-h6`, 새 터미널에서 실행 가능)

예상 시간은 기존 측정에서 계산한 추정이다.
- self-play: 6 worker 처리량 측정(`docs/h7c/h7c1_throughput.json`, 약 514판/시간)으로 400판 약 45–50분
- 라벨 · cache: 이 규모의 측정 기록이 없다. 수 분에서 수십 분으로 보고, 실제 시간을 기록한다.

각 블록은 `bash <<'SH' … SH` 하위 셸에서 `set -euo pipefail`로 실행한다. 실패하면 그 블록의 이후 단계가 실행되지 않고 터미널은 닫히지 않는다. 출력이 이미 있으면 삭제하거나 덮어쓰지 않고 멈춘다.

**A. 사전 검사**(수 분)

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
[ "$(git rev-parse --abbrev-ref HEAD)" = feat/h7-c1 ] || { echo "STOP: feat/h7-c1 브랜치가 아닙니다"; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "STOP: 작업 트리가 clean하지 않습니다"; exit 1; }
git pull --ff-only origin feat/h7-c1
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/feat/h7-c1)" ] || { echo "STOP: 원격과 다릅니다"; exit 1; }
python -m rt30.native.build --if-stale
python -m rt30.vcfcore.build --if-stale
python scripts/check_track_b_frozen.py
python - <<'PY'
from pathlib import Path
from hybrid.h7r_eval import load_contract
from hybrid.h7c_nmask import committed
root, path = Path.cwd(), Path('docs/h7r/contracts/R1-HDEV-1.json')
contract, sha = load_contract(path, root, check_files=True)   # c5 checkpoint · metadata SHA
assert contract['status'] == 'fixed' and committed(path, root), 'fixed, committed contract required'
h = contract['holdout']
assert (h['seed'], h['cycle'], h['games'], h['actor'], h['split']) == (8787, 1, 400, 'c5', 'dev')
print('PASS contract', sha)
PY
mkdir -p runs/h7r/R1-HDEV-1
[ ! -e runs/h7r/R1-HDEV-1/labels.json ] || { echo "STOP: 라벨이 이미 있습니다(이미 생성된 H-dev)"; exit 1; }
echo "사전 검사 완료: $(git rev-parse --short HEAD)"
SH
```

**B. c5 self-play 400판**(백그라운드, 약 45–50분)

```bash
cd ~/renju-h6 && source .venv/bin/activate && export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
STAMP=$(date +%Y%m%d-%H%M%S)
nohup python -u scripts/h7c_selfplay.py --cycle 1 --run-seed 8787 --games 400 --workers 6 \
  --policy-checkpoint runs/h3_policy_64x4/best.pt --value-checkpoint runs/h7c/c5/value/best.pt --device cuda \
  --holdout-contract docs/h7r/contracts/R1-HDEV-1.json --jsonl runs/h7r/R1-HDEV-1/games.jsonl \
  --output runs/h7r/R1-HDEV-1/selfplay-summary-$STAMP.json > runs/h7r/R1-HDEV-1/selfplay-$STAMP.log 2>&1 &
sleep 5; tail -f runs/h7r/R1-HDEV-1/selfplay-$STAMP.log   # 1분마다 [h7c-selfplay] 진행 줄. Ctrl-C는 tail만 끝낸다
```

- 재개: 중단되면 **같은 commit에서** B를 그대로 다시 실행한다.
  - 생성기가 `games.jsonl`의 identity(계약 SHA, 생성 commit 포함)를 대조하고 끝난 판을 건너뛴다.
  - 잘린 마지막 줄은 그 판만 다시 둔다.
  - 요약 파일은 실행마다 새 이름이라 덮어쓰지 않는다.
- 실행 중에는 `git pull`이나 커밋으로 HEAD를 바꾸지 않는다. 바뀌면 재개가 거부된다.

**C. 완료 검증 · 평가 전용 라벨과 cache · SHA manifest**(self-play가 끝난 뒤)

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
O=runs/h7r/R1-HDEV-1
C=docs/h7r/contracts/R1-HDEV-1.json
for p in "$O/rows.jsonl.gz" "$O/cache" "$O/labels.json"; do [ ! -e "$p" ] || { echo "STOP: $p 이미 있음"; exit 1; }; done
python - <<'PY'
from pathlib import Path
import h7c_labels as lab, h7r_holdout as ho
from hybrid.h7r_eval import load_contract
contract, sha = load_contract(Path('docs/h7r/contracts/R1-HDEV-1.json'), Path.cwd(), check_files=False)
games = lab.read_games([Path('runs/h7r/R1-HDEV-1/games.jsonl')])
first = ho.check_holdout_games(games, contract, sha)   # 0..399 전부, 중복 · 누락, 계약 SHA, seed, cycle, actor
assert first['models'] == contract['models']['c5']['identity']
print('PASS games', len(games), 'commit', first['generator_git']['commit'])
PY
python -u scripts/h7r_holdout.py labels --contract "$C" --games "$O/games.jsonl" \
  --value-output "$O/rows.jsonl.gz" --cache-dir "$O/cache" --workers 6 --output "$O/labels.json"
python - <<'PY'
import json
from pathlib import Path
import h7r_holdout as ho
from hybrid.h7r_eval import file_sha256
hc, hsha, games, binding = ho.checked_cache(Path('runs/h7r/R1-HDEV-1/cache'))   # 계약 → 대국 → 행 → cache
labels = json.loads(Path('runs/h7r/R1-HDEV-1/labels.json').read_text())
manifest = {'format': 'h7r-hdev-manifest-v1', 'contract_id': hc['contract_id'], 'contract_sha256': hsha,
            'games': len(games), 'generator_commit': games[0]['generator_git']['commit'],
            'games_sha256': labels['games_sha256'], 'rows_sha256': labels['rows_sha256'],
            'cache_labels_sha256': labels['cache_labels_sha256'], 'cache_splits': labels['cache_splits'],
            'cache_binding_sha256': file_sha256('runs/h7r/R1-HDEV-1/cache/holdout.json'),
            'labels_manifest_sha256': file_sha256('runs/h7r/R1-HDEV-1/labels.json'),
            'labels_commit': labels['git']['commit'], 'summary': labels['summary'],
            'note': 'hashes and counts only: the games, rows and cache stay under runs/ (never committed)'}
out = Path('docs/h7r/hdev/R1-HDEV-1-manifest.json')
assert not out.exists(), f'{out} exists'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=1) + '\n')
print('PASS cache binding; manifest', out)
PY
( cd "$O" && find . -type f ! -name 'SHA256SUMS' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
mkdir -p ~/renju-h6-backup
tar -czf ~/renju-h6-backup/R1-HDEV-1-$(date +%Y%m%d-%H%M%S).tar.gz "$O" "$C"
echo "완료: $O, docs/h7r/hdev/R1-HDEV-1-manifest.json, ~/renju-h6-backup/"
SH
```

- 라벨 단계는 재개 기능이 없고 기존 출력을 거부한다. 중간에 실패하면 남은 `rows.jsonl.gz` · `cache`를 지우지 말고 이름을 바꿔 보관한 뒤(예: `cache.failed-<시각>`) C를 다시 실행한다.

**D. 문서 산출물만 커밋**(해시와 개수만 담긴 manifest. 대국 · 행 · cache는 커밋하지 않는다)

```bash
cd ~/renju-h6
git status --short                       # docs/h7r/hdev/R1-HDEV-1-manifest.json 하나만 보여야 한다
git add docs/h7r/hdev/R1-HDEV-1-manifest.json
git commit -m "H7-R: R1-HDEV-1 generated (manifest only)"
git push origin feat/h7-c1
```

### 9.14 H-dev 생성 결과와 R2 계약 초안 (2026-10-30)

**H-dev 생성 기록**(`0065564`, `docs/h7r/hdev/R1-HDEV-1-manifest.json`)

- self-play: 400 / 400판, 39.3분이 걸렸다. 결과는 흑 211 / 백 174 / 무 15, 17,164수다.
- 생성 commit과 라벨 commit은 모두 `37dec66`이다(계약 확정 커밋). 계약 SHA `348fecda98c4…`가 manifest와 같다.
- 평가 cache는 396판, 15,985행이다.
  - outcome 13,805행: weight 0.1, weight 몫 38.8%
  - proof 2,180행: weight 1.0, weight 몫 61.2%
  - train / val은 0행이다.
- 4판은 benchmark probe 국면에 도달해 기존 라벨 규칙(`probe_game_min_ply` 4)으로 대국 전체가 빠졌다. 생성 실패가 아니다. 400판 전부 `check_holdout_games`를 통과했다.
- 무승부 15판(3.8%)에서 나온 t=0 outcome 행이 3,080개로 outcome 행의 22%다. 행 단위로 평균하면 이 몇 판이 결과를 좌우하므로, R2 주 지표는 게임 균등 평균으로 제안한다.

**R2 계약 초안**: `docs/h7r/r2/R2-DIAG-1-draft.md`(draft, 실행 불가)

- 이번 단계에서 하지 않은 것: R2-0A 평가, R2-0B 학습, 평가 도구 구현.
- 초안의 주요 제안
  - 주 지표: D4 / network의 게임 균등 손실. 구성 지표 세 개를 필수로 함께 보고한다.
  - bootstrap: 396게임 공통 재표집 20,000회, seed 20261030
  - ε = max(`c8:c5` 구간 반폭, 학습 seed 3개의 범위). 경험적 개발용 문턱이며 통계적 MDE가 아니다.
  - 학습 seed: 31001 / 31002 / 31003. seed 20261025 재현 대조 1회를 따로 둔다.
  - 개선 판정: 같은 seed의 기준 recipe와 비교한다.
- fixed로 바꾸기 전에 필요한 것
  - 사용자가 초안 §10의 선택 다섯 가지를 확정한다.
  - R2 평가 도구를 구현하고 테스트한다. 기존 `h7r_holdout.py eval`의 출력 의미는 바꾸지 않는다.

**2차 검토 반영과 R2-0A 도구**(같은 날)

- 초안에 반영한 것(`docs/h7r/r2/R2-DIAG-1-draft.md` §12)
  - seed가 replay 추출까지 바꾼다는 것과 ε_seed의 의미
  - 같은 seed가 같은 배정을 보장하는 조건(크기가 모두 같을 때)
  - 하위 실험별 직접 control
  - 재현 대조의 비교 항목, 판정, 중단 규칙
  - checkpoint에서 재계산하는 trunk hash
  - `mask:c8`이 선택 지표까지 다른 recipe 비교라는 점
- 구현: `scripts/h7r_r2.py eval`, `src/hybrid/h7r_r2.py`, 계약 draft `docs/h7r/r2/R2-DIAG-1.contract.json`, 테스트 `tests/test_h7r_r2.py`
  - 계약이 draft이므로 실행을 거부한다. R2-0A를 실행하려면 사용자가 §10의 선택을 확정하고, 계약을 `fixed` + `stages.R2-0A.approved: true`로 바꿔 커밋해야 한다. (§9.15에서 확정)
  - R2-0B 학습 도구와 재현 허용 오차는 아직 없다. 그 전에 계약과 도구를 추가한다.

### 9.15 R2-0A 계약 확정과 실행 명령 (2026-10-30)

**확정**(`docs/h7r/r2/R2-DIAG-1.contract.json`, 사용자 승인)

| 항목 | 값 |
|---|---|
| status | `fixed` |
| R2-0A | 승인: 기존 모델 6개를 H-dev에서 평가만 한다 |
| R2-0B | 미승인 |
| ε | null |
| 재현 허용 오차 | null |

- 주 지표·구성 지표 3개, 모델·비교 6개, bootstrap 20,000회 / seed 20261030, ply 구간은 초안대로다.
- 승인하지 않은 것: R2-0B 학습, ε 확정, D1–D5, 새 대국, Rapfi, H-conf / A-conf, champion 교체
- 구현 보완(자세한 내용은 초안 §13)
  - `preflight`: 검사만 한다.
  - `eval`: 모델 6개를 모두 검사한 뒤에 평가를 시작한다.
  - 재현 판정이 state hash 일치만으로 조기 통과하던 경로를 막았다.
- 원본 H6 v3 metadata의 `frozen_state_sha256` 선언은 이 저장소에서 확인할 수 없다(파일 없음). 아래 A가 데스크톱에서 먼저 확인한다. 선언이 없으면 멈추고, 호환 규칙을 정한 뒤 다시 실행한다.

이 세션 환경에는 H-dev와 모델 파일이 없어서 **R2-0A를 실행하지 않았다**. 아래 명령은 데스크톱(`~/renju-h6`)용이다. 각 블록은 실패하면 멈추며, 앞 블록이 PASS일 때만 다음으로 간다. 정식 실행에는 `--allow-dirty`를 쓰지 않는다. draft 우회는 명령행 옵션 자체가 없다.

**A. 상태 확인**(브랜치, clean tree, fixed 계약과 승인, frozen 검사, 모델 파일과 선언, 출력 없음)

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
git fetch origin feat/h7-c1
git checkout feat/h7-c1
git pull --ff-only origin feat/h7-c1
[ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: 브랜치"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "STOP: 추적 파일 변경 있음"; git status --short; exit 1; }
python scripts/check_track_b_frozen.py
python - <<'PY'
import json
from pathlib import Path
from hybrid import h7r_r2 as r2
from hybrid.h7c_nmask import committed
path = Path('docs/h7r/r2/R2-DIAG-1.contract.json')
c, sha = r2.load_contract(path)
assert c['status'] == 'fixed' and committed(path), 'contract not fixed / committed'
assert c['stages']['R2-0A']['approved'] and not c['stages']['R2-0B']['approved'], 'stage approvals'
assert c['epsilon']['value'] is None
files = {'c5': 'runs/h7c/c5/value', 'c6': 'runs/h7c/c6/value', 'c7': 'runs/h7c/c7/value',
         'c8': 'runs/h7c/c8/value', 'mask': 'runs/h7c/nmask/value', 'orig': 'runs/h6_value_head_v3'}
for name, folder in files.items():
    pt, meta = Path(folder) / 'best.pt', Path(folder) / 'best.json'
    spec = c['models'][name]
    ok = (r2.file_sha256(pt) == spec['value_checkpoint_sha256']
          and r2.file_sha256(meta) == spec['value_metadata_sha256'])
    declared = json.loads(meta.read_text()).get('frozen_state_sha256')
    print(f"{name:5s} files {'PASS' if ok else 'FAIL'}  declared frozen {declared}")
    assert ok, f'{name}: file SHA differs from the contract'
    assert declared == c['frozen_state_sha256'], f'{name}: declaration missing or different: STOP'
print('PASS contract', sha[:12], '+ six model files and declarations')
PY
for p in docs/h7r/r2/R2-DIAG-1/r2_0a.json runs/h7r/R2-DIAG-1/predictions.pt; do
  [ ! -e "$p" ] || { echo "STOP: $p 이미 있음 (지우지 말고 보존)"; exit 1; }
done
echo "A PASS"
SH
```

**B. preflight**(계약 → H-dev binding → 모델 6개의 파일 SHA와 checkpoint에서 재계산한 frozen state. 아무것도 쓰지 않는다)

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
python -u scripts/h7r_r2.py preflight --contract docs/h7r/r2/R2-DIAG-1.contract.json \
  --cache runs/h7r/R1-HDEV-1/cache --device cuda \
  --model c5=runs/h7c/c5/value/best.pt,runs/h7c/c5/value/best.json \
  --model c6=runs/h7c/c6/value/best.pt,runs/h7c/c6/value/best.json \
  --model c7=runs/h7c/c7/value/best.pt,runs/h7c/c7/value/best.json \
  --model c8=runs/h7c/c8/value/best.pt,runs/h7c/c8/value/best.json \
  --model mask=runs/h7c/nmask/value/best.pt,runs/h7c/nmask/value/best.json \
  --model orig=runs/h6_value_head_v3/best.pt,runs/h6_value_head_v3/best.json
SH
```

**C. R2-0A 평가**(한 번만 실행한다. 예상 시간은 수 분이며, 단계마다 진행 줄을 출력한다. 원시 예측은 `runs/`에 남는다)

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
mkdir -p runs/h7r/R2-DIAG-1
LOG=runs/h7r/R2-DIAG-1/r2_0a-$(date +%Y%m%d-%H%M%S).log
python -u scripts/h7r_r2.py eval --contract docs/h7r/r2/R2-DIAG-1.contract.json \
  --cache runs/h7r/R1-HDEV-1/cache --device cuda \
  --model c5=runs/h7c/c5/value/best.pt,runs/h7c/c5/value/best.json \
  --model c6=runs/h7c/c6/value/best.pt,runs/h7c/c6/value/best.json \
  --model c7=runs/h7c/c7/value/best.pt,runs/h7c/c7/value/best.json \
  --model c8=runs/h7c/c8/value/best.pt,runs/h7c/c8/value/best.json \
  --model mask=runs/h7c/nmask/value/best.pt,runs/h7c/nmask/value/best.json \
  --model orig=runs/h6_value_head_v3/best.pt,runs/h6_value_head_v3/best.json \
  --predictions runs/h7r/R2-DIAG-1/predictions.pt \
  --output docs/h7r/r2/R2-DIAG-1/r2_0a.json 2>&1 | tee "$LOG"
echo "log: $LOG"
SH
```

- 실패하면 남은 파일을 지우지 않는다. 그 출력을 그대로 보고하고 다음 블록으로 가지 않는다.

**D. 출력과 예측의 연결 확인, 요약 출력**

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts"
python - <<'PY'
import json
import subprocess
from pathlib import Path
from hybrid import h7r_r2 as r2
out = json.loads(Path('docs/h7r/r2/R2-DIAG-1/r2_0a.json').read_text())
c, sha = r2.load_contract('docs/h7r/r2/R2-DIAG-1.contract.json')
head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
assert out['format'] == r2.RESULT_FORMAT and out['stage'] == 'R2-0A'
assert out['contract_sha256'] == sha and out['contract_status'] == 'fixed'
assert out['git'] == {'commit': head, 'dirty': False}, out['git']
assert out['predictions']['sha256'] == r2.file_sha256(out['predictions']['path']), 'predictions SHA'
assert all(out['hdev'][k] == c['hdev'][k] for k in r2.HDEV_SHA_KEYS) and out['hdev']['games'] == 396
assert out['epsilon']['value'] is None
print('PASS links: contract', sha[:12], 'commit', head[:12], 'predictions', out['predictions']['sha256'][:12])
print('routes', out['route_counts'], 'minutes', out['minutes'])
fmt = lambda x: 'n/a' if x is None else f'{x:+.6f}'
for spec, s in out['summary'].items():
    items = [('primary', s['primary'])] + [(f"{m['metric']['group']}/{m['metric']['aggregate']}", m)
                                           for m in s['components']]
    for label, m in items:
        ci = m.get('ci95')
        ci_text = 'n/a' if ci is None else f'[{ci[0]:+.6f}, {ci[1]:+.6f}]'
        print(f"{spec:8s} {label:24s} {fmt(m['difference'])} CI95 {ci_text}  games {m['games']}"
              f"  zero-den {m.get('zero_denominator_resamples')}  readable {m['readable']}")
    print(f"{spec:8s} reversals: {', '.join(s['reversals']) or 'none'}")
print('epsilon: null; sample part', fmt(out['epsilon']['sample']))
PY
sha256sum docs/h7r/r2/R2-DIAG-1/r2_0a.json runs/h7r/R2-DIAG-1/predictions.pt
SH
```

**E. 요약만 커밋**(원시 예측과 로그는 `runs/`에 남고 커밋하지 않는다)

```bash
cd ~/renju-h6
git status --short                       # docs/h7r/r2/R2-DIAG-1/r2_0a.json 하나만 새로 보여야 한다
git add docs/h7r/r2/R2-DIAG-1/r2_0a.json
git commit -m "H7-R R2-0A: existing value models on H-dev (R2-DIAG-1, summary only)"
git push origin feat/h7-c1
```

- D의 출력(주 지표, 구성 지표, 방향 역전, 기여 게임 수, 분모 0 회차)을 그대로 보고한다. 이 결과로 ε를 확정하거나 승급·기력 향상을 판정하지 않는다.

### 9.16 R2-0A 결과, 수정안 A1(X1 탐색 부록 · R2-0B 준비) (2026-10-30)

**R2-0A**(`docs/h7r/r2/R2-DIAG-1/r2_0a.json`, 커밋 `4944acd`): 결과 표와 읽기는 초안 §14에 있다.

- 비교 6개 모두 주 지표 구간이 0을 포함하지 않는다.
- 과거 T-val과 결과가 다른 원인은 분리되지 않았다. 평가 분포와 지표 정의(raw/D4, 전체/network, pooled/macro)도 다르다.
- c8:c5의 pooled 구간은 0을 포함한다.
- ε는 null이고, 승급·기력 판정은 하지 않는다.

**수정안 A1**(`docs/h7r/r2/R2-DIAG-1-A1.json`, fixed): 고정 계약을 바꾸지 않고 덧붙인다.

- X1(승인): 차례별 기술적 분해와 게임 단위 교차 보정 평가. 사후 탐색 분석이다.
- R2-0B(미승인)
  - 재현 허용 오차 `abs_primary` ≤ 0.0002, `max_abs_prediction` ≤ 0.001을 운영상 허용치로 고정했다.
  - 학습 4회와 ε 확정은 별도 승인을 받아야 한다.
- 도구
  - `scripts/h7r_r2_x1.py`
  - `scripts/h7r_r2_seed.py`(`train` / `reproduce`, 승인 전에는 실행 거부)
  - 계산 함수: `src/hybrid/h7r_r2.py`
- 테스트(`tests/test_h7r_r2.py`)
  - 주 지표와 같은 가중치로 분해되는지(합 = 원 손실)
  - fold 밖 보정의 정의
  - X1 실행 전체
  - stand-in 기준으로 처음부터 끝까지
    - 승인 전 거부
    - seed 20261025 재실행이 CPU에서 `BITWISE_REPRODUCED`
    - 다른 seed는 다른 배정

**X1 명령**(데스크톱 `~/renju-h6`. 블록이 실패하면 멈추고 다음으로 가지 않는다. 출력이 이미 있으면 도구가 덮어쓰지 않는다)

A. 상태 확인과 입력 SHA

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
git fetch origin feat/h7-c1
git checkout feat/h7-c1
git pull --ff-only origin feat/h7-c1
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "STOP: 추적 파일 변경 있음"; git status --short; exit 1; }
python scripts/check_track_b_frozen.py
python - <<'PY'
from pathlib import Path
from hybrid import h7r_r2 as r2
from hybrid.h7c_nmask import committed
c, sha = r2.load_contract('docs/h7r/r2/R2-DIAG-1.contract.json')
a, asha = r2.load_amendment('docs/h7r/r2/R2-DIAG-1-A1.json', c, sha)
assert committed(Path('docs/h7r/r2/R2-DIAG-1-A1.json')) and a['x1']['approved'] and not a['r2_0b']['approved']
for name, pin in a['x1']['inputs'].items():
    assert r2.file_sha256(pin['path']) == pin['sha256'], f'{name}: SHA differs'
print('PASS contract', sha[:12], 'amendment', asha[:12], 'inputs pinned')
PY
[ ! -e docs/h7r/r2/R2-DIAG-1/x1_side_offset.json ] || { echo "STOP: X1 출력이 이미 있음 (보존)"; exit 1; }
echo "A PASS"
SH
```

B. X1 실행(CPU 계산과 경로 판정, 수 분 이내)

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
LOG=runs/h7r/R2-DIAG-1/x1-$(date +%Y%m%d-%H%M%S).log
python -u scripts/h7r_r2_x1.py --contract docs/h7r/r2/R2-DIAG-1.contract.json \
  --amendment docs/h7r/r2/R2-DIAG-1-A1.json --cache runs/h7r/R1-HDEV-1/cache \
  --result docs/h7r/r2/R2-DIAG-1/r2_0a.json --predictions runs/h7r/R2-DIAG-1/predictions.pt \
  --output docs/h7r/r2/R2-DIAG-1/x1_side_offset.json 2>&1 | tee "$LOG"
echo "log: $LOG"
SH
```

C. 연결 확인과 요약 출력

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts"
python - <<'PY'
import json
import subprocess
from pathlib import Path
from hybrid import h7r_r2 as r2
out = json.loads(Path('docs/h7r/r2/R2-DIAG-1/x1_side_offset.json').read_text())
c, sha = r2.load_contract('docs/h7r/r2/R2-DIAG-1.contract.json')
a, asha = r2.load_amendment('docs/h7r/r2/R2-DIAG-1-A1.json', c, sha)
head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
assert out['format'] == r2.X1_FORMAT and out['exploratory'] is True
assert out['contract_sha256'] == sha and out['amendment_sha256'] == asha
assert out['git'] == {'commit': head, 'dirty': False}, out['git']
assert out['inputs']['result_sha256'] == a['x1']['inputs']['result']['sha256']
assert out['inputs']['predictions_sha256'] == a['x1']['inputs']['predictions']['sha256']
print('PASS links: contract', sha[:12], 'amendment', asha[:12], 'commit', head[:12], 'folds', out['folds']['games'])
fmt = lambda x: 'n/a' if x is None else f'{x:+.6f}'
for p in ('d4', 'raw'):
    for g in ('network', 'network/outcome', 'network/proof'):
        print(f'== {p} {g}')
        for name, m in out['models'].items():
            cell = m[p][g]
            d = cell['descriptive']
            sides = '  '.join(f"{s}: share {d['sides'][s]['share']:.3f} mu {fmt(d['sides'][s]['mean_residual'])}"
                              for s in ('black', 'white'))
            print(f"  {name:5s} L {d['loss']:.6f} bias {d['bias']:.6f} var {d['variance']:.6f} "
                  f"L_cf {cell['crossfit']['loss']:.6f}  {sides}")
        for spec, comp in out['comparisons'].items():
            cell = comp[p][g]
            parts = []
            for key in ('loss', 'bias', 'variance', 'crossfit'):
                ci = cell[key]['ci95']
                ci_text = 'n/a' if ci is None else f'[{ci[0]:+.6f}, {ci[1]:+.6f}]'
                parts.append(f"{key} {fmt(cell[key]['difference'])} {ci_text} z{cell[key]['zero_denominator_resamples']}")
            print(f'  {spec:8s} ' + '  '.join(parts))
PY
sha256sum docs/h7r/r2/R2-DIAG-1/x1_side_offset.json
SH
```

D. 요약만 커밋

```bash
cd ~/renju-h6
git status --short                       # docs/h7r/r2/R2-DIAG-1/x1_side_offset.json 하나만 새로 보여야 한다
git add docs/h7r/r2/R2-DIAG-1/x1_side_offset.json
git commit -m "H7-R R2-0A X1: side offset decomposition (exploratory, summary only)"
git push origin feat/h7-c1
```

- X1은 사후 탐색이다. 결과가 한 방향을 보여도 D3을 바로 정하지 않는다. 원 지표, 교차 보정 지표, 종류별 결과, R2-0B의 seed 변동을 모두 모은 뒤 다음 하위 계약을 제안한다.
- R2-0B 명령은 승인할 때 준다. `scripts/h7r_r2_seed.py`는 지금 실행해도 "R2-0B is not approved"로 멈춘다.

### 9.17 X1 읽기 정정과 R2-0B 승인 · 실행 명령 (2026-10-31)

**X1 읽기**(초안 §15)

- X1이 배제한 것은 "차례별 상수 하나"로 network 차이를 설명하는 경우뿐이다.
  - 종류·ply·결과에 따른 보정 오차는 배제하지 못했다.
  - network 잔차 변동이 줄었다고 판별력 개선으로 읽지 않는다.
  - 교차 보정 뒤 손실이 커진 원인(모형 부적합, fold 추정 변동)은 가설로 남긴다.
  - proof의 음의 평균 잔차를 포화 부족으로 단정하지 않는다.
  - outcome에서 평균 잔차 감소는 손실 차이의 일부를 설명할 뿐 인과 근거가 아니다.
- 확인 범위
  - 데스크톱 실행: 입력 재해시와 행 정렬 검사를 했다.
  - 이 저장소의 검토: 커밋된 요약 기록과 SHA 연결만 확인했다.
- X1 결과와 판독은 바꾸지 않았다. X2는 실행하지 않는다.

**R2-0B 승인**(`docs/h7r/r2/R2-DIAG-1-A1.json`의 `r2_0b.approved: true`)

- 그대로 유지하는 것
  - recipe(c8 비 mask: parent c5, c8 cycle 자료·split·replay, seed만 바꾼 config, 기존 step·optimizer·augmentation·T-val 선택·H6 guard)
  - 학습 seed 31001 / 31002 / 31003, 재현 seed 20261025
  - 재현 허용 오차(`abs_primary` ≤ 0.0002, `max_abs_prediction` ≤ 0.001, raw와 D4 전체)
  - ε 산출식
- H-dev는 학습, early stopping, checkpoint 선택에 쓰지 않는다.
- 부모 계약과 R2-0A·X1 기록은 바꾸지 않았다.
- 도구: `scripts/h7r_r2_seed.py`(`train`, `reproduce`, `epsilon`). 순서와 중단 조건은 도구가 강제한다(초안 §15).
- 승인하지 않은 것: D1–D5, 새 대국, Rapfi, X2, H-conf / A-conf, champion 교체, V9·배포 변경

이 세션 환경에는 H-dev, 모델, cycle cache가 없어서 **R2-0B를 실행하지 않았다**. 아래는 데스크톱(`~/renju-h6`)용 명령이다.
- 블록은 A부터 G까지 순서대로 실행한다. 실패하면 멈추고 다음 블록으로 가지 않는다.
- 블록 A가 기록한 HEAD와 다르면 모든 블록이 멈춘다. 실행 중 `git pull`이나 checkout을 하지 않는다.
- 출력 폴더와 파일이 이미 있으면 도구가 거부한다. 지우지 말고 보존한 뒤 보고한다.

**A. 사전 검사**(clean commit, 승인 상태, 입력 파일, 출력 없음, HEAD 기록)

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
git fetch origin feat/h7-c1
git checkout feat/h7-c1
git pull --ff-only origin feat/h7-c1
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "STOP: 추적 파일 변경 있음"; git status --short; exit 1; }
python scripts/check_track_b_frozen.py
R=runs/h7r/R2-DIAG-1/r2_0b; D=docs/h7r/r2/R2-DIAG-1/r2_0b
for p in "$R" "$D"; do [ ! -e "$p" ] || { echo "STOP: $p 이미 있음 (보존)"; exit 1; }; done
python - <<'PY'
import json
from pathlib import Path
from hybrid import h7r_r2 as r2
from hybrid.h7c_nmask import committed
c, sha = r2.load_contract('docs/h7r/r2/R2-DIAG-1.contract.json')
a, asha = r2.load_amendment('docs/h7r/r2/R2-DIAG-1-A1.json', c, sha)
assert committed(Path('docs/h7r/r2/R2-DIAG-1-A1.json')) and a['r2_0b']['approved'], 'R2-0B not approved / committed'
ref = json.loads(Path(a['r2_0b']['reference_record']['path']).read_text())
assert r2.file_sha256('runs/h7c/c5/value/best.pt') == ref['inputs']['parent_checkpoint_sha256'], 'parent c5'
for cache, key in (('runs/h7c/c8/value_cache', 'cycle_labels_sha256'),
                   ('data/tactical/h6/value_cache_v3', 'replay_labels_sha256')):
    m = json.loads((Path(cache) / 'manifest.json').read_text())
    assert m['labels_sha256'] == ref['inputs'][key], cache
assert Path('runs/h7c/c8/value_rows.jsonl.gz').is_file() and Path('runs/h7r/R1-HDEV-1/cache/holdout.json').is_file()
print('PASS contract', sha[:12], 'amendment', asha[:12], 'tolerance', a['r2_0b']['reproduction']['tolerance'],
      'seeds', a['r2_0b']['training_seeds'], 'reproduction seed', a['r2_0b']['reproduction_seed'])
PY
mkdir -p "$R/logs"
git rev-parse HEAD > "$R/head.txt"
echo "A PASS: HEAD $(cat "$R/head.txt")"
SH
```

**B. ① 재현 학습**(seed 20261025. c8 recipe라 수 분 걸린다)

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
R=runs/h7r/R2-DIAG-1/r2_0b
[ "$(git rev-parse HEAD)" = "$(cat "$R/head.txt")" ] || { echo "STOP: HEAD 변경"; exit 1; }
python -u scripts/h7r_r2_seed.py train --contract docs/h7r/r2/R2-DIAG-1.contract.json \
  --amendment docs/h7r/r2/R2-DIAG-1-A1.json --seed 20261025 \
  --cycle-cache runs/h7c/c8/value_cache --cycle-rows runs/h7c/c8/value_rows.jsonl.gz \
  --replay-cache data/tactical/h6/value_cache_v3 --parent runs/h7c/c5/value/best.pt --device cuda \
  --out-dir "$R/seed-20261025" 2>&1 | tee "$R/logs/train-20261025.log"
echo "B PASS"
SH
```

**C. ② 재현 판정**(`NOT_REPRODUCED`면 여기서 멈춘다. 결과를 그대로 보고하고 D 이후는 실행하지 않는다)

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
R=runs/h7r/R2-DIAG-1/r2_0b; D=docs/h7r/r2/R2-DIAG-1/r2_0b
[ "$(git rev-parse HEAD)" = "$(cat "$R/head.txt")" ] || { echo "STOP: HEAD 변경"; exit 1; }
mkdir -p "$D"
python -u scripts/h7r_r2_seed.py reproduce --contract docs/h7r/r2/R2-DIAG-1.contract.json \
  --amendment docs/h7r/r2/R2-DIAG-1-A1.json --cache runs/h7r/R1-HDEV-1/cache \
  --reference-model runs/h7c/c8/value/best.pt,runs/h7c/c8/value/best.json \
  --run-dir "$R/seed-20261025" --device cuda --output "$D/reproduction.json" 2>&1 | tee "$R/logs/reproduce.log" \
  || { echo "STOP: 재현 판정 실패 (NOT_REPRODUCED 또는 입력 오류). seed 학습과 ε로 진행하지 않는다"; exit 1; }
echo "C PASS"
SH
```

**D. ③ seed 세 개 학습**(재현 기록이 통과했을 때만 실행된다. 하나라도 실패하면 멈춘다. 실패한 seed를 빼거나 대체하지 않는다)

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
R=runs/h7r/R2-DIAG-1/r2_0b; D=docs/h7r/r2/R2-DIAG-1/r2_0b
[ "$(git rev-parse HEAD)" = "$(cat "$R/head.txt")" ] || { echo "STOP: HEAD 변경"; exit 1; }
for SEED in 31001 31002 31003; do
  python -u scripts/h7r_r2_seed.py train --contract docs/h7r/r2/R2-DIAG-1.contract.json \
    --amendment docs/h7r/r2/R2-DIAG-1-A1.json --seed "$SEED" --reproduction "$D/reproduction.json" \
    --cycle-cache runs/h7c/c8/value_cache --cycle-rows runs/h7c/c8/value_rows.jsonl.gz \
    --replay-cache data/tactical/h6/value_cache_v3 --parent runs/h7c/c5/value/best.pt --device cuda \
    --out-dir "$R/seed-$SEED" 2>&1 | tee "$R/logs/train-$SEED.log" \
    || { echo "STOP: seed $SEED 실패 (기록은 $R/seed-$SEED 에 보존). 다음 seed와 ε로 진행하지 않는다"; exit 1; }
done
echo "D PASS"
SH
```

**E. ④ seed 모델 H-dev 평가와 ⑤ ε**

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts" PYTHONUNBUFFERED=1
R=runs/h7r/R2-DIAG-1/r2_0b; D=docs/h7r/r2/R2-DIAG-1/r2_0b
[ "$(git rev-parse HEAD)" = "$(cat "$R/head.txt")" ] || { echo "STOP: HEAD 변경"; exit 1; }
python -u scripts/h7r_r2_seed.py epsilon --contract docs/h7r/r2/R2-DIAG-1.contract.json \
  --amendment docs/h7r/r2/R2-DIAG-1-A1.json --cache runs/h7r/R1-HDEV-1/cache \
  --result docs/h7r/r2/R2-DIAG-1/r2_0a.json --reproduction "$D/reproduction.json" \
  --run-dir "$R/seed-31001" --run-dir "$R/seed-31002" --run-dir "$R/seed-31003" \
  --model c5=runs/h7c/c5/value/best.pt,runs/h7c/c5/value/best.json \
  --model c8=runs/h7c/c8/value/best.pt,runs/h7c/c8/value/best.json --device cuda \
  --predictions "$R/seed_predictions.pt" --output "$D/epsilon.json" 2>&1 | tee "$R/logs/epsilon.log" \
  || { echo "STOP: ε 계산 실패 또는 미완료"; exit 1; }
echo "E PASS"
SH
```

**F. 상태 확인, 원본 보존, manifest, 기록 사본**(checkpoint, tensor, 행별 예측, 로그는 `runs/`에 남는다)

```bash
cd ~/renju-h6 && bash <<'SH'
set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/scripts"
R=runs/h7r/R2-DIAG-1/r2_0b; D=docs/h7r/r2/R2-DIAG-1/r2_0b
[ "$(git rev-parse HEAD)" = "$(cat "$R/head.txt")" ] || { echo "STOP: HEAD 변경"; exit 1; }
( cd "$R" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
python - <<'PY'
import json
import shutil
from pathlib import Path
from hybrid import h7r_r2 as r2
R, D = Path('runs/h7r/R2-DIAG-1/r2_0b'), Path('docs/h7r/r2/R2-DIAG-1/r2_0b')
rep = json.loads((D / 'reproduction.json').read_text())
eps = json.loads((D / 'epsilon.json').read_text())
assert rep['verdict']['verdict'] in r2.REPRODUCED and eps['epsilon']['status'] == 'COMPUTED'
assert eps['reproduction_sha256'] == r2.file_sha256(D / 'reproduction.json')
assert eps['predictions']['sha256'] == r2.file_sha256(eps['predictions']['path'])
head = (R / 'head.txt').read_text().strip()
runs = {}
for seed in (20261025, 31001, 31002, 31003):
    run = R / f'seed-{seed}'
    rec = json.loads((run / 'r2_train.json').read_text())
    assert rec['git'] == {'commit': head, 'dirty': False} and rec['verdict'] == 'CANDIDATE', run
    copy = D / 'runs' / f'seed-{seed}'
    copy.mkdir(parents=True, exist_ok=False)
    for name in ('r2_train.json', 'record.json', 'config.json'):
        shutil.copy2(run / name, copy / name)
    runs[seed] = {name: r2.file_sha256(run / name) for name in ('best.pt', 'best.json', 'r2_train.json', 'record.json',
                                                                'config.json')}
manifest = {'format': 'h7r-r2-0b-manifest-v1', 'head': head, 'runs': runs,
            'reproduction_sha256': r2.file_sha256(D / 'reproduction.json'),
            'epsilon_sha256': r2.file_sha256(D / 'epsilon.json'),
            'seed_predictions': eps['predictions'], 'runs_sha256sums': r2.file_sha256(R / 'SHA256SUMS'),
            'note': 'hashes only: checkpoints, tensors, row predictions and logs stay under runs/ (never committed)'}
(D / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
e = eps['epsilon']
print('reproduction', rep['verdict'], 'state equal', rep['state_sha256']['reference'] == rep['state_sha256']['rerun'],
      'max|dv|', {p: v['max_abs'] for p, v in rep['predictions'].items()}, '|dL|', rep['primary']['abs_difference'])
print('epsilon', e['status'], e['value'], 'sample', e['sample'], 'seed', e['seed'], 'values', e['seed_values'],
      'mean', e['seed_mean'], 'sd', e['seed_sd'])
for spec, r in eps['r2_0a_reread'].items():
    print(f"  R2-0A {spec:8s} {r['difference']:+.6f} {r['ci95']} -> {r['reading']}")
for spec, cells in eps['paired'].items():
    p = cells[0]
    print(f"  {spec:14s} primary {p['difference']:+.6f} {p['ci95']} beyond_epsilon {p['beyond_epsilon']}")
PY
sha256sum "$D"/*.json
SH
```

**G. 집계와 기록 사본만 커밋**

```bash
cd ~/renju-h6
git status --short                       # docs/h7r/r2/R2-DIAG-1/r2_0b/ 아래 파일만 새로 보여야 한다
git add docs/h7r/r2/R2-DIAG-1/r2_0b
git commit -m "H7-R R2-0B: reproduction check, three seed runs, epsilon (records and hashes only)"
git push origin feat/h7-c1
```

- C, D, E의 실패 메시지와 F의 출력을 그대로 보고한다.
- ε는 경험적 개발용 문턱이다. 통계적 MDE, 기력 향상, 승급 기준이 아니다.
- R2-0A 비교의 `BEYOND_EPSILON`은 단일 실행에 대한 기술적 표시다. `DEVELOPMENT_SIGNAL`은 세 seed와 직접 control로만 판정한다.

### 9.18 R2-0B 기록 검증 보완 (2026-10-31)

- seed 학습과 ε 계산은 재현 기록의 `checks`를 그대로 신뢰하지 않는다. 재현에 사용한 c8 checkpoint·metadata와 H-dev 경로를 `verification_inputs`로 기록하고, 실제 모델과 cache에서 재현 비교를 읽기 전용으로 다시 계산한다. checks·상세 예측 오차·주 지표·state·배정·선택 step·곡선·판정이 원본과 맞아야 진행한다.
- 재계산의 부동소수점 값 대조에는 abs 1e-7 / rel 1e-6 반올림 허용만 적용한다. SHA·배정·step·판정은 정확히 일치해야 한다. 수정안의 재현 허용 오차(0.001 / 0.0002)는 바꾸지 않았고, 새로 계산한 판정도 통과해야 한다.
- seed별로 `config.json`과 `record.json` SHA를 다시 확인한다. seed 외 설정, parent·cycle·replay 출처, 배정 지문, 선택 step·적격성·곡선, metadata와 실제 checkpoint state도 함께 대조한다. ε 결과에는 학습 기록·config의 SHA를 추가한다.
- 이 검사는 학습 기록과 산출물의 일관성 확인이다. 악의적으로 모든 파일과 기록을 함께 다시 만드는 행위에 대한 서명·인증 체계는 아니다.
- §9.17 A–G 명령은 그대로 사용한다. seed 진입과 ε 계산 때 두 모델의 H-dev 추론을 다시 하므로 검증 시간이 추가된다. 실제 R2-0B 학습은 이번 보완에서 실행하지 않았다.
- `verification_inputs`가 없는 이전 재현 기록은 자동 보정하지 않고 거부한다. 기존 파일은 보존하고, 필요하면 새 출력 경로로 `reproduce`를 실행한다. 이미 진행 중인 실행의 HEAD를 바꾸지 않는다.

### 9.18 R2-0B 결과와 D4 baseline 초안 (2026-10-31)

- **R2-0B**(`docs/h7r/r2/R2-DIAG-1/r2_0b/`, 실행 commit `68df43b`)
  - 재현: `METRIC_ONLY`(\|ΔL\| 4.9e-10, 예측 \|Δv\| 최대 5.2e-7, 같은 배정과 step)
  - ε = 0.0019461(sample 부분이 정함, seed 부분 0.000673)
  - seed 세 개 모두 c5 대비 주 지표가 ε보다 크게 개선됐다. 직접 control 대비 `DEVELOPMENT_SIGNAL`은 아니다.
  - 자세한 내용: 초안 §16
- **기록 사본 폴더**: docs 아래 `runs/`는 `.gitignore`의 `runs/` 규칙에 걸린다(`4180beb`에서 강제 추가). 앞으로 사본은 `records/`에 둔다.
- **D4 baseline 하위 계약 초안**: `docs/h7r/r2/R2-D4-BASELINE-1-draft.md`(draft, 실행 불가)
  - c5 대 세 seed 모델의 수 선택과 Rapfi continuation을 H-dev 396판에서 본다.
  - 실행 전에 적어 둔 예상 정밀도: 비교당 약 420 관측, 구간 반폭 약 0.05. 가장 가능성이 높은 결과는 `NOT_ESTABLISHED`다.
  - 탐색과 Rapfi 비용은 계획용으로 약 2–3시간이다.
  - 문서 검토 → 구현·fixed 계약 → smoke·본 실행의 세 단계로 따로 승인받는다.
