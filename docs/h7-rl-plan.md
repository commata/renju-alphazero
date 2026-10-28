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
- arm(suite마다 같은 opening으로 묶음)
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
| `scripts/h7r_eval.py` | `suite`, `play`(계약의 한 arm, 재개, 예산 arm), `summarize`, `size`, `timing`, `selection`, `judge` |
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

### 9.2 비용 (실측과 추정을 나눠서)

- 컨테이너 smoke(CPU, stand-in 망이라 데스크톱 값이 아니다)
  - tested 쪽 수당 평균: S50 1.22초, S400 5.98초 → **4.9배**(SB 기록의 4.9배와 같다)
  - worker 시간: S50 arm 쌍당 75초, S400 arm 쌍당 191초 → **S400 arm은 S50 arm의 약 2.5배**
- 데스크톱 추정(6 worker)
  - S50 arm 쌍당 18.4초(c8 공식 50쌍 920.9초)
  - S400 arm 쌍당 약 46초(× 2.5)
- pilot 초안(suite 3 × (작은 차이 3 arm + 양성 대조 1 arm) × 100쌍)
  - S50 9 arm × 31분 + S400 3 arm × 77분 ≈ **8.5시간**
  - 데스크톱 smoke로 다시 잰다.
- 줄이는 선택지(결과 전에 정한다)
  - arm당 60쌍 → 약 5시간
  - 작은 차이 쌍을 2개로 → 약 7시간

### 9.3 본 pilot 전에 확정할 것

1. arm당 쌍 수(비용 9.2절)와 포함할 suite · 비교
2. suite 선택 문턱
   - 정보 비율은 (a)와 (b) 중 무엇의 하한으로 순위를 매길지
   - 흑백 점수 차 상한
3. 양성 대조군 보고 형식: 효과와 구간을 그대로 기록하고, 미검출 원인 구분을 보고서에 둔다. 재실행 없음.
4. m의 대상((a) / (b))과 값은 pilot 뒤 확인 계약에서 정한다.
5. holdout 계약(seed, seed용 cycle 번호, H-dev 400판, actor c5)과 그 생성 시점
6. pilot 계약 `R1-PILOT-1`의 `status: fixed` 전환과 커밋(그 전에는 도구가 실행하지 않는다)

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
  - `R1-PILOT-1`(`ab48eb996271`): **draft**. suite 3 × arm 4 × 100쌍, seed 8781 / 8782 / 8783(suite별 pairing). 9.3절 항목을 정한 뒤 `fixed`로 바꿔 커밋해야 실행된다.
  - `R1-SMOKE-1`(`42e9ac6b7b27`): fixed. 데스크톱 smoke용 c8 : c5 S50 2쌍(seed 8798), c5 S400 : c5 S50 1쌍(seed 8799). 판정에 쓰지 않는다.
  - `R1-SMOKE-HOLDOUT`(`a56b0ac6ccdf`): fixed. holdout 경로 확인용 c5 2판(seed 8797, cycle 1). 학습에 쓰이지 않는다.
- 남은 H7-R seed: 8784–8796(holdout · 확인 계약용)

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
