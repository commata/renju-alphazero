# R2-DIAG-1 — value 학습 병목 진단 계약 초안

작성일: 2026-10-30. 상태: 설계 문서. 계약 `docs/h7r/r2/R2-DIAG-1.contract.json`은 **fixed, R2-0A만 승인**(§13). R2-0B·ε 확정·D1–D5는 미승인. 연구 champion: c5. anchor: 원본 H3 policy / H6 v3 value.

이 문서는 계획과 구현을 대조한 검토용 초안이다. 기존 `h7r-eval-contract-v1` 형식의 Arena 계약이 아니므로 그 loader로 읽을 수 없다. 평가 전에 할 일은 세 가지다.

1. 아래 제안값을 검토해 확정한다.
2. 계약 형식과 검증기를 구현한다.
3. 계약을 fixed로 커밋한다.

검토 반영 내역은 §11(1차)과 §12(2차)에 있다. 2차 반영과 함께 R2-0A 평가 도구(`scripts/h7r_r2.py`, `src/hybrid/h7r_r2.py`)와 계약 draft(`docs/h7r/r2/R2-DIAG-1.contract.json`)를 구현했다(§9). 그 뒤 사용자 승인으로 계약을 fixed로 바꾸고 R2-0A만 승인했다(§13).

## 1. 목적과 범위

질문은 "value 학습의 변화가 작게 보이는 것이 측정 해상도 때문인가, 아니면 학습·자료·목표에 병목이 있기 때문인가"다. 모든 결과는 개발용 진단이다. H-dev 손실이 개선돼도 기력 향상, 승급, 독립 일반화의 근거로 읽지 않는다.

| 단계 | 내용 | 승인 |
|---|---|---|
| R2-0A | 기존 여섯 모델을 같은 H-dev에서 재측정한다. 평가만 하고 새 학습은 하지 않는다. | 별도 |
| R2-0B | 같은 recipe를 학습 seed 세 개로 재학습해 변동을 재고 ε를 보정한다. 앞 제안의 "R2-1"이며, 계획 §3.1에 맞춰 R2-0의 내부 단계로 이름을 통일한다. 재현 대조 1회를 포함한다(§7). | 별도. 재현 허용 오차와 하위 실험 control 규칙을 정한 뒤 |
| R2-D1~D5 | 원인별 진단이다. | 하위 계약별 별도 |

이 초안은 어떤 실행도 승인하지 않는다. 실행 권한과 계약 상태는 따로 기록한다.

범위 밖인 것: H-conf/A-conf, 확인 opening suite, champion 교체, V9·SER5·웹 변경, frozen 엔진·runtime·factory·canonical 변경.

## 2. 고정 자료와 출처

H-dev는 `R1-HDEV-1`로 생성·검증한 자료만 쓴다.

| 항목 | 고정 값 |
|---|---|
| 근거 커밋 | `0065564ff778c3ec4a66ddaf1cf4ed8717647f2e` |
| manifest | `docs/h7r/hdev/R1-HDEV-1-manifest.json` |
| 계약 SHA | `348fecda98c4122dd45abc918c6203683873837cdd5f5c8219f7ac9c187a68b0` |
| cache binding SHA | `9724d878eed46c9b2a803d212671bc1bebb2c024ed7ba44398468dfd243d4886` |
| 원본 대국 SHA | `93b05a037f20c60104268bc11fa572dc999496ec966bbede482394a3c4bab3f6` |
| gzip rows SHA | `ed0e4be2a212d7bcce72c501e4674d4376c1c8293fd1f079c7e17b4d99897131` |
| cache labels SHA | `c903d285f1e5e447a4053407edbb57bbc83ecdc61dd140657b1388e6502aa2ef` |
| test tensor SHA | `c3b389dad321b82ea98670d7effb1a717d3642abc95718d53b2692dfa5e556fd` |

**표본 구성**
- 생성한 400판 중 4판은 benchmark probe 국면에 도달해 기존 규칙으로 제외됐다(`scripts/h7c_labels.py`, `probe_game_min_ply` 4). 제외 규칙을 바꾸거나 4판을 보충하지 않는다.
- 평가 cache는 source game 396개, 15,985행이다. outcome 13,805행(weight 0.1), proof 2,180행(weight 1.0)이고, train/val은 0행이다.
- 행 기준 weight 몫은 outcome 38.8%, proof 61.2%다.
- 무승부 15판에서 나온 t=0 outcome 행이 3,080개로, outcome 행의 22%다.

**시작 전 검사와 접근 규칙**
- `checked_cache`로 다음 연결을 차례로 검사한다: 계약 → 400판 전체 → 평가 행 → 모든 cache 파일.
- cache의 `test`라는 이름은 최종 확인용이라는 뜻이 아니다. H-dev는 반복 개발·선택용이다.
- H-dev는 gradient, replay, distillation, 다른 학습 cache에 넣지 않는다.

## 3. 모델과 비교

여섯 모델을 모두 측정한다.

**사전 지정 비교 여섯 개**
- 기본 다섯 개: `c6:c5`, `c7:c5`, `c8:c5`, `mask:c5`, `orig:c5`
- 처방 비교 한 개: `mask:c8`

**부호 규칙**
- 손실 차이 Δ(A:B) = L(A) − L(B)다. 음수가 A의 개선이다.
- 개선량을 양수로 쓸 때는 I = −Δ로 따로 표기한다(§7).

| 이름 | value checkpoint SHA-256 | value metadata SHA-256 | 출처 기록 |
|---|---|---|---|
| c5 | `4e28a62a4a31e462ea116a28d625a80bf2833fdcb30dccbd308b7271c2803024` | `1fcb49ead057881e964c4fac81f99968ee86f7494701343c6083eb82516008f1` | `docs/h7c/c5/value_record.json` |
| c6 | `67e7817be8dc973c13f377821a7264a9bbb9f873a33b7669d00f46e0b5168482` | `a2592a5821cdce37100c4e6bff9277382e58839df51734db4229d7afb12252bb` | `docs/h7c/c6/value_record.json` |
| c7 | `b88ed5e92c3754b931921dac07a17d6ecd63f1c5ac39af580ae34b7946236769` | `8d021caba369d5520d70f5dbdee5449839413ad8d1f930744c12b52ef183c66d` | `docs/h7c/c7/value_record.json` |
| c8 | `7f665411889aee1e0a3438928c79c24a3e743b2298777d13d2295559c0d74923` | `d1345fce42561d41780431e4418902e44473f4bb69d715b3684fad64150c5aa2` | `docs/h7c/c8/value_record.json` |
| mask | `8efe85866378f30bc84593b1270a485937c4ffda549feaa8b750ba47053bfa4c` | `0babe95ea797dd79c77f0efed37d1fa6e4ddc6451bb2a40211d0a3765b797d86` | `docs/h7c/nmask/value_record.json` |
| orig | `a0fcccc401a833a08994e21cd7e963d5e7adc7a7cff715de14259bbed733ceca` | `51fd3fa4623314d415e1c75140c048a88daf9526098bd38ef771e11929d306ea` | `docs/external/x2_c0_prerl.json` (tbv3) |

공통 H3 policy checkpoint는 `efee483284dfdd87eb6caa59d0ea3fd015c9dd5862e5e26925b9a23095d0ce61`, metadata는 `5648b7fbe71dfa4ab92f6c9bc505ee9ec0af68414f86b97ea22744c9fc022609`다.

**identity 검사**
- value 예측은 value checkpoint와 그 metadata의 `model_config`로만 정해진다. 평가 도구는 policy checkpoint를 읽지 않는다. policy identity는 출처 기록으로만 남긴다.
- 모델마다 다음 세 가지를 대조한다. 하나라도 어긋나면 멈춘다.
  1. 명시한 checkpoint 파일과 metadata 파일의 SHA-256 = 계약의 값. metadata는 checkpoint 옆 `best.json`으로 추정하지 않고 경로를 따로 받는다.
  2. **로드한 checkpoint에서 재계산한** frozen state hash(`state_sha256(frozen_state(model))`, trunk·policy head의 parameter와 BN buffer)
  3. 계약의 기대값 `c2fbaac3597185a6ab751f08819ddc9e44644e873f68361036f7cbdb22e1b41d`, 그리고 metadata가 선언한 `frozen_state_sha256`
- metadata의 선언값만으로는 checkpoint의 실제 trunk가 같다고 볼 수 없다. 재계산한 값이 셋 모두와 같을 때만 모델 간 차이가 value head에서만 나온다고 말한다.

**비교 해석**
- c6/c7/c8은 모두 c5에서 출발한 형제 후보다. 학습 seed(20261025)와 설정이 같고, cycle 자료(self-play seed 8806/8807/8808)만 다르다. 순차 학습 추세로 읽지 않는다.
- mask는 c8과 parent, 자료, step, seed가 같다(`reproduction` 기록). 다른 것은 mask 처방이며, NMASK 계약은 이 처방을 **학습 loss와 cycle val 선택 지표 둘 다에** 적용했다. 그래서 선택된 step도 다르다(c8 147, mask 210).
- 따라서 `mask:c8`은 "학습과 checkpoint 선택에 같은 mask 처방을 적용한 recipe 비교"다. 학습 loss만 바꾼 효과를 분리한 실험이 아니다. 기존 실행 한 번씩의 비교이므로 seed 변동도 담지 않는다.
- 원본과의 비교는 기력의 우열을 뜻하지 않는다.

## 4. 지표 — 표본 수와 분모를 구분

**예측과 행 분류**
- 예측은 raw와 D4 평균을 둘 다 저장한다. D4 예측은 8방향 scalar 출력을 먼저 평균한 뒤 제곱오차를 계산한다. 방향별 손실의 평균과 혼동하지 않는다.
- 행은 두 축으로 교차 분류한다.
  - 경로: frozen `analysis.puct_value_v1.exact_leaf` 판정에 따른 network / exact_leaf. 모든 모델에 공통이다. root solver 전체 경로를 뜻하지 않는다.
  - label 종류: outcome / proof

**게임별 손실**

행 i의 오차는 e_i = (v_i − t_i)^2이고, 원래 label weight는 w_i다. 그룹 q에서 게임 g의 손실은 다음과 같다.

    L_g,q = Σ_{i∈(g,q)} w_i e_i / Σ_{i∈(g,q)} w_i

- 그룹에 양의 weight를 가진 행이 없는 게임은 그 그룹의 macro 평균 분모에 넣지 않고, 그 수를 보고한다. 모든 그룹이 항상 396판이라고 쓰지 않는다.

**주 지표(제안): D4 / network / outcome+proof의 게임 균등 평균**

    L_macro,q = (1/G_q) Σ_g L_g,q      (G_q = 그룹 q에 양의 weight 행이 있는 게임 수)

- 게임별 손실을 먼저 계산해 평균한다. 그래서 긴 대국(특히 무승부)이 행이 많다는 이유만으로 더 큰 비중을 얻지 않는다.
- 게임 안에서는 기존 label weight를 유지한다.
- 이 때문에 게임마다 proof와 outcome의 몫이 다르다. proof 행이 있는 결정 대국은 proof가 손실을 주도하고, proof 행이 없는 대국은 outcome만으로 정해진다. proof weight 문제까지 제거한 지표라고 주장하지 않는다.
- 이 혼합이 결과를 좌우하지 않는지 보도록, 아래의 **필수 구성 지표**를 함께 보고한다.

**필수 구성 지표**(주 지표와 같은 bootstrap으로 계산하며, 주 지표와 방향이 다르면 그 사실을 보고한다)
- D4 / network / outcome의 게임 균등 평균
- D4 / network / proof의 게임 균등 평균과 기여 게임 수 G
- D4 / network / outcome+proof의 기존 행 가중 WMSE(Σ w_i e_i / Σ w_i). 과거 trainer·평가와 정의를 맞추는 용도다.

**보조 지표**
- raw와 D4 각각에 대해 다음 그룹의 pooled WMSE: outcome, proof, network/outcome, network/proof, exact_leaf/outcome, exact_leaf/proof, all
- 게임 결과별(흑승/백승/무승부), side-to-move별, proof 세부 종류별 손실
- ply 구간별 손실과 게임·행 수
  - 구간: [5, 20), [20, 40), [40, 80), [80, ∞). 평가 라벨은 ply 5부터 만든다(`h7r_holdout.labels`의 `from_ply=5`).
  - 빈틈이 없는 반열린 구간이다. 5보다 작은 ply의 행이 있으면 도구가 멈춘다.
- 게임 균등 sign 정확도
  - t≠0인 행에서 v·t>0이면 맞음, v=0이면 실패로 센다.
  - trainer의 `sign_acc`는 `(v>0)==(t>0)`을 행 단위로 계산해 v=0, t<0을 맞음으로 센다. 이와 섞이지 않게 이름을 `sign_acc_strict_macro`로 구분한다.
- t=0 행: MSE와 평균 |v|를 보고한다. 0 예측 일치율을 일반 sign 정확도와 섞지 않는다.

**함께 기록할 것**
- 그룹별 games/rows/weight_sum, 그리고 제거·비유한·빈 그룹의 수를 반드시 기록한다.
- 행 단위 pooled 지표는 bootstrap을 게임별로 해도 긴 대국의 높은 비중이 그대로 남는다.

## 5. 통계와 판독 범위

**bootstrap 설정(제안값)**
- source-game paired bootstrap 20,000회, seed **20261030**, percentile 양측 95% 구간
- 기존 사용 seed와 겹치지 않는다: eval 기본값 20261028, pilot 20261029

**공통 재표집**
- 재표집 단위는 cache에 들어 있는 396개 게임 전체다.
- 회차마다 게임 index 396개(중복 포함)를 한 번 뽑고, 그 index를 모든 모델·비교·raw/D4·그룹에 공통으로 적용한다.
- 모델별·그룹별로 따로 재표집하지 않는다. 기존 `paired`는 그룹에 행이 있는 게임만 따로 뽑는 조건부 bootstrap이므로, 이 방식과 같다고 표시하지 않는다.
- 그룹에 행이 없는 게임은 그 회차에서 빈 기여를 한다.
- 어떤 회차에서 그룹 분모가 0이면 그 횟수를 기록하고, 그 그룹의 구간은 판독하지 않는다. 분모가 0인 회차를 버려서 구간을 좁히지 않는다.
- 모델마다 다른 행·게임 마스크를 적용하지 않는다.

**구현 방식(계산량)**
- 모델·그룹·게임마다 충분통계를 먼저 만든다: Σ w e, Σ w, 게임 손실 L_g, sign 분자·분모.
- 그러면 회차당 계산이 행 15,985개가 아니라 게임 396개의 합이 된다.
- macro와 pooled 지표 모두 이 통계로 정확히 계산할 수 있다. 행 단위 순수 Python 반복(기존 `paired`)으로는 20,000회 × 여러 그룹 × 여섯 비교가 수 시간 걸린다.

**판독 범위**
- 그룹의 실제 기여 게임이 20개 미만이면 기술통계만 보고하고 원인 판정에는 쓰지 않는다(제안값). 무승부 관련 그룹은 15판에서만 나오므로, 이 기준과 게임 수를 함께 본다.
- 모든 구간은 개발용 탐색 구간이다. 여러 비교·그룹 가운데 유리한 하나를 골라 확인적 유의성을 주장하지 않는다.
- 주 지표는 위에서 지정한 하나뿐이다. 구성·보조 지표는 원인 가설을 세우는 자료다.
- 최종 확인과 승급 판독은 별도 계약이 다룬다.

## 6. R2-0A — 기존 모델 재측정

- 평가만 한다. 학습, 새 대국, Rapfi, ε 수치 확정은 하지 않는다.
- 결과에는 모델별 주·구성·보조 지표와, 여섯 비교의 paired 차이와 구간을 모두 남긴다.
- 다음 방향 역전을 숨기지 않는다.
  - raw ↔ D4
  - macro ↔ pooled
  - 전체 ↔ network
  - outcome ↔ proof
- 한 지표만 유리하다는 이유로 모델을 승급하거나 계획을 바꾸지 않는다.
- 이 단계에서는 ε를 null로 남기고, "ε 초과 개선"이라는 판독을 내지 않는다.
- 예상 비용
  - GPU 추론: 여섯 모델 × 15,985행 × D4 8방향, 몇 분
  - `exact_leaf` 경로 판정: CPU 한 번
  - bootstrap(충분통계 방식): 몇 분
  - 실측값은 결과에 기록한다.

## 7. R2-0B — seed 변동과 ε 보정

별도 승인을 받은 뒤 학습한다.

**기준 recipe: 기존 c8의 비 mask recipe**
- 그대로 유지하는 것
  - parent c5
  - c8 cycle 자료(self-play seed 8808)와 split
  - replay와 H6 guard
  - step(210), optimizer, augmentation
  - T-val 기준 checkpoint 선택
- 바꾸는 것: 학습 seed만
- H-dev는 학습, early stopping, checkpoint 선택에 쓰지 않는다. 원래 T-val 선택을 마친 모델을 H-dev로 평가한다.

**학습 seed가 바꾸는 것**(`src/hybrid/h7c_train.py`)
- `mixed_epoch`: epoch마다 replay train에서 cycle 행 수만큼 비복원 추출하는 **replay 부분집합**과, 합친 행의 순서(생성기 seed × 1000 + epoch)
- D4 augmentation: batch마다 방향을 뽑는 생성기(seed)
- torch 전역 난수(`torch.manual_seed(seed)`)
- value head는 parent c5에서 이어 학습하므로, head 초기화의 변동은 없다.

따라서 ε_seed의 의미는 다음과 같다.

> 고정된 cycle 자료·replay 모집단·parent에서, replay 추출·batch 순서·augmentation과 실행 비결정성이 만드는 관측 변동. head 초기화와 새 self-play 표본의 변동은 포함하지 않는다.

**같은 seed의 짝맞춤이 보장하는 범위**
- cycle 행 수, replay 모집단 크기, batch 크기, step 수가 모두 같으면, 같은 seed는 epoch별 replay 추출·행 순서·augmentation 배정을 똑같이 만든다. 이것은 문구로 가정하지 않고 도구로 확인한다(`h7r_r2.data_assignment`가 trainer의 `mixed_epoch`를 그대로 불러 지문을 만든다).
- 이 가운데 하나라도 다르면(D1·D2·D5처럼 자료 크기나 혼합 비율이 바뀌는 조건), 같은 seed라도 replay 추출량과 batch 배치가 달라진다. 이때는 같은 부분(예: cycle 행 순서의 공통 접두, augmentation 수열)만 짝으로 기록하고, 나머지 차이를 결과에 적는다.
- "같은 seed이므로 자료 순서가 같다"는 일반 판단은 쓰지 않는다.

**seed 배정(제안)**
- 학습 seed: **31001 / 31002 / 31003**. 저장소 기록에서 충돌은 찾지 못했다. self-play 예약 seed와는 용도가 다르다.
- 세 seed 모두 새 실행 기록을 만든다. 기존 c8(seed 20261025)을 세 seed 중 하나로 대신하지 않는다.

**재현 대조 1회**
- seed 20261025로 c8 recipe를 현재 commit에서 한 번 더 학습한다. 이것은 재현 문제를 **발견하는 점검**이다. 한 번의 재실행으로 GPU 비결정성의 크기를 추정하지 않는다.
- 비교 항목(구분해서 기록)
  1. checkpoint 파일 SHA. 직렬화 방식 때문에 내용이 같아도 다를 수 있다.
  2. state_dict 내용의 동일성: 이름·dtype·shape·bytes의 hash(`state_sha256`). value head만 따로도 기록한다.
  3. 자료 배정 지문: replay 추출, 행 순서, augmentation(`data_assignment`)
  4. 선택된 step, 평가 시점별 T-val objective 곡선
  5. H-dev 예측: 행별 |Δv|의 최댓값과 평균(raw, D4)
  6. H-dev 주 지표 차이 |ΔL|
- 판정
  - `BITWISE_REPRODUCED`: 2의 state_dict 내용이 같다.
  - `METRIC_ONLY`: 2는 다르지만, 3이 같고, 4의 선택 step이 같고, 5와 6이 **실행 전에 고정한 허용 오차** 안이다.
  - `NOT_REPRODUCED`: 그 밖의 모든 경우. 허용 오차가 정해지지 않았거나 항목이 빠진 경우도 여기에 넣는다.
- 허용 오차(예측 |Δv| 최대, 주 지표 |ΔL|)는 R2-0B 실행 승인 전에 계약에 적는다. 이 초안에서는 null이며, null인 동안 도구는 `METRIC_ONLY`를 내지 않는다.
- `NOT_REPRODUCED`이면 **ε 확정과 그 뒤 진단 판독을 멈춘다.** 원인(GPU 비결정성, 코드 차이, 자료 차이)을 기록하고 별도 판단을 받는다. 도구의 `epsilon`은 재현 판정이 `BITWISE_REPRODUCED` 또는 `METRIC_ONLY`가 아니면 값을 내지 않는다.
- 이 대조 결과는 ε 계산 자체에는 넣지 않는다.

**ε 산출식(제안)**

주 지표의 단위는 squared value error다. 백분율이나 %p와 섞어 쓰지 않는다.

    Δ = L_macro,D4,network(c8) − L_macro,D4,network(c5)
    [lo, hi] = R2-0A의 해당 paired 95% bootstrap 구간
    epsilon_sample = max(|Δ − lo|, |hi − Δ|)
    epsilon_seed = max(L_31001, L_31002, L_31003) − min(L_31001, L_31002, L_31003)
    epsilon = max(epsilon_sample, epsilon_seed)

- epsilon_sample은 사전 지정한 `c8:c5` 주 비교만 쓴다. 여러 기존 비교 가운데 가장 유리한 구간으로 바꾸지 않는다.
- epsilon_seed는 세 seed의 mean/sd/min/max와 함께 보고한다. 정규분포 가정에서 세 값의 범위 기대치는 약 1.69σ이며, 표본이 세 개뿐이라 이 범위 자체가 크게 흔들린다.
- 이 ε는 **경험적 개발용 판별 문턱**이다. 정규성, 95% 모수 범위, 80% 검출력을 보장하는 통계적 MDE가 아니다. 계획 §3.1의 "최소 구별 가능 변화"도 이 제한과 함께 읽는다.
- ε가 0이거나 필요한 실행·구간이 빠지면 판독을 보류한다. 임의의 작은 상수를 넣지 않는다.

**이후 조건별 개선의 판독(제안)**

비교 기준은 c5도, 공통 c8 기준 실행도 아니다. **각 하위 계약이 지정한 직접 control**(§8 표)이다. control K는 조건 C와 바꾸는 변수 하나만 다르고, 학습 seed s ∈ {31001, 31002, 31003}마다 따로 학습한다(K_s, C_s).

    I_s = L(K_s) − L(C_s)           (양수가 개선)

- 예: D1의 넓은 head를 전체 자료로 학습한 c8 기준과 비교하면 자료량과 head 폭이 동시에 달라진다. 그래서 D1의 control은 같은 부분집합·같은 학습 예산의 기존 head다.
- 짝을 이룬 K_s와 C_s의 자료 배정이 실제로 얼마나 같은지는 위의 지문으로 기록한다.

`DEVELOPMENT_SIGNAL`은 세 조건을 모두 만족할 때다.
1. 세 seed 모두 I_s > ε
2. 세 seed 모두 paired Δ(C_s:K_s)의 95% 상한이 0보다 작다. 즉 I_s 구간의 하한이 0보다 크다.
3. 위 1–2가 주 지표에서 성립한다.

그 밖의 결과는 모두 `NOT_ESTABLISHED`다.

- c5 대비 결과와 공통 c8 기준 실행(B_s) 대비 결과도 공통 참고값으로 함께 보고한다. 판독에는 쓰지 않는다.
- ε는 c8 recipe의 seed 변동에서 나온 값이다. 자료 크기나 구조가 다른 control에서는 seed 변동이 다를 수 있으므로, 하위 계약은 control의 seed 범위(세 값의 max − min)도 함께 보고하고, 그것이 ε보다 크면 판독을 보류한다.
- 이 기준은 개발 단계의 선별용이다. 다중 실험의 확인적 오류 제어를 보장하지 않는다.
- 구성·보조 지표에 주 지표의 ε를 그대로 적용하지 않는다. 그 지표용으로 따로 보정하지 않았기 때문이다.

## 8. D1–D5 — 판단 구조 초안, 아직 실행할 수 있는 계약이 아님

**원칙**
- 전체 조합을 한꺼번에 실행하지 않는다.
- R2-0A/B 뒤에 조건, 연산량, 소스 SHA, 학습 시간 한도를 담은 하위 계약을 각각 fixed로 만든다.
- 평가를 본 뒤에 이 문서의 문턱을 유리한 쪽으로 바꾸지 않는다.
- 학습 조건은 모두 §7의 seed 세 개를 쓰고, 아래 표의 **직접 control**을 같은 seed로 학습해 짝을 이룬다. c8 기준 실행과 c5 대비 수치는 공통 참고값이다.

| 진단 | 직접 control | 제안 설계 | 결과와 다음 행동 |
|---|---|---|---|
| D1 표현/최적화 | head 폭: 같은 부분집합·같은 학습 예산의 기존 head. lr/BN: 같은 부분집합·같은 head의 원래 설정 | c8 T-train에서 게임 단위로 고정한 부분집합(1k / 5k / 전체에 가까운 규모). head 폭과 lr/BN은 따로 대비한다. 원래 val과 H-dev로는 학습하지 않는다. | 현재 head가 작은 부분집합을 못 맞추고 넓은 head가 더 잘 맞추면 용량 가설이다. 둘 다 못 맞추면 최적화나 target 모순도 조사한다. train 성공 문턱은 target 구성에 맞춘 절대 오차 기준으로 하위 계약에서 먼저 고정한다. H-dev용 ε를 그대로 가져다 쓰지 않는다. |
| D2 자료량/다양성 | 같은 구조·사전 지정 연산 예산(step 고정, epoch 고정 각각)의 기준 자료(c8만) | c8만, c6+c7+c8, 그리고 게임 단위로 중첩된 25/50% 부분집합. 모델과 최적화는 고정한다. gradient step을 고정한 자료 효과와, epoch을 고정해 생기는 추가 연산 효과를 분리한다. | 세 seed가 같은 방향이고 주 지표가 `DEVELOPMENT_SIGNAL`이면 자료 확대 후보다. train만 개선되면 과적합이나 분포 가설이다. 판 수가 늘었다는 것만으로 원인을 확정하지 않는다. |
| D3 label 목표 | 같은 자료·같은 학습 예산의 원래 loss(outcome weight 0.1) | outcome weight 0.1 / 0(= proof만), replay는 같게 둔다. cycle loss의 원래 분모를 유지해, 제거된 기여를 다른 행에 재분배하지 않는다. selection metric은 조건 간 공통 T-val 정의로 고정한다. | outcome/proof별 오차 크기만으로 label noise를 확정하지 않는다. 목표 변경이 주 지표에서 `DEVELOPMENT_SIGNAL`을 내는지, 종류별 교환관계가 어떤지 확인한다. |
| D4 예측→선택 | 그 조건을 낸 하위 계약의 control 모델 | 주 지표에서 `DEVELOPMENT_SIGNAL`이 나온 조건만 대상이다. H-dev의 사전 고정 부분표본에서 sensitivity와 Rapfi 개발 판정을 한다. | 선택이 바뀌지 않거나 개선 방향이 확인되지 않으면 "예측-선택 연결 미입증"으로 기록한다. 곧바로 "목표 불일치 확정"으로 쓰지 않는다. Rapfi 표본·예산·판독은 별도로 fixed한다. |
| D5 replay | 같은 총 step·batch의 원래 혼합 비율(50:50) | D1–D4 뒤에 필요하면 50:50 대 75:25 등을 비교한다. 총 batch/step을 고정하고 H6 guard를 유지한다. | 주 지표 신호와 guard를 둘 다 만족할 때만 다음 후보가 된다. replay를 줄이면서 늘어난 cycle 노출량을 함께 기록하고, 원인을 한 요인으로 과장하지 않는다. |

**하위 계약에서 확정해야 할 값**
- 모든 하위 계약: 직접 control의 정의, control과 조건의 자료 배정 지문 비교 방식
- D1: head 크기, 학습 epoch 한도, train 성공 문턱
- D2: 부분집합 목록과 연산 배정
- D3: 조건별 공통 선택 지표
- D4: 관측 기준과 Rapfi 기준
- D5: 비율

이 값이 정해지기 전에는 D1–D5 전체를 실행할 수 있는 fixed 상태로 표시하지 않는다. 진단 범주가 하나로 좁혀지지 않는 결과도 허용한다.

## 9. 코드·산출물·완료 기준

**R2-0A 도구(구현, 2차 검토 반영)**

- `src/hybrid/h7r_r2.py`(계산·검사)와 `scripts/h7r_r2.py eval`(실행)이다. 출력 형식은 `h7r-r2-eval-v1`, 지표 정의는 `h7r-r2-metrics-v1`이다.
- 기존 `h7r_holdout.py eval`의 출력과 의미는 바꾸지 않았다. 그 도구의 `routes`(frozen `exact_leaf`)와 `checked_cache`만 다시 쓴다.
- 실행 조건(하나라도 어긋나면 멈춤)
  - 계약이 `fixed`이고 커밋돼 있으며, `stages.R2-0A.approved`가 true다. R2-0B는 승인되지 않았다.
  - 작업 트리가 깨끗하다.
  - H-dev cache가 계약의 hash(계약, binding, 대국, 행, labels, test tensor)와 게임 396 / 행 15,985에 맞는다.
  - `--model NAME=checkpoint,metadata`가 계약의 모델 집합과 정확히 같다. 모델마다 파일 SHA, 로드한 checkpoint에서 재계산한 frozen state hash, metadata 선언값, 계약 기대값을 대조한다.
  - 예측에 NaN/Inf가 없다. 출력 파일과 예측 파일이 이미 있으면 덮어쓰지 않는다. 예측 파일은 `data/`나 `runs/` 아래에만 둔다.
- 계산
  - 그룹: all / route / kind / route×kind / t=0, 그리고 network 안의 결과·side·ply 구간·proof 세부 종류·t=0·t≠0
  - 모델 × raw/D4 × 그룹마다 macro, pooled, `sign_acc_strict_macro`, 평균 |v|, games/rows/weight_sum, 판독 가능 여부(기여 게임 ≥ 20), macro 구간
  - 계약 비교마다 macro / pooled 차이와 구간. 공통 재표집 20,000회를 게임별 합계로 계산하며, H-dev 규모에서 몇 초다.
  - 비교마다 주 지표, 구성 지표 3개, 방향 역전 목록(raw↔D4, macro↔pooled, 전체↔network, outcome↔proof)
  - ε: `value` null, 상태 `AWAITING_R2_0B`. `c8:c5` 구간의 반폭(sample 부분)만 기록한다.
- R2-0B용으로 미리 구현한 순수 함수(실행 도구는 아직 없음)
  - `epsilon`: 재현 판정이 통과하지 않으면 `BLOCKED_REPRODUCTION`으로 값을 내지 않는다.
  - `reproduction_verdict`: 자료 배정과 선택 step은 state hash가 같아도 반드시 있고 같아야 한다. 허용 오차가 null이거나 항목이 빠지면 `NOT_REPRODUCED`다.
  - `data_assignment` / `compare_assignments`: trainer의 `mixed_epoch`를 그대로 불러 replay 추출·행 순서 지문을 만들고, augmentation 수열을 trainer와 같은 순서로 뽑는다. 테스트가 실제 `train_value`의 batch와 symmetry를 가로채 지문과 같은지 확인한다.

**R2-0B 전에 더 구현할 것**
- seed별 학습 실행과 기록: 자료 배정 지문, 평가 곡선, 선택 step, state_dict 내용 hash
- 재현 대조 실행: 기준 실행과 재실행의 위 항목, H-dev 예측 |Δv|, 주 지표 |ΔL|
- 계약의 재현 허용 오차와 하위 실험 control 규칙

미구현 지표를 기존 명령 한 번으로 만들 수 있다고 쓰지 않는다.

**산출물 경로**
- 계약: `docs/h7r/r2/R2-DIAG-1.contract.json`(fixed, R2-0A 승인). R1의 `docs/h7r/contracts/`에 두지 않는다. 그 폴더의 JSON은 모두 R1 Arena 계약으로 읽혀 seed 충돌 검사를 받기 때문이다.
- 결과 요약: `docs/h7r/r2/R2-DIAG-1/`. 집계 수치와 SHA만 둔다.
- 행별 예측(`--predictions`), 대국·행·tensor·checkpoint: `runs/h7r/R2-DIAG-1/`. 커밋하지 않는다.

**fail closed**
- 다음 경우에는 멈춘다.
  - 계약·모델·metadata·cache SHA 불일치, `frozen_state_sha256` 불일치
  - 다른 게임이나 다른 행
  - NaN/Inf 예측·손실
  - 불완전한 실행
  - 결과 덮어쓰기
- 원본을 삭제하거나 수정해서 진행하지 않는다.
- 빈 그룹을 0 손실로 바꾸지 않는다.

**완료 기준**
- R2-0A: 여섯 모델과 여섯 비교의 공통 자료 재측정, 구간, 모든 출처, 범위 한계가 갖춰졌다. ε는 여전히 null일 수 있다.
- R2-0B: 다음을 모두 갖췄다.
  - 세 seed의 적격 학습과 H-dev 평가
  - 재현 대조 판정
  - ε 계산 근거
  - 실패한 seed를 조용히 빼거나 유리한 seed로 바꾸지 않았다는 것
  - seed 세 개가 거친 보정이라는 한계의 명시
- R2 전체: 계획 §3.3을 유지한다. D1–D4를 거쳐 병목 가설이나 미확정을 기록하고, R3에서 바꿀 변수 하나를 근거와 함께 제안한다.
- R2-0A 완료만으로 R2 전체 완료, 승급, 기력 향상을 선언하지 않는다.

## 10. 검토 요청 사항

fixed 전에 정할 핵심 선택은 다섯 가지다.
1. D4/network 게임 균등 평균을 주 지표로 삼고, 구성 지표 세 개를 필수로 할지
2. ε의 경험식과 학습 seed 세 개
3. 개발 신호를 하위 계약의 직접 control(같은 seed)과 비교하고, 3/3 seed·ε·paired 구간 조건으로 판정할지
4. 재현 대조 1회와 그 판정·중단 규칙을 R2-0B에 넣을지. 허용 오차 값은 R2-0B 승인 전에 정한다.
5. R2-0A만 먼저 실행 승인할지. 승인하면 계약을 `fixed`, `stages.R2-0A.approved: true`로 바꿔 커밋한 뒤 실행한다.

이 초안은 제안값을 표시한 설계 문서이며, 실행 승인을 대신하지 않는다.

## 11. 검토 반영 (2026-10-30)

| # | 원 초안 | 수정 | 이유 |
|---|---|---|---|
| 1 | 개선 판정 기준이 "parent 대비" | 같은 seed의 기준 recipe(B_s) 대비. c5 대비는 보고만 함(§12의 2차 반영에서 직접 control로 다시 바꿈) | c5 대비 차이에는 c8 recipe 학습 자체의 효과가 섞인다. 같은 seed로 짝지으면 자료 순서 변동도 줄어든다. |
| 2 | 부호 규칙이 혼재(Δ 음수 = 개선 vs "개선 하한 > 0") | Δ와 I = −Δ를 분리해 정의 | 판정식의 부호 오독을 막는다. |
| 3 | 주 지표 하나 | 구성 지표 세 개(outcome macro, proof macro, pooled WMSE) 필수 | 게임 안의 proof/outcome 혼합 비율이 게임마다 달라, 주 지표 하나로는 원인을 가릴 수 없다. |
| 4 | policy identity로 출처를 식별 | value metadata의 `frozen_state_sha256` 일치 검사와 명시적 metadata 경로 | 평가 도구는 policy checkpoint를 읽지 않는다. 차이가 value head에서만 나온다는 근거는 trunk hash다. |
| 5 | mask:c8의 성격 미기재 | parent·자료·step·seed가 같고 마스킹만 다른 비교라고 명시(단일 실행) | 기록(`nmask/value_record.json`의 `reproduction`)이 그렇게 말한다. |
| 6 | seed 변동의 의미 미기재 | seed가 바꾸는 것(순서·augmentation)과, 초기화 변동이 없다는 것을 명시 | ε_seed의 범위를 과장하지 않기 위해서다. |
| 7 | 없음 | seed 20261025 재현 대조 1회(ε 제외) | 비결정성이나 commit 차이가 seed 변동처럼 보이는 것을 막는다. 학습 비용은 수 분이다. |
| 8 | sign 정확도 | trainer `sign_acc`와 정의가 다르다는 점과 새 이름 명시 | v=0 처리 방식이 다르다. |
| 9 | ply 구간 5–19 … | cache 최소 ply부터 시작하는 빈틈 없는 반열린 구간 | 최소 ply를 확인하지 않은 경계라 빈틈이나 누락이 생길 수 있었다. |
| 10 | 공통 재표집만 명시 | 충분통계 방식과 계산량 명시 | 행 단위 Python 반복으로는 20,000회가 수 시간 걸린다. |
| 11 | 출처 근거 | 모델별 커밋된 기록 경로를 표에 넣음 | 근거 섹션의 서술을 대신한다. |

## 12. 2차 검토 반영 (2026-10-30)

| # | 지적 | 반영 |
|---|---|---|
| 1 | seed는 replay에서 뽑히는 자료도 바꾼다(`mixed_epoch`) | §7에 seed가 정하는 것(replay 부분집합, 행 순서, augmentation, 전역 난수)을 적고, ε_seed의 의미를 "고정된 cycle 자료·replay 모집단·parent에서의 관측 변동, head 초기화와 새 self-play 표본 변동 제외"로 고쳤다. |
| 2 | 같은 seed가 같은 자료 순서를 보장하지 않는다 | 크기가 모두 같을 때만 같은 배정이 된다는 것을 도구(`data_assignment`, 실제 trainer와 대조하는 테스트)로 확인하게 했다. 크기가 다르면 공통 부분만 짝으로 기록한다. 일반 문구는 지웠다. |
| 3 | 공통 c8 기준과 비교하면 원인이 섞인다 | 판정 기준을 하위 계약의 직접 control로 바꾸고, §8 표에 D1–D5별 control을 넣었다. c8 기준과 c5 대비는 공통 참고값이다. control의 seed 범위가 ε보다 크면 판독을 보류한다. |
| 4 | 재현 대조의 판정·중단 기준이 없다 | 비교 항목 여섯 개(파일 SHA, state_dict 내용, 자료 배정, 선택 step·곡선, 예측 차이, 주 지표 차이)와 판정 규칙을 정했다. 허용 오차는 실행 전에 고정하며, null이면 `METRIC_ONLY`가 나올 수 없다. `NOT_REPRODUCED`이면 ε와 후속 판독을 멈춘다(`epsilon`이 값을 내지 않음). 한 번의 재실행은 발견용 점검이라고 적었다. |
| 5 | trunk hash는 실제 모델에서 재계산해야 한다 | `check_model`이 로드한 checkpoint에서 `state_sha256(frozen_state(model))`을 다시 계산해 계약 기대값, metadata 선언값과 대조한다. trunk를 바꾸고 선언값만 맞춘 checkpoint를 거부하는 테스트가 있다. |
| 6 | `mask:c8`은 선택 지표도 다르다 | §3에 "학습과 checkpoint 선택에 같은 mask 처방을 적용한 recipe 비교, 학습 loss만의 효과가 아님"으로 고치고 선택 step(147 / 210)을 적었다. |
| — | 확인한 사실 | 평가 라벨은 ply 5부터 만든다(`from_ply=5`). ply 구간을 [5, 20) … [80, ∞)로 고정하고, 5보다 작은 ply의 행이 있으면 도구가 멈춘다. |

## 13. 계약 확정과 R2-0A 승인 (2026-10-30)

**사용자가 확정한 것**
- 주 지표: D4 / network 게임 균등 손실
- 구성 지표 3개, 모델 6개와 비교 6개
- bootstrap 20,000회, seed 20261030
- ply 구간 [5, 20) / [20, 40) / [40, 80) / [80, ∞)

**계약 상태**(`R2-DIAG-1.contract.json`)

| 항목 | 값 |
|---|---|
| status | `fixed` |
| R2-0A | 승인 |
| R2-0B | 미승인 |
| epsilon.value | null |
| 재현 허용 오차 | null |

- 학습 seed 31001–31003과 ε 산출식은 이후 단계의 설계값이다.
- 이 승인으로 학습이나 ε 계산을 할 수 있게 되지는 않는다. R2-0B 학습 도구가 없고, `epsilon`은 재현 판정이 통과해야 값을 낸다.

**구현 보완**
- `scripts/h7r_r2.py preflight`를 추가했다. 계약, H-dev cache, 모델 여섯 개를 검사만 하고 아무것도 쓰지 않는다.
- `eval`도 모델 여섯 개를 모두 검사한 뒤에야 경로 판정과 예측을 시작한다.
- 재현 판정이 state hash가 같다는 것만으로 다른 필수 검사를 건너뛰지 않게 고쳤다.

**원본 H6 v3 metadata의 `frozen_state_sha256`**
- 이 저장소에는 그 파일(`runs/h6_value_head_v3/best.json`)이 없어서 직접 확인하지 못했다.
- 있을 것이라는 근거는 두 가지다.
  - 그 파일을 쓴 trainer(`h6_value_train.metadata`)는 이 값을 기록한다.
  - `docs/h6-value.md`의 H6-2a′ 결과가 같은 `c2fbaac3…`를 보고한다.
- 그래서 호환 규칙은 넣지 않고 엄격한 대조를 유지한다. 선언이 없으면 preflight가 멈춘다. 그때는 metadata를 고치지 않고, 호환 규칙을 계약·문서에 먼저 적은 뒤 다시 실행한다.

## 14. R2-0A 결과와 수정안 A1 (2026-10-30)

**R2-0A 기록**: `docs/h7r/r2/R2-DIAG-1/r2_0a.json`(커밋 `4944acd`, sha256 `316be86c…`)

- 실행 조건: commit `bd21694`, clean tree, 계약 SHA `b8045043…`, H-dev hash 6개 일치. 원시 예측(`runs/h7r/R2-DIAG-1/predictions.pt`, `8ed43d05…`)은 커밋하지 않았다.
- 모델 6개 모두 재계산한 frozen state와 선언값이 `c2fbaac3…`로 같다. 원본 metadata에도 선언이 있어 호환 규칙은 필요 없었다.
- 실행 0.57분. 행 수는 network 15,279, exact_leaf 706이다. 판독 그룹의 분모 0 회차는 모두 0이다.
- cache의 무승부는 14판이다(probe 제외 4판 중 1판이 무승부). 흑승 209판과 백승 173판이 합쳐 396판이다. 무승부 그룹은 기여 게임이 20판 미만이라 판독하지 않는다.

| 비교 | 주 지표 Δ [95%] | outcome macro | proof macro (374판) | pooled |
|---|---|---|---|---|
| c6:c5 | −0.00118 [−0.00183, −0.00053] | −0.00583 | +0.00066 (0 포함) | −0.00068 |
| c7:c5 | −0.00437 [−0.00588, −0.00281] | −0.01197 | −0.00265 | −0.00171 |
| c8:c5 | −0.00455 [−0.00647, −0.00260] | −0.01894 | +0.00130 (0 포함) | −0.00189 (0 포함) |
| mask:c5 | −0.00209 [−0.00370, −0.00046] | −0.00620 | −0.00052 (0 포함) | −0.00087 (0 포함) |
| orig:c5 | +0.00316 [+0.00090, +0.00550] | −0.00870 | +0.00883 | +0.00533 |
| mask:c8 | +0.00246 [+0.00141, +0.00351] | +0.01274 | −0.00181 | +0.00102 (0 포함) |

- 방향 역전(outcome↔proof): c6:c5, c8:c5, orig:c5, mask:c8
- raw 예측에서 구간이 0을 포함하는 비교: c6:c5, mask:c5, mask:c8. 방향 역전 검사는 부호만 보기 때문에 이것을 표시하지 못한다.
- ε: null. sample 부분(c8:c5 구간 반폭)은 0.00195다.

**읽기(사용자 검토 반영)**

1. H-dev의 사전 지정 주 지표에서는 기존 후보 사이의 차이가 관측됐다. 과거 T-val과 결과가 다른 원인은 아직 분리되지 않았다.
   - 두 평가는 표본만 다른 것이 아니다. 평가 분포, raw/D4, 전체/network, pooled/macro 정의가 모두 다르다.
   - 그래서 "정체의 일부가 해상도 문제"라고 쓰지 않는다.
   - c8:c5의 pooled 구간은 0을 포함한다.
2. c7 / c8 / mask의 개선은 outcome 행과 흑승 대국에 몰려 있다. 백승 대국에서는 손실이 늘었다.
   - c5 self-play의 결과 비율과 맞는 차례별 평균 이동일 수 있다.
   - 이것이 국면 판별력의 변화인지 기저율 이동인지는 X1에서 나눠 본다. 지금은 원인으로 쓰지 않는다.
3. NMASK가 직접 제거한 것은 cycle train의 exact_leaf proof 행 1,710개(weight 1.0)의 손실 기여다. outcome 행의 수, weight, 노출량은 줄이지 않았다(`docs/h7c/nmask/mask_manifest.json`).
   - 지금 말할 수 있는 것: "exact_leaf proof 기여를 제거하고 선택 지표도 바꾼 recipe에서, outcome 예측과 proof 예측의 교환관계가 관측됐다."
   - 이것만으로 label noise나 기저율 이동을 원인으로 확정하지 않는다.
4. H-dev는 c5 actor의 대국이다. c6–c8은 c5 self-play로 학습해서 같은 분포이고, 원본은 그렇지 않다.

**수정안 A1**(`docs/h7r/r2/R2-DIAG-1-A1.json`, fixed)

- 고정 계약(`b8045043…`)을 바꾸지 않고 덧붙인다. 바꾸지 않는 것: 주 지표, 비교, R2-0A 판독, 승급 규칙.
- **X1 탐색 부록**(승인)
  - 입력은 SHA로 고정한다: R2-0A 기록 `316be86c…`, 예측 `8ed43d05…`, H-dev cache는 계약 hash로 고정.
  - 예측: D4와 raw. 그룹: network, network/outcome, network/proof
  - 기술적 분해
    - 그룹의 게임 균등 가중치(w_i / (W_g · G))를 그대로 쓰고, 차례별로 정확히 나눈다.
    - L = Σ_s Ω_s (μ_s² + σ_s²). Ω_s는 차례 s의 가중치 몫, μ_s는 평균 잔차, σ_s²는 잔차 변동이다.
    - 차례별 손실을 단순 평균하지 않는다.
  - 교차 보정 평가
    - 보정 모형: 차례별 상수 1개
    - fold: source game 5개, `int(sha256(game_id), 16) % 5`
    - 각 fold의 보정값은 다른 fold에서만 추정하고, 그 fold에서 평가한다.
    - bootstrap 회차마다 보정값을 다시 추정한다.
    - fold 안에 평가할 가중치는 있는데 바깥에 추정할 가중치가 없는 회차는 세고, 그 구간은 판독하지 않는다.
  - 비교 6개마다 원 손실, 평균 잔차 부분, 변동 부분, 교차 보정 손실의 차이와 구간을 낸다. bootstrap 설정은 계약과 같다.
  - 도구는 다시 계산한 원 손실이 R2-0A 값과 1e-9 안에서 같은지 확인하고, 다르면 멈춘다.
  - 판독: 사후 탐색이다. 독립 확인 근거, 승급, 주 지표 변경에 쓰지 않는다.
- **R2-0B 준비**(미승인)
  - 재현 허용 오차: `abs_primary` ≤ 0.0002, `max_abs_prediction` ≤ 0.001(raw와 D4 중 큰 값)
    - 운영상 허용치이며 통계적 보장값이 아니다.
    - 결과를 본 뒤 완화하지 않는다.
    - 같은 자료 배정, 같은 선택 step 조건은 그대로 필수다.
  - 기준 recipe: c8 기록(`docs/h7c/c8/value_record.json`)과 config(`configs/h7c/c1_value.json`)를 SHA로 고정한다.
- 상태는 수정안 파일이 기록한다. R2-0B 승인은 같은 파일의 `r2_0b.approved`를 바꾸는 커밋으로 한다. X1 기록에는 실행 당시의 수정안 SHA가 남는다.

**R2-0B 도구**(`scripts/h7r_r2_seed.py`, 승인 전에는 실행을 거부)

- `train`
  - 모든 입력이 c8 기록과 같은지 확인한다: parent c5, cycle cache와 rows, replay cache, config.
  - seed만 바꾼 config를 `runs/`에 쓰고, `h7c_train_value.run`을 그대로 호출한다. c8의 recipe decision은 config SHA(즉 seed)를 고정하므로 쓰지 않는다. decision은 검사만 더할 뿐 학습을 바꾸지 않는다.
  - 기록하는 것: 자료 배정 지문, 평가 곡선, 선택 step, state 내용 hash
  - H-dev는 읽지 않는다. NO_CANDIDATE여도 기록을 남긴다.
- `reproduce`
  - seed 20261025 재실행을 c8과 비교한다: 파일 SHA, state 내용 hash, 배정, step, 곡선, H-dev 예측 |Δv|, 주 지표 |ΔL|
  - 판정은 `reproduction_verdict`가 한다. `NOT_REPRODUCED`면 종료 코드 1이고, ε와 이후 판독을 멈춘다.
- ε 계산 도구(seed 모델의 H-dev 평가)는 R2-0B 승인 뒤에 추가한다.

**다음 하위 계약의 순서**: X1 결과 하나로 D3을 정하지 않는다. 다음을 모두 모은 뒤 제안한다.
- 원 지표
- 교차 보정 지표
- 종류별(outcome/proof) 결과
- R2-0B의 seed 변동과 ε

## 근거

- 계획: `docs/h7-rl-plan.md` §3, §9.12
- H-dev manifest: `docs/h7r/hdev/R1-HDEV-1-manifest.json`
- 현재 평가 구현: `scripts/h7r_holdout.py` (`routes`, `predictions`, `paired`, `evaluate`)
- 모델 기록
  - `docs/h7c/c5|c6|c7|c8/value_record.json`, `docs/h7c/nmask/value_record.json`
  - 원본은 `docs/external/x2_c0_prerl.json`의 tbv3 provenance
  - `frozen_state_sha256`은 각 value metadata(`value_best.json`, `docs/h6/h6_2a_best.json` 계열)
- fixed 시 로컬 모델·metadata 실파일을 다시 대조한다.
