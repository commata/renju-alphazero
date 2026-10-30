# R2-DIAG-1 — value 학습 병목 진단 계약 초안

작성일: 2026-10-30. 상태: **draft / 미승인 / 실행 불가**. 연구 champion: c5. anchor: 원본 H3 policy / H6 v3 value.

이 문서는 계획과 구현을 대조한 검토용 초안이다. 기존 `h7r-eval-contract-v1` 형식의 Arena 계약이 아니므로 그 loader로 읽을 수 없다. 평가 전에 할 일은 세 가지다.

1. 아래 제안값을 검토해 확정한다.
2. 계약 형식과 검증기를 구현한다.
3. 계약을 fixed로 커밋한다.

이 초안을 커밋한 단계에서는 평가, 학습, 평가 코드 변경을 하지 않았다. 검토 반영 내역은 §11에 있다.

## 1. 목적과 범위

질문은 "value 학습의 변화가 작게 보이는 것이 측정 해상도 때문인가, 아니면 학습·자료·목표에 병목이 있기 때문인가"다. 모든 결과는 개발용 진단이다. H-dev 손실이 개선돼도 기력 향상, 승급, 독립 일반화의 근거로 읽지 않는다.

| 단계 | 내용 | 승인 |
|---|---|---|
| R2-0A | 기존 여섯 모델을 같은 H-dev에서 재측정한다. 평가만 하고 새 학습은 하지 않는다. | 별도 |
| R2-0B | 같은 recipe를 학습 seed 세 개로 재학습해 변동을 재고 ε를 보정한다. 앞 제안의 "R2-1"이며, 계획 §3.1에 맞춰 R2-0의 내부 단계로 이름을 통일한다. 재현 대조 1회를 포함한다(§7). | 별도 |
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
- value 예측은 value checkpoint와 그 metadata의 `model_config`로만 정해진다. 평가 도구는 policy checkpoint를 읽지 않는다.
- 그래도 policy identity는 출처 기록으로 함께 남긴다.
- 대신 각 value metadata의 `frozen_state_sha256`(trunk·policy head)이 `c2fbaac3597185a6ab751f08819ddc9e44644e873f68361036f7cbdb22e1b41d`로 모두 같은지 확인한다. 그래야 모델 간 차이가 value head에서만 나온다고 말할 수 있다.
- metadata 파일은 checkpoint 옆의 `best.json`으로 추정하지 않는다. 명시한 경로를 SHA로 대조한다.

**비교 해석**
- c6/c7/c8은 모두 c5에서 출발한 형제 후보다. 학습 seed(20261025)와 설정이 같고, cycle 자료(self-play seed 8806/8807/8808)만 다르다. 순차 학습 추세로 읽지 않는다.
- mask는 c8과 parent, 자료, step, seed가 같고 cycle loss의 마스킹만 다르다(`reproduction` 기록). 그래서 `mask:c8`이 변수 하나만 바꾼 비교다. 다만 기존 실행 한 번씩의 비교이므로 seed 변동은 담지 않는다.
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
  - 구간: [cache 최소 ply, 20), [20, 40), [40, 80), [80, ∞)
  - 빈틈이 없는 반열린 구간을 쓰고, 경계는 평가 전에 확정한다.
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

**학습 seed가 바꾸는 것**
- trainer의 seed는 batch 순서, D4 augmentation 표본, torch 전역 난수를 정한다. value head는 parent c5에서 이어 학습하므로, 무작위 초기화의 변동은 들어 있지 않다.
- 따라서 ε_seed는 "같은 자료·같은 출발점에서 자료 순서와 augmentation이 만드는 변동"이다. 처음부터 학습할 때의 변동이나 자료 표본의 변동이 아니다.

**seed 배정(제안)**
- 학습 seed: **31001 / 31002 / 31003**. 저장소 기록에서 충돌은 찾지 못했다. self-play 예약 seed와는 용도가 다르다.
- 세 seed 모두 새 실행 기록을 만든다. 기존 c8(seed 20261025)을 세 seed 중 하나로 대신하지 않는다.

**재현 대조 1회**
- seed 20261025로 c8 recipe를 현재 commit에서 한 번 더 학습한다.
- c8의 checkpoint SHA와 H-dev 주 지표를 대조해 결과를 다음 중 하나로 기록한다.
  - `BITWISE_REPRODUCED`
  - `METRIC_ONLY`: SHA는 다르고, 주 지표 차이 |ΔL|를 함께 기록한다.
  - `NOT_REPRODUCED`
- 이 대조는 trainer의 결정성과 commit 간 동등성을 확인하는 용도이며, ε 계산에는 넣지 않는다.
- `NOT_REPRODUCED`이면 ε를 계산하기 전에 원인을 기록한다. 원인 후보는 GPU 비결정성, 코드 차이, 자료 차이다.

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

비교 기준은 c5가 아니라 **같은 seed의 기준 recipe 실행**(B_s, s ∈ {31001, 31002, 31003})이다. 조건 C의 seed s 실행 C_s는 B_s와 학습 seed가 같고, 조건이 바꾸는 변수 하나만 다르다.

    I_s = L(B_s) − L(C_s)           (양수가 개선)

`DEVELOPMENT_SIGNAL`은 세 조건을 모두 만족할 때다.
1. 세 seed 모두 I_s > ε
2. 세 seed 모두 paired Δ(C_s:B_s)의 95% 상한이 0보다 작다. 즉 I_s 구간의 하한이 0보다 크다.
3. 위 1–2가 주 지표에서 성립한다.

그 밖의 결과는 모두 `NOT_ESTABLISHED`다.

- c5 대비 결과(L(c5) − L(C_s))도 함께 보고한다. 다만 판독에는 쓰지 않는다. c5 대비 차이에는 "c8 recipe로 학습했다는 것" 자체의 효과가 섞이기 때문이다.
- 같은 seed끼리 짝을 지으면 자료 순서가 같아진다. 그래서 조건 효과와 순서 변동이 덜 섞인다.
- 이 기준은 개발 단계의 선별용이다. 다중 실험의 확인적 오류 제어를 보장하지 않는다.
- 구성·보조 지표에 주 지표의 ε를 그대로 적용하지 않는다. 그 지표용으로 따로 보정하지 않았기 때문이다.

## 8. D1–D5 — 판단 구조 초안, 아직 실행할 수 있는 계약이 아님

**원칙**
- 전체 조합을 한꺼번에 실행하지 않는다.
- R2-0A/B 뒤에 조건, 연산량, 소스 SHA, 학습 시간 한도를 담은 하위 계약을 각각 fixed로 만든다.
- 평가를 본 뒤에 이 문서의 문턱을 유리한 쪽으로 바꾸지 않는다.
- 학습 조건은 모두 §7의 seed 세 개와 같은 seed 짝의 기준 recipe를 쓴다.

| 진단 | 제안 설계 | 결과와 다음 행동 |
|---|---|---|
| D1 표현/최적화 | c8 T-train에서 게임 단위로 고정한 부분집합(1k / 5k / 전체에 가까운 규모). head 폭과 lr/BN은 따로 대비한다. 원래 val과 H-dev로는 학습하지 않는다. | 현재 head가 작은 부분집합을 못 맞추고 넓은 head가 더 잘 맞추면 용량 가설이다. 둘 다 못 맞추면 최적화나 target 모순도 조사한다. train 성공 문턱은 target 구성에 맞춘 절대 오차 기준으로 하위 계약에서 먼저 고정한다. H-dev용 ε를 그대로 가져다 쓰지 않는다. |
| D2 자료량/다양성 | c8만, c6+c7+c8, 그리고 게임 단위로 중첩된 25/50% 부분집합. 모델과 최적화는 고정한다. gradient step을 고정한 자료 효과와, epoch을 고정해 생기는 추가 연산 효과를 분리한다. | 세 seed가 같은 방향이고 주 지표가 `DEVELOPMENT_SIGNAL`이면 자료 확대 후보다. train만 개선되면 과적합이나 분포 가설이다. 판 수가 늘었다는 것만으로 원인을 확정하지 않는다. |
| D3 label 목표 | outcome weight 0.1 / 0(= proof만), replay는 같게 둔다. cycle loss의 원래 분모를 유지해, 제거된 기여를 다른 행에 재분배하지 않는다. selection metric은 조건 간 공통 T-val 정의로 고정한다. | outcome/proof별 오차 크기만으로 label noise를 확정하지 않는다. 목표 변경이 주 지표에서 `DEVELOPMENT_SIGNAL`을 내는지, 종류별 교환관계가 어떤지 확인한다. |
| D4 예측→선택 | 주 지표에서 `DEVELOPMENT_SIGNAL`이 나온 조건만 대상이다. H-dev의 사전 고정 부분표본에서 sensitivity와 Rapfi 개발 판정을 한다. | 선택이 바뀌지 않거나 개선 방향이 확인되지 않으면 "예측-선택 연결 미입증"으로 기록한다. 곧바로 "목표 불일치 확정"으로 쓰지 않는다. Rapfi 표본·예산·판독은 별도로 fixed한다. |
| D5 replay | D1–D4 뒤에 필요하면 50:50 대 75:25 등을 비교한다. 총 batch/step을 고정하고 H6 guard를 유지한다. | 주 지표 신호와 guard를 둘 다 만족할 때만 다음 후보가 된다. replay를 줄이면서 늘어난 cycle 노출량을 함께 기록하고, 원인을 한 요인으로 과장하지 않는다. |

**하위 계약에서 확정해야 할 값**
- D1: head 크기, 학습 epoch 한도, train 성공 문턱
- D2: 부분집합 목록과 연산 배정
- D3: 조건별 공통 선택 지표
- D4: 관측 기준과 Rapfi 기준
- D5: 비율

이 값이 정해지기 전에는 D1–D5 전체를 실행할 수 있는 fixed 상태로 표시하지 않는다. 진단 범주가 하나로 좁혀지지 않는 결과도 허용한다.

## 9. 코드·산출물·완료 기준

**현재 도구로 되는 것**

`h7r_holdout.py eval`이 제공하는 기능:
- 행 가중 WMSE
- 그룹별 조건부 source-game bootstrap
- raw/D4
- network/exact_leaf와 그 안의 outcome/proof 칸
- `--group` seed 변동(pooled WMSE의 mean/sd)

**추가 구현·테스트가 필요한 것**
- 게임 균등 macro 손실, `sign_acc_strict_macro`, t=0 지표. 전역 outcome/proof, ply, side, 결과별 집계
- 396게임 공통 재표집(충분통계 방식)과 빈 그룹 처리
- R2 전용 계약 형식
  - draft/fixed 상태와 승인 단계
  - 모델 identity 네 개(policy·value × checkpoint·metadata)
  - metadata 명시 경로
  - `frozen_state_sha256` 일치 검사
  - H-dev binding 검증
- metric 정의와 ε 산출식의 버전, 모든 입력·출력 SHA
- R2-0B 학습 기록: seed별 trainer 기록, 재현 대조 판정

미구현 지표를 기존 명령 한 번으로 만들 수 있다고 쓰지 않는다. 기존 기록의 metric 의미는 바꾸지 않고, 새 형식·버전으로 추가한다(기존 `h7r_holdout.py eval` 출력은 그대로 둔다).

**산출물 경로**
- 결과 요약: `docs/h7r/r2/R2-DIAG-1/`. 집계 수치와 SHA만 둔다.
- 대국·행·tensor·checkpoint·행별 예측: `runs/h7r/R2-DIAG-1/`. 커밋하지 않는다.
- 계약 예상 경로: `docs/h7r/contracts/R2-DIAG-1.json`. 파일, schema, loader는 아직 만들지 않았다.

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
3. 개발 신호를 같은 seed의 기준 recipe와 비교하고, 3/3 seed·ε·paired 구간 조건으로 판정할지
4. 재현 대조 1회를 R2-0B에 넣을지
5. R2-0A만 먼저 실행 승인할지

이 초안은 제안값을 표시한 설계 문서이며, 실행 승인을 대신하지 않는다.

## 11. 검토 반영 (2026-10-30)

| # | 원 초안 | 수정 | 이유 |
|---|---|---|---|
| 1 | 개선 판정 기준이 "parent 대비" | 같은 seed의 기준 recipe(B_s) 대비. c5 대비는 보고만 함 | c5 대비 차이에는 c8 recipe 학습 자체의 효과가 섞인다. 같은 seed로 짝지으면 자료 순서 변동도 줄어든다. |
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

## 근거

- 계획: `docs/h7-rl-plan.md` §3, §9.12
- H-dev manifest: `docs/h7r/hdev/R1-HDEV-1-manifest.json`
- 현재 평가 구현: `scripts/h7r_holdout.py` (`routes`, `predictions`, `paired`, `evaluate`)
- 모델 기록
  - `docs/h7c/c5|c6|c7|c8/value_record.json`, `docs/h7c/nmask/value_record.json`
  - 원본은 `docs/external/x2_c0_prerl.json`의 tbv3 provenance
  - `frozen_state_sha256`은 각 value metadata(`value_best.json`, `docs/h6/h6_2a_best.json` 계열)
- fixed 시 로컬 모델·metadata 실파일을 다시 대조한다.
