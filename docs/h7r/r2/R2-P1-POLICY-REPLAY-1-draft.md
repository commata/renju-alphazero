# R2-P1-POLICY-REPLAY-1: 정책 학습에 H3 정책 보존 replay를 더하는 대조 실험

- 상태: **fixed** (`R2-P1-POLICY-REPLAY-1/p1-contract.json`, 승인 플래그 모두 false)
  - 설계 기록: 이 문서와 `p1-contract.draft.json`(2차 검토 반영본)
  - 고정 계약은 초안에 `status`, `approved`, 소스 SHA 15개, 초안 SHA만 더한 것이다(테스트가 확인).
- 구현과 테스트는 끝났다(§9). 학습과 Arena는 실행하지 않았다.
- 목표: Track B 최강 기력 (사용자 결정, 2026-11-10)
## 1. 질문과 근거

- **질문**
  - 정책 fine-tune이 기존 정책의 판단을 잃는 것을 막는 항을 더하면, 같은 S50 예산에서 정책 후보가 연구 champion(H3 정책 + c5 value)보다 강해지는가.
  - 그리고 기존 학습법보다 나아지는가.
- **근거** (커밋된 기록. 원본 checkpoint·기보를 다시 실행한 검증은 아니다)

| 기록 | 사실 |
|---|---|
| c1 P-only (`docs/h7c/c1/arena_verdict.json`) | 0.36 [0.24, 0.49], 향상 [−0.26, −0.01], 25쌍 |
| c2 P-only (`docs/h7c/c2/arena_verdict.json`) | 0.36 [0.27, 0.45], 향상 [−0.23, −0.05], 25쌍. teacher는 S200 |
| c2 정책 학습 (`docs/h7c/c2/policy_record.json`) | 24,077행, 96 step(2 epoch), 선택 step 77. RenjuNet guard 하락 1.93%p(허용 2%p) |
| 정책 trainer (`src/hybrid/h7c_train.py`) | KL(H3 ‖ student)은 self-play 행에만 있다. RenjuNet은 val top-1 guard로만 쓴다. trunk를 포함해 학습하고, BN은 parent 통계를 유지한다. |
| c5 S400 대 S50 (`docs/h7c/e2/sb_c5_saturation.json`) | 150쌍 0.5767, 향상 [+2.5, +12.67]%p, 수당 시간 ×4.94. 탐색으로 얻을 기력은 있다. |

- **가설** (인과는 입증되지 않았다)
  - 탐색에서 배울 것이 있지만, 적은 self-play 자료에 맞추는 동안 기존 정책의 다른 판단까지 변한다. 그 손실이 이득을 상쇄하거나 넘는다.
  - 작은 확률 변화도 S50의 방문 배분과 선택을 크게 바꿀 수 있다. 그래서 KL 수치만으로 변화가 작았다고 단정하지 않는다.
  - RenjuNet 정확도 하락이 기력 저하의 원인이라는 보장도 없다.

## 2. 두 조건

| | 대조군 C | 처방군 T |
|---|---|---|
| cycle 학습 | c2 정책 학습법 그대로 | C와 같음 |
| 추가 항 | 없음 | RenjuNet **train** 국면에서 KL(H3 ‖ student), 온도 1 |

- **cycle 쪽은 두 군이 완전히 같다**
  - 입력: c2 `policy_rows.jsonl.gz`(`ba2e1bc9…`, 24,077 / 1,423행)
  - 출발점: H3(`efee4832…`), teacher: H3
  - 설정: batch 512, 2 epoch(96 step), lr 2e-4, weight decay 1e-4, min lr 비율 0.05, warm-up 5 step, KL λ 0.5(self-play 행), D4 증강, BN eval, evaluation 10회, guard 하락 0.02
  - cycle batch의 순서와 난수 생성기도 같다.
- **replay 항** (처방군만. 수식은 구현이 갈리지 않도록 고정한다)
  - 표본: 매 step 같은 cycle batch에 더해, RenjuNet train split에서 512개 국면을 뽑는다. 균등 복원 추출이다.
  - 난수: replay 전용 생성기(seed = 학습 seed + 1,000,003)가 국면 index와 D4 변환을 모두 정한다. cycle 생성기는 건드리지 않는다.
  - 입력: 같은 D4 변환을 적용한 같은 입력(planes, 합법 수 mask)을 H3와 student에 함께 넣는다.
  - 분포: 두 모델의 logit을 합법 수 밖에서 −∞로 가린 뒤 log-softmax한다. 합법 수 위에서 재정규화하는 셈이다.
  - KL: 국면마다 KL(p_H3 ‖ p_student) = Σ_합법 a p_H3(a) · (log p_H3(a) − log p_student(a))를 구하고, batch 512개의 평균을 낸다.
    - 기존 cycle KL(`h7c_train._policy_terms`)과 같은 형태다.
  - H3: eval 모드, gradient 차단(`torch.no_grad`)
  - 손실: L = L_cycle(기존 정의와 batch 평균 그대로) + μ · KL_replay, **μ = 0.5**
    - μ는 기존 self-play KL 계수와 같은 크기로 미리 고정한다. 결과를 보고 바꾸지 않는다.
  - cycle batch를 replay로 바꾸지 않는다. 그래서 cycle 학습 노출은 두 군이 같다.
  - 사람 착수 CE replay는 넣지 않는다. 다음 변수로 남긴다.
- **기록** (두 군 모두)
  - cycle KL: 기존 val 지표(`policy_objective`의 kl)
  - replay KL: 학습 중 step 평균. 평가 시점마다 RenjuNet val 전체에서 같은 식으로 한 번 더 잰다.
  - 두 KL은 따로 기록한다. 둘 다 진단값이며 선택에 쓰지 않는다.
  - C에서도 RenjuNet val KL을 잰다. no_grad·eval로 재므로 학습 경로와 난수에 영향이 없다.
- **checkpoint 선택** (두 군 동일)
  - 기존 규칙을 그대로 쓴다: guard(RenjuNet **val** top-1)가 H3 대비 0.02 이내인 평가 중, cycle val objective가 가장 낮은 것
  - replay 항은 선택 지표에 들어가지 않는다.
- **value**: c5(`4e28a62a…`)로 고정한다. 새 head 확대나 value 재학습은 섞지 않는다.

## 3. seed

| 용도 | 값 |
|---|---|
| 학습 seed (두 군 공통) | **20261024**(c2와 같음), 20261111, 20261112 |
| Arena 후보 seed (결과 전 고정) | **20261024** |
| Arena 대국 seed (H7-R 예약 seed) | O0 pairing **8788**, O1 pairing **8789**, 보조 O2 pairing **8790** |
| bootstrap | 20,000회, seed 20261131, opening 쌍 단위 percentile |

- **Arena 후보 seed**를 20261024로 둔 이유: 이 seed의 대조군은 실제 c2 정책을 재현한 것이다. 그래서 과거 기록과 직접 이어진다(§5 재현 gate).
- **Arena 대국 seed**
  - 기존 Arena 도구(`h7r_eval.check_contract`)는 예약 블록 8781–8799의 seed만 받는다.
  - 커밋된 계약이 쓰지 않은 것은 8788–8794다. 8795·8796은 테스트 fixture, 8787은 H-dev가 쓴다.
  - 이 계약이 8788–8790을 쓰면 확인 단계(H-conf, 확인 Arena, anchor)에 남는 seed는 8791–8794(4개)다.
  - 이 배정은 계약 확정 때 도구의 seed 검사로 다시 확인한다.
- 학습 seed와 bootstrap seed는 Arena seed가 아니므로 예약 블록과 무관하다.

## 4. 입력 검증 (prepare, 학습 전)

prepare는 값을 새로 받아들이는 단계가 아니다. 커밋된 기대값과 실제 파일을 대조한다.

- **c2 cycle 입력**
  - `runs/h7c/c2/policy_rows.jsonl.gz`의 파일 SHA `ba2e1bc9…`
  - 행 수 24,077 / 1,423(split별)
- **모델**
  - H3 정책 checkpoint `efee4832…`와 metadata `5648b7fb…`
  - c5 value checkpoint `4e28a62a…`와 metadata `1fcb49ea…`
  - 각 `best.json`의 SHA와 내용(stage, config)
- **RenjuNet cache** (`data/external/renjunet/h3_cache`)
  - 기대값은 H3 cache 보고(`docs/mcts-v8-results/h3_cache_report.json`)에 이미 있다.

| split | 텐서 해시(`tensor_sha256`) | 국면 | 게임 |
|---|---|---:|---:|
| train | `b48a8735bc6d1c87d7efaca52e7c8bf2838c3090b7639f5e1db2682a89430277` | 3,178,091 | 97,098 |
| val | `96f598810e46e9996317b11e13ff72d1e85a47f46666983cab719eca316bd3e7` | 127,316 | 5,354 |

  - manifest 형식, code identity(규칙 `8b5ed710…`, encoder, action index), H2 출력 `228f14b2…`도 같은 보고와 대조한다.
  - **추가 검사**: train과 val의 `game_id` 집합이 서로 겹치지 않는다.
- **c2 재현 대상**: `docs/h7c/c2/policy_record.json`과 `policy_best.json`의 SHA
- 모든 원본은 `data/`와 `runs/`에만 둔다. 커밋하는 것은 해시와 수치뿐이다(RenjuNet 파생 자료).

## 5. 단계와 gate

| 단계 | 하는 일 | 통과 조건 | 실패 시 |
|---|---|---|---|
| prepare | §4 검증 | 기대값과 모두 일치, split 비중첩 | 멈춤, 기록 |
| reproduce | 새 trainer, replay off, seed 20261024 | 아래 재현 조건 전부 | 멈춤. 환경이나 코드 차이로 보고, Arena 없음 |
| train | 3 seed × 2 군 = 6회 | 기록 완성 | 실패한 실행은 기록. Arena 후보 seed가 실패하면 멈춤 |
| link | Arena 후보 = seed 20261024의 T와 C 선택 checkpoint | 둘 다 `CANDIDATE` | 한쪽이 `NO_CANDIDATE`면 멈춤. 다른 seed로 바꾸지 않는다 |
| arena | §6 | 모든 arm 완료, 상대 불법 수 0 | 멈춤, 보존. 재실행·표본 보충 없음 |
| judge | §7 | — | — |

- **재현 조건** (c2 `policy_record.json`과 비교)
  - 정확히 같아야 하는 것
    - 입력 SHA, 설정, parent·teacher identity
    - 총 step 96, step/epoch 48, warm-up 5
    - 평가 step 목록 [10, 19, 29, 38, 48, 58, 67, 77, 86, 96]
    - 평가마다의 eligibility [T, T, F, F, T, T, F, T, F, F]
    - 선택 step 77
  - 1e-6 이내: 평가마다 val objective, CE, KL, guard top-1
  - checkpoint SHA 일치는 기록하되 조건은 아니다.
- **재현 gate의 전제**
  - c2 학습 커밋(`c7e4123`) 뒤 `h7c_train.py`는 세 번 바뀌었다(`f0b0022`, `0ced742`, `76e8a49`). 바뀐 것은 모두 value 쪽이고, 정책 학습 함수는 그대로다.
  - 그래도 torch·CUDA 버전이나 비결정성 때문에 재현이 안 될 수 있다. 그때는 위 표대로 멈추고 보고한다. 허용 오차를 사후에 넓히지 않는다.
- **실험 식별값**
  - prepare·train·link·Arena 기록은 승인 플래그를 뺀 계약의 `experiment_sha256`으로 서로 묶는다.
  - 각 실행 기록에는 그때의 계약 파일 SHA를 따로 남긴다.
  - 승인 커밋으로 계약 파일 SHA가 바뀌어도 앞 단계 기록은 그대로 유효하다(D1 D4 개정 A2의 교훈).
  - Arena용 h7r_eval 계약도 이 `experiment_sha256`을 담고, link 기록의 모델 SHA로 만든다.

## 6. Arena (주 판독)

- 상대: 연구 champion = H3 정책 + c5 value, S50
- 시험: T(seed 20261024 정책 + c5 value), C(같은 seed 정책 + c5 value), 둘 다 S50
- factory: `analysis.tb_v3_net`(v1). 정책은 frozen architecture이고, value는 c5다.
- **실행 구조**
  - 기존 Arena 도구(`scripts/h7r_eval.py play`)는 arm 하나에 suite 하나를 받는다. 그래서 suite별로 arm을 나눈다.

| arm | suite | 쌍 | pairing / 대국 seed |
|---|---|---:|---|
| `O0_T_vs_champion` | O0 (300) | 300 | `O0` / 8788 |
| `O0_C_vs_champion` | O0 (300) | 300 | `O0` / 8788 |
| `O1_T_vs_champion` | O1 (300) | 300 | `O1` / 8789 |
| `O1_C_vs_champion` | O1 (300) | 300 | `O1` / 8789 |

  - 실행 arm은 4개이고, 조건마다 600쌍이다.
  - 같은 suite 안에서 T와 C는 같은 opening, 같은 대국 seed, 같은 pairing을 쓴다.
  - 확인용 suite(`O1-confirm-main`, `O1-confirm-anchor`)는 쓰지 않는다.
  - 이전 pilot·Arena와 일부 수순이 겹친다. 개발 판독이라 허용하고, 기록에 적는다.
- **표본의 성격**
  - O0 300개는 대칭 동치류가 75개다(O1은 298개).
  - 그래서 600쌍을 독립 표본 600개로 해석하지 않는다.
  - 동치류 단위 민감도를 함께 보고한다(§7).
- **규모 근거** (계획값이며 보장이 아니다)
  - 측정된 작은 차이 arm의 pair SD 상한이 0.246–0.252다. 3%p를 검출력 0.8로 보려면 527–553쌍이 필요하다.
  - T − C 차이의 SD는 두 arm의 상관에 달려 있어 미리 알 수 없다.
- **비용**
  - R1 pilot 실측(S50 작은 차이 arm, 6 worker, 쌍당): O1 5.7–7.3초, O0 11.0–12.2초
  - 그대로 쓰면 O0 arm 하나에 55–61분, O1 arm 하나에 29–37분, 네 arm에 약 2.8–3.3시간이다.
  - 실제 비용은 각 arm 기록의 wall time으로 남긴다. 기존 `play`에는 arm 일부만 두는 smoke가 없어서, 초안의 "처음 10쌍 smoke"는 뺐다.
  - 측정한 비용으로 규모를 바꾸지 않는다.
- **보조 arm** (판독·안정성 확정·승급 근거로 쓰지 않는다)
  - seed 20261111과 20261112의 T와 C를 각각 champion과, O2 앞 25개 opening에서 둔다.
  - pairing `O2_aux`, seed 8790. arm 4개, 각 25쌍이다.
  - 학습 기록의 교사 일치율, RenjuNet val top-1, cycle KL, replay KL도 보조값이다.

## 7. 판독 (정책 후보용으로 새로 정한다)

- **통계**
  - 단측 α = 0.025, bootstrap 20,000회(seed 20261131), percentile
  - **층화 bootstrap**: suite 안에서만 opening을 다시 뽑는다(O0 300개, O1 300개를 각각 고정).
  - 뽑힌 opening의 T와 C 결과는 함께 움직인다.
  - 종합 추정치 = 0.5 × (O0 평균) + 0.5 × (O1 평균). 50:50 고정 가중이다.
  - suite별 `h7r_eval.arm_summary`도 함께 보고한다.
  - 주 목표 P와 짝 차이 D는 각각 이 수준에서 읽는다. 결정은 P 하나이므로 α를 나누지 않는다. D는 해석에만 쓴다.
- **P: T 대 champion** (종합 점수 − 0.5)
  - `STRONGER`: 단측 하한 > 0
  - `WEAKER`: 단측 상한 < 0
  - 그 밖은 `NOT_ESTABLISHED`
- **D: T − C** (같은 opening 쌍의 pair score 차이를 위와 같이 층화·가중)
  - `BETTER_THAN_CONTROL`: 단측 하한 > 0
  - `WORSE_THAN_CONTROL`: 단측 상한 < 0
  - 그 밖은 `NOT_ESTABLISHED`
  - 한 군만 champion 대비 유의하다는 이유로 두 군이 다르다고 판정하지 않는다. 두 군의 차이는 D로만 본다.
- **민감도** (판독이 아니라 함께 보고)
  - 동치류 단위 층화 bootstrap: suite 안에서 동치류를 다시 뽑고, 그 동치류의 opening을 모두 함께 움직인다. suite 가중은 50:50 그대로다.
  - P와 D의 판독이 동치류 단위에서도 같은지 기록한다.
- **해석** (기전은 정책 보존 가설과의 일치까지만 말한다)

| P | D | 기록할 문장 |
|---|---|---|
| STRONGER | 무엇이든 | 개발 후보다. 독립 확인 단계의 대상이고, champion은 바꾸지 않는다. |
| 그 밖 | BETTER_THAN_CONTROL | 정책 보존 처방이 기존 학습법보다 도움이 됐고, 지식 손실 가설과 일치한다. champion보다 강하다는 근거는 아니다(회귀 완화까지). |
| 그 밖 | NOT_ESTABLISHED | 이번 처방의 효과는 입증되지 않았다. 지식 손실 가설을 배제하지 않는다(가중치·표본·정밀도 때문에 놓쳤을 수 있다). |
| 그 밖 | WORSE_THAN_CONTROL | 이번 처방이 기존 학습법보다 나빴다. 가설의 배제가 아니라 이 처방의 결과다. |

- 기존 value ε, R2 H-dev 판독, D4 판독은 바꾸지도 쓰지도 않는다.
- **종료 조건**
  - P가 STRONGER가 아니면 이 계약은 끝난다.
  - 같은 자료로 가중치를 바꿔 다시 판정하지 않는다. 다른 μ나 사람 착수 CE는 새 계약이다.

## 8. 별도 단계로 남기는 것

- 독립 확인 평가: 확인 suite, 새 seed(8791–8794), H-conf 선택 품질
- anchor 비열등(원본 H3 / H6), 전술 v2, 수당 시간 gate(평균 +10%, p95 +20%)
- champion 변경, canonical·웹 변경
- S400 등 탐색 예산을 늘리는 엔진 후보. 학습 처방과 분리해 따로 검토한다.
- 강화학습 루프 연결: 이 처방이 위 단계를 통과한 뒤에만

## 9. 구현 범위 (구현 완료, 기존 trainer·기록 보존)

- **고치지 않은 것**
  - `src/hybrid/h7c_train.py`, `scripts/h7c_*`
  - `src/hybrid/h7r_eval.py`, `scripts/h7r_eval.py`
  - frozen engine·runtime 파일, `analysis.tb_v3_net`
  - 기존 기록
- **새로 만든 것**
  - `src/hybrid/h7r_p1_train.py`: `h7c_train`의 함수를 그대로 쓰고, 학습 loop만 replay 항(§2 수식)과 별도 KL 기록을 더한 판으로 둔다.
  - `scripts/h7r_p1.py`: `prepare`, `reproduce`, `train`, `link`, `arena-contract`, `judge`
    - 재현 기대값(step, 평가 step, eligibility, 선택 step)은 코드 상수가 아니라 계약(`stages.reproduce.expected`)에서 읽는다. c2 기록도 같은 값을 보여야 한다.
    - 대국은 기존 `scripts/h7r_eval.py play`로 둔다. judge는 그 arm 기록을 기존 `load_arm`으로 다시 검증해서 읽는다.
  - `tests/test_h7r_p1.py` (22개, §9.1)
- **테스트로 확인한 것**
  - trainer
    - replay off의 기록과 state가 `h7c_train.train_policy`와 같다. RenjuNet val KL 진단을 켜도 같다.
    - 처방군도 cycle batch의 index와 순서가 대조군과 같다.
    - replay KL은 합법 수 위의 softmax로 직접 계산한 값과 같고, H3에는 gradient가 가지 않는다.
  - 계약
    - 승인 플래그를 바꿔도 `experiment_sha256`이 그대로다. 승인 순서, `fixed`, 소스 SHA, 커밋 여부를 검사한다.
    - c2 설정이 계약 값과 정확히 같다. cache 기대값이 H3 cache 보고와 같다.
  - cache 검사: 해시, 국면 수, H2 identity가 다르거나 train과 val의 게임이 겹치면 거부한다.
  - 재현 규칙: eligibility, 선택 step, 평가 step, step 수, 1e-5 차이는 거부하고 1e-9 차이는 통과시킨다.
  - Arena 계약
    - 기존 `h7r_eval.check_contract`와 `load_contract`(seed 중복 검사 포함)가 받는다.
    - arm 8개, seed 8788 / 8789 / 8790이다.
    - 후보가 없는 보조 실행은 arm에서 빠진다.
  - judge
    - suite 50:50 가중이 쌍 수와 무관하다.
    - 동치류 단위 재표집을 한다.
    - T와 C의 opening이 다르면 거부한다.
    - STRONGER, WEAKER, BETTER, WORSE, NOT_ESTABLISHED 판독과 해석 문장을 확인했다.
  - **통합 테스트** (임시 저장소, 대역 입력)
    - prepare → reproduce → train 6회 → link → arena-contract를 실제 코드 경로로 잇는다.
    - 기존 trainer로 만든 c2 대역 기록을 새 trainer가 재현한다.
    - 승인 커밋으로 계약 파일이 바뀌어도 단계가 이어진다.
    - 재현이 실패하면 train을 거부한다.
- **실제 자료에서만 확인되는 것** (데스크톱)
  - c2 재현(CUDA), cache 해시 재계산, 학습 시간, Arena 비용

### 9.1 구현 검토 보완 (2026-11-10, `b4ce54f` 검토)

μ, 학습 seed, 후보 seed, 학습량, 표본 규모, 판독 기준은 바꾸지 않았다.

| 지적 | 보완 | 회귀 테스트 |
|---|---|---|
| 학습 기록 사본 경로 `docs/.../runs/`가 `.gitignore`의 `runs/`에 걸린다 | `docs/.../seed-records/{arm}-{seed}/record.json`로 바꿨다. 도구, 문서, 복사 절차 모두 | 실제 `.gitignore`에서 옛 경로는 무시되고 새 경로는 커밋되는 것 |
| prepare 뒤 cache를 그 manifest와 함께 바꿔도 학습됐다 | reproduce·train이 소비하는 모든 입력(행, H3, c5, cache)을 계약 기대값과 다시 대조한다. 학습은 대조한 그 객체로 한다. cache 텐서 해시는 계약 값과 직접 비교한다 | cache와 manifest를 함께 바꾸면 reproduce가 거부하고 기록이 남지 않는 것 |
| 실행 중 HEAD·dirty 변화가 기록되지 않았다 | 모든 명령이 시작할 때 HEAD(clean), 계약 파일, 소스, 읽는 파일을 기록하고, 게시 직전에 다시 확인한다. 바뀌면 checkpoint도 기록도 게시하지 않는다 | 학습 중 HEAD 이동, tracked 파일 수정, 입력 파일 변경 각각에서 기록·checkpoint가 없는 것 |
| judge가 P1 Arena 계획을 강제하지 않았다(주 arm 1쌍으로 `STRONGER`) | judge가 계약과 검증된 link(커밋된 seed-records로 다시 만든 값)에서 기대 Arena 계약을 재구성하고, 실제 계약과 전체가 같아야 한다. arm 기록의 쌍 수도 계약과 같아야 한다 | 쌍 축소, 모델 교체, arm 누락·추가, budget, 통계, link SHA 변경을 거부. 299쌍 arm 거부. 정상 arm이면 판독 |
| 학습 gate가 reproduce의 `passed`만 믿었다 | 저장된 재현 내용(run, settings, inputs)에 규칙을 다시 적용한다. seed와 c2 기록 SHA도 확인한다 | `passed: true`인 채 내용이 바뀐 기록, 다른 seed, 다른 c2 기록, 다른 설정을 거부 |
| 실행 블록이 실패 로그를 커밋하지 않고 고정 로그명을 덮어썼다 | 시각이 붙은 고유 로그, `__EXIT__` 코드, 성공·실패 모두 커밋, 실행마다 따로 커밋. 기록 보존 블록 p1-F를 추가했다. 블록 재실행은 커밋된 단계를 건너뛰고, 끝난 실행은 다시 학습하지 않고 복사만 한다 | 블록을 bash로 실제 임시 저장소(bare origin, 실제 `.gitignore`)에서 실행: prepare 실패→기록, 성공, 재실행 거부; 중단 뒤 p1-F; 실행 중 p1-F 거부; 학습 실패 뒤 재개; 재현 실패 시 학습 없음; Arena 블록 |

- 통합 테스트는 이제 git을 mock하지 않는다. 실제 임시 저장소에서 승인 커밋 → 기록 생성 → seed-records 복사 → staging → 커밋을 모든 단계에 걸쳐 확인한다.
- 블록 시험에서 하나를 더 찾아 고쳤다. p1-F의 실행 중 프로세스 검사(`pgrep`)가 블록을 실행하는 셸 자신의 명령줄과도 맞았다. 그래서 자기 자신을 제외하는 패턴으로 바꿨다.
- 테스트는 22개다. 연속 3회, 그리고 동시 2회 실행에서 모두 통과했다. 기존 trainer 동등성 테스트도 그대로 통과한다.

## 10. 승인 단계

1. 이 초안 검토
2. 구현과 테스트, 최종 소스 SHA를 채운 계약 확정(`status: fixed`) → `approved.prepare`
3. `approved.train` (재현 gate 포함)
4. `approved.arena` (모델 연결과 Arena 계약 생성 뒤)

## 11. 실행 절차 (데스크톱)

- 단계마다 승인 커밋을 받은 뒤 실행한다(자동 실행기 없음).
- **로그**
  - 명령마다 시각이 붙은 고유 로그를 `runs/.../logs/`에 남긴다. 끝에 `__EXIT__ <코드>`를 덧붙인다.
  - 로그 사본과 manifest를 `docs/.../stage-logs/`에 커밋한다. 성공·실패와 무관하게 커밋한다.
  - 이미 있는 로그·기록은 덮어쓰지 않는다.
- **기록 위치**
  - 학습 기록 사본은 `docs/.../seed-records/`에 둔다. `runs/`는 `.gitignore` 대상이라 커밋되지 않는다.
  - checkpoint는 커밋하지 않는다.
- **중단 뒤 복구**
  - 블록이 기록 커밋 전에 멈추면 **p1-F**를 실행한다.
  - p1-F는 아직 커밋되지 않은 로그 사본과, 끝난 기록(prepare / reproduce / link / judge, 학습 기록 사본)만 커밋한다. 아무것도 다시 실행하지 않는다.
  - 학습 블록은 다시 실행하면 커밋된 단계는 건너뛴다. 기록까지 끝났지만 커밋 전에 멈춘 실행은 다시 학습하지 않고 복사만 한다.
  - 중간에 끊긴 실행(파일은 있고 기록이 없음)은 도구가 거부한다. 지우지 않고 보고한다.
- **예상 시간**
  - prepare: 수 분(cache 해시 재계산 포함)
  - 재현과 학습: 7회 × 수 분(c2 학습은 96 step)
  - Arena: arm 4개 약 3시간과 보조 arm

p1-1. prepare (`approved.prepare` 뒤)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  D=docs/h7r/r2/R2-P1-POLICY-REPLAY-1
  L=runs/h7r/R2-P1-POLICY-REPLAY-1/logs
  if git ls-files --error-unmatch "$D/prepare.json" >/dev/null 2>&1; then echo "STOP: prepare is already recorded"; exit 1; fi
  if [ -e "$D/prepare.json" ]; then echo "STOP: an uncommitted prepare.json exists: run p1-F"; exit 1; fi
  mkdir -p "$L"
  B="prepare-$(date +%Y%m%d-%H%M%S)"
  if [ -e "$L/$B.log" ]; then echo "STOP: $L/$B.log exists"; exit 1; fi
  rc=0
  python -u scripts/h7r_p1.py prepare 2>&1 | tee "$L/$B.log" || rc=$?
  echo "__EXIT__ $rc" >> "$L/$B.log"
  python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$B.log" \
    --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$B.manifest.json"
  expect=(--expect "$D/stage-logs/$B.log" --expect "$D/stage-logs/$B.manifest.json")
  if [ "$rc" -eq 0 ]; then expect+=(--expect "$D/prepare.json"); fi
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  if [ "$rc" -eq 0 ]; then verdict=PASS; else verdict="FAIL (exit $rc)"; fi
  git commit -m "H7-R P1: prepare $verdict (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```

p1-F. 기록 보존 (어느 블록이든 기록 커밋 전에 멈췄을 때만. 아무것도 다시 실행하지 않는다)

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  D=docs/h7r/r2/R2-P1-POLICY-REPLAY-1
  R=runs/h7r/R2-P1-POLICY-REPLAY-1
  L="$R/logs"
  echo __START__
  # [1] / [l]: the patterns match the P1 commands but not this block's own command line
  if pgrep -af 'scripts/h7r_p[1][.]py|scripts/h7r_eva[l][.]py play'; then echo "STOP: a P1 command is still running"; exit 1; fi
  git reset -q -- "$D"
  test -z "$(git status --porcelain --untracked-files=no)"
  tracked(){ git ls-files --error-unmatch "$1" >/dev/null 2>&1; }
  expect=()
  shopt -s nullglob
  for f in "$L"/*.log; do
    b="$(basename "$f" .log)"
    tracked "$D/stage-logs/$b.log" && continue
    echo "== $f"; tail -n 2 "$f"
    python scripts/h7r_d4_records.py copy-logs --resume --src-dir "$L" --pattern "$b.log" \
      --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$b.manifest.json"
    expect+=(--expect "$D/stage-logs/$b.log" --expect "$D/stage-logs/$b.manifest.json")
  done
  for n in prepare reproduce link judge; do
    if [ -f "$D/$n.json" ] && ! tracked "$D/$n.json"; then expect+=(--expect "$D/$n.json"); fi
  done
  for rec in "$R"/[CT]-*/record.json; do
    run="$(basename "$(dirname "$rec")")"
    dst="$D/seed-records/$run/record.json"
    tracked "$dst" && continue
    mkdir -p "$(dirname "$dst")"
    cp --no-clobber "$rec" "$dst"
    cmp "$rec" "$dst"
    expect+=(--expect "$dst")
  done
  if [ "${#expect[@]}" -eq 0 ]; then
    git push origin HEAD:feat/h7-c1
    echo "nothing left to record; push checked"; exit 0
  fi
  for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
  python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
  git commit -m "H7-R P1: attempt records after an interruption (records only)"
  git push origin feat/h7-c1
  echo "__RECORDED__"
)
```

p1-2. 재현 gate와 학습 6회 (`approved.train` 뒤. 실행마다 따로 커밋한다)
   - 재현 결과는 PASS든 FAIL이든 커밋한다. FAIL이면 학습하지 않고 멈춘다.
   - 다시 실행하면 커밋된 단계는 건너뛴다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  D=docs/h7r/r2/R2-P1-POLICY-REPLAY-1
  R=runs/h7r/R2-P1-POLICY-REPLAY-1
  L="$R/logs"
  mkdir -p "$L"
  STAMP="$(date +%Y%m%d-%H%M%S)"
  tracked(){ git ls-files --error-unmatch "$1" >/dev/null 2>&1; }
  run(){  # log base, then the command; sets rc; the log ends with __EXIT__
    local b="$1"; shift
    if [ -e "$L/$b.log" ]; then echo "STOP: $L/$b.log exists"; exit 1; fi
    rc=0
    "$@" 2>&1 | tee "$L/$b.log" || rc=$?
    echo "__EXIT__ $rc" >> "$L/$b.log"
  }
  record(){  # message, log base ("" for none), then the record files
    local msg="$1" b="$2"; shift 2
    local expect=()
    if [ -n "$b" ]; then
      python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$b.log" \
        --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$b.manifest.json"
      expect+=(--expect "$D/stage-logs/$b.log" --expect "$D/stage-logs/$b.manifest.json")
    fi
    for f in "$@"; do expect+=(--expect "$f"); done
    for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
    python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
    git commit -m "$msg"
    git push origin feat/h7-c1
  }
  if ! tracked "$D/reproduce.json"; then
    if [ -e "$D/reproduce.json" ]; then echo "STOP: an uncommitted reproduce.json exists: run p1-F"; exit 1; fi
    b="reproduce-$STAMP"
    run "$b" python -u scripts/h7r_p1.py reproduce --device cuda
    files=()
    if [ -f "$D/reproduce.json" ]; then files+=("$D/reproduce.json"); fi
    record "H7-R P1: reproduction gate exit $rc (records only)" "$b" ${files[@]+"${files[@]}"}
    if [ "$rc" -ne 0 ]; then echo "STOP: the reproduction gate did not pass (exit $rc)"; exit 1; fi
  fi
  for s in 20261024 20261111 20261112; do
    for arm in C T; do
      dst="$D/seed-records/$arm-$s/record.json"
      if tracked "$dst"; then echo "done: $arm-$s"; continue; fi
      b=""
      rc=0
      if [ ! -f "$R/$arm-$s/record.json" ]; then
        b="train-$arm-$s-$STAMP"
        run "$b" python -u scripts/h7r_p1.py train --arm "$arm" --seed "$s" --device cuda
      fi
      files=()
      if [ -f "$R/$arm-$s/record.json" ]; then
        mkdir -p "$(dirname "$dst")"
        cp --no-clobber "$R/$arm-$s/record.json" "$dst"
        cmp "$R/$arm-$s/record.json" "$dst"
        files+=("$dst")
      fi
      record "H7-R P1: run $arm $s exit $rc (records only)" "$b" ${files[@]+"${files[@]}"}
      if [ "$rc" -ne 0 ]; then echo "STOP: run $arm $s failed (exit $rc)"; exit 1; fi
    done
  done
  echo "__RECORDED__"
)
```

p1-3. link와 Arena 계약 (클라우드 세션)
   - `link`는 커밋된 학습 기록만 읽으므로 클라우드에서 실행한다.
   - `approved.arena` 커밋 뒤 `arena-contract`로 `docs/h7r/contracts/R2-P1-ARENA-1.json`을 만들어 커밋한다.

p1-4. Arena와 judge (Arena 계약 커밋 뒤, tmux 권장)
   - arm이 중간에 멈추면 같은 블록을 다시 실행한다. 끝난 arm은 건너뛰고, 멈춘 arm은 `play`가 journal에서 이어서 둔다.
   - judge 결과를 보고 표본을 보충하거나 다시 두지 않는다.

```bash
(
  set -euo pipefail
  cd ~/renju-h6
  source .venv/bin/activate
  export PYTHONPATH="$PWD/src:$PWD/scripts"
  [ "$(git branch --show-current)" = feat/h7-c1 ] || { echo "STOP: wrong branch"; exit 1; }
  git pull --ff-only origin feat/h7-c1
  test -z "$(git status --porcelain --untracked-files=no)"
  python scripts/check_track_b_frozen.py
  D=docs/h7r/r2/R2-P1-POLICY-REPLAY-1
  A=docs/h7r/contracts/R2-P1-ARENA-1.json
  R=runs/h7r/R2-P1-POLICY-REPLAY-1
  L="$R/logs"
  mkdir -p "$L" "$R/arena"
  STAMP="$(date +%Y%m%d-%H%M%S)"
  if git ls-files --error-unmatch "$D/judge.json" >/dev/null 2>&1; then echo "STOP: the judge is already recorded"; exit 1; fi
  if [ -e "$D/judge.json" ]; then echo "STOP: an uncommitted judge.json exists: run p1-F"; exit 1; fi
  commit_logs(){  # message, then log bases
    local msg="$1"; shift
    local expect=()
    for b in "$@"; do
      python scripts/h7r_d4_records.py copy-logs --src-dir "$L" --pattern "$b.log" \
        --dest-dir "$D/stage-logs" --manifest "$D/stage-logs/$b.manifest.json"
      expect+=(--expect "$D/stage-logs/$b.log" --expect "$D/stage-logs/$b.manifest.json")
    done
    if [ "${#expect[@]}" -eq 0 ]; then return 0; fi
    if [ -f "$D/judge.json" ]; then expect+=(--expect "$D/judge.json"); fi
    for e in "${expect[@]}"; do [ "$e" = --expect ] || git add "$e"; done
    python scripts/h7r_d4_records.py verify-staged "${expect[@]}"
    git commit -m "$msg"
    git push origin feat/h7-c1
  }
  played=()
  for arm in $(python -c "import json, sys; print(' '.join(a['name'] for a in json.load(open(sys.argv[1]))['arms']))" "$A"); do
    if [ -e "$R/arena/$arm.json" ]; then echo "done: $arm"; continue; fi
    b="play-$arm-$STAMP"
    if [ -e "$L/$b.log" ]; then echo "STOP: $L/$b.log exists"; exit 1; fi
    rc=0
    python -u scripts/h7r_eval.py play --contract "$A" --arm "$arm" --device cuda --workers 6 \
      --jsonl "$R/arena/$arm.jsonl" --output "$R/arena/$arm.json" 2>&1 | tee "$L/$b.log" || rc=$?
    echo "__EXIT__ $rc" >> "$L/$b.log"
    played+=("$b")
    if [ "$rc" -ne 0 ]; then
      commit_logs "H7-R P1: Arena arm $arm exit $rc (logs only)" "${played[@]}"
      echo "STOP: arm $arm failed (exit $rc); rerun this block to resume"; exit 1
    fi
  done
  b="judge-$STAMP"
  rc=0
  python -u scripts/h7r_p1.py judge --arms-dir "$R/arena" 2>&1 | tee "$L/$b.log" || rc=$?
  echo "__EXIT__ $rc" >> "$L/$b.log"
  commit_logs "H7-R P1: Arena and judge exit $rc (records only)" ${played[@]+"${played[@]}"} "$b"
  echo "__RECORDED__ $rc"
  exit "$rc"
)
```
